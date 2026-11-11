"""RoboCasa U-unit tau re-selection and sequential resized frame cache."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path: sys.path.insert(0,str(HERE))
import experiment as ex
import p5_baselines as p5b
from expert import _video_frame

STATE=Path(os.environ.get("SKILLNET_STATE_DIR","/work/.skillnet-state"))
SOURCE=STATE/"outputs/robocasa-memory/memory-01/policy-order-v2"
OUT=STATE/"outputs/robocasa-memory/ctx-01/p5_baselines/tau"
FRAME_ROOT=STATE/"cache/robocasa-frames"
CAMERAS=("robot0_agentview_left","robot0_agentview_right","robot0_eye_in_hand")
TARGET_COUNTS=(95,200,282,500,1000)


def _load_grip():
    path=STATE/"outputs/robocasa-memory/ctx-01/p5_baselines/pools/U_gripper_features.npz"
    if path.exists():
        with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
    return {"grip_sign":np.ones(0,np.int8),"terminal_switch":np.zeros(0,bool),"internal_switch_count":np.zeros(0,np.int16)}


def _greedy_cover(distance,tau,*,return_assignment=True):
    """Deterministic ordered medoid ball-cover, every member within tau."""
    d=np.asarray(distance,np.float32);n=len(d);left=np.ones(n,bool);reps=[];assign=np.full(n,-1,np.int32)
    for i in range(n):
        if not left[i]:continue
        m=np.flatnonzero(left & (d[i] <= float(tau)))
        if not len(m):m=np.asarray([i])
        cid=len(reps);reps.append(i);assign[m]=cid;left[m]=False
    reps=np.asarray(reps,np.int32)
    return (reps,assign) if return_assignment else reps


def _count_at(distance,tau):
    return int(len(_greedy_cover(distance,tau,return_assignment=False)))


def _tau_for_count(distance,target):
    lo=0.;hi=float(np.max(distance));best=(hi,_count_at(distance,hi))
    for _ in range(26):
        mid=(lo+hi)*.5;count=_count_at(distance,mid)
        if abs(count-target)<abs(best[1]-target):best=(mid,count)
        if count>target:lo=mid
        else:hi=mid
    # Prefer a slightly smaller threshold when equally close, so the bound is conservative.
    for t in (lo,hi,(lo+hi)*.5):
        c=_count_at(distance,t)
        if abs(c-target)<abs(best[1]-target) or (abs(c-target)==abs(best[1]-target) and t<best[0]):best=(t,c)
    return float(best[0]),int(best[1])


def _query_key_errors(pool):
    q,episodes=p5b._make_queries()
    err=ex.shape_error_matrix(q["shape"],pool["shape"])
    return q,err


def _p5_representative_errors(q,pool,reps,err):
    import torch
    from expert import _load_net
    from experiment import _features
    torch.set_num_threads(8)
    out=[];candidate={k:np.asarray(v)[reps] for k,v in pool.items() if isinstance(v,np.ndarray) and len(v)==len(pool["key"])}
    for seed in (0,1,2):
        net,ck=_load_net(SOURCE/f"models/P5_U_seed{seed}.pt",pool,device="cpu")
        with torch.no_grad():
            ke=torch.nn.functional.normalize(net.key(torch.as_tensor(candidate["key"],dtype=torch.float32)),dim=-1)
            x=_features({"hist":q["hist"],"state":q["state"]},ck["norm"])
            qe=[]
            for s in range(0,len(x),256):
                qe.append(torch.nn.functional.normalize(net.query(torch.as_tensor(x[s:s+256],dtype=torch.float32)),dim=-1))
            scores=torch.cat(qe,0)@ke.T
            picks=scores.argmax(-1).numpy()
        out.append(float(err[np.arange(len(q["task"])),reps[picks]].mean()))
    return float(np.mean(out)),[float(x) for x in out]


def _save_pool(path,pool,reps,assign,tau,grip):
    members=[np.flatnonzero(assign==i).astype(np.int32) for i in range(len(reps))]
    offsets=np.r_[0,np.cumsum([len(x) for x in members])].astype(np.int64);flat=np.concatenate(members)
    action_seqs=[pool["actions_flat"][int(pool["action_offsets"][i]):int(pool["action_offsets"][i+1])] for i in reps]
    action_offsets=np.r_[0,np.cumsum([len(a) for a in action_seqs])].astype(np.int64)
    fields={k:v[reps] for k,v in pool.items() if isinstance(v,np.ndarray) and len(v)==len(pool["key"])}
    fields.update({"tau":np.asarray([tau],np.float32),"representative_unit":reps,"member_offsets":offsets,"member_units":flat,
        "actions_flat":np.concatenate(action_seqs).astype(np.float32),"action_offsets":action_offsets,
        "grip_sign":grip["grip_sign"][reps],"terminal_switch":grip["terminal_switch"][reps],
        "internal_switch_count":grip["internal_switch_count"][reps]})
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    np.savez_compressed(path,**fields)


def run_tau(out=OUT):
    out=Path(out);pooldir=out/"pools";pooldir.mkdir(parents=True,exist_ok=True)
    source_pool=dict(np.load(SOURCE/"pools/U_memory.npz",allow_pickle=False))
    errors=np.load(SOURCE/"cache/errors_U_memory.npy",mmap_mode="r")
    nearest=np.asarray(errors).copy();np.fill_diagonal(nearest,np.inf);nearest=nearest.min(axis=1)
    qs={str(q):float(np.quantile(nearest,q)) for q in (.10,.25,.50,.75)}
    # Use symmetric S-shared error as a metric for medoid coverage.
    distance=((np.asarray(errors,dtype=np.float32)+np.asarray(errors.T,dtype=np.float32))*.5)
    np.fill_diagonal(distance,0.)
    requests={f"nn_q{int(float(q)*100):02d}":v for q,v in qs.items()}
    for target in TARGET_COUNTS:
        t,c=_tau_for_count(distance,target);requests[f"target_{target}"]=t
    # Merge equivalent tau values but retain all reasons in report metadata.
    taus=[]
    for label,t in requests.items():
        match=next((x for x in taus if abs(x["tau"]-t)<1e-9),None)
        if match:match["selected_for"].append(label)
        else:taus.append({"tau":float(t),"selected_for":[label]})
    gripfile=STATE/"outputs/robocasa-memory/ctx-01/p5_baselines/pools/U_gripper_features.npz"
    with np.load(gripfile,allow_pickle=False) as z:grip={k:z[k] for k in z.files}
    q,qerr=_query_key_errors(source_pool);records=[]
    for item in taus:
        tau=item["tau"];reps,assign=_greedy_cover(distance,tau)
        if len(reps)>len(source_pool["key"]):raise AssertionError("invalid representative count")
        fname=f"U_tau_{tau:.8f}.npz";_save_pool(pooldir/fname,source_pool,reps,assign,tau,grip)
        qe=qerr[:,reps]
        oracle=np.min(qe,axis=1)
        p5mean,p5seeds=_p5_representative_errors(q,source_pool,reps,qerr)
        f0=q["frame"]==0
        item.update({"representatives":int(len(reps)),"member_median":float(np.median(np.bincount(assign))),
            "member_p90":float(np.quantile(np.bincount(assign),.9)),"pool_file":fname,
            "heldout_oracle_type_error":float(oracle.mean()),"heldout_oracle_frame0":float(oracle[f0].mean()),
            "p5_type_error_without_retraining":p5mean,"p5_seed_errors":p5seeds})
        records.append(item)
    result={"base_units":int(len(source_pool["key"])),"heldout_stride10_windows":int(len(q["task"])),
        "heldout_frame0_windows":int(np.sum(q["frame"]==0)),"nearest_neighbor_error_quantiles":qs,
        "candidate_requests":{k:float(v) for k,v in requests.items()},"pools":records,
        "clustering":"ordered greedy medoid ball-cover using symmetric pairwise S-shared type error; every assigned unit is within tau of its representative",
        "p5":"frozen three existing P5 checkpoints; score representatives using current candidate embeddings; no retraining and no heldout hyperparameter selection"}
    out.mkdir(parents=True,exist_ok=True)
    if (out/"tau_results.json").exists():raise FileExistsError(out/"tau_results.json")
    (out/"tau_results.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    return result


def read_frame_cache(task,episode,frame,camera,root=FRAME_ROOT):
    """Load one cached 224px RGB frame; camera accepts short or full stream name."""
    aliases={"left":"robot0_agentview_left","right":"robot0_agentview_right","wrist":"robot0_eye_in_hand"}
    c=aliases.get(camera,camera);path=Path(root)/str(task)/f"episode_{int(episode):06d}.npz"
    with np.load(path,allow_pickle=False) as z:
        if c not in z:raise KeyError(f"camera {c} missing in {path}")
        arr=z[c]
        i=int(frame)
        if i<0 or i>=len(arr):raise IndexError(f"frame {i} outside 0..{len(arr)-1}")
        return np.asarray(arr[i],np.uint8)


def _resize_frames(frames):
    import jax
    import jax.numpy as jnp
    from openpi.shared import image_tools
    n=len(frames);batch=16
    out=[]
    for s in range(0,n,batch):
        x=np.stack(frames[s:s+batch]).astype(np.uint8)
        valid=len(x)
        if valid<batch:x=np.concatenate([x,np.zeros((batch-valid,*x.shape[1:]),np.uint8)],axis=0)
        y=np.asarray(image_tools.resize_with_pad(jnp.asarray(x),224,224))
        out.append(y[:valid])
    return np.concatenate(out,axis=0)


def _extract_one(task,episode,root,cache_root,expected_frames):
    import av
    path=Path(cache_root)/task/f"episode_{int(episode):06d}.npz"
    if path.exists():
        with np.load(path,allow_pickle=False) as z:
            sizes={k:len(z[k]) for k in CAMERAS}
            if all(v==expected_frames for v in sizes.values()) and all(z[k].shape[1:]==(224,224,3) and z[k].dtype==np.uint8 for k in CAMERAS):
                return {"task":task,"episode":int(episode),"status":"reused","frames":expected_frames}
        raise ValueError(f"existing incomplete/invalid cache must not be overwritten: {path}")
    arrays={}
    for camera in CAMERAS:
        video=Path(root)/"videos/chunk-000"/f"observation.images.{camera}"/f"episode_{int(episode):06d}.mp4"
        container=av.open(str(video));stream=container.streams.video[0];frames=[]
        try:
            for frame in container.decode(stream):
                frames.append(frame.to_ndarray(format="rgb24"))
                if len(frames)%128==0:
                    pass
        finally:container.close()
        if len(frames)!=int(expected_frames):
            raise ValueError(f"video/action frame count mismatch {task}/{episode}/{camera}: {len(frames)} != {expected_frames}")
        arrays[camera]=_resize_frames(frames)
    if any(a.shape!=(expected_frames,224,224,3) or a.dtype!=np.uint8 for a in arrays.values()):raise AssertionError("bad resized frame cache shape/dtype")
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.stem+f".{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp,**arrays);tmp.replace(path)
    return {"task":task,"episode":int(episode),"status":"created","frames":expected_frames,"bytes":path.stat().st_size}


def run_frames(cache_root=FRAME_ROOT,workers=4):
    manifest=json.loads((SOURCE/"manifest.json").read_text());jobs=[]
    for d in manifest["demos"]:
        with np.load(SOURCE/"episodes"/f"task{int(d['task_id'])}_ep{int(d['episode']):06d}.npz",allow_pickle=False) as z:n=len(z["action"])
        jobs.append((str(d["task"]),int(d["episode"]),str(manifest["task_splits"][int(d["task_id"])]["root"]),str(cache_root),n))
    cache_root=Path(cache_root)
    cache_root.mkdir(parents=True,exist_ok=True)
    progress=cache_root/"progress.jsonl"
    if progress.exists():
        pass
    created=reused=0;started=time.time()
    # spawn avoids inheriting JAX decoder/thread state; affinity is inherited
    # from the parent, normally taskset 8-39 with four 8-thread workers.
    import multiprocessing as mp
    ctx=mp.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=int(workers),mp_context=ctx) as pool:
        futs=[pool.submit(_extract_one,*job) for job in jobs]
        with progress.open("a") as log:
            for i,f in enumerate(concurrent.futures.as_completed(futs),1):
                r=f.result();created+=r["status"]=="created";reused+=r["status"]=="reused"
                if i%10==0 or i==len(futs):
                    rec={"complete":i,"total":len(futs),"created":created,"reused":reused,"elapsed_sec":time.time()-started}
                    log.write(json.dumps(rec)+"\n");log.flush();print(json.dumps(rec),flush=True)
    return {"episodes":len(jobs),"created":created,"reused":reused,"cache_root":str(cache_root),"elapsed_sec":time.time()-started}


def verify_cache_sample(task="PickPlaceCounterToCabinet",episode=6,root=FRAME_ROOT):
    from openpi.shared import image_tools
    import jax.numpy as jnp
    manifest=json.loads((SOURCE/"manifest.json").read_text());d=next(x for x in manifest["demos"] if x["task"]==task and int(x["episode"])==int(episode))
    roots=Path(manifest["task_splits"][int(d["task_id"])]["root"])
    checks=[]
    for camera in CAMERAS:
        video=roots/"videos/chunk-000"/f"observation.images.{camera}"/f"episode_{int(episode):06d}.mp4"
        with np.load(Path(root)/task/f"episode_{int(episode):06d}.npz",allow_pickle=False) as z: arr=z[camera]
        indices=sorted(set([0,len(arr)//2,len(arr)-1]))
        for i in indices:
            raw=_video_frame(str(video),int(i))
            expected=np.asarray(image_tools.resize_with_pad(jnp.asarray(raw,dtype=jnp.uint8),224,224))
            got=read_frame_cache(task,episode,i,camera,root)
            equal=bool(np.array_equal(expected,got))
            checks.append({"camera":camera,"frame":int(i),"equal":equal})
            if not equal:raise AssertionError(f"cache differs from _video_frame for {camera} frame {i}")
    return checks


def write_report(out=OUT):
    out=Path(out);data=json.loads((out/"tau_results.json").read_text())
    status_path=out/"frame_cache_status.json"
    status=json.loads(status_path.read_text()) if status_path.exists() else {}
    complete=bool(status.get("cache_complete",False) and status.get("n_episodes",0)==600 and status.get("verified_frames_equal",0)==status.get("verified_frames_total",-1))
    first="実装: 完了" if complete else "実装: 未完了（画像キャッシュの確認が未完了）"
    lines=[first,"","# RoboCasa τ再設定と画像キャッシュ","",
      f"U単位 {data['base_units']}、保留 stride-10 {data['heldout_stride10_windows']}窓（frame 0 {data['heldout_frame0_windows']}）。最近傍型誤差の分位点を候補に含め、代表数目標でも閾値を探索した。クラスタ距離はS-shared型誤差を対称化したもの。", "",
      "## τプールと保留評価", "", "| τ | 採用理由 | 代表数 | メンバー数中央値 / p90 | oracle型誤差 平均 / frame0 | P5固定・代表内1位型誤差（seed平均） |", "|---:|---|---:|---:|---:|---:|"]
    for p in data["pools"]:
        lines.append(f"| {p['tau']:.8f} | {', '.join(p['selected_for'])} | {p['representatives']} | {p['member_median']:.1f} / {p['member_p90']:.1f} | {p['heldout_oracle_type_error']:.6f} / {p['heldout_oracle_frame0']:.6f} | {p['p5_type_error_without_retraining']:.6f} |")
    lines += ["",f"最近傍型誤差分位点: `{json.dumps(data['nearest_neighbor_error_quantiles'])}`。要求したプールは `pools/` に保存。P5値は既存3 seedを再学習せず代表キー集合上でスコアし直したもの。", "",
      "## 画像キャッシュ", "",f"キャッシュ先: `{FRAME_ROOT}`。全6タスク×100エピソード、3カメラを動画先頭から順次デコードし、既存 `image_tools.resize_with_pad`（224×224）で処理して uint8 保存する。読み出し関数は `tau_image_cache.read_frame_cache`。`_video_frame` とのフレーム一致テストの状態は `frame_cache_status.json` を参照。", ""]
    (out/"report.md").write_text("\n".join(lines))


def main():
    p=argparse.ArgumentParser();p.add_argument("stage",choices=("tau","frames","verify"));p.add_argument("--workers",type=int,default=4)
    a=p.parse_args()
    if a.stage=="tau":
        r=run_tau();print(json.dumps(r,indent=2));
        # Report is updated after cache verification, when complete.
    elif a.stage=="frames": print(json.dumps(run_frames(workers=a.workers),indent=2))
    else:
        checks=verify_cache_sample();path=OUT/"frame_cache_status.json"
        if path.exists():raise FileExistsError(path)
        files=list(FRAME_ROOT.glob("*/episode_*.npz"));by_task={p.parent.name:sum(1 for x in files if x.parent.name==p.parent.name) for p in files}
        if len(files)!=600 or any(n!=100 for n in by_task.values()):raise ValueError(f"frame cache incomplete: {len(files)} files, per task={by_task}")
        path.write_text(json.dumps({"checks":checks,"verified_frames_equal":sum(x['equal'] for x in checks),
            "verified_frames_total":len(checks),"n_episodes":len(files),"per_task":by_task,
            "bytes_total":sum(p.stat().st_size for p in files),"cache_complete":True},indent=2)+"\n")
        write_report();print(json.dumps(checks,indent=2))


if __name__=="__main__":main()
