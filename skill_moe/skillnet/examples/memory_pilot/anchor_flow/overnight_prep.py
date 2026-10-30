#!/usr/bin/env python3
"""Overnight S2 preparation: oracle-anchor ceiling and LIBERO-90 motion stats.

Does not run a closed-loop environment or modify the existing offline evaluator.
"""
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import time
from datetime import datetime, timezone

import numpy as np

HERE = Path(__file__).resolve().parent
EXAMPLES = HERE.parents[1]
for p in (HERE, EXAMPLES):
    if str(p) not in os.sys.path: os.sys.path.insert(0, str(p))

from key_check import DEFAULT_CHECKPOINT, ROW_FIELDS, _read_npz
from research_paths import REPO_ROOT as RESEARCH_REPO_ROOT, STATE_DIR as RESEARCH_STATE_DIR
from memory import load_memory

DEFAULT_OLD_RUN = str(RESEARCH_STATE_DIR / 'outputs/memory-pilot/anchor-v2-offline-20261029T164700Z')
DEFAULT_HELDOUT = str(RESEARCH_STATE_DIR / 'outputs/memory-pilot/anchor-s2-calibrated-20261028T173556Z/calibration/fit-v2/heldout-observations.npz')
DEFAULT_MEMORY = str(RESEARCH_STATE_DIR / 'outputs/memory-pilot/anchor-s2-train-20261028T072152Z/memory/m6.pkl')
DEFAULT_RLDS = Path(str(RESEARCH_STATE_DIR / 'data/libero/libero90_rlds/libero_90_openvla_processed/1.0.0'))
DEFAULT_RAW90 = Path(str(RESEARCH_REPO_ROOT / 'data/LIBERO/libero_90'))
T0S = (0.2, 0.4, 0.6, 0.8, 1.0)


def denoising_steps(t0):
    return max(2, round(10 * float(t0)))


def row_errors(pred, target):
    err=np.asarray(pred,np.float32)[...,:7]-np.asarray(target,np.float32)[...,:7]
    mse=np.mean(err**2,axis=(-2,-1))
    mm=np.linalg.norm(np.sum(err[...,:3]*np.asarray([.0111,.0122,.0110],np.float32),axis=-2),axis=-1)*1000
    return mse,mm


def _bootstrap_delta(values, episodes, seed, reps=4000):
    v=np.asarray(values,np.float64);ep=np.asarray(episodes).astype(str)
    uniq,inv=np.unique(ep,return_inverse=True)
    sums=np.bincount(inv,weights=v,minlength=len(uniq));counts=np.bincount(inv,minlength=len(uniq))
    rng=np.random.default_rng(seed);draw=rng.integers(0,len(uniq),size=(reps,len(uniq)))
    boot=sums[draw].sum(1)/counts[draw].sum(1)
    return float(v.mean()),float(np.quantile(boot,.025)),float(np.quantile(boot,.975))


def run_oracle(out: Path, held_path: str, memory_path: str, checkpoint: str, old_run: str,
               *, batch_size=32, reps=4000):
    import torch
    torch.set_num_threads(8)
    import jax
    import jax.numpy as jnp
    from openpi.shared import nnx_utils
    from smoke import load_model
    from train import observation_batch

    held = _read_npz(held_path, ROW_FIELDS + ("partition", "suite", "skills", "skill_mask"))
    actions = np.asarray(held["actions"], np.float32)
    state = np.asarray(held["state"], np.float32)
    eps_ids = np.asarray(held["episode_id"]).astype(str)
    init_ids = np.asarray(held["init_state_id"]).astype(str)
    n = len(actions)
    memory = load_memory(memory_path)
    mem_ep = np.asarray([str(x["meta"]["episode_id"]) for x in memory.memory])
    mem_init = np.asarray([str(x["meta"]["init_state_id"]) for x in memory.memory])
    mem_base = np.stack([np.asarray(x["base"], np.float32).reshape(-1) for x in memory.memory])
    mem_raw = np.stack([np.asarray(x["meta"]["raw_action"], np.float32) for x in memory.memory])
    if set(eps_ids) & set(mem_ep): raise ValueError("held-out episode appears in memory")
    if jax.default_backend() != "gpu": raise RuntimeError("oracle evaluation requires GPU")

    # Oracle 1: nearest latent base; Oracle 2: nearest complete raw action chunk.
    true_b, _ = memory.encode(actions, state)
    true_b = np.asarray(true_b, np.float32).reshape(n, -1)
    o1, o2 = np.empty(n, np.int64), np.empty(n, np.int64)
    raw_q = actions.reshape(n, -1)
    for i in range(n):
        eligible = (mem_ep != eps_ids[i]) & (mem_init != init_ids[i])
        if not eligible.any(): raise ValueError(f"no eligible memory for row {i}")
        idx = np.flatnonzero(eligible)
        o1[i] = idx[np.mean((mem_base[idx] - true_b[i]) ** 2, axis=1).argmin()]
        o2[i] = idx[np.mean((mem_raw[idx].reshape(len(idx), -1) - raw_q[i]) ** 2, axis=1).argmin()]
    # Use exactly the same tokenizer/image transforms and frozen checkpoint path.
    rows = {k: held[k] for k in ("image_base", "image_left_wrist", "state", "tokenized_prompt",
                                  "tokenized_prompt_mask", "skills", "skill_mask")}
    model, config = load_model(checkpoint)
    obs_all = observation_batch(rows, config)
    sample = nnx_utils.module_jit(model.sample_actions, static_argnames=("num_steps", "t0"))
    # Reuse the exact prefix keys and paired-noise artifact from the prior run.
    saved = np.load(Path(old_run) / "paired-errors.npz", allow_pickle=False)
    old_keys_path = Path(held_path).with_name("query-keys.npy")
    if not old_keys_path.exists():
        raise FileNotFoundError(f"paired query keys not found at {old_keys_path}")
    qkey = np.load(old_keys_path).astype(np.float32)
    qn = qkey / np.maximum(np.linalg.norm(qkey, axis=1, keepdims=True), 1e-8)
    _, prior_res = memory.prior(qn, state)
    anchors = np.stack([np.asarray(memory.decode(mem_base[o1], prior_res, qn, state), np.float32).reshape(n, 10, 32),
                        mem_raw[o2].reshape(n, 10, 32)])

    noise = np.stack([np.random.default_rng(20261029 + s).standard_normal((n, 10, 32)).astype(np.float32)
                      for s in range(3)])
    errors = np.zeros((3, 2, len(T0S), n), np.float32)
    for seed in range(3):
        for lo in range(0, n, batch_size):
            hi = min(n, lo + batch_size)
            obs = jax.tree.map(lambda x: x[lo:hi], obs_all)
            eps = jnp.asarray(noise[seed, lo:hi])
            for ti, t0 in enumerate(T0S):
                for oi in range(2):
                    if t0 == 1.0:
                        pred = sample(jax.random.key(0), obs, num_steps=10, noise=eps, x_src=eps, t0=1.0)
                    else:
                        a = jnp.asarray(anchors[oi, lo:hi])
                        xsrc = (1.0 - t0) * a + t0 * eps
                        pred = sample(jax.random.key(0), obs, num_steps=denoising_steps(t0), x_src=xsrc, t0=t0)
                    errors[seed, oi, ti, lo:hi] = row_errors(np.asarray(pred), actions[lo:hi])[0]

    partition = np.asarray(held["partition"]).astype(str)
    # Independent epsilon defines the reselected worst-30% subset. Reuse the prior
    # run's paired R/Ap measurements (same three epsilons as this oracle run).
    independent_error = np.zeros(n, np.float32)
    sep_rng = np.random.default_rng(20261102)
    for lo in range(0, n, batch_size):
        hi = min(n, lo + batch_size)
        obs = jax.tree.map(lambda x: x[lo:hi], obs_all)
        eps = jnp.asarray(sep_rng.standard_normal((hi-lo, 10, 32)).astype(np.float32))
        pred = sample(jax.random.key(0), obs, num_steps=10, noise=eps, x_src=eps, t0=1.0)
        independent_error[lo:hi] = row_errors(np.asarray(pred), actions[lo:hi])[0]
    worst = np.zeros(n, bool); worst[np.argsort(-independent_error)[:round(.3*n)]] = True
    mse_npz = saved["mse"]
    cfg_tau, cfg_anchor, cfg_t0 = saved["cfg_tau"], saved["cfg_anchor"], saved["cfg_t0"]
    old_cfgs = {}
    for i in range(len(cfg_tau)):
        old_cfgs[(str(cfg_anchor[i]), round(float(cfg_tau[i]), 5), round(float(cfg_t0[i]), 2))] = i
    old_res = {}
    # The existing run contains exactly the same three noise draws, and is the
    # reference for R/Ap's originally fit-selected taus.
    for name, tau in (("R", .1), ("Ap", .05)):
        i = old_cfgs[(name, tau, .2)]
        base = mse_npz[:, 0].mean(0)
        delta = (mse_npz[:, i] - mse_npz[:, 0]).mean(0)
        old_res[name] = {"tau": tau, "t0": .2,
                         "independent_worst30_delta_mse": _bootstrap_delta(delta[worst], eps_ids[worst], 90, reps),
                         "independent_worst30_mse": float((mse_npz[:, i].mean(0))[worst].mean()),
                         "independent_worst30_baseline_mse": float(base[worst].mean()),
                         "independent_worst30_n": int(worst.sum()),
                         "separate_epsilon_seed": 20261102}

    report = {"heldout": str(held_path), "memory": str(memory_path), "rows": n,
              "episodes": int(len(set(eps_ids))), "memory_entries": len(memory.memory),
              "partition_counts": {k: int(np.sum(partition == k)) for k in np.unique(partition)},
              "oracle_candidates": {"O1": "nearest true encoded base among memory rows excluding same episode and init_state", "O2": "nearest normalized raw action chunk among same eligible rows"},
              "t0": list(T0S), "seeds": 3, "episodes_bootstrap_reps": reps,
              "conditions": {}, "independent_bad30": {"rows": int(worst.sum()), "episodes": int(len(set(eps_ids[worst]))), "seed": 20261102, "R_Ap": old_res}}
    for oi, name in enumerate(("O1", "O2")):
        report["conditions"][name] = {}
        for ti, t0 in enumerate(T0S):
            delta = errors[:, oi, ti].mean(0) - errors[:, oi, -1].mean(0)
            report["conditions"][name][str(t0)] = {
                "mse": float(errors[:, oi, ti].mean()),
                "delta_vs_t0_1": _bootstrap_delta(delta, eps_ids, 1000 + oi*100 + ti, reps),
                "select_mse": float(errors[:, oi, ti, partition == "select"].mean()),
                "select_delta_vs_t0_1": _bootstrap_delta(delta[partition == "select"], eps_ids[partition == "select"], 2000 + oi*100 + ti, reps),
            }
    np.savez_compressed(out / "oracle-errors.npz", errors=errors, oracle_memory_indices=np.stack([o1,o2]),
                        separate_epsilon_bad30=worst, separate_epsilon_base_error=independent_error,
                        episode_id=eps_ids, partition=partition, t0=np.asarray(T0S))
    return report


def _segments_from_episode(actions, state, instruction, task_file, episode_id, fps):
    actions=np.asarray(actions,np.float32); state=np.asarray(state,np.float32)
    g=np.asarray(actions[:,6])
    # Change point is the first frame whose raw gripper command changes sign.
    sign=np.sign(g); sign[sign==0]=1
    bounds=np.r_[0, np.flatnonzero(sign[1:] != sign[:-1])+1, len(actions)]
    rows=[]
    for lo,hi in zip(bounds[:-1],bounds[1:]):
        if hi<=lo:continue
        pos=state[lo:hi,:3]
        rel=pos-pos[0]
        # Eight equally spaced points preserve coarse motion shape; append log duration.
        t=np.linspace(0,max(len(rel)-1,0),8)
        interp=np.stack([np.interp(t,np.arange(len(rel)),rel[:,d]) for d in range(3)],axis=1)
        delta=pos[-1]-pos[0]
        feat=np.r_[interp.reshape(-1),np.log1p(hi-lo)].astype(np.float32)
        first=instruction.strip().lower().split()
        verb=first[0].strip(".,;:") if first else "unknown"
        rows.append({"steps":int(hi-lo),"seconds":float((hi-lo)/fps),"dx":float(delta[0]),"dy":float(delta[1]),"dz":float(delta[2]),
                     "distance":float(np.linalg.norm(delta)),"task":task_file,"episode":episode_id,"instruction":instruction,"verb":verb,"feature":feat})
    return rows


def _kmeans_fit(x,k,rng,n_init=3,max_iter=50):
    best=None;best_inertia=float("inf")
    for _ in range(n_init):
        centers=np.asarray(x[rng.choice(len(x),k,replace=False)],np.float32).copy()
        for _ in range(max_iter):
            dist=np.maximum((x*x).sum(1,keepdims=True)+(centers*centers).sum(1)[None]-2*x@centers.T,0)
            labels=dist.argmin(1)
            new=centers.copy()
            for j in range(k):
                ix=labels==j
                if ix.any():new[j]=x[ix].mean(0)
                else:new[j]=x[rng.integers(len(x))]
            if np.allclose(new,centers,atol=1e-5):centers=new;break
            centers=new
        dist=np.maximum((x*x).sum(1,keepdims=True)+(centers*centers).sum(1)[None]-2*x@centers.T,0)
        labels=dist.argmin(1);inertia=float(dist[np.arange(len(x)),labels].sum())
        if inertia<best_inertia:best=(centers,labels);best_inertia=inertia
    return best[0],best[1],best_inertia


def _sample_silhouette(x,labels):
    if len(np.unique(labels))<2:return float("nan")
    d2=np.maximum((x*x).sum(1,keepdims=True)+(x*x).sum(1)[None]-2*x@x.T,0)
    d=np.sqrt(d2);vals=[];unique=np.unique(labels)
    for i in range(len(x)):
        own=labels==labels[i]
        a=float(d[i,own].sum()/max(int(own.sum())-1,1))
        b=min(float(d[i,labels==j].mean()) for j in unique if j!=labels[i])
        vals.append((b-a)/max(a,b,1e-12))
    return float(np.mean(vals))


def run_primitives(rlds: Path, *, raw90=DEFAULT_RAW90, fps=20.0, seed=20261103):
    os.environ["CUDA_VISIBLE_DEVICES"]=""
    import tensorflow as tf
    tf.config.set_visible_devices([],"GPU")
    tf.config.threading.set_intra_op_parallelism_threads(8);tf.config.threading.set_inter_op_parallelism_threads(2)
    info=json.loads((rlds/"dataset_info.json").read_text());split=next(x for x in info["splits"] if x["name"]=="train")
    lengths=split["shardLengths"]; files=sorted(rlds.glob("*train.tfrecord-*"))
    if len(files)!=len(lengths):raise ValueError("incomplete RLDS shard list")
    all_segments=[];task_counter={};episode_count=0
    for ei,serialized in enumerate(tf.data.TFRecordDataset([str(p) for p in files],num_parallel_reads=1)):
        f=tf.train.Example.FromString(serialized.numpy()).features.feature
        original=f["episode_metadata/file_path"].bytes_list.value[0].decode();task=Path(original).name
        acts=np.asarray(f["steps/action"].float_list.value,np.float32).reshape(-1,7)
        states=np.asarray(f["steps/observation/state"].float_list.value,np.float32).reshape(-1,8)
        langs=[x.decode() for x in f["steps/language_instruction"].bytes_list.value]
        if not(len(acts)==len(states)==len(langs)):raise ValueError(f"length mismatch in episode {ei}")
        instr=langs[0] if langs else ""
        segs=_segments_from_episode(acts,states,instr,task,f"rlds-{ei}",fps);all_segments.extend(segs)
        task_counter[task]=task_counter.get(task,0)+len(segs);episode_count+=1
        if ei%500==0:print(f"parsed {ei+1} episodes, {len(all_segments)} segments",flush=True)
    if episode_count!=sum(map(int,lengths)):raise ValueError("episode count differs from RLDS metadata")
    allowed_files={p.name for p in Path(raw90).glob("*.hdf5")}
    if len(allowed_files)!=90:raise ValueError(f"expected exactly 90 raw task files, got {len(allowed_files)}")
    missing_tasks=sorted(allowed_files-set(task_counter))
    x=np.stack([z["feature"] for z in all_segments]);mean=x.mean(0);scale=x.std(0);scale[scale<1e-6]=1
    x=(x-mean)/scale
    rng=np.random.default_rng(seed);sample_n=min(5000,len(x));sample=rng.choice(len(x),sample_n,replace=False)
    kmeans={}
    for k in range(4,17):
        centers,labels,inertia=_kmeans_fit(x,k,rng)
        summaries=[]
        for j in range(k):
            ids=np.flatnonzero(labels==j);tc={};vc={}
            for i in ids:
                task=all_segments[int(i)]["task"];verb=all_segments[int(i)]["verb"]
                tc[task]=tc.get(task,0)+1;vc[verb]=vc.get(verb,0)+1
            summaries.append({"cluster":j,"size":len(ids),"top_tasks":sorted(tc.items(),key=lambda q:-q[1])[:3],"top_verbs":sorted(vc.items(),key=lambda q:-q[1])[:3]})
        kmeans[str(k)]={"silhouette_sample":_sample_silhouette(x[sample],labels[sample]),"sample_n":sample_n,"cluster_sizes":np.bincount(labels,minlength=k).tolist(),"centers":centers.tolist(),"inertia":inertia,"clusters":summaries,"algorithm":"NumPy Lloyd k-means; 3 restarts, 50 iteration cap"}
    # M6-like DP-means with the same kappa scaling, support=10, episodes=3.
    pairs=rng.integers(0,len(x),(min(4096,max(64,2*len(x))),2));ds=np.mean((x[pairs[:,0]]-x[pairs[:,1]])**2,axis=1);ds=ds[ds>1e-8]
    eps=max(.1*float(np.median(ds)),1e-6);centers=[];groups=[]
    for ix in rng.permutation(len(x)):
        if not centers:centers=[x[ix].copy()];groups=[[int(ix)]];continue
        d=np.mean((np.stack(centers)-x[ix])**2,axis=1);j=int(d.argmin())
        if d[j]<=eps:groups[j].append(int(ix));centers[j]=x[groups[j]].mean(0)
        else:centers.append(x[ix].copy());groups.append([int(ix)])
    accepted=[g for g in groups if len(g)>=10 and len({all_segments[i]["episode"] for i in g})>=3]
    dp=[]
    for j,g in enumerate(accepted):
        verbs={};tasks={}
        for i in g:
            v=all_segments[i]["verb"];t=all_segments[i]["task"]
            verbs[v]=verbs.get(v,0)+1;tasks[t]=tasks.get(t,0)+1
        dp.append({"id":j,"size":len(g),"episodes":len({all_segments[i]["episode"] for i in g}),"tasks":len(tasks),
                   "top_tasks":[{"task":t,"n":c} for t,c in sorted(tasks.items(),key=lambda z:-z[1])[:5]],
                   "top_verbs":sorted(verbs.items(),key=lambda z:-z[1])[:5]})
    durations=np.asarray([z["steps"] for z in all_segments]);secs=durations/fps
    result={"source":str(rlds),"episodes":episode_count,"tasks":len(task_counter),"expected_tasks":90,"missing_tasks":missing_tasks,"segments":len(all_segments),"fps_assumed":fps,
            "segment_length_steps":{"q10":float(np.quantile(durations,.1)),"q25":float(np.quantile(durations,.25)),"median":float(np.median(durations)),"q75":float(np.quantile(durations,.75)),"q90":float(np.quantile(durations,.9))},
            "segment_length_seconds":{"q10":float(np.quantile(secs,.1)),"q25":float(np.quantile(secs,.25)),"median":float(np.median(secs)),"q75":float(np.quantile(secs,.75)),"q90":float(np.quantile(secs,.9))},
            "segments_per_task":{"min":int(min(task_counter.values())),"median":float(np.median(list(task_counter.values()))),"max":int(max(task_counter.values())),"counts":task_counter},
            "net_displacement":{"dx":np.quantile([z["dx"] for z in all_segments],[.1,.25,.5,.75,.9]).tolist(),"dy":np.quantile([z["dy"] for z in all_segments],[.1,.25,.5,.75,.9]).tolist(),"dz":np.quantile([z["dz"] for z in all_segments],[.1,.25,.5,.75,.9]).tolist(),"distance":np.quantile([z["distance"] for z in all_segments],[.1,.25,.5,.75,.9]).tolist()},
            "kmeans":kmeans,"dp_means":{"kappa":.1,"epsilon_mse":eps,"raw_candidate_clusters":len(groups),"supported_units":len(accepted),"min_support":10,"min_episodes":3,"units":dp}}
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument("--output-root",default=str(RESEARCH_STATE_DIR / 'outputs/memory-pilot'));p.add_argument("--heldout",default=DEFAULT_HELDOUT);p.add_argument("--memory",default=DEFAULT_MEMORY);p.add_argument("--checkpoint",default=DEFAULT_CHECKPOINT);p.add_argument("--old-run",default=DEFAULT_OLD_RUN);p.add_argument("--reuse-oracle",type=Path,default=None,help="reuse an earlier oracle result JSON instead of rerunning GPU inference");p.add_argument("--rlds",type=Path,default=DEFAULT_RLDS);p.add_argument("--raw90",type=Path,default=DEFAULT_RAW90);p.add_argument("--mode",choices=("all","oracle","primitives"),default="all");p.add_argument("--fps",type=float,default=20.0);p.add_argument("--batch-size",type=int,default=32);a=p.parse_args()
    stamp="overnight-prep-"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out=Path(a.output_root)/stamp;out.mkdir(parents=True,exist_ok=False)
    result={"run_id":stamp,"created_utc":datetime.now(timezone.utc).isoformat(),"closed_loop":False}
    if a.mode == "all":
        # TensorFlow/RLDS runs in an isolated CPU-only child; the parent then
        # initializes JAX on GPU for the oracle sampler.
        import shutil, subprocess, sys
        staging=out/".cpu-stage";staging.mkdir()
        env=os.environ.copy();env.update({"OMP_NUM_THREADS":"8","MKL_NUM_THREADS":"8","OPENBLAS_NUM_THREADS":"8","CUDA_VISIBLE_DEVICES":""})
        subprocess.run([sys.executable,str(Path(__file__).resolve()),"--mode","primitives","--output-root",str(staging),"--rlds",str(a.rlds),"--raw90",str(a.raw90),"--fps",str(a.fps)],check=True,env=env)
        stage_dirs=list(staging.glob("overnight-prep-*"))
        if len(stage_dirs)!=1:raise RuntimeError(f"unexpected CPU stage outputs: {stage_dirs}")
        result["primitives"]=json.loads((stage_dirs[0]/"results.json").read_text())["primitives"]
        shutil.rmtree(staging)
        if a.reuse_oracle:
            reused=json.loads(a.reuse_oracle.read_text());result["oracle"]=reused["oracle"]
            import shutil
            shutil.copy2(a.reuse_oracle.parent/"oracle-errors.npz",out/"oracle-errors.npz")
        else:result["oracle"]=run_oracle(out,a.heldout,a.memory,a.checkpoint,a.old_run,batch_size=a.batch_size)
    elif a.mode == "oracle":
        result["oracle"]=run_oracle(out,a.heldout,a.memory,a.checkpoint,a.old_run,batch_size=a.batch_size)
    else:
        result["primitives"]=run_primitives(a.rlds,raw90=a.raw90,fps=a.fps)
    (out/"results.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    report=["# Overnight prep", "", "閉ループ評価なし。", ""]
    if "oracle" in result:
        report += ["## Oracle 下書き", "", "| 比較 | t0 / subset | MSE | Δ vs t0=1 (95% episode bootstrap CI) | select MSE | select Δ (95% CI) |", "|---|---:|---:|---:|---:|---:|"]
        for name,grid in result["oracle"]["conditions"].items():
            for t,cell in grid.items():
                d=cell["delta_vs_t0_1"];sd=cell["select_delta_vs_t0_1"]
                report.append(f"| {name} | {t} | {cell['mse']:.4f} | {d[0]:+.4f} [{d[1]:+.4f},{d[2]:+.4f}] | {cell['select_mse']:.4f} | {sd[0]:+.4f} [{sd[1]:+.4f},{sd[2]:+.4f}] |")
        for name,v in result["oracle"]["independent_bad30"]["R_Ap"].items():
            d=v["independent_worst30_delta_mse"]
            report.append(f"| {name}, τ={v['tau']} | independent bad30, t0={v['t0']} | {v['independent_worst30_mse']:.4f} | {d[0]:+.4f} [{d[1]:+.4f},{d[2]:+.4f}] | — | — |")
    if "primitives" in result:
        z=result["primitives"];l=z["segment_length_steps"];s=z["segment_length_seconds"]
        report += ["", "## LIBERO-90 グリッパー区間", "", "| 分析 | 件数・設定 | 長さ / score | 正味変位・代表タスク（動詞は事後集計） |", "|---|---:|---|---|"]
        report.append(f"| RLDS区間・タスク | {z['segments']} intervals / {z['episodes']} episodes / {z['tasks']} tasks | step q10/median/q90 {l['q10']:.0f}/{l['median']:.0f}/{l['q90']:.0f}; sec {s['q10']:.2f}/{s['median']:.2f}/{s['q90']:.2f}; per-task {z['segments_per_task']['min']}/{z['segments_per_task']['median']:.0f}/{z['segments_per_task']['max']} | dx/dy/dz/distance q50 {z['net_displacement']['dx'][2]:.3f}/{z['net_displacement']['dy'][2]:.3f}/{z['net_displacement']['dz'][2]:.3f}/{z['net_displacement']['distance'][2]:.3f} |")
        best_k=max(z["kmeans"],key=lambda k:z["kmeans"][k]["silhouette_sample"])
        for k,v in z["kmeans"].items():
            rep="; ".join(f"c{c['cluster']} {c['top_tasks'][0][0]} ({','.join(w for w,_ in c['top_verbs'][:2])})" for c in v["clusters"]) if k==best_k else "—"
            report.append(f"| k-means | k={k} | silhouette {v['silhouette_sample']:.3f} (sample {v['sample_n']}) | {rep} |")
        dp=z["dp_means"];representative="; ".join(f"{u['id']}:{','.join(t['task'] for t in u['top_tasks'][:2])} ({','.join(v for v,_ in u['top_verbs'][:2])})" for u in dp["units"][:20])
        report.append(f"| M6型 DP-means | {dp['supported_units']} ({dp['raw_candidate_clusters']} candidates) | ε={dp['epsilon_mse']:.4g}, κ=.1 | {representative} |")
        report += ["", "変位のq10/25/50/75/90、クラスタごとの全タスク・動詞・サイズは results.json に保存。", ""]
    (out/"report.md").write_text("\n".join(report))
    print(out)


if __name__=="__main__":main()
