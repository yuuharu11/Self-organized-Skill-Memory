"""CPU-only test of whether similar demonstrated actions have nearby VLM keys.

Reads only selected numeric/string members of the archived npz, never images or
pickle. Episode-held-out query rows; no new VLM inference or policy training.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


def norm(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def select_queries(task, episode, per_task, seed):
    rng = np.random.default_rng(seed)
    chosen = []
    held = []
    for t in np.unique(task):
        episodes = np.unique(episode[task == t])
        n = min(per_task, max(1, len(episodes)//3))
        selected = sorted(rng.choice(episodes, n, replace=False))
        held.extend(selected)
        chosen.extend(int(rng.choice(np.flatnonzero(episode == e))) for e in selected)
    q = np.sort(np.asarray(chosen, int))
    ref = np.flatnonzero(~np.isin(episode, held))
    if set(episode[q]) & set(episode[ref]):
        raise ValueError("query/reference episode leakage")
    return q, ref


def squared_mse(query, reference):
    q = np.asarray(query, np.float64)
    r = np.asarray(reference, np.float64)
    return np.maximum((np.sum(q*q,axis=1)[:,None]+np.sum(r*r,axis=1)[None]-2*q@r.T)/q.shape[1],0)


def quantiles(x):
    x = np.asarray(x,dtype=float); x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0}
    return {"n": len(x), "mean": float(x.mean()),
            **dict(zip(("p10","p25","median","p75","p90"),map(float,np.quantile(x,[.1,.25,.5,.75,.9]))))}


def bootstrap_tasks(values, task, draws=1000):
    task_means = np.asarray([np.mean(values[task == t]) for t in np.unique(task)])
    rng = np.random.default_rng(20261104)
    bootstrap = task_means[rng.integers(0,len(task_means),size=(draws,len(task_means)))].mean(1)
    return {"task_equal_mean":float(task_means.mean()),"task_bootstrap_ci95":list(map(float,np.quantile(bootstrap,[.025,.975])))}


def plot_acceptance(result, output):
    stats=result["scopes"]["different_task"]
    fig,ax=plt.subplots(figsize=(9,4.6))
    bands=(.005,.01,.02)
    for offset,(key,label,color) in enumerate((
        ("near_coverage_{t}","Similar action exists in reference bank","#345B83"),
        ("top1_near_rate_{t}","Found by VLM top-1","#D37432"),
        ("top20_near_rate_{t}","Included in VLM top-20","#4B8A65"))):
        value=np.array([stats[key.format(t=t)]["task_equal_mean"] for t in bands])*100
        ci=np.array([stats[key.format(t=t)]["task_bootstrap_ci95"] for t in bands])*100
        pos=np.arange(3)+(offset-1)*.25
        bars=ax.bar(pos,value,.23,label=label,color=color,yerr=np.stack([value-ci[:,0],ci[:,1]-value]),capsize=3)
        ax.bar_label(bars,labels=[f"{v:.1f}%" for v in value],padding=6,fontsize=10)
    ax.set(xticks=np.arange(3),xticklabels=[str(t) for t in bands],xlabel="Allowed normalized action MSE (lower = stricter)",
           ylabel="Query episodes (%)",ylim=(0,107),title="Similar actions across different tasks: availability versus retrieval")
    ax.legend(frameon=False,loc="upper left",fontsize=9)
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="y",alpha=.18)
    ax.set_axisbelow(True)
    fig.text(.5,.015,"89 tasks; 712 held-out query episodes; 45,560 reference windows; current task excluded.\n"
             "Any action within tolerance counts as correct. Error bars: task-bootstrap 95% intervals.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.1,1,1))
    fig.savefig(output/"action_neighborhood_retrieval.png",dpi=200)
    fig.savefig(output/"action_neighborhood_retrieval.svg")
    plt.close(fig)


def policy_output_check(root, keys, task_rows, episode_rows, actions, saved):
    """Compare actual cached public-policy outputs and targets on the same rows."""
    with np.load(root/"prepared_eval_new.npz",allow_pickle=False) as z:
        row_ids=z["row_ids"].astype(int)
        target=z["target"][:,:,:7].astype(np.float32)
        if not np.allclose(target,actions[row_ids],atol=2e-6):
            raise ValueError("policy target cache does not align with source actions")
        if not np.array_equal(z["episode_id"].astype(str),episode_rows[row_ids]):
            raise ValueError("policy episode cache mismatch")
    with np.load(root/"predictions.npz",allow_pickle=False) as z:
        predictions=z["new__public_n10"][:,:,:,:7].astype(np.float32)
    if predictions.shape != (3,len(row_ids),10,7):
        raise ValueError("unexpected public-policy output shape")
    task,episode=task_rows[row_ids],episode_rows[row_ids]
    q,r=select_queries(task,episode,16,20261105)
    k=norm(keys[row_ids])
    center=k[r].mean(0,dtype=np.float64).astype(np.float32)
    cos=norm(k[q]-center)@norm(k[r]-center).T
    targets={"demonstrated_actions":target,"public_policy_mean_of_3":predictions.mean(0),
             **{f"public_policy_noise_{i}":predictions[i] for i in range(3)}}
    output={"source":str(root),"query_episodes":len(q),"reference_episodes":len(np.unique(episode[r])),
            "reference_rows":len(r),"tasks":len(np.unique(task[q])),"model":"public SkillNet n10; 3 cached noise draws",
            "cache_subset":"pre-existing gripper-big 1500-row/449-episode diagnostic; separate from 89-task primary analysis",
            "targets":{}}
    for name,values in targets.items():
        flat=values.reshape(len(values),-1)
        dist=squared_mse(flat[q],flat[r])
        out=[]
        for ix,query in enumerate(q):
            candidates=np.flatnonzero(task[r]!=task[query])
            dd,cc=dist[ix,candidates],cos[ix,candidates]
            best=int(np.argmin(dd));order=np.argsort(-cc,kind="stable")[:20]
            out.append({"nearest_output_mse":float(dd[best]),"nearest_output_vlm_rank":int(np.sum(cc>cc[best])+1),
                "nearest_output_vlm_percentile":float(np.mean(cc<=cc[best])),
                "vlm_top1_output_mse":float(dd[order[0]]),
                "rho_input_cos_vs_output_similarity":float(spearmanr(cc,-dd).statistic),
                **{f"has_near_{t}":bool(np.any(dd<=t)) for t in (.005,.01,.02,.04)},
                **{f"has_near_top{k}_{t}":bool(np.any(dd[order[:k]]<=t)) for t in (.005,.01,.02,.04) for k in (1,5,20)}})
        for key in out[0]:
            saved[f"policy_check|{name}|{key}"]=np.asarray([v[key] for v in out])
        output["targets"][name]={key:quantiles([v[key] for v in out]) for key in out[0]}
    saved["policy_check|query_rows"]=row_ids[q]
    saved["policy_check|query_task"]=task[q]
    saved["policy_check|reference_rows"]=row_ids[r]
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--episodes-per-task",type=int,default=8)
    p.add_argument("--seed",type=int,default=20261104)
    p.add_argument("--policy-root",type=Path)
    args=p.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU-only run requires CUDA_VISIBLE_DEVICES=''")
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    with np.load(args.cache,allow_pickle=False) as z:
        keys=z["key"].astype(np.float32)
        actions=z["actions"][:,:,:7].astype(np.float32)
        state=z["state"][:,:8].astype(np.float32)
        task=z["source_file"].astype(str)
        episode=z["episode_id"].astype(str)
        instruction=z["instruction"].astype(str)
        frame=z["frame"].astype(int)
        checkpoint=str(z["key_checkpoint"])
        action_normalization=str(z["normalization_sha256"])
    hashes={k:hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()
            for k,v in (("keys",keys),("active7_actions",actions),("task",task),("episode",episode),("frame",frame))}
    qidx,ridx=select_queries(task,episode,args.episodes_per_task,args.seed)
    qtask=task[qidx]; rtask=task[ridx]
    qi=instruction[qidx]; ri=instruction[ridx]
    qkeys,rkeys=norm(keys[qidx]),norm(keys[ridx])
    mean=rkeys.mean(0,dtype=np.float64).astype(np.float32)
    qcenter,rcenter=norm(qkeys-mean),norm(rkeys-mean)
    qa,ra=actions[qidx].reshape(len(qidx),-1),actions[ridx].reshape(len(ridx),-1)
    qa6,ra6=actions[qidx,:,:6].reshape(len(qidx),-1),actions[ridx,:,:6].reshape(len(ridx),-1)
    # Fixed random-pair controls from the reference bank, shared across scopes.
    rng=np.random.default_rng(args.seed+13)
    bands=(.005,.01,.02,.04)
    scopes=("cross_episode","different_task","different_instruction","same_task")
    records={s:[] for s in scopes}
    positives={s:[] for s in scopes}
    random_pairs={s:[] for s in scopes}
    saved={"query_row":qidx,"query_task":qtask,"query_episode":episode[qidx],"query_frame":frame[qidx],
           "reference_rows":ridx,"center_mean":mean}
    print(f"loaded {len(keys)} keys; queries {len(qidx)} across {len(np.unique(qtask))} tasks; reference {len(ridx)}",flush=True)
    for lo in range(0,len(qidx),16):
        hi=min(lo+16,len(qidx))
        d=squared_mse(qa[lo:hi],ra)
        d6=squared_mse(qa6[lo:hi],ra6)
        cosine=qcenter[lo:hi]@rcenter.T
        raw_cos=qkeys[lo:hi]@rkeys.T
        for j,ix in enumerate(range(lo,hi)):
            masks={"cross_episode":np.ones(len(ridx),bool),"different_task":rtask != qtask[ix],
                   "different_instruction":ri != qi[ix],"same_task":rtask == qtask[ix]}
            for scope,mask in masks.items():
                candidates=np.flatnonzero(mask)
                if len(candidates)<100:
                    raise ValueError(f"too few reference candidates in {scope}")
                dd=d[j,candidates]; cc=cosine[j,candidates]
                action_order=np.argsort(dd,kind="stable")[:32]
                vlm_order=np.argsort(-cc,kind="stable")[:100]
                nearest=candidates[action_order[0]]
                first=candidates[vlm_order[0]]
                random_local=rng.choice(len(candidates),min(256,len(candidates)),replace=False)
                random_index=candidates[random_local]
                rho=float(spearmanr(cc[random_local],-dd[random_local]).statistic)
                record={"query_index":ix,"candidates":len(candidates),"nearest_action_row":int(ridx[nearest]),
                    "nearest_action_mse":float(dd[action_order[0]]),
                    "nearest_motion6_mse":float(d6[j,candidates].min()),
                    "vlm_top1_row":int(ridx[first]),"vlm_top1_action_mse":float(d[j,first]),
                    "vlm_top5_blend_action_mse":float(np.mean((ra[candidates[vlm_order[:5]]].mean(0)-qa[ix])**2)),
                    "random_action_mse":float(dd[random_local].mean()),
                    "random_action_mean_prediction_mse":float(np.mean((ra[candidates].mean(0)-qa[ix])**2)),
                    "nearest_action_cosine":float(cc[action_order[0]]),
                    "nearest_action_raw_cosine":float(raw_cos[j,nearest]),
                    "nearest_action_rank_by_vlm":int(np.sum(cc>cc[action_order[0]])+1),
                    "nearest_action_vlm_percentile":float(np.mean(cc<=cc[action_order[0]])),
                    "random_pair_cosine_vs_action_similarity_spearman":rho,
                    "top1_cosine":float(cc[vlm_order[0]]),
                    "query_motion_rms":float(np.sqrt(np.mean(qa6[ix]**2))),
                    "action_nearby_32_input_cosine_median":float(np.median(cc[action_order])),
                    "action_nearby_32_fraction_below_random_input_median":float(np.mean(cc[action_order]<np.median(cc[random_local]))),
                    "action_nearby_32_output_mse_median":float(np.median(dd[action_order]))}
                for band in bands:
                    near=dd<=band
                    tag=str(band)
                    record[f"near_count_{tag}"]=int(near.sum())
                    record[f"near_distinct_tasks_{tag}"]=int(len(np.unique(rtask[candidates[near]])))
                    for k in (1,5,20,100):
                        record[f"has_near_in_top{k}_{tag}"]=bool(np.any(dd[vlm_order[:k]]<=band))
                    if band==.01 and near.any():
                        near_ix=candidates[np.flatnonzero(near)]
                        good=int(near_ix[np.argmax(cosine[j,near_ix])])
                        selected=near_ix if len(near_ix)<=32 else rng.choice(near_ix,32,replace=False)
                        positives[scope].extend((ix,int(ridx[v]),float(d[j,v]),float(cosine[j,v])) for v in selected)
                        record["nearest_action_within_band_input_cosine_min"]=float(np.min(cosine[j,near_ix]))
                        record["nearest_action_within_band_input_cosine_max"]=float(cosine[j,good])
                random_pairs[scope].extend((ix,int(ridx[v]),float(d[j,v]),float(cosine[j,v])) for v in random_index[:32])
                records[scope].append(record)
        if lo%160==0:
            print("processed",hi,"/",len(qidx),"elapsed",round(time.monotonic()-started,1),flush=True)
    result={"protocol":{"model":checkpoint,"action_normalization_sha256":action_normalization,
        "feature":"2048-D mean-pooled final VLM image+instruction prefix; raw per-observation keys, before unit averaging",
        "raw_tokens_available":False,"seed":args.seed,"queries":len(qidx),"reference_rows":len(ridx),
        "tasks":len(np.unique(qtask)),"query_episodes":len(np.unique(episode[qidx])),
        "reference_episodes":len(np.unique(episode[ridx])),"queries_per_episode":1,
        "target":"recorded demonstrated actions, normalized active7, 10 steps; no policy-generated predictions",
        "tolerances_mse":bands,"reference_mean_centering":"fit reference keys only",
        "different_task":"different source task file; conservative different initial-state groups",
        "same_task":"different episode; source initial-state IDs unavailable within task",
        "model_training":False,"vlm_inference":False,"cuda":False,"cpu_threads":1,
        "ci_scope":"descriptive task bootstrap conditional on cached model and fixed reference bank; overlapping bank windows are not independent samples"},
        "input_member_hashes":hashes,"scopes":{}}
    for scope,rr in records.items():
        metrics=sorted(set().union(*(r.keys() for r in rr)) - {"query_index","nearest_action_row","vlm_top1_row"})
        stats={k:quantiles([r.get(k,np.nan) for r in rr]) for k in metrics}
        for band in bands:
            tag=str(band)
            covered=np.array([r[f"near_count_{tag}"]>0 for r in rr])
            stats[f"near_coverage_{tag}"]=bootstrap_tasks(covered,qtask)
            for k in (1,5,20,100):
                hit=np.array([r[f"has_near_in_top{k}_{tag}"] for r in rr])
                stats[f"top{k}_near_rate_{tag}"]=bootstrap_tasks(hit,qtask)
                stats[f"top{k}_recall_given_near_exists_{tag}"]=float(hit[covered].mean()) if covered.any() else None
        stats["vlm_top1_minus_random_mse"]=bootstrap_tasks(np.array([r["vlm_top1_action_mse"]-r["random_action_mse"] for r in rr]),qtask)
        stats["vlm_top1_minus_oracle_mse"]=bootstrap_tasks(np.array([r["vlm_top1_action_mse"]-r["nearest_action_mse"] for r in rr]),qtask)
        result["scopes"][scope]=stats
        for k in metrics:
            saved[f"{scope}|{k}"]=np.asarray([r.get(k,np.nan) for r in rr])
        saved[f"{scope}|nearest_action_row"]=np.asarray([r["nearest_action_row"] for r in rr])
        saved[f"{scope}|vlm_top1_row"]=np.asarray([r["vlm_top1_row"] for r in rr])
        saved[f"{scope}|positive_pairs"]=np.asarray(positives[scope])
        saved[f"{scope}|random_pairs"]=np.asarray(random_pairs[scope])
    if args.policy_root is not None:
        result["policy_output_crosscheck"]=policy_output_check(args.policy_root,keys,task,episode,actions,saved)
    result["elapsed_seconds"]=time.monotonic()-started
    np.savez_compressed(args.output/"diagnostic_arrays.npz",**saved)
    (args.output/"summary.json").write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False))
    plt.rcParams.update({"font.size":11,"font.family":"DejaVu Sans","svg.fonttype":"none"})
    fig,axes=plt.subplots(1,2,figsize=(11,4.2))
    for scope,color,label in (("same_task","#345B83","Same task, different episode"),
                              ("different_task","#D37432","Different task"),
                              ("different_instruction","#4B8A65","Different instruction")):
        v=np.sort([r["nearest_action_mse"] for r in records[scope]])
        axes[0].plot(v,np.arange(1,len(v)+1)/len(v),label=label,color=color,linewidth=2)
        v=np.sort([r["nearest_action_vlm_percentile"] for r in records[scope]])
        axes[1].plot(v,np.arange(1,len(v)+1)/len(v),label=label,color=color,linewidth=2)
    axes[0].set(xlabel="Closest demonstrated-action MSE in reference bank",ylabel="Fraction of query episodes",title="Does a similar output exist?",xlim=(0,.1),ylim=(0,1))
    axes[1].set(xlabel="Input-cosine percentile of action-nearest reference",ylabel="Fraction of query episodes",title="Are similar outputs also nearby in VLM space?",xlim=(0,1),ylim=(0,1))
    for ax in axes:
        ax.legend(frameon=False,fontsize=8)
        ax.grid(alpha=.2)
        ax.spines[["top","right"]].set_visible(False)
    fig.suptitle("Per-observation VLM features versus demonstrated-action similarity",fontsize=14)
    fig.text(.5,.015,"One window per held-out episode; fixed reference bank. Action-nearest references use future targets for diagnosis only.\n"
             "Action similarity measures motor commands, not semantic skill or achieved effect. Left panel displays MSE up to 0.1.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.1,1,.92))
    fig.savefig(args.output/"vlm_action_geometry.png",dpi=200)
    fig.savefig(args.output/"vlm_action_geometry.svg")
    plt.close(fig)
    plot_acceptance(result,args.output)
    print("complete",args.output,"elapsed",round(time.monotonic()-started,1),flush=True)


if __name__=="__main__":
    main()
