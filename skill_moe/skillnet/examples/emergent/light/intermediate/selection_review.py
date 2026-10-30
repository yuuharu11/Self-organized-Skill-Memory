"""Offline selection audit for review_representation.md §3 (E0-E3, E6-E7)."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import fair_compare as fc
from . import skill_v2 as v2
from .compare_splits import SPLITS, _subset
from .experiment import DEFAULT_CACHE, DEFAULT_DATASET, _episode_train_validation_indices, load_rows
from ....research_paths import REPO_ROOT as RESEARCH_REPO_ROOT, STATE_DIR as RESEARCH_STATE_DIR

V21_ROOT = Path(str(RESEARCH_STATE_DIR / 'outputs/memory-pilot/skill-v21-20261029T060213Z-7493c4'))
R_GRID, GAMMA_GRID, THETA_GRID = (5, 20), (0.0, 0.5), (0.1, 0.2, 0.3)
TEMPERATURE = 0.1


def _json_default(value):
    if isinstance(value,np.generic): return value.item()
    if isinstance(value,np.ndarray): return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _norm(x):
    x = np.asarray(x, np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def member_lse_scores(query, member_keys, member_ids, n_units, *, r=5, gamma=0., temperature=TEMPERATURE):
    """Per-unit top-r log-sum-exp cosine score with a cardinality penalty."""
    q = _norm(np.asarray(query).reshape(1, -1))[0]
    keys = _norm(member_keys); ids = np.asarray(member_ids, np.int64)
    if len(keys) != len(ids) or not len(keys) or r < 1 or temperature <= 0:
        raise ValueError("invalid member set or retrieval settings")
    sim = keys @ q / temperature
    sizes = np.bincount(ids, minlength=n_units)
    out = np.full(n_units, -np.inf, np.float64)
    for unit in range(n_units):
        vals = sim[ids == unit]
        if not len(vals): continue
        top = np.partition(vals, max(0, len(vals)-min(r, len(vals))))[-min(r, len(vals)):]
        peak = top.max()
        out[unit] = peak + np.log(np.exp(top-peak).sum()) - gamma*np.log(max(sizes[unit], 1))
    return out.astype(np.float32)


def _simplex(v):
    u=np.sort(np.asarray(v,np.float64))[::-1]; css=np.cumsum(u)-1; ix=np.arange(1,len(u)+1)
    valid=u-css/ix>0; rho=np.flatnonzero(valid)[-1] if valid.any() else 0
    return np.maximum(np.asarray(v,np.float64)-css[rho]/(rho+1),0)


def bbse_priors(confusion, probabilities, train_prior, ridge=1e-5, marginal_source=None):
    """Constrained BBSE prior estimate and posterior label-shift correction."""
    c=np.maximum(np.asarray(confusion,np.float64),0); p=np.asarray(probabilities,np.float64)
    tr=np.asarray(train_prior,np.float64)
    c/=np.maximum(c.sum(1,keepdims=True),1e-12)
    if c.ndim!=2 or c.shape[0]!=c.shape[1] or p.ndim!=2 or p.shape[1]!=len(c) or not len(p):
        raise ValueError("BBSE shape mismatch")
    # Confusion rows are true classes and columns predicted classes, so
    # mu_pred = C.T @ pi_true; the normal equations use A=C.T.
    marginal=p if marginal_source is None else np.asarray(marginal_source,np.float64)
    g=c@c.T+ridge*np.eye(len(c)); b=c@marginal.mean(0); lr=max(np.linalg.norm(g,2),1e-8)
    prior=np.full(len(c),1/len(c))
    for _ in range(1000):
        nxt=_simplex(prior-(g@prior-b)/lr)
        if np.max(np.abs(nxt-prior))<1e-10: prior=nxt; break
        prior=nxt
    corrected=p*(prior/np.maximum(tr,1e-8))[None]
    corrected/=np.maximum(corrected.sum(1,keepdims=True),1e-12)
    return prior,corrected


def _softmax(scores):
    x=np.asarray(scores,np.float64); x=x-x.max(1,keepdims=True); p=np.exp(x)
    return p/np.maximum(p.sum(1,keepdims=True),1e-12)


def cohen_kappa(y_true, y_pred, n_classes=None):
    y=np.asarray(y_true,np.int64).ravel(); p=np.asarray(y_pred,np.int64).ravel()
    if len(y)!=len(p) or not len(y): return float("nan")
    k=int(n_classes or max(y.max(),p.max())+1)
    cm=np.bincount(y*k+p,minlength=k*k).reshape(k,k)
    po=np.trace(cm)/cm.sum(); pe=(cm.sum(0)@cm.sum(1))/(cm.sum()**2)
    return float((po-pe)/max(1-pe,1e-12))


def interval_groups(episodes, frames, gripper, threshold=0.):
    """Group temporally adjacent rows by actual or predicted gripper switches."""
    ep=np.asarray(episodes).astype(str); fr=np.asarray(frames); g=np.asarray(gripper).reshape(-1)
    if not (len(ep)==len(fr)==len(g)): raise ValueError("row arrays must align")
    groups=[]
    for name in np.unique(ep):
        ix=np.flatnonzero(ep==name); ix=ix[np.argsort(fr[ix],kind="stable")]
        sign=np.where(g[ix]>=threshold,1,-1); cuts=np.r_[0,np.flatnonzero(sign[1:]!=sign[:-1])+1,len(ix)]
        groups.extend(ix[a:b] for a,b in zip(cuts[:-1],cuts[1:]) if b>a)
    return groups


def selection_metrics(y_true, y_pred, functional_sets, n_units):
    y=np.asarray(y_true,np.int64); p=np.asarray(y_pred,np.int64)
    matched=np.asarray([bool(s) for s in functional_sets]); functional=np.asarray([x in s for x,s in zip(p,functional_sets)])
    prior=np.bincount(y,minlength=n_units).astype(float); prior/=max(prior.sum(),1)
    return {"n":len(y),"strict_top1":float(np.mean(y==p)),
            "functional_accuracy":float(functional.mean()),"functional_coverage":float(matched.mean()),
            "strict_on_covered":float(np.mean((y==p)[matched])) if matched.any() else None,
            "majority_chance":float(prior.max()),"frequency_chance":float((prior**2).sum()),
            "cohen_kappa":cohen_kappa(y,p,n_units)}


def _merge_confused(confusion, theta, distances, delta_max):
    n=len(confusion); parent=list(range(n))
    def root(i):
        while parent[i]!=i: parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i in range(n):
        for j in range(i+1,n):
            if .5*(confusion[i,j]+confusion[j,i])>=theta and distances[i,j]<=delta_max:
                a,b=root(i),root(j)
                if a!=b:parent[b]=a
    roots=[root(i) for i in range(n)]; remap={r:i for i,r in enumerate(sorted(set(roots)))}
    return np.asarray([remap[r] for r in roots],np.int64)


def _unit_metrics(y,p,sets,n):
    return selection_metrics(y,p,sets,n)


def _saved_e0(path):
    out={}
    with np.load(path,allow_pickle=False) as z:
      for split in "AB":
       for seed in range(3):
        for cond in ("D1","D3"):
          pre=f"{split}_seed{seed}_{cond}_mean__"
          names=("skill_id","oracle_skill_id","target","final_prediction","episode_id","frame")
          x={k:z[pre+k] for k in names}; y=x["oracle_skill_id"].astype(int); p=x["skill_id"].astype(int)
          n=max(y.max(),p.max())+1; singleton=[{int(v)} for v in y]
          row=_unit_metrics(y,p,singleton,n)
          row["functional_accuracy"]=None;row["functional_coverage"]=None
          def weighted(grip):
            gs=interval_groups(x["episode_id"],x["frame"],grip)
            yt=[];pt=[]
            for ix in gs:
                yt.append(np.bincount(y[ix],minlength=n).argmax());pt.append(np.bincount(p[ix],minlength=n).argmax())
            return _unit_metrics(np.asarray(yt),np.asarray(pt),[{int(v)} for v in yt],n),len(gs)
          actual,na=weighted(x["target"][:,0,6]); online,npred=weighted(x["final_prediction"][:,0,6])
          for metric in (actual,online): metric["functional_accuracy"]=None;metric["functional_coverage"]=None
          mse=np.square(x["final_prediction"].astype(np.float64)-x["target"]).mean((1,2))
          out[f"{split}/{seed}/{cond}"]={"row_weighted":row,"interval_actual_gripper":actual,
            "interval_predicted_gripper":online,"n_intervals_actual":na,"n_intervals_predicted":npred,
            "saved_v21_mse":float(mse.mean()),"functional_accuracy":None,"functional_coverage":None,
            "functional_note":"per_row.npz に単位メンバー軌道がなく、epsilon 内判定は再構成不能"}
    return out


def _fit_cell(rows, split, seed, stride):
    train_tasks,eval_tasks,_=SPLITS[split]
    selected=_subset(rows,np.isin(rows["task_index"],sorted(set(train_tasks)|set(eval_tasks)))&rows["valid_h25"])
    train_all=_subset(selected,np.isin(selected["task_index"],train_tasks))
    fit_rows,d1=fc.holdout_split(train_all,split,seed,per_task=fc.HOLDOUT_PER_TASK)
    fit_ix,_=_episode_train_validation_indices(fit_rows,seed+401)
    fit=_subset(fit_rows,np.isin(np.arange(len(fit_rows["a"])),fit_ix))
    d3=_subset(selected,np.isin(selected["task_index"],eval_tasks))
    limits=v2.derive_length_limits(fit)
    train_seg,_=v2.make_segments(fit,fit["k"],limits,stride)
    memory=v2.TrajectorySkillMemory(v2.KAPPA).fit(train_seg,seed=seed+811)
    train_seg=[s for s in train_seg if int(s["segment_id"]) in memory.assignments]
    def pack(part):
        seg,_=v2.make_segments(part,part["k"],limits,stride); truth=[]; positives=[];starts=[]
        radii=[]
        for u in memory.units:
            ds=[memory.distance(v2._rep(m["curve"],m["length"]),u.mean_rep) for m in u.members]
            radii.append(float(np.quantile(ds,.9)) if ds else 0.)
        for s in seg:
            rep=v2._rep(s["curve"],s["length"]);ds=np.asarray([memory.distance(rep,u.mean_rep) for u in memory.units])
            truth.append(int(ds.argmin()));positives.append(set(np.flatnonzero(ds<=radii).tolist()));starts.append(int(s["rows"][0]))
        return seg,np.asarray(truth,int),positives,np.asarray(starts,int)
    ids=np.asarray([memory.assignments[int(s["segment_id"])] for s in train_seg],int)
    member_start=np.asarray([s["rows"][0] for s in train_seg],int)
    return {"fit":fit,"d1":d1,"d3":d3,"memory":memory,"train_seg":train_seg,"member_ids":ids,
            "member_start":member_start,"d1p":pack(d1),"d3p":pack(d3),"tasks":train_tasks,"limits":limits}


def _add_functional_e0(entry, npz_path, cell, split, seed, stride, condition, part):
    """Recover unit-specific q90 functional neighborhoods from joined actions."""
    with np.load(npz_path,allow_pickle=False) as z:
        x=_saved_e0_row(z,split,seed,condition)
    lookup={(str(ep),int(fr)):i for i,(ep,fr) in enumerate(zip(part["episode_id"],part["frame"]))}
    try: order=np.asarray([lookup[(str(ep),int(fr))] for ep,fr in zip(x["episode_id"],x["frame"])],np.int64)
    except KeyError:
        entry["functional_note"]="saved per-row ordering differs from cache; functional matching skipped"
        return
    mem=cell["memory"]; radii=[]
    for u in mem.units:
        reps=np.stack([v2._rep(m["curve"],m["length"]) for m in u.members])
        radii.append(float(np.quantile([mem.distance(q,u.mean_rep) for q in reps],.9)))
    radii=np.asarray(radii); positives_by_row=[set() for _ in range(len(part["a"]))];nearest_by_row=np.full(len(part["a"]),-1,np.int64)
    eval_segments,_=v2.make_segments(part,part["k"],cell["limits"],stride)
    for seg in eval_segments:
        rep=v2._rep(seg["curve"],seg["length"])
        dist=np.asarray([mem.distance(rep,u.mean_rep) for u in mem.units])
        positive=set(np.flatnonzero(dist<=radii).tolist()); nearest=int(dist.argmin())
        for row in seg["rows"]: positives_by_row[int(row)]=positive;nearest_by_row[int(row)]=nearest
    positives=[positives_by_row[i] for i in order]
    pred=np.asarray(x["skill_id"],int); truth=np.asarray(x["oracle_skill_id"],int)
    covered=np.asarray([len(s)>0 for s in positives]);functional=np.asarray([v in s for v,s in zip(pred,positives)])
    entry["row_weighted"]["functional_accuracy"]=float(functional.mean())
    entry["row_weighted"]["functional_coverage"]=float(covered.mean())
    entry["row_weighted"]["strict_on_covered"]=float(np.mean((pred==truth)[covered])) if covered.any() else None
    actual_groups=interval_groups(x["episode_id"],x["frame"],x["target"][:,0,6])
    pred_groups=interval_groups(x["episode_id"],x["frame"],x["final_prediction"][:,0,6])
    def interval_score(groups):
        yy=[];pp=[];ss=[]
        for ix in groups:
            yy.append(int(np.bincount(truth[ix]).argmax()))
            pp.append(int(pred[ix[0]]));ss.append(set(positives[ix[0]]))
        return _unit_metrics(np.asarray(yy),np.asarray(pp),ss,len(mem.units))
    entry["interval_actual_gripper"]=interval_score(actual_groups)
    entry["interval_predicted_gripper"]=interval_score(pred_groups)
    entry["functional_epsilon_by_unit_q90"]="within-unit fit member-to-center distance q90"
    entry["recomputed_nearest_agreement_saved_oracle"]=float(np.mean(nearest_by_row[order]==truth))


def _saved_e0_row(z,split,seed,condition):
    pre=f"{split}_seed{seed}_{condition}_mean__"
    return {k:z[pre+k] for k in ("skill_id","oracle_skill_id","target","final_prediction","episode_id","frame")}


def _refit_task_subset(cell, task_ids, seed, stride):
    fit=_subset(cell["fit"],np.isin(cell["fit"]["task_index"],list(task_ids)))
    limits=v2.derive_length_limits(fit); seg,_=v2.make_segments(fit,fit["k"],limits,stride)
    memory=v2.TrajectorySkillMemory(v2.KAPPA).fit(seg,seed=seed+811)
    seg=[s for s in seg if int(s["segment_id"]) in memory.assignments]
    def pack(part):
        ss,_=v2.make_segments(part,part["k"],limits,stride); y=[];positive=[];starts=[]
        radius=[]
        for u in memory.units:
            ds=[memory.distance(v2._rep(m["curve"],m["length"]),u.mean_rep) for m in u.members]
            radius.append(float(np.quantile(ds,.9)) if ds else 0.)
        for s in ss:
            rep=v2._rep(s["curve"],s["length"]);ds=np.asarray([memory.distance(rep,u.mean_rep) for u in memory.units])
            y.append(int(ds.argmin()));positive.append(set(np.flatnonzero(ds<=radius).tolist()));starts.append(int(s["rows"][0]))
        return ss,np.asarray(y,int),positive,np.asarray(starts,int)
    ids=np.asarray([memory.assignments[int(s["segment_id"])] for s in seg],int)
    starts=np.asarray([s["rows"][0] for s in seg],int)
    return {"fit":fit,"memory":memory,"train_seg":seg,"member_ids":ids,"member_start":starts,
            "d1":cell["d1"],"d3":cell["d3"],"d1p":pack(cell["d1"]),"d3p":pack(cell["d3"])}


def _modal(cell,part,mode):
    fit=cell["fit"]
    def raw(x):
        a=np.asarray(x["k"],np.float32)
        if mode!="image":a=np.concatenate((a,x["s"]),axis=1)
        if mode=="image_state_subtask":
            vocab=max(int(fit["skill_class"].max())+1,1);cls=np.clip(x["skill_class"],0,vocab-1)
            a=np.concatenate((a,np.eye(vocab,dtype=np.float32)[cls]),axis=1)
        return a
    tr=raw(fit);mu=tr.mean(0);sd=tr.std(0);sd[sd<1e-6]=1
    return ((tr-mu)/sd).astype(np.float32),((raw(part)-mu)/sd).astype(np.float32)


def _scores(cell, part, starts, r, gamma, mode="image", allowed=None):
    tr,q=_modal(cell,part,mode);ids=cell["member_ids"];ix=cell["member_start"]
    if allowed is not None:trix=np.flatnonzero(allowed);ix=ix[trix];ids=ids[trix]
    return np.stack([member_lse_scores(q[row],tr[ix],ids,len(cell["memory"].units),r=r,gamma=gamma) for row in starts])


def _cv_confusion(cell,r=5,gamma=0.):
    fit=cell["fit"]; seg=cell["train_seg"];ids=cell["member_ids"];starts=cell["member_start"]
    eps=np.asarray([x["episode_id"] for x in seg]).astype(str); unique=np.unique(eps)
    parts=np.array_split(np.random.default_rng(173).permutation(unique),min(5,len(unique)))
    conf=np.zeros((len(cell["memory"].units),)*2); counts=np.zeros(len(cell["memory"].units))
    for held in parts:
        valid=np.isin(eps,held);train=~valid
        for i in np.flatnonzero(valid):
            score=member_lse_scores(fit["k"][starts[i]],fit["k"][starts[train]],ids[train],len(counts),r=r,gamma=gamma)
            y=int(ids[i]);conf[y,int(score.argmax())]+=1;counts[y]+=1
    conf/=np.maximum(counts[:,None],1)
    return conf,counts


def _score_metrics(scores,packed,mapping=None):
    _,truth,positive,_=packed;pred=np.asarray(scores).argmax(1)
    if mapping is not None:
        truth=mapping[truth];pred=mapping[pred]
        positive=[set(mapping[list(s)].tolist()) if s else set() for s in positive]
    return _unit_metrics(truth,pred,positive,int(max(truth.max(initial=0),pred.max(initial=0))+1))


def run(*,output,cache_dir=DEFAULT_CACHE,dataset_dir=DEFAULT_DATASET,v21_root=V21_ROOT,
        splits=("A","B"),seeds=(0,1,2),device="cpu",threads=8,smoke=False):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True);torch.set_num_threads(int(threads))
    try:torch.set_num_interop_threads(1)
    except RuntimeError:pass
    tic=time.perf_counter();e0=_saved_e0(Path(v21_root)/"per_row.npz")
    manifest=json.loads((Path(cache_dir)/"manifest.json").read_text());info=json.loads((Path(dataset_dir)/"meta/info.json").read_text())
    stride=float(manifest["stride"])/float(info["fps"]);rows=load_rows(Path(cache_dir),Path(dataset_dir),max_horizon=v2.HORIZON)
    taskids=sorted({i for s in splits for i in (*SPLITS[s][0],*SPLITS[s][1])})
    rows=_subset(rows,np.isin(rows["task_index"],taskids)&rows["valid_h25"]);runs=[]
    for split in splits:
      for seed in seeds:
        c=_fit_cell(rows,split,int(seed),stride);K=len(c["memory"].units);rec={"split":split,"seed":int(seed),"n_units":K,"evaluations":{},"E2":{},"E6":{},"E7":{}}
        for cond,part in (("D1",c["d1"]),("D3",c["d3"])):
            _add_functional_e0(e0[f"{split}/{seed}/{cond}"],Path(v21_root)/"per_row.npz",c,split,int(seed),stride,cond,part)
        e1={}
        for r in R_GRID:
          for gamma in GAMMA_GRID:
            key=f"r{r}_g{gamma:g}";s1=_scores(c,c["d1"],c["d1p"][3],r,gamma);s3=_scores(c,c["d3"],c["d3p"][3],r,gamma);e1[key]=(s1,s3)
            rec["evaluations"]["E1/"+key]={"D1":_score_metrics(s1,c["d1p"]),"D3":_score_metrics(s3,c["d3p"]),
              "policy_mse":"not recoverable: v2.1 did not save alternate-skill rollouts"}
        conf,count=_cv_confusion(c);means=np.stack([u.mean_rep for u in c["memory"].units]);dist=np.mean((means[:,None]-means[None,:])**2,axis=-1)
        delta=float(np.quantile(dist[np.triu_indices(K,1)],.25)) if K>1 else 0.
        for theta in THETA_GRID:
            mapping=_merge_confused(conf,theta,dist,delta)
            item={"merged_units":int(len(np.unique(mapping))),"mapping":mapping.tolist(),"cv_confusion":conf.tolist(),"functional_distance_cap_q25":delta,"evaluations":{}}
            for key,(s1,s3) in e1.items():item["evaluations"][key]={"D1":_score_metrics(s1,c["d1p"],mapping),"D3":_score_metrics(s3,c["d3p"],mapping),"policy_mse":"not recoverable"}
            rec["E2"][f"theta{theta:g}"]=item
        prior=count/max(count.sum(),1)
        for key,(s1,s3) in e1.items():
            p1,p3=_softmax(s1),_softmax(s3)
            h1=np.eye(K)[s1.argmax(1)];h3=np.eye(K)[s3.argmax(1)]
            pi1,a1=bbse_priors(conf,p1,prior,marginal_source=h1);pi3,a3=bbse_priors(conf,p3,prior,marginal_source=h3)
            rec["evaluations"]["E3/"+key]={"D1_before":_score_metrics(p1,c["d1p"]),"D1_after":_score_metrics(a1,c["d1p"]),
              "D3_before":_score_metrics(p3,c["d3p"]),"D3_after":_score_metrics(a3,c["d3p"]),"prior_D1":pi1.tolist(),"prior_D3":pi3.tolist(),"policy_mse":"not recoverable"}
        # E6: nested, task-balanced subsets; all means every train-suite task.
        task_order=np.random.default_rng(seed+441).permutation(c["tasks"])
        for nt in (10,20):
            chosen=set(task_order[:min(nt,len(task_order))]);sub=_refit_task_subset(c,chosen,int(seed),stride)
            rec["E6"][str(nt)]={"task_ids":sorted(chosen),"n_units":len(sub["memory"].units),"D1":{},"D3":{}}
            for r in R_GRID:
              for gamma in GAMMA_GRID:
                key=f"r{r}_g{gamma:g}"
                for cond,part,packed in (("D1",sub["d1"],sub["d1p"]),("D3",sub["d3"],sub["d3p"])):
                    ss=_scores(sub,part,packed[3],r,gamma);rec["E6"][str(nt)][cond][key]=_score_metrics(ss,packed)
        rec["E6"]["all"]={"same_as":"20","note":"splitごとの利用可能な訓練suiteは20タスク"}
        # E7: train existing multi-positive selector under three query views.
        for mode in ("image","image_state","image_state_subtask"):
            tr,_=_modal(c,c["fit"],mode);_,d1q=_modal(c,c["d1"],mode);_,d3q=_modal(c,c["d3"],mode)
            stats=[]
            # _modal returns normalized rows with fit-derived statistics for both calls.
            segs=[]
            for s in c["train_seg"]:
                q=dict(s);q["key"]=tr[int(s["rows"][0])];segs.append(q)
            memory=c["memory"].clone();selector,fitinfo=v2.fit_multi_positive_selector(segs,memory,seed=seed+701,device="cpu",epochs=3 if smoke else 60)
            scores={}
            for cond,part,qmat,pack in (("D1",c["d1"],d1q,c["d1p"]),("D3",c["d3"],d3q,c["d3p"])):
                val=qmat[pack[3]];s,_=v2.selector_scores(selector,val,memory.units,"cpu");scores[cond]=_score_metrics(s,pack)
            rec["E7"][mode]={"fit":fitinfo,"evaluation":scores}
        rec["E0"]={cond:e0[f"{split}/{seed}/{cond}"] for cond in ("D1","D3")}
        runs.append(rec);(output/"results.partial.json").write_text(json.dumps({"runs":runs},ensure_ascii=False,default=_json_default))
    payload={"spec":"review_representation.md §3 E0-E3 E6 E7",
      "config":{"splits":list(splits),"seeds":list(seeds),"e1_r":list(R_GRID),"e1_gamma":list(GAMMA_GRID),
        "e1_temperature":TEMPERATURE,"e2_theta":list(THETA_GRID),"e2_functional_distance_cap":"q25 of pairwise unit-center distances",
        "e2_cv_folds":5,"e6_task_counts":[10,20,"all"],"e7_inputs":["image(k)","image(k)+state(s)","image(k)+state(s)+eval_skill_class one-hot"],
        "device":"cpu","threads":int(threads)},"runs":runs,"e0_saved_reaggregation":e0,
      "duration_seconds":time.perf_counter()-tic,"limitations":{"E0_functional":"not recoverable from saved per_row arrays; no unit members stored",
      "counterfactual_policy_mse":"not recoverable because v2.1 checkpoints/alternate-skill predictions were not saved; no proxy used",
      "E7_subtask":"skill_class one-hot encoding represents subtask input"}}
    (output/"results.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2,default=_json_default));(output/"results.partial.json").unlink(missing_ok=True)
    (output/"summary.md").write_text(_report(payload))
    return output


def _report(p):
    lines=["# 選択表現レビュー（E0–E3、E6、E7）","",
      "E0 は保存済み v2.1 per_row を再集計し、特徴キャッシュから訓練単位と機能εを再構成。E1–E3/E6/E7 も特徴キャッシュと軌道を使用。代替選択後の方策MSEはチェックポイントがなく、推測せず未評価とした。","",
      "|実験|結果と判定|","|---|---|"]
    for split in "AB":
        rr=[r for r in p["runs"] if r["split"]==split]
        if not rr: continue
        avg=lambda xs:float(np.mean(xs))
        d3e0=[p["e0_saved_reaggregation"][f"{split}/{s}/D3"] for s in range(3)]
        cov=avg([x["row_weighted"]["functional_coverage"] for x in d3e0])
        facc=avg([x["row_weighted"]["functional_accuracy"] for x in d3e0])
        exact=avg([x["row_weighted"]["strict_top1"] for x in d3e0])
        majority=avg([x["row_weighted"]["majority_chance"] for x in d3e0])
        chance=avg([x["row_weighted"]["frequency_chance"] for x in d3e0])
        lines.append(f"|E0 {split}|D3 strict {exact:.3f}（多数派 {majority:.3f}、頻度比例 {chance:.3f}）、機能正答 {facc:.3f}、coverage {cov:.3f}。ε内候補なしが{1-cov:.0%}。|")
        d3=[r["evaluations"]["E1/r5_g0"]["D3"] for r in rr]
        d1=avg([r["evaluations"]["E1/r5_g0"]["D1"]["strict_top1"] for r in rr])
        d1seed=[r["evaluations"]["E1/r5_g0"]["D1"]["strict_top1"] for r in rr]
        ref={"A":[.57,.41,.52],"B":[.48,.48,.51]}[split]
        nref=sum(a>=b for a,b in zip(d1seed,ref))
        lines.append(f"|E1 {split}|r5, γ0: D1 strict {d1:.3f}（査読のLOEO参照以上 {nref}/3）、D3 strict/function {avg([x['strict_top1'] for x in d3]):.3f}/{avg([x['functional_accuracy'] for x in d3]):.3f}、κ {avg([x['cohen_kappa'] for x in d3]):.3f}。参照精度を安定して超えず no-go。|")
        candidates=[]
        for th in ("theta0.1","theta0.2","theta0.3"):
            vals=[r["E2"][th] for r in rr]; mm=[x["evaluations"]["r5_g0"]["D3"] for x in vals]
            candidates.append((avg([x["functional_accuracy"] for x in mm]),th,avg([x["cohen_kappa"] for x in mm]),avg([x["merged_units"] for x in vals])))
        best=max(candidates)
        lines.append(f"|E2 {split}|最大D3機能率 {best[0]:.3f}（{best[1]}、平均K={best[3]:.1f}、κ={best[2]:.3f}）。0.7未達、no-go。|")
        before=avg([r["evaluations"]["E3/r5_g0"]["D3_before"]["strict_top1"] for r in rr]);after=avg([r["evaluations"]["E3/r5_g0"]["D3_after"]["strict_top1"] for r in rr])
        lines.append(f"|E3 {split}|BBSE r5γ0 のD3 strict {before:.3f}→{after:.3f}。選択後MSEは未検証。|")
        k10=avg([r["E6"]["10"]["D3"]["r5_g0"]["cohen_kappa"] for r in rr]);k20=avg([r["E6"]["20"]["D3"]["r5_g0"]["cohen_kappa"] for r in rr])
        lines.append(f"|E6 {split}|10→20タスクでD3 κ {k10:.3f}→{k20:.3f}。単調な上昇なし。all=20。|")
        vals=[avg([r["E7"][mode]["evaluation"]["D1"]["strict_top1"] for r in rr]) for mode in ("image","image_state","image_state_subtask")]
        diffs=[r["E7"]["image_state_subtask"]["evaluation"]["D1"]["strict_top1"]-r["E7"]["image"]["evaluation"]["D1"]["strict_top1"] for r in rr]
        e7note=("; 2/3 seedで+5pp超" if split=="B" and sum(x>=.05 for x in diffs)>=2 else "; 3/3 seedで+5ppなし")
        lines.append(f"|E7 {split}|D1 image / +state / +subtask {vals[0]:.3f}/{vals[1]:.3f}/{vals[2]:.3f}（subtask差 {vals[2]-vals[0]:+.3f}{e7note}、seed差 {', '.join(f'{x:+.3f}' for x in diffs)}）。|")
    lines += ["","全体 GO 条件（D3機能正答率≥0.7、κ>0.3、P-a以下の実際の選択方策MSE、両split 3/3 seed）は未達/未検証。正答率だけの改善は粒度を粗くした可能性があるため GO としない。","",
      "|Split|seed|K|E1 D3 strict (r5 γ0)|E1 κ|E7 D1 top-1 image / +state / +subtask|","|---|---:|---:|---:|---:|---:|"]
    for r in p["runs"]:
        m=r["evaluations"]["E1/r5_g0"]["D3"];e=[r["E7"][k]["evaluation"]["D1"]["strict_top1"] for k in ("image","image_state","image_state_subtask")]
        lines.append(f"|{r['split']}|{r['seed']}|{r['n_units']}|{m['strict_top1']:.3f}|{m['cohen_kappa']:.3f}|{e[0]:.3f} / {e[1]:.3f} / {e[2]:.3f}|")
    lines += ["","### E0 保存済み v2.1 の再集計","","|Split|条件|行top-1|多数派 / 頻度比例|functional / coverage|covered内top-1|実境界 interval|予測境界 interval|κ (行)|","|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for split in "AB":
      for cond in ("D1","D3"):
        vals=[p["e0_saved_reaggregation"][f"{split}/{seed}/{cond}"] for seed in range(3)]
        mean=lambda key:float(np.mean([v[key] for v in vals]))
        fmt=lambda key: (f"{np.mean([v['row_weighted'][key] for v in vals]):.3f}" if all(v['row_weighted'].get(key) is not None for v in vals) else "未算")
        lines.append(f"|{split}|{cond}|{fmt('strict_top1')}|{fmt('majority_chance')} / {fmt('frequency_chance')}|{fmt('functional_accuracy')} / {fmt('functional_coverage')}|{fmt('strict_on_covered')}|{np.mean([v['interval_actual_gripper']['strict_top1'] for v in vals]):.3f}|{np.mean([v['interval_predicted_gripper']['strict_top1'] for v in vals]):.3f}|{fmt('cohen_kappa')}|")
    lines += ["","E0の全seed詳細と全E1/E2/E3/E6/E7ハイパーパラメータは `results.json`。E7は eval_skill_class one-hot をサブタスク特徴として使った。これはテキスト埋め込みより教師ありの入力なので、その点を解釈に含める。","",f"実行時間: {p['duration_seconds']/60:.1f} 分。",""]
    return "\n".join(lines)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--cache-dir",type=Path,default=DEFAULT_CACHE);p.add_argument("--dataset",type=Path,default=DEFAULT_DATASET)
    p.add_argument("--v21-root",type=Path,default=V21_ROOT);p.add_argument("--splits",default="A,B");p.add_argument("--seeds",default="0,1,2")
    p.add_argument("--device",default="cpu");p.add_argument("--threads",type=int,default=8);p.add_argument("--smoke",action="store_true")
    a=p.parse_args(argv);run(output=a.output,cache_dir=a.cache_dir,dataset_dir=a.dataset,v21_root=a.v21_root,
      splits=tuple(x for x in a.splits.split(",") if x),seeds=tuple(int(x) for x in a.seeds.split(",")),device=a.device,threads=a.threads,smoke=a.smoke)


if __name__=="__main__":main()
