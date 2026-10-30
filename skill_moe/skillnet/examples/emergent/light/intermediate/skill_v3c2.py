"""V3 continuation: open-ended expert count, soft competition and chance-aware scores."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import skill_v2 as v2
from .skill_v3_competition import (CompetitiveExperts, _v3_schedule, _phase_rows,
                                   _segment_model_data, _decoder_losses, _verb_from_plan)
from ....research_paths import REPO_ROOT as RESEARCH_REPO_ROOT, STATE_DIR as RESEARCH_STATE_DIR


CONFIGS=(
    {"name":"hard_q90_m5_e2","q":.90,"min_support":5,"min_episodes":2,"temperature":0.},
    {"name":"hard_q90_m3_e2","q":.90,"min_support":3,"min_episodes":2,"temperature":0.},
    {"name":"hard_q80_m5_e2","q":.80,"min_support":5,"min_episodes":2,"temperature":0.},
    {"name":"hard_q70_m5_e2","q":.70,"min_support":5,"min_episodes":2,"temperature":0.},
    {"name":"soft_q80_m5_e2_t002","q":.80,"min_support":5,"min_episodes":2,"temperature":.02},
    {"name":"soft_q80_m5_e2_t005","q":.80,"min_support":5,"min_episodes":2,"temperature":.05},
)


def _memory_from_experts(model,segments):
    units=[]
    for uid in range(len(model.experts)):
        members=[segments[i] for i,x in enumerate(model.assignments) if int(x)==uid]
        if not members: continue
        curves=np.stack([m["curve"] for m in members]);mean=curves.mean(0)
        key=np.mean(np.stack([m["key"] for m in members]),axis=0).astype(np.float32)
        u=v2.SkillUnit(uid,mean.astype(np.float32),mean.astype(np.float32),curves.var(0).astype(np.float32),
            float(np.median([m["length"] for m in members])),key)
        u.members=members;u.support=len(members);u.episodes={str(m["episode_id"]) for m in members}
        u.tasks={int(m["task_index"]) for m in members};u.classes=[int(m.get("skill_class",-1)) for m in members]
        u.mean_rep=v2._rep(u.mean,u.median_length);units.append(u)
    memory=v2.TrajectorySkillMemory(1.)
    memory.units=units
    memory.assignments={int(s["segment_id"]):int(model.assignments[i]) for i,s in enumerate(segments)}
    return memory


def _training_sources(rows,segments,memory,model,*,seed,stride=.5,device="cpu"):
    actions=rows["a"][:,:v2.HORIZON]
    n,h,da=actions.shape
    sources=np.zeros((5,n,h,da),np.float32);confidence=np.full((5,n),.5,np.float32)
    rng=np.random.default_rng(seed)
    for seg in segments:
        uid=memory.assignments[int(seg["segment_id"])]
        span=np.asarray(seg["rows"],np.int64);unit=memory.units[uid]
        for local,row in enumerate(span):
            phase=local/max(len(span)-1,1)
            scene=torch.as_tensor(rows["k"][row:row+1],dtype=torch.float32,device=device)
            state=torch.as_tensor(rows["s"][row:row+1],dtype=torch.float32,device=device)
            with torch.no_grad():
                expert=model.experts[uid](scene,state,torch.tensor([phase],dtype=torch.float32,device=device)).cpu().numpy()[0]
                shifted=model.experts[uid](scene,state,torch.tensor([np.clip(phase+.2,0,1)],dtype=torch.float32,device=device)).cpu().numpy()[0]
            sources[0,row]=np.repeat(expert[None],h,axis=0)
            sources[1,row]=np.repeat(shifted[None],h,axis=0)
            alternatives=[u for u in memory.units if u.skill_id!=uid]
            near=alternatives[0] if alternatives else unit
            random_unit=memory.units[int(rng.integers(len(memory.units)))]
            sources[2,row]=v2._trajectory_chunk(near.mean,phase,near.median_length,stride,h)
            sources[3,row]=v2._trajectory_chunk(random_unit.mean,phase,random_unit.median_length,stride,h)
    return sources,confidence


def _chance_and_verb(model,rows,segments,schedule,truth,train_verb_majority,*,cache_dir,annotations_path):
    from . import clip_retrieval as cr
    # These joins happen after fitting; labels never enter decoder/router/memory updates.
    post=cr.attach_subtask_texts(rows,cache_dir=cache_dir,annotations_path=annotations_path)
    indices=np.asarray([int(s["rows"][0]) for s in segments],np.int64)
    true_verbs=[_verb_from_plan(post["subtask_text"][i]) for i in indices]
    pred_ids=np.argmax(schedule["scores"][indices],axis=1)
    pred_verbs=[train_verb_majority.get(int(uid),"<unseen>") for uid in pred_ids]
    freq=np.bincount(np.asarray(truth,np.int64),minlength=len(model.experts)).astype(np.float64)
    prob=freq/max(freq.sum(),1.)
    chosen=schedule["top_ids"][indices]
    top1=float(np.mean(pred_ids==truth))
    top5=float(np.mean([int(t) in row for t,row in zip(truth,chosen)]))
    verb_acc=float(np.mean([a==b for a,b in zip(pred_verbs,true_verbs)]))
    return {"top1":top1,"top5":top5,"majority_top1_chance":float(prob.max()),
            "frequency_sampled_top1_chance":float(np.square(prob).sum()),
            "verb_accuracy":verb_acc,"truth_skill_frequency":prob.tolist(),
            "true_verb_counts":{str(k):int(v) for k,v in zip(*np.unique(true_verbs,return_counts=True))}}


def _posthoc_verb_majorities(memory,fit,*,cache_dir,annotations_path):
    from . import clip_retrieval as cr
    post=cr.attach_subtask_texts(fit,cache_dir=cache_dir,annotations_path=annotations_path)
    majority={};purities={};hist={}
    for u in memory.units:
        verbs=[_verb_from_plan(post["subtask_text"][int(m["rows"][0])]) for m in u.members if len(m.get("rows",[]))]
        if not verbs: majority[u.skill_id]="<empty>";purities[u.skill_id]=0.;continue
        val,cnt=np.unique(verbs,return_counts=True);ix=int(cnt.argmax())
        majority[u.skill_id]=str(val[ix]);purities[u.skill_id]=float(cnt[ix]/cnt.sum())
        hist[str(u.skill_id)]={str(k):int(v) for k,v in zip(val,cnt)}
    return majority,purities,hist


def _skill_output_difference(model,rows):
    device=next(model.experts[0].parameters()).device
    n=len(rows["a"]);phase=torch.zeros(n,dtype=torch.float32,device=device)
    scene=torch.as_tensor(rows["k"],dtype=torch.float32,device=device)
    state=torch.as_tensor(rows["s"],dtype=torch.float32,device=device)
    with torch.no_grad(): pred=torch.stack([e(scene,state,phase) for e in model.experts],dim=1).cpu().numpy()
    pairs=[]
    for i in range(pred.shape[1]):
      for j in range(i+1,pred.shape[1]):
        pairs.append({"left":i,"right":j,"mean_abs_action_difference":float(np.mean(np.abs(pred[:,i]-pred[:,j]))),
                      "action_mse":float(np.mean(np.square(pred[:,i]-pred[:,j])))})
    return {"pair_count":len(pairs),"mean_pairwise_abs_action_difference":float(np.mean([x["mean_abs_action_difference"] for x in pairs])) if pairs else 0.,
            "mean_pairwise_action_mse":float(np.mean([x["action_mse"] for x in pairs])) if pairs else 0.,"pairs":pairs}


def _fit_and_eval_variant(cfg,fit,val,d1,test,segments,eval_segments,queries,fit_bases,actions,
                          f_normal,norm_normal,eps,*,seed,device,cache_dir,annotations_path,
                          epochs_expert=18,epochs_flow=80,smoke=False):
    model=CompetitiveExperts(fit["k"].shape[1],fit["s"].shape[1],actions.shape[-1],q=cfg["q"],
        min_support=cfg["min_support"],min_episodes=cfg["min_episodes"],max_rounds=3,
        temperature=cfg["temperature"],device=device).fit(fit,segments,seed=seed+901,
            epochs=6 if smoke else epochs_expert)
    model.fit_router(fit,segments,seed=seed+991,epochs=8 if smoke else 40)
    memory=_memory_from_experts(model,segments)
    if not memory.units: raise RuntimeError(f"{cfg['name']}: no formed experts")
    # Remap any empty expert indices defensively (normally all are occupied).
    val_segments=eval_segments["VAL"]
    for s in val_segments:s["key"]=queries["VAL"][int(s["rows"][0])].copy()
    val_schedule=_v3_schedule(val,val_segments,model,device=device)
    zeros=np.zeros((len(actions),int(np.prod(actions.shape[1:])),),np.float32)
    sources,confidence=_training_sources(fit,segments,memory,model,seed=seed+1053,device=device)
    mix=v2._normalise_mix(v2.V21_SOURCE_MIX)
    flow,norm,flow_fit=v2._fit_skill_flow(fit,actions,sources,confidence,mix,seed=seed+907,
        epochs=12 if smoke else epochs_flow,patience=4 if smoke else 8,batch_size=512,device=device,
        initial_model=f_normal,initial_norm=norm_normal,teacher_model=f_normal,teacher_weight=2.)
    val_tau=v2._calibrate_tau(flow,norm,val,val_schedule,eps["VAL"],device=device)
    majority,purities,verb_hist=_posthoc_verb_majorities(memory,fit,cache_dir=cache_dir,annotations_path=annotations_path)
    result={"config":cfg,"n_skills":len(memory.units),"support":[u.support for u in memory.units],
            "episodes_per_skill":[len(u.episodes) for u in memory.units],"thresholds":model.thresholds,
            "births":model.births,"merges":model.merges,"expert_fit_history":model.fit_history,
            "soft_em_temperature":model.temperature,"responsibilities_entropy":float(np.mean(-np.sum(model.responsibilities*np.log(np.maximum(model.responsibilities,1e-8)),axis=1))),
            "verb_purity_mean":float(np.mean(list(purities.values()))) if purities else 0.,
            "verb_majority_by_skill":majority,"verb_counts_by_skill":verb_hist,
            "flow_fit":{"best_epoch":flow_fit["best_epoch"],"initialized_from":flow_fit["initialized_from"],
                        "initial_state_equal":flow_fit["initial_state_equal"],"tau_calibration":val_tau}}
    evals={}
    for name,rows in (("D1",d1),("D3",test)):
        segs=eval_segments[name]
        schedule=_v3_schedule(rows,segs,model,device=device)
        data=_segment_model_data(rows,segs,device=device)
        losses=np.stack([v2._decoder_losses(x,data) for x in model.experts],axis=1)
        truth=losses.argmin(1)
        sm=_chance_and_verb(model,rows,segs,schedule,truth,majority,cache_dir=cache_dir,annotations_path=annotations_path)
        noise=eps[name]
        tau,steps=v2.apply_confidence_map(val_tau,schedule["confidence"])
        pred=v2._predict(flow,norm,rows,schedule["anchor"],noise,tau,steps,device)
        target=rows["a"][:,:v2.HORIZON]
        metric=v2._metrics(pred,target)
        normal=v2._predict(f_normal,norm_normal,rows,np.zeros_like(schedule["anchor"]),noise,
            np.ones(len(rows["a"]),np.float32),np.full(len(rows["a"]),10),device)
        normal_row=v2._metrics(normal,target)["per_row_mse"]
        evals[name]={**sm,"mse":metric["mse"],"worse_fraction_vs_normal":float(np.mean(metric["per_row_mse"]>normal_row)),
                      "worse_5pct_fraction_vs_normal":float(np.mean(metric["per_row_mse"]>normal_row*1.05)),
                      "mean_tau":float(np.mean(tau)),"selected_skill_frequency":np.bincount(schedule["skill_id"],minlength=len(model.experts)).tolist()}
        if name=="D3":
            # Compare every expert on the same D3 observation/state/phase inputs.
            d3diff=_skill_output_difference(model,rows)
            rank=np.argsort(-schedule["scores"],axis=1)
            scene=torch.as_tensor(rows["k"],dtype=torch.float32,device=device);state=torch.as_tensor(rows["s"],dtype=torch.float32,device=device)
            phase=torch.zeros(len(rows["a"]),dtype=torch.float32,device=device)
            with torch.no_grad(): allpred=torch.stack([e(scene,state,phase) for e in model.experts],1).cpu().numpy()
            selected=rank[:,0];alternate=rank[:,1] if rank.shape[1]>1 else selected
            result["b_swap_action_delta"]=float(np.mean(np.abs(allpred[np.arange(len(selected)),selected]-allpred[np.arange(len(selected)),alternate])))
            result["same_scene_skill_output_difference"]=d3diff
        evals[name]=evals.get(name,{})
    result["evaluation"]=evals
    result["memory_metadata"]={"length_seconds_median":float(np.median([u.median_length*.5 for u in memory.units])),
        "tasks_per_skill_median":float(np.median([len(u.tasks) for u in memory.units])),"formation":memory.describe()}
    checkpoint={"experts":[{k:v.detach().cpu() for k,v in x.state_dict().items()} for x in model.experts],
        "router":{k:v.detach().cpu() for k,v in model.router.state_dict().items()},
        "flow":{k:v.detach().cpu() for k,v in flow.state_dict().items()},"flow_norm":norm,"tau_calibration":val_tau,
        "unit_means":np.stack([u.mean for u in memory.units]),"unit_keys":np.stack([u.key for u in memory.units]),
        "unit_lengths":np.asarray([u.median_length for u in memory.units]),"config":cfg}
    return result,checkpoint


def _summary(payload,out):
    lines=["# V3 拡張競合比較", "", "計画文由来のラベルは全学習終了後にのみ使用。偶然基準は評価区間の真の割当分布から計算。", "",
           "| Split/seed | 設定 | K | verb純度 | D1 acc/chance(majority,proportional) | D3 acc/chance(majority,proportional) | verb acc D1/D3 | D3 MSE | D1悪化率 | 専門家間同場面Δaction |",
           "|---|---|---:|---:|---|---|---:|---:|---:|---:|"]
    for run in payload["runs"]:
      for name,x in run["variants"].items():
        d1=x["evaluation"]["D1"];d3=x["evaluation"]["D3"]
        lines.append(f"| {run['split']}/{run['seed']} | {name} | {x['n_skills']} | {x['verb_purity_mean']:.3f} | {d1['top1']:.3f}/{d1['majority_top1_chance']:.3f},{d1['frequency_sampled_top1_chance']:.3f} | {d3['top1']:.3f}/{d3['majority_top1_chance']:.3f},{d3['frequency_sampled_top1_chance']:.3f} | {d1['verb_accuracy']:.3f}/{d3['verb_accuracy']:.3f} | {d3['mse']:.4f} | {d1['worse_fraction_vs_normal']:.3f} | {x['same_scene_skill_output_difference']['mean_pairwise_abs_action_difference']:.4f} |")
    lines += ["", "Outputs include D1/D3 top-5, D3 MSE, D1 harm, chance baselines, verb confusion counts and pairwise skill-output differences in results.json. See report.md for seed-averaged comparison with V3 q95 and interpretation.", ""]
    (out/"summary.md").write_text("\n".join(lines))


def run(*,output,splits=("A","B"),seeds=(0,1,2),device="cuda",threads=8,smoke=False,configs=CONFIGS):
    from . import anchor_flow as af,fair_compare as fc,clip_retrieval as cr
    from .compare_splits import SPLITS,_subset
    from .experiment import DEFAULT_CACHE,DEFAULT_DATASET,_episode_train_validation_indices,load_rows
    out=Path(output)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);(out/"checkpoints").mkdir()
    torch.set_num_threads(int(threads))
    try:torch.set_num_interop_threads(1)
    except RuntimeError:pass
    dev=torch.device(device);start=time.perf_counter();rows=load_rows(DEFAULT_CACHE,DEFAULT_DATASET,max_horizon=fc.HORIZON)
    allruns=[];basepath=Path(str(RESEARCH_STATE_DIR / 'outputs/memory-pilot/skill-v21-20261029T060213Z-7493c4/results.json'))
    v21=json.load(open(basepath));baseline_lookup={(r["split"],r["seed"]):r for r in v21["runs"]}
    for split in splits:
      train_tasks,eval_tasks,_=SPLITS[split]
      selected=_subset(rows,np.isin(rows["task_index"],sorted(set(train_tasks)|set(eval_tasks)))&rows["valid_h25"])
      train_all=_subset(selected,np.isin(selected["task_index"],train_tasks))
      for seed in seeds:
        tic=time.perf_counter();fit_rows,d1=fc.holdout_split(train_all,split,seed,per_task=2 if smoke else fc.HOLDOUT_PER_TASK)
        if smoke:
          fit_rows=af._select_episodes(fit_rows,train_tasks,12,seed+101);d1=af._select_episodes(d1,train_tasks,4,seed+311)
        fit_ix,val_ix=_episode_train_validation_indices(fit_rows,seed+401)
        fit=_subset(fit_rows,np.isin(np.arange(len(fit_rows["a"])),fit_ix));val=_subset(fit_rows,np.isin(np.arange(len(fit_rows["a"])),val_ix))
        test=_subset(selected,np.isin(selected["task_index"],eval_tasks))
        if smoke:test=af._select_episodes(test,eval_tasks,4,seed+211)
        m6,fit_bases,actions=af._fit_m6_backbone_and_units(fit,seed=seed+19,device=dev,epochs=12 if smoke else 60)
        fit_q=np.concatenate((fit["k"],fit["s"]),axis=-1).astype(np.float32)
        limits=v2.derive_length_limits(fit);segments,_=v2.make_segments(fit,fit_q,limits,.5)
        for s in segments:s["key"]=fit_q[int(s["rows"][0])].copy()
        def scene_key(part):return np.concatenate((part["k"],part["s"]),axis=-1).astype(np.float32)
        d1q,testq=scene_key(d1),scene_key(test)
        eval_segments={"D1":v2.make_segments(d1,d1q,limits,.5)[0],"D3":v2.make_segments(test,testq,limits,.5)[0],"VAL":None}
        for key,segs in eval_segments.items():
          if segs is not None:
            qv=d1q if key=="D1" else testq
            for s in segs:s["key"]=qv[int(s["rows"][0])].copy()
        train_action=actions
        dim=int(np.prod(train_action.shape[1:]));zero=np.zeros((len(train_action),dim),np.float32)
        normal_sources=(zero,np.zeros(len(train_action),np.float32),zero.copy(),np.zeros(len(train_action),np.float32))
        f_normal,norm_normal,_=af.fit_anchor_flow(fit,train_action,normal_sources,source_mix=fc.NORMAL_MIX,
            seed=seed+827,epochs=12 if smoke else 80,patience=4 if smoke else 8,batch_size=512,device=dev)
        rng=np.random.default_rng(seed*1_000_003+31)
        eps={"VAL":rng.standard_normal((fc.SAMPLES,len(val["a"]),dim)).astype(np.float32),
             "D1":rng.standard_normal((fc.SAMPLES,len(d1["a"]),dim)).astype(np.float32),
             "D3":rng.standard_normal((fc.SAMPLES,len(test["a"]),dim)).astype(np.float32)}
        variants={};checkpoints={}
        for cfg in configs:
          cfg=dict(cfg)
          model=CompetitiveExperts(fit["k"].shape[1],fit["s"].shape[1],train_action.shape[-1],q=cfg["q"],
            min_support=cfg["min_support"],min_episodes=cfg["min_episodes"],max_rounds=3,
            temperature=cfg["temperature"],device=dev).fit(fit,segments,seed=seed+901,epochs=6 if smoke else 18)
          model.fit_router(fit,segments,seed=seed+991,epochs=8 if smoke else 40)
          memory=_memory_from_experts(model,segments)
          # The unit count is an outcome, never a configured K.
          valq=scene_key(val);valsegs=v2.make_segments(val,valq,limits,.5)[0]
          for s in valsegs:s["key"]=valq[int(s["rows"][0])].copy()
          valsched=_v3_schedule(val,valsegs,model,device=dev)
          sources,conf=_training_sources(fit,segments,memory,model,seed=seed+1053,device=dev)
          flow,norm,flowfit=v2._fit_skill_flow(fit,train_action,sources,conf,v2._normalise_mix(v2.V21_SOURCE_MIX),
              seed=seed+907,epochs=12 if smoke else 80,patience=4 if smoke else 8,batch_size=512,device=dev,
              initial_model=f_normal,initial_norm=norm_normal,teacher_model=f_normal,teacher_weight=2.)
          taucal=v2._calibrate_tau(flow,norm,val,valsched,eps["VAL"],device=dev)
          majority,purity,hist=_posthoc_verb_majorities(memory,fit,cache_dir=DEFAULT_CACHE,annotations_path=cr.DEFAULT_ANNOTATIONS)
          ev={}
          for name,part,segs,qv in (("D1",d1,eval_segments["D1"],d1q),("D3",test,eval_segments["D3"],testq)):
            sched=_v3_schedule(part,segs,model,device=dev)
            dat=_segment_model_data(part,segs,device=dev)
            loss=np.stack([_decoder_losses(e,dat) for e in model.experts],axis=1)
            truth=loss.argmin(1)
            chance=_chance_and_verb(model,part,segs,sched,truth,majority,cache_dir=DEFAULT_CACHE,annotations_path=cr.DEFAULT_ANNOTATIONS)
            t,nsteps=v2.apply_confidence_map(taucal,sched["confidence"])
            pred=v2._predict(flow,norm,part,sched["anchor"],eps[name],t,nsteps,dev)
            target=part["a"][:,:v2.HORIZON];metrics=v2._metrics(pred,target)
            normal=v2._predict(f_normal,norm_normal,part,np.zeros_like(sched["anchor"]),eps[name],
                np.ones(len(part["a"]),np.float32),np.full(len(part["a"]),10),dev)
            normalrow=v2._metrics(normal,target)["per_row_mse"]
            ev[name]={**chance,"mse":metrics["mse"],"worse_fraction_vs_normal":float(np.mean(metrics["per_row_mse"]>normalrow)),
              "worse_5pct_fraction_vs_normal":float(np.mean(metrics["per_row_mse"]>normalrow*1.05)),"mean_tau":float(t.mean()),
              "selected_skill_frequency":np.bincount(sched["skill_id"],minlength=len(model.experts)).tolist()}
            if name=="D3":
              delta=_skill_output_difference(model,part)
              rank=np.argsort(-sched["scores"],axis=1);ids=rank[:,0];alt=rank[:,1] if rank.shape[1]>1 else ids
              sc=torch.as_tensor(part["k"],dtype=torch.float32,device=dev);st=torch.as_tensor(part["s"],dtype=torch.float32,device=dev);ph=torch.zeros(len(ids),device=dev)
              with torch.no_grad():outs=torch.stack([e(sc,st,ph) for e in model.experts],dim=1).cpu().numpy()
              bswap=float(np.mean(np.abs(outs[np.arange(len(ids)),ids]-outs[np.arange(len(ids)),alt])))
          vname=cfg["name"]
          variants[vname]={"config":cfg,"n_skills":len(memory.units),"support":[u.support for u in memory.units],
            "episodes_per_skill":[len(u.episodes) for u in memory.units],"verb_purity_mean":float(np.mean(list(purity.values()))) if purity else 0.,
            "verb_majority_by_skill":majority,"verb_counts_by_skill":hist,"thresholds":model.thresholds,
            "births":model.births,"merges":model.merges,"soft_em_temperature":model.temperature,
            "responsibilities_entropy":float(np.mean(-np.sum(model.responsibilities*np.log(np.maximum(model.responsibilities,1e-8)),axis=1))),
            "same_scene_skill_output_difference":delta,"b_swap_action_delta":bswap,"evaluation":ev,
            "flow_fit":{"best_epoch":flowfit["best_epoch"],"initialized_from":flowfit["initialized_from"],"initial_state_equal":flowfit["initial_state_equal"],"tau_calibration":taucal},
            "memory":{"median_length_seconds":float(np.median([u.median_length*.5 for u in memory.units])),
                      "median_tasks_per_skill":float(np.median([len(u.tasks) for u in memory.units])),"description":memory.describe()}}
          checkpoints[vname]={"experts":[{k:x.detach().cpu() for k,x in e.state_dict().items()} for e in model.experts],
            "router":{k:x.detach().cpu() for k,x in model.router.state_dict().items()},"flow":{k:x.detach().cpu() for k,x in flow.state_dict().items()},
            "flow_norm":norm,"tau_calibration":taucal,"unit_means":np.stack([u.mean for u in memory.units]),
            "unit_keys":np.stack([u.key for u in memory.units]),"unit_lengths":np.asarray([u.median_length for u in memory.units]),"config":cfg}
        torch.save(checkpoints,out/"checkpoints"/f"{split}_s{seed}.pt")
        base=baseline_lookup[(split,seed)]
        refs={k:base["conditions"]["D3"]["mean"][k]["mse"] for k in ("normal","raw_interval","primitive_Pa","skill_v2")}
        allruns.append({"split":split,"seed":seed,"variants":variants,"v21_D3_baselines":refs,
                        "segments":len(segments),"duration_seconds":time.perf_counter()-tic})
        payload={"schema_version":"skill-v3c2-v1","config":{"splits":list(splits),"seeds":list(seeds),"smoke":smoke,
          "device":str(dev),"threads":threads,"configs":list(configs),"language_free_inputs":"k+s","labels":"posthoc only"},
          "runs":allruns,"elapsed_seconds":time.perf_counter()-start}
        (out/"results.partial.json").write_text(json.dumps(payload,indent=2,ensure_ascii=False,default=str)+"\n")
    payload={"schema_version":"skill-v3c2-v1","config":{"splits":list(splits),"seeds":list(seeds),"smoke":smoke,
      "device":str(dev),"threads":threads,"configs":list(configs),"language_free_inputs":"k+s","labels":"posthoc only"},
      "runs":allruns,"elapsed_seconds":time.perf_counter()-start}
    (out/"results.json").write_text(json.dumps(payload,indent=2,ensure_ascii=False,default=str)+"\n")
    _summary(payload,out)
    return out


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--splits",default="A,B");p.add_argument("--seeds",default="0,1,2");p.add_argument("--device",default="cuda")
    p.add_argument("--threads",type=int,default=8);p.add_argument("--smoke",action="store_true")
    a=p.parse_args(argv);out=run(output=a.output,splits=tuple(a.splits.split(',')),seeds=tuple(map(int,a.seeds.split(','))),device=a.device,threads=a.threads,smoke=a.smoke);print(out)


if __name__=="__main__":main()
