"""Scene-conditioned latent skills (v3 V1/V2) for the light LIBERO cache.

This module deliberately owns its implementation.  The older experiment and
fair-comparison modules are imported read-only for data, segmentation and flow
evaluation helpers.
"""
from __future__ import annotations

import argparse
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import anchor_flow as af
from . import fair_compare as fc
from . import skill_v2 as v2
from .splits import SPLITS, subset_rows as _subset
from .experiment import (DEFAULT_CACHE, DEFAULT_DATASET,
                         _episode_train_validation_indices, _standardize,
                         _standardize_fit, load_rows)
from ....research_paths import REPO_ROOT as RESEARCH_REPO_ROOT, STATE_DIR as RESEARCH_STATE_DIR

LATENT_DIMS = (4, 8)
BETA_VALUES = (0.01, 0.1, 1.0)
V2_CONTRAST_WEIGHT = 0.1
SOURCE_MIX = v2.V21_SOURCE_MIX


class SceneConditionedVAE(nn.Module):
    """Encode motion alone; decode motion from latent, scene, state and phase."""
    def __init__(self, latent_dim: int, key_dim: int, state_dim: int, action_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(32 * 7, 256), nn.GELU(),
                                     nn.Linear(256, 128), nn.GELU())
        self.mu = nn.Linear(128, latent_dim)
        self.logvar = nn.Linear(128, latent_dim)
        self.key_proj = nn.Sequential(nn.Linear(key_dim, 64), nn.GELU())
        self.state_proj = nn.Sequential(nn.Linear(state_dim, 32), nn.GELU())
        self.decoder = nn.Sequential(nn.Linear(latent_dim + 64 + 32 + 1, 256), nn.GELU(),
                                     nn.Linear(256, 256), nn.GELU(),
                                     nn.Linear(256, action_dim))

    def encode(self, curve):
        h = self.encoder(curve.reshape(len(curve), -1))
        return self.mu(h), self.logvar(h).clamp(-8, 6)

    def decode(self, z, key, state, phase):
        h = torch.cat((z, self.key_proj(key), self.state_proj(state), phase.reshape(-1, 1)), -1)
        return self.decoder(h)

    def forward(self, curve, key, state, phase, *, sample=True):
        mu, logvar = self.encode(curve)
        z = mu + torch.randn_like(mu) * torch.exp(.5 * logvar) if sample else mu
        return self.decode(z, key, state, phase), mu, logvar


def _curve_norm(curves):
    mean = np.asarray(curves, np.float32).mean(axis=0)
    scale = np.asarray(curves, np.float32).std(axis=0)
    scale[scale < 1e-5] = 1.0
    return mean, scale


def make_interval_curves(rows, segments):
    """Unlabelled interval trajectory inputs, one first action per cache row."""
    return np.stack([s["curve"] for s in segments]).astype(np.float32)


def _segment_maps(segments, row_count):
    row_segment = np.full(row_count, -1, np.int64)
    row_phase = np.zeros(row_count, np.float32)
    for sid, seg in enumerate(segments):
        ix = np.asarray(seg["rows"], np.int64)
        row_segment[ix] = sid
        row_phase[ix] = np.linspace(0, 1, len(ix), dtype=np.float32) if len(ix) > 1 else 0
    return row_segment, row_phase


def fit_scene_vae(rows, segments, validation_rows, validation_segments, *, latent_dim,
                  beta, seed=0, device="cuda", epochs=30, batch_size=512,
                  contrastive=False, contrast_weight=V2_CONTRAST_WEIGHT):
    """Fit V1 or V2 and return standardized latent means and training history."""
    torch.manual_seed(int(seed))
    device = torch.device(device)
    curves = make_interval_curves(rows, segments)
    val_curves = make_interval_curves(validation_rows, validation_segments)
    cm, cs = _curve_norm(curves)
    curves = ((curves - cm) / cs).astype(np.float32)
    val_curves = ((val_curves - cm) / cs).astype(np.float32)
    actions = np.asarray(rows["a"][:, :v2.HORIZON], np.float32).reshape(len(rows["a"]), -1)
    am, ass = _standardize_fit(actions)
    actions = _standardize(actions, am, ass)
    val_actions = _standardize(np.asarray(validation_rows["a"][:, :v2.HORIZON]).reshape(len(validation_rows["a"]), -1), am, ass)
    km, ks = _standardize_fit(rows["k"]); sm, ss = _standardize_fit(rows["s"])
    keys = _standardize(rows["k"], km, ks); states = _standardize(rows["s"], sm, ss)
    val_keys = _standardize(validation_rows["k"], km, ks); val_states = _standardize(validation_rows["s"], sm, ss)
    rid, phase = _segment_maps(segments, len(rows["a"]))
    vrid, vphase = _segment_maps(validation_segments, len(validation_rows["a"]))
    if np.any(rid < 0) or np.any(vrid < 0):
        raise ValueError("every row must belong to one segmented interval")
    model = SceneConditionedVAE(latent_dim, keys.shape[1], states.shape[1], actions.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-5)
    rng = torch.Generator(device=device).manual_seed(seed + 17)
    tc = torch.as_tensor(curves, device=device)
    tvc = torch.as_tensor(val_curves, device=device)
    tr = torch.as_tensor(rid, device=device); tp = torch.as_tensor(phase, device=device)
    va = torch.as_tensor(actions, device=device); vk = torch.as_tensor(keys, device=device); vs = torch.as_tensor(states, device=device)
    vtr = torch.as_tensor(vrid, device=device); vtp = torch.as_tensor(vphase, device=device)
    vtarget = torch.as_tensor(val_actions, device=device); vkey = torch.as_tensor(val_keys, device=device); vstate = torch.as_tensor(val_states, device=device)
    valid_rows = np.arange(len(val_actions))
    contrast_partner=None
    if contrastive:
        ep=np.asarray(rows["episode_id"]).astype(str)
        contrast_partner=np.arange(len(ep),dtype=np.int64)
        for e in np.unique(ep):
            ix=np.flatnonzero(ep==e)
            if len(ix)>1:
                for pos,row in enumerate(ix): contrast_partner[row]=ix[(pos+1)%len(ix)]
    history, best, best_state, stale = [], float("inf"), None, 0
    # Episode-aware subset is already supplied by the caller; validation is never used for gradients.
    for epoch in range(int(epochs)):
        model.train(); order = torch.randperm(len(actions), generator=rng, device=device); losses=[]
        for lo in range(0, len(order), batch_size):
            ix = order[lo:lo + batch_size]
            sid = tr[ix]
            recon, mu, logvar = model(tc[sid], vk[ix], vs[ix], tp[ix], sample=True)
            rec = F.mse_loss(recon, va[ix])
            kl = -.5 * (1 + logvar - mu.square() - logvar.exp()).sum(-1).mean()
            loss = rec + float(beta) * kl
            if contrastive:
                # Same interval under a small temporal/noise perturbation is a positive;
                # distinct intervals from one recording are negatives.
                perturbed = tc[sid] + torch.randn_like(tc[sid]) * 0.015
                pos_mu, _ = model.encode(perturbed)
                pos_loss = (mu - pos_mu).square().sum(-1).mean()
                other_row=contrast_partner[ix.detach().cpu().numpy()]
                valid_pair=rid[other_row] != rid[ix.detach().cpu().numpy()]
                if np.any(valid_pair):
                    other_sid = tr[torch.as_tensor(other_row[valid_pair], device=device)]
                    neg_mu, _ = model.encode(tc[other_sid])
                    pos_mu=mu[torch.as_tensor(valid_pair,device=device)]
                    neg_dist = (pos_mu - neg_mu).square().sum(-1).sqrt()
                    neg_loss = F.relu(1.0 - neg_dist).square().mean()
                else:
                    neg_loss = mu.new_zeros(())
                loss = loss + float(contrast_weight) * (pos_loss + neg_loss)
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            vm, vl = model.encode(tvc[vtr]); vz = vm
            pred = model.decode(vz, vkey, vstate, torch.as_tensor(vphase, device=device))
            vloss = float(F.mse_loss(pred, vtarget).cpu())
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "validation_mse": vloss})
        if vloss < best - 1e-5:
            best, stale = vloss, 0; best_state = {k: x.detach().clone() for k,x in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 6: break
    if best_state: model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        z, _ = model.encode(torch.as_tensor(curves, device=device))
        vz, _ = model.encode(torch.as_tensor(val_curves, device=device))
    state = {"curve_mean": cm, "curve_scale": cs, "action_mean": am, "action_scale": ass,
             "key_mean": km, "key_scale": ks, "state_mean": sm, "state_scale": ss}
    return model, z.cpu().numpy().astype(np.float32), vz.cpu().numpy().astype(np.float32), state, {"best_validation_mse": best, "epochs": len(history), "history": history}


class LatentSkillBank:
    """Open-ended M6-style DP means, with support/episode minimums."""
    def __init__(self, kappa=.1, min_support=10, min_episodes=3):
        self.kappa, self.min_support, self.min_episodes = kappa, min_support, min_episodes
        self.centers = np.empty((0, 0), np.float32); self.members=[]; self.assignments=None; self.epsilon=0.0

    def fit(self, z, segments, *, seed=0):
        z=np.asarray(z,np.float32); rng=np.random.default_rng(seed)
        if len(z) != len(segments): raise ValueError("latent and segment count mismatch")
        pair=rng.integers(0,len(z),(min(4096,max(64,2*len(z))),2))
        # Match M6's mean-squared Euclidean distance and scale calibration.
        ds=np.mean(np.square(z[pair[:,0]]-z[pair[:,1]]),axis=1); ds=ds[ds>1e-8]
        self.epsilon=max(float(self.kappa*np.median(ds)) if len(ds) else 1e-6,1e-6)
        groups=[]; centers=[]
        for ix in rng.permutation(len(z)):
            if not centers: centers.append(z[ix].copy()); groups.append([int(ix)]); continue
            d=np.mean(np.square(np.stack(centers)-z[ix]),axis=1); j=int(d.argmin())
            if d[j]<=self.epsilon:
                groups[j].append(int(ix)); centers[j]=z[groups[j]].mean(0)
            else: centers.append(z[ix].copy()); groups.append([int(ix)])
        accepted=[]
        for group in groups:
            eps={segments[i]["episode_id"] for i in group}
            if len(group)>=self.min_support and len(eps)>=self.min_episodes: accepted.append(group)
        if not accepted: raise RuntimeError("latent clustering produced no supported skills")
        # Reassign every segment to the nearest supported prototype; this keeps
        # one ground-truth skill ID per interval while preserving open-ended K.
        self.members=accepted; self.centers=np.stack([z[g].mean(0) for g in accepted]).astype(np.float32)
        self.assignments=np.mean(np.square(z[:,None,:]-self.centers[None,:,:]),axis=-1).argmin(1).astype(np.int64)
        return self


def decoder_anchor(model, bank, unit_ids, phase, rows, norm, *, device="cuda", batch=512):
    """Decode one prototype at each row's scene/state and within-skill phase."""
    k=_standardize(rows["k"],norm["key_mean"],norm["key_scale"])
    s=_standardize(rows["s"],norm["state_mean"],norm["state_scale"])
    out=[]; model.eval()
    with torch.no_grad():
        for lo in range(0,len(rows["a"]),batch):
            hi=min(len(rows["a"]),lo+batch); ix=np.asarray(unit_ids[lo:hi],np.int64)
            z=torch.as_tensor(bank.centers[ix],device=device)
            kt=torch.as_tensor(k[lo:hi],device=device); st=torch.as_tensor(s[lo:hi],device=device)
            pt=torch.as_tensor(np.asarray(phase[lo:hi],np.float32),device=device)
            y=model.decode(z,kt,st,pt).cpu().numpy()
            y=y*norm["action_scale"]+norm["action_mean"]
            out.append(y.reshape(hi-lo,v2.HORIZON,7))
    return np.concatenate(out).astype(np.float32)


def evaluate_assignments(rows, query_keys, assignments, prototypes, anchor_builder, *,
                         truth_assignments=None, top_k=(1,5)):
    """Adapter API for V3/V4: score supplied skill IDs and build per-row anchors.

    `anchor_builder(unit_ids, row_indices)` returns (N,H,7). Other skill
    formation methods can pass their own assignments/prototypes without
    changing this evaluator. Returns top-k selection accuracy and anchors.
    """
    keys=np.asarray(prototypes["keys"],np.float32); q=np.asarray(query_keys,np.float32)
    q=q/np.maximum(np.linalg.norm(q,axis=1,keepdims=True),1e-8)
    keys=keys/np.maximum(np.linalg.norm(keys,axis=1,keepdims=True),1e-8)
    scores=q@keys.T; order=np.argsort(-scores,axis=1)
    selected=order[:,0]
    anchors=anchor_builder(selected,np.arange(len(q)))
    metrics={}
    if truth_assignments is not None:
        truth=np.asarray(truth_assignments,np.int64)
        for k in top_k: metrics[f"top{k}_accuracy"]=float(np.mean([truth[i] in order[i,:k] for i in range(len(q))]))
        metrics["top1_accuracy"]=float(np.mean(selected==truth))
    return {"selection":metrics,"scores":scores,"selected":selected,"anchors":anchors}


def build_skill_metadata(bank, segments):
    meta=[]
    for j,center in enumerate(bank.centers):
        ids=np.flatnonzero(bank.assignments==j); ss=[segments[i] for i in ids]
        tasks=len({int(s["task_index"]) for s in ss})
        classes=[int(s["skill_class"]) for s in ss if int(s["skill_class"])>=0]
        purity=float(max(np.bincount(classes))/len(classes)) if classes else None
        majority=int(np.bincount(classes).argmax()) if classes else -1
        meta.append({"skill_id":j,"support":len(ids),"episodes":len({s["episode_id"] for s in ss}),
                     "tasks":tasks,"verb_purity_posthoc":purity,
                     "majority_verb_class_posthoc":majority,
                     "median_length_seconds":float(np.median([s["length_seconds"] for s in ss]))})
    return meta


def _latent_means(model, segments, device, norm):
    curves=np.stack([s["curve"] for s in segments]).astype(np.float32)
    curves=_standardize(curves,norm["curve_mean"],norm["curve_scale"])
    with torch.no_grad():
        mu,_=model.encode(torch.as_tensor(curves,device=device))
    return mu.cpu().numpy().astype(np.float32)


def _row_keys(rows, key_stats=None):
    raw=np.concatenate([np.asarray(rows["k"],np.float32),np.asarray(rows["s"],np.float32)],axis=-1)
    if key_stats is None:
        mean,scale=_standardize_fit(raw)
    else: mean,scale=key_stats
    return _standardize(raw,mean,scale),(mean,scale)


def _make_flow_sources(model, bank, segments, rows, norm, row_phase, query, *, seed, device):
    row_segment,_=_segment_maps(segments,len(rows["a"]))
    truth=bank.assignments[row_segment]
    keys=np.stack([np.mean(np.stack([s["key"] for i,s in enumerate(segments) if bank.assignments[i]==j]),axis=0)
                   for j in range(len(bank.centers))]).astype(np.float32)
    qn=query/np.maximum(np.linalg.norm(query,axis=1,keepdims=True),1e-8)
    kn=keys/np.maximum(np.linalg.norm(keys,axis=1,keepdims=True),1e-8)
    sims=qn@kn.T
    near=np.argsort(-sims,axis=1)
    rng=np.random.default_rng(seed)
    alt=np.asarray([near[i,1] if len(near[i])>1 else near[i,0] for i in range(len(rows["a"]))],np.int64)
    rnd=rng.integers(len(bank.centers),size=len(rows["a"]))
    shifted=np.clip(row_phase+rng.choice([-1,1],len(row_phase))*rng.uniform(.15,.45,len(row_phase)),0,1).astype(np.float32)
    anchors=np.zeros((5,len(rows["a"]),v2.HORIZON,7),np.float32)
    anchors[0]=decoder_anchor(model,bank,truth,row_phase,rows,norm,device=device)
    anchors[1]=decoder_anchor(model,bank,truth,shifted,rows,norm,device=device)
    anchors[2]=decoder_anchor(model,bank,alt,row_phase,rows,norm,device=device)
    anchors[3]=decoder_anchor(model,bank,rnd,row_phase,rows,norm,device=device)
    confidence=np.zeros((5,len(rows["a"])),np.float32)
    for kind,ids in enumerate((truth,truth,alt,rnd)):
        confidence[kind]=np.clip((sims[np.arange(len(ids)),ids]+1)/2,0,1)
    return anchors,confidence,keys,truth


def _eval_v3(rows, query, q_stats, model, bank, skill_keys, flow, flow_norm, norm,
             normal_model, normal_norm, segments, eps, *, device, seed, skill_verbs=None, stream=False):
    row_segment,phase=_segment_maps(segments,len(rows["a"]))
    z=_latent_means(model,segments,device,norm)
    truth=np.linalg.norm(z[:,None,:]-bank.centers[None,:,:],axis=-1).argmin(1)
    truth_rows=truth[row_segment]
    qn=query/np.maximum(np.linalg.norm(query,axis=1,keepdims=True),1e-8)
    kn=skill_keys/np.maximum(np.linalg.norm(skill_keys,axis=1,keepdims=True),1e-8)
    sims=qn@kn.T; selected=sims.argmax(1)
    anchors=decoder_anchor(model,bank,selected,phase,rows,norm,device=device)
    conf=np.clip((sims[np.arange(len(selected)),selected]+1)/2,0,1)
    tau=np.clip(1-conf,.1,1).astype(np.float32)
    nsteps=np.maximum(2,np.rint(10*tau).astype(np.int64))
    pred=fc.rollout(flow,flow_norm,rows["f"],rows["s"],anchors,
                    eps,tau,n=nsteps,device=device)
    normal=fc.rollout(normal_model,normal_norm,rows["f"],rows["s"],None,
                      eps,np.ones(len(rows["a"]),np.float32),
                      n=np.full(len(rows["a"]),10,np.int64),device=device)
    target=rows["a"][:,:v2.HORIZON]
    row_error=np.mean((pred-target[None])**2,axis=(0,2,3))
    normal_row_error=np.mean((normal-target[None])**2,axis=(0,2,3))
    per_acc={f"top{k}":float(np.mean([truth_rows[i] in np.argsort(-sims[i])[:k] for i in range(len(rows["a"]))])) for k in (1,5)}
    verb_acc=None
    if skill_verbs is not None and "skill_class" in rows:
        labels=np.asarray(rows["skill_class"],np.int64)
        good=labels>=0
        verb_acc=float(np.mean(np.asarray(skill_verbs,np.int64)[selected[good]]==labels[good])) if np.any(good) else None
    return {"mse":float(np.mean((pred-target[None])**2)),
            "normal_mse":float(np.mean((normal-target[None])**2)),
            "D1_worse_row_fraction":float(np.mean(row_error>normal_row_error)),
            "selection_accuracy":per_acc,
            "verb_selection_accuracy_posthoc":verb_acc,
            "latent_swap_action_change":None,
            "selected_skill":selected,"truth_skill":truth_rows,"anchor":anchors,
            "prediction":pred.mean(0),"tau":tau,"tau_mean":float(tau.mean()),"steps_mean":float(nsteps.mean())}


def _variant_run(rows, split, seed, device, *, stride_seconds, latent_dim, beta, contrastive,
                 epochs, flow_epochs, batch_size, smoke=False):
    train_tasks, eval_tasks, _=SPLITS[split]
    train_all=_subset(rows,np.isin(rows["task_index"],train_tasks)&rows["valid_h25"])
    test=_subset(rows,np.isin(rows["task_index"],eval_tasks)&rows["valid_h25"])
    if smoke:
        train_all=af._select_episodes(train_all,train_tasks,10,seed+100)
        test=af._select_episodes(test,eval_tasks,3,seed+200)
    fit_rows,d1=fc.holdout_split(train_all,split,seed,per_task=2 if smoke else fc.HOLDOUT_PER_TASK)
    fit_ix,val_ix=_episode_train_validation_indices(fit_rows,seed+401)
    fit=_subset(fit_rows,np.isin(np.arange(len(fit_rows["a"])),fit_ix))
    val=_subset(fit_rows,np.isin(np.arange(len(fit_rows["a"])),val_ix))
    fit_q,qs=_row_keys(fit); val_q,_=_row_keys(val,qs); d1_q,_=_row_keys(d1,qs); test_q,_=_row_keys(test,qs)
    limits=v2.derive_length_limits(fit)
    train_segments,row_phase=v2.make_segments(fit,fit_q,limits,stride_seconds)
    val_segments,_=v2.make_segments(val,val_q,limits,stride_seconds)
    model,z,zv,norm,fit_info=fit_scene_vae(
        fit,train_segments,val,val_segments,latent_dim=latent_dim,beta=beta,
        seed=seed+521,device=device,epochs=epochs,batch_size=batch_size,
        contrastive=contrastive)
    bank=LatentSkillBank(min_support=1 if smoke else 10,
                         min_episodes=1 if smoke else 3).fit(z,train_segments,seed=seed+617)
    # Segment keys are the only retrieval index. Labels are used only after fit.
    skill_keys=np.stack([np.mean(np.stack([train_segments[i]["key"] for i in range(len(train_segments)) if bank.assignments[i]==j]),0)
                         for j in range(len(bank.centers))]).astype(np.float32)
    skill_meta=build_skill_metadata(bank,train_segments)
    skill_verbs=np.asarray([s["majority_verb_class_posthoc"] for s in skill_meta],np.int64)
    fit_anchors,fit_conf,_,fit_truth=_make_flow_sources(
        model,bank,train_segments,fit,norm,row_phase,fit_q,seed=seed+659,device=device)
    actions=fit["a"][:,:v2.HORIZON]
    dim=int(np.prod(actions.shape[1:])); zero=np.zeros((len(actions),dim),np.float32)
    normal_sources=(zero,np.zeros(len(actions),np.float32),zero.copy(),np.zeros(len(actions),np.float32))
    normal_model,normal_norm,normal_info=af.fit_anchor_flow(
        fit,actions,normal_sources,source_mix=fc.NORMAL_MIX,seed=seed+827,
        epochs=flow_epochs,patience=8,batch_size=batch_size,device=device)
    flow,flow_norm,flow_info=v2._fit_skill_flow(
        fit,actions,fit_anchors,fit_conf,np.asarray(SOURCE_MIX,np.float32),
        seed=seed+827,epochs=flow_epochs,patience=8,batch_size=batch_size,
        device=device,initial_model=normal_model,initial_norm=normal_norm,
        teacher_model=normal_model,teacher_weight=2.0)
    rng=np.random.default_rng(seed*10007+37)
    outputs={}; audit={}
    for name,part,q in (("D1",d1,d1_q),("D2",test,test_q),("D3",test,test_q)):
        eval_segments,_=v2.make_segments(part,q,limits,stride_seconds)
        eps=rng.standard_normal((fc.SAMPLES,len(part["a"]),v2.HORIZON*7)).astype(np.float32)
        result=_eval_v3(part,q,qs,model,bank,skill_keys,flow,flow_norm,norm,normal_model,normal_norm,
                        eval_segments,eps,device=device,seed=seed,skill_verbs=skill_verbs)
        outputs[name]={k:v for k,v in result.items() if isinstance(v,(int,float,dict))}
        # Swap with a different segment latent while keeping the same scene/state.
        if len(eval_segments)>1:
            swap=np.arange(len(eval_segments)); swap=np.roll(swap,1)
            swap_skill=np.asarray([bank.assignments[swap[sid]] for sid in range(len(eval_segments))])
            row_seg,_=_segment_maps(eval_segments,len(part["a"]))
            ids=swap_skill[row_seg]
            phase=_segment_maps(eval_segments,len(part["a"]))[1]
            swapped_anchor=decoder_anchor(model,bank,ids,phase,part,norm,device=device)
            swapped=fc.rollout(flow,flow_norm,part["f"],part["s"],swapped_anchor,
                               eps,result["tau"],
                               n=np.maximum(2,np.rint(10*result["tau"]).astype(np.int64)),device=device)
            audit[name]={"anchor_swap_mse":float(np.mean((swapped_anchor-result["anchor"])**2)),
                         "action_swap_mse":float(np.mean((swapped-result["prediction"][None])**2))}
    return {"split":split,"seed":seed,"variant":f"V{2 if contrastive else 1}_z{latent_dim}_beta{beta:g}",
            "latent_dim":latent_dim,"beta":beta,"contrastive":contrastive,
            "skill_count":len(bank.centers),"epsilon":bank.epsilon,
            "skills":skill_meta,"vae":fit_info,
            "normal_flow":normal_info,"skill_flow":flow_info,"conditions":outputs,"swap":audit}


def run(*, output_root=str(RESEARCH_STATE_DIR / 'outputs/memory-pilot'), cache_dir=DEFAULT_CACHE,
        dataset_dir=DEFAULT_DATASET, seeds=(0,1,2), splits=("A","B"), device_name="cuda",
        smoke=False, epochs=25, flow_epochs=50, batch_size=512, threads=8, resume_from=None,
        recompute=False):
    torch.set_num_threads(int(threads))
    try: torch.set_num_interop_threads(1)
    except RuntimeError: pass
    device=torch.device(device_name)
    stamp=datetime.now(timezone.utc).strftime("skill-v3-%Y%m%dT%H%M%SZ")+"-"+uuid.uuid4().hex[:6]
    if smoke: stamp+="-smoke"
    out=Path(output_root)/stamp; out.mkdir(parents=True,exist_ok=False)
    rows=load_rows(Path(cache_dir),Path(dataset_dir),max_horizon=v2.HORIZON)
    manifest=json.loads((Path(cache_dir)/"manifest.json").read_text())
    info=json.loads((Path(dataset_dir)/"meta"/"info.json").read_text())
    stride_seconds=float(manifest["stride"])/float(info["fps"])
    task_ids=sorted({t for sp in splits for t in (*SPLITS[sp][0],*SPLITS[sp][1])})
    rows=_subset(rows,np.isin(rows["task_index"],task_ids)&rows["valid_h25"])
    results=[]; t0=time.perf_counter()
    variants=[(4,0.1,False)] if smoke else [(d,b,c) for d in LATENT_DIMS for b in BETA_VALUES for c in (False,True)]
    previous_duration=0.0; previous_runs=[]
    if resume_from is not None:
        previous=json.loads((Path(resume_from)/"results.json").read_text())
        previous_duration=float(previous.get("duration_minutes",0.0))
        previous_runs=[r for r in previous["runs"] if r["split"] in splits and int(r["seed"]) in seeds]
        if not recompute: results=previous_runs.copy()
    text,text_emb=fc.precompute_texts(rows,("instruction",),Path(cache_dir))
    for split in splits:
        for seed in seeds:
            wanted={f"V{2 if c else 1}_z{d}_beta{b:g}" for d,b,c in variants}
            present={r["variant"] for r in results if r["split"]==split and int(r["seed"])==int(seed)}
            if wanted.issubset(present) and not recompute: continue
            old_combo=[r for r in previous_runs if r["split"]==split and int(r["seed"])==int(seed)]
            old_ref=old_combo[0].get("baseline_reference") if old_combo else None
            # Reuse v2's fixed baselines exactly; all v3 methods use the same held-out split/noise convention.
            if recompute and old_ref is not None:
                baseline_reference=old_ref
            else:
                baseline,_,_=v2.run_one(rows,split=split,seed=int(seed),texts_by_source=text,
                    text_embeddings=text_emb,stride_seconds=stride_seconds,device=device,smoke=smoke,
                    m1_epochs=30 if smoke else 60,flow_epochs=20 if smoke else 80,
                    batch_size=batch_size,v21=True,v22=False)
                baseline_reference={c:{k:baseline["conditions"][c]["mean"][k]["mse"]
                    for k in ("normal","raw_interval","primitive_Pa","skill_v2")} for c in ("D1","D2","D3")}
                (out/"baseline.partial.json").write_text(json.dumps({"split":split,"seed":int(seed),"baseline":baseline},ensure_ascii=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x)))
            for latent_dim,beta,contrastive in variants:
                try:
                    result=_variant_run(rows,split,int(seed),device,stride_seconds=stride_seconds,latent_dim=latent_dim,beta=beta,
                        contrastive=contrastive,epochs=epochs,flow_epochs=flow_epochs,
                        batch_size=batch_size,smoke=smoke)
                except Exception as exc:
                    result={"split":split,"seed":int(seed),"variant":f"V{2 if contrastive else 1}_z{latent_dim}_beta{beta:g}",
                            "latent_dim":latent_dim,"beta":beta,"contrastive":contrastive,
                            "error":f"{type(exc).__name__}: {exc}"}
                result["baseline_reference"]=baseline_reference
                results.append(result)
                (out/"results.partial.json").write_text(json.dumps({"runs":results},ensure_ascii=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x)))
    elapsed=(time.perf_counter()-t0)/60
    payload={"seeds":list(seeds),"splits":list(splits),"variants":variants,
             "smoke":smoke,"duration_minutes":elapsed,"cumulative_duration_minutes":elapsed+previous_duration,
             "resumed_from":str(resume_from) if resume_from else None,"runs":results,
             "assignment_api":"evaluate_assignments(rows, query_keys, assignments, prototypes, anchor_builder, truth_assignments=...)"}
    (out/"results.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x)))
    (out/"summary.md").write_text(summarize(payload))
    (out/"results.partial.json").unlink(missing_ok=True)
    (out/"baseline.partial.json").unlink(missing_ok=True)
    return out


def summarize(payload):
    lines=["# v3 scene-conditioned skills: V1/V2", "",
           f"実行: Split={payload['splits']}、seed={payload['seeds']}、所要 {payload.get('cumulative_duration_minutes',payload['duration_minutes']):.1f} 分。",
           "βはKL重み、V2はV1に同一エピソード内別区間への分離と同一区間摂動への近接項を加えた。",
           "ラベルは学習・割当てに使わず、動詞純度などの事後解釈だけに使用。", "",
           "| Split | 変種 | skills | top1 D1/D3 | top5 D1/D3 | 動詞純度 | 選択動詞一致 D1/D3 | 中央スキル長 秒 | 平均タスク数/skill | D3 MSE | v2.1 D3 | D1悪化行率 | b入替え行動差 MSE |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for split in payload["splits"]:
        runs=[r for r in payload["runs"] if r["split"]==split and "error" not in r]
        keys=sorted({r["variant"] for r in runs})
        for name in keys:
            rs=[r for r in runs if r["variant"]==name]
            mean=lambda f:float(np.mean([f(r) for r in rs]))
            top1=mean(lambda r:r["conditions"]["D1"]["selection_accuracy"]["top1"])
            top3=mean(lambda r:r["conditions"]["D3"]["selection_accuracy"]["top1"])
            top5=mean(lambda r:r["conditions"]["D1"]["selection_accuracy"]["top5"])
            top53=mean(lambda r:r["conditions"]["D3"]["selection_accuracy"]["top5"])
            verb1=mean(lambda r:r["conditions"]["D1"]["verb_selection_accuracy_posthoc"] or 0.0)
            verb3=mean(lambda r:r["conditions"]["D3"]["verb_selection_accuracy_posthoc"] or 0.0)
            purity=mean(lambda r:np.mean([s["verb_purity_posthoc"] for s in r["skills"] if s["verb_purity_posthoc"] is not None]))
            duration=mean(lambda r:np.median([s["median_length_seconds"] for s in r["skills"]]))
            taskspan=mean(lambda r:np.mean([s["tasks"] for s in r["skills"]]))
            mse=mean(lambda r:r["conditions"]["D3"]["mse"])
            base=mean(lambda r:r["baseline_reference"]["D3"]["skill_v2"])
            worse=mean(lambda r:r["conditions"]["D1"]["D1_worse_row_fraction"])
            change=mean(lambda r:r["swap"].get("D3",{}).get("action_swap_mse",0.0))
            lines.append(f"| {split} | {name} | {mean(lambda r:r['skill_count']):.1f} | {top1:.2f}/{top3:.2f} | {top5:.2f}/{top53:.2f} | {purity:.2f} | {verb1:.2f}/{verb3:.2f} | {duration:.2f} | {taskspan:.2f} | {mse:.4f} | {base:.4f} | {worse:.2f} | {change:.4f} |")
    lines += ["", "## D1 / D2 / D3 の MSE", "", "| Split | 変種 | 条件 | 通常 F | 生区間 | P-a | v2.1 | v3 |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for split in payload["splits"]:
        for name in sorted({r["variant"] for r in payload["runs"] if r["split"]==split}):
            rs=[r for r in payload["runs"] if r["split"]==split and r["variant"]==name]
            rs=[r for r in rs if "error" not in r]
            if not rs: continue
            for cond in ("D1","D2","D3"):
                base={k:float(np.mean([r["baseline_reference"][cond][k] for r in rs])) for k in ("normal","raw_interval","primitive_Pa","skill_v2")}
                method=float(np.mean([r["conditions"][cond]["mse"] for r in rs]))
                lines.append(f"| {split} | {name} | {cond} | {base['normal']:.4f} | {base['raw_interval']:.4f} | {base['primitive_Pa']:.4f} | {base['skill_v2']:.4f} | {method:.4f} |")
    failures=[r for r in payload["runs"] if "error" in r]
    lines += ["", "baseline_reference は同じ v2 評価ルーチンの比較値。V3 D2 は現時点でオンライン挿入をせず、D3 と同じ固定メモリで評価しているため、D2のメソッド比較は参考値.", ""]
    if failures:
        lines += ["学習/形成に失敗した変種（支持条件を満たす単位なし等）:"]
        lines += [f"- {r['split']} seed {r['seed']} {r['variant']}: {r['error']}" for r in failures]
    return "\n".join(lines)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds",default="0,1,2"); p.add_argument("--splits",default="A,B")
    p.add_argument("--smoke",action="store_true"); p.add_argument("--device",default="cuda")
    p.add_argument("--epochs",type=int,default=25); p.add_argument("--flow-epochs",type=int,default=50)
    p.add_argument("--batch-size",type=int,default=512); p.add_argument("--threads",type=int,default=8)
    p.add_argument("--output-root",default=str(RESEARCH_STATE_DIR / 'outputs/memory-pilot'))
    p.add_argument("--resume-from",default=None,help="reuse completed split/seed runs from a prior skill-v3 output")
    p.add_argument("--recompute",action="store_true",help="rerun v3 variants while reusing baseline values from --resume-from")
    args=p.parse_args(argv)
    out=run(seeds=tuple(map(int,args.seeds.split(','))),splits=tuple(args.splits.split(',')),
            smoke=args.smoke,device_name=args.device,epochs=args.epochs,flow_epochs=args.flow_epochs,
            batch_size=args.batch_size,threads=args.threads,output_root=args.output_root,resume_from=args.resume_from,
            recompute=args.recompute)
    print(out)


if __name__ == "__main__": main()
