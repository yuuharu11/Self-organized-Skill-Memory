"""Representation follow-up E4/E5 for the reviewer-defined V3 experiment."""
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
from . import skill_v3 as v3
from . import skill_v2 as v2
from .splits import SPLITS, subset_rows as _subset
from .experiment import (DEFAULT_CACHE, DEFAULT_DATASET,
                         _episode_train_validation_indices, _standardize,
                         _standardize_fit, load_rows)
from ....research_paths import REPO_ROOT as RESEARCH_REPO_ROOT, STATE_DIR as RESEARCH_STATE_DIR

ALPHAS = (0.5, 0.8)
Z_DIMS = (4, 8)
E5_BETA = 0.01
E4_TEMPERATURE = 0.1
E4_SINKHORN_ITERS = 3
E4_MAX_PROTOTYPES = 64
BASELINE_RESULTS = Path(str(RESEARCH_STATE_DIR / 'outputs/memory-pilot/skill-v3-20261029T132313Z-2cba70/results.json'))


def _kl_diag(qmu, qlv, pmu, plv):
    return 0.5 * (plv - qlv + (qlv.exp() + (qmu - pmu).square()) / plv.exp() - 1).sum(-1)


class ConditionalPriorVAE(nn.Module):
    """E5: q(z|A), p(z|O); decoder also sees O and phase."""
    def __init__(self, zdim, obs_dim, key_dim, state_dim, action_dim):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(224,256),nn.GELU(),nn.Linear(256,128),nn.GELU())
        self.qmu=nn.Linear(128,zdim); self.qlv=nn.Linear(128,zdim)
        self.prior=nn.Sequential(nn.Linear(obs_dim,256),nn.GELU(),nn.Linear(256,128),nn.GELU())
        self.pmu=nn.Linear(128,zdim); self.plv=nn.Linear(128,zdim)
        self.key_proj=nn.Sequential(nn.Linear(key_dim,64),nn.GELU())
        self.state_proj=nn.Sequential(nn.Linear(state_dim,32),nn.GELU())
        self.dec=nn.Sequential(nn.Linear(zdim+97,256),nn.GELU(),nn.Linear(256,256),nn.GELU(),nn.Linear(256,action_dim))

    def encode(self, curve):
        h=self.enc(curve.reshape(len(curve),-1)); return self.qmu(h),self.qlv(h).clamp(-8,6)
    def prior_params(self, obs):
        h=self.prior(obs); return self.pmu(h),self.plv(h).clamp(-8,6)
    def decode(self,z,key,state,phase):
        return self.dec(torch.cat((z,self.key_proj(key),self.state_proj(state),phase.reshape(-1,1)),dim=-1))
    def forward(self,curve,obs,key,state,phase):
        qm,ql=self.encode(curve); pm,pl=self.prior_params(obs)
        z=qm+torch.randn_like(qm)*torch.exp(.5*ql)
        return self.decode(z,key,state,phase),qm,ql,pm,pl


class CrossViewCluster(nn.Module):
    """E4 two-view SwAV/IIC style model with a context-conditioned decoder."""
    def __init__(self, obs_dim, key_dim, state_dim, action_dim, prototypes, embed_dim=64):
        super().__init__()
        self.g=nn.Sequential(nn.Linear(224,256),nn.GELU(),nn.Linear(256,128),nn.GELU(),nn.Linear(128,embed_dim))
        self.f=nn.Sequential(nn.Linear(obs_dim,256),nn.GELU(),nn.Linear(256,128),nn.GELU(),nn.Linear(128,embed_dim))
        self.prototypes=nn.Parameter(F.normalize(torch.randn(prototypes,embed_dim),dim=-1))
        self.key_proj=nn.Sequential(nn.Linear(key_dim,64),nn.GELU())
        self.state_proj=nn.Sequential(nn.Linear(state_dim,32),nn.GELU())
        self.dec=nn.Sequential(nn.Linear(embed_dim+97,256),nn.GELU(),nn.Linear(256,256),nn.GELU(),nn.Linear(256,action_dim))
    def action_embed(self,curve): return F.normalize(self.g(curve.reshape(len(curve),-1)),dim=-1)
    def obs_embed(self,obs): return F.normalize(self.f(obs),dim=-1)
    def logits_action(self,curve,proto=None):
        c=F.normalize(self.prototypes if proto is None else proto,dim=-1)
        return self.action_embed(curve)@c.T
    def logits_obs(self,obs,proto=None):
        c=F.normalize(self.prototypes if proto is None else proto,dim=-1)
        return self.obs_embed(obs)@c.T
    def decode(self,z,key,state,phase):
        return self.dec(torch.cat((z,self.key_proj(key),self.state_proj(state),phase.reshape(-1,1)),dim=-1))


def sinkhorn_assign(logits, temperature=E4_TEMPERATURE, iterations=E4_SINKHORN_ITERS):
    """Balanced batch assignments, rows sum to one; stable for small batches."""
    if logits.ndim!=2: raise ValueError("logits must be (batch, prototypes)")
    b,k=logits.shape
    q=torch.exp((logits/temperature-logits.max()).T).clamp_min(1e-12)
    q=q/q.sum()
    for _ in range(int(iterations)):
        q=q/q.sum(dim=1,keepdim=True).clamp_min(1e-12)/k
        q=q/q.sum(dim=0,keepdim=True).clamp_min(1e-12)/b
    return (q*b).T


def fit_conditional_prior(rows, segments, validation_rows, validation_segments, obs_train, obs_val, *,
                          zdim, alpha, beta=E5_BETA, seed=0, device="cuda", epochs=8, batch_size=512):
    """Fit E5 with Dreamer-style stop-gradient KL balancing."""
    torch.manual_seed(seed); device=torch.device(device)
    curves=np.stack([s["curve"] for s in segments]).astype(np.float32); vcurves=np.stack([s["curve"] for s in validation_segments]).astype(np.float32)
    cm,cs=v3._curve_norm(curves); curves=((curves-cm)/cs).astype(np.float32); vcurves=((vcurves-cm)/cs).astype(np.float32)
    actions=rows["a"][:,:v2.HORIZON].reshape(len(rows["a"]),-1).astype(np.float32); am,ass=_standardize_fit(actions); actions=_standardize(actions,am,ass)
    va=validation_rows["a"][:,:v2.HORIZON].reshape(len(validation_rows["a"]),-1).astype(np.float32); va=_standardize(va,am,ass)
    km,ks=_standardize_fit(rows["k"]); sm,ss=_standardize_fit(rows["s"])
    key=_standardize(rows["k"],km,ks); state=_standardize(rows["s"],sm,ss)
    vk=_standardize(validation_rows["k"],km,ks); vs=_standardize(validation_rows["s"],sm,ss)
    om,os=_standardize_fit(obs_train); obs=_standardize(obs_train,om,os); vobs=_standardize(obs_val,om,os)
    rid,phase=v3._segment_maps(segments,len(rows["a"])); vrid,vphase=v3._segment_maps(validation_segments,len(validation_rows["a"]))
    if np.any(rid<0) or np.any(vrid<0): raise ValueError("segment map has uncovered rows")
    model=ConditionalPriorVAE(zdim,obs.shape[1],key.shape[1],state.shape[1],actions.shape[1]).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=1e-5); gen=torch.Generator(device=device).manual_seed(seed+19)
    tc=torch.as_tensor(curves,device=device); tr=torch.as_tensor(rid,device=device); tp=torch.as_tensor(phase,device=device)
    vae=torch.as_tensor(actions,device=device); kt=torch.as_tensor(key,device=device); st=torch.as_tensor(state,device=device); ot=torch.as_tensor(obs,device=device)
    vtc=torch.as_tensor(vcurves,device=device); vtr=torch.as_tensor(vrid,device=device); vtp=torch.as_tensor(vphase,device=device)
    vkt=torch.as_tensor(vk,device=device); vst=torch.as_tensor(vs,device=device); vot=torch.as_tensor(vobs,device=device); vat=torch.as_tensor(va,device=device)
    best=float("inf"); best_state=None; stale=0; hist=[]
    for ep in range(epochs):
        order=torch.randperm(len(actions),generator=gen,device=device); losses=[]; model.train()
        for lo in range(0,len(order),batch_size):
            ix=order[lo:lo+batch_size]; sid=tr[ix]
            rec,qm,ql,pm,pl=model(tc[sid],ot[ix],kt[ix],st[ix],tp[ix])
            kl_prior=_kl_diag(qm.detach(),ql.detach(),pm,pl).mean()
            kl_post=_kl_diag(qm,ql,pm.detach(),pl.detach()).mean()
            loss=F.mse_loss(rec,vae[ix])+float(beta)*(float(alpha)*kl_prior+(1-float(alpha))*kl_post)
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            qmu,_=model.encode(vtc[vtr]); pred=model.decode(qmu,vkt,vst,vtp); val=float(F.mse_loss(pred,vat).cpu())
        hist.append({"epoch":ep+1,"train_loss":float(np.mean(losses)),"validation_reconstruction":val})
        if val<best-1e-5: best=val;best_state={k:x.detach().clone() for k,x in model.state_dict().items()};stale=0
        else:
            stale+=1
            if stale>=6: break
    if best_state:model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad(): z,_=model.encode(torch.as_tensor(curves,device=device)); vz,_=model.encode(vtc)
    stats={"curve_mean":cm,"curve_scale":cs,"action_mean":am,"action_scale":ass,"key_mean":km,"key_scale":ks,"state_mean":sm,"state_scale":ss,"obs_mean":om,"obs_scale":os}
    info={"best_validation_reconstruction":best,"epochs":len(hist),"history":hist,"alpha":alpha,"beta":beta}
    return model,z.cpu().numpy().astype(np.float32),vz.cpu().numpy().astype(np.float32),stats,info


def _e4_initial_prototypes(curves,segments,*,seed):
    # M6/DP-means supplies a data-derived active K; 64 is only a ceiling.
    flat=np.asarray(curves,np.float32).reshape(len(curves),-1)
    init=v3.LatentSkillBank(kappa=.1,min_support=10,min_episodes=3).fit(flat,segments,seed=seed)
    members=[np.flatnonzero(init.assignments==j) for j in range(len(init.centers))]
    members=sorted(members,key=len,reverse=True)[:E4_MAX_PROTOTYPES]
    return members


def fit_e4(rows,segments,validation_rows,validation_segments,obs_train,obs_val,*,seed=0,device="cuda",epochs=15,batch_size=512):
    torch.manual_seed(seed);device=torch.device(device)
    curves=np.stack([s["curve"] for s in segments]).astype(np.float32);vcurves=np.stack([s["curve"] for s in validation_segments]).astype(np.float32)
    cm,cs=v3._curve_norm(curves);curves=((curves-cm)/cs).astype(np.float32);vcurves=((vcurves-cm)/cs).astype(np.float32)
    groups=_e4_initial_prototypes(curves,segments,seed=seed+17); k=len(groups)
    if k<2: raise RuntimeError(f"E4 M6 initialization yielded only K={k}")
    actions=rows["a"][:,:v2.HORIZON].reshape(len(rows["a"]),-1).astype(np.float32);am,ass=_standardize_fit(actions);actions=_standardize(actions,am,ass)
    va=validation_rows["a"][:,:v2.HORIZON].reshape(len(validation_rows["a"]),-1).astype(np.float32);va=_standardize(va,am,ass)
    km,ks=_standardize_fit(rows["k"]);sm,ss=_standardize_fit(rows["s"]);key=_standardize(rows["k"],km,ks);state=_standardize(rows["s"],sm,ss)
    vk=_standardize(validation_rows["k"],km,ks);vs=_standardize(validation_rows["s"],sm,ss)
    om,os=_standardize_fit(obs_train);obs=_standardize(obs_train,om,os);vobs=_standardize(obs_val,om,os)
    rid,phase=v3._segment_maps(segments,len(rows["a"]));vrid,vphase=v3._segment_maps(validation_segments,len(validation_rows["a"]))
    if np.any(rid<0) or np.any(vrid<0):raise ValueError("segment map has uncovered rows")
    model=CrossViewCluster(obs.shape[1],key.shape[1],state.shape[1],actions.shape[1],k).to(device)
    with torch.no_grad():
        zz=model.action_embed(torch.as_tensor(curves,device=device))
        c=torch.stack([zz[ix].mean(0) for ix in groups]);model.prototypes.copy_(F.normalize(c,dim=-1))
    opt=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=1e-5);gen=torch.Generator(device=device).manual_seed(seed+29)
    tc=torch.as_tensor(curves,device=device);tr=torch.as_tensor(rid,device=device);tp=torch.as_tensor(phase,device=device)
    at=torch.as_tensor(actions,device=device);kt=torch.as_tensor(key,device=device);st=torch.as_tensor(state,device=device);ot=torch.as_tensor(obs,device=device)
    vtc=torch.as_tensor(vcurves,device=device);vtr=torch.as_tensor(vrid,device=device);vobs_t=torch.as_tensor(vobs,device=device)
    vkt=torch.as_tensor(vk,device=device);vst=torch.as_tensor(vs,device=device);vphase_t=torch.as_tensor(vphase,device=device);vat=torch.as_tensor(va,device=device)
    best=float("inf");best_state=None;stale=0;hist=[]
    vstarts=torch.as_tensor([s["rows"][0] for s in validation_segments],device=device,dtype=torch.long)
    for ep in range(epochs):
        order=torch.randperm(len(actions),generator=gen,device=device);losses=[];model.train()
        for lo in range(0,len(order),batch_size):
            ix=order[lo:lo+batch_size]; sid=tr[ix]
            # Batch includes one assignment per row; assignment views are repeated
            # for temporal rows but retain the same segment identity.
            al=model.logits_action(tc[sid]);ol=model.logits_obs(ot[sid])
            qa=sinkhorn_assign(al).detach();qo=sinkhorn_assign(ol).detach()
            cross=-(qa*F.log_softmax(ol/E4_TEMPERATURE,dim=-1)).sum(-1).mean()
            cross+=-(qo*F.log_softmax(al/E4_TEMPERATURE,dim=-1)).sum(-1).mean()
            code=qa@F.normalize(model.prototypes,dim=-1)
            rec=model.decode(code,kt[ix],st[ix],tp[ix])
            loss=cross+.5*F.mse_loss(rec,at[ix])
            opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step();losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            al=model.logits_action(vtc);ol=model.logits_obs(vobs_t[vstarts]);qa=sinkhorn_assign(al);qo=sinkhorn_assign(ol)
            pred=model.decode((qa[vtr]@F.normalize(model.prototypes,dim=-1)),vkt[vtr],vst[vtr],vphase_t)
            val=float(F.mse_loss(pred,vat).cpu())
        hist.append({"epoch":ep+1,"train_loss":float(np.mean(losses)),"validation_reconstruction":val})
        if val<best-1e-5:best=val;best_state={k:x.detach().clone() for k,x in model.state_dict().items()};stale=0
        else:
            stale+=1
            if stale>=6:break
    if best_state:model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_logits=model.logits_action(torch.as_tensor(curves,device=device));train_assignment=train_logits.argmax(-1).cpu().numpy()
        train_embedding=model.action_embed(torch.as_tensor(curves,device=device)).cpu().numpy().astype(np.float32)
        active=np.unique(train_assignment)
        eval_z=None
    # Remove unused prototypes after cross-view training.
    old_to_new={int(old):j for j,old in enumerate(active)}
    remapped=np.asarray([old_to_new[int(x)] for x in train_assignment],np.int64)
    active_centers=F.normalize(model.prototypes.detach()[torch.as_tensor(active,device=device)],dim=-1).cpu().numpy()
    bank=PrototypeBank(active_centers,remapped)
    stats={"curve_mean":cm,"curve_scale":cs,"action_mean":am,"action_scale":ass,"key_mean":km,"key_scale":ks,"state_mean":sm,"state_scale":ss,"obs_mean":om,"obs_scale":os,"active_prototypes":active.tolist()}
    return model,bank,train_embedding,stats,{"active_k":len(active),"initial_k":k,"history":hist,"best_validation_reconstruction":best}


class PrototypeBank:
    def __init__(self,centers,assignments): self.centers=np.asarray(centers,np.float32);self.assignments=np.asarray(assignments,np.int64)


def _observations(rows,stats=None):
    raw=np.concatenate([rows["k"],rows["s"],rows["subtask_embedding"]],axis=-1).astype(np.float32)
    if stats is None: m,s=_standardize_fit(raw)
    else:m,s=stats
    return _standardize(raw,m,s),(m,s)


def _context(row_slice,stats):
    k=_standardize(row_slice["k"],stats["key_mean"],stats["key_scale"])
    s=_standardize(row_slice["s"],stats["state_mean"],stats["state_scale"])
    return k,s


def _cluster_membership_metrics(train_z,train_assign,eval_z,selected,centers):
    train_z=np.asarray(train_z,np.float32);eval_z=np.asarray(eval_z,np.float32);centers=np.asarray(centers,np.float32)
    k=len(centers); dcenter=np.mean((eval_z[:,None,:]-centers[None,:,:])**2,axis=-1);truth=dcenter.argmin(1)
    eps=[];members=[];within=[]
    for j in range(k):
        ix=np.flatnonzero(np.asarray(train_assign)==j);members.append(ix)
        if not len(ix):eps.append(0.0);within.append(np.zeros(0));continue
        dist=np.mean((train_z[ix]-centers[j])**2,axis=-1);within.append(dist);eps.append(float(np.quantile(dist,.9)))
    nearest=np.full((len(eval_z),k),np.inf,np.float32)
    for j,ix in enumerate(members):
        if len(ix):nearest[:,j]=np.min(np.mean((eval_z[:,None,:]-train_z[ix][None,:,:])**2,axis=-1),axis=1)
    hit_matrix=nearest<=np.asarray(eps)[None,:]
    has_any=hit_matrix.any(1);sel=np.asarray(selected,np.int64);i=np.arange(len(sel))
    strict=(sel==truth);functional=hit_matrix[i,sel]
    pred_dist=np.bincount(sel,minlength=k)/max(len(sel),1);true_dist=np.bincount(truth,minlength=k)/max(len(truth),1)
    strict_kappa=float((strict.mean()-np.dot(pred_dist,true_dist))/(1-np.dot(pred_dist,true_dist)+1e-12))
    per_unit_function=hit_matrix.mean(0);chance_func=float(np.dot(pred_dist,per_unit_function))
    functional_kappa=float((functional.mean()-chance_func)/(1-chance_func+1e-12))
    train_dist=np.bincount(np.asarray(train_assign),minlength=k)/max(len(train_assign),1)
    freq_expected=float(np.dot(true_dist,train_dist))
    majority=float(np.max(true_dist))
    return {"truth":truth,"strict_accuracy":float(strict.mean()),"strict_kappa":strict_kappa,
            "functional_accuracy_all":float(functional.mean()),
            "functional_accuracy_covered":float(functional[has_any].mean()) if has_any.any() else None,
            "functional_coverage":float(has_any.mean()),"no_match_rate":float((~has_any).mean()),
            "functional_kappa":functional_kappa,"majority_baseline":majority,
            "frequency_proportional_baseline":freq_expected,"functional_excess_over_majority":float(functional.mean()-majority),
            "selected_members_min_distance":nearest[i,sel],"functional_eps_by_skill":eps}


def _encode_e5(model,segments,stats,device):
    curves=np.stack([s["curve"] for s in segments]).astype(np.float32);curves=_standardize(curves,stats["curve_mean"],stats["curve_scale"])
    with torch.no_grad(): mu,_=model.encode(torch.as_tensor(curves,device=device))
    return mu.cpu().numpy().astype(np.float32)


def _prior_segment_means(model,obs,segments,stats,device):
    starts=np.asarray([s["rows"][0] for s in segments],np.int64)
    raw=np.concatenate([obs[0][starts],obs[1][starts]],axis=-1) # placeholder; caller passes standardized prior vectors below
    with torch.no_grad(): pm,_=model.prior_params(torch.as_tensor(raw,device=device))
    return pm.cpu().numpy().astype(np.float32)


def _e5_prior_means(model,obs_vectors,segments,device):
    starts=np.asarray([s["rows"][0] for s in segments],np.int64)
    with torch.no_grad(): pm,_=model.prior_params(torch.as_tensor(obs_vectors[starts],device=device))
    return pm.cpu().numpy().astype(np.float32)


def _encode_e4_action(model,segments,stats,device):
    curves=np.stack([s["curve"] for s in segments]).astype(np.float32);curves=_standardize(curves,stats["curve_mean"],stats["curve_scale"])
    with torch.no_grad(): z=model.action_embed(torch.as_tensor(curves,device=device))
    return z.cpu().numpy().astype(np.float32)


def _e4_obs_predictions(model,obs_vectors,segments,active_proto,device):
    starts=np.asarray([s["rows"][0] for s in segments],np.int64)
    with torch.no_grad(): logits=model.logits_obs(torch.as_tensor(obs_vectors[starts],device=device))[:,torch.as_tensor(active_proto,device=device)]
    return logits.cpu().numpy()


def _make_anchor_sources(model,bank,segments,rows,norm,phase,query,*,seed,device,obs_scores=None):
    n=len(rows["a"]);row_seg,_=v3._segment_maps(segments,n);truth=bank.assignments[row_seg]
    qn=query/np.maximum(np.linalg.norm(query,axis=1,keepdims=True),1e-8)
    if obs_scores is None:
        skill_keys=np.stack([np.mean(np.stack([segments[i]["key"] for i in range(len(segments)) if bank.assignments[i]==j]),0) for j in range(len(bank.centers))])
        kn=skill_keys/np.maximum(np.linalg.norm(skill_keys,axis=1,keepdims=True),1e-8);scores=qn@kn.T
    else:scores=np.asarray(obs_scores,np.float32)
    ranking=np.argsort(-scores,axis=1);alt=np.asarray([ranking[i,1] if scores.shape[1]>1 else ranking[i,0] for i in range(n)])
    rng=np.random.default_rng(seed);random=rng.integers(len(bank.centers),size=n)
    shifted=np.clip(phase+rng.choice([-1,1],n)*rng.uniform(.15,.45,n),0,1).astype(np.float32)
    src=np.zeros((5,n,v2.HORIZON,7),np.float32)
    src[0]=v3.decoder_anchor(model,bank,truth,phase,rows,norm,device=device)
    src[1]=v3.decoder_anchor(model,bank,truth,shifted,rows,norm,device=device)
    src[2]=v3.decoder_anchor(model,bank,alt,phase,rows,norm,device=device)
    src[3]=v3.decoder_anchor(model,bank,random,phase,rows,norm,device=device)
    conf=np.zeros((5,n),np.float32)
    for t,ids in enumerate((truth,truth,alt,random)):
        conf[t]=np.clip((scores[np.arange(n),ids]-scores.min(1))/(scores.max(1)-scores.min(1)+1e-6),0,1)
    return src,conf,scores


def _eval_variant(rows,qobs,segments,model,bank,train_segments,train_z,train_assign,stats,flow,flow_norm,selection_scores,selector_kind,eps,*,device):
    n=len(rows["a"]);row_seg,phase=v3._segment_maps(segments,n)
    if selector_kind=="prior":
        chosen=np.mean((selection_scores[:,None,:]-bank.centers[None,:,:])**2,axis=-1).argmin(1)
    else:chosen=np.asarray(selection_scores).argmax(1)
    row_selected=chosen[row_seg]
    anchors=v3.decoder_anchor(model,bank,row_selected,phase,rows,stats,device=device)
    met=_cluster_membership_metrics(train_z,train_assign,_eval_variant.eval_z,chosen,bank.centers)
    conf=np.max(selection_scores,axis=1) if selector_kind!="prior" else -np.min(np.mean((selection_scores[:,None,:]-bank.centers[None])**2,axis=-1),axis=1)
    conf=np.asarray(conf,np.float32);conf=(conf-conf.min())/(conf.max()-conf.min()+1e-6);tau=np.clip(1-conf,.1,1)
    target=rows["a"][:,:v2.HORIZON];pred=fc.rollout(flow,flow_norm,rows["f"],rows["s"],anchors,eps,tau[row_seg],n=np.maximum(2,np.rint(10*tau[row_seg]).astype(np.int64)),device=device)
    normal=fc.rollout(_eval_variant.normal_model,_eval_variant.normal_norm,rows["f"],rows["s"],None,eps,np.ones(n,np.float32),n=np.full(n,10,np.int64),device=device)
    row_error=np.mean((pred-target[None])**2,axis=(0,2,3));normal_row=np.mean((normal-target[None])**2,axis=(0,2,3))
    oracle=met["truth"][row_seg];oracle_anchor=v3.decoder_anchor(model,bank,oracle,phase,rows,stats,device=device)
    oracle_pred=fc.rollout(flow,flow_norm,rows["f"],rows["s"],oracle_anchor,eps,tau[row_seg],n=np.maximum(2,np.rint(10*tau[row_seg]).astype(np.int64)),device=device)
    selected_mse=float(np.mean((pred-target[None])**2));oracle_mse=float(np.mean((oracle_pred-target[None])**2))
    met.update({"selected_mse":selected_mse,"oracle_mse":oracle_mse,"regret_mse":selected_mse-oracle_mse,
                "normal_mse":float(np.mean((normal-target[None])**2)),"D1_worse_row_fraction":float(np.mean(row_error>normal_row)),
                "skill_ids":chosen,"anchor":anchors,"prediction":pred.mean(0)})
    return met


def _run_variant(rows,obs_rows,split,seed,stride,device,*,kind,zdim=None,alpha=None,selector_mode="prior",epochs=8,flow_epochs=20,batch_size=512):
    train_tasks,eval_tasks,_=SPLITS[split];train_all=_subset(rows,np.isin(rows["task_index"],train_tasks)&rows["valid_h25"]);test=_subset(rows,np.isin(rows["task_index"],eval_tasks)&rows["valid_h25"])
    fit_rows,d1=fc.holdout_split(train_all,split,seed,per_task=fc.HOLDOUT_PER_TASK)
    fit_ix,val_ix=_episode_train_validation_indices(fit_rows,seed+401);fit=_subset(fit_rows,np.isin(np.arange(len(fit_rows["a"])),fit_ix));val=_subset(fit_rows,np.isin(np.arange(len(fit_rows["a"])),val_ix))
    # observation vectors are label-free: language-free image key, state and whole instruction embedding
    obsfit,obsstats=v3._row_keys(fit); obsfit=np.concatenate([fit["k"],fit["s"],fit["subtask_embedding"]],-1)
    obsv=np.concatenate([val["k"],val["s"],val["subtask_embedding"]],-1);obsd1=np.concatenate([d1["k"],d1["s"],d1["subtask_embedding"]],-1);obstest=np.concatenate([test["k"],test["s"],test["subtask_embedding"]],-1)
    om,os=_standardize_fit(obsfit);obsfit=_standardize(obsfit,om,os);obsv=_standardize(obsv,om,os);obstest=_standardize(obstest,om,os);obsd1=_standardize(obsd1,om,os)
    fit_q,_=v3._row_keys(fit);val_q,_=v3._row_keys(val,v3._row_keys(fit)[1]);d1_q,_=v3._row_keys(d1,v3._row_keys(fit)[1]);test_q,_=v3._row_keys(test,v3._row_keys(fit)[1])
    limits=v2.derive_length_limits(fit);train_seg,phase=v2.make_segments(fit,fit_q,limits,stride);val_seg,_=v2.make_segments(val,val_q,limits,stride)
    if kind=="E5":
        model,z,_,stats,learn=fit_conditional_prior(fit,train_seg,val,val_seg,obsfit,obsv,zdim=zdim,alpha=alpha,seed=seed+521,device=device,epochs=epochs,batch_size=batch_size)
        bank=v3.LatentSkillBank(kappa=.1,min_support=10,min_episodes=3).fit(z,train_seg,seed=seed+617)
        train_assign=bank.assignments;train_z=z; skill_meta=v3.build_skill_metadata(bank,train_seg)
    else:
        model,bank,train_z,stats,learn=fit_e4(fit,train_seg,val,val_seg,obsfit,obsv,seed=seed+521,device=device,epochs=epochs,batch_size=batch_size)
        train_assign=bank.assignments;skill_meta=[{"skill_id":j,"support":int(np.sum(train_assign==j)),"episodes":len({train_seg[i]["episode_id"] for i in np.flatnonzero(train_assign==j)}),"tasks":len({train_seg[i]["task_index"] for i in np.flatnonzero(train_assign==j)}),"verb_purity_posthoc":float(max(np.bincount([train_seg[i]["skill_class"] for i in np.flatnonzero(train_assign==j) if train_seg[i]["skill_class"]>=0]))/max(1,sum(train_seg[i]["skill_class"]>=0 for i in np.flatnonzero(train_assign==j)))) if any(train_seg[i]["skill_class"]>=0 for i in np.flatnonzero(train_assign==j)) else None,"median_length_seconds":float(np.median([train_seg[i]["length_seconds"] for i in np.flatnonzero(train_assign==j)]))} for j in range(len(bank.centers))]
    # Build image/state keyed training anchors and fit the shared v2.1 flow recipe.
    if kind=="E5":
        obs_train=torch.as_tensor(obsfit,device=device)
        with torch.no_grad(): seg_prior,_=model.prior_params(obs_train[torch.as_tensor([s["rows"][0] for s in train_seg],device=device)])
        prior=seg_prior.cpu().numpy(); key_scores=np.stack([np.mean(np.stack([s["key"] for i,s in enumerate(train_seg) if train_assign[i]==j]),0) for j in range(len(bank.centers))]); qstarts=np.stack([s["key"] for s in train_seg]); sim=qstarts@key_scores.T
        selector="prior"
        scores=sim
    else:
        selector="logits"
        starts=np.asarray([s["rows"][0] for s in train_seg]);scores=_e4_obs_predictions(model,obsfit,train_seg,stats["active_prototypes"],device)
        prior=None
    src,conf,_=_make_anchor_sources(model,bank,train_seg,fit,stats,phase,fit_q,seed=seed+659,device=device,obs_scores=scores.repeat_interleave(1,axis=0) if len(scores)==len(fit) else None)
    actions=fit["a"][:,:v2.HORIZON];dim=actions.shape[1]*actions.shape[2];zero=np.zeros((len(actions),dim),np.float32)
    normal_sources=(zero,np.zeros(len(actions),np.float32),zero.copy(),np.zeros(len(actions),np.float32))
    normal_model,normal_norm,_=af.fit_anchor_flow(fit,actions,normal_sources,source_mix=fc.NORMAL_MIX,seed=seed+827,epochs=flow_epochs,patience=8,batch_size=batch_size,device=device)
    flow,flow_norm,flow_info=v2._fit_skill_flow(fit,actions,src,conf,np.asarray(v2.V21_SOURCE_MIX,np.float32),seed=seed+827,epochs=flow_epochs,patience=8,batch_size=batch_size,device=device,initial_model=normal_model,initial_norm=normal_norm,teacher_model=normal_model,teacher_weight=2.0)
    results={};audit={};rng=np.random.default_rng(seed*10007+37)
    for cname,part,obs_part,q in (("D1",d1,obsd1,d1_q),("D2",test,obstest,test_q),("D3",test,obstest,test_q)):
        seg,_=v2.make_segments(part,q,limits,stride);eval_z=_encode_e5(model,seg,stats,device) if kind=="E5" else _encode_e4_action(model,seg,stats,device)
        starts=np.asarray([s["rows"][0] for s in seg],np.int64)
        if kind=="E5":
            with torch.no_grad(): pmean,_=model.prior_params(torch.as_tensor(obs_part[starts],device=device))
            prior_eval=pmean.cpu().numpy(); key_scores=np.stack([np.mean(np.stack([s["key"] for i,s in enumerate(train_seg) if train_assign[i]==j]),0) for j in range(len(bank.centers))]); qstarts=np.stack([s["key"] for s in seg]); old_scores=qstarts@(key_scores/np.maximum(np.linalg.norm(key_scores,axis=1,keepdims=True),1e-8)).T
            if selector_mode=="prior":scores_eval=prior_eval; selected=np.mean((prior_eval[:,None,:]-bank.centers[None,:,:])**2,axis=-1).argmin(1)
            elif selector_mode=="key_cosine":scores_eval=old_scores;selected=old_scores.argmax(1)
            else:raise ValueError(f"unknown E5 selector: {selector_mode}")
            mode=selector_mode
        else:
            scores_eval=_e4_obs_predictions(model,obs_part,seg,stats["active_prototypes"],device); selected=scores_eval.argmax(1);mode="logits"
        # Keep only one selected ID per segment; the evaluator accepts either selector form.
        _run_variant.eval_z=eval_z; _run_variant.normal_model=normal_model;_run_variant.normal_norm=normal_norm
        eps=rng.standard_normal((fc.SAMPLES,len(part["a"]),v2.HORIZON*7)).astype(np.float32)
        # Decode and denoise using selected skills, then compute reviewer metrics at interval granularity.
        row_seg,ph=v3._segment_maps(seg,len(part["a"])); rowids=selected[row_seg]
        anchors=v3.decoder_anchor(model,bank,rowids,ph,part,stats,device=device)
        strict_truth=np.mean((eval_z[:,None,:]-bank.centers[None,:,:])**2,axis=-1).argmin(1)
        distances=np.mean((eval_z[:,None,:]-bank.centers[None,:,:])**2,axis=-1)
        if kind=="E5":
            conf=-np.min(np.mean((pmean.cpu().numpy()[:,None,:]-bank.centers[None,:,:])**2,axis=-1),axis=1)
        else:conf=np.max(scores_eval,axis=1)
        conf=(conf-conf.min())/(conf.max()-conf.min()+1e-6);tau=np.clip(1-conf,.1,1).astype(np.float32)
        rowtau=tau[row_seg];pred=fc.rollout(flow,flow_norm,part["f"],part["s"],anchors,eps,rowtau,n=np.maximum(2,np.rint(10*rowtau).astype(np.int64)),device=device)
        normal=fc.rollout(normal_model,normal_norm,part["f"],part["s"],None,eps,np.ones(len(part["a"]),np.float32),n=np.full(len(part["a"]),10,np.int64),device=device)
        truth_rows=strict_truth
        metric=_cluster_membership_metrics(train_z,train_assign,eval_z,selected,bank.centers)
        target=part["a"][:,:v2.HORIZON];mse=float(np.mean((pred-target[None])**2));oracle=metric["truth"][row_seg];oracle_anchor=v3.decoder_anchor(model,bank,oracle,ph,part,stats,device=device)
        oracle_pred=fc.rollout(flow,flow_norm,part["f"],part["s"],oracle_anchor,eps,rowtau,n=np.maximum(2,np.rint(10*rowtau).astype(np.int64)),device=device)
        rowerr=np.mean((pred-target[None])**2,axis=(0,2,3));nerr=np.mean((normal-target[None])**2,axis=(0,2,3))
        metric.update({"selected_mse":mse,"oracle_mse":float(np.mean((oracle_pred-target[None])**2)),"regret_mse":mse-float(np.mean((oracle_pred-target[None])**2)),"normal_mse":float(np.mean((normal-target[None])**2)),"D1_worse_row_fraction":float(np.mean(rowerr>nerr))})
        # Same observation/phase, a different interval latent: measure how much the decoded action changes.
        if kind=="E5": swap_ids=np.roll(selected,1)
        else: swap_ids=np.roll(selected,1)
        swaprow=swap_ids[row_seg];swap_anchor=v3.decoder_anchor(model,bank,swaprow,ph,part,stats,device=device)
        swap_pred=fc.rollout(flow,flow_norm,part["f"],part["s"],swap_anchor,eps,rowtau,n=np.maximum(2,np.rint(10*rowtau).astype(np.int64)),device=device)
        metric["z_swap_action_mse"]=float(np.mean((swap_pred-pred)**2));metric["z_swap_anchor_mse"]=float(np.mean((swap_anchor-anchors)**2))
        results[cname]=metric
        if kind=="E4":
            # Previous selector comparison uses the same skill mean keys.
            mean_keys=np.stack([np.mean(np.stack([train_seg[i]["key"] for i in range(len(train_seg)) if train_assign[i]==j]),0) for j in range(len(bank.centers))])
            previous=np.stack([s["key"] for s in seg])@(mean_keys/np.maximum(np.linalg.norm(mean_keys,axis=1,keepdims=True),1e-8)).T
            audit[cname]={"e4_selected_top1":float(np.mean(selected==strict_truth)),"old_selector_top1":float(np.mean(previous.argmax(1)==strict_truth))}
    return {"split":split,"seed":seed,"method":kind,"selector":selector_mode if kind=="E5" else "observation_prototype",
            "variant":f"E5_z{zdim}_alpha{alpha}_{selector_mode}" if kind=="E5" else "E4_SwAV_IIC",
            "zdim":zdim,"alpha":alpha,"beta":E5_BETA if kind=="E5" else None,"skill_count":len(bank.centers),
            "epsilon":getattr(bank,"epsilon",None),"skill_metadata":skill_meta,"learning":learn,
            "flow_best_epoch":flow_info["best_epoch"],"conditions":results,"selector_audit":audit}


def _load_baseline(path):
    payload=json.loads(Path(path).read_text())
    return {(r["split"],int(r["seed"])):r["baseline_reference"] for r in payload["runs"] if "baseline_reference" in r}


def run(*,seeds=(0,1,2),splits=("A","B"),device_name="cuda",epochs=8,flow_epochs=20,batch_size=512,threads=8,
        cache_dir=DEFAULT_CACHE,dataset_dir=DEFAULT_DATASET,baseline_path=BASELINE_RESULTS,output_root=str(RESEARCH_STATE_DIR / 'outputs/memory-pilot')):
    torch.set_num_threads(int(threads))
    try:torch.set_num_interop_threads(1)
    except RuntimeError:pass
    device=torch.device(device_name);start=time.perf_counter()
    stamp="skill-v3e-"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"-"+uuid.uuid4().hex[:6]
    out=Path(output_root)/stamp;out.mkdir(parents=True,exist_ok=False)
    rows=load_rows(Path(cache_dir),Path(dataset_dir),max_horizon=v2.HORIZON)
    manifest=json.loads((Path(cache_dir)/"manifest.json").read_text());info=json.loads((Path(dataset_dir)/"meta"/"info.json").read_text());stride=float(manifest["stride"])/float(info["fps"])
    taskids=sorted({t for sp in splits for t in (*SPLITS[sp][0],*SPLITS[sp][1])});rows=_subset(rows,np.isin(rows["task_index"],taskids)&rows["valid_h25"])
    text_by_source,text_emb=fc.precompute_texts(rows,("instruction",),Path(cache_dir));rows=fc.attach_text(rows,text_by_source,text_emb,"instruction")
    baselines=_load_baseline(baseline_path);allruns=[]
    for split in splits:
        for seed in seeds:
            if (split,int(seed)) not in baselines:raise KeyError(f"missing v2.1 baseline: {split}/{seed}")
            configs=[("E5",d,a,s) for d in Z_DIMS for a in ALPHAS for s in ("prior","key_cosine")]+[("E4",None,None,None)]
            for kind,zdim,alpha,selector_mode in configs:
                try:
                    result=_run_variant(rows,rows,split,int(seed),stride,device,kind=kind,zdim=zdim,alpha=alpha,selector_mode=selector_mode or "prior",epochs=epochs,flow_epochs=flow_epochs,batch_size=batch_size)
                except Exception as exc:
                    result={"split":split,"seed":int(seed),"method":kind,"selector":selector_mode,"variant":f"E5_z{zdim}_alpha{alpha}_{selector_mode}" if kind=="E5" else "E4_SwAV_IIC","error":f"{type(exc).__name__}: {exc}"}
                result["baseline_reference"]=baselines[(split,int(seed))];allruns.append(result)
                (out/"results.partial.json").write_text(json.dumps({"runs":allruns},ensure_ascii=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x)))
    payload={"review":"review_representation.md §2 method 2/3, §3-4", "seeds":list(seeds),"splits":list(splits),"epochs":epochs,"flow_epochs":flow_epochs,"duration_minutes":(time.perf_counter()-start)/60,"runs":allruns}
    (out/"results.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x)))
    (out/"summary.md").write_text(summarize(payload));(out/"results.partial.json").unlink(missing_ok=True)
    return out


def summarize(payload):
    lines=["# E4/E5 review-driven V3 results","",f"Seeds={payload['seeds']}, splits={payload['splits']}, runtime={payload['duration_minutes']:.1f} min.",
           "ε_k: training members' latent-space mean-squared distance to their assigned skill center, 90th percentile. Functional match means the selected skill has at least one training member within ε_k. Strict accuracy remains nearest-center exact ID.","",
           "| Split | method/variant | K | D3 strict | functional | covered | none | majority | freq-proportional | κ strict/function | D3 MSE | P-a | D1 worse | z-swap action MSE |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for split in payload["splits"]:
        variants=sorted({r["variant"] for r in payload["runs"] if r["split"]==split})
        for variant in variants:
            rs=[r for r in payload["runs"] if r["split"]==split and r["variant"]==variant]
            failed=[r.get("error") for r in rs if "error" in r]
            if failed:
                lines.append(f"| {split} | {variant} | — | failed: {failed[0]} | — | — | — | — | — | — | — | — | — | — |");continue
            mean=lambda fn: float(np.mean([fn(r) for r in rs]))
            d3=[r["conditions"]["D3"] for r in rs];d1=[r["conditions"]["D1"] for r in rs]
            kappa=f"{np.mean([d['strict_kappa'] for d in d3]):.2f}/{np.mean([d['functional_kappa'] for d in d3]):.2f}"
            lines.append(f"| {split} | {variant} | {mean(lambda r:r['skill_count']):.1f} | {mean(lambda r:r['conditions']['D3']['strict_accuracy']):.2f} | {mean(lambda r:r['conditions']['D3']['functional_accuracy_all']):.2f} | {mean(lambda r:r['conditions']['D3']['functional_coverage']):.2f} | {mean(lambda r:r['conditions']['D3']['no_match_rate']):.2f} | {mean(lambda r:r['conditions']['D3']['majority_baseline']):.2f} | {mean(lambda r:r['conditions']['D3']['frequency_proportional_baseline']):.2f} | {kappa} | {mean(lambda r:r['conditions']['D3']['selected_mse']):.4f} | {mean(lambda r:r['baseline_reference']['D3']['primitive_Pa']):.4f} | {float(np.mean([d['D1_worse_row_fraction'] for d in d1])):.2f} | {mean(lambda r:r['conditions']['D3']['z_swap_action_mse']):.4f} |")
    failures=[r for r in payload["runs"] if "error" in r]
    go=not failures
    for r in payload["runs"]:
        if "error" in r:go=False;continue
        for cond in ("D3",):
            m=r["conditions"][cond];pa=r["baseline_reference"][cond]["primitive_Pa"]
            if m["functional_accuracy_all"]<.7 or m["functional_kappa"]<=.3 or m["selected_mse"]>pa:go=False
    lines += ["",f"Overall reviewer go criterion (D3 functional ≥.70, functional κ >.30, selected MSE ≤ P-a for all splits/seeds): **{'GO' if go else 'NO-GO'}**.",
              "Cohen κ(strict) uses exact nearest-center labels and prediction marginals. Functional κ chance correction uses prediction marginals against each selected unit's observed D3 functional-match prevalence. Majority is the D3 exact-label majority; frequency-proportional is expected exact accuracy from training label priors dotted with D3 truth frequencies.",
              "E5 beta=.01 (best-near V1); alpha=.5/.8; z=4/8. E4 uses M6/DP-means-derived initialization with K≤64, cross-view Sinkhorn 3 iterations at T=.1. D1 worsening is row-level against paired F_normal noise.",""]
    return "\n".join(lines)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--seeds",default="0,1,2");p.add_argument("--splits",default="A,B");p.add_argument("--device",default="cuda");p.add_argument("--epochs",type=int,default=8);p.add_argument("--flow-epochs",type=int,default=20);p.add_argument("--batch-size",type=int,default=512);p.add_argument("--threads",type=int,default=8);p.add_argument("--baseline-results",default=str(BASELINE_RESULTS));a=p.parse_args(argv)
    print(run(seeds=tuple(map(int,a.seeds.split(','))),splits=tuple(a.splits.split(',')),device_name=a.device,epochs=a.epochs,flow_epochs=a.flow_epochs,batch_size=a.batch_size,threads=a.threads,baseline_path=a.baseline_results))


if __name__=="__main__":main()
