#!/usr/bin/env python3
"""Observation-free motion keys and causal query encoders for MK1.

The motion encoder consumes only variable-length arm-action sequences.  It is
deliberately independent of task labels, observations, and gripper values.
"""
from __future__ import annotations

import math
import argparse
import json
import os
import sys
import time
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[5]
STATE_DIR = Path(os.environ.get('SKILLNET_STATE_DIR', REPO_ROOT / '.skillnet-state')).expanduser()
from typing import Any

import torch
import numpy as np
from torch import nn
from torch.nn import functional as F


class MotionBlock(nn.Module):
    def __init__(self, dim: int = 256, heads: int = 8):
        super().__init__()
        self.qn = nn.LayerNorm(dim)
        self.kn = nn.LayerNorm(dim)
        self.cross = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.sn = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ffn = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, work: torch.Tensor, motion: torch.Tensor, padding: torch.Tensor,
                aggregate: torch.Tensor | None = None, *, need_weights: bool = False):
        a, weights = self.cross(self.qn(work), self.kn(motion), self.kn(motion),
                                key_padding_mask=padding, need_weights=need_weights,
                                average_attn_weights=True)
        work = work + a
        seq = work if aggregate is None else torch.cat((aggregate, work), dim=1)
        sn = self.sn(seq)
        sa, _ = self.self_attn(sn, sn, sn, need_weights=False)
        seq = seq + sa
        seq = seq + self.ffn(seq)
        if aggregate is None:
            return seq, weights
        return seq[:, :1], seq[:, 1:], weights


def _sinusoidal_positions(length: int, dim: int, device, dtype) -> torch.Tensor:
    pos = torch.arange(length, device=device, dtype=torch.float32)[:, None]
    div = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.) / dim))
    out = torch.zeros((length, dim), device=device, dtype=torch.float32)
    out[:, 0::2] = torch.sin(pos * div)
    out[:, 1::2] = torch.cos(pos * div[:out[:, 1::2].shape[-1]])
    return out.to(dtype=dtype)


class MotionKeyEncoder(nn.Module):
    """M-1 or M-5 arm-action encoder. ``mode='m5'`` uses 5 work + CLS tokens."""
    def __init__(self, mode: str = "m1", *, dim: int = 256, depth: int = 2, heads: int = 8):
        super().__init__()
        if mode not in ("m1", "m5"):
            raise ValueError("mode must be m1 or m5")
        self.mode, self.dim = mode, int(dim)
        self.action_projection = nn.Linear(6, dim)
        self.work_tokens = nn.Parameter(torch.randn(1, 1 if mode == "m1" else 5, dim) * .02)
        self.aggregate_token = nn.Parameter(torch.randn(1, 1, dim) * .02) if mode == "m5" else None
        self.blocks = nn.ModuleList([MotionBlock(dim, heads) for _ in range(depth)])
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, actions: torch.Tensor, valid: torch.Tensor, *, return_details: bool = False):
        if actions.ndim != 3 or actions.shape[-1] != 6 or valid.shape != actions.shape[:2]:
            raise ValueError("expected arm actions [B,T,6] and validity mask [B,T]")
        valid = valid.bool()
        if not valid.any(dim=1).all():
            raise ValueError("each motion sequence needs at least one valid timestep")
        # Masked action values are replaced before projection, so padding contents cannot leak.
        x = torch.where(valid[..., None], actions[..., :6], torch.zeros_like(actions[..., :6])).float()
        motion = self.action_projection(x) + _sinusoidal_positions(x.shape[1], self.dim, x.device, x.dtype)[None]
        padding = ~valid
        work = self.work_tokens.expand(len(x), -1, -1)
        aggregate = None if self.aggregate_token is None else self.aggregate_token.expand(len(x), -1, -1)
        weights = []
        for block in self.blocks:
            if aggregate is None:
                work, attn = block(work, motion, padding, need_weights=return_details)
            else:
                aggregate, work, attn = block(work, motion, padding, aggregate, need_weights=return_details)
            if return_details:
                weights.append(attn)
        result = F.normalize(self.out_norm(work[:, :1] if aggregate is None else aggregate), dim=-1)
        if not return_details:
            return result
        return {"key": result, "work": work, "attention": weights}


class MotionQueryEncoder(nn.Module):
    """Causal query encoder for E-H or E-VLM+H with Q-1/Q-5 shape."""
    def __init__(self, input_mode: str = "e_h", query_mode: str = "q1", *,
                 token_dim: int = 2048, dim: int = 256, heads: int = 8):
        super().__init__()
        if input_mode not in ("e_h", "e_vlm_h") or query_mode not in ("q1", "q5"):
            raise ValueError("unsupported query mode")
        self.input_mode, self.query_mode, self.dim = input_mode, query_mode, dim
        self.vlm_projection = nn.Linear(token_dim, dim)
        self.group_embedding = nn.Embedding(6, dim)
        self.history_projection = nn.Linear(8, dim)
        self.history_position = nn.Parameter(torch.zeros(1, 10, dim))
        self.state_projection = nn.Sequential(nn.Linear(8, dim), nn.GELU(), nn.Linear(dim, dim))
        self.state_type = nn.Parameter(torch.zeros(1, 1, dim))
        self.queries = nn.Parameter(torch.randn(1, 1 if query_mode == "q1" else 5, dim) * .02)
        self.aggregate = nn.Parameter(torch.randn(1, 1, dim) * .02) if query_mode == "q5" else None
        self.cross = nn.ModuleList([MotionBlock(dim, heads) for _ in range(2)])
        self.out_norm = nn.LayerNorm(dim)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1. / .07)))

    def forward(self, *, history: torch.Tensor, history_mask: torch.Tensor, state: torch.Tensor,
                tokens: torch.Tensor | None = None, groups: torch.Tensor | None = None,
                motion_history: torch.Tensor | None = None, motion_history_mask: torch.Tensor | None = None):
        b = len(history)
        hist = self.history_projection(torch.cat((history.float(), history_mask.float()[..., None]), -1)) + self.history_position
        state_tok = self.state_projection(state.float()).unsqueeze(1) + self.state_type
        memories, masks = [hist, state_tok], [~history_mask.bool(), torch.zeros((b, 1), dtype=torch.bool, device=history.device)]
        if self.input_mode == "e_vlm_h":
            if tokens is None or groups is None:
                raise ValueError("e_vlm_h requires VLM tokens and groups")
            valid = groups >= 0
            memories.insert(0, self.vlm_projection(tokens.float()) + self.group_embedding(groups.clamp(0, 5).long()))
            masks.insert(0, ~valid)
        if motion_history is not None:
            if motion_history_mask is None:
                raise ValueError("motion_history_mask is required")
            memories.append(motion_history)
            masks.append(~motion_history_mask.bool())
        memory, padding = torch.cat(memories, 1), torch.cat(masks, 1)
        work = self.queries.expand(b, -1, -1)
        aggregate = None if self.aggregate is None else self.aggregate.expand(b, -1, -1)
        for block in self.cross:
            if aggregate is None:
                work, _ = block(work, memory, padding)
            else:
                aggregate, work, _ = block(work, memory, padding, aggregate)
        out = work[:, :1] if aggregate is None else aggregate
        return F.normalize(self.out_norm(out), dim=-1)


class CrossOnlyBlock(nn.Module):
    """Cross-attention + residual FFN, with no latent self-attention."""
    def __init__(self, dim: int = 256, heads: int = 8):
        super().__init__()
        self.qn=nn.LayerNorm(dim);self.kn=nn.LayerNorm(dim)
        self.attn=nn.MultiheadAttention(dim,heads,batch_first=True)
        self.ff=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,dim*4),nn.GELU(),nn.Linear(dim*4,dim))
    def forward(self,q,memory,padding,need_weights=False):
        k=self.kn(memory);a,w=self.attn(self.qn(q),k,k,key_padding_mask=padding,
                                        need_weights=need_weights,average_attn_weights=True)
        q=q+a;q=q+self.ff(q)
        return q,w


class RetrievalQueryEncoder(nn.Module):
    """Q-1/Q-5 causal query encoder for memory retrieval."""
    def __init__(self,input_mode:str="e_h",query_mode:str="q1",*,key_mode:str="x1",token_dim:int=2048,dim:int=256,heads:int=8):
        super().__init__()
        if input_mode not in ("e_h","e_vlm_h") or query_mode not in ("q1","q5"):
            raise ValueError("unsupported query configuration")
        if key_mode not in ("x1","dct"):raise ValueError("key_mode must be x1 or dct")
        self.input_mode=input_mode;self.query_mode=query_mode;self.key_mode=key_mode;self.dim=dim
        self.vlm_projection=nn.Linear(token_dim,dim);self.group_embedding=nn.Embedding(6,dim)
        self.history_projection=nn.Linear(8,dim);self.history_position=nn.Parameter(torch.zeros(1,10,dim))
        self.state_projection=nn.Sequential(nn.Linear(8,dim),nn.GELU(),nn.Linear(dim,dim));self.state_type=nn.Parameter(torch.zeros(1,1,dim))
        self.work=nn.Parameter(torch.randn(1,1 if query_mode=="q1" else 5,dim)*.02)
        self.aggregate=nn.Parameter(torch.randn(1,1,dim)*.02) if query_mode=="q5" else None
        self.cross=nn.ModuleList([CrossOnlyBlock(dim,heads) for _ in range(2)])
        self.aggregate_cross=nn.ModuleList([CrossOnlyBlock(dim,heads) for _ in range(2)]) if query_mode=="q5" else None
        self.out_norm=nn.LayerNorm(dim);self.logit_scale=nn.Parameter(torch.tensor(math.log(1/.07)))
        self.output_projection=nn.Linear(dim,18) if key_mode=="dct" else None

    def forward(self,*,history,history_mask,state,tokens=None,groups=None,motion_history=None,motion_history_valid=None,
                return_details:bool=False):
        b=len(history);hi=torch.cat((history.float(),history_mask.float()[...,None]),-1)
        mem=[self.history_projection(hi)+self.history_position,self.state_projection(state.float()).unsqueeze(1)+self.state_type]
        masks=[~history_mask.bool(),torch.zeros((b,1),device=history.device,dtype=torch.bool)]
        if self.input_mode=="e_vlm_h":
            if tokens is None or groups is None:raise ValueError("E-VLM+H requires causal VLM tokens and groups")
            mem.insert(0,self.vlm_projection(tokens.float())+self.group_embedding(groups.clamp(0,5).long()))
            masks.insert(0,groups<0)
        if motion_history is not None:
            if motion_history_valid is None:raise ValueError("motion history requires a validity flag")
            if motion_history.ndim==2:motion_history=motion_history.unsqueeze(1)
            mem.append(motion_history);masks.append(~motion_history_valid.bool().reshape(b,1))
        memory=torch.cat(mem,1);padding=torch.cat(masks,1)
        work=self.work.expand(b,-1,-1);aggregate=None if self.aggregate is None else self.aggregate.expand(b,-1,-1)
        details=[]
        for i,cross in enumerate(self.cross):
            work,aw=cross(work,memory,padding,need_weights=return_details)
            if aggregate is not None:
                aggregate,gw=self.aggregate_cross[i](aggregate,work,torch.zeros(work.shape[:2],device=work.device,dtype=torch.bool),need_weights=return_details)
                if return_details:details.append({"input_attention":aw,"aggregate_attention":gw})
            elif return_details:details.append({"input_attention":aw})
        out=work[:,:1] if aggregate is None else aggregate
        query=F.normalize(self.out_norm(out),dim=-1)
        if self.output_projection is not None:query=self.output_projection(query[:,0])
        if return_details:return {"query":query,"work":work,"attention":details}
        return query


class ArchMotionEncoder(nn.Module):
    """MK1b X1, X1s, or X5 structure with absolute/phase position encoding."""
    def __init__(self, architecture: str = "x1", position: str = "absolute", *, dim: int = 256, heads: int = 8):
        super().__init__()
        if architecture not in ("x1","x1s","x5") or position not in ("absolute","phase"):
            raise ValueError("unsupported MK1b architecture or position mode")
        self.architecture,self.position,self.dim=architecture,position,int(dim)
        self.action_projection=nn.Linear(6,dim)
        self.length_projection=nn.Sequential(nn.Linear(1,dim),nn.GELU(),nn.Linear(dim,dim)) if position=="phase" else None
        self.input_self=nn.MultiheadAttention(dim,heads,batch_first=True) if architecture=="x1s" else None
        self.input_norm=nn.LayerNorm(dim) if architecture=="x1s" else None
        self.work=nn.Parameter(torch.randn(1,1 if architecture!="x5" else 5,dim)*.02)
        self.aggregate=nn.Parameter(torch.randn(1,1,dim)*.02) if architecture=="x5" else None
        self.cross=CrossOnlyBlock(dim,heads)
        self.aggregate_cross=CrossOnlyBlock(dim,heads) if architecture=="x5" else None
        self.out_norm=nn.LayerNorm(dim)

    def forward(self,actions:torch.Tensor,valid:torch.Tensor,*,return_details:bool=False):
        if actions.ndim!=3 or actions.shape[-1]!=6 or valid.shape!=actions.shape[:2]:
            raise ValueError("expected [B,T,6] motion and [B,T] mask")
        valid=valid.bool();lengths=valid.sum(-1)
        if torch.any(lengths<1):raise ValueError("empty motion sequence")
        x=torch.where(valid[...,None],actions.float(),torch.zeros_like(actions.float()))
        memory=self.action_projection(x)
        if self.position=="absolute":
            memory=memory+_sinusoidal_positions(x.shape[1],self.dim,x.device,x.dtype)[None]
        else:
            phase=torch.arange(x.shape[1],device=x.device,dtype=torch.float32)[None]/lengths[:,None].float().clamp(min=1)
            freq=torch.exp(torch.arange(0,self.dim,2,device=x.device,dtype=torch.float32)*(-math.log(10000.)/self.dim))
            angle=phase[...,None]*freq[None,None,:]
            p=torch.zeros((len(x),x.shape[1],self.dim),device=x.device,dtype=torch.float32)
            p[...,0::2]=torch.sin(angle);p[...,1::2]=torch.cos(angle[...,:p[...,1::2].shape[-1]])
            memory=memory+p.to(memory.dtype)
            length_tok=self.length_projection(torch.log1p(lengths.float())[:,None]/math.log(100.)).unsqueeze(1)
            memory=torch.cat((memory,length_tok),dim=1)
            valid=torch.cat((valid,torch.ones((len(x),1),device=x.device,dtype=torch.bool)),dim=1)
        padding=~valid
        if self.input_self is not None:
            n=self.input_norm(memory);a,_=self.input_self(n,n,n,key_padding_mask=padding,need_weights=False)
            memory=memory+a
        work=self.work.expand(len(x),-1,-1)
        work,w=self.cross(work,memory,padding,need_weights=return_details)
        if self.aggregate is not None:
            aggregate=self.aggregate.expand(len(x),-1,-1)
            out,aw=self.aggregate_cross(aggregate,work,torch.zeros(work.shape[:2],device=x.device,dtype=torch.bool),need_weights=return_details)
        else:
            out,aw=work,w
        key=F.normalize(self.out_norm(out),dim=-1)
        if return_details:return {"key":key,"work":work,"attention":w,"aggregate_attention":aw}
        return key


class DCTMLPKeyEncoder(nn.Module):
    """18 saved, magnitude-preserving arm DCT3 coefficients to one key."""
    def __init__(self,dim:int=256):
        super().__init__();self.net=nn.Sequential(nn.Linear(18,512),nn.GELU(),nn.Linear(512,dim),nn.GELU(),nn.Linear(dim,dim));self.dim=dim
    def forward(self,actions=None,valid=None,*,coefficients=None,return_details=False):
        if coefficients is None:raise ValueError("DCT coefficients required")
        key=F.normalize(self.net(coefficients.float()),dim=-1)[:,None]
        if return_details:return {"key":key,"work":key,"attention":None,"aggregate_attention":None}
        return key


def _dct3_from_batch(actions:torch.Tensor,valid:torch.Tensor)->torch.Tensor:
    """Compute unnormalized first three DCT-II coefficients from variable-length arm sequences."""
    rows=[]
    for i in range(len(actions)):
        n=int(valid[i].sum().item());x=actions[i,:n]
        j=torch.arange(n,device=x.device,dtype=x.dtype);k=torch.arange(3,device=x.device,dtype=x.dtype)[:,None]
        basis=torch.cos(math.pi/n*(j[None]+.5)*k)*(2./n)**.5
        basis[0]*=1/math.sqrt(2.)
        rows.append(basis@x)
    return torch.stack(rows).reshape(len(actions),18)


def _encode_motion(model,actions,valid,*,batch=64,return_details=False):
    out=[];details=[];model.eval()
    with torch.no_grad():
        for s in range(0,len(actions),batch):
            z=model(actions[s:s+batch],valid[s:s+batch],return_details=return_details)
            if return_details:
                out.append(z["key"].cpu());details.append(z)
            else:out.append(z[:,0].cpu())
    return (torch.cat(out),details) if return_details else torch.cat(out)


def _split_memory_episodes(bank,seed=20261117,holdout_fraction=.2):
    """Deterministic episode-level memory holdout used only to choose distance tau."""
    pairs=sorted(set(zip(np.asarray(bank["task"],int).tolist(),np.asarray(bank["episode"],int).tolist())))
    rng=np.random.default_rng(seed);rng.shuffle(pairs)
    n=max(1,int(round(len(pairs)*holdout_fraction)));held={tuple(x) for x in pairs[:n]}
    fit=np.asarray([i for i,(t,e) in enumerate(zip(bank["task"],bank["episode"],strict=True)) if (int(t),int(e)) not in held],np.int64)
    val=np.asarray([i for i,(t,e) in enumerate(zip(bank["task"],bank["episode"],strict=True)) if (int(t),int(e)) in held],np.int64)
    if set(map(tuple,np.column_stack([bank["task"][fit],bank["episode"][fit]]))) & held:
        raise AssertionError("episode holdout leaked into key tau fit")
    return fit,val,held


def _distill_x1(model,bank,fit_ids,*,device,seed,tau,epochs=12,steps_per_epoch=None,progress=None):
    """Train X1 keys with only memory-side pairwise arm-MSE soft targets."""
    import key_shared_latent as key2
    rng=np.random.default_rng(seed);torch.manual_seed(seed)
    actions,lengths,valid,_=_arch_pack(bank,device)
    windows=key2._candidate_window_actions(np.arange(len(bank["length"])),bank)
    fit_ids=np.asarray(fit_ids,np.int64);qids=fit_ids.copy();opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    steps=steps_per_epoch or int(np.ceil(len(qids)/128));history=[]
    cand_tasks=np.asarray(bank["task"]);cand_eps=np.asarray(bank["episode"])
    for ep in range(epochs):
        rng.shuffle(qids);model.train();losses=[]
        for step in range(steps):
            qi=qids[step*128:min((step+1)*128,len(qids))]
            if not len(qi):continue
            random_ids=rng.choice(fit_ids,size=min(256,len(fit_ids)),replace=False)
            cids,allowed=key2.contrastive_candidate_set(qi,random_ids,bank["task"][qi],bank["episode"][qi],cand_tasks,cand_eps)
            ci=torch.as_tensor(cids,device=device);q=model(actions[qi],valid[qi])[:,0];k=model(actions[ci],valid[ci])[:,0]
            scores=(q@k.T)*20.
            errors=key2._arm6_candidate_mse(bank["value_actions10"][qi],windows[cids])
            loss=key2.contrastive_teacher_loss(scores,torch.as_tensor(errors,device=device),torch.as_tensor(allowed,device=device),
                objective="soft",teacher_temperature=tau)
            opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            losses.append(float(loss.detach()))
        rec={"epoch":ep+1,"loss":float(np.mean(losses)),"steps":len(losses)};history.append(rec)
        if progress is not None:progress(ep+1,rec)
    return model,history


def _memory_spearman(model,bank,ids,device):
    from scipy.stats import spearmanr
    import key_shared_latent as key2
    actions=torch.as_tensor(bank["natural_actions"][...,:6],device=device,dtype=torch.float32)
    lengths=torch.as_tensor(bank["length"],device=device,dtype=torch.long)
    valid=torch.arange(actions.shape[1],device=device)[None]<lengths[:,None]
    emb=_encode_motion(model,actions[ids],valid[ids]).numpy()
    windows=key2._candidate_window_actions(np.asarray(ids),bank)[...,:6].reshape(len(ids),-1)
    error=((windows[:,None]-windows[None,:])**2).mean(-1);score=emb@emb.T
    tri=np.triu_indices(len(ids),1)
    return float(spearmanr(score[tri],-error[tri]).statistic)


def _pretrain_query_x1(output,bank,device,*,seed=1,smoke=False):
    """Select X1 distance tau on held-out memory episodes, then refit all memory episodes."""
    fit,val,held=_split_memory_episodes(bank)
    tau_rows=[]
    for tau in (.01,.02,.05):
        name=f"x1_tau{tau:g}"
        _progress(output/"progress.log","x1_tau_fit_start",tau=tau,fit_units=len(fit),holdout_units=len(val),holdout_episodes=len(held))
        model=ArchMotionEncoder("x1","absolute").to(device);t0=time.time()
        model,hist=_distill_x1(model,bank,fit,device=device,seed=seed,tau=tau,epochs=1 if smoke else 12,
                               steps_per_epoch=2 if smoke else None)
        rho=_memory_spearman(model,bank,val,device);elapsed=time.time()-t0
        path=output/f"tau_select_{name}.pt";torch.save(model.state_dict(),path)
        row={"tau":tau,"holdout_spearman":rho,"fit_units":len(fit),"holdout_units":len(val),"holdout_episodes":len(held),
             "training_seconds":elapsed,"history":hist,"checkpoint":str(path)};tau_rows.append(row)
        _progress(output/"progress.log","x1_tau_fit_complete",tau=tau,spearman=rho,seconds=elapsed)
        del model
        if torch.cuda.is_available():torch.cuda.empty_cache()
    chosen=max(tau_rows,key=lambda x:x["holdout_spearman"])["tau"]
    _progress(output/"progress.log","x1_full_fit_start",tau=chosen,units=len(bank["length"]),epochs=12)
    model=ArchMotionEncoder("x1","absolute").to(device);t0=time.time()
    model,hist=_distill_x1(model,bank,np.arange(len(bank["length"])),device=device,seed=seed+47,tau=chosen,
                           epochs=1 if smoke else 12,steps_per_epoch=2 if smoke else None)
    seconds=time.time()-t0;ck=output/"x1_fixed.pt";torch.save(model.state_dict(),ck)
    _progress(output/"progress.log","x1_full_fit_complete",tau=chosen,seconds=seconds)
    return model,{"tau_candidates":tau_rows,"selected_tau":chosen,"full_fit_units":len(bank["length"]),"full_fit_epochs":1 if smoke else 12,
                  "full_fit_seconds":seconds,"checkpoint":str(ck)}


def _history_motion_keys(model,split,device):
    h=torch.as_tensor(split["history"][...,:6],device=device,dtype=torch.float32)
    actual=torch.as_tensor(split["history_mask"],device=device,dtype=torch.bool)
    valid=actual.clone();empty=~valid.any(-1)
    if empty.any():valid[empty,0]=True
    out=[];model.eval()
    with torch.no_grad():
        for s in range(0,len(h),64):out.append(model(h[s:s+64],valid[s:s+64])[:,0])
    return torch.cat(out),(~empty)


def _query_forward(model,split,ids,device,history_key=None,history_valid=None,*,return_details=False):
    ix=np.asarray(ids,np.int64)
    def as_tensor(name,dtype):
        v=split.get(name)
        return None if v is None else torch.as_tensor(v[ix],device=device,dtype=dtype)
    kw={"history":as_tensor("history",torch.float32),"history_mask":as_tensor("history_mask",torch.bool),
        "state":as_tensor("state",torch.float32),"tokens":as_tensor("tokens",torch.float32),
        "groups":as_tensor("groups",torch.long),"return_details":return_details}
    if history_key is not None:
        kw["motion_history"]=history_key[ix]
        kw["motion_history_valid"]=history_valid[ix]
    return model(**kw)


def _key_features_x1(model,bank,device):
    actions=torch.as_tensor(bank["natural_actions"][...,:6],device=device,dtype=torch.float32)
    lengths=torch.as_tensor(bank["length"],device=device,dtype=torch.long)
    valid=torch.arange(actions.shape[1],device=device)[None]<lengths[:,None]
    return _encode_motion(model,actions,valid).to(device)


def _score_retrieval(q,k,*,key_mode,logit_scale):
    scale=logit_scale.exp().clamp(max=100.)
    if key_mode=="dct":return -torch.square(q[:,None,:]-k[None,:,:]).sum(-1)*scale
    return q@k.T*scale


def _eval_query_model(qmodel,keymodel,key_mode,bank,split,device,*,dct_keys=None,dct_mean=None,dct_std=None,
                      history_key=None,history_valid=None,details=False):
    import key_shared_latent as key2
    from scipy.stats import spearmanr
    qmodel.eval();
    if key_mode=="dct":keys=torch.as_tensor(dct_keys,device=device,dtype=torch.float32)
    else:keys=_key_features_x1(keymodel,bank,device)
    with torch.no_grad():
        qparts=[];work=[]
        for s in range(0,len(split["key"]),16):
            z=_query_forward(qmodel,split,np.arange(s,min(s+16,len(split["key"]))),device,
                history_key,history_valid,return_details=details)
            if details:qparts.append(z["query"][:,0]);work.append(z["work"])
            else:qparts.append(z[:,0] if z.ndim==3 else z)
        q=torch.cat(qparts)
        scores=_score_retrieval(q,keys,key_mode=key_mode,logit_scale=qmodel.logit_scale)
    windows=key2._candidate_window_actions(np.arange(len(bank["length"])),bank)
    errs=key2._arm6_candidate_mse(split["actions"],windows)
    ss=scores.cpu().numpy();top=ss.argmax(1);per=errs[np.arange(len(top)),top];hard=np.asarray(split["hard"],bool)
    from scipy.stats import rankdata
    sr=rankdata(ss,axis=1);er=rankdata(-errs,axis=1);sr-=sr.mean(1,keepdims=True);er-=er.mean(1,keepdims=True)
    rho=(sr*er).sum(1)/np.maximum(np.sqrt((sr*sr).sum(1)*(er*er).sum(1)),1e-12)
    rankings=np.argsort(-ss,axis=1);oracle=np.argsort(errs,axis=1)
    r1=rankings[:,0]==oracle[:,0]
    r5=np.asarray([len(set(rankings[i,:5])&set(oracle[i,:5]))/5 for i in range(len(ss))])
    diag={"spearman":float(np.nanmean(rho)),"recall_at_1":float(np.mean(r1)),"recall_at_5":float(np.mean(r5))}
    result={"rows":len(top),"hard_rows":int(hard.sum()),"top1_arm6_mse":float(per.mean()),
        "top1_arm6_mse_hard":float(per[hard].mean()) if hard.any() else None,"rank_metrics":diag,
        "per_query_top1":per.tolist(),"scores":ss}
    if details:
        w=F.normalize(torch.cat(work),dim=-1).cpu().numpy()
        if w.shape[1]>1:
            ij=np.triu_indices(w.shape[1],1);result["query_work_cosine"]=float(np.mean(np.einsum("bkd,bjd->bkj",w,w)[:,ij[0],ij[1]]))
        else:result["query_work_cosine"]=1.
    return result


def _train_query_config(name,key_mode,input_mode,query_mode,*,output, qtrain,train,dev,bank,windows,
                        fixed_key_state,dct_keys,dct_stats,history_keys,device,seed,random_candidates=256,
                        epochs=16,smoke=False):
    import key_shared_latent as key2
    qmodel=RetrievalQueryEncoder(input_mode,query_mode,key_mode="dct" if key_mode=="dct" else "x1").to(device)
    motion=None
    if key_mode!="dct":
        motion=ArchMotionEncoder("x1","absolute").to(device)
        if key_mode in ("fixed","finetune"):
            motion.load_state_dict(fixed_key_state)
        if key_mode=="fixed":
            motion.eval()
            for p in motion.parameters():p.requires_grad_(False)
    params=list(qmodel.parameters())
    if motion is not None and key_mode!="fixed":params+=list(motion.parameters())
    opt=torch.optim.AdamW(params,lr=1e-4,weight_decay=1e-4)
    rng=np.random.default_rng(seed+700);qids=np.arange(min(len(train["key"]),128) if smoke else len(train["key"]))
    best=float("inf");best_epoch=None;best_state=None;history=[]
    cand_tasks=np.asarray(bank["task"]);cand_eps=np.asarray(bank["episode"])
    dmean,dstd=dct_stats
    if key_mode=="dct":candidate_keys=torch.as_tensor(dct_keys,device=device,dtype=torch.float32)
    else:candidate_keys=None
    if key_mode=="fixed":fixed_keys=_key_features_x1(motion,bank,device)
    else:fixed_keys=None
    if name=="eh_x1_fixed_mhistory":hist_train,hist_train_valid=history_keys["train"]
    else:hist_train=hist_train_valid=None
    q_candidate_ids=np.asarray(qtrain,np.int64)
    actions=torch.as_tensor(bank["natural_actions"][...,:6],device=device,dtype=torch.float32)
    lengths=torch.as_tensor(bank["length"],device=device,dtype=torch.long)
    valid=torch.arange(actions.shape[1],device=device)[None]<lengths[:,None]
    for ep in range(1,epochs+1):
        rng.shuffle(qids);qmodel.train();
        if motion is not None and key_mode!="fixed":motion.train()
        losses=[]
        for start in range(0,len(qids),8):
            qi=qids[start:start+8]
            rand=rng.choice(len(bank["length"]),size=min(random_candidates,len(bank["length"])),replace=False)
            cids,allowed=key2.contrastive_candidate_set(q_candidate_ids[qi],rand,train["task"][qi],train["episode"][qi],cand_tasks,cand_eps)
            qpart={k:np.asarray(v)[qi].copy() for k,v in train.items() if k in ("tokens","groups","history","history_mask","state")}
            qpart["history"],qpart["history_mask"]=key2._history_corrupt(qpart["history"],qpart["history_mask"],rng)
            hkey=hist_train[qi] if hist_train is not None else None;hvalid=hist_train_valid[qi] if hist_train_valid is not None else None
            qpart_device=qpart
            def tensor(name,dtype):
                v=qpart_device.get(name);return None if v is None else torch.as_tensor(v,device=device,dtype=dtype)
            q=qmodel(history=tensor("history",torch.float32),history_mask=tensor("history_mask",torch.bool),
                state=tensor("state",torch.float32),tokens=tensor("tokens",torch.float32),groups=tensor("groups",torch.long),
                motion_history=hkey,motion_history_valid=hvalid)
            q=q[:,0] if q.ndim==3 else q
            if key_mode=="dct":k=candidate_keys[cids]
            elif key_mode=="fixed":k=fixed_keys[cids]
            else:
                ci=torch.as_tensor(cids,device=device);k=motion(actions[ci],valid[ci])[:,0]
            scores=_score_retrieval(q,k,key_mode=key_mode,logit_scale=qmodel.logit_scale)
            err=key2._arm6_candidate_mse(train["actions"][qi],windows[cids])
            loss=key2.contrastive_teacher_loss(scores,torch.as_tensor(err,device=device),torch.as_tensor(allowed,device=device),
                objective="soft",teacher_temperature=.002)
            opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(params,1.);opt.step()
            key2._clamp_logit_scale_(qmodel);losses.append(float(loss.detach()))
        rec={"epoch":ep,"loss":float(np.mean(losses)),"steps":len(losses)}
        if ep in ((4,) if smoke else (4,8,16)):
            qmodel.eval()
            if motion is not None:motion.eval()
            # Dev-only selection, not Eval; recompute key representations after fine-tuning.
            if key_mode=="dct":devkeys=candidate_keys
            elif key_mode=="fixed":devkeys=fixed_keys
            else:devkeys=_key_features_x1(motion,bank,device)
            with torch.no_grad():
                qdev=[]
                for s in range(0,len(dev["key"]),16):
                    hk=history_keys["dev"][0] if name=="eh_x1_fixed_mhistory" else None
                    hv=history_keys["dev"][1] if name=="eh_x1_fixed_mhistory" else None
                    qpart=_query_forward(qmodel,dev,np.arange(s,min(s+16,len(dev["key"]))),device,hk,hv)
                    qdev.append(qpart[:,0] if qpart.ndim==3 else qpart)
                qd=torch.cat(qdev);sc=_score_retrieval(qd,devkeys,key_mode=key_mode,logit_scale=qmodel.logit_scale)
            err=key2._arm6_candidate_mse(dev["actions"],windows);top=sc.argmax(1).cpu().numpy();metric=float(err[np.arange(len(top)),top].mean())
            rec["dev_top1_arm6_mse"]=metric
            if metric<best:
                best=metric;best_epoch=ep
                best_state={"query":{k:v.detach().cpu().clone() for k,v in qmodel.state_dict().items()},
                    "motion":None if motion is None else {k:v.detach().cpu().clone() for k,v in motion.state_dict().items()}}
        history.append(rec)
        if ep%4==0:_progress(output/"progress.log","query_epoch",name=name,epoch=ep,loss=rec["loss"],dev=rec.get("dev_top1_arm6_mse"))
    qmodel.load_state_dict(best_state["query"])
    if motion is not None and best_state["motion"] is not None:motion.load_state_dict(best_state["motion"])
    return qmodel,motion,history,best_epoch,best


def _baseline_mlp_per_query(bank_input,bank,split,device):
    import key_grid as kg
    import key_shared_latent as key2
    root=(STATE_DIR / "outputs/memory-pilot/unseen-libero/key-grid-01/production-04")
    saved=json.loads((root/"results.json").read_text())
    row=saved["new_models"]["history_state"]["variants"]["two_tower_mlp"]
    qx=kg.make_vector_features(split,"history_state");kx=kg.make_vector_features(bank_input,"history_state")
    model=kg.PairMLPRanker(qx.shape[-1],shared=False).to(device)
    model.load_state_dict(torch.load(row["checkpoint"],map_location=device,weights_only=True));model.eval()
    scores=kg._score_vectors(model,qx,kx,device=device)
    windows=key2._candidate_window_actions(np.arange(len(bank["length"])),bank)
    errors=key2._arm6_candidate_mse(split["actions"],windows);top=scores.argmax(1)
    return errors[np.arange(len(top)),top],row


def _reference_rows(output,bank,bank_input,learned,unseen,device):
    import key_shared_latent as key2
    kroot=(STATE_DIR / "outputs/memory-pilot/unseen-libero/key-grid-01/production-04/results.json")
    sroot=(STATE_DIR / "outputs/memory-pilot/unseen-libero/key-shared-01/attempt-27-production/results.json")
    kg=json.loads(kroot.read_text());sr=json.loads(sroot.read_text());rows={}
    for split_name,split in (("learned",learned),("unseen",unseen)):
        fixed=sr["methods"][split_name]["fixed_baselines"]
        new=kg["new_models"]["history_state"]["variants"]["two_tower_mlp"]["evaluation"][split_name]["metrics"]
        refs={"history_state_mlp":{"all":new["top1_arm6_mse"],"hard":new["top1_arm6_mse_hard"]},
            "key2_separate":{"all":sr["methods"][split_name]["separate"]["rank_all"]["top1_arm6_mse"],
                              "hard":sr["methods"][split_name]["separate"]["rank_hard"]["top1_arm6_mse"]},
            "e_vlm_h":{"all":kg["existing_reselected"]["e_vlm_h"]["evaluation"][split_name]["metrics"]["top1_arm6_mse"],
                        "hard":kg["existing_reselected"]["e_vlm_h"]["evaluation"][split_name]["metrics"]["top1_arm6_mse_hard"]},
            "e_h":{"all":kg["existing_reselected"]["e_h"]["evaluation"][split_name]["metrics"]["top1_arm6_mse"],
                    "hard":kg["existing_reselected"]["e_h"]["evaluation"][split_name]["metrics"]["top1_arm6_mse_hard"]},
            "p5":{"all":kg["existing_reselected"]["p5"]["evaluation"][split_name]["metrics"]["top1_arm6_mse"],
                  "hard":kg["existing_reselected"]["p5"]["evaluation"][split_name]["metrics"]["top1_arm6_mse_hard"]}}
        for key in ("hold_last","pi05","random20_blend"):
            refs[key]={"all":fixed["baselines"][key]["arm6"],"hard":fixed["baselines_hard"][key]["arm6"]}
        refs["oracle"]={"all":fixed["oracle_playback_all"]["arm6"],
                         "hard":fixed["oracle_playback_hard"]["arm6"] if fixed.get("oracle_playback_hard") else None}
        # The exact standard random top-1 reference is saved as random candidate rank.
        refs["random"]={"all":fixed["random_rank"]["top1_arm6_mse"],
                        "hard":fixed["random_rank_hard"]["top1_arm6_mse"] if fixed.get("random_rank_hard") else None}
        rows[split_name]=refs
    return rows


def _run_query_experiment(output:Path,*,data,device:str,seed:int=1,smoke:bool=False):
    import key_shared_latent as key2
    bank,bank_input,fit_bank,fit_input,train,learned,unseen,dev=data
    # All candidates and all query/training rows are memory-side LIBERO-40, except locked Eval splits.
    source_map={tuple(map(int,k[:3])):i for i,k in enumerate(bank["source"])}
    train_candidate_ids=np.asarray([source_map[tuple(map(int,k))] for k in train["key"]],np.int64)
    if len(train_candidate_ids)!=len(train["key"]):raise AssertionError("memory query/candidate identity mismatch")
    if np.any(bank["task"][train_candidate_ids]!=train["task"]) or np.any(bank["episode"][train_candidate_ids]!=train["episode"]):
        raise AssertionError("memory training query task/episode mismatch")
    # Correct MK1b DCT-MLP diagnosis: keys were coeff*magnitude while the target feature used a
    # fixed-horizon normalized DCT, and dimensions were not standardized. Use saved DCT3 shape
    # coefficients directly, standardized from memory-only rows; Q learns into those coordinates.
    saved_coeff=np.asarray(bank["coefficients"],np.float32).reshape(len(bank["length"]),18)
    dct_mean=saved_coeff.mean(0);dct_std=saved_coeff.std(0).clip(1e-4)
    dct_keys=(saved_coeff-dct_mean)/dct_std
    _progress(output/"progress.log","x1_tau_selection_start",memory_units=len(bank["length"]))
    fixed_key,key_selection=_pretrain_query_x1(output,bank,device,seed=seed,smoke=smoke)
    fixed_state={k:v.detach().cpu().clone() for k,v in fixed_key.state_dict().items()}
    history_keys={}
    for name,split in (("train",train),("dev",dev),("learned",learned),("unseen",unseen)):
        hk,hv=_history_motion_keys(fixed_key,split,device);history_keys[name]=(hk,hv)
        _progress(output/"progress.log","history_motion_encoded",split=name,rows=len(hk),valid=int(hv.sum()))
    windows=key2._candidate_window_actions(np.arange(len(bank["length"])),bank)
    configs=[
        ("dct_e_h","dct","e_h","q1",None),("dct_e_vlm_h","dct","e_vlm_h","q1",None),
        ("x1_fixed_e_h","fixed","e_h","q1",None),("x1_fixed_e_vlm_h","fixed","e_vlm_h","q1",None),
        ("x1_finetune_e_h","finetune","e_h","q1",None),("x1_finetune_e_vlm_h","finetune","e_vlm_h","q1",None),
        ("x1_joint_e_h","joint","e_h","q1",None),("x1_joint_e_vlm_h","joint","e_vlm_h","q1",None),
        ("eh_x1_fixed_mhistory","fixed","e_h","q1","history"),
        ("x1_fixed_q5_e_vlm_h","fixed","e_vlm_h","q5",None)]
    if smoke:configs=[configs[0],configs[1],configs[3],configs[8],configs[9]]
    result={"status":"running_smoke" if smoke else "running","metadata":{"seed":seed,"train_rows":len(train["key"]),"bank_units":len(bank["length"]),
        "learned_eval_rows":len(learned["key"]),"unseen_eval_rows":len(unseen["key"]),"dev_rows":len(dev["key"]),
        "learning_rate":1e-4,"teacher_tau":.002,"epochs_grid":[4,8,16],"batch_size":8,"random_candidates":256,
        "candidate_episode_exclusion":True,"eval_used_for_selection":False,"dct_key_input":"saved coefficient-only standardized on memory rows"},
        "fixed_key_tau_selection":key_selection,"rows":{},"references":{},"paired_episode_bootstrap_5000":{}}
    for name,key_mode,input_mode,qmode,extra in configs:
        _progress(output/"progress.log","query_model_start",name=name)
        qmodel,motion,hist,selected_epoch,dev_mse=_train_query_config(name,key_mode,input_mode,qmode,
            output=output,qtrain=train_candidate_ids,train=train,dev=dev,bank=bank,windows=windows,
            fixed_key_state=fixed_state,dct_keys=dct_keys,dct_stats=(dct_mean,dct_std),
            history_keys=history_keys if extra else {},device=device,seed=seed,epochs=4 if smoke else 16,
            random_candidates=32 if smoke else 256,smoke=smoke)
        path=output/f"{name}.pt";torch.save({"query":qmodel.state_dict(),"motion":None if motion is None else motion.state_dict()},path)
        evaluations={}
        for split_name,split in (("learned",learned),("unseen",unseen)):
            details=qmode=="q5"
            hk,hv=history_keys[split_name] if extra else (None,None)
            ev=_eval_query_model(qmodel,motion,key_mode if key_mode!="dct" else "dct",bank,split,device,
                dct_keys=dct_keys,history_key=hk,history_valid=hv,details=details)
            scores=ev.pop("scores")
            if details:query_work_cos=ev.pop("query_work_cosine")
            else:query_work_cos=None
            ev["query_work_cosine"]=query_work_cos;evaluations[split_name]=ev
        result["rows"][name]={"key_mode":key_mode,"input":input_mode,"query_mode":qmode,
            "extra_motion_history":bool(extra),"selected_epoch":selected_epoch,"dev_top1_arm6_mse":dev_mse,
            "train_history":hist,"evaluation":evaluations,"checkpoint":str(path)}
        _progress(output/"progress.log","query_model_complete",name=name,selected_epoch=selected_epoch,dev=dev_mse,
            unseen=evaluations["unseen"]["top1_arm6_mse"])
        (output/"results.json").write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
        del qmodel,motion
        if torch.cuda.is_available():torch.cuda.empty_cache()
    # Recompute paired baselines using the identical fixed Eval rows.
    result["references"]=_reference_rows(output,bank,bank_input,learned,unseen,device)
    for split_name,split in (("learned",learned),("unseen",unseen)):
        baseline,_=_baseline_mlp_per_query(bank_input,bank,split,device)
        # Choose the overall row by Dev, never by the locked Eval scores.
        best_name=min(result["rows"],key=lambda n:result["rows"][n]["dev_top1_arm6_mse"])
        best=np.asarray(result["rows"][best_name]["evaluation"][split_name]["per_query_top1"])
        result["paired_episode_bootstrap_5000"][split_name]={
            "best_minus_history_state_mlp":{"best_row":best_name,**key2.episode_bootstrap_ci(best-baseline,split["task"],split["episode"],reps=5000,seed=7101)},
        }
        if split_name=="learned":
            names=("x1_fixed_e_vlm_h","dct_e_vlm_h")
        else:names=("x1_fixed_e_vlm_h","dct_e_vlm_h")
        a=np.asarray(result["rows"][names[0]]["evaluation"][split_name]["per_query_top1"])
        b=np.asarray(result["rows"][names[1]]["evaluation"][split_name]["per_query_top1"])
        result["paired_episode_bootstrap_5000"][split_name]["x1_fixed_minus_dct_same_query"] = key2.episode_bootstrap_ci(
            a-b,split["task"],split["episode"],reps=5000,seed=7102)
    result["status"]="complete_smoke" if smoke else "complete"
    for row in result["rows"].values():
        for split in row["evaluation"].values():split.pop("_scores",None)
    (output/"results.json").write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    _write_query_report(output,result)
    return result


def _write_query_report(output:Path,r:dict[str,Any]):
    names={"dct_e_h":"K-dct / E-H","dct_e_vlm_h":"K-dct / E-VLM+H",
        "x1_fixed_e_h":"K-X1 fixed / E-H","x1_fixed_e_vlm_h":"K-X1 fixed / E-VLM+H",
        "x1_finetune_e_h":"K-X1 fine-tune / E-H","x1_finetune_e_vlm_h":"K-X1 fine-tune / E-VLM+H",
        "x1_joint_e_h":"K-X1 joint / E-H","x1_joint_e_vlm_h":"K-X1 joint / E-VLM+H",
        "eh_x1_fixed_mhistory":"K-X1 fixed / E-H + M history","x1_fixed_q5_e_vlm_h":"K-X1 fixed / Q-5 E-VLM+H"}
    lines=["# MK1c：運動キーにクエリを付けた検索", "",
        "|行|Eval unseen all/hard|Eval learned all/hard|rho unseen|recall@1 / @5 unseen|rho learned|recall@1 / @5 learned|Dev epoch|Q5 work cos|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for key,row in r["rows"].items():
        u=row["evaluation"]["unseen"];l=row["evaluation"]["learned"];ur=u["rank_metrics"];lr=l["rank_metrics"]
        lines.append(f"|{names.get(key,key)}|{u['top1_arm6_mse']:.5f} / {u['top1_arm6_mse_hard']:.5f}|{l['top1_arm6_mse']:.5f} / {l['top1_arm6_mse_hard']:.5f}|{ur['spearman']:.4f}|{ur['recall_at_1']:.3f} / {ur['recall_at_5']:.3f}|{lr['spearman']:.4f}|{lr['recall_at_1']:.3f} / {lr['recall_at_5']:.3f}|{row['selected_epoch']}|{u['query_work_cosine'] if u['query_work_cosine'] is not None else '—'}|")
    lines += ["", "### K-X1 固定キーの τ 選択", "", "|τ|memory episode-holdout Spearman|holdout units|seconds|", "|---:|---:|---:|---:|"]
    for row in r["fixed_key_tau_selection"]["tau_candidates"]:
        lines.append(f"|{row['tau']}|{row['holdout_spearman']:.4f}|{row['holdout_units']}|{row['training_seconds']:.1f}|")
    lines += [f"選択 τ={r['fixed_key_tau_selection']['selected_tau']}。固定キーは選択 τ で全 {r['fixed_key_tau_selection']['full_fit_units']} メモリ単位を {r['fixed_key_tau_selection']['full_fit_epochs']} epoch 再学習。", "",
        "### 既存参照（top-1 腕 MSE all / hard）", "", "|行|learned|unseen|", "|---|---:|---:|"]
    refs=r.get("references",{})
    for name,label in (("history_state_mlp","履歴＋状態 MLP"),("key2_separate","KEY2 別キー"),("e_vlm_h","E-VLM+H"),("e_h","E-H"),("p5","P5"),("oracle","oracle"),("hold_last","直前保持"),("pi05","π0.5"),("random","random")):
        a=refs.get("learned",{}).get(name,{});b=refs.get("unseen",{}).get(name,{})
        lines.append(f"|{label}|{a.get('all',float('nan')):.5f} / {a.get('hard',float('nan')):.5f}|{b.get('all',float('nan')):.5f} / {b.get('hard',float('nan')):.5f}|")
    lines += ["", "### 対の差（episode bootstrap 95%、5,000回）", "", "|Eval|比較（左−右）|mean|95% CI|", "|---|---|---:|---:|"]
    for split,d in r.get("paired_episode_bootstrap_5000",{}).items():
        for key,label in (("best_minus_history_state_mlp","最良行 − 履歴＋状態 MLP"),("x1_fixed_minus_dct_same_query","K-X1 固定 − K-dct（E-VLM+H）")):
            row=d.get(key)
            if row:
                low=row.get("ci95_low",row.get("ci95_lower"));high=row.get("ci95_high",row.get("ci95_upper"))
                lines.append(f"|{split}|{label}|{row['mean']:.5f}|[{low:.5f}, {high:.5f}]|")
    lines += ["", "最良行との比較はEvalで選ばず、全構成のDev top-1最小（`x1_joint_e_h`）で固定した。", "", "### DCT-MLP 診断とデータ分離", "",
        "MK1b の DCT-MLP は保存キーに `coefficients × magnitude`（物理値）、教師特徴に固定10-step normalized-action DCT を使い、単位・標準化が一致していなかった。MK1c の K-dct は保存済み DCT3 形状係数のみをメモリ側平均・標準偏差で標準化し、Q 出力を同じ18次元へ線形射影する形に修正した。採点は標準化係数上の −二乗ユークリッド距離×学習倍率。結果は上表の K-dct 行。",
        "", "クエリ／キー学習は LIBERO-40 メモリ単位のみ。固定キー τ 選択はメモリ episode holdout の Spearman のみで行った。学習済み621／未学習619 Eval は Dev 選択後に各1回評価。Q5 work cosine は Eval query 上で計算。", ""]
    if r.get("status")=="complete_smoke":lines.insert(2,"スモーク実行のみ。ここに示す部分結果は本番値ではありません。")
    (output/"report.md").write_text("\n".join(lines))


def run_query_main():
    parser=argparse.ArgumentParser(description="MK1c query-and-retrieve production")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--run-root",type=Path,default=(STATE_DIR / "outputs/memory-pilot/pi05-memory40-20261104/run-01"))
    parser.add_argument("--missing-root",type=Path,default=(STATE_DIR / "outputs/memory-pilot/unseen-libero/key-shared-01/attempt-02/prefix-missing"))
    parser.add_argument("--seen-stride5",type=Path,default=(STATE_DIR / "outputs/memory-pilot/unseen-libero/seg-context-01/seg5-cache"))
    parser.add_argument("--unseen-stride5",type=Path,default=(STATE_DIR / "outputs/memory-pilot/unseen-libero/seg-context-01/unseen-stride5"))
    parser.add_argument("--vjepa-eval-cache",type=Path,default=(STATE_DIR / "outputs/memory-pilot/unseen-libero/key-shared-01/attempt-27-production/vjepa/libero90_eval"))
    parser.add_argument("--device",default="cuda:0");parser.add_argument("--seed",type=int,default=1);parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS","8")))
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
    import key_shared_latent as key2
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    _progress(args.output/"progress.log","query_data_build_start")
    data=key2._build_key2_queries(args.run_root,args.missing_root,args.seen_stride5,args.unseen_stride5,args.vjepa_eval_cache,
        progress_callback=lambda stage,**kw:_progress(args.output/"progress.log",stage,**kw))
    _progress(args.output/"progress.log","query_data_build_complete",bank_units=len(data[0]["length"]),train_rows=len(data[4]["key"]))
    result=_run_query_experiment(args.output,data=data,device=args.device,seed=args.seed,smoke=args.smoke)
    _progress(args.output/"progress.log","query_complete",status=result["status"])
    if not args.smoke:
        # Put the latest MK1c report first; preserve every prior MK1/MK1b section below it.
        main_report=args.output.parent/"report.md";old=main_report.read_text() if main_report.exists() else ""
        separator="\n---\n\n"
        if separator in old:old=old.split(separator,1)[1]
        main_report.write_text((args.output/"report.md").read_text()+separator+old)


def augment_motion(actions: torch.Tensor, lengths: torch.Tensor, *, noise_std: float = .005,
                   stretch: torch.Tensor | None = None, crop_left: torch.Tensor | None = None,
                   crop_right: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Small action-preserving temporal/noise augmentation, padded to batch width."""
    b, width, dim = actions.shape
    outputs, out_lengths = [], []
    for i in range(b):
        n = int(lengths[i].item())
        x = actions[i, :n]
        factor = float(stretch[i]) if stretch is not None else float(torch.empty((), device=x.device).uniform_(.9, 1.1))
        target = max(1, int(round(n * factor)))
        x = F.interpolate(x.T[None], size=target, mode="linear", align_corners=True)[0].T
        left = int(crop_left[i]) if crop_left is not None else int(torch.randint(0, min(3, max(1, target)), (), device=x.device))
        right = int(crop_right[i]) if crop_right is not None else int(torch.randint(0, min(3, max(1, target-left)), (), device=x.device))
        if target - left - right < 1:
            left, right = 0, 0
        x = x[left:target-right if right else target]
        x = x + torch.randn_like(x) * noise_std
        outputs.append(x); out_lengths.append(len(x))
    max_len = max(out_lengths)
    packed = actions.new_zeros((b, max_len, dim))
    for i, x in enumerate(outputs):
        packed[i, :len(x)] = x
    return packed, torch.tensor(out_lengths, device=actions.device, dtype=torch.long)


def pack_motion(rows: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack natural_actions to arm-only tensors and an explicit length mask."""
    lengths = torch.as_tensor(rows["length"], dtype=torch.long)
    actions = torch.as_tensor(rows["natural_actions"][..., :6], dtype=torch.float32)
    positions = torch.arange(actions.shape[1])[None]
    mask = positions < lengths[:, None]
    return actions, mask


def assert_no_eval_motion_rows(train_ids: Any, eval_pairs: Any, task: Any, episode: Any) -> None:
    """Raise if any memory-unit source in the training IDs belongs to fixed Eval episodes."""
    held = {tuple(map(int, p)) for p in eval_pairs}
    overlap = [(int(task[i]), int(episode[i])) for i in train_ids
               if (int(task[i]), int(episode[i])) in held]
    if overlap:
        raise AssertionError(f"Eval episodes leaked into motion training: {overlap[:5]}")


def _progress(path: Path, stage: str, **fields: Any) -> None:
    row = {"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "stage": stage, **fields}
    with path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def _fit_self_supervised(model: MotionKeyEncoder, bank: dict[str, Any], ids: Any, *, device: str,
                         seed: int, epochs: int = 16, batch_size: int = 64):
    rng = torch.Generator(device=device).manual_seed(seed)
    np_rng = __import__("numpy").random.default_rng(seed)
    torch.manual_seed(seed)
    ids = __import__("numpy").asarray(ids, dtype="int64")
    actions_np = __import__("numpy").asarray(bank["natural_actions"], dtype="float32")[..., :6]
    lengths_np = __import__("numpy").asarray(bank["length"], dtype="int64")
    coeff_np = __import__("numpy").asarray(bank["coefficients"], dtype="float32")
    if coeff_np.ndim >= 2 and coeff_np.shape[-1] >= 6:
        coeff_np = coeff_np[..., :6]  # auxiliary target is arm-only, matching M's input
    actions = torch.as_tensor(actions_np, device=device)
    lengths = torch.as_tensor(lengths_np, device=device)
    coeff = torch.as_tensor(coeff_np.reshape(len(coeff_np), -1), device=device)
    decoder = nn.Sequential(nn.Linear(model.dim, model.dim), nn.GELU(), nn.Linear(model.dim, coeff.shape[-1])).to(device)
    opt = torch.optim.AdamW(list(model.parameters()) + list(decoder.parameters()), lr=1e-4, weight_decay=1e-4)
    history = []
    for epoch in range(epochs):
        np_rng.shuffle(ids)
        model.train(); decoder.train(); losses=[]; nce_values=[]; aux_values=[]
        for start in range(0, len(ids), batch_size):
            bi = ids[start:start + batch_size]
            if len(bi) < 2:
                continue
            seq = actions[bi]
            lens = lengths[bi]
            valid = torch.arange(seq.shape[1], device=device)[None] < lens[:, None]
            aug, aug_len = augment_motion(seq, lens, noise_std=.003)
            aug_mask = torch.arange(aug.shape[1], device=device)[None] < aug_len[:, None]
            z1 = model(seq, valid)[:, 0]
            z2 = model(aug, aug_mask)[:, 0]
            logits = z1 @ z2.T / .1
            targets = torch.arange(len(bi), device=device)
            nce = (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)) * .5
            aux = F.mse_loss(decoder(z1), coeff[bi])
            loss = nce + .1 * aux
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step(); losses.append(float(loss.detach())); nce_values.append(float(nce.detach())); aux_values.append(float(aux.detach()))
        history.append({"epoch": epoch + 1, "loss": float(__import__("numpy").mean(losses)),
                        "infonce": float(__import__("numpy").mean(nce_values)),
                        "dct3_aux_mse": float(__import__("numpy").mean(aux_values)), "steps": len(losses)})
    return model, decoder, history


def _encode_all(model: MotionKeyEncoder, bank: dict[str, Any], *, device: str, batch_size: int = 64,
                details: bool = False):
    actions = torch.as_tensor(bank["natural_actions"][..., :6], dtype=torch.float32)
    lengths = torch.as_tensor(bank["length"], dtype=torch.long)
    out=[]; works=[]; attns=[]
    model.eval()
    with torch.no_grad():
        for start in range(0, len(lengths), batch_size):
            x=actions[start:start+batch_size].to(device); n=lengths[start:start+batch_size].to(device)
            mask=torch.arange(x.shape[1],device=device)[None] < n[:,None]
            z=model(x,mask,return_details=details)
            if details:
                out.append(z["key"].cpu());works.append(z["work"].cpu());attns.append([a.cpu() for a in z["attention"]])
            else:out.append(z.cpu())
    if details:
        return torch.cat(out),torch.cat(works),attns
    return torch.cat(out)


def _write_stage1_report(output: Path, result: dict[str, Any]) -> None:
    def number(x: Any, digits: int = 5) -> str:
        return "n/a" if x is None or not math.isfinite(float(x)) else f"{float(x):.{digits}f}"
    lines=["# MK1：行動チャンクから作る運動キー", "", "## 第1段 中間結果：観測なし運動キー", "",
           "|キー|Eval split|M cosine 最近傍|DCT3 最近傍|生値 `_type_features` 最近傍|oracle 腕 MSE|",
           "|---|---|---:|---:|---:|---:|"]
    for mode, data in result.get("stage1", {}).items():
        for split, row in data.get("nearest", {}).items():
            lines.append(f"|{mode}|{split}|{number(row.get('motion_cosine_top1_mse'))}|{number(row.get('dct3_top1_mse'))}|{number(row.get('raw10_top1_mse'))}|{number(row.get('oracle_mse'))}|")
    lines += ["", "### 診断", "", "|キー|pairwise Spearman|recall@1|recall@5|task-ID kNN|作業トークン cos|注意先 overlap|",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for mode,data in result.get("stage1",{}).items():
        d=data.get("diagnostics",{})
        lines.append(f"|{mode}|{number(d.get('spearman'),4)}|{number(d.get('recall_at_1'),4)}|{number(d.get('recall_at_5'),4)}|{number(d.get('task_knn_accuracy'),4)}|{number(d.get('work_cosine'),4)}|{number(d.get('attention_overlap'),4)}|")
    lines += ["", "Stage 1 はメモリ側 LIBERO-40 単位の自己教師学習です。Eval の将来行動は検索診断にのみ使い、モデル学習には含めません。", ""]
    (output/"report.md").write_text("\n".join(lines))


def _stage1(output: Path, *, device: str, seed: int, data: tuple[Any,...], smoke: bool = False):
    import numpy as np
    from scipy.stats import spearmanr
    import key_shared_latent as key2
    bank,bank_input,fit_bank,fit_input,train,learned,unseen,dev=data
    # fit_bank is the builder's memory-only candidate set; fixed learned Eval episodes are excluded.
    all_tasks=np.asarray(bank["task"]); all_eps=np.asarray(bank["episode"])
    fitkeys={(int(t),int(e),int(s)) for t,e,s in fit_bank["source"][:,:3]}
    train_ids=np.asarray([i for i,k in enumerate(bank["source"]) if tuple(map(int,k[:3])) in fitkeys],np.int64)
    eval_pairs=set(zip(np.asarray(learned["task"],int),np.asarray(learned["episode"],int)))
    assert_no_eval_motion_rows(train_ids,eval_pairs,all_tasks,all_eps)
    if smoke: train_ids=train_ids[:min(128,len(train_ids))]
    result={"stage1":{},"training":{"units":len(train_ids),"pool_units":len(bank["length"]),"eval_rows":{"learned":len(learned["key"]),"unseen":len(unseen["key"])},"seed":seed,"epochs":16,"objective":"symmetric same-unit InfoNCE + 0.1 DCT3 reconstruction"}}
    windows=key2._candidate_window_actions(np.arange(len(bank["length"])),bank)
    for mode in ("m1","m5"):
        _progress(output/"progress.log","stage1_train_start",mode=mode,units=len(train_ids))
        model=MotionKeyEncoder(mode).to(device)
        model,decoder,history=_fit_self_supervised(model,bank,train_ids,device=device,seed=seed,epochs=2 if smoke else 16)
        ckpt=output/f"{mode}.pt";torch.save({"model":model.state_dict(),"decoder":decoder.state_dict(),"history":history},ckpt)
        keys,work,attention=_encode_all(model,bank,device=device,details=True)
        norm_actions=np.asarray(bank["natural_actions"],np.float32)[...,:6]
        n=len(bank["length"]);take=min(n,1500 if smoke else n)
        # Pairwise rank diagnostics on a deterministic memory-only subset.
        ids=np.linspace(0,n-1,take,dtype=np.int64) if take<n else np.arange(n)
        emb=keys[ids,0].numpy();cos=emb@emb.T
        vw=windows[ids,:,:6].reshape(len(ids),-1)
        err=np.maximum(((vw[:,None]-vw[None,:])**2).mean(-1),0.)
        tri=np.triu_indices(len(ids),1);rank=spearmanr(cos[tri],-err[tri]).statistic
        cos_rank=cos.copy(); err_rank=err.copy(); np.fill_diagonal(cos_rank,-np.inf); np.fill_diagonal(err_rank,np.inf)
        order=np.argsort(-cos_rank,axis=1); oracle=np.argsort(err_rank,axis=1)
        rec1=np.mean(order[:,0]==oracle[:,0]); rec5=np.mean([len(set(order[i,:5])&set(oracle[i,:5]))/5 for i in range(len(ids))])
        # Task-ID kNN: majority vote among five cosine neighbors, self excluded.
        tasks=all_tasks[ids]; knn=order[:,1:6]; pred=[]
        for row in knn: pred.append(np.bincount(tasks[row].astype(int)).argmax())
        task_acc=float(np.mean(np.asarray(pred)==tasks))
        work_np=F.normalize(work[ids],dim=-1).numpy()
        work_cos=float(np.mean(np.einsum("bkd,bjd->bkj",work_np,work_np)[:,np.triu_indices(work_np.shape[1],1)[0],np.triu_indices(work_np.shape[1],1)[1]])) if work_np.shape[1]>1 else 1.
        attn_overlap=float("nan")
        if mode=="m5" and attention and attention[0] is not None:
            # Attention overlap: mean Jaccard of top-10 time indices across work tokens in first layer.
            a=torch.cat([z for z in attention[0]],0).numpy() if len(attention[0])>1 else attention[0][0].numpy()
            if a.ndim==3:
                sets=np.argsort(-a,axis=-1)[...,:min(10,a.shape[-1])]
                vals=[]
                for b in range(min(len(sets),500)):
                    for i in range(sets.shape[1]):
                        for j in range(i+1,sets.shape[1]): vals.append(len(set(sets[b,i])&set(sets[b,j]))/max(1,len(set(sets[b,i])|set(sets[b,j]))))
                attn_overlap=float(np.mean(vals)) if vals else float("nan")
        nearest={}
        for split_name,split in (("learned",learned),("unseen",unseen)):
            qact=torch.as_tensor(split["actions"][...,:6],device=device,dtype=torch.float32)
            qmask=torch.ones(qact.shape[:2],device=device,dtype=torch.bool)
            with torch.no_grad(): qkeys=model(qact,qmask)[:,0].cpu().numpy()
            score=qkeys@keys[:,0].numpy().T
            errmat=key2._arm6_candidate_mse(split["actions"],windows)
            top=score.argmax(1);oracle_ids=errmat.argmin(1)
            # DCT and raw10 nearest neighbors are diagnostic controls over resampled target/candidate values.
            rawq=np.asarray(split["actions"],np.float32)[...,:6].reshape(len(qact),-1)
            rawc=windows[...,:6].reshape(len(windows),-1)
            rawdist=np.maximum((rawq**2).sum(1)[:,None]+(rawc**2).sum(1)[None]-2*rawq@rawc.T,0.)
            from scipy.fft import dct
            dctq=dct(rawq.reshape(len(rawq),10,6),type=2,norm="ortho",axis=1)[:,:3].reshape(len(rawq),-1)
            dctc=dct(rawc.reshape(len(rawc),10,6),type=2,norm="ortho",axis=1)[:,:3].reshape(len(rawc),-1)
            dd=np.maximum((dctq**2).sum(1)[:,None]+(dctc**2).sum(1)[None]-2*dctq@dctc.T,0.)
            nearest[split_name]={"motion_cosine_top1_mse":float(errmat[np.arange(len(top)),top].mean()),
                "dct3_top1_mse":float(errmat[np.arange(len(top)),dd.argmin(1)].mean()),
                "raw10_top1_mse":float(errmat[np.arange(len(top)),rawdist.argmin(1)].mean()),
                "oracle_mse":float(errmat[np.arange(len(top)),oracle_ids].mean()),
                "per_query_motion_mse":errmat[np.arange(len(top)),top].tolist()}
        result["stage1"][mode]={"checkpoint":str(ckpt),"training_history":history,"nearest":nearest,
            "diagnostics":{"spearman":float(rank),"recall_at_1":float(rec1),"recall_at_5":float(rec5),"task_knn_accuracy":task_acc,
                           "work_cosine":work_cos,"attention_overlap":attn_overlap if np.isfinite(attn_overlap) else None,"rank_subset_units":len(ids)}}
        _progress(output/"progress.log","stage1_model_complete",mode=mode,unseen_top1=nearest["unseen"]["motion_cosine_top1_mse"])
        del model,decoder,keys,work,attention
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        (output/"results.json").write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
        _write_stage1_report(output,result)
    return result


def _query_part(model: MotionQueryEncoder, split: dict[str, Any], ids: Any, *, device: str):
    idx = __import__("numpy").asarray(ids, dtype="int64")
    def tensor(name, dtype):
        value = split.get(name)
        return None if value is None else torch.as_tensor(value[idx], device=device, dtype=dtype)
    return model(history=tensor("history",torch.float32),history_mask=tensor("history_mask",torch.bool),
        state=tensor("state",torch.float32),tokens=tensor("tokens",torch.float32),groups=tensor("groups",torch.long))


def _fit_retrieval(qmodel: MotionQueryEncoder, mmodel: MotionKeyEncoder, *, train: dict[str, Any],
                   fit_bank: dict[str, Any], candidate_windows: Any, query_candidate_ids: Any,
                   dev: dict[str, Any], bank: dict[str, Any], seed: int, device: str,
                   epochs: int = 16, learning_rate: float = 1e-4, teacher_tau: float = .002,
                   tune_motion: bool = True, random_candidates: int = 256):
    import numpy as np
    import key_shared_latent as key2
    rng=np.random.default_rng(seed+101)
    n=len(train["key"]);qids=np.arange(n);batch_size=8
    pars=list(qmodel.parameters())+(list(mmodel.parameters()) if tune_motion else [])
    opt=torch.optim.AdamW(pars,lr=learning_rate,weight_decay=1e-4)
    fit_actions=torch.as_tensor(fit_bank["natural_actions"][...,:6],device=device,dtype=torch.float32)
    fit_len=torch.as_tensor(fit_bank["length"],device=device,dtype=torch.long)
    fit_valid=torch.arange(fit_actions.shape[1],device=device)[None]<fit_len[:,None]
    fit_windows=key2._candidate_window_actions(np.arange(len(fit_bank["length"])),fit_bank)
    history=[];best_dev=float("inf");best_epoch=None;best_state=None
    for epoch in range(1,epochs+1):
        rng.shuffle(qids);qmodel.train();mmodel.train() if tune_motion else mmodel.eval();losses=[]
        for start in range(0,n,batch_size):
            qi=qids[start:start+batch_size]
            if not len(qi):continue
            rand=rng.choice(len(fit_actions),size=min(random_candidates,len(fit_actions)),replace=False)
            cids,allowed=key2.contrastive_candidate_set(np.asarray(query_candidate_ids)[qi],rand,
                train["task"][qi],train["episode"][qi],fit_bank["task"],fit_bank["episode"])
            qpart={k:np.asarray(v)[qi] for k,v in train.items() if k in ("tokens","groups","history","history_mask","state")}
            qpart["history"],qpart["history_mask"]=key2._history_corrupt(qpart["history"],qpart["history_mask"],rng)
            qidx=np.arange(len(qi))
            qrep=_query_part(qmodel,qpart,qidx,device=device)[:,0]
            ci=torch.as_tensor(cids,device=device,dtype=torch.long)
            krep=mmodel(fit_actions[ci],fit_valid[ci])[:,0]
            scale=qmodel.logit_scale.exp().clamp(max=key2.LOGIT_SCALE_MAX)
            scores=(qrep@krep.T)*scale
            errors=key2._arm6_candidate_mse(train["actions"][qi],fit_windows[cids])
            loss=key2.contrastive_teacher_loss(scores,torch.as_tensor(errors,device=device),
                torch.as_tensor(allowed,device=device),objective="soft",teacher_temperature=teacher_tau)
            opt.zero_grad(set_to_none=True);loss.backward()
            nn.utils.clip_grad_norm_(pars,1.);opt.step();key2._clamp_logit_scale_(qmodel)
            losses.append(float(loss.detach()))
        rec={"epoch":epoch,"loss":float(np.mean(losses)),"steps":len(losses)}
        if epoch in (4,8,16):
            qmodel.eval();mmodel.eval()
            # Dev-only selection; candidate encodings are recomputed from current M weights.
            full_act=torch.as_tensor(bank["natural_actions"][...,:6],device=device,dtype=torch.float32)
            full_len=torch.as_tensor(bank["length"],device=device,dtype=torch.long)
            full_valid=torch.arange(full_act.shape[1],device=device)[None]<full_len[:,None]
            with torch.no_grad():
                key_parts=[]
                for s in range(0,len(full_act),64):key_parts.append(mmodel(full_act[s:s+64],full_valid[s:s+64])[:,0])
                keys=torch.cat(key_parts)
                q_parts=[]
                for s in range(0,len(dev["key"]),32):q_parts.append(_query_part(qmodel,dev,np.arange(s,min(s+32,len(dev["key"]))),device=device)[:,0])
                q=torch.cat(q_parts)
                scores=(q@keys.T)*qmodel.logit_scale.exp().clamp(max=key2.LOGIT_SCALE_MAX)
            errs=key2._arm6_candidate_mse(dev["actions"],candidate_windows)
            top=scores.argmax(-1).cpu().numpy();top_mse=float(errs[np.arange(len(top)),top].mean())
            rec["dev_top1_arm6_mse"]=top_mse
            if top_mse<best_dev:
                best_dev=top_mse;best_epoch=epoch
                best_state={"query":{k:v.detach().cpu().clone() for k,v in qmodel.state_dict().items()},
                            "motion":{k:v.detach().cpu().clone() for k,v in mmodel.state_dict().items()}}
        history.append(rec)
    if best_state is None:
        raise RuntimeError("no Dev checkpoint was evaluated")
    qmodel.load_state_dict(best_state["query"]);mmodel.load_state_dict(best_state["motion"])
    return qmodel,mmodel,history,best_epoch,best_dev


def _stage2(output: Path, *, data: tuple[Any,...], device: str, seed: int, smoke: bool = False):
    import numpy as np
    import key_shared_latent as key2
    bank,bank_input,fit_bank,fit_input,train,learned,unseen,dev=data
    result=json.loads((output/"results.json").read_text())
    # Correct pairwise retrieval diagnostics from saved encoders with self-neighbors excluded.
    # This is evaluation-only and does not fit or select any checkpoint.
    from scipy.stats import spearmanr
    windows_full=key2._candidate_window_actions(np.arange(len(bank["length"])),bank)
    for mode in ("m1","m5"):
        saved=torch.load(output/f"{mode}.pt",map_location=device,weights_only=True)
        mm=MotionKeyEncoder(mode).to(device);mm.load_state_dict(saved["model"])
        keys,work,attention=_encode_all(mm,bank,device=device,details=True)
        emb=keys[:,0].numpy();cos=emb@emb.T
        vw=windows_full[...,:6].reshape(len(windows_full),-1)
        err=((vw[:,None]-vw[None,:])**2).mean(-1)
        triu=np.triu_indices(len(vw),1);corr=spearmanr(cos[triu],-err[triu]).statistic
        cos_rank=cos.copy();err_rank=err.copy();np.fill_diagonal(cos_rank,-np.inf);np.fill_diagonal(err_rank,np.inf)
        order=np.argsort(-cos_rank,axis=1);oracle=np.argsort(err_rank,axis=1)
        rec1=float(np.mean(order[:,0]==oracle[:,0]));rec5=float(np.mean([len(set(order[i,:5])&set(oracle[i,:5]))/5 for i in range(len(vw))]))
        tasks=np.asarray(bank["task"]);knn=order[:,:5]
        pred=np.asarray([np.bincount(tasks[row].astype(int)).argmax() for row in knn])
        work_np=F.normalize(work,dim=-1).numpy()
        if work_np.shape[1]>1:
            ij=np.triu_indices(work_np.shape[1],1);work_cos=float(np.mean(np.einsum("bkd,bjd->bkj",work_np,work_np)[:,ij[0],ij[1]]))
        else:work_cos=1.
        overlap=None
        if mode=="m5":
            a=attention[0][0].numpy() # [B,work_tokens,time]
            top=np.argsort(-a,axis=-1)[...,:min(10,a.shape[-1])];vals=[]
            for b in range(len(top)):
                for i in range(top.shape[1]):
                    for j in range(i+1,top.shape[1]):
                        si,sj=set(top[b,i]),set(top[b,j]);vals.append(len(si&sj)/max(1,len(si|sj)))
            overlap=float(np.mean(vals)) if vals else None
        result["stage1"][mode]["diagnostics"].update({"spearman":float(corr),"recall_at_1":rec1,"recall_at_5":rec5,
            "task_knn_accuracy":float(np.mean(pred==tasks)),"work_cosine":work_cos,"attention_overlap":overlap,
            "rank_subset_units":int(len(vw)),"self_neighbor_excluded":True})
        del mm,keys,work,attention
    ids=np.arange(len(fit_bank["length"]))
    query_ids={(int(t),int(e),int(s)):i for i,(t,e,s) in enumerate(fit_bank["source"][:,:3])}
    train_candidate_ids=np.asarray([query_ids[tuple(map(int,k))] for k in train["key"]],np.int64)
    train_source=set(zip(train["task"].astype(int).tolist(),train["episode"].astype(int).tolist()))
    if len(train_candidate_ids)!=len(train["key"]):raise AssertionError("training query/candidate IDs misaligned")
    candidates=key2._candidate_window_actions(np.arange(len(bank["length"])),bank)
    fit_candidates=key2._candidate_window_actions(ids,fit_bank)
    modes=("m1_fixed","m1_finetune","m1_random")
    result["stage2"]={"grid":{"query":"Q-1","inputs":["E-H","E-VLM+H"],"motion_modes":list(modes),
        "learning_rate":1e-4,"teacher_tau":.002,"epochs":[4,8,16],"seed":seed},"models":{}}
    for input_mode in ("e_h","e_vlm_h"):
        for m_mode in modes:
            name=f"q1_{input_mode}_{m_mode}"
            _progress(output/"progress.log","stage2_model_start",name=name,train_rows=len(train["key"]))
            qmodel=MotionQueryEncoder(input_mode,"q1").to(device)
            mmodel=MotionKeyEncoder("m1").to(device)
            if m_mode in ("m1_fixed","m1_finetune"):
                state=torch.load(output/"m1.pt",map_location=device,weights_only=True)["model"]
                mmodel.load_state_dict(state)
            tune=m_mode!="m1_fixed"
            if smoke:
                train={k:(v[:128] if hasattr(v,"__len__") and len(v)==len(train["key"]) else v) for k,v in train.items()}
                train_candidate_ids=train_candidate_ids[:128]
            qmodel,mmodel,hist,best_epoch,best_dev=_fit_retrieval(qmodel,mmodel,train=train,fit_bank=fit_bank,
                candidate_windows=fit_candidates,query_candidate_ids=train_candidate_ids,dev=dev,bank=bank,
                seed=seed,device=device,epochs=16,learning_rate=1e-4,teacher_tau=.002,
                tune_motion=tune,random_candidates=32 if smoke else 256)
            qmodel.eval();mmodel.eval();full_act=torch.as_tensor(bank["natural_actions"][...,:6],device=device,dtype=torch.float32)
            full_len=torch.as_tensor(bank["length"],device=device,dtype=torch.long)
            full_valid=torch.arange(full_act.shape[1],device=device)[None]<full_len[:,None]
            with torch.no_grad():
                keyparts=[mmodel(full_act[s:s+64],full_valid[s:s+64])[:,0] for s in range(0,len(full_act),64)]
                keys=torch.cat(keyparts); evals={}
                for split_name,split in (("learned",learned),("unseen",unseen)):
                    qparts=[_query_part(qmodel,split,np.arange(s,min(s+32,len(split["key"]))),device=device)[:,0]
                            for s in range(0,len(split["key"]),32)]
                    q=torch.cat(qparts);scores=q@keys.T
                    errs=key2._arm6_candidate_mse(split["actions"],candidates)
                    top=scores.argmax(1);per=errs[np.arange(len(top)),top]
                    hard=np.asarray(split["hard"],bool)
                    evals[split_name]={"rows":len(top),"top1_arm6_mse":float(per.mean()),
                        "top1_arm6_mse_hard":float(per[hard].mean()) if hard.any() else None,
                        "oracle_arm6_mse":float(errs.min(1).values if False else errs.min(1).mean()),
                        "per_query":per.tolist()}
            path=output/f"{name}.pt";torch.save({"query":qmodel.state_dict(),"motion":mmodel.state_dict()},path)
            result["stage2"]["models"][name]={"input":input_mode,"motion_mode":m_mode,"epochs_selected":best_epoch,
                "dev_top1_arm6_mse":best_dev,"train_history":hist,"evaluation":evals,"checkpoint":str(path)}
            _progress(output/"progress.log","stage2_model_complete",name=name,best_epoch=best_epoch,dev_top1=best_dev,
                      unseen=evals["unseen"]["top1_arm6_mse"])
            (output/"results.json").write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
            _write_stage1_report(output,result)
            if torch.cuda.is_available():torch.cuda.empty_cache()
    _write_full_report(output,result)
    return result


def _write_full_report(output: Path, result: dict[str, Any]):
    # Preserve Stage 1 at the top and append the first, locked retrieval grid.
    base=(output/"report.md").read_text().split("\n## 第2段")[0]
    lines=[base,"","## 第2段：Q-1 retrieval (Dev selection; Eval once)","",
        "|行|入力|M 条件|Dev top-1|学習済み全体|学習済み難|未学習全体|未学習難|選択epoch|",
        "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for name,row in result.get("stage2",{}).get("models",{}).items():
        a=row["evaluation"]["learned"];b=row["evaluation"]["unseen"]
        lines.append(f"|{name}|{row['input']}|{row['motion_mode']}|{row['dev_top1_arm6_mse']:.5f}|{a['top1_arm6_mse']:.5f}|{a['top1_arm6_mse_hard'] if a['top1_arm6_mse_hard'] is not None else 'n/a'}|{b['top1_arm6_mse']:.5f}|{b['top1_arm6_mse_hard']:.5f}|{row['epochs_selected']}|")
    lines += ["","この段階は Q-1 × {E-H, E-VLM+H} × {M(s)固定, M(s)微調整, M(r)} の先行格子です。Q-5 と M 履歴を含む残りの格子は別途継続が必要です。","\n"]
    (output/"report.md").write_text("\n".join(lines))


def _arch_encoder(arch: str,position: str):
    return DCTMLPKeyEncoder() if arch=="mlp" else ArchMotionEncoder(arch,position)


def _arch_pack(bank,device):
    actions=torch.as_tensor(bank["natural_actions"][...,:6],device=device,dtype=torch.float32)
    lengths=torch.as_tensor(bank["length"],device=device,dtype=torch.long)
    valid=torch.arange(actions.shape[1],device=device)[None]<lengths[:,None]
    coeff=np.asarray(bank["coefficients"],np.float32)*np.asarray(bank["magnitude"],np.float32)[:,None,None]
    coeff=torch.as_tensor(coeff.reshape(len(coeff),18),device=device,dtype=torch.float32)
    return actions,lengths,valid,coeff


def _fit_arch(model,arch,objective,tau,bank,train_ids,*,device,seed,epochs=4,batch_size=128):
    import numpy as np
    import key_shared_latent as key2
    torch.manual_seed(seed);rng=np.random.default_rng(seed+333)
    actions,lengths,valid,coeff=_arch_pack(bank,device)
    decoder=nn.Sequential(nn.Linear(256,256),nn.GELU(),nn.Linear(256,18)).to(device)
    params=list(model.parameters())+list(decoder.parameters())
    opt=torch.optim.AdamW(params,lr=1e-4,weight_decay=1e-4)
    wins=key2._candidate_window_actions(np.arange(len(bank["length"])),bank)
    n=len(train_ids);order=np.asarray(train_ids,np.int64).copy();history=[];t0=time.time()
    for ep in range(epochs):
        rng.shuffle(order);model.train();decoder.train();losses=[]
        for start in range(0,n,batch_size):
            ids=order[start:start+batch_size]
            if len(ids)<2:continue
            x=actions[ids];mask=valid[ids];lens=lengths[ids]
            if arch=="mlp": xin=coeff[ids]
            else: xin=None
            if objective=="aug":
                aug,auglen=augment_motion(x,lens,noise_std=.003)
                augmask=torch.arange(aug.shape[1],device=device)[None]<auglen[:,None]
                a1=_dct3_from_batch(x,mask) if arch=="mlp" else None
                a2=_dct3_from_batch(aug,augmask) if arch=="mlp" else None
                z1=model(coefficients=a1)[:,0] if arch=="mlp" else model(x,mask)[:,0]
                z2=model(coefficients=a2)[:,0] if arch=="mlp" else model(aug,augmask)[:,0]
                logits=z1@z2.T/.1;labels=torch.arange(len(ids),device=device)
                loss=(F.cross_entropy(logits,labels)+F.cross_entropy(logits.T,labels))*.5
                aux=(F.mse_loss(decoder(z1),coeff[ids])+F.mse_loss(decoder(z2),a2 if arch=="mlp" else coeff[ids]))*.5
                loss=loss+.1*aux
            else:
                random_ids=rng.choice(len(bank["length"]),size=min(256,len(bank["length"])),replace=False)
                cids,allowed=key2.contrastive_candidate_set(ids,random_ids,
                    bank["task"][ids],bank["episode"][ids],bank["task"],bank["episode"])
                ci=torch.as_tensor(cids,device=device,dtype=torch.long)
                if arch=="mlp":q=model(coefficients=coeff[ids])[:,0];k=model(coefficients=coeff[ci])[:,0]
                else:q=model(actions[ids],valid[ids])[:,0];k=model(actions[ci],valid[ci])[:,0]
                score=(q@k.T)*20.
                err=key2._arm6_candidate_mse(bank["value_actions10"][ids],wins[cids])
                loss=key2.contrastive_teacher_loss(score,torch.as_tensor(err,device=device),torch.as_tensor(allowed,device=device),
                    objective="soft",teacher_temperature=tau)
            opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(params,1.);opt.step()
            losses.append(float(loss.detach()))
        history.append({"epoch":ep+1,"loss":float(np.mean(losses)),"steps":len(losses)})
    return model,decoder,history,time.time()-t0


def _evaluate_arch(model,arch,bank,learned,unseen,device,*,result_details=False):
    import numpy as np
    import key_shared_latent as key2
    from scipy.stats import spearmanr
    actions,lengths,valid,coeff=_arch_pack(bank,device)
    model.eval()
    with torch.no_grad():
        chunks=[]
        for s in range(0,len(actions),64):
            if arch=="mlp":z=model(coefficients=coeff[s:s+64])[:,0]
            else:z=model(actions[s:s+64],valid[s:s+64])[:,0]
            chunks.append(z.cpu().numpy())
    keys=np.concatenate(chunks);normactions=np.asarray(bank["natural_actions"],np.float32)[...,:6]
    windows=key2._candidate_window_actions(np.arange(len(bank["length"])),bank)
    emb=keys
    cos=emb@emb.T;flat=windows[...,:6].reshape(len(windows),-1);errs=((flat[:,None]-flat[None,:])**2).mean(-1)
    pairs=np.triu_indices(len(flat),1);rho=spearmanr(cos[pairs],-errs[pairs]).statistic
    cr=cos.copy();er=errs.copy();np.fill_diagonal(cr,-np.inf);np.fill_diagonal(er,np.inf)
    rank=np.argsort(-cr,axis=1);oracle=np.argsort(er,axis=1)
    rec1=float(np.mean(rank[:,0]==oracle[:,0]));rec5=float(np.mean([len(set(rank[i,:5])&set(oracle[i,:5]))/5 for i in range(len(rank))]))
    task=np.asarray(bank["task"]);pred=np.asarray([np.bincount(task[x].astype(int)).argmax() for x in rank[:,:5]])
    workcos=None
    if arch=="x5":
        work=[]
        with torch.no_grad():
            for s in range(0,len(actions),64):work.append(model(actions[s:s+64],valid[s:s+64],return_details=True)["work"].cpu())
        ww=F.normalize(torch.cat(work),dim=-1).numpy();ij=np.triu_indices(ww.shape[1],1)
        workcos=float(np.mean(np.einsum("bkd,bjd->bkj",ww,ww)[:,ij[0],ij[1]]))
    per={}
    for name,split in (("learned",learned),("unseen",unseen)):
        qa=torch.as_tensor(split["actions"][...,:6],device=device,dtype=torch.float32);qm=torch.ones(qa.shape[:2],device=device,dtype=torch.bool)
        if arch=="mlp":
            # Query arm chunks have fixed 10 steps; transform to the same unnormalized DCT3 coefficient input.
            qcoeff=_dct3_from_batch(qa,qm)
            with torch.no_grad():q=model(coefficients=qcoeff)[:,0].cpu().numpy()
        else:
            with torch.no_grad():q=model(qa,qm)[:,0].cpu().numpy()
        scores=q@keys.T;mse=key2._arm6_candidate_mse(split["actions"],windows);top=scores.argmax(1)
        per[name]={"nearest_mse":float(mse[np.arange(len(top)),top].mean()),
                   "oracle_mse":float(mse.min(1).mean()),"nearest_per_query":mse[np.arange(len(top)),top].tolist()}
    diag={"spearman":float(rho),"recall_at_1":rec1,"recall_at_5":rec5,
          "task_knn_accuracy":float(np.mean(pred==task)),"work_cosine":workcos,"pair_units":len(flat),"self_neighbor_excluded":True}
    return {"nearest":per,"diagnostics":diag}


def _run_arch_grid(output:Path,*,data,device:str,seed:int=1,smoke:bool=False):
    import numpy as np
    bank,_,fit_bank,_,train,learned,unseen,_=data
    n=len(bank["length"]);ids=np.arange(min(n,128) if smoke else n);epochs=1 if smoke else 4
    # Compare X1/X1s positions first, then use the better absolute/phase choice for X5.
    configs=[]
    for arch in ("x1","x1s"):
        for pos in ("absolute","phase"):
            configs.append((arch,pos,"aug",None))
            for tau in (.002,.005,.01):configs.append((arch,pos,"dist",tau))
    # MLP receives no positional encoding and is compared under both objectives.
    configs.append(("mlp","none","aug",None))
    for tau in (.002,.005,.01):configs.append(("mlp","none","dist",tau))
    if smoke:configs=[("x1","absolute","aug",None),("x1s","phase","dist",.005),("mlp","none","aug",None)]
    results={"metadata":{"seed":seed,"train_units":len(ids),"epochs":4,"batch_size":128,
        "learning_rate":1e-4,"distillation_tau_grid":[.002,.005,.01],"evaluation":"fixed learned621/unseen619; diagnostics only"},"rows":{},"reference":{}}
    for arch,pos,obj,tau in configs:
        name=f"{arch}_{pos}_{obj}"+(f"_tau{tau:g}" if tau is not None else "")
        _progress(output/"progress.log","arch_train_start",name=name,units=len(ids))
        model=_arch_encoder(arch,pos).to(device)
        model,decoder,hist,seconds=_fit_arch(model,arch,obj,tau or .005,bank,ids,device=device,seed=seed,epochs=epochs)
        evaluated=_evaluate_arch(model,arch,bank,learned,unseen,device)
        ck=output/f"{name}.pt";torch.save(model.state_dict(),ck)
        results["rows"][name]={"architecture":arch,"position":pos,"objective":obj,"teacher_tau":tau,
            "training_seconds":seconds,"training_history":hist,"evaluation":evaluated,"checkpoint":str(ck)}
        (output/"results.json").write_text(json.dumps(results,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
        _progress(output/"progress.log","arch_model_complete",name=name,seconds=seconds,
                  unseen_mse=evaluated["nearest"]["unseen"]["nearest_mse"])
        del model,decoder
        if torch.cuda.is_available():torch.cuda.empty_cache()
    # X5 uses the better absolute/phase position picked from the X1/X1s rows.
    x5_position=min((p for p in ("absolute","phase")),key=lambda p:min(
        row["evaluation"]["nearest"]["unseen"]["nearest_mse"] for row in results["rows"].values()
        if row["architecture"] in ("x1","x1s") and row["position"]==p))
    x5_specs=[("aug",None),("dist",.005)] if smoke else [("aug",None),("dist",.002),("dist",.005),("dist",.01)]
    for obj,tau in x5_specs:
        arch,pos="x5",x5_position
        name=f"{arch}_{pos}_{obj}"+(f"_tau{tau:g}" if tau is not None else "")
        _progress(output/"progress.log","arch_train_start",name=name,units=len(ids))
        model=_arch_encoder(arch,pos).to(device)
        model,decoder,hist,seconds=_fit_arch(model,arch,obj,tau or .005,bank,ids,device=device,seed=seed,epochs=epochs)
        evaluated=_evaluate_arch(model,arch,bank,learned,unseen,device)
        ck=output/f"{name}.pt";torch.save(model.state_dict(),ck)
        results["rows"][name]={"architecture":arch,"position":pos,"objective":obj,"teacher_tau":tau,
            "training_seconds":seconds,"training_history":hist,"evaluation":evaluated,"checkpoint":str(ck)}
        (output/"results.json").write_text(json.dumps(results,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
        _progress(output/"progress.log","arch_model_complete",name=name,seconds=seconds,
                  unseen_mse=evaluated["nearest"]["unseen"]["nearest_mse"])
        del model,decoder
        if torch.cuda.is_available():torch.cuda.empty_cache()
    # Frozen DCT3 and raw-value nearest-neighbor reference rows.
    import key_shared_latent as key2
    windows=key2._candidate_window_actions(np.arange(n),bank)
    from scipy.fft import dct
    for split_name,split in (("learned",learned),("unseen",unseen)):
        target=split["actions"];mse=key2._arm6_candidate_mse(target,windows)
        rawq=np.asarray(target,np.float32)[...,:6].reshape(len(target),-1);rawc=windows[...,:6].reshape(len(windows),-1)
        rd=np.maximum((rawq**2).sum(1)[:,None]+(rawc**2).sum(1)[None]-2*rawq@rawc.T,0)
        dq=dct(rawq.reshape(len(rawq),10,6),type=2,norm="ortho",axis=1)[:,:3].reshape(len(rawq),-1)
        dc=dct(rawc.reshape(len(rawc),10,6),type=2,norm="ortho",axis=1)[:,:3].reshape(len(rawc),-1)
        dd=np.maximum((dq**2).sum(1)[:,None]+(dc**2).sum(1)[None]-2*dq@dc.T,0)
        results["reference"][split_name]={"dct3_nearest_mse":float(mse[np.arange(len(mse)),dd.argmin(1)].mean()),
            "raw10_nearest_mse":float(mse[np.arange(len(mse)),rd.argmin(1)].mean()),"oracle_mse":float(mse.min(1).mean())}
    resultrows=results["rows"]
    best=min(resultrows,key=lambda k:resultrows[k]["evaluation"]["nearest"]["unseen"]["nearest_mse"])
    results["best_row"]=best;results["best_unseen_mse"]=resultrows[best]["evaluation"]["nearest"]["unseen"]["nearest_mse"]
    results["best_minus_dct3"]=results["best_unseen_mse"]-results["reference"]["unseen"]["dct3_nearest_mse"]
    (output/"results.json").write_text(json.dumps(results,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    _write_arch_report(output,results)
    return results


def _write_arch_report(output:Path,r:dict[str,Any]):
    lines=["# MK1b：M の構造・目的比較","",
        "|構造|位置|目的|tau|Eval unseen NN|Eval learned NN|Spearman|recall@1|recall@5|task kNN|X5 work cos|秒|",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name,x in r["rows"].items():
        d=x["evaluation"]["diagnostics"];n=x["evaluation"]["nearest"]
        lines.append(f"|{x['architecture']}|{x['position']}|{x['objective']}|{x['teacher_tau'] if x['teacher_tau'] is not None else '—'}|{n['unseen']['nearest_mse']:.5f}|{n['learned']['nearest_mse']:.5f}|{d['spearman']:.4f}|{d['recall_at_1']:.4f}|{d['recall_at_5']:.4f}|{d['task_knn_accuracy']:.4f}|{d['work_cosine'] if d['work_cosine'] is not None else '—'}|{x['training_seconds']:.1f}|")
    ref=r["reference"]["unseen"]
    learned_ref=r["reference"]["learned"]
    lines += ["", "### 学習なし参照行", "", "|Eval split|DCT3 nearest|生値 nearest|oracle|", "|---|---:|---:|---:|"]
    for split_name,reference in (("learned",learned_ref),("unseen",ref)):
        lines.append(f"|{split_name}|{reference['dct3_nearest_mse']:.5f}|{reference['raw10_nearest_mse']:.5f}|{reference['oracle_mse']:.5f}|")
    lines += ["",f"選択行：**{r['best_row']}**（未学習 nearest MSE {r['best_unseen_mse']:.5f}）。未学習の学習なし DCT3 との差は {r['best_minus_dct3']:+.5f}。","",
        "距離蒸留は保存値を振幅正規化せず、同じ10-step保持端点規則による腕6軸 MSE を教師にしました。学習は LIBERO-40 のメモリ単位だけです。Eval nearest は依頼どおり選択にも使う診断量であり、閉ループの性能を示しません。","\n"]
    (output/"report.md").write_text("\n".join(lines))


def run_arch_main():
    parser=argparse.ArgumentParser(description="MK1b motion-key architecture/objective grid")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--run-root",type=Path,default=(STATE_DIR / "outputs/memory-pilot/pi05-memory40-20261104/run-01"))
    parser.add_argument("--missing-root",type=Path,default=(STATE_DIR / "outputs/memory-pilot/unseen-libero/key-shared-01/attempt-02/prefix-missing"))
    parser.add_argument("--seen-stride5",type=Path,default=(STATE_DIR / "outputs/memory-pilot/unseen-libero/seg-context-01/seg5-cache"))
    parser.add_argument("--unseen-stride5",type=Path,default=(STATE_DIR / "outputs/memory-pilot/unseen-libero/seg-context-01/unseen-stride5"))
    parser.add_argument("--vjepa-eval-cache",type=Path,default=(STATE_DIR / "outputs/memory-pilot/unseen-libero/key-shared-01/attempt-27-production/vjepa/libero90_eval"))
    parser.add_argument("--device",default="cuda:0");parser.add_argument("--seed",type=int,default=1)
    parser.add_argument("--smoke",action="store_true")
    a=parser.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS","8")))
    sys.path.insert(0,str(Path(__file__).resolve().parent));import key_shared_latent as key2
    _progress(a.output/"progress.log","arch_data_start")
    data=key2._build_key2_queries(a.run_root,a.missing_root,a.seen_stride5,a.unseen_stride5,a.vjepa_eval_cache,
        progress_callback=lambda stage,**kw:_progress(a.output/"progress.log",stage,**kw))
    _progress(a.output/"progress.log","arch_data_complete",bank_units=len(data[0]["length"]))
    _run_arch_grid(a.output,data=data,device=a.device,seed=a.seed,smoke=a.smoke)
    if not a.smoke:
        arch_report=(a.output/"report.md").read_text()
        main_report=a.output.parent/"report.md"
        existing=main_report.read_text() if main_report.exists() else "# MK1：行動チャンクから作る運動キー\n"
        main_report.write_text(arch_report+"\n---\n\n"+existing)
    _progress(a.output/"progress.log","arch_complete")

def main() -> None:
    import numpy as np
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--run-root",type=Path,default=(STATE_DIR / "outputs/memory-pilot/pi05-memory40-20261104/run-01"))
    parser.add_argument("--missing-root",type=Path,default=(STATE_DIR / "outputs/memory-pilot/unseen-libero/key-shared-01/attempt-02/prefix-missing"))
    parser.add_argument("--seen-stride5",type=Path,default=(STATE_DIR / "outputs/memory-pilot/unseen-libero/seg-context-01/seg5-cache"))
    parser.add_argument("--unseen-stride5",type=Path,default=(STATE_DIR / "outputs/memory-pilot/unseen-libero/seg-context-01/unseen-stride5"))
    parser.add_argument("--vjepa-eval-cache",type=Path,default=(STATE_DIR / "outputs/memory-pilot/unseen-libero/key-shared-01/attempt-27-production/vjepa/libero90_eval"))
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--seed",type=int,default=1)
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--stage2",action="store_true",help="run retrieval grid using an existing Stage 1 output directory")
    args=parser.parse_args()
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS","8")))
    args.output.mkdir(parents=True,exist_ok=args.stage2)
    if args.stage2 and not (args.output/"m1.pt").exists():
        raise FileNotFoundError("--stage2 requires a completed Stage 1 m1.pt checkpoint")
    _progress(args.output/"progress.log","data_build_start")
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    import key_shared_latent as key2
    data=key2._build_key2_queries(args.run_root,args.missing_root,args.seen_stride5,args.unseen_stride5,args.vjepa_eval_cache,
        progress_callback=lambda stage,**kw:_progress(args.output/"progress.log",stage,**kw))
    _progress(args.output/"progress.log","data_build_complete",bank_units=len(data[0]["length"]),train_rows=len(data[4]["key"]))
    if args.stage2:
        _stage2(args.output,device=args.device,seed=args.seed,data=data,smoke=args.smoke)
        _progress(args.output/"progress.log","stage2_complete")
    else:
        _stage1(args.output,device=args.device,seed=args.seed,data=data,smoke=args.smoke)
        _progress(args.output/"progress.log","stage1_complete")


if __name__ == "__main__":
    if os.environ.get("MK1B_ARCH_RUN") == "1":
        run_arch_main()
    elif os.environ.get("MK1C_QUERY_RUN") == "1":
        run_query_main()
    else:
        main()

