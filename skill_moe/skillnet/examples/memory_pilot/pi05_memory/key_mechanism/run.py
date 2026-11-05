#!/usr/bin/env python3
"""Train/select attention-based memory keys without Long leakage; evaluate once on Long."""
from __future__ import annotations

import argparse, json, math, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from models import M1AttentionKey, M2SalientTokens, M3CandidateJudge, DirectActionHead

import os as _os
_REPO = Path(__file__).resolve().parents[6]
_STATE_DIR = Path(_os.environ.get('SKILLNET_STATE_DIR', _REPO / '.skillnet-state'))
_PI05_ROOT = Path(_os.environ.get('SKILLNET_PI05_ROOT', _STATE_DIR / 'outputs/memory-pilot/pi05-memory40-20261104'))
_PI05_CODE = Path(__file__).resolve().parents[1]
ROOT=_PI05_ROOT
RUN=ROOT/'run-01'; MEMORY=ROOT/'skill-memories-01'/'raw-rows-noLong'
GAP=ROOT/'memory-gap-01'; TOPK=ROOT/'topk-long-01'/'production'; LOCAL=ROOT/'local-key-02'
OUT=ROOT/'key-mechanism-01'; Q80=.01549410422449; SEED=20261105
SUITES=((10,20,'goal'),(20,30,'object'),(30,40,'spatial'))
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'

def _load_rows():
    return {p.stem:np.load(p,mmap_mode='r',allow_pickle=False) for p in (RUN/'rows').glob('*.npy')}

def _history(rows):
    n=len(rows['frame']);lookup={(int(t),int(e),int(f)):i for i,(t,e,f) in enumerate(zip(rows['task'],rows['episode'],rows['frame']))}
    actions=np.asarray(rows['actions'][:,:,:7],np.float32);out=np.zeros((n,79),np.float32)
    for i,(e,f) in enumerate(zip(rows['episode'],rows['frame'])):
        j=lookup.get((int(rows['task'][i]),int(e),int(f)-10))
        if j is not None:out[i,:70]=actions[j].reshape(-1);out[i,78]=1.
    out[:,:8]=np.asarray(rows['state'][:,:8],np.float32)
    return out

def choose_holdout_tasks(seed=SEED):
    rng=np.random.default_rng(seed);picked=[]
    for lo,hi,_ in SUITES:picked.extend(sorted(rng.choice(np.arange(lo,hi),2,replace=False).tolist()))
    return np.asarray(picked,np.int32)

def assert_no_long(ids,rows,label):
    if np.any(np.asarray(rows['task'][ids])<10):raise AssertionError(f'Long data leaked into {label}')

def make_splits(rows,holdout=None):
    holdout=choose_holdout_tasks() if holdout is None else np.asarray(holdout,np.int32)
    task=np.asarray(rows['task']);mem=np.asarray(rows['memory'],bool);val=np.asarray(rows['validation'],bool)
    train=np.flatnonzero((task>=10)&~np.isin(task,holdout)&~mem&~val)
    dev=np.flatnonzero(np.isin(task,holdout)&val)
    final_mem=np.flatnonzero(mem&(task>=10))
    dev_mem=final_mem[~np.isin(task[final_mem],holdout)]
    long=np.flatnonzero(np.asarray(rows['validation'],bool)&(task<10))
    return {'holdout_tasks':holdout,'train':train,'dev':dev,'dev_mem':dev_mem,'memory':final_mem,'long':long}

def _norm_stats(path):
    z=np.load(path,allow_pickle=False);return z

def _gap_threshold(gap_path=GAP):
    """Hard-row threshold; prefers summary_corrected.json, falls back to summary.json."""
    for name in ('summary_corrected.json','summary.json'):
        candidate=gap_path/name
        if candidate.exists():return float(json.loads(candidate.read_text())['threshold'])
    raise FileNotFoundError(f'neither summary_corrected.json nor summary.json in {gap_path}')

def _cpu_state_dict(model):
    """CPU copy of a model's state dict without moving the live model."""
    return {k:v.cpu() for k,v in model.state_dict().items()}

def _torch_batch(ids,rows,tokens,groups,context,*,use_text=False,device=DEVICE):
    ids=np.asarray(ids,np.int64);n=712 if use_text else 512
    tok=np.asarray(tokens[ids,:n],np.float32)
    mask=(np.asarray(groups[ids,:n])!=-1)
    return (torch.as_tensor(tok,device=device),torch.as_tensor(context[ids],device=device),
            torch.as_tensor(mask,device=device,dtype=torch.bool))

def _action_flat(rows,ids):return np.asarray(rows['actions'][ids,:,:7],np.float32).reshape(len(ids),70)

def _pair_distance(a,b):
    a=np.asarray(a,np.float32).reshape(len(a),70);b=np.asarray(b,np.float32).reshape(len(b),70)
    return np.maximum((np.sum(a*a,1)[:,None]+np.sum(b*b,1)[None]-2*a@b.T)/70.,0.)

def _select_tau(dev_ids,rows,*,seed=SEED,target_rate=.05):
    rng=np.random.default_rng(seed+3);n=len(dev_ids)
    if n<2:raise ValueError('key-dev needs at least two rows')
    ii=rng.integers(n,size=50000);jj=rng.integers(n,size=50000);task=np.asarray(rows['task'][dev_ids])
    cross=task[ii]!=task[jj];a=_action_flat(rows,dev_ids[ii]);b=_action_flat(rows,dev_ids[jj])
    d=np.mean((a-b)**2,axis=1);rates={str(t):float(np.mean(d[cross]<=t)) for t in (.005,.01,.02)}
    tau=min((.005,.01,.02),key=lambda t:(abs(rates[str(t)]-target_rate),t))
    return float(tau),rates

def _positive_pairs(train_ids,rows,tau,*,seed=SEED,max_anchors=4096):
    rng=np.random.default_rng(seed+19);n=len(train_ids);anchors=rng.choice(train_ids,min(n,max_anchors),replace=False)
    train_actions=_action_flat(rows,train_ids);norm=np.sum(train_actions*train_actions,axis=1);task=np.asarray(rows['task'][train_ids])
    idpos={int(r):i for i,r in enumerate(train_ids)};pairs=[]
    for st in range(0,len(anchors),128):
        qids=anchors[st:st+128];q=_action_flat(rows,qids);d=np.maximum((np.sum(q*q,1)[:,None]+norm[None]-2*q@train_actions.T)/70.,0.)
        for qi,row in enumerate(qids):
            okay=task!=int(rows['task'][row]);d[qi,~okay]=np.inf
            valid=np.flatnonzero(d[qi]<=tau)
            if len(valid):
                # Sample among near cross-task positives rather than always taking an exact duplicate.
                near=valid[np.argsort(d[qi,valid])[:min(8,len(valid))]]
                pairs.append((int(row),int(train_ids[int(rng.choice(near))])))
    if len(pairs)<32:raise RuntimeError(f'only {len(pairs)} cross-task positive pairs for tau={tau}; increase selected tau')
    return np.asarray(pairs,np.int64)

def _contrastive_loss(z,actions,tasks,tau):
    sim=z@z.T/.12
    dist=torch.mean((actions[:,None]-actions[None])**2,dim=-1)
    same=tasks[:,None]==tasks[None,:];eye=torch.eye(len(z),device=z.device,dtype=torch.bool)
    pos=(dist<=tau)&~same&~eye
    valid=pos.any(1)
    if not bool(valid.any()):return z.sum()*0.
    sim=sim.masked_fill(eye,-1e4)
    den=torch.logsumexp(sim,dim=1);num=torch.logsumexp(sim.masked_fill(~pos,-1e4),dim=1)
    return (den[valid]-num[valid]).mean()

def _log(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:f.write(json.dumps(obj,ensure_ascii=False,allow_nan=True)+'\n')

def _train_m1(name,model,train_ids,dev_ids,dev_mem,rows,tokens,groups,context,tau,*,epochs=4,threads=16,smoke=False,progress=None):
    model.to(DEVICE).train();opt=torch.optim.AdamW(model.parameters(),lr=4e-4,weight_decay=1e-4)
    pairs=_positive_pairs(train_ids,rows,tau,seed=SEED+sum(map(ord,name)),max_anchors=(512 if smoke else 4096))
    rng=np.random.default_rng(SEED+41);pair_per_epoch=min(len(pairs),512 if smoke else 2048);batch_pairs=8 if smoke else 16
    history=[];best=None;best_dev=float('inf')
    for ep in range(epochs):
        model.train();order=rng.choice(len(pairs),pair_per_epoch,replace=len(pairs)<pair_per_epoch);losses=[]
        for st in range(0,len(order),batch_pairs):
            ps=pairs[order[st:st+batch_pairs]]
            if len(ps)<2:continue
            ids=np.r_[ps[:,0],ps[:,1]];tok,ctx,mask=_torch_batch(ids,rows,tokens,groups,context,use_text=model.token_count>512)
            aa=torch.as_tensor(_action_flat(rows,ids),device=DEVICE);tt=torch.as_tensor(rows['task'][ids],device=DEVICE)
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=DEVICE=='cuda'):
                z=model(tok,ctx,mask)
            loss=_contrastive_loss(z.float(),aa,tt,tau)
            opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();losses.append(float(loss.detach().cpu()))
        # key-Dev evaluation is task-held-out; candidates exclude the six held-out tasks.
        dev_key=_encode_keys(model,dev_ids,rows,tokens,groups,context,smoke=smoke)
        mem_key=_encode_keys(model,dev_mem,rows,tokens,groups,context,smoke=smoke)
        cand,score,errs=_retrieve(dev_key,mem_key,dev_ids,dev_mem,rows,rows['actions'],k=64)
        dm=_action_mse_rows(rows,dev_ids,dev_mem)
        d=float(errs[:,0].mean());history.append({'epoch':ep+1,'train_loss':float(np.mean(losses)),'dev_top1_mse':d})
        if d<best_dev:best_dev=d;best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if progress is not None:_log(progress,{'event':'epoch','model':name,**history[-1],'elapsed':time.time()})
    if best is not None:model.load_state_dict(best)
    model.to(DEVICE).eval()
    return model,{'tau':tau,'positive_pairs':len(pairs),'epochs':epochs,'history':history,'best_dev_top1':best_dev}

@torch.no_grad()
def _encode_keys(model,ids,rows,tokens,groups,context,*,batch=8,smoke=False,need_tokens=False,need_attention=False):
    model.eval();keys=[];h_all=[];a_all=[];masks=[]
    n=512 if getattr(model,'token_count',512)<=512 else 712
    ids=np.asarray(ids,np.int64)
    for st in range(0,len(ids),batch):
        q=ids[st:st+batch];tok,ctx,mask=_torch_batch(q,rows,tokens,groups,context,use_text=n>512)
        if need_tokens or need_attention:key,h,att=model.encode_tokens(tok,ctx,mask)
        else:key=model(tok,ctx,mask)
        keys.append(key.float().cpu().numpy())
        if need_tokens:h_all.append(h.half().cpu().numpy());masks.append(mask.cpu().numpy())
        if need_attention:a_all.append(att.float().cpu().numpy())
    result=[np.concatenate(keys,0)]
    if need_tokens:result += [np.concatenate(h_all,0),np.concatenate(masks,0)]
    if need_attention:result += [np.concatenate(a_all,0)]
    return result[0] if len(result)==1 else tuple(result)

def _retrieve(qfeat,mfeat,qids,mids,rows,actions,k=64):
    q=np.asarray(qfeat,np.float32);m=np.asarray(mfeat,np.float32);q=q/np.maximum(np.linalg.norm(q,axis=1,keepdims=True),1e-8);m=m/np.maximum(np.linalg.norm(m,axis=1,keepdims=True),1e-8)
    score=q@m.T;mt=np.asarray(rows['task'][mids]);cand=np.empty((len(qids),k),np.int64);sc=np.empty((len(qids),k),np.float32)
    for i,row in enumerate(qids):
        vals=score[i].copy();vals[mt==int(rows['task'][row])]=-np.inf
        ix=np.lexsort((mids,-vals))[:k];cand[i]=mids[ix];sc[i]=vals[ix]
    errs=_candidate_action_mse(rows,qids,cand)
    return cand,sc,errs

def _candidate_action_mse(rows,qids,cands):
    q=np.asarray(rows['actions'][qids,:,:7],np.float32);c=np.asarray(rows['actions'][cands,:,:7],np.float32)
    return np.mean((q[:,None]-c)**2,axis=(-1,-2))

def _action_mse_rows(rows,qids,mids,chunk=64):
    q=np.asarray(rows['actions'][qids,:,:7],np.float32).reshape(len(qids),70)
    mem=np.asarray(rows['actions'][mids,:,:7],np.float32).reshape(len(mids),70)
    out=np.empty((len(qids),len(mids)),np.float32)
    for st in range(0,len(qids),chunk):out[st:st+chunk]=np.mean((q[st:st+chunk,None]-mem[None])**2,axis=-1)
    return out

def _best_metrics(name,qids,cands,scores,rows,all_mse,conf=None):
    errs=_candidate_action_mse(rows,qids,cands);truth=np.asarray(rows['actions'][qids,:,:7],np.float32)
    pred=np.asarray(rows['actions'][cands[:,0],:,:7],np.float32)
    true_dir=truth[:,:,:3].sum(1);pred_dir=pred[:,:,:3].sum(1)
    den=np.linalg.norm(true_dir,axis=1)*np.linalg.norm(pred_dir,axis=1)
    direction=np.divide(np.sum(true_dir*pred_dir,axis=1),den,out=np.zeros(len(qids),np.float32),where=den>1e-8)
    grip=float(np.mean(np.sign(truth[:,:,6])==np.sign(pred[:,:,6])))
    if all_mse is None:
        recall=np.zeros(len(qids),np.float32);valid=np.zeros(len(qids),bool)
    else:
        rel=all_mse<=.01;denom=rel.sum(1);valid=denom>0
        recall=np.divide((errs[:,:8]<=.01).sum(1),denom,out=np.zeros(len(qids),np.float32),where=valid)
    from scipy.stats import spearmanr
    rho=[float(spearmanr(scores[i],errs[i]).statistic) if np.std(errs[i])>0 else float('nan') for i in range(len(qids))]
    out={'name':name,'candidate_rows':np.asarray(cands,np.int64),'scores':np.asarray(scores,np.float32),'candidate_mse':errs,
         'top1_mse':errs[:,0],'top5_best_mse':errs[:,:5].min(1),'recall8':recall,'recall_valid':valid,
         'spearman_within':np.asarray(rho,np.float32),'direction_cosine':direction,'gripper_sign_agreement':grip}
    if conf is not None:out['confidence']=np.asarray(conf,np.float32)
    return out

def _bootstrap_diff(a,b,episodes,*,seed=SEED,reps=5000):
    a=np.asarray(a,np.float64);b=np.asarray(b,np.float64);ep=np.asarray(episodes,np.int64);d=a-b;uniq=np.unique(ep);groups={u:np.flatnonzero(ep==u) for u in uniq}
    rng=np.random.default_rng(seed);vals=np.empty(reps)
    for i in range(reps):
        us=rng.choice(uniq,len(uniq),replace=True);ix=np.concatenate([groups[u] for u in us]);vals[i]=d[ix].mean()
    return {'mean':float(d.mean()),'ci95':[float(np.quantile(vals,.025)),float(np.quantile(vals,.975))],'episodes':len(uniq),'rows':len(a),'reps':reps}

def _risk_metrics(conf,error,episodes):
    from scipy.stats import spearmanr
    c=np.asarray(conf,np.float32);e=np.asarray(error,np.float32)
    rho=float(spearmanr(c,e).statistic);risk={};idx=np.argsort(-c)
    for pct in (20,40,60,80):
        keep=idx[:max(1,int(np.ceil(len(idx)*pct/100)))]
        risk[str(pct)]={'mean_error':float(e[keep].mean()),'fraction':pct/100}
    ix=np.argsort(-c)[:max(1,int(np.ceil(len(c)*.2)))];mask=np.zeros(len(c),bool);mask[ix]=True
    diff=e[mask].mean()-e.mean();ep=np.asarray(episodes);uniq=np.unique(ep);gr={u:np.flatnonzero(ep==u) for u in uniq};rng=np.random.default_rng(SEED+99);bs=[]
    for _ in range(5000):
        es=rng.choice(uniq,len(uniq),replace=True);ids=np.concatenate([gr[u] for u in es]);sel=mask[ids]
        if sel.any():bs.append(e[ids][sel].mean()-e[ids].mean())
    return {'spearman_confidence_error':rho,'spearman_confidence_error_quality':-rho,'risk_coverage':risk,
            'top20_minus_all':float(diff),'top20_minus_all_ci95':[float(np.quantile(bs,.025)),float(np.quantile(bs,.975))]}

def _metrics_summary(rows,qids,hard,all_mse,methods,episodes):
    from scipy.stats import spearmanr
    summary={};cis={};errors={}
    for name,m in methods.items():
        err=np.asarray(m['top1_mse'],np.float32);errors[name]=err
        summary[name]={'top1_all':float(err.mean()),'top1_hard':float(err[hard].mean()),
           'top5_best_all':float(np.mean(m['top5_best_mse'])) if m.get('top5_best_mse') is not None else None,
           'top5_best_hard':float(np.mean(m['top5_best_mse'][hard])) if m.get('top5_best_mse') is not None else None,
           'recall8_all':float(np.mean(m['recall8'][m['recall_valid']])) if m.get('recall8') is not None and np.any(m['recall_valid']) else None,
           'recall8_hard':float(np.mean(m['recall8'][hard&m['recall_valid']])) if m.get('recall8') is not None and np.any(hard&m['recall_valid']) else None,
           'spearman_within_all':float(np.nanmean(m['spearman_within'])) if m.get('spearman_within') is not None else None,
           'spearman_within_hard':float(np.nanmean(m['spearman_within'][hard])) if m.get('spearman_within') is not None else None,
           'direction_cosine_all':float(np.mean(m['direction_cosine'])) if m.get('direction_cosine') is not None else None,
           'direction_cosine_hard':float(np.mean(m['direction_cosine'][hard])) if m.get('direction_cosine') is not None else None,
           'gripper_sign_agreement':m.get('gripper_sign_agreement')}
    return summary

def _train_encoder(name,kind,train_ids,dev_ids,dev_mem,rows,tokens,groups,context,tau,*,epochs,smoke=False,progress=None):
    model=(M1AttentionKey(slots=int(name.rsplit('slots',1)[-1]),dim=96,token_count=512) if kind=='m1' else
           M1AttentionKey(slots=4,dim=96,token_count=712) if kind=='m1_text' else M2SalientTokens(dim=96,token_count=512))
    model,meta=_train_m1(name,model,train_ids,dev_ids,dev_mem,rows,tokens,groups,context,tau,epochs=epochs,smoke=smoke,progress=progress)
    if kind=='m2':
        # M2 selects candidates from pooled descriptors, then token-weighted MaxSim reranks the shortlist.
        qk=_encode_keys(model,dev_ids,rows,tokens,groups,context,smoke=smoke)
        mk=_encode_keys(model,dev_mem,rows,tokens,groups,context,smoke=smoke)
        qh,qw=_encode_tokens(model,dev_ids,rows,tokens,groups,context)
        mh,mw=_encode_tokens(model,dev_mem,rows,tokens,groups,context)
        c,_,_=_retrieve(qk,mk,dev_ids,dev_mem,rows,rows['actions'],k=64)
        _m2_rerank.mem_tokens=mh;_m2_rerank.mem_weights=mw
        c,score=_m2_rerank(qh,qw,dev_ids,dev_mem,c,rows)
        err=_candidate_action_mse(rows,dev_ids,c)
        meta['dev_selection_top1']=float(err[:,0].mean());meta['dev_reranked']=True
        meta['_dev_candidate_rows']=c;meta['_dev_scores']=score
    else:
        qk=_encode_keys(model,dev_ids,rows,tokens,groups,context,smoke=smoke);mk=_encode_keys(model,dev_mem,rows,tokens,groups,context,smoke=smoke)
        c,score,err=_retrieve(qk,mk,dev_ids,dev_mem,rows,rows['actions'],k=64)
        meta['dev_selection_top1']=float(err[:,0].mean());meta['_dev_candidate_rows']=c;meta['_dev_scores']=score
    return model,meta

@torch.no_grad()
def _encode_tokens(model,ids,rows,tokens,groups,context,*,batch=4):
    model.eval();hs=[];sal=[]
    use_text=getattr(model,'token_count',512)>512
    for st in range(0,len(ids),batch):
        q=np.asarray(ids[st:st+batch],np.int64);tok,ctx,mask=_torch_batch(q,rows,tokens,groups,context,use_text=use_text)
        _,h,w=model.encode_tokens(tok,ctx,mask);hs.append(h.half().cpu().numpy());sal.append(w.half().cpu().numpy())
    return np.concatenate(hs),np.concatenate(sal)

@torch.no_grad()
def _m2_rerank(qtokens,qtokweights,dev_ids,mem_ids,cand_rows,rows,*,batch=4):
    # qtokens [Q,N,D], qweights [Q,N]; candidate set rows [Q,K]. MaxSim over valid image patches.
    qh=torch.as_tensor(qtokens,device=DEVICE,dtype=torch.float32);qw=torch.as_tensor(qtokweights,device=DEVICE,dtype=torch.float32)
    mem_lookup={int(r):i for i,r in enumerate(mem_ids)};mi=np.asarray([[mem_lookup[int(r)] for r in c] for c in cand_rows],np.int64)
    # Caller supplies memory token tensors in a function attribute to avoid copying a full 400MB array per call.
    mh=torch.as_tensor(_m2_rerank.mem_tokens,device=DEVICE,dtype=torch.float32)
    mw=torch.as_tensor(_m2_rerank.mem_weights,device=DEVICE,dtype=torch.float32)
    scores=np.empty(cand_rows.shape,np.float32)
    for st in range(0,len(dev_ids),batch):
        en=min(st+batch,len(dev_ids));q=qh[st:en];wq=qw[st:en];idx=torch.as_tensor(mi[st:en],device=DEVICE)
        m=mh[idx];wm=mw[idx]
        # [B,K,Nq,Nm]; 4 query rows at a time keeps peak scratch under 1 GB.
        sim=torch.einsum('bnd,bkmd->bknm',q,m)
        q2m=sim.max(dim=-1).values
        m2q=sim.max(dim=-2).values
        a=torch.sum(q2m*wq[:,None,:],dim=-1)
        b=torch.sum(m2q*wm,dim=-1)
        scores[st:en]=((a+b)*.5).float().cpu().numpy()
    # Stable rank by score then dataset row id.
    order=np.argsort(-scores,axis=1,kind='stable')
    ranked=np.take_along_axis(cand_rows,order,axis=1)
    ranked_scores=np.take_along_axis(scores,order,axis=1)
    return ranked,ranked_scores

def _m2_candidate_search(model,qids,memids,rows,tokens,groups,context,*,batch=8):
    qk=_encode_keys(model,qids,rows,tokens,groups,context,batch=batch);mk=_encode_keys(model,memids,rows,tokens,groups,context,batch=batch)
    qm,qw=_encode_tokens(model,qids,rows,tokens,groups,context,batch=4);mm,mw=_encode_tokens(model,memids,rows,tokens,groups,context,batch=4)
    c,_,_=_retrieve(qk,mk,qids,memids,rows,rows['actions'],k=64)
    _m2_rerank.mem_tokens=mm;_m2_rerank.mem_weights=mw
    cand,score=_m2_rerank(qm,qw,qids,memids,c,rows)
    return cand,score

def _train_m3(base_model,train_ids,train_mem,rows,tokens,groups,context,tau,*,epochs=3,smoke=False,progress=None):
    rng=np.random.default_rng(SEED+331)
    qsample=np.asarray(rng.choice(train_ids,min(len(train_ids),256 if smoke else 2048),replace=False),np.int64)
    qk=_encode_keys(base_model,qsample,rows,tokens,groups,context,batch=4)
    mk=_encode_keys(base_model,train_mem,rows,tokens,groups,context,batch=4)
    c,_,_=_retrieve(qk,mk,qsample,train_mem,rows,rows['actions'],k=64)
    # Sample rank-stratified candidates so the judge sees easy and hard examples.
    ranks=np.asarray([0,1,3,7,15,31,47,63],np.int64);cand=c[:,ranks]
    model=M3CandidateJudge(dim=96).to(DEVICE);opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-4)
    qact=np.asarray(rows['actions'][qsample,:,:7],np.float32)
    cact=np.asarray(rows['actions'][cand,:,:7],np.float32)
    target=np.log(np.mean((qact[:,None]-cact)**2,axis=(-1,-2))+1e-5).astype(np.float32)
    state=np.asarray(rows['state'][:,:8],np.float32)
    for ep in range(epochs):
        model.train();order=rng.permutation(len(qsample));losses=[]
        for st in range(0,len(order),4):
            ix=order[st:st+4];ids=qsample[ix]
            tok,ctx,mask=_torch_batch(ids,rows,tokens,groups,context)
            with torch.no_grad():_,qh,_=base_model.encode_tokens(tok,ctx,mask)
            crows=cand[ix];ca=torch.as_tensor(cact[ix],device=DEVICE);cs=torch.as_tensor(state[crows],device=DEVICE)
            target_t=torch.as_tensor(target[ix],device=DEVICE)
            pred=model(qh.detach(),mask,ca,cs)
            loss=F.smooth_l1_loss(pred,target_t)
            opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();losses.append(float(loss.detach().cpu()))
        if progress is not None:_log(progress,{'event':'m3_epoch','epoch':ep+1,'loss':float(np.mean(losses)),'candidate_pairs':int(len(qsample)*8)})
    model.eval()
    return model,{'epochs':epochs,'training_queries':len(qsample),'candidate_pairs':int(len(qsample)*8),'candidate_ranks':ranks.tolist()}

@torch.no_grad()
def _m3_rerank(model,base_model,qids,memids,cands,rows,tokens,groups,context,*,batch=4):
    state=np.asarray(rows['state'][:,:8],np.float32);cand_action=np.asarray(rows['actions'][cands,:,:7],np.float32)
    scores=np.empty(cands.shape,np.float32)
    model.eval();base_model.eval()
    for st in range(0,len(qids),batch):
        q=qids[st:st+batch];cr=cands[st:st+batch]
        tok,ctx,mask=_torch_batch(q,rows,tokens,groups,context,use_text=getattr(base_model,'token_count',512)>512)
        _,qh,_=base_model.encode_tokens(tok,ctx,mask)
        ca=torch.as_tensor(cand_action[st:st+batch],device=DEVICE);cs=torch.as_tensor(state[cr],device=DEVICE)
        scores[st:st+batch]=(-model(qh,mask,ca,cs)).float().cpu().numpy()
    order=np.argsort(-scores,axis=1,kind='stable')
    return np.take_along_axis(cands,order,1),np.take_along_axis(scores,order,1)

def _train_direct(base_model,train_ids,rows,tokens,groups,context,*,epochs=8,smoke=False):
    model=DirectActionHead(96).to(DEVICE);opt=torch.optim.AdamW(model.parameters(),lr=7e-4,weight_decay=1e-4)
    rng=np.random.default_rng(SEED+114);ids=np.asarray(train_ids,np.int64)
    for ep in range(epochs):
        rng.shuffle(ids);losses=[];model.train()
        for st in range(0,len(ids),32 if smoke else 64):
            q=ids[st:st+(32 if smoke else 64)];tok,ctx,mask=_torch_batch(q,rows,tokens,groups,context)
            with torch.no_grad():key=base_model(tok,ctx,mask)
            truth=torch.as_tensor(np.asarray(rows['actions'][q,:,:7],np.float32),device=DEVICE)
            pred=model(key.detach(),ctx);loss=F.mse_loss(pred,truth)
            opt.zero_grad(set_to_none=True);loss.backward();opt.step();losses.append(float(loss.detach().cpu()))
    model.eval();return model,{'epochs':epochs,'train_rows':len(train_ids),'train_mse_last':float(np.mean(losses))}

@torch.no_grad()
def _direct_predict(model,base_model,qids,rows,tokens,groups,context,*,batch=16):
    out=[];model.eval();base_model.eval()
    for st in range(0,len(qids),batch):
        q=np.asarray(qids[st:st+batch]);tok,ctx,mask=_torch_batch(q,rows,tokens,groups,context)
        key=base_model(tok,ctx,mask);out.append(model(key,ctx).float().cpu().numpy())
    return np.concatenate(out)

def _baseline_predictions(rows,qids,pool,topk_file):
    lookup={(int(t),int(e),int(f)):i for i,(t,e,f) in enumerate(zip(rows['task'],rows['episode'],rows['frame']))}
    truth=np.asarray(rows['actions'][qids,:,:7],np.float32);hold=np.zeros_like(truth);prev=np.zeros_like(truth)
    for j,r in enumerate(qids):
        pr=lookup.get((int(rows['task'][r]),int(rows['episode'][r]),int(rows['frame'][r])-10))
        if pr is not None:
            act=np.asarray(rows['actions'][pr,:,:7],np.float32);hold[j]=act[-1][None].repeat(10,axis=0);prev[j]=act
    allowed=[pool[np.asarray(rows['task'][pool])!=int(rows['task'][r])] for r in qids]
    rng=np.random.default_rng(SEED+718);rand=np.asarray([np.asarray(rows['actions'][int(rng.choice(a)),:,:7],np.float32) for a in allowed])
    dm=_action_mse_rows(rows,qids,pool);dm[np.asarray(rows['task'][qids])[:,None]==np.asarray(rows['task'][pool])[None]]=np.inf
    order=np.argsort(dm,axis=1,kind='stable')[:,:64];oracle=np.asarray(pool)[order]
    return {'hold_last':hold,'previous_chunk_copy':prev,'random_one':rand,'oracle_rows':oracle,
            'oracle_mse':np.take_along_axis(dm,order,axis=1)}

def _load_local_baselines(qids):
    out={}
    for name,file in [('G','ranks_G.npz'),('wrist_lowerhalf+prev','ranks_wrist_lowerhalf_prev.npz'),('invariant_projected_local','ranks_invariant_projected_local.npz')]:
        z=np.load(LOCAL/file,allow_pickle=False);cand=np.asarray(z['cand_rows'],np.int64)
        if cand.ndim==1:cand=cand[:,None]
        # Map query IDs explicitly; never assume array index equals data row ID.
        q=np.asarray(np.load(LOCAL/'predictions.npz',allow_pickle=False)['query_rows'],np.int64)
        lookup={int(r):i for i,r in enumerate(q)};ix=np.asarray([lookup[int(r)] for r in qids],np.int64)
        cand=cand[ix]
        out[name]=cand
    return out

def _baseline_generated(rows,qids,topfile):
    z=np.load(topfile,allow_pickle=False)
    qmap={int(q):i for i,q in enumerate(z['query_rows'])};ix=np.asarray([qmap[int(q)] for q in qids],np.int64)
    zero=np.asarray(z['zero_predictions'][1,0,ix],np.float32) # t0=.4, first noise seed
    normal=np.asarray(z['normal_single'][ix],np.float32)
    return {'zero_draft':zero,'normal_pi05':normal}

def _episode_ids(rows,qids):return (np.asarray(rows['task'][qids],np.int64)*100000+np.asarray(rows['episode'][qids],np.int64)).astype(np.int64)

def _visualize_attention(out,rows,selected_m1,ids,hard,context,tokens,groups):
    from PIL import Image,ImageDraw,ImageFilter
    # Select one episode with many hard rows and one with many easy rows, then sample thirds.
    task=np.asarray(rows['task'][ids]);ep=np.asarray(rows['episode'][ids]);
    def choose(mask):
        subset=ids[mask]
        if not len(subset):subset=ids
        keys=[(int(rows['task'][r]),int(rows['episode'][r])) for r in subset]
        best=max(set(keys),key=keys.count);allrows=np.flatnonzero((np.asarray(rows['task'])==best[0])&(np.asarray(rows['episode'])==best[1]))
        allrows=allrows[np.argsort(np.asarray(rows['frame'][allrows]))]
        ix=np.unique(np.linspace(0,len(allrows)-1,3).round().astype(int));return allrows[ix]
    examples={'hard':choose(hard),'easy':choose(~hard)}
    stats={}
    for tag,ex in examples.items():
        _,_,_,att=_encode_keys(selected_m1,ex,rows,tokens,groups,context,batch=3,need_tokens=True,need_attention=True)
        panels=[]
        for j,r in enumerate(ex):
            weights=att[j]
            if weights.ndim>1:weights=weights.mean(0)
            for cam,(imgkey,offset) in enumerate((('image_base',0),('image_wrist',256))):
                im=Image.fromarray(np.asarray(rows[imgkey][r],np.uint8)).convert('RGB').resize((224,224))
                grid=np.zeros((16,16),np.float32);grid[:]=weights[offset:offset+256].reshape(16,16)
                if grid.max()>grid.min():grid=(grid-grid.min())/(grid.max()-grid.min())
                heat=Image.fromarray(np.uint8(grid*255)).resize((224,224),Image.Resampling.BILINEAR).filter(ImageFilter.GaussianBlur(7))
                color=Image.new('RGB',(224,224),(255,35,10));color.putalpha(heat.point(lambda x:int(x*.60)))
                im=Image.alpha_composite(im.convert('RGBA'),color).convert('RGB')
                d=ImageDraw.Draw(im);d.rectangle((0,0,223,18),fill=(0,0,0));d.text((4,3),f"{tag} {['early','middle','late'][j]} task {int(rows['task'][r])} frame {int(rows['frame'][r])} {'base' if cam==0 else 'wrist'}",fill='white')
                panels.append(im)
        canvas=Image.new('RGB',(448,672),'white')
        for i,im in enumerate(panels):canvas.paste(im,((i%2)*224,(i//2)*224))
        canvas.save(out/f'attention_{tag}.png',optimize=True)
        stats[tag]={'rows':ex.astype(int).tolist(),'frames':np.asarray(rows['frame'][ex]).astype(int).tolist()}
    return stats

def _prediction_only(pred,truth,name):
    pred=np.asarray(pred,np.float32);truth=np.asarray(truth,np.float32);mse=np.mean((pred-truth)**2,axis=(-1,-2))
    td=truth[:,:,:3].sum(1);pd=pred[:,:,:3].sum(1);den=np.linalg.norm(td,axis=1)*np.linalg.norm(pd,axis=1)
    dc=np.divide(np.sum(td*pd,axis=1),den,out=np.zeros(len(td),np.float32),where=den>1e-8)
    return {'name':name,'top1_mse':mse,'top5_best_mse':None,'recall8':None,'recall_valid':None,'spearman_within':None,
            'direction_cosine':dc,'gripper_sign_agreement':float(np.mean(np.sign(pred[:,:,6])==np.sign(truth[:,:,6]))),'prediction':pred}

def _load_row_rank(name,qids):
    z=np.load(LOCAL/f'ranks_{name.replace("+","_")}.npz',allow_pickle=False);cand=np.asarray(z['cand_rows'],np.int64)
    if cand.ndim==1:cand=cand[:,None]
    if len(cand)!=len(qids):raise AssertionError(f'local-key candidate rows mismatch for {name}')
    return cand

def _generate_after(out,rows,qids,hard,truth,methods,selected_name,top2_names,direct_pred,context,*,batch=4):
    import os
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false');os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION','.30')
    from pathlib import Path
    topk_path=_PI05_CODE / 'topk_long/run.py'
    import importlib.util
    spec=importlib.util.spec_from_file_location('key_mechanism_topk_runtime',topk_path);topk=importlib.util.module_from_spec(spec);spec.loader.exec_module(topk)
    jax,jnp,policy,base,baseline,bridge=topk._load_runtime()
    # Top two configs are preselected on key-Dev, before opening Long labels.
    selected_order=list(top2_names)
    names=list(selected_order)+['wrist_lowerhalf+prev','direct_regression','zero_draft']
    if len(set(names))!=len(names):raise AssertionError('duplicate generation draft names')
    n=len(qids);c=len(names);draft=np.zeros((n,c,10,32),np.float32)
    for i,name in enumerate(selected_order):
        cr=np.asarray(methods[name]['candidate_rows'],np.int64)[:,0]
        draft[:,i]=np.asarray(rows['actions'][cr],np.float32)
    localrows=_load_row_rank('wrist_lowerhalf+prev',qids)[:,0]
    draft[:,2]=np.asarray(rows['actions'][localrows],np.float32)
    draft[:,3,:,:7]=np.asarray(direct_pred,np.float32)
    # zero_draft remains zero in all 32 channels.
    pred=np.empty((2,n,c,10,7),np.float32);normal=np.empty((n,10,7),np.float32);t0s=(.4,.6);started=time.time()
    for st in range(0,n,batch):
        en=min(st+batch,n);ids=qids[st:en];obs=policy.obs_batch(rows,ids)
        eps=jnp.asarray(np.stack([topk._noise(jax,4101,int(r)) for r in ids]))
        normal[st:en]=np.asarray(baseline(jax.random.key(0),obs,num_steps=10,noise=eps),np.float32)[:,:,:7]
        for ti,t0 in enumerate(t0s):
            pred[ti,st:en]=topk._sample_bridge(bridge,jax,jnp,policy,rows,ids,draft[st:en],t0,4101,st,
                                               batch_size=batch,condition_batch=8)
        if en==n or (st//batch)%25==0:_log(out/'progress.jsonl',{'event':'generate','rows_done':en,'rows_total':n,'elapsed_s':time.time()-started})
    mse=np.mean((pred-truth[None,:,None])**2,axis=(-1,-2));normal_mse=np.mean((normal-truth)**2,axis=(-1,-2))
    result={'t0':list(t0s),'draft_names':names,'elapsed_seconds':time.time()-started,'gpu_memory_fraction':.30,'metrics':{}}
    episodes=_episode_ids(rows,qids)
    for ti,t0 in enumerate(t0s):
        result['metrics'][str(t0)]={}
        for ci,name in enumerate(names):
            err=mse[ti,:,ci]
            result['metrics'][str(t0)][name]={'all_mean':float(err.mean()),'hard_mean':float(err[hard].mean()),
                'all_vs_zero':_bootstrap_diff(err,mse[ti,:,names.index('zero_draft')],episodes),
                'hard_vs_zero':_bootstrap_diff(err[hard],mse[ti,hard,names.index('zero_draft')],episodes[hard])}
        result['metrics'][str(t0)]['normal_pi05']={'all_mean':float(normal_mse.mean()),'hard_mean':float(normal_mse[hard].mean())}
    np.savez_compressed(out/'generation_predictions.npz',query_rows=qids,hard_mask=hard,method_names=np.asarray(names),t0=np.asarray(t0s),
                        drafts=draft,generated=pred,generated_mse=mse,normal=normal,normal_mse=normal_mse,truth=truth)
    (out/'generation_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=True)+'\n')
    return result,selected_order

def _run(out,*,threads=16,smoke=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(threads);torch.manual_seed(SEED)
    rows=_load_rows();splits=make_splits(rows);hold=splits['holdout_tasks']
    # Verify the canonical memory row set before any modeling.
    final_mem=splits['memory'];ref=np.load(MEMORY/'units.npz',allow_pickle=False)['representative_row_ids']
    if len(final_mem)!=3857 or not np.array_equal(final_mem,ref):raise AssertionError('fixed raw-rows-noLong memory set changed')
    for k in ('train','dev','dev_mem','memory'):assert_no_long(splits[k],rows,k)
    if not np.all(np.asarray(rows['task'][splits['long']])<10):raise AssertionError('Long split contains a non-Long task')
    if np.intersect1d(np.asarray(rows['task'][splits['train']]),hold).size:raise AssertionError('key-dev task leaked into train')
    if np.intersect1d(np.asarray(rows['task'][splits['dev_mem']]),hold).size:raise AssertionError('key-dev task leaked into key-dev memory')
    if set(np.unique(rows['task'][splits['dev']]).tolist())!=set(hold.tolist()):raise AssertionError('key-dev task split mismatch')
    train_ids=splits['train'];dev_ids=splits['dev'];dev_mem=splits['dev_mem'];pool=final_mem
    gap=np.load(GAP/'predictions_and_errors.npz',allow_pickle=False);top=np.load(TOPK/'predictions_and_errors.npz',allow_pickle=False)
    gap_long=np.asarray(gap['row_ids'],np.int64)[np.asarray(rows['task'][gap['row_ids']])<10]
    if len(gap_long)!=548 or not np.all(np.asarray(rows['validation'][gap_long],bool)):
        raise AssertionError('Long Eval row set mismatch')
    long_ids=gap_long
    if not np.array_equal(long_ids,top['query_rows']):raise AssertionError('topk-long query row order mismatch')
    if len(long_ids)!=548 or int(top['hard_mask'].sum())!=63 or not np.isclose(_gap_threshold(GAP),Q80,atol=1e-10):
        raise AssertionError('fixed Long q80 split changed')
    hard=np.asarray(top['hard_mask'],bool)
    context=_history(rows)
    tokens=np.load(RUN/'tokens.npy',mmap_mode='r',allow_pickle=False);groups=np.load(RUN/'groups.npy',mmap_mode='r',allow_pickle=False)
    if tokens.shape!=(26581,712,2048) or tokens.dtype!=np.float16:raise AssertionError('unexpected token memmap')
    if not (np.all(groups[0,:256]==0) and np.all(groups[0,256:512]==1)):raise AssertionError('image patch ordering changed')
    tau,tau_rates=_select_tau(dev_ids,rows)
    if smoke:
        rng=np.random.default_rng(SEED+555);train_ids=rng.choice(train_ids,min(1024,len(train_ids)),replace=False)
        dev_ids=rng.choice(dev_ids,min(64,len(dev_ids)),replace=False);dev_mem=rng.choice(dev_mem,min(384,len(dev_mem)),replace=False)
        pool=np.asarray(pool);pool=pool[:min(512,len(pool))]
        long_ids=long_ids[:16];hard=hard[:16]
        tau=.02
    logpath=out/'progress.jsonl';configs={'M1_slots4':('m1',4),'M1_slots8':('m1',8)}
    if not smoke:configs['M1_text_slots4']=('m1_text',4)
    metas={};models={};dev_models={};started=time.time()
    for name,(kind,slots) in configs.items():
        torch.manual_seed(SEED+sum(ord(c) for c in name))
        epochs=1 if smoke else 4
        model,meta=_train_encoder(name,kind,train_ids,dev_ids,dev_mem,rows,tokens,groups,context,tau,epochs=epochs,smoke=smoke,progress=logpath)
        models[name]=model;metas[name]=meta;dev_models[name]=float(meta['dev_selection_top1'])
        _log(logpath,{'event':'model_done','name':name,'dev_top1':dev_models[name],'elapsed_s':time.time()-started})
    torch.manual_seed(SEED+242)
    m2,mm2=_train_encoder('M2_salient_maxsim','m2',train_ids,dev_ids,dev_mem,rows,tokens,groups,context,tau,
                           epochs=1 if smoke else 4,smoke=smoke,progress=logpath)
    models['M2_salient_maxsim']=m2;metas['M2_salient_maxsim']=mm2;dev_models['M2_salient_maxsim']=float(mm2['dev_selection_top1'])
    best_m1_name=min(('M1_slots4','M1_slots8'),key=lambda n:dev_models[n]);base_m1=models[best_m1_name]
    train_mem=pool[~np.isin(np.asarray(rows['task'][pool]),hold)]
    m3,m3meta=_train_m3(base_m1,train_ids,train_mem,rows,tokens,groups,context,tau,epochs=1 if smoke else 3,smoke=smoke,progress=logpath)
    # M3 model selection uses the held-out key-Dev task set only.
    dev_m1key=_encode_keys(base_m1,dev_ids,rows,tokens,groups,context,batch=4 if smoke else 8)
    dev_memkey=_encode_keys(base_m1,dev_mem,rows,tokens,groups,context,batch=4 if smoke else 8)
    dev_c,dev_sc,_=_retrieve(dev_m1key,dev_memkey,dev_ids,dev_mem,rows,rows['actions'],k=64)
    dev_m3c,dev_m3s=_m3_rerank(m3,base_m1,dev_ids,dev_mem,dev_c,rows,tokens,groups,context,batch=4)
    dev_m3err=_candidate_action_mse(rows,dev_ids,dev_m3c)
    m3meta['dev_selection_top1']=float(dev_m3err[:,0].mean());m3meta['_dev_candidate_rows']=dev_m3c;m3meta['_dev_scores']=dev_m3s
    models['M3_candidate_judge']=m3;metas['M3_candidate_judge']=m3meta;dev_models['M3_candidate_judge']=m3meta['dev_selection_top1']
    # One matched direct action predictor, trained on exactly the same 24-task rows and selected M1 representation.
    direct,dmeta=_train_direct(base_m1,train_ids,rows,tokens,groups,context,epochs=1 if smoke else 8,smoke=smoke)

    # Dev model order is fixed now; no refit after selection.
    eligible=['M1_slots4','M1_slots8','M2_salient_maxsim','M3_candidate_judge']
    dev_order=sorted(eligible,key=lambda n:dev_models[n]);selected_name=dev_order[0]
    if smoke:
        # smoke only verifies the pipeline; it intentionally does not create a report decision.
        pass

    # Final Long retrieval, using the full 3,857 noLong memory and same-task exclusion.
    method_results={};dev_predictions={};all_memory_error=None
    qids=long_ids;truth=np.asarray(rows['actions'][qids,:,:7],np.float32)
    for name in configs:
        model=models[name];qk=_encode_keys(model,qids,rows,tokens,groups,context,batch=4 if smoke else 8)
        mk=_encode_keys(model,pool,rows,tokens,groups,context,batch=4 if smoke else 8)
        c,s,_=_retrieve(qk,mk,qids,pool,rows,rows['actions'],k=64)
        method_results[name]=_best_metrics(name,qids,c,s,rows,None,conf=s[:,0])
    # M2 learned salience + token MaxSim reranking.
    m2c,m2s=_m2_candidate_search(m2,qids,pool,rows,tokens,groups,context,batch=4 if smoke else 8)
    method_results['M2_salient_maxsim']=_best_metrics('M2_salient_maxsim',qids,m2c,m2s,rows,None,conf=m2s[:,0])
    # M3 uses selected M1 coarse Top-64 then predicts candidate error; it returns one memory row.
    qk=_encode_keys(base_m1,qids,rows,tokens,groups,context,batch=4 if smoke else 8)
    mk=_encode_keys(base_m1,pool,rows,tokens,groups,context,batch=4 if smoke else 8)
    m1c,m1s,_=_retrieve(qk,mk,qids,pool,rows,rows['actions'],k=64)
    m3c,m3s=_m3_rerank(m3,base_m1,qids,pool,m1c,rows,tokens,groups,context,batch=4)
    method_results['M3_candidate_judge']=_best_metrics('M3_candidate_judge',qids,m3c,m3s,rows,None,conf=m3s[:,0])
    # Retain the text-enabled M1 only as an instruction-token reference, never as primary selection.
    if 'M1_text_slots4' in models:
        mod=models['M1_text_slots4'];qk=_encode_keys(mod,qids,rows,tokens,groups,context,batch=8);mk=_encode_keys(mod,pool,rows,tokens,groups,context,batch=8)
        c,s,_=_retrieve(qk,mk,qids,pool,rows,rows['actions'],k=64);method_results['M1_text_reference']=_best_metrics('M1_text_reference',qids,c,s,rows,None,conf=s[:,0])
    # Same rows for every action-distance denominator; never score same-task candidates.
    all_memory_error=_action_mse_rows(rows,qids,pool)
    all_memory_error[np.asarray(rows['task'][qids])[:,None]==np.asarray(rows['task'][pool])[None]]=np.inf
    for m in method_results.values():
        rel=all_memory_error<=.01;den=rel.sum(1);m['recall8']=np.divide((m['candidate_mse'][:,:8]<=.01).sum(1),den,out=np.zeros(len(qids),np.float32),where=den>0)
        m['recall_valid']=den>0
    # Local fixed keys and action-oracle/other retrieval baselines.
    for name,c in _load_local_baselines(qids).items():
        sc=np.tile(np.arange(c.shape[1],0,-1,dtype=np.float32),(len(qids),1))
        method_results[name]=_best_metrics(name,qids,c,sc,rows,all_memory_error,conf=sc[:,0])
    base=_baseline_predictions(rows,qids,pool,TOPK/'predictions_and_errors.npz')
    oracle_c=base['oracle_rows'];oracle_scores=-base['oracle_mse']
    method_results['action_oracle']=_best_metrics('action_oracle',qids,oracle_c,oracle_scores,rows,all_memory_error,conf=oracle_scores[:,0])
    random_rows=base['random_one'];# second-random candidates are provided by deterministic allowed-pool sampling for Top-5 diagnostics
    random_c=np.empty((len(qids),8),np.int64);rng=np.random.default_rng(SEED+919)
    for i,r in enumerate(qids):
        allowed=pool[np.asarray(rows['task'][pool])!=int(rows['task'][r])];random_c[i]=rng.choice(allowed,8,replace=False)
    method_results['random_one']=_best_metrics('random_one',qids,random_c,np.tile(np.arange(8,0,-1,dtype=np.float32),(len(qids),1)),rows,all_memory_error,conf=np.zeros(len(qids),np.float32))
    # Baselines that directly propose actions rather than a ranked memory row.
    direct_pred=_direct_predict(direct,base_m1,qids,rows,tokens,groups,context,batch=8)
    method_results['direct_regression']=_prediction_only(direct_pred,truth,'direct_regression')
    method_results['hold_last']=_prediction_only(base['hold_last'],truth,'hold_last')
    method_results['previous_chunk_copy']=_prediction_only(base['previous_chunk_copy'],truth,'previous_chunk_copy')
    genbase=_baseline_generated(rows,qids,TOPK/'predictions_and_errors.npz')
    for n,p in genbase.items():method_results[n]=_prediction_only(p,truth,n)

    # Key-Dev-selected top two model outputs, before reading Long errors.
    dev_order=sorted(eligible,key=lambda n:dev_models[n]);selected_name=dev_order[0]
    selected_conf=method_results[selected_name]['confidence']
    risk=_risk_metrics(selected_conf,method_results[selected_name]['top1_mse'],_episode_ids(rows,qids))
    episode=_episode_ids(rows,qids)
    comparisons={}
    for name in ('wrist_lowerhalf+prev','hold_last','zero_draft'):
        delta=method_results[selected_name]['top1_mse']-method_results[name]['top1_mse']
        comparisons[f'{selected_name}_minus_{name}_all']=_bootstrap_diff(method_results[selected_name]['top1_mse'],method_results[name]['top1_mse'],episode)
        comparisons[f'{selected_name}_minus_{name}_hard']=_bootstrap_diff(method_results[selected_name]['top1_mse'][hard],method_results[name]['top1_mse'][hard],episode[hard])
    selected_cis={}
    for mname in dev_order:
        for subset,mask in [('all',np.ones(len(qids),bool)),('hard',hard)]:
            selected_cis[f'{mname}_{subset}']=_bootstrap_diff(method_results[mname]['top1_mse'][mask],
                method_results['wrist_lowerhalf+prev']['top1_mse'][mask],episode[mask])
    summary_metrics=_metrics_summary(rows,qids,hard,all_memory_error,method_results,episode)

    # Write run protocol before GPU generation to leave a useful checkpoint if interrupted.
    protocol={'seed':SEED,'holdout_tasks':hold.tolist(),'holdout_tasks_by_suite':{s:[int(t) for t in hold if a<=t<b] for a,b,s in SUITES},
      'training_rows':len(train_ids),'training_tasks':np.unique(rows['task'][train_ids]).astype(int).tolist(),
      'key_dev_rows':len(dev_ids),'key_dev_tasks':np.unique(rows['task'][dev_ids]).astype(int).tolist(),
      'key_dev_memory_rows':len(dev_mem),'key_dev_memory_tasks':np.unique(rows['task'][dev_mem]).astype(int).tolist(),
      'final_memory_rows':len(pool),'final_memory_matches_raw_rows_noLong':True,'long_eval_rows':len(qids),'hard_rows':int(hard.sum()),
      'hard_threshold':Q80,'tau_selected_on_key_dev':tau,'tau_cross_task_positive_rates':tau_rates,
      'same_task_exclusion':'all retrieval and key-dev candidate searches exclude candidates with the query task',
      'long_used_for_fit_or_selection':False,'long_policy_pretraining_seen':True,
      'main_input':'base+wrist image tokens, position embeddings, normalized state[:8], prior executed action[10x7] (language tokens excluded)',
      'language_reference':'M1_text_slots4 uses valid instruction token positions and is excluded from primary model selection',
      'M1':{'variants':['M1_slots4','M1_slots8'],'soft_infoNCE_tau':tau,'epochs':1 if smoke else 4,'positive_pair_count':int(len(_positive_pairs(train_ids,rows,tau,seed=SEED+sum(map(ord,'M1_slots4')),max_anchors=512 if smoke else 4096)))},
      'M2':{'pooling':'learned salience-weighted token pooling for coarse top64, symmetric salience-weighted token MaxSim for rerank','epochs':1 if smoke else 4},
      'M3':{'coarse_candidates':'top64 M1','input':'query image token embeddings + candidate raw action[10x7] + candidate initial state[:8]','prediction':'log candidate-vs-query action MSE, no query target action input','epochs':1 if smoke else 3,
            **{k:v for k,v in m3meta.items() if not k.startswith('_')}},
      'direct_regression':'DirectActionHead over best key-dev M1 embedding plus the exact same state/history context; same train rows',
      'bootstrap_replicates':5000,'device':DEVICE,'cpu_threads':threads,'gpu_fraction':'<=.30','smoke':smoke}
    (out/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2,allow_nan=True)+'\n')

    if not smoke:
        gen,top2=_generate_after(out,rows,qids,hard,truth,method_results,selected_name,dev_order[:2],direct_pred,context)
        # K1: local-key comparison; K2: both hold-last subsets; K3: selected Dev winner vs zero; K4: confidence-quality correlation.
        k1=any(comparisons[f'{selected_name}_minus_wrist_lowerhalf+prev_{s}']['ci95'][1]<0 for s in ()) if False else (
            comparisons[f'{selected_name}_minus_wrist_lowerhalf+prev_all']['ci95'][1]<0 or
            comparisons[f'{selected_name}_minus_wrist_lowerhalf+prev_hard']['ci95'][1]<0)
        k2=(comparisons[f'{selected_name}_minus_hold_last_all']['ci95'][1]<0 and comparisons[f'{selected_name}_minus_hold_last_hard']['ci95'][1]<0)
        gmet=gen['metrics'];k3=any(gmet[str(t)][selected_name]['all_vs_zero']['ci95'][1]<0 or gmet[str(t)][selected_name]['hard_vs_zero']['ci95'][1]<0 for t in (.4,.6))
        k4=(risk['spearman_confidence_error_quality']>=.3 and risk['top20_minus_all_ci95'][1]<0)
        decision={'selected_model':selected_name,'key_dev_model_order':dev_order,'K1':bool(k1),'K2':bool(k2),'K3':bool(k3),'K4':bool(k4),
                  'K4_rule':'Spearman(confidence,-Top1MSE)>=0.3 and top-confidence-20% minus all error 95% CI upper<0; signed confidence-vs-error also reported'}
        (out/'decision.json').write_text(json.dumps(decision,ensure_ascii=False,indent=2)+'\n')
        protocol['generation']={'selected_dev_top2':top2,'t0':[.4,.6],'noise_seed':4101,'frozen_expert':str(RUN/'bridge/001500'),
                               'normal_pi05_included':True,'elapsed_seconds':gen['elapsed_seconds']}
        (out/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2,allow_nan=True)+'\n')
    else:gen=None;decision=None

    # Strip per-row arrays from the compact JSON; keep them in candidate_rows/predictions NPZ.
    compact={k:{kk:(float(v) if isinstance(v,(np.floating,np.integer)) else v) for kk,v in row.items() if kk not in
            ('candidate_rows','scores','candidate_mse','top1_mse','top5_best_mse','recall8','recall_valid','spearman_within','direction_cosine','confidence','prediction')}
             for k,row in method_results.items()}
    result={'protocol':protocol,'key_dev':{'dev_top1_mse':dev_models,'selected_order':dev_order,'m1_metadata':{n:{k:v for k,v in metas[n].items() if not k.startswith('_')} for n in metas},'m3_metadata':{k:v for k,v in m3meta.items() if not k.startswith('_')},'direct_fit':dmeta},
            'metrics':summary_metrics,'confidence':risk,'comparisons':comparisons,'selected_model_bootstrap_vs_local':selected_cis,
            'generation':gen,'decision':decision}
    (out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=True)+'\n')
    np.savez_compressed(out/'candidate_rows.npz',query_rows=qids,hard_mask=hard,
        **{f'{n}_rows':np.asarray(m['candidate_rows'],np.int64) for n,m in method_results.items() if m.get('candidate_rows') is not None},
        **{f'{n}_scores':np.asarray(m['scores'],np.float32) for n,m in method_results.items() if m.get('scores') is not None})
    np.savez_compressed(out/'predictions.npz',query_rows=qids,hard_mask=hard,truth=truth,
        **{f'{n}_top1_mse':np.asarray(m['top1_mse'],np.float32) for n,m in method_results.items()},
        **{f'{n}_confidence':np.asarray(m['confidence'],np.float32) for n,m in method_results.items() if m.get('confidence') is not None})
    torch.save({'models':{n:_cpu_state_dict(m) for n,m in models.items()},'m3':_cpu_state_dict(m3),'direct':_cpu_state_dict(direct),
                'selected_m1_for_m3':best_m1_name,'key_dev_order':dev_order,'selected_model':selected_name},out/'models.pt')
    if not smoke:
        attention_model=models[selected_name] if hasattr(models[selected_name],'encode_tokens') else base_m1
        viz=_visualize_attention(out,rows,attention_model,qids,hard,context,tokens,groups)
        result['attention_examples']=viz;(out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=True)+'\n')
    return result

def run(out=OUT,*,threads=16,smoke=False):
    out=Path(out)
    target=out/'smoke' if smoke else out/'production'
    if target.exists() and any(target.iterdir()):raise FileExistsError(f'refusing to overwrite output directory {target}')
    target.mkdir(parents=True,exist_ok=True)
    return _run(target,threads=threads,smoke=smoke)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=OUT);ap.add_argument('--threads',type=int,default=16);ap.add_argument('--smoke',action='store_true')
    a=ap.parse_args()
    if not 1<=a.threads<=16:raise ValueError('threads must be 1..16')
    if DEVICE=='cuda':torch.cuda.set_per_process_memory_fraction(.30)
    result=run(a.output,threads=a.threads,smoke=a.smoke)
    print(json.dumps({'output':str(a.output),'selected_model':result['decision']['selected_model'] if result['decision'] else 'smoke'},ensure_ascii=False))

if __name__=='__main__':main()
