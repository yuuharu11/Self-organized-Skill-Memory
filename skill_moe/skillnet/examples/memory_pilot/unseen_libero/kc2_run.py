#!/usr/bin/env python3
"""KC2 joint retrieval/gripper training on CPU using immutable KC1 assets."""
from __future__ import annotations
import argparse, json, math, os, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import kc1_run as K1

STATE_DIR = Path(os.environ.get('SKILLNET_STATE_DIR', Path(__file__).resolve().parents[5] / '.skillnet-state')).expanduser()
ROOT=STATE_DIR / 'outputs/memory-pilot/unseen-libero'
OUT=ROOT/'kc2-01'
K1_OUT=ROOT/'kc1-01'
BANK=K1.BANK_DIR
TAUS=(.03,.1,.3)
LAMBDAS=(0.,.1,.3,1.)
SEEDS=(1,2,3)
SPLITS=('eh','dev2','learned','unseen20')
EPOCHS=16
RAW_DEV_KC1=float(json.loads((K1_OUT/'offline_corrected.json').read_text())['selected']['dev_raw_coeff_sq'])
DEV_LIMIT=RAW_DEV_KC1*1.05


def log(event,**fields):
    OUT.mkdir(parents=True,exist_ok=True)
    row={'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'event':event,**fields}
    with (OUT/'progress.log').open('a') as f:f.write(json.dumps(row,ensure_ascii=False,default=float)+'\n')
    print(json.dumps(row,ensure_ascii=False,default=float),flush=True)


def paired_counts(kc,ref):
    """Return correctly oriented win counts: KC2-only, reference-only."""
    common=sorted(set(kc)&set(ref))
    left=np.asarray([bool(kc[x]) for x in common]);right=np.asarray([bool(ref[x]) for x in common])
    kw=int((left & ~right).sum());rw=int((right & ~left).sum());n=kw+rw
    p=1. if n==0 else min(1.,2.*sum(math.comb(n,i) for i in range(min(kw,rw)+1))/2**n)
    return {'pairs':len(common),'kc2_only':kw,'reference_only':rw,'mcnemar_p_exact':p,
            'success_difference_check':int(left.sum())-int(right.sum())==kw-rw}


def within_retrieval_constraint(dev_coeff_sq:float)->bool:
    return float(dev_coeff_sq)<=DEV_LIMIT


def select_joint_candidate(rows):
    eligible=[r for r in rows if float(r['lambda'])>0 and within_retrieval_constraint(r['dev_raw_coeff_sq'])]
    if not eligible:raise RuntimeError('no joint model within KC1 Dev coefficient error +5%')
    return min(eligible,key=lambda r:(r['dev_gripper_errors'],r['dev_raw_coeff_sq'],
        -r['dev_gripper_exact_accuracy'],r['lambda'],r['name']))


class KC2Net(nn.Module):
    """256-D query with learned coefficient map and q-only gripper head."""
    def __init__(self,q_name:str,score_mode:str):
        super().__init__()
        if score_mode not in ('dot','distance'):raise ValueError(score_mode)
        from kq2_run import QueryNet
        use_vlm=q_name=='iii'
        self.query=QueryNet('gru',1,1 if use_vlm else 0,use_vlm,history='w10',dct=False)
        self.project=nn.Linear(256,18)
        self.grip=nn.Sequential(nn.Linear(256,256),nn.ReLU(),nn.Linear(256,10))
        self.score_mode=score_mode
        self.q_name=q_name

    @property
    def logit_scale(self):return self.query.logit_scale

    def forward(self,gs,ix):return self.query(gs,ix)

    def grip_logits(self,q):return self.grip(q)

    def score(self,q,keys):
        v=self.project(q).float();k=keys.float();scale=self.query.logit_scale.exp().clamp(max=100.)
        if self.score_mode=='dot':return (v@k.T)*scale
        return -torch.cdist(v,k).square()*scale


def seed_init(seed,q_name,score_mode,tau):
    qi=0 if q_name=='i' else 1;si=0 if score_mode=='dot' else 1;ti=TAUS.index(tau)
    return int(seed*100000+qi*1000+si*100+ti*10)


def load_ctx():
    K1.OUT=OUT
    return K1._prepare_context()


def _gripper_labels(sp):
    acts=np.asarray(sp['actions'],np.float32)
    return (acts[...,6]>0).astype(np.float32)


def _hist_arrays(sp):
    d=OUT/'data'/('eh' if len(sp['key'])==12766 else 'dev2')
    history=np.load(d/'history.npy');mask=np.load(d/'history_mask.npy')
    last=history[:,9,6];valid=mask.any(1)&(last!=0)
    return last,valid


def _grip_metrics(probs,target,last,valid,th):
    import kq2cl_common as C
    decoded=C.decode_gripper(probs,np.where(valid,np.sign(last),-1.),th)
    targ=np.where(np.asarray(target)>0,1.,-1.)
    m=C.switch_metrics(decoded,targ,last,valid)
    return m|{'errors':int(m['missed_rows']+m['false_switch_rows']),
              'exact_step_accuracy':float((decoded==targ).mean())}


def evaluate_dev(model,gs,sp,target_raw,raw_keys,key_space,key_task,key_episode,labels,last,valid):
    model.eval();device=next(model.parameters()).device;scores=[];probs=[]
    with torch.no_grad():
        for start in range(0,gs.n,128):
            ix=torch.arange(start,min(start+128,gs.n),device=device);q=model(gs,ix)
            scores.append(model.score(q,key_space).float().cpu().numpy())
            probs.append(torch.sigmoid(model.grip_logits(q)).float().cpu().numpy())
    scores=np.concatenate(scores);probs=np.concatenate(probs)
    allow=K1.candidate_mask(sp['task'],sp['episode'],key_task,key_episode);scores[~allow]=-np.inf
    ids=np.argmax(scores,axis=1)
    err=np.square(np.asarray(target_raw,np.float32)-np.asarray(raw_keys,np.float32)[ids]).sum(-1)
    thresholds={str(t):_grip_metrics(probs,labels,last,valid,t) for t in (.3,.4,.5,.6,.7,.8,.9)}
    th=min(thresholds,key=lambda x:(thresholds[x]['errors'],-float(x)))
    return {'dev_raw_coeff_sq':float(err.mean()),'joint_head_threshold':float(th),
            'joint_head_metrics':thresholds[th],'dev_probs':probs}


def _config_name(q,score,tau,lam,seed):
    return f'q{q}_{score}_tau{tau:g}_lam{lam:g}_s{seed}'


def train_one(kr,ctx,targets,bank,*,q_name,score_mode,tau,lam,seed):
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','8')))
    units=np.asarray(bank['coefficients'],np.float32).reshape(len(bank['length']),18)
    kc1meta=json.loads((K1_OUT/'offline_corrected.json').read_text())['selected']
    kc1cfg=torch.load(kc1meta['checkpoint'],map_location='cpu',weights_only=False)['cfg']
    keys_np,mean,std=K1.coefficient_keys(units,standardize=True,mean=np.asarray(kc1cfg['mean']),std=np.asarray(kc1cfg['std']))
    key_space=torch.as_tensor(keys_np,dtype=torch.float32)
    name=_config_name(q_name,score_mode,tau,lam,seed)
    model_dir=OUT/'models'/f'seed{seed}';model_dir.mkdir(parents=True,exist_ok=True)
    mp=model_dir/f'{name}.pt';jp=model_dir/f'{name}.json'
    if jp.exists() and mp.exists():return json.loads(jp.read_text())
    if mp.exists() or jp.exists():raise FileExistsError(f'incomplete KC2 checkpoint: {name}')
    qseed=seed_init(seed,q_name,score_mode,tau)
    torch.manual_seed(qseed);rng=np.random.default_rng(qseed+707)
    model=KC2Net(q_name,score_mode).cpu()
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    train=ctx.splits['eh'];dev=ctx.splits['dev2'];target=targets['eh'][0];devtarget=targets['dev2'][0]
    gs=K1._as_gpusplit(kr,train,torch.device('cpu'),q_name=='iii')
    gd=K1._as_gpusplit(kr,dev,torch.device('cpu'),q_name=='iii')
    rawtarget=(target-mean)/std
    nearest=K1.nearest_allowed_coefficients(rawtarget,keys_np,train['task'],train['episode'],bank['task'],bank['episode'])
    mt=np.asarray(bank['task']);me=np.asarray(bank['episode']);labels=_gripper_labels(train)
    dpath=OUT/'data'/'eh';hist=np.load(dpath/'history.npy');hmask=np.load(dpath/'history_mask.npy')
    last=hist[:,9,6];valid=hmask.any(1)&(last!=0)
    ddev=OUT/'data'/'dev2';dh=np.load(ddev/'history.npy');dm=np.load(ddev/'history_mask.npy')
    dlast=dh[:,9,6];dvalid=dm.any(1)&(dlast!=0);devlabels=_gripper_labels(dev)
    best=None;best_any=None;history=[];start=time.time();n=len(target);batch=256
    for epoch in range(1,EPOCHS+1):
        model.train();order=rng.permutation(n);losses=[]
        for st in range(0,n,batch):
            ixn=order[st:st+batch];ix=torch.as_tensor(ixn,dtype=torch.long)
            q=model(gs,ix);qscore=model.score(q,key_space)
            allowed_np=K1.candidate_mask(train['task'][ixn],train['episode'][ixn],mt,me)
            random_ids=rng.choice(len(keys_np),size=min(256,len(keys_np)),replace=False)
            cand=np.unique(np.r_[nearest[ixn],random_ids]).astype(np.int64)
            candt=torch.as_tensor(cand,dtype=torch.long)
            allowed=torch.as_tensor(allowed_np[:,cand],dtype=torch.bool)
            if not bool(allowed.any(-1).all()):raise AssertionError('no legal in-batch candidates')
            norm_target=torch.as_tensor(rawtarget[ixn],dtype=torch.float32)
            dtrue=((norm_target[:,None]-key_space[candt][None])**2).sum(-1)
            td=torch.softmax((-dtrue/float(tau)).masked_fill(~allowed,-torch.inf),-1)
            logp=torch.log_softmax(qscore[:,cand].masked_fill(~allowed,-torch.inf),-1)
            retr=-(td*torch.nan_to_num(logp,nan=0.,neginf=0.)).sum(-1).mean()
            gl=model.grip_logits(q)
            bce=F.binary_cross_entropy_with_logits(gl,torch.as_tensor(labels[ixn],dtype=torch.float32))
            loss=retr+float(lam)*bce
            opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            with torch.no_grad():model.query.logit_scale.clamp_(max=float(np.log(100.)))
            losses.append((float(retr.detach()),float(bce.detach()),float(loss.detach())))
        devrec=evaluate_dev(model,gd,dev,devtarget,units,key_space,mt,me,devlabels,dlast,dvalid)
        eligible=devrec['dev_raw_coeff_sq']<=DEV_LIMIT
        key=((devrec['dev_raw_coeff_sq'],) if lam==0 else
             (devrec['joint_head_metrics']['errors'],devrec['dev_raw_coeff_sq'],-devrec['joint_head_threshold']))
        rec={'epoch':epoch,'retrieval_loss':float(np.mean([x[0] for x in losses])),
             'gripper_bce':float(np.mean([x[1] for x in losses])),'total_loss':float(np.mean([x[2] for x in losses])),
             'dev_raw_coeff_sq':devrec['dev_raw_coeff_sq'],'dev_gripper_errors':devrec['joint_head_metrics']['errors'],
             'dev_gripper_exact_accuracy':devrec['joint_head_metrics']['exact_step_accuracy'],
             'dev_threshold':devrec['joint_head_threshold'],'within_kc1_5pct':bool(eligible),
             'seconds':time.time()-start}
        history.append(rec)
        if best_any is None or devrec['dev_raw_coeff_sq']<best_any['rec']['dev_raw_coeff_sq']:
            best_any={'rec':rec,'state':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()}}
        if eligible and (best is None or key<best['key']):best={'key':key,'rec':rec,'state':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()}}
        if epoch in (1,4,8,12,16):log('train_epoch',name=name,**rec)
    if best is None:
        # Save the closest checkpoint for λ=0 controls and report its constraint failure.
        best=best_any;best['rec']['within_kc1_5pct']=False
    model_path=mp
    torch.save({'state_dict':best['state'],'config':{'q_name':q_name,'score_mode':score_mode,'tau':tau,
        'lambda':lam,'seed':seed,'standardize':True,'mean':mean,'std':std,'qseed':qseed},
        'selected_epoch':best['rec']['epoch']},model_path)
    result={'name':name,'q_name':q_name,'score_mode':score_mode,'tau':tau,'lambda':lam,'seed':seed,
        'checkpoint':str(model_path),'selected_epoch':best['rec']['epoch'],'dev_raw_coeff_sq':best['rec']['dev_raw_coeff_sq'],
        'dev_gripper_errors':best['rec']['dev_gripper_errors'],'dev_gripper_exact_accuracy':best['rec']['dev_gripper_exact_accuracy'],
        'dev_threshold':best['rec']['dev_threshold'],'within_kc1_5pct':best['rec']['within_kc1_5pct'],
        'history':history,'train_seconds':time.time()-start}
    jp.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    log('model_complete',name=name,dev_coeff=result['dev_raw_coeff_sq'],dev_grip_errors=result['dev_gripper_errors'])
    return result


def train_seed(seed):
    if seed not in SEEDS:raise ValueError(seed)
    kr,ctx,targets=load_ctx();bank=K1.memory_bank()
    results=[]
    for q in ('i','iii'):
      for score in ('dot','distance'):
       for tau in TAUS:
        for lam in LAMBDAS:
         results.append(train_one(kr,ctx,targets,bank,q_name=q,score_mode=score,tau=tau,lam=lam,seed=seed))
    return results


def smoke_objectives():
    """Small CPU backward pass through both retrieval scores and joint BCE."""
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','8')))
    from types import SimpleNamespace
    class Fake:
        def __init__(self,n):self.n=n
        def window(self,ix):
            n=len(ix);return torch.randn((n,10,16)),torch.ones((n,10),dtype=torch.bool)
    record=[]
    for mode in ('dot','distance'):
        m=KC2Net('i',mode).cpu();q=m(Fake(8),torch.arange(8));keys=torch.randn(24,18)
        score=m.score(q,keys);target=torch.randn(8,18);allowed=torch.ones((8,24),dtype=torch.bool)
        d=((target[:,None]-keys[None])**2).sum(-1);soft=torch.softmax(-d/.1,-1)
        retrieval=-(soft*torch.log_softmax(score,-1)).sum(-1).mean()
        labels=torch.randint(0,2,(8,10)).float();bce=F.binary_cross_entropy_with_logits(m.grip_logits(q),labels)
        loss=retrieval+.3*bce;loss.backward()
        if not torch.isfinite(loss):raise AssertionError('KC2 joint objective is nonfinite')
        if m.project.weight.grad is None or m.grip[0].weight.grad is None:raise AssertionError('missing retrieval/gripper gradients')
        record.append({'score_mode':mode,'retrieval':float(retrieval.detach()),'bce':float(bce.detach()),'loss':float(loss.detach())})
    log('smoke_objective_complete',modes=record)
    return record


def main():
    ap=argparse.ArgumentParser();ap.add_argument('cmd',choices=('train-seed','smoke'));ap.add_argument('--seed',type=int,choices=SEEDS)
    a=ap.parse_args()
    if a.cmd=='smoke':print(json.dumps(smoke_objectives(),ensure_ascii=False))
    else:
        if a.seed is None:raise SystemExit('--seed required for train-seed')
        print(json.dumps(train_seed(a.seed),ensure_ascii=False))

if __name__=='__main__':main()
