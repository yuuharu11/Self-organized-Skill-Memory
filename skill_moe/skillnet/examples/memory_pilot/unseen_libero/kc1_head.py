#!/usr/bin/env python3
"""Fit a KC1-specific gripper decoder on LIBERO-40 and select its threshold on Dev v2."""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn

import kc1_run as K
HERE=Path(__file__).resolve().parent
OUT=K.OUT


class KC1Head(nn.Module):
    def __init__(self, in_dim, out_dim=10):
        super().__init__()
        self.register_buffer('mu',torch.zeros(in_dim));self.register_buffer('sd',torch.ones(in_dim))
        self.net=nn.Sequential(nn.Linear(in_dim,256),nn.GELU(),nn.Dropout(.1),nn.Linear(256,64),nn.GELU(),nn.Linear(64,out_dim))
    def forward(self,x): return self.net((x-self.mu)/self.sd)


def retrieval_features(model, gs, sp, keys, key_task, key_episode):
    from kq2cl_common import summarize_scores
    device=next(model.parameters()).device; model.eval(); qrows=[]; score_rows=[]
    with torch.no_grad():
        for start in range(0,gs.n,128):
            ix=torch.arange(start,min(start+128,gs.n),device=device)
            q=model(gs,ix);qrows.append(q.float().cpu().numpy())
            score_rows.append(model.score(q,keys).float().cpu().numpy())
    q=np.concatenate(qrows);scores=np.concatenate(score_rows)
    allowed=K.candidate_mask(sp['task'],sp['episode'],key_task,key_episode)
    summ=summarize_scores(scores,allowed,k=50);ids=summ['ids'];logits=summ['logits']
    scale=float(model.logit_scale.exp().clamp(max=100.))
    topkeys=np.asarray(keys.cpu(),np.float32)[ids]
    norm=topkeys/np.maximum(np.linalg.norm(topkeys,axis=-1,keepdims=True),1e-6)
    pair=(norm[:,np.triu_indices(5,1)[0]]*norm[:,np.triu_indices(5,1)[1]]).sum(-1).mean(1)
    k1=norm[:,0];k5=norm[:,1:].mean(1);k1m=(k1*k5).sum(1)/np.maximum(np.linalg.norm(k5,axis=1),1e-6)
    gaps=lambda j: logits[:,0]-logits[:,j]
    scalar=np.stack([logits[:,0],gaps(1),gaps(4),gaps(9),gaps(49),summ['entropy'],
        np.log(np.maximum(summ['top1_prob'],1e-12)),summ['top5_mass'],logits[:,0]/scale,gaps(1)/scale,
        pair,k1m],1).astype(np.float32)
    ktop=np.asarray(keys.cpu(),np.float32)[ids[:,0]]
    return {'scalar':scalar,'q':q.astype(np.float32),'k1':ktop.astype(np.float32),
            'top_ids':ids,'top_logits':logits,'scores':scores}


def head_input(feats, gx, kind):
    pieces=[feats['scalar']]
    if kind=='full':pieces.extend([feats['q'],feats['k1']])
    pieces.append(np.asarray(gx,np.float32))
    return np.concatenate(pieces,1).astype(np.float32)


def _gripper_data(sp):
    import kq2cl_common as C
    d=OUT/'data'/'eh' if len(sp['key'])==12766 else OUT/'data'/'dev2'
    hist=np.load(d/'history.npy');hm=np.load(d/'history_mask.npy');state=np.load(d/'state.npy')
    last=hist[:,9,6];valid=hm.any(1)
    gx=C.grip_extra(last,valid,state[:,6:8])
    target=np.where(np.asarray(sp['actions'])[...,6]>0,1.,-1.).astype(np.float32)
    return gx,last,valid,target


def _metrics(prob, target, last, valid, th):
    import kq2cl_common as C
    decoded=C.decode_gripper(prob,np.where(valid,np.sign(last),-1.),th)
    m=C.switch_metrics(decoded,target,last,valid)
    return m|{'errors':int(m['missed_rows']+m['false_switch_rows'])}


def fit_head(kind, x, y, seed=1, epochs=30):
    torch.manual_seed(seed);torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','8')))
    model=KC1Head(x.shape[1]).cpu();xt=torch.as_tensor(x);yt=torch.as_tensor(y)
    model.mu.copy_(xt.mean(0));model.sd.copy_(xt.std(0).clamp_min(1e-4))
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    rng=np.random.default_rng(seed)
    for ep in range(epochs):
        model.train()
        for start in range(0,len(x),256):
            ix=torch.as_tensor(rng.integers(0,len(x),min(256,len(x))))
            loss=nn.functional.binary_cross_entropy_with_logits(model(xt[ix]),yt[ix])
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
    return model.eval()


def run():
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','8')))
    result_file=OUT/'offline_corrected.json'
    if not result_file.exists():raise FileNotFoundError('run KC1 offline first')
    result=json.loads(result_file.read_text()); selected=result['selected']
    import kq2_run as kr
    kr,ctx,targets=K._prepare_context()
    bank=K.memory_bank();config=torch.load(selected['checkpoint'],map_location='cpu',weights_only=False)['cfg']
    model=K._load_model(kr,selected['query'],'cpu',selected['seed'])
    model.load_state_dict(torch.load(selected['checkpoint'],map_location='cpu',weights_only=False)['query']);model.eval()
    kn,_,_=K.coefficient_keys(np.asarray(bank['coefficients']),standardize=selected['standardize'])
    keys=torch.as_tensor(kn,dtype=torch.float32)
    feats={}
    for split in ('eh','dev2'):
        sp=ctx.splits[split];gs=K._as_gpusplit(kr,sp,torch.device('cpu'),selected['query']=='iii')
        feats[split]=retrieval_features(model,gs,sp,keys,np.asarray(bank['task']),np.asarray(bank['episode']))
    xtr={};xdev={}
    gxtr,lasttr,validtr,ytr=_gripper_data(ctx.splits['eh'])
    gxdev,lastdev,validdev,ydev=_gripper_data(ctx.splits['dev2'])
    for kind in ('scalar','full'):
        xtr[kind]=head_input(feats['eh'],gxtr,kind);xdev[kind]=head_input(feats['dev2'],gxdev,kind)
    ytr_rows=(ytr>0).astype(np.float32).reshape(len(ytr),10)
    head_dir=OUT/'heads';head_dir.mkdir(exist_ok=True)
    if (head_dir/'bundle.pt').exists():raise FileExistsError(head_dir/'bundle.pt')
    rec={'status':'running','data_rows':{'eh':len(ytr),'dev2':len(ydev)},'features':'KC1 q-hidden + saved-coefficient retrieval scores/top-key + execution gripper state',
         'candidates':{}}
    for kind in ('scalar','full'):
        modelh=fit_head(kind,xtr[kind],ytr_rows,seed=1,epochs=30)
        with torch.no_grad():pclose=torch.sigmoid(modelh(torch.as_tensor(xdev[kind]))).numpy()
        thresholds={str(th):_metrics(pclose,ydev,lastdev,validdev,th) for th in (.5,.6,.7,.8,.9)}
        chosen=min(thresholds,key=lambda th:(thresholds[th]['errors'],-float(th)))
        rec['candidates'][kind]={'thresholds':thresholds,'selected_threshold':float(chosen)}
        rec['candidates'][kind]['model_state']=None
        rec['candidates'][kind]['dev_errors']=thresholds[chosen]['errors']
        log=OUT/'progress.log'
        with log.open('a') as f:f.write(json.dumps({'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'event':'head_dev_select','kind':kind,'threshold':float(chosen),'errors':thresholds[chosen]['errors']})+'\n')
        # Hold both fitted alternatives until Dev selects one; the full structure
        # keeps the q/key context available to the decoder.
        torch.save({'kind':kind,'state_dict':modelh.state_dict(),'threshold':float(chosen),
                    'selected_query':selected['name'],'input_dim':xtr[kind].shape[1]},head_dir/f'{kind}.pt')
    kind=min(rec['candidates'],key=lambda k:rec['candidates'][k]['dev_errors'])
    selected_head=torch.load(head_dir/f'{kind}.pt',map_location='cpu',weights_only=False)
    rec.update({'status':'complete','selected_kind':kind,'selected_threshold':selected_head['threshold'],
                'checkpoint':str(head_dir/f'{kind}.pt'),'selection':'Dev v2 missed+false switch rows'})
    (head_dir/'heads.json').write_text(json.dumps(rec,indent=2,ensure_ascii=False)+'\n')
    torch.save({'kind':kind,'state_dict':selected_head['state_dict'],'threshold':selected_head['threshold'],
                'input_dim':selected_head['input_dim'],'query_checkpoint':selected['checkpoint'],
                'query_config':config},head_dir/'bundle.pt')
    # Calibrate top-10 disagreement on Dev v2 to T5's established Dev fallback
    # mass; neither Eval split enters the threshold or radius coefficient.
    from t0x_common import calibrate_disagreement,topk_disagreement
    d2=ctx.splits['dev2']; fd=feats['dev2']; ids=fd['top_ids'][:,:10]; lg=fd['top_logits'][:,:10]
    units=np.asarray(bank['value_actions10'],np.float32)
    D,_=topk_disagreement(units[ids],lg,k=10)
    basecal=json.loads((K.ROOT/'t0x-01/t5/calibration.json').read_text())
    old_conf=np.load(K.ROOT/'kq2-cl-01/heads/pred_iii.npz')['dev2__ehat']
    b=float(basecal['b'])
    cal=calibrate_disagreement(D,old_conf,b)
    target_norm=(targets['dev2'][0]-np.asarray(config['mean']))/np.asarray(config['std']) if selected['standardize'] else targets['dev2'][0]
    code=np.asarray(bank['coefficients'],np.float32).reshape(len(bank['length']),18)
    if selected['standardize']:code=(code-np.asarray(config['mean']))/np.asarray(config['std'])
    true_coeff_error=np.square(target_norm-code[ids[:,0]]).sum(-1)
    from scipy.stats import spearmanr
    cal.update({'selected_query':selected['query'],'dev_coeff_error_spearman':float(spearmanr(D,true_coeff_error).statistic),
        'dev_D_quantiles_p05_p50_p95':np.quantile(D,[.05,.5,.95]).tolist(),
        'reference_rule':'T5 Dev v2 ehat>b fallback proportion; D_b uses inverted empirical CDF; T5 t0 radius formula'})
    (OUT/'calibration.json').write_text(json.dumps(cal,indent=2,ensure_ascii=False)+'\n')
    rec['calibration']=cal
    return rec


def log(): pass
if __name__=='__main__': print(json.dumps(run(),indent=2,ensure_ascii=False))
