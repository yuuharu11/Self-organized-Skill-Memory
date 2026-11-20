#!/usr/bin/env python3
"""Re-evaluate KC1 checkpoints in the requested raw coefficient metric.

The first offline pass used standardized-space errors while comparing a
standardized-key candidate with raw-key candidates. This correction keeps the
retrieval score in each candidate's key space, but selects and reports using
the same raw 18-D saved-coefficient squared error for every candidate.
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
import numpy as np
import torch

import kc1_run as K

OUT=K.OUT


def evaluate(model, gs, sp, target_raw, keyspace, key_raw, key_task, key_episode,
             actions_bank, *, seed=9181):
    model.eval(); device=next(model.parameters()).device
    scores=[]
    with torch.no_grad():
        for start in range(0,gs.n,128):
            ix=torch.arange(start,min(start+128,gs.n),device=device)
            scores.append(model.score(model(gs,ix),keyspace).float().cpu().numpy())
    scores=np.concatenate(scores)
    allow=K.candidate_mask(sp['task'],sp['episode'],key_task,key_episode)
    scores[~allow]=-np.inf
    order=np.argsort(-scores,axis=1,kind='stable')
    sq=np.square(np.asarray(target_raw,np.float32)[:,None,:]-np.asarray(key_raw,np.float32)[None]).sum(-1)
    ids=order[:,0]; top5=order[:,:min(5,order.shape[1])]
    rng=np.random.default_rng(seed); random_ids=np.empty(len(ids),np.int64)
    for i in range(len(ids)):
        random_ids[i]=rng.choice(np.flatnonzero(allow[i]))
    acts=np.asarray(sp['actions'],np.float32)
    amse=np.square(np.asarray(actions_bank,np.float32)[ids,:,:7]-acts[:,:,:7]).mean((1,2))
    return {'rows':int(len(ids)),
        'top1_coeff_sq':float(sq[np.arange(len(ids)),ids].mean()),
        'top5_coeff_sq_mean':float(sq[np.arange(len(ids))[:,None],top5].mean()),
        'random_coeff_sq':float(sq[np.arange(len(ids)),random_ids].mean()),
        'oracle_coeff_sq':float(np.where(allow,sq,np.inf).min(1).mean()),
        'top1_action_mse_active7':float(amse.mean()),
        'per_row_top1_coeff_sq':sq[np.arange(len(ids)),ids].astype(float).tolist(),
        'top1_ids':ids.astype(int).tolist()}


def run():
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','8')))
    dest=OUT/'offline_corrected.json'
    if dest.exists(): raise FileExistsError(dest)
    import kq2_run as kr
    kr,ctx,targets=K._prepare_context()
    bank=K.memory_bank(); raw_coeff=np.asarray(bank['coefficients'],np.float32).reshape(len(bank['length']),18)
    entries=[]; model_dir=OUT/'models'
    for meta_path in sorted(model_dir.glob('q*_std*_tau*_s*.json')):
        meta=json.loads(meta_path.read_text()); blob=torch.load(meta['checkpoint'],map_location='cpu',weights_only=False)
        qname=meta['query']; standardized=bool(meta['standardize'])
        model=K._load_model(kr,qname,'cpu',meta['seed']);model.load_state_dict(blob['query']);model.eval()
        keyspace,_,_=K.coefficient_keys(np.asarray(bank['coefficients']),standardize=standardized,
            mean=np.asarray(blob['cfg']['mean']),std=np.asarray(blob['cfg']['std']))
        keys=torch.as_tensor(keyspace,dtype=torch.float32)
        sp=ctx.splits['dev2'];gs=K._as_gpusplit(kr,sp,torch.device('cpu'),qname=='iii')
        metric=evaluate(model,gs,sp,targets['dev2'][0],keys,raw_coeff,np.asarray(bank['task']),
                        np.asarray(bank['episode']),bank['value_actions10'])
        entries.append({**meta,'dev_raw_coeff_sq':metric['top1_coeff_sq'],'dev_raw_metric':metric})
    if len(entries)!=36: raise AssertionError(f'expected 36 model configs, got {len(entries)}')
    selected=min(entries,key=lambda x:(x['dev_raw_coeff_sq'],x['name']))
    blob=torch.load(selected['checkpoint'],map_location='cpu',weights_only=False)
    model=K._load_model(kr,selected['query'],'cpu',selected['seed']);model.load_state_dict(blob['query']);model.eval()
    keyspace,mean,std=K.coefficient_keys(np.asarray(bank['coefficients']),standardize=selected['standardize'],
        mean=np.asarray(blob['cfg']['mean']),std=np.asarray(blob['cfg']['std']))
    keys=torch.as_tensor(keyspace,dtype=torch.float32)
    final={}
    for split in ('dev2','learned','unseen20'):
        sp=ctx.splits[split];gs=K._as_gpusplit(kr,sp,torch.device('cpu'),selected['query']=='iii')
        final[split]=evaluate(model,gs,sp,targets[split][0],keys,raw_coeff,np.asarray(bank['task']),
                              np.asarray(bank['episode']),bank['value_actions10'])
        final[split]['per_row_target_coeff_flat']=targets[split][0].astype(np.float32).tolist()
    x1={q:K._score_x1_reference(kr,ctx,targets,raw_coeff,q) for q in ('i','iii')}
    result={'status':'offline_complete_corrected','metric':'raw unstandardized squared Euclidean error in saved 18-D DCT3 shape coefficients',
        'selection':'minimum Dev v2 raw top-1 coefficient squared error over all 36 trained checkpoints; Eval not used',
        'selection_correction':'candidate score uses that model key space; candidate selection and all reported coefficient errors use raw saved coefficient space',
        'bank_audit':K.verify_saved_coefficients(bank),'bank_units':len(raw_coeff),
        'query_rows':{s:len(ctx.splits[s]['key']) for s in K.SPLITS},
        'settings_grid':{'standardize':[False,True],'tau':list(K.TAUS),'query':['i','iii'],'seed':list(K.SEEDS),'epochs_max':K.EPOCHS},
        'selected':selected,'selected_results':final,'candidate_dev_raw_metrics':entries,
        'existing_kx1_kq2_reference':x1,'target_rule':'anchor_spans suffix to next boundary, extend to at least five actions',
        'inference_inputs':'observed state/action history and (iii only) current VLM tokens; no target, task ID, or pi0.5 output'}
    dest.write_text(json.dumps(result,indent=2,ensure_ascii=False,default=float)+'\n')
    with (OUT/'progress.log').open('a') as f:f.write(json.dumps({'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'event':'offline_raw_metric_selection_complete','selected':selected['name'],
        'dev2':selected['dev_raw_coeff_sq'],'learned':final['learned']['top1_coeff_sq'],
        'unseen20':final['unseen20']['top1_coeff_sq']})+'\n')
    return result


if __name__=='__main__': print(json.dumps(run(),indent=2,ensure_ascii=False))
