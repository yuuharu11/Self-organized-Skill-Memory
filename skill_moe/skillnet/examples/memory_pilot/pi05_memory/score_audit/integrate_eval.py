#!/usr/bin/env python3
"""Efficient Eval-only integration for VLM/non-VLM keys and policy-error slices.

Uses evaluate_keys._metrics without recomputing the expensive per-task table.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(HERE.parent))
import evaluate_keys as ek
import run

ROOT=Path('/work/.skillnet-state/outputs/memory-pilot/pi05-memory40-20261104/run-01')
AUDIT=Path('/work/.skillnet-state/outputs/memory-pilot/pi05-memory40-20261104/score-audit-01')
OUT=AUDIT/'integrated-full'

def main():
    meta=np.load(OUT/'oracle_meta.npz',allow_pickle=False)
    qrows=meta['query_rows'];mrows=meta['memory_rows'];splits=meta['query_split']
    qtask=meta['query_task'];qep=meta['query_episode'];mtask=meta['memory_task'];mep=meta['memory_episode']
    oracle=np.load(OUT/'oracle_action_mse.npy',mmap_mode='r',allow_pickle=False)
    eval_ix=np.flatnonzero(splits=='eval'); dev_ix=np.flatnonzero(splits=='dev')
    all_eval=[]; high_eval=[]
    for path in sorted((OUT/'keys').glob('*.npz')):
        key0=np.load(path,allow_pickle=False)
        if not np.array_equal(key0['query_rows'],qrows): raise ValueError(f'query order mismatch: {path}')
        ex=str(key0['exclusion'].item())
        for label,ix in [('eval',eval_ix)]:
            key={k:(key0[k][ix] if getattr(key0[k],'ndim',0)>0 and len(key0[k])==len(qrows) else key0[k]) for k in key0.files}
            for tau in (.005,.01,.02):
                all_eval.append({'method':path.stem,'split':label,'exclusion':ex,**ek._metrics(
                    key,oracle[ix],qrows[ix],qtask[ix],qep[ix],mrows,mtask,mep,tau=tau,bootstraps=0)})

    rows=run.load_rows(ROOT)
    pred=np.load(AUDIT/'pi05_pred_actions.npy',mmap_mode='r',allow_pickle=False)
    own=np.mean((np.asarray(pred[qrows],np.float32)-np.asarray(rows['actions'][qrows,:,:7],np.float32))**2,axis=(1,2))
    cutoffs={str(q):float(np.quantile(own[dev_ix],q)) for q in (.75,.90)}
    # Query-row -> oracle row and memory-row -> oracle column maps.
    qpos={int(r):i for i,r in enumerate(qrows)}; mpos={int(r):i for i,r in enumerate(mrows)}
    own_lookup={int(r):float(own[i]) for i,r in enumerate(qrows)}
    for path in sorted((OUT/'keys').glob('*.npz')):
        key0=np.load(path,allow_pickle=False);ex=str(key0['exclusion'].item())
        for q,threshold in cutoffs.items():
            ix=np.asarray([i for i in eval_ix if own[i]>=threshold],np.int64)
            key={k:(key0[k][ix] if getattr(key0[k],'ndim',0)>0 and len(key0[k])==len(qrows) else key0[k]) for k in key0.files}
            high_eval.append({'method':path.stem,'split':'eval','policy_error_quantile':q,
                'threshold_from_dev':threshold,'exclusion':ex,**ek._metrics(
                    key,oracle[ix],qrows[ix],qtask[ix],qep[ix],mrows,mtask,mep,tau=.01,bootstraps=0)})

    # No-search baselines, aligned to identical Eval rows.
    eval_rows=qrows[eval_ix]
    pi_mse=own[eval_ix]
    pred_key=np.load(OUT/'keys/pi05_pred_to_memory_action_task.npz',allow_pickle=False)
    cand=pred_key['cand_rows'][eval_ix,0]
    pi_top=np.asarray([oracle[int(qi),mpos[int(cand[j])]] for j,qi in enumerate(eval_ix)],np.float32)
    bases=[]
    # Prior decode on CPU. Report both base-only (r=0) and full prior latent.
    sys.path.insert(0,str(HERE.parent))
    import torch
    torch.set_num_threads(8)
    model=run.restore_codec(ROOT)
    feats=np.load(ROOT/'pooled.npy',mmap_mode='r',allow_pickle=False)
    s=np.asarray(rows['state'][eval_rows],np.float32);f=np.asarray(feats[eval_rows],np.float32)
    b,r=model.prior(f,s)
    true=np.asarray(rows['actions'][eval_rows,:,:7],np.float32)
    for name,rr in [('codec_prior_b_r0',np.zeros_like(r)),('codec_prior_b_plus_prior_r',r)]:
        draft=model.decode(b,rr,f,s)[:,:,:7]
        mse=np.mean((draft-true)**2,axis=(1,2))
        bases.append({'name':name,'n':len(mse),'mean':float(mse.mean()),'median':float(np.median(mse)),
                      'le_0.01_rate':float(np.mean(mse<=.01))})
    summary={'n_eval':len(eval_ix),'pi05_policy_prediction':{'mean':float(pi_mse.mean()),'median':float(np.median(pi_mse)),
        'le_0.01_rate':float(np.mean(pi_mse<=.01))},
        'pi05_prediction_key_top1_true_action_mse':{'mean':float(pi_top.mean()),'median':float(np.median(pi_top)),
          'fraction_memory_top1_better_than_policy_prediction':float(np.mean(pi_top<pi_mse))},
        'codec_prior_no_search':bases}
    (OUT/'eval_tau_results.json').write_text(json.dumps({'definition':'evaluate_keys._metrics; eval split only; point estimates; tau=.005/.01/.02',
        'results':all_eval},indent=2,allow_nan=True)+'\n')
    (OUT/'policy_error_subsets.json').write_text(json.dumps({'thresholds_from_dev_policy_error':cutoffs,
        'results':high_eval},indent=2,allow_nan=True)+'\n')
    (OUT/'no_search_baselines.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'methods':len(list((OUT/'keys').glob('*.npz'))),'eval_rows':len(eval_ix),
        'high_error_rows':{q:int(sum(own[i]>=v for i in eval_ix)) for q,v in cutoffs.items()},'baselines':summary},indent=2))

if __name__=='__main__':main()
