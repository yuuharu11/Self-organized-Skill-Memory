#!/usr/bin/env python3
"""Frozen bridge comparison for corrected local-key drafts (no Expert training)."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np

import os as _os
_REPO = Path(__file__).resolve().parents[6]
_STATE_DIR = Path(_os.environ.get('SKILLNET_STATE_DIR', _REPO / '.skillnet-state'))
_PI05_ROOT = Path(_os.environ.get('SKILLNET_PI05_ROOT', _STATE_DIR / 'outputs/memory-pilot/pi05-memory40-20261104'))
_PI05_CODE = Path(__file__).resolve().parents[1]
ROOT=_PI05_ROOT
RUN=ROOT/'run-01'; OUT=ROOT/'local-key-02'; NOISE=4101; T0S=(.4,.6)
PI05=_PI05_CODE

def _load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec)
    assert spec.loader is not None;spec.loader.exec_module(mod);return mod

def run(out=OUT,batch_size=4):
    # The main corrected local key is selected on nonLong Dev. Add one key from
    # each local family and the learned local projection, all predeclared.
    summary=json.loads((out/'summary.json').read_text())
    pred=np.load(out/'predictions.npz',allow_pickle=False)
    ranks=out
    metric_names={m['name'] for m in summary['metrics']}
    dev_local=summary['protocol']['selected_local_variant']
    bnames=[n for n in metric_names if n.startswith('base_ee') and '+' not in n]
    # The better base-camera patch is selected by nonLong Dev, never Long Eval.
    bname=min(bnames,key=lambda n:summary['dev_region_selection'][n])
    draft_names=[]
    for n in (dev_local,bname,'invariant_projected_local'):
        if n not in draft_names:draft_names.append(n)
    direct_name='direct_regression'
    rows={p.stem:np.load(p,mmap_mode='r',allow_pickle=False) for p in (RUN/'rows').glob('*.npy')}
    qrows=np.asarray(pred['query_rows'],np.int64);hard=np.asarray(pred['hard_mask'],bool)
    truth=np.asarray(rows['actions'][qrows,:,:7],np.float32)
    drafts=[];names=[]
    for n in draft_names:
        f=out/f'ranks_{n.replace("+","_").replace("[","_").replace("]","").replace(",","_")}.npz'
        z=np.load(f,allow_pickle=False);cand=np.asarray(z['cand_rows'][:,0],np.int64)
        drafts.append(np.asarray(rows['actions'][cand],np.float32));names.append(n)
    direct7=np.asarray(pred['direct_action'],np.float32)
    if direct7.shape!=(len(qrows),10,7):raise AssertionError(direct7.shape)
    direct32=np.zeros((len(qrows),10,32),np.float32);direct32[:,:,:7]=direct7
    drafts.append(direct32);names.append(direct_name)
    zero=np.zeros_like(direct32);drafts.append(zero);names.append('zero')
    drafts=np.stack(drafts,axis=1)
    if drafts.shape[2:]!=(10,32):raise AssertionError(drafts.shape)

    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
    os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION','.20')
    topk=_load_module('local_key_topk_runtime',PI05/'topk_long'/'run.py')
    jax,jnp,policy,base,baseline,bridge=topk._load_runtime()
    n=len(qrows);c=len(names);outputs=np.empty((len(T0S),n,c,10,7),np.float32)
    normal=np.empty((n,10,7),np.float32);started=time.monotonic()
    for start in range(0,n,batch_size):
        end=min(start+batch_size,n);ids=qrows[start:end];b=end-start
        obs=policy.obs_batch(rows,ids)
        eps=jnp.asarray(np.stack([topk._noise(jax,NOISE,int(q)) for q in ids]))
        normal[start:end]=np.asarray(baseline(jax.random.key(0),obs,num_steps=10,noise=eps),np.float32)[:,:,:7]
        for ti,t0 in enumerate(T0S):
            outputs[ti,start:end]=topk._sample_bridge(bridge,jax,jnp,policy,rows,ids,drafts[start:end],t0,NOISE,start,
                                                      batch_size=batch_size,condition_batch=8)
        if end==n or (start//batch_size)%25==0:
            with (out/'generation_progress.jsonl').open('a') as f:
                f.write(json.dumps({'rows_done':end,'rows_total':n,'elapsed_s':time.monotonic()-started,
                                    'gpu_used_mib':topk._gpu_used_mib()})+'\n')
    mse=np.mean((outputs-truth[None,:,None])**2,axis=(-1,-2))
    normal_mse=np.mean((normal-truth)**2,axis=(-1,-2))
    hard_mse=mse[:,hard,:]
    summary_out={'query_rows':n,'noise_seed':NOISE,'t0':list(T0S),'draft_names':names,
                 'drafts':'top1 raw candidate actions for the selected local key, best base-EEF key by Dev, projected local; direct regression active7 zero-padded to 32; zero draft',
                 'expert':str(RUN/'bridge/001500'),'normal_policy':'pi05_libero base sample_actions with the frozen bridge checkpoint overlay',
                 'gpu_fraction':'.20','elapsed_seconds':time.monotonic()-started,'metrics':{}}
    for ti,t0 in enumerate(T0S):
        summary_out['metrics'][str(t0)]={}
        for ci,name in enumerate(names):
            summary_out['metrics'][str(t0)][name]={'all_mean_mse':float(mse[ti,:,ci].mean()),
                'hard_mean_mse':float(mse[ti,hard,ci].mean())}
        summary_out['metrics'][str(t0)]['normal_pi05']={'all_mean_mse':float(normal_mse.mean()),
            'hard_mean_mse':float(normal_mse[hard].mean())}
    np.savez_compressed(out/'generation.npz',query_rows=qrows,hard_mask=hard,draft_names=np.asarray(names),
                        t0=np.asarray(T0S),drafts=drafts,generated=outputs,generated_mse=mse,
                        normal=normal,normal_mse=normal_mse,truth=truth)
    (out/'generation_summary.json').write_text(json.dumps(summary_out,indent=2,ensure_ascii=False)+'\n')
    return summary_out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=OUT);ap.add_argument('--batch-size',type=int,default=4)
    a=ap.parse_args();r=run(a.output,a.batch_size);print(json.dumps({'output':str(a.output),'seconds':r['elapsed_seconds'],'methods':r['draft_names']}))

if __name__=='__main__':main()
