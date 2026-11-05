#!/usr/bin/env python3
"""Correct final ranking/summary from already extracted simulator features."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from core import assert_nonlong_roles, combine_blocks, normalize_block

import os as _os
_REPO = Path(__file__).resolve().parents[6]
_STATE_DIR = Path(_os.environ.get('SKILLNET_STATE_DIR', _REPO / '.skillnet-state'))
_PI05_ROOT = Path(_os.environ.get('SKILLNET_PI05_ROOT', _STATE_DIR / 'outputs/memory-pilot/pi05-memory40-20261104'))
ROOT = _PI05_ROOT
RUN = ROOT / 'run-01'
LOCAL = ROOT / 'local-key-02'
MEM = ROOT / 'skill-memories-01' / 'raw-rows-noLong'
OUT = ROOT / 'situation-oracle-01'
SEED = 20261105
REPS = 5000


def load_rows():
    return {p.stem: np.load(p, mmap_mode='r', allow_pickle=False)
            for p in (RUN / 'rows').glob('*.npy')}


def calibration_dev_episodes(rows):
    dev = []
    for task in np.unique(rows['task']):
        if int(task) < 10:
            continue
        eps = np.unique(rows['episode'][(rows['task'] == task) & rows['validation']])
        order = np.random.default_rng(SEED + int(task)).permutation(eps)
        dev.extend(order[:3].tolist())
    return np.asarray(dev, np.int64)


def weights_grid():
    grid = [(b, (b,), (1.,)) for b in ('E', 'R', 'G', 'H')]
    for w in ((.75,.25),(.5,.5),(.25,.75)):
        grid.append((f'R+G:{w[0]:g}/{w[1]:g}', ('R','G'), w))
    for w in ((.5,.25,.25),(.25,.5,.25),(.25,.25,.5),(1/3,1/3,1/3)):
        grid.append((f'R+G+H:{w[0]:.3f}/{w[1]:.3f}/{w[2]:.3f}',('R','G','H'),w))
    for w in ((.4,.2,.2,.2),(.2,.4,.2,.2),(.2,.2,.4,.2),(.2,.2,.2,.4),(.25,.25,.25,.25)):
        grid.append((f'E+R+G+H:{w[0]:.2f}/{w[1]:.2f}/{w[2]:.2f}/{w[3]:.2f}',('E','R','G','H'),w))
    return grid


def rank(scores, row_ids, k):
    order=np.lexsort((np.asarray(row_ids,np.int64),-np.asarray(scores,np.float32)))[:k]
    return np.asarray(row_ids,np.int64)[order]


def bootstrap(values, episodes, seed=SEED):
    x=np.asarray(values,np.float64);ep=np.asarray(episodes,np.int64);u=np.unique(ep)
    by={int(e):np.flatnonzero(ep==e) for e in u};rng=np.random.default_rng(seed)
    vals=np.empty(REPS,np.float64)
    for i in range(REPS):
        sample=rng.choice(u,len(u),replace=True);ix=np.concatenate([by[int(e)] for e in sample]);vals[i]=x[ix].mean()
    return {'mean':float(x.mean()),'ci95':[float(np.quantile(vals,.025)),float(np.quantile(vals,.975))],
            'episodes':int(len(u)),'rows':int(len(x)),'bootstrap_reps':REPS}


def per_query_metrics(qids, cands, pool_ids, rows, *, varying_reference=False):
    acts=np.asarray(rows['actions'][:,:,:7],np.float32)
    truth=np.asarray(acts[qids],np.float32)
    pool_map={int(r):i for i,r in enumerate(pool_ids)}
    arrays={k:np.full(len(qids),np.nan,np.float32) for k in
            ('top1_mse','top5_best_mse','recall8','xyz_cosine','gripper_sign_agreement','oracle_mse')}
    eligible=np.zeros(len(qids),bool)
    for i,rid in enumerate(qids):
        ids=np.asarray(cands[i],np.int64);ids=ids[ids>=0]
        if len(ids)==0: continue
        ca=np.asarray(acts[ids],np.float32);d=np.mean((ca-truth[i,None])**2,axis=(1,2))
        arrays['top1_mse'][i]=d[0];arrays['top5_best_mse'][i]=float(np.min(d[:5]))
        if varying_reference:
            allowed=pool_ids[(np.asarray(rows['task'][pool_ids])==int(rows['task'][rid])) &
                             (np.asarray(rows['episode'][pool_ids])!=int(rows['episode'][rid]))]
        else:
            allowed=pool_ids
        pa=np.asarray(acts[allowed],np.float32)
        full=np.mean((pa-truth[i,None])**2,axis=(1,2))
        arrays['oracle_mse'][i]=float(full.min())
        positive=full<=.01
        if positive.any():
            eligible[i]=True
            found=sum(int(np.mean((acts[x]-truth[i])**2)<=.01) for x in ids[:8])
            arrays['recall8'][i]=found/int(positive.sum())
        qxyz=truth[i,:,:3].sum(axis=0);pxyz=ca[0,:,:3].sum(axis=0)
        arrays['xyz_cosine'][i]=float(np.dot(qxyz,pxyz)/max(np.linalg.norm(qxyz)*np.linalg.norm(pxyz),1e-12))
        arrays['gripper_sign_agreement'][i]=float(np.mean(np.sign(truth[i,:,6])==np.sign(ca[0,:,6])))
    return arrays,eligible


def main():
    rows=load_rows();z=np.load(OUT/'simulator_features.npz',allow_pickle=False)
    row_ids=np.asarray(z['row_ids'],np.int64);lookup={int(r):i for i,r in enumerate(row_ids)}
    feats={b:np.asarray(z[b],np.float32) for b in ('E','R','G','H')}
    pred=np.load(LOCAL/'predictions.npz',allow_pickle=False)
    qids=np.asarray(pred['query_rows'],np.int64);hard=np.asarray(pred['hard_mask'],bool)
    qep=np.asarray(rows['episode'][qids],np.int64)
    pool=np.asarray(np.load(MEM/'units.npz',allow_pickle=False)['pool_row_ids'],np.int64)
    longmem=np.flatnonzero(np.asarray(rows['memory']) & (np.asarray(rows['task'])<10))
    dev_eps=calibration_dev_episodes(rows)
    dev=np.flatnonzero(np.isin(rows['episode'],dev_eps)&np.asarray(rows['validation'])&(np.asarray(rows['task'])>=10))
    assert_nonlong_roles(rows['task'], {'memory_pool': pool, 'dev_selection': dev})
    qnorm={};trnorm={};stats={}
    for b in feats:
        trnorm[b],allnorm,stats[b]=normalize_block(feats[b][np.asarray([lookup[int(r)] for r in pool])],feats[b])
        qnorm[b]=allnorm
    devpos=np.asarray([lookup[int(r)] for r in dev]);qpos=np.asarray([lookup[int(r)] for r in qids])
    all_actions=np.asarray(rows['actions'][:,:,:7],np.float32)
    dev_truth=all_actions[dev]
    dev_epos={int(r):i for i,r in enumerate(row_ids)}
    pool_task=np.asarray(rows['task'][pool]);dev_task=np.asarray(rows['task'][dev])
    grid=weights_grid();selection=[];best=None
    for name,blocks,w in grid:
        pk=combine_blocks({b:trnorm[b] for b in blocks},blocks,w)
        qk=combine_blocks({b:qnorm[b][devpos] for b in blocks},blocks,w)
        scores=qk@pk.T;top=np.empty(len(dev),np.int64)
        for i in range(len(dev)):
            s=scores[i].copy();s[pool_task==dev_task[i]]=-np.inf
            top[i]=rank(s,pool,1)[0]
        mse=np.mean((all_actions[top]-dev_truth)**2,axis=(1,2));item={'name':name,'blocks':list(blocks),'weights':list(w),'dev_top1_mse':float(mse.mean())}
        selection.append(item)
        if best is None or item['dev_top1_mse']<best['dev_top1_mse']:best=item
    key_specs=[(b,(b,),(1.,)) for b in ('E','R','G','H')]
    key_specs += [('R+G',('R','G'),(.5,.5)),('R+G+H',('R','G','H'),(1/3,1/3,1/3)),
                  ('E+R+G+H',('E','R','G','H'),(.25,.25,.25,.25)),
                  ('DEV_SELECTED',tuple(best['blocks']),tuple(best['weights']))]
    per_method={};ref_method={};rows_saved={'query_rows':qids,'episode':qep,'hard':hard}
    for name,blocks,w in key_specs:
        pk=combine_blocks({b:trnorm[b] for b in blocks},blocks,w)
        qk=combine_blocks({b:qnorm[b][qpos] for b in blocks},blocks,w)
        scores=qk@pk.T;top=np.asarray([rank(scores[i],pool,8) for i in range(len(qids))],np.int64)
        per_method[name],elig=per_query_metrics(qids,top,pool,rows)
        per_method[name]['eligible_recall8']=elig
        rows_saved[name+'_top8_rows']=top
        for met,v in per_method[name].items(): rows_saved[name+'_'+met]=v
        # Long same-task, different-episode reference pool. This is never used
        # to choose weights and is reported separately from the noLong bank.
        rpos=np.asarray([lookup[int(r)] for r in longmem],np.int64)
        rk=combine_blocks({b:qnorm[b][rpos] for b in blocks},blocks,w)
        rt=np.full((len(qids),8),-1,np.int64)
        for i,rid in enumerate(qids):
            allow=(np.asarray(rows['task'][longmem])==int(rows['task'][rid])) & (np.asarray(rows['episode'][longmem])!=int(rows['episode'][rid]))
            pos=np.flatnonzero(allow)
            if not len(pos): continue
            rt[i,:min(8,len(pos))]=rank(qk[i]@rk[pos].T,longmem[pos],min(8,len(pos)))
        ref_method[name],rel=per_query_metrics(qids,rt,longmem,rows,varying_reference=True)
        ref_method[name]['eligible_recall8']=rel
        rows_saved['ref_'+name+'_top8_rows']=rt
        for met,v in ref_method[name].items(): rows_saved['ref_'+name+'_'+met]=v
    # Non-learned baselines use the exact fixed row-ID mappings from run-01.
    localz=np.load(LOCAL/'ranks_wrist_lowerhalf_prev.npz',allow_pickle=False)
    lp=np.asarray(localz['cand_rows'],np.int64);lq=np.asarray(pred['query_rows'],np.int64);li={int(r):i for i,r in enumerate(lq)}
    local_top=np.stack([lp[li[int(r)],:8] for r in qids])
    local_m,le=per_query_metrics(qids,local_top,pool,rows);local_m['eligible_recall8']=le
    per_method['wrist_lowerhalf+prev']=local_m;rows_saved['local_top8_rows']=local_top
    for k,v in local_m.items():rows_saved['local_'+k]=v
    lookup_prev={(int(t),int(e),int(f)):i for i,(t,e,f) in enumerate(zip(rows['task'],rows['episode'],rows['frame']))}
    hold=np.zeros_like(all_actions[qids])
    for i,r in enumerate(qids):
        p=lookup_prev.get((int(rows['task'][r]),int(rows['episode'][r]),int(rows['frame'][r])-10))
        if p is not None:hold[i]=all_actions[p,-1][None]
    hold_mse=np.mean((hold-all_actions[qids])**2,axis=(1,2))
    rng=np.random.default_rng(SEED+718);rand=np.empty(len(qids),np.int64)
    for i,r in enumerate(qids):rand[i]=int(rng.choice(pool[np.asarray(rows['task'][pool])!=int(rows['task'][r])]))
    randtop=np.repeat(rand[:,None],8,axis=1)
    rand_m,re=per_query_metrics(qids,randtop,pool,rows);rand_m['eligible_recall8']=re
    per_method['random_one']=rand_m;rows_saved['random_rows']=rand
    for k,v in rand_m.items():rows_saved['random_'+k]=v
    oracle=np.asarray(pred['oracle_mse'],np.float32)
    per_method['hold_last']={'top1_mse':hold_mse,'top5_best_mse':hold_mse,'recall8':np.full(len(qids),np.nan),
        'xyz_cosine':np.full(len(qids),np.nan),'gripper_sign_agreement':np.mean(np.sign(hold[:,:,6])==np.sign(all_actions[qids,:,6]),axis=1),'oracle_mse':oracle}
    per_method['action_oracle']={'top1_mse':oracle,'top5_best_mse':oracle,'recall8':np.full(len(qids),np.nan),
        'xyz_cosine':np.full(len(qids),np.nan),'gripper_sign_agreement':np.full(len(qids),np.nan),'oracle_mse':oracle}
    rows_saved['hold_last_mse']=hold_mse
    score_selected=combine_blocks({b:trnorm[b] for b in best['blocks']},best['blocks'],best['weights'])
    q_selected=combine_blocks({b:qnorm[b][qpos] for b in best['blocks']},best['blocks'],best['weights'])
    order=np.argsort(-(q_selected@score_selected.T),axis=1,kind='stable')
    disp={}
    for k in (8,32):
        x=np.empty(len(qids),np.float32)
        for i in range(len(qids)):
            a=all_actions[pool[order[i,:k]]].reshape(k,-1)
            d=(np.sum(a*a,axis=1)[:,None]+np.sum(a*a,axis=1)[None]-2*a@a.T)/a.shape[1]
            x[i]=np.mean(d[np.triu_indices(k,1)])
        disp[str(k)]=x;rows_saved['dispersion_'+str(k)]=x
    summary={'protocol':{'queries':len(qids),'hard_queries':int(hard.sum()),'pool_rows':len(pool),
        'same_task_reference_rows':len(longmem),'dev_rows':len(dev),'dev_long_rows':0,'fit_or_weight_selection_used_long':False,
        'hdf5_action_windows_verified':8279,'episode_mappings':510,'hard_threshold':.01549410422449,
        'bootstrap_reps':REPS,'gpu_used':False,'recall8_definition':'top 8 retrievals intersect full eligible candidate pool with action MSE <= .01, divided by all eligible pool rows; mean/CIs only over queries with >=1 relevant candidate'},
        'selected_key':best,'weight_grid':selection,'metrics':{},'reference':{},'comparisons':{},'conditional_action_dispersion':{}}
    eps=np.asarray(rows['episode'][qids],np.int64)
    for name,d in per_method.items():
        # Reuse same raw vectors for summary intervals.
        for subset,mask in [('all_long',np.ones(len(qids),bool)),('hard',hard)]:
            rec=summary['metrics'].setdefault(subset,{}).setdefault(name,{})
            for k,v in d.items():
                if k=='eligible_recall8':continue
                a=np.asarray(v)
                ok=mask & np.isfinite(a)
                if ok.sum():
                    rec[k]=bootstrap(a[ok],eps[ok])
                    if k=='recall8':rec[k]['eligible_rows']=int(ok.sum())
    for name,d in ref_method.items():
        summary['reference'][name]={}
        for subset,mask in [('all_long',np.ones(len(qids),bool)),('hard',hard)]:
            rec=summary['reference'][name].setdefault(subset,{})
            for k,v in d.items():
                if k=='eligible_recall8':continue
                a=np.asarray(v);ok=mask & np.isfinite(a)
                if ok.sum():
                    rec[k]=bootstrap(a[ok],eps[ok])
                    if k=='recall8':rec[k]['eligible_rows']=int(ok.sum())
    for subset,mask in [('all_long',np.ones(len(qids),bool)),('hard',hard)]:
        summary['comparisons'][subset]={}
        for name in per_method:
            if 'top1_mse' not in per_method[name]:continue
            for baseline in ('wrist_lowerhalf+prev','hold_last'):
                diff=np.asarray(per_method[name]['top1_mse'])-np.asarray(per_method[baseline]['top1_mse'])
                summary['comparisons'][subset][f'{name}_minus_{baseline}']=bootstrap(diff[mask],eps[mask])
        summary['conditional_action_dispersion'][subset]={k:bootstrap(v[mask],eps[mask]) for k,v in disp.items()}
    np.savez_compressed(OUT/'row_metrics_final.npz',**rows_saved)
    (OUT/'results_final.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps({'selected':best,'all_H':summary['metrics']['all_long'].get('H',{}),
        'hard_H':summary['metrics']['hard'].get('H',{}),'output':str(OUT/'results_final.json')},ensure_ascii=False))


if __name__=='__main__':main()

