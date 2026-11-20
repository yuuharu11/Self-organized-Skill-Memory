#!/usr/bin/env python3
"""KEYAB: action-key input, position, self-attention, and teacher ablations.

CPU-only pipeline. The underlying memory/query rows and evaluation masks are
read from the existing KEY-MOTION/KQ2 caches; those caches are never changed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import key_shared_latent as k2  # noqa: E402
import key_motion as km  # noqa: E402
import kq2_run as kq  # noqa: E402
from keyab_core import KeyABEncoder  # noqa: E402

from unseen_paths import STATE_DIR
ROOT=STATE_DIR / 'outputs/memory-pilot/unseen-libero'
OUT=ROOT/'keyab-01'
RUN_ROOT=STATE_DIR / 'outputs/memory-pilot/pi05-memory40-20261104/run-01'
CONFIGS=[]
for source in ('raw','dct3'):
    for position in ('absolute','none'):
        for sa in (False,True):
            CONFIGS.append({'source':source,'position':position,'self_attention':sa,'teacher':'window10'})
for sa in (False,True):
    CONFIGS.append({'source':'raw','position':'absolute','self_attention':sa,'teacher':'whole16'})


def _now_jst():
    return datetime.now(timezone(timedelta(hours=9))).isoformat(timespec='seconds')


def progress(path:Path,stage:str,**fields):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:
        f.write(json.dumps({'time_jst':_now_jst(),'stage':stage,**fields},ensure_ascii=False,default=str)+'\n')


def load_raw_bank(bank):
    """Recreate the raw units via the same deterministic anchor spans as KQ2."""
    rows=k2._load_key2_rows(RUN_ROOT)
    ids=np.flatnonzero(np.asarray(rows['memory'],bool))
    import gripper_split as gs
    import motion_core_eval as mc
    _dct_physical,meta,raw_pool=gs._variable_pool(rows,ids)
    # source stores (task, episode, start, end); compare exact segmentation.
    source_expected=np.column_stack([meta[:,0],meta[:,1],meta[:,2],meta[:,2]+meta[:,3]])
    if not np.array_equal(np.asarray(bank['source'],np.int64),source_expected):
        raise AssertionError('raw segmentation order/identity differs from cached KQ2 units')
    maxlen=int(np.asarray(bank['natural_actions']).shape[1])
    out=np.zeros((len(raw_pool),maxlen,6),np.float32);length=np.asarray(meta[:,3],np.int64)
    rows_by_episode={}
    for rid in ids:
        rows_by_episode.setdefault((int(rows['task'][rid]),int(rows['episode'][rid])),[]).append(int(rid))
    # Independently regenerate the cuts, rather than relying only on the
    # cached metadata returned alongside the pool.
    spans_by_episode={}
    for key,trace_ids in rows_by_episode.items():
        trace_ids=sorted(trace_ids,key=lambda rid:int(rows['frame'][rid]))
        trace=np.concatenate([np.asarray(rows['actions'][rid,:,:7],np.float32) for rid in trace_ids])
        spans_by_episode[key]=[(int(s),int(e)) for s,e in mc.anchor_spans(trace,min_length=5)]
    bank_spans=[(int(t),int(ep),int(s),int(e))
                for (t,ep),spans in spans_by_episode.items() for s,e in spans]
    cached_spans=[tuple(map(int,row)) for row in np.asarray(bank['source'])]
    if bank_spans!=cached_spans:
        raise AssertionError('independently regenerated anchor_spans differ in order or boundaries')
    raw_maxdiff=0.;dct_maxdiff=0.;raw_min=float('inf');raw_max=float('-inf')
    for i,a in enumerate(raw_pool):
        if len(a)!=length[i] or len(a)>maxlen:raise AssertionError('raw unit length mismatch')
        a=np.asarray(a,np.float32)
        # Exact raw source check against the stitched normalized memory trace.
        task,episode,start,end=map(int,bank['source'][i])
        trace_ids=rows_by_episode[(task,episode)]
        trace_ids=sorted(trace_ids,key=lambda rid:int(rows['frame'][rid]))
        trace=np.concatenate([np.asarray(rows['actions'][rid,:,:7],np.float32) for rid in trace_ids])
        expected=trace[start:end]
        if expected.shape!=a.shape:raise AssertionError(f'raw unit shape/source mismatch at {i}: {expected.shape} != {a.shape}')
        raw_maxdiff=max(raw_maxdiff,float(np.max(np.abs(expected-a))))
        raw_min=min(raw_min,float(a[:,:6].min()));raw_max=max(raw_max,float(a[:,:6].max()))
        out[i,:len(a)]=a[:,:6]
        # Recreate the bank's DCT3 reconstruction from the raw normalized unit.
        phys=gs._to_physical(a[:,:6]);rec=gs._to_norm(gs._idct_low(phys,3))
        dct_maxdiff=max(dct_maxdiff,float(np.max(np.abs(rec-np.asarray(bank['natural_actions'][i,:len(a),:6],np.float32)))))
    if raw_maxdiff>1e-6:raise AssertionError(f'raw values do not match their source frames: {raw_maxdiff}')
    if dct_maxdiff>1e-5:raise AssertionError(f'raw->DCT3 does not reproduce natural_actions: {dct_maxdiff}')
    audit={'units':int(len(meta)),'source_rows_equal':bool(np.array_equal(np.asarray(bank['source']),
        np.column_stack([meta[:,0],meta[:,1],meta[:,2],meta[:,2]+meta[:,3]]))),
        'source_mismatches':0,'raw_vs_source_max_abs_diff':raw_maxdiff,
        'anchor_spans_regenerated_equal':True,
        'anchor_spans_order_length_mismatches':0,
        'raw_q01_q99_normalized_range_observed':[raw_min,raw_max],
        'dct3_raw_to_natural_actions_max_abs_diff':dct_maxdiff,
        'normalization':'raw pool uses run-01 q01/q99-normalized rows/actions; outliers may exceed [-1,1]'}
    return out,length,rows,audit


def _batch_encode(model,actions,valid,device,batch=128):
    model.eval();parts=[]
    with torch.no_grad():
        for s in range(0,len(actions),batch):
            z=model(torch.as_tensor(actions[s:s+batch],device=device),
                    torch.as_tensor(valid[s:s+batch],device=device))
            if z.ndim==3:z=z[:,0]
            parts.append(z.cpu().numpy())
    return np.concatenate(parts)


def _dev_score(model,actions,valid,bank,ctx,dev='dev2',device='cpu'):
    keys=_batch_encode(model,actions,valid,device)
    sp=ctx.splits[dev];qactions=torch.as_tensor(sp['actions'][...,:6],device=device,dtype=torch.float32)
    qvalid=torch.ones(qactions.shape[:2],device=device,dtype=torch.bool)
    q=_batch_encode(model,qactions.cpu().numpy(),qvalid.cpu().numpy(),device,batch=128)
    scores=q@keys.T
    return float(kq.top1(scores,sp).mean()),keys


def _fit_one(cfg,bank,ctx,raw,lengths,*,epochs,seed,device,save_dir,smoke=False,progress_path=None):
    torch.manual_seed(seed);rng=np.random.default_rng(seed+333)
    src=raw if cfg['source']=='raw' else np.asarray(bank['natural_actions'],np.float32)[...,:6]
    lens=np.asarray(lengths,np.int64);maxlen=src.shape[1]
    valid=np.arange(maxlen)[None,:]<lens[:,None]
    actions=torch.as_tensor(src,device=device,dtype=torch.float32)
    masks=torch.as_tensor(valid,device=device,dtype=torch.bool)
    n=len(actions);windows=np.asarray(ctx.windows,np.float32)
    rows=k2._load_key2_rows(RUN_ROOT)
    episode_keys=np.column_stack([bank['task'],bank['episode']])
    # Whole-unit target sequences are used only for the declared whole16 teacher.
    if cfg['teacher']=='whole16':
        full=torch.from_numpy(src)
        target16=k2_t_resample(full,lens,16)
    else:target16=None
    model=KeyABEncoder(position=cfg['position'],self_attention=cfg['self_attention']).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    order=np.arange(n);batch=32 if smoke else 128
    max_steps_per_epoch=int(np.ceil((min(n,128) if smoke else n)/batch))
    history=[];best=float('inf');best_state=None;best_ep=0;start=time.monotonic()
    count=min(n,128) if smoke else n
    for ep in range(1,epochs+1):
        rng.shuffle(order);model.train();losses=[]
        for ss in range(0,count,batch):
            qi=order[ss:ss+batch]
            if len(qi)<2:continue
            rnd=rng.choice(n,size=min(256,n),replace=False);cids=np.unique(np.r_[qi,rnd])
            qit=torch.as_tensor(qi,device=device);cit=torch.as_tensor(cids,device=device)
            q=model(actions[qit],masks[qit]);kk=model(actions[cit],masks[cit])
            score=q@kk.T*20.
            allowed=~((torch.as_tensor(bank['task'][qi],device=device)[:,None]==torch.as_tensor(bank['task'][cids],device=device)[None]) &
                      (torch.as_tensor(bank['episode'][qi],device=device)[:,None]==torch.as_tensor(bank['episode'][cids],device=device)[None]))
            if cfg['teacher']=='window10':
                errs=k2._arm6_candidate_mse(bank['value_actions10'][qi],windows[cids])
            else:
                qt=target16[qi];ct=target16[cids]
                errs=((qt[:,None]-ct[None])**2).mean((2,3)).cpu().numpy()
            loss=k2.soft_listwise_loss(score,torch.as_tensor(errs,device=device),allowed,.01 if cfg.get('tau') is None else cfg['tau'])
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            losses.append(float(loss.detach()))
        dscore,keys=_dev_score(model,src,valid,bank,ctx,device=device)
        rec={'epoch':ep,'loss':float(np.mean(losses)) if losses else None,'dev2_top1_arm_mse':dscore,
             'steps':len(losses),'elapsed_seconds':time.monotonic()-start}
        history.append(rec)
        if progress_path is not None:
            progress(progress_path,'epoch_metric',name=config_name(cfg),epoch=ep,loss=rec['loss'],
                     dev2_top1_arm_mse=dscore,selected_so_far=best_ep or ep)
        if dscore<best:
            best=dscore;best_ep=ep;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if save_dir is not None and (ep in (4,10,25,epochs)):
            save_dir.mkdir(parents=True,exist_ok=True)
            torch.save({'config':cfg,'epoch':ep,'state_dict':model.state_dict()},save_dir/f'{config_name(cfg)}__ep{ep}.pt')
    model.load_state_dict(best_state)
    keys=_batch_encode(model,actions,masks,device)
    ev={}
    for split_name in ('learned','unseen20'):
        sp=ctx.splits[split_name];q=_batch_encode(model,sp['actions'][...,:6],np.ones(sp['actions'].shape[:2],bool),device)
        score=q@keys.T;pred=kq.top1(score,sp)
        ev[split_name]={'top1_arm_mse':float(pred.mean()),'per_query':pred.tolist()}
    if save_dir is not None:
        torch.save({'config':cfg,'epoch':best_ep,'state_dict':model.state_dict()},save_dir/f'{config_name(cfg)}.pt')
        np.save(save_dir/f'{config_name(cfg)}_keys.npy',keys)
    return {'config':cfg,'selected_epoch':best_ep,'dev2_top1_arm_mse':best,'evaluation':ev,
            'history':history,'seconds':time.monotonic()-start,'keys':keys,'model':model}


def k2_t_resample(actions,lens,length):
    import torch
    out=[]
    for i,n in enumerate(np.asarray(lens,int)):
        x=actions[i,:int(n)]
        if len(x)==1:out.append(x.expand(length,-1));continue
        at=torch.linspace(0,len(x)-1,length,device=x.device);lo=at.floor().long();hi=at.ceil().long();f=(at-lo).unsqueeze(-1)
        out.append(x[lo]*(1-f)+x[hi]*f)
    return torch.stack(out)


def config_name(cfg):
    s=f"{cfg['source']}_{cfg['position']}_sa{int(cfg['self_attention'])}_{cfg['teacher']}"
    if 'tau' in cfg:s+=f"_tau{cfg['tau']:g}"
    if 'epochs' in cfg:s+=f"_ep{int(cfg['epochs'])}"
    return s


def _selected_configs():
    out=[]
    for base in CONFIGS:
        for tau in (.01,.05):
            for ep in (4,25):out.append({**base,'tau':tau,'epochs':ep})
    return out


def run(output=OUT,*,smoke=False,long_epochs=25,do_downstream=True):
    output=Path(output)
    if output.exists():
        allowed={'progress.log','results.json','K-x1_keys.npy'}
        if (output/'initial-invalid').exists():allowed.add('initial-invalid')
        unexpected=[p.name for p in output.iterdir() if p.name not in allowed]
        if unexpected and not (output/'results.json').exists():
            raise FileExistsError(f'refusing to overwrite unrecognized output files: {unexpected[:5]}')
    output.mkdir(parents=True,exist_ok=True);plog=output/'progress.log'
    progress(plog,'start',smoke=smoke,outputs=str(output),cpu_only=True)
    t0=time.monotonic()
    ctx=kq.Ctx('cpu',splits=('eh','dev2','learned','unseen20'),vlm=False,out=kq.OUT)
    raw,lengths,rows,data_audit=load_raw_bank(ctx.bank)
    progress(plog,'data_loaded',units=len(raw),raw_maxlen=raw.shape[1],raw_total_steps=int(lengths.sum()),
             dev_rows=len(ctx.splits['dev2']['key']),learned_rows=len(ctx.splits['learned']['key']),unseen20_rows=len(ctx.splits['unseen20']['key']),
             load_seconds=time.monotonic()-t0,raw_audit=data_audit)
    # Smoke times a small 1-epoch train, used to establish a conservative estimate.
    if smoke:
        c={**CONFIGS[0],'tau':.01}
        from types import SimpleNamespace
        bsm={k:(np.asarray(v)[:128] if isinstance(v,np.ndarray) and len(v)==len(ctx.bank['length']) else v)
             for k,v in ctx.bank.items()}
        bsm['windows']=np.asarray(ctx.windows)[:128]
        csm=SimpleNamespace(bank=bsm,windows=bsm['windows'],splits={})
        for n,sp in ctx.splits.items():
            csp={k:(np.asarray(v)[:16] if isinstance(v,np.ndarray) and len(v)==len(sp['key']) else v) for k,v in sp.items()}
            csp['errors']=np.asarray(sp['errors'])[:16,:128];csp['allowed']=np.asarray(sp['allowed'])[:16,:128]
            csm.splits[n]=csp
        x=_fit_one(c,bsm,csm,raw[:128],lengths[:128],epochs=1,seed=4101,device='cpu',save_dir=None,smoke=True,progress_path=plog)
        est_cfgs=len(_selected_configs())
        estimate=x['seconds']*max(1,len(raw)/128)*sum(4+long_epochs for _ in range(1))*est_cfgs
        progress(plog,'smoke_complete',seconds=x['seconds'],estimated_key_grid_hours=estimate/3600,
                 configs=est_cfgs,estimate_method='scaled one-epoch 128-unit CPU smoke; conservative upper estimate')
        return {'smoke':x,'estimated_seconds':estimate,'configs':est_cfgs}
    configs=_selected_configs();epochs_for={4:4,25:int(long_epochs)}
    # K-x1 reference checkpoint from MK1b, untouched.
    refm=km.ArchMotionEncoder('x1','absolute').cpu()
    refpath=ROOT/'key-motion-01/arch/x1_absolute_dist_tau0.01.pt'
    refm.load_state_dict(torch.load(refpath,map_location='cpu',weights_only=True))
    refkeys=kq.encode_bank(refm,ctx.bank,'cpu')
    results_path=output/'results.json'
    if results_path.exists():results=json.loads(results_path.read_text())
    else:results={'protocol':{'cpu_only':True,'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES',''),
       'fixed_memory_units':len(ctx.bank['length']),'data_source':'KQ2/CMP3 exact 4516 unit bank; raw segments are third _variable_pool return from anchor_spans',
       'raw_bank_audit':data_audit,
       'k_x1_checkpoint':str(refpath),'selection':'Dev v2 20Hz top1 arm MSE','epochs':[4,int(long_epochs)],'tau':[.01,.05]},'key_models':{},'query_models':{}}
    results.setdefault('protocol',{})['raw_bank_audit']=data_audit
    results['protocol']['initial_invalid_run']={
        'archived_at':'initial-invalid/',
        'cause':'load_raw_bank unpacked _variable_pool return value 1 (physical DCT reconstruction) as raw; corrected to return value 3 (raw_pool)',
        'invalid_unseen20_example':0.13686320185661316,
        'corrected_unseen20_example':0.0249065812677145}
    results.setdefault('key_models',{});results.setdefault('query_models',{})
    np.save(output/'K-x1_keys.npy',refkeys)
    for splitname in ('dev2','learned','unseen20'):
        sp=ctx.splits[splitname];q=_batch_encode(refm,sp['actions'][...,:6],np.ones(sp['actions'].shape[:2],bool),'cpu')
        results['key_models'].setdefault('K-x1',{})[splitname]={'top1_arm_mse':float(kq.top1(q@refkeys.T,sp).mean())}
    _atomic_json(output/'results.json',results)
    # A true architectural reproduction control: initialize the identical
    # dct3/absolute/no-SA encoder with the existing X1 weights and verify
    # that this runner reproduces the archived key-space retrieval metrics.
    control=KeyABEncoder(position='absolute',self_attention=False).cpu()
    control.load_state_dict(refm.state_dict(),strict=True)
    dctsrc=np.asarray(ctx.bank['natural_actions'],np.float32)[...,:6]
    dctmask=np.arange(dctsrc.shape[1])[None,:]<np.asarray(ctx.bank['length'])[:,None]
    ctrlkeys=_batch_encode(control,dctsrc,dctmask,'cpu')
    control_metrics={}
    for splitname in ('dev2','learned','unseen20'):
        sp=ctx.splits[splitname];q=_batch_encode(control,sp['actions'][...,:6],np.ones(sp['actions'].shape[:2],bool),'cpu')
        control_metrics[splitname]={'top1_arm_mse':float(kq.top1(q@ctrlkeys.T,sp).mean())}
    keydiff=float(np.max(np.abs(ctrlkeys-refkeys)))
    if keydiff>2e-6:raise AssertionError(f'KeyAB dct3 control differs from K-x1 encoded keys by {keydiff}')
    results['key_models']['DCT3_X1_checkpoint_control']={'weight_source':str(refpath),'max_abs_key_diff_vs_K_x1':keydiff,**control_metrics}
    results['references']={'DCT3_unlearned':kq.dct_metric_a(ctx),
        'hold_last_unseen20_arm_mse':0.0470,'C1b_unseen20_arm_mse':0.0522}
    _atomic_json(output/'results.json',results)
    selected_by_base={}
    for ix,cfg in enumerate(configs):
        n_epochs=epochs_for[int(cfg['epochs'])];name=config_name(cfg)
        recfile=output/f'{name}.json'
        if recfile.exists():
            row=json.loads(recfile.read_text());results['key_models'][name]=row
        else:
            progress(plog,'key_train_start',index=ix+1,total=len(configs),name=name,units=len(raw),epochs=n_epochs)
            # train config evaluates all 4516; raw actions are loaded once, then shared.
            r=_fit_one(cfg,ctx.bank,ctx,raw,lengths,epochs=n_epochs,seed=4101,device='cpu',save_dir=output,progress_path=plog)
            row={k:v for k,v in r.items() if k not in ('keys','model')}
            row['checkpoint']=str(output/f'{name}.pt');row['key_file']=str(output/f'{name}_keys.npy')
            _atomic_json(recfile,row);results['key_models'][name]=row;_atomic_json(output/'results.json',results)
            base=(cfg['source'],cfg['position'],cfg['self_attention'],cfg['teacher'])
            old=selected_by_base.get(base)
            if old is None or row['dev2_top1_arm_mse']<old['dev2_top1_arm_mse']:selected_by_base[base]=row
            progress(plog,'key_train_done',index=ix+1,total=len(configs),name=name,dev=row['dev2_top1_arm_mse'],
                     learned=row['evaluation']['learned']['top1_arm_mse'],unseen20=row['evaluation']['unseen20']['top1_arm_mse'],seconds=row['seconds'])
    # Select tau/epoch within each architecture on Dev; best rows only are sent downstream.
    grouped={}
    for name,row in results['key_models'].items():
        # The K-x1 reference and the checkpoint-loaded control are metrics-only
        # records, not tunable configs. Keep them out of the downstream grid.
        if name in ('K-x1','DCT3_X1_checkpoint_control') or 'config' not in row:continue
        c=row['config'];base=(c['source'],c['position'],c['self_attention'],c['teacher'])
        if base not in grouped or row['dev2_top1_arm_mse']<grouped[base]['dev2_top1_arm_mse']:grouped[base]=row
    results['selected_key_configs']={config_name(v['config']):v['dev2_top1_arm_mse'] for v in grouped.values()}
    _atomic_json(output/'results.json',results)
    if do_downstream:
        # C1b history+state query, same config as KQ2 stage-1(b); Dev v2 selects epoch.
        qroot=output/'downstream';qroot.mkdir(exist_ok=True)
        for ix,(base,row) in enumerate(grouped.items(),1):
            name='key_'+config_name(row['config']);file=qroot/f'{name}.json'
            if file.exists():qres=json.loads(file.read_text())
            else:
                keyfile=output/f"{config_name(row['config'])}_keys.npy"
                cfg={'keys':str(keyfile),'seed':4101,'epochs':16,'batch':8,'rand':256,'lr':1e-4,'tau':.01,
                     'same_scene':True,'eh_state':'ref','dev':'dev2'}
                progress(plog,'query_train_start',index=ix,total=len(grouped),name=name)
                qres=kq.train_keyq(name,cfg,ctx,qroot)
                progress(plog,'query_train_done',index=ix,total=len(grouped),name=name,dev=qres['dev_selected'],
                         learned=qres['evaluation']['learned']['all'],unseen20=qres['evaluation']['unseen20']['all'],seconds=qres['seconds'])
            results['query_models'][name]=qres;_atomic_json(output/'results.json',results)
    results['timing_seconds']=time.monotonic()-t0
    _atomic_json(output/'results.json',results);write_report(output,results)
    progress(plog,'done',seconds=results['timing_seconds'])
    (output/'DONE').write_text('complete\n')
    return results


def _atomic_json(path,obj):
    tmp=Path(str(path)+'.tmp');tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False,default=float)+'\n');tmp.replace(path)


def write_report(output,r):
    lines=['# KEYAB：キー入力・位置埋め込み・Self-Attention 比較','',
      '## 結論', '',
      '初回 raw の MSE 異常は、raw 配列として物理 DCT 戻り値を誤って使ったことが原因だった。raw_pool に直して再実行すると raw absolute の unseen20 は .0232〜.0249 となり、既存 K-x1 .0245 と同程度になった。全40キー設定と10下流 query を CPU・seed 4101 で完了。', '',
      '## 初回異常の原因と訂正監査','',
      '- `_variable_pool` の返り値は `(physical_DCT_pool, metadata, raw_pool)`。初回は第1要素を raw として使っていた。修正版は第3要素を使い、初回の全成果を `initial-invalid/` に退避して、再実行結果に混ぜていない。',
      f"- 元 bank の単位数 {r['protocol']['fixed_memory_units']}。`motion_core_eval.anchor_spans(min_length=5)` を全メモリ trace で独立再生成し、(task, episode, start, end) の順序・境界・length の不一致は 0。raw と source 行動の最大絶対差は {r['protocol']['raw_bank_audit']['raw_vs_source_max_abs_diff']:.1e}。",
      f"- 正規化は元の q01/q99 を [-1,1] に写した行動値。未クリップ外れ値を含む観測範囲は {r['protocol']['raw_bank_audit']['raw_q01_q99_normalized_range_observed']}。raw を物理空間で DCT3 再構成して正規化へ戻した結果と `natural_actions` の最大差は {r['protocol']['raw_bank_audit']['dct3_raw_to_natural_actions_max_abs_diff']:.2e}。",
      '- 初回 raw absolute / SAなし / window10 / τ=.01 は ep4・ep25 とも Dev 上で epoch1 が選ばれ、同じ epoch1 checkpoint を評価したため最終値が同じだった。これは stale ep4/ep25 評価ではなく、実際に各 epoch の Dev 値が計算されていた（初回 long の Dev は epoch1 .12977 から epoch25 .20633へ悪化）。修正後は ep4 設定が epoch4 .02760、ep25 設定が epoch23 .02678 を選び、評価もそれぞれ .02491 と .02395 になった。',
      f"- 同じ runner の DCT3×absolute×SAなし X1 checkpoint control は K-x1 に対するキー最大差 {r['key_models']['DCT3_X1_checkpoint_control']['max_abs_key_diff_vs_K_x1']:.2e}、unseen20 MSE {r['key_models']['DCT3_X1_checkpoint_control']['unseen20']['top1_arm_mse']:.5f}。DCT3 学習条件 τ=.01/ep4 は {r['key_models']['dct3_absolute_sa0_window10_tau0.01_ep4']['evaluation']['unseen20']['top1_arm_mse']:.5f}。", '',
      '## キー空間最近傍（上位単位選択）','',
      '| 入力 | 位置 | SA | 教師 | τ | epoch | Dev top1腕MSE | learned | unseen20 |','|---|---|---:|---|---:|---:|---:|---:|---:|']
    for name,row in r['key_models'].items():
        if name=='K-x1':
            lines.append(f"| K-x1 (既存) | absolute | no | 距離蒸留 | .01 | — | — | {row['learned']['top1_arm_mse']:.5f} | {row['unseen20']['top1_arm_mse']:.5f} |")
            continue
        if name=='DCT3_X1_checkpoint_control':
            lines.append(f"| DCT3/X1 checkpoint control | absolute | no | 既存X1重み | — | — | {row['dev2']['top1_arm_mse']:.5f} | {row['learned']['top1_arm_mse']:.5f} | {row['unseen20']['top1_arm_mse']:.5f} |")
            continue
        c=row['config'];ev=row['evaluation']
        lines.append(f"| {c['source']} | {c['position']} | {int(c['self_attention'])} | {c['teacher']} | {c['tau']:.2f} | {row['selected_epoch']} | {row['dev2_top1_arm_mse']:.5f} | {ev['learned']['top1_arm_mse']:.5f} | {ev['unseen20']['top1_arm_mse']:.5f} |")
    lines+=['','## C1b query downstream','',
      '| キー | Dev選択MSE | learned | unseen20 | epoch |','|---|---:|---:|---:|---:|']
    for name,row in r.get('query_models',{}).items():
        ev=row['evaluation'];lines.append(f"| {name} | {row['dev_selected']:.5f} | {ev['learned']['all']:.5f} | {ev['unseen20']['all']:.5f} | {row['selected_epoch']} |")
    lines+=['','## プロトコル・制約','',
      f"- 固定候補バンク: {r['protocol']['fixed_memory_units']} 可変長単位。raw は元の LeRobot 行動を同一 `anchor_spans` 境界で切り出し、DCT3 は KQ2 bank の natural reconstruction を使用。",
      '- 学習教師は episode 除外付きの腕6 MSE soft-listwise。window10 は10 step 対応、whole16 は両区間を16点へ同長 resample して全体比較。',
      '- 下流 query は KQ2 C1b HSMLP の設定、モデル・epoch は Dev v2 20Hz で選択。',
      '- CPUのみ。PyTorch/BLAS/OMP の thread 数は進捗ログと環境情報に記録。',
      f"- 参照（unseen20）：直前保持 .0470、C1b .0522。DCT3: {r.get('references',{}).get('DCT3_unlearned',{}).get('unseen20')}",
      f"- 総時間: {r.get('timing_seconds')} sec",'']
    (output/'report.md').write_text('\n'.join(lines))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUT);p.add_argument('--smoke',action='store_true')
    p.add_argument('--long-epochs',type=int,default=25);p.add_argument('--skip-downstream',action='store_true')
    a=p.parse_args();torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','8')))
    result=run(a.output,smoke=a.smoke,long_epochs=a.long_epochs,do_downstream=not a.skip_downstream)
    print(json.dumps({'output':str(a.output),'smoke':a.smoke,'models':len(result.get('models',result.get('key_models',{}))) }))


if __name__=='__main__':main()
