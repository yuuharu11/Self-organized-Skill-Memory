#!/usr/bin/env python3
"""RCX6 heads over the frozen E-VLM+H representation.

Only the small grip and error heads are fitted here.  The selected E-VLM+H
retriever, its unit bank, and the bridge Expert are read-only inputs.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import latent_key as lk  # noqa: E402
import latent_skill as ls  # noqa: E402
from confidence_gripper import t0_from_error  # noqa: E402

OUT = Path('/work/.skillnet-state/outputs/robocasa-memory/latent-skill-01')
DEST = OUT / 'latent-heads'
SEED = 6117


def future_grip(rows, manifest, out=OUT):
    """Read causal next-50 grip labels; callers must pass the intended split."""
    memory = {(int(d['task_id']), int(d['episode'])) for d in manifest['demos'] if d['split'] == 'memory'}
    ys = np.empty((len(rows['task']), 50), np.float32)
    cache={}
    for i, (t, e, f) in enumerate(zip(rows['task'], rows['episode'], rows['frame'])):
        key = (int(t), int(e))
        if key not in memory and rows.get('_split') == 'train':
            raise AssertionError(f'non-memory label in train: {key}')
        if key not in cache:
            cache[key]=np.load(Path(out) / 'episodes' / f'task{key[0]}_ep{key[1]:06d}.npz')
        ep=cache[key]
        ix = np.flatnonzero(ep['frame'] == int(f))
        if len(ix) != 1:
            raise ValueError(f'row key does not resolve uniquely: {key}/{f}')
        ix = np.minimum(np.arange(int(ix[0]), int(ix[0]) + 50), len(ep['action']) - 1)
        ys[i] = np.where(ep['action'][ix, 11] > 0, 1., -1.)
        if i < 32:
            assert np.array_equal(ys[i, :10], np.where(rows['target'][i, :, 6] > 0, 1., -1.))
    return ys


def transition_metrics(pred, target, tolerance=2):
    """Sign-switch recall/miss and nearest switch-time error over paired rows."""
    pred = np.asarray(pred) > 0; target = np.asarray(target) > 0
    errors, missed, ntrue, npred = [], 0, 0, 0
    for p, y in zip(pred, target):
        pt = np.flatnonzero(p[1:] != p[:-1]) + 1
        yt = np.flatnonzero(y[1:] != y[:-1]) + 1
        ntrue += len(yt); npred += len(pt)
        for t in yt:
            if len(pt):
                errors.append(int(np.min(np.abs(pt - t))))
            else:
                errors.append(len(y)); missed += 1
            if not len(pt) or np.min(np.abs(pt - t)) > tolerance:
                missed += 0 if (not len(pt)) else 1
    return {'true_switches': int(ntrue), 'pred_switches': int(npred),
            'switch_recall_within_2': float(1 - missed / max(ntrue, 1)),
            'mean_nearest_time_error_steps': float(np.mean(errors)) if errors else None,
            'windows_with_switch': int(np.sum(np.any(target[:, 1:] != target[:, :-1], axis=1)))}


def _json(path, obj):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=lambda x: x.item() if hasattr(x, 'item') else str(x)) + '\n')


def _load_e(mode='e_vlm_h', device='cpu'):
    import torch
    sel = json.loads((OUT / 'key/selected_configs.json').read_text())['selected'][mode]
    inp = lk.Inputs(OUT, mode, bank_device=None)
    model = lk.new_model(mode).to(device)
    model.load_state_dict(torch.load(OUT / 'key/runs' / sel['run'] / 'model.pt', map_location=device, weights_only=False))
    model.eval()
    return inp, model, sel


def _representation(inp, model, rows, device, split, chunk=64, exclude_self=False):
    """Return q tokens, selected unit ids/key tokens, top-10 scores and full top1 scores."""
    import torch
    bank = inp.bank_in
    with torch.no_grad():
        keys = torch.cat([lk.encode(model, inp.batch(bank, np.arange(s, min(s + chunk, len(bank['task']))), device), key=True)
                          for s in range(0, len(bank['task']), chunk)])
        qlist=[]; ids=[]; top10=[]; top10s=[]; top1s=[]
        for s in range(0, len(rows['task']), chunk):
            ix=np.arange(s,min(s+chunk,len(rows['task'])))
            q=lk.encode(model, inp.batch(rows, ix, device), key=False)
            scores=lk.token_scores(q, keys).float()
            if exclude_self:
                same=(torch.as_tensor(rows['task'][ix],device=device)[:,None] == torch.as_tensor(bank['task'],device=device)[None]) & (torch.as_tensor(rows['episode'][ix],device=device)[:,None] == torch.as_tensor(bank['episode'],device=device)[None])
                scores=scores.masked_fill(same,-1e9)
            ti=torch.topk(scores, k=10, dim=1)
            qlist.append(q.cpu().numpy()); ids.append(ti.indices[:,0].cpu().numpy())
            top10.append(ti.indices.cpu().numpy()); top10s.append(ti.values.cpu().numpy()); top1s.append(scores.max(1).values.cpu().numpy())
    q=np.concatenate(qlist); ids=np.concatenate(ids); t10=np.concatenate(top10); s10=np.concatenate(top10s)
    return {'q':q.astype(np.float32),'top1':ids.astype(np.int32),'key':keys[torch.as_tensor(ids,device=keys.device)].cpu().numpy().astype(np.float32),
            'top10':t10,'scores10':s10,'score1':np.concatenate(top1s).astype(np.float32),'bank_keys':keys.cpu().numpy().astype(np.float32)}


def _p5_scores(rows, device='cpu', chunk=64):
    import torch
    inp, model, sel = _load_e('p5', device)
    bank=inp.bank_in
    with torch.no_grad():
        keys=torch.cat([lk.encode(model,inp.batch(bank,np.arange(s,min(s+chunk,len(bank['task']))),device),key=True)
                        for s in range(0,len(bank['task']),chunk)])
        out=[]
        for s in range(0,len(rows['task']),chunk):
            ix=np.arange(s,min(s+chunk,len(rows['task'])))
            q=lk.encode(model,inp.batch(rows,ix,device),key=False)
            sc=lk.token_scores(q,keys).float()
            if rows.get('_split')=='train':
                task=torch.as_tensor(rows['task'][ix],device=device); ep=torch.as_tensor(rows['episode'][ix],device=device)
                same=(task[:,None]==torch.as_tensor(bank['task'],device=device)[None])&(ep[:,None]==torch.as_tensor(bank['episode'],device=device)[None])
                sc=sc.masked_fill(same,-1e9)
            out.append(sc.cpu().numpy())
    return np.concatenate(out), model, inp


def build_signals(rows, repr_e, scores_p5, inp_e):
    """RCX3-A's seven runtime-only signals, computed from cached ranking results."""
    bank=inp_e.bank; sel=repr_e['top1']; ids=repr_e['top10']; ss=repr_e['scores10']
    # Scores from the E model are token cosines. Recover its trained logit scale from checkpoint.
    import torch
    e_run=json.loads((OUT/'key/selected_configs.json').read_text())['selected']['e_vlm_h']['run']
    ck=torch.load(OUT/'key/runs'/e_run/'model.pt',map_location='cpu',weights_only=False)
    scale=float(np.exp(min(float(ck['logit_scale']),np.log(lk.LOGIT_SCALE_MAX))))
    coeff=np.asarray(bank['coeff'],np.float32)
    spread=np.sqrt(np.mean((coeff[ids]-coeff[sel,None])**2,axis=(1,2,3)))
    p5top=scores_p5.argmax(1)
    cross=np.sqrt(np.mean((coeff[sel]-coeff[p5top])**2,axis=(1,2)))
    jump=np.zeros(len(sel),np.float32)
    hist=rows['history']
    for i in range(len(sel)):
        valid=np.flatnonzero(rows['history_mask'][i])
        last=hist[i,valid[-1],:6] if len(valid) else np.zeros(6,np.float32)
        jump[i]=np.sqrt(np.mean((bank['window10'][sel[i],0,:6]-last)**2))
    repeat=np.zeros(len(sel),np.float32)
    for i in range(1,len(sel)):
        repeat[i]=float(rows['task'][i]==rows['task'][i-1] and rows['episode'][i]==rows['episode'][i-1] and sel[i]==sel[i-1])
    prob=np.exp((ss*scale)-np.max(ss*scale,axis=1,keepdims=True)); prob/=prob.sum(1,keepdims=True)
    ent=-(prob*np.log(np.maximum(prob,1e-30))).sum(1)
    gap=ss[:,0]-ss[:,1]
    return np.stack([repr_e['score1'],gap,ent,spread,cross,jump,repeat],axis=1).astype(np.float32)


def extract(device='cpu'):
    """Extract fixed-E representations and top-1/top-10 IDs for train/dev/eval."""
    import torch
    if str(device).startswith('cuda') and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(.25)
    dest=DEST/'features'; dest.mkdir(parents=True,exist_ok=True)
    if (dest/'train.npz').exists() and (dest/'dev.npz').exists() and (dest/'eval.npz').exists():
        return {'reused':True}
    torch.set_num_threads(8)
    inp, model, sel=_load_e('e_vlm_h',device)
    splits={s:dict(np.load(OUT/f'rows/{s}.npz')) for s in ('train','dev','eval')}
    for s in splits.values(): s['_split']=s is splits['train'] and 'train' or ('dev' if s is splits['dev'] else 'eval')
    p5_cache={}
    for name,rows in splits.items():
        r=_representation(inp,model,rows,device,name,exclude_self=name=='train')
        if name in ('dev','eval'):
            ref=np.load(OUT/f'key/eval/e_vlm_h_{name}_top1.npy')
            if not np.array_equal(r['top1'],ref):
                raise AssertionError(f'frozen E top-1 changed on {name}; refusing misaligned head labels')
        p5sc,_,_=_p5_scores(rows,device)
        sig=build_signals(rows,r,p5sc,inp)
        np.savez_compressed(dest/f'{name}.npz',q=r['q'],key=r['key'],top1=r['top1'],top10=r['top10'],scores10=r['scores10'],score1=r['score1'],signals7=sig,
                            task=rows['task'],episode=rows['episode'],frame=rows['frame'],target=rows['target'],history=rows['history'],history_mask=rows['history_mask'],state=rows['state'])
        print(json.dumps({'split':name,'rows':len(rows['task']),'feature_shape':list(r['q'].shape),'top1_checksum':int(r['top1'].sum())}),flush=True)
    _json(dest/'manifest.json',{'frozen_mode':'e_vlm_h','selected_run':sel['run'],'representation':'query/key E tokens [5,256], shared frozen encoder','candidate_selection':'top-1; same episode excluded in train only','train_rows':len(splits['train']['task']),'dev_rows':len(splits['dev']['task']),'eval_rows':len(splits['eval']['task'])})
    return {'reused':False,'selected_run':sel['run']}


class ErrorHead:
    @staticmethod
    def make(dim,hidden=256):
        import torch
        from torch import nn
        return nn.Sequential(nn.Linear(dim,hidden),nn.GELU(),nn.Linear(hidden,hidden//2),nn.GELU(),nn.Linear(hidden//2,1))


class GripHead:
    @staticmethod
    def make(dim=1280,hidden=128):
        from confidence_gripper import make_gripper_model
        return make_gripper_model(dim,hidden)


def fit():
    import torch
    from scipy.stats import spearmanr
    torch.set_num_threads(8)
    dest=DEST; feat=dest/'features'
    if (dest/'heads.pt').exists(): raise FileExistsError(f'head checkpoint exists: {dest}/heads.pt')
    manifest=json.loads((OUT/'manifest.json').read_text())
    train_rows=dict(np.load(OUT/'rows/train.npz')); train_rows['_split']='train'
    dev_rows=dict(np.load(OUT/'rows/dev.npz')); dev_rows['_split']='dev'
    eval_rows=dict(np.load(OUT/'rows/eval.npz')); eval_rows['_split']='eval'
    fx={s:dict(np.load(feat/f'{s}.npz')) for s in ('train','dev','eval')}
    mem={(int(d['task_id']),int(d['episode'])) for d in manifest['demos'] if d['split']=='memory'}
    assert all((int(t),int(e)) in mem for t,e in zip(train_rows['task'],train_rows['episode']))
    # Grip labels are queried only for train and Dev here; Eval is deferred to evaluate().
    yt=future_grip(train_rows,manifest); yd=future_grip(dev_rows,manifest)
    xt=np.asarray(fx['train']['q']).reshape(len(yt),-1); xd=np.asarray(fx['dev']['q']).reshape(len(yd),-1)
    gc=xt.mean(0); gs=np.maximum(xt.std(0),1e-4); xt=((xt-gc)/gs).astype(np.float32); xd=((xd-gc)/gs).astype(np.float32)
    candidates=[]; states=[]
    pos=float((yt>0).mean()); pw=torch.tensor([(1-pos)/max(pos,1e-4)])
    for j,(hidden,lr,epochs) in enumerate([(64,1e-3,8),(128,1e-3,10)]):
        torch.manual_seed(SEED+j); rng=np.random.default_rng(SEED+j)
        model=GripHead.make(xt.shape[1],hidden); opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
        lossfn=torch.nn.BCEWithLogitsLoss(pos_weight=pw); X=torch.from_numpy(xt); Y=torch.from_numpy((yt>0).astype(np.float32))
        for ep in range(epochs):
            for st in range(0,len(X),128):
                ix=rng.permutation(len(X))[st:st+128]; pred=model(X[ix]); loss=lossfn(pred,Y[ix]); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        with torch.no_grad(): logits=model(torch.from_numpy(xd)).numpy()
        pred=np.where(logits>=0,1.,-1.); bacc=[]
        for k in range(50):
            y=yd[:,k]>0; p=pred[:,k]>0
            tpr=p[y].mean() if y.any() else 0.
            tnr=(~p[~y]).mean() if (~y).any() else 0.
            bacc.append(.5*(tpr+tnr))
        met=transition_metrics(pred,yd)
        score=float(np.mean(bacc)); candidates.append({'hidden':hidden,'lr':lr,'epochs':epochs,'dev_balanced_accuracy50':score,**met})
        states.append({k:v.detach().cpu() for k,v in model.state_dict().items()})
    gi=max(range(len(candidates)),key=lambda i:(candidates[i]['switch_recall_within_2'],candidates[i]['dev_balanced_accuracy50']))

    # Error training target: log arm6 MSE between selected stored unit and the true next 10 actions.
    bank=dict(np.load(OUT/'units/memory_units.npz')); top=np.asarray(fx['train']['top1'],int)
    actual=np.asarray(train_rows['target'],np.float32); drafts=np.asarray(bank['window10'][top],np.float32)
    err=np.maximum(((drafts[:,:,:6]-actual[:,:,:6])**2).sum((1,2))/(10*7),1e-8); logerr=np.log(err).astype(np.float32)
    ye=np.asarray(fx['dev']['target'],np.float32); dd=np.asarray(bank['window10'][np.asarray(fx['dev']['top1'],int)],np.float32)
    deverr=np.maximum(((dd[:,:,:6]-ye[:,:,:6])**2).sum((1,2))/(10*7),1e-8); devlog=np.log(deverr).astype(np.float32)
    qtr=np.asarray(fx['train']['q'],np.float32); ktr=np.asarray(fx['train']['key'],np.float32); qdv=np.asarray(fx['dev']['q'],np.float32); kdv=np.asarray(fx['dev']['key'],np.float32)
    def pair(q,k): return np.concatenate([q.reshape(len(q),-1),k.reshape(len(k),-1),(q-k).reshape(len(q),-1),(q*k).reshape(len(q),-1)],1).astype(np.float32)
    base_t=pair(qtr,ktr); base_d=pair(qdv,kdv)
    sigtr=np.asarray(fx['train']['signals7'],np.float32); sigdv=np.asarray(fx['dev']['signals7'],np.float32)
    err_results={}; err_states={}
    for name,aa,bb in [('latent',base_t,base_d),('latent_plus_7',np.concatenate([base_t,sigtr],1),np.concatenate([base_d,sigdv],1))]:
        center=aa.mean(0); scale=np.maximum(aa.std(0),1e-4); aa=((aa-center)/scale).astype(np.float32); bb=((bb-center)/scale).astype(np.float32)
        torch.manual_seed(SEED); model=ErrorHead.make(aa.shape[1]); opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-4)
        X=torch.from_numpy(aa); Y=torch.from_numpy(logerr[:,None]); rng=np.random.default_rng(SEED)
        for ep in range(40):
            order=rng.permutation(len(X));
            for st in range(0,len(X),256):
                ix=order[st:st+256]; pred=model(X[ix]); loss=torch.nn.functional.smooth_l1_loss(pred,Y[ix]); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        with torch.no_grad(): pred=model(torch.from_numpy(bb)).numpy().ravel()
        rho=float(spearmanr(pred,devlog).statistic); rmse=float(np.sqrt(np.mean((pred-devlog)**2)))
        err_results[name]={'dev_spearman_log_error':rho,'dev_log_rmse':rmse,'features_dim':int(aa.shape[1])}
        err_states[name]={'state':{k:v.detach().cpu() for k,v in model.state_dict().items()},'center':center.astype(np.float32),'scale':scale.astype(np.float32),'input_dim':int(aa.shape[1])}
    selected_error=max(err_results,key=lambda n:(err_results[n]['dev_spearman_log_error'],-err_results[n]['dev_log_rmse']))
    ck={'seed':SEED,'grip':{'state':states[gi],'hidden':candidates[gi]['hidden'],'input_dim':1280,'center':gc.astype(np.float32),'scale':gs.astype(np.float32),'candidates':candidates,'selected':candidates[gi]},
        'error':{**err_states[selected_error],'selected':selected_error,'candidates':err_results},
        'tail_values':[-1.,0.,0.,0.,0.,-1.], 't0_quantiles':json.loads((OUT/'expert/train/key-agnostic-01-retry-fraction035/config.json').read_text())['t0_map_quantiles'],
        't0_range':[.1,.9]}
    torch.save(ck,dest/'heads.pt')
    _json(dest/'selection.json',{'selection_uses':'Dev only','gripper_candidates':candidates,'gripper_selected':candidates[gi],
                                  'error_candidates':err_results,'error_selected':selected_error,
                                  'error_target':'log retrieval-draft arm6 squared-error sum / 70 over first 10 steps',
                                  'fallback_thresholds_dev':{str(rate):float(np.quantile(np.exp(pred),1-rate/100)) for rate in (0,10,20,30)},
                                  'default_fallback_rate':20,'t0_map':'Expert training config t0_map_quantiles, empirical CDF mapped to [0.1,0.9]'})
    return ck


def _head_predict(ck, f):
    import torch
    errc=ck['error']; q=np.asarray(f['q'],np.float32); k=np.asarray(f['key'],np.float32)
    x=np.concatenate([q.reshape(len(q),-1),k.reshape(len(k),-1),(q-k).reshape(len(q),-1),(q*k).reshape(len(q),-1)],1)
    if ck['error']['selected']=='latent_plus_7': x=np.concatenate([x,np.asarray(f['signals7'],np.float32)],1)
    x=(x-errc['center'])/errc['scale']; em=ErrorHead.make(errc['input_dim']); em.load_state_dict(errc['state']); em.eval()
    gm=GripHead.make(1280,ck['grip']['hidden']); gm.load_state_dict(ck['grip']['state']); gm.eval()
    gx=np.asarray(f['q'],np.float32).reshape(len(q),-1); gx=(gx-ck['grip']['center'])/ck['grip']['scale']
    with torch.no_grad(): ep=em(torch.from_numpy(x.astype(np.float32))).numpy().ravel(); gl=gm(torch.from_numpy(gx.astype(np.float32))).numpy()
    return np.exp(np.clip(ep,-20,20)), np.where(gl>=0,1.,-1.).astype(np.float32)


def evaluate():
    import torch
    torch.set_num_threads(8)
    if (DEST/'results.json').exists(): raise FileExistsError('Eval has already been written')
    ck=torch.load(DEST/'heads.pt',map_location='cpu',weights_only=False)
    cfg=json.loads((OUT/'expert/train/key-agnostic-01-retry-fraction035/config.json').read_text())
    qtls=np.asarray(cfg['t0_map_quantiles'],np.float64)
    result={'default_rate':20,'splits':{},'t0_dev_candidates':{},'limits':'Eval evaluated once; b and head chosen from Dev only.'}
    devf=dict(np.load(DEST/'features/dev.npz')); dev_error,dev_grip=_head_predict(ck,devf)
    for rate in (0,10,20,30):
        threshold=float(np.quantile(dev_error,1-rate/100)) if rate else float(np.inf)
        d0=np.asarray(t0_from_error(dev_error,qtls)); active=dev_error<threshold
        result['t0_dev_candidates'][str(rate)]={'threshold':threshold,'fallback_rate':float((~active).mean()),'t0_quantiles':np.quantile(d0[active],[0,.1,.25,.5,.75,.9,1]).tolist() if active.any() else [],'mean_t0':float(d0[active].mean()) if active.any() else None}
    for split in ('dev','eval'):
        f=dict(np.load(DEST/f'features/{split}.npz')); rows=dict(np.load(OUT/f'rows/{split}.npz')); ehat,grip=_head_predict(ck,f)
        threshold=float(np.quantile(dev_error,.8)); t0=np.asarray(t0_from_error(ehat,qtls)); fallback=ehat>=threshold
        targ=np.asarray(rows['target'],np.float32); ygr=future_grip(rows,json.loads((OUT/'manifest.json').read_text()))[:,:10]
        first=np.load(OUT/f'expert/eval/{split}/first10.npz'); per=np.load(OUT/f'expert/eval/{split}/per_row.npz')
        pred=np.empty_like(targ)
        tgrid=np.asarray(per['t0_grid']); grid=np.round(tgrid,1)
        for i,v in enumerate(t0):
            j=int(np.argmin(np.abs(grid-v))); pred[i]=first[f'expert__e_vlm_h__{grid[j]:.1f}'][i]
        pi=np.asarray(first['pi05'],np.float32)
        pred[fallback]=pi[fallback]
        # Replace predicted gripper channel only on the E+Expert path; π0.5 fallback is untouched.
        nonfb=~fallback; pred[nonfb,:,6]=grip[nonfb,:10]
        def arm_mse(a): return ((a[:,:,:6]-targ[:,:,:6])**2).sum((1,2))/(10*7)
        ysign=np.where(ygr>0,1.,-1.)
        head10=grip[:,:10]; hold=np.repeat(np.where(rows['history'][:,-1,6,None]>0,1.,-1.),10,axis=1)
        saved=np.where(np.load(OUT/'units/memory_units.npz')['natural_actions'][f['top1'],:10,6]>0,1.,-1.)
        pi_g=np.where(pi[:,:,6]>0,1.,-1.)
        metrics={}
        for nm,aa in [('latent_t0_griphead',pred),('draft',np.asarray(np.load(OUT/f'expert/drafts/{split}_e_vlm_h.npz')['draft'][:,:10],np.float32)),('pi05',pi)]:
            m=arm_mse(aa); metrics[nm]={'arm6_mse_mean':float(m.mean()),'arm6_mse_median':float(np.median(m))}
        for nm,pp in [('latent_griphead',head10),('hold_last',hold),('saved_unit',saved),('pi05',pi_g)]:
            metrics[nm+'_grip']={'sign_accuracy_first10':float(np.mean(pp==ysign)),'transition_first10':transition_metrics(pp,ysign)}
        # Existing dist selector is a reference, using the already-saved π0.5 distance.
        selector=json.loads((OUT/'closed-loop/selector_e_vlm_h.json').read_text()); W=np.asarray(selector['W'],np.float32); tg=np.asarray(selector['t0_grid'],np.float32)
        dist=np.asarray(per['draft__e_vlm_h__dist_pi05'])
        # Dist selector stores one W row per t0: choose minimum predicted arm error.
        scores=np.stack([np.array([1.,np.log(max(x,1e-6))],np.float32)@W for x in dist])
        di=scores.argmin(1); de=np.asarray([first[f'expert__e_vlm_h__{tg[j]:.1f}'][i] for i,j in enumerate(di)])
        metrics['dist_expert']={'arm6_mse_mean':float(arm_mse(de).mean())}
        result['splits'][split]={'n':len(targ),'threshold_b20':threshold,'fallback_rate':float(fallback.mean()),'mean_t0_nonfallback':float(t0[~fallback].mean()) if (~fallback).any() else None,
                                  't0_quantiles_nonfallback':np.quantile(t0[~fallback],[0,.1,.25,.5,.75,.9,1]).tolist() if (~fallback).any() else [],'metrics':metrics}
        if split=='dev': np.savez_compressed(DEST/'dev_predictions.npz',ehat=ehat,t0=t0,fallback=fallback,grip=grip)
        else: np.savez_compressed(DEST/'eval_predictions.npz',ehat=ehat,t0=t0,fallback=fallback,grip=grip,actions=pred)
    _json(DEST/'results.json',result)
    return result


def dev_threshold_sweep():
    """Dev-only b calibration table, including arm error and t0 distribution."""
    import torch
    torch.set_num_threads(8)
    ck=torch.load(DEST/'heads.pt',map_location='cpu',weights_only=False)
    f=dict(np.load(DEST/'features/dev.npz')); rows=dict(np.load(OUT/'rows/dev.npz'))
    err,_=_head_predict(ck,f); qtls=np.asarray(ck['t0_quantiles'],np.float64)
    first=np.load(OUT/'expert/eval/dev/first10.npz'); grid=np.asarray(np.load(OUT/'expert/eval/dev/per_row.npz')['t0_grid'])
    target=np.asarray(rows['target'],np.float32); pi=np.asarray(first['pi05'],np.float32)
    report={}
    for rate in (0,10,20,30):
        b=float(np.quantile(err,1-rate/100)) if rate else np.inf
        t0=np.asarray(t0_from_error(err,qtls)); fb=err>=b; pred=np.empty_like(target)
        for i,v in enumerate(t0):
            j=int(np.argmin(np.abs(grid-v))); pred[i]=first[f'expert__e_vlm_h__{grid[j]:.1f}'][i]
        pred[fb]=pi[fb]
        mse=((pred[:,:,:6]-target[:,:,:6])**2).sum((1,2))/(10*7)
        report[str(rate)]={'threshold_b':float(b),'fallback_rate':float(fb.mean()),'arm6_mse':float(mse.mean()),
                           'arm6_mse_median':float(np.median(mse)), 'mean_t0_nonfallback':float(t0[~fb].mean()) if (~fb).any() else None,
                           't0_quantiles_nonfallback':np.quantile(t0[~fb],[0,.1,.25,.5,.75,.9,1]).tolist() if (~fb).any() else []}
    _json(DEST/'dev_threshold_sweep.json',report)
    return report


def report():
    r=json.loads((DEST/'results.json').read_text()); s=r['splits']; lines=['# RCX6：潜在トークンのグリッパー・t0 ヘッド','',
    '## 主結果','', '| split | 新ヘッド arm6 MSE | Expert dist arm6 MSE | 下書き arm6 MSE | π0.5 arm6 MSE | fallback | 平均 t0 | grip accuracy | switch recall ±2 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for sp in ('dev','eval'):
        d=s[sp]; m=d['metrics']; h=m['latent_griphead_grip']; lines.append(f"| {sp} | {m['latent_t0_griphead']['arm6_mse_mean']:.4f} | {m['dist_expert']['arm6_mse_mean']:.4f} | {m['draft']['arm6_mse_mean']:.4f} | {m['pi05']['arm6_mse_mean']:.4f} | {d['fallback_rate']:.1%} | {d['mean_t0_nonfallback']:.3f} | {h['sign_accuracy_first10']:.3f} | {h['transition_first10']['switch_recall_within_2']:.3f} |")
    lines += ['', '## Dev 選択・較正','',f"誤差ヘッド選択：`{json.loads((DEST/'selection.json').read_text())['error_selected']}`。グリッパーの Dev 選択と候補比較は `selection.json` に記録。フォールバック閾値は Dev 推定誤差の80百分位（既定20%）。t0 は Expert 訓練 `t0_map_quantiles` の経験CDFから [0.1,0.9] に写像。",'', '## グリッパー比較','']
    for sp in ('dev','eval'):
        for nm,v in s[sp]['metrics'].items():
            if nm.endswith('_grip'): lines.append(f"- {sp} {nm}: sign accuracy {v['sign_accuracy_first10']:.4f}; transitions {v['transition_first10']}")
    lines += ['', '## 条件・制限','', 'E-VLM+H と Expert の重みは既存チェックポイントを読み取り専用で使用。新規学習はグリッパー・誤差ヘッドのみ。π0.5 出力、正解、タスクIDはヘッド入力に含めていない。Dev の正解はモデル選択・閾値の決定だけ、Eval は選択確定後の一度だけ採点に使用した。7〜12ch は grip/base0..3/mode 順。RCX5 の学習デモ集計に従い非fallback時の tail は [-1,0,0,0,0,-1]。', '', '閉ループ関門・本番実行は別ステップで同じ設定を使用する。']
    (DEST/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('stage',choices=['extract','fit','evaluate','dev_thresholds','report','all']); p.add_argument('--device',default='cuda')
    a=p.parse_args(); DEST.mkdir(parents=True,exist_ok=True)
    if a.stage in ('extract','all'): extract(a.device)
    if a.stage in ('fit','all'): fit()
    if a.stage in ('evaluate','all'): evaluate()
    if a.stage=='dev_thresholds': dev_threshold_sweep()
    if a.stage in ('report','all'): report()

if __name__=='__main__': main()
