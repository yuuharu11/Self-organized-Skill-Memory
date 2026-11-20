#!/usr/bin/env python3
"""XF2: KC2-consistent retrieval drafts, confidence-calibrated flow Expert, and CL1 runner.

CPU preparation is restartable: `prepare` refits the selected KC2 query on five episode folds,
scores heldout rows, and builds crossfit isotonic error estimates. `train` fits one KQ2-CL K1 Expert
to those retrieved drafts. `offline` selects variable/fixed t0 on Dev v2 only; `closed` runs both
strategies with the KC2 Gdec and fallback protocol.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import kc1_run as K1
import kc2_run as K2
import kq2cl_common as C

STATE_DIR = Path(os.environ.get('SKILLNET_STATE_DIR', Path(__file__).resolve().parents[5] / '.skillnet-state')).expanduser()
ROOT = STATE_DIR / 'outputs/memory-pilot/unseen-libero'
OUT = ROOT / 'xf2-01'
XF1 = ROOT / 'xf1-01'
KC2 = ROOT / 'kc2-01'
KFOLDS = 5
EPOCHS = 16
TAU = .03
LAMBDA = 1.
SEED = 3
T0_K = (.5, 1., 2.)
T0_FIXED = (.15, .2, .25, .3, .35, .4)
NOISE_OFFSET = {'learned': 0, 'unseen20': 50_000_000, 'dev2': 60_000_000}


def log(event, **fields):
    OUT.mkdir(parents=True, exist_ok=True)
    row = {'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'event': event, **fields}
    with (OUT / 'progress.log').open('a') as f:
        f.write(json.dumps(row, ensure_ascii=False, default=float) + '\n')
    print(json.dumps(row, ensure_ascii=False, default=float), flush=True)


def fold_partition(query_task, query_episode, memory_task, memory_episode, assignment, fold):
    """Return fit/heldout rows and fold memory IDs, asserting episode isolation."""
    qfold = np.asarray([assignment[(int(t), int(e))] for t, e in zip(query_task, query_episode)], np.int8)
    mfold = np.asarray([assignment[(int(t), int(e))] for t, e in zip(memory_task, memory_episode)], np.int8)
    fit = np.flatnonzero(qfold != fold)
    held = np.flatnonzero(qfold == fold)
    mem = np.flatnonzero(mfold != fold)
    held_pairs = set(zip(np.asarray(query_task)[held].astype(int), np.asarray(query_episode)[held].astype(int)))
    fit_pairs = set(zip(np.asarray(query_task)[fit].astype(int), np.asarray(query_episode)[fit].astype(int)))
    mem_pairs = set(zip(np.asarray(memory_task)[mem].astype(int), np.asarray(memory_episode)[mem].astype(int)))
    if held_pairs & fit_pairs or held_pairs & mem_pairs:
        raise AssertionError(f'fold {fold}: heldout episode leaked into query fit or memory')
    return fit, held, mem


def fit_oof_isotonic(spread, error, folds):
    """Crossfit E[error|spread]. Each fold's predictions use other folds only."""
    x = np.asarray(spread, np.float64).reshape(-1)
    y = np.asarray(error, np.float64).reshape(-1)
    f = np.asarray(folds, np.int64).reshape(-1)
    if x.shape != y.shape or x.shape != f.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('invalid isotonic arrays')
    pred = np.empty_like(y)
    models = {}
    for held in np.unique(f):
        tr = f != held
        te = ~tr
        if not tr.any() or not te.any():
            raise ValueError(f'fold {held} has empty train or heldout set')
        models[int(held)] = IsotonicRegression(out_of_bounds='clip').fit(x[tr], y[tr])
        pred[te] = models[int(held)].predict(x[te])
    full = IsotonicRegression(out_of_bounds='clip').fit(x, y)
    return pred.astype(np.float32), full


def variable_t0(ehat, k=1.):
    e = np.maximum(np.asarray(ehat, np.float64), 0.)
    return np.clip(float(k) * np.sqrt(e) / (1. + float(k) * np.sqrt(e)), .05, .6).astype(np.float32)


def calibration_threshold(values, fallback_rate=.204, spread=None):
    """Calibrate target fallback fraction; break isotonic plateaus by spread10."""
    x = np.asarray(values, np.float64)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError('invalid calibration values')
    if spread is None:
        threshold=float(np.quantile(x,1.-fallback_rate,method='higher'));tie=None;mask=x>threshold
    else:
        s=np.asarray(spread,np.float64)
        if s.shape!=x.shape or not np.isfinite(s).all():raise ValueError('invalid spread tie-break array')
        n=max(1,min(len(x),int(round(fallback_rate*len(x)))))
        order=np.lexsort((-s,-x));cut=int(order[n-1]);threshold=float(x[cut]);tie=float(s[cut])
        mask=(x>threshold)|((x==threshold)&(s>=tie))
    return {'threshold':threshold,'spread_tiebreak_threshold':tie,'target_rate':float(fallback_rate),
            'realized_rate':float(np.mean(mask)),'realized_count':int(mask.sum()),
            'tie_rule':'among equal isotonic ehat values, fallback by spread10 descending' if tie is not None else None,
            'rule':'fallback iff ehat > threshold; spread10 breaks isotonic plateaus'}


def fallback_mask(ehat,spread,calibration):
    e=np.asarray(ehat,np.float64);s=np.asarray(spread,np.float64);thr=float(calibration['threshold'])
    mask=e>thr;tie=calibration.get('spread_tiebreak_threshold')
    if tie is not None:mask|=(e==thr)&(s>=float(tie))
    return mask


def apply_isotonic(meta,values):
    return np.interp(np.asarray(values,np.float64),np.asarray(meta['x'],np.float64),np.asarray(meta['y'],np.float64))


def calibrate() -> dict:
    """Write plateau-aware Dev calibration to a new artifact; preserve prepare outputs."""
    old=json.loads((OUT/'calibration.json').read_text());dev=np.load(OUT/'rows_dev2.npz',allow_pickle=False)
    ehat=apply_isotonic(old['isotonic'],dev['spread10'])
    fb=calibration_threshold(ehat,.204,spread=dev['spread10'])
    result={'isotonic':old['isotonic'],'fallback':fb,'p_b':.204,
      'threshold_source':'Dev v2 ehat quantile; spread10 breaks isotonic plateaus only',
      'source_prepare_calibration':str(OUT/'calibration.json')}
    dest=OUT/'calibration_v2.json'
    if dest.exists():raise FileExistsError(dest)
    dest.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    log('calibrate_done',fallback_threshold=fb['threshold'],spread_tiebreak=fb['spread_tiebreak_threshold'],
        fallback_realized_rate=fb['realized_rate'],fallback_count=fb['realized_count'])
    return result


def _load_kc2_context():
    if not (KC2 / 'DONE').is_file():
        raise RuntimeError(f'KC2 closed loop is not complete: missing {KC2 / "DONE"}')
    return K2.load_ctx()


def _split_fold_assignment(sp, bank):
    manifest = json.loads((XF1 / 'manifest.json').read_text())
    raw = manifest['episode_assignments']
    assignment = {tuple(map(int, key.split(':'))): int(fold) for key, fold in raw.items()}
    query_pairs = set(zip(np.asarray(sp['task']).astype(int), np.asarray(sp['episode']).astype(int)))
    bank_pairs = set(zip(np.asarray(bank['task']).astype(int), np.asarray(bank['episode']).astype(int)))
    if query_pairs != bank_pairs or query_pairs != set(assignment):
        raise AssertionError(f'XF1 fold assignment mismatch query={len(query_pairs)} bank={len(bank_pairs)} assign={len(assignment)}')
    return assignment


def _kc2_assets(bank):
    chosen = json.loads((KC2 / 'offline_results.json').read_text())['selected']
    if (chosen['q_name'], chosen['score_mode'], float(chosen['tau']), float(chosen['lambda'])) != ('i', 'distance', .03, 1.):
        raise AssertionError(f'KC2 selected configuration changed: {chosen}')
    checkpoint = torch.load(chosen['checkpoint'], map_location='cpu', weights_only=False)
    # KC2's checkpoint records the exact standardized coefficient-space moments
    # inherited from the selected KC1 query. K1.OUT is mutated by K2.load_ctx().
    kc1cfg = checkpoint['config']
    coeff = np.asarray(bank['coefficients'], np.float32).reshape(len(bank['length']), 18)
    keys, mean, std = K1.coefficient_keys(coeff, standardize=True,
                                           mean=np.asarray(kc1cfg['mean']), std=np.asarray(kc1cfg['std']))
    return chosen, checkpoint, keys, mean, std


def _model_from_checkpoint(blob, device='cpu'):
    model = K2.KC2Net('i', 'distance').to(device)
    state = blob.get('state_dict', blob.get('model'))
    model.load_state_dict(state)
    model.eval()
    return model


def _score_rows(model, gs, sp, bank, keys_np, row_ids, memory_ids=None, *, batch=128):
    """KC2 negative-squared-distance retrieval, score/top10/spread/gripper and raw draft."""
    device = next(model.parameters()).device
    mids = np.arange(len(bank['length']), dtype=np.int64) if memory_ids is None else np.asarray(memory_ids, np.int64)
    keypart = np.asarray(keys_np[mids], np.float32)
    key_t = torch.as_tensor(keypart, dtype=torch.float32, device=device)
    ids_out=[]; score_out=[]; grip_out=[]; spread_out=[]; draft_out=[]; err_out=[]
    all_actions = np.asarray(bank['value_actions10'], np.float32)
    # KC2 split bundles already carry the aligned history arrays. Using the
    # split-local arrays avoids silently reusing Dev history for other splits.
    hist = np.asarray(sp['history'])
    hmask = np.asarray(sp['history_mask'])
    last = np.asarray(hist[row_ids, 9, 6], np.float32)
    valid = np.asarray(hmask[row_ids]).any(1) & (last != 0)
    prev_sign = np.where(valid, np.sign(last), -1.).astype(np.float32)
    labels = K2._gripper_labels(sp)
    threshold = float(json.loads((KC2 / 'offline_results.json').read_text())['selected']['dev_threshold'])
    model.eval()
    with torch.inference_mode():
        for lo in range(0, len(row_ids), batch):
            rows = np.asarray(row_ids[lo:lo+batch], np.int64)
            ix = torch.as_tensor(rows, dtype=torch.long, device=device)
            q = model(gs, ix)
            logits = model.score(q, key_t).float().cpu().numpy()
            probs = torch.sigmoid(model.grip_logits(q)).float().cpu().numpy()
            allow = K1.candidate_mask(sp['task'][rows], sp['episode'][rows],
                                      np.asarray(bank['task'])[mids], np.asarray(bank['episode'])[mids])
            logits[~allow] = -np.inf
            order = np.argsort(-logits, axis=1, kind='stable')[:, :10]
            top_scores = np.take_along_axis(logits, order, axis=1)
            ks = keypart[order]
            spread = np.var(ks, axis=1).mean(axis=1)
            ids = mids[order]
            gripper = C.decode_gripper(probs, prev_sign[lo:lo+len(rows)], threshold)
            draft = all_actions[ids[:, 0]].copy()
            draft[:, :, 6] = gripper
            err = C.arm6(draft, sp['actions'][rows])
            ids_out.append(ids); score_out.append(top_scores); grip_out.append(probs)
            spread_out.append(spread); draft_out.append(draft); err_out.append(err)
    return {'top_ids': np.concatenate(ids_out), 'top_scores': np.concatenate(score_out).astype(np.float32),
            'spread10': np.concatenate(spread_out).astype(np.float32),
            'gripper_probs': np.concatenate(grip_out).astype(np.float32),
            'draft': np.concatenate(draft_out).astype(np.float32),
            'e_arm': np.concatenate(err_out).astype(np.float32)}


def _train_kc2_fold(fold, kr, ctx, targets, bank, assignment, keys_np, mean, std):
    sp = ctx.splits['eh']; outdir = OUT / 'query-folds'; outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f'fold{fold}_epoch16.pt'
    rowpath = OUT / f'rows_fold{fold}.npz'
    if rowpath.exists() and outpath.exists():
        with np.load(rowpath, allow_pickle=False) as z: return {k:z[k] for k in z.files}
    fit, held, mem = fold_partition(sp['task'], sp['episode'], bank['task'], bank['episode'], assignment, fold)
    from kc2_run import KC2Net
    name_seed = K2.seed_init(SEED, 'i', 'distance', TAU)
    model_path = outdir / f'fold{fold}_epoch16.pt'
    ckpt_existing = sorted(outdir.glob(f'fold{fold}_epoch*.pt'), key=lambda p:int(p.stem.split('epoch')[-1]))
    torch.manual_seed(name_seed)
    model = KC2Net('i', 'distance').cpu()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    start_epoch=0
    if ckpt_existing:
        last=ckpt_existing[-1]
        ck=torch.load(last,map_location='cpu',weights_only=False)
        model.load_state_dict(ck['state_dict']); start_epoch=int(ck['epoch'])
        if 'optimizer' in ck: optimizer.load_state_dict(ck['optimizer'])
    else:
        pass
    mem_task=np.asarray(bank['task'])[mem];mem_ep=np.asarray(bank['episode'])[mem]
    fold_keys=keys_np[mem]
    target_raw=np.asarray(targets['eh'][0],np.float32)
    target_norm=(target_raw[fit]-mean)/std
    nearest=K1.nearest_allowed_coefficients(target_norm,fold_keys,sp['task'][fit],sp['episode'][fit],mem_task,mem_ep)
    gs=K1._as_gpusplit(kr,sp,torch.device('cpu'),False)
    labels=K2._gripper_labels(sp)
    rng=np.random.default_rng(name_seed + 1009*fold)
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','8')))
    for epoch in range(start_epoch+1,EPOCHS+1):
        model.train(); order=rng.permutation(len(fit)); losses=[]
        for st in range(0,len(order),256):
            pos=order[st:st+256]; rows=fit[pos]
            ix=torch.as_tensor(rows,dtype=torch.long);q=model(gs,ix)
            nearest_rows=nearest[pos]
            row_allowed=K1.candidate_mask(sp['task'][rows],sp['episode'][rows],mem_task,mem_ep)
            random_ids=rng.choice(len(mem),size=min(K1.RANDOM_CANDIDATES,len(mem)),replace=False)
            cand=np.unique(np.r_[nearest_rows,random_ids]).astype(np.int64)
            allow=torch.as_tensor(row_allowed[:,cand],dtype=torch.bool)
            target_t=torch.as_tensor((target_raw[rows]-mean)/std,dtype=torch.float32)
            allkey=torch.as_tensor(fold_keys,dtype=torch.float32)
            dtrue=torch.square(target_t[:,None]-allkey[cand][None]).sum(-1)
            score=model.score(q,allkey[cand])
            soft=torch.softmax((-dtrue/TAU).masked_fill(~allow,-torch.inf),-1)
            logp=torch.log_softmax(score.masked_fill(~allow,-torch.inf),-1)
            retr=-(soft*torch.nan_to_num(logp,nan=0.,neginf=0.)).sum(-1).mean()
            grip=K2.F.binary_cross_entropy_with_logits(model.grip_logits(q),torch.as_tensor(labels[rows],dtype=torch.float32))
            loss=retr+LAMBDA*grip
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            with torch.no_grad():model.query.logit_scale.clamp_(max=float(np.log(100.)))
            losses.append((float(retr.detach()),float(grip.detach()),float(loss.detach())))
        ck={'fold':int(fold),'epoch':epoch,'state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),
            'memory_ids':mem,'setting':{'q_name':'i','score_mode':'distance','tau':TAU,'lambda':LAMBDA,
            'seed':SEED,'epoch':EPOCHS,'kc2_selected_epoch':16}}
        tmp=outdir/f'fold{fold}_epoch{epoch}.tmp.pt';torch.save(ck,tmp);tmp.replace(outdir/f'fold{fold}_epoch{epoch}.pt')
        log('query_fold_epoch',fold=fold,epoch=epoch,train_rows=len(fit),heldout_rows=len(held),memory_units=len(mem),
            retrieval_loss=float(np.mean([x[0] for x in losses])),gripper_bce=float(np.mean([x[1] for x in losses])))
    model.load_state_dict(torch.load(model_path,map_location='cpu',weights_only=False)['state_dict']);model.eval()
    gs=K1._as_gpusplit(kr,sp,torch.device('cpu'),False)
    result=_score_rows(model,gs,sp,bank,keys_np,held,memory_ids=mem)
    result.update(task=np.asarray(sp['task'][held]),episode=np.asarray(sp['episode'][held]),frame=np.asarray(sp['key'][held])[:,2],
                  row_id=held.astype(np.int32),fold=np.full(len(held),fold,np.int8))
    tmp=rowpath.with_suffix('.tmp.npz');np.savez_compressed(tmp,**result);tmp.replace(rowpath)
    return result


def _build_eh_observations(ctx):
    """Cache exact 224px π0.5 observations for all EH keys, restartable by row."""
    import kq2cl_obs as O
    import expert_ex3 as EX
    import vq_sources as VS
    sp=ctx.splits['eh']; keys=np.asarray(sp['key'],np.int32); d=OUT/'obs_eh';d.mkdir(parents=True,exist_ok=True)
    expected={'key':keys}
    shapes={'image_base':((224,224,3),np.uint8),'image_wrist':((224,224,3),np.uint8),
            'state':((32,),np.float32),'prompt':((200,),np.int32),'prompt_mask':((200,),bool)}
    for name,(shape,dtype) in shapes.items():
        p=d/f'{name}.npy'
        if p.exists():
            a=np.load(p,mmap_mode='r')
            if a.shape!=(len(keys),*shape):raise AssertionError(f'bad observation cache shape {name}: {a.shape}')
        else: np.lib.format.open_memmap(p,mode='w+',dtype=dtype,shape=(len(keys),*shape)).flush()
    if (d/'key.npy').exists():
        if not np.array_equal(np.load(d/'key.npy'),keys):raise AssertionError('XF2 EH observation key mismatch')
    else:np.save(d/'key.npy',keys)
    donepath=d/'done.npy'
    done=np.load(donepath,mmap_mode='r+') if donepath.exists() else np.lib.format.open_memmap(donepath,mode='w+',dtype=np.bool_,shape=(len(keys),))
    if not donepath.exists():done[:]=False;done.flush()
    _,base=EX._base();rows40=base.load_rows(C.RUN40)
    rowkey=np.stack([rows40['task'],rows40['episode'],rows40['frame']],axis=1).astype(np.int32)
    rowmap={tuple(k):i for i,k in enumerate(rowkey)}
    trs=O.transforms();reader=VS.ObsReader()
    output={n:np.load(d/f'{n}.npy',mmap_mode='r+') for n in shapes}
    # Existing run-01 observations provide exact source-parity rows; read unmatched 5-frame starts from source parquet.
    for i,key in enumerate(keys):
        if done[i]:continue
        tup=tuple(map(int,key))
        if tup in rowmap:
            rid=rowmap[tup]
            values={'image_base':rows40['image_base'][rid],'image_wrist':rows40['image_wrist'][rid],
                    'state':rows40['state'][rid],'prompt':rows40['prompt'][rid],'prompt_mask':rows40['prompt_mask'][rid]}
        else:
            raw=reader.read(key,'libero40');item=O.apply_transforms(trs,raw)
            values={'image_base':item['image']['base_0_rgb'],'image_wrist':item['image']['left_wrist_0_rgb'],
                    'state':item['state'],'prompt':item['tokenized_prompt'],'prompt_mask':item['tokenized_prompt_mask']}
        for n,v in values.items():output[n][i]=np.asarray(v,dtype=shapes[n][1])
        done[i]=True
        if (i+1)%250==0:
            for a in output.values():a.flush()
            done.flush();log('eh_obs_extract',completed=int(np.asarray(done).sum()),total=len(keys))
    for a in output.values():a.flush()
    done.flush()
    if not np.asarray(done).all():raise RuntimeError('EH obs cache is incomplete')
    # Normalized state parity with KC2's row input is checked for every row.
    state=np.load(d/'state.npy',mmap_mode='r')
    expected_state=np.asarray(sp['state'],np.float32)
    if expected_state.shape[1]>=8:
        diff=float(np.max(np.abs(state[:,:8]-expected_state[:,:8])))
        if diff>2e-4:raise AssertionError(f'EH state parity failed max_abs={diff}')
    meta={'rows':len(keys),'key_order_sha256':__import__('hashlib').sha256(keys.tobytes()).hexdigest(),
          'matching_run40_rows':int(sum(tuple(map(int,k)) in rowmap for k in keys)),
          'source_read_rows':int(len(keys)-sum(tuple(map(int,k)) in rowmap for k in keys)),
          'state_max_abs_diff_vs_kc2':diff,'image_resolution':[224,224],
          'source':'run-01 cached observations where exact key exists; otherwise vq_sources.ObsReader(libero40)+kq2cl_obs.transforms'}
    (d/'manifest.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False)+'\n')
    return d


def prepare() -> dict:
    if not (KC2/'DONE').is_file():raise RuntimeError('KC2 DONE gate is required before XF2 preparation')
    OUT.mkdir(parents=True,exist_ok=True)
    for p in (OUT/'OFFLINE_DONE',OUT/'DONE'):
        if p.exists():raise FileExistsError(f'refusing to overwrite XF2 completion marker {p}')
    kr,ctx,targets=_load_kc2_context();bank=K1.memory_bank();sp=ctx.splits['eh']
    assignment=_split_fold_assignment(sp,bank)
    chosen,fullck,keys_np,mean,std=_kc2_assets(bank)
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','8')))
    arrays=[]
    for f in range(KFOLDS):
        arrays.append(_train_kc2_fold(f,kr,ctx,targets,bank,assignment,keys_np,mean,std))
    cv={k:np.concatenate([a[k] for a in arrays],axis=0) for k in arrays[0] if isinstance(arrays[0][k],np.ndarray)}
    order=np.argsort(cv['row_id']);cv={k:v[order] for k,v in cv.items()}
    folds=cv['fold'];ehat_oof,iso=fit_oof_isotonic(cv['spread10'],cv['e_arm'],folds)
    cv['ehat_oof']=ehat_oof
    full_meta={'x':np.asarray(iso.X_thresholds_,np.float64).tolist(),'y':np.asarray(iso.y_thresholds_,np.float64).tolist(),
               'out_of_bounds':'clip','source':'XF2 KC2 query out-of-fold heldout rows only'}
    np.savez_compressed(OUT/'rows_eh_crossfit.npz',**cv)
    (OUT/'isotonic.json').write_text(json.dumps(full_meta,indent=2)+'\n')
    # KC2 production query on Dev/learned/unseen20, same full memory and Gdec as its selected policy.
    fullmodel=_model_from_checkpoint(fullck)
    split_rows={}
    for split in ('dev2','learned','unseen20'):
        splitsp=ctx.splits[split];gs=K1._as_gpusplit(kr,splitsp,torch.device('cpu'),False)
        val=_score_rows(fullmodel,gs,splitsp,bank,keys_np,np.arange(gs.n))
        val.update(task=np.asarray(splitsp['task']),episode=np.asarray(splitsp['episode']),frame=np.asarray(splitsp['key'])[:,2])
        np.savez_compressed(OUT/f'rows_{split}.npz',**val);split_rows[split]=val
    dev_ehat=apply_isotonic(full_meta,split_rows['dev2']['spread10']).astype(np.float32)
    fb=calibration_threshold(dev_ehat,.204,spread=split_rows['dev2']['spread10'])
    calibration={'isotonic':full_meta,'fallback':fb,'p_b':.204,
                 'threshold_source':'KC2 full-memory query predictions on Dev v2; empirical ehat quantile only',
                 'kc2_checkpoint':chosen['checkpoint'],'kc2_threshold':float(chosen['dev_threshold'])}
    (OUT/'calibration.json').write_text(json.dumps(calibration,indent=2,ensure_ascii=False)+'\n')
    trainrows={'row_id':cv['row_id'],'fold':folds,'task':cv['task'],'episode':cv['episode'],
               'draft':cv['draft'],'target':np.asarray(sp['actions'][cv['row_id']],np.float32),
               'ehat_oof':ehat_oof,'spread10':cv['spread10'],'e_arm':cv['e_arm']}
    np.savez_compressed(OUT/'expert_train_rows.npz',**trainrows)
    obsdir=_build_eh_observations(ctx)
    manifest={'status':'prepared','split_seed':json.loads((XF1/'manifest.json').read_text())['split_seed'],
        'fold_assignment_source':str(XF1/'manifest.json'),'folds':KFOLDS,'kc2_selected':chosen,
        'query_refit':{'q_name':'i','score_mode':'distance','tau':TAU,'lambda':LAMBDA,'seed':SEED,'epochs':EPOCHS,
                       'episode_folds':True,'memory_and_query_train_exclude_heldout_fold':True},
        'row_counts':{'eh_crossfit':len(cv['row_id']),'dev2':len(split_rows['dev2']['spread10']),
                      'learned':len(split_rows['learned']['spread10']),'unseen20':len(split_rows['unseen20']['spread10'])},
        'eh_observation_manifest':str(obsdir/'manifest.json'),'calibration':calibration,
        't0_formula':'clip(k*sqrt(ehat)/(1+k*sqrt(ehat)), .05, .6); train k=1 and multiply U(.8,1.25)',
        'inference_signal':'spread10 from top ten standardized saved coefficient keys only'}
    mpath=OUT/'manifest.json'
    if mpath.exists():
        old=json.loads(mpath.read_text())
        if old.get('fold_assignment_source')!=manifest['fold_assignment_source'] or old.get('kc2_selected',{}).get('name')!=chosen['name']:
            raise AssertionError('existing XF2 manifest incompatible with this run')
    mpath.write_text(json.dumps(manifest,indent=2,ensure_ascii=False,default=float)+'\n')
    log('prepare_done',**manifest['row_counts'],fallback_threshold=fb['threshold'],fallback_realized_rate=fb['realized_rate'])
    return manifest


def _oof_training_arrays():
    z=np.load(OUT/'expert_train_rows.npz',allow_pickle=False)
    return {k:z[k] for k in z.files}


def train_expert(*,updates=5000,batch=8,microbatch=8,save_every=500):
    import jax, jax.numpy as jnp, flax.nnx as nnx, optax, orbax.checkpoint as ocp
    import expert_ex3 as EX
    import skill_expert_refine as E3
    OUT.mkdir(parents=True,exist_ok=True)
    rows=_oof_training_arrays();obsdir=OUT/'obs_eh'
    obs={k:np.load(obsdir/f'{k}.npy',mmap_mode='r') for k in ('image_base','image_wrist','state','prompt','prompt_mask')}
    ehat=np.asarray(rows['ehat_oof'],np.float32);draft=np.asarray(rows['draft'],np.float32);target=np.asarray(rows['target'],np.float32)
    row_ids=np.asarray(rows['row_id'],np.int64)
    if len(rows['fold'])!=len(row_ids) or len(set(row_ids.tolist()))!=len(row_ids):raise AssertionError('expert train row IDs are not unique')
    train_dir=OUT/'expert'/'train-XF2-K1';ck= train_dir/'checkpoints';ck.mkdir(parents=True,exist_ok=True)
    exp3,base=EX._base();model,_=base.load_policy(C.PI05);filt=EX._filter()
    optimizer=optax.chain(optax.clip_by_global_norm(1.),optax.adamw(2.5e-5,b1=.9,b2=.95,weight_decay=1e-5))
    state=nnx.state(model,filt);opt_state=optimizer.init(state)
    meta_path=train_dir/'train.json';meta=json.loads(meta_path.read_text()) if meta_path.exists() else {
        'status':'running','updates':updates,'batch':batch,'microbatch':microbatch,'save_every':save_every,
        'rows':len(row_ids),'draft':'KC2 selected query top1, fold-specific memory exclusion, KC2 q-only Gdec',
        'confidence':'out-of-fold isotonic E[e_arm|spread10]','training_t0':'k=1 formula × U(.8,1.25), clipped .05-.6',
        'loss':'KQ2-CL K1 bridge flow, normalized arm6 only, gripper weight 0','optimizer':'AdamW 2.5e-5 b(.9,.95) wd1e-5 clip1',
        'seed':20261120,'param_filter':EX.PARAM_REGEX,'history':[]}
    start=0;finished=sorted(int(p.name) for p in ck.iterdir() if p.name.isdigit() and (ck/f'{p.name}.opt').exists())
    if finished:
        start=finished[-1]
        with ocp.PyTreeCheckpointer() as cp:
            restored=cp.restore(ck/f'{start:06d}')
            st=nnx.state(model,filt);st.replace_by_pure_dict(jax.tree.map(jnp.asarray,restored));nnx.update(model,st)
            opt_state=cp.restore(ck/f'{start:06d}.opt',item=opt_state)
    w=jnp.asarray([1.]*6+[0.],jnp.float32)
    @nnx.jit
    def grad_step(current,ob,action,d,t0,u,eps):
        source=(1.-t0[:,None,None])*d+t0[:,None,None]*eps
        x=(1.-u[:,None,None])*action+u[:,None,None]*source
        target_v=(source-action)/jnp.maximum(t0[:,None,None],1e-3)
        def loss_fn(m):
            pred=m.velocity(ob,x,t0*u)
            return jnp.mean(jnp.sum(jnp.square(pred[:,:,:7]-target_v[:,:,:7])*w,-1))/jnp.sum(w)
        return nnx.value_and_grad(loss_fn,argnums=nnx.DiffState(0,filt))(current)
    np_rng=np.random.default_rng(20261120+start);rng=jax.random.fold_in(jax.random.key(20261120),start)
    losslog=[];started=time.time()
    for step in range(start+1,updates+1):
        bi=np_rng.integers(0,len(row_ids),size=batch)
        d=np.zeros((batch,10,32),np.float32);d[:,:,:7]=draft[bi]
        a=np.zeros((batch,10,32),np.float32);a[:,:,:7]=target[bi]
        tbase=variable_t0(ehat[bi],1.)
        t0=np.clip(tbase*np_rng.uniform(.8,1.25,size=batch),.05,.6).astype(np.float32)
        rng,uk,ek=jax.random.split(rng,3);u=jax.random.uniform(uk,(batch,),minval=.01,maxval=1.)
        eps=jax.random.normal(ek,(batch,10,32))
        grads=None;lv=0.
        for lo in range(0,batch,microbatch):
            mb=np.arange(lo,min(lo+microbatch,batch));source_ids=row_ids[bi[mb]]
            ob=base.obs_batch(obs,source_ids)
            val,g=grad_step(model,ob,jnp.asarray(a[mb]),jnp.asarray(d[mb]),jnp.asarray(t0[mb]),u[mb],eps[mb])
            weight=len(mb)/batch;lv+=float(val)*weight;g=jax.tree.map(lambda x:x*weight,g)
            grads=g if grads is None else jax.tree.map(jnp.add,grads,g)
        st=nnx.state(model,filt);upd,opt_state=optimizer.update(grads,opt_state,st);nnx.update(model,optax.apply_updates(st,upd))
        losslog.append(lv)
        if step%100==0:log('expert_train_step',step=step,loss=float(np.mean(losslog[-100:])),seconds=round(time.time()-started))
        if step%save_every==0 or step==updates:
            EX._save_state(model,filt,ck/f'{step:06d}',ocp)
            with ocp.PyTreeCheckpointer() as cp:cp.save(ck/f'{step:06d}.opt',opt_state,force=True)
            meta['history'].append({'step':step,'loss_last':float(np.mean(losslog[-save_every:])),'checkpoint':str(ck/f'{step:06d}')})
            tmp=meta_path.with_suffix('.tmp');tmp.write_text(json.dumps(meta,indent=2,ensure_ascii=False)+'\n');tmp.replace(meta_path)
    meta['status']='complete';meta['training_seconds']=time.time()-started
    meta_path.write_text(json.dumps(meta,indent=2,ensure_ascii=False)+'\n')
    log('expert_train_done',checkpoint=str(ck/f'{updates:06d}'),seconds=meta['training_seconds'])
    return meta


def offline(*,batch=16):
    import expert_ex3 as EX
    import kq2cl_obs as O
    import skill_expert_refine as E3
    import t0x_common as T5
    if (OUT/'OFFLINE_DONE').exists():raise FileExistsError('OFFLINE_DONE already exists; preserving existing evaluation')
    selected=K2.OFFLINE if hasattr(K2,'OFFLINE') else KC2/'offline_results.json'
    kc2sel=json.loads((KC2/'offline_results.json').read_text())['selected']
    new_ck=OUT/'expert'/'train-XF2-K1'/'checkpoints'/'005000'
    if not new_ck.exists():raise FileNotFoundError(f'XF2 K1 training checkpoint missing: {new_ck}')
    calpath=OUT/'calibration_v2.json' if (OUT/'calibration_v2.json').exists() else OUT/'calibration.json'
    cal=json.loads(calpath.read_text());iso=IsotonicRegression(out_of_bounds='clip')
    iso.X_thresholds_=np.asarray(cal['isotonic']['x'],np.float64);iso.y_thresholds_=np.asarray(cal['isotonic']['y'],np.float64)
    devsp=np.load(OUT/'rows_dev2.npz',allow_pickle=False);dev_ehat=apply_isotonic(cal['isotonic'],devsp['spread10'])
    fallback=calibration_threshold(dev_ehat,.204,spread=devsp['spread10'])
    oldcal=json.loads((KC2/'calibration.json').read_text());D_b=float(oldcal['D_b'])
    ex=EX.LiveExpert();_,base=EX._base();bank=K1.memory_bank();targets={};allres={};metrics={}
    checkpoints={'XF2':new_ck,'K1_old':STATE_DIR / 'outputs/memory-pilot/unseen-libero/kq2-cl-01/expert/train-K1/checkpoints/000500'}
    for split in ('dev2','learned','unseen20'):
        rows=O.load_obs(split);spdir=KC2/'data'/split
        target=np.load(spdir/'actions.npy');targets[split]=target
        rr=np.load(OUT/f'rows_{split}.npz',allow_pickle=False)
        n=len(target);obs=base.obs_batch(rows,np.arange(n));noise=E3._seed_noise(np.arange(n)+{'learned':0,'unseen20':50_000_000,'dev2':60_000_000}[split])
        ehat=apply_isotonic(cal['isotonic'],rr['spread10']).astype(np.float32)
        candidates={f'V_k{k:g}':variable_t0(ehat,k) for k in T0_K}
        candidates.update({f'F_t{t:g}':np.full(n,t,np.float32) for t in T0_FIXED})
        errors={};draft=rr['draft'];outputs={}
        for mode,checkpoint in checkpoints.items():
            ex.load(checkpoint)
            for name,t0 in candidates.items():
                out=[]
                for st in range(0,n,batch):
                    ix=slice(st,min(st+batch,n))
                    out.append(ex.bridge(obs, draft[ix],noise[ix],t0[ix],batch=batch))
                pred=np.concatenate(out).astype(np.float32)
                errors[f'{mode}_{name}']=C.arm6(pred,target)
                outputs[f'{mode}_{name}']=pred
        # Old KC2 Expert comparator: KC2 top-10 disagreement calibration and same fallback policy.
        top_actions=np.asarray(bank['value_actions10'])[rr['top_ids']]
        dd,_=T5.topk_disagreement(top_actions,rr['top_scores'],k=10)
        old_t0=T5.t0_from_disagreement(dd,oldcal)
        old_fallback=dd>D_b
        ex.load(checkpoints['K1_old']);oldpred=[]
        for st in range(0,n,batch):
            ix=slice(st,min(st+batch,n));oldpred.append(ex.bridge(obs,draft[ix],noise[ix],old_t0[ix],batch=batch))
        oldpred=np.concatenate(oldpred).astype(np.float32)
        pi=np.asarray(np.load(C.OUT/'pi05'/f'{split}.npy'),np.float32)
        if pi.shape[0]!=n:raise AssertionError(f'{split}: π0.5 reference row count mismatch')
        # Apply the same Dev-calibrated fallback gate to both XF2 policies in
        # offline selection and reporting, matching online serving semantics.
        xf2_fallback=fallback_mask(ehat,rr['spread10'],fallback)
        for name in list(errors):
            if name.startswith('XF2_'):
                outputs[name][xf2_fallback]=pi[xf2_fallback]
                errors[name]=C.arm6(outputs[name],target)
        oldpred[old_fallback]=pi[old_fallback]
        errors['K1_old_KC2_policy']=C.arm6(oldpred,target);outputs['K1_old_KC2_policy']=oldpred
        arrays={'spread10':rr['spread10'],'ehat':ehat,'fallback':xf2_fallback,
                'old_disagreement':dd,'old_fallback':old_fallback,'old_t0':old_t0}
        arrays.update({f'error__{k}':v for k,v in errors.items()})
        arrays.update({f'prediction__{k}':v for k,v in outputs.items()})
        np.savez_compressed(OUT/f'offline_{split}.npz',**arrays)
        metrics[split]={k:{'mean_arm6':float(v.mean()),'p50':float(np.quantile(v,.5)),'p90':float(np.quantile(v,.9))}
                        for k,v in errors.items()}
        allres[split]={'ehat':ehat,'errors':errors}
        log('offline_split_done',split=split,rows=n,best_dev=min((k,v['mean_arm6']) for k,v in metrics[split].items() if k.startswith('XF2_')))
    vnames=[f'V_k{k:g}' for k in T0_K];fnames=[f'F_t{t:g}' for t in T0_FIXED]
    devmeans={n:metrics['dev2'][f'XF2_{n}']['mean_arm6'] for n in vnames+fnames}
    best_v=min(vnames,key=lambda n:devmeans[n]);best_f=min(fnames,key=lambda n:devmeans[n])
    dev_e=allres['dev2']['ehat'];order=np.argsort(dev_e,kind='stable');groups=np.array_split(order,5)
    quintile=[]
    for i,ix in enumerate(groups):
        quintile.append({'bin':i+1,'n':len(ix),'mean_spread10':float(np.mean(np.load(OUT/'rows_dev2.npz')['spread10'][ix])),
                         'mean_ehat':float(dev_e[ix].mean()),'V_mean_arm6':float(np.mean(allres['dev2']['errors'][f'XF2_{best_v}'][ix])),
                         'F_mean_arm6':float(np.mean(allres['dev2']['errors'][f'XF2_{best_f}'][ix]))})
    result={'status':'offline_complete','dev_selection':{'best_variable':best_v,'best_fixed':best_f,
            'dev_mean_arm6':{k:devmeans[k] for k in devmeans},'fallback':fallback},
            'selected_settings':{'variable_k':float(best_v.split('k')[1]),'fixed_t0':float(best_f.split('t')[1])},
            'metrics':metrics,'dev_spread10_quintiles':quintile,'fallback_threshold':fallback,
            't0_formula':'clip(k*sqrt(ehat)/(1+k*sqrt(ehat)),.05,.6)',
            'old_k1_rule':'KC2 top10 disagreement t0 from kc2-01/calibration.json, fallback D>D_b; same observation and noise',
            'kc2_query_checkpoint':kc2sel['checkpoint'],'xf2_expert_checkpoint':str(new_ck),
            'expert_training_rows':len(np.load(OUT/'expert_train_rows.npz')['row_id']),
            'learned_gdec_optimism':'KC2 Gdec was fit using the full LIBERO-40 data, including the crossfit training episodes.'}
    (OUT/'offline_results.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    (OUT/'OFFLINE_DONE').write_text('XF2 offline selection/evaluation complete\n')
    log('offline_done',best_variable=best_v,best_fixed=best_f,dev_v=devmeans[best_v],dev_f=devmeans[best_f])
    _write_report(result,closed=None)
    return result


def _write_report(offline_result,closed):
    lines=['# XF2：検索下書きと信頼度連動 t0', '',
           'KC2 のクエリ構成で検索した下書きを使い、XF1 のエピソード分割を保って Expert を学習した。未学習20Hzと学習済み窓は記述的評価。閉ループの比較は同一 task/trial の対応付けで計算した。', '']
    if offline_result:
        lines += ['## オフライン評価', '',
            '| Split | 条件 | 腕6軸 MSE 平均 | p50 | p90 | 行数 |',
            '|---|---|---:|---:|---:|---:|']
        for split, metrics in offline_result['metrics'].items():
            for name,m in metrics.items():
                n=len(np.load(OUT/f'offline_{split}.npz',allow_pickle=False)[f'error__{name}'])
                lines.append(f"| {split} | {name} | {m['mean_arm6']:.6f} | {m['p50']:.6f} | {m['p90']:.6f} | {n} |")
        sel=offline_result['dev_selection']
        lines += ['', f"Dev で選択：可変 `{sel['best_variable']}` (MSE {sel['dev_mean_arm6'][sel['best_variable']]:.6f})、固定 `{sel['best_fixed']}` (MSE {sel['dev_mean_arm6'][sel['best_fixed']]:.6f})。フォールバック閾値 {sel['fallback']['threshold']:.6g}、Dev 実現率 {sel['fallback']['realized_rate']:.3f}（目標 .204）。", '',
          '## 学習用下書きの誤差', '',
          '| 指標 | 値 |', '|---|---:|']
        tr=np.load(OUT/'expert_train_rows.npz',allow_pickle=False)
        for label,arr in [('検索した下書き腕 MSE',tr['e_arm']),('out-of-fold 推定誤差 ê',tr['ehat_oof'])]:
            lines.append(f'| {label} 平均 / p50 / p90 | {np.mean(arr):.6f} / {np.median(arr):.6f} / {np.quantile(arr,.9):.6f} |')
    if closed:
        lines += ['', '## 閉ループ', '', '| Suite | 条件 | 成功 | XF2のみ | 比較のみ | McNemar exact p |', '|---|---|---:|---:|---:|---:|']
        for suite, data in closed['suites'].items():
            for cond in ('XF2-V','XF2-F'):
                c=data['conditions'][cond]
                for ref,paired in c['paired'].items():
                    lines.append(f"| {suite} | {cond} vs {ref} | {c['successes']}/100 vs {c['references'][ref]['successes']}/100 | {paired['xf2_only']} | {paired['reference_only']} | {paired['mcnemar_p_exact']:.6g} |")
    lines += ['', '## 再現性・注意', '',
              '- Query は fold ごとに再学習し、評価エピソードの単位を fold memory と query training の双方から除外した。Gdec は KC2 の全学習データで学習済みなので、クロスフィット学習行に対して楽観的になりうる。',
              '- t0 可変は `clip(k√ê/(1+k√ê), .05, .6)`、固定 t0 と k は Dev v2 だけで選択した。',
              '- Discordant pair の2列は順に XF2 条件のみ成功、対照のみ成功。', '']
    p=OUT/'report.md';p.write_text('\n'.join(lines),encoding='utf-8')


def _success_map(path):
    out={}
    p=Path(path)
    if not p.exists():return out
    for line in p.read_text().splitlines():
        if not line.strip():continue
        r=json.loads(line)
        if r.get('status')=='complete':out[(int(r['task_id']),int(r['trial_index']))]=bool(r['success'])
    return out


def _trial_paths():
    from t0x_report import baseline_for_suite
    refs={}
    for suite in ('libero90','libero10'):
        kc=KC2/'closed-loop'/suite/'KC2-dot-Gdec'/'trials.jsonl'
        t5=ROOT/'t0x-01'/'t5'/'closed-loop'/'closed-loop'/suite/'AtopkGdec'/'trials.jsonl'
        _,pirows=baseline_for_suite(suite)
        refs[suite]={'KC2':_success_map(kc),'T5_AtopkGdec':_success_map(t5),
            'pi05':{(int(k[0]),int(k[1])):bool(v['success']) for k,v in pirows.items() if v.get('status')=='complete'}}
    return refs


def run_closed(*,fraction=.2,workers=8):
    """Run XF2-V/F through the CL1 harness; each condition resumes its own trial JSONL."""
    import subprocess
    import kq2cl_closed as CL
    if not (OUT/'OFFLINE_DONE').is_file():raise RuntimeError('offline selection must finish before closed loop')
    if (OUT/'DONE').exists():raise FileExistsError('DONE already exists; preserving completed XF2 run')
    gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    used,total=map(int,gpu.stdout.strip().split(',')[:2])
    if used>min(8192,int(total*fraction*.65)):raise RuntimeError(f'GPU memory use too high before XF2 server: {used}/{total} MiB, fraction={fraction}')
    CL.C.OUT=OUT
    # The official harness launches two conditions concurrently. XF2 serializes them to
    # keep the H200 fraction bounded and avoid server contention.
    real_popen=CL.subprocess.Popen
    def launch(args,*a,**kw):
        if isinstance(args,(list,tuple)):
            args=list(args)
            for i,v in enumerate(args):
                if str(v).endswith('/kq2cl_serve.py'):
                    args[i]=str(HERE/'xf2_serve.py');break
        return real_popen(args,*a,**kw)
    CL.subprocess.Popen=launch
    oldenv={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')}
    os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    try:
        spec_path=OUT/'closed-spec.json'
        if not spec_path.exists():
            sel=json.loads((KC2/'offline_results.json').read_text())['selected']; checkpoint=OUT/'expert/train-XF2-K1/checkpoints/005000'
            cal=OUT/'calibration_v2.json' if (OUT/'calibration_v2.json').exists() else OUT/'calibration.json'
            calpath=OUT/'calibration_v2.json' if (OUT/'calibration_v2.json').exists() else OUT/'calibration.json'
            spec={n:{'mode':'memory','query':'i','expert':str(checkpoint),'bundle':str(sel['checkpoint']),
                'tpol':{'calibration':str(calpath),'strategy':n}} for n in ('XF2-V','XF2-F')}
            C.write_json(spec_path,spec)
        else:spec=json.loads(spec_path.read_text())
        refs=_trial_paths();suites={}
        for suite in ('libero90','libero10'):
            CL.PLAN[suite]=dict(CL.PLAN[suite])
            for cond in ('XF2-V','XF2-F'):
                CL.PLAN[suite][cond]=range(10)
                log('closed_condition_start',suite=suite,condition=cond,gpu_used_mib=used,total_mib=total,fraction=fraction)
                CL.run([cond],workers=workers,fraction=fraction,smoke=False,cpus=['24-39'],spec_path=str(spec_path),suite=suite)
            cdata={}
            for cond in ('XF2-V','XF2-F'):
                trialfile=OUT/'closed-loop'/suite/cond/'trials.jsonl'
                values=_success_map(trialfile)
                if len(values)!=100:raise RuntimeError(f'{suite}/{cond} incomplete: {len(values)}/100; resume by rerunning closed')
                paired={name:paired_counts(values,rv) for name,rv in refs[suite].items()}
                cdata[cond]={'successes':sum(values.values()),'completed':len(values),'paired':paired}
                cdata[cond]['references']={name:{'completed':len(rv),'successes':sum(rv.values())} for name,rv in refs[suite].items()}
            cdata['V_vs_F']=paired_counts(_success_map(OUT/'closed-loop'/suite/'XF2-V'/'trials.jsonl'),_success_map(OUT/'closed-loop'/suite/'XF2-F'/'trials.jsonl'))
            suites[suite]={'conditions':cdata}
        result={'status':'complete','paired_protocol':'CL1, task/trial paired, seed_base=7, wait=10, replan=5, max_steps=520',
                'fallback_threshold':json.loads((OUT/'calibration.json').read_text())['fallback'],
                'fraction':fraction,'gpu_initial_mib':used,'gpu_total_mib':total,'suites':suites}
        target=OUT/'closed_loop_results.json'
        if target.exists():raise FileExistsError(target)
        target.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
        _write_report(json.loads((OUT/'offline_results.json').read_text()),result)
        (OUT/'DONE').write_text('XF2 closed-loop complete\n')
        log('closed_done',successes={s:{c:v['successes'] for c,v in x['conditions'].items() if c.startswith('XF2-')} for s,x in suites.items()})
        return result
    finally:
        CL.subprocess.Popen=real_popen;CL.C.OUT=ROOT/'kq2-cl-01'
        for k,v in oldenv.items():
            if v is None:os.environ.pop(k,None)
            else:os.environ[k]=v


def paired_counts(xf2_success, ref_success):
    """Correctly oriented paired result: XF2-only then reference-only successes."""
    keys=sorted(set(xf2_success)&set(ref_success))
    x=np.asarray([bool(xf2_success[k]) for k in keys]);r=np.asarray([bool(ref_success[k]) for k in keys])
    xonly=int((x&~r).sum());ronly=int((r&~x).sum());n=xonly+ronly
    p=1. if n==0 else min(1.,2.*sum(math.comb(n,i) for i in range(min(xonly,ronly)+1))/2**n)
    return {'pairs':len(keys),'xf2_only':xonly,'reference_only':ronly,'mcnemar_p_exact':p,
            'success_difference_check':int(x.sum())-int(r.sum())==xonly-ronly}


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('stage',choices=('prepare','calibrate','train','offline','closed','all'))
    ap.add_argument('--updates',type=int,default=5000);ap.add_argument('--fraction',type=float,default=.25)
    a=ap.parse_args()
    if a.stage in ('prepare','all'):prepare()
    if a.stage=='calibrate':calibrate()
    if a.stage in ('train','all'):train_expert(updates=a.updates)
    if a.stage in ('offline','all'):offline()
    if a.stage in ('closed','all'):run_closed(fraction=a.fraction)


if __name__=='__main__':main()
