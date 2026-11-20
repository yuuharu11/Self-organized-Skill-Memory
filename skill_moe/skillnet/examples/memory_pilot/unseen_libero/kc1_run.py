#!/usr/bin/env python3
"""KC1: query learning and offline evaluation against saved DCT3 shape codes.

All learned work in this file runs on CPU. The only closed-loop GPU work is
deliberately left to kc1_closed.py after the T5 DONE gate.
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
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
STATE_DIR = Path(os.environ.get('SKILLNET_STATE_DIR', Path(__file__).resolve().parents[5] / '.skillnet-state')).expanduser()
ROOT = STATE_DIR / 'outputs/memory-pilot/unseen-libero'
OUT = ROOT / 'kc1-01'
BANK_DIR = ROOT / 'key-motion-01/final/data'
KQ2 = ROOT / 'kq2-01'
SPLITS = ('eh', 'dev2', 'learned', 'unseen20')
TAUS = (.03, .1, .3)
SEEDS = (1, 2, 3)
EPOCHS = 16
RANDOM_CANDIDATES = 256
_BANK_CACHE = None


def log(event: str, **fields):
    OUT.mkdir(parents=True, exist_ok=True)
    rec = {'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'event': event, **fields}
    with (OUT / 'progress.log').open('a') as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=float) + '\n')
    print(json.dumps(rec, ensure_ascii=False, default=float), flush=True)


def memory_bank(data_dir: Path = BANK_DIR) -> dict:
    """Load only immutable coefficients and metadata from the memory bank."""
    global _BANK_CACHE
    if Path(data_dir) == BANK_DIR and _BANK_CACHE is not None:
        return _BANK_CACHE
    result = {k: np.load(data_dir / f'bank__{k}.npy', mmap_mode='r', allow_pickle=False)
              for k in ('coefficients', 'magnitude', 'length', 'task', 'episode', 'source',
                        'natural_actions', 'value_actions10')}
    if Path(data_dir) == BANK_DIR:
        _BANK_CACHE = result
    return result


def coefficient_keys(coefficients: np.ndarray, *, standardize: bool = False,
                     mean: np.ndarray | None = None, std: np.ndarray | None = None):
    c = np.asarray(coefficients, np.float32).reshape(len(coefficients), 18)
    if not standardize:
        return c.copy(), np.zeros(18, np.float32), np.ones(18, np.float32)
    if mean is None or std is None:
        mean = c.mean(0)
        std = c.std(0)
    mean = np.asarray(mean, np.float32).reshape(18)
    std = np.asarray(std, np.float32).reshape(18)
    std = np.maximum(std, 1e-6)
    return ((c - mean) / std).astype(np.float32), mean, std


def verify_saved_coefficients(bank: dict, *, atol=2e-5) -> dict:
    """Assert saved codes equal segment_value_encode on their original full units."""
    import seg_context_retrieval as scr
    errors = np.zeros(len(bank['length']), np.float64)
    for i, length in enumerate(np.asarray(bank['length'], np.int64)):
        c, mag, n = scr.segment_value_encode(np.asarray(bank['natural_actions'][i, :length]), coefficients=3)
        if n != int(length):
            raise AssertionError(f'unit {i}: encoded length {n} != stored {length}')
        errors[i] = np.max(np.abs(c - bank['coefficients'][i]))
        if not np.isclose(mag, float(bank['magnitude'][i]), atol=atol, rtol=atol):
            raise AssertionError(f'unit {i}: magnitude mismatch')
    if np.max(errors, initial=0.) > atol:
        raise AssertionError(f'saved coefficient mismatch max_abs={errors.max()}')
    return {'units': int(len(errors)), 'max_coefficient_abs_diff': float(errors.max(initial=0.)),
            'max_abs_tolerance': float(atol), 'exact_source': 'bank__natural_actions[:,:length]'}


def target_from_spans(actions: np.ndarray, start: int, spans: list[tuple[int, int]], *, min_tail=5):
    """Encode the future remainder to the next anchor boundary, extending short tails."""
    import seg_context_retrieval as scr
    a = np.asarray(actions, np.float32)
    t = int(start)
    if t < 0 or t >= len(a):
        raise ValueError(f'query start outside episode: {t}/{len(a)}')
    idx = next((i for i, (s, e) in enumerate(spans) if s <= t < e), None)
    if idx is None:
        raise ValueError(f'no anchor span contains query start {t}')
    end = spans[idx][1]
    j = idx + 1
    while end - t < min_tail and j < len(spans):
        end = spans[j][1]
        j += 1
    c, _, _ = scr.segment_value_encode(a[t:end], coefficients=3)
    return c.reshape(18), (t, int(end))


def targets_for_split(sp: dict, *, minimum_length=5):
    """Build the coefficient target at each query start from its own episode trace."""
    import motion_core_eval as motion
    ep_slot = np.asarray(sp['ep_slot'], np.int64)
    offsets = np.asarray(sp['ep_offset'], np.int64)
    lengths = np.asarray(sp['ep_length'], np.int64)
    starts = np.asarray(sp['start'], np.int64)
    seq = np.asarray(sp['seq_actions'], np.float32)
    targets = np.empty((len(starts), 18), np.float32)
    target_spans = np.empty((len(starts), 2), np.int64)
    for slot in np.unique(ep_slot):
        rows = np.flatnonzero(ep_slot == slot)
        off, n = int(offsets[slot]), int(lengths[slot])
        acts = seq[off:off+n]
        spans = motion.anchor_spans(acts, min_length=minimum_length)
        for row in rows:
            targets[row], target_spans[row] = target_from_spans(acts, int(starts[row]), spans,
                                                                  min_tail=minimum_length)
    return targets, target_spans


def candidate_mask(query_task, query_episode, memory_task, memory_episode):
    """Only same-episode candidates are excluded; task labels never enter the query net."""
    qt, qe = np.asarray(query_task), np.asarray(query_episode)
    mt, me = np.asarray(memory_task), np.asarray(memory_episode)
    return (qt[:, None] != mt[None, :]) | (qe[:, None] != me[None, :])


def nearest_allowed_coefficients(target, keys, query_task, query_episode, memory_task, memory_episode,
                                 *, chunk=128):
    """Precompute each training row's best legal in-batch candidate in bounded memory."""
    q=np.asarray(target,np.float32);k=np.asarray(keys,np.float32)
    qn=np.square(q).sum(-1);kn=np.square(k).sum(-1)
    result=np.empty(len(q),np.int64)
    mt,me=np.asarray(memory_task),np.asarray(memory_episode)
    qt,qe=np.asarray(query_task),np.asarray(query_episode)
    for start in range(0,len(q),chunk):
        end=min(start+chunk,len(q))
        d=qn[start:end,None]+kn[None]-2*q[start:end]@k.T
        allow=candidate_mask(qt[start:end],qe[start:end],mt,me)
        d[~allow]=np.inf
        result[start:end]=np.argmin(d,axis=1)
    return result


def _prepare_context():
    import kq2_run as kr
    kr.KF_DATA = BANK_DIR
    kr.ROOT = ROOT
    kr.OUT = OUT
    kr.EVAL = ('dev2', 'learned', 'unseen20')
    OUT.mkdir(parents=True, exist_ok=True)
    for name in SPLITS:
        dst, src = OUT / 'data' / name, KQ2 / 'data' / name
        if not src.is_dir():
            raise FileNotFoundError(src)
        if dst.exists() or dst.is_symlink():
            if not dst.is_symlink() or dst.resolve() != src.resolve():
                raise FileExistsError(f'refusing to replace KC1 split path {dst}')
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.symlink_to(src, target_is_directory=True)
    ctx = kr.Ctx('cpu', splits=SPLITS, vlm=True, out=OUT)
    target = {s: targets_for_split(ctx.splits[s]) for s in SPLITS}
    return kr, ctx, target


def _as_gpusplit(kr, sp, device, use_vlm):
    return kr.GPUSplit(sp, device, use_vlm, errors=False)


def _eval_model(model, gs, sp, target, keys, key_task, key_episode, *, random_seed=8181):
    model.eval(); device = next(model.parameters()).device
    with torch.no_grad():
        scores = []
        for start in range(0, gs.n, 128):
            ix = torch.arange(start, min(start+128, gs.n), device=device)
            q = model(gs, ix)
            scores.append(model.score(q, keys).float().cpu().numpy())
    scores = np.concatenate(scores)
    allowed = candidate_mask(sp['task'], sp['episode'], key_task, key_episode)
    scores[~allowed] = -np.inf
    order = np.argsort(-scores, axis=1, kind='stable')
    target = np.asarray(target, np.float32)
    sq = ((target[:, None, :] - np.asarray(keys.cpu() if torch.is_tensor(keys) else keys)[None])**2).sum(-1)
    top1 = order[:, 0]
    valid5 = order[:, :min(5, order.shape[1])]
    rng = np.random.default_rng(random_seed)
    random_ids = np.empty(len(target), np.int64)
    for i in range(len(target)):
        eligible = np.flatnonzero(allowed[i])
        random_ids[i] = rng.choice(eligible)
    arm_rows = np.asarray(sp['actions'], np.float32)
    mem_actions = np.asarray(memory_bank()['value_actions10'], np.float32)
    mse_action = ((mem_actions[top1, :, :7] - arm_rows[:, :, :7])**2).mean((1, 2))
    return {'rows': int(len(target)), 'top1_coeff_sq': float(sq[np.arange(len(target)), top1].mean()),
            'top5_coeff_sq_mean': float(sq[np.arange(len(target))[:, None], valid5].mean()),
            'random_coeff_sq': float(sq[np.arange(len(target)), random_ids].mean()),
            'oracle_coeff_sq': float(np.where(allowed, sq, np.inf).min(1).mean()),
            'top1_action_mse_active7': float(mse_action.mean()),
            'per_row_top1_coeff_sq': sq[np.arange(len(target)), top1].astype(np.float32).tolist(),
            'top1_ids': top1.astype(int).tolist()}


def _dev_top1(model, gs, sp, target, keys, key_task, key_episode):
    """Fast Dev checkpoint metric: only score winners and their coefficient error."""
    model.eval(); device=next(model.parameters()).device
    allow=candidate_mask(sp['task'],sp['episode'],key_task,key_episode)
    pred=[]
    with torch.no_grad():
        for start in range(0,gs.n,128):
            ix=torch.arange(start,min(start+128,gs.n),device=device)
            q=model(gs,ix)
            sc=model.score(q,keys).float().cpu().numpy()
            pred.extend(np.argmax(np.where(allow[start:start+len(sc)],sc,-np.inf),axis=1).tolist())
    pred=np.asarray(pred,np.int64); target=np.asarray(target,np.float32)
    key_np=keys.detach().cpu().numpy() if torch.is_tensor(keys) else np.asarray(keys)
    err=np.square(target-key_np[pred]).sum(-1)
    return float(err.mean())


def _load_model(kr, q_name, device, seed):
    from kq2_run import QueryNet
    use_vlm = q_name == 'iii'
    model = QueryNet('gru', 1, 1 if use_vlm else 0, use_vlm, history='w10', dct=True).to(device)
    return model


def _evaluate_dev(model, kr, ctx, targets, keys, key_task, key_episode, q_name, *, mean, std, standardize):
    gs = _as_gpusplit(kr, ctx.splits['dev2'], torch.device('cpu'), q_name == 'iii')
    raw=targets['dev2'][0]
    target=(raw-mean)/std if standardize else raw
    return _dev_top1(model,gs,ctx.splits['dev2'],target,keys,key_task,key_episode)


def _train_one(kr, ctx, targets, raw_coeff, key_task, key_episode, nearest_ids, *, q_name, standardize, tau, seed, out_dir):
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '8')))
    device = torch.device('cpu')
    keys_np, mean, std = coefficient_keys(raw_coeff, standardize=standardize)
    key_tensor = torch.as_tensor(keys_np, dtype=torch.float32, device=device)
    mem_task = np.asarray(key_task, np.int64); mem_ep = np.asarray(key_episode, np.int64)
    train_sp = ctx.splits['eh']; train_target = targets['eh'][0]
    gs = _as_gpusplit(kr, train_sp, device, q_name == 'iii')
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = _load_model(kr, q_name, device, seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    best_dev = float('inf'); best_ep = None; best_state = None; history=[]; started=time.time()
    n = len(train_target); epochs = EPOCHS; batch = 256
    for epoch in range(1, epochs+1):
        model.train(); permutation=rng.permutation(n); losses=[]
        for s in range(0,n,batch):
            bix=permutation[s:s+batch]
            ix=torch.as_tensor(bix,device=device)
            q=model(gs,ix)
            if standardize:
                target_t=torch.as_tensor((train_target[bix]-mean)/std,dtype=torch.float32,device=device)
            else:
                target_t=torch.as_tensor(train_target[bix],dtype=torch.float32,device=device)
            # Candidate pool = in-batch nearest saved keys + 256 random bank keys.
            row_allowed=candidate_mask(train_sp['task'][bix],train_sp['episode'][bix],mem_task,mem_ep)
            nearest=np.asarray(nearest_ids)[bix]
            random_ids=rng.choice(len(keys_np),size=min(RANDOM_CANDIDATES,len(keys_np)),replace=False)
            cand=np.unique(np.r_[nearest,random_ids]).astype(np.int64)
            cand_t=torch.as_tensor(cand,device=device)
            allowed=torch.as_tensor(row_allowed[:,cand],device=device,dtype=torch.bool)
            if not bool(allowed.any(-1).all()):
                raise AssertionError('candidate sampler produced a query with no episode-legal memory')
            dtrue=((target_t[:,None]-key_tensor[cand_t][None])**2).sum(-1)
            logits=model.score(q,key_tensor[cand_t])
            target_dist=torch.softmax((-dtrue/float(tau)).masked_fill(~allowed,-torch.inf),-1)
            target_dist=torch.nan_to_num(target_dist,nan=0.)
            logprob=torch.log_softmax(logits.masked_fill(~allowed,-torch.inf),-1)
            loss=-(target_dist*torch.nan_to_num(logprob,nan=0.,neginf=0.)).sum(-1).mean()
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            with torch.no_grad(): model.logit_scale.clamp_(max=float(np.log(100.)))
            losses.append(float(loss.detach()))
        dev=_evaluate_dev(model,kr,ctx,targets,key_tensor,mem_task,mem_ep,q_name,
                          mean=mean,std=std,standardize=standardize)
        rec={'epoch':epoch,'loss':float(np.mean(losses)),'dev_top1_coeff_sq':dev,'seconds':time.time()-started}
        history.append(rec)
        if dev < best_dev:
            best_dev=dev;best_ep=epoch
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if epoch in (1,4,8,12,16):
            log('train_epoch',query=q_name,standardize=standardize,tau=tau,seed=seed,**rec)
    model.load_state_dict(best_state)
    name=f"q{q_name}_std{int(standardize)}_tau{tau:g}_s{seed}"
    model_path=out_dir/f'{name}.pt'
    if model_path.exists():
        raise FileExistsError(f'refusing to overwrite partial checkpoint {model_path}')
    torch.save({'query':best_state,'cfg':{'query':q_name,'standardize':standardize,'tau':tau,'seed':seed,
        'mean':mean,'std':std,'epochs':epochs,'lr':1e-4,'candidate_pool':'in-batch nearest + random 256'}},model_path)
    row={'name':name,'query':q_name,'standardize':standardize,'tau':tau,'seed':seed,'selected_epoch':best_ep,
         'dev_top1_coeff_sq':best_dev,'history':history,'checkpoint':str(model_path)}
    (out_dir/f'{name}.json').write_text(json.dumps(row,indent=2,ensure_ascii=False)+'\n')
    return row


def _score_x1_reference(kr, ctx, targets, raw_coeff, q_name):
    """The existing K-x1 + KQ2 query, scored in KC1 coefficient space."""
    from kq2_run import QueryNet
    ckpt=KQ2/'stage2/rows'/('iii_G1X2F_s1.pt' if q_name=='iii' else 'i_noV_G1N_s1.pt')
    if not ckpt.exists():
        return {'status':'missing_reference_checkpoint','checkpoint':str(ckpt)}
    blob=torch.load(ckpt,map_location='cpu',weights_only=False); cfg=blob['cfg']
    model=QueryNet(cfg['enc'],cfg['gru_layers'],cfg['xattn'],cfg['ffn'],cfg['history']).cpu().eval()
    model.load_state_dict(blob['query'])
    keys=torch.as_tensor(np.load(KQ2/'stage1/K-x1_keys.npy'),dtype=torch.float32)
    units=memory_bank()
    result={}
    for split in ('learned','unseen20','dev2'):
        sp=ctx.splits[split]; use_vlm=q_name=='iii'; gs=_as_gpusplit(kr,sp,torch.device('cpu'),use_vlm)
        with torch.no_grad():
            sc=[]
            for st in range(0,gs.n,128):
                ix=torch.arange(st,min(st+128,gs.n)); q=model(gs,ix)
                sc.append(model.score(q,keys).numpy())
        scores=np.concatenate(sc); allow=candidate_mask(sp['task'],sp['episode'],units['task'],units['episode'])
        scores[~allow]=-np.inf; ids=np.argmax(scores,1)
        target=targets[split][0]
        coeff=np.asarray(units['coefficients']).reshape(len(units['length']),18)
        err=((target-coeff[ids])**2).sum(-1)
        act=((np.asarray(units['value_actions10'])[ids,:,:7]-np.asarray(sp['actions'])[:,:,:7])**2).mean((1,2))
        result[split]={'rows':len(ids),'top1_coeff_sq':float(err.mean()),'top1_action_mse_active7':float(act.mean()),
                       'per_row_top1_coeff_sq':err.astype(float).tolist()}
    return result


def run_offline():
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','8')))
    OUT.mkdir(parents=True,exist_ok=True); log('offline_start',device='cpu')
    bank=memory_bank(); coeff=np.asarray(bank['coefficients'],np.float32)
    audit=verify_saved_coefficients(bank)
    audit_path=OUT/'bank_audit.json'
    if audit_path.exists():
        if json.loads(audit_path.read_text()) != audit:
            raise AssertionError('saved bank audit does not match current immutable coefficients')
    else:
        audit_path.write_text(json.dumps(audit,indent=2)+'\n')
    kr,ctx,targets=_prepare_context()
    target_path=OUT/'query_targets.npz'
    target_arrays={**{f'{s}__coeff':targets[s][0] for s in SPLITS},
                   **{f'{s}__span':targets[s][1] for s in SPLITS}}
    if target_path.exists():
        old=np.load(target_path,allow_pickle=False)
        if any(not np.array_equal(old[k],v) for k,v in target_arrays.items()):
            raise AssertionError('existing KC1 target cache differs from recomputed causal targets')
    else:
        np.savez_compressed(target_path,**target_arrays)
    candidates=[]; models_dir=OUT/'models';models_dir.mkdir(exist_ok=True)
    nearest_cache={}
    for q_name in ('i','iii'):
        for standardize in (False,True):
            keys_np,mean,std=coefficient_keys(coeff,standardize=standardize)
            if standardize not in nearest_cache:
                nearest_cache[standardize]=nearest_allowed_coefficients(
                    (targets['eh'][0]-mean)/std if standardize else targets['eh'][0], keys_np,
                    ctx.splits['eh']['task'],ctx.splits['eh']['episode'],bank['task'],bank['episode'])
            nearest=nearest_cache[standardize]
            for tau in TAUS:
                for seed in SEEDS:
                    name=f"q{q_name}_std{int(standardize)}_tau{tau:g}_s{seed}"
                    meta_path=models_dir/f'{name}.json'
                    if meta_path.exists():
                        candidates.append(json.loads(meta_path.read_text()));continue
                    started=time.time()
                    row=_train_one(kr,ctx,targets,coeff,np.asarray(bank['task']),np.asarray(bank['episode']),nearest,
                                   q_name=q_name,standardize=standardize,tau=tau,seed=seed,out_dir=models_dir)
                    row['train_seconds']=time.time()-started;candidates.append(row)
    selected=min(candidates,key=lambda x:x['dev_top1_coeff_sq'])
    blob=torch.load(selected['checkpoint'],map_location='cpu',weights_only=False)
    model=_load_model(kr,selected['query'],'cpu',selected['seed']);model.load_state_dict(blob['query']);model.eval()
    keys_np,mean,std=coefficient_keys(coeff,standardize=selected['standardize'])
    keys=torch.as_tensor(keys_np,dtype=torch.float32)
    final={}
    for split in ('dev2','learned','unseen20'):
        sp=ctx.splits[split];gs=_as_gpusplit(kr,sp,torch.device('cpu'),selected['query']=='iii')
        raw_target=targets[split][0]
        target=(raw_target-mean)/std if selected['standardize'] else raw_target
        final[split]=_eval_model(model,gs,sp,target,keys,np.asarray(bank['task']),np.asarray(bank['episode']),random_seed=9181)
        # Save independent reference result and per-row paired vectors.
        final[split]['per_row_target_coeff_flat']=raw_target.astype(np.float32).tolist()
    x1={q:_score_x1_reference(kr,ctx,targets,coeff,q) for q in ('i','iii')}
    result={'status':'offline_complete','metric':'squared Euclidean error in stored 18-D DCT3 shape coefficients',
        'bank_audit':audit,'bank_units':len(coeff),'query_rows':{s:len(ctx.splits[s]['key']) for s in SPLITS},
        'settings_grid':{'standardize':[False,True],'tau':list(TAUS),'query':['i','iii'],'seed':list(SEEDS),'epochs_max':EPOCHS},
        'selection':'minimum Dev v2 top-1 coefficient squared error; no learned/unseen20 scores used to select',
        'selected':selected,'selected_results':final,'existing_kx1_kq2_reference':x1,
        'target_rule':'anchor_spans from each episode; encode current-frame suffix to next boundary, extend to >=5 actions',
        'inference_inputs':'state/action history and (iii only) current VLM tokens; no target, task ID, or pi0.5 output'}
    dest=OUT/'offline_results.json'
    if dest.exists(): raise FileExistsError(dest)
    dest.write_text(json.dumps(result,indent=2,ensure_ascii=False,default=float)+'\n')
    (OUT/'OFFLINE_DONE').write_text('offline evaluation complete\n')
    log('offline_complete',selected=selected['name'],dev2=selected['dev_top1_coeff_sq'],
        unseen20=final['unseen20']['top1_coeff_sq'],learned=final['learned']['top1_coeff_sq'])
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('cmd',choices=('offline','audit'))
    a=p.parse_args()
    if a.cmd=='audit': print(json.dumps(verify_saved_coefficients(memory_bank()),indent=2))
    else: print(json.dumps(run_offline(),indent=2,ensure_ascii=False,default=float))


if __name__=='__main__': main()
