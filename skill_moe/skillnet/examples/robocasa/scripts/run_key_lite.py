"""Offline context/history key comparison using only cached RoboCasa features."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn

ROOT = Path(os.environ.get('SKILLNET_REPO_ROOT', Path(__file__).resolve().parents[5]))
STATE = Path(os.environ.get('SKILLNET_STATE_DIR', Path.home() / '.skillnet-state'))
OUT = Path(os.environ.get('ROBOCASA_KEYLITE_OUT', STATE / 'outputs/robocasa-memory/key-lite-01'))
CTX = Path(os.environ.get('ROBOCASA_CTX_OUT', STATE / 'outputs/robocasa-memory/ctx-01'))
SOURCE = Path(os.environ.get('ROBOCASA_MEMORY_OUT', STATE / 'outputs/robocasa-memory/memory-01/policy-order-v2'))
sys.path.insert(0, str(ROOT / 'skill_moe/skillnet/examples/memory_pilot/robocasa'))
import experiment as ex
import expert

SEED = 20261111
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '8')))
if DEVICE == 'cuda':
    torch.cuda.set_per_process_memory_fraction(.20)


def log(stage, **kw):
    rec = {'time': time.time(), 'stage': stage, **kw}
    with (OUT / 'progress.jsonl').open('a') as f:
        f.write(json.dumps(rec) + '\n')
    print(json.dumps(rec), flush=True)


class RankNet(nn.Module):
    def __init__(self, qdim, kdim):
        super().__init__()
        self.query = nn.Sequential(nn.Linear(qdim, 512), nn.GELU(), nn.Linear(512, 512), nn.GELU(), nn.Linear(512, 256))
        self.key = nn.Sequential(nn.Linear(kdim, 512), nn.GELU(), nn.Linear(512, 256))
        self.log_scale = nn.Parameter(torch.tensor(float(np.log(20.))))

    def forward(self, query, keys):
        return nn.functional.normalize(self.query(query), dim=-1) @ nn.functional.normalize(self.key(keys), dim=-1).T * self.log_scale.exp()


def load_data():
    z = np.load(CTX / 'cache/rows.npz', allow_pickle=False)
    pool = dict(np.load(CTX / 'cache/pool.npz', allow_pickle=False))
    p5 = np.load(CTX / 'p5_baselines/p5_baseline_rows.npz', allow_pickle=False)
    teacher = np.load(CTX / 'cache/train_teacher_final.npz', allow_pickle=False)
    feat = np.asarray(np.load(CTX / 'cache/prefix_mean_fp16.npy', mmap_mode='r'), np.float32)
    keys = dict(z)
    error_path = OUT / 'heldout_type_errors.npy'
    if error_path.exists():
        errors = np.load(error_path, mmap_mode='r')
        if errors.shape != (len(z['heldout_task']), len(pool['task'])):
            raise AssertionError(f'heldout error alignment mismatch: {errors.shape}')
    else:
        # The generic memory cache's held-out matrix covers only 1,809 rows;
        # this experiment uses the full 2,892-row Context cache. Recompute the
        # matching matrix from those held-out targets and the shared U pool.
        targets = np.asarray(z['heldout_target'], np.float32)
        qshape = np.stack([ex.encode_shape(a)[0] for a in targets])
        errors = ex.shape_error_matrix(qshape, pool['shape'])
        np.save(error_path, np.asarray(errors, np.float32))
    if teacher['error'].shape != (len(z['memory_task']), len(pool['task'])):
        raise AssertionError(f'train teacher alignment mismatch: {teacher["error"].shape}')
    for name, arr in (('task', 'heldout_task'), ('episode', 'heldout_local_episode'), ('frame', 'heldout_frame')):
        if not np.array_equal(p5[name], z[arr]):
            raise AssertionError(f'P5 cached row order differs: {name}')
    if len(pool['task']) != 7062 or len(z['heldout_task']) != 2892:
        raise AssertionError('unexpected cached candidate/evaluation row count')
    row = np.arange(len(z['heldout_task']))
    cached_p5 = np.asarray(p5['type_error_correct'][0], np.float32)
    recomputed_p5 = np.asarray(errors[row, np.asarray(p5['pick_correct'][0], np.int64)], np.float32)
    if not np.allclose(cached_p5, recomputed_p5, rtol=1e-5, atol=1e-7):
        raise AssertionError(f'full heldout error recomputation does not match P5 cache; max={np.max(np.abs(cached_p5-recomputed_p5))}')
    if len(z['memory_task']) != 11314:
        log('cache_count_note', observed_memory_queries=int(len(z['memory_task'])), user_expected_memory_side=17221)
    if len(np.unique(pool['task'])) != 6:
        raise AssertionError('expected six task ids in U candidate pool')
    log('loaded', candidates=len(pool['task']), memory_queries=len(z['memory_task']), heldout_queries=len(z['heldout_task']), device=DEVICE)
    return z, pool, p5, errors, teacher, feat


def build_history_rows(z, split):
    hist = np.asarray(z[f'{split}_hist'], np.float32)
    # Cache histories are in LeRobot channel order; key input follows policy/fork order.
    hist = ex.action_storage_to_policy(hist)
    ep = np.asarray(z[f'{split}_episode'], np.int64)
    task = np.asarray(z[f'{split}_task'], np.int64)
    frame = np.asarray(z[f'{split}_frame'], np.int64)
    mask = np.zeros((len(frame), 10), bool)
    for i, t in enumerate(frame):
        valid = min(max(int(t), 0), 10)
        if valid:
            mask[i, 10-valid:] = True
    lookup = {(int(e), int(f)): i for i, (e, f) in enumerate(zip(ep, frame, strict=True))}
    task_eps: dict[int, list[int]] = {}
    episode_rows: dict[int, list[int]] = {}
    for t, e in zip(task, ep, strict=True):
        task_eps.setdefault(int(t), []).append(int(e))
    for i, e in enumerate(ep):
        episode_rows.setdefault(int(e), []).append(i)
    task_eps = {t: sorted(set(v)) for t, v in task_eps.items()}
    shifted = {}
    for delta in (-20, -10, 10, 20):
        ids = np.full(len(frame), -1, np.int64)
        for i, (e, f) in enumerate(zip(ep, frame, strict=True)):
            ids[i] = lookup.get((int(e), int(f) + delta), -1)
        shifted[delta] = ids
    replaced = np.full(len(frame), -1, np.int64)
    rng = np.random.default_rng(SEED + (1 if split == 'memory' else 2))
    for i, (t, e, f) in enumerate(zip(task, ep, frame, strict=True)):
        others = [x for x in task_eps[int(t)] if x != int(e)]
        if others:
            other = others[int(rng.integers(len(others)))]
            candidates = episode_rows.get(other, [])
            if candidates:
                replaced[i] = min(candidates, key=lambda j: abs(int(frame[j])-int(f)))
    return {'hist': hist, 'mask': mask, 'task': task, 'episode': ep, 'frame': frame,
            'shifted': shifted, 'replaced': replaced}


def corrupt_history(h, rows, ids, rng):
    """Per query: 50% empty, 25% deliberately corrupted, 25% correct."""
    n = len(ids)
    out = h['hist'][ids].copy()
    mask = h['mask'][ids].copy()
    u = rng.random(n)
    no = u < .50
    bad = (u >= .50) & (u < .75)
    out[no] = 0.; mask[no] = False
    bad_ids = np.flatnonzero(bad)
    for b in bad_ids:
        source = int(ids[b])
        option = int(rng.integers(5))
        src = int(h['shifted'][(-20, -10, 10, 20)[option]][source]) if option < 4 else int(h['replaced'][source])
        if src < 0:
            out[b] = 0.; mask[b] = False
        else:
            out[b] = h['hist'][src]
            mask[b] = h['mask'][src]
    return out, mask


def context_inputs(z, feat, fit_ids, candidate=False):
    if candidate:
        ix = np.asarray(z['candidate_context'], np.int64)
        state = np.asarray(z['candidate_state'], np.float32)
    else:
        ix = np.asarray(z['memory_context'], np.int64)[fit_ids]
        state = np.asarray(z['memory_state'], np.float32)[fit_ids]
    mean = feat[ix].mean(0).astype(np.float32)
    std = np.maximum(feat[ix].std(0), .01).astype(np.float32)
    sm = state.mean(0).astype(np.float32)
    ss = np.maximum(state.std(0), 1e-4).astype(np.float32)
    return mean, std, sm, ss


def assemble_context(z, feat, norm, split):
    mean, std, sm, ss = norm
    cix = np.asarray(z[f'{split}_context'], np.int64)
    state = np.asarray(z[f'{split}_state'], np.float32)
    q = np.concatenate([(feat[cix]-mean)/std, (state-sm)/ss], axis=1).astype(np.float32)
    cix = np.asarray(z['candidate_context'], np.int64)
    cs = np.asarray(z['candidate_state'], np.float32)
    k = np.concatenate([(feat[cix]-mean)/std, (cs-sm)/ss], axis=1).astype(np.float32)
    return q, k


def evaluate_scores(scores, errors, allowed, tasks, frames, label):
    scores = np.asarray(scores, np.float32).copy()
    scores[~allowed] = -np.inf
    picks = scores.argmax(1)
    picked_error = errors[np.arange(len(errors)), picks]
    order5 = np.argpartition(-scores, 5-1, axis=1)[:, :5]
    order5_scores = np.take_along_axis(scores, order5, axis=1)
    order5 = np.take_along_axis(order5, np.argsort(-order5_scores, axis=1, kind='stable'), axis=1)
    oracle_order = np.argsort(np.where(allowed, errors, np.inf), axis=1, kind='stable')
    oracle_top1pct = np.zeros_like(allowed)
    for i in range(len(errors)):
        n = max(1, int(np.ceil(.01 * allowed[i].sum())))
        oracle_top1pct[i, oracle_order[i, :n]] = True
    hit5 = oracle_top1pct[np.arange(len(errors))[:, None], order5].any(1)
    f0 = np.asarray(frames) == 0
    return {'label': label, 'n': len(picks), 'top1_type_error': float(picked_error.mean()),
            'frame0_type_error': float(picked_error[f0].mean()), 'r_at_5_oracle_top1pct': float(hit5.mean()),
            'frame0_r_at_5_oracle_top1pct': float(hit5[f0].mean()),
            'task_match_diagnostic': float((np.asarray(tasks) == np.asarray(pool_task_for_metrics)[picks]).mean()),
            'pick': picks.astype(np.int32), 'top5': order5.astype(np.int32), 'row_error': picked_error.astype(np.float32)}


pool_task_for_metrics = None


def train_ranker(z, feat, pool, teacher, mem_hist, *, input_history, exclude_task=None, tag='model', max_epochs=12):
    task = np.asarray(z['memory_task'], np.int64)
    eps = np.asarray(z['memory_episode'], np.int64)
    fit_rows = np.ones(len(task), bool) if exclude_task is None else task != int(exclude_task)
    eligible = np.flatnonzero(fit_rows)
    if not len(eligible): raise ValueError('empty LOTO training set')
    unique_ep = np.unique(eps[eligible])
    rng = np.random.default_rng(SEED + (0 if exclude_task is None else 100 + int(exclude_task)) + (7 if input_history else 0))
    perm = rng.permutation(unique_ep)
    val_eps = set(map(int, perm[:max(1, len(perm)//5)]))
    val = np.asarray([int(e) in val_eps for e in eps[eligible]])
    fit_ids = eligible[~val]; val_ids = eligible[val]
    if not len(val_ids): raise ValueError('empty validation episodes')
    norm = context_inputs(z, feat, fit_ids)
    qctx_mem, keys = assemble_context(z, feat, norm, 'memory')
    qctx_eval, _ = assemble_context(z, feat, norm, 'heldout')
    # The normalization above is fit on the eligible training queries only.
    if input_history:
        hist = mem_hist['hist']; hmask = mem_hist['mask']
        hmean = hist[fit_ids][hmask[fit_ids]].mean(0) if hmask[fit_ids].any() else np.zeros(12, np.float32)
        hstd = hist[fit_ids][hmask[fit_ids]].std(0) if hmask[fit_ids].any() else np.ones(12, np.float32)
        hstd = np.maximum(hstd, 1e-4).astype(np.float32)
        qhist = ((hist-hmean)/hstd).astype(np.float32) * hmask[..., None]
        qmem = np.concatenate([qctx_mem, qhist.reshape(len(hist), -1), hmask.astype(np.float32)], axis=1).astype(np.float32)
        qeval = qctx_eval
        hnorm = {'mean':hmean,'std':hstd}
    else:
        qmem, qeval = qctx_mem, qctx_eval
        qhist = None; hnorm = None
    errors = np.asarray(teacher['error'], np.float32)
    allowed = np.asarray(teacher['allowed'], bool)
    train_err = errors[fit_ids]
    train_allowed = allowed[fit_ids]
    gap = np.mean(np.where(train_allowed, train_err, 0.), axis=1) - np.min(np.where(train_allowed, train_err, np.inf), axis=1)
    tau = max(.1 * float(np.median(gap)), 1e-8)
    torch.manual_seed(SEED + (0 if exclude_task is None else 900 + int(exclude_task)) + (31 if input_history else 0))
    net = RankNet(qmem.shape[1], keys.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    kt = torch.as_tensor(keys, dtype=torch.float32, device=DEVICE)
    et = torch.as_tensor(errors, dtype=torch.float32, device=DEVICE)
    at = torch.as_tensor(allowed, dtype=torch.bool, device=DEVICE)
    qct = torch.as_tensor(qctx_mem, dtype=torch.float32, device=DEVICE)
    qvt = torch.as_tensor(qctx_eval, dtype=torch.float32, device=DEVICE)
    best = (float('inf'), None, 0); curve=[]; stale=0
    batch = 128
    for epoch in range(1, max_epochs+1):
        net.train(); order = rng.permutation(fit_ids); losses=[]
        if input_history:
            corrupted, cmask = corrupt_history(mem_hist, z, order, rng)
            ch = ((corrupted-hnorm['mean'])/hnorm['std']).astype(np.float32) * cmask[...,None]
            xepoch = np.concatenate([qctx_mem[order], ch.reshape(len(order),-1), cmask.astype(np.float32)],axis=1).astype(np.float32)
        else:
            xepoch = None
        for st in range(0, len(order), batch):
            ids = order[st:st+batch]
            bx = (torch.as_tensor(xepoch[st:st+batch],dtype=torch.float32,device=DEVICE) if input_history
                  else qct[torch.as_tensor(ids,dtype=torch.long,device=DEVICE)])
            ii = torch.as_tensor(ids,dtype=torch.long,device=DEVICE)
            logits=net(bx,kt).masked_fill(~at[ii],-1e9)
            target=torch.softmax((-et[ii]/tau).masked_fill(~at[ii],-1e9),dim=-1)
            loss=-(target*torch.log_softmax(logits,dim=-1)).sum(-1).mean()
            opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
            losses.append(float(loss.detach().cpu()))
        net.eval()
        with torch.no_grad():
            vals=[]
            for st in range(0,len(val_ids),batch):
                ids=val_ids[st:st+batch]
                vx=(torch.as_tensor(qmem[ids],dtype=torch.float32,device=DEVICE) if input_history
                    else qct[torch.as_tensor(ids,dtype=torch.long,device=DEVICE)])
                vals.append(net(vx,kt).masked_fill(~at[torch.as_tensor(ids,device=DEVICE)],-1e9).cpu().numpy())
            vs=np.concatenate(vals,axis=0);vp=vs.argmax(1)
        metric=float(errors[val_ids,vp].mean())
        curve.append({'epoch':epoch,'train_ce':float(np.mean(losses)),'dev_top1_type_error':metric})
        log('train_epoch', model=tag, epoch=epoch, train_ce=curve[-1]['train_ce'], dev_top1_type_error=metric)
        if metric < best[0]:
            best=(metric,{k:v.detach().cpu().clone() for k,v in net.state_dict().items()},epoch);stale=0
        else:
            stale+=1
            if stale >= 3 and epoch >= 5:
                break
    ck={'state':best[1],'norm':norm,'hist_norm':hnorm,'q_dim':qmem.shape[1],'k_dim':keys.shape[1],
        'input_history':input_history,'tau':tau,'best_epoch':best[2],'dev_top1_type_error':best[0],
        'dev_episodes':sorted(val_eps),'curve':curve,'exclude_task':exclude_task}
    return net,ck


def predict_ranker(ck, z, feat, pool, *, split, hist_data=None, history_condition='correct', exclude_episode=True):
    qctx, keys = assemble_context(z, feat, ck['norm'], split)
    if ck['input_history']:
        h = hist_data
        raw = h['hist'].copy(); mask=h['mask'].copy()
        if history_condition == 'none': raw[:]=0; mask[:]=False
        elif history_condition.startswith('shift_'):
            delta=int(history_condition.split('_')[1]); original=raw.copy(); om=mask.copy()
            for i,j in enumerate(h['shifted'][delta]):
                if j<0: raw[i]=0;mask[i]=False
                else: raw[i]=original[j];mask[i]=om[j]
        elif history_condition == 'replace':
            original=raw.copy();om=mask.copy()
            for i,j in enumerate(h['replaced']):
                if j<0:raw[i]=0;mask[i]=False
                else:raw[i]=original[j];mask[i]=om[j]
        hnorm=ck['hist_norm']; hs=((raw-hnorm['mean'])/hnorm['std']).astype(np.float32)*mask[...,None]
        q=np.concatenate([qctx,hs.reshape(len(hs),-1),mask.astype(np.float32)],axis=1).astype(np.float32)
    else:q=qctx
    net=RankNet(ck['q_dim'],ck['k_dim']);net.load_state_dict(ck['state']);net.to(DEVICE).eval()
    with torch.no_grad():
        kt=torch.as_tensor(keys,dtype=torch.float32,device=DEVICE)
        parts=[]
        for st in range(0,len(q),128):
            parts.append(net(torch.as_tensor(q[st:st+128],dtype=torch.float32,device=DEVICE),kt).cpu().numpy())
    scores=np.concatenate(parts,axis=0)
    if exclude_episode:
        qep=np.asarray(z[f'{split}_episode'],np.int64)
        ce=np.asarray(z['candidate_episode'],np.int64)
        scores[qep[:,None]==ce[None,:]]=-np.inf
    return scores


def orientation_rerank(scores, errors, z, pool, feat_k=(5,20)):
    offsets=np.asarray(pool['action_offsets'],np.int64);flat=np.asarray(pool['actions_flat'],np.float32)
    refs=np.asarray(z['eval_pi05_actions'],np.float32) # stored LeRobot order
    lens=np.clip(np.asarray(pool['length'],np.int64),1,50)
    out={}
    masked=scores.copy();masked[np.asarray(z['heldout_episode'])[:,None]==np.asarray(pool['global_episode'])[None,:]]=-np.inf
    f0=np.asarray(z['heldout_frame'])==0
    for k in feat_k:
        picks=np.empty(len(refs),np.int32)
        top_r5=np.argpartition(-masked,4,axis=1)[:,:5]
        order=np.argsort(-np.take_along_axis(masked,top_r5,axis=1),axis=1,kind='stable')
        top_r5=np.take_along_axis(top_r5,order,axis=1)
        for i in range(len(refs)):
            cand=np.argpartition(-masked[i],k-1)[:k]
            cand=cand[np.argsort(-masked[i,cand],kind='stable')]
            best=-np.inf;best_id=int(cand[0])
            for j in cand:
                L=int(lens[j]);a=flat[offsets[j]:offsets[j+1],5:8][:L]
                r=refs[i,:L,5:8]
                den=float(np.linalg.norm(a)*np.linalg.norm(r))
                cos=float(np.sum(a*r)/max(den,1e-12))
                if cos>best:best,best_id=cos,int(j)
            picks[i]=best_id
        e=np.asarray(errors[np.arange(len(refs)),picks],np.float32)
        best1=np.argsort(np.where(masked>-np.inf,errors,np.inf),axis=1,kind='stable')
        oracle_top=np.zeros_like(masked,dtype=bool)
        for i in range(len(refs)):
            m=np.isfinite(masked[i]);n=max(1,int(np.ceil(.01*m.sum())))
            oracle_top[i,best1[i,:n]]=True
        top5_hit=oracle_top[np.arange(len(refs))[:,None],top_r5].any(1)
        out[f'context_top{k}_direction_rerank']={'n':len(e),'top1_type_error':float(e.mean()),
            'frame0_type_error':float(e[f0].mean()),'r_at_5_oracle_top1pct':float(top5_hit.mean()),
            'frame0_r_at_5_oracle_top1pct':float(top5_hit[f0].mean()),'pick':picks}
    return out


def p5_scores_for_history(z, pool, hist_data, condition, *, device=DEVICE):
    p5source=np.load(CTX/'p5_baselines/p5_baseline_rows.npz',allow_pickle=False)
    # Existing three seeds and baseline row-level picks are retained as the primary P5 results.
    # For R@5 compute exact score rankings from those frozen checkpoints.
    from experiment import _features
    h=hist_data['hist'].copy();mask=hist_data['mask'].copy()
    if condition=='none':h[:]=0;mask[:]=False
    elif condition.startswith('shift_'):
        delta=int(condition.split('_')[1]);orig=h.copy();om=mask.copy()
        for i,j in enumerate(hist_data['shifted'][delta]):
            if j<0:h[i]=0;mask[i]=False
            else:h[i]=orig[j];mask[i]=om[j]
    elif condition=='replace':
        orig=h.copy();om=mask.copy()
        for i,j in enumerate(hist_data['replaced']):
            if j<0:h[i]=0;mask[i]=False
            else:h[i]=orig[j];mask[i]=om[j]
    # p5 expects stored action/state order.
    q={'hist':ex.action_policy_to_storage(h), 'state':np.asarray(z['heldout_storage_state'],np.float32)}
    values=[]
    for seed in range(3):
        net,ck=expert._load_net(SOURCE/f'models/P5_U_seed{seed}.pt',pool,device=device)
        x=_features(q,ck['norm'])
        x[:,:120]=(x[:,:120].reshape(len(x),10,12)*mask[...,None]).reshape(len(x),120)
        with torch.no_grad():
            s=net(torch.as_tensor(x,dtype=torch.float32,device=device),torch.as_tensor(pool['key'],dtype=torch.float32,device=device)).cpu().numpy()
        allowed=np.asarray(z['heldout_episode'])[:,None] != np.asarray(pool['global_episode'])[None,:]
        s[~allowed]=-np.inf
        values.append(s)
    return np.mean(values,axis=0),p5source


def metrics_from_pick(pick, errors, allowed, tasks, pool, frames, label):
    n=len(pick);e=np.asarray(errors[np.arange(n),pick]);f0=np.asarray(frames)==0
    # Recall@5 here measures whether the score top-5 includes an oracle top-1% unit;
    # caller may pass a full score order in the separate R@5 fields.
    return {'condition':label,'n':n,'top1_type_error':float(e.mean()),'frame0_type_error':float(e[f0].mean()),
            'task_match_diagnostic':float((np.asarray(tasks)==np.asarray(pool['task'])[pick]).mean()),
            'pick':np.asarray(pick,np.int32),'row_error':e.astype(np.float32)}


def loto(z, pool, teacher, feat, mem_hist, eval_hist, errors):
    result=[]
    mem_task=np.asarray(z['memory_task']);eval_task=np.asarray(z['heldout_task'])
    qsrc=dict(np.load(SOURCE/'queries/U_memory.npz',allow_pickle=False))
    psrc=dict(np.load(SOURCE/'pools/U_memory.npz',allow_pickle=False))
    if not (np.array_equal(qsrc['task'],psrc['task']) and np.array_equal(qsrc['episode'],psrc['episode']) and np.array_equal(qsrc['frame'],psrc['frame'])):
        raise AssertionError('source P5 memory query and pool alignment mismatch')
    for key in ('task','episode','global_episode','frame'):
        if not np.array_equal(np.asarray(pool[key]), np.asarray(psrc[key])):
            raise AssertionError(f'P5 candidate pool differs from Context pool at {key}')
    for held_task in sorted(map(int,np.unique(eval_task))):
        ids=np.flatnonzero(eval_task==held_task)
        fold={'task_id':held_task,'n_eval':len(ids)}
        for hist_on,label in ((False,'context'),(True,'context_history')):
            net,ck=train_ranker(z,feat,pool,teacher,mem_hist,input_history=hist_on,exclude_task=held_task,
                                tag=f'loto-{label}-task{held_task}',max_epochs=9)
            scores=predict_ranker(ck,z,feat,pool,split='heldout',hist_data=eval_hist,history_condition='correct')
            allowed=np.asarray(z['heldout_episode'])[:,None]!=np.asarray(pool['global_episode'])[None,:]
            pick=scores.argmax(1);e=np.asarray(errors[np.arange(len(errors)),pick])
            score=scores[ids].copy();score[~allowed[ids]]=-np.inf
            err_sub=np.asarray(errors[ids]);pick_sub=score.argmax(1);e_sub=err_sub[np.arange(len(ids)),pick_sub]
            fold[label]={'top1_type_error':float(e_sub.mean()),'task_match':float((pool['task'][pick[ids]]==held_task).mean()),'best_epoch':ck['best_epoch']}
            d=OUT/'loto'/f'{label}-task{held_task}.pt';d.parent.mkdir(parents=True,exist_ok=True);torch.save(ck,d)
        # Train one P5 model with all queries from this task excluded; query targets remain type-only.
        held_pairs=[(int(t),int(e)) for t,e in zip(qsrc['task'],qsrc['episode'],strict=True) if int(t)==held_task]
        p5path=OUT/'loto'/f'p5-task{held_task}.pt'
        p5ck=expert._fit_p5_fold(qsrc,psrc,held_pairs,output=p5path,seed=SEED+held_task,
                                 epochs=8,device=DEVICE)
        net,pck=expert._load_net(p5path,psrc,device=DEVICE)
        from experiment import _features
        hstore=np.asarray(z['heldout_hist'],np.float32)[ids]
        sstore=np.asarray(z['heldout_storage_state'],np.float32)[ids]
        x=_features({'hist':hstore,'state':sstore},pck['norm'])
        x[:,:120]=(x[:,:120].reshape(len(x),10,12)*eval_hist['mask'][ids][...,None]).reshape(len(x),120)
        with torch.no_grad(): ps=net(torch.as_tensor(x,dtype=torch.float32,device=DEVICE),torch.as_tensor(psrc['key'],dtype=torch.float32,device=DEVICE)).cpu().numpy()
        allow=np.asarray(z['heldout_episode'])[ids,None]!=np.asarray(pool['global_episode'])[None,:]
        ps[~allow]=-np.inf;pp=ps.argmax(1);ee=np.asarray(errors[ids]);pe=ee[np.arange(len(ids)),pp]
        fold['p5_retrained']={'top1_type_error':float(pe.mean()),'task_match':float((np.asarray(pool['task'])[pp]==held_task).mean()),'best_epoch':p5ck['summary']['best_epoch']}
        rng=np.random.default_rng(SEED+held_task*101);rp=np.empty(len(ids),np.int32)
        for j,i in enumerate(ids):
            opts=np.flatnonzero(np.asarray(pool['global_episode'])!=int(z['heldout_episode'][i]))
            rp[j]=int(rng.choice(opts))
        re=np.asarray(errors[ids])[np.arange(len(ids)),rp]
        fold['random']={'top1_type_error':float(re.mean())}
        result.append(fold)
        log('loto_fold_complete',task_id=held_task,context=fold['context']['top1_type_error'],context_history=fold['context_history']['top1_type_error'],p5=fold['p5_retrained']['top1_type_error'],random=fold['random']['top1_type_error'])
        (OUT/'loto_progress.json').write_text(json.dumps(result,indent=2)+'\n')
        write_report(OUT)
    return result


def main():
    global pool_task_for_metrics
    z,pool,p5,errors,teacher,feat=load_data(); pool_task_for_metrics=np.asarray(pool['task'])
    mem_hist=build_history_rows(z,'memory'); eval_hist=build_history_rows(z,'heldout')
    allowed=np.asarray(z['heldout_episode'])[:,None] != np.asarray(pool['global_episode'])[None,:]
    scores={}
    # Existing Context checkpoint, frozen, with its original all-task normalization.
    ck=torch.load(CTX/'context_model.pt',map_location='cpu',weights_only=False)
    norm=(ck['norm']['feature_mean'],ck['norm']['feature_std'],ck['norm']['state_mean'],ck['norm']['state_std'])
    qctx,keys=assemble_context(z,feat,norm,'heldout');cctx,_=assemble_context(z,feat,norm,'candidate')
    net=RankNet(ck['input_dim'],ck['key_dim']);net.load_state_dict(ck['state'],strict=False);net.to(DEVICE).eval()
    with torch.no_grad():
        vals=[];kt=torch.as_tensor(cctx,dtype=torch.float32,device=DEVICE)
        for st in range(0,len(qctx),128):vals.append(net(torch.as_tensor(qctx[st:st+128],dtype=torch.float32,device=DEVICE),kt).cpu().numpy())
    scores['Context']=np.concatenate(vals);scores['Context'][~allowed]=-np.inf
    all_results=[]
    frame=np.asarray(z['heldout_frame']);task=np.asarray(z['heldout_task'])
    context_pick=scores['Context'].argmax(1)
    r=metrics_from_pick(context_pick,errors,allowed,task,pool,frame,'Context_existing')
    r['r_at_5_oracle_top1pct']=float(json.loads((CTX/'evaluation.json').read_text())['R_at_5_oracle_best_1pct'])
    all_results.append(r)
    # Existing P5 3-seed baselines: cache exactly matched condition rows.
    p5row=p5
    p5_summary={}
    for cond in ('correct','none','shift_-20','shift_-10','shift_+10','shift_+20','replace'):
        e=np.asarray(p5row[f'type_error_{cond}'],np.float32)
        f0=frame==0
        p5_summary[cond]={'n':len(frame),'top1_type_error':float(e.mean()),'frame0_type_error':float(e[:,f0].mean()),
                          'seed_mean':e.mean(1).tolist(),'seed_frame0_mean':e[:,f0].mean(1).tolist()}
    # Context direction rerank K=5 and 20 (the exact Context top-K set is used).
    rerank=orientation_rerank(scores['Context'],errors,z,pool)
    for name,val in rerank.items():
        all_results.append({'condition':name,**{k:v for k,v in val.items() if k not in ('pick',)}})
    # Train the history-aware Context key on all memory tasks, and evaluate each prescribed history corruption.
    histnet,hck=train_ranker(z,feat,pool,teacher,mem_hist,input_history=True,tag='context-history-full',max_epochs=12)
    torch.save(hck,OUT/'context_history.pt')
    hist_results={}
    for cond in ('correct','none','shift_-20','shift_-10','shift_+10','shift_+20','replace'):
        ss=predict_ranker(hck,z,feat,pool,split='heldout',hist_data=eval_hist,history_condition=cond)
        ss[~allowed]=-np.inf
        met=evaluate_scores(ss,errors,allowed,task,frame,f'Context+history/{cond}')
        hist_results[cond]={k:v for k,v in met.items() if k not in ('pick','top5','row_error')}
        all_results.append({k:v for k,v in met.items() if k not in ('pick','top5','row_error')})
    # Deterministic random ceiling reference and P5 top-5 diagnostic under true history.
    rng=np.random.default_rng(SEED+5);random_pick=np.empty(len(frame),np.int32)
    for i in range(len(frame)):
        random_pick[i]=rng.choice(np.flatnonzero(allowed[i]))
    random_error=np.asarray(errors[np.arange(len(frame)),random_pick])
    # Top-5 P5 recall is calculated for the correct-history condition by exact frozen model scores.
    p5_scores,_=p5_scores_for_history(z,pool,eval_hist,'correct')
    p5scores=p5_scores.copy();p5scores[~allowed]=-np.inf
    p5top=np.argpartition(-p5scores,4,axis=1)[:,:5]
    p5top=np.take_along_axis(p5top,np.argsort(-np.take_along_axis(p5scores,p5top,axis=1),axis=1,kind='stable'),axis=1)
    oracle_top=np.zeros_like(allowed)
    for i in range(len(frame)):
        ids=np.flatnonzero(allowed[i]); n=max(1,int(np.ceil(.01*len(ids))))
        best=ids[np.argsort(errors[i,ids],kind='stable')[:n]];oracle_top[i,best]=True
    p5_r5=float(oracle_top[np.arange(len(frame))[:,None],p5top].any(1).mean())
    (OUT/'main_results.json').write_text(json.dumps({'n_heldout':len(frame),'memory_queries':len(z['memory_task']),
       'candidate_units':len(pool['task']),'Context':{k:v for k,v in r.items() if k not in ('pick','row_error')},
       'Context+history':hist_results,'P5_existing_three_seed':p5_summary,
       'Context_direction_rerank':{k:{a:b for a,b in v.items() if a!='pick'} for k,v in rerank.items()},
       'P5_correct_history_R_at_5':p5_r5,'random':{'top1_type_error':float(random_error.mean()),'frame0_type_error':float(random_error[frame==0].mean())},
       'data_note':'ctx-01 cache contains 11,314 memory stride-10 query rows (not 17,221) and 7,062 U units; held-out 2,892 rows.'},indent=2)+'\n')
    log('main_complete',context_history_best_epoch=hck['best_epoch'],context_history_dev_error=hck['dev_top1_type_error'])
    write_report(OUT)
    # Leave-one-task-out training: task labels only define the held-out fold, never the ranking target.
    loto_result=loto(z,pool,teacher,feat,mem_hist,eval_hist,errors)
    (OUT/'results.json').write_text(json.dumps({'main':json.loads((OUT/'main_results.json').read_text()),'loto':loto_result},indent=2)+'\n')
    write_report(OUT)
    log('complete',loto_folds=len(loto_result))


def write_report(out):
    d=json.loads((out/'main_results.json').read_text())
    loto=json.loads((out/'loto_progress.json').read_text()) if (out/'loto_progress.json').exists() else []
    done = len(loto) == 6 and len(json.loads((out/'loto_progress.json').read_text())) == 6
    lines=[('実装: 完了' if done else '実装: 未完了'),'','# RM-KEY-LITE：文脈キーと履歴キーのオフライン比較','','## 主結果','',
           f"対象は保留120デモの {d['n_heldout']} stride-10 窓。候補は6タスクの未統合 U {d['candidate_units']}単位。キャッシュ内の memory query は {d['memory_queries']}行（依頼文の約17,221行より少ない。既存キャッシュを再利用したため追加行を生成していない）。保留行は採点のみ。閉ループ・Expert は未実施。",'',
           '|手法 / 履歴条件|top-1 型誤差|frame0|R@5 oracle top1%|','|---|---:|---:|---:|']
    p5=d['P5_existing_three_seed']
    for cond,m in d['Context+history'].items():
        lines.append(f"|Context+history / {cond}|{m['top1_type_error']:.6f}|{m['frame0_type_error']:.6f}|{m['r_at_5_oracle_top1pct']:.3f}|")
    c=d['Context'];lines.append(f"|Context既存|{c['top1_type_error']:.6f}|{c['frame0_type_error']:.6f}|{c['r_at_5_oracle_top1pct']:.3f}|")
    for cond,m in p5.items():
        r5 = f"{d['P5_correct_history_R_at_5']:.3f}" if cond == 'correct' else '—'
        lines.append(f"|P5既存3seed / {cond}|{m['top1_type_error']:.6f}|{m['frame0_type_error']:.6f}|{r5}|")
    for name,m in d['Context_direction_rerank'].items():
        lines.append(f"|{name}|{m['top1_type_error']:.6f}|{m['frame0_type_error']:.6f}|{m['r_at_5_oracle_top1pct']:.3f}|")
    lines += ['', '## Leave-one-task-out','', '|外したtask id|Context|Context+history|P5再学習|random|','|---:|---:|---:|---:|---:|']
    for x in loto:
        task_name = ex.TASKS[int(x['task_id'])]
        lines.append(f"|{task_name} (id {x['task_id']})|{x['context']['top1_type_error']:.6f}|{x['context_history']['top1_type_error']:.6f}|{x['p5_retrained']['top1_type_error']:.6f}|{x['random']['top1_type_error']:.6f}|")
    lines += ['', '## 判断','']
    if len(loto)==6:
        ctx_l=np.mean([x['context']['top1_type_error'] for x in loto]);hist_l=np.mean([x['context_history']['top1_type_error'] for x in loto]);p5_l=np.mean([x['p5_retrained']['top1_type_error'] for x in loto]);rand_l=np.mean([x['random']['top1_type_error'] for x in loto])
        lines.append(f"LOTO平均は Context {ctx_l:.6f}、Context+history {hist_l:.6f}、再学習P5 {p5_l:.6f}、random {rand_l:.6f}。")
    else:
        lines.append(f"LOTO は {len(loto)}/6 task まで完了。未完 fold はこの表から除外しており、全タスクの比較結論は保留。")
    if len(loto)==6:
        lines.append('値の小さいキーを候補にするが、今回はオフライン型誤差であり成功率の根拠ではない。改善が履歴条件で変わらない場合は、履歴追加の価値が確認できたとは扱わない。')
    lines += ['', 'P5 は既存3 seedのキャッシュ結果をそのまま再利用。Context+history は今回の固定seed・listwise soft target MLPで学習し、履歴を50%ゼロ、25%破損（同デモ±10/±20または別エピソード）、25%正しい履歴とした。LOTOの候補プールは常に全6タスクU単位、クエリ学習行だけ1 taskを除外した。P5 LOTOは各fold1 seed、8 epochで再学習。', '']
    (out/'report.md').write_text('\n'.join(lines))


if __name__=='__main__':
    main()
