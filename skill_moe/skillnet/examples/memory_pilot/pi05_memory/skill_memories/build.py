#!/usr/bin/env python3
"""Build raw-action and M1-base M6-style skill memories without accelerator use."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
PI05 = HERE.parent
import os as _os
_REPO = Path(__file__).resolve().parents[6]
_STATE_DIR = Path(_os.environ.get('SKILLNET_STATE_DIR', _REPO / '.skillnet-state'))
_PI05_ROOT = Path(_os.environ.get('SKILLNET_PI05_ROOT', _STATE_DIR / 'outputs/memory-pilot/pi05-memory40-20261104'))
RUN = _PI05_ROOT / 'run-01'
GAP = _PI05_ROOT / 'memory-gap-01'
RETRIEVAL = _PI05_ROOT / 'retrieval-4b-01'
TOPK = _PI05_ROOT / 'topk-long-01/production'
OUT = _PI05_ROOT / 'skill-memories-01'
SEED = 20261104
DEV_SEED = 20261105
MIN_SUPPORT = 10
MIN_EPISODES = 3
PENDING_BATCH = 200
PENDING_MAX_AGE = 1000
RANDOM_PAIRS = 64
PROBE_COUNT = 64
KAPPA_GRID = (0.0125, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8)
LONG_TASKS = set(range(10))


def load_rows(run: Path = RUN):
    return {p.stem: np.load(p, mmap_mode='r', allow_pickle=False)
            for p in (run / 'rows').glob('*.npy')}


def dev_row_ids(rows: dict[str, np.ndarray]) -> np.ndarray:
    """Task-stratified validation episodes; Long tasks are never returned."""
    selected: list[tuple[int, int]] = []
    for task in sorted(int(t) for t in np.unique(rows['task']) if int(t) not in LONG_TASKS):
        episodes = np.unique(np.asarray(rows['episode'])[
            (np.asarray(rows['task']) == task) & np.asarray(rows['validation'], bool)])
        order = np.random.default_rng(DEV_SEED + task).permutation(episodes)
        selected.extend((task, int(ep)) for ep in order[:3])
    mask = np.zeros(len(rows['task']), bool)
    task_all = np.asarray(rows['task']); episode_all = np.asarray(rows['episode'])
    for task, episode in selected:
        mask |= ((task_all == task) & (episode_all == episode) & np.asarray(rows['validation'], bool))
    ids = np.flatnonzero(mask)
    if not len(ids) or np.any(np.isin(task_all[ids], list(LONG_TASKS))):
        raise AssertionError('invalid/Long Dev rows')
    return ids


def long_eval_ids(rows, gap_path: Path = GAP, topk_path: Path = TOPK) -> tuple[np.ndarray, np.ndarray]:
    """Use the fixed memory-gap Eval rows and assert topk-long's Long/hard split."""
    gap = np.load(gap_path / 'predictions_and_errors.npz', allow_pickle=False)
    names = gap['condition_names'].tolist()
    normal_i = names.index('A_normal')
    normal = np.asarray(gap['row_mse'][normal_i], np.float64).mean(axis=0)
    threshold = float(json.loads((gap_path / 'summary_corrected.json').read_text())['threshold'])
    all_ids = np.asarray(gap['row_ids'], np.int64)
    task = np.asarray(rows['task'])[all_ids]
    ids = all_ids[task < 10]
    hard = normal[task < 10] >= threshold
    z = np.load(topk_path / 'predictions_and_errors.npz', allow_pickle=False)
    if not np.array_equal(ids, z['query_rows']):
        raise AssertionError('Long Eval rows/order differ from topk-long')
    if not np.array_equal(hard, z['hard_mask']):
        raise AssertionError('Long hard-row mask differs from topk-long')
    if len(ids) != 548 or int(hard.sum()) != 63:
        raise AssertionError((len(ids), int(hard.sum())))
    return ids, hard


def assert_no_long(pool_ids, eval_ids, rows, *, context: str) -> None:
    ids = np.asarray(pool_ids, np.int64)
    eval_ids = np.asarray(eval_ids, np.int64)
    if len(np.intersect1d(ids, eval_ids)):
        raise AssertionError(f'{context}: Long Eval rows leaked')
    if np.any(np.isin(np.asarray(rows['task'])[ids], list(LONG_TASKS))):
        raise AssertionError(f'{context}: Long task rows leaked')


def sha256_inputs(row_ids, values, tasks, episodes, extra=None) -> str:
    h = hashlib.sha256()
    for arr in (np.asarray(row_ids, np.int64), np.asarray(tasks, np.int32),
                np.asarray(episodes, np.int32), np.asarray(values, np.float32)):
        h.update(str(arr.shape).encode())
        h.update(np.ascontiguousarray(arr).view(np.uint8))
    if extra is not None:
        arr = np.asarray(extra, np.float32)
        h.update(str(arr.shape).encode()); h.update(np.ascontiguousarray(arr).view(np.uint8))
    return h.hexdigest()


class RawFunctional:
    """M6 functional metric for raw active-7 action windows: exact action MSE."""
    def __init__(self):
        self.dim = 70
        self.cache: dict[bytes, np.ndarray] = {}

    def embedding(self, value):
        x = np.asarray(value, np.float32).reshape(-1)
        if x.shape != (70,):
            raise ValueError(f'raw functional vector must have 70 values, got {x.shape}')
        key = x.tobytes()
        if key not in self.cache:
            self.cache[key] = x / np.sqrt(70.0)
        return self.cache[key]

    def distance(self, left, right):
        delta = self.embedding(left) - self.embedding(right)
        return float(np.dot(delta, delta))

    def nearest(self, value, centers):
        if not centers:
            return None, float('inf')
        c = np.asarray(centers, np.float32)
        l2 = np.linalg.norm(c - np.asarray(value, np.float32)[None], axis=1)
        ix = np.argsort(l2, kind='stable')[:min(10, len(l2))]
        d = [self.distance(value, c[i]) for i in ix]
        j = min(range(len(ix)), key=lambda k: (d[k], int(ix[k])))
        return int(ix[j]), float(d[j])


class CodecFunctional:
    """M6 decoder-probe functional distance for stored M1 bases."""
    def __init__(self, codec, probe_features, probe_states, probe_residuals, *, batch_bases=32):
        self.codec = codec
        self.features = np.asarray(probe_features, np.float32)
        self.states = np.asarray(probe_states, np.float32)
        self.residuals = np.asarray(probe_residuals, np.float32)
        if not (len(self.features) == len(self.states) == len(self.residuals) == PROBE_COUNT):
            raise ValueError('M6 functional probe must contain exactly 64 aligned rows')
        self.batch_bases = int(batch_bases)
        self.cache: dict[bytes, np.ndarray] = {}

    def embeddings(self, values):
        import torch
        torch.set_num_threads(8)
        x = np.asarray(values, np.float32)
        if x.ndim == 1: x = x[None]
        result = [None] * len(x)
        missing, missing_keys = [], []
        for i, v in enumerate(x):
            key = np.ascontiguousarray(v).tobytes()
            if key in self.cache:
                result[i] = self.cache[key]
            else:
                missing.append(v); missing_keys.append((i, key))
        if missing:
            missing = np.asarray(missing, np.float32)
            for start in range(0, len(missing), self.batch_bases):
                end = min(start + self.batch_bases, len(missing)); b = missing[start:end]
                nb, npb = len(b), len(self.features)
                br = np.repeat(b[:, None, :], npb, axis=1).reshape(nb*npb, -1)
                rr = np.tile(self.residuals, (nb, 1))
                ff = np.tile(self.features, (nb, 1))
                ss = np.tile(self.states, (nb, 1))
                decoded = np.asarray(self.codec.decode(br, rr, ff, ss), np.float32)
                if decoded.ndim != 3 or decoded.shape[1:] != (10, 32):
                    raise ValueError(f'unexpected M1 decoder output {decoded.shape}')
                emb = decoded.reshape(nb, -1) / np.sqrt(float(decoded[0].size))
                for j, vec in enumerate(emb):
                    global_missing = start + j
                    i, key = missing_keys[global_missing]
                    vec = vec.astype(np.float32, copy=False)
                    self.cache[key] = vec
                    result[i] = vec
        return np.stack(result).astype(np.float32)

    def embedding(self, value):
        return self.embeddings(np.asarray(value, np.float32)[None])[0]

    def distance(self, left, right):
        a, b = self.embedding(left), self.embedding(right)
        delta = a - b
        return float(np.dot(delta, delta))

    def nearest(self, value, centers):
        if not centers:
            return None, float('inf')
        c = np.asarray(centers, np.float32)
        l2 = np.linalg.norm(c - np.asarray(value, np.float32)[None], axis=1)
        ix = np.argsort(l2, kind='stable')[:min(10, len(l2))]
        left = self.embedding(value); right = self.embeddings(c[ix])
        d = np.sum(np.square(right - left[None], dtype=np.float64), axis=1)
        j = min(range(len(ix)), key=lambda k: (float(d[k]), int(ix[k])))
        return int(ix[j]), float(d[j])

    def pairwise(self, values):
        e = self.embeddings(np.asarray(values, np.float32))
        norms = np.einsum('ij,ij->i', e, e)
        d = norms[:, None] + norms[None, :] - 2.0 * (e @ e.T)
        return np.maximum(d, 0.0)


def m6_fit(values, row_ids, tasks, episodes, metric, *, kappa, seed=SEED,
           min_support=MIN_SUPPORT, min_episodes=MIN_EPISODES,
           pending_batch=PENDING_BATCH, pending_max_age=PENDING_MAX_AGE,
           epsilon_override=None):
    """Offline mirror of M6's seeded assignment/pending/registration lifecycle."""
    x = np.asarray(values, np.float32)
    row_ids = np.asarray(row_ids, np.int64); tasks = np.asarray(tasks, np.int32)
    episodes = np.asarray(episodes, np.int32)
    if x.ndim != 2 or len(x) != len(row_ids) or len(x) != len(tasks) or len(x) != len(episodes):
        raise ValueError('M6 arrays must align and values must be [N,D]')
    if len(x) == 0: raise ValueError('cannot fit an empty pool')
    rng = np.random.default_rng(int(seed) + 6046)
    pair = rng.integers(0, len(x), size=(RANDOM_PAIRS, 2))
    random_d = np.asarray([metric.distance(x[a], x[b]) for a, b in pair], np.float64)
    median = float(np.median(random_d))
    epsilon = float(epsilon_override if epsilon_override is not None else max(float(kappa)*median, 1e-12))
    centers: list[np.ndarray] = []
    support: list[int] = []
    unit_eps: list[set[tuple[int,int]]] = []
    members: list[list[int]] = []
    row_units = np.full(len(x), -1, np.int32)
    pending: list[dict] = []
    pending_age: dict[int,int] = {}
    pending_since = 0; next_pending = 0; expired = 0

    def assign_pending(force=False):
        nonlocal pending, pending_since, expired, centers, support, unit_eps, members
        if not pending or (not force and pending_since < pending_batch):
            return
        pending_since = 0
        clusters: list[dict] = []
        for item in pending:
            value = item['value']
            if not clusters:
                clusters.append({'center': value.copy(), 'values': [value], 'items': [item],
                                 'eps': {item['episode_key']}})
                continue
            nearest, dist = metric.nearest(value, [c['center'] for c in clusters])
            if nearest is not None and dist <= epsilon:
                c = clusters[nearest]; c['values'].append(value); c['items'].append(item)
                c['center'] = np.mean(c['values'], axis=0).astype(np.float32); c['eps'].add(item['episode_key'])
            else:
                clusters.append({'center': value.copy(), 'values': [value], 'items': [item],
                                 'eps': {item['episode_key']}})
        registered_ids: dict[int,int] = {}
        for c in clusters:
            count, neps = len(c['items']), len(c['eps'])
            if count < min_support or neps < min_episodes: continue
            nearest, dist = metric.nearest(c['center'], centers)
            if nearest is not None and dist <= epsilon:
                old = support[nearest]; total = old + count
                centers[nearest] = ((centers[nearest]*old + c['center']*count)/total).astype(np.float32)
                support[nearest] = total; unit_eps[nearest].update(c['eps'])
                members[nearest].extend(item['local'] for item in c['items'])
                uid = nearest
            else:
                uid = len(centers); centers.append(c['center'].copy()); support.append(count)
                unit_eps.append(set(c['eps'])); members.append([item['local'] for item in c['items']])
            for item in c['items']: registered_ids[item['id']] = uid
        retained=[]
        for item in pending:
            if item['id'] in registered_ids:
                row_units[item['local']] = registered_ids[item['id']]
                pending_age.pop(item['id'],None)
                continue
            pid=item['id']; pending_age[pid]=pending_age.get(pid,0)+pending_batch
            if pending_age[pid]>=pending_max_age:
                pending_age.pop(pid,None); expired+=1
            else: retained.append(item)
        pending=retained

    for local in rng.permutation(len(x)):
        value=x[local]
        nearest,dist=metric.nearest(value,centers)
        episode_key=(int(tasks[local]),int(episodes[local]))
        if nearest is not None and dist<=epsilon:
            old=support[nearest]; new=old+1
            centers[nearest]=((centers[nearest]*old+value)/new).astype(np.float32)
            support[nearest]=new; unit_eps[nearest].add(episode_key)
            members[nearest].append(int(local)); row_units[local]=nearest
        else:
            pending.append({'id':next_pending,'local':int(local),'value':value.copy(),'episode_key':episode_key})
            pending_age[next_pending]=0; next_pending+=1; pending_since+=1
        assign_pending()
    assign_pending(force=True)
    return {'row_units':row_units,'centers':np.asarray(centers,np.float32).reshape(-1,x.shape[1]),
            'members_local':[np.asarray(m,np.int64) for m in members],
            'support':np.asarray(support,np.int32),'episode_sets':unit_eps,
            'functional_median':median,'epsilon':epsilon,'pending_local':np.asarray([p['local'] for p in pending],np.int64),
            'expired_pending':int(expired),'registered':len(centers),'seed':int(seed)}


def choose_kappas_from_dev(dev_actions, dev_row_ids, dev_tasks, dev_episodes, *, seed=SEED):
    """Select finer/coarser kappas around M6's .1 using Dev registrations only."""
    metric=RawFunctional(); values=np.asarray(dev_actions,np.float32).reshape(len(dev_actions),-1)
    records=[]
    for kappa in KAPPA_GRID:
        fit=m6_fit(values,dev_row_ids,dev_tasks,dev_episodes,metric,kappa=kappa,seed=seed)
        records.append({'kappa':float(kappa),'dev_units':int(fit['registered']),
                        'dev_pending':int(len(fit['pending_local'])),'epsilon_dev':float(fit['epsilon'])})
    lower=[r for r in records if r['kappa']<0.1]
    upper=[r for r in records if r['kappa']>0.1]
    central=next(r for r in records if r['kappa']==0.1)
    # Target several tens/hundreds on Long-excluded Dev; lower epsilon -> finer units.
    ka=min(lower,key=lambda r:(abs(r['dev_units']-300),r['kappa']))
    kc=min(upper,key=lambda r:(abs(r['dev_units']-30),r['kappa']))
    chosen=[ka,central,kc]
    if not (chosen[0]['kappa']<chosen[1]['kappa']<chosen[2]['kappa']):
        raise AssertionError('kappa values are not increasing')
    return [float(r['kappa']) for r in chosen],records


def unit_medoid(members_local, values, metric):
    ix=np.asarray(members_local,np.int64)
    if not len(ix): raise ValueError('empty M6 unit')
    x=np.asarray(values[ix],np.float32)
    if len(ix)==1: return int(ix[0])
    if isinstance(metric,RawFunctional):
        emb=metric.embeddings(x) if hasattr(metric,'embeddings') else x/np.sqrt(float(x.shape[1]))
        norm=np.einsum('ij,ij->i',emb,emb); d=norm[:,None]+norm[None,:]-2*(emb@emb.T)
        d=np.maximum(d,0.)
    else:
        emb=metric.embeddings(x); norm=np.einsum('ij,ij->i',emb,emb)
        d=np.maximum(norm[:,None]+norm[None,:]-2*(emb@emb.T),0.)
    medoid_local=int(np.argmin(d.mean(axis=1)))
    return int(ix[medoid_local])


def summarize_units(fit, pool_ids, rows, values, metric):
    rep_local=np.asarray([unit_medoid(m,values,metric) for m in fit['members_local']],np.int64)
    rep_rows=np.asarray(pool_ids,np.int64)[rep_local]
    return rep_local,rep_rows


def summarize_movement(action, grip_threshold=.05):
    a=np.asarray(action,np.float32)
    delta=a[:,:3].mean(axis=0)
    axes=('x','y','z'); j=int(np.argmax(np.abs(delta)))
    direction=('+' if delta[j]>=0 else '-')+axes[j]
    g=np.asarray(a[:,6])
    state=lambda v: 'open' if v>=grip_threshold else ('close' if v<=-grip_threshold else 'hold')
    return {'dominant_mean_delta_axis':direction,'mean_delta_xyz_normalized':[float(x) for x in delta],
            'grip_start_end':[float(g[0]),float(g[-1])],
            'grip_start_end_label':[state(float(g[0])),state(float(g[-1]))]}


def raw_diagnostics(query_actions, representatives, *, batch=32):
    q=np.asarray(query_actions,np.float32).reshape(len(query_actions),70)
    r=np.asarray(representatives,np.float32).reshape(len(representatives),70)
    out1=np.empty(len(q),np.float32); out5=np.empty(len(q),np.float32)
    rn=np.einsum('ij,ij->i',r,r)
    for start in range(0,len(q),batch):
        qb=q[start:start+batch]; qn=np.einsum('ij,ij->i',qb,qb)
        d=np.maximum(qn[:,None]+rn[None,:]-2*(qb@r.T),0.)/70.
        part=np.partition(d,min(4,d.shape[1]-1),axis=1)
        out1[start:start+len(qb)]=d.min(axis=1)
        out5[start:start+len(qb)]=part[:,:min(5,d.shape[1])].min(axis=1)
    return out1,out5


def b_diagnostics(codec, rep_b, query_feat, query_state, query_actions, *, batch_queries=4, batch_reps=128):
    import torch
    torch.set_num_threads(8)
    B=np.asarray(rep_b,np.float32); F=np.asarray(query_feat,np.float32); S=np.asarray(query_state,np.float32)
    A=np.asarray(query_actions,np.float32)[:,:,:7]
    one=np.empty(len(A),np.float32); five=np.empty(len(A),np.float32)
    for qs in range(0,len(A),batch_queries):
        qe=min(qs+batch_queries,len(A)); f=F[qs:qe]; s=S[qs:qe]
        _,r=codec.prior(f,s); target=A[qs:qe]
        all_d=np.empty((qe-qs,len(B)),np.float32)
        for rs in range(0,len(B),batch_reps):
            re=min(rs+batch_reps,len(B)); k=re-rs; nq=qe-qs
            bb=np.broadcast_to(B[None,rs:re],(nq,k,B.shape[1])).reshape(nq*k,-1)
            rr=np.broadcast_to(r[:,None,:],(nq,k,r.shape[-1])).reshape(nq*k,-1)
            ff=np.broadcast_to(f[:,None,:],(nq,k,f.shape[-1])).reshape(nq*k,-1)
            ss=np.broadcast_to(s[:,None,:],(nq,k,s.shape[-1])).reshape(nq*k,-1)
            pred=np.asarray(codec.decode(bb,rr,ff,ss),np.float32).reshape(nq,k,10,32)[:,:,:,:7]
            d=np.mean((pred-target[:,None])**2,axis=(-1,-2))
            all_d[:,rs:re]=d
        p=np.partition(all_d,min(4,all_d.shape[1]-1),axis=1)
        one[qs:qe]=all_d.min(axis=1); five[qs:qe]=p[:,:min(5,p.shape[1])].min(axis=1)
        print(f'Long b-query diagnostics {qe}/{len(A)}',flush=True)
    return one,five


def unit_stats(unit_members, pool_ids, rows, rep_rows, actions):
    sizes=np.asarray([len(x) for x in unit_members],np.int32)
    task_counts=[]; episode_counts=[]; examples=[]
    tasks=np.asarray(rows['task']); eps=np.asarray(rows['episode'])
    for uid,(members,rep) in enumerate(zip(unit_members,rep_rows,strict=True)):
        ids=np.asarray(pool_ids)[members]
        ts=np.unique(tasks[ids]); es=set(zip(tasks[ids].tolist(),eps[ids].tolist()))
        task_counts.append(len(ts)); episode_counts.append(len(es))
        if uid < 5 or len(members) >= (int(np.median(sizes)) if len(sizes) else 0)*2:
            examples.append({'unit_id':uid,'member_count':int(len(members)),'task_count':int(len(ts)),
                             'episode_count':int(len(es)),'representative_row_id':int(rep),
                             **summarize_movement(actions[int(rep)])})
    def quant(v):
        if not len(v):return {'min':0,'p10':0,'median':0,'p90':0,'max':0}
        return {'min':int(np.min(v)),'p10':float(np.quantile(v,.1)),'median':float(np.median(v)),
                'p90':float(np.quantile(v,.9)),'max':int(np.max(v))}
    return {'unit_count':int(len(sizes)),'member_count':int(sizes.sum()),'member_count_distribution':quant(sizes),
            'episode_count_distribution':quant(episode_counts),'task_count_distribution':quant(task_counts),
            'units_crossing_multiple_tasks':int(np.sum(np.asarray(task_counts)>1)),
            'fraction_cross_task':float(np.mean(np.asarray(task_counts)>1)) if len(task_counts) else 0.,
            'representative_examples':examples}


def save_row_memory(outdir, name, pool_ids, rows, actions, wrist, *, pool_name, input_hash, eval_ids, hard):
    dest=outdir/name
    if dest.exists(): raise FileExistsError(dest)
    dest.mkdir(parents=True)
    ids=np.asarray(pool_ids,np.int64); acts=np.asarray(actions[ids],np.float32)
    if np.any(np.isin(np.asarray(rows['task'])[ids],list(LONG_TASKS))) and pool_name=='noLong':
        raise AssertionError(f'Long row in {name}')
    n=len(ids); member_offsets=np.arange(n+1,dtype=np.int64)
    tasks=np.asarray(rows['task'])[ids]; eps=np.asarray(rows['episode'])[ids]
    np.savez_compressed(dest/'units.npz',unit_ids=np.arange(n,dtype=np.int32),
        representative_row_ids=ids,representative_action=acts,
        member_row_ids=ids.copy(),member_offsets=member_offsets,
        unit_member_count=np.ones(n,np.int32),unit_episode_count=np.ones(n,np.int32),
        unit_task_count=np.ones(n,np.int32),pool_row_ids=ids,pool_unit_ids=np.arange(n,dtype=np.int32))
    rep_state=np.asarray(rows['state'])[ids,:8].astype(np.float32)
    rep_wrist=np.asarray(wrist[ids],np.float32)
    np.savez_compressed(dest/'keys.npz',representative_row_ids=ids,
        representative_wrist=rep_wrist,representative_state=rep_state,
        router_input=np.concatenate([rep_wrist,rep_state],axis=1).astype(np.float32))
    diag1,diag5=raw_diagnostics(np.asarray(actions[eval_ids])[:,:,:7],acts[:,:,:7])
    tarr=np.asarray(rows['task'])[ids];
    report={'kind':'raw-row-episodic-memory','unit_stats':{'unit_count':n,'member_count':n,
        'member_count_distribution':{'min':1,'p10':1.,'median':1.,'p90':1.,'max':1},
        'task_count_distribution':{'min':1,'p10':1.,'median':1.,'p90':1.,'max':1},
        'units_crossing_multiple_tasks':0,'fraction_cross_task':0.,'representative_examples':[
            {'unit_id':i,'member_count':1,'task_count':1,'representative_row_id':int(ids[i]),**summarize_movement(acts[i])}
            for i in range(min(5,n))]},
        'long_eval_nearest_mean_mse':float(diag1.mean()),'long_eval_top5_best_mean_mse':float(diag5.mean()),
        'long_hard_nearest_mean_mse':float(diag1[hard].mean()),'long_hard_top5_best_mean_mse':float(diag5[hard].mean()),
        'diagnostic_rows':len(eval_ids),'hard_rows':int(hard.sum())}
    manifest={'memory_id':name,'pool':pool_name,'value':'raw normalized actions [10,32], functional distance/diagnostics use active7 MSE',
        'kappa':None,'epsilon':None,'min_support':None,'min_episodes':None,'unit_count':n,'unassigned_rate':0.,
        'input_sha256':input_hash,'seed':SEED,'row_stride':10,'chunk_length':10}
    (dest/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    (dest/'diagnostics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return report


def save_m6_memory(outdir, name, pool_ids, values, rows, actions, wrist, fit, metric, *,
                   pool_name, kappa, input_hash, eval_ids, hard, value_kind,
                   rep_actions=None, rep_b=None, codec=None, query_features=None, query_states=None):
    dest=outdir/name
    if dest.exists(): raise FileExistsError(dest)
    dest.mkdir(parents=True)
    pool_ids=np.asarray(pool_ids,np.int64); vals=np.asarray(values,np.float32)
    members_local=fit['members_local']; rep_local,rep_rows=summarize_units(fit,pool_ids,rows,vals,metric)
    member_rows=np.concatenate([pool_ids[np.asarray(m,np.int64)] for m in members_local]) if members_local else np.empty(0,np.int64)
    offsets=np.zeros(len(members_local)+1,np.int64)
    if members_local: offsets[1:]=np.cumsum([len(m) for m in members_local])
    unit_count=len(members_local); tasks=np.asarray(rows['task']); episodes=np.asarray(rows['episode'])
    rep_action=np.asarray(actions[rep_rows],np.float32)
    unit_ep=np.asarray([len(set(zip(tasks[pool_ids[m]].tolist(),episodes[pool_ids[m]].tolist()))) for m in members_local],np.int32)
    unit_task=np.asarray([len(np.unique(tasks[pool_ids[m]])) for m in members_local],np.int32)
    unit_sizes=np.asarray([len(m) for m in members_local],np.int32)
    extras={}
    if rep_b is not None: extras['representative_b']=np.asarray(rep_b,np.float32)[rep_rows]
    np.savez_compressed(dest/'units.npz',unit_ids=np.arange(unit_count,dtype=np.int32),
        representative_row_ids=rep_rows,representative_action=rep_action,
        member_row_ids=member_rows,member_offsets=offsets,unit_member_count=unit_sizes,
        unit_episode_count=unit_ep,unit_task_count=unit_task,
        pool_row_ids=pool_ids,pool_unit_ids=fit['row_units'].astype(np.int32),**extras)
    rep_wrist=np.asarray(wrist[rep_rows],np.float32); rep_state=np.asarray(rows['state'])[rep_rows,:8].astype(np.float32)
    np.savez_compressed(dest/'keys.npz',representative_row_ids=rep_rows,
        representative_wrist=rep_wrist,representative_state=rep_state,
        router_input=np.concatenate([rep_wrist,rep_state],axis=1).astype(np.float32))
    if value_kind=='b12':
        if codec is None or query_features is None or query_states is None: raise ValueError('missing b diagnostic inputs')
        nearest,top5=b_diagnostics(codec,extras['representative_b'],query_features,query_states,
                                   np.asarray(actions[eval_ids],np.float32))
    else:
        nearest,top5=raw_diagnostics(np.asarray(actions[eval_ids])[:,:,:7],rep_action[:,:,:7])
    stats=unit_stats(members_local,pool_ids,rows,rep_rows,actions)
    assigned=int(np.sum(fit['row_units']>=0)); unassigned=1.-assigned/max(len(pool_ids),1)
    report={'unit_stats':stats,'assigned_rows':assigned,'pending_rows':int(len(fit['pending_local'])),
        'expired_pending':int(fit['expired_pending']),'unassigned_rate':float(unassigned),
        'long_eval_nearest_mean_mse':float(nearest.mean()),'long_eval_top5_best_mean_mse':float(top5.mean()),
        'long_hard_nearest_mean_mse':float(nearest[hard].mean()),'long_hard_top5_best_mean_mse':float(top5[hard].mean()),
        'long_eval_nearest_per_row':nearest.tolist(),'long_eval_top5_per_row':top5.tolist(),
        'diagnostic_rows':len(eval_ids),'hard_rows':int(hard.sum())}
    manifest={'memory_id':name,'pool':pool_name,'value':value_kind,
        'functional_distance':'active7 10x7 MSE' if value_kind=='raw' else 'M6 decoder functional MSE over fixed 64 nonLong memory probes; base-L2 top10 prefilter',
        'kappa':float(kappa),'functional_median_random_pairs':fit['functional_median'],'epsilon':fit['epsilon'],
        'min_support':MIN_SUPPORT,'min_episodes':MIN_EPISODES,'pending_batch_size':PENDING_BATCH,
        'pending_max_age':PENDING_MAX_AGE,'unit_count':unit_count,'unassigned_rate':float(unassigned),
        'input_sha256':input_hash,'seed':SEED,'row_stride':10,'chunk_length':10,
        'warning':'The reused run-01 M1 codec was trained on non-validation rows across all 40 tasks, including Long; this noLong build excludes Long rows from its pool, assignments, medoids, and diagnostics but does not refit codec weights.' if value_kind=='b12' else None}
    (dest/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    (dest/'diagnostics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return report


def reference_metrics(topk_path, hard):
    z=np.load(topk_path/'predictions_and_errors.npz',allow_pickle=False)
    zero=np.asarray(z['zero_mse'],np.float64)
    tvals=np.asarray(z['t0_values'],np.float64)
    result={'source':str(topk_path/'predictions_and_errors.npz'),
            'normal_single_all':float(np.mean(z['normal_single_mse'])),
            'normal_single_hard':float(np.mean(z['normal_single_mse'][hard])),
            'zero_by_t0':{str(float(t)): {'all':float(zero[i].mean()),'hard':float(zero[i][:,hard].mean())}
                          for i,t in enumerate(tvals)}}
    return result



def key_ranked_top5_existing(run, retrieval, output):
    """Supplement saved memories with runtime-key Top-5 action coverage; no model fit."""
    output=Path(output); dest=output/'key_top5_diagnostics.json'; npz_path=output/'key_top5_diagnostics.npz'
    if dest.exists() or npz_path.exists(): raise FileExistsError('key Top-5 diagnostic already exists')
    rows=load_rows(run); eval_ids,hard=long_eval_ids(rows)
    wrist=np.load(retrieval/'wrist_token_mean.npy',mmap_mode='r',allow_pickle=False)
    stats=np.load(retrieval/'wrist_state_normalization.npz',allow_pickle=False)
    mean=np.asarray(stats['mean'],np.float32);std=np.asarray(stats['std'],np.float32)
    qraw=np.concatenate([np.asarray(wrist[eval_ids],np.float32),np.asarray(rows['state'][eval_ids,:8],np.float32)],axis=1)
    q=(qraw-mean)/np.where(std<1e-5,1.,std);q/=np.maximum(np.linalg.norm(q,axis=1,keepdims=True),1e-8)
    truth=np.asarray(rows['actions'][eval_ids,:,:7],np.float32)
    names=[]; candidate_rows=[]; top1_errors=[]; best5_errors=[]; all_scores=[]; details={}
    import torch
    torch.set_num_threads(8)
    codec=None; qfeat=None
    sys.path.insert(0,str(PI05))
    from run import restore_codec
    for folder in sorted(p for p in output.iterdir() if p.is_dir() and (p/'units.npz').exists()):
        unit=np.load(folder/'units.npz',allow_pickle=False); key=np.load(folder/'keys.npz',allow_pickle=False)
        rep_rows=np.asarray(unit['representative_row_ids'],np.int64); rawkey=np.asarray(key['router_input'],np.float32)
        if len(rep_rows)<5: raise ValueError(f'{folder.name} has fewer than five candidates')
        m=(rawkey-mean)/np.where(std<1e-5,1.,std);m/=np.maximum(np.linalg.norm(m,axis=1,keepdims=True),1e-8)
        score=q@m.T
        ix=np.argsort(-score,axis=1,kind='stable')[:,:5]
        selected_rows=rep_rows[ix]; selected_actions=np.asarray(unit['representative_action'],np.float32)[ix][:,:,:,:7]
        if 'representative_b' in unit.files:
            if codec is None:
                codec=restore_codec(run);qfeat=np.asarray(np.load(run/'pooled.npy',mmap_mode='r')[eval_ids],np.float32)
            bases=np.asarray(unit['representative_b'],np.float32)[ix]
            d=np.empty((len(eval_ids),5),np.float32)
            for start in range(0,len(eval_ids),4):
                end=min(start+4,len(eval_ids)); f=qfeat[start:end];state=np.asarray(rows['state'][eval_ids[start:end]],np.float32)
                _,r=codec.prior(f,state);k=5*(end-start)
                pred=np.asarray(codec.decode(bases[start:end].reshape(k,-1),np.repeat(r,5,axis=0),
                    np.repeat(f,5,axis=0),np.repeat(state,5,axis=0)),np.float32).reshape(end-start,5,10,32)[:,:,:,:7]
                d[start:end]=np.mean((pred-truth[start:end,None])**2,axis=(-1,-2))
        else:
            d=np.mean((selected_actions-truth[:,None])**2,axis=(-1,-2))
        names.append(folder.name);candidate_rows.append(selected_rows);top1_errors.append(d[:,0]);best5_errors.append(d.min(axis=1));all_scores.append(np.take_along_axis(score,ix,axis=1))
        details[folder.name]={'key':'retrieval-4b wrist_state_cos (train-standardized wrist token mean + state[:8], cosine)',
           'top1_action_mse_all':float(d[:,0].mean()),'top1_action_mse_hard':float(d[hard,0].mean()),
           'top5_best_action_mse_all':float(d.min(axis=1).mean()),'top5_best_action_mse_hard':float(d[hard].min(axis=1).mean()),
           'candidate_count':int(len(rep_rows))}
        print(f'key Top-5 {folder.name} done',flush=True)
    np.savez_compressed(npz_path,memory_names=np.asarray(names),candidate_rows=np.asarray(candidate_rows),
        cosine_scores=np.asarray(all_scores),top1_action_mse=np.asarray(top1_errors),top5_best_action_mse=np.asarray(best5_errors),
        eval_rows=eval_ids,hard_mask=hard)
    report={'key_top5_definition':'rank units/items by retrieval-4b wrist_state_cos; evaluate normalized active7 action MSE of top1 and best among key-ranked top5',
       'eval_rows':len(eval_ids),'hard_rows':int(hard.sum()),'memories':details,'npz':str(npz_path)}
    dest.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return report

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',type=Path,default=RUN);ap.add_argument('--gap',type=Path,default=GAP)
    ap.add_argument('--retrieval',type=Path,default=RETRIEVAL);ap.add_argument('--topk',type=Path,default=TOPK)
    ap.add_argument('--output',type=Path,default=OUT)
    ap.add_argument('--key-top5-existing',action='store_true',help='score key-ranked Top-5 from existing memory folders')
    ap.add_argument('--max-m6-rows',type=int,default=None,
        help='CPU test/smoke only; do not use for production')
    args=ap.parse_args();out=args.output
    if args.key_top5_existing:
        key_ranked_top5_existing(args.run,args.retrieval,out)
        return
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()): raise FileExistsError(f'refusing non-empty output root {out}')
    import torch
    torch.set_num_threads(8)
    rows=load_rows(args.run); actions=np.load(args.run/'rows/actions.npy',mmap_mode='r',allow_pickle=False)
    bases=np.load(args.run/'bases.npy',mmap_mode='r',allow_pickle=False)
    wrist=np.load(args.retrieval/'wrist_token_mean.npy',mmap_mode='r',allow_pickle=False)
    if len(actions)!=len(rows['task']) or len(bases)!=len(actions) or wrist.shape!=(len(actions),2048):
        raise ValueError('run-01/retrieval-4b arrays do not align')
    long_ids,hard=long_eval_ids(rows,args.gap,args.topk)
    dev_ids=dev_row_ids(rows)
    if np.any(np.isin(np.asarray(rows['task'])[dev_ids],list(LONG_TASKS))) or np.intersect1d(dev_ids,long_ids).size:
        raise AssertionError('Long/Eval rows entered Dev selection')
    memory_ids=np.flatnonzero(np.asarray(rows['memory'],bool))
    task_all=np.asarray(rows['task'])
    no_long_ids=memory_ids[~np.isin(task_all[memory_ids],list(LONG_TASKS))]
    if len(no_long_ids)==0 or np.any(np.isin(task_all[no_long_ids],list(LONG_TASKS))):
        raise AssertionError('invalid noLong pool')
    assert_no_long(no_long_ids,long_ids,rows,context='noLong pool')
    assert_no_long(dev_ids,long_ids,rows,context='noLong Dev')
    if args.max_m6_rows:
        no_long_ids=no_long_ids[:args.max_m6_rows]
        # Test mode cannot claim production leakage/count coverage; still preserve no-Long check.
    dev_a=np.asarray(actions[dev_ids,:,:7],np.float32)
    kappas,dev_calibration=choose_kappas_from_dev(dev_a,dev_ids,np.asarray(rows['task'])[dev_ids],
                                                   np.asarray(rows['episode'])[dev_ids])
    ka,kb,kc=kappas
    all40_ids=memory_ids
    raw_metric=RawFunctional()
    no_long_actions=np.asarray(actions[no_long_ids,:,:7],np.float32)
    all_actions=np.asarray(actions[all40_ids,:,:7],np.float32)
    raw_records={}
    for name,pool_ids,pool_values,pool in [('raw-rows-noLong',no_long_ids,None,'noLong'),
                                          ('raw-rows-all40',all40_ids,None,'all40')]:
        hv=np.asarray(actions[pool_ids],np.float32)
        sig=sha256_inputs(pool_ids,hv,task_all[pool_ids],np.asarray(rows['episode'])[pool_ids],
                          np.asarray(wrist[pool_ids],np.float32))
        raw_records[name]=save_row_memory(out,name,pool_ids,rows,actions,wrist,pool_name=pool,
                                           input_hash=sig,eval_ids=long_ids,hard=hard)
        print(f'created {name}: {len(pool_ids)} rows',flush=True)

    build_list=[(f'raw-m6-noLong-k{str(k).replace(".","p")}',no_long_ids,no_long_actions,k,'noLong','raw')
                for k in kappas]
    build_list.append(('raw-m6-all40',all40_ids,all_actions,kb,'all40','raw'))
    # M6 raw-distance scale and streaming assignments use no labels, no task IDs as features.
    for name,pool_ids,pool_values,kappa,pool,kind in build_list:
        metric=RawFunctional();fit=m6_fit(pool_values.reshape(len(pool_values),-1),pool_ids,task_all[pool_ids],
            np.asarray(rows['episode'])[pool_ids],metric,kappa=kappa,seed=SEED)
        if pool=='noLong': assert_no_long(pool_ids,long_ids,rows,context=name)
        sig=sha256_inputs(pool_ids,pool_values,task_all[pool_ids],np.asarray(rows['episode'])[pool_ids],
                          np.asarray(wrist[pool_ids],np.float32))
        raw_records[name]=save_m6_memory(out,name,pool_ids,pool_values.reshape(len(pool_values),-1),rows,actions,wrist,
            fit,metric,pool_name=pool,kappa=kappa,input_hash=sig,eval_ids=long_ids,hard=hard,value_kind='raw')
        # Store compact M6 summaries for the final README.
        print(f'created {name}: units={fit["registered"]} pending={len(fit["pending_local"])} eps={fit["epsilon"]:.6g}',flush=True)

    # M1 base memory: fixed run-01 codec and bases, no accelerator use.
    sys.path.insert(0,str(PI05))
    from run import restore_codec
    codec=restore_codec(args.run)
    probe_rng=np.random.default_rng(SEED+6046)
    probe_local=probe_rng.choice(len(no_long_ids),size=PROBE_COUNT,replace=len(no_long_ids)<PROBE_COUNT)
    probe_rows=no_long_ids[probe_local]
    _,probe_r=codec.prior(np.asarray(np.load(args.run/'pooled.npy',mmap_mode='r')[probe_rows],np.float32),
                          np.asarray(rows['state'])[probe_rows])
    bmetric=CodecFunctional(codec,np.asarray(np.load(args.run/'pooled.npy',mmap_mode='r')[probe_rows],np.float32),
                            np.asarray(rows['state'])[probe_rows],probe_r)
    b_values=np.asarray(bases[no_long_ids],np.float32)
    b_fit=m6_fit(b_values,no_long_ids,task_all[no_long_ids],np.asarray(rows['episode'])[no_long_ids],
                 bmetric,kappa=kb,seed=SEED)
    assert_no_long(no_long_ids,long_ids,rows,context='b12 noLong pool')
    sig=sha256_inputs(no_long_ids,b_values,task_all[no_long_ids],np.asarray(rows['episode'])[no_long_ids],
                      np.asarray(wrist[no_long_ids],np.float32))
    qfeatures=np.asarray(np.load(args.run/'pooled.npy',mmap_mode='r')[long_ids],np.float32)
    qstates=np.asarray(rows['state'])[long_ids]
    raw_records['b12-m6-noLong']=save_m6_memory(out,'b12-m6-noLong',no_long_ids,b_values,rows,actions,wrist,
        b_fit,bmetric,pool_name='noLong',kappa=kb,input_hash=sig,eval_ids=long_ids,hard=hard,value_kind='b12',
        rep_b=bases,codec=codec,query_features=qfeatures,query_states=qstates)
    print(f'created b12-m6-noLong: units={b_fit["registered"]} pending={len(b_fit["pending_local"])} eps={b_fit["epsilon"]:.6g}',flush=True)

    # Baseline values reused from the completed topk-long run, no policy inference.
    refs=reference_metrics(args.topk,hard)
    summary={'long_eval_rows':len(long_ids),'long_hard_rows':int(hard.sum()),'hard_threshold':0.01549410422449,
        'dev_rows':len(dev_ids),'dev_episode_pairs':sorted({(int(task),int(ep)) for task,ep in zip(task_all[dev_ids],np.asarray(rows['episode'])[dev_ids])}),
        'kappa_selection':{'grid':list(KAPPA_GRID),'dev_calibration':dev_calibration,'selected':[ka,kb,kc],
                           'rule':'fixed .1 center; lower count nearest 300, upper count nearest 30; Dev only'},
        'references':refs,'memories':raw_records,
        'warning':'NoLong excludes Long memory/Eval rows from construction. Existing run-01 M1 codec weights/bases were reused; the codec was fit on non-validation data across all 40 tasks, including Long.',
        'gpu_used':False,'expert_training':False}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    protocol={'source_run':str(args.run),'retrieval_features':str(args.retrieval/'wrist_token_mean.npy'),
       'eval_source':str(args.gap/'predictions_and_errors.npz'),'normal_zero_source':str(args.topk/'predictions_and_errors.npz'),
       'eval_rows':long_ids.tolist(),'hard_mask':hard.tolist(),'dev_rows':dev_ids.tolist(),
       'kappa_calibration_did_not_use_long':True,'long_rows_excluded_from_noLong':True,
       'metric_raw':'normalized active7 action MSE over 10x7','metric_b':'M6 decoded functional MSE using 64 fixed nonLong memory probes',
       'kappa_grid':list(KAPPA_GRID),'selected_kappas':[ka,kb,kc],'seed':SEED,
       'min_support':MIN_SUPPORT,'min_episodes':MIN_EPISODES,'pending_batch_size':PENDING_BATCH,'pending_max_age':PENDING_MAX_AGE,
       'fixed_interval':'10 action steps, stride 10','expert_training':False,'gpu_inference':False}
    (out/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
    (out/'DONE').write_text('complete\n')
    print(json.dumps({'output':str(out),'selected_kappas':kappas,
        'unit_counts':{k:v['unit_stats']['unit_count'] for k,v in raw_records.items()},
        'long_hard_nearest':{k:v['long_hard_nearest_mean_mse'] for k,v in raw_records.items()}},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    main()
