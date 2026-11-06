#!/usr/bin/env python3
"""Oracle upper-bound diagnostic for action compression and M6 clustering.

All row identifiers in this module are dataset row IDs. No retrieval score is
used: the true target action selects the medoid (intentionally oracle).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
PI05 = HERE.parent
sys.path.insert(0, str(PI05))
from compress_vs_raw.common import base_module, fair_module, require_new_artifacts

REPO_ROOT = Path(__file__).resolve().parents[6]
STATE_DIR = Path(os.environ.get('SKILLNET_STATE_DIR', REPO_ROOT / '.skillnet-state')).expanduser()
ROOT = Path(os.environ.get('SKILLNET_PI05_ROOT', STATE_DIR / 'outputs/memory-pilot/pi05-memory40-20261104')).expanduser()
RUN = ROOT / 'run-01'
MEMORIES = ROOT / 'skill-memories-01'
CODEC_DIR = ROOT / 'compress-vs-raw-01/dimension-extension-01'
OUT = ROOT / 'compress-cluster-oracle-01'
SEED = 20261104
LONG_TASKS = set(range(10))
KAPPAS = (.002, .006, .016, .04, .1, .25, .64, 1.6)
LEVELS = ('none', 'fine', 'mid', 'coarse', 'm6default')
REPS = 5000


def save_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=True) + '\n')
    tmp.replace(path)


def publish_pre_generation():
    """Normalize mixed-version cell metadata and publish the fixed 5x5 table."""
    pre_path=OUT/'pre_generation.json'
    if not pre_path.exists(): raise FileNotFoundError(pre_path)
    if (OUT/'results.md').exists(): raise FileExistsError(OUT/'results.md')
    pre=json.loads(pre_path.read_text())
    cells=pre.get('cells',[])
    if len(cells)!=25: raise AssertionError(f'expected 25 pre-generation cells, got {len(cells)}')
    n=int(pre['rows'])
    for r in cells:
        s=r.setdefault('storage',{})
        if 'memory_store_scalar_values_total' not in s:
            dim=70 if r['representation']=='raw' else int(r['representation'][1:])
            units=int(r['unit_count'])
            r['storage']={'fit_pool_representation_scalars':n*dim,
                'stored_medoid_representation_scalars':units*dim,'stored_medoid_row_ids':units,
                'stored_pool_row_ids':n,'stored_row_to_unit_ids':n,
                'memory_store_scalar_values_total':units*dim+units+2*n}
    # The first interrupted publish used a legacy raw-none storage schema.
    # Update only this run's own just-created pre-generation artifact.
    save_json(pre_path,pre)
    lines=['# 圧縮・クラスタリングの生成前診断','',
        'これは正解行動を候補選択に使う上限診断で、検索性能・汎化評価ではない。Long 548行はクラスタfitとメモリに含み、圧縮・クラスタリングだけの下書き誤差を示す。','',
        f"プール={n:,} unique rows（train {pre['pool_parts']['expert_train']:,} + memory {pre['pool_parts']['memory']:,} + heldout Dev {pre['pool_parts']['heldout_dev']:,} + Long {pre['pool_parts']['long_eval']:,}、重複除去後）。Long hard={pre['hard_rows']}行。",'',
        '| 表現 | 水準 | 単位数 | κ | 未割当率 | 保存スカラー値 | 圧縮のみMSE | 主下書きMSE (all / hard) | 主中央値 | XYZ方向cos | Grip符号 | 副下書きMSE |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in cells:
        lines.append(f"| {r['representation']} | {r['level']} | {r['unit_count']} | {r['kappa']:.4g} | {r['unassigned_rate_before_fallback']:.3f} | {r['storage']['memory_store_scalar_values_total']} | {r['compression_only_mean_mse']:.5f} | {r['primary_all']['mean_mse']:.5f} / {r['primary_hard']['mean_mse']:.5f} | {r['primary_mse_median']:.5f} | {r['primary_all']['xyz_direction_cos_mean']:.3f} | {r['primary_all']['gripper_sign_match']:.3f} | {r['secondary_all']['mean_mse']:.5f} |")
    lines += ['', 'Top-32候補絞り込みは代表の自己条件付き復元を使い、Long行で候補内をquery条件で厳密評価した。抜き取り全件照合のmiss数を各セルJSONに記録。','',
        '次段のExpert学習はこの表が確定した後に開始する。']
    (OUT/'results.md').write_text('\n'.join(lines)+'\n')
    docs=REPO_ROOT/'docs/research/07_pi05_memory40/compress_cluster_oracle/results.md'
    docs.parent.mkdir(parents=True,exist_ok=True)
    if docs.exists(): raise FileExistsError(docs)
    docs.write_text('# 圧縮・クラスタリングの生成前表\n\n'+(OUT/'results.md').read_text()+'\nExpert生成前。正解行動を候補選択に使う上限診断である。\n')
    (OUT/'.pre_generation.lock').unlink(missing_ok=True)


def compose_pool(train_ids, memory_ids, dev_ids, long_ids):
    """Union dataset row IDs; caller retains explicit query identity."""
    parts = [np.asarray(x, np.int64) for x in (train_ids, memory_ids, dev_ids, long_ids)]
    if any(x.ndim != 1 for x in parts):
        raise ValueError('pool members must be 1D dataset row IDs')
    return np.unique(np.concatenate(parts)).astype(np.int64)


def row_id_to_local(pool_ids):
    ids = np.asarray(pool_ids, np.int64)
    if ids.ndim != 1 or len(np.unique(ids)) != len(ids):
        raise ValueError('pool row IDs must be unique and one-dimensional')
    return {int(r): i for i, r in enumerate(ids)}


def map_row_ids(pool_ids, query_ids):
    lookup = row_id_to_local(pool_ids)
    q = np.asarray(query_ids, np.int64)
    try:
        return np.asarray([lookup[int(r)] for r in q.reshape(-1)], np.int64).reshape(q.shape)
    except KeyError as e:
        raise AssertionError(f'query row {e.args[0]} is absent from memory pool') from e


def assert_pool_coverage(pool_ids, query_ids):
    map_row_ids(pool_ids, query_ids)
    return True


def assert_long_excluded_from_expert(train_ids, rows):
    ids = np.asarray(train_ids, np.int64)
    if np.any(np.isin(np.asarray(rows['task'])[ids], list(LONG_TASKS))):
        raise AssertionError('Long rows entered Expert training')
    return True


def active7_mse(left, right):
    a, b = np.asarray(left, np.float32), np.asarray(right, np.float32)
    if a.shape != b.shape or a.shape[-1] != 7:
        raise ValueError(f'active7 arrays must have equal shapes ending in 7: {a.shape}, {b.shape}')
    if a.ndim == 2:
        return float(np.mean(np.square(a-b)))
    return np.mean(np.square(a - b), axis=tuple(range(1, a.ndim)))


def stable_other_task_shuffle(source_rows, query_rows, tasks, seed=4101):
    src, q, task = np.asarray(source_rows, np.int64), np.asarray(query_rows, np.int64), np.asarray(tasks)
    if src.shape != q.shape:
        raise ValueError('source and query row arrays must align')
    rng = np.random.default_rng(seed)
    from scipy.optimize import linear_sum_assignment
    source_tasks=task[src]; query_tasks=task[q]
    # Find a bijection subject to source_task != query_task. Random jitter
    # preserves exchangeability among valid assignments without hoping for an
    # astronomically unlikely unconstrained permutation.
    cost=np.where(query_tasks[:,None]==source_tasks[None,:],1e6,0.).astype(np.float64)
    cost += rng.uniform(0.,1e-3,size=cost.shape)
    row_ind,col_ind=linear_sum_assignment(cost)
    if not np.array_equal(row_ind,np.arange(len(q))) or np.any(source_tasks[col_ind]==query_tasks):
        raise RuntimeError('the selected source set has no all-cross-task bijection')
    return src[col_ind],col_ind


def _raw_pairwise(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    an = np.einsum('ij,ij->i', a, a); bn = np.einsum('ij,ij->i', b, b)
    return np.maximum(an[:, None] + bn[None, :] - 2 * (a @ b.T), 0.) / a.shape[1]


def _cos_xyz(pred, target):
    p = np.asarray(pred, np.float64)[..., :3].reshape(-1, 10, 3).sum(axis=1)
    t = np.asarray(target, np.float64)[..., :3].reshape(-1, 10, 3).sum(axis=1)
    den = np.linalg.norm(p, axis=1) * np.linalg.norm(t, axis=1)
    return np.divide(np.sum(p*t, axis=1), den, out=np.zeros_like(den), where=den > 1e-12)


def _metrics(pred, target):
    p, y = np.asarray(pred, np.float32), np.asarray(target, np.float32)
    mse = np.mean((p-y)**2, axis=(1,2))
    return {'n': int(len(y)), 'mean_mse': float(mse.mean()), 'median_mse': float(np.median(mse)),
            'xyz_direction_cos_mean': float(_cos_xyz(p, y).mean()),
            'gripper_sign_match': float(np.mean(np.sign(p[:,:,6]) == np.sign(y[:,:,6])))}


def _bootstrap_diff(a, b, episode, *, reps=REPS, seed=20261105):
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    ep = np.asarray(episode)
    uniq = np.unique(ep)
    groups = {x: np.flatnonzero(ep == x) for x in uniq}
    rng = np.random.default_rng(seed)
    means = np.empty(reps, np.float64)
    for i in range(reps):
        draw = rng.choice(uniq, len(uniq), replace=True)
        ix = np.concatenate([groups[x] for x in draw])
        means[i] = d[ix].mean()
    return {'mean': float(d.mean()), 'ci95': [float(np.quantile(means,.025)), float(np.quantile(means,.975))],
            'episodes': int(len(uniq)), 'rows': int(len(d)), 'reps': int(reps)}


def _load_input():
    base = base_module(); fair = fair_module()
    rows = base.load_rows(RUN)
    actions = np.load(RUN/'rows/actions.npy', mmap_mode='r', allow_pickle=False)
    features = np.load(RUN/'pooled.npy', mmap_mode='r', allow_pickle=False)
    train = fair.training_rows(rows)
    dev = fair.validation_rows(rows)
    memory = np.asarray(np.load(MEMORIES/'raw-rows-all40/units.npz', allow_pickle=False)['pool_row_ids'], np.int64)
    longz = np.load(ROOT/'topk-long-01/production/predictions_and_errors.npz', allow_pickle=False)
    long = np.asarray(longz['query_rows'], np.int64)
    hard = np.asarray(longz['hard_mask'], bool)
    if (len(train), len(memory), len(dev), len(long), int(hard.sum())) != (11712, 6583, 213, 548, 63):
        raise AssertionError((len(train),len(memory),len(dev),len(long),int(hard.sum())))
    assert_long_excluded_from_expert(train, rows)
    pool = compose_pool(train, memory, dev, long)
    assert_pool_coverage(pool, np.r_[train, memory, dev, long])
    if np.any(np.asarray(rows['task'])[long] >= 10):
        raise AssertionError('Long query list contains non-Long rows')
    if not np.array_equal(long, np.asarray(np.load(ROOT/'memory-gap-01/predictions_and_errors.npz',allow_pickle=False)['row_ids'])[
            np.asarray(rows['task'][np.load(ROOT/'memory-gap-01/predictions_and_errors.npz',allow_pickle=False)['row_ids']]) < 10]):
        raise AssertionError('Long row ordering differs from memory-gap fixed Eval')
    return base, fair, rows, actions, features, train, dev, memory, long, hard, pool


def load_codec(name):
    import torch
    torch.set_num_threads(8)
    sys.path.insert(0, str(PI05/'compress_vs_raw'))
    from dimension_extension import load_codec as restore
    dims = {'b12': ('b12m',12), 'b24': ('b24',24), 'b48': ('b48',48), 'b96': ('b96',96)}
    key, dim = dims[name]
    path = CODEC_DIR/'codecs'/f'{key}.pt'
    trained = CODEC_DIR/f'E-{key}'/'trained.json'
    if not path.is_file() or not trained.is_file():
        raise FileNotFoundError(f'codec not complete yet: {path} and {trained} are required')
    payload = torch.load(path, map_location='cpu', weights_only=False)
    codec = restore(payload, dim)
    return codec, path


def _make_metric(values, kind, codec, rows, features, pool_ids, probe_local):
    from skill_memories.build import RawFunctional, CodecFunctional
    if kind == 'raw':
        return RawFunctional()
    probe_ids = np.asarray(pool_ids, np.int64)[np.asarray(probe_local, np.int64)]
    _, probe_r = codec.prior(np.asarray(features[probe_ids], np.float32), np.asarray(rows['state'][probe_ids], np.float32))
    # Probe selection was made on nonLong rows only and is then frozen for every fit.
    return CodecFunctional(codec, features[probe_ids], np.asarray(rows['state'][probe_ids]), np.asarray(probe_r), batch_bases=16)


def _nonlong_probe(pool_ids, rows, seed=SEED):
    mask = np.asarray(rows['task'][np.asarray(pool_ids, np.int64)]) >= 10
    local = np.flatnonzero(mask)
    if len(local) < 64:
        raise AssertionError('not enough non-Long rows for fixed 64 decoder probes')
    return np.sort(np.random.default_rng(seed+911).choice(local, 64, replace=False)).astype(np.int64)


def _safe_metric_nearest(metric, value, centers):
    """Array-safe equivalent of the existing M6 metric nearest lookup."""
    if len(centers) == 0:
        return None, float('inf')
    c = np.asarray(centers, np.float32)
    l2 = np.linalg.norm(c - np.asarray(value, np.float32)[None], axis=1)
    ix = np.argsort(l2, kind='stable')[:min(10, len(l2))]
    d = [metric.distance(value, c[i]) for i in ix]
    j = min(range(len(ix)), key=lambda k: (d[k], int(ix[k])))
    return int(ix[j]), float(d[j])


def _encode_values(kind, actions, rows, features, pool_ids, codec):
    ids = np.asarray(pool_ids, np.int64)
    if kind == 'raw':
        return np.asarray(actions[ids, :, :7], np.float32).reshape(len(ids), 70)
    b, _ = codec.encode(np.asarray(actions[ids], np.float32), np.asarray(rows['state'][ids], np.float32))
    return np.asarray(b, np.float32)


def _fit_unit(values, pool_ids, rows, metric, name, nonlong_local, kappa_cache=None):
    from skill_memories.build import m6_fit
    # skill_memories.build.M6Metric.nearest uses ``if not centers`` and raises
    # on NumPy center arrays in the default assignment/fallback path. Keep its
    # distance semantics, while normalizing only that container check here.
    class _SafeM6Metric:
        def distance(self, left, right):
            return metric.distance(left, right)
        def nearest(self, value, centers):
            return _safe_metric_nearest(metric, value, centers)
    safe_metric = _SafeM6Metric()
    ids=np.asarray(pool_ids,np.int64); tasks=np.asarray(rows['task'][ids]); eps=np.asarray(rows['episode'][ids])
    nl=np.asarray(nonlong_local,np.int64)
    # The scale calibration, including its random-pair median, sees no Long rows.
    rng=np.random.default_rng(SEED+6046)
    pair=rng.integers(0,len(nl),size=(64,2))
    med=float(np.median([metric.distance(values[nl[a]],values[nl[b]]) for a,b in pair]))
    if name == 'none':
        return {'row_units':np.arange(len(ids),dtype=np.int32),'centers':values.copy(),
                'members_local':[np.asarray([i],np.int64) for i in range(len(ids))],
                'support':np.ones(len(ids),np.int32),'epsilon':0.,'kappa':0.,'median':med,
                'pending_local':np.empty(0,np.int64),'expired_pending':0}
    if name == 'm6default':
        fit=m6_fit(values,ids,tasks,eps,safe_metric,kappa=.1,seed=SEED,min_support=10,min_episodes=3,
                   epsilon_override=max(.1*med,1e-12))
        fit.update(kappa=.1,median=med)
        return fit
    target={'fine':3000,'mid':300,'coarse':30}[name]
    # κ is selected only from nonLong examples. The selected value is then fit
    # once on the full pool, including Long queries as explicitly requested.
    if kappa_cache is not None and target in kappa_cache:
        kappa,nl_count,table=kappa_cache[target]
    else:
        table=[]
        for kappa0 in KAPPAS:
            fit_nl=m6_fit(values[nl],ids[nl],tasks[nl],eps[nl],safe_metric,kappa=kappa0,seed=SEED,
                          min_support=1,min_episodes=1,epsilon_override=max(kappa0*med,1e-12))
            table.append((abs(int(fit_nl['registered'])-target),float(kappa0),int(fit_nl['registered'])))
        _,kappa,nl_count=min(table,key=lambda x:(x[0],x[1]))
        if kappa_cache is not None: kappa_cache[target]=(kappa,nl_count,table)
    fit=m6_fit(values,ids,tasks,eps,safe_metric,kappa=kappa,seed=SEED,min_support=1,min_episodes=1,
               epsilon_override=max(kappa*med,1e-12))
    fit.update(kappa=kappa,median=med,selection_target=target,nonlong_units=nl_count,
               kappa_grid=[{'kappa':k,'units':n} for _,k,n in sorted(table,key=lambda x:x[1])])
    return fit


def _unit_reps(fit, values, metric, pool_ids):
    from skill_memories.build import unit_medoid
    reps_local=np.asarray([unit_medoid(m,values,metric) for m in fit['members_local']],np.int64)
    return reps_local,np.asarray(pool_ids,np.int64)[reps_local]


def _codec_decode(codec, bases, query_ids, rows, features, actions, reps, batch=64):
    qids=np.asarray(query_ids,np.int64); reps=np.asarray(reps,np.int64)
    out=np.empty((len(qids),len(reps),10,32),np.float32)
    for st in range(0,len(qids),batch):
        sl=slice(st,min(st+batch,len(qids))); q=qids[sl]
        _,r=codec.prior(np.asarray(features[q],np.float32),np.asarray(rows['state'][q],np.float32))
        nq=len(q); k=len(reps)
        bb=np.broadcast_to(np.asarray(bases[reps],np.float32)[None],(nq,k,bases.shape[1])).reshape(nq*k,-1)
        rr=np.broadcast_to(np.asarray(r,np.float32)[:,None,:],(nq,k,r.shape[-1])).reshape(nq*k,-1)
        ff=np.broadcast_to(np.asarray(features[q],np.float32)[:,None,:],(nq,k,features.shape[1])).reshape(nq*k,-1)
        ss=np.broadcast_to(np.asarray(rows['state'][q],np.float32)[:,None,:],(nq,k,rows['state'].shape[1])).reshape(nq*k,-1)
        out[sl]=np.asarray(codec.decode(bb,rr,ff,ss),np.float32).reshape(nq,k,10,32)
    return out


def _decode_variable_candidates(codec, bases, query_ids, rows, features, candidate_rows, *, chunk=1024):
    """Decode Q different K-candidate lists in chunks, conditioning on each query."""
    q=np.asarray(query_ids,np.int64); c=np.asarray(candidate_rows,np.int64)
    if c.ndim!=2 or c.shape[0]!=len(q): raise ValueError('candidate row matrix must be [Q,K]')
    _,r=codec.prior(np.asarray(features[q],np.float32),np.asarray(rows['state'][q],np.float32))
    qn,k=c.shape
    bb=np.asarray(bases[c],np.float32).reshape(qn*k,-1)
    rr=np.broadcast_to(np.asarray(r,np.float32)[:,None,:],(qn,k,r.shape[-1])).reshape(qn*k,-1)
    ff=np.broadcast_to(np.asarray(features[q],np.float32)[:,None,:],(qn,k,features.shape[1])).reshape(qn*k,-1)
    ss=np.broadcast_to(np.asarray(rows['state'][q],np.float32)[:,None,:],(qn,k,rows['state'].shape[1])).reshape(qn*k,-1)
    out=np.empty((qn*k,10,32),np.float32)
    for st in range(0,qn*k,chunk):
        sl=slice(st,min(st+chunk,qn*k))
        out[sl]=np.asarray(codec.decode(bb[sl],rr[sl],ff[sl],ss[sl]),np.float32)
    return out.reshape(qn,k,10,32)


def _predictions_for_queries(kind, codec, bases, values, fit, reps_local, reps_rows,
                             pool_ids, query_ids, rows, actions, features, *, full_scan=False):
    q=np.asarray(query_ids,np.int64); target=np.asarray(actions[q,:,:7],np.float32)
    if kind=='raw':
        rep_actions=np.asarray(actions[reps_rows,:,:7],np.float32)
        dist=np.empty((len(q),len(reps_rows)),np.float32)
        for st in range(0,len(q),64):
            sl=slice(st,min(st+64,len(q)))
            dist[sl]=_raw_pairwise(target[sl].reshape(-1,70),rep_actions.reshape(-1,70))
        chosen=dist.argmin(axis=1)
        primary=np.asarray(actions[reps_rows[chosen]],np.float32)
        # Row-to-unit map preserves dataset row-ID semantics for own membership.
        qlocal=map_row_ids(pool_ids,q); ownu=np.asarray(fit['row_units'])[qlocal]
        ownidx=np.maximum(ownu,0)
        secondary=np.asarray(actions[reps_rows[ownidx]],np.float32)
        secondary[ownu<0]=np.nan
        return primary,secondary,reps_rows[chosen],{'full_scan':True,'missed_by_prefilter':0}

    # Candidate prefilter: self-conditioned reconstruction against target.
    reps=np.asarray(reps_rows,np.int64)
    selfr=codec.prior(np.asarray(features[reps],np.float32),np.asarray(rows['state'][reps],np.float32))[1]
    selfpred=np.asarray(codec.decode(np.asarray(bases[reps],np.float32),np.asarray(selfr,np.float32),
                            np.asarray(features[reps],np.float32),np.asarray(rows['state'][reps],np.float32)),np.float32)[:,:,:7]
    k=len(reps) if full_scan else min(32,len(reps))
    primary=np.empty((len(q),10,32),np.float32); chosen=np.empty(len(q),np.int64)
    for st in range(0,len(q),64):
        sl=slice(st,min(st+64,len(q)))
        pre=np.mean((target[sl,None]-selfpred[None])**2,axis=(2,3))
        shortlist=np.argpartition(pre,k-1,axis=1)[:,:k]
        candidate_rows=reps[shortlist]
        pred=_decode_variable_candidates(codec,bases,q[sl],rows,features,candidate_rows)[:,:,:,:7]
        d=np.mean((pred-target[sl,None])**2,axis=(2,3))
        best=np.argmin(d,axis=1); chosen[sl]=shortlist[np.arange(len(best)),best]
        primary[sl]=_decode_variable_candidates(codec,bases,q[sl],rows,features,reps[chosen[sl],None])[:,0]
    # Full-scan check on a fixed subset. No output selection is changed by this diagnostic.
    check_n=min(16,len(q)); missed=0
    if not full_scan and check_n:
        cq=q[:check_n]; ct=target[:check_n]
        pre=np.mean((ct[:,None]-selfpred[None])**2,axis=(2,3))
        shortlist=np.argpartition(pre,k-1,axis=1)[:,:k]
        shortlist_pred=_decode_variable_candidates(codec,bases,cq,rows,features,reps[shortlist])[:,:,:,:7]
        shortlist_d=np.mean((shortlist_pred-ct[:,None])**2,axis=(2,3)).min(axis=1)
        exact_best=np.full(check_n,np.inf,np.float32)
        for ci in range(0,len(reps),max(1,1024//check_n)):
            chunk_rows=reps[ci:ci+max(1,1024//check_n)]
            cr=np.broadcast_to(chunk_rows,(check_n,len(chunk_rows)))
            dec=_decode_variable_candidates(codec,bases,cq,rows,features,cr)[:,:,:,:7]
            exact_best=np.minimum(exact_best,np.mean((dec-ct[:,None])**2,axis=(2,3)).min(axis=1))
        missed=int(np.sum(exact_best < shortlist_d-1e-7))
    qlocal=map_row_ids(pool_ids,q); ownu=np.asarray(fit['row_units'])[qlocal]
    ownidx=np.maximum(ownu,0); secondary_rows=np.asarray(reps_rows[ownidx],np.int64)
    secondary=np.empty_like(primary)
    for st in range(0,len(q),64):
        sl=slice(st,min(st+64,len(q)))
        secondary[sl]=_decode_variable_candidates(
            codec,bases,q[sl],rows,features,secondary_rows[sl,None])[:,0]
    secondary[ownu<0]=np.nan
    return primary,secondary,np.asarray(reps_rows)[chosen],{'full_scan':bool(full_scan),'prefilter_k':k,
        'missed_by_prefilter':missed,'full_scan_check_rows':int(check_n)}


def _condition_fit(name, kind, codec, rows, actions, features, pool, train, dev, long, hard,
                   values, metric, fit, reps_local, reps_rows, primary_by_id, secondary_by_id):
    # Worker called by training stage to create exact row-aligned drafts.
    pass


def _dev_t0_predictions(model, bridge, obs, draft, noise, target, batch=8):
    import jax
    import jax.numpy as jnp
    outs={}
    for t0 in (.2,.4,.6):
        pred=np.empty_like(target,np.float32)
        for st in range(0,len(target),batch):
            sl=slice(st,min(st+batch,len(target)))
            y=bridge(jax.random.key(0),jax.tree.map(lambda x:x[sl],obs),jnp.asarray(draft[sl]),
                     jnp.full((len(target[sl]),),t0),num_steps=10,noise=jnp.asarray(noise[sl]))
            pred[sl]=np.asarray(y,np.float32)[:,:,:7]
        outs[t0]=float(np.mean((pred-target)**2))
    return outs


def build_missing_draft_caches(*, threads=8):
    """Create row-aligned train/dev primary+secondary drafts for queue cells."""
    import torch
    torch.set_num_threads(threads)
    base,fair,rows,actions,features,train,dev,memory,long,hard,pool=_load_input()
    jobs=[(k,l) for k in ('raw','b12','b48') for l in ('none','mid','coarse','m6default')]
    jobs += [(k,l) for k in ('b24','b96') for l in ('none','coarse')]
    jobs += [('raw','fine'),('b12','fine'),('b24','mid'),('b48','mid'),('b96','mid')]
    for kind,level in jobs:
        cache=np.load(OUT/f'pre_{kind}_{level}.npz',allow_pickle=False)
        rep_rows=np.asarray(cache['representative_rows'],np.int64)
        pool_ids=np.asarray(cache['pool_row_ids'],np.int64)
        fit={'row_units':np.asarray(cache['unit_ids'],np.int32)}
        if kind=='raw':
            values=np.asarray(actions[pool_ids,:,:7],np.float32).reshape(len(pool_ids),70)
            codec=bases=None
        else:
            codec,_=load_codec(kind)
            encoded,_=codec.encode(np.asarray(actions[pool_ids],np.float32),np.asarray(rows['state'][pool_ids],np.float32))
            bases=np.zeros((len(actions),np.asarray(encoded).shape[-1]),np.float32)
            bases[pool_ids]=np.asarray(encoded,np.float32)
            values=bases[pool_ids]
        for split,ids in (('train',train),('dev',dev)):
            path=OUT/f'drafts_{kind}_{level}_{split}.npz'
            if path.exists():
                z=np.load(path,allow_pickle=False)
                if not np.array_equal(z['row_ids'],ids): raise AssertionError(f'cached draft IDs mismatch: {path}')
                continue
            primary,secondary,selected,_=_predictions_for_queries(kind,codec,bases,values,fit,
                np.arange(len(rep_rows),dtype=np.int64),rep_rows,pool_ids,ids,rows,actions,features)
            np.savez_compressed(path,row_ids=ids,primary=primary,secondary=secondary,
                                selected_representative_rows=selected)
            print(json.dumps({'stage':'draft_cache','representation':kind,'level':level,
                              'split':split,'rows':len(ids)}),flush=True)


def train_one(kind, level, *, threads=8, attempt=1):
    """Train a fair-expert-matched repair Expert using the oracle primary draft."""
    exp=OUT/'experts'/f'{kind}_{level}'/f'attempt-{attempt:02d}'
    if (exp/'trained.json').exists(): return json.loads((exp/'trained.json').read_text())
    if exp.exists() and any(exp.iterdir()):
        raise FileExistsError(f'incomplete Expert artifacts exist; inspect rather than overwrite: {exp}')
    import torch
    torch.set_num_threads(threads)
    import jax
    import jax.numpy as jnp
    import flax.nnx as nnx
    import optax
    import orbax.checkpoint as ocp
    from openpi.shared.nnx_utils import PathRegex, module_jit
    base,fair,rows,actions,features,train,dev,memory,long,hard,pool=_load_input()
    train_cache=np.load(OUT/f'drafts_{kind}_{level}_train.npz',allow_pickle=False)
    dev_cache=np.load(OUT/f'drafts_{kind}_{level}_dev.npz',allow_pickle=False)
    if not np.array_equal(train_cache['row_ids'],train) or not np.array_equal(dev_cache['row_ids'],dev):
        raise AssertionError('train/dev oracle drafts row IDs do not match fair-expert split')
    if np.any(np.isin(np.asarray(rows['task'][train]),list(LONG_TASKS))):
        raise AssertionError('Long entered Expert training')
    batch_ids=np.load(ROOT/'fair-expert-01/batch_row_ids.npy',allow_pickle=False)
    if batch_ids.shape!=(4000,8): raise AssertionError(f'fair-expert minibatch sequence changed: {batch_ids.shape}')
    if not np.all(np.isin(batch_ids,train)): raise AssertionError('batch row outside fixed fair-expert training rows')
    train_lookup={int(r):i for i,r in enumerate(train_cache['row_ids'])}
    batch_drafts=np.asarray(train_cache['primary'],np.float32)
    if batch_drafts.shape!=(len(train),10,32): raise AssertionError(f'bad draft shape {batch_drafts.shape}')
    start=time.monotonic()
    model,_=base.load_policy(Path(os.environ.get('SKILLNET_PI05_CHECKPOINT',STATE_DIR/'checkpoints/openpi-assets/checkpoints/pi05_libero')).expanduser())
    filt=nnx.All(nnx.Param,PathRegex(r'.*(llm.*_1|action_in_proj|time_mlp|action_out_proj).*'))
    state=nnx.state(model,filt)
    if sum(x.value.size for x in state.flat_state().values())<100_000_000:
        raise AssertionError('trainable action-expert filter selected too few parameters')
    tx=optax.chain(optax.clip_by_global_norm(1.),optax.adamw(2.5e-5,b1=.9,b2=.95,weight_decay=1e-5))
    opt=tx.init(state); step_fn=base.make_bridge_step(filt,tx); key=jax.random.key(SEED)
    exp.mkdir(parents=True); (exp/'bridge').mkdir()
    dev_target=np.asarray(actions[dev,:,:7],np.float32)
    dev_draft=np.asarray(dev_cache['primary'],np.float32)
    dev_noise=np.stack([np.asarray(jax.random.normal(jax.random.fold_in(jax.random.key(88261),int(r)),(10,32))) for r in dev])
    dev_obs=base.obs_batch(rows,dev)
    best=float('inf'); best_info=None; logs=[]
    for step,ids in enumerate(batch_ids,start=1):
        di=np.asarray([train_lookup[int(r)] for r in ids],np.int64)
        draft=jnp.asarray(batch_drafts[di])
        key,sub=jax.random.split(key)
        loss,opt=step_fn(model,opt,base.obs_batch(rows,ids),jnp.asarray(actions[ids]),draft,sub)
        loss=float(loss)
        if not np.isfinite(loss): raise FloatingPointError(f'{kind}/{level} nonfinite at {step}')
        if step%500==0:
            ckpt=exp/'bridge'/f'{step:06d}'
            ocp.PyTreeCheckpointer().save(ckpt,nnx.state(model,filt).to_pure_dict())
            bridge=module_jit(model.sample_bridge,static_argnames=('num_steps',))
            vals=_dev_t0_predictions(model,bridge,dev_obs,dev_draft,dev_noise,dev_target)
            t0=min(vals,key=vals.get); vmse=vals[t0]
            record={'step':step,'train_loss':loss,'dev_mse':{str(k):v for k,v in vals.items()},
                    'selected_t0':float(t0),'elapsed_s':time.monotonic()-start}
            logs.append(record)
            if vmse<best:
                best=vmse; best_info={'path':str(ckpt),'steps':step,'t0':float(t0),
                    'validation_generated_active7_mse':float(vmse),'selection':'heldout-six-task Dev primary oracle draft only; no Long'}
                save_json(exp/'best_bridge.json',best_info)
            save_json(exp/'training_progress.json',{'status':'running','kind':kind,'level':level,'steps_done':step,
                       'logs':logs,'best':best_info,'seed':SEED})
            print(json.dumps({'stage':'train','kind':kind,'level':level,**record}),flush=True)
    trained={'representation':kind,'level':level,'steps':4000,'seed':SEED,'batch_size':8,
        'train_rows':len(train),'dev_rows':len(dev),'batch_row_ids_sha256':hashlib.sha256(batch_ids.tobytes()).hexdigest(),
        'trainable_filter':'llm.*_1|action_in_proj|time_mlp|action_out_proj',
        'optimizer':'AdamW lr=2.5e-5 b1=.9 b2=.95 wd=1e-5; clip=1',
        'loss':'same fair-expert bridge velocity MSE; only primary one-row draft differs',
        'best':best_info,'seconds':time.monotonic()-start,'logs':logs}
    save_json(exp/'trained.json',trained)
    save_json(exp/'training_progress.json',{'status':'complete','kind':kind,'level':level,'steps_done':4000,'best':best_info})
    return trained


def evaluate_one(kind, level, *, threads=8, batch=8, attempt=1):
    dest=OUT/'experts'/f'{kind}_{level}'/f'attempt-{attempt:02d}'
    result_path=dest/'evaluation.json'
    if result_path.exists(): return json.loads(result_path.read_text())
    import torch
    torch.set_num_threads(threads)
    import jax
    import jax.numpy as jnp
    from openpi.shared.nnx_utils import module_jit
    base,fair,rows,actions,features,train,dev,memory,long,hard,pool=_load_input()
    trained=json.loads((dest/'trained.json').read_text()); best=trained['best']
    exp=Path(best['path'])
    if not exp.exists(): raise FileNotFoundError(exp)
    data=np.load(OUT/f'pre_{kind}_{level}.npz',allow_pickle=False)
    if not np.array_equal(data['query_rows'],long): raise AssertionError('Long row order mismatch')
    pc=np.load(OUT/f'drafts_{kind}_{level}_dev.npz',allow_pickle=False) if False else None
    prep=np.load(OUT/f'pre_{kind}_{level}.npz',allow_pickle=False)
    primary=np.asarray(prep['primary'],np.float32); secondary=np.asarray(prep['secondary'],np.float32)
    if primary.shape[0]!=len(long): raise AssertionError('primary Long draft row mismatch')
    task=np.asarray(rows['task'][long]); ep=np.asarray(rows['episode'][long]); cluster=task.astype(np.int64)*100000+ep
    shuffled_rows,perm=stable_other_task_shuffle(np.asarray(prep['primary_rep_rows'],np.int64),long,rows['task'],seed=SEED+4101)
    # Drafts are generated from the same selected row IDs as the primary oracle picks.
    # Recover those dataset rows' raw actions/encodings with the output cache or codec below.
    if kind=='raw':
        shuffled=np.asarray(actions[shuffled_rows],np.float32)
    else:
        codec,_=load_codec(kind)
        bases=np.zeros((len(actions),int(np.asarray(np.load(OUT/f'pre_{kind}_{level}.npz',allow_pickle=False)['representative_rows']).dtype==np.int64)),np.float32) if False else None
        # A selected medoid's query-conditioned decode is not transferable across queries;
        # generate the shuffled candidate under the recipient query's context.
        all_pool=np.asarray(prep['pool_row_ids'],np.int64)
        vals_path=OUT/'representations'/f'{kind}_bases.npy'
        if not vals_path.exists():
            ids=all_pool; vals=np.asarray(codec.encode(np.asarray(actions[ids],np.float32),np.asarray(rows['state'][ids],np.float32))[0],np.float32)
            bases_full=np.zeros((len(actions),vals.shape[1]),np.float32); bases_full[ids]=vals
            vals_path.parent.mkdir(exist_ok=True); np.save(vals_path,bases_full)
        else: bases_full=np.load(vals_path,mmap_mode='r',allow_pickle=False)
        shuffled=np.asarray(codec.decode(np.asarray(bases_full[shuffled_rows],np.float32),
             np.asarray(codec.prior(np.asarray(features[long],np.float32),np.asarray(rows['state'][long],np.float32))[1],np.float32),
             np.asarray(features[long],np.float32),np.asarray(rows['state'][long],np.float32)),np.float32)
    zero=np.zeros((len(long),10,32),np.float32)
    official=np.load(ROOT/'fair-expert-01/evaluation-final/offline_predictions.npz',allow_pickle=False)
    if not np.array_equal(official['row_ids'],long): raise AssertionError('fair-expert reference row mismatch')
    refs={'official_normal':np.asarray(official['pred_official_normal'],np.float32)}
    em2_best=json.loads((ROOT/'fair-expert-01/E-M2/best_bridge.json').read_text())
    ez_best=json.loads((ROOT/'fair-expert-01/E-zero/best_bridge.json').read_text())
    refs['E-M2_M2_top1']=np.asarray(official[f"pred_E-M2_M2_top1_t0_{float(em2_best['t0']):.1f}"],np.float32)
    refs['E-zero_zero']=np.asarray(official[f"pred_E-zero_zero_t0_{float(ez_best['t0']):.1f}"],np.float32)
    policy,_=base.load_policy(Path(os.environ.get('SKILLNET_PI05_CHECKPOINT',STATE_DIR/'checkpoints/openpi-assets/checkpoints/pi05_libero')).expanduser())
    from_path=Path(best['path'])
    # The module's checked loader restores only the selected action expert leaves.
    sys.path.insert(0,str(PI05/'skill_expert'))
    from skill_expert.run import load_expert
    load_expert(policy,from_path)
    bridge=module_jit(policy.sample_bridge,static_argnames=('num_steps',)); obs=base.obs_batch(rows,long)
    target=np.asarray(actions[long,:,:7],np.float32)
    regenerated=np.stack([np.asarray(jax.random.normal(jax.random.fold_in(jax.random.key(4101),int(r)),(10,32))) for r in long])
    # Reuse the exact fair-expert bank; regenerated CPU values can differ by a
    # few ulps from the original accelerator path while preserving row seeds.
    noise=np.asarray(official['noise'],np.float32)
    noise_max_abs=float(np.max(np.abs(regenerated-noise)))
    if noise_max_abs>1e-6: raise AssertionError(f'noise bank differs from seed=4101 fold_in(row_id): {noise_max_abs}')
    candidates={'primary':primary,'secondary':secondary,'shuffle':shuffled,'zero':zero}
    predictions={}; errors={}
    for name,draft in candidates.items():
        for t0 in (.2,.4,.6):
            pred=np.empty((len(long),10,7),np.float32)
            for st in range(0,len(long),batch):
                sl=slice(st,min(st+batch,len(long)))
                y=bridge(jax.random.key(0),jax.tree.map(lambda x:x[sl],obs),jnp.asarray(draft[sl]),
                    jnp.full((len(long[sl]),),t0),num_steps=10,noise=jnp.asarray(noise[sl]))
                pred[sl]=np.asarray(y,np.float32)[:,:,:7]
            key=f'{name}_t{t0:.1f}'; predictions[key]=pred
            errors[key]=np.mean((pred-target)**2,axis=(1,2)).astype(np.float32)
    referr={name:np.mean((pred-target)**2,axis=(1,2)).astype(np.float32) for name,pred in refs.items()}
    selected_t0=float(best['t0']); selected=errors[f'primary_t{selected_t0:.1f}']
    boot={}
    for subset,mask in [('all',np.ones(len(long),bool)),('hard',hard)]:
        for against,arr in [('same_expert_shuffle',errors[f'shuffle_t{selected_t0:.1f}']),
                            ('E-zero_zero',referr['E-zero_zero']),('official_normal',referr['official_normal'])]:
            boot[f'{subset}_primary_minus_{against}']=_bootstrap_diff(selected[mask],arr[mask],cluster[mask],seed=4101)
    all_arrays={f'pred_{k}':v for k,v in predictions.items()}
    all_arrays.update({f'mse_{k}':v for k,v in errors.items()})
    all_arrays.update({f'ref_pred_{k}':v for k,v in refs.items()})
    all_arrays.update({f'ref_mse_{k}':v for k,v in referr.items()})
    np.savez_compressed(dest/'evaluation_rows.npz',row_ids=long,hard_mask=hard,target=target,
        primary_draft=primary,secondary_draft=secondary,shuffle_draft=shuffled,noise=noise,
        **all_arrays)
    summary={'representation':kind,'level':level,'selected_t0':selected_t0,'rows':len(long),'hard_rows':int(hard.sum()),
        'metrics':{k:{'all_mean':float(v.mean()),'all_median':float(np.median(v)),
            'hard_mean':float(v[hard].mean()),'hard_median':float(np.median(v[hard]))} for k,v in errors.items()},
        'components':{k:{'xyz_mse':float(np.mean((v[:,:,:3]-target[:,:,:3])**2)),
            'rotation_mse':float(np.mean((v[:,:,3:6]-target[:,:,3:6])**2)),
            'gripper_mse':float(np.mean((v[:,:,6]-target[:,:,6])**2)),
            'xyz_hard_mse':float(np.mean((v[hard,:,:3]-target[hard,:,:3])**2)),
            'rotation_hard_mse':float(np.mean((v[hard,:,3:6]-target[hard,:,3:6])**2)),
            'gripper_hard_mse':float(np.mean((v[hard,:,6]-target[hard,:,6])**2))} for k,v in predictions.items()},
        'references':{k:{'mean':float(v.mean()),'hard_mean':float(v[hard].mean())} for k,v in referr.items()},
        'paired_bootstrap':boot,'shuffle_cross_task':bool(np.all(np.asarray(rows['task'])[shuffled_rows]!=task)),
        'output_sensitivity':{str(t):{
            'primary_vs_shuffle_rmse':float(np.sqrt(np.mean((predictions[f'primary_t{t:.1f}']-predictions[f'shuffle_t{t:.1f}'])**2))),
            'primary_vs_zero_rmse':float(np.sqrt(np.mean((predictions[f'primary_t{t:.1f}']-predictions[f'zero_t{t:.1f}'])**2)))}
            for t in (.2,.4,.6)},
        'episode_bootstrap_reps':REPS,'noise':'reuse fair-expert fixed bank generated by fold_in(key(4101), dataset_row_id)',
        'regenerated_noise_max_abs_difference':noise_max_abs}
    save_json(result_path,summary)
    return summary


def aggregate():
    """Append final paired contrasts and the morning-facing conclusion."""
    import json
    if (OUT/'summary.json').exists() or (OUT/'summary.md').exists():
        raise FileExistsError('refusing to overwrite completed summary artifacts')
    cells={}; artifact_dirs={}
    for p in sorted((OUT/'experts').glob('*/attempt-*/evaluation.json')) if (OUT/'experts').exists() else []:
        z=json.loads(p.read_text()); key=f"{z['representation']}_{z['level']}"
        cells[key]=z; artifact_dirs[key]=p.parent
    by={(v['representation'],v['level']):v for v in cells.values()}
    if not by: raise RuntimeError('no evaluated Experts yet')
    pre_table=json.loads((OUT/'pre_generation.json').read_text())['cells']
    pre_by={(x['representation'],x['level']):x for x in pre_table}
    # Report per-row selected-t0 errors for compression/cluster contrasts.
    diffs={}
    for kind in ('b12','b24','b48','b96'):
        for level in LEVELS:
            if (kind,level) in by and ('raw','none') in by:
                # Bootstrap from saved per-row arrays, using the matched Long row identities.
                aa=np.load(artifact_dirs[f'{kind}_{level}']/'evaluation_rows.npz',allow_pickle=False)
                bb=np.load(artifact_dirs['raw_none']/'evaluation_rows.npz',allow_pickle=False)
                ta=float(by[(kind,level)]['selected_t0']); tb=float(by[('raw','none')]['selected_t0'])
                target=aa['target']; mask=aa['hard_mask']
                rows=_load_input()[2]; qids=np.asarray(aa['row_ids'],np.int64)
                ep=np.asarray(rows['task'][qids],np.int64)*100000+np.asarray(rows['episode'][qids],np.int64)
                mse_a=aa[f'mse_primary_t{ta:.1f}']; mse_b=bb[f'mse_primary_t{tb:.1f}']
                diffs[f'{kind}_{level}_minus_raw_none']={'all':_bootstrap_diff(mse_a,mse_b,ep,seed=20261106),
                    'hard':_bootstrap_diff(mse_a[mask],mse_b[mask],ep[mask],seed=20261106)}
    for level in LEVELS:
        if ('raw',level) in by and ('raw','none') in by and level!='none':
            aa=np.load(artifact_dirs[f'raw_{level}']/'evaluation_rows.npz',allow_pickle=False)
            bb=np.load(artifact_dirs['raw_none']/'evaluation_rows.npz',allow_pickle=False)
            ta=float(by[('raw',level)]['selected_t0']); tb=float(by[('raw','none')]['selected_t0'])
            rows=_load_input()[2]; qids=np.asarray(aa['row_ids'],np.int64)
            ep=np.asarray(rows['task'][qids],np.int64)*100000+np.asarray(rows['episode'][qids],np.int64)
            mask=np.asarray(aa['hard_mask'],bool)
            a=aa[f'mse_primary_t{ta:.1f}']; b=bb[f'mse_primary_t{tb:.1f}']
            diffs[f'raw_{level}_cluster_loss_minus_raw_none']={'all':_bootstrap_diff(a,b,ep,seed=20261107),
                'hard':_bootstrap_diff(a[mask],b[mask],ep[mask],seed=20261107)}
    diagnostics={}
    if ('raw','none') in pre_by:
        base=pre_by[('raw','none')]
        diagnostics['raw_none_pre_generation_mse']=base['primary_all']['mean_mse']
        diagnostics['compression_only_long_error']={k:pre_by[(k,'none')]['compression_only_mean_mse']
            for k in ('b12','b24','b48','b96') if (k,'none') in pre_by}
        diagnostics['compression_loss_pre_generation']={k:pre_by[(k,'none')]['primary_all']['mean_mse']-base['primary_all']['mean_mse']
            for k in ('b12','b24','b48','b96') if (k,'none') in pre_by}
        diagnostics['raw_cluster_loss_pre_generation']={l:pre_by[('raw',l)]['primary_all']['mean_mse']-base['primary_all']['mean_mse']
            for l in LEVELS if ('raw',l) in pre_by and l!='none'}
    save_json(OUT/'summary.json',{'experts':cells,'paired_compression_cluster_contrasts':diffs,
        'pre_generation_diagnostics':diagnostics,
        'limits':['正解行動を見て候補を選ぶ上限診断','評価行自身がクラスタfit/メモリに含まれる','既知タスク・オフラインのみ','単一seed',
                  'codecは事前学習済みでLongを含む学習行から作られた','閉ループ性能・汎化性能を示さない']})

    summary_lines=['# 圧縮・クラスタリングの結果','',
        '## 朝の要約','',
        '本実験は各行の正解を含むプールから正解行動MSE最小の代表を選ぶ、検索なしの上限診断である。圧縮・クラスタリングだけの損失と、同じ候補を修正Expertがデノイズした後に残る差を分けて読む。Long行をプールにもクラスタfitにも含むため、汎化の主張には使わない。']
    if diagnostics:
        comp=', '.join(f"{k}: {v:+.5f}" for k,v in diagnostics.get('compression_loss_pre_generation',{}).items()) or 'なし'
        clust=', '.join(f"{k}: {v:+.5f}" for k,v in diagnostics.get('raw_cluster_loss_pre_generation',{}).items()) or 'なし'
        summary_lines += [f"- 圧縮のみ（none、Long主下書きMSEの差 bN−raw）: {comp}。codecのprior残差を使う自己復元誤差は {json.dumps(diagnostics.get('compression_only_long_error',{}),ensure_ascii=False)}。",
            f"- rawのクラスタ化（Long主下書きMSEの差 水準−none）: {clust}。",
            '- 修正Expert後にどれだけ縮まったかは下表とepisode bootstrap差で判断する。差のCIが0をまたぐ場合に同等とは結論しない。']
    summary_lines += ['',
        '## 生成前の25セル','',
        (OUT/'results.md').read_text() if (OUT/'results.md').exists() else '生成前表は未完成。','',
        '## Expert後の行動MSE','',
        '| 表現 | 水準 | 選択t0 | primary all / hard | secondary all / hard | shuffle all | zero all | 通常π0.5 | E-zero | E-M2 |','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for key,v in cells.items():
        t=v['selected_t0']; metr=v['metrics']; refs=v['references']
        summary_lines.append(f"| {v['representation']} | {v['level']} | {t:.1f} | {metr[f'primary_t{t:.1f}']['all_mean']:.5f} / {metr[f'primary_t{t:.1f}']['hard_mean']:.5f} | {metr[f'secondary_t{t:.1f}']['all_mean']:.5f} / {metr[f'secondary_t{t:.1f}']['hard_mean']:.5f} | {metr[f'shuffle_t{t:.1f}']['all_mean']:.5f} | {metr[f'zero_t{t:.1f}']['all_mean']:.5f} | {refs['official_normal']['mean']:.5f} | {refs['E-zero_zero']['mean']:.5f} | {refs['E-M2_M2_top1']['mean']:.5f} |")
    summary_lines += ['', '## 対差と限界','', 'episode単位bootstrap 5,000回の対差CIは `summary.json` に保存。差の区間が0をまたぐ場合は「同等」と解釈しない。',
        '', '限界：正解を見た上限、評価行自身がクラスタfit/メモリに含まれる、既知タスク、単一seed、Longを含む既成codec、オフライン評価である。検索の実現可能性やシミュレーションでの閉ループ性能を示さない。']
    final_text='\n'.join(summary_lines)+'\n'
    (OUT/'summary.md').write_text(final_text)
    with (OUT/'results.md').open('a') as f:
        f.write('\n\n---\n\n## Expert後の結果\n\n参照: `summary.md` と `summary.json`。\n')
    docs_results=REPO_ROOT/'docs/research/07_pi05_memory40/compress_cluster_oracle/results.md'
    docs_results.parent.mkdir(parents=True,exist_ok=True)
    docs_results.write_text(final_text)


def _build(args):
    start=time.monotonic()
    base,fair,rows,actions,features,train,dev,memory,long,hard,pool=_load_input()
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'pre_generation.json').exists():
        raise FileExistsError(f'refusing to overwrite completed pre-generation table: {OUT / "pre_generation.json"}')
    pool_local=row_id_to_local(pool)
    nonlong_local=np.flatnonzero(np.asarray(rows['task'][pool])>=10)
    query=np.asarray(long,np.int64)
    targets=np.asarray(actions[query,:,:7],np.float32)
    out_cells={}; artifacts={}; all_table=[]
    for name in (f'cell_{k}_{l}.json' for k in ('raw','b12','b24','b48','b96') for l in LEVELS):
        path=OUT/name
        if path.exists():
            record=json.loads(path.read_text())
            if 'memory_store_scalar_values_total' not in record.get('storage',{}):
                dim=70 if record['representation']=='raw' else int(record['representation'][1:])
                units=int(record['unit_count'])
                record['storage']={'fit_pool_representation_scalars':int(len(pool)*dim),
                    'stored_medoid_representation_scalars':units*dim,'stored_medoid_row_ids':units,
                    'stored_pool_row_ids':int(len(pool)),'stored_row_to_unit_ids':int(len(pool)),
                    'memory_store_scalar_values_total':int(units*dim+units+2*len(pool))}
            all_table.append(record)
    target_pairs={(k,l) for k in ('raw','b12','b48') for l in ('none','mid','coarse','m6default')}
    target_pairs |= {(k,l) for k in ('b24','b96') for l in ('none','coarse')}
    kinds=('raw','b12','b24','b48','b96')
    if args.representation:
        if args.representation not in kinds: raise ValueError(f'unknown representation: {args.representation}')
        kinds=(args.representation,)
    for kind in kinds:
        codec=None; codecpath=None
        if kind!='raw': codec,codecpath=load_codec(kind)
        values=_encode_values(kind,actions,rows,features,pool,codec)
        bases=None; ownpred=None; compress=None
        if kind!='raw':
            bases=np.zeros((len(actions),values.shape[1]),np.float32); bases[pool]=values
            ownb,_=codec.encode(np.asarray(actions[pool],np.float32),np.asarray(rows['state'][pool],np.float32))
            _,ownr=codec.prior(np.asarray(features[pool],np.float32),np.asarray(rows['state'][pool],np.float32))
            ownpred=np.asarray(codec.decode(ownb,ownr,np.asarray(features[pool],np.float32),np.asarray(rows['state'][pool],np.float32)),np.float32)
            compress=np.mean((ownpred[:,:,:7]-np.asarray(actions[pool,:,:7],np.float32))**2,axis=(1,2))
        probes=_nonlong_probe(pool,rows)
        metric=_make_metric(values,kind,codec,rows,features,pool,probes)
        kappa_cache={}
        for level in LEVELS:
            cell_path=OUT/f'cell_{kind}_{level}.json'
            if cell_path.exists():
                continue
            print(json.dumps({'stage':'fit_cluster','representation':kind,'level':level,'utc':time.strftime('%FT%TZ',time.gmtime())}),flush=True)
            fit=_fit_unit(values,pool,rows,metric,level,nonlong_local,kappa_cache)
            reps_local,reps_rows=_unit_reps(fit,values,metric,pool)
            unassigned_rate=float(np.mean(np.asarray(fit['row_units'])<0))
            if level=='m6default' and np.any(np.asarray(fit['row_units'])<0):
                # Preserve the unassigned fraction, but define the requested
                # nearest-unit fallback for the secondary candidate.
                fit=dict(fit); ru=np.asarray(fit['row_units']).copy()
                for ix in np.flatnonzero(ru<0):
                    uid,_=_safe_metric_nearest(metric,values[ix],fit['centers'])
                    ru[ix]=int(uid)
                fit['row_units']=ru
            primary,secondary,chosen,pref=_predictions_for_queries(kind,codec,bases,values,fit,reps_local,reps_rows,pool,query,rows,actions,features)
            if (kind,level) in target_pairs:
                for split,ids in [('train',train),('dev',dev)]:
                    draft_path=OUT/f'drafts_{kind}_{level}_{split}.npz'
                    if draft_path.exists():
                        cached=np.load(draft_path,allow_pickle=False)
                        if not np.array_equal(cached['row_ids'],ids): raise AssertionError(f'draft cache row mismatch: {draft_path}')
                        continue
                    pp,ss,cc,pr=_predictions_for_queries(kind,codec,bases,values,fit,reps_local,reps_rows,pool,ids,rows,actions,features)
                    np.savez_compressed(draft_path,row_ids=ids,primary=pp,secondary=ss,selected_representative_rows=cc)
            if kind=='raw': compress=np.zeros(len(pool),np.float32)
            local={int(row):i for i,row in enumerate(pool)}
            longloc=np.asarray([local[int(r)] for r in long],np.int64)
            hardloc=longloc[hard]
            hmask=np.asarray(hard,bool)
            primary_mse=np.mean((primary[:,:,:7]-targets)**2,axis=(1,2))
            secondary_mse=np.nanmean((secondary[:,:,:7]-targets)**2,axis=(1,2))
            # Candidate count/storage counts are actual scalar payload values, not 3857-row estimates.
            unit_count=len(reps_rows); dim=70 if kind=='raw' else int(values.shape[1])
            storage={'fit_pool_representation_scalars':int(len(pool)*dim),
                     'stored_medoid_representation_scalars':int(unit_count*dim),
                     'stored_medoid_row_ids':int(unit_count),
                     'stored_pool_row_ids':int(len(pool)),
                     'stored_row_to_unit_ids':int(len(pool)),
                     'memory_store_scalar_values_total':int(unit_count*dim+unit_count+2*len(pool))}
            record={'representation':kind,'level':level,'unit_count':int(unit_count),
                    'unassigned_rate_before_fallback':unassigned_rate,
                    'kappa':float(fit.get('kappa',0.)),'epsilon':float(fit.get('epsilon',0.)),
                    'nonlong_units_for_kappa_selection':fit.get('nonlong_units'),
                    'pending_rows':int(len(fit.get('pending_local',[]))),
                    'expired_pending':int(fit.get('expired_pending',0)),
                    'storage':storage,'compression_only_mean_mse':float(np.mean(compress[longloc])),
                    'compression_only_hard_mean_mse':float(np.mean(compress[hardloc])),
                    'primary_all':_metrics(primary[:,:,:7],targets),
                    'primary_hard':_metrics(primary[hmask,:,:7],targets[hmask]),
                    'secondary_all':_metrics(secondary[~np.isnan(secondary).any((1,2)),:,:7],targets[~np.isnan(secondary).any((1,2))]),
                    'secondary_hard':_metrics(secondary[hmask & ~np.isnan(secondary).any((1,2)),:,:7],targets[hmask & ~np.isnan(secondary).any((1,2))]),
                    'primary_mse_median':float(np.median(primary_mse)),
                    'secondary_mse_median':float(np.nanmedian(secondary_mse)),
                    'prefilter_validation':pref,'codec_path':str(codecpath) if codecpath else None}
            all_table.append(record)
            save_json(cell_path,record)
            np.savez_compressed(OUT/f'pre_{kind}_{level}.npz',query_rows=query,hard_mask=hard,
                primary=primary,secondary=secondary,primary_mse=primary_mse,secondary_mse=secondary_mse,
                primary_rep_rows=chosen,pool_row_ids=pool,unit_ids=np.asarray(fit['row_units'],np.int32),
                representative_rows=reps_rows,kappa=np.asarray(record['kappa']),epsilon=np.asarray(record['epsilon']))
            if (kind,level) in {(k,l) for k in ('raw','b12','b48') for l in ('none','mid','coarse','m6default')}:
                artifacts[f'{kind}_{level}']={'representation':kind,'level':level,'unit_rows':reps_rows.tolist(),
                                              'row_units':np.asarray(fit['row_units'],np.int32).tolist(),
                                              'pool_row_ids':pool.tolist(),'kappa':record['kappa'],'epsilon':record['epsilon']}
            print(json.dumps({'stage':'cell_done',**{k:record[k] for k in ('representation','level','unit_count','kappa','epsilon')},
                              'primary_long_mse':record['primary_all']['mean_mse'],'elapsed_s':time.monotonic()-start}),flush=True)
            with (OUT/'progress.jsonl').open('a') as pf:
                pf.write(json.dumps({'cell':f'{kind}_{level}','status':'complete','utc':time.strftime('%FT%TZ',time.gmtime()),
                                     'seconds':time.monotonic()-start},allow_nan=True)+'\n')
        del values, metric, codec
        if kind!='raw':
            import gc; gc.collect()
    # Concurrent representation workers write disjoint cell files. Only the
    # worker that observes all 25 cells may publish the shared table; reload
    # records to avoid a stale per-process view.
    expected=[OUT/f'cell_{k}_{l}.json' for k in ('raw','b12','b24','b48','b96') for l in LEVELS]
    all_table=[json.loads(p.read_text()) for p in expected if p.exists()]
    if len(all_table)<25:
        print(json.dumps({'stage':'representation_done','representation':args.representation,
                          'cell_count':len(all_table),'needed':25}),flush=True)
        return all_table
    lock=OUT/'.pre_generation.lock'
    try:
        fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    except FileExistsError:
        return all_table
    os.close(fd)
    if (OUT/'pre_generation.json').exists():
        lock.unlink(missing_ok=True)
        raise FileExistsError(f'refusing to overwrite completed pre-generation table: {OUT / "pre_generation.json"}')
    save_json(OUT/'pre_generation.json',{'rows':len(pool),'pool_parts':{'expert_train':len(train),'memory':len(memory),'heldout_dev':len(dev),'long_eval':len(long)},
        'pool_unique':len(pool),'long_eval_rows':len(long),'hard_rows':int(hard.sum()),'cells':all_table,
        'interpretation':'oracle upper bound; true target chooses candidate; evaluation rows are included in pool; not retrieval/generalization'})
    # The requested descriptive table is written now, before any Expert generation.
    lines=['# 圧縮・クラスタリングの生成前診断','',
           'これは正解行動を候補選択に使う上限診断で、検索性能・汎化評価ではない。Long 548行はクラスタfitとメモリに含み、圧縮・クラスタリングだけの下書き誤差を示す。','',
           f"プール={len(pool):,} unique rows（train {len(train):,} + memory {len(memory):,} + heldout Dev {len(dev):,} + Long {len(long):,}、重複除去後）。Long hard={int(hard.sum())}行。",'',
           '| 表現 | 水準 | 単位数 | κ | 未割当率 | 保存スカラー値 | 圧縮のみMSE | 主下書きMSE (all / hard) | 主中央値 | XYZ方向cos | Grip符号 | 副下書きMSE |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in all_table:
        lines.append(f"| {r['representation']} | {r['level']} | {r['unit_count']} | {r['kappa']:.4g} | {r['unassigned_rate_before_fallback']:.3f} | {r['storage']['memory_store_scalar_values_total']} | {r['compression_only_mean_mse']:.5f} | {r['primary_all']['mean_mse']:.5f} / {r['primary_hard']['mean_mse']:.5f} | {r['primary_mse_median']:.5f} | {r['primary_all']['xyz_direction_cos_mean']:.3f} | {r['primary_all']['gripper_sign_match']:.3f} | {r['secondary_all']['mean_mse']:.5f} |")
    lines += ['', 'Top-32候補絞り込みは代表の自己条件付き復元を使い、Long行で候補内をquery条件で厳密評価した。抜き取り全件照合のmiss数を各セルJSONに記録。', '',
              '次段のExpert学習はこの表が確定した後に開始する。']
    (OUT/'results.md').write_text('\n'.join(lines)+'\n')
    docs_pre=REPO_ROOT/'docs/research/07_pi05_memory40/compress_cluster_oracle/results.md'
    docs_pre.parent.mkdir(parents=True,exist_ok=True)
    docs_pre.write_text('# 圧縮・クラスタリングの生成前表\n\n'+(OUT/'results.md').read_text()+'\nExpert生成前。正解行動を候補選択に使う上限診断である。\n')
    lock.unlink(missing_ok=True)
    return all_table


def main():
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['build','train','evaluate','finalize','publish','drafts'])
    p.add_argument('--representation',default=None)
    p.add_argument('--level',default=None)
    p.add_argument('--threads',type=int,default=8)
    p.add_argument('--batch',type=int,default=8)
    p.add_argument('--attempt',type=int,default=1)
    args=p.parse_args()
    import torch; torch.set_num_threads(args.threads)
    if args.stage=='build':
        _build(args)
    elif args.stage=='train':
        if not args.representation or not args.level: p.error('train requires --representation and --level')
        train_one(args.representation,args.level,threads=args.threads,attempt=args.attempt)
    elif args.stage=='evaluate':
        if not args.representation or not args.level: p.error('evaluate requires --representation and --level')
        evaluate_one(args.representation,args.level,threads=args.threads,batch=args.batch,attempt=args.attempt)
    elif args.stage=='finalize': aggregate()
    elif args.stage=='publish': publish_pre_generation()
    elif args.stage=='drafts': build_missing_draft_caches(threads=args.threads)


if __name__=='__main__':
    main()
