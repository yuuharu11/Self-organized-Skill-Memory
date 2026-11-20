#!/usr/bin/env python3
"""KQ2-CL stage R: task-fold query models, retrieval caches and the retriever's rank law.

  folds     5 task folds of LIBERO-40 (8 tasks each). For (iii) and (i) the KQ2 recipe (same cfg,
            seed 1, epoch chosen on Dev v2 20 Hz) is retrained on the eh rows of the other 32 tasks.
  retrieve  per query in {iii, i}:
              eh      insE  production query, own episode masked
                      insT  production query, every unit of the own task masked
                      oofE  fold query (task never seen), own episode masked
                      oofT  fold query, own task masked (unseen-like)
              learned / unseen20 / dev2: production query, own episode masked (KQ2 rule)
            saves q, top-50 ids/logits, entropy, top-1 prob, top-1 true rank / arm6 error.
  measure   rank / error law of the top-1 (recall@k, quantiles) for every set; KS vs Dev v2.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

import kq2cl_common as C

FOLD_SEED = 20261119
NFOLD = 5


def fold_of_task(tasks: np.ndarray) -> dict[int, int]:
    u = np.unique(tasks)
    perm = np.random.default_rng(FOLD_SEED).permutation(u)
    return {int(t): i % NFOLD for i, t in enumerate(perm)}


def subset_split(sp: dict, mask: np.ndarray) -> dict:
    n = len(sp['key'])
    out = {}
    for k, v in sp.items():
        if k in ('seq_actions', 'seq_states', 'ep_offset', 'ep_length'):
            out[k] = v
        elif isinstance(v, np.ndarray) and v.ndim and len(v) == n:
            out[k] = v[mask]
        elif k == 'vlm':
            out[k] = np.asarray(v[np.flatnonzero(mask)]) if v is not None else None
        else:
            out[k] = v
    return out


def ctx_and_gpu(device: str, vlm: bool):
    import kq2_run as kr
    ctx = kr.Ctx(device, vlm=vlm)
    gs = {n: kr.GPUSplit(ctx.splits[n], ctx.device, vlm) for n in ('eh',) + C.EVAL}
    return ctx, gs


def folds(device: str, queries: list[str]) -> None:
    import torch
    import kq2_run as kr
    save = C.OUT / 'folds'; save.mkdir(parents=True, exist_ok=True)
    need_vlm = 'iii' in queries
    ctx, gs = ctx_and_gpu(device, need_vlm)
    eh = ctx.splits['eh']
    fmap = fold_of_task(eh['task'])
    C.write_json(save / 'folds.json', {'seed': FOLD_SEED, 'task_to_fold': fmap})
    for qn in queries:
        cfg = torch.load(C.QUERY_CK[qn], map_location='cpu', weights_only=False)['cfg']
        for f in range(NFOLD):
            name = f'{qn}_fold{f}'
            if (save / f'{name}.json').exists():
                continue
            mask = np.asarray([fmap[int(t)] != f for t in eh['task']])
            sub = subset_split(eh, mask)
            g2 = dict(gs); g2['eh'] = kr.GPUSplit(sub, ctx.device, bool(cfg['xattn']))
            C.log('fold_start', name=name, train_rows=int(mask.sum()))
            r = kr.train_query(name, dict(cfg), ctx, g2, save)
            C.log('fold_done', name=name, epoch=r['selected_epoch'], dev=r['dev_selected'],
                  unseen20=r['evaluation']['unseen20']['all'])
            del g2['eh']; torch.cuda.empty_cache()


def _run_query(model, gsplit, ix, keys, torch):
    q = model(gsplit, ix)
    return q, model.score(q, keys)


def retrieve(device: str, queries: list[str]) -> None:
    import torch
    import kq2_run as kr
    save = C.OUT / 'retrieval'; save.mkdir(parents=True, exist_ok=True)
    need_vlm = 'iii' in queries
    ctx, gs = ctx_and_gpu(device, need_vlm)
    keys = torch.as_tensor(np.load(C.KEYS), device=ctx.device, dtype=torch.float32)
    fmap = C.read_json(C.OUT / 'folds/folds.json')['task_to_fold']
    fmap = {int(k): int(v) for k, v in fmap.items()}
    bt = ctx.bt

    def run(model, split_name, rows, extra_mask=None):
        sp = ctx.splits[split_name]; g = gs[split_name]
        qs, lg = [], []
        with torch.no_grad():
            for s in range(0, len(rows), 512):
                ix = torch.as_tensor(rows[s:s + 512], device=ctx.device)
                q, sc = _run_query(model, g, ix, keys, torch)
                qs.append(q.float().cpu().numpy()); lg.append(sc.float().cpu().numpy())
        q = np.concatenate(qs); logits = np.concatenate(lg)
        allowed = sp['allowed'][rows].copy()
        if extra_mask is not None:
            allowed &= extra_mask
        summ = C.summarize_scores(logits, allowed)
        err = sp['errors'][rows]
        top1 = summ['ids'][:, 0].astype(np.int64)
        rank = C.true_rank(err, top1, allowed)
        e1 = err[np.arange(len(rows)), top1]
        oracle = np.where(allowed, err, np.inf).min(1)
        return {'rows': np.asarray(rows, np.int64), 'q': q.astype(np.float32), **summ,
                'rank': rank.astype(np.int64), 'err': e1.astype(np.float32), 'oracle_err': oracle.astype(np.float32),
                'n_allowed': allowed.sum(1).astype(np.int32),
                'logit_scale': np.float32(float(model.logit_scale.exp().clamp(max=100.)))}

    for qn in queries:
        prod, _ = C.load_query(qn, ctx.device)
        eh = ctx.splits['eh']; n = len(eh['key'])
        own_task = eh['task'][:, None] != bt[None]
        out = {}
        for split in C.EVAL:
            p = save / f'{qn}__{split}.npz'
            if not p.exists():
                np.savez_compressed(p, **run(prod, split, np.arange(len(ctx.splits[split]['key']))))
                C.log('retrieve', query=qn, split=split)
        for v, mask in (('insE', None), ('insT', own_task)):
            p = save / f'{qn}__eh_{v}.npz'
            if not p.exists():
                np.savez_compressed(p, **run(prod, 'eh', np.arange(n), mask))
                C.log('retrieve', query=qn, split=f'eh_{v}')
        for v in ('oofE', 'oofT'):
            p = save / f'{qn}__eh_{v}.npz'
            if p.exists():
                continue
            parts = []
            for f in range(NFOLD):
                fm, _ = C.load_query(C.OUT / 'folds' / f'{qn}_fold{f}.pt', ctx.device)
                rows = np.flatnonzero([fmap[int(t)] == f for t in eh['task']])
                parts.append(run(fm, 'eh', rows, own_task[rows] if v == 'oofT' else None))
            order = np.argsort(np.concatenate([x['rows'] for x in parts]))
            merged = {k: (np.concatenate([x[k] for x in parts])[order] if np.ndim(parts[0][k]) else parts[0][k])
                      for k in parts[0]}
            merged['logit_scale_per_row'] = np.concatenate([np.full(len(x['rows']), float(x['logit_scale'])) for x in parts])[order]
            assert np.array_equal(merged['rows'], np.arange(n))
            np.savez_compressed(p, **merged)
            C.log('retrieve', query=qn, split=f'eh_{v}')
        torch.cuda.empty_cache()


def measure(queries: list[str]) -> dict:
    from scipy.stats import ks_2samp
    R = C.OUT / 'retrieval'
    res = {}
    for qn in queries:
        dev = np.load(R / f'{qn}__dev2.npz')
        for s in ('eh_insE', 'eh_insT', 'eh_oofE', 'eh_oofT', 'dev2', 'learned', 'unseen20'):
            z = np.load(R / f'{qn}__{s}.npz')
            r, e = z['rank'], z['err']
            rec = {'rows': int(len(r)), 'rank': C.quantiles(r), 'arm6': C.quantiles(e),
                   'recall@1': float((r == 1).mean()), 'recall@10': float((r <= 10).mean()),
                   'recall@100': float((r <= 100).mean()), 'oracle_arm6_mean': float(z['oracle_err'].mean()),
                   'top1_arm6_mean': float(e.mean())}
            if s != 'dev2':
                rec['ks_vs_dev2'] = {'log_rank': float(ks_2samp(np.log(r), np.log(dev['rank'])).statistic),
                                     'arm6': float(ks_2samp(e, dev['err']).statistic)}
            res[f'{qn}:{s}'] = rec
    C.write_json(C.OUT / 'measure.json', res)
    C.log('measure_done', **{k: {'rank_med': v['rank']['0.5'], 'arm6': round(v['top1_arm6_mean'], 4)} for k, v in res.items()})
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=('folds', 'retrieve', 'measure'))
    ap.add_argument('--queries', default='iii,i')
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    import torch
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '8')))
    qs = a.queries.split(',')
    C.log('start', stage='retrieve.' + a.cmd, queries=qs)
    if a.cmd == 'folds':
        folds(a.device, qs)
    elif a.cmd == 'retrieve':
        retrieve(a.device, qs)
    else:
        measure(qs)
    C.log('end', stage='retrieve.' + a.cmd)


if __name__ == '__main__':
    main()
