#!/usr/bin/env python3
"""EX3: Expert retrained on the draft distribution the KQ-final retriever actually produces,
with the gripper channel learned by the Expert (no π0.5 anywhere at inference).

Why: EX2's Expert was trained on near-truth drafts (action-space top-50, geometric p=.16,
10% uniform; mostly rank 1-10) while KQ-final picks the true-best unit at median rank ~78 on
unseen tasks. EX3 keeps the Expert key-agnostic (it only sees drafts) but samples the training
draft's *true action-space rank* from the rank distribution measured for the KQ-final
retrievers on LIBERO-40 memory rows with every own-task unit masked (unseen-like).

Pipeline (subcommands, each resumable / skip-if-done):
  measure   true rank + arm-6 MSE of the KQ-final top-1 (C1a attention / C1b MLP query) on
            memory rows: (a) task-fold cross-fitted query (EKQ crossfit, own task masked)
            (b) production query, own task masked. Dev (production query) for comparison.
  prepare   training rows (= EX2 Expert A rows, LIBERO-40 only), per-row action-MSE ordering of
            all eligible units (own task masked), hold-last gripper, sampler preview.
  train     EX2-A model/filter/optimizer; fresh rank-sampled draft per step; loss on arm6+gripper.
  select    corrected Dev scoring (live NNX state) of every checkpoint; C1a+C1b Dev drafts.
  devgrid   per-row Dev Expert curves over a t0 grid (EX2 and EX3) -> t0 maps.
  offline   learned 621 / unseen 619 main table.
  closed    LIBERO-90 Eval closed loop (serve_expert_ex3.py), summarize -> McNemar.
  report    report.md / results.json.

Metrics: arm6 = sum over 10x6 squared normalized error / 70 (project standard); active7 adds the
gripper channel (sum over 10 squared / 70).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[5]
STATE_DIR = Path(os.environ.get('SKILLNET_STATE_DIR', REPO_ROOT / '.skillnet-state')).expanduser()
sys.path[:0] = [str(HERE), str(HERE.parent / 'pi05_memory')]
ROOT = (STATE_DIR / 'outputs/memory-pilot/unseen-libero')
OUT = ROOT / 'expert-ex3-01'
KQF = ROOT / 'key-motion-01/final'
EX2 = ROOT / 'expert-ex2-01/production-05'
EX2_CK = ROOT / 'closed-loop-kq-01/expert-select/train-A/checkpoints/002500'
EKQ = ROOT / 'expert-kq-01'
CL1 = ROOT / 'closed-loop-kq-01'
RUN40 = (STATE_DIR / 'outputs/memory-pilot/pi05-memory40-20261104/run-01')
PI05 = (STATE_DIR / 'checkpoints/openpi-assets/checkpoints/pi05_libero')
QUERIES = {'C1a': KQF / 'best/Afix_x1_fixed_e_h_cteff.pt', 'C1b': KQF / 'best/H_x1_mk1b_cteff_tau0.01_s1.pt'}
EXPORT = {'C1a': 'Afix_x1_fixed_e_h_cteff', 'C1b': 'H_x1_mk1b_cteff_tau0.01_s1'}
PARAM_REGEX = r'.*(llm.*_1|action_in_proj|time_mlp|action_out_proj).*'
T0_GRID = (.02, .05, .1, .15, .2, .25, .3, .35, .4, .5, .6, .7, .8, .9)
SCALES = (.5, 1., 1.5, 2., 3.)
DEFAULT_GRIP = -1.0   # LIBERO normalized gripper at episode start (open)


# ---------------------------------------------------------------------------
# pure helpers (tested in tests/unseen_libero/test_expert_ex3.py)
# ---------------------------------------------------------------------------

def arm6(pred, target) -> np.ndarray:
    p = np.asarray(pred, np.float64); t = np.asarray(target, np.float64)
    return np.square(p[..., :6] - t[..., :6]).sum((-1, -2)) / 70.


def grip7(pred, target) -> np.ndarray:
    p = np.asarray(pred, np.float64); t = np.asarray(target, np.float64)
    return np.square(p[..., 6] - t[..., 6]).sum(-1) / 70.


def candidate_mse(targets, units) -> np.ndarray:
    """[N,M] arm6 MSE (sum/70) between each target and every stored unit window."""
    y = np.asarray(targets, np.float64)[..., :6].reshape(len(targets), -1)
    c = np.asarray(units, np.float64)[..., :6].reshape(len(units), -1)
    d = np.square(y).sum(1, keepdims=True) + np.square(c).sum(1)[None] - 2 * y @ c.T
    return np.maximum(d, 0) / 70.


def true_rank(mse_all: np.ndarray, picked: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    """1-based rank of the picked unit among eligible units by action MSE (ties count as better)."""
    m = np.asarray(mse_all, np.float64); e = np.asarray(eligible, bool); p = np.asarray(picked, np.int64)
    if m.shape != e.shape or p.shape != (len(m),):
        raise ValueError(f'bad shapes {m.shape} {e.shape} {p.shape}')
    if not e[np.arange(len(p)), p].all():
        raise ValueError('picked unit is not eligible')
    pm = m[np.arange(len(p)), p]
    return 1 + ((m < pm[:, None]) & e).sum(1)


def hold_last_gripper(last: np.ndarray, valid: np.ndarray, default: float = DEFAULT_GRIP) -> np.ndarray:
    """[N,10] draft gripper = sign of the last executed gripper (default open when no history)."""
    v = np.asarray(last, np.float64); ok = np.asarray(valid, bool)
    s = np.where(ok & (v != 0), np.sign(v), default)
    return np.repeat(s[:, None], 10, 1).astype(np.float32)


def sample_ranks(rng: np.random.Generator, n: int, measured: np.ndarray, n_eligible: np.ndarray, *,
                 near_fraction: float = .15, near_p: float = .16, near_k: int = 50,
                 rank_scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Draw 1-based draft ranks: (1-near_fraction) from the measured retriever rank law (bootstrap
    of ``measured``), near_fraction from a truncated geometric over the top ``near_k`` (keeps good
    drafts intact). ``rank_scale`` multiplies the measured-law ranks (ceil) before clipping; it is
    the one Dev-calibrated knob (memory rows are easier than unseen ones). Ranks are clipped to the
    row's eligible-unit count."""
    r = np.asarray(measured, np.int64); ne = np.asarray(n_eligible, np.int64)
    if r.ndim != 1 or len(r) == 0 or np.any(r < 1) or ne.shape != (n,) or np.any(ne < 1):
        raise ValueError('bad rank sampler inputs')
    near = rng.random(n) < near_fraction
    probs = near_p * (1 - near_p) ** np.arange(near_k); probs /= probs.sum()
    out = r[rng.integers(0, len(r), size=n)]
    if rank_scale != 1.0:
        out = np.ceil(out * float(rank_scale)).astype(np.int64)
    out[near] = 1 + rng.choice(near_k, size=int(near.sum()), p=probs)
    return np.minimum(out, ne), near


def select_t0(error, c: float, lo: float = .05, hi: float = .6) -> np.ndarray:
    e = np.asarray(error, np.float64)
    if np.any(e < 0) or not np.isfinite(e).all():
        raise ValueError('bad error')
    return np.clip(c * np.sqrt(e), lo, hi).astype(np.float32)


def interp_curve(curve, grid, t0) -> np.ndarray:
    g = np.asarray(grid, np.float64); c = np.asarray(curve, np.float64); t = np.clip(np.asarray(t0, np.float64), g[0], g[-1])
    j = np.clip(np.searchsorted(g, t, side='right') - 1, 0, len(g) - 2)
    a = (t - g[j]) / (g[j + 1] - g[j]); r = np.arange(len(t))
    return (1 - a) * c[r, j] + a * c[r, j + 1]


def switch_metrics(pred_grip, target_grip, last, valid) -> dict[str, Any]:
    """Gripper switch timing vs the last executed gripper sign (rows without history skipped)."""
    p = np.sign(np.asarray(pred_grip, np.float64)); t = np.sign(np.asarray(target_grip, np.float64))
    ok = np.asarray(valid, bool) & (np.asarray(last) != 0); g0 = np.sign(np.asarray(last, np.float64))

    def first(x):
        d = x != g0[:, None]
        return np.where(d.any(1), d.argmax(1), -1)
    kp, kt = first(p), first(t)
    gt = ok & (kt >= 0); pr = ok & (kp >= 0); both = gt & pr
    err = np.abs(kp[both] - kt[both]); nsw = int(gt.sum()); nno = int((ok & ~gt).sum())
    return {'rows_with_history': int(ok.sum()), 'gt_switch_rows': nsw,
            'missed_switch_rate': float((gt & ~pr).sum() / max(nsw, 1)) if nsw else None,
            'false_switch_rate': float((ok & ~gt & pr).sum() / max(nno, 1)),
            'timing_error_mean_steps': float(err.mean()) if len(err) else None,
            'within1_step_rate': float((err <= 1).sum() / max(nsw, 1)) if nsw else None,
            'sign_step_error_rate': float((p[ok] != t[ok]).mean()) if ok.any() else None}


def bootstrap_ci(values, groups, *, seed=4101, reps=5000) -> list[float]:
    x = np.asarray(values, np.float64); _, inv = np.unique(np.asarray(groups), return_inverse=True)
    s = np.bincount(inv, weights=x); c = np.bincount(inv)
    d = np.random.default_rng(seed).integers(len(s), size=(reps, len(s)))
    m = s[d].sum(1) / c[d].sum(1)
    return [float(np.quantile(m, .025)), float(np.quantile(m, .975))]


def mcnemar_exact(a, b) -> dict[str, Any]:
    a = np.asarray(a, bool); b = np.asarray(b, bool)
    w = int((b & ~a).sum()); l = int((a & ~b).sum()); n = w + l
    if n == 0:
        return {'b_only': w, 'a_only': l, 'p': 1.0}
    k = min(w, l)
    return {'b_only': w, 'a_only': l, 'p': min(1., 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)}


def quantiles(x, qs=(.1, .25, .5, .75, .9)) -> dict[str, float]:
    x = np.asarray(x, np.float64)
    return {str(q): float(np.quantile(x, q)) for q in qs} | {'mean': float(x.mean())}


# ---------------------------------------------------------------------------
# io
# ---------------------------------------------------------------------------

def utc() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def log(event: str, **kw) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rec = {'utc': utc(), 'event': event, **kw}
    with (OUT / 'progress.log').open('a') as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=float) + '\n')
    print(json.dumps(rec, ensure_ascii=False, default=float), flush=True)


def write_json(path: Path, value: Any) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=float) + '\n'); tmp.replace(path)


def kqf(name: str) -> np.ndarray:
    return np.load(KQF / 'data' / f'{name}.npy')


def load_split(name: str) -> dict[str, np.ndarray]:
    s = {k: kqf(f'{name}__{k}') for k in ('key', 'history', 'history_mask', 'state', 'actions', 'task', 'episode')}
    s['history'] = s['history'].astype(np.float32); s['history_mask'] = s['history_mask'].astype(bool)
    s['actions'] = s['actions'].astype(np.float32); s['state'] = s['state'].astype(np.float32)
    hp = KQF / 'data' / f'{name}__hard.npy'
    s['hard'] = np.load(hp).astype(bool) if hp.exists() else np.zeros(len(s['key']), bool)
    # history is right-aligned: last executed step is index 9 when any history exists
    s['last_grip'] = s['history'][:, 9, 6]; s['last_valid'] = s['history_mask'].any(1)
    return s


def bank() -> dict[str, np.ndarray]:
    return {'actions': kqf('bank_actions').astype(np.float32), 'task': kqf('bank_task').astype(np.int64),
            'episode': kqf('bank_episode').astype(np.int64)}


def export_top1(qname: str, split: str) -> np.ndarray:
    with np.load(KQF / 'export' / f'{EXPORT[qname]}__{split}_scores.npz') as z:
        if not np.array_equal(z['key'], kqf(f'{split}__key')):
            raise AssertionError(f'{qname}/{split} export order mismatch')
        return np.asarray(z['top1_unit'], np.int64)


# ---------------------------------------------------------------------------
# 1. measure the retriever's rank / error law (memory side, own task masked)
# ---------------------------------------------------------------------------

def measure() -> dict[str, Any]:
    dest = OUT / 'measure.json'
    import torch
    import key_final as kf
    torch.set_num_threads(8)
    b = bank(); eh = load_split('eh'); dev = load_split('dev')
    eligible_eh = eh['task'][:, None] != b['task'][None]
    mse_eh = np.concatenate([candidate_mse(eh['actions'][s:s + 1024], b['actions']) for s in range(0, len(eh['key']), 1024)])
    mse_dev = candidate_mse(dev['actions'], b['actions'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    keys = torch.as_tensor(kqf('x1_keys'), device=device, dtype=torch.float32)
    out: dict[str, Any] = {'rows': {'eh': int(len(eh['key'])), 'dev': int(len(dev['key']))},
                           'candidate_set_memory': 'all 4,516 LIBERO-40 units minus every unit of the row task (own episode included)',
                           'candidate_set_dev': 'all 4,516 units (Dev tasks are not in memory)'}
    arrays: dict[str, np.ndarray] = {}
    for qname, path in QUERIES.items():
        blob = torch.load(path, map_location='cpu', weights_only=False)
        model = kf.make_query(blob['cfg']).to(device); model.load_state_dict(blob['query']); model.eval()
        top = []
        for s in range(0, len(eh['key']), 256):
            ix = np.arange(s, min(s + 256, len(eh['key'])))
            with torch.no_grad():
                sc = kf.score(model(**kf.pack(eh, ix, blob['cfg'], device)), keys, 'dot', model.logit_scale).double().cpu().numpy()
            sc[~eligible_eh[ix]] = -np.inf
            top.append(sc.argmax(1))
        variants = {'ins': np.concatenate(top)}
        ret = EKQ / 'retrieval' / f'{qname}.npz'
        if ret.exists():
            with np.load(ret) as z:
                variants['oofT'] = np.asarray(z['eh_oofT__ids'][:, 0], np.int64)
        dev_top = export_top1(qname, 'dev')
        for v, picked in variants.items():
            r = true_rank(mse_eh, picked, eligible_eh); e = mse_eh[np.arange(len(picked)), picked]
            arrays[f'{qname}__{v}__rank'] = r; arrays[f'{qname}__{v}__mse'] = e
            out[f'{qname}:{v}'] = {'rank': quantiles(r), 'arm6': quantiles(e), 'recall@1': float((r == 1).mean()),
                                   'recall@10': float((r <= 10).mean()), 'recall@100': float((r <= 100).mean())}
        r = true_rank(mse_dev, dev_top, np.ones_like(mse_dev, bool)); e = mse_dev[np.arange(len(dev_top)), dev_top]
        arrays[f'{qname}__dev__rank'] = r; arrays[f'{qname}__dev__mse'] = e
        out[f'{qname}:dev'] = {'rank': quantiles(r), 'arm6': quantiles(e), 'recall@1': float((r == 1).mean()),
                               'recall@10': float((r <= 10).mean()), 'recall@100': float((r <= 100).mean())}
    # Old EX2 training law on the same rows, for reference (geometric p=.16 top-50 + 10% uniform)
    rng = np.random.default_rng(0); n = len(eh['key'])
    probs = .16 * .84 ** np.arange(50); probs /= probs.sum()
    old = 1 + rng.choice(50, size=n, p=probs); uni = rng.random(n) < .1
    old[uni] = 1 + rng.integers(0, eligible_eh.sum(1)[uni])
    srt = np.sort(np.where(eligible_eh, mse_eh, np.inf), 1)
    arrays['ex2_law__rank'] = old; arrays['ex2_law__mse'] = srt[np.arange(n), old - 1]
    out['ex2_training_law'] = {'rank': quantiles(old), 'arm6': quantiles(arrays['ex2_law__mse'])}
    out['oracle'] = {'memory_own_task_masked_arm6': quantiles(srt[:, 0]), 'dev_arm6': quantiles(mse_dev.min(1))}
    # Similarity of each memory law to the Dev law (two-sample KS on log rank and on arm6)
    from scipy.stats import ks_2samp
    for qname in QUERIES:
        for v in ('ins', 'oofT'):
            if f'{qname}__{v}__rank' not in arrays:
                continue
            out[f'{qname}:{v}']['ks_vs_dev'] = {
                'log_rank': float(ks_2samp(np.log(arrays[f'{qname}__{v}__rank']), np.log(arrays[f'{qname}__dev__rank'])).statistic),
                'arm6': float(ks_2samp(arrays[f'{qname}__{v}__mse'], arrays[f'{qname}__dev__mse']).statistic)}
    out['ex2_training_law']['ks_vs_dev'] = {
        'log_rank': float(ks_2samp(np.log(old), np.log(np.r_[arrays['C1a__dev__rank'], arrays['C1b__dev__rank']])).statistic),
        'arm6': float(ks_2samp(arrays['ex2_law__mse'], np.r_[arrays['C1a__dev__mse'], arrays['C1b__dev__mse']]).statistic)}
    np.savez_compressed(OUT / 'measure.npz', **arrays)
    write_json(dest, out)
    log('measure_done', **{k: v['rank']['0.5'] for k, v in out.items() if isinstance(v, dict) and 'rank' in v})
    return out


# ---------------------------------------------------------------------------
# 2. training rows + sampler
# ---------------------------------------------------------------------------

def prepare(variant: str = 'oofT', near_fraction: float = .15) -> dict[str, Any]:
    dest = OUT / 'prepare'
    if (dest / 'prepare.json').exists():
        log('prepare_skip'); return json.loads((dest / 'prepare.json').read_text())
    dest.mkdir(parents=True, exist_ok=True)
    rows = {k: np.load(RUN40 / 'rows' / f'{k}.npy', mmap_mode='r') for k in ('task', 'episode', 'frame', 'actions', 'memory')}
    with np.load(EX2 / 'rank-samples/train_a.npz') as z:
        ids = np.asarray(z['row_ids'], np.int64)
    if rows['memory'][ids].any():
        raise AssertionError('memory rows in training set')
    t = np.asarray(rows['task']); e = np.asarray(rows['episode']); f = np.asarray(rows['frame'])
    index = {(int(a), int(b_), int(c)): i for i, (a, b_, c) in enumerate(zip(t, e, f))}
    prev = np.asarray([index.get((int(t[i]), int(e[i]), int(f[i]) - 10), -1) for i in ids], np.int64)
    if np.any((prev < 0) & (f[ids] > 0)):
        raise AssertionError('missing previous stride-10 row')
    acts = np.asarray(rows['actions'][ids][:, :, :7], np.float32)
    last = np.where(prev >= 0, np.asarray(rows['actions'])[np.maximum(prev, 0), 9, 6], 0.).astype(np.float32)
    b = bank()
    order = np.empty((len(ids), len(b['task'])), np.int16); n_el = np.empty(len(ids), np.int32)
    for s in range(0, len(ids), 1024):
        m = candidate_mse(acts[s:s + 1024], b['actions'])
        m[t[ids[s:s + 1024], None] == b['task'][None]] = np.inf
        order[s:s + 1024] = np.argsort(m, 1, kind='stable').astype(np.int16)
        n_el[s:s + 1024] = np.isfinite(m).sum(1)
    meas = np.load(OUT / 'measure.npz')
    measured = np.concatenate([meas[f'{q}__{variant}__rank'] for q in QUERIES]).astype(np.int64)
    np.save(dest / 'order.npy', order)
    np.savez_compressed(dest / 'rows.npz', row_ids=ids, target=acts, last_grip=last, last_valid=prev >= 0,
                        n_eligible=n_el, task=t[ids], measured_rank=measured)
    # Preview: one draw per row from the sampler; compare its arm6 law with memory / Dev laws.
    rng = np.random.default_rng(1)
    r, near = sample_ranks(rng, len(ids), measured, n_el, near_fraction=near_fraction)
    picked = order[np.arange(len(ids)), r - 1].astype(np.int64)
    pre = arm6(b['actions'][picked], acts)
    hl = hold_last_gripper(last, prev >= 0)
    sw = (np.sign(acts[..., 6]) != hl).any(1)
    from scipy.stats import ks_2samp
    dev_m = np.r_[meas['C1a__dev__mse'], meas['C1b__dev__mse']]
    res = {'rows': int(len(ids)), 'tasks': int(len(np.unique(t[ids]))), 'variant_for_rank_law': variant,
           'near_fraction': near_fraction, 'measured_rank_pool': int(len(measured)),
           'n_eligible': quantiles(n_el), 'preview_rank': quantiles(r), 'preview_arm6': quantiles(pre),
           'preview_t0_train_scale2': quantiles(select_t0(pre, 2.)),
           'preview_t0_at_cap_.6': float((select_t0(pre, 2.) >= .6).mean()),
           'preview_ks_arm6_vs_dev': float(ks_2samp(pre, dev_m).statistic),
           'preview_ks_arm6_vs_memory_law': float(ks_2samp(pre, np.concatenate([meas[f'{q}__{variant}__mse'] for q in QUERIES])).statistic),
           'gripper_switch_within_chunk_rows': float(sw.mean()), 'rows_without_history': float((prev < 0).mean()),
           'draft_gripper': 'hold-last executed gripper sign (previous stride-10 row action[9,6]); -1 (open) at frame 0'}
    write_json(dest / 'prepare.json', res); log('prepare_done', **{k: v for k, v in res.items() if not isinstance(v, dict)})
    return res


# ---------------------------------------------------------------------------
# 3. training
# ---------------------------------------------------------------------------

def _filter():
    import flax.nnx as nnx
    from openpi.shared.nnx_utils import PathRegex
    return nnx.All(nnx.Param, PathRegex(PARAM_REGEX))


def _base():
    import skill_expert_refine as exp3
    return exp3, exp3._module(HERE.parent / 'pi05_memory/run.py', f'ex3_base_{os.getpid()}')


def train(name: str, *, updates: int = 5000, batch: int = 8, microbatch: int = 8, save_every: int = 500,
          grip_weight: float = 1.0, train_scale: float = 2.0, near_fraction: float = .15, rank_scale: float = 1.0,
          seed: int = 20261118) -> dict[str, Any]:
    import jax, jax.numpy as jnp, flax.nnx as nnx, optax, orbax.checkpoint as ocp
    exp3, base = _base()
    tdir = OUT / f'train-{name}'; ck = tdir / 'checkpoints'; ck.mkdir(parents=True, exist_ok=True)
    prep = dict(np.load(OUT / 'prepare/rows.npz')); order = np.load(OUT / 'prepare/order.npy', mmap_mode='r')
    units = bank()['actions']
    rows40 = base.load_rows(RUN40)
    ids = prep['row_ids']; target = prep['target']; n_el = prep['n_eligible']; measured = prep['measured_rank']
    hl = hold_last_gripper(prep['last_grip'], prep['last_valid'])
    model, _ = base.load_policy(PI05)
    filt = _filter()
    optimizer = optax.chain(optax.clip_by_global_norm(1.), optax.adamw(2.5e-5, b1=.9, b2=.95, weight_decay=1e-5))
    state = nnx.state(model, filt); opt_state = optimizer.init(state)
    meta_path = tdir / 'train.json'
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {
        'status': 'running', 'name': name, 'updates': updates, 'batch': batch, 'rows': int(len(ids)),
        'sample_rule': f'{1 - near_fraction:.0%} true rank bootstrapped from the measured KQ-final law (memory rows, own task masked); '
                       f'{near_fraction:.0%} truncated geometric p=.16 over top-50; rank over own-task-masked units by arm6 action MSE',
        'draft_gripper': 'hold-last executed gripper', 'loss': f'mean over arm6 velocity sq + {grip_weight} x gripper velocity sq, / (6+{grip_weight}) per element',
        'grip_weight': grip_weight, 'rank_scale': rank_scale, 'near_fraction': near_fraction, 't0_rule_train': f'clip({train_scale}*sqrt(pre_arm6)*U(.9,1.1),.05,.6)',
        'optimizer': 'AdamW 2.5e-5 b(.9,.95) wd1e-5 clip1 (EX2 A)', 'param_filter': PARAM_REGEX, 'seed': seed, 'history': []}
    start = 0
    done = sorted(int(p.name) for p in ck.iterdir() if p.name.isdigit() and (ck / f'{p.name}.opt').exists())
    if done:
        start = done[-1]
        with ocp.PyTreeCheckpointer() as c:
            restored = c.restore(ck / f'{start:06d}')
            st = nnx.state(model, filt); st.replace_by_pure_dict(jax.tree.map(jnp.asarray, restored)); nnx.update(model, st)
            opt_state = c.restore(ck / f'{start:06d}.opt', item=opt_state)
        log('train_resume', name=name, step=start)
    wv = jnp.asarray([1.] * 6 + [float(grip_weight)], jnp.float32)

    @nnx.jit
    def grad_step(current, obs, action, draft, t0, u, eps):
        source = (1. - t0[:, None, None]) * draft + t0[:, None, None] * eps
        x = (1. - u[:, None, None]) * action + u[:, None, None] * source
        target_v = (source - action) / jnp.maximum(t0[:, None, None], 1e-3)

        def loss_fn(m):
            pred = m.velocity(obs, x, t0 * u)
            sq = jnp.square(pred[:, :, :7] - target_v[:, :, :7])
            return jnp.mean(jnp.sum(sq * wv, -1)) / jnp.sum(wv)
        return nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, filt))(current)

    np_rng = np.random.default_rng(seed + start)
    rng = jax.random.fold_in(jax.random.key(seed), start)
    losses = []; t_start = time.time()
    for step in range(start + 1, updates + 1):
        bi = np_rng.integers(0, len(ids), size=batch)
        r, _ = sample_ranks(np_rng, batch, measured, n_el[bi], near_fraction=near_fraction, rank_scale=rank_scale)
        picked = np.asarray(order[bi, r - 1], np.int64)
        d = np.zeros((batch, 10, 32), np.float32); d[..., :6] = units[picked, :, :6]; d[..., 6] = hl[bi]
        a = np.zeros((batch, 10, 32), np.float32); a[..., :7] = target[bi]
        pre = arm6(d, a)
        t0 = jnp.asarray(select_t0(pre * np_rng.uniform(.9, 1.1, size=batch) ** 2, train_scale))
        rng, uk, ek = jax.random.split(rng, 3)
        u = jax.random.uniform(uk, (batch,), minval=.01, maxval=1.); eps = jax.random.normal(ek, (batch, 10, 32))
        grads = None; lv = 0.
        for lo in range(0, batch, microbatch):
            mb = np.arange(lo, min(lo + microbatch, batch))
            obs = base.obs_batch(rows40, ids[bi[mb]])
            val, g = grad_step(model, obs, jnp.asarray(a[mb]), jnp.asarray(d[mb]), t0[mb], u[mb], eps[mb])
            w = len(mb) / batch; lv += float(val) * w
            g = jax.tree.map(lambda x: x * w, g); grads = g if grads is None else jax.tree.map(jnp.add, grads, g)
        st = nnx.state(model, filt)
        upd, opt_state = optimizer.update(grads, opt_state, st); nnx.update(model, optax.apply_updates(st, upd))
        losses.append(lv)
        if step % 100 == 0:
            log('train_step', name=name, step=step, loss=float(np.mean(losses[-100:])), sec=round(time.time() - t_start))
        if step % save_every == 0 or step == updates:
            exp3._save_state(model, filt, ck / f'{step:06d}', ocp)
            with ocp.PyTreeCheckpointer() as c:
                c.save(ck / f'{step:06d}.opt', opt_state, force=True)
            meta['history'].append({'step': step, 'loss_last500': float(np.mean(losses[-save_every:])), 'checkpoint': str(ck / f'{step:06d}')})
            write_json(meta_path, meta)
    meta['status'] = 'complete'; write_json(meta_path, meta)
    log('train_done', name=name)
    return meta


# ---------------------------------------------------------------------------
# Expert evaluation helpers (live NNX state; fixes EX2's frozen module_jit scoring)
# ---------------------------------------------------------------------------

class LiveExpert:
    def __init__(self):
        import jax, jax.numpy as jnp, flax.nnx as nnx, orbax.checkpoint as ocp
        self.jax, self.jnp, self.nnx, self.ocp = jax, jnp, nnx, ocp
        self.exp3, self.base = _base()
        self.model, _ = self.base.load_policy(PI05)
        self.filt = _filter(); self.loaded = None

        @nnx.jit
        def run(current, ob, draft, starts, eps):
            return current.sample_bridge(jax.random.key(0), ob, draft, starts, num_steps=10, noise=eps)
        self._run = run

    def load(self, checkpoint: Path) -> None:
        checkpoint = Path(checkpoint)
        if self.loaded == checkpoint:
            return
        with self.ocp.PyTreeCheckpointer() as c:
            restored = c.restore(checkpoint)
        st = self.nnx.state(self.model, self.filt)
        st.replace_by_pure_dict(self.jax.tree.map(self.jnp.asarray, restored)); self.nnx.update(self.model, st)
        self.loaded = checkpoint

    def bridge(self, obs, draft7, noise, t0, batch: int = 16) -> np.ndarray:
        d = np.zeros((len(draft7), 10, 32), np.float32); d[..., :7] = draft7
        t = np.asarray(t0, np.float32).reshape(-1)
        out = []
        for s in range(0, len(d), batch):
            sl = slice(s, min(s + batch, len(d)))
            ob = self.jax.tree.map(lambda x: x[sl], obs)
            out.append(np.asarray(self._run(self.model, ob, self.jnp.asarray(d[sl]), self.jnp.asarray(t[sl]), self.jnp.asarray(noise[sl])), np.float32))
        return np.concatenate(out)[..., :7]


def _dev_inputs(ex: LiveExpert):
    dev = load_split('dev')
    with np.load(EX2 / 'retrieval/dev_exact_observations.npz') as z:
        rows = {k: z[k] for k in z.files}
    if not np.array_equal(rows['row_keys'], dev['key']):
        raise AssertionError('Dev observation mismatch')
    ids = np.arange(len(dev['key']), dtype=np.int64)
    return dev, ex.base.obs_batch(rows, ids), ex.exp3._seed_noise(ids + 60_000_000)


def _drafts(split: str, sp: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    units = bank()['actions']; hl = hold_last_gripper(sp['last_grip'], sp['last_valid'])
    out = {}
    for q in QUERIES:
        d = units[export_top1(q, split)].copy(); d[..., 6] = hl; out[q] = d
    return out


def select() -> dict[str, Any]:
    """Score every EX3 checkpoint (and EX2 step 2500) on Dev: C1a+C1b top-1 drafts, t0 from the true
    draft error with c in SCALES; select (checkpoint, c) by Dev arm6 after Expert. Eval unused."""
    dest = OUT / 'select.json'
    prev = json.loads(dest.read_text()) if dest.exists() else {'scores': {}}
    ex = LiveExpert(); dev, obs, noise = _dev_inputs(ex)
    drafts = _drafts('dev', dev); tgt = dev['actions']
    cks = {'EX2:2500': EX2_CK}
    for tdir in sorted(OUT.glob('train-*')):
        for p in sorted((tdir / 'checkpoints').iterdir()):
            if p.name.isdigit():
                cks[f'{tdir.name[6:]}:{int(p.name)}'] = p
    for label, path in cks.items():
        if label in prev['scores']:
            continue
        ex.load(path); rec = {'checkpoint': str(path), 'dev_arm6_by_scale': {}, 'per_query': {}}
        for c in SCALES:
            vals = []
            for q, d in drafts.items():
                t0 = select_t0(arm6(d, tgt), c); o = ex.bridge(obs, d, noise, t0); v = arm6(o, tgt); vals.append(v)
                rec['per_query'].setdefault(q, {})[str(c)] = float(v.mean())
                if c == 1. and not label.startswith('EX2'):
                    rec.setdefault('gripper_c1', {})[q] = switch_metrics(o[..., 6], tgt[..., 6], dev['last_grip'], dev['last_valid']) | \
                        {'grip_mse_sign': float(grip7(np.sign(o), tgt).mean()), 'grip_mse_raw': float(grip7(o, tgt).mean())}
            rec['dev_arm6_by_scale'][str(c)] = float(np.concatenate(vals).mean())
        prev['scores'][label] = rec; write_json(dest, prev)
        log('select_scored', label=label, best=min(rec['dev_arm6_by_scale'].values()))
    ex3 = {k: v for k, v in prev['scores'].items() if not k.startswith('EX2')}
    best = min(ex3, key=lambda k: min(ex3[k]['dev_arm6_by_scale'].values()))
    sc = ex3[best]['dev_arm6_by_scale']; c = min(sc, key=sc.get)
    e2 = prev['scores']['EX2:2500']['dev_arm6_by_scale']; c2 = min(e2, key=e2.get)
    prev['selected'] = {'label': best, 'checkpoint': ex3[best]['checkpoint'], 'scale': float(c), 'dev_arm6': sc[c]}
    prev['best_by_run'] = {}
    for run in sorted({k.split(':')[0] for k in ex3}):
        kk = min([k for k in ex3 if k.split(':')[0] == run], key=lambda k: min(ex3[k]['dev_arm6_by_scale'].values()))
        s2 = ex3[kk]['dev_arm6_by_scale']; c3 = min(s2, key=s2.get)
        prev['best_by_run'][run] = {'label': kk, 'checkpoint': ex3[kk]['checkpoint'], 'scale': float(c3), 'dev_arm6': s2[c3]}
    prev['ex2_reference'] = {'checkpoint': str(EX2_CK), 'scale': float(c2), 'dev_arm6': e2[c2]}
    prev['draft_dev_arm6'] = {q: float(arm6(d, tgt).mean()) for q, d in drafts.items()}
    prev['rule'] = 'Dev 217 rows x {C1a,C1b} production top-1 drafts, hold-last gripper; t0=clip(c*sqrt(true arm6 err),.05,.6); live NNX state'
    write_json(dest, prev); log('select_done', selected=prev['selected'], ex2=prev['ex2_reference'])
    return prev


# ---------------------------------------------------------------------------
# 4. Dev t0 grid -> confidence maps (EKQ π0.5-free ê, or retrieval-only fallback)
# ---------------------------------------------------------------------------

FROZEN = OUT / 'ekq-frozen'


def _ekq_ready() -> bool:
    """EKQ π0.5-free confidence heads + retrieval summaries, frozen into OUT on first use so a later
    EKQ refit cannot change EX3 numbers mid-way."""
    import shutil
    if all((FROZEN / f'{q}_confidence.pt').exists() and (FROZEN / f'{q}.npz').exists() for q in QUERIES):
        return True
    if not all((EKQ / 'heads' / f'{q}_confidence.pt').exists() and (EKQ / 'retrieval' / f'{q}.npz').exists() for q in QUERIES):
        return False
    FROZEN.mkdir(parents=True, exist_ok=True)
    for q in QUERIES:
        shutil.copy2(EKQ / 'heads' / f'{q}_confidence.pt', FROZEN / f'{q}_confidence.pt')
        shutil.copy2(EKQ / 'retrieval' / f'{q}.npz', FROZEN / f'{q}.npz')
    log('ekq_frozen', files=sorted(p.name for p in FROZEN.iterdir()))
    return True


def ehat(qname: str, split: str, sp: dict[str, np.ndarray]) -> np.ndarray:
    """EKQ π0.5-free predicted arm6 error of the top-1 draft (expert_kq_runtime conf head)."""
    import expert_kq_runtime as ekq
    ret = dict(np.load(FROZEN / f'{qname}.npz'))
    if not np.array_equal(ret[f'{split}__ids'][:, 0], export_top1(qname, split)):
        raise AssertionError(f'EKQ retrieval top-1 differs from KQ-final export ({qname}/{split})')
    import torch
    c = torch.load(FROZEN / f'{qname}_confidence.pt', map_location='cpu', weights_only=False)
    m = ekq.new_conf_head(c['kind']); m.load_state_dict(c['state_dict']); m.eval()
    f = ekq.confidence_features(ret[f'{split}__ids'], ret[f'{split}__vals'], ret[f'{split}__entropy'], ret[f'{split}__top1_prob'],
                                bank()['actions'], sp['history'], sp['history_mask'], sp['state'], ret[f'{split}__qnorm'])
    return ekq.predict_conf(m, f)


def devgrid() -> dict[str, Any]:
    sel = json.loads((OUT / 'select.json').read_text())
    dest = OUT / 'devgrid.npz'
    ex = LiveExpert(); dev, obs, noise = _dev_inputs(ex); drafts = _drafts('dev', dev); tgt = dev['actions']
    res = dict(np.load(dest)) if dest.exists() else {}
    for tag, ck in (('EX3', sel['selected']['checkpoint']), ('EX2', str(EX2_CK))):
        if f'{tag}__C1b__curve' in res:
            continue
        ex.load(Path(ck))
        for q, d in drafts.items():
            cur = np.zeros((len(d), len(T0_GRID)))
            for j, t in enumerate(T0_GRID):
                cur[:, j] = arm6(ex.bridge(obs, d, noise, np.full(len(d), t, np.float32)), tgt)
            res[f'{tag}__{q}__curve'] = cur
            log('devgrid', tag=tag, q=q, mean_by_t0=[round(float(x), 4) for x in cur.mean(0)])
        np.savez_compressed(dest, **res)
    for q, d in drafts.items():
        res[f'{q}__true'] = arm6(d, tgt)
        if _ekq_ready():
            res[f'{q}__ehat'] = ehat(q, 'dev', dev)
    np.savez_compressed(dest, **res)
    # maps: t0 = clip(c*sqrt(ê), lo, hi); no π0.5 fallback
    out = {'ekq_confidence_available': _ekq_ready(), 'maps': {}}
    for tag in ('EX3', 'EX2'):
        rows = []
        for c in (0., .25, .5, .75, 1., 1.25, 1.5, 2., 2.5, 3., 4.):
            for lo in (.02, .05, .1, .15, .2, .3):
                for hi in (.4, .6, .8, .9):
                    if hi <= lo or (c == 0 and hi != .4):
                        continue
                    rec = {'c': c, 'lo': lo, 'hi': hi}
                    for src in ('true', 'ehat'):
                        if f'C1a__{src}' not in res:
                            continue
                        v = np.concatenate([interp_curve(res[f'{tag}__{q}__curve'], T0_GRID, select_t0(res[f'{q}__{src}'], c, lo, hi)) for q in QUERIES])
                        rec[src] = float(v.mean())
                    rows.append(rec)
        best = {src: min([r for r in rows if src in r], key=lambda r: (round(r[src], 5), r['c'])) for src in ('true', 'ehat') if any(src in r for r in rows)}
        cur = np.concatenate([res[f'{tag}__{q}__curve'] for q in QUERIES])
        out['maps'][tag] = {'best': best, 'per_row_oracle_t0': float(cur.min(1).mean()),
                            'constant_t0': {str(t): float(x) for t, x in zip(T0_GRID, cur.mean(0))},
                            'draft': float(np.concatenate([res[f'{q}__true'] for q in QUERIES]).mean())}
    if _ekq_ready():
        from scipy.stats import spearmanr
        out['ehat_dev_spearman'] = {q: float(spearmanr(res[f'{q}__ehat'], res[f'{q}__true']).correlation) for q in QUERIES}
    write_json(OUT / 'devgrid.json', out); log('devgrid_done', maps={k: v['best'] for k, v in out['maps'].items()})
    return out


# ---------------------------------------------------------------------------
# 5. offline Eval
# ---------------------------------------------------------------------------

def offline() -> dict[str, Any]:
    sel = json.loads((OUT / 'select.json').read_text()); grid = json.loads((OUT / 'devgrid.json').read_text())
    ex = LiveExpert(); exp3 = ex.exp3
    srcs = {'learned': ex.base.load_rows(exp3.RUN), 'unseen': exp3._load_rows(exp3.UNSEEN)}
    units = bank()['actions']
    experts = {'EX2': (str(EX2_CK), sel['ex2_reference']['scale']), 'EX3': (sel['selected']['checkpoint'], sel['selected']['scale'])}
    for run, rec in sel.get('best_by_run', {}).items():
        if rec['checkpoint'] != sel['selected']['checkpoint']:
            experts[f'EX3alt_{run}'] = (rec['checkpoint'], rec['scale'])   # non-selected run: reference rows (oracle t0 only)
    result: dict[str, Any] = {'experts': {k: {'checkpoint': v[0], 'oracle_scale_c_dev': v[1],
                                              'confidence_map_dev': grid['maps'].get(k, {}).get('best', {}).get('ehat')} for k, v in experts.items()},
                              'ekq_confidence_available': grid['ekq_confidence_available'], 'splits': {}}
    preds_all = {}
    for split in ('learned', 'unseen'):
        sp = load_split(split)
        with np.load(EX2 / f'retrieval/{split}_retrieval.npz') as z:
            if not np.array_equal(z['key'], sp['key']):
                raise AssertionError('row order mismatch')
            ids = np.asarray(z['row_ids'], np.int64); pi05 = np.asarray(z['pi05'], np.float32)[..., :7]
        obs = ex.base.obs_batch(srcs[split], ids); noise = exp3._seed_noise(ids + (0 if split == 'learned' else 50_000_000))
        tgt = sp['actions']; hard = sp['hard']; ep = sp['task'] * 10_000 + sp['episode']
        hl = hold_last_gripper(sp['last_grip'], sp['last_valid'])
        drafts = _drafts(split, sp)
        oid, _ = exp3.nearest_unit_ids(tgt, units, bank()['task'])
        od = units[oid].copy(); od[..., 6] = hl
        preds: dict[str, np.ndarray] = {'pi05_reference': pi05, 'oracle_draft': od}
        eh = {q: ehat(q, split, sp) for q in QUERIES} if grid['ekq_confidence_available'] else {}
        for q, d in drafts.items():
            preds[f'{q}:draft'] = d
        for tag, (ck, c) in experts.items():
            ex.load(Path(ck))
            for q, d in drafts.items():
                o = ex.bridge(obs, d, noise, select_t0(arm6(d, tgt), c))
                preds[f'{q}:{tag}:oracle_t0'] = o
                m = grid['maps'].get(tag, {}).get('best', {}).get('ehat')
                if q in eh and m:
                    t = select_t0(eh[q], m['c'], m['lo'], m['hi'])
                    preds[f'{q}:{tag}:conf_t0'] = ex.bridge(obs, d, noise, t)
                    preds[f'{q}:{tag}:conf_t0__t0'] = t
            preds[f'oracle_draft:{tag}:oracle_t0'] = ex.bridge(obs, od, noise, select_t0(arm6(od, tgt), c))
            log('offline_expert', split=split, tag=tag)
        metrics = {}
        for name, p in preds.items():
            if name.endswith('__t0'):
                continue
            p = np.asarray(p, np.float32).copy()
            if ':EX2:' in name:
                p[..., 6] = hl          # EX2 was never trained on the gripper: report hold-last
            elif ':EX3' in name:
                p[..., 6] = np.where(p[..., 6] >= 0, 1., -1.)   # runtime executes the sign of the Expert gripper
            a = arm6(p, tgt); g = grip7(p, tgt)
            metrics[name] = {'arm6': float(a.mean()), 'arm6_hard': float(a[hard].mean()), 'arm6_ci95': bootstrap_ci(a, ep),
                             'active7': float((a + g).mean()), 'active7_hard': float((a + g)[hard].mean()),
                             'gripper': switch_metrics(p[..., 6], tgt[..., 6], sp['last_grip'], sp['last_valid'])}
        paired = {}
        for q in list(QUERIES) + ['oracle_draft']:
            for v in ('oracle_t0', 'conf_t0'):
                a, b = f'{q}:EX3:{v}', f'{q}:EX2:{v}'
                if a in preds and b in preds:
                    dlt = arm6(preds[a], tgt) - arm6(preds[b], tgt)
                    paired[f'{a} - {b}'] = {'mean': float(dlt.mean()), 'ci95': bootstrap_ci(dlt, ep), 'hard_mean': float(dlt[hard].mean())}
                ref = f'{q}:draft' if q != 'oracle_draft' else 'oracle_draft'
                for tag in ('EX2', 'EX3'):
                    k = f'{q}:{tag}:{v}'
                    if k in preds:
                        dlt = arm6(preds[k], tgt) - arm6(preds[ref], tgt)
                        paired[f'{k} - {ref}'] = {'mean': float(dlt.mean()), 'ci95': bootstrap_ci(dlt, ep)}
        raw_grip = {k: float(grip7(preds[k], tgt).mean()) for k in preds if ':EX3:' in k and not k.endswith('__t0')}
        result['splits'][split] = {'rows': int(len(ids)), 'hard_rows': int(hard.sum()), 'metrics': metrics, 'paired': paired,
                                   'ex3_gripper_raw_mse_before_sign': raw_grip,
                                   'hold_last_gripper': switch_metrics(hl, tgt[..., 6], sp['last_grip'], sp['last_valid']) | {'grip_mse': float(grip7(np.concatenate([tgt[..., :6], hl[..., None]], -1), tgt).mean())},
                                   't0_quantiles': {k: quantiles(v) for k, v in preds.items() if k.endswith('__t0')}}
        preds_all.update({f'{split}/{k}': v for k, v in preds.items()})
        write_json(OUT / 'offline.json', result)
    np.savez_compressed(OUT / 'offline_predictions.npz', **preds_all)
    log('offline_done')
    return result


# ---------------------------------------------------------------------------
# 6. closed loop (LIBERO-90 Eval 10x10, CL1 seeds / initial states)
# ---------------------------------------------------------------------------

CL_CONDITIONS = {'X3a': dict(mode='C1a', label='EX3＋C1a（K-x1 × E-H 注意クエリ）、π0.5 なし'),
                 'X3b': dict(mode='C1b', label='EX3＋C1b（K-x1 × 履歴＋状態MLP）、π0.5 なし')}


def closed_loop(conds: list[str], *, workers: int = 10, fraction: float = .22, timeout: int = 1800, smoke: bool = False) -> None:
    import subprocess, threading
    import run_cl1_closed_loop as cl1
    import run_ex1_scope_changed as ex1run
    sys.path.insert(0, str(HERE.parent))
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from anchor_flow.run_case import S2WebsocketClientPolicy, _run_trial
    sel = json.loads((OUT / 'select.json').read_text()); grid = json.loads((OUT / 'devgrid.json').read_text())
    root = OUT / ('closed-loop-smoke' if smoke else 'closed-loop'); root.mkdir(parents=True, exist_ok=True)
    m = grid['maps']['EX3']['best'].get('ehat')
    conf_src = 'ekq' if m else 'retrieval_only'
    if not m:
        raise RuntimeError('EKQ confidence head is not available; fit the retrieval-only Dev map first')

    def one(name, cpus):
        c = CL_CONDITIONS[name]; tasks, baseline = ex1run.load_baseline('libero90')
        cdir = root / name; (cdir / 'trials').mkdir(parents=True, exist_ok=True)
        existing = {(int(r['task_id']), int(r['trial_index'])): r for r in cl1.read_trials(cdir)}
        want = [(t, i) for t in tasks for i in range(10)]
        if smoke:
            want = [w for w in want if w[0]['task_id'] == 10 and w[1] == 0]
        pending = [(t, i) for t, i in want if existing.get((t['task_id'], i), {}).get('status') != 'complete']
        log('cl_condition_start', condition=name, pending=len(pending))
        if not pending:
            return
        port = cl1.free_port(); env = os.environ.copy()
        env.update(SKILLNET_SUBTASK_GENERATE='0', XLA_PYTHON_CLIENT_PREALLOCATE='false', XLA_PYTHON_CLIENT_MEM_FRACTION=str(fraction),
                   TORCH_MEM_FRACTION=str(fraction), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
        (root / 'servers').mkdir(exist_ok=True)
        slog = (root / 'servers' / f'{name}-{port}.log').open('a')
        cmd = ['taskset', '-c', cpus, 'nice', '-n', '10', sys.executable, str(HERE / 'serve_expert_ex3.py'), '--query', c['mode'],
               '--expert-checkpoint', sel['selected']['checkpoint'], '--map', json.dumps(m), '--port', str(port)]
        proc = subprocess.Popen(cmd, env=env, stdout=slog, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + 900
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(f'{name} server exited; see {slog.name}')
                if time.monotonic() > deadline:
                    raise TimeoutError(name)
                try:
                    cl = S2WebsocketClientPolicy('127.0.0.1', port); cl.close(); break
                except (ConnectionError, OSError, TimeoutError):
                    time.sleep(2)
            log('cl_server_ready', condition=name, port=port)
            jobs = [{'host': '127.0.0.1', 'port': port, 'suite': 'libero_90', 'task_id': int(t['task_id']), 'task_key': t['name'],
                     'trial_index': int(i), 'seed_base': 7, 'max_steps': 520, 'wait_steps': 10, 'replan_steps': 5,
                     'condition': f'EX3_{name}', 'checkpoint': sel['selected']['checkpoint'], 'timeout': timeout,
                     'memory_snapshot': 'LIBERO-40 4,516 variable-length DCT3 units (EX2 bank order)',
                     'query_text_source': 'instruction_only'} for t, i in pending]
            with ProcessPoolExecutor(max_workers=1 if smoke else workers) as pool:
                futs = {pool.submit(_run_trial, j): j for j in jobs}
                for fut in as_completed(futs):
                    j = futs[fut]
                    try:
                        r = fut.result()
                    except Exception as exc:
                        log('cl_trial_error', condition=name, task_id=j['task_id'], trial=j['trial_index'], error=repr(exc)[:400]); continue
                    r['condition'] = f'EX3_{name}'; r['confidence_source'] = conf_src
                    r['baseline_success'] = bool(baseline[(int(r['task_id']), int(r['trial_index']))]['success'])
                    with (cdir / 'trials.jsonl').open('a') as f:
                        f.write(json.dumps(r, ensure_ascii=False) + '\n'); f.flush(); os.fsync(f.fileno())
                    (cdir / 'trials' / f"task{int(r['task_id']):02d}-trial{int(r['trial_index']):02d}.json").write_text(json.dumps(r, ensure_ascii=False) + '\n')
                    log('cl_trial', condition=name, task_id=int(r['task_id']), trial=int(r['trial_index']), status=r.get('status'), success=bool(r.get('success')))
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, 15)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, 9)
            slog.close()
        log('cl_condition_end', condition=name)

    cpus = ['32-39', '40-47']
    ths = [threading.Thread(target=one, args=(n, cpus[i % 2])) for i, n in enumerate(conds)]
    [t.start() for t in ths]; [t.join() for t in ths]
    if not smoke:
        summarize_closed()


def summarize_closed() -> dict[str, Any]:
    import run_cl1_closed_loop as cl1
    import run_ex1_scope_changed as ex1run
    tasks, baseline = ex1run.load_baseline('libero90')
    keys = [(t['task_id'], i) for t in tasks for i in range(10)]
    succ, rows = {}, {}
    for n in CL_CONDITIONS:
        rows[n] = {(int(r['task_id']), int(r['trial_index'])): r for r in cl1.read_trials(OUT / 'closed-loop' / n) if r.get('status') == 'complete'}
        succ[n] = {k: bool(v['success']) for k, v in rows[n].items()}
    for n in ('C1a', 'C1b', 'C2', 'C3'):
        succ[n] = {(int(r['task_id']), int(r['trial_index'])): bool(r['success']) for r in cl1.read_trials(CL1 / n) if r.get('status') == 'complete'}
    for n in ('E1a', 'E1b', 'E1a_pg'):
        d = EKQ / 'closed-loop' / n
        if d.exists():
            succ[n] = {(int(r['task_id']), int(r['trial_index'])): bool(r['success']) for r in cl1.read_trials(d) if r.get('status') == 'complete'}
    succ['pi05'] = {k: bool(baseline[k]['success']) for k in keys}
    res = {'setting': 'LIBERO-90 Eval 10 task x 10, seed_base 7, initial states 0..9, replan 5, wait 10, 520 steps (CL1 と同じ)',
           'conditions': {}, 'mcnemar': {}, 'diagnostics': {}}
    for n, s in succ.items():
        res['conditions'][n] = {'completed': len(s), 'successes': int(sum(s.values())),
                                'tasks': {str(t['task_id']): int(sum(s.get((t['task_id'], i), False) for i in range(10))) for t in tasks}}
    for a, b in (('C1a', 'X3a'), ('C1b', 'X3b'), ('E1a', 'X3a'), ('E1b', 'X3b'), ('pi05', 'X3a'), ('pi05', 'X3b'), ('C3', 'X3a'), ('X3a', 'X3b')):
        common = [k for k in keys if k in succ.get(a, {}) and k in succ.get(b, {})]
        if common:
            res['mcnemar'][f'{b} - {a}'] = {'pairs': len(common), **mcnemar_exact([succ[a][k] for k in common], [succ[b][k] for k in common])}
    for n in CL_CONDITIONS:
        qs = [q for r in rows[n].values() for q in r.get('selected_units_and_t0', []) if isinstance(q, dict) and 'unit_id' in q]
        if not qs:
            continue
        t0 = np.asarray([q['t0'] for q in qs]); eh = np.asarray([q['confidence_error'] for q in qs])
        gsw = []
        for r in rows[n].values():
            g = [np.sign(q['gripper_chunk'][0]) for q in r.get('selected_units_and_t0', []) if isinstance(q, dict) and q.get('gripper_chunk')]
            gsw.append(int(sum(x != y for x, y in zip(g, g[1:]))))
        res['diagnostics'][n] = {'queries': int(len(qs)), 't0': quantiles(t0, (.05, .5, .95)), 'ehat': quantiles(eh, (.05, .5, .95)),
                                 'pi05_used': int(sum(bool(q.get('pi05_sampled')) for q in qs)),
                                 'gripper_changes_per_trial_mean': float(np.mean(gsw)) if gsw else None,
                                 'gripper_changes_success_mean': float(np.mean([x for x, r in zip(gsw, rows[n].values()) if r['success']] or [np.nan])),
                                 'gripper_changes_failure_mean': float(np.mean([x for x, r in zip(gsw, rows[n].values()) if not r['success']] or [np.nan]))}
    res['status'] = 'complete' if all(res['conditions'][n]['completed'] == 100 for n in CL_CONDITIONS) else 'partial'
    write_json(OUT / 'closed_loop.json', res)
    return res


def _f(x, nd=4):
    return '—' if x is None else (f'{x:.{nd}f}' if isinstance(x, float) else str(x))


def report() -> str:
    """report.md (auto tables + optional prose from report_prose.md) and merged results.json."""
    J = lambda n: json.loads((OUT / n).read_text()) if (OUT / n).exists() else {}
    meas, prep, sel, grid, off, cl = J('measure.json'), J('prepare/prepare.json'), J('select.json'), J('devgrid.json'), J('offline.json'), J('closed_loop.json')
    trains = {p.parent.name: json.loads(p.read_text()) for p in sorted(OUT.glob('train-*/train.json'))}
    L = ['# EX3：検索の崩れ方に合わせた Expert の再学習（グリッパー込み）', '']
    prose = OUT / 'report_prose.md'
    if prose.exists():
        L += [prose.read_text().strip(), '']
    if meas:
        L += ['## 1. 検索が実際に出す下書きの順位と誤差', '',
              '真の順位＝候補（自タスクの単位を全部除外）を行動 MSE で並べたときの、検索 top-1 の順位。腕6軸 MSE は標準指標（二乗和/70）。',
              '', '|集合|検索|順位 中央値 [25%, 75%, 90%]|recall@1 / @10 / @100|腕 MSE 中央値 [75%, 90%]|Dev との KS（log順位 / MSE）|', '|---|---|---|---|---|---|']
        names = {'ins': 'メモリ行・本番クエリ（自タスク除外）', 'oofT': 'メモリ行・タスク交差学習クエリ（自タスク除外）', 'dev': 'Dev（LIBERO-90 2タスク）'}
        for q in QUERIES:
            for v in ('ins', 'oofT', 'dev'):
                r = meas.get(f'{q}:{v}')
                if not r:
                    continue
                ks = r.get('ks_vs_dev'); rk = r['rank']; a = r['arm6']
                kss = '—' if not ks else '{:.2f} / {:.2f}'.format(ks['log_rank'], ks['arm6'])
                L.append(f"|{names[v]}|{q}|{rk['0.5']:.0f} [{rk['0.25']:.0f}, {rk['0.75']:.0f}, {rk['0.9']:.0f}]|{r['recall@1']:.3f} / {r['recall@10']:.3f} / {r['recall@100']:.3f}|"
                         f"{a['0.5']:.4f} [{a['0.75']:.4f}, {a['0.9']:.4f}]|{kss}|")
        e = meas['ex2_training_law']
        L.append(f"|EX2 の学習則（上位50の幾何 p=.16＋10% 一様）|—|{e['rank']['0.5']:.0f} [{e['rank']['0.25']:.0f}, {e['rank']['0.75']:.0f}, {e['rank']['0.9']:.0f}]|—|{e['arm6']['0.5']:.4f} [{e['arm6']['0.75']:.4f}, {e['arm6']['0.9']:.4f}]|{e['ks_vs_dev']['log_rank']:.2f} / {e['ks_vs_dev']['arm6']:.2f}|")
        L.append('')
    if trains:
        L += ['## 2. 学習', '', '|run|順位則|近傍の割合|グリッパー重み|更新|状態|', '|---|---|---:|---:|---:|---|']
        for k, t in trains.items():
            L.append(f"|{k}|測った則 × {t.get('rank_scale', 1.0)}|{t.get('near_fraction', .15)}|{t.get('grip_weight')}|{t.get('updates')}|{t.get('status')}|")
        L.append('')
    if sel.get('scores'):
        L += ['## 3. Dev での checkpoint 選択（C1a＋C1b の Dev top-1 下書き、t0 は真の誤差から c 倍、生の NNX 状態で採点）', '',
              '|checkpoint|' + '|'.join(f'c={c}' for c in SCALES) + '|', '|---|' + '---:|' * len(SCALES)]
        for k, v in sel['scores'].items():
            L.append(f'|{k}|' + '|'.join(_f(v['dev_arm6_by_scale'].get(str(c))) for c in SCALES) + '|')
        if 'selected' in sel:
            L += ['', f"選択：{sel['selected']['label']}（c={sel['selected']['scale']}、Dev {sel['selected']['dev_arm6']:.4f}）。EX2 step 2500：c={sel['ex2_reference']['scale']}、Dev {sel['ex2_reference']['dev_arm6']:.4f}。Dev 下書き：" +
                  ', '.join(f'{q} {v:.4f}' for q, v in sel['draft_dev_arm6'].items()) + '。']
        L.append('')
    if grid:
        L += ['## 4. 信頼度から t0 への写像（Dev、π0.5 なし、退避なし）', '']
        for tag, m in grid['maps'].items():
            b = m['best']
            L.append(f"- {tag}: 真の誤差での最良 {b.get('true')}、ê（EKQ 信頼度）での最良 {b.get('ehat')}、行ごと最適 t0 {m['per_row_oracle_t0']:.4f}、下書き {m['draft']:.4f}。")
        if grid.get('ehat_dev_spearman'):
            L.append(f"- ê と真の下書き誤差の Dev Spearman：{grid['ehat_dev_spearman']}")
        L.append('')
    if off:
        L += ['## 5. オフライン主表（LIBERO 学習済み 621 行／未学習 619 行）', '',
              '腕6軸 MSE（全体 / 難）と、グリッパーを含む7次元 MSE（全体）。EX2 はグリッパーを学習していないので、EX2 行のグリッパーは保持（hold-last）。EX3 行は Expert のグリッパーの符号。',
              '', '|行|学習済み 腕 全体 / 難|学習済み 7次元|未学習 腕 全体 [95%CI] / 難|未学習 7次元|', '|---|---|---:|---|---:|']
        rows = []
        for q in QUERIES:
            rows += [f'{q}:draft'] + [f'{q}:{t}:{v}' for t in ['EX2', 'EX3'] + sorted(k for k in off['experts'] if k.startswith('EX3alt')) for v in ('oracle_t0', 'conf_t0')]
        rows += ['oracle_draft'] + [f'oracle_draft:{t}:oracle_t0' for t in ['EX2', 'EX3'] + sorted(k for k in off['experts'] if k.startswith('EX3alt'))] + ['pi05_reference']
        for r in rows:
            ml = off['splits'].get('learned', {}).get('metrics', {}).get(r); mu = off['splits'].get('unseen', {}).get('metrics', {}).get(r)
            if not ml and not mu:
                continue
            g = lambda m, k: _f(m.get(k)) if m else '—'
            ci = f" [{mu['arm6_ci95'][0]:.4f}, {mu['arm6_ci95'][1]:.4f}]" if mu else ''
            L.append(f"|{r}|{g(ml, 'arm6')} / {g(ml, 'arm6_hard')}|{g(ml, 'active7')}|{g(mu, 'arm6')}{ci} / {g(mu, 'arm6_hard')}|{g(mu, 'active7')}|")
        L += ['', '対比較（腕6軸、episode bootstrap 95%）：', '']
        for split in ('learned', 'unseen'):
            for k, v in off['splits'].get(split, {}).get('paired', {}).items():
                L.append(f"- {split} / {k}: {v['mean']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]")
        L += ['', '### グリッパー（直前に実行したグリッパーからの切り替え）', '', '|分割|行|切替のある行|見逃し率|誤切替率|時刻誤差（step）|±1 step 以内|', '|---|---|---:|---:|---:|---:|---:|']
        for split in ('learned', 'unseen'):
            sp = off['splits'].get(split, {})
            items = [('hold-last', sp.get('hold_last_gripper'))] + [(k, v['gripper']) for k, v in sp.get('metrics', {}).items() if ':EX3:' in k or k == 'pi05_reference']
            for name, m in items:
                if m:
                    L.append(f"|{split}|{name}|{m['gt_switch_rows']}|{_f(m['missed_switch_rate'], 3)}|{_f(m['false_switch_rate'], 3)}|{_f(m['timing_error_mean_steps'], 2)}|{_f(m['within1_step_rate'], 3)}|")
        L.append('')
    if cl:
        L += ['## 6. 閉ループ（LIBERO-90 Eval 10タスク×10試行、CL1 と同じ seed・初期状態）', '', '|条件|成功/完了|' + '|'.join(f't{t}' for t in next(iter(cl['conditions'].values()))['tasks']) + '|',
              '|---|---:|' + '---:|' * len(next(iter(cl['conditions'].values()))['tasks'])]
        for n, v in cl['conditions'].items():
            L.append(f"|{n}|{v['successes']}/{v['completed']}|" + '|'.join(str(x) for x in v['tasks'].values()) + '|')
        L += ['', '|比較 B − A|対|B だけ成功|A だけ成功|p|', '|---|---:|---:|---:|---:|']
        for k, v in cl['mcnemar'].items():
            L.append(f"|{k}|{v['pairs']}|{v['b_only']}|{v['a_only']}|{v['p']:.3f}|")
        for n, d in cl.get('diagnostics', {}).items():
            L.append(f"- {n}: 問い合わせ {d['queries']}、t0 {d['t0']}、ê {d['ehat']}、π0.5 使用 {d['pi05_used']} 回、試行あたりグリッパー切替 {_f(d['gripper_changes_per_trial_mean'], 2)}（成功 {_f(d['gripper_changes_success_mean'], 2)}／失敗 {_f(d['gripper_changes_failure_mean'], 2)}）")
        L.append('')
    text = '\n'.join(L) + '\n'
    (OUT / 'report.md').write_text(text)
    write_json(OUT / 'results.json', {'measure': meas, 'prepare': prep, 'train': trains, 'select': {k: v for k, v in sel.items() if k != 'scores'} | {'scores': {k: v['dev_arm6_by_scale'] for k, v in sel.get('scores', {}).items()}},
                                      'devgrid': grid, 'offline': off, 'closed_loop': cl})
    return text


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command', choices=('measure', 'prepare', 'train', 'select', 'devgrid', 'offline', 'closed', 'closed-smoke', 'summarize', 'report'))
    p.add_argument('--name', default='G1'); p.add_argument('--updates', type=int, default=5000)
    p.add_argument('--grip-weight', type=float, default=1.0); p.add_argument('--microbatch', type=int, default=8)
    p.add_argument('--near-fraction', type=float, default=.15); p.add_argument('--rank-scale', type=float, default=1.0); p.add_argument('--variant', default='oofT')
    p.add_argument('--conditions', default='X3a,X3b'); p.add_argument('--workers', type=int, default=10)
    p.add_argument('--fraction', type=float, default=.22)
    a = p.parse_args()
    log('start', command=a.command, argv=sys.argv[1:])
    if a.command == 'measure':
        measure()
    elif a.command == 'prepare':
        prepare(a.variant, a.near_fraction)
    elif a.command == 'train':
        train(a.name, updates=a.updates, grip_weight=a.grip_weight, microbatch=a.microbatch, near_fraction=a.near_fraction, rank_scale=a.rank_scale)
    elif a.command == 'select':
        select()
    elif a.command == 'devgrid':
        devgrid()
    elif a.command == 'offline':
        offline()
    elif a.command in ('closed', 'closed-smoke'):
        closed_loop(a.conditions.split(','), workers=a.workers, fraction=a.fraction, smoke=a.command == 'closed-smoke')
    elif a.command == 'report':
        report()
    elif a.command == 'summarize':
        print(json.dumps({k: v['successes'] for k, v in summarize_closed()['conditions'].items()}))
    log('end', command=a.command)


if __name__ == '__main__':
    main()
