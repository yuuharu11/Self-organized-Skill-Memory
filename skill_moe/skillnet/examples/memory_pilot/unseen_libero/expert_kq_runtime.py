#!/usr/bin/env python3
"""EKQ: Expert runtime without π0.5 for confidence (t0) and gripper, refit for KQ-final.

The Expert weights (EX2 production-05 A step 2500) are frozen. Only what sits around
them is refit:

* confidence ê: predicts log arm-6 MSE of the retrieved draft from quantities that exist
  at inference without π0.5 (score margin/entropy on the KQ-final scale, agreement among the
  top-k unit drafts, continuity between the draft start and the last executed action,
  history/state magnitude, query norm).
* t0 map: t0 = clip(c * sqrt(ê), lo, hi), (c, lo, hi, optional π0.5 fallback threshold)
  chosen on Dev by top-1 arm-6 MSE after the Expert.
* gripper: a small head over action/gripper history, state and the retrieved draft,
  10 per-step open/close logits.

Data rules: fitting uses LIBERO-40 memory-side rows only (the KQ-final ``eh`` rows, 400
memory episodes, disjoint from the learned Eval episodes). Dev (217 LIBERO-90 rows) selects.
learned 621 / unseen 619 are scored once. No task id, ground truth or future enters
inference. π0.5 appears only as a reference column and as an optional fallback.

Memory rows are scored three ways (``variant``):
  ins   production query, own episode masked (in-sample for the query: it was trained on them)
  oofE  task-fold cross-fitted query (query never trained on the row's task), own episode masked
  oofT  same cross-fitted query, every candidate of the row's own task masked (unseen-like)

Subcommands: crossfit, retrieve, fit, dev-grid, select, eval, report.
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
sys.path.insert(0, str(HERE))
ROOT = (STATE_DIR / 'outputs/memory-pilot/unseen-libero')
OUT = ROOT / 'expert-kq-01'
KQF = ROOT / 'key-motion-01/final'
CL1 = ROOT / 'closed-loop-kq-01'
EX2_RET = CL1 / 'expert-select/retrieval'
EX2_CONF = ROOT / 'expert-ex2-01/production-05/confidence/confidence_head.pt'
QUERIES = {'C1a': KQF / 'best/Afix_x1_fixed_e_h_cteff.pt', 'C1b': KQF / 'best/H_x1_mk1b_cteff_tau0.01_s1.pt'}
SELECTED_EPOCH = {'C1a': 8, 'C1b': 14}   # KQ-final Dev-selected epochs of the production rows
N_FOLDS = 5
TOPK = 20
KCONF = 5
T0_GRID = (.02, .05, .1, .15, .2, .25, .3, .35, .4, .5, .6, .7, .8, .9)
CONF_FEATURES = ('margin12', 'entropy', 'log_top1_prob', 'top1_score', 'log_topk_pair_mse', 'log_topk_weighted_var',
                 'log_start_jump', 'history_valid_frac', 'log_history_arm_abs', 'log_history_last_change',
                 'log_draft_rms', 'log_query_norm', 'state_norm')


def utc() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def log(event: str, out: Path = OUT, **kw: Any) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rec = {'utc': utc(), 'event': event, **kw}
    with (out / 'progress.log').open('a') as f:
        f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print(json.dumps(rec, ensure_ascii=False), flush=True)


def write_json(path: Path, value: Any) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


# ---------------------------------------------------------------------------
# pure helpers (shared with the server; tested)
# ---------------------------------------------------------------------------

def arm6_mse(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Standard metric: sum of 10x6 squared arm errors / 70."""
    p = np.asarray(pred, np.float64); t = np.asarray(target, np.float64)
    return (np.square(p[..., :6] - t[..., :6]).sum((-1, -2)) / 70.).astype(np.float64)


def grip_mse(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Gripper contribution to active-7 MSE: sum over 10 steps of squared error / 70."""
    p = np.asarray(pred, np.float64); t = np.asarray(target, np.float64)
    p = p[..., 6] if p.ndim == 3 else p
    t = t[..., 6] if t.ndim == 3 else t
    if p.shape != t.shape or p.shape[-1] != 10:
        raise ValueError(f'gripper arrays must be [N,10] or [N,10,>=7]: {p.shape}/{t.shape}')
    return np.square(p - t).sum(-1) / 70.


def score_summary(scores: np.ndarray, k: int = TOPK) -> dict[str, np.ndarray]:
    """Top-k ids/scores, normalized softmax entropy and top-1 probability.

    Masked candidates are -inf; entropy is normalized by log(#finite candidates), which equals
    the EX2 ``confidence_features`` entropy when nothing is masked.
    """
    s = np.asarray(scores, np.float64)
    if s.ndim != 2 or s.shape[1] < 2:
        raise ValueError(f'scores must be [N, M>=2], got {s.shape}')
    finite = np.isfinite(s)
    if np.any(finite.sum(1) < 2) or np.isnan(s).any() or np.isposinf(s).any():
        raise ValueError('each row needs two finite scores and no NaN/+inf')
    order = np.argsort(-s, axis=1, kind='stable')[:, :k]
    vals = np.take_along_axis(s, order, 1)
    z = s - s.max(1, keepdims=True)
    p = np.exp(z); p /= p.sum(1, keepdims=True)
    ent = -(p * np.log(np.maximum(p, 1e-30))).sum(1) / np.log(np.maximum(finite.sum(1), 2))
    return {'ids': order.astype(np.int32), 'vals': vals.astype(np.float32), 'entropy': ent.astype(np.float32),
            'top1_prob': p.max(1).astype(np.float32)}


def confidence_features(top_ids: np.ndarray, top_vals: np.ndarray, entropy: np.ndarray, top1_prob: np.ndarray,
                        units: np.ndarray, history: np.ndarray, history_mask: np.ndarray, state: np.ndarray,
                        query_norm: np.ndarray, k: int = KCONF) -> np.ndarray:
    """π0.5-free inference features for the draft-error regressor (see CONF_FEATURES).

    ``units`` are the stored [M,10,>=6] candidate windows; ``history`` is the last ten executed
    normalized actions (zero where ``history_mask`` is False); ``state`` is the 8-d normalized
    proprio state. No target, task, episode or π0.5 output is accepted.
    """
    ids = np.asarray(top_ids, np.int64)[:, :k]; v = np.asarray(top_vals, np.float64)[:, :k]
    u = np.asarray(units, np.float64)[..., :6]
    h = np.asarray(history, np.float64); m = np.asarray(history_mask, bool); st = np.asarray(state, np.float64)
    n = len(ids)
    if ids.shape[1] < 2 or h.shape != (n, 10, 7) or m.shape != (n, 10) or st.shape[0] != n:
        raise ValueError(f'bad shapes {ids.shape} {h.shape} {m.shape} {st.shape}')
    drafts = u[ids]                                   # [N,k,10,6]
    top1 = drafts[:, 0]
    pair = np.square(drafts[:, 1:] - top1[:, None]).sum((2, 3)).mean(1) / 70.
    w = np.exp(v - v.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
    mean = (w[:, :, None, None] * drafts).sum(1)
    wvar = (w[:, :, None, None] * np.square(drafts - mean[:, None])).sum(1).sum((1, 2)) / 70.
    valid = m.sum(1)
    last_idx = np.where(valid > 0, 9, 0)              # history is right-aligned
    last = h[np.arange(n), last_idx, :6]
    jump = np.where(valid > 0, np.square(top1[:, 0] - last).mean(1), 0.)
    habs = np.where(valid > 0, (np.abs(h[..., :6]) * m[..., None]).sum((1, 2)) / np.maximum(valid * 6, 1), 0.)
    change = np.where(valid > 1, np.square(h[:, 9, :6] - h[:, 8, :6]).mean(1), 0.)
    rms = np.sqrt(np.square(top1).mean((1, 2)))
    feats = np.stack([v[:, 0] - v[:, 1], np.asarray(entropy, np.float64), np.log(np.asarray(top1_prob, np.float64) + 1e-12),
                      v[:, 0], np.log(pair + 1e-6), np.log(wvar + 1e-6), np.log(jump + 1e-6), valid / 10.,
                      np.log(habs + 1e-6), np.log(change + 1e-6), np.log(rms + 1e-6),
                      np.log(np.asarray(query_norm, np.float64) + 1e-6), np.linalg.norm(st[:, :8], axis=1)], 1)
    if not np.isfinite(feats).all():
        raise ValueError('nonfinite confidence features')
    return feats.astype(np.float32)


def gripper_inputs(top_ids: np.ndarray, top_vals: np.ndarray, units: np.ndarray, history: np.ndarray,
                   history_mask: np.ndarray, state: np.ndarray, k: int = KCONF) -> np.ndarray:
    """Inputs of the gripper head: history (70) + mask (10) + state (8) + top-1 draft arm (60)
    + top-1 stored gripper (10) + score-weighted top-k 'unit flips relative to its first step' (10)."""
    ids = np.asarray(top_ids, np.int64)[:, :k]; v = np.asarray(top_vals, np.float64)[:, :k]
    u = np.asarray(units, np.float32)
    h = np.asarray(history, np.float32); m = np.asarray(history_mask, np.float32); st = np.asarray(state, np.float32)
    n = len(ids)
    top1 = u[ids[:, 0]]
    g = u[ids][..., 6]                                 # [N,k,10]
    flips = (np.sign(g) != np.sign(g[..., :1])).astype(np.float32)
    w = np.exp(v - v.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
    wf = (w[:, :, None] * flips).sum(1)
    x = np.concatenate([h.reshape(n, -1), m.reshape(n, -1), st[:, :8], top1[..., :6].reshape(n, -1), top1[..., 6], wf], 1)
    return x.astype(np.float32)


FLIP_THRESHOLDS = (.5, .6, .7, .8, .9)


def switch_error_count(sm: dict[str, Any]) -> int:
    n_sw = sm['gt_switch_rows']; n_no = sm['rows_with_history'] - n_sw
    return int(round(sm['missed_switch_rate'] * n_sw + sm['false_switch_rate'] * n_no))


GRIP_INPUT_DIM = 70 + 10 + 8 + 60 + 10 + 10
DEFAULT_START_GRIP = -1.0  # LIBERO normalized gripper at episode start (open)


def hold_last(history: np.ndarray, history_mask: np.ndarray, default: float = DEFAULT_START_GRIP) -> np.ndarray:
    h = np.asarray(history, np.float32); m = np.asarray(history_mask, bool)
    last = np.where(m.any(1), h[:, 9, 6], default)
    last = np.where(last == 0, default, np.sign(last))
    return np.repeat(last[:, None], 10, 1).astype(np.float32)


def unit_flip_rule(history: np.ndarray, history_mask: np.ndarray, unit_grip: np.ndarray,
                   default: float = DEFAULT_START_GRIP) -> np.ndarray:
    """Hold the last executed gripper, flipping wherever the retrieved unit flips."""
    base = hold_last(history, history_mask, default)
    ug = np.asarray(unit_grip, np.float32)
    flip = np.sign(ug) != np.sign(ug[:, :1])
    return np.where(flip, -base, base).astype(np.float32)


def select_t0(error, c: float, lo: float, hi: float) -> np.ndarray:
    e = np.asarray(error, np.float64)
    if np.any(e < 0) or not np.isfinite(e).all() or not 0 <= lo < hi <= 1 or c < 0:
        raise ValueError('bad t0 map inputs')
    return np.clip(c * np.sqrt(e), lo, hi).astype(np.float32)


def interp_curve(curve: np.ndarray, grid, t0: np.ndarray) -> np.ndarray:
    """Per-row linear interpolation of an MSE-vs-t0 curve [N,G] at t0 [N]."""
    g = np.asarray(grid, np.float64); c = np.asarray(curve, np.float64); t = np.clip(np.asarray(t0, np.float64), g[0], g[-1])
    j = np.clip(np.searchsorted(g, t, side='right') - 1, 0, len(g) - 2)
    a = (t - g[j]) / (g[j + 1] - g[j])
    r = np.arange(len(t))
    return (1 - a) * c[r, j] + a * c[r, j + 1]


def switch_metrics(pred: np.ndarray, target: np.ndarray, history: np.ndarray, history_mask: np.ndarray) -> dict[str, Any]:
    """Gripper switch timing vs the last executed gripper (rows without history are skipped).

    A switch is the first step whose sign differs from the last executed sign.
    """
    p = np.sign(np.asarray(pred, np.float64)); t = np.sign(np.asarray(target, np.float64))
    if p.ndim == 3:
        p = p[..., 6]
    if t.ndim == 3:
        t = t[..., 6]
    h = np.asarray(history, np.float64); m = np.asarray(history_mask, bool)
    ok = m.any(1) & (h[:, 9, 6] != 0)
    g0 = np.sign(h[:, 9, 6])

    def first(x):
        d = x != g0[:, None]
        return np.where(d.any(1), d.argmax(1), -1)
    kp, kt = first(p), first(t)
    gt_sw = ok & (kt >= 0); pr_sw = ok & (kp >= 0)
    both = gt_sw & pr_sw
    err = np.abs(kp[both] - kt[both])
    n_sw = int(gt_sw.sum()); n_no = int((ok & ~gt_sw).sum())
    return {'rows_with_history': int(ok.sum()), 'gt_switch_rows': n_sw,
            'missed_switch_rate': float((gt_sw & ~pr_sw).sum() / max(n_sw, 1)),
            'false_switch_rate': float((ok & ~gt_sw & pr_sw).sum() / max(n_no, 1)),
            'timing_error_mean_steps': float(err.mean()) if len(err) else None,
            'timing_error_median_steps': float(np.median(err)) if len(err) else None,
            'within1_step_rate': float((err <= 1).sum() / max(n_sw, 1)),
            'sign_step_error_rate': float((p[ok] != t[ok]).mean()) if ok.any() else None}


def spearman(a, b) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).correlation)


def calibration_table(pred: np.ndarray, true: np.ndarray, bins: int = 5) -> list[dict[str, float]]:
    p = np.asarray(pred, np.float64); t = np.asarray(true, np.float64)
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    out = []
    for i in range(bins):
        m = (p >= edges[i]) & ((p <= edges[i + 1]) if i == bins - 1 else (p < edges[i + 1]))
        if m.any():
            out.append({'bin': i, 'rows': int(m.sum()), 'pred_median': float(np.median(p[m])), 'true_median': float(np.median(t[m])),
                        'true_mean': float(t[m].mean())})
    return out


# ---------------------------------------------------------------------------
# torch heads
# ---------------------------------------------------------------------------

def new_conf_head(kind: str, dim: int = len(CONF_FEATURES)):
    import torch
    from torch import nn

    class ConfHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.kind = kind
            self.net = nn.Linear(dim, 1) if kind == 'linear' else nn.Sequential(
                nn.Linear(dim, 64), nn.GELU(), nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))
            self.register_buffer('x_mean', torch.zeros(dim)); self.register_buffer('x_std', torch.ones(dim))
            self.register_buffer('bias', torch.zeros(()))   # additive log-space calibration offset

        def forward(self, x):
            return self.net((x - self.x_mean) / self.x_std).squeeze(-1) + self.bias
    return ConfHead()


def new_grip_head(hidden: int = 128, dim: int = GRIP_INPUT_DIM):
    import torch
    from torch import nn

    class GripHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 10))
            self.register_buffer('x_mean', torch.zeros(dim)); self.register_buffer('x_std', torch.ones(dim))
            # flip only when P(opposite of the last executed gripper) exceeds this (selected on memory CV)
            self.register_buffer('flip_threshold', torch.tensor(.5))

        def forward(self, x):
            return self.net((x - self.x_mean) / self.x_std)
    return GripHead()


def fit_conf_head(x, y_log, kind, *, steps=3000, lr=2e-3, wd=1e-3, seed=0, loss='l2'):
    import torch
    torch.manual_seed(seed)
    xt = torch.as_tensor(np.asarray(x, np.float32)); yt = torch.as_tensor(np.asarray(y_log, np.float32))
    m = new_conf_head(kind)
    m.x_mean.copy_(xt.mean(0)); m.x_std.copy_(xt.std(0).clamp_min(1e-4))
    opt = torch.optim.AdamW(m.net.parameters(), lr=lr, weight_decay=wd)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        ix = torch.randint(0, len(xt), (min(1024, len(xt)),), generator=g)
        pr = m(xt[ix])
        lo = torch.nn.functional.mse_loss(pr, yt[ix]) if loss == 'l2' else torch.nn.functional.smooth_l1_loss(pr, yt[ix])
        opt.zero_grad(); lo.backward(); opt.step()
    return m.eval()


def predict_conf(m, x) -> np.ndarray:
    import torch
    with torch.no_grad():
        return np.exp(m(torch.as_tensor(np.asarray(x, np.float32))).numpy().astype(np.float64)).clip(1e-6, 10.)


def fit_grip_head(x, y_sign, *, hidden=128, steps=4000, lr=1e-3, wd=1e-4, seed=0, pos_weight=1.0):
    import torch
    torch.manual_seed(seed)
    xt = torch.as_tensor(np.asarray(x, np.float32)); yt = torch.as_tensor((np.asarray(y_sign) > 0).astype(np.float32))
    m = new_grip_head(hidden)
    m.x_mean.copy_(xt.mean(0)); m.x_std.copy_(xt.std(0).clamp_min(1e-4))
    opt = torch.optim.AdamW(m.net.parameters(), lr=lr, weight_decay=wd)
    g = torch.Generator().manual_seed(seed)
    pw = torch.tensor(float(pos_weight))
    for _ in range(steps):
        ix = torch.randint(0, len(xt), (min(512, len(xt)),), generator=g)
        lo = torch.nn.functional.binary_cross_entropy_with_logits(m(xt[ix]), yt[ix], pos_weight=pw)
        opt.zero_grad(); lo.backward(); opt.step()
    return m.eval()


def grip_close_prob(m, x) -> np.ndarray:
    import torch
    with torch.no_grad():
        return torch.sigmoid(m(torch.as_tensor(np.asarray(x, np.float32)))).numpy().astype(np.float64)


def grip_from_prob(p_close, last_sign, threshold: float = .5) -> np.ndarray:
    """Hold the last executed sign unless P(opposite) > threshold at that step."""
    p = np.asarray(p_close, np.float64); g0 = np.asarray(last_sign, np.float64).reshape(-1, 1)
    p_opp = np.where(g0 < 0, p, 1 - p)
    return np.where(p_opp > threshold, -g0, g0).astype(np.float32)


def predict_grip(m, x, history=None, history_mask=None) -> np.ndarray:
    """Gripper chunk in {-1,+1}. With history, uses the head's stored flip threshold around hold-last."""
    p = grip_close_prob(m, x)
    if history is None:
        return np.where(p > .5, 1., -1.).astype(np.float32)
    g0 = hold_last(history, history_mask)[:, 0]
    return grip_from_prob(p, g0, float(m.flip_threshold))


# ---------------------------------------------------------------------------
# 1. task-fold cross-fitting of the KQ-final queries (memory side only)
# ---------------------------------------------------------------------------

def fold_of_task(task) -> np.ndarray:
    return np.mod(np.asarray(task, np.int64), N_FOLDS)


def crossfit(qname: str, folds: list[int], device: str = 'cuda') -> None:
    import torch
    import key_final as kf
    import key_shared_latent as k2
    from torch import nn
    blob = torch.load(QUERIES[qname], map_location='cpu', weights_only=False)
    cfg = dict(blob['cfg'])
    ctx = kf.RowContext(KQF, torch.device(device), False)
    train = ctx.splits['eh']
    tasks = np.asarray(train['task'], np.int64)
    for f in folds:
        path = OUT / 'crossfit' / f'{qname}_fold{f}.pt'
        if path.exists():
            log('crossfit_skip', qname=qname, fold=f); continue
        seed = int(cfg['seed']) + 17 * f
        torch.manual_seed(seed); rng = np.random.default_rng(seed)
        qmodel = kf.make_query(cfg).to(device)
        keys = torch.as_tensor(np.asarray(ctx.fixed['x1_mk1b']), device=device, dtype=torch.float32)
        opt = torch.optim.AdamW(qmodel.parameters(), lr=float(cfg['lr']), weight_decay=1e-4)
        rows = np.flatnonzero(fold_of_task(tasks) != f)
        pool = np.flatnonzero(fold_of_task(ctx.bt) != f)   # held-out tasks' units never appear in training
        t_start = time.time()
        for ep in range(1, SELECTED_EPOCH[qname] + 1):
            rng.shuffle(rows); qmodel.train(); losses = []
            for s in range(0, len(rows), cfg['batch']):
                qi = rows[s:s + cfg['batch']]
                rand = rng.choice(pool, size=min(cfg['rand'], len(pool)), replace=False)
                own = train['candidate_id'][qi]
                cids = np.unique(np.r_[own[own >= 0], rand])
                allowed = ~((train['task'][qi, None] == ctx.bt[cids][None, :]) & (train['episode'][qi, None] == ctx.be[cids][None, :]))
                q = qmodel(**kf.pack(train, qi, cfg, device))
                sc = kf.score(q, keys[torch.as_tensor(cids, device=device)], 'dot', qmodel.logit_scale)
                err = torch.as_tensor(k2._arm6_candidate_mse(np.asarray(train['actions'][qi]), ctx.windows[cids]), device=device)
                am = torch.as_tensor(allowed, device=device)
                loss = k2.soft_listwise_loss(sc, err, am, float(cfg['tau']))
                if cfg['same_scene']:
                    same = (ctx.bt[cids][None, :] == train['task'][qi, None]) & (ctx.be[cids][None, :] != train['episode'][qi, None])
                    sm = torch.as_tensor(same, device=device) & am
                    negid = err.masked_fill(~sm, -torch.inf).argmax(-1); posid = err.masked_fill(~am, torch.inf).argmin(-1)
                    has = sm.any(-1)
                    if has.any():
                        pos = sc.gather(1, posid[:, None]).squeeze(1); neg = sc.gather(1, negid[:, None]).squeeze(1)
                        loss = loss + .1 * torch.nn.functional.softplus(neg[has] - pos[has] + .1).mean()
                opt.zero_grad(set_to_none=True); loss.backward()
                nn.utils.clip_grad_norm_(qmodel.parameters(), 1.); opt.step()
                with torch.no_grad():
                    qmodel.logit_scale.clamp_(max=float(np.log(100.)))
                losses.append(float(loss.detach()))
            log('crossfit_epoch', qname=qname, fold=f, epoch=ep, loss=float(np.mean(losses)), seconds=round(time.time() - t_start))
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'cfg': cfg, 'query': {k: v.detach().cpu() for k, v in qmodel.state_dict().items()}, 'fold': f,
                    'heldout_tasks': sorted(set(tasks[fold_of_task(tasks) == f].tolist())), 'epochs': SELECTED_EPOCH[qname],
                    'train_rows': int(len(rows)), 'candidate_pool': int(len(pool))}, path)
        log('crossfit_done', qname=qname, fold=f, rows=int(len(rows)))


# ---------------------------------------------------------------------------
# 2. retrieval summaries for every split / variant
# ---------------------------------------------------------------------------

def _load_query(path, device):
    import torch, key_final as kf
    blob = torch.load(path, map_location='cpu', weights_only=False)
    m = kf.make_query(blob['cfg']).to(device); m.load_state_dict(blob['query']); m.eval()
    return m, blob['cfg']


def _score_rows(model, cfg, split, ids, keys, device, mask_fn=None):
    import torch, key_final as kf
    outs = {'ids': [], 'vals': [], 'entropy': [], 'top1_prob': [], 'qnorm': [], 'margin_full': []}
    for s in range(0, len(ids), 256):
        ix = ids[s:s + 256]
        with torch.no_grad():
            q = model(**kf.pack(split, ix, cfg, device))
            sc = kf.score(q, keys, 'dot', model.logit_scale).double().cpu().numpy()
        if mask_fn is not None:
            sc[mask_fn(ix)] = -np.inf
        summ = score_summary(sc)
        for k in ('ids', 'vals', 'entropy', 'top1_prob'):
            outs[k].append(summ[k])
        outs['qnorm'].append(q.float().norm(dim=-1).cpu().numpy())
    return {k: np.concatenate(v) for k, v in outs.items() if v}


def retrieve(device: str = 'cuda') -> None:
    import torch, key_final as kf
    torch.set_num_threads(8)
    ctx = kf.RowContext(KQF, torch.device(device), False)
    keys = torch.as_tensor(np.asarray(ctx.fixed['x1_mk1b']), device=device, dtype=torch.float32)
    bt, be = ctx.bt, ctx.be
    for qname, qpath in QUERIES.items():
        dest = OUT / 'retrieval' / f'{qname}.npz'
        if dest.exists():
            log('retrieve_skip', qname=qname); continue
        res = {}
        model, cfg = _load_query(qpath, device)
        for split in ('dev', 'learned', 'unseen'):
            sp = ctx.splits[split]
            r = _score_rows(model, cfg, sp, np.arange(len(sp['key'])), keys, device)   # server path: no mask
            for k, v in r.items():
                res[f'{split}__{k}'] = v
        eh = ctx.splits['eh']; n = len(eh['key'])
        own_ep = lambda ix: (eh['task'][ix, None] == bt[None]) & (eh['episode'][ix, None] == be[None])
        own_task = lambda ix: eh['task'][ix, None] == bt[None]
        r = _score_rows(model, cfg, eh, np.arange(n), keys, device, own_ep)
        for k, v in r.items():
            res[f'eh_ins__{k}'] = v
        folds = fold_of_task(eh['task'])
        for variant, mfn in (('oofE', own_ep), ('oofT', own_task)):
            acc = {}
            for f in range(N_FOLDS):
                fm, fcfg = _load_query(OUT / 'crossfit' / f'{qname}_fold{f}.pt', device)
                ix = np.flatnonzero(folds == f)
                r = _score_rows(fm, fcfg, eh, ix, keys, device, mfn)
                for k, v in r.items():
                    acc.setdefault(k, np.zeros((n,) + v.shape[1:], v.dtype))[ix] = v
            for k, v in acc.items():
                res[f'eh_{variant}__{k}'] = v
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dest, **res)
        units = ctx.windows
        msg = {}
        for split in ('dev', 'learned', 'unseen'):
            msg[split] = float(arm6_mse(units[res[f'{split}__ids'][:, 0]], ctx.splits[split]['actions']).mean())
        for v in ('ins', 'oofE', 'oofT'):
            msg['eh_' + v] = float(arm6_mse(units[res[f'eh_{v}__ids'][:, 0]], eh['actions']).mean())
        log('retrieve_done', qname=qname, top1_arm6=msg)


# ---------------------------------------------------------------------------
# 3. fit confidence and gripper heads (CPU), select on Dev
# ---------------------------------------------------------------------------

def load_split(name):
    d = KQF / 'data'
    s = {k: np.load(d / f'{name}__{k}.npy') for k in ('key', 'history', 'history_mask', 'state', 'actions', 'task', 'episode')}
    s['history'] = s['history'].astype(np.float32); s['history_mask'] = s['history_mask'].astype(bool)
    s['state'] = s['state'].astype(np.float32); s['actions'] = s['actions'].astype(np.float32)
    hp = d / f'{name}__hard.npy'
    s['hard'] = np.load(hp).astype(bool) if hp.exists() else np.zeros(len(s['key']), bool)
    return s


def units_bank():
    return np.load(KQF / 'data/bank_actions.npy').astype(np.float32)


def split_inputs(ret, prefix, sp, units):
    feats = confidence_features(ret[prefix + '__ids'], ret[prefix + '__vals'], ret[prefix + '__entropy'], ret[prefix + '__top1_prob'],
                                units, sp['history'], sp['history_mask'], sp['state'], ret[prefix + '__qnorm'])
    gin = gripper_inputs(ret[prefix + '__ids'], ret[prefix + '__vals'], units, sp['history'], sp['history_mask'], sp['state'])
    draft = units[ret[prefix + '__ids'][:, 0]]
    return feats, gin, draft


def fit_heads() -> dict[str, Any]:
    import torch
    torch.set_num_threads(8)
    units = units_bank()
    sp = {n: load_split(n) for n in ('eh', 'dev', 'learned', 'unseen')}
    eh_ep = sp['eh']['task'] * 1000 + sp['eh']['episode']
    out = {'confidence': {}, 'gripper': {}}
    for qname in QUERIES:
        ret = dict(np.load(OUT / 'retrieval' / f'{qname}.npz'))
        dev_f, dev_g, dev_d = split_inputs(ret, 'dev', sp['dev'], units)
        dev_true = arm6_mse(dev_d, sp['dev']['actions'])
        # ---- confidence -------------------------------------------------
        grid = []
        heads = {}
        for variant in ('ins', 'oofE', 'oofT', 'oofET'):
            if variant == 'oofET':
                parts = [split_inputs(ret, 'eh_oofE', sp['eh'], units), split_inputs(ret, 'eh_oofT', sp['eh'], units)]
                x = np.concatenate([p[0] for p in parts]); y = np.concatenate([arm6_mse(p[2], sp['eh']['actions']) for p in parts])
            else:
                x, _, d = split_inputs(ret, f'eh_{variant}', sp['eh'], units)
                y = arm6_mse(d, sp['eh']['actions'])
            for kind in ('linear', 'mlp'):
                m = fit_conf_head(x, np.log(y + 1e-6), kind, seed=1)
                p = predict_conf(m, dev_f)
                row = {'variant': variant, 'kind': kind, 'train_rows': int(len(x)), 'train_true_median': float(np.median(y)),
                       'dev_spearman': spearman(p, dev_true), 'dev_pred_median': float(np.median(p)),
                       'dev_true_median': float(np.median(dev_true)),
                       'dev_log_mae': float(np.mean(np.abs(np.log(p) - np.log(dev_true + 1e-6))))}
                grid.append(row); heads[(variant, kind)] = m
        # Dev selection: Spearman first (ranking drives which rows get deep noise), calibration as tiebreak.
        best = max(grid, key=lambda r: (round(r['dev_spearman'], 2), -r['dev_log_mae']))
        m = heads[(best['variant'], best['kind'])]
        ck = OUT / 'heads' / f'{qname}_confidence.pt'; ck.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'state_dict': m.state_dict(), 'kind': best['kind'], 'variant': best['variant'], 'features': CONF_FEATURES,
                    'target': 'log(arm6 MSE + 1e-6) of the top-1 draft'}, ck)
        evals = {}
        for split in ('dev', 'learned', 'unseen'):
            f, _, d = split_inputs(ret, split, sp[split], units)
            p = predict_conf(m, f); t = arm6_mse(d, sp[split]['actions'])
            evals[split] = {'spearman': spearman(p, t), 'pred_median': float(np.median(p)), 'true_median': float(np.median(t)),
                            'calibration': calibration_table(p, t)}
        # memory-side calibration of the in-sample vs cross-fitted rows under the chosen head
        for v in ('ins', 'oofE', 'oofT'):
            f, _, d = split_inputs(ret, f'eh_{v}', sp['eh'], units)
            p = predict_conf(m, f); t = arm6_mse(d, sp['eh']['actions'])
            evals['memory_' + v] = {'spearman': spearman(p, t), 'pred_median': float(np.median(p)), 'true_median': float(np.median(t))}
        out['confidence'][qname] = {'grid': grid, 'selected': best, 'checkpoint': str(ck), 'evaluation': evals}
        log('confidence_fit', qname=qname, selected=best)
        # ---- gripper ----------------------------------------------------
        ggrid = []
        ghead = {}
        dev_hist, dev_mask = sp['dev']['history'], sp['dev']['history_mask']
        dev_gt = sp['dev']['actions'][..., 6]
        base_rules = {'hold_last': hold_last(dev_hist, dev_mask),
                      'unit_stored': units[ret['dev__ids'][:, 0]][..., 6],
                      'unit_flip_rule': unit_flip_rule(dev_hist, dev_mask, units[ret['dev__ids'][:, 0]][..., 6])}
        for name, pr in base_rules.items():
            ggrid.append({'name': name, 'dev_grip_mse': float(grip_mse(pr, dev_gt).mean()),
                          **{k: v for k, v in switch_metrics(pr, dev_gt, dev_hist, dev_mask).items() if k in ('missed_switch_rate', 'false_switch_rate', 'timing_error_mean_steps')}})
        for variant in ('oofT', 'ins'):
            _, gx, _ = split_inputs(ret, f'eh_{variant}', sp['eh'], units)
            gy = sp['eh']['actions'][..., 6]
            for hidden in (64, 256):
                for steps in (2000, 6000):
                    for pw in (1.0, 2.0):
                        name = f'head_{variant}_h{hidden}_s{steps}_pw{pw}'
                        gm = fit_grip_head(gx, gy, hidden=hidden, steps=steps, seed=1, pos_weight=pw)
                        pr = predict_grip(gm, dev_g)
                        ggrid.append({'name': name, 'variant': variant, 'hidden': hidden, 'steps': steps, 'pos_weight': pw,
                                      'dev_grip_mse': float(grip_mse(pr, dev_gt).mean()),
                                      **{k: v for k, v in switch_metrics(pr, dev_gt, dev_hist, dev_mask).items() if k in ('missed_switch_rate', 'false_switch_rate', 'timing_error_mean_steps')}})
                        ghead[name] = gm
        heads_only = [r for r in ggrid if r['name'].startswith('head_')]
        gbest = min(heads_only, key=lambda r: (r['dev_grip_mse'], r['name']))
        gm = ghead[gbest['name']]
        gck = OUT / 'heads' / f'{qname}_gripper.pt'
        torch.save({'state_dict': gm.state_dict(), 'hidden': gbest['hidden'], 'config': gbest, 'input_dim': GRIP_INPUT_DIM}, gck)
        rule_best = min([r for r in ggrid if not r['name'].startswith('head_')], key=lambda r: r['dev_grip_mse'])
        out['gripper'][qname] = {'grid': ggrid, 'selected_head': gbest, 'best_rule_on_dev': rule_best, 'checkpoint': str(gck),
                                 'head_beats_rules_on_dev': bool(gbest['dev_grip_mse'] < rule_best['dev_grip_mse'])}
        log('gripper_fit', qname=qname, head=gbest, best_rule=rule_best)
    write_json(OUT / 'fit.json', out)
    return out


def fit_gripper_cv() -> dict[str, Any]:
    """Gripper head selection by memory-side task-fold CV.

    Dev (217 rows, 2 tasks) contains no gripper switch inside any 10-step chunk (hold-last is exact
    there), so it cannot select a gripper model. Selection instead uses the LIBERO-40 memory rows:
    train on 4 task folds, score the held-out fold (its retrieval is the cross-fitted oofT variant,
    i.e. unseen-like). The chosen config is refit on all memory rows. Eval is not touched.
    """
    import torch
    torch.set_num_threads(8)
    units = units_bank(); eh = load_split('eh'); dev = load_split('dev')
    folds = fold_of_task(eh['task']); gy = eh['actions'][..., 6]
    out = {}
    for qname in QUERIES:
        ret = dict(np.load(OUT / 'retrieval' / f'{qname}.npz'))
        xs = {v: split_inputs(ret, f'eh_{v}', eh, units)[1] for v in ('oofT', 'ins')}
        x_eval = xs['oofT']                                   # held-out fold is always scored on unseen-like retrieval
        top1 = units[ret['eh_oofT__ids'][:, 0]][..., 6]
        rules = {'hold_last': hold_last(eh['history'], eh['history_mask']), 'unit_stored': top1,
                 'unit_flip_rule': unit_flip_rule(eh['history'], eh['history_mask'], top1)}
        g0 = rules['hold_last'][:, 0]
        grid = []
        for name, pr in rules.items():
            sm = switch_metrics(pr, gy, eh['history'], eh['history_mask'])
            grid.append({'name': name, 'cv_grip_mse': float(grip_mse(pr, gy).mean()), **sm, 'switch_errors': switch_error_count(sm)})
        for variant in ('oofT', 'ins'):
            for hidden in (64, 256):
                for steps in (1000, 2000, 4000):
                    prob = np.zeros(gy.shape, np.float64)
                    for f in range(N_FOLDS):
                        tr = folds != f; te = folds == f
                        gm = fit_grip_head(xs[variant][tr], gy[tr], hidden=hidden, steps=steps, seed=1)
                        prob[te] = grip_close_prob(gm, x_eval[te])
                    for th in FLIP_THRESHOLDS:
                        pred = grip_from_prob(prob, g0, th)
                        sm = switch_metrics(pred, gy, eh['history'], eh['history_mask'])
                        row = {'name': f'head_{variant}_h{hidden}_s{steps}_th{th}', 'variant': variant, 'hidden': hidden, 'steps': steps,
                               'threshold': th, 'cv_grip_mse': float(grip_mse(pred, gy).mean()), **sm,
                               'switch_errors': switch_error_count(sm)}
                        grid.append(row); log('gripper_cv', qname=qname, **{k: row[k] for k in ('name', 'cv_grip_mse', 'missed_switch_rate', 'false_switch_rate', 'timing_error_mean_steps', 'switch_errors')})
        heads = [r for r in grid if r['name'].startswith('head_')]
        # A missed switch can still be caught at the next replan (5 steps later); a false switch drops/
        # releases objects. Select by switch-event errors (missed + false rows), tiebreak gripper MSE.
        best = min(heads, key=lambda r: (r['switch_errors'], r['cv_grip_mse'], r['name']))
        gm = fit_grip_head(xs[best['variant']], gy, hidden=best['hidden'], steps=best['steps'], seed=1)
        gm.flip_threshold.fill_(float(best['threshold']))
        dg = split_inputs(ret, 'dev', dev, units)[1]
        dev_pred = predict_grip(gm, dg, dev['history'], dev['history_mask'])
        torch.save({'state_dict': gm.state_dict(), 'hidden': best['hidden'], 'config': best, 'input_dim': GRIP_INPUT_DIM,
                    'selection': 'memory task-fold CV gripper MSE (Dev has no switches)'}, OUT / 'heads' / f'{qname}_gripper.pt')
        out[qname] = {'grid': grid, 'selected': best, 'rule_best': min([r for r in grid if not r['name'].startswith('head_')], key=lambda r: r['cv_grip_mse']),
                      'dev_check': {'grip_mse': float(grip_mse(dev_pred, dev['actions'][..., 6]).mean()),
                                    **switch_metrics(dev_pred, dev['actions'][..., 6], dev['history'], dev['history_mask'])}}
        log('gripper_selected', qname=qname, selected=best['name'], cv=best['cv_grip_mse'], rule=out[qname]['rule_best']['name'], rule_cv=out[qname]['rule_best']['cv_grip_mse'])
    write_json(OUT / 'fit_gripper.json', out)
    for qname in QUERIES:
        out[qname]['grid'] = sorted(out[qname]['grid'], key=lambda r: r['switch_errors'])
    return out


def load_heads(qname):
    import torch
    c = torch.load(OUT / 'heads' / f'{qname}_confidence.pt', map_location='cpu', weights_only=False)
    cm = new_conf_head(c['kind']); cm.load_state_dict(c['state_dict']); cm.eval()
    g = torch.load(OUT / 'heads' / f'{qname}_gripper.pt', map_location='cpu', weights_only=False)
    gm = new_grip_head(g['hidden']); gm.load_state_dict(g['state_dict']); gm.eval()
    return cm, gm


# ---------------------------------------------------------------------------
# 4. Expert (frozen) on Dev / Eval
# ---------------------------------------------------------------------------

def _expert(tag):
    sys.path.insert(0, str(HERE.parent / 'pi05_memory'))
    import skill_expert_refine as exp3
    import expert_ex2 as ex2
    base = exp3._module(HERE.parent / 'pi05_memory/run.py', f'ekq_{tag}')
    sched = json.loads((CL1 / 'schedule.json').read_text())
    model, _, _ = ex2._load_expert_for_eval(base, Path(sched['expert']['checkpoint']))
    return base, model, exp3, ex2


def _old_features_and_eh(ex2, torch, draft7_pi05grip, pi05, ret, prefix):
    """EX2 runtime confidence: features use π0.5 output; margin/entropy on the KQ score scale."""
    conf = torch.load(EX2_CONF, map_location='cpu', weights_only=True)
    reg = ex2._new_confidence_model(); reg.load_state_dict(conf['state_dict']); reg.eval()
    d = np.asarray(draft7_pi05grip, np.float32); r = np.asarray(pi05, np.float32)[..., :7]
    diff = np.square(d[:, :, :6] - r[:, :, :6]).mean((1, 2))
    ratio = np.linalg.norm(d[:, :, :6].reshape(len(d), -1), axis=1) / np.maximum(np.linalg.norm(r[:, :, :6].reshape(len(r), -1), axis=1), 1e-6)
    v = ret[prefix + '__vals']
    feats = np.stack([np.log(diff + 1e-6), v[:, 0] - v[:, 1], ret[prefix + '__entropy'], np.log(ratio + 1e-6)], 1).astype(np.float32)
    return ex2.predict_error(reg, feats)


def dev_grid() -> None:
    import torch
    torch.set_num_threads(8)
    dest = OUT / 'dev_grid.npz'
    if dest.exists():
        log('dev_grid_skip'); return
    base, model, exp3, ex2 = _expert('dev')
    units = units_bank(); sp = load_split('dev')
    with np.load(EX2_RET / 'dev_retrieval.npz') as z:
        if not np.array_equal(z['key'], sp['key']):
            raise AssertionError('Dev row order mismatch')
        pi05 = np.asarray(z['pi05'], np.float32)[..., :7]; ids = np.asarray(z['row_ids'], np.int64)
    with np.load(EX2_RET / 'dev_exact_observations.npz') as z:
        rows = {k: z[k] for k in z.files}
    if not np.array_equal(rows['row_keys'], sp['key']):
        raise AssertionError('Dev observation mismatch')
    obs = base.obs_batch(rows, ids); noise = exp3._seed_noise(ids + 70_000_000)
    res = {'pi05_arm6': arm6_mse(pi05, sp['actions'])}
    for qname in QUERIES:
        ret = dict(np.load(OUT / 'retrieval' / f'{qname}.npz'))
        cm, gm = load_heads(qname)
        f, g, d = split_inputs(ret, 'dev', sp, units)
        grip = predict_grip(gm, g, sp['history'], sp['history_mask'])
        for gname, gcol in (('newgrip', grip), ('pi05grip', pi05[..., 6])):
            d32 = np.zeros((len(ids), 10, 32), np.float32); d32[..., :6] = d[..., :6]; d32[..., 6] = gcol
            curve = np.zeros((len(ids), len(T0_GRID)))
            for j, t in enumerate(T0_GRID):
                out = ex2._bridge_with_row_t0(model, obs, d32, noise, np.full(len(ids), t, np.float32), batch=8)
                curve[:, j] = arm6_mse(out, sp['actions'])
            res[f'{qname}__{gname}__curve'] = curve
            log('dev_grid', qname=qname, grip=gname, mean_by_t0=[round(float(x), 4) for x in curve.mean(0)])
        res[f'{qname}__draft_arm6'] = arm6_mse(d, sp['actions'])
        res[f'{qname}__ehat_new'] = predict_conf(cm, f)
        dpi = d.copy(); dpi[..., 6] = pi05[..., 6]
        res[f'{qname}__ehat_old'] = _old_features_and_eh(ex2, torch, dpi, pi05, ret, 'dev')
    np.savez_compressed(dest, **res)


def select_map() -> dict[str, Any]:
    z = dict(np.load(OUT / 'dev_grid.npz'))
    rows = []
    cs = (0., .25, .5, .75, 1., 1.25, 1.5, 2., 2.5, 3., 4.)
    los = (.02, .05, .1, .15, .2, .3)
    his = (.4, .6, .8, .9)
    eh_all = np.concatenate([z[f'{q}__ehat_new'] for q in QUERIES])
    thresholds = [None] + [float(np.quantile(eh_all, q)) for q in (.7, .8, .9, .95)]
    for c in cs:
        for lo in los:
            for hi in his:
                if hi <= lo or (c == 0 and hi != his[0]):
                    continue
                for th in thresholds:
                    vals = []; fb = []
                    for q in QUERIES:
                        e = z[f'{q}__ehat_new']; t = select_t0(e, c, lo, hi)
                        v = interp_curve(z[f'{q}__newgrip__curve'], T0_GRID, t)
                        f = np.zeros(len(e), bool) if th is None else e > th
                        v = np.where(f, z['pi05_arm6'], v); vals.append(v); fb.append(f)
                    rows.append({'c': c, 'lo': lo, 'hi': hi, 'threshold': th, 'dev_arm6': float(np.concatenate(vals).mean()),
                                 'fallback_rate': float(np.concatenate(fb).mean()),
                                 'per_query': {q: float(v.mean()) for q, v in zip(QUERIES, vals)}})
    nofb = [r for r in rows if r['threshold'] is None]
    best = min(rows, key=lambda r: (round(r['dev_arm6'], 5), r['threshold'] is not None, r['c']))
    best_nofb = min(nofb, key=lambda r: (round(r['dev_arm6'], 5), r['c']))
    # Fallback only if it buys a clear Dev gain (>= .002) over the best no-fallback map.
    chosen = best if best['threshold'] is not None and best_nofb['dev_arm6'] - best['dev_arm6'] >= .002 else best_nofb
    ref = {}
    for q in QUERIES:
        cur = z[f'{q}__newgrip__curve']
        ref[q] = {'draft': float(z[f'{q}__draft_arm6'].mean()), 'oracle_t0_per_row': float(cur.min(1).mean()),
                  'best_constant_t0': {str(T0_GRID[int(cur.mean(0).argmin())]): float(cur.mean(0).min())},
                  'old_rule_c1': float(interp_curve(z[f'{q}__pi05grip__curve'], T0_GRID, select_t0(z[f'{q}__ehat_old'], 1., .05, .6)).mean()),
                  'chosen_map': float(interp_curve(cur, T0_GRID, select_t0(z[f'{q}__ehat_new'], chosen['c'], chosen['lo'], chosen['hi'])).mean()),
                  'ehat_new_median': float(np.median(z[f'{q}__ehat_new'])), 'ehat_old_median': float(np.median(z[f'{q}__ehat_old'])),
                  'draft_true_median': float(np.median(z[f'{q}__draft_arm6'])),
                  'curve_mean_by_t0': {str(t): float(x) for t, x in zip(T0_GRID, cur.mean(0))},
                  'curve_pi05grip_mean_by_t0': {str(t): float(x) for t, x in zip(T0_GRID, z[f'{q}__pi05grip__curve'].mean(0))}}
    sel = {'chosen': chosen, 'best_with_fallback': best, 'best_no_fallback': best_nofb, 'thresholds': thresholds,
           'pi05_dev_arm6': float(z['pi05_arm6'].mean()), 'reference': ref,
           'grid_top10': sorted(rows, key=lambda r: r['dev_arm6'])[:10],
           'rule': 'Dev 217 rows x {C1a,C1b}; MSE at mapped t0 by linear interpolation of the per-row Expert curve on T0_GRID; Eval unused'}
    write_json(OUT / 'map.json', sel)
    log('map_selected', chosen=chosen)
    return sel


def eval_offline() -> dict[str, Any]:
    import torch
    torch.set_num_threads(8)
    base, model, exp3, ex2 = _expert('eval')
    units = units_bank()
    m = json.loads((OUT / 'map.json').read_text())['chosen']
    srcs = {'learned': base.load_rows(exp3.RUN), 'unseen': exp3._load_rows(exp3.UNSEEN)}
    result = {'map': m, 'splits': {}}
    for split in ('learned', 'unseen'):
        sp = load_split(split)
        with np.load(EX2_RET / f'{split}_retrieval.npz') as z:
            if not np.array_equal(z['key'], sp['key']):
                raise AssertionError(f'{split} row order mismatch')
            pi05 = np.asarray(z['pi05'], np.float32)[..., :7]; ids = np.asarray(z['row_ids'], np.int64)
        obs = base.obs_batch(srcs[split], ids); noise = exp3._seed_noise(ids + (0 if split == 'learned' else 50_000_000))
        tgt = sp['actions']; hard = sp['hard']; ep = sp['task'] * 10_000 + sp['episode']
        hl = hold_last(sp['history'], sp['history_mask'])
        sres = {'rows': int(len(ids)), 'hard_rows': int(hard.sum()), 'pi05_arm6': float(arm6_mse(pi05, tgt).mean()),
                'pi05_active7': float((arm6_mse(pi05, tgt) + grip_mse(pi05[..., 6], tgt[..., 6])).mean()), 'queries': {}}
        preds = {'pi05': pi05}
        gripper = {'pi05': switch_metrics(pi05[..., 6], tgt[..., 6], sp['history'], sp['history_mask']) |
                   {'grip_mse': float(grip_mse(pi05[..., 6], tgt[..., 6]).mean())},
                   'hold_last': switch_metrics(hl, tgt[..., 6], sp['history'], sp['history_mask']) |
                   {'grip_mse': float(grip_mse(hl, tgt[..., 6]).mean())}}
        for qname in QUERIES:
            ret = dict(np.load(OUT / 'retrieval' / f'{qname}.npz'))
            cm, gm = load_heads(qname)
            f, g, d = split_inputs(ret, split, sp, units)
            grip = predict_grip(gm, g, sp['history'], sp['history_mask'])
            ustored = d[..., 6]; uflip = unit_flip_rule(sp['history'], sp['history_mask'], ustored)
            for gname, gp in (('head', grip), ('unit_stored', ustored), ('unit_flip_rule', uflip)):
                gripper[f'{qname}:{gname}'] = switch_metrics(gp, tgt[..., 6], sp['history'], sp['history_mask']) | {'grip_mse': float(grip_mse(gp, tgt[..., 6]).mean())}
            # old runtime: π0.5 features, c=1 clip(.05,.6), π0.5 gripper in draft and output
            dpi = np.zeros((len(ids), 10, 32), np.float32); dpi[..., :6] = d[..., :6]; dpi[..., 6] = pi05[..., 6]
            eh_old = _old_features_and_eh(ex2, torch, dpi[..., :7], pi05, ret, split)
            t_old = select_t0(eh_old, 1., .05, .6)
            old = ex2._bridge_with_row_t0(model, obs, dpi, noise, t_old, batch=8)[..., :7]; old[..., 6] = pi05[..., 6]
            # new runtime
            eh_new = predict_conf(cm, f)
            t_new = select_t0(eh_new, m['c'], m['lo'], m['hi'])
            fb = np.zeros(len(ids), bool) if m['threshold'] is None else eh_new > m['threshold']
            dn = np.zeros((len(ids), 10, 32), np.float32); dn[..., :6] = d[..., :6]; dn[..., 6] = grip
            new = ex2._bridge_with_row_t0(model, obs, dn, noise, t_new, batch=8)[..., :7]
            new_arm_only = new.copy()
            new[fb] = pi05[fb]
            new_head = new.copy(); new_head[~fb, :, 6] = grip[~fb]
            new_pi05g = new.copy(); new_pi05g[..., 6] = pi05[..., 6]
            new_hold = new.copy(); new_hold[~fb, :, 6] = hl[~fb]
            # ablation for CL: new confidence, π0.5 gripper in draft and output
            dab = dn.copy(); dab[..., 6] = pi05[..., 6]
            abl = ex2._bridge_with_row_t0(model, obs, dab, noise, t_new, batch=8)[..., :7]; abl[..., 6] = pi05[..., 6]; abl[fb] = pi05[fb]
            # diagnostic: Expert arm MSE on the t0 grid (never used for selection)
            curve = np.zeros((len(ids), len(T0_GRID)))
            for j, t in enumerate(T0_GRID):
                o = ex2._bridge_with_row_t0(model, obs, dn, noise, np.full(len(ids), t, np.float32), batch=8)
                curve[:, j] = arm6_mse(o, tgt)
            draft7 = d.copy(); draft7[..., 6] = grip
            q = {}
            for name, p in (('draft', draft7), ('old_runtime', old), ('new_runtime', new_head), ('new_conf_pi05grip', abl)):
                a = arm6_mse(p, tgt)
                q[name] = {'arm6': float(a.mean()), 'arm6_hard': float(a[hard].mean()), 'arm6_ci95': ex2._bootstrap_mean(a, ep)}
            for name, p in (('new_runtime+head_grip', new_head), ('new_runtime+pi05_grip', new_pi05g), ('new_runtime+hold_last', new_hold),
                            ('old_runtime(pi05_grip)', old)):
                q.setdefault('active7', {})[name] = float((arm6_mse(p, tgt) + grip_mse(p[..., 6], tgt[..., 6])).mean())
            for (a_name, a), (b_name, b) in ((('new_runtime', new_head), ('old_runtime', old)), (('new_runtime', new_head), ('draft', draft7)),
                                             (('new_conf_pi05grip', abl), ('old_runtime', old))):
                diff = arm6_mse(a, tgt) - arm6_mse(b, tgt)
                q.setdefault('paired', {})[f'{a_name} - {b_name}'] = {'mean': float(diff.mean()), 'ci95': ex2._bootstrap_mean(diff, ep)}
            true_err = arm6_mse(d, tgt)
            q['confidence'] = {'new': {'spearman': spearman(eh_new, true_err), 'pred_median': float(np.median(eh_new)),
                                       'true_median': float(np.median(true_err)), 'calibration': calibration_table(eh_new, true_err)},
                               'old': {'spearman': spearman(eh_old, true_err), 'pred_median': float(np.median(eh_old)),
                                       'calibration': calibration_table(eh_old, true_err)}}
            q['t0'] = {'new_quantiles': [float(x) for x in np.quantile(t_new, [.05, .5, .95])],
                       'old_quantiles': [float(x) for x in np.quantile(t_old, [.05, .5, .95])],
                       'fallback_rate': float(fb.mean())}
            q['diag_t0_curve'] = {str(t): float(x) for t, x in zip(T0_GRID, curve.mean(0))}
            q['diag_oracle_t0_per_row'] = float(curve.min(1).mean())
            q['diag_arm_without_fallback'] = float(arm6_mse(new_arm_only, tgt).mean())
            sres['queries'][qname] = q
            preds.update({f'{qname}__draft': draft7, f'{qname}__old': old, f'{qname}__new': new_head, f'{qname}__abl': abl,
                          f'{qname}__t0_new': t_new, f'{qname}__t0_old': t_old, f'{qname}__ehat_new': eh_new, f'{qname}__ehat_old': eh_old,
                          f'{qname}__curve': curve})
            log('eval_query', split=split, qname=qname, arm6={k: q[k]['arm6'] for k in ('draft', 'old_runtime', 'new_runtime', 'new_conf_pi05grip')})
        sres['gripper'] = gripper
        result['splits'][split] = sres
        np.savez_compressed(OUT / f'{split}_predictions.npz', **preds)
    write_json(OUT / 'offline.json', result)
    return result


def write_bundle() -> None:
    """Runtime bundle for serve_expert_ex2.py --ekq-bundle."""
    import torch
    m = json.loads((OUT / 'map.json').read_text())['chosen']
    for qname in QUERIES:
        c = torch.load(OUT / 'heads' / f'{qname}_confidence.pt', map_location='cpu', weights_only=False)
        g = torch.load(OUT / 'heads' / f'{qname}_gripper.pt', map_location='cpu', weights_only=False)
        torch.save({'confidence': c, 'gripper': g, 'map': m, 'topk': TOPK, 'kconf': KCONF}, OUT / 'heads' / f'{qname}_bundle.pt')
    log('bundle_written', map=m)


def _f(x, n=4):
    return '—' if x is None else f'{x:.{n}f}'


def build_report() -> str:
    """report.md + results.json from the stage outputs (closed loop included when present)."""
    fit = json.loads((OUT / 'fit.json').read_text())
    fg = json.loads((OUT / 'fit_gripper.json').read_text())
    mp = json.loads((OUT / 'map.json').read_text())
    off = json.loads((OUT / 'offline.json').read_text())
    clp = OUT / 'closed-loop/results.json'
    cl = json.loads(clp.read_text()) if clp.exists() else None
    L = ['# EKQ：π0.5 を使わない信頼度とグリッパー', '']
    L += ['## 主表：top-1 腕6軸 MSE（オフライン、LIBERO 学習済み 621 行／未学習 619 行）', '',
          '|検索|分割|下書き|Expert 旧実行時（π0.5 信頼度・c=1）|Expert 新実行時（π0.5 なし）|新 − 旧 [95% CI]|π0.5|', '|---|---|---:|---:|---:|---|---:|']
    for q in QUERIES:
        for sp in ('learned', 'unseen'):
            r = off['splits'][sp]['queries'][q]; d = r['paired']['new_runtime - old_runtime']
            L.append(f"|{q}|{sp}|{_f(r['draft']['arm6'])}|{_f(r['old_runtime']['arm6'])}|{_f(r['new_runtime']['arm6'])}|"
                     f"{d['mean']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]|{_f(off['splits'][sp]['pi05_arm6'])}|")
    L += ['', '## 7 次元 MSE（グリッパー込み、腕は新実行時）', '', '|検索|分割|新グリッパーヘッド|π0.5 グリッパー（参考）|直前保持|旧実行時（π0.5 グリッパー）|', '|---|---|---:|---:|---:|---:|']
    for q in QUERIES:
        for sp in ('learned', 'unseen'):
            a = off['splits'][sp]['queries'][q]['active7']
            L.append(f"|{q}|{sp}|{_f(a['new_runtime+head_grip'])}|{_f(a['new_runtime+pi05_grip'])}|{_f(a['new_runtime+hold_last'])}|{_f(a['old_runtime(pi05_grip)'])}|")
    L += ['', '## 信頼度の較正（ê と下書きの真の腕誤差）', '', '|検索|分割|新 Spearman|新 予測中央値|旧 Spearman|旧 予測中央値|真の中央値|', '|---|---|---:|---:|---:|---:|---:|']
    for q in QUERIES:
        ev = fit['confidence'][q]['evaluation']
        L.append(f"|{q}|Dev|{_f(ev['dev']['spearman'],3)}|{_f(ev['dev']['pred_median'])}|—|{_f(mp['reference'][q]['ehat_old_median'])}|{_f(ev['dev']['true_median'])}|")
        for sp in ('learned', 'unseen'):
            c = off['splits'][sp]['queries'][q]['confidence']
            L.append(f"|{q}|{sp}|{_f(c['new']['spearman'],3)}|{_f(c['new']['pred_median'])}|{_f(c['old']['spearman'],3)}|{_f(c['old']['pred_median'])}|{_f(c['new']['true_median'])}|")
    L += ['', '分位ビン（未学習、新 ê の5分位ごとの予測中央値 → 真の中央値）', '']
    for q in QUERIES:
        cal = off['splits']['unseen']['queries'][q]['confidence']['new']['calibration']
        L.append(f"- {q}: " + '、'.join(f"{_f(b['pred_median'])}→{_f(b['true_median'])}" for b in cal))
    L += ['', '## グリッパー（スイッチは直前実行値からの最初の符号変化、10 step チャンク）', '',
          '|分割|方式|grip MSE|切替行|見逃し率|誤切替率|タイミング誤差（平均 step）|±1 step 以内|', '|---|---|---:|---:|---:|---:|---:|---:|']
    for sp in ('learned', 'unseen'):
        for name, g in off['splits'][sp]['gripper'].items():
            L.append(f"|{sp}|{name}|{_f(g['grip_mse'])}|{g['gt_switch_rows']}|{_f(g['missed_switch_rate'],3)}|{_f(g['false_switch_rate'],3)}|"
                     f"{_f(g['timing_error_mean_steps'],2)}|{_f(g['within1_step_rate'],3)}|")
    L += ['', '## t0 写像（Dev で選択）', '', f"選択：t0 = clip({mp['chosen']['c']}·√ê, {mp['chosen']['lo']}, {mp['chosen']['hi']})、"
          f"π0.5 退避 {'なし' if mp['chosen']['threshold'] is None else mp['chosen']['threshold']}（Dev 腕 MSE {mp['chosen']['dev_arm6']:.4f}、Dev の π0.5 {mp['pi05_dev_arm6']:.4f}）", '',
          '|検索|Dev 下書き|旧規則|選んだ写像|最良の定数 t0|行ごと最良 t0（oracle）|', '|---|---:|---:|---:|---|---:|']
    for q in QUERIES:
        r = mp['reference'][q]; (bt, bv), = r['best_constant_t0'].items()
        L.append(f"|{q}|{_f(r['draft'])}|{_f(r['old_rule_c1'])}|{_f(r['chosen_map'])}|{bt}: {_f(bv)}|{_f(r['oracle_t0_per_row'])}|")
    L += ['', '未学習の t0 別 Expert 腕 MSE（診断、選択には未使用）', '']
    for q in QUERIES:
        r = off['splits']['unseen']['queries'][q]
        L.append(f"- {q}: " + '、'.join(f"{t}:{v:.4f}" for t, v in r['diag_t0_curve'].items()) + f"；行ごと最良 {r['diag_oracle_t0_per_row']:.4f}；"
                 f"t0 中央値 新 {r['t0']['new_quantiles'][1]:.3f}／旧 {r['t0']['old_quantiles'][1]:.3f}；退避率 {r['t0']['fallback_rate']:.3f}")
    if cl:
        L += ['', '## 閉ループ（LIBERO-90 Eval 10×10、CL1 と同じ seed・初期状態）', '', '|条件|成功/完了|', '|---|---:|']
        for n, v in cl['conditions'].items():
            L.append(f"|{n}（{v['label']}）|{v['successes']}/{v['completed']}|")
        L += ['', '|比較 B − A|対|B だけ成功|A だけ成功|p|', '|---|---:|---:|---:|---:|']
        for k, v in cl['mcnemar'].items():
            L.append(f"|{k}|{v['pairs']}|{v['b_only']}|{v['a_only']}|{v['p']:.3f}|")
        tids = list(next(iter(cl['conditions'].values()))['tasks'].keys())
        L += ['', '|task|' + '|'.join(cl['conditions']) + '|', '|---|' + '---:|' * len(cl['conditions'])]
        for t in tids:
            L.append(f'|{t}|' + '|'.join(str(v['tasks'].get(t, '')) for v in cl['conditions'].values()) + '|')
        L += ['', '|条件|問い合わせ|t0 中央値 [5%, 95%]|ê 中央値|退避率|']
        L += ['|---|---:|---|---:|---:|']
        for n, d in cl['diagnostics'].items():
            t = d['t0_quantiles']; e = d['ehat_quantiles']
            L.append(f"|{n}|{d['queries']}|{t['0.5']:.3f} [{t['0.05']:.3f}, {t['0.95']:.3f}]|{e['0.5']:.4f}|{d['fallback_rate']:.3f}|")
    L += ['', '## 設定', '',
          '- Expert：EX2 production-05 A step 2500（CL1 と同じ、重みは不変）。下書きは KQ-final C1a（K-x1 × E-H 注意）／C1b（K-x1 × 履歴＋状態 MLP）の top-1。',
          '- 信頼度：特徴 ' + ', '.join(CONF_FEATURES) + '。目的変数 log 腕6軸 MSE。学習はメモリ側 eh 12,766 行のみ。'
          '検索は ins（本番クエリ・自 episode 除外）／oofE（タスク 5 分割の交差学習クエリ・自 episode 除外）／oofT（同・自タスク全除外）。線形/MLP × 学習行の組を Dev 217 行の Spearman で選択。',
          f"  - 選択：C1a {fit['confidence']['C1a']['selected']['variant']}/{fit['confidence']['C1a']['selected']['kind']}、C1b {fit['confidence']['C1b']['selected']['variant']}/{fit['confidence']['C1b']['selected']['kind']}",
          '- グリッパー：入力 = 実行済み行動10×7・有効フラグ・状態8・top-1 下書き腕・top-1 単位の保存グリッパー・top-5 単位の切替（スコア重み）。出力 10 step の閉確率。直前値から反転するのは反対側の確率が閾値を超えた step だけ。'
          'Dev 217 行にはチャンク内の切替が 1 つもない（直前保持が誤差 0）ため、選択はメモリ側のタスク 5 分割 CV（保持タスクは oofT 検索）で、切替誤り数（見逃し＋誤切替）最小。',
          f"  - 選択：C1a {fg['C1a']['selected']['name']}（CV grip MSE {fg['C1a']['selected']['cv_grip_mse']:.4f}、直前保持 {fg['C1a']['rule_best']['cv_grip_mse']:.4f}）、C1b {fg['C1b']['selected']['name']}（{fg['C1b']['selected']['cv_grip_mse']:.4f}）",
          '- t0：Dev 217 行 × C1a/C1b で、各行の Expert 出力を t0 格子 ' + str(T0_GRID) + ' で計算し、写像の t0 で線形補間した腕 MSE の平均で (c, 下限, 上限, π0.5 退避閾値) を選択。退避は Dev で .002 以上良くなるときだけ採用。',
          '- Eval（learned 621 / unseen 619）は各列 1 回だけ採点。π0.5 は参考列と退避候補にのみ使用。',
          '- コード：unseen_libero/expert_kq_runtime.py、run_ekq_closed_loop.py、serve_expert_ex2.py（--ekq-bundle / --ekq-gripper を追加、既存モードは不変）、tests/unseen_libero/test_expert_kq_runtime.py。']
    text = '\n'.join(L) + '\n'
    write_json(OUT / 'results.json', {'fit_confidence': fit['confidence'], 'fit_gripper': fg, 'map': mp, 'offline': off, 'closed_loop': cl})
    return text


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command', choices=('crossfit', 'retrieve', 'fit', 'fit-gripper', 'dev-grid', 'select', 'eval', 'bundle', 'report'))
    p.add_argument('--query', default='C1a'); p.add_argument('--folds', default='0,1,2,3,4')
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    log('start', command=a.command, query=a.query if a.command == 'crossfit' else None)
    if a.command == 'crossfit':
        crossfit(a.query, [int(x) for x in a.folds.split(',')])
    elif a.command == 'retrieve':
        retrieve()
    elif a.command == 'fit':
        fit_heads()
    elif a.command == 'fit-gripper':
        fit_gripper_cv()
    elif a.command == 'dev-grid':
        dev_grid()
    elif a.command == 'select':
        select_map()
    elif a.command == 'eval':
        eval_offline()
    elif a.command == 'bundle':
        write_bundle()
    elif a.command == 'report':
        t = build_report(); (OUT / 'report_tables.md').write_text(t); print(t)
    log('end', command=a.command)


if __name__ == '__main__':
    main()
