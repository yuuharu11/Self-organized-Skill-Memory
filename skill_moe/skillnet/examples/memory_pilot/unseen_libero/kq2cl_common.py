#!/usr/bin/env python3
"""KQ2-CL shared helpers: paths, logging, the error->t0 map, head features, metrics.

KQ2-CL = closed-loop test of the KQ2 queries.
  (iii) GRU(h of the 10-step state+action history) -> h cross-attends 34 pi0.5 VLM tokens (KQ2 stage 2)
  (i)   the same GRU h only (no VLM)
over fixed K-x1 motion keys and the raw 4,516 LIBERO-40 units (no clusters).

Everything here is torch/numpy only (no JAX) so it can be imported by every stage.
"""
from __future__ import annotations

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

ROOT = STATE_DIR / 'outputs/memory-pilot/unseen-libero'
OUT = ROOT / 'kq2-cl-01'
KQ2 = ROOT / 'kq2-01'
KEYS = KQ2 / 'stage1/K-x1_keys.npy'
QUERY_CK = {'iii': KQ2 / 'stage2/rows/iii_G1X2F_s1.pt', 'i': KQ2 / 'stage2/rows/i_noV_G1N_s1.pt'}
QUERY_LABEL = {'iii': '(iii) GRU h x VLM cross-attention (KQ2 iii_G1X2F_s1)',
               'i': '(i) GRU h only, no VLM (KQ2 i_noV_G1N_s1)'}
SPLITS = ('eh', 'learned', 'unseen20', 'dev2')
EVAL = ('dev2', 'learned', 'unseen20')
EX3_CK = ROOT / 'expert-ex3-01/train-G2/checkpoints/004000'
EX3_SCALE = .5   # EX3 Dev choice (c for the true-error rule)
PI05 = STATE_DIR / 'checkpoints/openpi-assets/checkpoints/pi05_libero'
RUN40 = STATE_DIR / 'outputs/memory-pilot/pi05-memory40-20261104/run-01'
T0_GRID = (.02, .05, .1, .15, .2, .25, .3, .35, .4, .5, .6, .7, .8, .9)
DEFAULT_GRIP = -1.0


# ---------------------------------------------------------------------------
# io
# ---------------------------------------------------------------------------

def utc() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def log(event: str, **kw) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rec = {'utc': utc(), 'event': event, **kw}
    line = json.dumps(rec, ensure_ascii=False, default=float)
    with (OUT / 'progress.log').open('a') as f:
        f.write(line + '\n')
    print(line, flush=True)


def write_json(path: Path, value: Any) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=float) + '\n'); tmp.replace(path)


def read_json(path: Path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def arm6(pred, target) -> np.ndarray:
    p = np.asarray(pred, np.float64); t = np.asarray(target, np.float64)
    return np.square(p[..., :6] - t[..., :6]).sum((-1, -2)) / 70.


def quantiles(x, qs=(.05, .1, .25, .5, .75, .9, .95)) -> dict[str, float]:
    x = np.asarray(x, np.float64)
    if x.size == 0:
        return {}
    return {str(q): float(np.quantile(x, q)) for q in qs} | {'mean': float(x.mean())}


def bootstrap_ci(values, groups, *, seed=4101, reps=2000) -> list[float]:
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


def hold_last_gripper(last, valid, default: float = DEFAULT_GRIP) -> np.ndarray:
    v = np.asarray(last, np.float64); ok = np.asarray(valid, bool)
    s = np.where(ok & (v != 0), np.sign(v), default)
    return np.repeat(s[:, None], 10, 1).astype(np.float32)


def switch_metrics(pred_grip, target_grip, last, valid) -> dict[str, Any]:
    """Gripper switch timing vs the last executed gripper sign (rows without history skipped).
    Same definition as EX3/EKQ (first sign change within the 10-step chunk)."""
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
            'false_switch_rows': int((ok & ~gt & pr).sum()), 'missed_rows': int((gt & ~pr).sum()),
            'timing_error_mean_steps': float(err.mean()) if len(err) else None,
            'within1_step_rate': float((err <= 1).sum() / max(nsw, 1)) if nsw else None,
            'sign_step_error_rate': float((p[ok] != t[ok]).mean()) if ok.any() else None}


# ---------------------------------------------------------------------------
# error -> t0 map (EX3 form): t0 = clip(c * sqrt(err), lo, hi)
# training uses c=2, [.05, .6] on the TRUE draft error (x U(.9,1.1)^2 jitter), as EX3.
# ---------------------------------------------------------------------------

class T0Map:
    def __init__(self, c: float = 2.0, lo: float = .05, hi: float = .6):
        self.c, self.lo, self.hi = float(c), float(lo), float(hi)

    def __call__(self, err) -> np.ndarray:
        e = np.maximum(np.asarray(err, np.float64), 0)
        return np.clip(self.c * np.sqrt(e), self.lo, self.hi).astype(np.float32)

    def as_dict(self):
        return {'c': self.c, 'lo': self.lo, 'hi': self.hi}


TRAIN_T0MAP = T0Map(2.0, .05, .6)


def interp_curve(curve, grid, t0) -> np.ndarray:
    g = np.asarray(grid, np.float64); c = np.asarray(curve, np.float64); t = np.clip(np.asarray(t0, np.float64), g[0], g[-1])
    j = np.clip(np.searchsorted(g, t, side='right') - 1, 0, len(g) - 2)
    a = (t - g[j]) / (g[j + 1] - g[j]); r = np.arange(len(t))
    return (1 - a) * c[r, j] + a * c[r, j + 1]


def true_rank(err_row_major: np.ndarray, picked: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    m = np.asarray(err_row_major, np.float64); e = np.asarray(allowed, bool); p = np.asarray(picked, np.int64)
    pm = m[np.arange(len(p)), p]
    return 1 + ((m < pm[:, None]) & e).sum(1)


# ---------------------------------------------------------------------------
# head features: only quantities from the query q, the retrieved keys and the scores
# ---------------------------------------------------------------------------

SCALAR_NAMES = ('top1_logit', 'gap2', 'gap5', 'gap10', 'gap50', 'entropy_norm', 'log_top1_prob', 'top5_mass',
                'dot1', 'dot_gap2', 'top5_key_pairdot', 'k1_vs_top5mean')


def head_features(q: np.ndarray, top_ids: np.ndarray, top_logits: np.ndarray, entropy: np.ndarray,
                  top1_prob: np.ndarray, top5_mass: np.ndarray, keys: np.ndarray, scale: float) -> dict[str, np.ndarray]:
    """q [N,256] (normalized), top_ids/top_logits [N,>=50] sorted desc (allowed only),
    entropy (normalized by log #allowed), top1 prob, top-5 softmax mass, keys [M,256], logit scale."""
    q = np.asarray(q, np.float32); ids = np.asarray(top_ids, np.int64); lg = np.asarray(top_logits, np.float64)
    k1 = keys[ids[:, 0]]
    k5 = keys[ids[:, :5]]                                   # [N,5,256]
    g = np.einsum('nid,njd->nij', k5, k5)
    iu = np.triu_indices(5, 1)
    pair = g[:, iu[0], iu[1]].mean(1)
    m5 = k5[:, 1:].mean(1)
    k1m = (k1 * m5).sum(1) / np.maximum(np.linalg.norm(m5, axis=1), 1e-6)
    sc = np.stack([lg[:, 0], lg[:, 0] - lg[:, 1], lg[:, 0] - lg[:, 4], lg[:, 0] - lg[:, 9], lg[:, 0] - lg[:, 49],
                   np.asarray(entropy, np.float64), np.log(np.maximum(np.asarray(top1_prob, np.float64), 1e-12)),
                   np.asarray(top5_mass, np.float64), lg[:, 0] / scale, (lg[:, 0] - lg[:, 1]) / scale, pair, k1m], 1)
    return {'scalar': sc.astype(np.float32), 'q': q, 'k1': k1.astype(np.float32)}


def summarize_scores(logits: np.ndarray, allowed: np.ndarray, k: int = 50) -> dict[str, np.ndarray]:
    """Per-row top-k ids/logits (allowed only), normalized entropy, top-1 prob, top-5 mass."""
    s = np.where(np.asarray(allowed, bool), np.asarray(logits, np.float64), -np.inf)
    ids = np.argsort(-s, axis=1, kind='stable')[:, :k]
    top = np.take_along_axis(s, ids, 1)
    z = s - s.max(1, keepdims=True)
    p = np.exp(z); p /= p.sum(1, keepdims=True)
    ent = -(np.where(p > 0, p * np.log(np.maximum(p, 1e-300)), 0)).sum(1) / np.log(np.maximum(np.isfinite(s).sum(1), 2))
    pt = np.take_along_axis(p, ids[:, :5], 1)
    return {'ids': ids.astype(np.int32), 'logits': top.astype(np.float32), 'entropy': ent.astype(np.float32),
            'top1_prob': pt[:, 0].astype(np.float32), 'top5_mass': pt.sum(1).astype(np.float32)}


# ---------------------------------------------------------------------------
# small torch heads
# ---------------------------------------------------------------------------

def make_head(kind: str, out_dim: int, extra: int = 0):
    """kind: 'scalar' (12 score features) or 'full' (scalar + q + k1). extra: additional inputs (gripper)."""
    import torch
    from torch import nn
    d_in = len(SCALAR_NAMES) + extra + (512 if kind == 'full' else 0)

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.kind = kind
            self.register_buffer('mu', torch.zeros(d_in)); self.register_buffer('sd', torch.ones(d_in))
            self.net = nn.Sequential(nn.Linear(d_in, 256), nn.GELU(), nn.Dropout(.1), nn.Linear(256, 64), nn.GELU(),
                                     nn.Linear(64, out_dim))

        def forward(self, x):
            return self.net((x - self.mu) / self.sd)
    return Head()


def head_input(feats: dict[str, np.ndarray], kind: str, extra: np.ndarray | None = None) -> np.ndarray:
    parts = [feats['scalar']]
    if extra is not None:
        parts.append(np.asarray(extra, np.float32))
    if kind == 'full':
        parts += [feats['q'], feats['k1']]
    return np.concatenate(parts, 1).astype(np.float32)


def grip_extra(last_cmd: np.ndarray, has_hist: np.ndarray, grip_state: np.ndarray) -> np.ndarray:
    """Execution-time gripper inputs: last executed gripper command (sign, -1 at start), has-history
    flag, current gripper qpos (normalized state dims 6,7)."""
    lc = np.where(np.asarray(has_hist, bool), np.sign(np.asarray(last_cmd, np.float32)), DEFAULT_GRIP)
    lc = np.where(lc == 0, DEFAULT_GRIP, lc)
    return np.concatenate([lc[:, None], np.asarray(has_hist, np.float32)[:, None], np.asarray(grip_state, np.float32)], 1)


def decode_gripper(prob_close: np.ndarray, last_sign: np.ndarray, th: float) -> np.ndarray:
    """10-step +-1 gripper (LIBERO normalized: +1 close, -1 open). Start from the last executed sign;
    switch once, at the first step whose opposite-side probability exceeds th."""
    p = np.asarray(prob_close, np.float64); g0 = np.where(np.asarray(last_sign) >= 0, 1., -1.)
    opp = np.where(g0[:, None] > 0, 1 - p, p)
    hit = opp > th
    k = np.where(hit.any(1), hit.argmax(1), 10)
    steps = np.arange(10)[None]
    return np.where(steps >= k[:, None], -g0[:, None], g0[:, None]).astype(np.float32)


# ---------------------------------------------------------------------------
# query models (KQ2 QueryNet) and an inline split for single closed-loop queries
# ---------------------------------------------------------------------------

def load_query(name_or_path, device):
    import torch
    import kq2_run as kr
    p = QUERY_CK[name_or_path] if name_or_path in QUERY_CK else Path(name_or_path)
    blob = torch.load(p, map_location=device, weights_only=False)
    cfg = blob['cfg']
    m = kr.QueryNet(cfg['enc'], cfg['gru_layers'], cfg['xattn'], cfg['ffn'], cfg['history'],
                    dct=bool(cfg.get('dct', False))).to(device)
    m.load_state_dict(blob['query']); m.eval()
    return m, cfg


class InlineSplit:
    """Duck-typed kq2_run.GPUSplit for precomputed history tokens (closed loop / parity)."""
    def __init__(self, x, valid, vlm=None):
        self.x, self.valid, self.vlm = x, valid, vlm
        self.n = len(x)

    def window(self, ix, L: int = 10):
        return self.x[ix], self.valid[ix]


def window_tokens(seq_a_norm: np.ndarray, seq_s_norm: np.ndarray, t: int, L: int = 10):
    """Tokens for tau = t-L+1..t: [a_{tau-1} (7), s_tau (8), a-valid]; valid = tau >= 0.
    seq_s_norm[tau] for tau=0..t, seq_a_norm[tau] for tau=0..t-1 (executed). Same as GPUSplit.window."""
    x = np.zeros((L, 16), np.float32); valid = np.zeros(L, bool)
    for j in range(L):
        tau = t - L + 1 + j
        if tau < 0:
            continue
        valid[j] = True
        x[j, 7:15] = seq_s_norm[tau]
        if tau >= 1:
            x[j, :7] = seq_a_norm[tau - 1]; x[j, 15] = 1.
    return x, valid


def oracle_ids(split: str) -> np.ndarray:
    """Best stored unit by arm6 (own (task, episode) masked, KQ2 rule) for a KQ2 data split."""
    import key_shared_latent as k2
    D = KQ2 / 'data' / split
    tgt = np.load(D / 'actions.npy'); t = np.load(D / 'task.npy').astype(np.int64); e = np.load(D / 'episode.npy').astype(np.int64)
    kd = ROOT / 'key-motion-01/final/data'
    win = np.load(kd / 'bank_actions.npy').astype(np.float32)
    bt = np.load(kd / 'bank__task.npy').astype(np.int64); be = np.load(kd / 'bank__episode.npy').astype(np.int64)
    err = k2._arm6_candidate_mse(tgt, win)
    err[(t[:, None] == bt[None]) & (e[:, None] == be[None])] = np.inf
    return err.argmin(1)


def load_curves(split: str) -> dict:
    """Merge curves/<split>__<tag>.npz (Expert t0-grid curves, raw gripper, draft errors)."""
    out = {}
    for p in sorted((OUT / 'curves').glob(f'{split}__*.npz')):
        if p.name.endswith('.tmp.npz'):
            continue
        with np.load(p) as z:
            out.update({k: z[k] for k in z.files if k != 'checkpoint'})
    return out
