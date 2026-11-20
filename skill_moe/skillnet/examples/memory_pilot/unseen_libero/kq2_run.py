#!/usr/bin/env python3
"""KQ2: motion-key comparison (stage 1) and GRU-h x VLM cross-attention query (stage 2).

Rules (same as KQ-final / VQ-A):
  * training data = LIBERO-40 only (memory units for keys; stride-5 memory frames `eh` for queries);
  * selection (key tau / epoch / architecture, query epoch / setting) on Dev v2 20 Hz only;
  * learned 621 / unseen20 619 are scored once per trained row;
  * score = top-1 unit's stored arm-6 window vs the true next 10 steps (k2._arm6_candidate_mse),
    own (task, episode) excluded; the gripper is not scored;
  * no skill clusters: every raw memory unit (4,516) is a candidate;
  * no observation-side augmentation.

Stage 1 (keys):  python kq2_run.py keys --plan plan_keys.json [--names ...]
                 python kq2_run.py keyq --plan plan_keyq.json [--names ...]   (C1b query per key, metric b)
Stage 2 (query): python kq2_run.py query --plan plan_query.json [--names ...]
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
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import key_shared_latent as k2  # noqa: E402  (read-only)
import key_motion as km  # noqa: E402  (read-only)
import kq2_data as kd  # noqa: E402

ROOT = kd.ROOT
OUT = kd.OUT
KF_DATA = kd.KF_DATA
KF_ROWS = ROOT / 'key-motion-01/final/rows'
X1_MK1B = ROOT / 'key-motion-01/arch/x1_absolute_dist_tau0.01.pt'
C1B = [KF_ROWS / f'H_x1_mk1b_cteff_tau0.01_s{i}.pt' for i in (1, 2, 3)]
EVAL = ('dev2', 'learned', 'unseen20')

log = kd.log
write_json = kd.write_json


# ----------------------------------------------------------------------------
# data context
# ----------------------------------------------------------------------------

def load_bank() -> dict[str, np.ndarray]:
    b = {k: np.load(KF_DATA / f'bank__{k}.npy') for k in
         ('task', 'episode', 'natural_actions', 'length', 'source', 'value_actions10')}
    b['windows'] = np.load(KF_DATA / 'bank_actions.npy').astype(np.float32)
    b['task'] = b['task'].astype(np.int64); b['episode'] = b['episode'].astype(np.int64)
    return b


def subsets_for(name: str, sp: dict) -> dict[str, np.ndarray]:
    """VQ-A subsets: first_unit / post_switch / dir_change (+ 'middle' = none of them)."""
    import dev_v2
    if name == 'dev2':
        d = dev_v2.load_dev_v2_20()
        if not np.array_equal(d['key'], sp['key']):
            raise AssertionError('dev2 order')
        m = {k[4:]: d[k] for k in d if k.startswith('sub_')}
    else:
        m = dev_v2.eval_subsets(sp)
    m = {k: np.asarray(v, bool) for k, v in m.items()}
    m['middle'] = ~(m['first_unit'] | m['post_switch'] | m['dir_change'])
    return m


class Ctx:
    def __init__(self, device: str, splits=('eh',) + EVAL, vlm: bool = False, out: Path = OUT):
        self.device = torch.device(device)
        self.out = out
        self.bank = load_bank()
        self.bt, self.be = self.bank['task'], self.bank['episode']
        self.windows = self.bank['windows']
        self.splits = {}
        for n in splits:
            d = out / 'data' / n
            sp = {p.stem: np.load(p) for p in d.glob('*.npy') if p.stem != 'vlm'}
            sp['task'] = sp['task'].astype(np.int64); sp['episode'] = sp['episode'].astype(np.int64)
            sp['errors'] = k2._arm6_candidate_mse(sp['actions'], self.windows)
            sp['allowed'] = ~((sp['task'][:, None] == self.bt[None]) & (sp['episode'][:, None] == self.be[None]))
            if vlm:
                sp['vlm'] = np.load(d / 'vlm.npy', mmap_mode='r')
            if n != 'eh':
                sp['subsets'] = subsets_for(n, sp)
            sp['epkey'] = sp['task'] * 10_000_000 + sp['episode']
            self.splits[n] = sp
        if 'eh' in self.splits:
            import key_final as kf
            self.splits['eh']['candidate_id'] = kf.candidate_ids_for(self.splits['eh'], self.bank)


def top1(scores: np.ndarray, sp: dict) -> np.ndarray:
    s = np.where(sp['allowed'], scores, -np.inf)
    return sp['errors'][np.arange(len(s)), s.argmax(1)]


def block(per: np.ndarray, sp: dict) -> dict[str, Any]:
    out = {'all': float(per.mean()), 'per_query_top1': per.astype(np.float64).tolist()}
    if 'subsets' in sp:
        out['subsets'] = {k: {'rows': int(m.sum()), 'top1': float(per[m].mean()) if m.any() else None}
                          for k, m in sp['subsets'].items()}
    return out


# ----------------------------------------------------------------------------
# stage 1: key encoders
# ----------------------------------------------------------------------------

class FFN(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x):
        return x + self.net(x)


class SABlock(nn.Module):
    def __init__(self, dim: int, heads: int, ffn: bool):
        super().__init__()
        self.n = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ffn = FFN(dim) if ffn else None

    def forward(self, x, pad):
        n = self.n(x)
        a, _ = self.attn(n, n, n, key_padding_mask=pad, need_weights=False)
        x = x + a
        return self.ffn(x) if self.ffn is not None else x


class XBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ffn: bool):
        super().__init__()
        self.qn, self.kn = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ffn = FFN(dim) if ffn else None

    def forward(self, q, mem, pad, need_weights=False):
        k = self.kn(mem)
        a, w = self.attn(self.qn(q), k, k, key_padding_mask=pad, need_weights=need_weights, average_attn_weights=True)
        q = q + a
        return (self.ffn(q) if self.ffn is not None else q), w


class SAKey(nn.Module):
    """K-SA: arm actions [B,T,6] (masked) -> Linear + absolute sinusoid -> self-attention x depth
    -> learnable latent (Q) cross-attends the self-attention output (K,V) x cross -> LN -> L2."""
    def __init__(self, sa_depth: int = 1, cross: int = 1, ffn: bool = True, dim: int = 256, heads: int = 8):
        super().__init__()
        self.dim = dim
        self.proj = nn.Linear(6, dim)
        self.sa = nn.ModuleList([SABlock(dim, heads, ffn) for _ in range(sa_depth)])
        self.latent = nn.Parameter(torch.randn(1, 1, dim) * .02)
        self.cross = nn.ModuleList([XBlock(dim, heads, ffn) for _ in range(cross)])
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, actions, valid, return_details=False):
        valid = valid.bool()
        x = torch.where(valid[..., None], actions.float(), torch.zeros_like(actions.float()))
        m = self.proj(x) + km._sinusoidal_positions(x.shape[1], self.dim, x.device, x.dtype)[None]
        pad = ~valid
        for b in self.sa:
            m = b(m, pad)
        q = self.latent.expand(len(x), -1, -1)
        for b in self.cross:
            q, _ = b(q, m, pad)
        return F.normalize(self.out_norm(q), dim=-1)


def make_key(cfg: dict) -> nn.Module:
    if cfg['arch'] == 'x1':
        return km.ArchMotionEncoder('x1', 'absolute')
    return SAKey(int(cfg['sa']), int(cfg['cross']), bool(cfg['ffn']))


def nparams(m: nn.Module) -> int:
    return int(sum(p.numel() for p in m.parameters()))


def bank_tensors(bank, device):
    a = torch.as_tensor(bank['natural_actions'][..., :6], device=device, dtype=torch.float32)
    ln = torch.as_tensor(bank['length'], device=device, dtype=torch.long)
    return a, torch.arange(a.shape[1], device=device)[None] < ln[:, None]


@torch.no_grad()
def encode_bank(model, bank, device) -> np.ndarray:
    model.eval()
    a, v = bank_tensors(bank, device)
    return torch.cat([model(a[s:s + 512], v[s:s + 512])[:, 0] for s in range(0, len(a), 512)]).float().cpu().numpy()


@torch.no_grad()
def key_metric_a(model, ctx: Ctx, keys: np.ndarray) -> dict[str, Any]:
    """MK1b metric: encode the TRUE next 10 arm steps with the key encoder, take the nearest key."""
    model.eval()
    res = {}
    for n in EVAL:
        sp = ctx.splits[n]
        a = torch.as_tensor(sp['actions'][..., :6], device=ctx.device, dtype=torch.float32)
        q = model(a, torch.ones(a.shape[:2], device=ctx.device, dtype=torch.bool))[:, 0].float().cpu().numpy()
        res[n] = float(top1(q @ keys.T, sp).mean())
    return res


def dct3(x: np.ndarray) -> np.ndarray:
    from scipy.fft import dct
    return dct(np.asarray(x, np.float64)[..., :6], type=2, norm='ortho', axis=1)[:, :3].reshape(len(x), -1).astype(np.float32)


def dct_metric_a(ctx: Ctx) -> dict[str, float]:
    kc = dct3(ctx.windows)
    out = {}
    for n in EVAL:
        sp = ctx.splits[n]
        qc = dct3(sp['actions'])
        d = (qc ** 2).sum(1)[:, None] + (kc ** 2).sum(1)[None] - 2 * qc @ kc.T
        out[n] = float(top1(-d, sp).mean())
    return out


def key_diagnostics(keys: np.ndarray, bank) -> dict[str, float]:
    from scipy.stats import spearmanr
    rng = np.random.default_rng(0)
    ix = rng.choice(len(keys), 1500, replace=False)
    w = bank['windows'][ix][..., :6].reshape(len(ix), -1)
    err = ((w[:, None] - w[None]) ** 2).mean(-1); cos = keys[ix] @ keys[ix].T
    tri = np.triu_indices(len(ix), 1)
    c = cos[tri]
    return {'spearman_vs_neg_mse': float(spearmanr(c, -err[tri]).statistic), 'cos_mean': float(c.mean()),
            'cos_p05': float(np.percentile(c, 5)), 'cos_p95': float(np.percentile(c, 95))}


def train_key(name: str, cfg0: dict, ctx: Ctx, save: Path) -> dict:
    """MK1b distance distillation (key_motion._fit_arch, objective 'dist'), run much longer.

    batch 128 memory units as queries (target = the unit's own next 10 steps), candidates = batch
    + 256 random units (own episode masked), score = cos * 20, soft target softmax(-armMSE / tau)."""
    cfg = {'arch': 'sa', 'sa': 1, 'cross': 1, 'ffn': True, 'tau': .01, 'seed': 1, 'epochs': 200,
           'ckpt_epochs': [4, 10, 25, 50, 100, 150, 200], 'lr': 1e-4, 'batch': 128, **cfg0}
    dev = ctx.device; seed = int(cfg['seed'])
    torch.manual_seed(seed); rng = np.random.default_rng(seed + 333)
    model = make_key(cfg).to(dev)
    bank = ctx.bank
    a, v = bank_tensors(bank, dev)
    n = len(bank['length'])
    # errors between each unit's own next 10 steps and every candidate window (fixed teacher)
    E = torch.as_tensor(k2._arm6_candidate_mse(bank['value_actions10'], ctx.windows), device=dev)
    bt = torch.as_tensor(ctx.bt, device=dev); be = torch.as_tensor(ctx.be, device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg['lr']), weight_decay=1e-4)
    order = np.arange(n); hist = []; ck = {}
    t0 = time.time()
    for ep in range(1, int(cfg['epochs']) + 1):
        rng.shuffle(order); model.train(); losses = []
        for s in range(0, n, int(cfg['batch'])):
            ids = order[s:s + int(cfg['batch'])]
            if len(ids) < 2:
                continue
            rnd = rng.choice(n, size=min(256, n), replace=False)
            cids = np.unique(np.r_[ids, rnd])
            qi = torch.as_tensor(ids, device=dev); ci = torch.as_tensor(cids, device=dev)
            q = model(a[qi], v[qi])[:, 0]; k = model(a[ci], v[ci])[:, 0]
            sc = (q @ k.T) * 20.
            allowed = ~((bt[qi][:, None] == bt[ci][None]) & (be[qi][:, None] == be[ci][None]))
            loss = k2.soft_listwise_loss(sc, E[qi][:, ci], allowed, float(cfg['tau']))
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.); opt.step()
            losses.append(float(loss.detach()))
        rec = {'epoch': ep, 'loss': float(np.mean(losses)), 'seconds': time.time() - t0}
        if ep in cfg['ckpt_epochs']:
            keys = encode_bank(model, bank, dev)
            rec['a'] = key_metric_a(model, ctx, keys)
            p = save / f'{name}_ep{ep}.pt'
            torch.save(model.state_dict(), p); np.save(save / f'{name}_ep{ep}_keys.npy', keys)
            ck[ep] = {'checkpoint': str(p), 'keys': str(save / f'{name}_ep{ep}_keys.npy'), 'a': rec['a']}
            log('key_ckpt', ctx.out, name=name, epoch=ep, loss=rec['loss'], a=rec['a'])
        hist.append(rec)
    best = min(ck, key=lambda e: ck[e]['a']['dev2'])
    model.load_state_dict(torch.load(ck[best]['checkpoint'], map_location=dev, weights_only=True))
    keys = np.load(ck[best]['keys'])
    row = {'name': name, 'cfg': cfg, 'params': nparams(model), 'history': hist, 'checkpoints': ck,
           'selected_epoch': best, 'a': ck[best]['a'], 'keys': ck[best]['keys'], 'checkpoint': ck[best]['checkpoint'],
           'diagnostics': key_diagnostics(keys, bank), 'seconds': time.time() - t0}
    write_json(save / f'{name}.json', row)
    return row


def reference_keys(ctx: Ctx, save: Path) -> dict:
    """K-x1 (existing MK1b checkpoint, loaded as is) and DCT3 reference rows for metric (a)."""
    p = save / 'reference.json'
    if p.exists():
        return json.loads(p.read_text())
    m = km.ArchMotionEncoder('x1', 'absolute').to(ctx.device)
    m.load_state_dict(torch.load(X1_MK1B, map_location=ctx.device, weights_only=True))
    keys = encode_bank(m, ctx.bank, ctx.device)
    old = np.load(KF_DATA / 'x1_keys.npy')
    np.save(save / 'K-x1_keys.npy', keys)
    kc = dct3(ctx.windows); np.save(save / 'DCT3_keys.npy', kc)
    res = {'K-x1': {'a': key_metric_a(m, ctx, keys), 'params': nparams(m), 'keys': str(save / 'K-x1_keys.npy'),
                    'max_abs_diff_vs_kqfinal_cache': float(np.abs(keys - old).max()),
                    'diagnostics': key_diagnostics(keys, ctx.bank)},
           'DCT3': {'a': dct_metric_a(ctx), 'params': 0, 'keys': str(save / 'DCT3_keys.npy')},
           'oracle': {n: float(ctx.splits[n]['errors'].min(1).mean()) for n in EVAL}}
    write_json(p, res)
    return res


# ----------------------------------------------------------------------------
# stage 1 (b): the C1b query (history+state MLP, KQ-final CT recipe) trained on a fixed key
# ----------------------------------------------------------------------------

class HSMLP(nn.Module):
    """key_final.HSMLPQuery (88 -> 512 -> 256, normalized); optional 18-d head for the DCT3 key."""
    def __init__(self, dct: bool = False):
        super().__init__()
        self.tower = nn.Sequential(nn.Linear(88, 512), nn.GELU(), nn.Linear(512, 256))
        self.dct_projection = nn.Linear(256, 18) if dct else None
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / .07)))

    def forward(self, hs):
        q = F.normalize(self.tower(hs.float()), dim=-1)
        return self.dct_projection(q) if self.dct_projection is not None else q


def hs_features(sp: dict, state_key: str) -> np.ndarray:
    h = np.asarray(sp['history'], np.float32).reshape(len(sp['history']), -1)
    m = np.asarray(sp['history_mask'], np.float32)
    return np.concatenate([h, m, np.asarray(sp[state_key], np.float32)], -1).astype(np.float32)


def kq_score(q, k, dct: bool, scale):
    s = scale.exp().clamp(max=100.)
    if dct:
        return -torch.cdist(q.float(), k.float()).square() * s
    return (q @ k.T) * s


@torch.no_grad()
def kq_scores(model, hs: np.ndarray, keys: torch.Tensor, dct: bool, device) -> np.ndarray:
    model.eval()
    x = torch.as_tensor(hs, device=device)
    return torch.cat([kq_score(model(x[s:s + 512]), keys, dct, model.logit_scale) for s in range(0, len(x), 512)]).float().cpu().numpy()


def train_keyq(name: str, cfg0: dict, ctx: Ctx, save: Path) -> dict:
    """Same loop as key_final.train_row with the C1b config (rows='eh', same_scene, tau .01, batch 8,
    rand 256, lr 1e-4, 16 epochs, Dev every 2 epochs), but errors are precomputed and the Dev used
    for epoch selection is Dev v2 20 Hz.  `eh_state`='ref' keeps the C1b training inputs exactly
    (the KQ-final eh cache, see report), 'fixed' uses the corrected state."""
    cfg = {'keys': None, 'dct': False, 'seed': 4101, 'epochs': 16, 'batch': 8, 'rand': 256, 'lr': 1e-4, 'tau': .01,
           'same_scene': True, 'eh_state': 'ref', 'dev': 'dev2', **cfg0}
    dev = ctx.device; seed = int(cfg['seed'])
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    dct = bool(cfg['dct'])
    model = HSMLP(dct).to(dev)
    keys = torch.as_tensor(np.load(cfg['keys']), device=dev, dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg['lr']), weight_decay=1e-4)
    tr = ctx.splits['eh']
    hs_tr = torch.as_tensor(hs_features(tr, 'state_ref' if cfg['eh_state'] == 'ref' else 'state'), device=dev)
    E = torch.as_tensor(tr['errors'], device=dev)
    ttask = tr['task']; tep = tr['episode']; cand = tr['candidate_id']
    bt, be = ctx.bt, ctx.be
    btt = torch.as_tensor(bt, device=dev); bet = torch.as_tensor(be, device=dev)
    qn = len(ttask); ids = np.arange(qn); nb = len(bt)
    dsp = ctx.splits[cfg['dev']] if cfg['dev'] in ctx.splits else None
    hs_dev = hs_features(dsp, 'state')
    best, best_ep, best_state, hist = float('inf'), None, None, []
    t0 = time.time()
    for ep in range(1, int(cfg['epochs']) + 1):
        rng.shuffle(ids); model.train(); losses = []
        for s in range(0, qn, int(cfg['batch'])):
            qi = ids[s:s + int(cfg['batch'])]
            rand = rng.choice(nb, size=min(int(cfg['rand']), nb), replace=False)
            own = cand[qi]
            cids = np.unique(np.r_[own[own >= 0], rand])
            qit = torch.as_tensor(qi, device=dev); cit = torch.as_tensor(cids, device=dev)
            am = ~((torch.as_tensor(ttask[qi], device=dev)[:, None] == btt[cit][None]) &
                   (torch.as_tensor(tep[qi], device=dev)[:, None] == bet[cit][None]))
            q = model(hs_tr[qit])
            sc = kq_score(q, keys[cit], dct, model.logit_scale)
            err = E[qit][:, cit]
            loss = k2.soft_listwise_loss(sc, err, am, float(cfg['tau']))
            if cfg['same_scene']:
                same = (btt[cit][None] == torch.as_tensor(ttask[qi], device=dev)[:, None]) & \
                       (bet[cit][None] != torch.as_tensor(tep[qi], device=dev)[:, None])
                sm = same & am
                negid = err.masked_fill(~sm, -torch.inf).argmax(-1)
                posid = err.masked_fill(~am, torch.inf).argmin(-1)
                has = sm.any(-1)
                if has.any():
                    pos = sc.gather(1, posid[:, None]).squeeze(1); neg = sc.gather(1, negid[:, None]).squeeze(1)
                    loss = loss + .1 * F.softplus(neg[has] - pos[has] + .1).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.); opt.step()
            with torch.no_grad():
                model.logit_scale.clamp_(max=float(np.log(100.)))
            losses.append(float(loss.detach()))
        rec = {'epoch': ep, 'loss': float(np.mean(losses)), 'seconds': time.time() - t0}
        if ep % 2 == 0 or ep == int(cfg['epochs']):
            rec['dev'] = float(top1(kq_scores(model, hs_dev, keys, dct, dev), dsp).mean())
            if rec['dev'] < best:
                best, best_ep = rec['dev'], ep
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        hist.append(rec)
    model.load_state_dict(best_state)
    torch.save({'cfg': cfg, 'query': best_state}, save / f'{name}.pt')
    ev = {}
    for n in EVAL:
        sp = ctx.splits[n]
        ev[n] = block(top1(kq_scores(model, hs_features(sp, 'state'), keys, dct, dev), sp), sp)
    row = {'name': name, 'cfg': cfg, 'selected_epoch': best_ep, 'dev_selected': best, 'history': hist,
           'evaluation': ev, 'seconds': time.time() - t0}
    write_json(save / f'{name}.json', row)
    return row


# ----------------------------------------------------------------------------
# stage 2: query = GRU(h) over state+action history, h cross-attends the VLM tokens
# ----------------------------------------------------------------------------

class GPUSplit:
    """Split tensors on the GPU: per-episode sequences, row start, VLM tokens, errors."""
    def __init__(self, sp: dict, device, vlm: bool, errors: bool = True):
        t = lambda x, dt=None: torch.as_tensor(np.asarray(x), device=device, dtype=dt)
        self.seq_a = t(sp['seq_actions'], torch.float32); self.seq_s = t(sp['seq_states'], torch.float32)
        self.off = t(sp['ep_offset'], torch.long); self.slot = t(sp['ep_slot'], torch.long); self.start = t(sp['start'], torch.long)
        self.g = self.off[self.slot] + self.start
        self.n = len(sp['key'])
        self.vlm = None
        if vlm:
            v = np.asarray(sp['vlm'])
            self.vlm = t(v, torch.float16)
        self.errors = t(sp['errors'], torch.float32) if errors else None
        self.allowed = t(sp['allowed'], torch.bool)
        self.task = t(sp['task'], torch.long); self.episode = t(sp['episode'], torch.long)

    def window(self, ix: torch.Tensor, L: int = 10):
        """Left-padded L-step tokens: token j <-> tau = t-L+1+j: [a_{tau-1}, s_tau, a-valid]; mask valid tau>=0."""
        t = self.start[ix][:, None]; g = self.g[ix][:, None]
        rel = torch.arange(L, device=ix.device)[None] - (L - 1)  # -L+1 .. 0
        tau = t + rel
        valid = tau >= 0
        gs = (g + rel).clamp(min=0)
        s = self.seq_s[gs] * valid[..., None]
        av = tau >= 1
        a = self.seq_a[(g + rel - 1).clamp(min=0)] * av[..., None]
        x = torch.cat([a, s, av[..., None].float()], -1)
        return x, valid

    def full(self, ix: torch.Tensor, max_len: int | None = None):
        """Right-padded tokens from the episode start (tau = 0..t) and lengths."""
        t = self.start[ix]; g = self.g[ix]
        ln = t + 1
        if max_len is not None:
            ln = ln.clamp(max=max_len)
        T = int(ln.max())
        j = torch.arange(T, device=ix.device)[None]
        tau = (t + 1 - ln)[:, None] + j  # first token = max(0, t+1-max_len)
        valid = j < ln[:, None]
        gs = (g - t)[:, None] + tau
        gs = torch.where(valid, gs, g[:, None])
        s = self.seq_s[gs] * valid[..., None]
        av = (tau >= 1) & valid
        a = self.seq_a[(gs - 1).clamp(min=0)] * av[..., None]
        return torch.cat([a, s, av[..., None].float()], -1), ln


class QueryNet(nn.Module):
    def __init__(self, enc: str = 'gru', gru_layers: int = 1, xattn: int = 1, ffn: bool = True, history: str = 'w10',
                 dim: int = 256, heads: int = 8, in_dim: int = 16, dct: bool = False):
        super().__init__()
        self.enc, self.xattn_n, self.history = enc, int(xattn), history
        self.dct_projection = nn.Linear(dim, 18) if dct else None
        if enc == 'gru':
            self.emb = nn.Sequential(nn.Linear(in_dim, dim), nn.GELU())
            self.gru = nn.GRU(dim, dim, num_layers=int(gru_layers), batch_first=True)
        else:  # C1b MLP tower on the same 10-step window (+ mask)
            self.mlp = nn.Sequential(nn.Linear(10 * in_dim + 10, 512), nn.GELU(), nn.Linear(512, dim))
        if self.xattn_n:
            self.kv = nn.Sequential(nn.LayerNorm(2048), nn.Linear(2048, dim))
            self.attn = nn.ModuleList([nn.MultiheadAttention(dim, heads, batch_first=True) for _ in range(self.xattn_n)])
            self.post = nn.ModuleList([nn.LayerNorm(dim) for _ in range(self.xattn_n)])
            self.ffn = nn.ModuleList([FFN(dim) for _ in range(self.xattn_n)]) if ffn else None
        else:
            self.post0 = nn.LayerNorm(dim)
            self.ffn0 = FFN(dim) if ffn else None
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / .07)))
        self.diag = {}

    def score(self, q, keys):
        s = self.logit_scale.exp().clamp(max=100.)
        if self.dct_projection is not None:  # unlearned DCT3 key: negative squared distance (as key_final 'dct')
            return -torch.cdist(self.dct_projection(q).float(), keys.float()).square() * s
        return (q @ keys.T) * s

    def encode_h(self, gs: GPUSplit, ix: torch.Tensor):
        if self.enc == 'gru':
            if self.history == 'full':
                x, ln = gs.full(ix)
                packed = nn.utils.rnn.pack_padded_sequence(self.emb(x), ln.cpu(), batch_first=True, enforce_sorted=False)
                _, hn = self.gru(packed)
            else:
                x, valid = gs.window(ix)
                ln = valid.sum(1)
                # move valid tokens to the front (right padding) for packing
                order = torch.argsort(torch.where(valid, 0, 1), dim=1, stable=True)
                x = torch.gather(x, 1, order[..., None].expand_as(x))
                packed = nn.utils.rnn.pack_padded_sequence(self.emb(x), ln.cpu(), batch_first=True, enforce_sorted=False)
                _, hn = self.gru(packed)
            return hn[-1]
        x, valid = gs.window(ix)
        return self.mlp(torch.cat([x.flatten(1), valid.float()], -1))

    def forward(self, gs: GPUSplit, ix: torch.Tensor, diag: bool = False):
        h = self.encode_h(gs, ix)
        if not self.xattn_n:
            x = self.post0(h)
            if self.ffn0 is not None:
                x = self.ffn0(x)
            if diag:
                self.diag = {'h_norm': h.norm(dim=-1)}
            return F.normalize(x, dim=-1)
        kv = self.kv(gs.vlm[ix].float())
        x = h[:, None]
        d = {'h_norm': h.norm(dim=-1)}
        for l in range(self.xattn_n):
            a, w = self.attn[l](x, kv, kv, need_weights=diag, average_attn_weights=False)
            if diag:
                wm = w.mean(1)[:, 0]  # [B, 34] mean over heads
                ent_head = -(w.clamp_min(1e-12) * w.clamp_min(1e-12).log()).sum(-1)[:, :, 0]  # [B, heads]
                d[f'l{l}_entropy_headmean'] = ent_head.mean(1) / math.log(kv.shape[1])
                d[f'l{l}_entropy_of_mean'] = -(wm.clamp_min(1e-12) * wm.clamp_min(1e-12).log()).sum(-1) / math.log(kv.shape[1])
                d[f'l{l}_attn_norm_ratio'] = a[:, 0].norm(dim=-1) / x[:, 0].norm(dim=-1)
                d[f'l{l}_weights'] = wm
            x = self.post[l](x + a)
            if self.ffn is not None:
                x = self.ffn[l](x)
        if diag:
            self.diag = d
        return F.normalize(x[:, 0], dim=-1)


def query_loss(sc, err, allowed, same_bad, objective: str, tau: float, margin_w: float):
    if objective == 'onehot':
        loss = k2.hard_info_nce_loss(sc, err, allowed)
    else:
        loss = k2.soft_listwise_loss(sc, err, allowed, tau)
    if margin_w > 0:
        has = same_bad.any(-1)
        if has.any():
            posid = err.masked_fill(~allowed, torch.inf).argmin(-1)
            pos = sc.gather(1, posid[:, None]).squeeze(1)
            neg = sc.masked_fill(~same_bad, -torch.inf).max(-1).values
            loss = loss + margin_w * F.softplus(neg[has] - pos[has] + .1).mean()
    return loss


@torch.no_grad()
def query_scores(model: QueryNet, gs: GPUSplit, keys: torch.Tensor, batch: int = 512, diag: bool = False):
    model.eval()
    out, dg = [], {}
    for s in range(0, gs.n, batch):
        ix = torch.arange(s, min(s + batch, gs.n), device=keys.device)
        q = model(gs, ix, diag=diag)
        out.append(model.score(q, keys).float())
        if diag:
            for k, v in model.diag.items():
                dg.setdefault(k, []).append(v.float().cpu())
    sc = torch.cat(out).masked_fill(~gs.allowed, -torch.inf)
    return sc, ({k: torch.cat(v).numpy() for k, v in dg.items()} if diag else None)


def train_query(name: str, cfg0: dict, ctx: Ctx, gsplits: dict, save: Path) -> dict:
    cfg = {'keys': None, 'enc': 'gru', 'gru_layers': 1, 'xattn': 1, 'ffn': True, 'history': 'w10',
           'objective': 'soft', 'tau': .01, 'margin_w': .1, 'seed': 1, 'epochs': 40, 'batch': 256, 'lr': 3e-4,
           'wd': 1e-4, **cfg0}
    dev = ctx.device; seed = int(cfg['seed'])
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    keys = torch.as_tensor(np.load(cfg['keys']), device=dev, dtype=torch.float32)
    model = QueryNet(cfg['enc'], cfg['gru_layers'], cfg['xattn'], cfg['ffn'], cfg['history'], dct=bool(cfg.get('dct', False))).to(dev)
    tr = gsplits['eh']
    if cfg['xattn'] and tr.vlm is None:
        raise ValueError('VLM tokens not loaded')
    steps_per_epoch = int(math.ceil(tr.n / int(cfg['batch'])))
    total = steps_per_epoch * int(cfg['epochs'])
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg['lr']), weight_decay=float(cfg['wd']))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1., (s + 1) / steps_per_epoch) *
                                              .5 * (1 + math.cos(math.pi * min(1., s / total))))
    bt = torch.as_tensor(ctx.bt, device=dev); be = torch.as_tensor(ctx.be, device=dev)
    # same scene (same task), other episode, clearly worse than the best allowed unit: hard negatives
    same_task = tr.task[:, None] == bt[None]
    best_state, best, best_ep, hist = None, float('inf'), None, []
    dsp = ctx.splits['dev2']
    t0 = time.time()
    for ep in range(1, int(cfg['epochs']) + 1):
        model.train(); perm = rng.permutation(tr.n); losses = []
        for s in range(0, tr.n, int(cfg['batch'])):
            ix = torch.as_tensor(perm[s:s + int(cfg['batch'])], device=dev)
            q = model(tr, ix)
            sc = model.score(q, keys)
            err = tr.errors[ix]; allowed = tr.allowed[ix]
            same = same_task[ix] & allowed
            med = torch.nanquantile(err.masked_fill(~same, float('nan')), .5, dim=1, keepdim=True)
            same_bad = same & (err > med)
            loss = query_loss(sc, err, allowed, same_bad, cfg['objective'], float(cfg['tau']), float(cfg['margin_w']))
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step(); sched.step()
            with torch.no_grad():
                model.logit_scale.clamp_(max=float(np.log(100.)))
            losses.append(float(loss.detach()))
        sc, _ = query_scores(model, gsplits['dev2'], keys)
        d = float(top1(sc.cpu().numpy(), dsp).mean())
        rec = {'epoch': ep, 'loss': float(np.mean(losses)), 'dev': d, 'logit_scale': float(model.logit_scale.exp()),
               'seconds': time.time() - t0}
        if d < best:
            best, best_ep = d, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        hist.append(rec)
    model.load_state_dict(best_state)
    torch.save({'cfg': cfg, 'query': best_state}, save / f'{name}.pt')
    ev = {}
    for n in EVAL:
        sc, dg = query_scores(model, gsplits[n], keys, diag=True)
        sp = ctx.splits[n]
        ev[n] = block(top1(sc.cpu().numpy(), sp), sp)
        if dg:
            ev[n]['diag'] = {k: float(np.mean(v)) for k, v in dg.items() if k != 'l0_weights' and k != 'l1_weights' and v.ndim == 1}
            if 'l0_weights' in dg:
                w = dg['l0_weights']
                ev[n]['diag']['l0_weight_image_base'] = float(w[:, :16].sum(1).mean())
                ev[n]['diag']['l0_weight_image_wrist'] = float(w[:, 16:32].sum(1).mean())
                ev[n]['diag']['l0_weight_text'] = float(w[:, 32:].sum(1).mean())
            np.savez_compressed(save / f'{name}__{n}_diag.npz', **{k: v.astype(np.float16) for k, v in dg.items()})
    row = {'name': name, 'cfg': cfg, 'params': nparams(model), 'selected_epoch': best_ep, 'dev_selected': best,
           'history': hist, 'evaluation': ev, 'seconds': time.time() - t0}
    write_json(save / f'{name}.json', row)
    return row


@torch.no_grad()
def vlm_shuffle_ablation(name: str, ctx: Ctx, gsplits: dict, save: Path, seed: int = 9001) -> dict:
    """Test-time: VLM tokens permuted across rows (same split) -> how much does top-1 move."""
    blob = torch.load(save / f'{name}.pt', map_location=ctx.device, weights_only=False)
    cfg = blob['cfg']
    model = QueryNet(cfg['enc'], cfg['gru_layers'], cfg['xattn'], cfg['ffn'], cfg['history'], dct=bool(cfg.get('dct', False))).to(ctx.device)
    model.load_state_dict(blob['query'])
    keys = torch.as_tensor(np.load(cfg['keys']), device=ctx.device, dtype=torch.float32)
    res = {}
    for n in EVAL:
        gs = gsplits[n]; sp = ctx.splits[n]
        sc, _ = query_scores(model, gs, keys)
        base = top1(sc.cpu().numpy(), sp)
        perm = torch.as_tensor(np.random.default_rng(seed).permutation(gs.n), device=ctx.device)
        orig = gs.vlm
        gs.vlm = orig[perm]
        try:
            sc2, _ = query_scores(model, gs, keys)
        finally:
            gs.vlm = orig
        sh = top1(sc2.cpu().numpy(), sp)
        res[n] = {'intact': float(base.mean()), 'vlm_shuffled': float(sh.mean()),
                  'top1_changed_fraction': float((sc.argmax(1) != sc2.argmax(1)).float().mean())}
    return res


# ----------------------------------------------------------------------------
# baselines
# ----------------------------------------------------------------------------

def baselines(ctx: Ctx, save: Path) -> dict:
    p = save / 'baselines.json'
    if p.exists():
        return json.loads(p.read_text())
    res = {}
    for n in EVAL:
        sp = ctx.splits[n]
        hold = k2._hold_last_actions(sp)
        res.setdefault('hold_last', {})[n] = block(k2.action_errors(hold, sp['actions'])['arm6'], sp)
        res.setdefault('oracle', {})[n] = block(np.where(sp['allowed'], sp['errors'], np.inf).min(1), sp)
    keys = torch.as_tensor(np.load(KF_DATA / 'x1_keys.npy'), device=ctx.device, dtype=torch.float32)
    for i, ck in enumerate(C1B, 1):
        blob = torch.load(ck, map_location=ctx.device, weights_only=False)
        m = HSMLP(False).to(ctx.device); m.load_state_dict(blob['query'])
        for n in EVAL:
            sp = ctx.splits[n]
            res.setdefault(f'C1b_s{i}', {})[n] = block(top1(kq_scores(m, hs_features(sp, 'state'), keys, False, ctx.device), sp), sp)
    write_json(p, res)
    log('baselines', ctx.out, **{k: {n: v[n]['all'] for n in EVAL} for k, v in res.items()})
    return res


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=('baselines', 'keys', 'keyq', 'query', 'ablate'))
    ap.add_argument('--plan', type=Path)
    ap.add_argument('--names', nargs='*')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--out', type=Path, default=OUT)
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '8')))
    if a.cmd == 'baselines':
        ctx = Ctx(a.device, splits=EVAL)
        sub = a.out / 'stage1'; sub.mkdir(parents=True, exist_ok=True)
        baselines(ctx, a.out); reference_keys(Ctx(a.device, splits=EVAL), sub)
        return
    plan = json.loads(a.plan.read_text())
    names = a.names or list(plan)
    sub = a.out / {'keys': 'stage1/keys', 'keyq': 'stage1/keyq', 'query': 'stage2/rows', 'ablate': 'stage2/rows'}[a.cmd]
    sub.mkdir(parents=True, exist_ok=True)
    if a.cmd == 'keys':
        ctx = Ctx(a.device, splits=EVAL)
    elif a.cmd == 'keyq':
        ctx = Ctx(a.device)
    else:
        need_vlm = any(plan[n].get('xattn', 1) for n in names)
        ctx = Ctx(a.device, vlm=need_vlm)
        gsplits = {n: GPUSplit(ctx.splits[n], ctx.device, need_vlm) for n in ('eh',) + EVAL}
    if a.cmd == 'ablate':
        res = {}
        p = a.out / 'stage2' / 'vlm_shuffle.json'
        if p.exists():
            res = json.loads(p.read_text())
        for n in names:
            res[n] = vlm_shuffle_ablation(n, ctx, gsplits, sub)
            write_json(p, res)
        return
    for n in names:
        if (sub / f'{n}.json').exists():
            continue
        lock = sub / f'{n}.lock'
        try:  # several workers may share one name list; the first to create the lock runs the row
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode()); os.close(fd)
        except FileExistsError:
            pid = int(lock.read_text() or 0)
            if pid and os.path.exists(f'/proc/{pid}'):
                continue
            lock.write_text(str(os.getpid()))
        log('row_start', a.out, cmd=a.cmd, name=n, cfg=plan[n])
        if a.cmd == 'keys':
            r = train_key(n, plan[n], ctx, sub)
            log('row_done', a.out, cmd=a.cmd, name=n, epoch=r['selected_epoch'], a=r['a'], seconds=r['seconds'])
        elif a.cmd == 'keyq':
            r = train_keyq(n, plan[n], ctx, sub)
            log('row_done', a.out, cmd=a.cmd, name=n, epoch=r['selected_epoch'], dev=r['dev_selected'],
                learned=r['evaluation']['learned']['all'], unseen20=r['evaluation']['unseen20']['all'], seconds=r['seconds'])
        else:
            r = train_query(n, plan[n], ctx, gsplits, sub)
            log('row_done', a.out, cmd=a.cmd, name=n, epoch=r['selected_epoch'], dev=r['dev_selected'],
                learned=r['evaluation']['learned']['all'], unseen20=r['evaluation']['unseen20']['all'], seconds=r['seconds'])
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
