#!/usr/bin/env python3
"""KQ-final: one code path for the MK1c / CT1 motion-key + query comparison.

Everything (CT1 baseline reproduction, MK1c reproduction, one-factor swaps,
corrected MK1c grid, history+state MLP query rows, KEYGRID re-runs) is trained
by ``train_row`` so that a factor that differs between two rows is exactly the
factor named in the row config.  Existing modules are imported read-only.

Data rules: training rows are LIBERO-40 memory-side only (unit starts, or the
stride-5 dense memory frames); Dev 217 rows select epochs / tau; the locked
learned 621 / unseen 619 rows are scored once per row.  No task id, future or
ground truth enters a query.  Eval actions are used only for scoring.
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
REPO_ROOT = Path(__file__).resolve().parents[5]
STATE_DIR = Path(os.environ.get('SKILLNET_STATE_DIR', REPO_ROOT / '.skillnet-state')).expanduser()
sys.path.insert(0, str(HERE))
import key_motion as km  # noqa: E402  (read-only)
import key_shared_latent as k2  # noqa: E402  (read-only)

ROOT = (STATE_DIR / 'outputs/memory-pilot/unseen-libero')
OUT = ROOT / 'key-motion-01/final'
CT_CACHE = ROOT / 'key-contrastive-01/data/aligned.npz'
RUN_ROOT = (STATE_DIR / 'outputs/memory-pilot/pi05-memory40-20261104/run-01')
MISSING_ROOT = ROOT / 'key-shared-01/attempt-02/prefix-missing'
SEEN5 = ROOT / 'seg-context-01/seg5-cache'
UNSEEN5 = ROOT / 'seg-context-01/unseen-stride5'
X1_MK1B = ROOT / 'key-motion-01/arch/x1_absolute_dist_tau0.01.pt'
X1_MK1C = ROOT / 'key-motion-01/query/x1_fixed.pt'
KEYGRID_RESULTS = ROOT / 'key-grid-01/production-04/results.json'
SPLITS = ('start', 'eh', 'dev', 'learned', 'unseen')
FIELDS = ('key', 'history', 'history_mask', 'state', 'actions', 'task', 'episode')


def log(event: str, out: Path = OUT, **kw: Any) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'progress.log').open('a') as f:
        f.write(json.dumps({'time': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'stage': event, **kw},
                           ensure_ascii=False) + '\n')


def write_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------

def prep_data(out: Path = OUT) -> None:
    """Unpack the CT1 aligned cache to memmap-able .npy and add start-row VLM tokens."""
    d = out / 'data'
    d.mkdir(parents=True, exist_ok=True)
    z = np.load(CT_CACHE, allow_pickle=False)
    keep = [k for k in z.files if k.startswith(('bank__', 'start__', 'eh__', 'dev__', 'learned__', 'unseen__'))
            or k in ('bank_actions', 'bank_task', 'bank_episode', 'x1_keys', 'dct_keys', 'dct_mean', 'dct_std')]
    for k in keep:
        if 'vjepa' in k or k.endswith('__pi05'):
            continue
        if not (d / f'{k}.npy').exists():
            np.save(d / f'{k}.npy', z[k], allow_pickle=False)
        log('prep_array', out, name=k)
    if not (d / 'start__tokens.npy').exists():
        keys = np.load(d / 'start__key.npy')
        store = k2.ExactFeatureStore(RUN_ROOT, MISSING_ROOT, SEEN5, UNSEEN5)
        tok, grp = store.get(keys, domain='libero40',
                             progress_callback=lambda s, **kw: log(s, out, **kw) if kw.get('completed', 0) % 1024 == 0 else None,
                             progress_name='start_vlm')
        np.save(d / 'start__groups.npy', grp, allow_pickle=False)
        np.save(d / 'start__tokens.npy', tok, allow_pickle=False)
    # MK1c fixed key (tau .05 refit) encoded on the same bank.
    if not (d / 'x1_keys_mk1c.npy').exists():
        bank = load_bank(out)
        enc = km.ArchMotionEncoder('x1', 'absolute')
        enc.load_state_dict(torch.load(X1_MK1C, map_location='cpu', weights_only=True))
        np.save(d / 'x1_keys_mk1c.npy', encode_bank(enc, bank, 'cpu'), allow_pickle=False)
    write_json(d / 'manifest.json', {'complete': True, 'source': str(CT_CACHE), 'x1_mk1b': str(X1_MK1B),
                                     'x1_mk1c': str(X1_MK1C)})
    log('prep_complete', out)


def _npy(out: Path, name: str, mmap: bool = False):
    return np.load(out / 'data' / f'{name}.npy', mmap_mode='r' if mmap else None, allow_pickle=False)


def load_bank(out: Path = OUT) -> dict[str, np.ndarray]:
    b = {k: _npy(out, 'bank__' + k) for k in ('task', 'episode', 'natural_actions', 'length', 'source', 'coefficients')}
    b['actions'] = _npy(out, 'bank_actions')  # [4516,10,7] stored-value windows
    return b


def load_split(name: str, out: Path = OUT, *, tokens: bool = False) -> dict[str, np.ndarray]:
    s = {k: _npy(out, f'{name}__{k}') for k in FIELDS}
    hp = out / 'data' / f'{name}__hard.npy'
    s['hard'] = np.load(hp) if hp.exists() else np.zeros(len(s['key']), bool)
    s['task'] = s['task'].astype(np.int64)
    s['episode'] = s['episode'].astype(np.int64)
    if tokens:
        s['tokens'] = _npy(out, f'{name}__tokens', mmap=True)
        s['groups'] = _npy(out, f'{name}__groups', mmap=True)
    return s


def encode_bank(enc: nn.Module, bank: dict[str, np.ndarray], device) -> np.ndarray:
    actions = torch.as_tensor(np.asarray(bank['natural_actions'])[..., :6], device=device, dtype=torch.float32)
    lengths = torch.as_tensor(np.asarray(bank['length']), device=device, dtype=torch.long)
    valid = torch.arange(actions.shape[1], device=device)[None] < lengths[:, None]
    enc = enc.to(device)
    return km._encode_motion(enc, actions, valid, batch=128).numpy().astype(np.float32)


def candidate_ids_for(split: dict[str, np.ndarray], bank: dict[str, np.ndarray]) -> np.ndarray:
    cmap = {tuple(map(int, k[:3])): i for i, k in enumerate(np.asarray(bank['source']))}
    return np.asarray([cmap.get(tuple(map(int, k)), -1) for k in split['key']], np.int64)


def hs_features(split: dict[str, np.ndarray]) -> np.ndarray:
    """Exactly key_grid.make_vector_features(part, 'history_state'): 70 + 10 + 8 = 88."""
    h = np.asarray(split['history'], np.float32).reshape(len(split['history']), -1)
    m = np.asarray(split['history_mask'], np.float32).reshape(len(h), -1)
    return np.concatenate([h, m, np.asarray(split['state'], np.float32)], -1).astype(np.float32)


# ----------------------------------------------------------------------------
# models
# ----------------------------------------------------------------------------

class HSMLPQuery(nn.Module):
    """KEYGRID two-tower query tower (88 -> 512 -> 256), normalized; optional DCT head."""
    def __init__(self, key_mode: str, input_dim: int = 88, dim: int = 256):
        super().__init__()
        self.key_mode = key_mode
        self.tower = nn.Sequential(nn.Linear(input_dim, 512), nn.GELU(), nn.Linear(512, dim))
        self.dct_projection = nn.Linear(dim, 18) if key_mode == 'dct' else None
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / .07)))

    def forward(self, *, hs: torch.Tensor, **_: Any) -> torch.Tensor:
        q = F.normalize(self.tower(hs.float()), dim=-1)
        return self.dct_projection(q) if self.dct_projection is not None else q


class KeyGridKey(nn.Module):
    """KEYGRID two-tower key tower on candidate-start history+state (5 tokens, mean cosine)."""
    def __init__(self, input_dim: int = 88, dim: int = 256, tokens: int = 5):
        super().__init__()
        self.tokens, self.dim = tokens, dim
        self.tower = nn.Sequential(nn.Linear(input_dim, 512), nn.GELU(), nn.Linear(512, dim * tokens))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # mean_k <q, k_k> == <q, mean_k k_k> for an expanded single query vector.
        return F.normalize(self.tower(x.float()).reshape(-1, self.tokens, self.dim), dim=-1).mean(1)


class MQEQuery(nn.Module):
    """CT1 query: key_motion.MotionQueryEncoder (cross + self-attn blocks) + optional DCT head."""
    def __init__(self, input_mode: str, key_mode: str):
        super().__init__()
        self.encoder = km.MotionQueryEncoder(input_mode, 'q1')
        self.dct_projection = nn.Linear(256, 18) if key_mode == 'dct' else None

    @property
    def logit_scale(self):
        return self.encoder.logit_scale

    def forward(self, *, history, history_mask, state, tokens=None, groups=None, motion_history=None,
                motion_history_valid=None, **_: Any):
        if motion_history is not None:
            raise ValueError('MQE row does not take motion history')
        q = self.encoder(history=history, history_mask=history_mask, state=state, tokens=tokens, groups=groups)[:, 0]
        return self.dct_projection(q) if self.dct_projection is not None else q


class RQEQuery(nn.Module):
    """MK1c query: key_motion.RetrievalQueryEncoder (cross-only blocks)."""
    def __init__(self, input_mode: str, key_mode: str, query_mode: str = 'q1'):
        super().__init__()
        self.encoder = km.RetrievalQueryEncoder(input_mode, query_mode, key_mode='dct' if key_mode == 'dct' else 'x1')

    @property
    def logit_scale(self):
        return self.encoder.logit_scale

    def forward(self, *, history, history_mask, state, tokens=None, groups=None, motion_history=None,
                motion_history_valid=None, **_: Any):
        q = self.encoder(history=history, history_mask=history_mask, state=state, tokens=tokens, groups=groups,
                         motion_history=motion_history, motion_history_valid=motion_history_valid)
        return q[:, 0] if q.ndim == 3 else q


def make_query(cfg: dict[str, Any]) -> nn.Module:
    km_ = 'dct' if cfg['key'] == 'dct' else 'x1'
    if cfg['query'] == 'mqe':
        return MQEQuery(cfg['input'], km_)
    if cfg['query'] == 'rqe':
        return RQEQuery(cfg['input'], km_, cfg.get('qmode', 'q1'))
    if cfg['query'] == 'hsmlp':
        return HSMLPQuery(km_)
    raise ValueError(cfg['query'])


def score(q: torch.Tensor, k: torch.Tensor, key: str, scale: torch.Tensor) -> torch.Tensor:
    s = scale.exp().clamp(max=100.)
    if key == 'dct':
        return -torch.cdist(q.float(), k.float()).square() * s
    return (q @ k.T) * s


# ----------------------------------------------------------------------------
# training
# ----------------------------------------------------------------------------

DEFAULT = dict(query='mqe', input='e_h', key='x1_mk1b', tau=.005, aug=False, rows='start', seed=4101,
               epochs=16, batch=8, rand=256, lr=1e-4, dev_epochs='every2', same_scene=False, mhistory=False,
               qmode='q1', scale_detached=False)


class RowContext:
    """Per-process tensors shared by rows."""
    def __init__(self, out: Path, device, need_tokens: bool):
        self.out, self.device = out, device
        self.bank = load_bank(out)
        self.windows = np.asarray(self.bank['actions'], np.float32)
        self.bt = np.asarray(self.bank['task'], np.int64)
        self.be = np.asarray(self.bank['episode'], np.int64)
        self.splits = {n: load_split(n, out, tokens=need_tokens and n != 'eh') for n in SPLITS}
        for n in ('dev', 'learned', 'unseen'):
            self.splits[n]['errors'] = k2._arm6_candidate_mse(self.splits[n]['actions'], self.windows)
        for n in ('start', 'eh'):
            self.splits[n]['candidate_id'] = candidate_ids_for(self.splits[n], self.bank)
        self.fixed = {'x1_mk1b': _npy(out, 'x1_keys'), 'x1_mk1c': _npy(out, 'x1_keys_mk1c'), 'dct': _npy(out, 'dct_keys')}
        # Candidate-start H+S (bank order == start order; asserted).
        if not np.array_equal(candidate_ids_for(self.splits['start'], self.bank), np.arange(len(self.bt))):
            raise AssertionError('start rows are not in bank order')
        self.bank_hs = hs_features(self.splits['start'])
        for n in SPLITS:
            self.splits[n]['hs'] = hs_features(self.splits[n])
        self.dct_mean, self.dct_std = _npy(out, 'dct_mean'), _npy(out, 'dct_std')


def pack(split: dict[str, np.ndarray], ids: np.ndarray, cfg: dict[str, Any], device, *, rng=None, augment=False,
         mhist=None) -> dict[str, Any]:
    ix = np.asarray(ids, np.int64)
    hist = np.asarray(split['history'][ix], np.float32)
    mask = np.asarray(split['history_mask'][ix], bool)
    if augment:
        hist, mask = k2._history_corrupt(hist, mask, rng)
    out = dict(history=torch.as_tensor(hist, device=device), history_mask=torch.as_tensor(mask, device=device),
               state=torch.as_tensor(np.asarray(split['state'][ix], np.float32), device=device))
    if cfg['query'] == 'hsmlp':
        hs = np.concatenate([hist.reshape(len(ix), -1), mask.astype(np.float32), np.asarray(split['state'][ix], np.float32)], -1)
        out['hs'] = torch.as_tensor(hs, device=device)
    if cfg['input'] == 'e_vlm_h':
        out['tokens'] = torch.as_tensor(np.asarray(split['tokens'][ix]), device=device, dtype=torch.float32)
        out['groups'] = torch.as_tensor(np.asarray(split['groups'][ix]), device=device, dtype=torch.long)
    if mhist is not None:
        out['motion_history'] = mhist[0][torch.as_tensor(ix, device=mhist[0].device)]
        out['motion_history_valid'] = mhist[1][torch.as_tensor(ix, device=mhist[1].device)]
    return out


class KeySide(nn.Module):
    """Candidate keys: fixed cache, trainable X1 (finetune / joint), or KEYGRID tower."""
    def __init__(self, cfg: dict[str, Any], ctx: RowContext):
        super().__init__()
        self.kind = cfg['key']
        self.ctx = ctx
        dev = ctx.device
        if self.kind in ('x1_mk1b', 'x1_mk1c', 'dct'):
            self.fixed = torch.as_tensor(np.asarray(ctx.fixed[self.kind]), device=dev, dtype=torch.float32)
            self.net = None
        elif self.kind in ('x1_finetune', 'x1_finetune_mk1b', 'x1_joint'):
            self.net = km.ArchMotionEncoder('x1', 'absolute').to(dev)
            if self.kind.startswith('x1_finetune'):
                init = X1_MK1B if self.kind.endswith('mk1b') else X1_MK1C
                self.net.load_state_dict(torch.load(init, map_location=dev, weights_only=True))
            a = torch.as_tensor(np.asarray(ctx.bank['natural_actions'])[..., :6], device=dev, dtype=torch.float32)
            ln = torch.as_tensor(np.asarray(ctx.bank['length']), device=dev, dtype=torch.long)
            self.actions, self.valid = a, torch.arange(a.shape[1], device=dev)[None] < ln[:, None]
        elif self.kind == 'keygrid':
            self.net = KeyGridKey().to(dev)
            self.feat = torch.as_tensor(ctx.bank_hs, device=dev)
        else:
            raise ValueError(self.kind)

    def score_mode(self) -> str:
        return 'dct' if self.kind == 'dct' else 'dot'

    def keys(self, cids=None) -> torch.Tensor:
        if self.net is None:
            return self.fixed if cids is None else self.fixed[torch.as_tensor(cids, device=self.fixed.device)]
        if self.kind == 'keygrid':
            x = self.feat if cids is None else self.feat[torch.as_tensor(cids, device=self.feat.device)]
            return self.net(x)
        if cids is None:
            out = [self.net(self.actions[s:s + 256], self.valid[s:s + 256])[:, 0] for s in range(0, len(self.actions), 256)]
            return torch.cat(out)
        ci = torch.as_tensor(cids, device=self.actions.device)
        return self.net(self.actions[ci], self.valid[ci])[:, 0]


def motion_history_keys(ctx: RowContext, split: dict[str, np.ndarray]):
    enc = km.ArchMotionEncoder('x1', 'absolute').to(ctx.device)
    enc.load_state_dict(torch.load(X1_MK1C, map_location=ctx.device, weights_only=True))
    hk, hv = km._history_motion_keys(enc, split, ctx.device)
    return hk.to(ctx.device), hv.to(ctx.device)


@torch.no_grad()
def score_split(qmodel, keyside: KeySide, cfg, ctx: RowContext, split: dict[str, np.ndarray], mhist=None,
                batch: int = 64) -> tuple[np.ndarray, np.ndarray]:
    qmodel.eval()
    if keyside.net is not None:
        keyside.net.eval()
    kall = keyside.keys()
    qs, out = [], []
    for s in range(0, len(split['key']), batch):
        ids = np.arange(s, min(s + batch, len(split['key'])))
        q = qmodel(**pack(split, ids, cfg, ctx.device, mhist=mhist))
        qs.append(q.float().cpu().numpy())
        out.append(score(q, kall, keyside.score_mode(), qmodel.logit_scale).float().cpu().numpy())
    sc = np.concatenate(out).astype(np.float32)
    allowed = ~((split['task'][:, None] == ctx.bt[None, :]) & (split['episode'][:, None] == ctx.be[None, :]))
    sc[~allowed] = -1e9
    return sc, np.concatenate(qs)


def metrics(scores: np.ndarray, errors: np.ndarray, split: dict[str, np.ndarray], bank_task: np.ndarray) -> dict[str, Any]:
    n = len(scores)
    top = scores.argmax(1)
    per = errors[np.arange(n), top]
    best = errors.argmin(1)
    # rank (1-based) of the oracle-best unit in the model ranking
    rank = (scores > scores[np.arange(n), best][:, None]).sum(1) + 1
    from scipy.stats import rankdata
    sr = rankdata(scores, axis=1); er = rankdata(-errors, axis=1)
    sr -= sr.mean(1, keepdims=True); er -= er.mean(1, keepdims=True)
    rho = (sr * er).sum(1) / np.maximum(np.sqrt((sr * sr).sum(1) * (er * er).sum(1)), 1e-12)
    hard = np.asarray(split['hard'], bool)

    def block(m):
        return {'top1_arm6_mse': float(per[m].mean()), 'spearman': float(np.mean(rho[m])),
                'oracle_rank_median': float(np.median(rank[m])),
                'recall_at': {str(k): float(np.mean(rank[m] <= k)) for k in (1, 5, 20, 100)},
                'other_task_top1_fraction': float(np.mean(bank_task[top[m]] != np.asarray(split['task'])[m]))}
    res = {'all': block(np.ones(n, bool)), 'rows': int(n), 'hard_rows': int(hard.sum())}
    res['hard'] = block(hard) if hard.any() else None
    res['per_query_top1'] = per.astype(np.float64).tolist()
    res['top1_unit'] = top.astype(int).tolist()
    return res


def train_row(name: str, cfg0: dict[str, Any], ctx: RowContext, *, save_dir: Path, final_eval: bool = True,
              save_scores: bool = False) -> dict[str, Any]:
    cfg = {**DEFAULT, **cfg0}
    dev = ctx.device
    seed = int(cfg['seed'])
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    qmodel = make_query(cfg).to(dev)
    keyside = KeySide(cfg, ctx)
    params = list(qmodel.parameters()) + ([] if keyside.net is None else list(keyside.net.parameters()))
    opt = torch.optim.AdamW(params, lr=float(cfg['lr']), weight_decay=1e-4)
    train = ctx.splits[cfg['rows']]
    mh = {}
    if cfg['mhistory']:
        for n in ('start', 'dev', 'learned', 'unseen'):
            mh[n] = motion_history_keys(ctx, ctx.splits[n])
    qn = len(train['key'])
    ids = np.arange(qn)
    dev_epochs = set(range(2, cfg['epochs'] + 1, 2)) | {cfg['epochs']} if cfg['dev_epochs'] == 'every2' else set(cfg['dev_epochs'])
    best, best_ep, best_state, hist = float('inf'), None, None, []
    t0 = time.time()
    nb = len(ctx.bt)
    for ep in range(1, cfg['epochs'] + 1):
        rng.shuffle(ids)
        qmodel.train()
        if keyside.net is not None:
            keyside.net.train()
        losses = []
        for s in range(0, qn, cfg['batch']):
            qi = ids[s:s + cfg['batch']]
            rand = rng.choice(nb, size=min(cfg['rand'], nb), replace=False)
            own = train['candidate_id'][qi]
            cids = np.unique(np.r_[own[own >= 0], rand])
            allowed = ~((train['task'][qi, None] == ctx.bt[cids][None, :]) & (train['episode'][qi, None] == ctx.be[cids][None, :]))
            qin = pack(train, qi, cfg, dev, rng=rng, augment=bool(cfg['aug']),
                       mhist=mh.get('start') if cfg['mhistory'] else None)
            q = qmodel(**qin)
            kt = keyside.keys(cids)
            scale = qmodel.logit_scale.detach() if cfg['scale_detached'] else qmodel.logit_scale
            sc = score(q, kt, keyside.score_mode(), scale)
            err = torch.as_tensor(k2._arm6_candidate_mse(np.asarray(train['actions'][qi]), ctx.windows[cids]), device=dev)
            am = torch.as_tensor(allowed, device=dev)
            loss = k2.soft_listwise_loss(sc, err, am, float(cfg['tau']))
            if cfg['same_scene']:
                same = (ctx.bt[cids][None, :] == train['task'][qi, None]) & (ctx.be[cids][None, :] != train['episode'][qi, None])
                sm = torch.as_tensor(same, device=dev) & am
                negid = err.masked_fill(~sm, -torch.inf).argmax(-1)
                posid = err.masked_fill(~am, torch.inf).argmin(-1)
                has = sm.any(-1)
                if has.any():
                    pos = sc.gather(1, posid[:, None]).squeeze(1); neg = sc.gather(1, negid[:, None]).squeeze(1)
                    loss = loss + .1 * F.softplus(neg[has] - pos[has] + .1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.)
            opt.step()
            with torch.no_grad():
                qmodel.logit_scale.clamp_(max=float(np.log(100.)))
            losses.append(float(loss.detach()))
        rec = {'epoch': ep, 'loss': float(np.mean(losses)), 'seconds': time.time() - t0}
        if ep in dev_epochs:
            sc, _ = score_split(qmodel, keyside, cfg, ctx, ctx.splits['dev'], mhist=mh.get('dev'))
            d = ctx.splits['dev']['errors'][np.arange(len(sc)), sc.argmax(1)]
            rec['dev_top1_arm6_mse'] = float(d.mean())
            if rec['dev_top1_arm6_mse'] < best:
                best, best_ep = rec['dev_top1_arm6_mse'], ep
                best_state = {'query': {k: v.detach().cpu().clone() for k, v in qmodel.state_dict().items()},
                              'key': None if keyside.net is None else {k: v.detach().cpu().clone() for k, v in keyside.net.state_dict().items()}}
        hist.append(rec)
        log('epoch', save_dir.parent if save_dir.name == 'rows' else save_dir, name=name, **rec)
    qmodel.load_state_dict(best_state['query'])
    if keyside.net is not None:
        keyside.net.load_state_dict(best_state['key'])
    ck = save_dir / f'{name}.pt'
    torch.save({'cfg': cfg, **best_state}, ck)
    row = {'name': name, 'cfg': cfg, 'selected_epoch': best_ep, 'dev_top1_arm6_mse': best, 'history': hist,
           'checkpoint': str(ck), 'train_rows': int(qn)}
    if final_eval:
        row['evaluation'] = evaluate_and_export(name, qmodel, keyside, cfg, ctx, save_dir, mh, save_scores)
    write_json(save_dir / f'{name}.json', row)
    return row


def evaluate_and_export(name, qmodel, keyside, cfg, ctx: RowContext, save_dir: Path, mh=None, save_scores=False):
    mh = mh or {}
    ev = {}
    for sname in ('dev', 'learned', 'unseen'):
        sp = ctx.splits[sname]
        sc, q = score_split(qmodel, keyside, cfg, ctx, sp, mhist=mh.get(sname))
        ev[sname] = metrics(sc, sp['errors'], sp, ctx.bt)
        if save_scores:
            o = np.argsort(-sc, 1)[:, :20]
            np.savez_compressed(save_dir / f'{name}__{sname}_scores.npz', top1_unit=sc.argmax(1).astype(np.int32),
                                top1_score=sc.max(1), top20_unit=o.astype(np.int32),
                                top20_score=np.take_along_axis(sc, o, 1), scores=sc.astype(np.float16),
                                query=q.astype(np.float32), key=np.asarray(sp['key']), task=sp['task'],
                                episode=sp['episode'], hard=np.asarray(sp['hard'], bool),
                                top1_arm6_mse=np.asarray(ev[sname]['per_query_top1'], np.float32),
                                bank_source=np.asarray(ctx.bank['source']))
    if save_scores:
        # recorded query for each memory unit: its start observation (history+state[+VLM]) through the query net
        _, qmem = score_split(qmodel, keyside, cfg, ctx, ctx.splits['start'], mhist=mh.get('start'))
        with torch.no_grad():
            kall = keyside.keys().float().cpu().numpy()
        np.savez_compressed(save_dir / f'{name}__memory_units.npz', recorded_query=qmem.astype(np.float32),
                            unit_key=kall.astype(np.float32), source=np.asarray(ctx.bank['source']),
                            task=ctx.bt, episode=ctx.be, length=np.asarray(ctx.bank['length']))
    return ev


def export_row(path: Path, ctx: RowContext, dest: Path, name: str):
    cfg, q, ks = load_row_model(path, ctx)
    mh = {n: motion_history_keys(ctx, ctx.splits[n]) for n in ('start', 'dev', 'learned', 'unseen')} if cfg['mhistory'] else {}
    return evaluate_and_export(name, q, ks, cfg, ctx, dest, mh, True)


# ----------------------------------------------------------------------------
# KEYGRID H+S two-tower reproduction (detached scale, its own RNG convention)
# ----------------------------------------------------------------------------

def train_keygrid_original(name: str, seed: int, lr: float, tau: float, ctx: RowContext, save_dir: Path,
                           epochs=(4, 8, 16)) -> dict[str, Any]:
    """Reimplements key_grid._train_one exactly, with Dev checks at the epoch grid (same RNG prefix)."""
    import key_grid as kg
    dev = ctx.device
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 101)
    train = ctx.splits['start']
    fit_q, fit_k = train['hs'], ctx.bank_hs
    model = kg.PairMLPRanker(88, shared=False).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n = len(train['key']); qids = np.arange(n); bs = 8; steps = int(np.ceil(n / bs))
    cand = train['candidate_id']
    rows = []
    for ep in range(1, max(epochs) + 1):
        model.train(); rng.shuffle(qids)
        for step in range(steps):
            qi = qids[step * bs:min((step + 1) * bs, n)]
            random_ids = rng.choice(len(fit_k), size=min(256, len(fit_k)), replace=False)
            cids, allowed = k2.contrastive_candidate_set(cand[qi], random_ids, train['task'][qi], train['episode'][qi], ctx.bt, ctx.be)
            q = torch.as_tensor(fit_q[qi], device=dev); k = torch.as_tensor(fit_k[cids], device=dev)
            sc = model(q, k) * model.logit_scale.detach().exp().clamp(max=k2.LOGIT_SCALE_MAX)
            err = k2._arm6_candidate_mse(np.asarray(train['actions'][qi]), ctx.windows[cids])
            loss = k2.contrastive_teacher_loss(sc, torch.as_tensor(err, device=dev), torch.as_tensor(allowed, device=dev),
                                               objective='soft', teacher_temperature=tau)
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.); opt.step()
            k2._clamp_logit_scale_(model)
        if ep in epochs:
            sc = kg._score_vectors(model, ctx.splits['dev']['hs'], ctx.bank_hs, device=str(dev))
            d = float(ctx.splits['dev']['errors'][np.arange(len(sc)), sc.argmax(1)].mean())
            ck = save_dir / f'{name}_ep{ep}.pt'
            torch.save(model.state_dict(), ck)
            rows.append({'epoch': ep, 'dev_top1_arm6_mse': d, 'checkpoint': str(ck)})
    out = {'name': name, 'seed': seed, 'lr': lr, 'tau': tau, 'grid': rows}
    write_json(save_dir / f'{name}.json', out)
    return out


def eval_keygrid_checkpoint(ck: str, ctx: RowContext) -> dict[str, Any]:
    import key_grid as kg
    model = kg.PairMLPRanker(88, shared=False).to(ctx.device)
    model.load_state_dict(torch.load(ck, map_location=ctx.device, weights_only=True))
    ev = {}
    for sname in ('dev', 'learned', 'unseen'):
        sp = ctx.splits[sname]
        sc = kg._score_vectors(model, sp['hs'], ctx.bank_hs, device=str(ctx.device))
        ev[sname] = metrics(sc, sp['errors'], sp, ctx.bt)
    return ev


# ----------------------------------------------------------------------------
# key-space nearest-neighbour upper bounds (key faithfulness)
# ----------------------------------------------------------------------------

def key_upper_bound(ctx: RowContext, key: str) -> dict[str, float]:
    """Encode the TRUE future chunk with the key encoder and take its nearest key (diagnostic only)."""
    out = {}
    for sname in ('learned', 'unseen'):
        sp = ctx.splits[sname]
        a = torch.as_tensor(np.asarray(sp['actions'])[..., :6], device=ctx.device, dtype=torch.float32)
        m = torch.ones(a.shape[:2], device=ctx.device, dtype=torch.bool)
        if key == 'dct':
            # same encoder as the stored keys: magnitude-free physical-6 DCT3 shape of the chunk
            import seg_context_retrieval as scr
            c = np.stack([scr.segment_value_encode(np.asarray(x, np.float32), coefficients=3)[0].reshape(18)
                          for x in np.asarray(sp['actions'])])
            c = (c - ctx.dct_mean) / ctx.dct_std
            k = np.asarray(ctx.fixed['dct'])
            sc = -((c[:, None, :] - k[None]) ** 2).sum(-1)
        else:
            enc = km.ArchMotionEncoder('x1', 'absolute').to(ctx.device)
            enc.load_state_dict(torch.load(X1_MK1B if key == 'x1_mk1b' else X1_MK1C, map_location=ctx.device, weights_only=True))
            enc.eval()
            with torch.no_grad():
                q = enc(a, m)[:, 0].cpu().numpy()
            sc = q @ np.asarray(ctx.fixed[key]).T
        top = sc.argmax(1)
        out[sname] = float(sp['errors'][np.arange(len(top)), top].mean())
    return out


def load_row_model(path: Path, ctx: RowContext):
    blob = torch.load(path, map_location=ctx.device, weights_only=False)
    cfg = blob['cfg']
    q = make_query(cfg).to(ctx.device); q.load_state_dict(blob['query'])
    ks = KeySide(cfg, ctx)
    if ks.net is not None:
        ks.net.load_state_dict(blob['key'])
    return cfg, q, ks


def input_ablation(path: Path, ctx: RowContext, seed: int = 9001) -> dict[str, Any]:
    """Which query input carries the retrieval? Score Eval with one input destroyed at test time."""
    cfg, q, ks = load_row_model(path, ctx)
    out = {}
    for sname in ('learned', 'unseen'):
        sp = ctx.splits[sname]
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(sp['key']))
        variants = {
            'intact': sp,
            'history_zeroed': {**sp, 'history': np.zeros_like(np.asarray(sp['history'])),
                               'history_mask': np.zeros_like(np.asarray(sp['history_mask']))},
            'history_xyz_reversed': {**sp, 'history': np.asarray(sp['history']) * np.r_[-1, -1, -1, 1, 1, 1, 1].astype(np.float32)},
            'state_shuffled': {**sp, 'state': np.asarray(sp['state'])[perm]},
        }
        if cfg['input'] == 'e_vlm_h':
            variants['vlm_shuffled'] = {**sp, 'tokens': _Perm(sp['tokens'], perm), 'groups': _Perm(sp['groups'], perm)}
        res = {}
        mh = motion_history_keys(ctx, sp) if cfg['mhistory'] else None
        for vname, v in variants.items():
            sc, _ = score_split(q, ks, cfg, ctx, v, mhist=mh)
            res[vname] = float(sp['errors'][np.arange(len(sc)), sc.argmax(1)].mean())
        out[sname] = res
    return out


class _Perm:
    def __init__(self, arr, perm):
        self.arr, self.perm = arr, np.asarray(perm)

    def __getitem__(self, ix):
        return np.asarray(self.arr[np.sort(self.perm[ix])])[np.argsort(np.argsort(self.perm[ix]))]


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=('prep', 'rows', 'keygrid'))
    ap.add_argument('--plan', type=Path, help='JSON file {name: cfg}')
    ap.add_argument('--names', nargs='*', default=None)
    ap.add_argument('--out', type=Path, default=OUT)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--save-scores', action='store_true')
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '8')))
    if a.cmd == 'prep':
        return prep_data(a.out)
    plan = json.loads(a.plan.read_text())
    names = a.names or list(plan)
    sub = a.out / ('rows' if a.cmd == 'rows' else 'keygrid')
    sub.mkdir(parents=True, exist_ok=True)
    need_tokens = any(plan[n].get('input') == 'e_vlm_h' for n in names)
    ctx = RowContext(a.out, torch.device(a.device), need_tokens)
    for n in names:
        if (sub / f'{n}.json').exists():
            continue
        log('row_start', a.out, name=n, cfg=plan[n])
        try:
            if a.cmd == 'rows':
                r = train_row(n, plan[n], ctx, save_dir=sub, save_scores=a.save_scores or plan[n].get('save_scores', False))
                log('row_complete', a.out, name=n, dev=r['dev_top1_arm6_mse'], epoch=r['selected_epoch'],
                    unseen=r['evaluation']['unseen']['all']['top1_arm6_mse'],
                    learned=r['evaluation']['learned']['all']['top1_arm6_mse'])
            else:
                c = plan[n]
                r = train_keygrid_original(n, int(c['seed']), float(c['lr']), float(c['tau']), ctx, sub)
                log('keygrid_complete', a.out, name=n, grid=r['grid'])
        except Exception as e:  # keep the worker alive for other rows
            log('row_failed', a.out, name=n, error=repr(e))
            raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
