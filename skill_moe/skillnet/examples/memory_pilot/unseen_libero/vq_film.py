#!/usr/bin/env python3
"""VQ-A: VLM information bottleneck x FiLM on the C1b history+state query.

Query  : C1b tower h = MLP(history10 + mask + state) (88 -> 512 -> 256), then
         h' = gamma * h + beta with (gamma, beta) = g(z) initialised to identity
         (gamma = 1, beta = 0; last layer of g is zero), q = normalize(h').
z      : bottlenecked summary of the frozen pi0.5 prefix tokens (532 x 2048):
         'resampler' = Perceiver-style K learned latents x cross-attention -> K x d,
         'pooled'    = masked mean -> MLP -> d.  Optional VIB (Gaussian z, KL to N(0,I)).
Key    : X1 (MK1b, tau .01) fixed.  Training recipe = C1b (key_final row
         H_x1_mk1b_cteff_tau0.01): LIBERO-40 stride-5 memory frames (12,766), teacher
         softmax(-armMSE/.01), soft InfoNCE (batch 8 + random 256, own episode
         excluded) + same-scene hard negative margin, lr 1e-4, 16 epochs, no
         observation augmentation.  Epoch / setting selection on Dev v2 only.

Commands: prep (join eh VLM tokens), validate (Dev v2 vs Eval rank check on KQ-final
rows), rows --plan (train configs), diag (why-analysis).
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
import key_final as kf  # noqa: E402  (read-only)
import key_shared_latent as k2  # noqa: E402  (read-only)
import dev_v2  # noqa: E402

ROOT = kf.ROOT
OUT = ROOT / 'vq-a-01'
KQ = kf.OUT
EVAL_SPLITS = ('dev2', 'dev2_10', 'learned', 'unseen20', 'unseen')
VQC = ROOT / 'vq-c-01'


def log(event: str, out: Path = OUT, **kw: Any) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'progress.log').open('a') as f:
        f.write(json.dumps({'time': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'stage': event, **kw},
                           ensure_ascii=False) + '\n')


write_json = kf.write_json


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------

def prep_eh_tokens(out: Path = OUT) -> None:
    """Join exact pi0.5 prefix tokens for the 12,766 C1b training frames (read-only caches)."""
    d = out / 'data'; d.mkdir(parents=True, exist_ok=True)
    if (d / 'eh_tokens.done').exists():
        return
    keys = np.load(KQ / 'data' / 'eh__key.npy')
    store = k2.ExactFeatureStore(kf.RUN_ROOT, kf.MISSING_ROOT, kf.SEEN5, kf.UNSEEN5)
    tp, gp = d / 'eh_tokens.npy', d / 'eh_groups.npy'
    mode = 'r+' if tp.exists() else 'w+'
    tok = np.lib.format.open_memmap(tp, mode=mode, dtype=np.float16, shape=(len(keys), 532, 2048))
    grp = np.lib.format.open_memmap(gp, mode=mode, dtype=np.int8, shape=(len(keys), 532))
    prog = d / 'eh_tokens.progress'
    done = int(prog.read_text()) if prog.exists() else 0
    missing = []
    for s in range(done, len(keys), 256):
        ks = keys[s:s + 256]
        try:
            t, g = store.get(ks, domain='libero40')
        except KeyError:
            t = np.zeros((len(ks), 532, 2048), np.float16); g = np.full((len(ks), 532), -1, np.int8)
            for j, k in enumerate(ks):
                try:
                    a, b = store.get(k[None], domain='libero40'); t[j], g[j] = a[0], b[0]
                except KeyError:
                    missing.append(k.tolist())
        tok[s:s + len(ks)] = t; grp[s:s + len(ks)] = g
        tok.flush(); grp.flush(); prog.write_text(str(s + len(ks)))
        log('eh_tokens', out, completed=s + len(ks), total=len(keys))
    write_json(d / 'eh_tokens_manifest.json', {'rows': int(len(keys)), 'missing': missing})
    (d / 'eh_tokens.done').write_text('ok\n')


class Ctx(kf.RowContext):
    """key_final.RowContext + Dev v2 split + eh tokens + subset masks."""
    def __init__(self, device, need_tokens: bool, out: Path = OUT):
        super().__init__(KQ, device, need_tokens)
        # dev2 = Dev v2 at the memory rate (20 Hz, selection); dev2_10 = first (10 Hz) build, kept for continuity.
        # unseen20 / dev_old20 = VQ-C's 20 Hz rebuilds of the locked unseen 619 / old Dev 217 (tokens reused by row).
        self.splits['dev2'] = dev_v2.load_dev_v2_20(tokens=need_tokens)
        self.splits['dev2_10'] = dev_v2.load_dev_v2(tokens=need_tokens)
        for name, src, base in (('unseen20', 'unseen20', 'unseen'), ('dev_old20', 'dev20', 'dev')):
            d = VQC / src
            sp = {k: np.load(d / f'{k}.npy') for k in ('key', 'task', 'episode', 'hard', 'history', 'history_mask', 'state', 'actions', 'row')}
            sp['task'] = sp['task'].astype(np.int64); sp['episode'] = sp['episode'].astype(np.int64)
            if need_tokens:
                sp['tokens'] = dev_v2._Rows(self.splits[base]['tokens'], sp['row'])
                sp['groups'] = np.asarray(self.splits[base]['groups'])[sp['row']]
            self.splits[name] = sp
        for n in ('dev2', 'dev2_10', 'unseen20', 'dev_old20'):
            s = self.splits[n]
            s['errors'] = k2._arm6_candidate_mse(s['actions'], self.windows)
            s['hs'] = kf.hs_features(s)
        if need_tokens and (out / 'data' / 'eh_tokens.done').exists():
            self.splits['eh']['tokens'] = np.load(out / 'data' / 'eh_tokens.npy', mmap_mode='r')
            self.splits['eh']['groups'] = np.load(out / 'data' / 'eh_groups.npy', mmap_mode='r')
        self.subsets = {}
        for n in ('dev2', 'dev2_10', 'learned', 'unseen20', 'unseen'):
            sp = self.splits[n]
            self.subsets[n] = ({k[4:]: sp[k] for k in sp if k.startswith('sub_')} if n.startswith('dev2')
                               else dev_v2.eval_subsets(sp))


# ----------------------------------------------------------------------------
# model
# ----------------------------------------------------------------------------

class Resampler(nn.Module):
    def __init__(self, k: int, d: int, width: int = 256, heads: int = 4, vib: bool = False):
        super().__init__()
        self.ln = nn.LayerNorm(2048)
        self.proj = nn.Linear(2048, width)
        self.lat = nn.Parameter(torch.randn(k, width) * .02)
        self.att = nn.MultiheadAttention(width, heads, batch_first=True)
        self.ff = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 2), nn.GELU(), nn.Linear(width * 2, width))
        self.out = nn.Linear(width, d * (2 if vib else 1))
        self.k, self.d, self.vib = k, d, vib
        self.last_attn = None

    def forward(self, tokens, groups):
        x = self.proj(self.ln(tokens.float()))
        pad = groups < 0
        q = self.lat[None].expand(len(x), -1, -1)
        y, w = self.att(q, x, x, key_padding_mask=pad, need_weights=True)
        self.last_attn = w.detach()
        y = y + self.lat[None]
        y = y + self.ff(y)
        return self.out(y)  # [B, K, d or 2d]


class PooledZ(nn.Module):
    def __init__(self, d: int, vib: bool = False):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(2048), nn.Linear(2048, 256), nn.GELU(), nn.Linear(256, d * (2 if vib else 1)))
        self.k, self.d, self.vib = 1, d, vib

    def forward(self, tokens, groups):
        v = (groups >= 0).float()[..., None]
        p = (tokens.float() * v).sum(1) / v.sum(1).clamp(min=1)
        return self.net(p)[:, None]


class FiLMQuery(nn.Module):
    """C1b tower + FiLM from a bottlenecked VLM summary z.  bottleneck='none' == C1b."""
    def __init__(self, bottleneck: str = 'none', k: int = 1, d: int = 32, vib: bool = False, film_at: str = 'out',
                 keep_groups=None):
        super().__init__()
        self.keep_groups = None if keep_groups is None else tuple(int(x) for x in keep_groups)  # 0 base, 1 wrist, 2 language
        self.tower = nn.Sequential(nn.Linear(88, 512), nn.GELU(), nn.Linear(512, 256))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / .07)))
        self.bottleneck, self.vib, self.film_at = bottleneck, vib, film_at
        self.z = None
        if bottleneck == 'resampler':
            self.z = Resampler(k, d, vib=vib)
        elif bottleneck == 'pooled':
            self.z = PooledZ(d, vib=vib)
        if self.z is not None:
            fd = 256 if film_at == 'out' else 512
            self.film = nn.Sequential(nn.Linear(self.z.k * d, 256), nn.GELU(), nn.Linear(256, 2 * fd))
            nn.init.zeros_(self.film[-1].weight); nn.init.zeros_(self.film[-1].bias)
        self.kl = torch.zeros(())
        self.last = {}

    def encode_z(self, tokens, groups):
        r = self.z(tokens, groups)
        if self.vib:
            mu, lv = r.chunk(2, -1); lv = lv.clamp(-8, 8)
            self.kl = .5 * (mu.square() + lv.exp() - lv - 1).sum((1, 2)).mean()
            z = mu + torch.randn_like(mu) * (.5 * lv).exp() if self.training else mu
        else:
            z = r; self.kl = torch.zeros((), device=r.device)
        return z.flatten(1)

    def forward(self, *, hs, tokens=None, groups=None, **_: Any):
        if self.z is None:
            return F.normalize(self.tower(hs.float()), dim=-1)
        if self.keep_groups is not None:
            keep = torch.zeros_like(groups, dtype=torch.bool)
            for g_ in self.keep_groups:
                keep |= groups == g_
            groups = groups.masked_fill(~keep, -1)
        z = self.encode_z(tokens, groups)
        gb = self.film(z); dg, b = gb.chunk(2, -1)
        if self.film_at == 'out':
            h = self.tower(hs.float()); h2 = (1 + dg) * h + b
        else:
            a = self.tower[1](self.tower[0](hs.float())); h = a
            h2 = self.tower[2]((1 + dg) * a + b)
        self.last = {'z': z.detach(), 'dgamma': dg.detach(), 'beta': b.detach(), 'h': h.detach()}
        return F.normalize(h2, dim=-1)


def pack(split, ids, device, need_tokens):
    ix = np.asarray(ids, np.int64)
    hist = np.asarray(split['history'][ix], np.float32); mask = np.asarray(split['history_mask'][ix], bool)
    hs = np.concatenate([hist.reshape(len(ix), -1), mask.astype(np.float32), np.asarray(split['state'][ix], np.float32)], -1)
    out = {'hs': torch.as_tensor(hs, device=device)}
    if need_tokens:
        o = np.argsort(ix); six = ix[o]; inv = np.argsort(o)
        out['tokens'] = torch.as_tensor(np.asarray(split['tokens'][six])[inv], device=device, dtype=torch.float32)
        out['groups'] = torch.as_tensor(np.asarray(split['groups'][six])[inv], device=device, dtype=torch.long)
    return out


DEFAULT = dict(bottleneck='none', k=1, d=32, vib=False, kl=0.0, film_at='out', keep_groups=None, tau=.01, seed=4101, epochs=16,
               batch=8, rand=256, lr=1e-4, same_scene=True, dev_split='dev2')


@torch.no_grad()
def score_split(q, keys, ctx, split, need_tokens, batch=64, collect=False):
    q.eval()
    out, qs, extra = [], [], {'z': [], 'dgamma': [], 'beta': [], 'h': []}
    for s in range(0, len(split['key']), batch):
        ids = np.arange(s, min(s + batch, len(split['key'])))
        v = q(**pack(split, ids, ctx.device, need_tokens))
        qs.append(v.float().cpu().numpy())
        out.append((v @ keys.T * q.logit_scale.exp().clamp(max=100.)).float().cpu().numpy())
        if collect and q.last:
            for k_ in extra:
                extra[k_].append(q.last[k_].float().cpu().numpy())
    sc = np.concatenate(out)
    allowed = ~((split['task'][:, None] == ctx.bt[None, :]) & (split['episode'][:, None] == ctx.be[None, :]))
    sc[~allowed] = -1e9
    ex = {k_: np.concatenate(v) for k_, v in extra.items() if v} if collect else {}
    return sc, np.concatenate(qs), ex


def metrics(sc, split, sub, bt):
    m = kf.metrics(sc, split['errors'], split, bt)
    per = np.asarray(m['per_query_top1'])
    m['subsets'] = {n: {'rows': int(v.sum()), 'top1_arm6_mse': float(per[v].mean()) if v.any() else None}
                    for n, v in sub.items()}
    return m


def train_row(name, cfg0, ctx, save_dir: Path):
    cfg = {**DEFAULT, **cfg0}
    dev = ctx.device; seed = int(cfg['seed'])
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    need = cfg['bottleneck'] != 'none'
    q = FiLMQuery(cfg['bottleneck'], int(cfg['k']), int(cfg['d']), bool(cfg['vib']), cfg['film_at'], cfg['keep_groups']).to(dev)
    keys = torch.as_tensor(np.asarray(ctx.fixed['x1_mk1b']), device=dev, dtype=torch.float32)
    opt = torch.optim.AdamW(q.parameters(), lr=float(cfg['lr']), weight_decay=1e-4)
    train = ctx.splits['eh']; qn = len(train['key']); ids = np.arange(qn); nb = len(ctx.bt)
    dev_epochs = set(range(2, cfg['epochs'] + 1, 2)) | {cfg['epochs']}
    best, best_ep, best_state, hist = float('inf'), None, None, []
    t0 = time.time()
    for ep in range(1, cfg['epochs'] + 1):
        rng.shuffle(ids); q.train(); losses, kls = [], []
        for s in range(0, qn, cfg['batch']):
            qi = ids[s:s + cfg['batch']]
            rand = rng.choice(nb, size=min(cfg['rand'], nb), replace=False)
            own = train['candidate_id'][qi]
            cids = np.unique(np.r_[own[own >= 0], rand])
            allowed = ~((train['task'][qi, None] == ctx.bt[cids][None, :]) & (train['episode'][qi, None] == ctx.be[cids][None, :]))
            v = q(**pack(train, qi, dev, need))
            sc = (v @ keys[torch.as_tensor(cids, device=dev)].T) * q.logit_scale.exp().clamp(max=100.)
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
            if cfg['vib'] and cfg['kl'] > 0:
                loss = loss + float(cfg['kl']) * q.kl
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(list(q.parameters()), 1.); opt.step()
            with torch.no_grad():
                q.logit_scale.clamp_(max=float(np.log(100.)))
            losses.append(float(loss.detach())); kls.append(float(q.kl.detach()))
        rec = {'epoch': ep, 'loss': float(np.mean(losses)), 'kl': float(np.mean(kls)), 'seconds': time.time() - t0}
        if ep in dev_epochs:
            sp = ctx.splits[cfg['dev_split']]
            sc, _, _ = score_split(q, keys, ctx, sp, need)
            rec['dev_top1_arm6_mse'] = float(sp['errors'][np.arange(len(sc)), sc.argmax(1)].mean())
            if rec['dev_top1_arm6_mse'] < best:
                best, best_ep = rec['dev_top1_arm6_mse'], ep
                best_state = {k_: v_.detach().cpu().clone() for k_, v_ in q.state_dict().items()}
        hist.append(rec)
        log('epoch', save_dir.parent, name=name, **rec)
    q.load_state_dict(best_state)
    ck = save_dir / f'{name}.pt'
    torch.save({'cfg': cfg, 'query': best_state}, ck)
    ev = {}
    for sn in EVAL_SPLITS:
        sp = ctx.splits[sn]
        sc, _, ex = score_split(q, keys, ctx, sp, need, collect=True)
        ev[sn] = metrics(sc, sp, ctx.subsets.get(sn, {}), ctx.bt)
        if ex:
            ev[sn]['film'] = film_stats(ex)
            np.savez_compressed(save_dir / f'{name}__{sn}_z.npz', **{k_: v_.astype(np.float16) for k_, v_ in ex.items()})
    row = {'name': name, 'cfg': cfg, 'selected_epoch': best_ep, 'dev_top1_arm6_mse': best, 'history': hist,
           'checkpoint': str(ck), 'train_rows': int(qn), 'evaluation': ev}
    write_json(save_dir / f'{name}.json', row)
    return row


def film_stats(ex):
    dg, b, h = ex['dgamma'].astype(np.float64), ex['beta'].astype(np.float64), ex['h'].astype(np.float64)
    hn = np.linalg.norm(h, axis=1)
    mod = np.linalg.norm(dg * h + b, axis=1)
    return {'mean_abs_dgamma': float(np.abs(dg).mean()), 'mean_abs_beta': float(np.abs(b).mean()),
            'rel_change_norm': float(np.mean(mod / np.maximum(hn, 1e-8))),
            'dgamma_across_row_sd': float(dg.std(0).mean()), 'beta_across_row_sd': float(b.std(0).mean())}


def load_model(path: Path, device):
    blob = torch.load(path, map_location=device, weights_only=False)
    c = {**DEFAULT, **blob['cfg']}
    q = FiLMQuery(c['bottleneck'], int(c['k']), int(c['d']), bool(c['vib']), c['film_at'], c['keep_groups']).to(device)
    q.load_state_dict(blob['query'])
    return c, q


# ----------------------------------------------------------------------------
# Dev v2 validation (rank agreement with unseen Eval on existing KQ-final rows)
# ----------------------------------------------------------------------------

def validate(out: Path = OUT, device='cuda:0') -> dict[str, Any]:
    from scipy.stats import spearmanr, kendalltau
    vd = dev_v2.OUT
    ctx = Ctx(torch.device(device), need_tokens=True)
    sp = ctx.splits['dev2']
    rows = sorted((KQ / 'rows').glob('*.json'))
    res_path = vd / 'validation20_rows.json'
    res = json.loads(res_path.read_text()) if res_path.exists() else {}
    for p in rows:
        if p.stem in res:
            continue
        r = json.loads(p.read_text())
        if 'evaluation' not in r or not Path(r['checkpoint']).exists():
            continue
        cfg, qm, ks = kf.load_row_model(Path(r['checkpoint']), ctx)
        mh = kf.motion_history_keys(ctx, sp) if cfg['mhistory'] else None
        sc, _ = kf.score_split(qm, ks, cfg, ctx, sp, mhist=mh)
        per = sp['errors'][np.arange(len(sc)), sc.argmax(1)]
        sub = {n: float(per[m].mean()) for n, m in ctx.subsets['dev2'].items()}
        early = np.asarray(sp['frame']) <= 80
        extra = {}
        for nm in ('unseen20', 'dev_old20'):
            s2 = ctx.splits[nm]
            mh2 = kf.motion_history_keys(ctx, s2) if cfg['mhistory'] else None
            sc2, _ = kf.score_split(qm, ks, cfg, ctx, s2, mhist=mh2)
            extra[nm] = float(s2['errors'][np.arange(len(sc2)), sc2.argmax(1)].mean())
        res[p.stem] = {'dev2': float(per.mean()), 'dev2_frame_le80': float(per[early].mean()), 'dev2_subsets': sub,
                       'unseen20': extra['unseen20'], 'dev_old20': extra['dev_old20'], 'unseen10': r['evaluation']['unseen']['all']['top1_arm6_mse'],
                       'dev_old': r['evaluation']['dev']['all']['top1_arm6_mse'],
                       'learned': r['evaluation']['learned']['all']['top1_arm6_mse'],
                       'aug': bool(r['cfg'].get('aug')), 'per_task': {int(t): float(per[sp['task'] == t].mean()) for t in np.unique(sp['task'])}}
        write_json(res_path, res)
        log('validate_row', vd, name=p.stem, **{k_: v_ for k_, v_ in res[p.stem].items() if k_ != 'per_task' and k_ != 'dev2_subsets'})
        torch.cuda.empty_cache()
    names = sorted(res)
    target = 'unseen20'

    def corr(keyd, sel):
        x = np.array([res[n][keyd] for n in sel]); y = np.array([res[n][target] for n in sel])
        return {'n': len(sel), 'spearman': float(spearmanr(x, y).statistic), 'kendall': float(kendalltau(x, y).statistic)}
    groups = {'all': names, 'no_aug': [n for n in names if not res[n]['aug']],
              'B_mk1c_orig': [n for n in names if n.startswith('B') and res[n]['aug']],
              'H_mlp': [n for n in names if n.startswith('H_')]}
    summ = {}
    for g, sel in groups.items():
        if len(sel) >= 4:
            summ[g] = {'dev_old10_vs_unseen20': corr('dev_old', sel), 'dev_old20_vs_unseen20': corr('dev_old20', sel),
                       'dev2_20_vs_unseen20': corr('dev2', sel), 'dev2_20_frame_le80': corr('dev2_frame_le80', sel)}
    # leave-one-task-out stability of dev2 ranking
    tasks = sorted(next(iter(res.values()))['per_task'])
    lo = []
    for t in tasks:
        x = []
        for n in names:
            pt = res[n]['per_task']; cnt = {int(k_): int((sp['task'] == int(k_)).sum()) for k_ in pt}
            num = sum(pt[k_] * cnt[int(k_)] for k_ in pt if int(k_) != int(t)); den = sum(cnt[int(k_)] for k_ in pt if int(k_) != int(t))
            x.append(num / den)
        lo.append(float(spearmanr(x, [res[n]['unseen20'] for n in names]).statistic))
    summ['dev2_leave_one_task_out_spearman'] = {'min': min(lo), 'max': max(lo)}
    # pairwise decision accuracy on close pairs (|unseen diff| >= .003)
    def pair_acc(keyd):
        ok = tot = 0
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = res[names[i]], res[names[j]]
                if abs(a['unseen20'] - b['unseen20']) < .003:
                    continue
                tot += 1; ok += (a[keyd] - b[keyd]) * (a['unseen20'] - b['unseen20']) > 0
        return {'pairs': tot, 'agree': ok / max(tot, 1)}
    summ['pairwise_agreement'] = {'dev_old10': pair_acc('dev_old'), 'dev_old20': pair_acc('dev_old20'), 'dev2_20': pair_acc('dev2')}
    x = np.array([res[n]['unseen10'] for n in names]); y = np.array([res[n]['unseen20'] for n in names])
    summ['unseen10_vs_unseen20_spearman'] = float(spearmanr(x, y).statistic)
    write_json(vd / 'validation20.json', summ)
    return summ


# ----------------------------------------------------------------------------
# rescoring KQ-final rows (C1b / concat E-VLM+H) on subsets
# ----------------------------------------------------------------------------

def rescore_kq(names, ctx, out: Path = OUT):
    res = {}
    for n in names:
        r = json.loads((KQ / 'rows' / f'{n}.json').read_text())
        cfg, qm, ks = kf.load_row_model(Path(r['checkpoint']), ctx)
        ev = {}
        for sn in EVAL_SPLITS:
            sp = ctx.splits[sn]
            sc, _ = kf.score_split(qm, ks, cfg, ctx, sp)
            ev[sn] = metrics(sc, sp, ctx.subsets.get(sn, {}), ctx.bt)
        res[n] = ev
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=('prep', 'validate', 'rows', 'rescore'))
    ap.add_argument('--plan', type=Path)
    ap.add_argument('--names', nargs='*')
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '8')))
    if a.cmd == 'prep':
        return prep_eh_tokens()
    if a.cmd == 'validate':
        print(json.dumps(validate(device=a.device), indent=1)); return
    if a.cmd == 'rescore':
        ctx = Ctx(torch.device(a.device), need_tokens=True)
        write_json(OUT / 'rescore_kq.json', rescore_kq(a.names, ctx)); return
    plan = json.loads(a.plan.read_text())
    names = a.names or list(plan)
    sub = OUT / 'rows20'; sub.mkdir(parents=True, exist_ok=True)  # rows/ = first grid, selected on the 10 Hz Dev
    need = any(plan[n].get('bottleneck', 'none') != 'none' for n in names)
    ctx = Ctx(torch.device(a.device), need_tokens=need)
    for n in names:
        if (sub / f'{n}.json').exists():
            continue
        lock = sub / f'{n}.lock'
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.close(fd)
        except FileExistsError:
            continue
        log('row_start', OUT, name=n, cfg=plan[n])
        try:
            r = train_row(n, plan[n], ctx, sub)
            log('row_complete', OUT, name=n, dev2=r['dev_top1_arm6_mse'], epoch=r['selected_epoch'],
                unseen20=r['evaluation']['unseen20']['all']['top1_arm6_mse'], unseen10=r['evaluation']['unseen']['all']['top1_arm6_mse'],
                learned=r['evaluation']['learned']['all']['top1_arm6_mse'])
        except Exception as e:
            log('row_failed', OUT, name=n, error=repr(e)); lock.unlink(missing_ok=True); raise
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
