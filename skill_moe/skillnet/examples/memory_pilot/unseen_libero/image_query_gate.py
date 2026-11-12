"""I2: learn to fall back on the image when the action history points the wrong way.

Score of candidate k for a query x:
    P5aug   s * cos(q(hist, state), key_k)                        (S1's P5 network, trained with corrupted history)
    G       a(x) * z(P5net(hist, state))_k + b(x) * z(nn)_k      (P5net trained jointly)
    Gfz     same, P5net frozen = S1's selected P5 model of the seed (only the gate is trained)
  nn_k   = -(S-shared type error between pi0.5's chunk and candidate k)   (training-free, image only)
  z(.)   = per-row z-score over the allowed candidates
  a, b   = softplus(MLP([hist, state, pi0.5 chunk, agreement features]))
  agreement = cosines / norms of the history's recent translation & rotation vs pi0.5's first steps, has-history flag.
Corrupted history during training (state kept): none .5, swap (random other row) .2, reverse translation .15, zero .15.
Loss: 'ce' = S1's soft teacher (frac 1/10, S1's selected value for every W|S-shared model), 'corr' = ce + (1 - Pearson
between the scores and -type error over the allowed candidates).
Selection (declared before Eval): epoch with the best LIBERO-40 holdout-task Spearman, holdout rows half true / half
corrupted with a fixed seed.  Eval: true / swap / reverse / zero history (image_query_robust.py's corruption).
Eval targets are read only for scoring.  Output: image-query-01/gate/.
"""
from __future__ import annotations

import json, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path: sys.path.insert(0, str(HERE))
import image_query as iq  # noqa: E402

G = iq.OUT / 'gate'
FRAC = 0.1
PROBS = (('none', .5), ('swap', .2), ('reverse', .15), ('zero', .15))


def nn_train_scores():
    p = G / 'nn_train.npy'
    if p.exists(): return np.load(p, mmap_mode='r')
    G.mkdir(parents=True, exist_ok=True); s = iq.nn_scores('train').astype(np.float32); np.save(p, s); return s


def agreement(hist, act):
    """[N, 7] agreement features (numpy or torch), normalized action space."""
    import torch
    th = torch if isinstance(hist, torch.Tensor) else np
    def cos(u, v):
        nu = th.sqrt((u ** 2).sum(-1) + 1e-8); nv = th.sqrt((v ** 2).sum(-1) + 1e-8)
        return (u * v).sum(-1) / nu / nv, nu, nv
    h = hist[:, -3:]; a = act[:, :3]
    ct, nht, nat = cos(h[..., :3].mean(1), a[..., :3].mean(1)); cr, nhr, nar = cos(h[..., 3:6].mean(1), a[..., 3:6].mean(1))
    has = (th.abs(hist).sum((1, 2)) > 0) * 1.0
    return th.stack([ct, cr, nht, nat, nhr, nar, has], 1)


def corrupt(hist, mode, rng, task=None):
    h = hist.copy()
    if mode == 'swap':
        n = len(h); perm = np.empty(n, np.int64)
        for i in range(n):
            c = np.flatnonzero(task != task[i]) if task is not None else np.arange(n); perm[i] = rng.choice(c)
        h = hist[perm]
    elif mode == 'reverse': h[:, :, :3] *= -1
    elif mode == 'zero': h[:] = 0
    return h


def make_model(kind, in_q, key_dim, in_g):
    import torch, torch.nn as nn
    import scale_only_query as so
    F = nn.functional

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.p5 = so.make_net('W', 'P5', in_q, key_dim)
            if kind != 'P5aug':
                self.gate = nn.Sequential(nn.Linear(in_g, 256), nn.GELU(), nn.Linear(256, 2))
                nn.init.zeros_(self.gate[-1].weight); nn.init.constant_(self.gate[-1].bias, 1.0)

        def forward(self, xq, kin, xg=None, znn=None, allowed=None):
            s = self.p5(xq, kin)
            if kind == 'P5aug': return s
            ab = F.softplus(self.gate(xg)) * 3
            return ab[:, :1] * zt(s, allowed) + ab[:, 1:] * znn
    return M()


def zt(s, A):
    import torch
    m = A.float(); n = m.sum(1, keepdim=True)
    mu = (s * m).sum(1, keepdim=True) / n; sd = torch.sqrt((((s - mu) * m) ** 2).sum(1, keepdim=True) / n + 1e-9)
    return (s - mu) / sd


def spearman_t(sc, E, A):
    """Per-row Spearman over allowed candidates (GPU; no ties)."""
    import torch
    lo = torch.finfo(sc.dtype).min
    r1 = sc.masked_fill(~A, lo).argsort(1).argsort(1).float(); r2 = (-E).masked_fill(~A, lo).argsort(1).argsort(1).float()
    return pearson_t(r1, r2, A)


def pearson_t(x, y, A):
    import torch
    m = A.float(); n = m.sum(1)
    xm = x - (x * m).sum(1, keepdim=True) / n[:, None]; ym = y - (y * m).sum(1, keepdim=True) / n[:, None]
    return (xm * ym * m).sum(1) / torch.sqrt((xm ** 2 * m).sum(1) * (ym ** 2 * m).sum(1) + 1e-12)


def load(split):
    import scale_only_query as so
    P = iq.S1 / 'prep'; d = dict(np.load(P / f'W_inputs_{split}.npz')); pool = dict(np.load(P / 'W_pool.npz'))
    act = np.load(iq._pf(split) / f'{split}_actions.npy').astype(np.float32)
    return d, pool, act


def train_one(kind, loss, seed, epochs=20, device='cuda'):
    import torch, scale_only_query as so
    path = G / f'{kind}_{loss}_s{seed}.pt'
    if path.exists(): return torch.load(path, weights_only=False)
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    d, pool, act = load('train'); P = iq.S1 / 'prep'
    KI = torch.as_tensor(so.key_inputs('W', pool, iq.COND), device=device)
    E = np.load(P / f'W_err_{iq.COND}_train.npz')['err']; A = so.allowed_matrix(d['task'], pool['task'], pool['keep'], True)
    hold = np.isin(d['task'], so.HOLDOUT); fit, va = np.flatnonzero(~hold), np.flatnonzero(hold)
    tau = so.teacher_tau(E[fit], A[fit], FRAC)
    Et = torch.as_tensor(np.where(A, E, 1e3).astype(np.float32), device=device); At = torch.as_tensor(A, device=device)
    NN = torch.as_tensor(np.asarray(nn_train_scores()), device=device); ZNN = zt(NN, At)
    hist0 = torch.as_tensor(d['hist'], device=device); state = torch.as_tensor(d['state'], device=device)
    actt = torch.as_tensor(act, device=device); task = torch.as_tensor(d['task'], device=device)
    gmu = torch.cat([actt.reshape(len(actt), -1)], 1).mean(0); gsd = actt.reshape(len(actt), -1).std(0) + 1e-6

    def xs(h, idx):
        xq = torch.cat([h.reshape(len(h), -1), state[idx]], 1)
        xg = torch.cat([xq, (actt[idx].reshape(len(idx), -1) - gmu) / gsd, agreement(h, actt[idx])], 1)
        return xq, xg
    # fixed corrupted holdout for selection: half true, half corrupted (modes cycle)
    vr = np.random.default_rng(777); hv = d['hist'][va].copy(); modes = vr.choice(['swap', 'reverse', 'zero'], len(va))
    keep_true = vr.random(len(va)) < .5
    for mname in ('swap', 'reverse', 'zero'):
        sel = (~keep_true) & (modes == mname)
        if sel.any(): hv[sel] = corrupt(d['hist'][va], mname, vr, d['task'][va])[sel]
    hv = torch.as_tensor(hv, device=device); vat = torch.as_tensor(va, device=device)
    net = make_model(kind, 78, KI.shape[1], 78 + 70 + 7).to(device)
    if kind == 'Gfz':
        sel_ = json.loads((iq.S1 / 'selection.json').read_text())['selected'][f'W|{iq.COND}|P5|{seed}']['holdout']
        r = torch.load(iq.S1 / 'models' / f"W_{iq.COND}_P5_s{seed}_f{sel_['frac']:.4f}.pt", weights_only=False)
        net.p5.load_state_dict(r['holdout']['state'])
        for p_ in net.p5.parameters(): p_.requires_grad_(False)
    opt = torch.optim.AdamW([p_ for p_ in net.parameters() if p_.requires_grad], 1e-3, weight_decay=1e-4)
    nb = (len(fit) + 255) // 256; sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=epochs * nb, pct_start=.05)
    names, pr = zip(*PROBS); best = (-np.inf, None, 0); curve = []
    for ep in range(epochs):
        net.train(); perm = rng.permutation(fit)
        for i in range(0, len(perm), 256):
            b = torch.as_tensor(perm[i:i + 256], device=device); h = hist0[b].clone()
            m = rng.choice(len(names), len(b), p=pr); mt = torch.as_tensor(m, device=device)
            sw = b[torch.randperm(len(b), device=device)]
            h = torch.where((mt == 1)[:, None, None], hist0[sw], h)
            rev = h.clone(); rev[:, :, :3] *= -1; h = torch.where((mt == 2)[:, None, None], rev, h)
            h = torch.where((mt == 3)[:, None, None], torch.zeros_like(h), h)
            xq, xg = xs(h, b)
            logit = net(xq, KI, xg, ZNN[b], At[b]).masked_fill(~At[b], -1e9)
            Eb = Et[b]; teacher = torch.softmax(-(Eb - Eb.min(1, keepdim=True).values) / tau, -1) * At[b]
            teacher = teacher / teacher.sum(1, keepdim=True)
            l = -(teacher * torch.log_softmax(logit, -1)).sum(-1).mean()
            if loss == 'corr': l = l + (1 - pearson_t(logit.masked_fill(~At[b], 0), -Eb, At[b])).mean()
            opt.zero_grad(); l.backward(); opt.step(); sched.step()
        net.eval()
        with torch.no_grad():
            xq, xg = xs(hv, vat); sc = net(xq, KI, xg, ZNN[vat], At[vat])
            sp = spearman_t(sc, Et[vat], At[vat]); v = float(sp.mean())
            vt = float(sp[torch.as_tensor(keep_true, device=device)].mean())
        curve.append({'epoch': ep + 1, 'holdout_mixed_spearman': v, 'holdout_true_spearman': vt, 'loss': float(l)})
        if v > best[0]: best = (v, {k: t.detach().cpu().clone() for k, t in net.state_dict().items()}, ep + 1)
    res = {'kind': kind, 'loss': loss, 'seed': seed, 'tau': tau, 'curve': curve, 'best': best[0], 'epoch': best[2], 'state': best[1],
           'gmu': gmu.cpu(), 'gsd': gsd.cpu()}
    torch.save(res, path); iq.log(G, stage='train', kind=kind, loss=loss, seed=seed, holdout_mixed=round(best[0], 4), epoch=best[2])
    return res


def evaluate(configs, seeds=(4101, 4102, 4103)):
    import torch, scale_only_query as so
    cd, vs, kq = so._mods(); d, pool, act = load('eval'); n = len(d['task'])
    ua = np.load(kq.UNSEEN / 'actions.npy', mmap_mode='r'); y = np.asarray(ua[d['row'], :, :6], np.float32)   # scoring only
    E = so.type_errors(vs.to_physical(y).astype(np.float32), pool['phys'], iq.COND, 'cuda')
    A = so.allowed_matrix(d['task'], pool['task'], pool['keep'], False); hard = d['hard'].astype(bool)
    orc = np.where(A, E, np.inf).min(1); rnd = np.where(A, E, 0).sum(1) / A.sum(1)
    At = torch.as_tensor(A); KI = torch.as_tensor(so.key_inputs('W', pool, iq.COND))
    ZNN = zt(torch.as_tensor(iq.nn_scores('eval')), At); actt = torch.as_tensor(act); state = torch.as_tensor(d['state'])
    rng = np.random.default_rng(4101)            # identical to image_query_robust.py
    perm = np.array([rng.choice(np.flatnonzero(d['task'] != d['task'][i])) for i in range(n)])
    hists = {'true': d['hist'], 'swap': d['hist'][perm]}
    h = d['hist'].copy(); h[:, :, :3] *= -1; hists['reverse'] = h; hists['zero'] = np.zeros_like(d['hist'])

    def m(sc):
        sp = iq.spearman_rows(sc, E, A); pick = np.where(A, sc, -np.inf).argmax(1); te = E[np.arange(n), pick]
        return {'spearman': float(sp.mean()), 'spearman_hard': float(sp[hard].mean()),
                'gap_recovered': float(((rnd - te) / np.maximum(rnd - orc, 1e-9)).mean()),
                'strict_top1': float((te <= orc + .1 * (rnd - orc)).mean())}
    res = json.loads((G / 'results.json').read_text()) if (G / 'results.json').exists() else {}
    for kind, loss in configs:
        per = {hn: [] for hn in hists}; gates = {hn: [] for hn in hists}
        for s in seeds:
            r = train_one(kind, loss, s)
            net = make_model(kind, 78, KI.shape[1], 155); net.load_state_dict(r['state']); net.eval()
            for hn, hh in hists.items():
                ht = torch.as_tensor(hh); xq = torch.cat([ht.reshape(n, -1), state], 1)
                xg = torch.cat([xq, (actt.reshape(n, -1) - r['gmu']) / r['gsd'], agreement(ht, actt)], 1)
                with torch.no_grad():
                    sc = net(xq, KI, xg, ZNN, At).numpy()
                    if kind != 'P5aug':
                        ab = torch.nn.functional.softplus(net.gate(xg)).numpy() * 3; gates[hn].append(float((ab[:, 1] / ab.sum(1)).mean()))
                per[hn].append(m(sc))
        key = f'{kind}|{loss}'; res[key] = {}
        for hn in hists:
            res[key][hn] = {k: float(np.mean([x[k] for x in per[hn]])) for k in per[hn][0]}
            if gates[hn]: res[key][hn]['image_weight_share'] = float(np.mean(gates[hn]))
            iq.log(G, stage='evaluate', config=key, history=hn, **{k: round(v, 4) for k, v in res[key][hn].items()})
    (G / 'results.json').write_text(json.dumps(res, indent=1) + '\n')


if __name__ == '__main__':
    G.mkdir(parents=True, exist_ok=True)
    cfgs = [tuple(c.split(':')) for c in (sys.argv[1] if len(sys.argv) > 1 else 'Gfz:ce,G:ce,G:corr,P5aug:ce,P5aug:corr').split(',')]
    evaluate(cfgs)
