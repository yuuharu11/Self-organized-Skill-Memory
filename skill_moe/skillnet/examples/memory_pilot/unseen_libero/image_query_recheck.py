"""I4: re-score every query of I1-I3 with the three agreed metrics (type error = s*-fitted S-shared error).

  top1      mean type error of the top-scored allowed candidate (main)
  sp10      Spearman(score, -E) over the candidates in the window's true best 10%
  pe10      Pearson(score, -E) over the same subset
  pelog     Pearson(score, -log(E + 1e-6)) over all allowed candidates
  (sp_full kept for reference.)  Constant scores on a subset -> correlation 0 (counted).
History variants: true / swap / reverse / zero (image_query_robust.py) and phase -20/-10/+10/+20 (image_query_metrics.py).
Queries: S1 P5/P6/P0, pi0.5-NN, the 21 I1 learned modes, I1 fusions (weights as selected on LIBERO-40),
P5+NN w1/w3/w6, I2 gates (Gfz trained on LIBERO-40; linear Dev gate re-trained exactly as image_query_devgate.py).
Eval targets are used for scoring only.  Output: image-query-01/recheck.json and recheck.md.
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path: sys.path.insert(0, str(HERE))
import image_query as iq  # noqa: E402
import image_query_gate as g  # noqa: E402

SEEDS = (4101, 4102, 4103)
I1_MODES = ('A', 'A+H', 'E', 'E+H', 'E1+H', 'V+H', 'M6', 'M10', 'M14', 'M15', 'M16', 'M17', 'M6+H', 'M10+H', 'M14+H',
            'M15+H', 'M16+H', 'M17+H', 'ML16+H', 'A+E+H', 'A+M16+H')
I1_FUSIONS = {'A': .6, 'E': .5, 'M16': .5, 'M14': .4, 'V+H': .4}      # w selected on the LIBERO-40 holdout (results.json)


def metric_rows(sc, E, A, rows):
    """Per-row metrics on GPU. sc, E: [N,K] numpy; returns dict of [n] arrays."""
    import torch
    S = torch.as_tensor(sc[rows], dtype=torch.float64, device='cuda'); Et = torch.as_tensor(E[rows], dtype=torch.float64, device='cuda')
    At = torch.as_tensor(A[rows], device='cuda')
    q10 = torch.nanquantile(Et.masked_fill(~At, float('nan')), .1, dim=1, keepdim=True)
    T = At & (Et <= q10)
    def pearson(x, y, m):
        mf = m.double(); n = mf.sum(1)
        xm = x - (x * mf).sum(1, keepdim=True) / n[:, None]; ym = y - (y * mf).sum(1, keepdim=True) / n[:, None]
        vx = (xm ** 2 * mf).sum(1); vy = (ym ** 2 * mf).sum(1)
        r = (xm * ym * mf).sum(1) / torch.sqrt(vx * vy).clamp_min(1e-30)
        return torch.where(vx < 1e-20, torch.zeros_like(r), r), vx < 1e-20
    def ranks(x, m):                        # average ranks (ties averaged) within m; outside m -> lowest block
        lo = torch.finfo(x.dtype).min; xs = x.masked_fill(~m, lo)
        order = xs.argsort(1); srt = xs.gather(1, order); N, K = xs.shape
        new = torch.ones_like(srt, dtype=torch.bool); new[:, 1:] = srt[:, 1:] != srt[:, :-1]
        gid = torch.cumsum(new.long(), 1) - 1 + torch.arange(N, device=x.device)[:, None] * K
        pos = torch.arange(K, device=x.device, dtype=x.dtype).expand(N, K).reshape(-1)
        lo_ = torch.full((N * K,), float('inf'), device=x.device, dtype=x.dtype).scatter_reduce(0, gid.reshape(-1), pos, 'amin')
        hi_ = torch.full((N * K,), -float('inf'), device=x.device, dtype=x.dtype).scatter_reduce(0, gid.reshape(-1), pos, 'amax')
        avg = ((lo_ + hi_) / 2)[gid.reshape(-1)].reshape(N, K)
        out = torch.empty_like(xs); out.scatter_(1, order, avg); return out
    sp10, c1 = pearson(ranks(S, T), ranks(-Et, T), T)
    spf, _ = pearson(ranks(S, At), ranks(-Et, At), At)
    pe10, c2 = pearson(S, -Et, T)
    pel, _ = pearson(S, -torch.log(Et + 1e-6), At)
    top1 = Et.gather(1, S.masked_fill(~At, -float('inf')).argmax(1, keepdim=True))[:, 0]
    return {'top1': top1.cpu().numpy(), 'sp10': sp10.cpu().numpy(), 'pe10': pe10.cpu().numpy(), 'pelog': pel.cpu().numpy(),
            'sp_full': spf.cpu().numpy(), 'const_subset': (c1 | c2).cpu().numpy()}


def main():
    import torch, scale_only_query as so
    cd, vs, kq = so._mods(); P = iq.S1 / 'prep'; pool = dict(np.load(P / 'W_pool.npz')); ev = dict(np.load(P / 'W_inputs_eval.npz'))
    ua = np.load(kq.UNSEEN / 'actions.npy', mmap_mode='r'); y = np.asarray(ua[ev['row'], :, :6], np.float32)       # scoring only
    E = so.type_errors(vs.to_physical(y).astype(np.float32), pool['phys'], iq.COND, 'cuda')
    A = so.allowed_matrix(ev['task'], pool['task'], pool['keep'], False); hard = ev['hard'].astype(bool); n = len(E)
    # ---------------- history variants (same constructions and seeds as I2/I3)
    rng = np.random.default_rng(4101)
    perm = np.array([rng.choice(np.flatnonzero(ev['task'] != ev['task'][i])) for i in range(n)])
    H = {'true': (ev['hist'], ev['has_hist'], np.arange(n)), 'swap': (ev['hist'][perm], ev['has_hist'][perm], np.arange(n))}
    r = ev['hist'].copy(); r[:, :, :3] *= -1; H['reverse'] = (r, ev['has_hist'], np.arange(n))
    H['zero'] = (np.zeros_like(ev['hist']), np.zeros_like(ev['has_hist']), np.arange(n))
    key = {(int(t), int(e), int(f)): i for i, (t, e, f) in enumerate(zip(ev['task'], ev['episode'], ev['frame']))}
    for dt in (-20, -10, 10, 20):
        src = np.array([key.get((int(t), int(e), int(f) + dt), -1) for t, e, f in zip(ev['task'], ev['episode'], ev['frame'])])
        ok = np.flatnonzero(src >= 0); hh = ev['hist'].copy(); hs = ev['has_hist'].copy()
        hh[ok] = ev['hist'][src[ok]]; hs[ok] = ev['has_hist'][src[ok]]; H[f'phase{dt:+d}'] = (hh, hs, ok)
    KI = torch.as_tensor(so.key_inputs('W', pool, iq.COND)); At = torch.as_tensor(A)
    nn = iq.nn_scores('eval'); znn = iq.zrow(nn, A); act = np.load(iq._pf('eval') / 'eval_actions.npy').astype(np.float32)

    def s1net(mode, s):
        sel = json.loads((iq.S1 / 'selection.json').read_text())['selected'][f'W|{iq.COND}|{mode}|{s}']['holdout']
        rr = torch.load(iq.S1 / 'models' / f"W_{iq.COND}_{mode}_s{s}_f{sel['frac']:.4f}.pt", weights_only=False)
        d = dict(ev); m = so.make_net('W', mode, so.query_inputs(d, mode).shape[1], KI.shape[1]); m.load_state_dict(rr['holdout']['state']); m.eval(); return m
    S1N = {(m, s): s1net(m, s) for m in ('P5', 'P6', 'P0') for s in SEEDS}
    I1N = {}
    for mode in I1_MODES:
        X = iq.inputs(mode, ('train', 'eval'))[0]['eval']
        for s in SEEDS:
            frac = json.loads((iq.OUT / 'selection.json').read_text())[f'{mode}|{s}']['frac']
            rr = torch.load(iq.OUT / 'models' / f'{mode}_s{s}_f{frac:.4f}.pt', weights_only=False)
            m = iq.make_net(X.shape[1], KI.shape[1]); m.load_state_dict(rr['holdout']['state']); m.eval(); I1N[(mode, s)] = (m, X)
    GF = {}
    for s in SEEDS:
        rr = torch.load(g.G / f'Gfz_ce_s{s}.pt', weights_only=False)
        m = g.make_model('Gfz', 78, KI.shape[1], 155); m.load_state_dict(rr['state']); m.eval(); GF[s] = (m, rr['gmu'], rr['gsd'])
    DG = dev_gates(so, pool, KI, S1N)

    def scores_for(hh, hs):
        d = {**ev, 'hist': hh, 'has_hist': hs}; out = {}
        with torch.no_grad():
            for (mode, s), m in S1N.items(): out.setdefault(mode, []).append(m(torch.as_tensor(so.query_inputs(d, mode)), KI).numpy())
            out['pi05_nn'] = [nn]
            for (mode, s), (m, X) in I1N.items():
                Xv = X.copy()
                if MODES_H(mode): Xv[:, -78:-8] = hh.reshape(n, -1)
                out.setdefault(mode, []).append(m(torch.as_tensor(Xv), KI).numpy())
            for w in (1, 3, 6): out[f'P5+NN w{w}'] = [iq.zrow(p, A) + w * znn for p in out['P5']]
            for mode, w in I1_FUSIONS.items():
                out[f'P5+{mode} w{w}'] = [iq.zrow(p, A) + w * iq.zrow(q, A) for p, q in zip(out['P5'], out[mode])]
            ht = torch.as_tensor(hh); xq = torch.cat([ht.reshape(n, -1), torch.as_tensor(ev['state'])], 1); znt = torch.as_tensor(znn)
            for s, (m, mu, sd) in GF.items():
                xg = torch.cat([xq, (torch.as_tensor(act).reshape(n, -1) - mu) / sd, g.agreement(ht, torch.as_tensor(act))], 1)
                out.setdefault('Gate LIBERO-40 (Gfz)', []).append(m(xq, KI, xg, znt, At).numpy())
            for s, (lin, mu, sd) in DG.items():
                p5 = g.zt(S1N[('P5', s)](xq, KI), At)
                ab = torch.nn.functional.softplus(lin((g.agreement(ht, torch.as_tensor(act)) - mu) / sd)) * 3
                out.setdefault('Gate Dev (linear)', []).append((ab[:, :1] * p5 + ab[:, 1:] * znt).numpy())
        return out
    res = {'variant_rows': {k: int(len(v[2])) for k, v in H.items()}}
    for hn, (hh, hs, rows) in H.items():
        sc = scores_for(hh, hs); res[hn] = {}
        for name, lst in sc.items():
            ms = [metric_rows(x, E, A, rows) for x in lst]; hr = hard[rows]
            agg = {}
            for k in ('top1', 'sp10', 'pe10', 'pelog', 'sp_full'):
                agg[k] = float(np.mean([m[k].mean() for m in ms])); agg[k + '_hard'] = float(np.mean([m[k][hr].mean() for m in ms]))
                agg[k + '_seed_range'] = [float(min(m[k].mean() for m in ms)), float(max(m[k].mean() for m in ms))]
            agg['const_subset_rows'] = float(np.mean([m['const_subset'].sum() for m in ms]))
            res[hn][name] = agg
        print(hn, 'done', flush=True)
    oracle = np.where(A, E, np.inf).min(1); rnd = np.where(A, E, 0).sum(1) / A.sum(1)
    res['reference'] = {'oracle_top1': float(oracle.mean()), 'oracle_top1_hard': float(oracle[hard].mean()),
                        'random_top1': float(rnd.mean()), 'random_top1_hard': float(rnd[hard].mean())}
    (iq.OUT / 'recheck.json').write_text(json.dumps(res, indent=1, ensure_ascii=False) + '\n'); write_md(res)


def MODES_H(mode): return iq.MODES[mode][1]


def dev_gates(so, pool, KI, S1N):
    """Linear Dev gate, identical recipe to image_query_devgate.py (P5 frozen, 7 agreement features, Dev, 300 steps)."""
    import torch
    P = iq.S1 / 'prep'; dv = dict(np.load(P / 'W_inputs_dev.npz')); Ed = torch.as_tensor(np.load(P / f'W_err_{iq.COND}_dev.npz')['err'])
    Ad = torch.as_tensor(so.allowed_matrix(dv['task'], pool['task'], pool['keep'], False)); actd = torch.as_tensor(np.load(iq._pf('dev') / 'dev_actions.npy').astype(np.float32))
    znd = g.zt(torch.as_tensor(iq.nn_scores('dev')), Ad); rng = np.random.default_rng(5); n = len(dv['task'])
    perm = np.array([rng.choice(np.flatnonzero(dv['task'] != dv['task'][i])) if (dv['task'] != dv['task'][i]).any() else rng.integers(n) for i in range(n)])
    h = dv['hist']; rv = h.copy(); rv[:, :, :3] *= -1
    HD = {'true': h, 'swap': h[perm], 'reverse': rv, 'zero': np.zeros_like(h)}; mix = {'true': .5, 'swap': .2, 'reverse': .15, 'zero': .15}
    tau = so.teacher_tau(Ed.numpy(), Ad.numpy(), .1)
    teacher = torch.softmax(-(Ed - Ed.masked_fill(~Ad, 1e3).min(1, keepdim=True).values) / tau, -1) * Ad; teacher /= teacher.sum(1, keepdim=True)
    out = {}
    for s in SEEDS:
        net = S1N[('P5', s)]
        with torch.no_grad():
            zpd = {k: g.zt(net(torch.cat([torch.as_tensor(v).reshape(n, -1), torch.as_tensor(dv['state'])], 1), KI), Ad) for k, v in HD.items()}
        fd = {k: g.agreement(torch.as_tensor(v), actd) for k, v in HD.items()}; mu = fd['true'].mean(0); sd = fd['true'].std(0) + 1e-6
        torch.manual_seed(s); lin = torch.nn.Linear(7, 2); torch.nn.init.zeros_(lin.weight); torch.nn.init.constant_(lin.bias, .5)
        opt = torch.optim.Adam(lin.parameters(), 1e-2, weight_decay=1e-2)
        for _ in range(300):
            l = 0
            for k, p in mix.items():
                ab = torch.nn.functional.softplus(lin((fd[k] - mu) / sd)) * 3
                sc = (ab[:, :1] * zpd[k] + ab[:, 1:] * znd).masked_fill(~Ad, -1e9)
                l = l + p * (-(teacher * torch.log_softmax(sc, -1)).sum(-1).mean() + (1 - g.pearson_t(sc.masked_fill(~Ad, 0), -Ed, Ad)).mean())
            opt.zero_grad(); l.backward(); opt.step()
        lin.eval(); out[s] = (lin, mu, sd)
    return out


def write_md(res):
    V = list(res['variant_rows']); ref = res['reference']
    names = list(res['true'])
    L = ['# I4 再採点（型誤差 = s* を当てた S-shared 誤差、未学習 Eval 619 窓、3 seed 平均）', '',
         f"oracle の1位の型誤差 {ref['oracle_top1']:.4f}（難 {ref['oracle_top1_hard']:.4f}）、ランダム {ref['random_top1']:.4f}（難 {ref['random_top1_hard']:.4f}）。",
         '行数：' + '、'.join(f'{k} {v}' for k, v in res['variant_rows'].items()), '']
    for k, lab in (('top1', '1位の型誤差の平均（小さいほど良い）'), ('sp10', 'Spearman（上位10%の候補内）'), ('pe10', 'Pearson（上位10%の候補内）'),
                   ('pelog', 'Pearson（全候補、−log 型誤差）'), ('sp_full', '参考：Spearman（全候補）')):
        L += [f'## {lab}', '', '| クエリ | 真 | 真・難 | ' + ' | '.join(V[1:]) + ' |', '|---|' + '---:|' * (len(V) + 1)]
        order = sorted(names, key=lambda m: res['true'][m]['top1']) if k == 'top1' else names
        for m in order:
            f = (lambda x: f'{x:.4f}') if k == 'top1' else (lambda x: f'{x:.3f}')
            L.append(f'| {m} | {f(res["true"][m][k])} | {f(res["true"][m][k + "_hard"])} | ' + ' | '.join(f(res[v][m][k]) for v in V[1:]) + ' |')
        L.append('')
    (iq.OUT / 'recheck.md').write_text('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
