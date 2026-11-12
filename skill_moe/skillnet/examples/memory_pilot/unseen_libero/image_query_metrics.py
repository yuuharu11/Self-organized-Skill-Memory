"""I3: re-score I1/I2 queries with top-focused metrics and phase-shifted histories.

Metrics per Eval window (allowed candidates only; Eval targets used for scoring only):
  sp_full   Spearman(score, -type error) over all allowed candidates (S1 / I1 metric)
  sp_top10  same, restricted to the candidates whose true type error is in the window's best 10%
            (constant scores on the subset -> 0; happens for pi0.5-NN when the chunk fits none of them, scale clamps at 0)
  R@k       a 'good' candidate (err <= oracle + 0.1 (random - oracle), = strict top1) is within the top-k scores
  ndcg10    relevance exp(-(e - e_oracle) / tau_q), tau_q = 0.1 (random - oracle)
  gap@1/5   gap recovered by the best of the top-1 / top-5
History variants: true, swap / reverse / zero (image_query_robust.py), and phase shifts: the history of the window
dt frames later/earlier in the same episode (dt = +-10, +-20; windows are stride 10).  Rows without such a window
are dropped for that variant (counts reported).  Output: image-query-01/metrics.json.
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path: sys.path.insert(0, str(HERE))
import image_query as iq  # noqa: E402


def metrics(sc, E, A, rows, hard):
    from scipy.stats import rankdata
    sc, E, A = sc[rows], E[rows], A[rows]; h = hard[rows]; n = len(rows)
    orc = np.where(A, E, np.inf).min(1); rnd = np.where(A, E, 0).sum(1) / A.sum(1); tq = .1 * (rnd - orc)
    out = {k: np.zeros(n) for k in ('sp_full', 'sp_top10', 'R@1', 'R@5', 'R@20', 'ndcg10', 'gap@1', 'gap@5')}
    for i in range(n):
        a = np.flatnonzero(A[i]); s, e = sc[i, a], E[i, a]
        out['sp_full'][i] = np.corrcoef(rankdata(s), rankdata(-e))[0, 1]
        top = e <= np.quantile(e, .1)
        out['sp_top10'][i] = np.corrcoef(rankdata(s[top]), rankdata(-e[top]))[0, 1]
        order = np.argsort(-s); good = e <= orc[i] + tq[i]
        for k in (1, 5, 20): out[f'R@{k}'][i] = good[order[:k]].any()
        rel = np.exp(-(e - orc[i]) / max(tq[i], 1e-9)); disc = 1 / np.log2(np.arange(2, 12))
        out['ndcg10'][i] = (rel[order[:10]] * disc).sum() / (np.sort(rel)[::-1][:10] * disc).sum()
        for k in (1, 5): out[f'gap@{k}'][i] = (rnd[i] - e[order[:k]].min()) / max(rnd[i] - orc[i], 1e-9)
    out = {k: np.nan_to_num(v) for k, v in out.items()}       # constant scores in the subset = no ranking information -> 0
    res = {k: float(v.mean()) for k, v in out.items()}
    res.update({f'{k}_hard': float(v[h].mean()) for k, v in out.items() if k in ('sp_top10', 'R@5', 'gap@1')})
    res['rows'] = int(n); return res


def main():
    import torch, scale_only_query as so
    cd, vs, kq = so._mods(); P = iq.S1 / 'prep'; pool = dict(np.load(P / 'W_pool.npz')); ev = dict(np.load(P / 'W_inputs_eval.npz'))
    ua = np.load(kq.UNSEEN / 'actions.npy', mmap_mode='r'); y = np.asarray(ua[ev['row'], :, :6], np.float32)       # scoring only
    E = so.type_errors(vs.to_physical(y).astype(np.float32), pool['phys'], iq.COND, 'cuda')
    A = so.allowed_matrix(ev['task'], pool['task'], pool['keep'], False); hard = ev['hard'].astype(bool); n = len(E)
    # history variants -> (hist, has_hist, rows)
    rng = np.random.default_rng(4101)                                    # identical to image_query_robust.py
    perm = np.array([rng.choice(np.flatnonzero(ev['task'] != ev['task'][i])) for i in range(n)])
    H = {'true': (ev['hist'], ev['has_hist'], np.arange(n)), 'swap': (ev['hist'][perm], ev['has_hist'][perm], np.arange(n))}
    r = ev['hist'].copy(); r[:, :, :3] *= -1; H['reverse'] = (r, ev['has_hist'], np.arange(n))
    H['zero'] = (np.zeros_like(ev['hist']), np.zeros_like(ev['has_hist']), np.arange(n))
    key = {(int(t), int(e), int(f)): i for i, (t, e, f) in enumerate(zip(ev['task'], ev['episode'], ev['frame']))}
    for dt in (-20, -10, 10, 20):
        src = np.array([key.get((int(t), int(e), int(f) + dt), -1) for t, e, f in zip(ev['task'], ev['episode'], ev['frame'])])
        ok = np.flatnonzero(src >= 0); hh = ev['hist'].copy(); hs = ev['has_hist'].copy()
        hh[ok] = ev['hist'][src[ok]]; hs[ok] = ev['has_hist'][src[ok]]
        H[f'phase{dt:+d}'] = (hh, hs, ok)
    frame_step = np.diff(np.sort(np.unique(ev['frame'])))[:3]
    Ae = torch.as_tensor(A); KI = torch.as_tensor(so.key_inputs('W', pool, iq.COND))
    nn = iq.nn_scores('eval'); znn = iq.zrow(nn, A)
    p5 = {}
    for s in (4101, 4102, 4103):
        net = iq.scores.__globals__  # noqa (keep import side effects)
        sel = json.loads((iq.S1 / 'selection.json').read_text())['selected'][f'W|{iq.COND}|P5|{s}']['holdout']
        rr = torch.load(iq.S1 / 'models' / f"W_{iq.COND}_P5_s{s}_f{sel['frac']:.4f}.pt", weights_only=False)
        m = so.make_net('W', 'P5', 78, KI.shape[1]); m.load_state_dict(rr['holdout']['state']); m.eval(); p5[s] = m
    learned = {}
    for mode in ('A+H', 'M17+H', 'M6+H'):
        X = iq.inputs(mode, ('train', 'eval'))[0]['eval']; nets = []
        for s in (4101, 4102, 4103):
            frac = json.loads((iq.OUT / 'selection.json').read_text())[f'{mode}|{s}']['frac']
            rr = torch.load(iq.OUT / 'models' / f'{mode}_s{s}_f{frac:.4f}.pt', weights_only=False)
            m = iq.make_net(X.shape[1], KI.shape[1]); m.load_state_dict(rr['holdout']['state']); m.eval(); nets.append(m)
        learned[mode] = (X, nets)
    res = {'variant_rows': {k: int(len(v[2])) for k, v in H.items()}, 'frame_step': frame_step.tolist()}
    with torch.no_grad():
        for hn, (hh, hs, rows) in H.items():
            d = {**ev, 'hist': hh, 'has_hist': hs}; xq = torch.as_tensor(so.query_inputs(d, 'P5'))
            per = {}
            for s, m in p5.items():
                sp5 = m(xq, KI).numpy(); z5 = iq.zrow(sp5, A)
                per.setdefault('P5', []).append(metrics(sp5, E, A, rows, hard))
                per.setdefault('P5+nn w3 (LIBERO-40)', []).append(metrics(z5 + 3 * znn, E, A, rows, hard))
                per.setdefault('P5+nn w6 (Dev)', []).append(metrics(z5 + 6 * znn, E, A, rows, hard))
                per.setdefault('P5+nn w1', []).append(metrics(z5 + znn, E, A, rows, hard))
            per['pi05_nn'] = [metrics(nn, E, A, rows, hard)]
            for mode, (X, nets) in learned.items():
                Xv = X.copy(); Xv[:, -78:-8] = hh.reshape(n, -1)
                for m in nets: per.setdefault(mode, []).append(metrics(m(torch.as_tensor(Xv), KI).numpy(), E, A, rows, hard))
            res[hn] = {k: {mm: float(np.mean([x[mm] for x in v])) for mm in v[0]} for k, v in per.items()}
            print(hn, len(rows), {k: (round(v['sp_top10'], 3), round(v['R@5'], 3), round(v['gap@1'], 3)) for k, v in res[hn].items()}, flush=True)
    (iq.OUT / 'metrics.json').write_text(json.dumps(res, indent=1) + '\n')


if __name__ == '__main__':
    main()
