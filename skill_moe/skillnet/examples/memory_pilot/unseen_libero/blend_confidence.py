"""BC1: with the P5 query, (a) pick one type or blend the top-k, (b) which confidence tracks the error,
(c) what changes when the target is DCT-3 low-passed (high frequencies left to the denoiser).

Same setting as I4 (image_query_recheck.py): W pool 1,595, S-shared, unseen Eval 619 windows, P5 x 3 seeds.
Ground-truth actions are used for scoring only (type error, 6-axis MSE, oracle).  Blend hyper-parameters
(k, temperature) are reported as a full grid on Eval and *selected on Dev* (LIBERO-90 tasks 6, 7).

Blend  w_j = softmax(score_j / T) over the P5 top-k (T relative to the row's score std over the allowed set).
  BT  blend of scale-free types (type_of) -> one size.   BS  blend of stored windows (stored size kept).
  Size: s* (denoising assumed optimal; = type error), hist speed, pi0.5-chunk speed, stored.
Confidence (no ground truth): margin, z-margin, p_max, entropy, top-5 type spread, seed agreement,
  pi0.5-chunk fit of the pick, history fit of the pick, pi0.5-NN rank of the pick, history speed.
Targets: top-1 type error (abs), relative error (/target energy), regret (top1 - oracle), rank percentile.
Output: image-query-01/blend/results.json and results.md."""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np, torch
from scipy.fft import dct, idct
from scipy.stats import spearmanr
from scipy.stats import rankdata


def roc_auc_score(y, x):
    y = np.asarray(y, bool); r = rankdata(x); n1 = y.sum(); n0 = len(y) - n1
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import image_query as iq, scale_only_query as so  # noqa: E402

SEEDS = (4101, 4102, 4103)
KS = (1, 2, 3, 5, 10, 20)
TS = (0.1, 0.3, 1.0, 3.0, 1e9)           # x row score std; 1e9 = uniform over top-k
OUT = iq.OUT / 'blend'
cd, vs, kq = so._mods(); P = iq.S1 / 'prep'; pool = dict(np.load(P / 'W_pool.npz'))
U = pool['phys'].astype(np.float64)                         # [K,10,6]
UT = so.type_of(U, 'S-shared')                              # scale-free types
KI = torch.as_tensor(so.key_inputs('W', pool, iq.COND))
ua = np.load(kq.UNSEEN / 'actions.npy', mmap_mode='r')


def lowpass(x, k=3):
    c = dct(np.asarray(x, np.float64), type=2, norm='ortho', axis=-2); c[..., k:, :] = 0
    return idct(c, type=2, norm='ortho', axis=-2)


def p5(seed):
    sel = json.loads((iq.S1 / 'selection.json').read_text())['selected'][f'W|{iq.COND}|P5|{seed}']['holdout']
    rr = torch.load(iq.S1 / 'models' / f"W_{iq.COND}_P5_s{seed}_f{sel['frac']:.4f}.pt", weights_only=False)
    m = so.make_net('W', 'P5', 78, KI.shape[1]); m.load_state_dict(rr['holdout']['state']); m.eval(); return m
NETS = {s: p5(s) for s in SEEDS}


def scores(d, hh=None):
    hh = d['hist'] if hh is None else hh
    with torch.no_grad():
        x = torch.cat([torch.as_tensor(hh).reshape(len(hh), -1), torch.as_tensor(d['state'])], 1)
        return {s: m(x, KI).numpy().astype(np.float64) for s, m in NETS.items()}


def split(name):
    d = dict(np.load(P / f'W_inputs_{name}.npz')); y = np.asarray(ua[d['row'], :, :6], np.float32)
    yp = vs.to_physical(y).astype(np.float64)
    A = so.allowed_matrix(d['task'], pool['task'], pool['keep'], False)
    E = so.type_errors(yp.astype(np.float32), pool['phys'], iq.COND, 'cuda').astype(np.float64)
    El = so.type_errors(lowpass(yp).astype(np.float32), pool['phys'], iq.COND, 'cuda').astype(np.float64)
    pi = np.load(iq._pf(name) / f'{name}_actions.npy').astype(np.float32)[:, :, :6]
    return dict(d=d, y=y, yp=yp, A=A, E=E, El=El, pi=vs.to_physical(pi).astype(np.float64), nn=iq.nn_scores(name).astype(np.float64),
                hard=d.get('hard', np.zeros(len(y))).astype(bool))


def fit_err(u, q):
    """S-shared error with the optimal non-negative scale; u, q [N,T,6] -> [N]."""
    s = np.maximum((u * q).sum((1, 2)), 0) / np.maximum((u * u).sum((1, 2)), 1e-12)
    return ((s[:, None, None] * u - q) ** 2).mean((1, 2)), s


def topk(sc, A, k):
    m = np.where(A, sc, -np.inf); idx = np.argsort(-m, 1)[:, :k]; return idx, np.take_along_axis(m, idx, 1)


def weights(vals, sc, A, T):
    sd = np.nanstd(np.where(A, sc, np.nan), 1, keepdims=True) + 1e-12
    z = (vals - vals[:, :1]) / (T * sd); w = np.exp(z); return w / w.sum(1, keepdims=True)


def blend(S, sc, k, T, base):
    idx, v = topk(sc, S['A'], k); w = weights(v, sc, S['A'], T)
    return (w[:, :, None, None] * base[idx]).sum(1)


def six(pred_phys, S):
    return so.six_axis(vs.to_normalized(pred_phys), S['y'])


def sized(u, S, rule):
    if rule == 'sstar': return fit_err(u, S['yp'])[0]          # type error (denoise = optimal size)
    if rule == 'sstar_low': return fit_err(u, lowpass(S['yp']))[0]
    if rule == 'stored': return six(u, S)
    if rule == 'hist':
        ref = vs.to_physical(S['d']['hist'][:, :, :6]); o, _ = so.match_speed(u, ref, 'S-shared')
        o = np.where(S['d']['has_hist'].astype(bool)[:, None, None], o, u); return six(o, S)
    if rule == 'pi05': return six(so.match_speed(u, S['pi'], 'S-shared')[0], S)
    if rule == 'sstar6': return six(so.apply_scales(u, fit_err(u, S['yp'])[1][:, None], 'S-shared'), S)
    raise ValueError(rule)
RULES = ('sstar', 'sstar_low', 'sstar6', 'hist', 'pi05', 'stored')


def blend_grid(S, SC):
    out = {}
    for base_name, base in (('BT', UT), ('BS', U)):
        for k in KS:
            for T in (TS if k > 1 else (1.0,)):
                key = f'{base_name}|k{k}|T{T:g}'; per = {r: [] for r in RULES}
                for s in SEEDS:
                    u = blend(S, SC[s], k, T, base)
                    if base_name == 'BT':                         # types carry no size: give them the stored rms of the blend members
                        idx, v = topk(SC[s], S['A'], k); w = weights(v, SC[s], S['A'], T)
                        u = u * (w * so.rms_speed(U[idx])).sum(1)[:, None, None] / np.maximum(so.rms_speed(u), 1e-9)[:, None, None]
                    for r in RULES: per[r].append(sized(u, S, r))
                out[key] = {r: np.mean(per[r], 0) for r in RULES}
    return out


def confidences(S, SC):
    A = S['A']; n = len(A); C = {}
    for s in SEEDS:
        sc = SC[s]; idx, v = topk(sc, A, 5); pick = idx[:, 0]
        sd = np.nanstd(np.where(A, sc, np.nan), 1) + 1e-12
        m = np.where(A, sc, -np.inf); p = np.exp(m - m.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        ent = -(np.where(p > 0, p * np.log(np.maximum(p, 1e-300)), 0)).sum(1)
        t5 = UT[idx]; spread = np.mean([((t5[:, i] - t5[:, j]) ** 2).mean((1, 2)) for i in range(5) for j in range(i + 1, 5)], 0)
        pif, _ = fit_err(U[pick], S['pi']); pirel = pif / np.maximum((S['pi'] ** 2).mean((1, 2)), 1e-12)
        h = vs.to_physical(S['d']['hist'][:, :, :6]).astype(np.float64); hf, _ = fit_err(U[pick], h)
        hrel = np.where(S['d']['has_hist'].astype(bool), hf / np.maximum((h ** 2).mean((1, 2)), 1e-12), 1.)
        nnr = np.array([(S['nn'][i, A[i]] > S['nn'][i, pick[i]]).mean() for i in range(n)])
        rows = {'margin': v[:, 0] - v[:, 1], 'z_margin': (v[:, 0] - v[:, 1]) / sd, 'p_max': p.max(1), 'neg_entropy': -ent,
                'neg_top5_spread': -spread, 'neg_pi05_fit_rel': -pirel, 'neg_pi05_fit_abs': -pif, 'neg_hist_fit_rel': -hrel,
                'neg_nn_rank': -nnr, 'neg_hist_speed': -so.rms_speed(h)}
        for k_, v_ in rows.items(): C.setdefault(k_, []).append(v_)
    C = {k: np.mean(v, 0) for k, v in C.items()}
    picks = np.stack([topk(SC[s], A, 1)[0][:, 0] for s in SEEDS])
    agree = np.mean([((UT[picks[i]] - UT[picks[j]]) ** 2).mean((1, 2)) for i in range(3) for j in range(i + 1, 3)], 0)
    C['neg_seed_spread'] = -agree
    ens = sum(iq.zrow(SC[s], A) for s in SEEDS); C['ens_z_margin'] = -np.diff(topk(ens, A, 2)[1], axis=1)[:, 0]
    return C


def targets(S, SC, E):
    A = S['A']; o = np.where(A, E, np.inf).min(1); en = (S['yp'] ** 2).mean((1, 2))
    t1 = np.mean([E[np.arange(len(E)), topk(SC[s], A, 1)[0][:, 0]] for s in SEEDS], 0)
    rk = np.mean([np.array([(E[i, A[i]] < E[i, topk(SC[s], A, 1)[0][i, 0]]).mean() for i in range(len(E))]) for s in SEEDS], 0)
    return {'abs': t1, 'rel': t1 / np.maximum(en, 1e-12), 'regret': t1 - o, 'rank_pct': rk}


def corr_table(C, T):
    res = {}
    for c, cv in C.items():
        res[c] = {}
        for t, tv in T.items():
            bad = tv > np.quantile(tv, .75)
            r = spearmanr(cv, -tv).correlation
            res[c][t] = {'spearman': float(0 if np.isnan(r) else r), 'auroc_bad25': float(roc_auc_score(~bad, cv)),
                         'risk_top50': float(tv[cv >= np.median(cv)].mean() / tv.mean())}
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True); res = {}
    # ---- DCT-3 subspace check
    hf = lambda x: float(((x - lowpass(x)) ** 2).sum() / (x ** 2).sum())
    EV, DV = split('eval'), split('dev')
    res['dct'] = {'pool_highfreq_energy': hf(U), 'eval_target_highfreq_energy': hf(EV['yp']),
                  'eval_target_highfreq_energy_per_row_median': float(np.median(((EV['yp'] - lowpass(EV['yp'])) ** 2).sum((1, 2)) / (EV['yp'] ** 2).sum((1, 2))))}
    SCE, SCD = scores(EV['d']), scores(DV['d'])
    A = EV['A']; o = np.where(A, EV['E'], np.inf).argmin(1); ol = np.where(A, EV['El'], np.inf).argmin(1)
    rho = [spearmanr(EV['E'][i, A[i]], EV['El'][i, A[i]]).correlation for i in range(len(A))]
    t1 = lambda E, sc: E[np.arange(len(E)), np.where(A, sc, -np.inf).argmax(1)]
    res['dct'].update({'oracle_same_pick': float((o == ol).mean()), 'row_spearman_E_vs_Elow_median': float(np.median(rho)),
                       'oracle_full': float(EV['E'][np.arange(len(A)), o].mean()), 'oracle_low': float(EV['El'][np.arange(len(A)), ol].mean()),
                       'oracle_low_pick_scored_full': float(EV['E'][np.arange(len(A)), ol].mean()),
                       'P5_top1_full': float(np.mean([t1(EV['E'], SCE[s]).mean() for s in SEEDS])),
                       'P5_top1_low': float(np.mean([t1(EV['El'], SCE[s]).mean() for s in SEEDS])),
                       'random_full': float((np.where(A, EV['E'], 0).sum(1) / A.sum(1)).mean()),
                       'random_low': float((np.where(A, EV['El'], 0).sum(1) / A.sum(1)).mean())})
    print('dct', json.dumps(res['dct']), flush=True)
    # ---- single vs blend
    ge, gd = blend_grid(EV, SCE), blend_grid(DV, SCD); hard = EV['hard']
    res['blend'] = {k: {r: {'all': float(v[r].mean()), 'hard': float(v[r][hard].mean())} for r in RULES} for k, v in ge.items()}
    res['blend_dev'] = {k: {r: float(v[r].mean()) for r in RULES} for k, v in gd.items()}
    single = ge['BT|k1|T1']
    for r in RULES:                                   # paired episode bootstrap of best-on-Dev blend minus single
        keyd = min((k for k in gd if k.startswith('BT')), key=lambda k: gd[k][r].mean())
        diff = ge[keyd][r] - single[r]; ep = EV['d']['task'] * 1000 + EV['d']['episode']; ue = np.unique(ep)
        rng = np.random.default_rng(4101); bs = []
        for _ in range(2000):
            pick = rng.choice(ue, len(ue)); bs.append(np.concatenate([diff[ep == e] for e in pick]).mean())
        res.setdefault('dev_selected', {})[r] = {'key': keyd, 'single': float(single[r].mean()), 'blend': float(ge[keyd][r].mean()),
                                                 'diff': float(diff.mean()), 'ci95': [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
                                                 'single_hard': float(single[r][hard].mean()), 'blend_hard': float(ge[keyd][r][hard].mean())}
    # ---- confidence
    C = confidences(EV, SCE)
    res['conf_full'] = corr_table(C, targets(EV, SCE, EV['E'])); res['conf_low'] = corr_table(C, targets(EV, SCE, EV['El']))
    # does confidence tell when blending helps? (k5 T1 type blend, s* size)
    gain = single['sstar'] - ge['BT|k5|T1']['sstar']
    res['conf_vs_blend_gain'] = {c: float(spearmanr(-cv, gain).correlation) for c, cv in C.items()}
    # corrupted history: does confidence drop?
    rng = np.random.default_rng(4101); d = EV['d']; n = len(d['task'])
    perm = np.array([rng.choice(np.flatnonzero(d['task'] != d['task'][i])) for i in range(n)])
    rv = d['hist'].copy(); rv[:, :, :3] *= -1
    res['conf_detect_corrupt'] = {}
    for hn, hh in (('swap', d['hist'][perm]), ('reverse', rv)):
        Sx = {**EV, 'd': {**d, 'hist': hh, 'has_hist': d['has_hist'][perm] if hn == 'swap' else d['has_hist']}}
        Cx = confidences(Sx, scores(Sx['d'], hh))
        res['conf_detect_corrupt'][hn] = {c: float(roc_auc_score(np.r_[np.ones(n), np.zeros(n)], np.r_[C[c], Cx[c]])) for c in C}
    (OUT / 'results.json').write_text(json.dumps(res, indent=1) + '\n'); print('done', flush=True)


if __name__ == '__main__':
    main()
