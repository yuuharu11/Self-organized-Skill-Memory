"""PH2: how fine must the 'current state' be?  Motion-mode clusters of the W pool types (k-means on S-shared types,
K = 4/8/16/32).  Label of a window = cluster of its best candidate (min type error; LIBERO-40 labels for training,
Dev labels for lam, Eval labels for scoring / oracle only).  Predictor: image (+ state, never history) -> K-way.
score = z(P5) + lam * log P(cluster(candidate)); lam on Dev (mixed histories).  Output image-query-01/phase/clusters.json."""
import json, sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import image_query as iq, scale_only_query as so, phase_probe as pp
from sklearn.cluster import KMeans
SEEDS = (4101, 4102, 4103)
cd, vs, kq = so._mods(); P = iq.S1 / 'prep'; pool = dict(np.load(P / 'W_pool.npz'))
ev = dict(np.load(P / 'W_inputs_eval.npz')); dv = dict(np.load(P / 'W_inputs_dev.npz')); trn = dict(np.load(P / 'W_inputs_train.npz'))
ua = np.load(kq.UNSEEN / 'actions.npy', mmap_mode='r'); y = np.asarray(ua[ev['row'], :, :6], np.float32)
E = so.type_errors(vs.to_physical(y).astype(np.float32), pool['phys'], iq.COND, 'cuda')
A = so.allowed_matrix(ev['task'], pool['task'], pool['keep'], False); hard = ev['hard'].astype(bool)
Ed = np.load(P / f'W_err_{iq.COND}_dev.npz')['err']; Ad = so.allowed_matrix(dv['task'], pool['task'], pool['keep'], False)
Et = np.load(P / f'W_err_{iq.COND}_train.npz')['err']; At = so.allowed_matrix(trn['task'], pool['task'], pool['keep'], True)
top1 = lambda sc, EE, AA: EE[np.arange(len(EE)), np.where(AA, sc, -np.inf).argmax(1)]
KI = torch.as_tensor(so.key_inputs('W', pool, iq.COND))
def p5(seed):
    sel = json.loads((iq.S1 / 'selection.json').read_text())['selected'][f'W|{iq.COND}|P5|{seed}']['holdout']
    rr = torch.load(iq.S1 / 'models' / f"W_{iq.COND}_P5_s{seed}_f{sel['frac']:.4f}.pt", weights_only=False)
    m = so.make_net('W', 'P5', 78, KI.shape[1]); m.load_state_dict(rr['holdout']['state']); m.eval(); return m
P5 = {s: p5(s) for s in SEEDS}
def sc5(m, hh, st):
    with torch.no_grad(): return m(torch.cat([torch.as_tensor(hh).reshape(len(hh), -1), torch.as_tensor(st)], 1), KI).numpy()
def variants(d, seed):   # same as phase_probe / I4
    rng = np.random.default_rng(seed); m = len(d['task'])
    perm = np.array([rng.choice(np.flatnonzero(d['task'] != d['task'][i])) if (d['task'] != d['task'][i]).any() else rng.integers(m) for i in range(m)])
    H = {'true': (d['hist'], np.arange(m)), 'swap': (d['hist'][perm], np.arange(m))}
    r = d['hist'].copy(); r[:, :, :3] *= -1; H['reverse'] = (r, np.arange(m)); H['zero'] = (np.zeros_like(d['hist']), np.arange(m))
    key = {(int(t), int(e), int(f)): i for i, (t, e, f) in enumerate(zip(d['task'], d['episode'], d['frame']))}
    for dt in (-20, 20):
        src = np.array([key.get((int(t), int(e), int(f) + dt), -1) for t, e, f in zip(d['task'], d['episode'], d['frame'])])
        ok = np.flatnonzero(src >= 0); hh = d['hist'].copy(); hh[ok] = d['hist'][src[ok]]; H[f'phase{dt:+d}'] = (hh, ok)
    return H
HE = variants(ev, 4101); HD = variants(dv, 5)
mix = {'true': .5, 'swap': .1, 'reverse': .1, 'zero': .1, 'phase-20': .1, 'phase+20': .1}
typ = so.type_of(pool['phys'], iq.COND).reshape(len(pool['phys']), -1)
fit = ~np.isin(trn['task'], so.HOLDOUT)
res = {}
for K in (4, 8, 16, 32):
    cl = KMeans(K, n_init=10, random_state=0).fit_predict(typ).astype(np.int64)
    lab = lambda EE, AA: cl[np.where(AA, EE, np.inf).argmin(1)]
    ytr, ydv, yev = lab(Et, At), lab(Ed, Ad), lab(E, A)
    Aor = A & (cl[None, :] == yev[:, None]); out = {'oracle': {}, 'pred': {}}
    for hn, (hh, rws) in HE.items():
        out['oracle'][hn] = float(np.mean([top1(sc5(P5[s], hh, ev['state']), E, Aor)[rws].mean() for s in SEEDS]))
    for feat in ('vlm', 'mid17', 'act'):
        X = {s: np.concatenate([iq.feature(s, feat), d['state']], 1).astype(np.float32) for s, d in (('train', trn), ('dev', dv), ('eval', ev))}
        mu, sd = X['train'][fit].mean(0), X['train'][fit].std(0) + 1e-6; X = {k: (v - mu) / sd for k, v in X.items()}
        lps = {'dev': [], 'eval': []}; acc = []
        for s in SEEDS:
            torch.manual_seed(s); net = nn.Sequential(nn.Dropout(.2), nn.Linear(X['train'].shape[1], 256), nn.GELU(), nn.Linear(256, K))
            opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-2); xt = torch.as_tensor(X['train'][fit]); yt = torch.as_tensor(ytr[fit])
            xh = torch.as_tensor(X['train'][~fit]); best = (-1, None)
            for ep in range(15):
                net.train(); pm = torch.randperm(len(xt))
                for i in range(0, len(xt), 256):
                    b = pm[i:i + 256]; l = nn.functional.cross_entropy(net(xt[b]), yt[b]); opt.zero_grad(); l.backward(); opt.step()
                net.eval()
                with torch.no_grad(): a = float((net(xh).argmax(1).numpy() == ytr[~fit]).mean())
                if a > best[0]: best = (a, {k: v.clone() for k, v in net.state_dict().items()})
            net.load_state_dict(best[1]); net.eval()
            with torch.no_grad():
                for sp in ('dev', 'eval'): lps[sp].append(torch.log_softmax(net(torch.as_tensor(X[sp])), 1).numpy())
            acc.append((best[0], float((lps['eval'][-1].argmax(1) == yev).mean())))
        def comb(j, s, H, d, EE, AA, lp, lam, rws): return top1(iq.zrow(sc5(P5[s], H, d['state']), AA) + lam * lp[j][:, cl], EE, AA)[rws].mean()
        lams = (0, .25, .5, 1, 2, 4, 8, 16, 64)
        dsc = {lam: np.mean([sum(w * comb(j, s, HD[k][0], dv, Ed, Ad, lps['dev'], lam, HD[k][1]) for k, w in mix.items()) for j, s in enumerate(SEEDS)]) for lam in lams}
        lam = min(dsc, key=dsc.get)
        ev_res = {hn: float(np.mean([comb(j, s, hh, ev, E, A, lps['eval'], lam, rws) for j, s in enumerate(SEEDS)])) for hn, (hh, rws) in HE.items()}
        img_only = float(np.mean([top1(lps['eval'][j][:, cl] + 1e-3 * iq.zrow(iq.nn_scores('eval'), A), E, A).mean() for j in range(3)]))
        out['pred'][feat] = {'acc_holdout': float(np.mean([a[0] for a in acc])), 'acc_eval': float(np.mean([a[1] for a in acc])),
                             'chance_eval': float(np.bincount(yev, minlength=K).max() / len(yev)), 'lam_dev': lam, 'eval': ev_res}
        print(K, feat, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out['pred'][feat].items() if k != 'eval'}, {k: round(v, 4) for k, v in ev_res.items()}, flush=True)
    print(K, 'oracle', {k: round(v, 4) for k, v in out['oracle'].items()}, flush=True)
    res[K] = out
(pp.OUT / 'clusters.json').write_text(json.dumps(res, indent=1) + '\n')
