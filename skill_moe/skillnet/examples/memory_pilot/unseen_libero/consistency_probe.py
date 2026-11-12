"""PH3: a consistency detector 'does this action history fit the current image/state?' and a hard switch.
Detector input: image features (one of vlm / mid17 / act) + state + history (+ agreement features of image_query_gate).
Label: 1 = history corrupted (swap .25 / reverse .25 / zero .15 / phase shift +-10/+-20 .35 within the same episode),
0 = true.  Trained on LIBERO-40 fit rows, epoch chosen on LIBERO-40 holdout AUC (no Dev/Eval choice).
Switch: P5 if p(corrupt) < thr else pi0.5-NN; thr chosen on Dev (mixed histories, top-1 type error).
Also reported: oracle switch (knows whether the history was corrupted).  Output image-query-01/phase/consistency.json."""
import json, sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import image_query as iq, scale_only_query as so, phase_probe as pp, image_query_gate as g
from sklearn.metrics import roc_auc_score
SEEDS = (4101, 4102, 4103)
cd, vs, kq = so._mods(); P = iq.S1 / 'prep'; pool = dict(np.load(P / 'W_pool.npz'))
ev = dict(np.load(P / 'W_inputs_eval.npz')); dv = dict(np.load(P / 'W_inputs_dev.npz')); trn = dict(np.load(P / 'W_inputs_train.npz'))
ua = np.load(kq.UNSEEN / 'actions.npy', mmap_mode='r'); y = np.asarray(ua[ev['row'], :, :6], np.float32)
E = so.type_errors(vs.to_physical(y).astype(np.float32), pool['phys'], iq.COND, 'cuda')
A = so.allowed_matrix(ev['task'], pool['task'], pool['keep'], False); hard = ev['hard'].astype(bool)
Ed = np.load(P / f'W_err_{iq.COND}_dev.npz')['err']; Ad = so.allowed_matrix(dv['task'], pool['task'], pool['keep'], False)
top1 = lambda sc, EE, AA: EE[np.arange(len(EE)), np.where(AA, sc, -np.inf).argmax(1)]
KI = torch.as_tensor(so.key_inputs('W', pool, iq.COND))
def p5(seed):
    sel = json.loads((iq.S1 / 'selection.json').read_text())['selected'][f'W|{iq.COND}|P5|{seed}']['holdout']
    rr = torch.load(iq.S1 / 'models' / f"W_{iq.COND}_P5_s{seed}_f{sel['frac']:.4f}.pt", weights_only=False)
    m = so.make_net('W', 'P5', 78, KI.shape[1]); m.load_state_dict(rr['holdout']['state']); m.eval(); return m
P5 = {s: p5(s) for s in SEEDS}
def sc5(m, hh, st):
    with torch.no_grad(): return m(torch.cat([torch.as_tensor(hh).reshape(len(hh), -1), torch.as_tensor(st)], 1), KI).numpy()
def variants(d, seed, shifts=(-20, -10, 10, 20)):
    rng = np.random.default_rng(seed); m = len(d['task'])
    perm = np.array([rng.choice(np.flatnonzero(d['task'] != d['task'][i])) if (d['task'] != d['task'][i]).any() else rng.integers(m) for i in range(m)])
    H = {'true': (d['hist'], np.arange(m)), 'swap': (d['hist'][perm], np.arange(m))}
    r = d['hist'].copy(); r[:, :, :3] *= -1; H['reverse'] = (r, np.arange(m)); H['zero'] = (np.zeros_like(d['hist']), np.arange(m))
    key = {(int(t), int(e), int(f)): i for i, (t, e, f) in enumerate(zip(d['task'], d['episode'], d['frame']))}
    for dt in shifts:
        src = np.array([key.get((int(t), int(e), int(f) + dt), -1) for t, e, f in zip(d['task'], d['episode'], d['frame'])])
        ok = np.flatnonzero(src >= 0); hh = d['hist'].copy(); hh[ok] = d['hist'][src[ok]]; H[f'phase{dt:+d}'] = (hh, ok)
    return H
HE = variants(ev, 4101); HD = variants(dv, 5)
mix = {'true': .5, 'swap': .1, 'reverse': .1, 'zero': .1, 'phase-20': .05, 'phase-10': .05, 'phase+10': .05, 'phase+20': .05}
nn_ev, nn_dv = iq.nn_scores('eval'), iq.nn_scores('dev')
fit = ~np.isin(trn['task'], so.HOLDOUT)
# training corruptions on LIBERO-40 (phase shifts within the same episode)
HT = variants(trn, 11)
res = {'oracle_switch': {}, 'det': {}}
for hn, (hh, rws) in HE.items():
    pick = [top1(sc5(P5[s], hh, ev['state']), E, A) if hn == 'true' else top1(nn_ev, E, A) for s in SEEDS]
    res['oracle_switch'][hn] = float(np.mean([p[rws].mean() for p in pick]))
print('oracle switch', {k: round(v, 4) for k, v in res['oracle_switch'].items()}, flush=True)
act = {s: np.load(iq._pf(s) / f'{s}_actions.npy').astype(np.float32) for s in ('train', 'dev', 'eval')}
for feat in ('vlm', 'mid17', 'act'):
    F = {s: iq.feature(s, feat) for s in ('train', 'dev', 'eval')}
    def X(split, d, hh):
        return np.concatenate([F[split], d['state'], hh.reshape(len(hh), -1), g.agreement(hh, act[split])], 1).astype(np.float32)
    trX, trY = [], []
    corr_w = {'swap': .25, 'reverse': .25, 'zero': .15, 'phase-20': .0875, 'phase-10': .0875, 'phase+10': .0875, 'phase+20': .0875}
    rng = np.random.default_rng(0)
    base_rows = np.flatnonzero(fit)
    for k, w in corr_w.items():
        rws = np.intersect1d(HT[k][1], base_rows); take = rng.choice(rws, int(w * len(base_rows)), replace=len(rws) < int(w * len(base_rows)))
        trX.append(X('train', trn, HT[k][0])[take]); trY.append(np.ones(len(take)))
    trX.append(X('train', trn, trn['hist'])[base_rows]); trY.append(np.zeros(len(base_rows)))
    trX = np.concatenate(trX); trY = np.concatenate(trY)
    mu, sd = trX.mean(0), trX.std(0) + 1e-6
    hold = np.flatnonzero(~fit); hX = np.concatenate([X('train', trn, trn['hist'])[hold]] + [X('train', trn, HT[k][0])[np.intersect1d(HT[k][1], hold)] for k in corr_w])
    hY = np.concatenate([np.zeros(len(hold))] + [np.ones(len(np.intersect1d(HT[k][1], hold))) for k in corr_w])
    out = {'auc_eval': {}, 'eval': {}}; dets = []
    for s in SEEDS:
        torch.manual_seed(s); net = nn.Sequential(nn.Dropout(.2), nn.Linear(trX.shape[1], 256), nn.GELU(), nn.Linear(256, 1))
        opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-2)
        xt = torch.as_tensor((trX - mu) / sd); yt = torch.as_tensor(trY, dtype=torch.float32); xh = torch.as_tensor((hX - mu) / sd); best = (-1, None)
        for ep in range(10):
            net.train(); pm = torch.randperm(len(xt))
            for i in range(0, len(xt), 512):
                b = pm[i:i + 512]; l = nn.functional.binary_cross_entropy_with_logits(net(xt[b])[:, 0], yt[b]); opt.zero_grad(); l.backward(); opt.step()
            net.eval()
            with torch.no_grad(): auc = roc_auc_score(hY, net(xh)[:, 0].numpy())
            if auc > best[0]: best = (auc, {k: v.clone() for k, v in net.state_dict().items()})
        net.load_state_dict(best[1]); net.eval(); dets.append((net, best[0]))
    def prob(net, split, d, hh):
        with torch.no_grad(): return torch.sigmoid(net(torch.as_tensor((X(split, d, hh) - mu) / sd))[:, 0]).numpy()
    for hn, (hh, rws) in HE.items():
        if hn == 'true': continue
        out['auc_eval'][hn] = float(np.mean([roc_auc_score(np.r_[np.zeros(len(rws)), np.ones(len(rws))],
                                np.r_[prob(nt, 'eval', ev, ev['hist'])[rws], prob(nt, 'eval', ev, hh)[rws]]) for nt, _ in dets]))
    # threshold on Dev
    thrs = (.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.01)
    def sw(j, s, split, d, H, EE, AA, nnsc, thr, rws):
        pr = prob(dets[j][0], split, d, H); a = top1(sc5(P5[s], H, d['state']), EE, AA); b = top1(nnsc, EE, AA)
        return np.where(pr >= thr, b, a)[rws].mean()
    dsc = {t: np.mean([sum(w * sw(j, s, 'dev', dv, HD[k][0], Ed, Ad, nn_dv, t, HD[k][1]) for k, w in mix.items()) for j, s in enumerate(SEEDS)]) for t in thrs}
    thr = min(dsc, key=dsc.get); out['thr_dev'] = thr; out['auc_holdout'] = float(np.mean([d[1] for d in dets]))
    for hn, (hh, rws) in HE.items():
        out['eval'][hn] = float(np.mean([sw(j, s, 'eval', ev, hh, E, A, nn_ev, thr, rws) for j, s in enumerate(SEEDS)]))
        out['eval'][hn + '_switch_rate'] = float(np.mean([(prob(dets[j][0], 'eval', ev, hh)[rws] >= thr).mean() for j in range(3)]))
    out['dev_curve'] = {str(t): float(v) for t, v in dsc.items()}
    out['eval_curve_diagnostic'] = {str(t): {hn: float(np.mean([sw(j, s, 'eval', ev, hh, E, A, nn_ev, t, rws) for j, s in enumerate(SEEDS)])) for hn, (hh, rws) in HE.items()} for t in (.3, .5, .7, .9)}
    for t, v in out['eval_curve_diagnostic'].items(): print('   diag thr', t, {k: round(x, 4) for k, x in v.items()})
    print('   dev curve', {k: round(v, 4) for k, v in out['dev_curve'].items()})
    res['det'][feat] = out
    print(feat, 'AUC hold', round(out['auc_holdout'], 3), 'AUC eval', {k: round(v, 3) for k, v in out['auc_eval'].items()}, 'thr', thr)
    print('   top1', {k: round(v, 4) for k, v in out['eval'].items() if 'rate' not in k}, '\n   switch rate', {k[:-12]: round(v, 2) for k, v in out['eval'].items() if 'rate' in k}, flush=True)
(pp.OUT / 'consistency.json').write_text(json.dumps(res, indent=1) + '\n')
