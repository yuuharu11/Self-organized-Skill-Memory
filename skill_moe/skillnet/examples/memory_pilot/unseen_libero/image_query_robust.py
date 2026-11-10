"""I1 robustness: does the image part keep ranking when the action history points the wrong way?

Offline Eval windows carry the demonstrator's own past actions, so 'continue the history' is always right there.
Here the images (and pi0.5's output computed from them) stay true while only the query's history is corrupted:
  swap     history (and the 'has history' flag) of a random Eval window from another task
  reverse  translation axes (0-2) of the history negated
  zero     history removed (as at an episode start)
State is kept.  Also: does the P5 / pi0.5-NN rank disagreement flag windows where P5's top-1 is wrong?
Eval targets are read for scoring only.  Output: image-query-01/robust.json.
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path: sys.path.insert(0, str(HERE))
import image_query as iq  # noqa: E402


def p5_scores(d, seed):
    import torch, scale_only_query as so
    P = iq.S1 / 'prep'; pool = dict(np.load(P / 'W_pool.npz')); KI = torch.as_tensor(so.key_inputs('W', pool, iq.COND))
    s = json.loads((iq.S1 / 'selection.json').read_text())['selected'][f'W|{iq.COND}|P5|{seed}']['holdout']
    r = torch.load(iq.S1 / 'models' / f"W_{iq.COND}_P5_s{seed}_f{s['frac']:.4f}.pt", weights_only=False)
    X = torch.as_tensor(so.query_inputs(d, 'P5')); net = so.make_net('W', 'P5', X.shape[1], KI.shape[1])
    net.load_state_dict(r['holdout']['state']); net.eval()
    with torch.no_grad(): return net(X, KI).numpy()


def main(seeds=(4101, 4102, 4103), w=3.0):
    import scale_only_query as so
    from scipy.stats import spearmanr
    cd, vs, kq = so._mods(); P = iq.S1 / 'prep'; pool = dict(np.load(P / 'W_pool.npz'))
    ev = dict(np.load(P / 'W_inputs_eval.npz'))
    ua = np.load(kq.UNSEEN / 'actions.npy', mmap_mode='r'); y = np.asarray(ua[ev['row'], :, :6], np.float32)
    E = so.type_errors(vs.to_physical(y).astype(np.float32), pool['phys'], iq.COND, 'cuda')
    A = so.allowed_matrix(ev['task'], pool['task'], pool['keep'], False); hard = ev['hard'].astype(bool)
    orc = np.where(A, E, np.inf).min(1); rnd = np.where(A, E, 0).sum(1) / A.sum(1)
    nn = iq.nn_scores('eval'); znn = iq.zrow(nn, A)
    rng = np.random.default_rng(4101); n = len(ev['task'])
    perm = np.empty(n, np.int64)
    for i in range(n):
        c = np.flatnonzero(ev['task'] != ev['task'][i]); perm[i] = rng.choice(c)
    variants = {'true': dict(ev)}
    v = dict(ev); v['hist'] = ev['hist'][perm]; v['has_hist'] = ev['has_hist'][perm]; variants['swap'] = v
    v = dict(ev); h = ev['hist'].copy(); h[:, :, :3] *= -1; v['hist'] = h; variants['reverse'] = v
    v = dict(ev); v['hist'] = np.zeros_like(ev['hist']); v['has_hist'] = np.zeros_like(ev['has_hist']); variants['zero'] = v

    def m(sc):
        sp = iq.spearman_rows(sc, E, A); pick = np.where(A, sc, -np.inf).argmax(1); te = E[np.arange(n), pick]
        return {'spearman': float(sp.mean()), 'spearman_hard': float(sp[hard].mean()),
                'gap_recovered': float(((rnd - te) / np.maximum(rnd - orc, 1e-9)).mean()),
                'strict_top1': float((te <= orc + .1 * (rnd - orc)).mean())}, te
    res = {'pi05_nn': m(nn)[0]}
    det = {}
    for name, d in variants.items():
        ms5, msf = [], []
        for s in seeds:
            sc = p5_scores(d, s); a, te5 = m(sc); ms5.append(a); msf.append(m(iq.zrow(sc, A) + w * znn)[0])
            if name == 'true':                        # detection: disagreement vs P5's top-1 being bad
                dis = 1 - np.array([spearmanr(sc[i, A[i]], nn[i, A[i]]).statistic for i in range(n)])
                bad = te5 > orc + .1 * (rnd - orc)
                from sklearn.metrics import roc_auc_score
                det[s] = {'auc_disagreement_vs_p5_wrong': float(roc_auc_score(bad, dis)),
                          'spearman_disagreement_vs_p5_err': float(spearmanr(dis, te5 - orc).statistic),
                          'p5_wrong_rate': float(bad.mean())}
        avg = lambda ms: {k: float(np.mean([x[k] for x in ms])) for k in ms[0]}
        res[f'P5|{name}'] = avg(ms5); res[f'P5+pi05nn(w={w})|{name}'] = avg(msf)
        print(name, 'P5', {k: round(v, 3) for k, v in res[f'P5|{name}'].items()}, 'fused', {k: round(v, 3) for k, v in res[f'P5+pi05nn(w={w})|{name}'].items()}, flush=True)
    res['detection'] = det; print('pi05_nn', res['pi05_nn']); print('detection', det)
    (iq.OUT / 'robust.json').write_text(json.dumps(res, indent=1) + '\n')


if __name__ == '__main__':
    main(w=float(sys.argv[1]) if len(sys.argv) > 1 else 3.0)
