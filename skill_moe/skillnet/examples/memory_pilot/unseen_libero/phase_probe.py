"""PH1: does a task-agnostic phase, read from the image, stop P5 from drifting with a wrong history?

Phase (no human labels; from the demo's own gripper commands, task-agnostic):
  0 = no close command yet (approach), 1 = gripper closed (holding), 2 = open again after having closed (after release)
  taken at the window start from the commands of the frames before it (episodes start open).
Memory side: each W pool window gets the phase of its LIBERO-40 source row.  Query side (Eval): the true phase is
used ONLY for scoring and for the oracle upper bound.
Phase predictor: image (+ state, never the action history) -> 3-way softmax, trained on LIBERO-40 rows (fit tasks),
features one of: VLM group means (S1 'mean'), mid layer 17 / 14 / 10 image means, pi0.5 chunk, Expert hidden.
Combination: score = z(P5) + lam * log P(phase(candidate) | image, state);  lam chosen on Dev (true history
half, corrupted half; grid), Eval never used for choices.
Eval: top-1 type error (main), with true / swap / reverse / zero / phase-shift histories.
Output: image-query-01/phase/.
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path: sys.path.insert(0, str(HERE))
import image_query as iq  # noqa: E402

OUT = iq.OUT / 'phase'
SEEDS = (4101, 4102, 4103)


def phases_of(task, episode, frame, actions):
    """Phase at each window start from the gripper commands (dim 6, +1 close / -1 open) of earlier frames.
    Windows tile each episode with stride 10, so the episode's command sequence is the concatenation."""
    task, episode, frame = map(np.asarray, (task, episode, frame)); ph = np.zeros(len(task), np.int64)
    for key in set(zip(task.tolist(), episode.tolist())):
        idx = np.flatnonzero((task == key[0]) & (episode == key[1])); idx = idx[np.argsort(frame[idx])]
        closed_ever = False; closed = False
        for i in idx:
            ph[i] = 1 if closed else (2 if closed_ever else 0)
            for gcmd in np.asarray(actions[i, :, 6]):
                closed = gcmd > 0; closed_ever |= closed
    return ph


def main():
    import torch, torch.nn as nn, scale_only_query as so
    OUT.mkdir(parents=True, exist_ok=True)
    cd, vs, kq = so._mods(); P = iq.S1 / 'prep'; pool = dict(np.load(P / 'W_pool.npz'))
    base, rows, unseen, splits = kq._production_data_splits(include_eval_targets=False)
    tid = np.asarray(splits['train'][1]); dev_ids = np.asarray(splits['dev'][1]); ev_ids = np.asarray(splits['eval'][1])
    # ---- phases: LIBERO-40 rows (all rows, so that episodes are complete), pool windows via their source rows
    ra = np.load(cd.RUN / 'rows' / 'actions.npy', mmap_mode='r')
    ph40 = phases_of(rows['task'], rows['episode'], rows['frame'], ra)
    import theta_regression as tr
    p0 = tr.candidate_shape_pool(rep='C_vm_none'); keep = ~np.isin(p0['source_task'], so.HOLDOUT)
    pool_ph = ph40[p0['source_row'][keep]]
    if len(pool_ph) != len(pool['task']) or not np.array_equal(p0['source_task'][keep], pool['task']): raise AssertionError('pool misaligned')
    ua = np.load(kq.UNSEEN / 'actions.npy', mmap_mode='r')                   # Dev labels: choices; Eval labels: scoring only
    ut, ue, uf = (np.asarray(np.load(kq.UNSEEN / f'{k}.npy')) for k in ('task', 'episode', 'frame'))
    phu = phases_of(ut, ue, uf, ua)
    ph_tr, ph_dev, ph_ev = ph40[tid], phu[dev_ids], phu[ev_ids]
    info = {'pool_phase_counts': np.bincount(pool_ph, minlength=3).tolist(), 'train_phase_counts': np.bincount(ph_tr, minlength=3).tolist(),
            'eval_phase_counts': np.bincount(ph_ev, minlength=3).tolist(), 'dev_phase_counts': np.bincount(ph_dev, minlength=3).tolist()}
    print(info, flush=True)
    # ---- Eval scoring material
    ev = dict(np.load(P / 'W_inputs_eval.npz')); dv = dict(np.load(P / 'W_inputs_dev.npz')); trn = dict(np.load(P / 'W_inputs_train.npz'))
    y = np.asarray(ua[ev['row'], :, :6], np.float32)
    E = so.type_errors(vs.to_physical(y).astype(np.float32), pool['phys'], iq.COND, 'cuda')
    A = so.allowed_matrix(ev['task'], pool['task'], pool['keep'], False); hard = ev['hard'].astype(bool); n = len(E)
    Ed = np.load(P / f'W_err_{iq.COND}_dev.npz')['err']; Ad = so.allowed_matrix(dv['task'], pool['task'], pool['keep'], False)
    top1 = lambda sc, EE, AA: EE[np.arange(len(EE)), np.where(AA, sc, -np.inf).argmax(1)]
    # history variants (as I4)
    def variants(d, seed):
        rng = np.random.default_rng(seed); m = len(d['task'])
        perm = np.array([rng.choice(np.flatnonzero(d['task'] != d['task'][i])) if (d['task'] != d['task'][i]).any() else rng.integers(m) for i in range(m)])
        H = {'true': (d['hist'], np.arange(m)), 'swap': (d['hist'][perm], np.arange(m))}
        r = d['hist'].copy(); r[:, :, :3] *= -1; H['reverse'] = (r, np.arange(m)); H['zero'] = (np.zeros_like(d['hist']), np.arange(m))
        key = {(int(t), int(e), int(f)): i for i, (t, e, f) in enumerate(zip(d['task'], d['episode'], d['frame']))}
        for dt in (-20, -10, 10, 20):
            src = np.array([key.get((int(t), int(e), int(f) + dt), -1) for t, e, f in zip(d['task'], d['episode'], d['frame'])])
            ok = np.flatnonzero(src >= 0); hh = d['hist'].copy(); hh[ok] = d['hist'][src[ok]]; H[f'phase{dt:+d}'] = (hh, ok)
        return H
    HE = variants(ev, 4101); HD = variants(dv, 5)
    KI = torch.as_tensor(so.key_inputs('W', pool, iq.COND))
    def p5(seed):
        sel = json.loads((iq.S1 / 'selection.json').read_text())['selected'][f'W|{iq.COND}|P5|{seed}']['holdout']
        rr = torch.load(iq.S1 / 'models' / f"W_{iq.COND}_P5_s{seed}_f{sel['frac']:.4f}.pt", weights_only=False)
        m = so.make_net('W', 'P5', 78, KI.shape[1]); m.load_state_dict(rr['holdout']['state']); m.eval(); return m
    P5 = {s: p5(s) for s in SEEDS}
    def p5_scores(m, hh, state):
        with torch.no_grad(): return m(torch.cat([torch.as_tensor(hh).reshape(len(hh), -1), torch.as_tensor(state)], 1), KI).numpy()
    # ---- phase predictors (image + state only)
    feats = {'vlm': 'vlm', 'mid17': 'mid17', 'mid14': 'mid14', 'mid10': 'mid10', 'act': 'act', 'hfin': 'hfin'}
    def X_of(name, split):
        x = iq.feature(split, feats[name]) if name != 'state' else None
        st = {'train': trn, 'dev': dv, 'eval': ev}[split]['state']
        return np.concatenate([x, st], 1).astype(np.float32)
    fit = ~np.isin(trn['task'], so.HOLDOUT)
    res = {'info': info, 'oracle_phase': {}, 'predictors': {}}
    # oracle: restrict candidates to the true phase (upper bound of a perfect phase detector)
    Aor = A & (pool_ph[None, :] == ph_ev[:, None])
    for hn, (hh, rws) in HE.items():
        v = [top1(p5_scores(P5[s], hh, ev['state']), E, A)[rws].mean() for s in SEEDS]
        vo = [top1(p5_scores(P5[s], hh, ev['state']), E, Aor)[rws].mean() for s in SEEDS]
        res['oracle_phase'][hn] = {'P5': float(np.mean(v)), 'P5_true_phase_mask': float(np.mean(vo)),
                                   'pi05_nn': float(top1(iq.nn_scores('eval'), E, A)[rws].mean()),
                                   'pi05_nn_true_phase_mask': float(top1(iq.nn_scores('eval'), E, Aor)[rws].mean())}
        print('oracle', hn, {k: round(x, 4) for k, x in res['oracle_phase'][hn].items()}, flush=True)
    res['oracle_phase']['best_possible_true_phase'] = float(np.where(Aor, E, np.inf).min(1).mean())
    dev_mix = {'true': .5, 'swap': .1, 'reverse': .1, 'zero': .1, 'phase-20': .05, 'phase-10': .05, 'phase+10': .05, 'phase+20': .05}
    for name in feats:
        Xtr, Xdv, Xev = X_of(name, 'train'), X_of(name, 'dev'), X_of(name, 'eval')
        mu, sd = Xtr[fit].mean(0), Xtr[fit].std(0) + 1e-6
        Xtr, Xdv, Xev = [(x - mu) / sd for x in (Xtr, Xdv, Xev)]
        out = {'acc': {}, 'eval': {}}
        logp_ev, logp_dv = [], []
        for s in SEEDS:
            torch.manual_seed(s)
            net = nn.Sequential(nn.Dropout(.2), nn.Linear(Xtr.shape[1], 256), nn.GELU(), nn.Linear(256, 3))
            opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-2); xt = torch.as_tensor(Xtr[fit]); yt = torch.as_tensor(ph_tr[fit])
            xh = torch.as_tensor(Xtr[~fit]); yh = ph_tr[~fit]; best = (-1, None)
            for ep in range(15):
                net.train(); perm = torch.randperm(len(xt))
                for i in range(0, len(xt), 256):
                    b = perm[i:i + 256]; l = nn.functional.cross_entropy(net(xt[b]), yt[b]); opt.zero_grad(); l.backward(); opt.step()
                net.eval()
                with torch.no_grad(): acc = float((net(xh).argmax(1).numpy() == yh).mean())
                if acc > best[0]: best = (acc, {k: v.clone() for k, v in net.state_dict().items()})
            net.load_state_dict(best[1]); net.eval()
            with torch.no_grad():
                lev = torch.log_softmax(net(torch.as_tensor(Xev)), 1).numpy(); ldv = torch.log_softmax(net(torch.as_tensor(Xdv)), 1).numpy()
            logp_ev.append(lev); logp_dv.append(ldv)
            out['acc'].setdefault('libero40_holdout', []).append(best[0])
            out['acc'].setdefault('dev', []).append(float((ldv.argmax(1) == ph_dev).mean()))
            out['acc'].setdefault('eval', []).append(float((lev.argmax(1) == ph_ev).mean()))
        out['acc'] = {k: float(np.mean(v)) for k, v in out['acc'].items()}
        out['acc']['eval_majority'] = float(np.bincount(ph_ev, minlength=3).max() / n)
        # lam on Dev (top-1 type error, mixed histories)
        lams = (0, .25, .5, 1, 2, 4, 8, 16)
        def comb(m, hh, state, lp, AA, EE, rws, lam):
            z = iq.zrow(p5_scores(m, hh, state), AA); return top1(z + lam * lp[:, pool_ph], EE, AA)[rws].mean()
        dsc = {lam: np.mean([sum(w * comb(P5[s], HD[k][0], dv['state'], logp_dv[j], Ad, Ed, HD[k][1], lam) for k, w in dev_mix.items())
                              for j, s in enumerate(SEEDS)]) for lam in lams}
        lam = min(dsc, key=dsc.get); out['lam_dev'] = lam; out['dev_curve'] = {str(k): float(v) for k, v in dsc.items()}
        for hn, (hh, rws) in HE.items():
            out['eval'][hn] = {'P5+phase': float(np.mean([comb(P5[s], hh, ev['state'], logp_ev[j], A, E, rws, lam) for j, s in enumerate(SEEDS)])),
                               'P5+phase_hard': float(np.mean([comb(P5[s], hh, ev['state'], logp_ev[j], A, E, np.intersect1d(rws, np.flatnonzero(hard)), lam) for j, s in enumerate(SEEDS)]))}
        res['predictors'][name] = out
        print(name, 'acc', {k: round(v, 3) for k, v in out['acc'].items()}, 'lam', lam,
              {hn: round(v['P5+phase'], 4) for hn, v in out['eval'].items()}, flush=True)
    (OUT / 'results.json').write_text(json.dumps(res, indent=1) + '\n')


if __name__ == '__main__':
    main()
