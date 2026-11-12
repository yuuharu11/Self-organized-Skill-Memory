"""PD1: start pi0.5's flow from the type P5 picked (anchor), let it correct, and re-pick the memory type.

Draft (no ground truth): P5's top-1 W type, size by a run-time rule (stored | hist = speed of the executed history |
pi05 = speed of pi0.5's own chunk), gripper by S1's G1 (hold the last executed command), normalized, padded to 32.
Partial denoising: public pi05_libero (frozen), x_t0 = (1 - t0) draft + t0 eps, 10 Euler steps t0 -> 0
(pi05_memory.core.sample_bridge), fixed noise fold_in(4101, 2_000_000 + unseen row id).
Re-pick: the W type nearest (S-shared type error) to the corrected chunk -> scored by its true type error.
Detector: d = S-shared type error between draft and corrected chunk (how hard pi0.5 pulls the draft away).
Switch: corrected re-pick if d >= q else P5's pick; q = Dev quantile.
Choices (t0, size rule, q) on Dev (temporary; L90X selection tasks later).  Eval targets for scoring only.
Grid: seed 4101 over t0 x rule x {true, reverse, phase-20}; the Dev-chosen setting with 3 seeds and all histories.
Output: image-query-01/partial-denoise/.
"""
from __future__ import annotations

import json, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path: sys.path.insert(0, str(HERE))
import image_query as iq  # noqa: E402

OUT = iq.OUT / 'partial-denoise'
T0S = (.1, .2, .3, .5, .7, 1.0)
RULES = ('stored', 'hist', 'pi05')
SEEDS = (4101, 4102, 4103)
NOISE_OFFSET = 2_000_000


class Ctx:
    def __init__(self):
        import torch, jax, scale_only_query as so, compressed_denoise as cd
        self.so, self.cd, self.torch = so, cd, torch
        _, self.vs, self.kq = so._mods(); P = iq.S1 / 'prep'
        self.pool = dict(np.load(P / 'W_pool.npz')); self.KI = torch.as_tensor(so.key_inputs('W', self.pool, iq.COND))
        self.thr = json.loads((P / 'prep.json').read_text())['gripper_threshold']
        base = cd._module(cd.PI05 / 'run.py', 'pd_base'); self.base = base
        self.model, _ = base.load_policy(cd.CHECKPOINT)
        self.unseen = {p.stem: np.load(p, mmap_mode='r', allow_pickle=False) for p in cd.UNSEEN_ROWS.glob('*.npy')}
        self.split = {}
        for split in ('dev', 'eval'):
            d = dict(np.load(P / f'W_inputs_{split}.npz')); d['pi05'] = np.load(iq._pf(split) / f'{split}_actions.npy').astype(np.float32)
            ids = np.asarray(d['row'], np.int64)
            eps = np.stack([np.asarray(jax.random.normal(jax.random.fold_in(jax.random.key(4101), NOISE_OFFSET + int(i)), (10, 32))) for i in ids])
            A = so.allowed_matrix(d['task'], self.pool['task'], self.pool['keep'], False)
            y = np.asarray(self.unseen['actions'][ids], np.float32)[:, :, :6]       # scoring (Eval) / choices (Dev) only
            E = so.type_errors(self.vs.to_physical(y).astype(np.float32), self.pool['phys'], iq.COND, 'cuda')
            self.split[split] = {'d': d, 'ids': ids, 'eps': eps, 'A': A, 'E': E, 'obs': base.obs_batch(self.unseen, ids)}
        self.p5 = {}
        for s in SEEDS:
            sel = json.loads((iq.S1 / 'selection.json').read_text())['selected'][f'W|{iq.COND}|P5|{s}']['holdout']
            rr = torch.load(iq.S1 / 'models' / f"W_{iq.COND}_P5_s{s}_f{sel['frac']:.4f}.pt", weights_only=False)
            m = so.make_net('W', 'P5', 78, self.KI.shape[1]); m.load_state_dict(rr['holdout']['state']); m.eval(); self.p5[s] = m

    def variants(self, d, seed, names):
        rng = np.random.default_rng(seed); m = len(d['task']); H = {}
        perm = np.array([rng.choice(np.flatnonzero(d['task'] != d['task'][i])) if (d['task'] != d['task'][i]).any() else rng.integers(m) for i in range(m)])
        r = d['hist'].copy(); r[:, :, :3] *= -1
        base = {'true': d['hist'], 'swap': d['hist'][perm], 'reverse': r, 'zero': np.zeros_like(d['hist'])}
        key = {(int(t), int(e), int(f)): i for i, (t, e, f) in enumerate(zip(d['task'], d['episode'], d['frame']))}
        for nm in names:
            if nm in base: H[nm] = (base[nm], np.arange(m)); continue
            dt = int(nm[5:]); src = np.array([key.get((int(t), int(e), int(f) + dt), -1) for t, e, f in zip(d['task'], d['episode'], d['frame'])])
            ok = np.flatnonzero(src >= 0); hh = d['hist'].copy(); hh[ok] = d['hist'][src[ok]]; H[nm] = (hh, ok)
        return H

    def p5_pick(self, s, d, hh, A):
        torch = self.torch
        with torch.no_grad():
            sc = self.p5[s](torch.cat([torch.as_tensor(hh).reshape(len(hh), -1), torch.as_tensor(d['state'])], 1), self.KI).numpy()
        return np.where(A, sc, -np.inf).argmax(1)

    def draft(self, pick, d, hh, rule):
        inp = {**d, 'hist': hh}
        six = self.so.generate_w(self.pool, pick, iq.COND, rule, inp)                # normalized [N,10,6], no ground truth
        g = self.so.gripper_g1(inp, self.thr, 'W')
        out = np.zeros((len(pick), 10, 32), np.float32); out[:, :, :6] = six; out[:, :, 6] = g[:, None]; return out

    def denoise(self, split, draft, t0):
        S = self.split[split]
        return self.cd._predict_action(self.model, S['obs'], draft, S['eps'], t0)[:, :, :7]

    def repick(self, chunk, A):
        e = self.so.type_errors(self.vs.to_physical(chunk[:, :, :6]).astype(np.float32), self.pool['phys'], iq.COND, 'cuda')
        return np.where(A, -e, -np.inf).argmax(1)

    def draft_gap(self, draft, chunk):
        """S-shared type error between the draft's and the corrected chunk's 6-axis motion (per row)."""
        a = self.vs.to_physical(draft[:, :, :6]).astype(np.float32); b = self.vs.to_physical(chunk[:, :, :6]).astype(np.float32)
        return np.array([self.so.scale_only_error(b[i:i + 1], a[i:i + 1], iq.COND)[0, 0] for i in range(len(a))])


def main():
    OUT.mkdir(parents=True, exist_ok=True); C = Ctx(); so = C.so
    log = lambda **r: (print(json.dumps(r, ensure_ascii=False), flush=True), open(OUT / 'progress.log', 'a').write(json.dumps(r, ensure_ascii=False) + '\n'))
    res = {'grid': {}, 'chosen': {}, 'baselines': {}}
    for split in ('dev', 'eval'):
        S = C.split[split]; E, A = S['E'], S['A']
        res['baselines'][split] = {'oracle': float(np.where(A, E, np.inf).min(1).mean()),
                                   'pi05_nn': float(E[np.arange(len(E)), C.repick(S['d']['pi05'], A)].mean())}
    # ---------------- grid (seed 4101)
    names = ('true', 'reverse', 'phase-20')
    for split in ('dev', 'eval'):
        S = C.split[split]; d, E, A = S['d'], S['E'], S['A']; H = C.variants(d, 4101 if split == 'eval' else 5, names)
        for hn, (hh, rws) in H.items():
            pick = C.p5_pick(4101, d, hh, A); te5 = E[np.arange(len(E)), pick]
            res['grid'].setdefault(split, {}).setdefault(hn, {})['P5'] = float(te5[rws].mean())
            for rule in RULES:
                dr = C.draft(pick, d, hh, rule)
                for t0 in T0S:
                    t = time.time(); ch = C.denoise(split, dr, t0); rp = C.repick(ch, A); te = E[np.arange(len(E)), rp]
                    gap = C.draft_gap(dr, ch)
                    res['grid'][split][hn][f'{rule}|{t0}'] = {'repick': float(te[rws].mean()), 'gap_mean': float(gap[rws].mean())}
                    log(split=split, hist=hn, rule=rule, t0=t0, repick=round(float(te[rws].mean()), 4), P5=round(float(te5[rws].mean()), 4),
                        sec=round(time.time() - t, 1))
                    np.savez(OUT / f'grid_{split}_{hn}_{rule}_{t0}.npz', chunk=ch, repick=rp, pick=pick, gap=gap)
        (OUT / 'results.json').write_text(json.dumps(res, indent=1) + '\n')
    # ---------------- choose (rule, t0) on Dev: mean over the three histories (true weighted .5)
    w = {'true': .5, 'reverse': .25, 'phase-20': .25}
    dv = res['grid']['dev']
    best = min(((r, t) for r in RULES for t in T0S), key=lambda k: sum(w[h] * dv[h][f'{k[0]}|{k[1]}']['repick'] for h in w))
    res['chosen']['setting'] = {'rule': best[0], 't0': best[1], 'by': 'Dev, top-1 type error, true .5 / reverse .25 / phase-20 .25'}
    log(chosen=res['chosen']['setting'])
    (OUT / 'results.json').write_text(json.dumps(res, indent=1) + '\n')


if __name__ == '__main__':
    main()
