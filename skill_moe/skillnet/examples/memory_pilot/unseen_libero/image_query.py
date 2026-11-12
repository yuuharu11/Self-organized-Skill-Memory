"""I1: which image-derived query ranks the stored motion types best on unseen LIBERO-90?

Same setting as S1 (scale_only_query.py, run scale-only-02): W pool (1,595 ten-frame windows of LIBERO-40),
S-shared type (one scale for the 6 axes removed, orientation kept), Eval = 619 unseen LIBERO-90 windows with the
corrected (upright) images.  The metric is S1's: per Eval window, Spearman between the query's candidate scores
and -(true type error) over the allowed candidates, averaged over windows (also on the hard rows).

Stages
  extract   pi0.5 Expert hidden states and prefix mid-layer V (K1 recipe, key_query_probe.extract_policy_feature_stage)
            on the corrected Dev / Eval rows only.  The LIBERO-40 train features of K1 are upright already and reused.
            The stored pi0.5 actions must equal orientation-fix-01's (same fixed noise key) -> checked.
  train     query networks with S1's recipe (soft teacher over LIBERO-40 type errors, 3 teacher temperatures,
            selection on the LIBERO-40 holdout tasks; Eval never used for selection).
  fuse      training-free / late fusion: z-scored P5 logits + w * z-scored score of an image query; w chosen on
            the LIBERO-40 holdout tasks.
  evaluate  Spearman / Pearson / top-1 type error / 6-axis MSE with stored scale for every query.

Eval ground truth is read only in `evaluate` (scoring).  Outputs: $SKILLNET_STATE_DIR/outputs/memory-pilot/
unseen-libero/image-query-01/.
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path: sys.path.insert(0, str(HERE))
from unseen_paths import UNSEEN_ROOT, UNSEEN_ROWS  # noqa: E402

OUT = UNSEEN_ROOT / 'image-query-01'
S1 = UNSEEN_ROOT / 'scale-only-02' / 'production'
K1PF = UNSEEN_ROOT / 'key-query-01' / 'policy_features'
FIX = UNSEEN_ROOT / 'orientation-fix-01'
COND = 'S-shared'
MID_LAYERS = (6, 10, 14, 15, 16, 17)


def log(out, **rec):
    rec = {'t': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **rec}
    with open(Path(out) / 'progress.log', 'a') as f: f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print(json.dumps(rec, ensure_ascii=False), flush=True)


# ======================================================================== extract
def extract(out, batch=4):
    import key_query_probe as kq
    orig = kq._production_data_splits

    def dev_eval_only(*, include_eval_targets=False):
        base, rows, unseen, splits = orig(include_eval_targets=include_eval_targets)
        return base, rows, unseen, {k: splits[k] for k in ('dev', 'eval')}
    kq._production_data_splits = dev_eval_only
    if Path(kq.UNSEEN).resolve() != Path(UNSEEN_ROWS).resolve(): raise AssertionError('key_query_probe reads the legacy rows')
    summary = kq.extract_policy_feature_stage(output=out, batch=batch)
    pf = Path(out) / 'policy_features'; check = {}
    for split in ('dev', 'eval'):
        a = np.load(pf / f'{split}_actions.npy').astype(np.float32); z = np.load(FIX / 'pi05' / f'{split}_k1.npz')
        check[f'{split}_actions_vs_orientation_fix_maxabs'] = float(np.max(np.abs(a - z['actions'][:, :, :7])))
    (Path(out) / 'extract_check.json').write_text(json.dumps(check, indent=1) + '\n'); log(out, stage='extract', **check)
    return summary


MULTI_OFFSETS = (100_000, 200_000, 300_000, 400_000)     # extra fixed noise keys fold_in(4101, offset + row)


def pi05_multi(out, batch=8):
    """pi0.5 predictions with 4 extra fixed noises on the corrected Dev/Eval rows (no targets)."""
    from openpi.shared.nnx_utils import module_jit
    import theta_regression as tr, orientation_fix as of
    base = tr._module(tr.PI05 / 'run.py', 'image_query_run')
    model, _ = base.load_policy(tr.CHECKPOINT); fn = module_jit(model.sample_actions, static_argnames=('num_steps',))
    rows = of.rows_of(UNSEEN_ROWS); ids = of.split_ids(rows); D = Path(out) / 'pi05_multi'; D.mkdir(parents=True, exist_ok=True)
    for split in ('dev', 'eval'):
        p = D / f'{split}.npz'
        if p.exists(): continue
        acts = [np.load(_pf(split) / f'{split}_actions.npy').astype(np.float32)]
        for off in MULTI_OFFSETS:
            t0 = time.time(); acts.append(of.pi05_predictions(model, fn, rows, ids[split], off, batch))
            log(out, stage='pi05_multi', split=split, offset=off, sec=round(time.time() - t0, 1))
        np.savez(p, actions=np.stack(acts, 1), row=ids[split], offsets=np.array((0,) + MULTI_OFFSETS))
    p = D / 'train.npz'                     # LIBERO-40 train rows (upright images), same ids as K1 policy_features
    if not p.exists():
        import key_query_probe as kq
        _, trows, _, splits = kq._production_data_splits(include_eval_targets=False); tid = np.asarray(splits['train'][1])
        acts = [np.load(_pf('train') / 'train_actions.npy').astype(np.float32)]
        for off in MULTI_OFFSETS:
            t0 = time.time(); acts.append(of.pi05_predictions(model, fn, trows, tid, off, batch))
            log(out, stage='pi05_multi', split='train', offset=off, sec=round(time.time() - t0, 1))
        np.savez(p, actions=np.stack(acts, 1), row=tid, offsets=np.array((0,) + MULTI_OFFSETS))


# ======================================================================== query features
def _pf(split):
    return (K1PF if split == 'train' else OUT / 'policy_features')


def mid_image_means(split, layer, chunk=1024):
    """[N, 2*256]: per-camera mean of the 2x2-pooled image V tokens of prefix layer `layer` (64 tokens per camera)."""
    li = MID_LAYERS.index(layer); v = np.load(_pf(split) / f'{split}_mid_v.npy', mmap_mode='r'); out = []
    for i in range(0, len(v), chunk):
        x = np.asarray(v[i:i + chunk, li, :128], np.float32)
        out.append(np.concatenate([x[:, :64].mean(1), x[:, 64:128].mean(1)], 1))
    return np.concatenate(out)


def mid_lang_mean(split, layer, chunk=1024):
    li = MID_LAYERS.index(layer); v = np.load(_pf(split) / f'{split}_mid_v.npy', mmap_mode='r')
    m = np.load(_pf(split) / f'{split}_mid_mask.npy'); out = []
    for i in range(0, len(v), chunk):
        x = np.asarray(v[i:i + chunk, li, 128:], np.float32); mm = m[i:i + chunk, 128:, None]
        out.append((x * mm).sum(1) / np.maximum(mm.sum(1), 1))
    return np.concatenate(out)


def feature(split, name):
    """Image-derived query block (no target actions).  Cached in OUT/features/."""
    F = OUT / 'features'; F.mkdir(parents=True, exist_ok=True); p = F / f'{split}_{name}.npy'
    if p.exists(): return np.load(p)
    if name == 'act':                              # pi0.5 chunk, normalized action space, 10x7
        x = np.load(_pf(split) / f'{split}_actions.npy').astype(np.float32).reshape(-1, 70)
    elif name in ('hfin', 'ht1'):                  # Expert hidden (horizon mean + first step)
        h = np.load(_pf(split) / f"{split}_hidden_{'final' if name == 'hfin' else 't1'}.npy", mmap_mode='r')
        x = np.concatenate([np.asarray(h[:, :, :], np.float32).mean(1), np.asarray(h[:, 0], np.float32)], 1)
    elif name.startswith('mid'):                   # mid<layer> image means, midL<layer> language mean
        lay = int(name.lstrip('midL')); x = mid_lang_mean(split, lay) if name.startswith('midL') else mid_image_means(split, lay)
    elif name == 'vlm':                            # S1's VLM group means (theta_regression recipe)
        d = np.load(S1 / 'prep' / f'W_inputs_{split}.npz'); x = np.asarray(d['mean'], np.float32).reshape(len(d['mean']), -1)
    else: raise ValueError(name)
    np.save(p, x.astype(np.float32)); return x.astype(np.float32)


def standardize(blocks_train, blocks_other):
    mu = blocks_train.mean(0); sd = blocks_train.std(0) + 1e-6
    return (blocks_train - mu) / sd, [(b - mu) / sd for b in blocks_other]


# query modes: list of image blocks; '+H' = also history (10x7) + state (P5's inputs)
MODES = {
    'A': (['act'], False), 'A+H': (['act'], True),
    'E': (['hfin'], False), 'E+H': (['hfin'], True), 'E1+H': (['ht1'], True),
    'V+H': (['vlm'], True),
    **{f'M{l}': ([f'mid{l}'], False) for l in MID_LAYERS},
    **{f'M{l}+H': ([f'mid{l}'], True) for l in MID_LAYERS},
    'ML16+H': (['midL16'], True),
    'A+E+H': (['act', 'hfin'], True), 'A+M16+H': (['act', 'mid16'], True),
}


def inputs(mode, splits=('train', 'dev', 'eval')):
    blocks, hist = MODES[mode]; P = S1 / 'prep'; res = {}
    raw = {s: dict(np.load(P / f'W_inputs_{s}.npz')) for s in splits}
    img = {s: np.concatenate([feature(s, b) for b in blocks], 1) for s in splits}
    tr, others = standardize(img['train'], [img[s] for s in splits if s != 'train'])
    img = dict(zip(splits, [tr] + others))
    for s in splits:
        x = img[s]
        if hist:
            h = np.concatenate([raw[s]['hist'].reshape(len(x), -1), raw[s]['state']], 1).astype(np.float32)
            x = np.concatenate([x, h], 1)
        res[s] = x.astype(np.float32)
    return res, raw


# ======================================================================== train (S1 recipe)
def make_net(in_dim, key_dim, d=256):
    import torch, torch.nn as nn
    F = nn.functional

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            hid = 768 if in_dim > 300 else 512
            self.query = nn.Sequential(nn.Dropout(0.1 if in_dim > 300 else 0.), nn.Linear(in_dim, hid), nn.GELU(), nn.Linear(hid, d))
            self.key = nn.Sequential(nn.Linear(key_dim, 512), nn.GELU(), nn.Linear(512, d))
            self.log_scale = nn.Parameter(torch.tensor(float(np.log(10.))))

        def forward(self, x, kin):
            return F.normalize(self.query(x), dim=-1) @ F.normalize(self.key(kin), dim=-1).T * self.log_scale.exp()
    return Net()


def train_mode(mode, seeds=(4101, 4102, 4103), fracs=(1 / 3, 1 / 10, 1 / 30), epochs=20, device='cuda'):
    import torch, scale_only_query as so
    M = OUT / 'models'; M.mkdir(parents=True, exist_ok=True)
    X, raw = inputs(mode); P = S1 / 'prep'
    pool = dict(np.load(P / 'W_pool.npz')); KI = so.key_inputs('W', pool, COND)
    E = np.load(P / f'W_err_{COND}_train.npz')['err']; Edev = np.load(P / f'W_err_{COND}_dev.npz')['err']
    A = so.allowed_matrix(raw['train']['task'], pool['task'], pool['keep'], True)
    hold = np.isin(raw['train']['task'], so.HOLDOUT)
    Adev = so.allowed_matrix(raw['dev']['task'], pool['task'], pool['keep'], False)
    so.make_net = lambda pn, md, i, k, d=256: make_net(i, k, d)          # train_one builds nets through so.make_net
    data = {'X': X['train'], 'KI': KI, 'E': E, 'A': A, 'fit': np.flatnonzero(~hold), 'va': np.flatnonzero(hold),
            'Xdev': X['dev'], 'Adev': Adev, 'Edev': Edev}
    for seed in seeds:
        best = None
        for frac in fracs:
            path = M / f'{mode}_s{seed}_f{frac:.4f}.pt'
            if not path.exists():
                t0 = time.time(); r = so.train_one('W', mode, COND, seed, frac, data, epochs=epochs, device=device)
                r.update(mode=mode, seed=seed, in_dim=int(X['train'].shape[1])); torch.save(r, path)
                log(OUT, stage='train', mode=mode, seed=seed, frac=round(frac, 4), holdout=r['holdout']['value'],
                    epoch=r['holdout']['epoch'], dev=r['dev']['value'], sec=round(time.time() - t0, 1))
            r = torch.load(path, weights_only=False)
            if best is None or r['holdout']['value'] < best[0]: best = (r['holdout']['value'], frac)
        sel = json.loads((OUT / 'selection.json').read_text()) if (OUT / 'selection.json').exists() else {}
        sel[f'{mode}|{seed}'] = {'frac': best[1], 'holdout': best[0]}
        (OUT / 'selection.json').write_text(json.dumps(sel, indent=1) + '\n')


def scores(mode, seed, split):
    """[N, 1595] logits of the holdout-selected state; mode 'P5' loads S1's model."""
    import torch, scale_only_query as so
    P = S1 / 'prep'; pool = dict(np.load(P / 'W_pool.npz')); KI = torch.as_tensor(so.key_inputs('W', pool, COND))
    if mode in ('P5', 'P6', 'P0'):
        s = json.loads((S1 / 'selection.json').read_text())['selected'][f'W|{COND}|{mode}|{seed}']['holdout']
        r = torch.load(S1 / 'models' / f"W_{COND}_{mode}_s{seed}_f{s['frac']:.4f}.pt", weights_only=False)
        d = dict(np.load(P / f'W_inputs_{split}.npz')); X = torch.as_tensor(so.query_inputs(d, mode))
        net = so.make_net('W', mode, X.shape[1], KI.shape[1])
    else:
        frac = json.loads((OUT / 'selection.json').read_text())[f'{mode}|{seed}']['frac']
        r = torch.load(OUT / 'models' / f'{mode}_s{seed}_f{frac:.4f}.pt', weights_only=False)
        X = torch.as_tensor(inputs(mode, ('train', split) if split != 'train' else ('train',))[0][split])
        net = make_net(X.shape[1], KI.shape[1])
    net.load_state_dict(r['holdout']['state']); net.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), 4096): out.append(net(X[i:i + 4096], KI).numpy())
    return np.concatenate(out)


def nn_scores(split):
    """Training-free: -(S-shared type error between the pi0.5 chunk and each candidate)."""
    import scale_only_query as so
    cd, vs, kq = so._mods(); pool = dict(np.load(S1 / 'prep' / 'W_pool.npz'))
    a = np.load(_pf(split) / f'{split}_actions.npy').astype(np.float32)[:, :, :6]
    return -so.type_errors(vs.to_physical(a).astype(np.float32), pool['phys'], COND, 'cuda')


# ======================================================================== metrics
def spearman_rows(sc, E, A):
    from scipy.stats import rankdata
    out = np.empty(len(E))
    for i in range(len(E)):
        a = rankdata(sc[i, A[i]]); b = rankdata(-E[i, A[i]])
        out[i] = np.corrcoef(a, b)[0, 1]
    return out


def zrow(s, A):
    m = np.where(A, s, np.nan); mu = np.nanmean(m, 1, keepdims=True); sd = np.nanstd(m, 1, keepdims=True) + 1e-9
    return (s - mu) / sd


def fuse_weight(sa, sb, E, A, rows, ws=np.linspace(0, 3, 31)):
    za, zb = zrow(sa[rows], A[rows]), zrow(sb[rows], A[rows])
    res = [(float(spearman_rows(za + w * zb, E[rows], A[rows]).mean()), float(w)) for w in ws]
    return max(res)[1], res


def evaluate(modes, fusions, seeds=(4101, 4102, 4103)):
    import scale_only_query as so
    cd, vs, kq = so._mods(); P = S1 / 'prep'; pool = dict(np.load(P / 'W_pool.npz'))
    ev = dict(np.load(P / 'W_inputs_eval.npz')); tr = dict(np.load(P / 'W_inputs_train.npz'))
    ua = np.load(kq.UNSEEN / 'actions.npy', mmap_mode='r'); y = np.asarray(ua[ev['row'], :, :6], np.float32)   # scoring only
    Eev = so.type_errors(vs.to_physical(y).astype(np.float32), pool['phys'], COND, 'cuda')
    Aev = so.allowed_matrix(ev['task'], pool['task'], pool['keep'], False); hard = ev['hard'].astype(bool)
    Etr = np.load(P / f'W_err_{COND}_train.npz')['err']; Atr = so.allowed_matrix(tr['task'], pool['task'], pool['keep'], True)
    hold = np.flatnonzero(np.isin(tr['task'], so.HOLDOUT))
    orc = np.where(Aev, Eev, np.inf).min(1); rnd = np.where(Aev, Eev, 0).sum(1) / Aev.sum(1)

    def metrics(sc):
        sp = spearman_rows(sc, Eev, Aev); pick = np.where(Aev, sc, -np.inf).argmax(1); te = Eev[np.arange(len(pick)), pick]
        pred = so.generate_w(pool, pick, COND, 'stored', ev)
        mse = ((pred[:, :, :6] - y) ** 2).mean((1, 2))
        return {'spearman': float(sp.mean()), 'spearman_hard': float(sp[hard].mean()), 'top1_type_err': float(te.mean()),
                'gap_recovered': float(((rnd - te) / np.maximum(rnd - orc, 1e-9)).mean()),
                'strict_top1': float((te <= orc + .1 * (rnd - orc)).mean()), 'mse6': float(mse.mean()), 'mse6_hard': float(mse[hard].mean())}
    res = json.loads((OUT / 'results.json').read_text()) if (OUT / 'results.json').exists() else {}
    cache = {}

    def sc_of(mode, seed, split):
        k = (mode, seed, split)
        if k not in cache: cache[k] = nn_scores(split) if mode == 'pi05_nn' else scores(mode, seed, split)
        return cache[k]
    for mode in modes:
        ms = [metrics(sc_of(mode, s, 'eval')) for s in ((None,) if mode == 'pi05_nn' else seeds)]
        res[mode] = {k: [float(np.mean([m[k] for m in ms])), float(np.min([m[k] for m in ms])), float(np.max([m[k] for m in ms]))] for k in ms[0]}
        log(OUT, stage='evaluate', mode=mode, **{k: round(v[0], 4) for k, v in res[mode].items()})
    for a, b in fusions:                     # z(a) + w z(b), w picked on the LIBERO-40 holdout (per seed)
        ms, wsel = [], []
        for s in seeds:
            sa_tr = sc_of(a, s, 'train'); sb_tr = sc_of(b, None if b == 'pi05_nn' else s, 'train')
            w, _ = fuse_weight(sa_tr, sb_tr, Etr, Atr, hold); wsel.append(w)
            sa, sb = sc_of(a, s, 'eval'), sc_of(b, None if b == 'pi05_nn' else s, 'eval')
            ms.append(metrics(zrow(sa, Aev) + w * zrow(sb, Aev)))
        key = f'{a}⊕{b}'
        res[key] = {k: [float(np.mean([m[k] for m in ms])), float(np.min([m[k] for m in ms])), float(np.max([m[k] for m in ms]))] for k in ms[0]}
        res[key]['w_holdout'] = wsel
        log(OUT, stage='evaluate', mode=key, w=wsel, **{k: round(v[0], 4) for k, v in res[key].items() if k != 'w_holdout'})
    (OUT / 'results.json').write_text(json.dumps(res, indent=1, ensure_ascii=False) + '\n')
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=('extract', 'pi05multi', 'train', 'evaluate'))
    ap.add_argument('--modes', default=''); ap.add_argument('--fusions', default=''); ap.add_argument('--batch', type=int, default=4)
    a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    modes = [m for m in a.modes.split(',') if m]
    if a.stage == 'extract': extract(OUT, a.batch)
    elif a.stage == 'pi05multi': pi05_multi(OUT, a.batch)
    elif a.stage == 'train':
        for m in modes: train_mode(m)
    else: evaluate(modes, [tuple(f.split(':')) for f in a.fusions.split(',') if f])


if __name__ == '__main__':
    main()
