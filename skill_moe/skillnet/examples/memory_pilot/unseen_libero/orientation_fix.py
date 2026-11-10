"""Recompute the image-dependent inputs of the unseen LIBERO-90 Dev/Eval rows after the orientation fix.

The first offline rows (oracle-01/unseen_rows) had their images rotated 180 degrees; the corrected rows are
``unseen_paths.UNSEEN_ROWS``.  The LIBERO-40 side was always correct, so nothing trained there is redone.
This script re-extracts, on the corrected Dev (113) / Eval (619) rows, only what was computed from images:

  pi05      pi0.5 (public pi05_libero) 10-step predictions with the same fixed noise as before:
              key 'k1'     fold_in(key(4101), unseen row id)                 (K1 policy_features -> S1 'pi05')
              key 'global' fold_in(key(4101), len(LIBERO-40 rows) + row id)  (theta_regression._normal_eval_predictions,
                                                                             = oracle-01/official_normal_eval.npz)
            plus a pipeline check: the 'k1' predictions on the LEGACY rows must reproduce the stored K1 values.
  features  frozen VLM prefix group means [base camera, wrist camera, language] x 2048 (the 6144-d 'mean' of
            K1/K3/S1; theta_regression.extract_features recipe).

No ground-truth action is read.  Outputs go to a new directory (default $U/orientation-fix-01) and are never
overwritten.

    python orientation_fix.py {pi05,features,all} [--out DIR] [--batch 4]
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path: sys.path.insert(0, str(HERE))
from unseen_paths import UNSEEN_ROOT, UNSEEN_ROWS, LEGACY_UNSEEN_ROWS  # noqa: E402

OUT = UNSEEN_ROOT / 'orientation-fix-01'
K1_POLICY = UNSEEN_ROOT / 'key-query-01' / 'policy_features'
SEED = 4101
FIELDS = ('image_base', 'image_wrist', 'state', 'prompt', 'prompt_mask')


def log(out, **rec):
    rec = {'time_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **rec}
    print(json.dumps(rec, ensure_ascii=False), flush=True)
    with open(Path(out) / 'progress.log', 'a') as f: f.write(json.dumps(rec, ensure_ascii=False) + '\n')


def _save_new(path, **arrays):
    path = Path(path)
    if path.exists(): raise FileExistsError(f'{path} exists; outputs are never overwritten')
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_name(path.stem + '.tmp.npz'); np.savez(tmp, **arrays); tmp.replace(path)


def rows_of(root):
    return {k: np.load(Path(root) / f'{k}.npy', mmap_mode='r', allow_pickle=False) for k in FIELDS + ('is_dev',)}


def split_ids(rows):
    is_dev = np.asarray(rows['is_dev'], bool)
    dev, ev = np.flatnonzero(is_dev).astype(np.int64), np.flatnonzero(~is_dev).astype(np.int64)
    if len(dev) != 113 or len(ev) != 619: raise ValueError('fixed Dev/Eval split changed')
    return {'dev': dev, 'eval': ev}


def noise_offset(key):
    """Row-id offset of the fixed noise key (see module docstring)."""
    if key == 'k1': return 0
    if key == 'global':
        import theta_regression as tr
        return int(len(np.load(tr.RUN / 'rows' / 'actions.npy', mmap_mode='r')))
    raise ValueError(key)


def pi05_predictions(model, fn, rows, ids, offset, batch):
    """theta_regression._normal_eval_predictions with an explicit model (noise fold_in(4101, offset + id))."""
    import jax, jax.numpy as jnp
    import theta_regression as tr
    base = tr._module(tr.PI05 / 'run.py', 'orientation_fix_run'); out = []
    for st in range(0, len(ids), batch):
        q = ids[st:st + batch]; obs = base.obs_batch(rows, q)
        keys = jax.vmap(lambda r: jax.random.fold_in(jax.random.key(SEED), r))(jnp.asarray(offset + q, jnp.uint32))
        noise = jax.vmap(lambda k: jax.random.normal(k, (model.action_horizon, model.action_dim)))(keys)
        out.append(np.asarray(fn(jax.random.PRNGKey(0), obs, num_steps=10, noise=noise))[:, :, :7])
    return np.concatenate(out).astype(np.float32)


def stage_pi05(out, batch=4):
    from openpi.shared.nnx_utils import module_jit
    import theta_regression as tr
    base = tr._module(tr.PI05 / 'run.py', 'orientation_fix_run')
    model, _ = base.load_policy(tr.CHECKPOINT); fn = module_jit(model.sample_actions, static_argnames=('num_steps',))
    new, old = rows_of(UNSEEN_ROWS), rows_of(LEGACY_UNSEEN_ROWS); ids = split_ids(new)
    if not all(np.array_equal(new[k], old[k]) for k in ('state', 'prompt', 'prompt_mask', 'is_dev')):
        raise AssertionError('corrected rows differ from the legacy rows outside the images')
    check = {}
    for split in ('dev', 'eval'):           # pipeline check on the legacy (rotated) rows: must match K1
        p = out / 'pi05' / f'legacy_{split}_k1.npz'
        if not p.exists():
            t0 = time.time(); a = pi05_predictions(model, fn, old, ids[split], 0, batch); _save_new(p, actions=a, row=ids[split])
            log(out, stage='pi05', rows='legacy', split=split, key='k1', sec=round(time.time() - t0, 1))
        a = np.load(p)['actions']; k1 = np.asarray(np.load(K1_POLICY / f'{split}_actions.npy'), np.float32)
        check[split] = float(np.abs(a - k1).max())
    for key in ('k1', 'global'):
        off = noise_offset(key)
        for split in ('dev', 'eval'):
            p = out / 'pi05' / f'{split}_{key}.npz'
            if p.exists(): continue
            t0 = time.time(); a = pi05_predictions(model, fn, new, ids[split], off, batch)
            _save_new(p, actions=a, row=ids[split], noise_offset=np.int64(off))
            log(out, stage='pi05', rows='corrected', split=split, key=key, offset=off, sec=round(time.time() - t0, 1))
    meta = {'rows': str(UNSEEN_ROWS), 'legacy_rows': str(LEGACY_UNSEEN_ROWS), 'checkpoint': str(tr.CHECKPOINT), 'num_steps': 10,
            'noise': {'k1': 'fold_in(key(4101), unseen row id)', 'global': f'fold_in(key(4101), {noise_offset("global")} + unseen row id)'},
            'check_legacy_k1_vs_key_query_01_maxabs': check, 'target_actions_used': False}
    (out / 'pi05' / 'meta.json').write_text(json.dumps(meta, indent=1) + '\n'); log(out, stage='pi05', done=True, check=check)


def stage_features(out, batch=8):
    from openpi.shared.nnx_utils import module_jit
    import theta_regression as tr
    base = tr._module(tr.PI05 / 'run.py', 'orientation_fix_run')
    model, _ = base.load_policy(tr.CHECKPOINT); fn = module_jit(model.prefix_features)
    new = rows_of(UNSEEN_ROWS); ids = split_ids(new); D = out / 'features'; D.mkdir(parents=True, exist_ok=True)
    for split in ('dev', 'eval'):
        p = D / f'{split}.npy'
        if p.exists(): continue
        t0 = time.time(); arr = np.zeros((len(ids[split]), 3, 2048), np.float32)
        for st in range(0, len(ids[split]), batch):
            q = ids[split][st:st + batch]; h, g = fn(base.obs_batch(new, q)); h, g = np.asarray(h), np.asarray(g)
            for k in range(len(q)): arr[st + k] = tr.grouped_means(h[k], g[k])
        tmp = D / f'{split}.tmp.npy'; np.save(tmp, arr); tmp.replace(p)
        log(out, stage='features', split=split, rows=len(arr), sec=round(time.time() - t0, 1))
    # the legacy rows reproduce the stored theta-regression features (pipeline check, first 16 Eval rows)
    old = rows_of(LEGACY_UNSEEN_ROWS); q = ids['eval'][:16]; h, g = fn(base.obs_batch(old, q)); h, g = np.asarray(h), np.asarray(g)
    ref = np.load(tr.OUT / 'production' / 'features' / 'eval.npy', mmap_mode='r')[:16]
    chk = float(max(np.abs(tr.grouped_means(h[k], g[k]) - ref[k]).max() for k in range(len(q))))
    meta = {'rows': str(UNSEEN_ROWS), 'groups': ['base_camera', 'wrist_camera', 'language'], 'shape': [3, 2048],
            'recipe': 'theta_regression.extract_features (prefix_features, grouped_means)', 'target_actions_used': False,
            'check_legacy_eval16_vs_theta_regression_maxabs': chk}
    (D / 'meta.json').write_text(json.dumps(meta, indent=1) + '\n'); log(out, stage='features', done=True, check=chk)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=('pi05', 'features', 'all'))
    ap.add_argument('--out', type=Path, default=OUT); ap.add_argument('--batch', type=int, default=4); a = ap.parse_args()
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false'); a.out.mkdir(parents=True, exist_ok=True)
    if a.stage in ('pi05', 'all'): stage_pi05(a.out, a.batch)
    if a.stage in ('features', 'all'): stage_features(a.out, max(a.batch, 8))


if __name__ == '__main__':
    main()
