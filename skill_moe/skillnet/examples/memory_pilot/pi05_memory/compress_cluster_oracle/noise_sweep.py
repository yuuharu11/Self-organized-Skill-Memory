#!/usr/bin/env python3
"""Fixed-checkpoint t0 sweep for the compress/cluster oracle diagnostics.

This script only evaluates already-trained Experts. Existing t0=.2/.4/.6
per-row predictions are reused after an exact-path recomputation check.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[6]
STATE_DIR = Path(os.environ.get('SKILLNET_STATE_DIR', REPO_ROOT / '.skillnet-state')).expanduser()
ROOT = Path(os.environ.get('SKILLNET_PI05_ROOT', STATE_DIR / 'outputs/memory-pilot/pi05-memory40-20261104')).expanduser()
RUN = ROOT / 'run-01'
EXPERIMENT = HERE / 'experiment.py'
EXISTING = ROOT / 'compress-cluster-oracle-01'
FAIR = ROOT / 'fair-expert-01'
OUT = Path(os.environ.get('NOISE_SWEEP_OUT', EXISTING / 'noise-sweep-01'))
DOC_FIG = REPO_ROOT / 'docs/research/07_pi05_memory40/compress_cluster_oracle/figures'
T0S = (.05, .1, .2, .3, .4, .5, .6, .8, 1.0)
OLD_T0S = (.2, .4, .6)
REPS = 5000
THREADS = int(os.environ.get('OMP_NUM_THREADS', '8'))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _bootstrap_mean_ci(values, episodes, *, seed):
    v = np.asarray(values, np.float64)
    ep = np.asarray(episodes)
    good = np.isfinite(v)
    v, ep = v[good], ep[good]
    uniq, inv = np.unique(ep, return_inverse=True)
    sums = np.bincount(inv, weights=v, minlength=len(uniq))
    counts = np.bincount(inv, minlength=len(uniq)).astype(np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(uniq), size=(REPS, len(uniq)))
    den = counts[draws].sum(axis=1)
    boot = sums[draws].sum(axis=1) / den
    return float(v.mean()), float(np.quantile(boot, .025)), float(np.quantile(boot, .975))


def _active_mse(pred, target):
    return np.mean((np.asarray(pred, np.float32) - np.asarray(target, np.float32)) ** 2,
                   axis=(1, 2)).astype(np.float32)


def _sensitivity(pred, zero_pred):
    return np.sqrt(np.mean((np.asarray(pred, np.float32) - np.asarray(zero_pred, np.float32)) ** 2,
                           axis=(1, 2))).astype(np.float32)


def _run_grid(policy, bridge, obs, noise, drafts, *, batch=8):
    """Return [draft, t0, row, horizon, active7] predictions."""
    import jax
    import jax.numpy as jnp
    names = tuple(drafts)
    out = np.empty((len(names), len(T0S), len(noise), 10, 7), np.float32)
    for di, name in enumerate(names):
        draft = np.asarray(drafts[name], np.float32)
        if draft.shape != (len(noise), 10, 32):
            raise AssertionError(f'{name} draft shape {draft.shape}')
        for ti, t0 in enumerate(T0S):
            for start in range(0, len(noise), batch):
                sl = slice(start, min(start + batch, len(noise)))
                y = bridge(jax.random.key(0), jax.tree.map(lambda x: x[sl], obs),
                           jnp.asarray(draft[sl]), jnp.full((len(noise[sl]),), t0),
                           num_steps=10, noise=jnp.asarray(noise[sl]))
                out[di, ti, sl] = np.asarray(y, np.float32)[:, :, :7]
    return out


def main():
    import torch
    torch.set_num_threads(THREADS)
    if OUT.exists():
        raise FileExistsError(f'refusing to overwrite {OUT}')
    OUT.mkdir(parents=True)
    (OUT / 'progress.jsonl').touch()

    exp = _load_module('compress_cluster_exp_for_noise', EXPERIMENT)
    base, fair, rows, actions, features, train, dev, memory, long, hard, pool = exp._load_input()
    official = np.load(FAIR / 'evaluation-final/offline_predictions.npz', allow_pickle=False)
    if not np.array_equal(official['row_ids'], long):
        raise AssertionError('official reference rows are not the fixed Long order')
    target = np.asarray(actions[long, :, :7], np.float32)
    task = np.asarray(rows['task'][long], np.int64)
    episode = np.asarray(rows['episode'][long], np.int64)
    episode_key = task * 100000 + episode
    subset_masks = {'all': np.ones(len(long), bool), 'hard': np.asarray(hard, bool)}
    if int(subset_masks['hard'].sum()) != 63:
        raise AssertionError('fixed hard subset must have 63 rows')
    noise = np.asarray(official['noise'], np.float32)
    if noise.shape != (len(long), 10, 32):
        raise AssertionError(f'fixed noise bank shape {noise.shape}')

    import jax
    from openpi.shared.nnx_utils import module_jit
    sys.path.insert(0, str(HERE.parent / 'skill_expert'))
    from skill_expert.run import load_expert

    # Restore public weights once; each checkpoint only replaces the declared
    # action-expert leaves, matching the original evaluation loader.
    policy, _ = base.load_policy(Path(os.environ.get('SKILLNET_PI05_CHECKPOINT', STATE_DIR / 'checkpoints/openpi-assets/checkpoints/pi05_libero')).expanduser())
    obs = base.obs_batch(rows, long)
    ref_normal = np.asarray(official['pred_official_normal'], np.float32)
    ref_normal_mse = _active_mse(ref_normal, target)

    # The E-M2 baseline uses the already-fixed Top-1 candidate list. Row IDs
    # are dataset IDs, never offsets into the memory pool.
    cand_cache = np.load(FAIR / 'candidate_lists.npz', allow_pickle=False)
    if not np.array_equal(cand_cache['long_query_rows'], long):
        raise AssertionError('M2 candidate list row order mismatch')
    m2_ids = np.asarray(cand_cache['long_candidate_rows'][:, 0], np.int64)
    m2_draft = np.asarray(actions[m2_ids], np.float32)
    if m2_draft.shape != (len(long), 10, 32):
        raise AssertionError('M2 draft shape mismatch')

    curve_rows = []
    row_records = []
    verify = {}

    def emit(expert, representation, level, draft_name, pred_by_t, zero_by_t):
        mse = np.stack([_active_mse(pred_by_t[i], target) for i in range(len(T0S))])
        sens = np.stack([_sensitivity(pred_by_t[i], zero_by_t[i]) for i in range(len(T0S))])
        row_records.append({'expert': expert, 'representation': representation,
                            'level': level, 'draft': draft_name,
                            'mse_by_t0': mse, 'sensitivity_by_t0': sens})
        for ti, t0 in enumerate(T0S):
            for subset, mask in subset_masks.items():
                mean, lo, hi = _bootstrap_mean_ci(mse[ti, mask], episode_key[mask],
                                                   seed=4101 + ti * 31 + int(mask.sum()))
                curve_rows.append({'expert': expert, 'representation': representation,
                                   'level': level, 'draft': draft_name, 't0': t0,
                                   'subset': subset, 'mse_mean': mean, 'ci_low': lo,
                                   'ci_high': hi,
                                   'sensitivity': float(np.mean(sens[ti, mask]))})

    # Load and evaluate each existing Expert once; no optimizer or training path
    # is called. Existing t0 results are used in the output after verification.
    eval_files = sorted(EXISTING.glob('experts/*/attempt-*/evaluation_rows.npz'))
    if len(eval_files) != 20:
        raise AssertionError(f'expected the 20 unique completed Experts, found {len(eval_files)}')
    for index, cache_path in enumerate(eval_files, start=1):
        key = cache_path.parent.parent.name
        kind, level = key.rsplit('_', 1)
        cached = np.load(cache_path, allow_pickle=False)
        if not np.array_equal(cached['row_ids'], long) or not np.array_equal(cached['hard_mask'], hard):
            raise AssertionError(f'cached row identity mismatch: {cache_path}')
        drafts = {'primary': np.asarray(cached['primary_draft'], np.float32),
                  'shuffle': np.asarray(cached['shuffle_draft'], np.float32),
                  'zero': np.zeros((len(long), 10, 32), np.float32)}
        best = json.loads((cache_path.parent / 'best_bridge.json').read_text())
        ckpt = Path(best['path'])
        if not ckpt.exists():
            raise FileNotFoundError(ckpt)
        load_expert(policy, ckpt)
        bridge = module_jit(policy.sample_bridge, static_argnames=('num_steps',))
        preds = _run_grid(policy, bridge, obs, noise, drafts)
        # Compare the new path at every preexisting t0 to the original per-row
        # outputs, then source those three points from the old cache.
        maxdiff = 0.0
        diffs = {}
        reused = preds.copy()
        for di, dname in enumerate(drafts):
            for ti, t0 in enumerate(T0S):
                if t0 not in OLD_T0S:
                    continue
                old = np.asarray(cached[f'pred_{dname}_t{t0:.1f}'], np.float32)
                dmax = float(np.max(np.abs(preds[di, ti] - old)))
                diffs[f'{dname}/t0={t0:.1f}'] = dmax
                maxdiff = max(maxdiff, dmax)
                reused[di, ti] = old
        if maxdiff > 3e-5:
            raise AssertionError(f'{key} old t0 prediction mismatch: maxabs={maxdiff}; details={diffs}')
        verify[key] = {'recomputed_old_t0_max_abs': maxdiff,
                       'reused_t0': list(OLD_T0S), 'checkpoint': str(ckpt)}
        zero_pred = reused[2]
        for di, dname in enumerate(('primary', 'shuffle', 'zero')):
            # Every preexisting t0 prediction is reused after the recomputation
            # check, preserving the original per-row result exactly.
            emit(key, kind, level, dname, reused[di], zero_pred)
        with (OUT / 'progress.jsonl').open('a') as f:
            f.write(json.dumps({'stage': 'expert', 'expert': key, 'index': index,
                                'of': len(eval_files), 'seconds_utc': time.strftime('%FT%TZ', time.gmtime()),
                                'max_abs_old_t0': maxdiff}) + '\n')
        print(json.dumps({'stage': 'expert_done', 'expert': key,
                          'index': index, 'max_abs_old_t0': maxdiff}), flush=True)

    # Baselines use the fair-expert checkpoints and identical noise bank.
    for expert, ckpt_root, drafts, rep, level in (
        ('E-zero', FAIR / 'E-zero', {'zero': np.zeros((len(long), 10, 32), np.float32)}, 'baseline', 'zero'),
        ('E-M2', FAIR / 'E-M2', {'M2': m2_draft}, 'baseline', 'M2')):
        binfo = json.loads((ckpt_root / 'best_bridge.json').read_text())
        ckpt = Path(binfo['path'])
        load_expert(policy, ckpt)
        bridge = module_jit(policy.sample_bridge, static_argnames=('num_steps',))
        # For M2, also evaluate a zero draft solely to measure draft-induced
        # output sensitivity; only M2 is published as a curve.
        if expert == 'E-M2':
            all_drafts = {'M2': m2_draft, 'zero': np.zeros((len(long), 10, 32), np.float32)}
            p = _run_grid(policy, bridge, obs, noise, all_drafts)
            m2p, zerop = p[0], p[1]
        else:
            zerop = _run_grid(policy, bridge, obs, noise, drafts)[0]
            m2p = None
        # Verify all reusable baseline points against the saved fair-expert outputs.
        maxdiff = 0.0
        reused_main = m2p.copy() if expert == 'E-M2' else zerop.copy()
        for t0 in OLD_T0S:
            if expert == 'E-zero':
                partial = FAIR / f'evaluation-final/eval_partials/E-zero_zero_t0{t0:.1f}.npz'
            else:
                partial = FAIR / f'evaluation-final/eval_partials/E-M2_M2_top1_t0{t0:.1f}.npz'
            old = np.load(partial, allow_pickle=False)['prediction']
            ti = T0S.index(t0)
            cur = zerop[ti] if expert == 'E-zero' else m2p[ti]
            maxdiff = max(maxdiff, float(np.max(np.abs(cur - old))))
            reused_main[ti] = old
        if maxdiff > 3e-5:
            raise AssertionError(f'{expert} baseline old-t0 mismatch: {maxdiff}')
        verify[expert] = {'recomputed_old_t0_max_abs': maxdiff,
                          'reused_t0': list(OLD_T0S), 'checkpoint': str(ckpt)}
        if expert == 'E-zero':
            emit(expert, rep, level, 'zero', reused_main, reused_main)
        else:
            emit(expert, rep, level, 'M2', reused_main, zerop)

    # Official normal policy is a no-t0 horizontal reference.
    for subset, mask in subset_masks.items():
        mean, lo, hi = _bootstrap_mean_ci(ref_normal_mse[mask], episode_key[mask], seed=4401 + int(mask.sum()))
        curve_rows.append({'expert': 'official_normal', 'representation': 'baseline', 'level': 'none',
                           'draft': 'none', 't0': '', 'subset': subset, 'mse_mean': mean,
                           'ci_low': lo, 'ci_high': hi, 'sensitivity': 0.0})
    row_records.append({'expert': 'official_normal', 'representation': 'baseline',
                        'level': 'none', 'draft': 'none',
                        'mse_by_t0': ref_normal_mse[None, :],
                        'sensitivity_by_t0': np.zeros((1, len(long)), np.float32)})
    # Preserve all raw outputs in compact row-wise arrays.
    arrays = {}
    for i, rec in enumerate(row_records):
        stem = f"{rec['expert']}__{rec['draft']}"
        arrays[f'mse__{stem}'] = rec['mse_by_t0']
        arrays[f'sensitivity__{stem}'] = rec['sensitivity_by_t0']
    np.savez_compressed(OUT / 'noise_curves_rows.npz', row_ids=long, task_ids=task,
                        episode_ids=episode, episode_key=episode_key, hard_mask=hard,
                        t0=np.asarray(T0S, np.float32), **arrays)
    with (OUT / 'noise_curves.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=('expert', 'representation', 'level', 'draft', 't0',
                                               'subset', 'mse_mean', 'ci_low', 'ci_high', 'sensitivity'))
        writer.writeheader(); writer.writerows(curve_rows)
    (OUT / 'verification.json').write_text(json.dumps({'status': 'complete', 'checks': verify,
        'old_t0_reused': list(OLD_T0S), 'new_t0_evaluated': [x for x in T0S if x not in OLD_T0S],
        'noise': 'same cached fair-expert bank: seed=4101, fold_in(dataset row_id)',
        'episode_bootstrap_reps': REPS}, indent=2) + '\n')
    print(json.dumps({'status': 'complete', 'expert_count': len(eval_files),
                      'rows_csv': len(curve_rows), 'out': str(OUT)}, indent=2))


if __name__ == '__main__':
    main()
