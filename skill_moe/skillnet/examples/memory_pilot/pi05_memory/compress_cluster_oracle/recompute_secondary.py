#!/usr/bin/env python3
"""Recompute compressed secondary drafts and their frozen-Expert results.

This is a correction-only pass. It never trains or writes into the original
compress-cluster-oracle run directory.
"""
from __future__ import annotations

import csv
import hashlib
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
SOURCE = ROOT / 'compress-cluster-oracle-01'
FAIR = ROOT / 'fair-expert-01'
OUT = Path(os.environ.get('SECONDARY_FIX_OUT', SOURCE / 'secondary-correction-01'))
T0S = (.2, .4, .6)
BOOT_REPS = 5000
THREADS = int(os.environ.get('OMP_NUM_THREADS', '8'))


def load_experiment():
    spec = importlib.util.spec_from_file_location('compress_cluster_exp_secondary_fix', HERE / 'experiment.py')
    if spec is None or spec.loader is None:
        raise ImportError('cannot load experiment.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def episode_bootstrap(values, episode_keys, *, seed):
    values = np.asarray(values, np.float64)
    episode_keys = np.asarray(episode_keys)
    ok = np.isfinite(values)
    values, episode_keys = values[ok], episode_keys[ok]
    unique, inverse = np.unique(episode_keys, return_inverse=True)
    sums = np.bincount(inverse, weights=values, minlength=len(unique))
    counts = np.bincount(inverse, minlength=len(unique)).astype(np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(unique), size=(BOOT_REPS, len(unique)))
    boot = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return {'mean': float(values.mean()), 'ci_low': float(np.quantile(boot, .025)),
            'ci_high': float(np.quantile(boot, .975)), 'episodes': int(len(unique)),
            'rows': int(len(values)), 'bootstrap_reps': BOOT_REPS}


def active_mse(pred, target):
    return np.mean((np.asarray(pred, np.float32) - np.asarray(target, np.float32)) ** 2,
                   axis=(1, 2)).astype(np.float32)


def run_grid(bridge, obs, noise, draft, *, batch=8):
    import jax
    import jax.numpy as jnp
    out = {}
    for t0 in T0S:
        pred = np.empty((len(noise), 10, 7), np.float32)
        for start in range(0, len(noise), batch):
            sl = slice(start, min(start + batch, len(noise)))
            y = bridge(jax.random.key(0), jax.tree.map(lambda x: x[sl], obs),
                       jnp.asarray(draft[sl]), jnp.full((len(noise[sl]),), t0),
                       num_steps=10, noise=jnp.asarray(noise[sl]))
            pred[sl] = np.asarray(y, np.float32)[:, :, :7]
        out[t0] = pred
    return out


def main():
    import torch
    torch.set_num_threads(THREADS)
    if OUT.exists():
        raise FileExistsError(f'refusing to overwrite {OUT}')
    OUT.mkdir(parents=True)
    exp = load_experiment()
    base, _, rows, actions, features, _, _, _, long, hard, _ = exp._load_input()
    official = np.load(FAIR / 'evaluation-final/offline_predictions.npz', allow_pickle=False)
    if not np.array_equal(official['row_ids'], long):
        raise AssertionError('fixed Long row order mismatch')
    if len(long) != 548 or int(np.sum(hard)) != 63:
        raise AssertionError('Long/hard row counts changed')
    target = np.asarray(actions[long, :, :7], np.float32)
    episode_key = np.asarray(rows['task'][long], np.int64) * 100000 + np.asarray(rows['episode'][long], np.int64)
    noise = np.asarray(official['noise'], np.float32)
    if noise.shape != (len(long), 10, 32):
        raise AssertionError(f'noise shape changed: {noise.shape}')

    expert_files = sorted(SOURCE.glob('experts/*/attempt-*/evaluation_rows.npz'))
    if len(expert_files) != 20:
        raise AssertionError(f'expected 20 trained Expert rows, got {len(expert_files)}')
    expert_data = {}
    source_hashes = {}
    for path in expert_files:
        key = path.parent.parent.name
        expert_data[key] = path
        source_hashes[str(path)] = sha256(path)

    # Load each source representation and map every query dataset row ID to its
    # fitted unit's medoid row. Decode exactly one candidate per query.
    corrected_drafts = {}
    corrected_ids = {}
    pre_metrics = []
    codec_cache = {}
    bkeys = sorted(k for k in expert_data if k.startswith('b'))
    draft_keys = sorted(p.stem[len('pre_'):] for p in SOURCE.glob('pre_b*.npz'))
    if len(draft_keys) != 20:
        raise AssertionError(f'expected all 20 compressed draft cells, got {len(draft_keys)}')
    for key in draft_keys:
        kind, level = key.rsplit('_', 1)
        prep_path = SOURCE / f'pre_{kind}_{level}.npz'
        prep = np.load(prep_path, allow_pickle=False)
        if not np.array_equal(prep['query_rows'], long) or not np.array_equal(prep['hard_mask'], hard):
            raise AssertionError(f'pre table row order changed for {key}')
        pool_ids = np.asarray(prep['pool_row_ids'], np.int64)
        units = np.asarray(prep['unit_ids'], np.int64)
        reps = np.asarray(prep['representative_rows'], np.int64)
        qlocal = exp.map_row_ids(pool_ids, long)
        own_units = units[qlocal]
        own_index = np.maximum(own_units, 0)
        secondary_rows = reps[own_index]
        if kind not in codec_cache:
            codec_cache[kind], _ = exp.load_codec(kind)
        bases = np.load(SOURCE / 'representations' / f'{kind}_bases.npy', mmap_mode='r', allow_pickle=False)
        if bases.shape[0] != len(actions):
            raise AssertionError(f'bases row count mismatch for {kind}: {bases.shape}')
        secondary = exp._decode_variable_candidates(
            codec_cache[kind], bases, long, rows, features, secondary_rows[:, None])[:, 0]
        secondary = np.asarray(secondary, np.float32)
        secondary[own_units < 0] = np.nan
        if not np.isfinite(secondary).all():
            raise AssertionError(f'non-finite corrected secondary draft in {key}')
        corrected_drafts[key] = secondary
        corrected_ids[key] = secondary_rows
        if level == 'none':
            max_abs = float(np.max(np.abs(secondary - np.asarray(prep['primary'], np.float32))))
            primary_mse = np.mean((secondary[:, :, :7] - np.asarray(prep['primary'], np.float32)[:, :, :7]) ** 2,
                                  axis=(1, 2))
            close_fraction = float(np.mean(primary_mse <= 1e-6))
            mean_diff = float(primary_mse.mean())
            # Oracle selection can occasionally find another encoded row with
            # lower reconstruction error than the query's own row. The expected
            # relation is therefore near-equality, not a forced ID match.
            if close_fraction < .99 or mean_diff > 2e-4:
                raise AssertionError(f'{key} b×none primary/secondary too far: '
                                     f'fraction={close_fraction}, mean_mse={mean_diff}')
            none_check = {'max_abs': max_abs, 'mean_active7_mse': mean_diff,
                          'fraction_rows_mse_le_1e-6': close_fraction}
        else:
            none_check = None
        draft_err = active_mse(secondary[:, :, :7], target)
        for subset, mask in (('all', np.ones(len(long), bool)), ('hard', hard)):
            ci = episode_bootstrap(draft_err[mask], episode_key[mask], seed=8101 + len(pre_metrics) * 17 + int(mask.sum()))
            pre_metrics.append({'expert': key, 'representation': kind, 'level': level,
                                'draft': 'secondary_corrected', 'stage': 'draft', 't0': '',
                                'subset': subset, 'mse_mean': ci['mean'],
                                'ci_low': ci['ci_low'], 'ci_high': ci['ci_high'],
                                'b_none_primary_consistency': json.dumps(none_check) if none_check else ''})

    # The public policy is restored once. Only the fixed action-expert leaves
    # are swapped in per condition; no optimizer/training code is called.
    import jax
    from openpi.shared.nnx_utils import module_jit
    sys.path.insert(0, str(HERE.parent / 'skill_expert'))
    from skill_expert.run import load_expert
    policy, _ = base.load_policy(Path(os.environ.get('SKILLNET_PI05_CHECKPOINT', STATE_DIR / 'checkpoints/openpi-assets/checkpoints/pi05_libero')).expanduser())
    obs = base.obs_batch(rows, long)
    generated = {}
    predictions = {}
    eval_rows = []
    original_rows_by_key = {}
    for key in bkeys:
        path = expert_data[key]
        old = np.load(path, allow_pickle=False)
        if not np.array_equal(old['row_ids'], long) or not np.array_equal(old['hard_mask'], hard):
            raise AssertionError(f'Expert evaluation row IDs changed for {key}')
        prep = np.load(SOURCE / f'pre_{key.split("_")[0]}_{key.rsplit("_", 1)[1]}.npz', allow_pickle=False)
        if not np.array_equal(old['primary_draft'], prep['primary']):
            raise AssertionError(f'primary draft differs from original for {key}')
        binfo = json.loads((path.parent / 'best_bridge.json').read_text())
        ckpt = Path(binfo['path'])
        load_expert(policy, ckpt)
        bridge = module_jit(policy.sample_bridge, static_argnames=('num_steps',))
        pred_grid = run_grid(bridge, obs, noise, corrected_drafts[key])
        errors = {t: active_mse(pred, target) for t, pred in pred_grid.items()}
        generated[key] = errors
        predictions[key] = pred_grid
        selected_t0 = float(binfo['t0'])
        for t0 in T0S:
            for subset, mask in (('all', np.ones(len(long), bool)), ('hard', hard)):
                ci = episode_bootstrap(errors[t0][mask], episode_key[mask],
                                       seed=9101 + int(round(t0 * 100)) * 31 + int(mask.sum()))
                eval_rows.append({'expert': key, 'representation': key.split('_')[0],
                                  'level': key.rsplit('_', 1)[1], 'draft': 'secondary_corrected',
                                  'stage': 'after_expert', 't0': t0, 'subset': subset,
                                  'mse_mean': ci['mean'], 'ci_low': ci['ci_low'],
                                  'ci_high': ci['ci_high'], 'b_none_primary_consistency': ''})
        print(json.dumps({'stage': 'expert_done', 'expert': key,
                          'selected_t0': selected_t0,
                          'secondary_all_selected_t0': float(errors[selected_t0].mean())}), flush=True)
        original_rows_by_key[key] = path

    # At each Expert's own selected t0, pair each compressed representation
    # with raw at the same clustering level. Raw outputs were unaffected and
    # are read from the original evaluation cache.
    contrasts = []
    contrast_summary = {}
    for key in bkeys:
        kind, level = key.rsplit('_', 1)
        raw_key = f'raw_{level}'
        if raw_key not in expert_data:
            continue
        binfo = json.loads((expert_data[key].parent / 'best_bridge.json').read_text())
        rbinfo = json.loads((expert_data[raw_key].parent / 'best_bridge.json').read_text())
        t_comp, t_raw = float(binfo['t0']), float(rbinfo['t0'])
        raw_npz = np.load(expert_data[raw_key], allow_pickle=False)
        if not np.array_equal(raw_npz['row_ids'], long):
            raise AssertionError(f'raw paired row IDs changed for {raw_key}')
        raw_err = np.asarray(raw_npz[f'mse_secondary_t{t_raw:.1f}'], np.float32)
        comp_err = generated[key][t_comp]
        entry = {'representation': kind, 'level': level, 'compressed_t0': t_comp,
                 'raw_t0': t_raw}
        for subset, mask in (('all', np.ones(len(long), bool)), ('hard', hard)):
            stats = exp._bootstrap_diff(comp_err[mask], raw_err[mask], episode_key[mask],
                                        reps=BOOT_REPS, seed=10001 + len(contrasts) * 13 + int(mask.sum()))
            entry[f'{subset}_mean_diff'] = stats['mean']
            entry[f'{subset}_ci_low'] = stats['ci95'][0]
            entry[f'{subset}_ci_high'] = stats['ci95'][1]
            entry[f'{subset}_episodes'] = stats['episodes']
        contrasts.append(entry)
        contrast_summary[f'{kind}_{level}_minus_raw_{level}'] = entry

    arrays = {'row_ids': long, 'hard_mask': hard, 'target': target,
              'task_ids': np.asarray(rows['task'][long]),
              'episode_ids': np.asarray(rows['episode'][long]),
              'episode_key': episode_key}
    for key in draft_keys:
        arrays[f'draft__{key}'] = corrected_drafts[key]
        arrays[f'secondary_row__{key}'] = corrected_ids[key]
    for key in bkeys:
        for t0 in T0S:
            arrays[f'pred__{key}__t0_{t0:.1f}'] = predictions[key][t0]
            arrays[f'mse__{key}__t0_{t0:.1f}'] = generated[key][t0]
    np.savez_compressed(OUT / 'corrected_secondary_rows.npz', **arrays)
    metrics = pre_metrics + eval_rows
    with (OUT / 'secondary_metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=('expert', 'representation', 'level', 'draft',
            'stage', 't0', 'subset', 'mse_mean', 'ci_low', 'ci_high', 'b_none_primary_consistency'))
        writer.writeheader(); writer.writerows(metrics)
    with (OUT / 'compressed_minus_raw.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(contrasts[0]))
        writer.writeheader(); writer.writerows(contrasts)

    # Verify no original trained evaluation artifact was changed.
    for path in expert_files:
        if sha256(path) != source_hashes[str(path)]:
            raise AssertionError(f'original evaluation artifact changed: {path}')
    summary = {'status': 'complete', 'rows': len(long), 'hard_rows': int(hard.sum()),
        'experts_corrected': bkeys, 'expert_count': len(bkeys),
        'compressed_draft_cells_corrected': draft_keys, 'draft_cell_count': len(draft_keys),
        't0_values': list(T0S),
        'bootstrap_reps': BOOT_REPS, 'noise_source': 'same cached fair-expert noise (seed 4101; fold_in row_id)',
        'primary_expert_training_rerun': False, 'primary_draft_cache_unchanged': True,
        'original_expert_evaluation_hashes_unchanged': True,
        'bN_none_consistency_vs_primary': {r['expert']: r['b_none_primary_consistency'] for r in pre_metrics
                                        if r['level'] == 'none' and r['subset'] == 'all'},
        'compressed_minus_raw_at_each_selected_t0': contrast_summary,
        'source_evaluation_sha256': source_hashes}
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'status': 'complete', 'out': str(OUT), 'experts': len(bkeys),
                      'metric_rows': len(metrics), 'contrast_rows': len(contrasts)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
