#!/usr/bin/env python3
"""Matched-distribution E-M2 vs cross-task shuffled retrieval drafts."""
from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location('fair_expert_runtime', Path(__file__).with_name('run.py'))
fair = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fair)


def main():
    import jax
    import jax.numpy as jnp
    from openpi.shared.nnx_utils import module_jit

    root = fair.OUT_DEFAULT
    out = root / 'evaluation-rankmix'
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    base, skill = fair._base_run(), fair._skill_run()
    rows = base.load_rows(fair.RUN)
    gap = np.load(fair.GAP, allow_pickle=False)
    all_eval = np.asarray(gap['row_ids'], np.int64)
    qids = all_eval[np.asarray(rows['task'][all_eval]) < 10]
    if len(qids) != 548:
        raise AssertionError('fixed Long Eval rows changed')
    saved = np.load(fair.M2_DIR / 'candidate_rows.npz', allow_pickle=False)
    if not np.array_equal(saved['query_rows'], qids):
        raise AssertionError('M2 candidate row order changed')
    candidates = np.asarray(saved['M2_salient_maxsim_rows'], np.int64)
    pool = np.asarray(np.load(fair.MEMORY / 'units.npz', allow_pickle=False)['pool_row_ids'])
    fair.validate_candidates(candidates, qids, rows, pool)
    sampled, ranks = fair.sample_m2_training_draft(candidates, np.random.default_rng(fair.SEED + 811))
    perm = np.random.default_rng(fair.SEED + 812).permutation(len(qids))
    shuffled = sampled[perm]
    if np.any(rows['task'][sampled] == rows['task'][qids]) or np.any(rows['task'][shuffled] == rows['task'][qids]):
        raise AssertionError('source candidates must be different task from query')
    actions = np.load(fair.RUN / 'rows' / 'actions.npy', mmap_mode='r')
    drafts = {'M2_rank_mix': fair.candidate_drafts(actions, sampled),
              'shuffle_rank_mix': fair.candidate_drafts(actions, shuffled)}
    targets = np.asarray(actions[qids, :, :7], np.float32)
    noise = np.stack([np.asarray(jax.random.normal(jax.random.fold_in(jax.random.key(4101), int(r)), (10, 32)))
                      for r in qids])
    selected = json.loads((root / 'E-M2' / 'best_bridge.json').read_text())
    policy, _ = base.load_policy(fair.CHECKPOINT)
    skill.load_expert(policy, Path(selected['path']))
    fn = module_jit(policy.sample_bridge, static_argnames=('num_steps',))
    obs = base.obs_batch(rows, qids)
    preds = {}
    started = time.monotonic()
    for name, draft in drafts.items():
        for t0 in fair.T0S:
            pred = np.empty_like(targets)
            for st in range(0, len(qids), 8):
                sl = slice(st, min(st + 8, len(qids)))
                ob = jax.tree.map(lambda x: x[sl], obs)
                pred[sl] = np.asarray(fn(jax.random.key(0), ob, jnp.asarray(draft[sl]),
                    jnp.full((len(qids[sl]),), t0), num_steps=10,
                    noise=jnp.asarray(noise[sl])))[:, :, :7]
            preds[f'{name}/t0={t0:.1f}'] = pred
    mse = {k: np.mean((v - targets) ** 2, axis=(1, 2)).astype(np.float32) for k, v in preds.items()}
    episodes = np.asarray(rows['task'][qids], np.int64) * 100000 + np.asarray(rows['episode'][qids], np.int64)
    pair = {}
    for t0 in fair.T0S:
        a, b = mse[f'M2_rank_mix/t0={t0:.1f}'], mse[f'shuffle_rank_mix/t0={t0:.1f}']
        pair[str(t0)] = fair._bootstrap_diff(a, b, episodes, reps=5000, seed=4101)
    _save = out / 'rankmix_results.npz'
    np.savez_compressed(_save, row_ids=qids, task_ids=np.asarray(rows['task'][qids]),
        episode_ids=np.asarray(rows['episode'][qids]), ranks=ranks, selected_rows=sampled,
        shuffle_rows=shuffled, shuffle_permutation=perm, noise=noise, targets=targets,
        m2_draft=drafts['M2_rank_mix'], shuffle_draft=drafts['shuffle_rank_mix'],
        **{f'pred_{k.replace("/", "_").replace("=", "_")}': v for k, v in preds.items()},
        **{f'mse_{k.replace("/", "_").replace("=", "_")}': v for k, v in mse.items()})
    summary = {'rows': len(qids), 'rank_counts': {str(k): int(np.sum(ranks == k)) for k in range(5)},
        'selected_checkpoint': selected, 'M2_vs_shuffle_bootstrap': pair,
        'M2_rank_mix_mean_mse': {str(t): float(mse[f'M2_rank_mix/t0={t:.1f}'].mean()) for t in fair.T0S},
        'shuffle_rank_mix_mean_mse': {str(t): float(mse[f'shuffle_rank_mix/t0={t:.1f}'].mean()) for t in fair.T0S},
        'seconds': time.monotonic() - started}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
