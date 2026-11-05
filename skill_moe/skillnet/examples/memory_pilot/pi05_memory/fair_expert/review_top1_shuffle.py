#!/usr/bin/env python3
"""Re-evaluate top-1-matched shuffle without overwriting fair-expert-01."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '.25')

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('fair_expert_review_runtime', HERE / 'run.py')
fair = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(fair)


def evaluate(output: Path, batch: int = 8, source: Path = fair.OUT_DEFAULT):
    source = Path(source)
    if output.exists():
        raise FileExistsError(f'refusing output overwrite: {output}')
    output.mkdir(parents=True)
    saved = np.load(source / 'offline_predictions.npz', allow_pickle=False)
    rows = fair._base_run().load_rows(fair.RUN)
    qids = np.asarray(saved['row_ids'], np.int64)
    candidate_rows = np.asarray(saved['m2_candidate_rows'], np.int64)
    if len(qids) != 548 or candidate_rows.shape != (548, 64):
        raise AssertionError('fixed Long query/candidate arrays changed')
    if not np.array_equal(candidate_rows,
            np.load(fair.M2_DIR / 'candidate_rows.npz', allow_pickle=False)['M2_salient_maxsim_rows']):
        raise AssertionError('saved M2 candidates differ from production row IDs')
    top1_rows = candidate_rows[:, 0]
    shuffled_rows, permutation = fair.shuffle_other_task(
        candidate_rows[:, :1], qids, rows, seed=fair.SEED + 812)
    shuffled_rows = np.asarray(shuffled_rows, np.int64)
    if not np.array_equal(np.sort(top1_rows), np.sort(shuffled_rows)):
        raise AssertionError('Top-1 shuffle changed the candidate multiset')
    if np.any(np.asarray(rows['task'])[shuffled_rows] == np.asarray(rows['task'])[qids]):
        raise AssertionError('shuffled candidate has the query task')

    actions = np.load(fair.RUN / 'rows' / 'actions.npy', mmap_mode='r')
    target = np.asarray(saved['target'], np.float32)
    if not np.array_equal(target, np.asarray(actions[qids, :, :7], np.float32)):
        raise AssertionError('saved evaluation targets do not match dataset row IDs')
    noise = np.asarray(saved['noise'], np.float32)
    draft = fair.candidate_drafts(actions, shuffled_rows)
    original_m2 = np.asarray(saved['m2_draft'], np.float32)
    np.testing.assert_array_equal(original_m2, fair.candidate_drafts(actions, top1_rows))

    import jax
    import jax.numpy as jnp
    from openpi.shared.nnx_utils import module_jit

    base, skill = fair._base_run(), fair._skill_run()
    selected = json.loads((source / 'E-M2' / 'best_bridge.json').read_text())
    old = json.loads((fair.RUN / 'best_bridge.json').read_text())
    specs = {'E-M2': Path(selected['path']), 'old': Path(old['path'])}
    predictions = {}
    for expert_name, checkpoint in specs.items():
        policy, _ = base.load_policy(fair.CHECKPOINT)
        skill.load_expert(policy, checkpoint)
        fn = module_jit(policy.sample_bridge, static_argnames=('num_steps',))
        for t0 in fair.T0S:
            pred = np.empty_like(target)
            for start in range(0, len(qids), batch):
                end = min(start + batch, len(qids))
                obs = base.obs_batch(rows, qids[start:end])
                pred[start:end] = np.asarray(fn(
                    jax.random.key(0), obs, jnp.asarray(draft[start:end]),
                    jnp.full((end - start,), t0), num_steps=10,
                    noise=jnp.asarray(noise[start:end])))[:, :, :7]
            key = f'{expert_name}_shuffle_top1_t0_{t0:.1f}'
            predictions[key] = pred
        del fn, policy
        jax.clear_caches()

    hard = np.asarray(saved['hard_mask'], bool)
    mse = {name: np.mean((pred - target) ** 2, axis=(1, 2)).astype(np.float32)
           for name, pred in predictions.items()}
    episode_ids = (np.asarray(rows['task'][qids], np.int64) * 100000 +
                   np.asarray(rows['episode'][qids], np.int64))
    selected_t0 = float(selected['t0'])
    f2 = {}
    for subset, mask in [('all_long', np.ones(len(qids), bool)), ('hard', hard)]:
        m2 = np.asarray(saved[f'mse_E-M2_M2_top1_t0_{selected_t0:.1f}'], np.float32)
        shuf = mse[f'E-M2_shuffle_top1_t0_{selected_t0:.1f}']
        f2[subset] = fair._bootstrap_diff(m2[mask], shuf[mask], episode_ids[mask], reps=5000, seed=4101)
    metrics = {}
    for t0 in fair.T0S:
        for name in ('E-M2', 'old'):
            k = f'{name}_shuffle_top1_t0_{t0:.1f}'
            err = mse[k]
            metrics[k] = {'all_mean': float(err.mean()), 'hard_mean': float(err[hard].mean())}
    decision = {'F1': json.loads((source / 'decision.json').read_text())['F1'],
                'F2': bool(any(v['ci95'][1] < 0 for v in f2.values())),
                'selected_t0': selected_t0,
                'F2_definition': 'E-M2 with its Top-1 candidates minus E-M2 with the exact same Top-1 candidate multiset permuted across Long queries'}
    np.savez_compressed(output / 'top1_shuffle_predictions.npz', query_rows=qids,
        hard_mask=hard, source_candidate_rows=top1_rows, shuffled_candidate_rows=shuffled_rows,
        permutation=permutation, draft=draft, target=target, noise=noise,
        **{f'pred_{k}': v for k, v in predictions.items()},
        **{f'mse_{k}': v for k, v in mse.items()})
    result = {'source_evaluation': str(source), 'selected': selected,
        'candidate_multiset_preserved': True, 'all_shuffled_candidates_cross_task': True,
        'shuffle_rank_counts': {'top1': int(len(qids)), 'other_ranks': 0},
        'metrics': metrics, 'F2_selected_t0': f2, 'decision': decision}
    (output / 'top1_shuffle_summary.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    (output / 'decision.json').write_text(json.dumps(decision, indent=2, ensure_ascii=False) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path,
        default=fair.OUT_DEFAULT / 'corrected-top1-shuffle-01')
    parser.add_argument('--source', type=Path, default=fair.OUT_DEFAULT,
        help='evaluate-stage output directory holding offline_predictions.npz, decision.json and E-M2/best_bridge.json')
    parser.add_argument('--batch', type=int, default=8)
    args = parser.parse_args()
    result = evaluate(args.output, args.batch, args.source)
    print(json.dumps(result['decision'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
