"""Evaluate every saved RM-EXP3 checkpoint at the requested fixed t0 grid."""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import expert
import expert2_train as base

ROOT = Path('/work/.skillnet-state/outputs/robocasa-memory/expert-03')
PREP = ROOT / 'prep'
OUT = ROOT / 'train'
T0_GRID = (.2, .3, .4, .5)


def _json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                              default=expert.ex._json_default) + '\n')
    tmp.replace(path)


def validate(*, threads=8, memory_fraction=.35):
    import jax
    import jax.numpy as jnp
    import flax.nnx as nnx
    import torch
    from flax import serialization
    from openpi.shared.nnx_utils import PathRegex
    from openpi.shared import normalize

    torch.set_num_threads(int(threads))
    torch.cuda.set_per_process_memory_fraction(float(memory_fraction))
    base.PREP = PREP
    base.OUT = OUT
    expert_dir = OUT / 'expert'
    final_path = expert_dir / 'validation_grid.json'
    if final_path.exists():
        previous = json.loads(final_path.read_text())
        if previous.get('status') == 'complete':
            return previous
        raise FileExistsError(f'incomplete validation result exists: {final_path}')

    data = np.load(PREP / 'drafts/memory_dynamic.npz', allow_pickle=False)
    _, val_ids, selected = base.prepare_full_cache(OUT, threads=threads)
    if len(val_ids) != 300:
        raise AssertionError(f'fixed validation set must contain 300 rows, got {len(val_ids)}')
    tasks = np.asarray(data['task'])
    cache = {}
    tree_def = None
    for task in sorted(np.unique(tasks[selected]).tolist()):
        payload = pickle.load((OUT / 'obs_cache' / f'task_{int(task)}.pkl').open('rb'))
        if tree_def is None:
            tree_def = payload['tree_def']
        elif payload['tree_def'] != tree_def:
            raise AssertionError('observation cache PyTrees differ')
        for pos, row in enumerate(payload['row_ids']):
            cache[int(row)] = [np.asarray(x[pos]) for x in payload['leaves']]

    stats = normalize.load(base.CHECKPOINT / 'assets')['actions']
    raw_drafts = np.asarray(data['draft'], np.float32)
    raw_targets = np.asarray(data['target'], np.float32)
    draft = np.stack([expert._draft_to_model(x, stats) for x in raw_drafts]).astype(np.float32)
    target = np.stack([expert._draft_to_model(x, stats) for x in raw_targets]).astype(np.float32)
    lengths = np.asarray(data['first_unit_length'], np.int32)

    model, _ = expert._load_expert_model()
    filt = nnx.All(nnx.Param, PathRegex(r'.*(llm.*_1|action_in_proj|time_mlp|action_out_proj).*'))

    @nnx.jit
    def evaluate(m, obs, dn, eps, starts):
        return m.sample_bridge(jax.random.key(0), obs, dn, starts,
                               num_steps=10, noise=eps)

    def make_batch(ids):
        leaves = [jnp.asarray(np.stack([cache[int(i)][j] for i in ids]))
                  for j in range(len(cache[int(ids[0])]))]
        return jax.tree.unflatten(tree_def, leaves)

    checkpoints = sorted(expert_dir.glob('expert_[0-9]*.msgpack'))
    if not checkpoints:
        raise FileNotFoundError(f'no checkpoints found in {expert_dir}')
    progress_path = expert_dir / 'validation_progress.json'
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {
        'status': 'running', 'rows': len(val_ids), 't0_grid': list(T0_GRID),
        'checkpoints': {}, 'selection_metric': 'mean normalized MSE6 over first 5 steps'}
    all_started = time.monotonic()
    for cp in checkpoints:
        cp_result = progress['checkpoints'].setdefault(cp.name, {})
        # Replace all trainable leaves with this checkpoint, preserving the
        # frozen VLM and the corrected dynamic NNX validation path.
        selected_state = nnx.state(model, filt)
        loaded = serialization.from_bytes(selected_state.to_pure_dict(), cp.read_bytes())
        selected_state.replace_by_pure_dict(loaded)
        nnx.update(model, selected_state)
        for t0 in T0_GRID:
            tkey = str(t0)
            if tkey in cp_result:
                continue
            predictions = []
            for start in range(0, len(val_ids), 4):
                bid = np.asarray(val_ids[start:start + 4], np.int64)
                obs = make_batch(bid)
                dn = jnp.asarray(draft[bid])
                noises = np.stack([
                    np.asarray(jax.random.normal(
                        jax.random.fold_in(jax.random.key(4101), int(row)),
                        (1, 50, 32)))[0]
                    for row in bid
                ]).astype(np.float32)
                starts = jnp.full((len(bid),), float(t0), jnp.float32)
                pred = np.asarray(evaluate(model, obs, dn, jnp.asarray(noises), starts))
                predictions.extend(pred)
            pred = np.asarray(predictions, np.float32)
            truth = target[val_ids]
            first5 = float(np.mean((pred[:, :5, :6] - truth[:, :5, :6]) ** 2))
            unit_err = [float(np.mean((p[:int(np.clip(lengths[row], 1, 50)), :6] -
                                       target[row, :int(np.clip(lengths[row], 1, 50)), :6]) ** 2))
                         for p, row in zip(pred, val_ids, strict=True)]
            cp_result[tkey] = {'rows': len(val_ids), 'first5_mse6': first5,
                               'unit_mse6': float(np.mean(unit_err))}
            progress.update({'latest_checkpoint': cp.name, 'latest_t0': float(t0),
                             'elapsed_seconds': time.monotonic() - all_started})
            _json(progress_path, progress)
            print(json.dumps({'stage': 'validate', 'checkpoint': cp.name,
                              't0': float(t0), **cp_result[tkey]}), flush=True)

    rows = []
    for cp in checkpoints:
        for t0 in T0_GRID:
            metrics = progress['checkpoints'].get(cp.name, {}).get(str(t0))
            if metrics is not None:
                rows.append({'checkpoint': cp.name, 't0': float(t0), **metrics})
    best = {}
    for t0 in T0_GRID:
        candidates = [r for r in rows if r['t0'] == t0]
        if candidates:
            best[str(t0)] = min(candidates, key=lambda r: r['first5_mse6'])
    result = {'status': 'complete' if len(rows) == len(checkpoints) * len(T0_GRID) else 'partial',
        'rows': len(val_ids), 'validation_row_ids': val_ids.tolist(),
        'noise_seed': 4101, 'num_steps': 10, 't0_grid': list(T0_GRID),
        'metric': 'mean of normalized active6 squared error over first 5 action steps',
        'unit_metric': 'mean normalized active6 error over each selected first-unit length',
        'weights_loaded_dynamically': True, 'results': rows, 'best_by_t0': best,
        'elapsed_seconds': time.monotonic() - all_started}
    if result['status'] == 'complete':
        _json(final_path, result)
        progress['status'] = 'complete'
        _json(progress_path, progress)
    else:
        progress['status'] = 'partial'
        _json(progress_path, progress)
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--fraction', type=float, default=.35)
    args = parser.parse_args()
    print(json.dumps(validate(threads=args.threads, memory_fraction=args.fraction)), flush=True)
