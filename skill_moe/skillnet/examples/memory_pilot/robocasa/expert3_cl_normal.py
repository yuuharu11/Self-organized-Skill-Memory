"""Compute π0.5 normal-generation metrics on RM-EXP3's fixed 300 windows."""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import expert
import expert2_train as train

ROOT = Path('/work/.skillnet-state/outputs/robocasa-memory/expert-03')
PREP = ROOT / 'prep'
OUT = ROOT / 'train'
DEST = ROOT / 'closedloop/normal_baseline300.json'


def run(*, threads=8, fraction=.25, batch_size=4):
    import jax
    import jax.numpy as jnp
    import flax.nnx as nnx
    import torch

    if DEST.exists():
        raise FileExistsError(f'refusing to overwrite {DEST}')
    torch.set_num_threads(int(threads))
    torch.cuda.set_per_process_memory_fraction(float(fraction))
    train.PREP, train.OUT = PREP, OUT
    data = np.load(PREP / 'drafts/memory_dynamic.npz', allow_pickle=False)
    train_ids, val_ids, _ = train.prepare_full_cache(OUT, threads=threads)
    if len(val_ids) != 300:
        raise AssertionError(f'expected the fixed 300 windows, got {len(val_ids)}')

    tasks = np.asarray(data['task'])
    cache = {}
    tree_def = None
    # Keep only the selected validation rows from each large observation cache.
    for task in sorted(np.unique(tasks[val_ids]).tolist()):
        payload = pickle.load((OUT / 'obs_cache' / f'task_{int(task)}.pkl').open('rb'))
        if tree_def is None:
            tree_def = payload['tree_def']
        elif payload['tree_def'] != tree_def:
            raise AssertionError('observation cache PyTrees differ')
        rows = np.asarray(payload['row_ids'], np.int64)
        selected = val_ids[tasks[val_ids] == task]
        positions = np.searchsorted(rows, selected)
        if np.any(positions >= len(rows)) or not np.array_equal(rows[positions], selected):
            raise AssertionError(f'validation rows missing from task-{task} cache')
        for pos, row in zip(positions, selected, strict=True):
            cache[int(row)] = [np.asarray(leaf[int(pos)]) for leaf in payload['leaves']]
        del payload

    stats = __import__('openpi.shared.normalize', fromlist=['load']).load(
        train.CHECKPOINT / 'assets')['actions']
    physical = np.asarray(data['target'], np.float32)
    targets = np.stack([expert._draft_to_model(x, stats) for x in physical]).astype(np.float32)
    lengths = np.asarray(data['first_unit_length'], np.int32)
    model, _ = expert._load_expert_model()

    @nnx.jit
    def normal(m, obs, noise):
        return m.sample_actions(jax.random.key(0), obs, num_steps=10, noise=noise)

    def make_batch(ids):
        leaves = [jnp.asarray(np.stack([cache[int(row)][j] for row in ids]))
                  for j in range(len(cache[int(ids[0])]))]
        return jax.tree.unflatten(tree_def, leaves)

    preds = []
    for start in range(0, len(val_ids), int(batch_size)):
        ids = np.asarray(val_ids[start:start + int(batch_size)], np.int64)
        obs = make_batch(ids)
        noise = np.stack([
            np.asarray(jax.random.normal(jax.random.fold_in(jax.random.key(4101), int(row)),
                                         (1, 50, 32)))[0]
            for row in ids
        ]).astype(np.float32)
        preds.extend(np.asarray(normal(model, obs, jnp.asarray(noise)), np.float32))
    pred = np.asarray(preds, np.float32)
    truth = targets[val_ids]
    first5 = float(np.mean((pred[:, :5, :6] - truth[:, :5, :6]) ** 2))
    row_unit = [float(np.mean((p[:int(np.clip(lengths[row], 1, 50)), :6] -
                               truth[i, :int(np.clip(lengths[row], 1, 50)), :6]) ** 2))
                for i, (row, p) in enumerate(zip(val_ids, pred, strict=True))]
    result = {
        'condition': 'pi05_normal_from_noise', 'rows': len(val_ids),
        'validation_row_ids': val_ids.tolist(), 'noise_seed': 4101,
        'num_steps': 10, 'first5_mse6': first5, 'unit_mse6': float(np.mean(row_unit)),
        'metric': 'normalized active6 MSE; first5 over first 5 actions; unit over clipped first_unit_length',
        'mean_first_unit_length': float(np.mean(np.clip(lengths[val_ids], 1, 50))),
        'same_validation_rows_as_expert3': True,
    }
    DEST.parent.mkdir(parents=True, exist_ok=True)
    tmp = DEST.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(result, indent=2) + '\n')
    tmp.replace(DEST)
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--fraction', type=float, default=.25)
    p.add_argument('--batch-size', type=int, default=4)
    a = p.parse_args()
    run(threads=a.threads, fraction=a.fraction, batch_size=a.batch_size)
