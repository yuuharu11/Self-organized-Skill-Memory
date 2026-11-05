#!/usr/bin/env python3
"""Re-select E-M2 checkpoint on a fixed Dev sample from its train draft mix."""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

import importlib.util

_spec = importlib.util.spec_from_file_location('fair_expert_runtime', Path(__file__).with_name('run.py'))
fair = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fair)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=fair.OUT_DEFAULT)
    p.add_argument('--run', type=Path, default=fair.RUN)
    p.add_argument('--checkpoint', type=Path, default=fair.CHECKPOINT)
    p.add_argument('--batch', type=int, default=8)
    args = p.parse_args()

    import jax
    import jax.numpy as jnp
    import orbax.checkpoint as ocp
    from openpi.shared.nnx_utils import module_jit

    base = fair._base_run()
    skill = fair._skill_run()
    rows = base.load_rows(args.run)
    dev = fair.validation_rows(rows)
    qmap, qids, cand, _scores = fair._load_candidates(args, rows)
    dev_cands = np.stack([cand[qmap[int(r)]] for r in dev])
    fair.validate_candidates(dev_cands, dev, rows,
        np.asarray(np.load(fair.MEMORY / 'units.npz', allow_pickle=False)['pool_row_ids'], np.int64))
    val_ids, val_ranks = fair.sample_m2_training_draft(
        dev_cands, np.random.default_rng(fair.SEED + 1902))
    actions = np.load(args.run / 'rows' / 'actions.npy', mmap_mode='r')
    val_drafts = fair.candidate_drafts(actions, val_ids)
    val_target = np.asarray(actions[dev, :, :7], np.float32)
    val_eps = np.stack([np.asarray(jax.random.normal(
        jax.random.fold_in(jax.random.key(88261), int(r)), (10, 32))) for r in dev])
    val_obs = base.obs_batch(rows, dev)

    model, _ = base.load_policy(args.checkpoint)
    checkpoints = sorted((args.output / 'E-M2' / 'bridge').iterdir())
    records = []
    started = time.monotonic()
    for checkpoint in checkpoints:
        with ocp.PyTreeCheckpointer() as ckpt:
            partial = ckpt.restore(checkpoint)
        skill.load_expert(model, checkpoint)
        bridge = module_jit(model.sample_bridge, static_argnames=('num_steps',))
        vals = {}
        for t0 in fair.T0S:
            row_losses = []
            for st in range(0, len(dev), args.batch):
                ix = slice(st, min(st + args.batch, len(dev)))
                obs = jax.tree.map(lambda x: x[ix], val_obs)
                pred = np.asarray(bridge(jax.random.key(0), obs, jnp.asarray(val_drafts[ix]),
                    jnp.full((len(dev[ix]),), t0), num_steps=10,
                    noise=jnp.asarray(val_eps[ix])))[:, :, :7]
                row_losses.extend(np.mean((pred - val_target[ix]) ** 2, axis=(1, 2)).tolist())
            vals[str(t0)] = float(np.mean(row_losses))
        best_t0 = min(fair.T0S, key=lambda t: vals[str(t)])
        records.append({'checkpoint': str(checkpoint), 'steps': int(checkpoint.name),
                        'mse_by_t0': vals, 'best_t0': best_t0,
                        'best_mse': vals[str(best_t0)]})
        del bridge
        jax.clear_caches()

    best = min(records, key=lambda x: x['best_mse'])
    selected = {'path': best['checkpoint'], 'steps': best['steps'], 't0': best['best_t0'],
        'validation_generated_active7_mse': best['best_mse'],
        'selection': 'fixed Dev sample from the train-time per-query rank mix; seed 20263006; six withheld tasks; no Long'}
    edir = args.output / 'E-M2'
    prior = edir / 'best_bridge.json'
    if prior.exists() and not (edir / 'best_bridge_top1_dev_provisional.json').exists():
        shutil.copy2(prior, edir / 'best_bridge_top1_dev_provisional.json')
    prior.write_text(json.dumps(selected, indent=2) + '\n')
    (edir / 'dev_reselection.json').write_text(json.dumps({
        'method': 'post-training fixed-rank-mix Dev selection; E-M2 training unchanged',
        'dev_rows': len(dev), 'rank_counts': {str(k): int(np.sum(val_ranks == k)) for k in range(5)},
        'selected': selected, 'checkpoints': records, 'seconds': time.monotonic() - started,
    }, indent=2) + '\n')
    trained_path = edir / 'trained.json'
    trained = json.loads(trained_path.read_text())
    trained['best'] = selected
    trained['dev_selection_source'] = 'reselected after noticing original validation used Top-1 only; fixed mixture draws saved in dev_reselection.json'
    trained['validation_draft_rank_counts'] = {str(k): int(np.sum(val_ranks == k)) for k in range(5)}
    trained_path.write_text(json.dumps(trained, indent=2) + '\n')
    print(json.dumps({'selected': selected, 'rank_counts': trained['validation_draft_rank_counts'],
                      'checkpoints': len(records)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
