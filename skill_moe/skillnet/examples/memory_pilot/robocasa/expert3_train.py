"""RM-EXP3 Expert fit with a per-example uniformly sampled bridge start."""
from __future__ import annotations

import json
import os
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

STATE = Path(os.environ.get('SKILLNET_STATE_DIR', '/work/.skillnet-state'))
OUTPUT_ROOT = Path(os.environ.get('ROBOCASA_OUTPUT_ROOT', str(STATE / 'outputs/robocasa-memory')))
ROOT = Path(os.environ.get('ROBOCASA_EXPERT3_ROOT', str(OUTPUT_ROOT / 'expert-03')))
PREP = Path(os.environ.get('ROBOCASA_EXPERT3_PREP', str(ROOT / 'prep')))
OUT = Path(os.environ.get('ROBOCASA_EXPERT3_TRAIN', str(ROOT / 'train')))
SEED = 20261111
T0_LOW, T0_HIGH = 0.2, 0.5


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                              default=expert.ex._json_default) + '\n')
    tmp.replace(path)


def _variable_grad(filt):
    import jax
    import jax.numpy as jnp
    import flax.nnx as nnx

    @nnx.jit
    def step(model, obs, target, draft, key):
        kt, kp, ke = jax.random.split(key, 3)
        b = target.shape[0]
        t0 = jax.random.uniform(kt, (b,), minval=T0_LOW, maxval=T0_HIGH)
        progress = jax.random.uniform(kp, (b,), minval=.01, maxval=1.)
        noise = jax.random.normal(ke, target.shape)
        # This is the same bridge path used by Expert.sample_bridge: model time
        # is t=t0*progress, with x interpolating target toward the t0 endpoint.
        xt, model_t, velocity = expert._bridge_training_sample(target, draft, noise, t0, progress)
        weight = jnp.asarray([1.] * 7 + [.1] * 5 + [0.] * 20, jnp.float32)

        def loss_fn(m):
            pred = m.velocity(obs, xt, model_t)
            return jnp.sum((pred - velocity) ** 2 * weight[None, None, :]) / (
                jnp.sum(weight) * target.shape[1] * b)

        loss, grads = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, filt))(model)
        return loss, grads

    return step


def train(*, steps=3000, effective_batch=32, microbatch=8, threads=8,
          memory_fraction=.35, time_budget_hours=3.25, resume=True):
    import jax
    import jax.numpy as jnp
    import flax.nnx as nnx
    import optax
    import torch
    from flax import serialization
    from openpi.shared.nnx_utils import PathRegex

    torch.set_num_threads(int(threads))
    if effective_batch % microbatch:
        raise ValueError('effective batch must divide by microbatch')
    torch.cuda.set_per_process_memory_fraction(float(memory_fraction))
    if not (0 < time_budget_hours <= 4):
        raise ValueError('time budget must be in (0,4] hours')

    base.PREP = PREP
    base.OUT = OUT
    out = OUT
    data_path = PREP / 'drafts/memory_dynamic.npz'
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    data = np.load(data_path, allow_pickle=False)
    train_ids, val_ids, _ = base.prepare_full_cache(out, threads=threads)

    cache_map = {}
    tree_def = None
    selected = np.r_[train_ids, val_ids]
    for task in sorted(np.unique(np.asarray(data['task'])[selected]).tolist()):
        payload = pickle.load((out / 'obs_cache' / f'task_{task}.pkl').open('rb'))
        if tree_def is None:
            tree_def = payload['tree_def']
        elif payload['tree_def'] != tree_def:
            raise AssertionError('cached observation PyTrees differ')
        for pos, row in enumerate(payload['row_ids']):
            cache_map[int(row)] = [np.asarray(x[pos]) for x in payload['leaves']]
    if not set(map(int, selected)).issubset(cache_map):
        raise AssertionError('full observation cache incomplete')

    stats = __import__('openpi.shared.normalize', fromlist=['load']).load(
        base.CHECKPOINT / 'assets')['actions']
    raw_drafts = np.asarray(data['draft'], np.float32)
    raw_targets = np.asarray(data['target'], np.float32)
    drafts = np.stack([expert._draft_to_model(row, stats) for row in raw_drafts]).astype(np.float32)
    targets = np.stack([expert._draft_to_model(row, stats) for row in raw_targets]).astype(np.float32)

    model, _ = expert._load_expert_model()  # frozen pi05 checkpoint at 75k
    filt = nnx.All(nnx.Param, PathRegex(r'.*(llm.*_1|action_in_proj|time_mlp|action_out_proj).*'))
    state = nnx.state(model, filt)
    tx = optax.chain(optax.clip_by_global_norm(1.), optax.adamw(
        2.5e-5, b1=.9, b2=.95, weight_decay=1e-5))
    opt = tx.init(state)
    grad = _variable_grad(filt)

    fit = np.asarray(train_ids, np.int64)
    order_path = out / 'train_row_order.npy'
    if order_path.exists():
        order = np.load(order_path, mmap_mode='r')
        if order.shape != (steps, effective_batch) or not np.isin(order, fit).all():
            raise ValueError('saved deterministic train-row order has wrong shape/IDs')
    else:
        rng = np.random.default_rng(SEED)
        order = rng.choice(fit, (steps, effective_batch), replace=True).astype(np.int32)
        tmp = order_path.with_suffix('.npy.tmp')
        with tmp.open('wb') as f:
            np.save(f, order)
        tmp.replace(order_path)

    expert_dir = out / 'expert'
    expert_dir.mkdir(parents=True, exist_ok=True)
    resume_path = expert_dir / 'trainer_state.pkl'
    progress_path = expert_dir / 'progress.json'
    start_step = 0
    if resume and resume_path.exists():
        saved = pickle.load(resume_path.open('rb'))
        if saved.get('steps') != steps or saved.get('effective_batch') != effective_batch:
            raise ValueError('resume state training configuration mismatch')
        state.replace_by_pure_dict(saved['model_state'])
        nnx.update(model, state)
        opt = saved['optimizer_state']
        start_step = int(saved['step'])
        print(json.dumps({'stage': 'resume', 'step': start_step}), flush=True)
    elif any(expert_dir.glob('expert_*.msgpack')):
        raise FileExistsError('checkpoints exist without resume state; refusing an unsafe restart')

    def make_batch(ids):
        leaves = [jnp.asarray(np.stack([cache_map[int(i)][j] for i in ids]))
                  for j in range(len(cache_map[int(ids[0])]))]
        return jax.tree.unflatten(tree_def, leaves)

    # Keep each memory footprint attempt auditable. A restart after a failure
    # may append duplicate step numbers, but the attempt-specific file and
    # checkpoint state make the exact continuation unambiguous.
    log_path = expert_dir / f'train_steps_mb{microbatch}.jsonl'
    mode = 'a' if log_path.exists() else 'x'
    if mode == 'a' and start_step == 0:
        with log_path.open('a') as marker:
            marker.write(json.dumps({'event': 'restart_from_initial_weights',
                                     'time': time.time(), 'microbatch': microbatch}) + '\n')
    started = time.monotonic()
    deadline = float(time_budget_hours) * 3600.
    actual_steps = start_step
    with log_path.open(mode) as log:
        for step0 in range(start_step, steps):
            ids = np.asarray(order[step0], np.int64)
            accum = None
            losses = []
            for batch_start in range(0, effective_batch, microbatch):
                sub = ids[batch_start:batch_start + microbatch]
                obs = make_batch(sub)
                key = jax.random.fold_in(jax.random.key(SEED), step0 * 17 + batch_start)
                loss, grads = grad(model, obs, jnp.asarray(targets[sub]),
                                   jnp.asarray(drafts[sub]), key)
                accum = grads if accum is None else jax.tree.map(lambda a, b: a + b, accum, grads)
                losses.append(float(loss))
            accum = jax.tree.map(lambda g: g / (effective_batch // microbatch), accum)
            state = nnx.state(model, filt)
            updates, opt = tx.update(accum, opt, state)
            nnx.update(model, optax.apply_updates(state, updates))
            actual_steps = step0 + 1
            rec = {'step': actual_steps, 'train_loss': float(np.mean(losses)),
                   't0_low': T0_LOW, 't0_high': T0_HIGH,
                   'microbatch': microbatch, 'effective_batch': effective_batch}
            log.write(json.dumps(rec) + '\n')
            if actual_steps % 10 == 0:
                log.flush()
            if actual_steps % 100 == 0:
                print(json.dumps(rec), flush=True)
                _json(progress_path, {'status': 'running', 'latest_step': actual_steps,
                    'steps_requested': steps, 'optimizer': 'AdamW 2.5e-5, clip 1.0',
                    'effective_batch': effective_batch, 'microbatch': microbatch,
                    't0_sampling': 'independent U[0.2,0.5) per training example',
                    'VLM_frozen': True, 'base_checkpoint': str(base.CHECKPOINT),
                    'elapsed_seconds': time.monotonic() - started})
            if actual_steps % 500 == 0 or actual_steps == steps:
                cp = expert_dir / f'expert_{actual_steps:06d}.msgpack'
                if cp.exists():
                    raise FileExistsError(f'refusing to overwrite {cp}')
                cp.write_bytes(serialization.to_bytes(nnx.state(model, filt).to_pure_dict()))
                resume_data = {'step': actual_steps, 'steps': steps,
                    'effective_batch': effective_batch,
                    'model_state': jax.device_get(nnx.state(model, filt).to_pure_dict()),
                    'optimizer_state': jax.device_get(opt)}
                tmp = resume_path.with_suffix('.pkl.tmp')
                with tmp.open('wb') as f:
                    pickle.dump(resume_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                tmp.replace(resume_path)
                print(json.dumps({'stage': 'checkpoint', 'step': actual_steps,
                                  'path': str(cp), 'resume': str(resume_path)}), flush=True)
            if time.monotonic() - started >= deadline:
                break

    status = 'complete' if actual_steps >= steps else 'time_budget_reached'
    result = {'status': status, 'latest_step': actual_steps, 'steps_requested': steps,
        't0_sampling': 'independent U[0.2,0.5) per example/update',
        'bridge_time': 't = t0 * progress; x=(1-progress)*target+progress*((1-t0)*draft+t0*noise)',
        'optimizer': 'AdamW lr=2.5e-5, b1=.9, b2=.95, weight_decay=1e-5, clip_norm=1',
        'effective_batch': effective_batch, 'microbatch': microbatch, 'threads': threads,
        'memory_fraction': memory_fraction, 'VLM_frozen': True,
        'base_checkpoint': str(base.CHECKPOINT), 'training_seconds': time.monotonic() - started,
        'checkpoint_interval': 500, 'resumable_optimizer_state': str(resume_path)}
    _json(expert_dir / 'train_summary.json', result)
    _json(progress_path, result)
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--microbatch', type=int, default=8)
    parser.add_argument('--effective-batch', type=int, default=32)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--fraction', type=float, default=.35)
    parser.add_argument('--time-budget-hours', type=float, default=3.25)
    parser.add_argument('--no-resume', action='store_true')
    args = parser.parse_args()
    train(steps=args.steps, effective_batch=args.effective_batch,
          microbatch=args.microbatch, threads=args.threads,
          memory_fraction=args.fraction,
          time_budget_hours=args.time_budget_hours, resume=not args.no_resume)
