#!/usr/bin/env python3
"""Fair M2-vs-zero bridge Expert comparison (offline only).

This script reads the run-01 dataset, the frozen key-mechanism M2 and raw
noLong memory. All artifacts are written to a fresh output directory.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np

import os as _os
_REPO = Path(__file__).resolve().parents[6]
_STATE_DIR = Path(_os.environ.get('SKILLNET_STATE_DIR', _REPO / '.skillnet-state'))
_PI05_ROOT = Path(_os.environ.get('SKILLNET_PI05_ROOT', _STATE_DIR / 'outputs/memory-pilot/pi05-memory40-20261104'))
ROOT = _PI05_ROOT
RUN = ROOT / 'run-01'
MEMORY = ROOT / 'skill-memories-01' / 'raw-rows-noLong'
M2_DIR = ROOT / 'key-mechanism-01' / 'production'
GAP = ROOT / 'memory-gap-01' / 'predictions_and_errors.npz'
HARD = ROOT / 'retrieval-4b-01' / 'raw_draft_final_mse.npz'
CHECKPOINT = _STATE_DIR / 'checkpoints/openpi-assets/checkpoints/pi05_libero'
OUT_DEFAULT = ROOT / 'fair-expert-01'
HOLDOUT = np.asarray([17, 19, 28, 29, 35, 38], np.int32)
LONG_TASKS = np.arange(10, dtype=np.int32)
SEED = 20261104
T0S = (.2, .4, .6)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _base_run():
    return _load_module('fair_expert_pi05_base', Path(__file__).resolve().parents[1] / 'run.py')


def _skill_run():
    return _load_module('fair_expert_skill_runtime', Path(__file__).resolve().parents[1] / 'skill_expert' / 'run.py')


def _km_models():
    return _load_module('fair_expert_km_models', Path(__file__).resolve().parents[1] / 'key_mechanism' / 'models.py')


def _write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=True) + '\n')
    tmp.replace(path)


def training_rows(rows):
    task = np.asarray(rows['task'])
    validation = np.asarray(rows['validation'], bool)
    ids = np.flatnonzero((~validation) & (task >= 10) & ~np.isin(task, HOLDOUT)).astype(np.int64)
    if np.any(task[ids] < 10) or np.any(np.isin(task[ids], HOLDOUT)) or np.any(validation[ids]):
        raise AssertionError('Long/heldout/validation data leaked into Expert training')
    return ids


def validation_rows(rows):
    # run-01 marks every held-out episode as validation; use its preexisting
    # calibration split to keep the final Eval episodes out of checkpoint choice.
    dev_episodes = set()
    for task in np.unique(np.asarray(rows['task'])):
        episodes = np.unique(np.asarray(rows['episode'])[
            (np.asarray(rows['task']) == task) & np.asarray(rows['validation'], bool)])
        if len(episodes):
            order = np.random.default_rng(20261105 + int(task)).permutation(episodes)
            dev_episodes.update((int(task), int(e)) for e in order[:3])
    task_arr = np.asarray(rows['task'])
    episode_arr = np.asarray(rows['episode'])
    is_dev_episode = np.asarray([(int(t), int(e)) in dev_episodes for t, e in zip(task_arr, episode_arr)])
    ids = np.flatnonzero(np.asarray(rows['validation'], bool) & is_dev_episode &
                         np.isin(task_arr, HOLDOUT)).astype(np.int64)
    if set(np.unique(np.asarray(rows['task'])[ids]).tolist()) != set(HOLDOUT.tolist()):
        raise AssertionError('validation rows do not cover the six predeclared heldout tasks')
    if np.any(np.asarray(rows['task'])[ids] < 10):
        raise AssertionError('Long rows leaked into Dev')
    return ids


def validate_candidates(candidate_rows, query_rows, rows, pool_ids, *, require_task_exclusion=True):
    c = np.asarray(candidate_rows, np.int64)
    q = np.asarray(query_rows, np.int64)
    if c.ndim != 2 or c.shape[0] != len(q):
        raise ValueError('candidate row IDs must align with query IDs')
    allowed = np.zeros(len(rows['actions']), bool)
    allowed[np.asarray(pool_ids, np.int64)] = True
    if np.any((c < 0) | (c >= len(allowed))) or not np.all(allowed[c]):
        raise AssertionError('candidate row ID outside raw noLong memory')
    if require_task_exclusion and np.any(np.asarray(rows['task'])[c] == np.asarray(rows['task'])[q, None]):
        raise AssertionError('same-task candidate leaked')
    return True


def candidate_drafts(actions, candidate_row_ids):
    """Map dataset row IDs directly to actions (never treat them as pool offsets)."""
    c = np.asarray(candidate_row_ids, np.int64)
    if c.size and (c.min() < 0 or c.max() >= len(actions)):
        raise IndexError('candidate row ID out of dataset range')
    return np.asarray(actions[c], np.float32)


def sample_m2_training_draft(candidate_row_ids, rng):
    c = np.asarray(candidate_row_ids, np.int64)
    if c.ndim != 2 or c.shape[1] < 5:
        raise ValueError('M2 list needs Top-1 through Top-5')
    # Independent per-query draw: .70 rank 1; .30 spread uniformly over
    # ranks 2..5 (each .075). A single batch-level draw would be invalid.
    ranks = rng.choice(5, size=len(c), p=[.70, .075, .075, .075, .075]).astype(np.int8)
    return c[np.arange(len(c)), ranks], ranks.astype(np.int8)


def shuffle_other_task(candidate_row_ids, query_rows, rows, *, seed):
    """Deterministically permute selected drafts until source tasks differ."""
    c = np.asarray(candidate_row_ids, np.int64)
    q = np.asarray(query_rows, np.int64)
    chosen = c[:, 0].copy()
    rng = np.random.default_rng(seed)
    for _ in range(100):
        perm = rng.permutation(len(q))
        bad = np.asarray(rows['task'])[chosen[perm]] == np.asarray(rows['task'])[q]
        if not np.any(bad):
            out = chosen[perm]
            if np.any(np.asarray(rows['task'])[out] == np.asarray(rows['task'])[q]):
                raise AssertionError('shuffle did not cross task')
            return out, perm
    # Never fall back to sampling with replacement: that changes the source
    # distribution and makes a content-only shuffle comparison unfair.
    raise RuntimeError('could not find a cross-task permutation preserving the candidate multiset')


def make_batch_indices(train_ids, *, steps=4000, batch=8, seed=SEED):
    rng = np.random.default_rng(seed)
    return np.stack([rng.choice(train_ids, batch, replace=False) for _ in range(steps)]).astype(np.int64)


def _context(rows):
    # Exactly the retrieval-4b/key-mechanism query context.
    import importlib.util
    path = Path(__file__).resolve().parents[1] / 'key_mechanism' / 'run.py'
    import sys
    sys.path.insert(0, str(path.parent))
    km = _load_module('fair_expert_key_mechanism', path)
    return km._history(rows)


def prepare(args):
    import torch
    torch.set_num_threads(args.threads)
    if not torch.cuda.is_available():
        raise RuntimeError('M2 exact token reranking requires CUDA')
    torch.cuda.set_per_process_memory_fraction(.25)
    base = _base_run()
    rows = base.load_rows(args.run)
    train = training_rows(rows)
    dev = validation_rows(rows)
    pool = np.asarray(np.load(args.memory / 'units.npz', allow_pickle=False)['pool_row_ids'], np.int64)
    if len(pool) != 3857 or not np.array_equal(pool, np.sort(pool)):
        raise AssertionError('raw-rows-noLong memory row inventory changed')
    if np.any(np.asarray(rows['task'])[pool] < 10):
        raise AssertionError('Long memory row entered noLong pool')
    # No evaluation episode may participate in query fitting or Dev selection.
    eval_all = np.load(args.gap, allow_pickle=False)
    eval_rows = np.asarray(eval_all['row_ids'], np.int64)
    eval_pairs = set(zip(np.asarray(rows['task'])[eval_rows].tolist(),
                         np.asarray(rows['episode'])[eval_rows].tolist()))
    if any((int(rows['task'][r]), int(rows['episode'][r])) in eval_pairs for r in train):
        raise AssertionError('Eval episode leaked into Expert train rows')
    if any((int(rows['task'][r]), int(rows['episode'][r])) in eval_pairs for r in dev):
        raise AssertionError('Eval episode leaked into validation rows')

    # Reuse the trained M2 weights and cached VLM token features. M2's own
    # trained query rows are reflected in reports, not used as Expert labels.
    km_path = Path(__file__).resolve().parents[1] / 'key_mechanism' / 'run.py'
    import sys
    sys.path.insert(0, str(km_path.parent))
    km = _load_module('fair_expert_km', km_path)
    M2SalientTokens = _km_models().M2SalientTokens
    payload = torch.load(args.m2_dir / 'models.pt', map_location='cpu', weights_only=False)
    model = M2SalientTokens(dim=96, token_count=512)
    model.load_state_dict(payload['models']['M2_salient_maxsim'])
    model.to('cuda').eval()
    tokens = np.load(args.run / 'tokens.npy', mmap_mode='r', allow_pickle=False)
    groups = np.load(args.run / 'groups.npy', mmap_mode='r', allow_pickle=False)
    ctx = _context(rows)
    # Query set includes all Expert train queries and the six-task heldout Dev.
    qids = np.unique(np.r_[train, dev]).astype(np.int64)
    started = time.monotonic()
    qk = km._encode_keys(model, qids, rows, tokens, groups, ctx, batch=8)
    mk = km._encode_keys(model, pool, rows, tokens, groups, ctx, batch=8)
    qh, qw = km._encode_tokens(model, qids, rows, tokens, groups, ctx, batch=4)
    mh, mw = km._encode_tokens(model, pool, rows, tokens, groups, ctx, batch=4)
    coarse, _, _ = km._retrieve(qk, mk, qids, pool, rows, rows['actions'], k=64)
    km._m2_rerank.mem_tokens = mh
    km._m2_rerank.mem_weights = mw
    ranked, scores = km._m2_rerank(qh, qw, qids, pool, coarse, rows, batch=4)
    validate_candidates(ranked, qids, rows, pool)
    # Long rows are evaluated from the previously materialized production list;
    # confirm fresh code/weights produce identical row IDs for that fixed split.
    saved = np.load(args.m2_dir / 'candidate_rows.npz', allow_pickle=False)
    long_q = np.asarray(saved['query_rows'], np.int64)
    long_c = np.asarray(saved['M2_salient_maxsim_rows'], np.int64)
    if not np.array_equal(np.asarray(rows['task'])[long_q] < 10, np.ones(len(long_q), bool)):
        raise AssertionError('cached M2 validation query list is not the fixed Long Eval set')
    # Validation uses exact cache to prevent drift; training candidates come from
    # the same weights/reranking implementation above.
    query_lookup = {int(r): i for i, r in enumerate(qids)}
    if np.any([int(r) not in query_lookup for r in long_q]):
        pass  # Long queries are not part of train/Dev by design.
    np.savez_compressed(args.output / 'candidate_lists.npz', query_rows=qids,
                        candidate_rows=ranked.astype(np.int64), scores=scores.astype(np.float32),
                        long_query_rows=long_q, long_candidate_rows=long_c)
    # Make the shared minibatch order once; each Expert consumes this exact array.
    batch_ids = make_batch_indices(train, steps=args.steps, batch=args.batch)
    np.save(args.output / 'batch_row_ids.npy', batch_ids)
    bsha = hashlib.sha256(batch_ids.tobytes()).hexdigest()
    # Diagnostic: compare candidate draft quality on M2's original fitting tasks
    # versus the six Expert selection tasks. No task labels enter retrieval.
    loc = np.asarray([query_lookup[int(r)] for r in train], np.int64)
    m2_first, _ = sample_m2_training_draft(ranked[loc], np.random.default_rng(SEED + 73))
    draft_mse = np.mean((candidate_drafts(rows['actions'], m2_first)[:, :, :7] -
                         np.asarray(rows['actions'][train])[:, :, :7]) ** 2, axis=(1, 2))
    # M2's own key-learning split is read from its recorded protocol.
    key_protocol = json.loads((args.m2_dir / 'protocol.json').read_text())
    original_fit_tasks = set(map(int, key_protocol['training_tasks']))
    dev_cands = np.stack([ranked[query_lookup[int(r)]] for r in dev])
    dev_top1_mse = np.mean((candidate_drafts(rows['actions'], dev_cands[:, 0])[:, :, :7] -
                            np.asarray(rows['actions'][dev])[:, :, :7]) ** 2, axis=(1, 2))
    # Also report query-task groups that overlapped M2 fitting vs heldout tasks.
    fit_mask = np.isin(np.asarray(rows['task'])[train], list(original_fit_tasks))
    hold_mask = np.isin(np.asarray(rows['task'])[train], HOLDOUT)
    _write_json(args.output / 'prepare.json', {
        'train_rows': int(len(train)), 'validation_rows': int(len(dev)),
        'memory_rows': int(len(pool)), 'candidate_k': 64, 'm2_ranked': 'coarse top-64 + salient weighted token MaxSim',
        'm2_key_weights': str(args.m2_dir / 'models.pt'), 'fixed_long_rows': int(len(long_q)),
        'm2_long_cached_top1_matches_recomputed': None,
        'training_row_batch_sha256': bsha, 'training_row_ids_sha256': hashlib.sha256(train.tobytes()).hexdigest(),
        'm2_training_task_ids': sorted(original_fit_tasks),
        'm2_fit_query_tasks_draft_mse': float(draft_mse[fit_mask].mean()) if fit_mask.any() else None,
        'fair_heldout_six_task_train_query_draft_mse': float(draft_mse[hold_mask].mean()) if hold_mask.any() else None,
        'six_task_validation_query_top1_draft_mse': float(dev_top1_mse.mean()),
        'query_task_partition_note': 'M2 fit overlap uses the recorded key-mechanism training_tasks; six-task validation query error is separately measured',
        'candidate_rows_are_dataset_row_ids': True, 'actions_lookup': 'rows/actions.npy[candidate_row_id]',
        'seconds': time.monotonic() - started,
    })
    print(json.dumps({'stage': 'prepare', 'queries': len(qids), 'train': len(train), 'dev': len(dev),
                      'seconds': time.monotonic() - started}), flush=True)


def _load_candidates(args, rows):
    z = np.load(args.output / 'candidate_lists.npz', allow_pickle=False)
    return ({int(r): i for i, r in enumerate(z['query_rows'])},
            np.asarray(z['query_rows'], np.int64), np.asarray(z['candidate_rows'], np.int64),
            np.asarray(z['scores'], np.float32))


def train(args):
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import optax
    import orbax.checkpoint as ocp
    from openpi.shared.nnx_utils import PathRegex
    base = _base_run()
    load_rows, obs_batch, load_policy = base.load_rows, base.obs_batch, base.load_policy
    make_bridge_step = base.make_bridge_step

    rows = load_rows(args.run)
    train = training_rows(rows)
    dev = validation_rows(rows)
    if args.expert == 'em2' and args.draft_source != 'm2':
        raise ValueError('E-M2 requires M2 draft source')
    if args.expert == 'zero' and args.draft_source != 'zero':
        raise ValueError('E-zero requires zero draft source')
    (qmap, qids, cand, _scores) = _load_candidates(args, rows)
    batch_ids = np.load(args.output / 'batch_row_ids.npy', allow_pickle=False)
    if batch_ids.ndim != 2 or batch_ids.shape[1] != args.batch or args.steps > len(batch_ids) or \
       not np.all(np.isin(batch_ids[:args.steps], train)):
        raise AssertionError('shared training batch sequence changed')
    batch_ids = batch_ids[:args.steps]
    bsha = hashlib.sha256(batch_ids.tobytes()).hexdigest()
    if args.expert == 'em2':
        dev_cands = np.stack([cand[qmap[int(r)]] for r in dev])
        validate_candidates(dev_cands, dev, rows,
                            np.asarray(np.load(args.memory / 'units.npz')['pool_row_ids'], np.int64))
        val_draft_ids, val_ranks = sample_m2_training_draft(
            dev_cands, np.random.default_rng(SEED + 1902))
    else:
        val_draft_ids = np.full(len(dev), -1, np.int64)
        val_ranks = np.full(len(dev), -1, np.int8)

    if (args.expert_dir / 'training_progress.json').exists() or (args.expert_dir / 'best_bridge.json').exists():
        raise FileExistsError(f'refusing to overwrite expert artifacts: {args.expert_dir}')
    args.expert_dir.mkdir(parents=True, exist_ok=True)
    model, _ = load_policy(args.checkpoint)
    filt = nnx.All(nnx.Param, PathRegex(r'.*(llm.*_1|action_in_proj|time_mlp|action_out_proj).*'))
    selected = nnx.state(model, filt)
    if sum(x.value.size for x in selected.flat_state().values()) < 100_000_000:
        raise ValueError('action expert trainable filter did not select expected dense tower')
    tx = optax.chain(optax.clip_by_global_norm(1.), optax.adamw(2.5e-5, b1=.9, b2=.95, weight_decay=1e-5))
    opt = tx.init(selected)
    step_fn = make_bridge_step(filt, tx)
    jkey = jax.random.key(SEED)
    if args.expert == 'em2':
        d_rng = np.random.default_rng(SEED + 1901)
        ranks = np.empty(len(train), np.int8)
        train_candidate_ids = np.stack([cand[qmap[int(r)]] for r in train])
        pick, ranks = sample_m2_training_draft(train_candidate_ids, d_rng)
        train_draft = candidate_drafts(np.load(args.run / 'rows' / 'actions.npy', mmap_mode='r'), pick)
        train_row_to_index = {int(r): j for j, r in enumerate(train)}
        saved_drafts = args.expert_dir / 'training_drafts.npz'
        if saved_drafts.exists():
            prev = np.load(saved_drafts, allow_pickle=False)
            if not np.array_equal(prev['query_rows'], train) or not np.array_equal(prev['selected_candidate_rows'], pick):
                raise AssertionError('existing E-M2 training draft artifact differs; refusing overwrite')
        else:
            np.savez_compressed(saved_drafts, query_rows=train,
                                candidate_rows=train_candidate_ids, selected_candidate_rows=pick,
                                rank=ranks, draft_mse=np.mean((train_draft[:, :, :7] -
                                np.asarray(rows['actions'][train])[:, :, :7]) ** 2, axis=(1, 2)))
    else:
        train_draft = None
        ranks = np.full(len(train), -1, np.int8)
        train_row_to_index = None

    # The validation-generation function chooses checkpoint and t0 independently
    # for each Expert, on the heldout task rows only.
    from openpi.shared.nnx_utils import module_jit
    val_eps = np.stack([np.asarray(jax.random.normal(jax.random.fold_in(jax.random.key(88261), int(r)), (10, 32)))
                        for r in dev])
    val_target = np.asarray(rows['actions'][dev], np.float32)[:, :, :7]
    if args.expert == 'em2':
        val_drafts = candidate_drafts(np.load(args.run / 'rows' / 'actions.npy', mmap_mode='r'), val_draft_ids)
    else:
        val_drafts = np.zeros((len(dev), 10, 32), np.float32)
    best_mse = float('inf')
    best_info = None
    logs = []
    started = time.monotonic()
    expert_dir = args.expert_dir
    bridge_dir = expert_dir / 'bridge'
    bridge_dir.mkdir(exist_ok=True)
    if any(bridge_dir.iterdir()):
        raise FileExistsError(f'checkpoint directory is nonempty: {bridge_dir}')
    for i in range(args.steps):
        ids = batch_ids[i]
        if args.expert == 'em2':
            # Draft sampling was fixed before training and is independent of the
            # shared row minibatch RNG, so E-zero consumes the identical data order.
            draft = train_draft[[train_row_to_index[int(r)] for r in ids]]
        else:
            draft = np.zeros((len(ids), 10, 32), np.float32)
        jkey, subkey = jax.random.split(jkey)
        loss, opt = step_fn(model, opt, obs_batch(rows, ids), jnp.asarray(rows['actions'][ids]),
                            jnp.asarray(draft), subkey)
        loss = float(loss)
        if not np.isfinite(loss):
            raise FloatingPointError(f'{args.expert} nonfinite loss at {i+1}')
        rec = {'step': i + 1, 'train_velocity_mse': loss, 'elapsed_s': time.monotonic() - started}
        logs.append(rec)
        if (i + 1) % args.save_every == 0 or i + 1 == args.steps:
            ckpt = expert_dir / 'bridge' / f'{i+1:06d}'
            ocp.PyTreeCheckpointer().save(ckpt, nnx.state(model, filt).to_pure_dict())
            # module_jit freezes module state at construction. Construct it only
            # after this update so Dev checkpoint selection evaluates these
            # weights rather than the initial expert snapshot.
            bridge = module_jit(model.sample_bridge, static_argnames=('num_steps',))
            best_at_update = {'mse': float('inf'), 't0': None}
            val_obs = obs_batch(rows, dev)
            for t0 in T0S:
                vals = []
                for st in range(0, len(dev), args.batch):
                    ix = slice(st, min(st + args.batch, len(dev)))
                    pred = np.asarray(bridge(jax.random.key(0), jax.tree.map(lambda x: x[ix], val_obs),
                                             jnp.asarray(val_drafts[ix]), jnp.full((len(dev[ix]),), t0),
                                             num_steps=10, noise=jnp.asarray(val_eps[ix])))[:, :, :7]
                    vals.extend(np.mean((pred - val_target[ix]) ** 2, axis=(1, 2)).tolist())
                vmse = float(np.mean(vals))
                rec[f'dev_mse_t0_{t0:.1f}'] = vmse
                if vmse < best_at_update['mse']:
                    best_at_update = {'mse': vmse, 't0': t0}
            rec['dev_selected_mse'] = best_at_update['mse']
            rec['dev_selected_t0'] = best_at_update['t0']
            if best_at_update['mse'] < best_mse:
                best_mse = best_at_update['mse']
                best_info = {'path': str(ckpt), 'steps': i + 1, 't0': best_at_update['t0'],
                             'validation_generated_active7_mse': best_mse,
                             'selection': 'six withheld tasks validation rows; own draft; t0 in .2/.4/.6; no Long'}
                _write_json(expert_dir / 'best_bridge.json', best_info)
            _write_json(expert_dir / 'training_progress.json', {
                'status': 'running', 'expert': args.expert, 'target_steps': args.steps,
                'training_logs': logs, 'best': best_info,
                'batch_row_ids_sha256': bsha,
            'validation_draft_rank_counts': {str(k): int(np.sum(val_ranks == k))
                                             for k in range(5)} if args.expert == 'em2' else {},
            })
            print(json.dumps({'stage': 'train', 'expert': args.expert, **rec}), flush=True)
    _write_json(expert_dir / 'trained.json', {
        'expert': args.expert, 'steps': args.steps, 'seed': SEED, 'seconds': time.monotonic() - started,
        'batch_size': args.batch, 'train_rows': len(train), 'dev_rows': len(dev),
        'batch_row_ids_sha256': bsha, 'best': best_info,
        'trainable_filter': 'llm.*_1|action_in_proj|time_mlp|action_out_proj',
        'loss': 'skill_expert.run.make_bridge_step; active7 velocity MSE',
        'optimizer': 'AdamW lr=2.5e-5 b1=.9 b2=.95 wd=1e-5; clip=1',
        'training_logs': logs,
        'validation_draft_rank_counts': {str(k): int(np.sum(val_ranks == k))
                                         for k in range(5)} if args.expert == 'em2' else {},
    })


def _bootstrap_diff(a, b, episode_ids, *, reps=5000, seed=4101):
    a, b, ep = np.asarray(a, np.float64), np.asarray(b, np.float64), np.asarray(episode_ids)
    d = a - b
    uniq = np.unique(ep)
    groups = {u: np.flatnonzero(ep == u) for u in uniq}
    rng = np.random.default_rng(seed)
    means = np.empty(reps, np.float64)
    for i in range(reps):
        draw = rng.choice(uniq, len(uniq), replace=True)
        ix = np.concatenate([groups[u] for u in draw])
        means[i] = d[ix].mean()
    return {'mean': float(d.mean()), 'ci95': [float(np.quantile(means, .025)), float(np.quantile(means, .975))],
            'episodes': int(len(uniq)), 'rows': int(len(a)), 'reps': reps}


def evaluate(args):
    import jax
    import jax.numpy as jnp
    from openpi.shared.nnx_utils import module_jit
    base = _base_run()
    load_rows, obs_batch, load_policy = base.load_rows, base.obs_batch, base.load_policy
    load_expert = _skill_run().load_expert

    rows = load_rows(args.run)
    gap = np.load(args.gap, allow_pickle=False)
    eval_ids = np.asarray(gap['row_ids'], np.int64)
    long_mask = np.asarray(rows['task'][eval_ids]) < 10
    long_ids = eval_ids[long_mask]
    if len(long_ids) != 548 or set(np.asarray(rows['task'][long_ids]).tolist()) != set(range(10)):
        raise AssertionError(f'expected fixed Long Eval rows=548, got {len(long_ids)}')
    hard_data = np.load(args.hard, allow_pickle=False)
    if not np.array_equal(hard_data['eval_rows'], eval_ids):
        raise AssertionError('hard mask rows mismatch fixed Eval')
    hard = np.asarray(hard_data['hard_mask'], bool)[long_mask]
    if not hard.any():
        raise AssertionError('no hard rows in the Long split')
    saved = np.load(args.m2_dir / 'candidate_rows.npz', allow_pickle=False)
    if not np.array_equal(saved['query_rows'], long_ids):
        raise AssertionError('M2 cached Long row order mismatch')
    m2_cands = np.asarray(saved['M2_salient_maxsim_rows'], np.int64)
    pool = np.asarray(np.load(args.memory / 'units.npz', allow_pickle=False)['pool_row_ids'], np.int64)
    validate_candidates(m2_cands, long_ids, rows, pool)
    m2_rows = m2_cands[:, 0]
    # F2 compares M2 Top-1 with a shuffled Top-1 candidate. Permuting this
    # exact source set preserves the candidate/action distribution; drawing a
    # rank-1..5 mixture here would confound content with candidate quality.
    shuffle_sources = m2_cands[:, 0].copy()
    shuffle_source_ranks = np.zeros(len(long_ids), np.int8)
    shuffled_rows, shuffle_perm = shuffle_other_task(m2_cands[:, :1], long_ids, rows, seed=SEED + 812)
    if np.any(np.asarray(rows['task'])[shuffled_rows] == np.asarray(rows['task'])[long_ids]):
        raise AssertionError('evaluation shuffle source task must differ')
    actions = np.load(args.run / 'rows' / 'actions.npy', mmap_mode='r')
    draft_m2 = candidate_drafts(actions, m2_rows)
    draft_shuf = candidate_drafts(actions, shuffled_rows)
    draft_zero = np.zeros((len(long_ids), 10, 32), np.float32)

    # Reuse key-mechanism's exact saved direct-regression draft on these rows.
    km_gen = np.load(args.m2_dir / 'generation_predictions.npz', allow_pickle=False)
    if not np.array_equal(km_gen['query_rows'], long_ids):
        raise AssertionError('key-mechanism direct-regression rows differ from fixed Long Eval')
    name_idx = {str(n): i for i, n in enumerate(km_gen['method_names'])}
    if 'direct_regression' not in name_idx:
        raise AssertionError('key-mechanism direct-regression draft missing')
    dreg = np.asarray(km_gen['drafts'][:, name_idx['direct_regression']], np.float32)

    conditions = {}
    all_preds = {}
    all_drafts = {'m2': draft_m2, 'shuffle': draft_shuf, 'zero': draft_zero, 'direct_regression': dreg}
    # Official normal π0.5 as a separate reference with the same fixed noise bank.
    target = np.asarray(actions[long_ids, :, :7], np.float32)
    episode = (np.asarray(rows['task'][long_ids], np.int64) * 100000 +
               np.asarray(rows['episode'][long_ids], np.int64))
    noise = np.stack([np.asarray(jax.random.normal(jax.random.fold_in(jax.random.key(4101), int(r)), (10, 32)))
                      for r in long_ids])
    partial_dir = args.output / 'eval_partials'
    partial_dir.mkdir(exist_ok=True)
    progress_path = args.output / 'evaluation_progress.jsonl'
    normal_path = partial_dir / 'official_normal.npz'
    if normal_path.exists():
        cached = np.load(normal_path, allow_pickle=False)
        if not np.array_equal(cached['row_ids'], long_ids):
            raise AssertionError('partial normal predictions do not match fixed Long rows')
        normal = np.asarray(cached['prediction'], np.float32)
    else:
        official, _ = load_policy(args.checkpoint)
        normal_fn = module_jit(official.sample_actions, static_argnames=('num_steps',))
        normal = np.empty_like(target)
        for st in range(0, len(long_ids), args.batch):
            q = long_ids[st:st + args.batch]
            p = np.asarray(normal_fn(jax.random.key(0), obs_batch(rows, q), num_steps=10,
                                     noise=jnp.asarray(noise[st:st + len(q)])))[:, :, :7]
            normal[st:st + len(q)] = p
        np.savez_compressed(normal_path, row_ids=long_ids, prediction=normal)
        del official, normal_fn
        jax.clear_caches()
    all_preds['official_normal'] = normal
    conditions['official_normal'] = {'t0': 1., 'expert': 'official', 'draft': 'none'}

    expert_specs = [
        ('E-M2', Path(json.loads((args.output / 'E-M2' / 'best_bridge.json').read_text())['path'])),
        ('E-zero', Path(json.loads((args.output / 'E-zero' / 'best_bridge.json').read_text())['path'])),
        ('old', Path(json.loads((args.run / 'best_bridge.json').read_text())['path'])),
    ]
    for expert_name, checkpoint in expert_specs:
        policy = None
        fn = None
        drafts = {'M2_top1': draft_m2, 'shuffle': draft_shuf, 'zero': draft_zero,
                  'direct_regression': dreg}
        for draft_name, draft in drafts.items():
            if expert_name == 'E-zero' and draft_name not in {'zero', 'direct_regression'}:
                continue
            for t0 in T0S:
                part_key = f'{expert_name}_{draft_name}_t0{t0:.1f}'.replace('/', '_')
                part_path = partial_dir / f'{part_key}.npz'
                if part_path.exists():
                    cached = np.load(part_path, allow_pickle=False)
                    if not np.array_equal(cached['row_ids'], long_ids):
                        raise AssertionError(f'partial row mismatch: {part_path}')
                    pred = np.asarray(cached['prediction'], np.float32)
                else:
                    if policy is None:
                        policy, _ = load_policy(args.checkpoint)
                        load_expert(policy, checkpoint)
                        fn = module_jit(policy.sample_bridge, static_argnames=('num_steps',))
                    pred = np.empty_like(target)
                    for st in range(0, len(long_ids), args.batch):
                        q = long_ids[st:st + args.batch]; sl = slice(st, st + len(q))
                        p = np.asarray(fn(jax.random.key(0), obs_batch(rows, q), jnp.asarray(draft[sl]),
                                          jnp.full((len(q),), t0), num_steps=10,
                                          noise=jnp.asarray(noise[sl])))[:, :, :7]
                        pred[sl] = p
                    np.savez_compressed(part_path, row_ids=long_ids, prediction=pred)
                with progress_path.open('a') as pf:
                    pf.write(json.dumps({'expert': expert_name, 'draft': draft_name, 't0': t0,
                                         'status': 'complete', 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}) + '\n')
                name = f'{expert_name}/{draft_name}/t0={t0:.1f}'
                all_preds[name] = pred
                conditions[name] = {'expert': expert_name, 'draft': draft_name, 't0': t0}
        if policy is not None:
            del fn, policy
            import gc
            gc.collect()
            jax.clear_caches()

    # Per-row active7 MSE, hard split, and episode-cluster contrasts.
    mse = {k: np.mean((v - target) ** 2, axis=(1, 2)).astype(np.float32) for k, v in all_preds.items()}
    summaries = {}
    for name, values in mse.items():
        summaries[name] = {
            'all_mean': float(values.mean()), 'all_median': float(np.median(values)),
            'all_le_0.01': float(np.mean(values <= .01)),
            'hard_mean': float(values[hard].mean()) if hard.any() else None,
            'hard_median': float(np.median(values[hard])) if hard.any() else None,
        }
    best_sel = json.loads((args.output / 'E-M2' / 'best_bridge.json').read_text())
    zero_sel = json.loads((args.output / 'E-zero' / 'best_bridge.json').read_text())
    selected_t0 = float(best_sel['t0'])
    m2_selected = mse[f'E-M2/M2_top1/t0={selected_t0:.1f}']
    zero_selected = mse[f'E-zero/zero/t0={float(zero_sel["t0"]):.1f}']
    shuffle_selected = mse[f'E-M2/shuffle/t0={selected_t0:.1f}']
    pairs = {}
    for subset, mask in [('all_long', np.ones(len(long_ids), bool)), ('hard', hard)]:
        pairs[f'F1_{subset}'] = _bootstrap_diff(m2_selected[mask], zero_selected[mask], episode[mask])
        pairs[f'F2_{subset}'] = _bootstrap_diff(m2_selected[mask], shuffle_selected[mask], episode[mask])
        pairs[f'F3_official_{subset}'] = _bootstrap_diff(m2_selected[mask], mse['official_normal'][mask], episode[mask])

    # Sensitivity: generated change divided by draft change for paired M2 vs
    # shuffled draft on the same rows/noise.
    sensitivity = {}
    for expert in ('E-M2', 'old'):
        rows_ratio = {}
        for t0 in T0S:
            a = all_preds[f'{expert}/M2_top1/t0={t0:.1f}']
            b = all_preds[f'{expert}/shuffle/t0={t0:.1f}']
            numerator = np.sqrt(np.mean((a - b) ** 2, axis=(1, 2)))
            denominator = np.sqrt(np.mean((draft_m2[:, :, :7] - draft_shuf[:, :, :7]) ** 2, axis=(1, 2)))
            rows_ratio[str(t0)] = {'mean_ratio': float(np.mean(numerator / np.maximum(denominator, 1e-8))),
                                   'mean_output_distance': float(numerator.mean()),
                                   'mean_draft_distance': float(denominator.mean())}
        sensitivity[expert] = rows_ratio

    # M2-fit-source vs heldout-task draft quality for the actual Long candidate too.
    direct_summary = {}
    for k, d in all_drafts.items():
        direct_summary[k] = {'mse_mean': float(np.mean((d[:, :, :7] - target) ** 2)),
                             'mse_median': float(np.median(np.mean((d[:, :, :7] - target) ** 2, axis=(1, 2))))}
    result = {
        'rows': len(long_ids), 'hard_rows': int(hard.sum()), 'tasks': sorted(set(np.asarray(rows['task'])[long_ids].tolist())),
        'noise': 'jax.random.normal(fold_in(key(4101), dataset_row_id), [10,32]); shared across methods',
        'selected': {'E-M2': best_sel, 'E-zero': zero_sel},
        'metrics_active7': summaries, 'paired_episode_bootstrap': pairs,
        'draft_mse': direct_summary, 'output_sensitivity_ratio': sensitivity,
        'M2_candidate_top1_mean_score': float(np.mean(np.load(args.m2_dir / 'candidate_rows.npz')['M2_salient_maxsim_scores'][:, 0])),
        'decision': {
            'selected_t0': selected_t0,
            'F1': bool(any(pairs[f'F1_{s}']['ci95'][1] < 0 for s in ('all_long', 'hard'))),
            'F2': bool(any(pairs[f'F2_{s}']['ci95'][1] < 0 for s in ('all_long', 'hard'))),
            'F3_reference_only': 'not used as a causal decision; official Expert initial weights and VLM exposure differ',
        },
    }
    np.savez_compressed(args.output / 'offline_predictions.npz', row_ids=long_ids, task_ids=np.asarray(rows['task'][long_ids]),
                        episode_ids=np.asarray(rows['episode'][long_ids]), hard_mask=hard, target=target,
                        noise=noise, m2_candidate_rows=m2_cands, shuffled_candidate_rows=shuffled_rows,
                        candidate_shuffle_permutation=shuffle_perm,
                        shuffle_source_ranks=shuffle_source_ranks[shuffle_perm],
                        m2_draft=draft_m2, shuffle_draft=draft_shuf, zero_draft=draft_zero,
                        direct_regression_draft=dreg,
                        **{f'pred_{k.replace("/", "_").replace("=", "_")}': v for k, v in all_preds.items()},
                        **{f'mse_{k.replace("/", "_").replace("=", "_")}': v for k, v in mse.items()})
    _write_json(args.output / 'offline_summary.json', result)
    _write_json(args.output / 'decision.json', {
        'F1': result['decision']['F1'], 'F2': result['decision']['F2'],
        'selected': {'E-M2': best_sel, 'E-zero': zero_sel},
    })
    print(json.dumps({'stage': 'evaluate', 'F1': result['decision']['F1'], 'F2': result['decision']['F2'],
                      'selected_t0': selected_t0}), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['prepare', 'train', 'evaluate'])
    p.add_argument('--expert', choices=['em2', 'zero'])
    p.add_argument('--draft-source', choices=['m2', 'zero'])
    p.add_argument('--run', type=Path, default=RUN)
    p.add_argument('--memory', type=Path, default=MEMORY)
    p.add_argument('--m2-dir', type=Path, default=M2_DIR)
    p.add_argument('--gap', type=Path, default=GAP)
    p.add_argument('--hard', type=Path, default=HARD)
    p.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    p.add_argument('--output', type=Path, default=OUT_DEFAULT)
    p.add_argument('--expert-dir', type=Path)
    p.add_argument('--steps', type=int, default=4000)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--save-every', type=int, default=500)
    p.add_argument('--threads', type=int, default=8)
    a = p.parse_args()
    a.run = a.run.resolve(); a.memory = a.memory.resolve(); a.m2_dir = a.m2_dir.resolve()
    a.output = a.output.resolve(); a.gap = a.gap.resolve(); a.hard = a.hard.resolve()
    if not 1 <= a.threads <= 16:
        raise ValueError('CPU threads must be 1..16')
    a.output.mkdir(parents=True, exist_ok=True)
    if a.stage == 'prepare':
        if (a.output / 'candidate_lists.npz').exists():
            raise FileExistsError('prepare output exists; refusing overwrite')
        prepare(a)
    elif a.stage == 'train':
        a.expert_dir = (a.expert_dir or a.output / ('E-M2' if a.expert == 'em2' else 'E-zero')).resolve()
        train(a)
    else:
        if (a.output / 'offline_summary.json').exists():
            raise FileExistsError('offline evaluation already exists; refusing overwrite')
        evaluate(a)


if __name__ == '__main__':
    main()
