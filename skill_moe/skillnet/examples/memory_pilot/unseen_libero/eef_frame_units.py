#!/usr/bin/env python3
"""EEF-frame memory diagnostic (EEF1; discontinued by user decision, no results).

The transform is deliberately explicit: a unit is expressed in the EEF frame at
its *source start*, and a retrieved unit is returned to the base frame using
the query-start pose. Histories are transformed using the same query pose.

Run the small geometry smoke and tests with:
  cd /work && source env/activate.sh && taskset -c 8-47 nice -n 10 env \
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false python skill_moe/skillnet/examples/memory_pilot/unseen_libero/eef_frame_units.py smoke \
      --out /work/.skillnet-state/outputs/memory-pilot/unseen-libero/eef-frame-01-smoke-<UTC> \
      --limit 8 --train-limit 128 --epochs 1 --seeds 4101

Full cached Dev/Eval production command (LIBERO-40 memory-out remains marked unsupported):
  cd /work && source env/activate.sh && taskset -c 8-47 nice -n 10 env \
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false timeout 48h \
    python skill_moe/skillnet/examples/memory_pilot/unseen_libero/eef_frame_units.py production \
      --out /work/.skillnet-state/outputs/memory-pilot/unseen-libero/eef-frame-01-production-<UTC>
Resume a matching interrupted run by adding `--resume` to that command.

`production` runs the full cached Dev/Eval experiment. `--limit` and
`--epochs` are intended only for the same-code-path smoke; they change data
volume, not the implementation route. Each trained model is checkpointed and
can be resumed by rerunning with the identical arguments and `--resume`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero')
SCALE = ROOT / 'scale-only-02' / 'production'
OUT = ROOT / 'eef-frame-01'


def axis_angle_matrix(rotvec):
    """Rodrigues rotation matrix for [...,3] axis-angle vectors."""
    v = np.asarray(rotvec, dtype=np.float64)
    theta = np.linalg.norm(v, axis=-1)
    axis = v / np.maximum(theta[..., None], 1e-12)
    x, y, z = np.moveaxis(axis, -1, 0)
    k = np.zeros(v.shape[:-1] + (3, 3), np.float64)
    k[..., 0, 1] = -z; k[..., 0, 2] = y
    k[..., 1, 0] = z; k[..., 1, 2] = -x
    k[..., 2, 0] = -y; k[..., 2, 1] = x
    eye = np.broadcast_to(np.eye(3), k.shape)
    r = eye + np.sin(theta)[..., None, None] * k + (1 - np.cos(theta))[..., None, None] * (k @ k)
    return np.where((theta < 1e-10)[..., None, None], eye, r).astype(np.float32)


def rotate_six(x, rotation):
    """Apply row-vector R to position and rotation deltas [...,T,6]."""
    a = np.asarray(x, np.float32)
    r = np.asarray(rotation, np.float32)
    def apply(v):
        if v.ndim == r.ndim - 1:
            return np.einsum('...j,...ji->...i', v, r)
        return np.einsum('...tj,...ji->...ti', v, r)
    return np.concatenate((apply(a[..., :3]), apply(a[..., 3:6])), axis=-1)


def to_eef_physical(actions, source_rotvec):
    """Apply R0^T to column-vector deltas (row arrays multiply by R0)."""
    r = axis_angle_matrix(source_rotvec)
    return rotate_six(actions, r)


def from_eef_physical(actions_eef, query_rotvec):
    """Apply Rq to column-vector deltas (row arrays multiply by Rq^T)."""
    return rotate_six(actions_eef, np.swapaxes(axis_angle_matrix(query_rotvec), -1, -2))


def normalized_state_rotvec(state_norm, stats):
    """Decode policy-normalized state xyz+axis-angle using official quantile stats."""
    s = np.asarray(state_norm, np.float32)[..., :6]
    q01 = np.asarray(stats['q01'], np.float32)[..., :6]
    q99 = np.asarray(stats['q99'], np.float32)[..., :6]
    physical = (s + 1.) * .5 * (q99 - q01 + 1e-6) + q01
    return physical[..., 3:6]


def to_eef_normalized(actions_norm, state_norm, action_stats, state_stats):
    """Transform normalized actions, preserving policy normalization and gripper."""
    a = np.asarray(actions_norm, np.float32).copy()
    q01 = np.asarray(action_stats['q01'], np.float32)[..., :6]
    q99 = np.asarray(action_stats['q99'], np.float32)[..., :6]
    p = (a[..., :6] + 1.) * .5 * (q99 - q01 + 1e-6) + q01
    p = to_eef_physical(p, normalized_state_rotvec(state_norm, state_stats))
    a[..., :6] = (p - q01) / (q99 - q01 + 1e-6) * 2. - 1.
    return a


def to_eef_normalized_at_rotvec(actions_norm, rotvec, action_stats):
    """Transform normalized actions using an exact per-frame physical EEF pose."""
    a = np.asarray(actions_norm, np.float32).copy()
    q01 = np.asarray(action_stats['q01'], np.float32)[..., :6]
    q99 = np.asarray(action_stats['q99'], np.float32)[..., :6]
    p = (a[..., :6] + 1.) * .5 * (q99 - q01 + 1e-6) + q01
    p = to_eef_physical(p, rotvec)
    a[..., :6] = (p - q01) / (q99 - q01 + 1e-6) * 2. - 1.
    return a


def from_eef_normalized(actions_norm, state_norm, action_stats, state_stats):
    a = np.asarray(actions_norm, np.float32).copy()
    q01 = np.asarray(action_stats['q01'], np.float32)[..., :6]
    q99 = np.asarray(action_stats['q99'], np.float32)[..., :6]
    p = (a[..., :6] + 1.) * .5 * (q99 - q01 + 1e-6) + q01
    p = from_eef_physical(p, normalized_state_rotvec(state_norm, state_stats))
    a[..., :6] = (p - q01) / (q99 - q01 + 1e-6) * 2. - 1.
    return a


def low_frequency_fraction(actions, k=3):
    """Fraction of six-axis DCT-II energy in the first k temporal coefficients."""
    from scipy.fft import dct
    x = np.asarray(actions, np.float64)
    c = dct(x[..., :6], type=2, norm='ortho', axis=-2)
    en = c * c
    return en[..., :k, :].sum(axis=(-2, -1)) / np.maximum(en.sum(axis=(-2, -1)), 1e-15)


def _cluster_separation_ratio(phase, clusters, seed):
    from sklearn.cluster import MiniBatchKMeans
    x = np.asarray(phase, np.float32).reshape(len(phase), -1)
    k = min(int(clusters), len(x))
    model = MiniBatchKMeans(n_clusters=k, random_state=int(seed),
                            batch_size=min(1024, max(64, k * 4)), n_init=3,
                            max_iter=100, reassignment_ratio=.01)
    labels = model.fit_predict(x)
    centers = model.cluster_centers_
    if len(centers) > 1:
        delta = centers[:, None, :] - centers[None, :, :]
        between = float(np.mean(np.sum(delta * delta, axis=-1)[np.triu_indices(len(centers), 1)]))
    else:
        between = 0.
    residual = x - centers[labels]
    within = float(np.mean(np.sum(residual * residual, axis=1)))
    return {'k_requested': int(clusters), 'k_used': int(k),
            'between_center_sqdist': between, 'within_sqdist': within,
            'ratio': float(between / max(within, 1e-20)),
            'occupied_clusters': int(len(np.unique(labels)))}


def d1_summary(pool_actions, pool_task, *, counts=(32, 95, 256), seed=20261114):
    """CMP2-D type-cluster separation and full-pool cross-task nearest share.

    ``counts`` are the requested number of clusters, not sample sizes.  Base and EEF
    summaries use every unit from the same pool, so the sample count is matched.
    """
    from scale_only_query import type_of
    x = np.asarray(pool_actions, np.float32)
    task = np.asarray(pool_task)
    if len(x) != len(task) or not len(x):
        raise ValueError(f'pool action/task mismatch: {len(x)} vs {len(task)}')
    key = type_of(x, 'S-shared').reshape(len(x), -1)
    typed = key.reshape(len(x), x.shape[1], x.shape[2])
    src = np.linspace(0., 1., typed.shape[1]); dst = np.linspace(0., 1., 32)
    phase = np.stack([np.stack([np.interp(dst, src, a[:, j]) for j in range(a.shape[1])], -1)
                      for a in typed])
    cluster = {str(int(k)): _cluster_separation_ratio(phase, int(k), seed)
               for k in counts}

    # Exact nearest neighbours over the full pool, in blocks to avoid an N x N matrix.
    sq = np.sum(key * key, axis=1)
    nearest_d = np.full(len(key), np.inf, np.float64)
    nearest_i = np.full(len(key), -1, np.int64)
    block = 256
    for start in range(0, len(key), block):
        end = min(start + block, len(key))
        d = np.maximum(sq[start:end, None] + sq[None, :] - 2 * key[start:end] @ key.T, 0.)
        d[np.arange(end - start), np.arange(start, end)] = np.inf
        nearest_i[start:end] = d.argmin(axis=1)
        nearest_d[start:end] = d.min(axis=1)
    return {'unit_count': int(len(x)), 'cluster_separation': cluster,
            'within_nearest_squared_mean': float(nearest_d.mean()),
            'nearest_cross_task_fraction': float(np.mean(task[nearest_i] != task))}


def eef_runtime_action(unit_eef_norm, query_state_norm, action_stats, state_stats):
    """Closed-loop hook: return the stored EEF unit in the query's base frame."""
    return from_eef_normalized(unit_eef_norm, query_state_norm, action_stats, state_stats)


def _u_source_rotations(pool, rows_dir, cd, state_stats):
    """Read exact U unit start poses from unsegmented LeRobot demos.

    Anchor cuts are generally off the stride-10 sampling grid. Reading the state from
    the row containing a cut uses a stale pose, so use the exact source frame.
    """
    import json
    import pyarrow.parquet as pq
    manifest = json.loads((Path(cd.RUN) / 'data_manifest.json').read_text())
    dataset = Path(manifest['source'])
    pairs = list(zip(np.asarray(pool['task'], np.int64), np.asarray(pool['episode'], np.int64),
                     np.asarray(pool['start'], np.int64), strict=True))
    by_episode = {}
    out = np.empty((len(pairs), 3), np.float32)
    for i, (task, episode, start) in enumerate(pairs):
        episode, task, start = int(episode), int(task), int(start)
        if episode not in by_episode:
            path = dataset / 'data' / f'chunk-{episode // 1000:03d}' / f'episode_{episode:06d}.parquet'
            table = pq.read_table(path, columns=['state', 'frame_index', 'task_index'])
            frame_index = np.asarray(table['frame_index'].to_numpy(), np.int64)
            states = np.asarray(table['state'].to_pylist(), np.float32)
            task_index = np.asarray(table['task_index'].to_numpy(), np.int64)
            if not np.array_equal(frame_index, np.arange(len(frame_index))):
                raise ValueError(f'non-contiguous source frames in {path}')
            by_episode[episode] = (states, task_index)
        states, task_index = by_episode[episode]
        if start < 0 or start >= len(states):
            raise IndexError(f'U unit start {(task, episode, start)} outside source episode')
        if int(task_index[start]) != task:
            raise ValueError(f'U unit task mismatch at {(task, episode, start)}: {task_index[start]}')
        out[i] = states[start, 3:6]
    return out, pairs


def _unseen_segment_rotvecs(inputs, unseen_root):
    """Read exact query-start EEF axis-angle from LIBERO-90 HDF5 demos."""
    import json
    import h5py
    root = Path(unseen_root)
    manifest = json.loads((root / 'manifest.json').read_text())
    records = {(int(r['task_id']), int(r['episode'])): r for r in manifest['source_records']}
    opened = {}
    out = np.empty((len(inputs['task']), 3), np.float32)
    try:
        for i, (task, episode, frame) in enumerate(zip(inputs['task'], inputs['episode'], inputs['start'], strict=True)):
            key = (int(task), int(episode))
            rec = records[key]
            path = str(rec['source_hdf5'])
            if path not in opened:
                opened[path] = h5py.File(path, 'r')
            demo = opened[path]['data'][rec['source_demo']]
            raw_frame = 2 * int(frame)  # unseen rows sample every other HDF5 frame
            ori = np.asarray(demo['obs']['ee_ori'][raw_frame], np.float32)
            if ori.shape != (3,):
                raise ValueError(f'expected axis-angle [3] at {key}/{frame}, got {ori.shape}')
            out[i] = ori
    finally:
        for f in opened.values():
            f.close()
    return out


def transform_u_pool(pool, rotations, sq):
    """Transform decoded U units in their source-start EEF frames and re-encode DCT3."""
    out = dict(pool)
    out['phys16'] = np.stack([to_eef_physical(x, r) for x, r in zip(pool['phys16'], rotations, strict=True)])
    out['values'] = np.stack([sq.dct3_values(x) for x in out['phys16']]).astype(np.float32)
    # Values are the DCT-3 representation consumed by the S-shared P5 keys.
    out['canon_values'] = out['values'].copy()
    return out


def _segment_errors(preds, targets):
    """Six-axis MSE per segment, holding the final prediction over uncovered frames."""
    out = np.zeros(len(preds), np.float64)
    first5 = np.zeros_like(out); cont = np.zeros_like(out)
    for i, (p, y) in enumerate(zip(preds, targets, strict=True)):
        ext = p[np.minimum(np.arange(len(y)), len(p) - 1)]
        out[i] = sq_six_axis(ext, y)
        first5[i] = sq_six_axis(ext[:min(5, len(y))], y[:5])
        if len(y) > 5:
            cont[i] = sq_six_axis(ext[5:min(10, len(y))], y[5:10])
        else:
            cont[i] = np.nan
    return {'full': out, 'first5': first5, 'continuation5_9': cont}


def sq_six_axis(pred, target):
    # Match scale_only_query.six_axis: normalized squared error sum over six axes / 7.
    return float(((np.asarray(pred)[..., :6] - np.asarray(target)[..., :6]) ** 2).sum(-1).mean() / 7.)


def _write_once(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f'write-once output already exists: {path}')
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(payload)
    tmp.replace(path)


def run_experiment(out=OUT, *, limit=32, train_limit=1536, epochs=2,
                   seeds=None, resume=False):
    """Shared bounded/full offline implementation; ``None`` limits mean full data."""
    import scale_only_query as sq
    seeds = tuple(sq.SEEDS if seeds is None else seeds)
    row_slice = slice(None) if limit is None else slice(0, int(limit))
    def bounded(d): return {k: np.asarray(v)[row_slice] for k, v in d.items()}
    out = Path(out)
    if out.exists() and any((out / name).exists() for name in ('results.json', 'report.md')):
        raise FileExistsError(f'write-once final output already exists: {out}')
    out.mkdir(parents=True, exist_ok=True)
    fracs = (1 / 3, 1 / 10, 1 / 30)
    epoch_by_pool = ({'W': 20, 'U': 40} if epochs is None else {'W': int(epochs), 'U': int(epochs)})
    manifest = {'limit': limit, 'train_limit': train_limit, 'epochs': epoch_by_pool,
                'seeds': list(seeds), 'fracs': list(fracs), 'route': 'run_experiment'}
    manifest_path = out / 'run_manifest.json'
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if not resume or old != manifest:
            raise ValueError('existing run requires --resume with exactly matching run parameters')
    else:
        if resume and any(out.iterdir()):
            raise ValueError('--resume requested but no matching run_manifest.json exists')
        _write_once(manifest_path, json.dumps(manifest, indent=2) + '\n')
    pool = dict(np.load(SCALE / 'prep/W_pool.npz', allow_pickle=False))
    inp = dict(np.load(SCALE / 'prep/W_inputs_eval.npz', allow_pickle=False))
    p0 = __import__('theta_regression').candidate_shape_pool(rep='C_vm_none')
    hold = tuple(sq.HOLDOUT)
    keep = ~np.isin(p0['source_task'], hold)
    rows_path = sq._mods()[0].RUN / 'rows'
    states = np.load(rows_path / 'state.npy', mmap_mode='r', allow_pickle=False)
    # Source orientation belongs to each candidate's source row (not its target query).
    source_rot = np.asarray(states[p0['source_row'][keep], 3:6], np.float32)
    # State arrays in the frozen memory cache are policy-normalized; decode using official stats.
    from openpi.shared.normalize import load
    stat_path = Path('/work/.skillnet-state/checkpoints/openpi-assets/checkpoints/pi05_libero/assets/physical-intelligence/libero')
    stats = load(stat_path)
    # The source states here use the same normalized state convention as the policy rows.
    state_stats = {'q01': stats['state'].q01, 'q99': stats['state'].q99}
    # D1 compares W types under independent random unit-frame changes.
    base_d1 = d1_summary(pool['phys'], pool['task'])
    pool_rot = normalized_state_rotvec(states[p0['source_row'][keep]], state_stats)
    rel = np.stack([to_eef_physical(a, r) for a, r in zip(pool['phys'], pool_rot, strict=True)])
    eef_d1 = d1_summary(rel, pool['task'])
    rng = np.random.default_rng(441)
    random_rot = rng.normal(size=(len(pool['phys']), 3))
    random_rot *= (rng.uniform(0, np.pi, len(random_rot)) / np.maximum(np.linalg.norm(random_rot, axis=1), 1e-12))[:, None]
    rel_random = np.stack([to_eef_physical(a, r) for a, r in zip(pool['phys'], random_rot, strict=True)])
    lf_delta = float(np.max(np.abs(low_frequency_fraction(pool['phys']) - low_frequency_fraction(rel_random))))
    # D2/D3 W smoke: transform source units and query histories with their own
    # source/query poses, fit the same small P5 model on a bounded train subset,
    # and score only the first 32 Eval windows.
    cd, vs, kq = sq._mods()
    action_stats = stats['actions']
    # Stats objects expose quantile arrays; convert to the lightweight mapping API.
    action_stats = {'q01': action_stats.q01, 'q99': action_stats.q99}
    pool_eef = dict(pool)
    pool_eef['phys'] = np.stack([to_eef_physical(a, r) for a, r in zip(pool['phys'], pool_rot, strict=True)])
    pool_eef['keep'] = np.asarray(pool['keep'], bool)
    tr_rows = np.load(rows_path / 'actions.npy', mmap_mode='r', allow_pickle=False)
    tr_states = np.load(rows_path / 'state.npy', mmap_mode='r', allow_pickle=False)
    tr_pool_rows = np.load(rows_path / 'task.npy', mmap_mode='r', allow_pickle=False)
    train_ids_all = np.flatnonzero(~np.asarray(np.load(rows_path / 'validation.npy', mmap_mode='r', allow_pickle=False), bool))
    train_cache_all = dict(np.load(SCALE / 'prep/W_inputs_train.npz', allow_pickle=False))
    for field, source in (('task', tr_pool_rows),
                          ('episode', np.load(rows_path / 'episode.npy', mmap_mode='r', allow_pickle=False)),
                          ('frame', np.load(rows_path / 'frame.npy', mmap_mode='r', allow_pickle=False))):
        if not np.array_equal(train_cache_all[field], np.asarray(source)[train_ids_all]):
            raise AssertionError(f'W training cache {field} order does not match run-01 rows')
    n_train = len(train_ids_all) if train_limit is None else min(int(train_limit), len(train_ids_all))
    train_sub_ix = np.linspace(0, len(train_ids_all) - 1, n_train).astype(np.int64)
    train_ids = train_ids_all[train_sub_ix]
    train_q = vs.to_physical(np.asarray(tr_rows[train_ids, :, :6], np.float32))
    train_rot = normalized_state_rotvec(tr_states[train_ids], state_stats)
    train_q = np.stack([to_eef_physical(a, r) for a, r in zip(train_q, train_rot, strict=True)])
    # Base and EEF use the identical deterministic subset and cached row order.
    train_input_base = {k: v[train_sub_ix] for k, v in train_cache_all.items()}
    train_input = {k: v.copy() for k, v in train_input_base.items()}
    train_input['hist'] = to_eef_normalized(train_input['hist'], train_input['state'], action_stats, state_stats)
    train_input['task'] = np.asarray(tr_pool_rows)[train_ids]
    train_pool_task = np.asarray(pool['task'])
    train_E = sq.type_errors(train_q, pool_eef['phys'], 'S-shared')
    train_A = sq.allowed_matrix(train_input['task'], train_pool_task, pool_eef['keep'], True)
    holdout = np.isin(train_input['task'], sq.HOLDOUT)
    dev_in = bounded(inp)
    unseen = kq.UNSEEN
    unseen_actions = np.load(unseen / 'actions.npy', mmap_mode='r', allow_pickle=False)
    eval_y_base = np.asarray(unseen_actions[inp['row'][row_slice], :, :7], np.float32)
    eval_rot = normalized_state_rotvec(dev_in['state'], state_stats)
    eval_q_eef = np.stack([to_eef_physical(vs.to_physical(a[:, :6]), r)
                           for a, r in zip(eval_y_base, eval_rot, strict=True)])
    eval_target_norm = np.stack([vs.to_normalized(a) for a in eval_q_eef]).astype(np.float32)
    dev_ids = np.load(SCALE / 'prep/W_inputs_dev.npz', allow_pickle=False)['row'][row_slice]
    dev_all = dict(np.load(SCALE / 'prep/W_inputs_dev.npz', allow_pickle=False))
    dev_in = bounded(dev_all)
    dev_input_base = {k: v.copy() for k, v in dev_in.items()}
    dev_actions = np.asarray(unseen_actions[dev_ids, :, :6], np.float32)
    dev_q = np.stack([to_eef_physical(vs.to_physical(a), r) for a, r in zip(dev_actions,
                      normalized_state_rotvec(dev_in['state'], state_stats), strict=True)])
    dev_E = sq.type_errors(dev_q, pool_eef['phys'], 'S-shared')
    dev_A = sq.allowed_matrix(dev_in['task'], pool_eef['task'], pool_eef['keep'], False)
    dev_hist = to_eef_normalized(dev_in['hist'], dev_in['state'], action_stats, state_stats)
    dev_in['hist'] = dev_hist
    Xdev = sq.query_inputs(dev_in, 'P5')
    train_X = sq.query_inputs(train_input, 'P5')
    train_K = sq.key_inputs('W', pool_eef, 'S-shared')
    train_data = {'X': train_X, 'KI': train_K, 'E': train_E, 'A': train_A,
                  'fit': np.flatnonzero(~holdout), 'va': np.flatnonzero(holdout),
                  'Xdev': Xdev, 'Adev': dev_A, 'Edev': dev_E}
    train_q_base = vs.to_physical(np.asarray(tr_rows[train_ids, :, :6], np.float32))
    train_E_base = sq.type_errors(train_q_base, pool['phys'], 'S-shared')
    train_A_base = sq.allowed_matrix(train_input_base['task'], pool['task'], pool['keep'], True)
    dev_q_base = np.stack([vs.to_physical(a) for a in dev_actions])
    dev_E_base = sq.type_errors(dev_q_base, pool['phys'], 'S-shared')
    dev_A_base = sq.allowed_matrix(dev_in['task'], pool['task'], pool['keep'], False)
    train_data_base = {'X': sq.query_inputs(train_input_base, 'P5'), 'KI': sq.key_inputs('W', pool, 'S-shared'),
                       'E': train_E_base, 'A': train_A_base,
                       'fit': np.flatnonzero(~holdout), 'va': np.flatnonzero(holdout),
                       'Xdev': sq.query_inputs(dev_input_base, 'P5'), 'Adev': dev_A_base, 'Edev': dev_E_base}
    # Do not silently train on an empty validation partition if the bounded
    # deterministic subset happens not to contain a held-out task.
    if not len(train_data['va']) or not len(train_data['fit']):
        raise RuntimeError('smoke training subset must contain both fit and holdout tasks')
    import torch
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '8')))
    eval_in = bounded(dict(np.load(SCALE / 'prep/W_inputs_eval.npz', allow_pickle=False)))
    eval_in['hist'] = to_eef_normalized(eval_in['hist'], eval_in['state'], action_stats, state_stats)
    eval_in['pi05'] = to_eef_normalized(eval_in['pi05'], eval_in['state'], action_stats, state_stats)
    eval_A = sq.allowed_matrix(eval_in['task'], pool_eef['task'], pool_eef['keep'], False)
    eval_E = sq.type_errors(eval_q_eef, pool_eef['phys'], 'S-shared')
    eef_scores = {}
    runtime_model_state = None
    # Oracle type upper bound for D2. Only type selection uses Eval labels.
    oracle_pick_eef = np.where(eval_A, eval_E, np.inf).argmin(1)
    pred_oracle_eef = sq.generate_w(pool_eef, oracle_pick_eef, 'S-shared', 'stored', eval_in)
    d2_eef = float(((pred_oracle_eef[:, :, :6] - eval_target_norm) ** 2).mean() * 6 / 7)
    eval_base_in = bounded(dict(np.load(SCALE / 'prep/W_inputs_eval.npz', allow_pickle=False)))
    eval_base_q = np.stack([vs.to_physical(a[:, :6]) for a in eval_y_base])
    base_E = sq.type_errors(eval_base_q, pool['phys'], 'S-shared')
    base_A = sq.allowed_matrix(eval_base_in['task'], pool['task'], pool['keep'], False)
    oracle_pick_base = np.where(base_A, base_E, np.inf).argmin(1)
    target_base_norm = eval_y_base[:, :, :6]
    pred_oracle_base = sq.generate_w(pool, oracle_pick_base, 'S-shared', 'stored', eval_base_in)
    d2_base = float(((pred_oracle_base[:, :, :6] - target_base_norm) ** 2).mean() * 6 / 7)
    d2w_base_vec = sq.six_axis(pred_oracle_base, target_base_norm)
    d2w_eef_vec = sq.six_axis(pred_oracle_eef, eval_target_norm)
    base_scores = {}; base_w_vectors = {}; eef_w_vectors = {}; dev_w_vectors = {'base': {}, 'eef': {}}
    dev_target_norm = np.stack([vs.to_normalized(a) for a in dev_q]).astype(np.float32)
    dev_rot = normalized_state_rotvec(bounded(dev_all)['state'], state_stats)
    dev_in_eef = dict(dev_all)
    dev_in_eef = bounded(dev_in_eef)
    dev_in_eef['hist'] = to_eef_normalized(dev_in_eef['hist'], dev_in_eef['state'], action_stats, state_stats)
    dev_in_eef['pi05'] = to_eef_normalized(dev_in_eef['pi05'], dev_in_eef['state'], action_stats, state_stats)
    def train_cached(pool_name, coordinate, seed, data):
        """Atomic model checkpoint so an interrupted run can resume at model granularity."""
        import torch
        results = []
        for frac in fracs:
            path = out / 'models' / f'{pool_name}_P5_S-shared_{coordinate}_s{seed}_f{frac:.4f}.pt'
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                if not resume: raise FileExistsError(f'checkpoint exists; pass --resume: {path}')
                payload = torch.load(path, map_location='cpu', weights_only=False)
                if payload.get('manifest') != manifest: raise ValueError(f'checkpoint configuration mismatch: {path}')
                result = payload['result']
            else:
                result = sq.train_one(pool_name, 'P5', 'S-shared', seed, frac, data,
                                      epochs=epoch_by_pool[pool_name], device='cpu')
                tmp = path.with_suffix('.tmp')
                torch.save({'manifest': manifest, 'result': result}, tmp)
                tmp.replace(path)
            results.append(result)
        # Same selection rule as scale-only-02: each seed selects the lowest Dev
        # type error across the three teacher temperatures; Eval is never used.
        selected = min(results, key=lambda r: (r['dev']['value'], r['frac']))
        return selected

    trained_models = {}
    for seed in seeds:
        trained_base = train_cached('W', 'base', seed, train_data_base)
        trained_models[('W', 'base', seed)] = trained_base
        net_base = sq.make_net('W', 'P5', train_data_base['X'].shape[1], train_data_base['KI'].shape[1])
        net_base.load_state_dict(trained_base['dev']['state']); net_base.eval()
        with torch.no_grad():
            base_pick = net_base(torch.as_tensor(sq.query_inputs(eval_base_in, 'P5')),
                                 torch.as_tensor(train_data_base['KI'])).masked_fill(~torch.as_tensor(base_A), -1e9).argmax(-1).numpy()
            base_pick_dev = net_base(torch.as_tensor(train_data_base['Xdev']),
                                     torch.as_tensor(train_data_base['KI'])).masked_fill(~torch.as_tensor(dev_A_base), -1e9).argmax(-1).numpy()
        base_scores[str(seed)] = {}
        for rule in ('stored', 'hist', 'pi05'):
            base_pred = sq.generate_w(pool, base_pick, 'S-shared', rule, eval_base_in)
            base_scores[str(seed)][f'D3_{rule}_first5'] = float(((base_pred[:, :5, :6] - target_base_norm[:, :5]) ** 2).mean() * 6 / 7)
            base_scores[str(seed)][f'D3_{rule}_continuation5_9'] = float(((base_pred[:, 5:10, :6] - target_base_norm[:, 5:10]) ** 2).mean() * 6 / 7)
            base_w_vectors.setdefault(rule, {})[str(seed)] = sq.six_axis(base_pred, target_base_norm)
            dev_base_pred = sq.generate_w(pool, base_pick_dev, 'S-shared', rule, dev_input_base)
            dev_w_vectors['base'].setdefault(rule, {})[str(seed)] = sq.six_axis(dev_base_pred, np.stack([vs.to_normalized(vs.to_physical(y[:, :6])) for y in np.load(kq.UNSEEN / 'actions.npy', mmap_mode='r', allow_pickle=False)[dev_ids]]))
        trained = train_cached('W', 'eef', seed, train_data)
        trained_models[('W', 'eef', seed)] = trained
        net = sq.make_net('W', 'P5', train_X.shape[1], train_K.shape[1])
        net.load_state_dict(trained['dev']['state']); net.eval()
        if seed == seeds[0]: runtime_model_state = trained['dev']['state']
        with torch.no_grad():
            Xeval = torch.as_tensor(sq.query_inputs(eval_in, 'P5'))
            pick_eef = net(Xeval,
                           torch.as_tensor(train_K)).masked_fill(~torch.as_tensor(eval_A), -1e9).argmax(-1).numpy()
            pick_dev = net(torch.as_tensor(Xdev), torch.as_tensor(train_K)).masked_fill(~torch.as_tensor(dev_A), -1e9).argmax(-1).numpy()
        eef_scores[str(seed)] = {}
        for rule in ('stored', 'hist', 'pi05'):
            pred = sq.generate_w(pool_eef, pick_eef, 'S-shared', rule, eval_in)
            eef_scores[str(seed)][f'D3_{rule}_first5'] = float(((pred[:, :5, :6] - eval_target_norm[:, :5]) ** 2).mean() * 6 / 7)
            eef_scores[str(seed)][f'D3_{rule}_continuation5_9'] = float(((pred[:, 5:10, :6] - eval_target_norm[:, 5:10]) ** 2).mean() * 6 / 7)
            eef_w_vectors.setdefault(rule, {})[str(seed)] = sq.six_axis(pred, eval_target_norm)
            pred_dev_eef = sq.generate_w(pool_eef, pick_dev, 'S-shared', rule, dev_in_eef)
            dev_w_vectors['eef'].setdefault(rule, {})[str(seed)] = sq.six_axis(pred_dev_eef, dev_target_norm)
        eef_scores[str(seed)]['D3_dev_top1_type_error'] = float(dev_E[np.arange(len(dev_E)), pick_dev].mean())
    # --------------------------- U pool: D1, upper bound, and short P5 smoke.
    u_pool = dict(np.load(SCALE / 'prep/U_pool.npz', allow_pickle=False))
    u_rot, u_source_rows = _u_source_rotations(u_pool, rows_path, cd, state_stats)
    u_eef = transform_u_pool(u_pool, u_rot, sq)
    u_d1 = {'base': d1_summary(u_pool['phys16'][u_pool['keep']], u_pool['task'][u_pool['keep']]),
            'eef': d1_summary(u_eef['phys16'][u_eef['keep']], u_eef['task'][u_eef['keep']])}
    # Build aligned variable-length targets only for the smoke subset.
    _, mem_rows, _, splits = kq._production_data_splits(include_eval_targets=False)
    train_ids_all = np.asarray(splits['train'][1], np.int64)
    train_u_in, train_u_tgt = sq.build_segments(mem_rows['actions'], np.asarray(mem_rows['task']),
        np.asarray(mem_rows['episode']), np.asarray(mem_rows['frame']), mem_rows['state'], train_ids_all)
    cached_u_train = dict(np.load(SCALE / 'prep/U_inputs_train.npz', allow_pickle=False))
    if not (np.array_equal(train_u_in['start'], cached_u_train['start']) and np.array_equal(train_u_in['task'], cached_u_train['task'])):
        raise AssertionError('U training smoke rows do not match scale-only-02')
    u_train_n = len(train_u_in['task']) if train_limit is None else min(int(train_limit), len(train_u_in['task']))
    u_train_ix = np.linspace(0, len(train_u_in['task']) - 1, u_train_n).astype(np.int64)
    u_train_base = {k: np.asarray(v)[u_train_ix].copy() for k, v in train_u_in.items()}
    u_train = {k: v.copy() for k, v in u_train_base.items()}
    u_train_q_base = train_u_tgt['target16'][u_train_ix]
    u_train_q = u_train_q_base.copy()
    u_train_rot, _ = _u_source_rotations(u_train_base, rows_path, cd, state_stats)
    u_train_q = np.stack([to_eef_physical(q, r) for q, r in zip(u_train_q, u_train_rot, strict=True)])
    u_train['hist'] = to_eef_normalized_at_rotvec(u_train['hist'], u_train_rot, action_stats)
    u_train['past'] = to_eef_normalized_at_rotvec(u_train['past'], u_train_rot, action_stats)
    u_train['last'] = to_eef_normalized_at_rotvec(u_train['last'], u_train_rot, action_stats)
    # LIBERO-90 Dev/Eval segment targets are rebuilt from corrected unseen rows.
    unseen = kq.UNSEEN
    unseen_arrays = {k: np.load(unseen / f'{k}.npy', mmap_mode='r', allow_pickle=False)
                     for k in ('actions', 'task', 'episode', 'frame', 'state')}
    eval_w_all = dict(np.load(SCALE / 'prep/W_inputs_eval.npz', allow_pickle=False))
    eval_row_ids = np.asarray(eval_w_all['row'], np.int64)
    eval_u_in_built, eval_u_t = sq.build_segments(unseen_arrays['actions'], unseen_arrays['task'], unseen_arrays['episode'],
        unseen_arrays['frame'], unseen_arrays['state'], eval_row_ids, hard_by_row={int(r): bool(h) for r, h in zip(eval_w_all['row'], eval_w_all['hard'])})
    eval_u_in = dict(np.load(SCALE / 'prep/U_inputs_eval.npz', allow_pickle=False))
    if not (np.array_equal(eval_u_in_built['start'], eval_u_in['start']) and np.array_equal(eval_u_in_built['episode'], eval_u_in['episode'])):
        raise AssertionError('rebuilt U Eval windows differ from scale-only-02')
    # Dev targets already belong to the fixed scale-only-02 Dev split.
    dev_u_in = dict(np.load(SCALE / 'prep/U_inputs_dev.npz', allow_pickle=False))
    dev_u_t = dict(np.load(SCALE / 'prep/U_dev_targets.npz', allow_pickle=False))
    dev_u_ix = np.arange(len(dev_u_in['task']) if limit is None else min(int(limit), len(dev_u_in['task'])))
    eval_u_ix = np.arange(len(eval_u_in['task']) if limit is None else min(int(limit), len(eval_u_in['task'])))
    eval_u = {k: np.asarray(v)[eval_u_ix] for k, v in eval_u_in.items()}
    dev_u = {k: np.asarray(v)[dev_u_ix] for k, v in dev_u_in.items()}
    eval_u_q16 = eval_u_t['target16'][eval_u_ix]
    eval_u_rot = _unseen_segment_rotvecs(eval_u, unseen)
    eval_u_q16_eef = np.stack([to_eef_physical(q, r) for q, r in zip(eval_u_q16, eval_u_rot, strict=True)])
    eval_u_E = sq.type_errors(eval_u_q16_eef, u_eef['phys16'], 'S-shared')
    eval_u_A = sq.allowed_matrix(eval_u['task'], u_eef['task'], u_eef['keep'], False)
    eval_u_pick = np.where(eval_u_A, eval_u_E, np.inf).argmin(1)
    eval_u_tgt_base = [sq._unflatten(eval_u_t['target_flat'], eval_u_t['target_off'], int(i)) for i in eval_u_ix]
    eval_u_tgt_eef = [to_eef_normalized_at_rotvec(t, eval_u_rot[i], action_stats) for i, t in enumerate(eval_u_tgt_base)]
    eval_u_in_eef = dict(eval_u)
    for key in ('hist', 'past', 'last', 'pi05_now'):
        if key in eval_u_in_eef:
            eval_u_in_eef[key] = to_eef_normalized_at_rotvec(eval_u_in_eef[key], eval_u_rot, action_stats)
    d2u_base_pred = sq.generate_u(u_pool, eval_u_pick, 'S-shared', 'stored', eval_u)
    d2u_eef_pick = np.where(eval_u_A, eval_u_E, np.inf).argmin(1)
    d2u_eef_pred = sq.generate_u(u_eef, d2u_eef_pick, 'S-shared', 'stored', eval_u_in_eef)
    d2u_base = _segment_errors(d2u_base_pred, eval_u_tgt_base)['full']
    d2u_eef = _segment_errors(d2u_eef_pred, eval_u_tgt_eef)['full']
    d2u_oracle_base_pred = sq.oracle_u(u_pool, eval_u_pick, 'S-shared', eval_u_tgt_base)
    d2u_oracle_eef_pred = sq.oracle_u(u_eef, d2u_eef_pick, 'S-shared', eval_u_tgt_eef)
    d2u_oracle_base = _segment_errors(d2u_oracle_base_pred, eval_u_tgt_base)['full']
    d2u_oracle_eef = _segment_errors(d2u_oracle_eef_pred, eval_u_tgt_eef)['full']
    dev_u_tgt16 = dev_u_t['target16'][dev_u_ix]
    dev_u_rot = _unseen_segment_rotvecs(dev_u, unseen)
    dev_u_q16_eef = np.stack([to_eef_physical(q, r) for q, r in zip(dev_u_tgt16, dev_u_rot, strict=True)])
    dev_u_E = sq.type_errors(dev_u_q16_eef, u_eef['phys16'], 'S-shared')
    dev_u_A = sq.allowed_matrix(dev_u['task'], u_eef['task'], u_eef['keep'], False)
    dev_u_q_base = dev_u_tgt16
    dev_u_E_base = sq.type_errors(dev_u_q_base, u_pool['phys16'], 'S-shared')
    dev_u_A_base = sq.allowed_matrix(dev_u['task'], u_pool['task'], u_pool['keep'], False)
    dev_u_eef = dict(dev_u)
    for key in ('hist', 'past', 'last', 'pi05_now'):
        if key in dev_u_eef:
            dev_u_eef[key] = to_eef_normalized_at_rotvec(dev_u_eef[key], dev_u_rot, action_stats)
    u_keys = sq.key_inputs('U', u_eef, 'S-shared')
    u_keys_base = sq.key_inputs('U', u_pool, 'S-shared')
    u_X_base = sq.query_inputs(u_train_base, 'P5')
    u_X = sq.query_inputs(u_train, 'P5')
    u_Xdev_base = sq.query_inputs(dev_u, 'P5')
    u_Xdev = sq.query_inputs(dev_u_eef, 'P5')
    u_E_base = sq.type_errors(u_train_q_base, u_pool['phys16'], 'S-shared')
    u_E = sq.type_errors(u_train_q, u_eef['phys16'], 'S-shared')
    u_A_base = sq.allowed_matrix(u_train_base['task'], u_pool['task'], u_pool['keep'], True)
    u_A = sq.allowed_matrix(u_train['task'], u_eef['task'], u_eef['keep'], True)
    u_hold = np.isin(u_train['task'], sq.HOLDOUT)
    u_data_base = {'X': u_X_base, 'KI': u_keys_base, 'E': u_E_base, 'A': u_A_base,
                   'fit': np.flatnonzero(~u_hold), 'va': np.flatnonzero(u_hold),
                   'Xdev': u_Xdev_base, 'Adev': dev_u_A_base, 'Edev': dev_u_E_base}
    u_data = {'X': u_X, 'KI': u_keys, 'E': u_E, 'A': u_A,
              'fit': np.flatnonzero(~u_hold), 'va': np.flatnonzero(u_hold),
              'Xdev': u_Xdev, 'Adev': dev_u_A, 'Edev': dev_u_E}
    if not len(u_data['fit']) or not len(u_data['va']):
        raise RuntimeError('U smoke training subset must contain fit and holdout tasks')
    u_d3 = {'base': {}, 'eef': {}}
    eval_u_in_base = eval_u
    for seed in seeds:
        trained_u_base = train_cached('U', 'base', seed, u_data_base)
        trained_models[('U', 'base', seed)] = trained_u_base
        net_u_base = sq.make_net('U', 'P5', u_X_base.shape[1], u_keys_base.shape[1])
        net_u_base.load_state_dict(trained_u_base['dev']['state']); net_u_base.eval()
        with torch.no_grad():
            base_pick = net_u_base(torch.as_tensor(sq.query_inputs(eval_u_in_base, 'P5')),
                                   torch.as_tensor(u_keys_base)).masked_fill(~torch.as_tensor(eval_u_A), -1e9).argmax(-1).numpy()
        u_d3['base'][str(seed)] = {}
        for rule in ('stored', 'hist', 'pi05'):
            base_preds = sq.generate_u(u_pool, base_pick, 'S-shared', rule, eval_u_in_base)
            u_d3['base'][str(seed)][rule] = _segment_errors(base_preds, eval_u_tgt_base)
        trained_u = train_cached('U', 'eef', seed, u_data)
        trained_models[('U', 'eef', seed)] = trained_u
        net_u = sq.make_net('U', 'P5', u_X.shape[1], u_keys.shape[1]); net_u.load_state_dict(trained_u['dev']['state']); net_u.eval()
        with torch.no_grad():
            pu = net_u(torch.as_tensor(sq.query_inputs(eval_u_in_eef, 'P5')), torch.as_tensor(u_keys)).masked_fill(~torch.as_tensor(eval_u_A), -1e9).argmax(-1).numpy()
        u_d3['eef'][str(seed)] = {}
        for rule in ('stored', 'hist', 'pi05'):
            pred_u = sq.generate_u(u_eef, pu, 'S-shared', rule, eval_u_in_eef)
            u_d3['eef'][str(seed)][rule] = _segment_errors(pred_u, eval_u_tgt_eef)

    # LIBERO-40 memory-out: the 8 holdout tasks are excluded from gradient
    # updates and the candidate pool, but their demonstration rows/segments are
    # scored separately from LIBERO-90 Dev/Eval.
    mem_ix = np.flatnonzero(np.isin(train_cache_all['task'], sq.HOLDOUT))
    mem_raw = train_ids_all[mem_ix]
    mem_w_base = {k: np.asarray(v)[mem_ix].copy() for k, v in train_cache_all.items()}
    mem_y_base = np.asarray(tr_rows[mem_raw, :, :7], np.float32)
    mem_rot = normalized_state_rotvec(tr_states[mem_raw], state_stats)
    mem_q_base = vs.to_physical(mem_y_base[:, :, :6]).astype(np.float32)
    mem_q_eef = np.stack([to_eef_physical(y, r) for y, r in zip(mem_q_base, mem_rot, strict=True)])
    mem_y_eef = mem_y_base.copy()
    mem_y_eef[:, :, :6] = np.stack([vs.to_normalized(y) for y in mem_q_eef])
    mem_w_eef = {k: v.copy() for k, v in mem_w_base.items()}
    mem_w_eef['hist'] = to_eef_normalized(mem_w_eef['hist'], mem_w_eef['state'], action_stats, state_stats)
    mem_Aw = sq.allowed_matrix(mem_w_base['task'], pool['task'], pool['keep'], True)
    mem_w_result = {'rows': int(len(mem_ix)), 'base': {'D2': {}, 'D3': {}}, 'eef': {'D2': {}, 'D3': {}}}
    mem_w_ep = np.asarray(mem_w_base['task'], np.int64) * 100000 + np.asarray(mem_w_base['episode'], np.int64)
    mem_w_hard = np.asarray(mem_w_base['hard'], bool)
    for coord, q, y, d, candidate in (
        ('base', mem_q_base, mem_y_base, mem_w_base, pool),
        ('eef', mem_q_eef, mem_y_eef, mem_w_eef, pool_eef),
    ):
        E = sq.type_errors(q, candidate['phys'], 'S-shared')
        pick = np.where(mem_Aw, E, np.inf).argmin(1)
        pred = sq.generate_w(candidate, pick, 'S-shared', 'stored', d)
        err = sq.six_axis(pred, y)
        mem_w_result[coord]['D2'] = {'stored': sq.boot(err, mem_w_ep),
                                     'hard_mean': float(err[mem_w_hard].mean()) if mem_w_hard.any() else None}
    for seed in seeds:
        for coord, candidate, data, d, y in (
            ('base', pool, train_data_base, mem_w_base, mem_y_base),
            ('eef', pool_eef, train_data, mem_w_eef, mem_y_eef),
        ):
            model = sq.make_net('W', 'P5', data['X'].shape[1], data['KI'].shape[1])
            model.load_state_dict(trained_models[('W', coord, seed)]['dev']['state']); model.eval()
            with torch.no_grad():
                pick = model(torch.as_tensor(sq.query_inputs(d, 'P5')), torch.as_tensor(data['KI']))\
                    .masked_fill(~torch.as_tensor(mem_Aw), -1e9).argmax(-1).numpy()
            mem_w_result[coord]['D3'][str(seed)] = {}
            for rule in ('stored', 'hist', 'pi05'):
                pred = sq.generate_w(candidate, pick, 'S-shared', rule, d)
                err = sq.six_axis(pred, y)
                mem_w_result[coord]['D3'][str(seed)][rule] = {
                    'mean': float(err.mean()), 'ci95': sq.boot(err, mem_w_ep)['ci95'],
                    'hard_mean': float(err[mem_w_hard].mean()) if mem_w_hard.any() else None}

    mem_u_ix = np.flatnonzero(np.isin(train_u_in['task'], sq.HOLDOUT))
    mem_u = {k: np.asarray(v)[mem_u_ix].copy() for k, v in train_u_in.items()}
    mem_u_ybase = [np.asarray(train_u_tgt['target_norm'][i], np.float32) for i in mem_u_ix]
    mem_u_qbase = np.asarray(train_u_tgt['target16'], np.float32)[mem_u_ix]
    mem_u_rot, _ = _u_source_rotations(mem_u, rows_path, cd, state_stats)
    mem_u_qeef = np.stack([to_eef_physical(q, r) for q, r in zip(mem_u_qbase, mem_u_rot, strict=True)])
    mem_u_yeef = [to_eef_normalized_at_rotvec(y, mem_u_rot[i], action_stats)
                  for i, y in enumerate(mem_u_ybase)]
    mem_u_eef = {k: v.copy() for k, v in mem_u.items()}
    for key in ('hist', 'past', 'last', 'pi05_now'):
        if key in mem_u_eef: mem_u_eef[key] = to_eef_normalized_at_rotvec(mem_u_eef[key], mem_u_rot, action_stats)
    mem_Au = sq.allowed_matrix(mem_u['task'], u_pool['task'], u_pool['keep'], True)
    mem_u_ep = np.asarray(mem_u['task'], np.int64) * 100000 + np.asarray(mem_u['episode'], np.int64)
    mem_u_hard = np.asarray(mem_u['hard'], bool)
    mem_u_result = {'segments': int(len(mem_u_ix)), 'base': {'D2': {}, 'D3': {}}, 'eef': {'D2': {}, 'D3': {}}}
    for coord, q, ys, candidate, d in (
        ('base', mem_u_qbase, mem_u_ybase, u_pool, mem_u),
        ('eef', mem_u_qeef, mem_u_yeef, u_eef, mem_u_eef),
    ):
        E = sq.type_errors(q, candidate['phys16'], 'S-shared')
        pick = np.where(mem_Au, E, np.inf).argmin(1)
        pred = sq.generate_u(candidate, pick, 'S-shared', 'stored', d)
        err = _segment_errors(pred, ys)
        mem_u_result[coord]['D2'] = {k: sq.boot(v, mem_u_ep) for k, v in err.items()}
    for seed in seeds:
        for coord, candidate, data, d, ys in (
            ('base', u_pool, u_data_base, mem_u, mem_u_ybase),
            ('eef', u_eef, u_data, mem_u_eef, mem_u_yeef),
        ):
            model = sq.make_net('U', 'P5', data['X'].shape[1], data['KI'].shape[1])
            model.load_state_dict(trained_models[('U', coord, seed)]['dev']['state']); model.eval()
            with torch.no_grad():
                pick = model(torch.as_tensor(sq.query_inputs(d, 'P5')), torch.as_tensor(data['KI']))\
                    .masked_fill(~torch.as_tensor(mem_Au), -1e9).argmax(-1).numpy()
            mem_u_result[coord]['D3'][str(seed)] = {}
            for rule in ('stored', 'hist', 'pi05'):
                pred = sq.generate_u(candidate, pick, 'S-shared', rule, d)
                err = _segment_errors(pred, ys)
                mem_u_result[coord]['D3'][str(seed)][rule] = {
                    k: sq.boot(v[np.isfinite(v)], mem_u_ep[np.isfinite(v)]) for k, v in err.items()}

    ep_u = np.asarray(eval_u['task'], np.int64) * 100000 + np.asarray(eval_u['episode'], np.int64)
    hard_u = np.asarray(eval_u['hard'], bool)
    def boot_metric(values, mask=None):
        values = np.asarray(values, np.float64)
        keep = np.isfinite(values) if mask is None else (np.asarray(mask, bool) & np.isfinite(values))
        if not keep.any(): return {'mean': None, 'ci95': None, 'n': 0}
        b = sq.boot(values[keep], ep_u[keep])
        return {'mean': float(values[keep].mean()), 'ci95': b['ci95'], 'n': int(keep.sum())}
    def paired_boot_arrays(left, right, episodes):
        left, right = np.asarray(left, np.float64), np.asarray(right, np.float64)
        valid = np.isfinite(left) & np.isfinite(right)
        if not valid.any(): return {'mean': None, 'ci95': None, 'episodes': 0, 'rows': 0, 'reps': 5000}
        return sq.boot_diff(left[valid], right[valid], np.asarray(episodes)[valid])
    u_boot = {'D2_stored_base': boot_metric(d2u_base), 'D2_stored_eef': boot_metric(d2u_eef),
              'D2_oracle_scale_base': boot_metric(d2u_oracle_base), 'D2_oracle_scale_eef': boot_metric(d2u_oracle_eef),
              'D2_base_vs_eef_difference': sq.boot_diff(d2u_eef, d2u_base, ep_u),
              'hard_base': boot_metric(d2u_base, hard_u), 'hard_eef': boot_metric(d2u_eef, hard_u)}
    u_d3_summary = {'base': {}, 'eef': {}, 'paired_eef_minus_base': {}}
    for seed in seeds:
        for rule in ('stored', 'hist', 'pi05'):
            b = u_d3['base'][str(seed)][rule]; e = u_d3['eef'][str(seed)][rule]
            u_d3_summary['base'].setdefault(rule, {})[str(seed)] = {k: boot_metric(v) for k, v in b.items()}
            u_d3_summary['eef'].setdefault(rule, {})[str(seed)] = {k: boot_metric(v) for k, v in e.items()}
            u_d3_summary['paired_eef_minus_base'].setdefault(rule, {})[str(seed)] = {
                k: paired_boot_arrays(e[k], b[k], ep_u) for k in ('full', 'first5', 'continuation5_9')}
    # Read exact scale-only-02 frozen reference, kept as reproduction anchor.
    source_results = json.loads((SCALE / 'results.json').read_text())
    # Compare against the persisted per-query arrays, not the compact results
    # table: the latter's list currently mixes an aggregate with seed rows.
    # Keep both values in the review artifact so that this upstream reporting
    # inconsistency remains visible without changing scale-only-02 outputs.
    stored_table = source_results['tables']['W|eval']['S-shared|P5|holdout|stored']
    eval_rows = dict(np.load(SCALE / 'evaluation/rows.npz', allow_pickle=False))
    # Independently reproduce the frozen base P5 reference from its saved models.
    # The matched two-epoch smoke models above are used for the paired base/EEF D3
    # comparison; this full cached path is only the scale-only-02 reproduction check.
    ref_in = dict(np.load(SCALE / 'prep/W_inputs_eval.npz', allow_pickle=False))
    ref_target = np.asarray(unseen_actions[ref_in['row'], :, :6], np.float32)
    ref_hard = np.asarray(ref_in['hard'], bool)
    recomputed_all, recomputed_hard, saved_all, saved_hard = [], [], [], []
    # The reproduction anchor is the published three-seed scale-only-02
    # aggregate, even when this invocation trains only one smoke seed.
    for seed in sq.SEEDS:
        ref_pick = sq.score_models(SCALE, 'W', 'S-shared', 'P5', seed, 'eval')['holdout']
        ref_pred = sq.generate_w(pool, ref_pick, 'S-shared', 'stored', ref_in)
        ref_err = sq.six_axis(ref_pred, ref_target)
        recomputed_all.append(float(ref_err.mean()))
        recomputed_hard.append(float(ref_err[ref_hard].mean()))
        key = f'W|eval||W|S-shared|P5|s{seed}|holdout|stored'
        if key not in eval_rows:
            raise AssertionError(f'missing frozen per-query scale-only-02 result: {key}')
        row_err = np.asarray(eval_rows[key], np.float64)
        saved_all.append(float(row_err.mean()))
        saved_hard.append(float(row_err[ref_hard].mean()))
    if not np.allclose(recomputed_all, saved_all, atol=1e-6, rtol=0) or not np.allclose(recomputed_hard, saved_hard, atol=1e-6, rtol=0):
        raise AssertionError(f'base scale-only-02 model reproduction mismatch: {recomputed_all}/{recomputed_hard} vs saved rows {saved_all}/{saved_hard}')
    # The result table's first entry is the mean and subsequent entries are
    # inconsistent with the persisted per-query arrays. Flag rather than
    # silently treating those entries as three seed results.
    table_consistent = (np.allclose(stored_table['all'], saved_all, atol=1e-6, rtol=0)
                        and np.allclose(stored_table['hard'], saved_hard, atol=1e-6, rtol=0))
    # P5 parity anchor is the independently reproduced per-query mean.
    base_anchor = {'all': float(np.mean(recomputed_all)), 'hard': float(np.mean(recomputed_hard)),
                   'recomputed_all_by_seed': recomputed_all, 'recomputed_hard_by_seed': recomputed_hard,
                   'saved_rows_all_by_seed': saved_all, 'saved_rows_hard_by_seed': saved_hard,
                   'source_results_table_consistent': bool(table_consistent),
                   'source_results_table_all': stored_table['all'], 'source_results_table_hard': stored_table['hard'],
                   'expected_all': .11078, 'expected_hard': .22600}
    if abs(base_anchor['all'] - base_anchor['expected_all']) > .002 or abs(base_anchor['hard'] - base_anchor['expected_hard']) > .002:
        raise AssertionError(f'base scale-only-02 anchor mismatch: {base_anchor}')
    meta = json.loads((SCALE / 'prep/prep.json').read_text())
    w_ep = np.asarray(eval_base_in['task'], np.int64) * 100000 + np.asarray(eval_base_in['episode'], np.int64)
    dev_ep = np.asarray(dev_in_eef['task'], np.int64) * 100000 + np.asarray(dev_in_eef['episode'], np.int64)
    w_boot = {'D2_oracle_stored_base': sq.boot(d2w_base_vec, w_ep),
              'D2_oracle_stored_eef': sq.boot(d2w_eef_vec, w_ep),
              'D2_eef_minus_base': paired_boot_arrays(d2w_eef_vec, d2w_base_vec, w_ep), 'D3': {}}
    for rule in ('stored', 'hist', 'pi05'):
        w_boot['D3'][rule] = {}
        for seed in seeds:
            w_boot['D3'][rule][str(seed)] = paired_boot_arrays(eef_w_vectors[rule][str(seed)], base_w_vectors[rule][str(seed)], w_ep)
    dev_gate = {}
    for rule in ('stored', 'hist', 'pi05'):
        dev_gate[rule] = {}
        for seed in seeds:
            dev_gate[rule][str(seed)] = paired_boot_arrays(dev_w_vectors['eef'][rule][str(seed)], dev_w_vectors['base'][rule][str(seed)], dev_ep)
    dev_gate['pass'] = any(all(v['ci95'][1] < 0 for v in dev_gate[rule].values()) for rule in ('stored', 'hist', 'pi05'))
    dev_rule = min(('stored', 'hist', 'pi05'),
                   key=lambda r: (float(np.mean([np.mean(dev_w_vectors['eef'][r][str(s)]) for s in seeds])), r))
    model_dir = out / 'models'; model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f'eef_w_p5_seed{seeds[0]}.pt'
    if model_path.exists():
        if not resume: raise FileExistsError(f'model exists; use --resume: {model_path}')
        prior = torch.load(model_path, map_location='cpu', weights_only=False)
        if prior.get('manifest') != manifest: raise ValueError(f'runtime checkpoint configuration mismatch: {model_path}')
    else:
        tmp_model = model_path.with_suffix('.tmp')
        torch.save({'manifest': manifest, 'state': runtime_model_state, 'seed': seeds[0], 'rule': dev_rule,
                    'selection': 'lowest mean Dev D3 six-axis MSE across all Dev rows',
                    'pool_phys_eef': pool_eef['phys'], 'pool_phys_base': pool['phys'],
                    'pool_task': pool_eef['task'], 'pool_keep': pool_eef['keep'],
                    'key': train_K, 'query_dim': train_X.shape[1], 'key_dim': train_K.shape[1]}, tmp_model)
        tmp_model.replace(model_path)
    result = {'status': 'offline_complete', 'run_manifest': manifest,
              'pool': 'W=1,595 and U=3,504 kept S-shared units; source split excludes 8 held-out LIBERO-40 tasks',
              'eval_rows_total': meta['W_eval'], 'eval_rows_scored': int(len(eval_in['task'])),
              'smoke_eval_unique_tasks': int(len(np.unique(eval_in['task']))),
              'smoke_eval_unique_episodes': int(len(set(zip(eval_in['task'].tolist(), eval_in['episode'].tolist())))),
              'D1_base': base_d1, 'D1_eef': eef_d1, 'D1_note': 'nearest-neighbour compactness diagnostic uses matched random subsets; score scale is normalized physical 10x6 delta.',
              'D2_oracle_type_stored_scale_mse': {'base': d2_base, 'eef': d2_eef},
              'D3_W_smoke': {'base': base_scores, 'eef': eef_scores},
              'W_episode_bootstrap': w_boot, 'W_dev_gate_smoke': dev_gate, 'W_dev_selected_amplitude': dev_rule,
              'D1_U_smoke': u_d1,
              'D2_U_smoke': {'stored_scale': {'base': boot_metric(d2u_base), 'eef': boot_metric(d2u_eef)},
                             'oracle_scale_reference': {'base': boot_metric(d2u_oracle_base), 'eef': boot_metric(d2u_oracle_eef)},
                             'paired_eef_minus_base': u_boot['D2_base_vs_eef_difference'],
                             'hard': {'base': u_boot['hard_base'], 'eef': u_boot['hard_eef']}},
              'D3_U_smoke': u_d3_summary,
              'U_eval_windows_total': meta['U_eval'], 'U_eval_smoke_windows': int(len(eval_u_ix)),
              'U_smoke_unique_tasks': int(len(np.unique(eval_u['task']))),
              'U_smoke_unique_episodes': int(len(set(zip(eval_u['task'].tolist(), eval_u['episode'].tolist())))),
              'U_episode_bootstrap': u_boot,
              'low_frequency_max_abs_change_random_rotations': lf_delta,
              'base_W_P5_stored_scale_only_02_reference': base_anchor,
              'reproduced_from_frozen_scale_only_02_rows': True,
              'scale_only_02_compact_table_consistent_with_rows': bool(base_anchor['source_results_table_consistent']),
              'memory_out_LIBERO40': {'status': 'complete', 'W': mem_w_result, 'U': mem_u_result,
                                      'holdout_tasks': list(sq.HOLDOUT)},
              'deviation_note': 'Memory-out rows are the LIBERO-40 holdout-task rows withheld from gradient updates; teacher-temperature selection still follows the same holdout/Dev procedure as scale-only-02.',
              'closed_loop': {'status': 'not_run', 'reason': 'runtime one-trial and server parity remain outstanding'} }
    _write_once(out / 'results.json', json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    report = ['# EEF1 オフライン実験', '',
              'D1: CMP2-D と同じ32位相表現の between-center / within-cluster 比（k=32/95/256）と、全プールでの最近傍別タスク率。base/EEF は同じ単位数・同じ行の単位を使用。',
              '| プール | k | 単位数 | base 比 | EEF 比 | base 最近傍別タスク率 | EEF 最近傍別タスク率 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for pool_name, base_d, eef_d in (('W', base_d1, eef_d1), ('U', u_d1['base'], u_d1['eef'])):
        for k in ('32', '95', '256'):
            report.append(f"| {pool_name} | {k} | {base_d['unit_count']} | {base_d['cluster_separation'][k]['ratio']:.4f} | {eef_d['cluster_separation'][k]['ratio']:.4f} | {base_d['nearest_cross_task_fraction']:.4f} | {eef_d['nearest_cross_task_fraction']:.4f} |")
    report += [
              f"D2/D3: 正規化6軸MSE（6軸二乗誤差を7次元平均スケールへ換算）、先頭5/続き5–9。Eval窓={len(eval_in['task'])}、Dev窓={len(dev_in_eef['task'])}。", '',
              '| D2 oracle type・保存スケール | base | EEF |', '|---|---:|---:|',
              f'| 6軸MSE | {d2_base:.4f} | {d2_eef:.4f} |', '',
              '| D3 P5 保存スケール | base first5 / cont. | EEF first5 / cont. |', '|---|---:|---:|']
    for seed in seeds:
        b, e = base_scores[str(seed)], eef_scores[str(seed)]
        report.append(f"| {seed} | {b['D3_stored_first5']:.4f} / {b['D3_stored_continuation5_9']:.4f} | {e['D3_stored_first5']:.4f} / {e['D3_stored_continuation5_9']:.4f} |")
    report += ['', f"scale-only-02 全 Eval 参照: W/P5 保存スケール all={base_anchor['all']:.5f} (期待 .111), hard={base_anchor['hard']:.5f} (期待 .226)。保存モデルから独立再計算し、per-query rows と照合。results.json 集約表との一致={base_anchor['source_results_table_consistent']}。低周波率の回転差={lf_delta:.2g}。", '',
              f"U: プール3,504単位。Eval 全{meta['U_eval']}区間中{len(eval_u_ix)}区間（{result['U_smoke_unique_tasks']}タスク、{result['U_smoke_unique_episodes']}エピソード）。D2 base/EEF 保存スケール平均={u_boot['D2_stored_base']['mean']:.4f}/{u_boot['D2_stored_eef']['mean']:.4f}、paired 95% CI={u_boot['D2_base_vs_eef_difference']['ci95']}。", '',
              'LIBERO-40 memory-out の評価は未実装。関門はDevだけで選択し、Evalは報告用。', '']
    _write_once(out / 'report.md', '\n'.join(report))
    return result


class EEFMemorySelector:
    """Runtime adapter consumed by memory_closed_loop.make_policy without editing it."""
    def __init__(self):
        import torch
        import scale_only_query as sq
        import variable_shape_memory as vs
        from openpi.shared.normalize import load
        model_path = Path(os.environ['EEF_SMOKE_MODEL'])
        ck = torch.load(model_path, map_location='cpu', weights_only=False)
        torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '8')))
        self.torch, self.vs = torch, vs
        self.pool = {'phys': np.asarray(ck['pool_phys_eef'], np.float32),
                     'task': np.asarray(ck['pool_task'], np.int64),
                     'keep': np.asarray(ck['pool_keep'], bool)}
        self.pool_base = np.asarray(ck['pool_phys_base'], np.float32)
        self.key = torch.as_tensor(np.asarray(ck['key'], np.float32))
        self.net = sq.make_net('W', 'P5', int(ck['query_dim']), int(ck['key_dim']))
        self.net.load_state_dict(ck['state']); self.net.eval()
        self.rule = ck['rule']
        stats_path = Path('/work/.skillnet-state/checkpoints/openpi-assets/checkpoints/pi05_libero/assets/physical-intelligence/libero')
        stats = load(stats_path)
        self.action_stats = {'q01': stats['actions'].q01, 'q99': stats['actions'].q99}
        self.state_stats = {'q01': stats['state'].q01, 'q99': stats['state'].q99}
        self.shapes = np.concatenate((vs.to_normalized(self.pool_base), np.zeros((len(self.pool_base), 10, 1), np.float32)), -1)
        self.last_scores = None
        self._history = self._state = None

    def pick(self, history, state, exclude_task=None):
        import scale_only_query as sq
        h = np.asarray(history, np.float32); s = np.asarray(state, np.float32)
        he = to_eef_normalized(h, s, self.action_stats, self.state_stats)
        x = sq.query_inputs({'hist': he, 'state': s}, 'P5')
        with self.torch.no_grad():
            score = self.net(self.torch.as_tensor(x), self.key)
            allowed = np.broadcast_to(self.pool['keep'], (len(s), len(self.pool['keep']))).copy()
            if exclude_task is not None: allowed &= self.pool['task'][None] != int(exclude_task)
            score = score.masked_fill(~self.torch.as_tensor(allowed), -1e9)
            self.last_scores = score.numpy()
            pick = self.last_scores.argmax(-1)
        self._history, self._state = h, s
        return pick

    def pose_to(self, idx, ref_norm):
        import scale_only_query as sq
        if self._state is None or self._history is None: raise RuntimeError('pick must run before pose_to')
        pi = np.asarray(ref_norm, np.float32)
        pi_eef = to_eef_normalized(pi, self._state, self.action_stats, self.state_stats)
        hist_eef = to_eef_normalized(self._history, self._state, self.action_stats, self.state_stats)
        inp = {'hist': hist_eef, 'state': self._state, 'pi05': pi_eef,
               'has_hist': np.abs(self._history).sum((1, 2)) > 0}
        chosen = sq.generate_w(self.pool, np.asarray(idx), 'S-shared', self.rule, inp)
        return from_eef_normalized(chosen, self._state, self.action_stats, self.state_stats)

    @staticmethod
    def confidence(scores, temperature=.05):
        v = np.asarray(scores)[np.asarray(scores) > -1e8]
        z = (v - v.max()) / temperature; p = np.exp(z); p /= p.sum()
        top = np.sort(v)[::-1]
        return {'p_top1': float(p.max()), 'margin': float(top[0] - top[1]),
                'eff_candidates': float(np.exp(-(p * np.log(p + 1e-30)).sum())), 'cos_top1': float(top[0])}


def make_eef_policy(checkpoint):
    import memory_closed_loop as legacy
    # Dynamic class substitution affects only the new server process.
    legacy.MemorySelector = EEFMemorySelector
    return legacy.make_policy(checkpoint)


def serve(a):
    import torch
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '8')))
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    policy = make_eef_policy(a.checkpoint)
    WebsocketPolicyServer(policy, host='127.0.0.1', port=a.port, metadata={'policy': 'eef_frame_memory_smoke'}).serve_forever()


def _runtime_parity(model_path):
    """Compare offline drafting with the selector used by the websocket policy on 10 Dev rows."""
    import scale_only_query as sq
    import torch
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '8')))
    os.environ['EEF_SMOKE_MODEL'] = str(model_path)
    selector = EEFMemorySelector()
    from openpi.shared.normalize import load
    st = load(Path('/work/.skillnet-state/checkpoints/openpi-assets/checkpoints/pi05_libero/assets/physical-intelligence/libero'))
    action_stats = {'q01': st['actions'].q01, 'q99': st['actions'].q99}
    state_stats = {'q01': st['state'].q01, 'q99': st['state'].q99}
    data = dict(np.load(SCALE / 'prep/W_inputs_dev.npz', allow_pickle=False))
    ids = np.arange(min(10, len(data['task'])))
    h, s, pi = data['hist'][ids], data['state'][ids], data['pi05'][ids]
    picks = selector.pick(h, s)
    runtime = selector.pose_to(picks, pi)
    pi_eef = to_eef_normalized(pi, s, action_stats, state_stats)
    h_eef = to_eef_normalized(h, s, action_stats, state_stats)
    inp = {'hist': h_eef, 'state': s, 'pi05': pi_eef, 'has_hist': np.abs(h).sum((1, 2)) > 0}
    offline_eef = sq.generate_w(selector.pool, picks, 'S-shared', selector.rule, inp)
    offline = from_eef_normalized(offline_eef, s, action_stats, state_stats)
    maxdiff = float(np.max(np.abs(runtime[..., :6] - offline[..., :6])))
    if maxdiff > 1e-5: raise AssertionError(f'websocket/offline draft mismatch {maxdiff:.3g} > 1e-5')
    return {'rows': len(ids), 'six_axis_max_abs_difference': maxdiff, 'limit': 1e-5}


def closed_loop_smoke(out, *, checkpoint=Path('/work/.skillnet-state/checkpoints/openpi-assets/checkpoints/pi05_libero')):
    """Run one real LIBERO-90 trial with the EEF selector and verify offline parity first."""
    import subprocess
    import time
    import run as base
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    model_path = out / 'models/eef_w_p5_seed4101.pt'
    if not model_path.exists(): raise FileNotFoundError(f'run all/smoke first; missing {model_path}')
    parity = _runtime_parity(model_path)
    plan = json.loads((ROOT / 'baseline-01/plan.json').read_text())
    task_id = int(plan['selected_tasks'][0]['task_id'])
    port = base.free_port()
    log_path = out / 'eef_policy_server.log'
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8',
               XLA_PYTHON_CLIENT_PREALLOCATE='false', XLA_PYTHON_CLIENT_MEM_FRACTION='.15',
               EEF_SMOKE_MODEL=str(model_path), MEM_T0='0')
    with log_path.open('w') as log:
        proc = subprocess.Popen(['taskset', '-c', '8-47', 'nice', '-n', '10', sys.executable,
                                str(Path(__file__).resolve()), 'serve', '--checkpoint', str(checkpoint), '--port', str(port)],
                               env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            base.ready(proc, port)
            job = {'host': '127.0.0.1', 'port': port, 'suite': 'libero_90', 'task_id': task_id,
                   'trial_index': 0, 'seed_base': 7, 'condition': 'memory', 'memory_snapshot': 'none',
                   'query_text_source': 'instruction_only', 'checkpoint': str(checkpoint), 'wait_steps': 10,
                   'replan_steps': 5, 'max_steps': 520, 'record': False}
            attempt = out / 'closed_loop' / f'task{task_id:02d}-init00'
            attempt.parent.mkdir(parents=True, exist_ok=True)
            result = _one_trial(job, attempt, timeout=1800)
        finally:
            base.terminate(proc)
    payload = {'parity': parity, 'trial': result, 'task_id': task_id, 'init_id': 0,
               'rule': torch_rule(model_path), 'gpu_fraction': .15}
    _write_once(out / 'closed_loop_smoke.json', json.dumps(payload, indent=2, ensure_ascii=False) + '\n')
    return payload


def torch_rule(model_path):
    import torch
    return torch.load(model_path, map_location='cpu', weights_only=False)['rule']


def _one_trial(job, folder, *, timeout):
    import subprocess
    import time
    import run as base
    folder.mkdir(parents=True, exist_ok=False)
    job_path, result_path = folder / 'job.json', folder / 'result.json'
    job_path.write_text(json.dumps(job))
    env = os.environ.copy(); env.update(OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8', XLA_PYTHON_CLIENT_PREALLOCATE='false')
    log_path = folder / 'worker.log'
    with log_path.open('w') as log:
        proc = subprocess.Popen(['taskset', '-c', '8-47', 'nice', '-n', '10', sys.executable,
                                 str(Path(__file__).resolve().parents[1] / 'unseen_libero/worker.py'), str(job_path), str(result_path)],
                                env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try: proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            base.terminate(proc)
            return {'status': 'timeout', 'success': False, 'error': f'trial exceeded {timeout}s'}
    if proc.returncode != 0 or not result_path.exists():
        return {'status': 'error', 'success': False, 'returncode': proc.returncode,
                'worker_log': str(log_path)}
    return json.loads(result_path.read_text())


def smoke(out=OUT):
    """Small-data alias for the same run_experiment implementation used by production."""
    return run_experiment(out, limit=32, train_limit=1536, epochs=2, resume=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('smoke', 'production', 'closed_loop_smoke', 'serve'))
    p.add_argument('--out', type=Path, default=ROOT / 'eef-frame-01-production-exp-20261114T120000Z')
    p.add_argument('--checkpoint', type=Path, default=Path('/work/.skillnet-state/checkpoints/openpi-assets/checkpoints/pi05_libero'))
    p.add_argument('--port', type=int)
    p.add_argument('--limit', type=int, default=None, help='max Dev/Eval rows per split; production default is all')
    p.add_argument('--train-limit', type=int, default=None, help='max fit examples per pool; production default is all')
    p.add_argument('--epochs', type=int, default=None, help='override full settings W=20/U=40; use only for smoke')
    p.add_argument('--seeds', type=str, default=None, help='comma-separated subset; production default is scale-only-02 3 seeds')
    p.add_argument('--resume', action='store_true', help='resume matching run from atomic per-model checkpoints')
    a = p.parse_args()
    if a.command == 'serve':
        if a.port is None: p.error('serve requires --port')
        serve(a)
    elif a.command == 'smoke':
        limit = 32 if a.limit is None else a.limit
        train_limit = 1536 if a.train_limit is None else a.train_limit
        epochs = 2 if a.epochs is None else a.epochs
        seeds = tuple(int(v) for v in a.seeds.split(',')) if a.seeds else (4101,)
        print(json.dumps(run_experiment(a.out, limit=limit, train_limit=train_limit,
                                        epochs=epochs, seeds=seeds, resume=a.resume),
                         indent=2, ensure_ascii=False))
    elif a.command == 'production':
        seeds = None if a.seeds is None else tuple(int(v) for v in a.seeds.split(','))
        print(json.dumps(run_experiment(a.out, limit=a.limit, train_limit=a.train_limit,
                                        epochs=a.epochs, seeds=seeds, resume=a.resume),
                         indent=2, ensure_ascii=False))
    elif a.command == 'closed_loop_smoke':
        print(json.dumps(closed_loop_smoke(a.out, checkpoint=a.checkpoint), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
