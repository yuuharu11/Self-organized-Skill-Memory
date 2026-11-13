#!/usr/bin/env python3
"""MAG1-L: offline amplitude/duration bounds with oracle motion type.

Run from /work after ``source env/activate.sh``. The default command runs both
the 95-unit and 1,595-window pools and writes a new
``outputs/memory-pilot/unseen-libero/magnitude-01`` directory. It uses frozen
LIBERO rows, cached pi0.5 chunks, and the existing A-vm-08 bridge checkpoint;
it does not train or extract image features. Example (GPU capped at 15%):

    taskset -c 8-47 nice -n 10 env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
      OPENBLAS_NUM_THREADS=8 XLA_PYTHON_CLIENT_PREALLOCATE=false \
      XLA_PYTHON_CLIENT_MEM_FRACTION=.15 timeout 8h python magnitude_bounds.py

Type selection is privileged: per query it chooses the allowed memory unit
whose orientation/scale-free motion is closest to the target. Its fitted
orientation is carried through all amplitude rules, so this is an upper bound,
not an inference policy. T1 is the target segment's remaining length according
to the reviewed speed-minimum/gripper-switch segmentation, not a prediction.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compressed_denoise as cd
import scale_only_query as sq
import variable_shape_memory as vs

ROOT = Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero')
OUT = ROOT / 'magnitude-01'
ROWS = ROOT / 'oracle-01/unseen_rows'
SCALE = ROOT / 'scale-only-02/production'
DEN = ROOT / 'compressed-denoise-01/production'
POOL = ROOT / 'cluster-pools-01'
HOLDOUT = (2, 7, 12, 17, 22, 27, 32, 37)
AMP = ('A1_oracle', 'A2_history', 'A3_pi05', 'A4_stored', 'A5_expert_t02', 'A5_expert_t04', 'A6_procrustes')
DUR = ('T1_oracle', 'T2_stored', 'T3_ten', 'T4_history')
EPS = 1e-12


def _json(path: Path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def _npz(path: Path, **kwargs):
    tmp = path.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, **kwargs)
    tmp.replace(path)


def normalize_shape(x, n=10):
    """Resample a [T,6] or [T,7] motion while keeping its coordinate frame."""
    x = np.asarray(x, np.float64)
    at = np.linspace(0, len(x) - 1, n)
    return np.stack([np.interp(at, np.arange(len(x)), x[:, k]) for k in range(x.shape[1])], 1)


def scalar_fit(template, reference):
    """Nonnegative least-squares scalar over the first six action coordinates."""
    x = np.asarray(template, np.float64)[..., :6]
    y = np.asarray(reference, np.float64)[..., :6]
    return np.maximum((x * y).sum(axis=(-2, -1)), 0) / np.maximum((x * x).sum(axis=(-2, -1)), EPS)


def scale_template(template, scale):
    """Scale motion axes; retain the stored gripper channel unchanged."""
    out = np.asarray(template, np.float64).copy()
    out[..., :6] *= np.asarray(scale)[..., None, None]
    return out


def duration_values(t1, stored_length, hist_speed, type_distance):
    """T1 oracle cut length, T2 source length, T3 current 10, T4 distance/speed."""
    speed = np.maximum(np.asarray(hist_speed, np.float64), 1e-8)
    return {
        'T1_oracle': np.maximum(np.asarray(t1, np.int64), 1),
        'T2_stored': np.maximum(np.asarray(stored_length, np.int64), 1),
        'T3_ten': np.full(len(t1), 10, np.int64),
        'T4_history': np.maximum(np.rint(np.asarray(type_distance) / speed).astype(np.int64), 1),
    }


def paired_bootstrap(values, episode, hard=None, *, reps=2000, seed=4101):
    """Episode-cluster bootstrap mean and 95% interval."""
    v = np.asarray(values, np.float64)
    ep = np.asarray(episode)
    if hard is not None:
        keep = np.asarray(hard, bool)
        v, ep = v[keep], ep[keep]
    uniq, inv = np.unique(ep, return_inverse=True)
    if not len(uniq):
        return {'rows': 0, 'episodes': 0, 'mean': None, 'ci95': None}
    sums = np.bincount(inv, weights=v)
    counts = np.bincount(inv)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(uniq), (reps, len(uniq)))
    boot = sums[draws].sum(1) / counts[draws].sum(1)
    return {'rows': int(len(v)), 'episodes': int(len(uniq)), 'mean': float(v.mean()),
            'ci95': [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))]}


def episode_cluster(task, episode):
    """Make episode IDs globally unique because LIBERO episode numbers reset per task."""
    task = np.asarray(task, np.int64)
    episode = np.asarray(episode, np.int64)
    if task.shape != episode.shape:
        raise ValueError('task/episode shape mismatch')
    return task * 1_000_000 + episode


def best_oracle_types(target_physical, candidates, *, pool_task=None, query_task=None, chunk=128):
    """Return per-row candidate index, Procrustes rotation and A1 scale.

    This uses target motion to choose type and is intentionally an upper bound.
    ``candidates`` are canonical physical [K,10,7] motions.
    """
    target = np.asarray(target_physical, np.float64)
    pool = np.asarray(candidates, np.float64)
    allowed = np.ones((len(target), len(pool)), bool)
    if pool_task is not None and query_task is not None:
        allowed &= np.asarray(pool_task)[None, :] != np.asarray(query_task)[:, None]
        if not allowed.any(1).all():
            raise ValueError('same-task exclusion leaves a query with no candidate')
    choices = np.empty(len(target), np.int64)
    rots = np.empty((len(target), 3, 3), np.float64)
    amps = np.empty(len(target), np.float64)
    errs = np.empty(len(target), np.float64)
    for i, y in enumerate(target):
        best = (np.inf, None, None, None)
        for st in range(0, len(pool), chunk):
            p = pool[st:st + chunk]
            cross = p[:, :, :3].transpose(0, 2, 1) @ y[:, :3] + p[:, :, 3:6].transpose(0, 2, 1) @ y[:, 3:6]
            u, _, vt = np.linalg.svd(cross)
            fix = np.broadcast_to(np.eye(3), (len(p), 3, 3)).copy()
            fix[:, -1, -1] = np.linalg.det(u @ vt)
            r = u @ fix @ vt
            aligned = np.concatenate([p[:, :, :3] @ r, p[:, :, 3:6] @ r], axis=2)
            scale = np.maximum((aligned * y).sum((1, 2)) / np.maximum((aligned * aligned).sum((1, 2)), EPS), 0)
            residual = ((aligned * scale[:, None, None] - y) ** 2).mean((1, 2))
            residual[~allowed[i, st:st + len(p)]] = np.inf
            j = int(np.argmin(residual))
            if residual[j] < best[0]:
                best = (residual[j], st + j, r[j], scale[j])
        errs[i], choices[i], rots[i], amps[i] = best
    return choices, rots, amps, errs


def apply_rotation(shape, rotation):
    x = np.asarray(shape, np.float64)
    r = np.asarray(rotation, np.float64)
    return np.concatenate([x[..., :3] @ r, x[..., 3:6] @ r], axis=-1)


def extend_type(shape, n):
    """Repeat-interpolate canonical type to the chosen action horizon."""
    n = int(max(1, n))
    return normalize_shape(shape, n)


def rollout(template, amplitude, horizon, gripper=None, rotation=None):
    """Render scaled type to a horizon and hold its last action beyond horizon."""
    base = extend_type(template, int(horizon))
    motion = vs.to_physical(base[:, :6])
    if rotation is not None:
        motion = apply_rotation(motion[None], rotation)[0]
    motion = motion * float(amplitude)
    base[:, :6] = vs.to_normalized(motion)
    if base.shape[1] < 7:
        base = np.column_stack([base, np.full(len(base), 0. if gripper is None else gripper)])
    return base


def _load_rows(split):
    return {k: np.load(SCALE / 'prep' / f'W_inputs_{split}.npz')[k] for k in
            ('row', 'task', 'episode', 'frame', 'hist', 'pi05', 'hard') if
            (SCALE / 'prep' / f'W_inputs_{split}.npz').exists() and
            k in np.load(SCALE / 'prep' / f'W_inputs_{split}.npz').files}


def _traces():
    actions = np.load(ROWS / 'actions.npy', mmap_mode='r', allow_pickle=False)
    task = np.load(ROWS / 'task.npy', mmap_mode='r', allow_pickle=False)
    episode = np.load(ROWS / 'episode.npy', mmap_mode='r', allow_pickle=False)
    frame = np.load(ROWS / 'frame.npy', mmap_mode='r', allow_pickle=False)
    result = {}
    for key in sorted(set(zip(np.asarray(task).tolist(), np.asarray(episode).tolist()))):
        ids = np.flatnonzero((task == key[0]) & (episode == key[1]))
        ids = ids[np.argsort(frame[ids])]
        if not np.array_equal(frame[ids], frame[ids[0]] + 10 * np.arange(len(ids))):
            continue
        result[key] = (np.asarray(actions[ids, :, :7]).reshape(-1, 7), ids, int(frame[ids[0]]))
    return result


def _segments_by_row(split_rows, traces):
    """Map row to current anchor span and return exact remaining segment targets."""
    import motion_core_eval as motion
    all_task = np.load(ROWS / 'task.npy', mmap_mode='r', allow_pickle=False)
    all_ep = np.load(ROWS / 'episode.npy', mmap_mode='r', allow_pickle=False)
    all_frame = np.load(ROWS / 'frame.npy', mmap_mode='r', allow_pickle=False)
    ids = np.asarray(split_rows['row'], np.int64)
    spans = {}
    for key, (trace, row_ids, frame0) in traces.items():
        cuts = motion.anchor_spans(trace.astype(np.float32), min_length=5)
        for row_pos, row in enumerate(row_ids):
            # Every row stores a 10-action chunk; manifest ``frame`` is the
            # source observation index and is not the within-chunk offset.
            start = int(row_pos * 10)
            hit = next(((s, e) for s, e in cuts if s <= start < e), None)
            if hit is not None:
                spans[int(row)] = (trace, start, hit[1])
    out = []
    for row in ids:
        if int(row) not in spans:
            out.append((np.zeros((1, 7), np.float32), 1))
        else:
            trace, start, end = spans[int(row)]
            # Keep the rest of the episode as the scoring target; T1 separately
            # records the distance to this segment's end.
            out.append((trace[start:], end - start))
    return out


def _stored_scales():
    """Rebuild raw source RMS scales for pool_08's selected canonical units."""
    import cluster_pools as cp
    from scipy.spatial.transform import Rotation  # imported by helper path in shared env
    rows = {'task': np.load(cd.RUN / 'rows/task.npy', mmap_mode='r'),
            'episode': np.load(cd.RUN / 'rows/episode.npy', mmap_mode='r'),
            'frame': np.load(cd.RUN / 'rows/frame.npy', mmap_mode='r')}
    mem_ids = np.asarray(np.load(cd.RUN / 'memory_row_ids.npy'), np.int64)
    bank = cd._bank_from_memory(rows, mem_ids, mode='variable', form='rms', tau=.08)
    if bank['unit_count'] != 133:
        raise AssertionError(f'pool_08 raw medoid count mismatch: {bank["unit_count"]}')
    actions = np.load(cd.RUN / 'rows/actions.npy', mmap_mode='r')
    traces, _ = cd._stitch(actions, rows['task'], rows['episode'], rows['frame'], mem_ids)
    scales = []
    decoded = cd._bank_decode(bank)
    source_rows = []
    for task, ep, start, length, decoded_unit in zip(bank['task'], bank['episode'], bank['start'], bank['memory'].lengths, decoded):
        # Reapply the same medoid filter used by cluster-pools-01.
        if task in HOLDOUT:
            continue
        if not np.allclose(normalize_shape(decoded_unit, 10), np.load(POOL / 'pool_unit_08.npz')['shape'][len(scales)], atol=2e-5):
            raise AssertionError('reconstructed pool order/shape differs from pool_unit_08')
        from variable_shape_memory import pose_of
        trace = traces[(int(task), int(ep))][int(start):int(start + length), :6]
        _, size = pose_of(trace, 'rms')
        scales.append(float(size))
    if len(scales) != 95:
        raise AssertionError(f'source-scale count mismatch: {len(scales)}')
    return np.asarray(scales)


def _expert_outputs(split, drafts, t0_values=(.2, .4)):
    """Frozen A-vm-08 bridge inference for A5; row-keyed noise mirrors DN1."""
    import torch
    torch.set_num_threads(8)
    import jax
    import flax.nnx as nnx
    import orbax.checkpoint as ocp
    import compressed_denoise as expert
    import openpi.shared.nnx_utils as nnx_utils

    base = expert._module(expert.PI05 / 'run.py', f'mag1_base_{split}')
    model, _ = base.load_policy(expert.CHECKPOINT)
    trained = json.loads((DEN / 'predictions/A-vm-08/result.json').read_text())
    checkpoint = Path(trained['selected_checkpoint']['path'])
    with ocp.PyTreeCheckpointer() as checkpointer:
        part = checkpointer.restore(checkpoint)
    state = nnx.state(model, nnx.All(nnx.Param, nnx_utils.PathRegex(r'.*(llm.*_1|action_in_proj|time_mlp|action_out_proj).*')))
    state.replace_by_pure_dict(jax.tree.map(lambda x: __import__('jax.numpy', fromlist=['asarray']).asarray(x), part))
    nnx.update(model, state)
    inp = _load_rows(split)
    unseen = {p.stem: np.load(p, mmap_mode='r', allow_pickle=False) for p in ROWS.glob('*.npy')}
    rows_table = base.load_rows(cd.RUN)
    obs = base.obs_batch(unseen, np.asarray(inp['row'], np.int64))
    global_ids = len(rows_table['actions']) + np.asarray(inp['row'], np.int64)
    eps = np.stack([np.asarray(jax.random.normal(jax.random.fold_in(jax.random.key(cd.NOISE_SEED), int(i)), (10, 32))) for i in global_ids])
    out = {}
    for t0 in t0_values:
        out[t0] = expert._predict_action(model, obs, np.asarray(drafts, np.float32), eps, t0)[:, :, :7]
    return out


def _metric(pred, target, axis_count):
    return ((np.asarray(pred)[:, :5, :axis_count] - np.asarray(target)[:, :5, :axis_count]) ** 2).mean((1, 2))


def _episode_hard(split, inputs):
    """Difficulty threshold from Dev pi0.5 error q80, then carry to Eval."""
    target_all = np.load(ROWS / 'actions.npy', mmap_mode='r', allow_pickle=False)
    target = np.asarray(target_all[np.asarray(inputs['row'], np.int64), :, :7])
    p = np.asarray(inputs['pi05'], np.float64)
    err = _metric(p, target, 7)
    if split == 'dev':
        threshold = float(np.quantile(err, .8))
    else:
        dev = _load_rows('dev')
        dt = np.asarray(target_all[np.asarray(dev['row'], np.int64), :, :7])
        threshold = float(np.quantile(_metric(dev['pi05'], dt, 7), .8))
    return err >= threshold, threshold


def run(out=OUT, *, include_a5=True):
    out = Path(out)
    if out.exists():
        raise FileExistsError(f'refusing to overwrite existing output: {out}')
    out.mkdir(parents=True)
    inputs = {s: _load_rows(s) for s in ('dev', 'eval')}
    actions = np.load(ROWS / 'actions.npy', mmap_mode='r', allow_pickle=False)
    targets = {s: np.asarray(actions[np.asarray(inputs[s]['row'], np.int64), :, :7], np.float32) for s in inputs}
    traces = _traces()
    spans = {s: _segments_by_row(inputs[s], traces) for s in inputs}
    # Normalized actions -> physical coordinates for memory type selection.
    phys_target = {s: vs.to_physical(targets[s][:, :, :6]).astype(np.float32) for s in inputs}
    unitz = np.load(POOL / 'pool_unit_08.npz')
    unit_phys = np.asarray(unitz['shape'][:, :, :6], np.float32)
    # Pool shapes are canonical (orientation/size removed). Fit a pose per query.
    chosen = {s: best_oracle_types(phys_target[s], unit_phys, pool_task=unitz['source_task'], query_task=inputs[s]['task']) for s in inputs}
    source_scale = _stored_scales()
    if not np.array_equal(unitz['source_task'], np.load(POOL / 'pool_unit_08.npz')['source_task']):
        raise AssertionError('pool metadata changed')

    results, row_payload = [], {}
    for split in ('dev', 'eval'):
        x = inputs[split]; y = targets[split]; pick, rot, a1, type_err = chosen[split]
        hard, hard_threshold = _episode_hard(split, x)
        templates_phys = apply_rotation(unit_phys[pick], rot)
        templates = np.concatenate([vs.to_normalized(templates_phys).astype(np.float32),
                                    unitz['shape'][pick, :, 6:7].astype(np.float32)], axis=2)
        # A2: RMS speed of the prior ten executed actions; A3: fit pi0.5 amplitude.
        hist_phys = vs.to_physical(np.asarray(x['hist'])[:, :, :6])
        hist_speed = np.sqrt(np.mean(np.sum(hist_phys ** 2, axis=2), axis=1))
        template_speed = np.sqrt(np.mean(np.sum(templates_phys ** 2, axis=2), axis=1))
        a2 = hist_speed / np.maximum(template_speed, 1e-8)
        pi05_phys = vs.to_physical(x['pi05'][:, :, :6])
        a3 = scalar_fit(templates_phys, pi05_phys)
        a4 = source_scale[pick]
        # A6 reproduces the current joint rotation+scale Procrustes fit to pi0.5.
        a6 = np.empty(len(y), np.float64)
        a6_rot = np.empty((len(y), 3, 3), np.float64)
        for i in range(len(y)):
            r_pi, s_pi = sq.procrustes_rs(templates_phys[i], pi05_phys[i])
            a6[i] = s_pi
            a6_rot[i] = r_pi
        amplitudes = {'A1_oracle': a1, 'A2_history': a2, 'A3_pi05': a3,
                      'A4_stored': a4, 'A6_procrustes': a6}
        amp_rotations = {'A6_procrustes': a6_rot}
        if include_a5:
            drafts = np.zeros((len(y), 10, 32), np.float32)
            for i in range(len(y)):
                drafts[i, :, :7] = rollout(templates[i], a2[i], 10).astype(np.float32)
            # Existing A-vm-08 Expert only; no gradients or checkpoint changes.
            exp = _expert_outputs(split, drafts)
            amplitudes['A5_expert_t02'] = scalar_fit(templates_phys, vs.to_physical(exp[.2][:, :, :6]))
            amplitudes['A5_expert_t04'] = scalar_fit(templates_phys, vs.to_physical(exp[.4][:, :, :6]))
        seg = spans[split]
        t1 = np.asarray([v[1] for v in seg], np.int64)
        t2 = unitz['length'][pick]
        type_distance = np.sqrt(np.sum(templates_phys ** 2, axis=(1, 2)))
        t = duration_values(t1, t2, hist_speed, type_distance)
        # Frame-exact action targets from the query point through the anchor cut.
        future = [v[0][:max(int(t1[i]), 10)] for i, v in enumerate(seg)]
        # score common 10-frame regions; shorter predictions hold their final command.
        combos = {}
        for ak, amp in amplitudes.items():
            for tk, horizons in t.items():
                pred = np.zeros_like(y, dtype=np.float32)
                truth = np.zeros_like(y, dtype=np.float32)
                for i, (h, source) in enumerate(zip(horizons, future)):
                    r = rollout(templates[i], amp[i], h,
                                rotation=amp_rotations.get(ak, [None] * len(y))[i])
                    upto = min(10, len(r), len(source))
                    pred[i, :upto] = r[:upto]
                    truth[i, :upto] = source[:upto]
                    if upto < 10:
                        pred[i, upto:] = pred[i, max(upto - 1, 0)]
                        truth[i, upto:] = truth[i, max(upto - 1, 0)]
                full = np.mean((pred[:, :, :7] - truth[:, :, :7]) ** 2, axis=(1, 2))
                for name, axes in (('6axis', 6), ('active7', 7)):
                    first = _metric(pred, truth, axes)
                    cont = ((pred[:, 5:10, :axes] - truth[:, 5:10, :axes]) ** 2).mean((1, 2))
                    combos[f'{ak}|{tk}|{name}'] = {'first5': first, 'continuation5_9': cont, 'full10': full}
                results.append((split, ak, tk, combos))
        # Baselines and exact target oracle are row-aligned and scored with same selected target horizon.
        baselines = {'pi05': np.asarray(x['pi05']), 'previous_hold': np.repeat(x['hist'][:, -1:, :7], 10, axis=1), 'action_oracle': y}
        base_metrics = {}
        for name, p in baselines.items():
            for axes_name, axes in (('6axis', 6), ('active7', 7)):
                first = _metric(p, truth, axes)
                cont = ((p[:, 5:10, :axes] - truth[:, 5:10, :axes]) ** 2).mean((1, 2))
                base_metrics[f'{name}|{axes_name}'] = {'first5': first, 'continuation5_9': cont}
        _npz(out / f'{split}_rows.npz', row=x['row'], task=x['task'], episode=x['episode'], hard=hard,
             selected_unit=pick, oracle_type_error=type_err, oracle_amp=a1, duration_t1=t1,
             duration_t2=t2, duration_t3=t['T3_ten'], duration_t4=t['T4_history'],
             target=targets[split], template=templates, pred_amp=np.stack([amplitudes[k] for k in amplitudes]))
        row_payload[split] = {'combo': combos, 'baseline': base_metrics, 'hard': hard,
                              'episode': episode_cluster(x['task'], x['episode']), 'threshold': hard_threshold}

    # Pool comparison: all 1,595 fixed-window candidates, still oracle-picked by type.
    wpool = np.load(SCALE / 'prep/W_pool.npz')
    w_phys = np.asarray(wpool['phys'], np.float32)
    scale_pool_metrics = {}
    for split in ('dev', 'eval'):
        pick, rot, a1, err = best_oracle_types(phys_target[split], w_phys, pool_task=wpool['task'], query_task=inputs[split]['task'])
        pred = np.zeros((len(pick), 10, 7), np.float32)
        for i, j in enumerate(pick):
            aligned = apply_rotation(w_phys[j:j+1], rot[i])[0]
            pred[i, :, :6] = vs.to_normalized(aligned * a1[i]).astype(np.float32)
            pred[i, :, 6] = inputs[split]['hist'][i, -1, 6]
        row_mse = _metric(pred, targets[split], 7)
        row_mse6 = _metric(pred, targets[split], 6)
        ep_cluster = episode_cluster(inputs[split]['task'], inputs[split]['episode'])
        scale_pool_metrics[split] = {
            'active7_first5': paired_bootstrap(row_mse, ep_cluster),
            'axis6_first5': paired_bootstrap(row_mse6, ep_cluster),
            'axis6_first5_converted_to_active7': float(row_mse6.mean() * 6 / 7),
        }
        _npz(out / f'{split}_scale_only_1595_type.npz', selected_window=pick, type_error=err,
             oracle_amp=a1, prediction=pred, first5_active7_mse=row_mse, first5_axis6_mse=row_mse6)

    # Select the best realistic (no target-derived amplitude/duration) pair on Dev only.
    selectable = [a for a in amplitudes if a not in ('A1_oracle',)]
    selectable_t = ('T2_stored', 'T3_ten', 'T4_history')
    # A5 must be available in this run to be considered; A6 is the same cached pi05 scale fit.
    dev = row_payload['dev']
    choices = []
    for ak in selectable:
        for tk in selectable_t:
            v = dev['combo'][f'{ak}|{tk}|active7']['first5']
            choices.append((float(np.mean(v)), ak, tk))
    chosen_nonoracle = min(choices)
    table_rows = []
    summary_combos = {
        'amplitude oracle only': ('A1_oracle', chosen_nonoracle[2]),
        'duration oracle only': (chosen_nonoracle[1], 'T1_oracle'),
        'both oracle': ('A1_oracle', 'T1_oracle'),
        'best non-oracle by Dev': (chosen_nonoracle[1], chosen_nonoracle[2]),
    }
    # Full matrix and report use row-level episode bootstrap.
    matrix = []
    for split in ('dev', 'eval'):
        pdata = row_payload[split]
        for ak in amplitudes:
            for tk in DUR:
                for axes_name in ('6axis', 'active7'):
                    rec = pdata['combo'][f'{ak}|{tk}|{axes_name}']
                    for metric in ('first5', 'continuation5_9'):
                        base_compare = {}
                        for base_name in ('pi05', 'previous_hold', 'action_oracle'):
                            bv = pdata['baseline'][f'{base_name}|{axes_name}'][metric]
                            base_compare[base_name] = paired_bootstrap(bv, pdata['episode'])
                            base_compare[f'{base_name}_hard'] = paired_bootstrap(bv, pdata['episode'], pdata['hard'])
                        matrix.append({'split': split, 'amp': ak, 'duration': tk, 'axes': axes_name,
                            'window': metric,
                            **paired_bootstrap(rec[metric], pdata['episode']),
                            'hard': paired_bootstrap(rec[metric], pdata['episode'], pdata['hard']),
                            'baselines': base_compare})
    for label, (ak, tk) in summary_combos.items():
        for split in ('dev', 'eval'):
            pdata = row_payload[split]
            rec = pdata['combo'][f'{ak}|{tk}|active7']['first5']
            table_rows.append({'summary': label, 'split': split, 'amp': ak, 'duration': tk,
                               **paired_bootstrap(rec, pdata['episode']),
                               'pi05': paired_bootstrap(pdata['baseline']['pi05|active7']['first5'], pdata['episode']),
                               'previous_hold': paired_bootstrap(pdata['baseline']['previous_hold|active7']['first5'], pdata['episode']),
                               'action_oracle': paired_bootstrap(pdata['baseline']['action_oracle|active7']['first5'], pdata['episode']),
                               'pi05_hard': paired_bootstrap(pdata['baseline']['pi05|active7']['first5'], pdata['episode'], pdata['hard']),
                               'previous_hold_hard': paired_bootstrap(pdata['baseline']['previous_hold|active7']['first5'], pdata['episode'], pdata['hard']),
                               'action_oracle_hard': paired_bootstrap(pdata['baseline']['action_oracle|active7']['first5'], pdata['episode'], pdata['hard']),
                               'hard': paired_bootstrap(rec, pdata['episode'], pdata['hard'])})
    normal_check = np.load(DEN / 'predictions/A-vm-08/eval_rows.npz')
    eval_rows_aligned = np.array_equal(inputs['eval']['row'], normal_check['row_ids'])
    if not eval_rows_aligned:
        raise AssertionError('cached A-vm-08 Eval rows do not match the fixed Eval split')
    official_mse = np.mean((normal_check['official_pi05_prediction'] - normal_check['target']) ** 2, axis=(1, 2))
    official_mse6 = np.mean((normal_check['official_pi05_prediction'][:, :, :6] - normal_check['target'][:, :, :6]) ** 2, axis=(1, 2))
    previous_eval = np.repeat(inputs['eval']['hist'][:, -1:, :7], 10, axis=1)
    previous_full = np.mean((previous_eval - targets['eval']) ** 2, axis=(1, 2))
    previous_full6 = np.mean((previous_eval[:, :, :6] - targets['eval'][:, :, :6]) ** 2, axis=(1, 2))
    report = {
        'experiment': 'MAG1-L', 'generated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'type_selection': 'target-oracle minimum scale/rotation Procrustes error; same task excluded where present; episodes cannot overlap because the pool is LIBERO-40 and query rows are heldout LIBERO-90 tasks',
        'type_pool': 'cluster-pools-01/pool_unit_08.npz: 95 units',
        'comparison_pool': 'scale-only-02/production/prep/W_pool.npz: 1,595 fixed 10-frame windows',
        'hard_definition': 'pi0.5 active7 10-step MSE >= Dev q80; threshold is fixed from Dev and reused for Eval',
        'duration_definition': {'T1': 'remaining steps to motion_core_eval.anchor_spans segment end',
            'T2': 'source unit original frame count', 'T3': 10,
            'T4': 'canonical type total physical displacement divided by previous-history RMS speed'},
        'amplitude_definition': {'A1': 'least-squares scale to oracle action after oracle type orientation',
            'A2': 'match prior 10 executed action physical RMS speed', 'A3': 'least-squares scale fit to cached pi0.5 motion, orientation held to oracle type',
            'A4': 'source unit RMS scale rebuilt from frozen memory demos',
            'A5': 'A2 draft passed through frozen A-vm-08 bridge at t0=.2/.4; fit output scale to fixed type',
            'A6': 'pi0.5 Procrustes scale on fixed oracle type orientation'},
        'dev_selected_nonoracle': {'amp': chosen_nonoracle[1], 'duration': chosen_nonoracle[2], 'dev_active7_first5_mse': chosen_nonoracle[0]},
        'summary': table_rows, 'all_combinations': matrix,
        'baselines': {s: {k: {'first5_active7': paired_bootstrap(v['first5'], row_payload[s]['episode']),
                              'continuation5_9_active7': paired_bootstrap(v['continuation5_9'], row_payload[s]['episode'])}
                          for k, v in row_payload[s]['baseline'].items()} for s in row_payload},
        'scale_only_1595_pool': scale_pool_metrics,
        'reference_checks': {'reference_claimed': {'type_oracle': .079, 'amplitude_oracle': .065, 'orientation_oracle': .037,
            'previous_hold': .114, 'pi05': .192},
            'eval_rows_match_A_vm_08_cached_eval': bool(eval_rows_aligned),
            'cached_normal_pi05_active7_full10': float(official_mse.mean()),
            'cached_normal_pi05_6axis_full10': float(official_mse6.mean()),
            'cached_normal_pi05_6axis_converted_to_7axis': float(official_mse6.mean() * 6 / 7),
            'previous_hold_active7_full10': float(previous_full.mean()),
            'previous_hold_6axis_full10': float(previous_full6.mean()),
            'previous_hold_6axis_full10_converted_to_7axis': float(previous_full6.mean() * 6 / 7),
            'action_oracle_active7_full10': float(np.mean((targets['eval'] - targets['eval']) ** 2)),
            'note': 'First5 and continuation metrics are reported separately above; full10 checks use the same fixed Eval row IDs and normalized active7 actions. Six-axis mean is converted by 6/7 when comparing its total contribution to active7.'},
    }
    _json(out / 'report.json', report)
    (out / 'report.md').write_text(render_report(report))
    (out / 'DONE').write_text('complete\n')
    return report


def render_report(report):
    lines = ['# MAG1-L result', '',
             f"Dev selected non-oracle pair: **{report['dev_selected_nonoracle']['amp']} + {report['dev_selected_nonoracle']['duration']}** (Dev active7 first-5 MSE {report['dev_selected_nonoracle']['dev_active7_first5_mse']:.6f}).",
             '', '## Four-row summary (active7, first five steps)', '',
             '| Summary | Split | Amplitude | Duration | MSE [95% CI] | π0.5 | Previous hold | Action oracle | Hard MSE | Hard π0.5 | Hard hold |',
             '|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in report['summary']:
        ci = r['ci95']
        lines.append(f"| {r['summary']} | {r['split']} | {r['amp']} | {r['duration']} | {r['mean']:.6f} [{ci[0]:.6f}, {ci[1]:.6f}] | {r['pi05']['mean']:.6f} | {r['previous_hold']['mean']:.6f} | {r['action_oracle']['mean']:.6f} | {r['hard']['mean']:.6f} | {r['pi05_hard']['mean']:.6f} | {r['previous_hold_hard']['mean']:.6f} |")
    lines += ['', '## Full combinations', '', '| Split | Amplitude | Duration | Axes | Window | Mean MSE | 95% CI | π0.5 | Previous hold | Action oracle | Hard mean | Hard π0.5 | Hard hold |', '|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in report['all_combinations']:
        ci = r['ci95']
        b = r['baselines']
        lines.append(f"| {r['split']} | {r['amp']} | {r['duration']} | {r['axes']} | {r['window']} | {r['mean']:.6f} | [{ci[0]:.6f}, {ci[1]:.6f}] | {b['pi05']['mean']:.6f} | {b['previous_hold']['mean']:.6f} | {b['action_oracle']['mean']:.6f} | {r['hard']['mean']:.6f} | {b['pi05_hard']['mean']:.6f} | {b['previous_hold_hard']['mean']:.6f} |")
    lines += ['', '## Baselines', '', '```json', json.dumps(report['baselines'], indent=2), '```', '',
              '## 1,595-candidate scale-only comparison', '',
              '| Split | A1 active7 first-5 | 95% CI | A1 6-axis first-5 ×6/7 |', '|---|---:|---:|---:|']
    for split, rec in report['scale_only_1595_pool'].items():
        a7 = rec['active7_first5']
        lines.append(f"| {split} | {a7['mean']:.6f} | [{a7['ci95'][0]:.6f}, {a7['ci95'][1]:.6f}] | {rec['axis6_first5_converted_to_active7']:.6f} |")
    lines += ['', '## Reference checks', '', '```json', json.dumps(report['reference_checks'], indent=2), '```', '',
              'Type selection and T1/A1 use target actions and are offline upper bounds. No Eval results participate in selection.', '']
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=OUT)
    p.add_argument('--no-a5', action='store_true', help='run other variants without the frozen bridge diagnostic')
    a = p.parse_args()
    run(a.out, include_a5=not a.no_a5)


if __name__ == '__main__':
    main()
