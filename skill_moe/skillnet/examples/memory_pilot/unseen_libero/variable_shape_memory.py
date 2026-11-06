#!/usr/bin/env python3
"""Merge motion segments by shape, with direction and size removed before the DCT.

Every result gives the query its own direction and size, then picks the memory unit
closest to the correct action. That makes it an upper bound on what denoising could
recover, not a generator. Nothing is denoised and no retrieval is involved.

    forms     as_is   normalized actions, nothing removed (the earlier comparison)
              net     rotate the net xyz motion onto +x, divide by its length
              rms     the same rotation, divide by the RMS of the motion
    segments  fixed10, or the segment_demo.cut rule at a given epsilon
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.fft import dct, idct

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'pi05_memory'), str(HERE.parent / 'pi05_memory' / 'shape_skills')]
from draft_reconstruction import ROOT, RUN, UNSEEN, summarize  # noqa: E402
from oracle_repr import RepresentationMetric  # noqa: E402
from representations import rotation_to_x  # noqa: E402
from segment_demo import Q01, Q99, cut  # noqa: E402
from skill_memories.build import m6_fit, unit_medoid  # noqa: E402
from variable_length_error import paired_interval, rows_from, stitch  # noqa: E402

K = 3
FORMS = ('as_is', 'net', 'rms')
SEGMENTS = ('fixed10', 'e0.005', 'e0.02')
KAPPAS = (.2, .1, .05, .025)
SUPPORTS = {'default': (10, 3), 'keep_all': (1, 1)}
SPAN = (Q99[:6] - Q01[:6] + 1e-6).astype(np.float64)
LOW = Q01[:6].astype(np.float64)


def to_physical(motion):
    return (np.asarray(motion, np.float64) + 1) * .5 * SPAN + LOW


def to_normalized(motion):
    return (np.asarray(motion, np.float64) - LOW) / SPAN * 2 - 1


def pose_of(motion, form):
    """Return (R, scale) for a normalized [L,6] motion."""
    p = to_physical(motion)
    total = p[:, :3].sum(0)
    rotation = np.asarray(rotation_to_x(total), np.float64)
    rms = float(np.sqrt(np.mean(p * p)))
    scale = float(np.linalg.norm(total)) if form == 'net' else rms
    return rotation, scale if scale >= 1e-5 else max(rms, 1.) if rms < 1e-5 else rms


def to_shape(motion, form):
    if form == 'as_is':
        return np.asarray(motion, np.float64)
    rotation, scale = pose_of(motion, form)
    p = to_physical(motion)
    out = np.column_stack([p[:, :3] @ rotation.T, p[:, 3:6] @ rotation.T]) / scale
    # A net-scaled shape sums to one, so multiply by length to compare across lengths.
    return out * len(p) if form == 'net' else out


def from_shape(shape, form, pose):
    """Apply a query's pose to [M,L,6] shapes; returns normalized motion."""
    if form == 'as_is':
        return shape
    rotation, scale = pose
    s = shape / shape.shape[1] if form == 'net' else shape
    return to_normalized(np.concatenate([s[..., :3] @ rotation, s[..., 3:6] @ rotation], -1) * scale)


def encode(trace, start, end, form, grip_k):
    """[K*6 shape values, grip_k gripper values], independent of length."""
    a = trace[start:end]
    full = dct(to_shape(a[:, :6], form), type=2, norm='ortho', axis=0)
    c = np.zeros((K, 6), np.float64)
    c[:min(K, len(a))] = full[:K]
    g = np.zeros(grip_k, np.float64)
    gripper = dct(np.asarray(a[:, 6], np.float64), type=2, norm='ortho')
    g[:min(grip_k, len(a))] = gripper[:grip_k]
    return np.r_[c.ravel(), g] / np.sqrt(len(a))


def decode(values, length, form, pose, grip_k):
    v = np.atleast_2d(values)
    keep = min(K, length)
    c = np.zeros((len(v), length, 6), np.float64)
    c[:, :keep] = v[:, :K * 6].reshape(-1, K, 6)[:, :keep] * np.sqrt(length)
    g = np.zeros((len(v), length), np.float64)
    keep = min(grip_k, length)
    g[:, :keep] = v[:, K * 6:K * 6 + keep] * np.sqrt(length)
    out = np.empty((len(v), length, 7), np.float64)
    out[:, :, :6] = from_shape(idct(c, type=2, norm='ortho', axis=1), form, pose)
    out[:, :, 6] = idct(g, type=2, norm='ortho', axis=1)
    return out


def spans_of(trace, segment):
    if segment == 'fixed10':
        return [(i, i + 10) for i in range(0, len(trace), 10)]
    return [(s['start'], s['end']) for s in cut(trace, epsilon=float(segment[1:]), k=K)]


def grip_values(segment):
    # A variable segment holds one gripper sign; a fixed window may cross a switch.
    return K if segment == 'fixed10' else 1


def set_coefficients(k):
    """Change how many DCT values each axis keeps, for this process."""
    global K
    K = int(k)


def straight_candidates(form, grip_k):
    """The shape-free control: constant motion along +x, open or closed."""
    level = {'net': 1., 'rms': np.sqrt(6.)}.get(form, 0.)
    out = []
    for grip in (-1., 1.):
        v = np.zeros(K * 6 + grip_k)
        v[0] = level
        v[K * 6] = grip
        out.append(v)
    return np.stack(out)


def score(candidates, form, segment, traces, where, positions, target, hard, *, seed=None):
    rebuilt, used, selections = {}, set(), 0
    rng = np.random.default_rng(seed) if seed is not None else None
    grip_k = grip_values(segment)
    for ep, trace in traces.items():
        out = np.empty_like(trace, dtype=np.float64)
        for start, end in spans_of(trace, segment):
            motion = trace[start:end]
            pose = None if form == 'as_is' else pose_of(motion[:, :6], form)
            if rng is not None:
                pick = int(rng.integers(len(candidates)))
                out[start:end] = decode(candidates[pick], end - start, form, pose, grip_k)[0]
            else:
                drafts = decode(candidates, end - start, form, pose, grip_k)
                pick = int(np.argmin(np.mean((drafts - motion[None]) ** 2, axis=(1, 2))))
                out[start:end] = drafts[pick]
            used.add(pick)
            selections += 1
        rebuilt[ep] = out
    pred = rows_from(rebuilt, where, positions)
    return {'units_used': len(used), 'selections': selections,
            **summarize(pred, target, hard)}, np.mean((pred - target) ** 2, axis=(1, 2))


class Shared:
    """Data the worker processes inherit."""


def run_one(config):
    form, segment, kappa, support = config
    values, keys = [], []
    grip_k = grip_values(segment)
    for key, trace in Shared.memory.items():
        for start, end in spans_of(trace, segment):
            values.append(encode(trace, start, end, form, grip_k))
            keys.append(key)
    values, keys = np.stack(values).astype(np.float32), np.asarray(keys)
    args = (Shared.traces, Shared.where, Shared.positions, Shared.target, Shared.hard)
    entry = {'form': form, 'segment': segment, 'kappa': kappa, 'support': support, 'segments': len(values),
             'values_per_unit': K * 6 + grip_k}
    if kappa is None:
        result, error = score(values.astype(np.float64), form, segment, *args)
        entry.update(units=len(values), unassigned=0., **result)
        random, _ = score(values.astype(np.float64), form, segment, *args, seed=20261104)
        entry['random_all'], entry['random_hard'] = random['all'], random['hard']
        straight, _ = score(straight_candidates(form, grip_k), form, segment, *args)
        entry['straight_all'], entry['straight_hard'] = straight['all'], straight['hard']
    else:
        metric = RepresentationMetric(values.shape[1])
        least, episodes = SUPPORTS[support]
        fit = m6_fit(values, np.arange(len(values)), keys[:, 0].astype(np.int32), keys[:, 1].astype(np.int32),
                     metric, kappa=kappa, seed=20261104, min_support=least, min_episodes=episodes)
        medoids = np.asarray([unit_medoid(m, values, metric) for m in fit['members_local']], np.int64)
        result, error = score(values[medoids].astype(np.float64), form, segment, *args)
        entry.update(units=len(medoids), unassigned=float(np.mean(np.asarray(fit['row_units']) < 0)), **result)
    print(json.dumps({k: entry[k] for k in ('form', 'segment', 'kappa', 'support', 'units', 'unassigned', 'all', 'hard')}), flush=True)
    return entry, error


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=ROOT / 'unseen-libero' / 'draft-reconstruction-01' / 'variable_shape.json')
    ap.add_argument('--workers', type=int, default=16)
    args = ap.parse_args()
    rows = {k: np.load(RUN / 'rows' / f'{k}.npy', mmap_mode='r') for k in ('actions', 'task', 'episode', 'frame', 'memory')}
    memory = np.flatnonzero(np.asarray(rows['memory'], bool))
    key = np.stack([np.asarray(rows['task'])[memory], np.asarray(rows['episode'])[memory]], 1)
    Shared.memory, _ = stitch(np.asarray(rows['actions'][memory]), key, np.asarray(rows['frame'])[memory])
    un = {k: np.load(UNSEEN / 'unseen_rows' / f'{k}.npy', mmap_mode='r') for k in ('actions', 'task', 'episode', 'frame', 'is_dev')}
    ev = np.flatnonzero(~np.asarray(un['is_dev']))
    raw = np.load(UNSEEN / 'drafts' / 'raw_none_eval.npz', allow_pickle=False)
    if not np.array_equal(raw['query_ids'], ev + len(rows['task'])):
        raise AssertionError('unseen eval rows differ from the draft cache')
    threshold = json.loads((UNSEEN / 'hard_threshold.json').read_text())['threshold']
    Shared.hard = raw['pre_mse'] >= threshold
    ukey = np.stack([np.asarray(un['task']), np.asarray(un['episode'])], 1)[ev]
    Shared.traces, Shared.where = stitch(np.asarray(un['actions'])[ev], ukey, np.asarray(un['frame'])[ev])
    Shared.positions = np.arange(len(ev))
    Shared.target = rows_from(Shared.traces, Shared.where, Shared.positions)

    configs = [(f, s, None, 'none') for f in FORMS for s in SEGMENTS]
    configs += [(f, s, k, n) for f in FORMS for s in SEGMENTS for k in KAPPAS for n in SUPPORTS]
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    with ProcessPoolExecutor(args.workers) as pool:
        done = list(pool.map(run_one, configs))
    errors = {(e['form'], e['segment'], e['kappa'], e['support']): v for e, v in done}
    base = np.mean((np.asarray(np.load(UNSEEN / 'drafts' / 'raw_m6default_eval.npz')['draft'])[:, :, :7] - Shared.target) ** 2, axis=(1, 2))
    entries = []
    for entry, error in done:
        entry['minus_fixed_raw_m6_all'] = paired_interval(error, base, ukey)
        entry['minus_fixed_raw_m6_hard'] = paired_interval(error[Shared.hard], base[Shared.hard], ukey[Shared.hard])
        same = errors.get(('as_is', entry['segment'], entry['kappa'], entry['support']))
        entry['minus_as_is_all'] = paired_interval(error, same, ukey)
        entry['minus_as_is_hard'] = paired_interval(error[Shared.hard], same[Shared.hard], ukey[Shared.hard])
        entries.append(entry)
    args.output.write_text(json.dumps({'rows': len(ev), 'hard_rows': int(Shared.hard.sum()), 'results': entries}, ensure_ascii=False, indent=2) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
