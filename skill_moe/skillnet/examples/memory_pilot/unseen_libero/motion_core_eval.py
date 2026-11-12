#!/usr/bin/env python3
"""Try ways to define motion units before denoising, using stored units only.

    frame    world, or the end-effector frame of the state when a unit starts
    gripper  joint (in the unit and in the distance), or separate (motion alone)
    cut      fixed10, the error rule, speed minima with gripper switches, or re-segmented
             over the whole memory with the units themselves

The correct actions of an unseen demo only choose units and score the result; the state
used for the end-effector frame is known at run time. No retrieval and no denoising.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.fft import dct, idct
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'pi05_memory'), str(HERE.parent / 'pi05_memory' / 'shape_skills')]
import dct_skill_draft_eval as ev  # noqa: E402
import dct_tolerance_merge as merge  # noqa: E402
import optimal_units_eval as opt  # noqa: E402
import stored_units_eval as stored  # noqa: E402
import variable_shape_memory as vs  # noqa: E402
from draft_reconstruction import ROOT, RUN, UNSEEN, UNSEEN_ROWS  # noqa: E402
from segment_demo import cut  # noqa: E402

K = 3
EPSILON = .005
MIN_LENGTH = 5
LEVELS = (None, .04, .08, .16)
STATE_Q01 = np.array([-.3524, -.2682, .0408, 1.5318, -2.7152, -1.0765])
STATE_Q99 = np.array([.1389, .3252, 1.2569, 3.2628, 2.4437, .5638])
# (cut, frame, gripper)
CONFIGS = [('fixed10', 'world', 'joint'), ('fixed10', 'world', 'separate'), ('fixed10', 'ee', 'joint'), ('fixed10', 'ee', 'separate'),
           ('error', 'world', 'joint'), ('error', 'world', 'separate'), ('error', 'ee', 'joint'), ('error', 'ee', 'separate'),
           ('anchor', 'world', 'joint'), ('anchor', 'ee', 'joint'), ('resegment', 'world', 'joint')]
RUNS = ((10, False), (20, True))  # (frames per selection, stretching allowed)


def rotations_from_state(state):
    """World-from-end-effector rotation for each row of normalized state."""
    orientation = (state[:, :6] + 1) * .5 * (STATE_Q99 - STATE_Q01 + 1e-6) + STATE_Q01
    return Rotation.from_rotvec(orientation[:, 3:6]).as_matrix()


def to_frame(motion, rotation):
    """Normalized world motion to normalized motion in the given frame."""
    p = vs.to_physical(motion)
    return vs.to_normalized(np.concatenate([p[..., :3] @ rotation, p[..., 3:6] @ rotation], -1))


def from_frame(motion, rotation):
    p = vs.to_physical(motion)
    return vs.to_normalized(np.concatenate([p[..., :3] @ rotation.T, p[..., 3:6] @ rotation.T], -1))


def anchor_spans(trace, min_length=MIN_LENGTH):
    """Cut at gripper switches and at local minima of smoothed speed."""
    speed = np.linalg.norm(vs.to_physical(trace[:, :6])[:, :3], axis=1)
    smooth = np.convolve(speed, np.ones(3) / 3, mode='same')
    n = len(trace)
    minima = [t for t in range(1, n - 1) if smooth[t] <= smooth[t - 1] and smooth[t] < smooth[t + 1]]
    sign = np.sign(trace[:, 6])
    switches = (np.flatnonzero(sign[1:] != sign[:-1]) + 1).tolist()
    bounds = [0]
    for t in sorted(set(minima) | set(switches)):
        if t in switches or t - bounds[-1] >= min_length:
            bounds.append(t)
    if n - bounds[-1] < min_length and len(bounds) > 1 and bounds[-1] not in switches:
        bounds.pop()
    return list(zip(bounds, bounds[1:] + [n]))


def spans_for(trace, kind):
    if kind == 'fixed10':
        return [(i, i + 10) for i in range(0, len(trace), 10)]
    if kind == 'anchor':
        return anchor_spans(trace)
    return [(s['start'], s['end']) for s in cut(trace, epsilon=EPSILON, k=K)]


def encode(trace, rotations, start, end, frame, grip_k):
    a = trace[start:end]
    motion = a[:, :6] if frame == 'world' else to_frame(a[:, :6], rotations[start])
    c = np.zeros((K, 6))
    c[:min(K, len(a))] = dct(motion, type=2, norm='ortho', axis=0)[:K]
    g = np.zeros(grip_k)
    g[:min(grip_k, len(a))] = dct(a[:, 6].astype(np.float64), type=2, norm='ortho')[:grip_k]
    return np.r_[c.ravel(), g] / np.sqrt(len(a))


def decode(values, length, grip_k):
    v = np.atleast_2d(values)
    keep = min(K, length)
    c = np.zeros((len(v), length, 6))
    c[:, :keep] = v[:, :K * 6].reshape(-1, K, 6)[:, :keep] * np.sqrt(length)
    g = np.zeros((len(v), length))
    keep = min(grip_k, length)
    g[:, :keep] = v[:, K * 6:K * 6 + keep] * np.sqrt(length)
    return np.concatenate([idct(c, type=2, norm='ortho', axis=1), idct(g, type=2, norm='ortho', axis=1)[..., None]], -1)


def memory_values(segments_of, frame, grip_k):
    values, lengths = [], []
    for key, trace in ev.Shared.memory.items():
        for start, end in segments_of(key, trace):
            values.append(encode(trace, Data.memory_rotations[key], start, end, frame, grip_k))
            lengths.append(end - start)
    return np.stack(values), np.asarray(lengths)


def choose_units(values, level, gripper):
    if level is None:
        return np.arange(len(values))
    motion = values[:, :K * 6]
    grip = values[:, K * 6:] if gripper == 'joint' else np.zeros((len(values), 0))
    axes = 7 if gripper == 'joint' else 6
    within = np.empty((len(values), len(values)), bool)
    for i in range(len(values)):
        within[i] = (np.sum((motion - motion[i]) ** 2, axis=1) + np.sum((grip - grip[i]) ** 2, axis=1)) / axes <= level
    return merge.greedy_cover(within)[0]


def drafts_by_length(values, lengths, stretch, grip_k):
    groups = {}
    for u, own in enumerate(lengths):
        for factor in (opt.STRETCHES['free'] if stretch else (1.,)):
            n = max(1, int(round(int(own) * factor)))
            groups.setdefault((int(own), n), []).append(u)
    out = {}
    for (own, n), units in groups.items():
        d = decode(values[units], n, grip_k)
        if n != own:
            d[:, :, :6] = vs.to_normalized(vs.to_physical(d[:, :, :6]) * own / n)
        out.setdefault(n, []).append(d)
    return {n: np.concatenate(v) for n, v in out.items()}


def cost_table(trace, rotations, drafts, frame, axes):
    table = {}
    n = len(trace)
    for start in range(n):
        for length, candidates in drafts.items():
            covered = min(length, n - start)
            piece = candidates[:, :covered]
            if frame == 'ee':
                piece = np.concatenate([from_frame(piece[:, :, :6], rotations[start]), piece[:, :, 6:]], -1)
            errors = np.sum((piece[:, :, axes] - trace[start:start + covered][None, :, axes]) ** 2, axis=(1, 2))
            i = int(np.argmin(errors))
            table[start, length] = float(errors[i]), piece[i]
    return table


def metrics(pred, target, hard):
    """Separate motion, direction, size, integrated position and gripper."""
    motion = np.mean((pred[:, :, :6] - target[:, :, :6]) ** 2, axis=(1, 2))
    p, t = vs.to_physical(pred[:, :, :6])[:, :, :3], vs.to_physical(target[:, :, :6])[:, :, :3]
    ade = np.mean(np.linalg.norm(np.cumsum(p, 1) - np.cumsum(t, 1), axis=2), axis=1)
    fde = np.linalg.norm(p.sum(1) - t.sum(1), axis=1)
    u, v = p.sum(1), t.sum(1)
    nu, nv = np.linalg.norm(u, axis=1), np.linalg.norm(v, axis=1)
    moving = nv > 1e-3
    cos = np.sum(u * v, axis=1)[moving] / np.maximum(nu * nv, 1e-8)[moving]
    all_axes = np.mean((pred[:, :, :7] - target[:, :, :7]) ** 2, axis=(1, 2))
    out = {'all': float(all_axes.mean()), 'hard': float(all_axes[hard].mean()), 'motion_all': float(motion.mean()),
           'motion_hard': float(motion[hard].mean()), 'ade_all': float(ade.mean()), 'ade_hard': float(ade[hard].mean()),
           'fde_all': float(fde.mean()), 'fde_hard': float(fde[hard].mean()), 'direction_cos': float(cos.mean()),
           'size_ratio_median': float(np.median(nu[moving] / nv[moving])),
           'gripper_match': float(np.mean(np.sign(pred[:, :, 6]) == np.sign(target[:, :, 6])))}
    return out


def score(values, lengths, frame, gripper, grip_k, per, stretch):
    S = ev.Shared
    drafts = drafts_by_length(values, lengths, stretch, grip_k)
    keys = sorted(drafts)
    axes = np.arange(7) if gripper == 'joint' else np.arange(6)
    pred = np.empty_like(S.target, dtype=np.float64)
    selections, frames = 0, 0
    for key, trace in S.traces.items():
        table = cost_table(trace, Data.query_rotations[key], drafts, frame, axes)
        _, sequence = opt.best_sequence(table, keys, len(trace), max(1, math.ceil(len(trace) / per)))
        if not sequence:
            return None
        joined = np.empty_like(trace, dtype=np.float64)
        for start, length in sequence:
            piece = table[start, length][1]
            joined[start:start + len(piece)] = piece
        pred[S.rows_of[key]] = joined.reshape(-1, 10, 7)
        selections += len(sequence)
        frames += len(trace)
    return {'frames_per_selection': frames / selections, **metrics(pred, S.target, S.hard)}


def resegmented_spans(tau, rounds=2):
    """Alternate between units and cuts: cut each memory demo with the current units."""
    spans = {key: spans_for(trace, 'error') for key, trace in ev.Shared.memory.items()}
    axes = np.arange(7)
    for _ in range(rounds):
        values, lengths = memory_values(lambda key, trace: spans[key], 'world', 1)
        units = choose_units(values, tau, 'joint')
        drafts = drafts_by_length(values[units], lengths[units], False, 1)
        keys = sorted(drafts)
        for key, trace in ev.Shared.memory.items():
            table = cost_table(trace, Data.memory_rotations[key], drafts, 'world', axes)
            _, sequence = opt.best_sequence(table, keys, len(trace), max(1, math.ceil(len(trace) / 15)))
            if sequence:
                spans[key] = [(s, min(s + n, len(trace))) for s, n in sequence]
    return spans


class Data:
    """Rotations per frame, inherited by workers."""


def run_one(config):
    (kind, frame, gripper), level = config
    grip_k = K if kind == 'fixed10' else 1
    if kind == 'resegment':
        if level is None:
            return []
        spans = resegmented_spans(level)
        segments_of = lambda key, trace: spans[key]
    else:
        segments_of = lambda key, trace: spans_for(trace, kind)
    values, lengths = memory_values(segments_of, frame, grip_k)
    units = choose_units(values, level, gripper)
    out = []
    for per, stretch in RUNS:
        result = score(values[units], lengths[units], frame, gripper, grip_k, per, stretch)
        if result is None:
            continue
        entry = {'cut': kind, 'frame': frame, 'gripper': gripper, 'level': level, 'per': per, 'stretch': stretch,
                 'units': len(units), 'values_per_unit': K * 6 + (0 if gripper == 'separate' else grip_k) + (0 if kind == 'fixed10' else 1),
                 'length_median': float(np.median(lengths[units])), **result}
        out.append(entry)
        print(json.dumps({k: entry[k] for k in ('cut', 'frame', 'gripper', 'level', 'per', 'units', 'all', 'hard', 'motion_all', 'motion_hard')}), flush=True)
    return out


def rotations_per_frame(traces, rows_key, rows_frame, state):
    """State is known at each window start; hold it for the frames of the window."""
    out = {}
    all_rotations = rotations_from_state(state)
    for key, trace in traces.items():
        mask = np.flatnonzero((rows_key == key).all(1))
        mask = mask[np.argsort(rows_frame[mask])]
        out[key] = np.repeat(all_rotations[mask], 10, axis=0)[:len(trace)]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=ROOT / 'unseen-libero' / 'draft-reconstruction-01' / 'motion_core.json')
    ap.add_argument('--workers', type=int, default=16)
    args = ap.parse_args()
    ukey = stored.load_data()
    S = ev.Shared
    rows = {k: np.load(RUN / 'rows' / f'{k}.npy', mmap_mode='r') for k in ('state', 'task', 'episode', 'frame', 'memory')}
    memory = np.flatnonzero(np.asarray(rows['memory'], bool))
    key = np.stack([np.asarray(rows['task'])[memory], np.asarray(rows['episode'])[memory]], 1)
    Data.memory_rotations = rotations_per_frame(S.memory, key, np.asarray(rows['frame'])[memory], np.asarray(rows['state'][memory], np.float64))
    un = {k: np.load(UNSEEN_ROWS / f'{k}.npy', mmap_mode='r') for k in ('state', 'frame', 'is_dev')}
    ev_rows = np.flatnonzero(~np.asarray(un['is_dev']))
    Data.query_rotations = rotations_per_frame(S.traces, ukey, np.asarray(un['frame'])[ev_rows], np.asarray(un['state'][ev_rows], np.float64))
    jobs = [(c, l) for c in CONFIGS for l in LEVELS]
    with ProcessPoolExecutor(args.workers) as pool:
        results = [e for part in pool.map(run_one, jobs) for e in part]
    args.output.write_text(json.dumps({'rows': len(S.target), 'hard_rows': int(S.hard.sum()), 'results': results}, ensure_ascii=False, indent=2) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
