#!/usr/bin/env python3
"""Merge DCT segments under the same error tolerance used to cut them.

The error of rebuilding segment i from unit j, with i's own direction and size, follows
directly from the DCT values because the transform is orthonormal. Units are chosen
greedily: take the segment that newly covers the most segments within tau, until every
segment is covered. The unit count follows from tau and no segment is discarded.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import variable_shape_memory as vs  # noqa: E402
from draft_reconstruction import ROOT, RUN, UNSEEN, UNSEEN_ROWS  # noqa: E402
from variable_length_error import rows_from, stitch  # noqa: E402

TAUS = (.005, .01, .02, .04, .08, .16)
FORMS = ('as_is', 'rms')
SEGMENTS = ('fixed10', 'e0.005', 'e0.02')


def pose_matrix(motion, form):
    """Map a shape difference to a normalized action difference."""
    if form == 'as_is':
        return np.eye(6)
    rotation, scale = vs.pose_of(motion, form)
    block = np.zeros((6, 6))
    block[:3, :3] = block[3:, 3:] = rotation
    return block * scale @ np.diag(2 / vs.SPAN)


def rebuild_errors(values, matrices, grip_k):
    """E[i,j]: active-7 MSE of rebuilding segment i from segment j's values."""
    n = len(values)
    k = vs.K
    motion = values[:, :k * 6].reshape(n, k, 6)
    errors = np.empty((n, n), np.float32)
    for i in range(n):
        delta = (motion - motion[i]) @ matrices[i]
        grip = values[:, k * 6:] - values[i, k * 6:]
        errors[i] = (np.sum(delta ** 2, axis=(1, 2)) + np.sum(grip ** 2, axis=1)) / 7
    return errors


def greedy_cover(within):
    """Pick units until every row is covered; returns unit indices and how many each newly took."""
    n = len(within)
    counts = within.sum(0).astype(np.int64)
    uncovered = np.ones(n, bool)
    units, sizes = [], []
    while uncovered.any():
        j = int(np.argmax(counts))
        fresh = uncovered & within[:, j]
        units.append(j)
        sizes.append(int(fresh.sum()))
        counts -= within[fresh].sum(0)
        uncovered &= ~fresh
    return np.asarray(units), np.asarray(sizes)


class Shared:
    """Data the worker processes inherit."""


def run_one(config):
    form, segment = config
    grip_k = vs.grip_values(segment)
    values, matrices = [], []
    for trace in Shared.memory.values():
        for start, end in vs.spans_of(trace, segment):
            values.append(vs.encode(trace, start, end, form, grip_k))
            matrices.append(pose_matrix(trace[start:end, :6], form))
    values, matrices = np.stack(values), np.stack(matrices)
    errors = rebuild_errors(values, matrices, grip_k)
    out = []
    for tau in TAUS:
        units, sizes = greedy_cover(errors <= tau)
        result, _ = vs.score(values[units], form, segment, Shared.traces, Shared.where, Shared.positions, Shared.target, Shared.hard)
        sorted_sizes = np.sort(sizes)[::-1]
        out.append({'form': form, 'segment': segment, 'tau': tau, 'segments': len(values), 'units': len(units),
                    'singletons': int(np.sum(sizes == 1)), 'units_with_10': int(np.sum(sizes >= 10)),
                    'covered_by_units_with_10': float(sizes[sizes >= 10].sum() / len(values)),
                    'largest_unit': int(sorted_sizes[0]), **result})
        print(form, segment, tau, len(units), round(out[-1]['all'], 4), round(out[-1]['hard'], 4), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=ROOT / 'unseen-libero' / 'draft-reconstruction-01' / 'dct_tolerance_merge.json')
    args = ap.parse_args()
    rows = {k: np.load(RUN / 'rows' / f'{k}.npy', mmap_mode='r') for k in ('actions', 'task', 'episode', 'frame', 'memory')}
    memory = np.flatnonzero(np.asarray(rows['memory'], bool))
    key = np.stack([np.asarray(rows['task'])[memory], np.asarray(rows['episode'])[memory]], 1)
    Shared.memory, _ = stitch(np.asarray(rows['actions'][memory]), key, np.asarray(rows['frame'])[memory])
    un = {k: np.load(UNSEEN_ROWS / f'{k}.npy', mmap_mode='r') for k in ('actions', 'task', 'episode', 'frame', 'is_dev')}
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
    configs = [(f, s) for f in FORMS for s in SEGMENTS]
    with ProcessPoolExecutor(len(configs)) as pool:
        results = [e for part in pool.map(run_one, configs) for e in part]
    args.output.write_text(json.dumps({'rows': len(ev), 'hard_rows': int(Shared.hard.sum()), 'results': results}, ensure_ascii=False, indent=2) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
