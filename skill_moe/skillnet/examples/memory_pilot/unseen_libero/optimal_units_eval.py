#!/usr/bin/env python3
"""Rebuild unseen demos from stored units with the best sequence under a selection budget.

For a budget of one selection per `frames` frames, dynamic programming finds the unit
sequence with the lowest total error over each demo. The correct actions choose the units
and score the result; they place no cuts other than through the chosen units' lengths, and
give no direction or size. Scores: frame-wise MSE, and DTW with a band that tolerates
timing shifts. Nothing is denoised and no retrieval is involved.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'pi05_memory')]
import dct_skill_draft_eval as ev  # noqa: E402
import stored_units_eval as stored  # noqa: E402
import variable_shape_memory as vs  # noqa: E402
from draft_reconstruction import ROOT, summarize  # noqa: E402
from variable_length_error import paired_interval  # noqa: E402

SEGMENTS = ('fixed10', 'e0.005', 'e0.02')
LEVELS = (None, .04, .08, .16)
STRETCHES = {'none': (1.,), 'free': (.5, .75, 1., 1.5, 2.)}
FRAMES_PER_SELECTION = (5, 10, 20)
BAND = 5


def stretched_units(memory, factors):
    """{length: [units,length,7] drafts}. Stretching keeps the path: slower means smaller actions."""
    groups = {}
    for unit, own in enumerate(memory.lengths):
        for factor in factors:
            length = max(1, int(round(int(own) * factor)))
            ratio = length / int(own)
            groups.setdefault((int(own), length, ratio), []).append(unit)
    drafts = {}
    for (own, length, ratio), units in groups.items():
        d = vs.decode(memory.values[units], length, 'as_is', None, memory.grip_k)
        if ratio != 1.:
            d[:, :, :6] = vs.to_normalized(vs.to_physical(d[:, :, :6]) / ratio)
        drafts.setdefault(length, []).append(d)
    return {length: np.concatenate(parts) for length, parts in drafts.items()}


def best_cost_table(trace, drafts):
    """For each start frame and unit length: the lowest summed error and its draft."""
    table = {}
    n = len(trace)
    for start in range(n):
        for length, candidates in drafts.items():
            covered = min(length, n - start)
            errors = np.sum((candidates[:, :covered] - trace[start:start + covered][None]) ** 2, axis=(1, 2))
            i = int(np.argmin(errors))
            table[start, length] = float(errors[i]), candidates[i, :covered]
    return table


def best_sequence(table, lengths, frames, budget):
    """Lowest-error cover of `frames` with at most `budget` units; returns (cost, [(start,length)])."""
    cost = np.full((budget + 1, frames + 1), np.inf)
    cost[0, 0] = 0.
    back = {}
    for used in range(budget):
        for start in np.flatnonzero(np.isfinite(cost[used, :frames])):
            for length in lengths:
                stop = min(start + length, frames)
                value = cost[used, start] + table[int(start), length][0]
                if value < cost[used + 1, stop]:
                    cost[used + 1, stop] = value
                    back[used + 1, stop] = (used, int(start), length)
    finals = cost[:, frames]
    used = int(np.argmin(finals))
    if not np.isfinite(finals[used]):
        return np.inf, []
    sequence, stop = [], frames
    while used > 0:
        prior, start, length = back[used, stop]
        sequence.append((start, length))
        used, stop = prior, start
    return float(finals.min()), sequence[::-1]


def dtw_frame_errors(pred, target, band=BAND):
    """Per-frame squared error over active axes after alignment within a band."""
    n = len(target)
    local = np.full((n, n), np.inf)
    for i in range(n):
        lo, hi = max(0, i - band), min(n, i + band + 1)
        local[i, lo:hi] = np.mean((pred[lo:hi] - target[i]) ** 2, axis=1)
    total = np.full((n, n), np.inf)
    total[0, 0] = local[0, 0]
    for i in range(n):
        for j in range(max(0, i - band), min(n, i + band + 1)):
            if i == 0 and j == 0:
                continue
            prior = min(total[i - 1, j] if i else np.inf, total[i, j - 1] if j else np.inf,
                        total[i - 1, j - 1] if i and j else np.inf)
            total[i, j] = local[i, j] + prior
    out = np.full(n, np.inf)
    i, j = n - 1, n - 1
    while True:
        out[i] = min(out[i], local[i, j])
        if i == 0 and j == 0:
            break
        moves = [(total[a, b], a, b) for a, b in ((i - 1, j - 1), (i - 1, j), (i, j - 1)) if a >= 0 and b >= 0]
        _, i, j = min(moves)
    return out


def run_one(config):
    segment, level, stretch = config
    S = ev.Shared
    memory = stored.make_memory(segment, level)
    drafts = stretched_units(memory, STRETCHES[stretch])
    lengths = sorted(drafts)
    tables = {key: best_cost_table(trace, drafts) for key, trace in S.traces.items()}
    out, errors = [], {}
    for per in FRAMES_PER_SELECTION:
        pred = np.empty_like(S.target, dtype=np.float64)
        aligned = np.empty(len(S.target))
        selections, frames, chosen_lengths, jumps = 0, 0, [], []
        feasible = True
        for key, trace in S.traces.items():
            budget = max(1, math.ceil(len(trace) / per))
            _, sequence = best_sequence(tables[key], lengths, len(trace), budget)
            if not sequence:
                feasible = False
                break
            joined = np.empty_like(trace, dtype=np.float64)
            for start, length in sequence:
                piece = tables[key][start, length][1]
                joined[start:start + len(piece)] = piece
            selections += len(sequence)
            frames += len(trace)
            chosen_lengths += [length for _, length in sequence]
            jumps += [float(np.mean((joined[s, :6] - joined[s - 1, :6]) ** 2)) for s, _ in sequence[1:]]
            rows = S.rows_of[key]
            pred[rows] = joined.reshape(-1, 10, 7)
            per_frame = dtw_frame_errors(joined, trace)
            aligned[rows] = per_frame.reshape(-1, 10).mean(1)
        if not feasible:
            continue
        mse = np.mean((pred - S.target) ** 2, axis=(1, 2))
        entry = {'segment': segment, 'level': level, 'stretch': stretch, 'budget_frames': per,
                 'units': len(memory.values), 'selections': selections, 'frames_per_selection': frames / selections,
                 'length_median': float(np.median(chosen_lengths)), 'jump': float(np.mean(jumps)) if jumps else None,
                 'dtw_all': float(aligned.mean()), 'dtw_hard': float(aligned[S.hard].mean()),
                 **summarize(pred, S.target, S.hard)}
        out.append(entry)
        errors[per] = (mse, aligned)
        print(json.dumps({k: entry[k] for k in ('segment', 'level', 'stretch', 'budget_frames', 'units', 'frames_per_selection', 'all', 'hard', 'dtw_all', 'dtw_hard')}), flush=True)
    return out, errors


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=ROOT / 'unseen-libero' / 'draft-reconstruction-01' / 'optimal_units.json')
    ap.add_argument('--workers', type=int, default=16)
    args = ap.parse_args()
    ukey = stored.load_data()
    hard = ev.Shared.hard
    configs = [(s, l, t) for s in SEGMENTS for l in LEVELS for t in STRETCHES]
    with ProcessPoolExecutor(args.workers) as pool:
        done = list(pool.map(run_one, configs))
    errors = {}
    results = []
    for config, (entries, part) in zip(configs, done):
        for per, value in part.items():
            errors[(*config, per)] = value
        results += entries
    for e in results:
        base = errors.get(('fixed10', e['level'], e['stretch'], e['budget_frames']))
        if base is None or e['segment'] == 'fixed10':
            continue
        mine = errors[(e['segment'], e['level'], e['stretch'], e['budget_frames'])]
        for i, name in enumerate(('mse', 'dtw')):
            e[f'minus_fixed_{name}_all'] = paired_interval(mine[i], base[i], ukey)
            e[f'minus_fixed_{name}_hard'] = paired_interval(mine[i][hard], base[i][hard], ukey[hard])
    args.output.write_text(json.dumps({'rows': len(ev.Shared.target), 'hard_rows': int(hard.sum()), 'results': results}, ensure_ascii=False, indent=2) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
