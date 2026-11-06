#!/usr/bin/env python3
"""Round-trip error of the demo cutting rule, on the rows of draft_reconstruction.

Each episode is cut with segment_demo.cut, each segment is reduced to k temporal DCT
coefficients per motion axis plus one gripper value, and the episode is rebuilt. No
denoising and no retrieval. The comparison keeps each episode's segment count and its
gripper cuts, but places the remaining cuts by halving the longest segment, so the
difference between the two isolates where the error-driven cuts land.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from draft_reconstruction import ROOT, RUN, TOPK, UNSEEN, summarize  # noqa: E402
from segment_demo import cut, low_freq  # noqa: E402

EPSILONS = (.005, .01, .02, .04)
COEFFICIENTS = (2, 3, 4, 5)
BOOTSTRAP = 5000


def stitch(actions, episode, frame):
    """Join stride-10 windows into per-episode traces; returns traces and row locations."""
    traces, where = {}, np.empty(len(actions), dtype=object)
    for key in np.unique(episode, axis=0):
        ix = np.flatnonzero((episode == key).all(1))
        ix = ix[np.argsort(frame[ix])]
        first, width = int(frame[ix[0]]), actions.shape[1]
        if not np.array_equal(frame[ix], first + width * np.arange(len(ix))):
            raise AssertionError(f'episode {tuple(key)} has a gap or an overlap')
        traces[tuple(key.tolist())] = np.concatenate(np.asarray(actions[ix])[:, :, :7])
        for pos, row in enumerate(ix):
            where[row] = (tuple(key.tolist()), pos * width)
    return traces, where


def equal_cuts(actions, count):
    """Keep every gripper cut, then halve the longest segment until there are `count`."""
    sign = np.sign(actions[:, 6])
    bounds = sorted({0, len(actions), *(np.flatnonzero(sign[1:] != sign[:-1]) + 1).tolist()})
    while len(bounds) - 1 < count:
        i = int(np.argmax(np.diff(bounds)))
        if bounds[i + 1] - bounds[i] < 2:
            break
        bounds.insert(i + 1, (bounds[i] + bounds[i + 1]) // 2)
    return list(zip(bounds[:-1], bounds[1:]))


def rebuild(actions, spans, k):
    """Rebuild a trace from k DCT values per motion axis and one gripper value per span."""
    out = np.empty_like(actions, dtype=np.float64)
    values = 0
    for start, end in spans:
        out[start:end, :6] = low_freq(actions[start:end, :6], k)
        out[start:end, 6] = actions[start:end, 6].mean()
        values += 6 * min(k, end - start) + 2  # the gripper value and the length
    return out, values


def rows_from(traces, where, ids):
    return np.stack([traces[where[i][0]][where[i][1]:where[i][1] + 10] for i in ids])


def paired_interval(a, b, groups, seed=0):
    """Episode bootstrap 95% interval of mean(a - b)."""
    labels, inverse = np.unique(groups, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    diff = a - b
    total, count = np.bincount(inverse, diff), np.bincount(inverse)
    draws = np.random.default_rng(seed).integers(0, len(labels), (BOOTSTRAP, len(labels)))
    means = total[draws].sum(1) / count[draws].sum(1)
    return float(diff.mean()), float(np.quantile(means, .025)), float(np.quantile(means, .975))


def evaluate(traces, where, eval_rows, groups, hard):
    target = rows_from(traces, where, eval_rows)
    frames = sum(len(a) for a in traces.values())

    def score(spans_by_episode, k):
        rebuilt, values = {}, 0
        for key, a in traces.items():
            rebuilt[key], v = rebuild(a, spans_by_episode[key], k)
            values += v
        pred = rows_from(rebuilt, where, eval_rows)
        return pred, {'values_per_10_frames': 10 * values / frames, **summarize(pred, target, hard)}

    def row_error(pred):
        return np.mean((pred - target) ** 2, axis=(1, 2))

    out = {'frames': frames, 'episodes': len(traces), 'grip_only': {}, 'levels': []}
    gripper = {key: equal_cuts(a, 0) for key, a in traces.items()}
    for k in COEFFICIENTS:
        out['grip_only'][f'DCT {k}'] = {'segments': sum(map(len, gripper.values())), **score(gripper, k)[1]}
    for epsilon in EPSILONS:
        for k in COEFFICIENTS:
            found = {key: cut(a, epsilon=epsilon, k=k) for key, a in traces.items()}
            spans = {key: [(s['start'], s['end']) for s in v] for key, v in found.items()}
            equal = {key: equal_cuts(traces[key], len(v)) for key, v in spans.items()}
            if any(len(equal[key]) != len(spans[key]) for key in spans):
                raise AssertionError('the equal cuts differ in segment count')
            lengths = np.array([e - s for v in spans.values() for s, e in v])
            pred_var, var = score(spans, k)
            pred_eq, eq = score(equal, k)
            e_var, e_eq = row_error(pred_var), row_error(pred_eq)
            out['levels'].append({
                'epsilon': epsilon, 'k': k, 'segments': int(len(lengths)),
                'length_median': float(np.median(lengths)), 'length_p10': float(np.quantile(lengths, .1)),
                'length_p90': float(np.quantile(lengths, .9)),
                'short_under_5': float(np.mean(lengths < 5)), 'variable': var, 'equal': eq,
                'variable_minus_equal_all': paired_interval(e_var, e_eq, groups),
                'variable_minus_equal_hard': paired_interval(e_var[hard], e_eq[hard], groups[hard])})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=ROOT / 'unseen-libero' / 'draft-reconstruction-01' / 'variable_length.json')
    args = ap.parse_args()
    rows = {k: np.load(RUN / 'rows' / f'{k}.npy', mmap_mode='r') for k in ('actions', 'task', 'episode', 'frame')}
    z = np.load(TOPK, allow_pickle=False)
    long_rows = np.flatnonzero(np.asarray(rows['task']) < 10)
    long_key = np.stack([np.asarray(rows['task'])[long_rows], np.asarray(rows['episode'])[long_rows]], 1)
    all_where = np.empty(len(rows['task']), dtype=object)
    traces, where = stitch(np.asarray(rows['actions'][long_rows]), long_key, np.asarray(rows['frame'])[long_rows])
    all_where[long_rows] = where
    ids, hard = z['query_rows'], z['hard_mask'].astype(bool)
    groups = np.stack([np.asarray(rows['task'])[ids], np.asarray(rows['episode'])[ids]], 1)
    result = {'long': {'rows': len(ids), 'hard_rows': int(hard.sum()), **evaluate(traces, all_where, ids, groups, hard)}}

    un = {k: np.load(UNSEEN / 'unseen_rows' / f'{k}.npy', mmap_mode='r') for k in ('actions', 'task', 'episode', 'frame', 'is_dev')}
    ev = np.flatnonzero(~np.asarray(un['is_dev']))
    raw = np.load(UNSEEN / 'drafts' / 'raw_none_eval.npz', allow_pickle=False)
    if not np.array_equal(raw['query_ids'], ev + len(rows['task'])):
        raise AssertionError('unseen eval rows differ from the draft cache')
    threshold = json.loads((UNSEEN / 'hard_threshold.json').read_text())['threshold']
    hard = raw['pre_mse'] >= threshold
    key = np.stack([np.asarray(un['task']), np.asarray(un['episode'])], 1)
    traces, where = stitch(np.asarray(un['actions']), key, np.asarray(un['frame']))
    result['unseen'] = {'rows': len(ev), 'hard_rows': int(hard.sum()), **evaluate(traces, where, ev, key[ev], hard)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
