#!/usr/bin/env python3
"""Compare LIBERO-40 motion memories on unseen rows, choosing by the correct action.

Fixed memories come from the 10-frame oracle caches. Variable memories cut the same
memory demos with segment_demo.cut; each segment keeps three DCT values per motion axis
and one gripper value, divided by sqrt(length) so that it can be stretched to any length.
A query is an unseen stretch of frames, replaced by the memory segment closest to it.
Nothing is denoised and no retrieval is involved.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.fft import dct, idct

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'pi05_memory')]
from draft_reconstruction import ROOT, RUN, UNSEEN, summarize  # noqa: E402
from oracle_repr import m6_default_medoids  # noqa: E402
from segment_demo import cut  # noqa: E402
from variable_length_error import paired_interval, rows_from, stitch  # noqa: E402

EPSILONS = (.005, .01, .02, .04)
K = 3
FIXED = ('raw', 'dct21', 'b12', 'pca24')


def segment_values(trace, start, end):
    """[K*6 motion values, gripper] of one segment, independent of its length."""
    a = trace[start:end]
    c = np.zeros((K, 6), np.float64)
    full = dct(np.asarray(a[:, :6], np.float64), type=2, norm='ortho', axis=0)
    c[:min(K, len(a))] = full[:K]
    return np.r_[c.ravel() / np.sqrt(len(a)), a[:, 6].mean()]


def decode(values, length):
    """Stretch [M,K*6+1] values to [M,length,7]."""
    v = np.atleast_2d(values)
    c = np.zeros((len(v), length, 6), np.float64)
    keep = min(K, length)
    c[:, :keep] = v[:, :K * 6].reshape(-1, K, 6)[:, :keep] * np.sqrt(length)
    out = np.empty((len(v), length, 7), np.float64)
    out[:, :, :6] = idct(c, type=2, norm='ortho', axis=1)
    out[:, :, 6] = v[:, -1, None]
    return out


def memory_segments(traces, epsilon):
    values, keys, lengths = [], [], []
    for key, trace in traces.items():
        for s in cut(trace, epsilon=epsilon, k=K):
            values.append(segment_values(trace, s['start'], s['end']))
            keys.append(key)
            lengths.append(s['length'])
    return np.stack(values), np.asarray(keys), np.asarray(lengths)


def replace_with_nearest(trace, spans, candidates):
    out = np.empty_like(trace, dtype=np.float64)
    picks = []
    for start, end in spans:
        drafts = decode(candidates, end - start)
        errors = np.mean((drafts - trace[start:end][None]) ** 2, axis=(1, 2))
        best = int(np.argmin(errors))
        out[start:end] = drafts[best]
        picks.append(best)
    return out, picks


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=ROOT / 'unseen-libero' / 'draft-reconstruction-01' / 'variable_memory.json')
    args = ap.parse_args()
    rows = {k: np.load(RUN / 'rows' / f'{k}.npy', mmap_mode='r') for k in ('actions', 'task', 'episode', 'frame', 'memory')}
    memory = np.flatnonzero(np.asarray(rows['memory'], bool))
    fit = json.loads((UNSEEN / 'fit-raw-none.json').read_text())
    if len(memory) != fit['unit_count']:
        raise AssertionError('the memory rows differ from the fixed oracle pool')
    key = np.stack([np.asarray(rows['task'])[memory], np.asarray(rows['episode'])[memory]], 1)
    memory_traces, _ = stitch(np.asarray(rows['actions'][memory]), key, np.asarray(rows['frame'])[memory])

    un = {k: np.load(UNSEEN / 'unseen_rows' / f'{k}.npy', mmap_mode='r') for k in ('actions', 'task', 'episode', 'frame', 'is_dev')}
    ev = np.flatnonzero(~np.asarray(un['is_dev']))
    raw = np.load(UNSEEN / 'drafts' / 'raw_none_eval.npz', allow_pickle=False)
    if not np.array_equal(raw['query_ids'], ev + len(rows['task'])):
        raise AssertionError('unseen eval rows differ from the draft cache')
    threshold = json.loads((UNSEEN / 'hard_threshold.json').read_text())['threshold']
    hard = raw['pre_mse'] >= threshold
    ukey = np.stack([np.asarray(un['task']), np.asarray(un['episode'])], 1)
    traces, where = stitch(np.asarray(un['actions'])[ev], ukey[ev], np.asarray(un['frame'])[ev])
    # stitch numbers rows inside the eval subset, which is what rows_from receives below.
    positions = np.arange(len(ev))
    target = rows_from(traces, where, positions)
    groups = ukey[ev]
    frames = sum(len(t) for t in traces.values())

    fixed = {}
    for name in FIXED:
        for level in ('none', 'm6default'):
            z = np.load(UNSEEN / 'drafts' / f'{name}_{level}_eval.npz', allow_pickle=False)
            info = json.loads((UNSEEN / f'fit-{name}-{level}.json').read_text())
            per_unit = {'raw': 70, 'dct21': 21, 'b12': 12, 'pca24': 24}[name]
            pred = np.asarray(z['draft'])[:, :, :7]
            if not np.allclose(np.mean((pred - target) ** 2, axis=(1, 2)), z['pre_mse'], atol=1e-4):
                raise AssertionError(f'{name} {level}: the cache does not match the target rows')
            fixed[f'{name} {level}'] = {'units': info['unit_count'], 'values_per_unit': per_unit,
                                        'stored_values': info['unit_count'] * per_unit,
                                        'unassigned': info['unassigned_rate'],
                                        **summarize(pred, target, hard)}
    base = {k: np.mean((np.asarray(np.load(UNSEEN / 'drafts' / f'raw_{k}_eval.npz')['draft'])[:, :, :7] - target) ** 2, axis=(1, 2))
            for k in ('none', 'm6default')}

    variable = []
    for epsilon in EPSILONS:
        values, keys, lengths = memory_segments(memory_traces, epsilon)
        fit = m6_default_medoids(values.astype(np.float32), keys[:, 0], keys[:, 1], np.arange(len(values)))
        pools = {'none': values, 'm6default': values[fit['medoid_local']]}
        for level, candidates in pools.items():
            for query in ('own_cuts', 'every_10_frames'):
                rebuilt, selections = {}, 0
                used = []
                for ep, trace in traces.items():
                    if query == 'own_cuts':
                        spans = [(s['start'], s['end']) for s in cut(trace, epsilon=epsilon, k=K)]
                    else:
                        spans = [(i, i + 10) for i in range(0, len(trace), 10)]
                    rebuilt[ep], picks = replace_with_nearest(trace, spans, candidates)
                    selections += len(picks)
                    used += picks
                pred = rows_from(rebuilt, where, positions)
                e = np.mean((pred - target) ** 2, axis=(1, 2))
                variable.append({
                    'epsilon': epsilon, 'level': level, 'query': query, 'units': len(candidates),
                    'values_per_unit': K * 6 + 1, 'stored_values': len(candidates) * (K * 6 + 1),
                    'memory_segments': len(values), 'memory_length_median': float(np.median(lengths)),
                    'unassigned': fit['unassigned_rate'] if level != 'none' else 0.,
                    'selections_per_10_frames': 10 * selections / frames,
                    'units_used': len(set(used)), **summarize(pred, target, hard),
                    'minus_raw_same_level_all': paired_interval(e, base[level], groups),
                    'minus_raw_same_level_hard': paired_interval(e[hard], base[level][hard], groups[hard])})
                print(epsilon, level, query, len(candidates), round(variable[-1]['all'], 4), round(variable[-1]['hard'], 4), flush=True)
    result = {'rows': len(ev), 'hard_rows': int(hard.sum()), 'frames': frames, 'fixed': fixed, 'variable': variable}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
