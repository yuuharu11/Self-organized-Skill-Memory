#!/usr/bin/env python3
"""Rebuild unseen demos from stored units only, choosing the best unit at each step.

The correct actions choose the unit and score the result. They do not place cuts and do
not supply direction or size: a unit plays back as stored, at its own length, and the next
unit is chosen when it ends. Nothing is denoised and no retrieval is involved.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'pi05_memory')]
import dct_skill_draft_eval as ev  # noqa: E402
from dct_skill_draft import SkillMemory  # noqa: E402
from draft_reconstruction import ROOT, RUN, UNSEEN, summarize  # noqa: E402
from oracle_repr import m6_default_medoids  # noqa: E402
from variable_length_error import paired_interval, stitch  # noqa: E402
from variable_shape_memory import spans_of as ev_spans  # noqa: E402

# None keeps every segment; 'm6' uses the existing registration rule.
LEVELS = (None, .04, .08, .16, 'm6')
SEGMENTS = ('fixed10', 'e0.005', 'e0.02')
SELECT_TAU = {None: .04, 'm6': .08}
FIXED_CACHES = ('raw', 'b12', 'pca24', 'dct21')


def make_memory(segment, level):
    traces = list(ev.Shared.memory.values())
    if level is None:
        return SkillMemory.segments(traces, segment=segment, form='as_is')[0]
    if level != 'm6':
        return SkillMemory.build(traces, segment=segment, tau=level, form='as_is')
    keys = []
    for key, trace in ev.Shared.memory.items():
        keys += [key] * len(ev_spans(trace, segment))
    keys = np.asarray(keys)
    memory = SkillMemory.segments(traces, segment=segment, form='as_is')[0]
    fit = m6_default_medoids(memory.values.astype(np.float32), keys[:, 0], keys[:, 1], np.arange(len(keys)))
    return memory.subset(fit['medoid_local'])


def run_one(config):
    segment, level = config
    memory = make_memory(segment, level)
    tau = SELECT_TAU.get(level, level)
    pred = np.empty_like(ev.Shared.target, dtype=np.float64)
    selections, frames, jumps, demo_jumps = 0, 0, [], []
    for key, trace in ev.Shared.traces.items():
        drafts, used, chosen = ev.roll(memory, trace, tau, None)
        pred[ev.Shared.rows_of[key]] = drafts
        selections += chosen
        frames += len(trace)
        joined = drafts.reshape(-1, 7)
        for _, first in used[1:]:
            if 0 < first < len(joined):
                jumps.append(np.mean((joined[first, :6] - joined[first - 1, :6]) ** 2))
                demo_jumps.append(np.mean((trace[first, :6] - trace[first - 1, :6]) ** 2))
    entry = {'segment': segment, 'level': level, 'select_tau': tau, 'units': len(memory.values),
             'values_per_unit': int(memory.values.shape[1]), 'selections': selections,
             'frames_per_selection': frames / selections, 'jump': float(np.mean(jumps)),
             'jump_in_demo': float(np.mean(demo_jumps)), **summarize(pred, ev.Shared.target, ev.Shared.hard)}
    print(json.dumps({k: entry[k] for k in ('segment', 'level', 'units', 'frames_per_selection', 'all', 'hard')}), flush=True)
    return entry, np.mean((pred - ev.Shared.target) ** 2, axis=(1, 2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=ROOT / 'unseen-libero' / 'draft-reconstruction-01' / 'stored_units.json')
    args = ap.parse_args()
    rows = {k: np.load(RUN / 'rows' / f'{k}.npy', mmap_mode='r') for k in ('actions', 'task', 'episode', 'frame', 'memory')}
    memory = np.flatnonzero(np.asarray(rows['memory'], bool))
    key = np.stack([np.asarray(rows['task'])[memory], np.asarray(rows['episode'])[memory]], 1)
    S = ev.Shared
    S.memory, _ = stitch(np.asarray(rows['actions'][memory]), key, np.asarray(rows['frame'])[memory])
    un = {k: np.load(UNSEEN / 'unseen_rows' / f'{k}.npy', mmap_mode='r') for k in ('actions', 'task', 'episode', 'frame', 'is_dev')}
    rows_ev = np.flatnonzero(~np.asarray(un['is_dev']))
    raw = np.load(UNSEEN / 'drafts' / 'raw_none_eval.npz', allow_pickle=False)
    if not np.array_equal(raw['query_ids'], rows_ev + len(rows['task'])):
        raise AssertionError('unseen eval rows differ from the draft cache')
    S.hard = raw['pre_mse'] >= json.loads((UNSEEN / 'hard_threshold.json').read_text())['threshold']
    ukey = np.stack([np.asarray(un['task']), np.asarray(un['episode'])], 1)[rows_ev]
    frame = np.asarray(un['frame'])[rows_ev]
    S.traces, _ = stitch(np.asarray(un['actions'])[rows_ev], ukey, frame)
    S.rows_of = {k: np.flatnonzero((ukey == k).all(1))[np.argsort(frame[(ukey == k).all(1)])] for k in S.traces}
    S.target = np.asarray(un['actions'])[rows_ev][:, :, :7].astype(np.float64)

    caches, base = {}, {}
    for name in FIXED_CACHES:
        for level in ('none', 'm6default'):
            pred = np.asarray(np.load(UNSEEN / 'drafts' / f'{name}_{level}_eval.npz')['draft'])[:, :, :7]
            info = json.loads((UNSEEN / f'fit-{name}-{level}.json').read_text())
            caches[f'{name} {level}'] = {'units': info['unit_count'], **summarize(pred, S.target, S.hard)}
            base[f'{name} {level}'] = np.mean((pred - S.target) ** 2, axis=(1, 2))
    normal = np.load(UNSEEN / 'official_normal_eval.npz', allow_pickle=False)
    caches['normal pi05'] = summarize(np.asarray(normal['prediction'])[:, :, :7], S.target, S.hard)

    configs = [(s, l) for s in SEGMENTS for l in LEVELS]
    with ProcessPoolExecutor(8) as pool:
        done = list(pool.map(run_one, configs))
    results = []
    for entry, error in done:
        entry['minus_raw_none_all'] = paired_interval(error, base['raw none'], ukey)
        entry['minus_raw_none_hard'] = paired_interval(error[S.hard], base['raw none'][S.hard], ukey[S.hard])
        entry['minus_raw_m6_all'] = paired_interval(error, base['raw m6default'], ukey)
        entry['minus_raw_m6_hard'] = paired_interval(error[S.hard], base['raw m6default'][S.hard], ukey[S.hard])
        results.append(entry)
    args.output.write_text(json.dumps({'rows': len(rows_ev), 'hard_rows': int(S.hard.sum()), 'caches': caches, 'results': results}, ensure_ascii=False, indent=2) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
