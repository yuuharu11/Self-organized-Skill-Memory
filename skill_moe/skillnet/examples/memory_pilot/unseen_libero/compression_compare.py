#!/usr/bin/env python3
"""Compare compression methods under one memory, one merge rule and one selection rule.

All methods keep what they store from LIBERO-40 only. Merging takes units until every
memory item is rebuilt within tau. On unseen demos the correct actions only choose units
and score the result: fixed units are chosen per 10-frame window, variable units by the
lowest-error sequence with one selection per 10 frames. Units play as stored, at their own
length. No retrieval, no denoising.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'pi05_memory'), str(HERE.parent / 'pi05_memory' / 'shape_skills')]
import dct_skill_draft_eval as ev  # noqa: E402
import dct_tolerance_merge as merge  # noqa: E402
import optimal_units_eval as opt  # noqa: E402
import stored_units_eval as stored  # noqa: E402
from draft_reconstruction import (ROOT, RUN, UNSEEN, UNSEEN_ROWS, CODECS, PI05, dct_roundtrip, keyframe_roundtrip,  # noqa: E402
                                  pca_basis, pca_roundtrip, summarize)
from representations import resample  # noqa: E402
from segment_demo import cut  # noqa: E402
from variable_length_error import stitch  # noqa: E402

LEVELS = (None, .04, .08, .16)
FIXED = {'raw': 70, 'keyframe3': 21, 'dct3': 21, 'pca12': 12, 'pca24': 24, 'm1b12': 12, 'm1b24': 24}
VARIABLE = {'var_raw': None, 'var_pca12': 12, 'var_pca24': 24}
CODEC_FILES = {'m1b12': ('b12m', 12), 'm1b24': ('b24', 24)}
PHASE = 32
EPSILON = .005


def cover_units(decoded, tau):
    """Greedy cover on a [N,...] array, compared by mean squared difference."""
    flat = np.asarray(decoded, np.float32).reshape(len(decoded), -1)
    within = np.empty((len(flat), len(flat)), bool)
    for i in range(len(flat)):
        within[i] = np.mean((flat - flat[i]) ** 2, axis=1) <= tau
    return merge.greedy_cover(within)[0]


class Data:
    """Inherited by worker processes."""


def load_codec(name):
    import torch
    torch.set_num_threads(1)
    spec = importlib.util.spec_from_file_location('compare_dim', PI05 / 'compress_vs_raw' / 'dimension_extension.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    file, dim = CODEC_FILES[name]
    return mod.load_codec(torch.load(CODECS / f'{file}.pt', map_location='cpu', weights_only=False), dim)


def run_fixed(config):
    name, level = config
    S, target = ev.Shared, ev.Shared.target
    memory = Data.memory_actions
    per_unit = FIXED[name]
    if name.startswith('m1'):
        codec = load_codec(name)
        state, feature = Data.memory_state, Data.memory_feature
        b, _ = codec.encode(memory, state)
        b = np.asarray(b)
        decoded = []
        for lo in range(0, len(b), 256):
            _, r = codec.prior(feature[lo:lo + 256], state[lo:lo + 256])
            decoded.append(np.asarray(codec.decode(b[lo:lo + 256], np.asarray(r), feature[lo:lo + 256], state[lo:lo + 256]))[:, :, :7])
        decoded = np.concatenate(decoded)
        units = np.arange(len(b)) if level is None else cover_units(decoded, level)
        pred = np.empty_like(target)
        for row in range(len(target)):
            f, s = Data.query_feature[row:row + 1], Data.query_state[row:row + 1]
            _, r = codec.prior(f, s)
            n = len(units)
            draft = np.asarray(codec.decode(b[units], np.repeat(np.asarray(r), n, 0), np.repeat(f, n, 0), np.repeat(s, n, 0)))[:, :, :7]
            pred[row] = draft[np.argmin(np.mean((draft - target[row][None]) ** 2, axis=(1, 2)))]
    else:
        active = memory[:, :, :7].astype(np.float64)
        if name == 'raw':
            decoded = active
        elif name == 'dct3':
            decoded = dct_roundtrip(active, 3)
        elif name == 'keyframe3':
            decoded = keyframe_roundtrip(active, 3)
        else:
            decoded = pca_roundtrip(active, Data.pca_mean, Data.pca_vt, per_unit)
        units = np.arange(len(decoded)) if level is None else cover_units(decoded, level)
        drafts = decoded[units]
        pred = np.empty_like(target)
        for row in range(len(target)):
            pred[row] = drafts[np.argmin(np.mean((drafts - target[row][None]) ** 2, axis=(1, 2)))]
    out = {'method': name, 'cut': 'fixed10', 'level': level, 'units': len(units), 'values_per_unit': per_unit,
           'stored_values': len(units) * per_unit, **summarize(pred, target, S.hard)}
    print(json.dumps({k: out[k] for k in ('method', 'level', 'units', 'all', 'hard')}), flush=True)
    return out


def phase_view(a):
    return resample(np.asarray(a, np.float32), PHASE).astype(np.float64)


def variable_decode(phase_values, length):
    return resample(np.asarray(phase_values, np.float32), length).astype(np.float64)


def run_variable(config):
    name, level = config
    S = ev.Shared
    k = VARIABLE[name]
    items, lengths = Data.var_items, Data.var_lengths
    if k is None:
        coded = items
    else:
        flat = items.reshape(len(items), -1)
        coded = (((flat - Data.var_mean) @ Data.var_vt[:k].T @ Data.var_vt[:k]) + Data.var_mean).reshape(items.shape)
    units = np.arange(len(coded)) if level is None else cover_units(coded, level)
    groups = {}
    for u in units:
        groups.setdefault(int(lengths[u]), []).append(variable_decode(coded[u], int(lengths[u])))
    drafts = {length: np.stack(v) for length, v in groups.items()}
    keys = sorted(drafts)
    pred = np.empty_like(S.target, dtype=np.float64)
    selections, frames = 0, 0
    for key, trace in S.traces.items():
        table = opt.best_cost_table(trace, drafts)
        budget = max(1, math.ceil(len(trace) / 10))
        _, sequence = opt.best_sequence(table, keys, len(trace), budget)
        joined = np.empty_like(trace, dtype=np.float64)
        for start, length in sequence:
            piece = table[start, length][1]
            joined[start:start + len(piece)] = piece
        pred[S.rows_of[key]] = joined.reshape(-1, 10, 7)
        selections += len(sequence)
        frames += len(trace)
    per_unit = (PHASE * 7 if k is None else k) + 1
    out = {'method': name, 'cut': f'e{EPSILON}', 'level': level, 'units': len(units), 'values_per_unit': per_unit,
           'stored_values': len(units) * per_unit, 'frames_per_selection': frames / selections,
           **summarize(pred, S.target, S.hard)}
    print(json.dumps({k2: out[k2] for k2 in ('method', 'level', 'units', 'all', 'hard')}), flush=True)
    return out


def variable_self_error(k):
    """Round trip of each unseen demo cut by the same rule, with no memory."""
    S = ev.Shared
    pred = np.empty_like(S.target, dtype=np.float64)
    for key, trace in S.traces.items():
        out = np.empty_like(trace, dtype=np.float64)
        for s in cut(trace, epsilon=EPSILON, k=3):
            flat = phase_view(trace[s['start']:s['end']]).ravel()
            coded = (((flat - Data.var_mean) @ Data.var_vt[:k].T @ Data.var_vt[:k]) + Data.var_mean).reshape(PHASE, 7)
            out[s['start']:s['end']] = variable_decode(coded, s['length'])
        pred[S.rows_of[key]] = out.reshape(-1, 10, 7)
    return summarize(pred, S.target, S.hard)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=ROOT / 'unseen-libero' / 'draft-reconstruction-01' / 'compression_compare.json')
    ap.add_argument('--workers', type=int, default=16)
    args = ap.parse_args()
    stored.load_data()
    S = ev.Shared
    rows = {k: np.load(RUN / 'rows' / f'{k}.npy', mmap_mode='r') for k in ('actions', 'state', 'task', 'episode', 'frame', 'memory', 'validation')}
    pooled = np.load(RUN / 'pooled.npy', mmap_mode='r')
    memory = np.flatnonzero(np.asarray(rows['memory'], bool))
    train = np.flatnonzero(~np.asarray(rows['validation'], bool))
    Data.memory_actions = np.asarray(rows['actions'][memory], np.float32)
    Data.memory_state = np.asarray(rows['state'][memory], np.float32)
    Data.memory_feature = np.asarray(pooled[memory], np.float32)
    Data.pca_mean, Data.pca_vt = pca_basis(np.asarray(rows['actions'][train])[:, :, :7])
    un = {k: np.load(UNSEEN_ROWS / f'{k}.npy', mmap_mode='r') for k in ('state', 'pooled', 'is_dev')}
    ev_rows = np.flatnonzero(~np.asarray(un['is_dev']))
    Data.query_state = np.asarray(un['state'][ev_rows], np.float32)
    Data.query_feature = np.asarray(un['pooled'][ev_rows], np.float32)

    task, episode, frame = (np.asarray(rows[k]) for k in ('task', 'episode', 'frame'))
    train_key = np.stack([task[train], episode[train]], 1)
    train_traces, _ = stitch(np.asarray(rows['actions'][train]), train_key, frame[train])
    phases = [phase_view(t[s['start']:s['end']]).ravel() for t in train_traces.values() for s in cut(t, epsilon=EPSILON, k=3)]
    Data.var_mean, Data.var_vt = pca_basis(np.stack(phases).reshape(len(phases), PHASE, 7))
    items, lengths = [], []
    for trace in S.memory.values():
        for s in cut(trace, epsilon=EPSILON, k=3):
            items.append(phase_view(trace[s['start']:s['end']]))
            lengths.append(s['length'])
    Data.var_items, Data.var_lengths = np.stack(items), np.asarray(lengths)
    print('variable segments', len(items), 'fit segments', len(phases), flush=True)

    self_errors = {f'var_pca{k}': variable_self_error(k) for k in (12, 24)}
    with ProcessPoolExecutor(args.workers) as pool:
        fixed = pool.map(run_fixed, [(m, l) for m in FIXED for l in LEVELS])
        variable = pool.map(run_variable, [(m, l) for m in VARIABLE for l in LEVELS])
        results = list(fixed) + list(variable)
    args.output.write_text(json.dumps({'rows': len(S.target), 'hard_rows': int(S.hard.sum()), 'self': self_errors, 'results': results}, ensure_ascii=False, indent=2) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
