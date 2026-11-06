#!/usr/bin/env python3
"""Recompute fixed raw-nearest and native normalized PCA24 controls from rows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data import HELDOUT_TASKS, LONG_TASKS, training_row_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--tolerance', type=float, default=0.001)
    a = ap.parse_args()
    rows = {p.stem: np.load(p, mmap_mode='r', allow_pickle=False)
            for p in (a.run / 'rows').glob('*.npy')}
    z = np.load(a.root / 'topk-long-01/production/predictions_and_errors.npz', allow_pickle=False)
    long_ids = np.asarray(z['query_rows'], np.int64)
    if len(long_ids) != 548:
        raise AssertionError(f'expected 548 Long rows, got {len(long_ids)}')
    hard = np.asarray(z['hard_mask'], bool)
    if hard.sum() != 63:
        raise AssertionError(f'expected 63 hard rows, got {hard.sum()}')
    target = np.asarray(rows['actions'][long_ids, :, :7], np.float32)
    task = np.asarray(rows['task'])
    # This is the established raw-rows-noLong memory pool: all memory rows
    # outside tasks 0--9, including heldout-task memory episodes.
    mem = np.flatnonzero(np.asarray(rows['memory'], bool) & (task >= 10))
    if len(mem) != 3857:
        raise AssertionError(f'expected 3857 non-Long/non-heldout memory rows, got {len(mem)}')
    mem_actions = np.asarray(rows['actions'][mem, :, :7], np.float32)
    # Exact per-window active7 mean squared distance. Bounded temporary matrix.
    raw_mse = []
    for q in target:
        d = np.mean((mem_actions - q[None]) ** 2, axis=(1, 2))
        raw_mse.append(float(d.min()))
    raw_mse = np.asarray(raw_mse, np.float64)

    fit_ids = np.flatnonzero(training_row_mask(rows))
    if np.any(np.isin(task[fit_ids], LONG_TASKS + HELDOUT_TASKS)):
        raise AssertionError('Long/heldout entered PCA fitting')
    fit = np.asarray(rows['actions'][fit_ids, :, :7], np.float64).reshape(len(fit_ids), -1)
    # Native 70-value q01/q99-normalized S0 windows; no phase resampling.
    mean = fit.mean(0)
    _, _, vt = np.linalg.svd(fit - mean, full_matrices=False)
    comp = vt[:24]
    q = target.astype(np.float64).reshape(len(target), -1)
    rec = ((q - mean) @ comp.T) @ comp + mean
    pca_mse = np.mean((rec - q) ** 2, axis=1)

    def summary(v):
        return {'all_mean': float(np.mean(v)), 'hard_mean': float(np.mean(v[hard])),
                'all_median': float(np.median(v)), 'hard_n': int(hard.sum())}
    result = {
        'long_row_ids': long_ids.tolist(), 'hard_mask': hard.tolist(),
        'memory_row_ids_count': int(len(mem)), 'fit_row_ids_count': int(len(fit_ids)),
        'raw_nearest': summary(raw_mse), 'pca24_native_q01q99': summary(pca_mse),
        'raw_nearest_row_mse': raw_mse.tolist(), 'pca24_row_mse': pca_mse.tolist(),
        'expected': {'raw_nearest': [0.0290, 0.0351], 'pca24': [0.0009, 0.0021]},
        'tolerance': 0.001,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.out.exists():
        raise FileExistsError(a.out)
    a.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('raw_nearest', 'pca24_native_q01q99',
                                               'memory_row_ids_count', 'fit_row_ids_count')}, indent=2))
    for name, key in (('raw_nearest', 'raw_nearest'), ('pca24', 'pca24_native_q01q99')):
        got = result[key]
        exp = result['expected'][name]
        if abs(got['all_mean'] - exp[0]) > a.tolerance or abs(got['hard_mean'] - exp[1]) > a.tolerance:
            raise AssertionError(f'{name} mismatch: got {got}, expected {exp}, tolerance {a.tolerance}')


if __name__ == '__main__':
    main()
