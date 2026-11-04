#!/usr/bin/env python3
"""Build the action-only query-by-memory oracle distance cache."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STATE = Path(os.environ.get('SKILLNET_STATE_DIR', HERE.parents[5] / '.skillnet-state')).expanduser()
ROOT = STATE / 'outputs/memory-pilot/pi05-memory40-20261104/run-01'
OUT = STATE / 'outputs/memory-pilot/pi05-memory40-20261104/score-audit-01'
sys.path.insert(0, str(HERE.parent))
import run  # noqa: E402


def query_split(rows: dict[str, np.ndarray], query_rows: np.ndarray) -> np.ndarray:
    dev_eps, eval_eps = run.calibration_episodes(rows)
    labels: dict[tuple[int, int], str] = {}
    for task in np.unique(rows['task']):
        for ep in dev_eps:
            if np.any((rows['task'] == task) & (rows['episode'] == ep) & rows['validation']):
                labels[int(task), int(ep)] = 'dev'
        for ep in eval_eps:
            if np.any((rows['task'] == task) & (rows['episode'] == ep) & rows['validation']):
                labels[int(task), int(ep)] = 'eval'
    return np.asarray([labels[int(rows['task'][i]), int(rows['episode'][i])] for i in query_rows])


def build(limit: int | None = None, out: Path = OUT) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    if any((out / name).exists() for name in ('oracle_action_mse.npy', 'oracle_meta.npz', 'query_meta.npz')):
        raise FileExistsError(f'refusing to overwrite oracle cache in {out}')
    rows = run.load_rows(ROOT)
    mem_rows = np.load(ROOT / 'memory_row_ids.npy', mmap_mode='r', allow_pickle=False)
    queries = np.flatnonzero(rows['validation'])
    if limit is not None:
        queries = queries[:limit]
    q_actions = np.asarray(rows['actions'][queries, :, :7], np.float32).reshape(len(queries), -1)
    m_actions = np.asarray(rows['actions'][mem_rows, :, :7], np.float32).reshape(len(mem_rows), -1)
    m_norm2 = np.einsum('ij,ij->i', m_actions, m_actions, dtype=np.float32)
    q_norm2 = np.einsum('ij,ij->i', q_actions, q_actions, dtype=np.float32)
    dist = np.lib.format.open_memmap(out / 'oracle_action_mse.npy', mode='w+', dtype='float16',
                                     shape=(len(queries), len(mem_rows)))
    t0 = time.monotonic()
    block = 64
    for start in range(0, len(queries), block):
        end = min(len(queries), start + block)
        dots = q_actions[start:end] @ m_actions.T
        d2 = np.maximum(q_norm2[start:end, None] + m_norm2[None, :] - 2 * dots, 0) / q_actions.shape[1]
        dist[start:end] = d2.astype(np.float16)
    dist.flush()
    qtask = np.asarray(rows['task'][queries], dtype=np.int32)
    qep = np.asarray(rows['episode'][queries], dtype=np.int32)
    mt = np.asarray(rows['task'][mem_rows], dtype=np.int32)
    me = np.asarray(rows['episode'][mem_rows], dtype=np.int32)
    meta = out / 'oracle_meta.npz'
    np.savez_compressed(meta, query_rows=queries.astype(np.int64), memory_rows=mem_rows.astype(np.int64),
                        query_task=qtask, query_episode=qep, query_split=query_split(rows, queries),
                        memory_task=mt, memory_episode=me,
                        threshold_values=np.asarray([.005, .01, .02], np.float32))
    np.savez(out / 'query_meta.npz', query_rows=queries.astype(np.int64),
             split=query_split(rows, queries), task=qtask, episode=qep)
    summary = {
        'queries': int(len(queries)), 'memory_rows': int(len(mem_rows)),
        'distance_shape': list(dist.shape), 'distance_dtype': 'float16',
        'seconds': time.monotonic() - t0, 'limit': limit,
        'threshold_existence_rate': {},
        'definition': 'MSE of normalized rows.actions[:, :, :7], flattened 70 dims; no codec involved',
    }
    for tau in (.005, .01, .02):
        summary['threshold_existence_rate'][str(tau)] = float(np.mean(np.any(dist <= tau, axis=1)))
    (out / 'oracle_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    print(json.dumps(build(args.limit, args.output), indent=2))


if __name__ == '__main__':
    main()
