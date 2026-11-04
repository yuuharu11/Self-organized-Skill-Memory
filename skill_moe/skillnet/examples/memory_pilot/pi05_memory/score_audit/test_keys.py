"""Small CPU tests for non-VLM retrieval key generation."""
from __future__ import annotations

import tempfile
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import keys
import learned_key


def test_history_uses_only_preceding_executed_chunk_and_frame_zero_flag():
    rows = {
        'episode': np.asarray([0, 0, 1]),
        'frame': np.asarray([0, 10, 0]),
        'actions': np.arange(3*10*32, dtype=np.float32).reshape(3, 10, 32),
    }
    values, present = keys.history_features(rows)
    assert present.tolist() == [0., 1., 0.]
    np.testing.assert_array_equal(values[0], np.zeros(70, np.float32))
    np.testing.assert_array_equal(values[1], rows['actions'][0, :, :7].reshape(-1))


def test_top64_candidate_generation_respects_query_exclusion(tmp_path):
    n = 75
    rows = {'task': np.zeros(n, np.int32), 'episode': np.arange(n, dtype=np.int32),
            'validation': np.zeros(n, bool), 'memory': np.ones(n, bool)}
    rows['validation'][0] = True
    rows['memory'][0] = False
    qids = np.asarray([0], np.int64)
    mem = np.flatnonzero(rows['memory'])
    out = tmp_path / 'audit'
    item = keys.write_key('toy', qids, rows, mem, lambda q, ids: -ids.astype(np.float32),
                          'episode', 'synthetic', out)
    payload = np.load(item['path'], allow_pickle=False)
    assert payload['query_rows'].tolist() == [0]
    assert payload['cand_rows'].shape == (1, 64)
    assert payload['score'].shape == (1, 64)
    assert np.all(rows['episode'][payload['cand_rows'][0]] != rows['episode'][0])
    assert np.all(np.diff(payload['score'][0]) <= 0)


def test_task_exclusion_removes_same_task():
    rows = {'task': np.asarray([1, 1, 2]), 'episode': np.asarray([10, 11, 12])}
    memory = np.asarray([0, 1, 2])
    assert keys._eligible(memory, rows, 0, 'task').tolist() == [2]


def test_pair_pool_contains_only_cross_task_examples():
    rows = {'task': np.asarray([0, 0, 1, 2])}
    actions = np.zeros((4, 10, 7), np.float32)
    actions[2] = .01
    left, right, mse = learned_key.pair_pool(np.arange(4), rows, actions, count=100, seed=2)
    assert len(left) > 0
    assert np.all(rows['task'][left] != rows['task'][right])
    assert mse.shape == left.shape

