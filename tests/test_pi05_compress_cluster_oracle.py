from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_path = (Path(__file__).resolve().parents[1] / 'skill_moe/skillnet/examples/memory_pilot/'
         'pi05_memory/compress_cluster_oracle/experiment.py')
_spec = importlib.util.spec_from_file_location('compress_cluster_oracle_test_module', _path)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)

_queue_path = (Path(__file__).resolve().parents[1] / 'skill_moe/skillnet/examples/memory_pilot/'
               'pi05_memory/compress_cluster_oracle/run_queue.py')
_queue_spec = importlib.util.spec_from_file_location('compress_cluster_oracle_queue_test_module', _queue_path)
_queue_mod = importlib.util.module_from_spec(_queue_spec)
assert _queue_spec.loader is not None
_queue_spec.loader.exec_module(_queue_mod)


def test_each_query_action_row_is_in_pool_and_row_ids_are_not_pool_offsets():
    pool = _mod.compose_pool(np.array([20, 24]), np.array([2, 9]), np.array([31]), np.array([40]))
    assert _mod.assert_pool_coverage(pool, np.array([40, 2, 31]))
    assert _mod.map_row_ids(pool, np.array([40, 2, 31])).tolist() == [5, 0, 4]
    with pytest.raises(AssertionError):
        _mod.assert_pool_coverage(pool, np.array([41]))


def test_raw_none_can_select_exact_truth_with_zero_mse():
    truth = np.arange(70, dtype=np.float32).reshape(10, 7)
    pool = np.stack([truth + 1, truth, truth - 2])
    distances = np.asarray([_mod.active7_mse(truth, candidate) for candidate in pool])
    assert int(np.argmin(distances)) == 1
    assert distances[1] == 0.0


def test_raw_none_pipeline_returns_own_action_and_actual_row_id():
    rng = np.random.default_rng(3)
    actions = np.zeros((30, 10, 32), np.float32)
    actions[5] = rng.normal(size=(10, 32))
    actions[20] = rng.normal(size=(10, 32))
    pool = np.asarray([5, 20], np.int64)
    values = actions[pool, :, :7].reshape(2, 70)
    fit = {'row_units': np.asarray([0, 1], np.int32)}
    primary, secondary, chosen, _ = _mod._predictions_for_queries(
        'raw', None, None, values, fit, np.asarray([0, 1]), pool, pool,
        np.asarray([20]), {'task': np.zeros(30, np.int32)}, actions, np.zeros((30, 2), np.float32))
    assert chosen.tolist() == [20]
    np.testing.assert_array_equal(primary[0], actions[20])
    np.testing.assert_array_equal(secondary[0], actions[20])


def test_long_rows_cannot_enter_expert_training():
    rows = {'task': np.asarray([10, 11, 2, 0])}
    assert _mod.assert_long_excluded_from_expert([0, 1], rows)
    with pytest.raises(AssertionError):
        _mod.assert_long_excluded_from_expert([0, 3], rows)


def test_shuffle_sources_are_from_other_tasks():
    tasks = np.asarray([10, 10, 11, 11, 12, 12])
    source = np.asarray([0, 2, 4, 1, 3, 5])
    query = np.asarray([0, 1, 2, 3, 4, 5])
    shuffled, _ = _mod.stable_other_task_shuffle(source, query, tasks, seed=12)
    assert np.all(tasks[shuffled] != tasks[query])


def test_m6default_handles_numpy_center_arrays_without_changing_distance():
    from skill_memories.build import RawFunctional
    rng = np.random.default_rng(17)
    values = np.broadcast_to(rng.normal(size=(1, 70)).astype(np.float32), (24, 70)).copy()
    rows = {'task': np.repeat(np.arange(4), 6),
            'episode': np.tile(np.repeat(np.arange(3), 2), 4)}
    ids = np.arange(24, dtype=np.int64)
    fit = _mod._fit_unit(values, ids, rows, RawFunctional(), 'm6default', np.arange(24))
    assert fit['row_units'].shape == (24,)
    assert len(fit['centers']) >= 1


def test_b12_none_secondary_uses_each_queries_own_row_and_matches_primary():
    class TinyCodec:
        def prior(self, feature, state):
            n = len(feature)
            return np.zeros((n, 1), np.float32), np.zeros((n, 1), np.float32)

        def decode(self, base, residual, feature, state):
            return np.broadcast_to(np.asarray(base)[:, :1, None], (len(base), 10, 32)).copy()

    nrows = 4
    actions = np.zeros((nrows, 10, 32), np.float32)
    actions[1, :, :7] = 1.0
    actions[2, :, :7] = 2.0
    actions[3, :, :7] = 3.0
    bases = np.arange(nrows, dtype=np.float32)[:, None]
    features = np.zeros((nrows, 2), np.float32)
    rows = {'state': np.zeros((nrows, 3), np.float32)}
    pool = np.asarray([1, 2, 3], np.int64)
    reps = pool.copy()
    fit = {'row_units': np.asarray([0, 1, 2], np.int32)}
    primary, secondary, chosen, _ = _mod._predictions_for_queries(
        'b12', TinyCodec(), bases, bases[pool], fit, np.arange(3), reps, pool,
        np.asarray([1, 2, 3]), rows, actions, features)
    assert chosen.tolist() == [1, 2, 3]
    np.testing.assert_array_equal(secondary, primary)


def test_queue_has_no_duplicate_expert_cells():
    base = _queue_mod.BASE
    extra = _queue_mod.EXTRA
    all_cells = base + extra
    assert len(base) == 16
    assert len(extra) == 4
    assert len(set(all_cells)) == len(all_cells) == 20
    assert not set(base).intersection(extra)
