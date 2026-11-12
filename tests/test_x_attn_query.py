import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'skill_moe/skillnet/examples/memory_pilot/unseen_libero'))
import x_attn_query as xq


def test_detector_swap_respects_fit_task_rows_only():
    # Row 2 is a held-out task and must not become a synthetic training source.
    ds = {'task': np.array([0, 1, 2]), 'episode': np.zeros(3, int),
          'frame': np.zeros(3, int),
          'hist': np.repeat(np.arange(3, dtype=np.float32)[:, None, None], 70, axis=1).reshape(3, 10, 7)}
    out, actual = xq._corrupt_histories_by_code(
        ds, np.array([0]), np.array(['swap']), seed=3, swap_pool_ids=np.array([0, 1]))
    np.testing.assert_array_equal(out[0], ds['hist'][1])
    np.testing.assert_array_equal(actual, ['swap'])


def test_missing_phase_corruption_is_not_mislabeled():
    ds = {'task': np.array([0, 1]), 'episode': np.zeros(2, int),
          'frame': np.array([0, 0]), 'hist': np.ones((2, 10, 7), np.float32)}
    out, actual = xq._corrupt_histories_by_code(ds, np.array([0]), np.array(['phase+10']), seed=1)
    np.testing.assert_array_equal(out, np.zeros((1, 10, 7), np.float32))
    np.testing.assert_array_equal(actual, ['zero'])


def test_missing_selection_phase_stays_corrupted(monkeypatch):
    ds = {'task': np.array([0]), 'episode': np.array([0]), 'frame': np.array([0]),
          'hist': np.ones((1, 10, 7), np.float32)}
    monkeypatch.setattr(xq, 'corruption_codes', lambda n, **kwargs: np.array(['phase']))
    out, modes = xq._mixed_history(ds, np.array([0]), seed=1, selection=True)
    np.testing.assert_array_equal(out, np.zeros((1, 10, 7), np.float32))
    np.testing.assert_array_equal(modes, ['zero'])


def test_phase_corruption_uses_same_episode_neighbor():
    hist = np.stack([np.full((10, 7), i, np.float32) for i in range(4)])
    ds = {'task': np.array([0, 0, 0, 1]), 'episode': np.array([1, 1, 1, 1]),
          'frame': np.array([0, 10, 20, 10]), 'hist': hist}
    out, actual = xq._corrupt_histories_by_code(ds, np.array([1]), np.array(['phase']), seed=2)
    assert actual[0] == 'phase'
    assert out[0, 0, 0] in (0., 2.)


def test_top10_correlations_report_constant_window_counts():
    error_row = np.arange(1, 21, dtype=np.float64)
    score = np.stack([np.ones(20), -error_row])
    error = np.stack([error_row, error_row])
    rows = xq._retrieval_row_metrics(score, error)
    assert rows['conditional_spearman'][0] == 0.
    assert rows['conditional_spearman_valid'][0] == 0.
    assert rows['pearson_top10'][0] == 0.
    assert rows['pearson_top10_valid'][0] == 0.
    assert rows['conditional_spearman_valid'][1] == 1.

