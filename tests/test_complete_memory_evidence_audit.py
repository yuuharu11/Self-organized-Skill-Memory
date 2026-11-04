"""Tests for the restored-model diagnostic's sampling and integration contract."""
import importlib.util
from pathlib import Path

import numpy as np
import torch
from torch import nn

spec = importlib.util.spec_from_file_location(
    "complete_memory_evidence", Path(__file__).parents[1] / "scripts/complete_memory_evidence_audit.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_episode_selection_ignores_future_actions_and_uses_one_window():
    rows = {"task": np.repeat([1, 2], 12),
            "episode": np.array([f"t{t}-e{i//3}" for t in (1, 2) for i in range(12)]),
            "a": np.zeros((24, 25, 7))}
    selected = audit.select_episode_windows(rows, 3)
    rows["a"][:] = 999
    assert np.array_equal(selected, audit.select_episode_windows(rows, 3))
    assert len(selected) == len(np.unique(rows["episode"][selected])) == 6
    assert all(np.sum(rows["task"][selected] == t) == 3 for t in (1, 2))


def test_gold_rank_and_functionally_equivalent_recall_are_different():
    scores = np.array([[3., 2., 1.], [1., 2., 3.]])
    distances = np.array([[.01, .02, .03], [2., 3., 4.]])
    result, rank, _ = audit.retrieval_metrics(scores, np.array([1, 0]), distances, .1)
    assert rank.tolist() == [2, 3]
    assert result["hits"]["1"] == 0.
    assert result["functional_coverage"] == .5
    assert result["functional_recall_on_covered"]["1"] == 1.
    assert result["hits"]["5"] == 1.


class ConstantVelocity(nn.Module):
    def forward(self, x):
        return torch.ones((len(x), 175), dtype=x.dtype)


def test_noise_and_anchor_integration_use_each_rows_actual_time():
    rows = {"f": np.zeros((2, 1), np.float32), "s": np.zeros((2, 1), np.float32)}
    norm = {"f_mean": 0., "f_scale": 1., "s_mean": 0., "s_scale": 1.,
            "a_mean": 2., "a_scale": 3.}
    eps = np.ones((2, 2, 175), np.float32)
    anchors = np.full((2, 25, 7), 8., np.float32)
    output = audit.flow_predict(ConstantVelocity(), norm, rows, eps, anchors,
                                np.array([.2, .6]), np.array([1, 3]))
    # Standardized anchor=2; x_start=t+(1-t)*2; Euler integral=-t.
    expected = 2.+3.*(2.-2.*np.array([.2, .6]))
    assert np.allclose(output[:, :, 0, 0], expected[None])
    pure = audit.flow_predict(ConstantVelocity(), norm, rows, eps, None, 1., np.array([2, 5]))
    assert np.allclose(pure, 2., atol=1e-6)


def test_mse_averages_time_and_action_axes_but_keeps_noise_and_rows():
    pred = np.zeros((3, 2, 25, 7), np.float32)
    pred[:, 1] = 2.
    errors = audit.sample_errors(pred, np.zeros((2, 25, 7), np.float32))
    assert errors.shape == (3, 2)
    assert np.array_equal(errors, np.tile([0., 4.], (3, 1)))
