import numpy as np

from skill_moe.skillnet.examples.emergent.light.skill_bench import (
    _coverage_outside,
    _unit_separation,
)


class _ToyContinuous:
    # Exercise the M6 functional-unit evaluator without fitting a neural net.
    __module__ = "skill_moe.skillnet.examples.emergent.light.methods.m0"

    def __init__(self):
        self._bench_m6_unitizer = _ToyM6Unitizer()

    def base_features(self, values):
        return np.asarray(values, np.float32)

    def decode(self, base, residual, feature, state):
        values = np.asarray(base, np.float32).reshape(len(feature), -1)[:, :1]
        return values[:, None, :]


class _ToyM6Unitizer:
    def __init__(self):
        self._centers = [np.asarray([0.0], np.float32), np.asarray([10.0], np.float32)]
        self.base_dim = 1
        self.epsilon = 1.0

    def _nearest(self, base, centers=None):
        centers = self._centers if centers is None else centers
        distances = [float(np.mean((np.asarray(base) - center) ** 2)) for center in centers]
        index = int(np.argmin(distances))
        return index, distances[index]


def test_r3_uses_train_q95_and_prequantized_base_distance():
    method = _ToyContinuous()
    train_base = np.asarray([[0.0], [0.0], [10.0], [10.0]], np.float32)
    query_base = np.asarray([[0.0], [10.0]], np.float32)
    train_raw = np.asarray([[0.0], [0.1], [10.0], [10.1]], np.float32)
    query_raw = np.asarray([[0.05], [10.2]], np.float32)
    metric = _coverage_outside(method, train_base, query_base, train_raw, query_raw)
    assert np.isclose(metric["threshold_train_q95"], 0.01)
    assert np.isclose(metric["outside_rate"], 0.5)


def test_r2_reports_between_over_within_action_distance():
    method = _ToyContinuous()
    fit_base = np.asarray([[0.0], [0.0], [10.0], [10.0]], np.float32)
    query_base = fit_base.copy()
    table = {"step": np.arange(4), "base": query_base,
             "feature": np.zeros((4, 1), np.float32),
             "state": np.zeros((4, 1), np.float32)}
    own_pred = np.asarray([0.0, 0.5, 10.0, 10.5], np.float32)[:, None, None]
    metric = _unit_separation(method, table, np.zeros((4, 1), np.float32), fit_base, own_pred)
    assert metric["between_unit_action_mse"] > metric["within_unit_action_mse"]
    assert metric["separation_ratio"] > 1.0
    assert metric["redundant_pair_fraction"] == 0.0
