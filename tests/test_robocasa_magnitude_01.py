from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = (Path(__file__).parents[1] / "skill_moe/skillnet/examples/memory_pilot/robocasa/magnitude_01.py")
SPEC = importlib.util.spec_from_file_location("magnitude_01", SCRIPT)
MAG = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MAG)


def test_shared_nonnegative_scale_is_least_squares():
    x = np.array([[1., 2.], [3., 4.]])
    y = 2.5 * x
    assert np.isclose(MAG._scale_fit(x, y), 2.5)
    assert MAG._scale_fit(x, -y) == 0.0


def test_duration_rules_and_caps():
    c = np.zeros((20, 6), np.float32)
    c[:, 0] = 1.
    assert MAG._duration(20, 7, c, 1., "T1") == 7
    assert MAG._duration(20, 7, c, 1., "T2") == 20
    assert MAG._duration(20, 7, c, 1., "T3") == 10
    assert MAG._duration(20, 7, c, 1., "T4") == 20
    assert MAG._duration(80, 80, c, 0., "T1") == 50


def test_amplitude_definitions_are_distinct_and_oracle_is_nonnegative():
    c = np.ones((10, 12), np.float32)
    y = c.copy(); y[:, :6] *= 3.
    pi = c.copy(); pi[:, :6] *= 2.
    assert np.isclose(MAG._amplitude(c, y, pi, 1., "A1"), 3.)
    assert np.isclose(MAG._amplitude(c, y, pi, 1., "A3"), 2.)
    assert MAG._amplitude(c, y, pi, 0., "A2") == 0.
    assert MAG._amplitude(c, y, pi, 1., "A4") == 1.


def test_policy_and_storage_order_round_trip():
    a = np.arange(24, dtype=np.float32).reshape(2, 12)
    assert np.array_equal(ex_action_roundtrip(a), a)


def test_ood_dev_tasks_are_selected_by_name_not_numeric_assumption():
    names = ["OOD0", "EvalA", "OOD1", "EvalB", "OOD2"]
    assert MAG._ood_task_ids(names, {"EvalA", "EvalB"}) == [0, 2, 4]


def test_nonoracle_selection_uses_equal_weight_first_and_continuation_active6():
    metrics = {
        "A2_T2": {"first5_active6_mse": 0.10, "continuation5_9_active6_mse": 1.0},
        "A4_T2": {"first5_active6_mse": 0.5, "continuation5_9_active6_mse": 0.5},
        "A6_T2": {"first5_active6_mse": 0.8, "continuation5_9_active6_mse": 0.8},
        "A2_T1": {"first5_active6_mse": 0.0, "continuation5_9_active6_mse": 0.0},
        "A1_T2": {"first5_active6_mse": 0.0, "continuation5_9_active6_mse": 0.0},
    }
    assert MAG._select_nonoracle_active6(metrics) == "A4_T2"


def test_axis_summary_reports_all_seven_normalized_channels():
    values = np.asarray([[1, 2, 3, 4, 5, 6, 7], [3, 4, 5, 6, 7, 8, 9]], np.float64)
    summary = MAG._axis_segment_summary(values, [0, 0], [10, 11])
    assert list(summary) == ["x", "y", "z", "rx", "ry", "rz", "gripper"]
    assert summary["gripper"]["mean"] == 8.0
    assert len(summary["gripper"]["bootstrap95"]) == 2


def test_a5_progress_key_keeps_all_duration_rules_for_each_t0():
    keys = {MAG._a5_progress_key({"row": 5, "rule": f"A5_t0.2_T{i}"}) for i in range(1, 5)}
    assert len(keys) == 4
    assert MAG._a5_progress_key({"row": 5, "rule": "A5_t0.2_T1"}) != MAG._a5_progress_key(
        {"row": 5, "rule": "A5_t0.4_T1"})


def ex_action_roundtrip(a):
    import sys
    sys.path.insert(0, str(SCRIPT.parent))
    import experiment as ex
    return ex.action_policy_to_storage(ex.action_storage_to_policy(a))
