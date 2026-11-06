import importlib.util
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parents[1] / "skill_moe/skillnet/examples/memory_pilot/unseen_libero/segment_demo.py"
spec = importlib.util.spec_from_file_location("segment_demo", CODE)
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


def _actions(n=60):
    a = np.zeros((n, 7), np.float32)
    a[:, 6] = -1.
    return a


def test_segments_cover_the_demo_without_gaps():
    a = _actions(); a[:, 0] = np.sin(np.arange(60) / 2)
    s = demo.cut(a, epsilon=.005)
    assert s[0]["start"] == 0 and s[-1]["end"] == 60
    assert all(x["end"] == y["start"] for x, y in zip(s, s[1:]))


def test_gripper_switch_always_cuts_even_inside_the_minimum_length():
    a = _actions(); a[3:, 6] = 1.
    s = demo.cut(a, epsilon=1.)
    assert s[0]["end"] == 3 and s[0]["reason"] == "gripper"


def test_smooth_motion_stays_in_one_segment_at_large_epsilon():
    a = _actions(); a[:, 0] = np.linspace(-1, 1, 60)
    assert len(demo.cut(a, epsilon=.04)) == 1


def test_smaller_epsilon_never_gives_fewer_segments_on_rough_motion():
    a = _actions(); a[:, 0] = np.sin(np.arange(60))
    assert len(demo.cut(a, epsilon=.001)) >= len(demo.cut(a, epsilon=.5))


def _error_module():
    import sys
    sys.path.insert(0, str(CODE.parent))
    import variable_length_error
    return variable_length_error


def test_equal_cuts_keep_gripper_switches_and_reach_the_count():
    a = _actions(); a[20:, 6] = 1.
    spans = _error_module().equal_cuts(a, 4)
    assert len(spans) == 4 and any(s == 20 for s, _ in spans)
    assert spans[0][0] == 0 and spans[-1][1] == 60
    assert all(x[1] == y[0] for x, y in zip(spans, spans[1:]))


def test_rebuild_counts_values_and_keeps_a_smooth_motion():
    a = _actions(); a[:, 0] = np.linspace(-1, 1, 60)
    out, values = _error_module().rebuild(a, [(0, 30), (30, 60)], 3)
    assert values == 2 * (6 * 3 + 2)
    assert np.mean((out[:, 0] - a[:, 0]) ** 2) < 2e-3 and np.all(out[:, 6] == -1.)


def test_segment_values_round_trip_at_its_own_length_and_stretch():
    import sys
    sys.path[:0] = [str(CODE.parent), str(CODE.parent.parent / "pi05_memory")]
    import variable_memory_oracle as vm
    a = _actions(30); a[:, 0] = np.cos(np.pi * (np.arange(30) + .5) / 30) * .5 + .2
    v = vm.segment_values(a, 0, 30)
    np.testing.assert_allclose(vm.decode(v, 30)[0], a, atol=1e-6)
    assert vm.decode(v, 60).shape == (1, 60, 7)
    assert abs(vm.decode(v, 60)[0, :, 0].mean() - .2) < 1e-6
