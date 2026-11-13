import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "skill_moe/skillnet/examples/memory_pilot/unseen_libero/denoise_eval.py"
spec = importlib.util.spec_from_file_location("dn1_denoise_eval_test", SCRIPT)
dn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dn)


def test_six_axis_scale_matches_active7_when_gripper_error_zero():
    rng = np.random.default_rng(4)
    target = rng.normal(size=(5, 10, 7)).astype(np.float32)
    pred = target.copy()
    pred[:, :, :6] += .3
    assert np.allclose(dn.six_mse(pred, target), dn.active7_mse(pred, target))


def test_padding_preserves_active_actions_and_zeroes_unused_dimensions():
    a = np.ones((2, 10, 7), np.float32)
    padded = dn.pad_actions(a)
    assert padded.shape == (2, 10, 32)
    np.testing.assert_array_equal(padded[:, :, :7], a)
    assert np.count_nonzero(padded[:, :, 7:]) == 0


def test_hold_last_uses_previous_window_last_action_and_zero_at_start():
    actions = np.zeros((3, 10, 32), np.float32)
    actions[0, -1, :7] = np.arange(7) + 1
    actions[1, -1, :7] = np.arange(7) + 11
    result = dn.hold_last(actions, [0, 0, 0], [5, 5, 5], [10, 0, 20])
    np.testing.assert_array_equal(result[1], np.zeros((10, 7), np.float32))
    np.testing.assert_array_equal(result[0], np.repeat(actions[1, -1:, :7], 10, axis=0))
    np.testing.assert_array_equal(result[2], np.repeat(actions[0, -1:, :7], 10, axis=0))


def test_oracle_type_selection_uses_type_and_pi05_gripper():
    import sys
    sys.path.insert(0, str(SCRIPT.parent))
    import variable_shape_memory as vs

    t = np.linspace(0, 1, 10, dtype=np.float32)
    physical = np.stack([np.stack([.1 + t * .03, .03 + t * .01, t * .005,
                                   t * .002, t * .003, t * .004], axis=1)])
    target = np.zeros((1, 10, 7), np.float32)
    target[:, :, :6] = vs.to_normalized(physical)[0]
    target[:, :, 6] = np.where(t < .5, -1, 1)
    pool = np.zeros((2, 10, 7), np.float32)
    pool[0, :, :6] = vs.to_normalized(physical[0] * .5)
    pool[1, :, :6] = vs.to_normalized(physical[0][::-1] + np.array([0, .2, 0, 0, 0, 0]))
    pool[:, :, 6] = -0.25
    normal = target.copy()
    normal[:, :, 6] = .75
    chosen, err, posed = dn.pose_oracle_types(target, normal, pool, vs)
    assert chosen.tolist() == [0]
    assert err[0] < 1e-6
    np.testing.assert_allclose(posed[:, :, 6], normal[:, :, 6])

