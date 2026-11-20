import numpy as np

from skill_moe.skillnet.examples.memory_pilot.unseen_libero.t0x_common import (
    calibrate_disagreement,
    t0_from_disagreement,
    topk_disagreement,
)


def test_topk_disagreement_is_softmax_weighted_arm_mse_to_top1():
    # Query x candidate x time x (arm6 + gripper); gripper must not affect D.
    actions = np.zeros((1, 3, 2, 7), np.float32)
    actions[0, 1, :, 0] = 2.
    actions[0, 2, :, 0] = 4.
    actions[0, :, :, 6] = np.asarray([100., -100., 1000.])[:, None]
    scores = np.log(np.asarray([[1., 2., 1.]], np.float32))
    d, p = topk_disagreement(actions, scores, k=3)
    expected_p = np.asarray([[.25, .5, .25]])
    # arm6 MSE per step: 0, 4/6, 16/6; average under the within-top-k softmax.
    np.testing.assert_allclose(p, expected_p)
    np.testing.assert_allclose(d, [.25 * 0 + .5 * (4 / 6) + .25 * (16 / 6)])


def test_disagreement_calibration_matches_ehat_tail_and_hits_radius_anchor():
    cal = calibrate_disagreement([1., 2., 3., 4.], [0., 0., 1., 1.], b=.5)
    assert cal["p_b"] == .5
    assert cal["D_b"] == 2.
    np.testing.assert_allclose(t0_from_disagreement([0., 2.], cal), [.15, .35], atol=1e-7)
    assert np.mean(np.asarray([1., 2., 3., 4.]) > cal["D_b"]) == .5


def test_disagreement_helpers_reject_bad_shapes_and_invalid_threshold():
    try:
        topk_disagreement(np.zeros((2, 2, 10, 6)), np.zeros((1, 2)))
    except ValueError:
        pass
    else:
        raise AssertionError("query axes must agree")
    try:
        calibrate_disagreement([0., 1.], [0., 1.], b=0.)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid fallback threshold must fail")
