import numpy as np

from skill_moe.skillnet.examples.memory_pilot.unseen_libero.denoise_oracle_adaptive import (
    arm6_mse,
    apply_fixed_pi05_gripper,
    apply_quantile_policy,
    fit_quantile_policy,
)


def test_quantile_policy_selects_minimum_dev_t0_by_bin():
    feature = np.arange(8, dtype=np.float64)
    losses = np.column_stack([feature + 1, 8 - feature, np.ones_like(feature) * 20])
    p = fit_quantile_policy(feature, losses, t0_values=[0.0, 0.1, 0.2], quantiles=(.5,))
    bins, indices, t0 = apply_quantile_policy(feature, p)
    assert p['edges'] == [3.5]
    assert bins.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert indices.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert np.allclose(t0, [0.0] * 4 + [0.1] * 4)


def test_arm6_mse_matches_sum_over_seven_times_ten():
    target = np.zeros((2, 10, 7), dtype=np.float32)
    pred = np.zeros_like(target)
    pred[:, :, :6] = 1.0
    assert np.allclose(arm6_mse(pred, target), 60 / 70)


def test_duplicate_quantile_edges_are_safe():
    feature = np.ones(5)
    losses = np.tile(np.array([[3., 1.]]), (5, 1))
    policy = fit_quantile_policy(feature, losses, t0_values=[0., .1])
    bins, indices, t0 = apply_quantile_policy(feature, policy)
    assert np.all(indices == 1)
    assert np.all(t0 == .1)
    assert np.all(bins == 1)


def test_fixed_gripper_changes_only_the_last_action_channel():
    draft = np.zeros((3, 10, 7), dtype=np.float32)
    normal = np.ones_like(draft)
    fixed = apply_fixed_pi05_gripper(draft, normal)
    np.testing.assert_array_equal(fixed[:, :, :6], draft[:, :, :6])
    np.testing.assert_array_equal(fixed[:, :, 6], normal[:, :, 6])
