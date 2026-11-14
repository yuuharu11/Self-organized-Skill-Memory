import importlib.util
from pathlib import Path

import numpy as np


MODULE = Path(__file__).parents[2] / 'skill_moe/skillnet/examples/memory_pilot/unseen_libero/denoise_oracle_drafts.py'
SPEC = importlib.util.spec_from_file_location('denoise_oracle_drafts_test', MODULE)
den = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(den)


def test_arm_only_selection_does_not_use_gripper():
    target = np.zeros((1, 10, 7), np.float32)
    cand = np.zeros((2, 10, 7), np.float32)
    cand[0, :, 6] = 1
    cand[1, :, 0] = .1
    idx, _ = den.nearest_arm_candidates(target, cand)
    assert idx.tolist() == [0]


def test_arm6_and_gripper_contributions_sum_to_active7():
    target = np.zeros((3, 10, 7), np.float32)
    pred = np.ones_like(target)
    m = den.action_metrics(pred, target)
    np.testing.assert_allclose(m['arm6_contrib'] + m['grip_contrib'], m['active7'])
    np.testing.assert_allclose(m['arm6_contrib'], 6 / 7)
    np.testing.assert_allclose(m['grip_contrib'], 1 / 7)


def test_dct3_keeps_fixed_action_shape_and_values_are_unmatched():
    import compression_matched as cmp2
    x = np.random.default_rng(1).normal(size=(4, 10, 7)).astype(np.float32)
    y = cmp2.dct_decode(cmp2.dct_encode(x, 3), 10).astype(np.float32)
    assert y.shape == x.shape
    # No fitted scale, rotation, clipping or channel substitution.
    assert np.isfinite(y).all()
    assert not np.array_equal(y, x)


def test_zero_noise_grid_and_selection_values_are_fixed():
    assert den.T0S == (0.0, 0.1, 0.2, 0.4, 0.6, 1.0)


def test_bootstrap_groups_rows_by_task_and_episode():
    values = np.array([0.0, 2.0, 8.0, 10.0])
    task = np.array([3, 3, 3, 3])
    episode = np.array([11, 11, 29, 29])
    result = den._ep_bootstrap(values, task, episode, reps=2000, seed=9)
    assert result['episodes'] == 2
    assert result['rows'] == 4
    assert result['mean'] == 5.0
    # Cluster resampling can only draw episode means 1 and 9, not row means.
    assert result['ci95'][0] > 0.0
    assert result['ci95'][1] < 10.0


def test_final_gripper_is_copied_from_ordinary_pi05_for_all_conditions():
    rng = np.random.default_rng(10)
    denoised = rng.normal(size=(3, 10, 7)).astype(np.float32)
    ordinary = rng.normal(size=(3, 10, 7)).astype(np.float32)
    fixed = den.use_pi05_gripper(denoised, ordinary)
    np.testing.assert_array_equal(fixed[:, :, :6], denoised[:, :, :6])
    np.testing.assert_array_equal(fixed[:, :, 6], ordinary[:, :, 6])
    assert not np.shares_memory(fixed, denoised)
    target = np.zeros_like(fixed)
    m = den.action_metrics(fixed, target)
    np.testing.assert_allclose(m['arm6_contrib'] + m['grip_contrib'], m['active7'])
    paired = den.summarize_condition(fixed, target, np.array([True, False, True]),
                                     np.array([1, 1, 2]), np.array([4, 4, 7]), fixed, ordinary)
    assert 'mse6_sum_div7' in paired['error_delta_vs']['ordinary_pi05']['all']

