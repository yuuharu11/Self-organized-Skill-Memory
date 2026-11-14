import numpy as np

from skill_moe.skillnet.examples.memory_pilot.unseen_libero.gripper_split import (
    GripperPolicyHook,
    action_parts_mse,
    dct_roundtrip,
    episode_codes,
    exclude_episode_rows,
    memory_external_nonvalidation_rows,
    majority_gripper_sign,
    nearest_candidates,
    top_error_mask,
    tau_neighbor_matrix,
    variable_dct3_parts,
)


def test_arm_and_gripper_mse_sum_to_active7():
    rng = np.random.default_rng(4)
    target = rng.normal(size=(5, 10, 7))
    pred = rng.normal(size=(5, 10, 7))
    arm, grip, total = action_parts_mse(pred, target)
    np.testing.assert_allclose(arm + grip, total, atol=1e-12)
    np.testing.assert_allclose(total, ((pred-target)**2).mean(axis=(1,2)))


def test_nearest_candidate_uses_only_requested_axes():
    target = np.zeros((1, 10, 7), np.float32)
    candidates = np.ones((2, 10, 7), np.float32)
    candidates[0, :, 6] = 10
    candidates[1, :, :6] = 2
    idx_arm, _ = nearest_candidates(target, candidates, np.arange(6))
    idx_all, _ = nearest_candidates(target, candidates, np.arange(7))
    assert idx_arm[0] == 0
    assert idx_all[0] == 1


def test_fixed_dct_roundtrip_preserves_shape_and_constant_grip():
    x = np.zeros((3, 10, 7), np.float32)
    x[..., 6] = -1
    y = dct_roundtrip(x, 3)
    assert y.shape == x.shape
    np.testing.assert_allclose(y[..., 6], -1, atol=1e-6)


class _MockPolicy:
    def __init__(self):
        self.hist = {'e': [np.zeros(7, np.float32) for _ in range(5)]}
    def transform(self, raw):
        return {'state': np.zeros(8, np.float32)}
    def infer(self, obs):
        return {'actions': np.zeros((10,7), np.float32), 's2_debug': {'unit':0}}
    def unnormalize(self, x):
        return {'actions': np.asarray(x['actions']).copy()}


def test_hook_changes_only_gripper_and_updates_history():
    policy = _MockPolicy()
    hook = GripperPolicyHook(policy, 'unit', np.array([-1.]))
    obs={'observation/image':0,'observation/wrist_image':0,'observation/state':0,'prompt':'',
         's2_episode_id':'e','s2_query_id':1}
    result=hook.infer(obs)
    np.testing.assert_array_equal(result['actions'][:,:6],0)
    np.testing.assert_array_equal(result['actions'][:,6],-1)
    np.testing.assert_array_equal([a[6] for a in policy.hist['e']],-1)


def test_bootstrap_cluster_ids_include_task_namespace():
    codes = episode_codes(np.array([1, 2, 1, 2]), np.array([7, 7, 8, 8]))
    assert len(np.unique(codes)) == 4
    assert codes[0] != codes[1]


def test_classifier_training_excludes_whole_heldout_episodes():
    task = np.array([0, 0, 0, 1, 1, 1])
    episode = np.array([4, 4, 5, 4, 4, 5])
    candidates = np.arange(6)
    kept = exclude_episode_rows(candidates, task, episode, np.array([0, 3]))
    assert kept.tolist() == [2, 5]


def test_memory_external_split_excludes_demo_if_any_row_is_in_memory():
    rows = {'task': np.array([0, 0, 0, 1, 1]),
            'episode': np.array([2, 2, 3, 2, 2]),
            'memory': np.array([True, False, False, False, False]),
            'validation': np.zeros(5, bool)}
    assert memory_external_nonvalidation_rows(rows).tolist() == [2, 3, 4]


def test_variable_dct_gripper_reconstruction_is_measured_not_assumed_zero():
    x = np.zeros((5, 7), np.float32)
    x[:, 6] = np.array([-1, -1, 1, 1, 1], np.float32)
    arm, grip, total = variable_dct3_parts(x)
    assert grip > 0
    np.testing.assert_allclose(total, arm + grip)


def test_hard_rows_are_upper_quintile_within_each_evaluation_set():
    hard = top_error_mask(np.arange(600, dtype=np.float64), .2)
    assert int(hard.sum()) == 120
    assert np.all(np.flatnonzero(hard) >= 480)


def test_hook_resets_previous_grip_at_episode_restart():
    policy = _MockPolicy()
    hook = GripperPolicyHook(policy, 'previous', np.array([-1.]))
    hook.last_grip['e'] = 1.
    obs={'observation/image':0,'observation/wrist_image':0,'observation/state':0,'prompt':'',
         's2_episode_id':'e','s2_query_id':0}
    result=hook.infer(obs)
    np.testing.assert_array_equal(result['actions'][:,6],0.)


def test_tied_unit_gripper_uses_same_positive_tie_break_as_offline_encoding():
    np.testing.assert_array_equal(majority_gripper_sign(np.array([-0.2, 0., 0.2])), [-1., 1., 1.])
    policy = _MockPolicy()
    hook = GripperPolicyHook(policy, 'unit', np.array([0.]))
    obs={'observation/image':0,'observation/wrist_image':0,'observation/state':0,'prompt':'',
         's2_episode_id':'e','s2_query_id':1}
    result=hook.infer(obs)
    np.testing.assert_array_equal(result['actions'][:,6],1.)


def test_tau_cover_uses_six_axis_denominator_without_gripper():
    coeff = np.zeros((2, 19), np.float64)
    coeff[1, 0] = np.sqrt(.5)
    with_grip = tau_neighbor_matrix(coeff, include_gripper=True, tau=.08)
    without_grip = tau_neighbor_matrix(coeff, include_gripper=False, tau=.08)
    assert with_grip[0, 1]
    assert not without_grip[0, 1]
