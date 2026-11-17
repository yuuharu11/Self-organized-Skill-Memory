import numpy as np
import pytest
import torch

from skill_moe.skillnet.examples.memory_pilot.unseen_libero import gripper_from_latent as gh1


def test_head_consumes_latents_and_current_gripper_state_only():
    model = gh1.LatentGripperHead(12, hidden_dim=16, heads=4, auxiliary_switch=True)
    out = model(torch.randn(3, 5, 12), torch.randn(3, 2))
    assert out['logits'].shape == (3, 10)
    assert out['switch_horizon_logits'].shape == (3, 11)
    gh1.validate_model_inputs({'latent_tokens': np.zeros((1, 5, 12)), 'gripper_state': np.zeros((1, 2))})


@pytest.mark.parametrize('forbidden', ['target', 'pi05_actions', 'task_id', 'future_state'])
def test_model_input_guard_rejects_targets_pi05_and_identity(forbidden):
    inputs = {'latent_tokens': np.zeros((1, 5, 12)), 'gripper_state': np.zeros((1, 2)), forbidden: 0}
    with pytest.raises(ValueError, match='forbidden'):
        gh1.validate_model_inputs(inputs)


def test_mlp_baseline_has_ten_logits_and_requires_two_state_values():
    model = gh1.LatentGripperMLP(8, hidden_dim=16)
    assert model(torch.randn(2, 5, 8), torch.randn(2, 2))['logits'].shape == (2, 10)
    assert model(torch.randn(2, 8), torch.randn(2, 2))['logits'].shape == (2, 10)
    with pytest.raises(ValueError):
        model(torch.randn(2, 5, 8), torch.randn(2, 3))


def test_pi05_control_uses_prefix_mean_and_two_current_gripper_state_values():
    tokens = np.arange(2 * 3 * 4, dtype=np.float16).reshape(2, 3, 4)
    groups = np.array([[0, 1, -1], [0, -1, -1]], np.int8)
    part = {'tokens': tokens, 'groups': groups, 'state': np.arange(16, dtype=np.float32).reshape(2, 8),
            'actions': np.zeros((2, 10, 7), np.float32), 'task': np.array([1, 2]),
            'history': np.zeros((2, 10, 7), np.float32), 'history_mask': np.ones((2, 10), bool)}
    got = gh1.pooled_pi05_features({'train': part})['train']
    np.testing.assert_allclose(got['latent_tokens'][0], tokens[0, :2].astype(np.float32).mean(0))
    np.testing.assert_array_equal(got['gripper_state'], part['state'][:, 6:8])
    assert got['target'].shape == (2, 10)


def test_grip_mse_sign_mapping_and_switch_metrics():
    target = np.array([[-1] * 5 + [1] * 5, [1] * 10], np.float32)
    logits = np.array([[-2] * 5 + [2] * 5, [2] * 10], np.float32)
    got = gh1.evaluate_logits(logits, target)
    assert got['gripper_contribution_mse_sign'] == pytest.approx(0.)
    assert got['switch_f1'] == pytest.approx(1.)
    assert got['switch_time_error_steps'] == pytest.approx(0.)
    assert got['false_switch_rate'] == pytest.approx(0.)


def test_active7_reuses_grip1_scoring_for_frozen_arm_plus_gripper():
    target = np.zeros((2, 10, 7), np.float32)
    arm = np.ones((2, 10, 6), np.float32)
    grip = -np.ones((2, 10), np.float32)
    got = gh1.active7_with_gripper(arm, target, grip)
    assert got['arm6_mse'] == pytest.approx(6 / 7)
    assert got['gripper_contribution_mse'] == pytest.approx(1 / 7)
    assert got['active7_mse'] == pytest.approx(1.)


def test_joint_retrieval_arm_metric_uses_scored_top1_candidate():
    scores = np.array([[.1, .9, -np.inf], [.8, .2, .1]])
    errors = np.array([[.3, .05, .01], [.07, .2, .4]])
    assert gh1.top1_candidate_arm6_mse(scores, errors) == pytest.approx(.06)
    with pytest.raises(ValueError, match='finite candidate'):
        gh1.top1_candidate_arm6_mse(np.full((1, 2), -np.inf), np.zeros((1, 2)))


def test_switch_horizon_uses_first_future_transition_or_no_switch_class():
    y = np.array([[-1, -1, 1, 1, 1, 1, 1, 1, 1, 1], [1] * 10])
    np.testing.assert_array_equal(gh1.switch_horizon_labels(y), [2, 10])


def test_switch_presence_f1_and_false_switch_rate_are_window_level():
    target = np.array([[-1] * 5 + [1] * 5, [1] * 10], np.float32)
    pred = np.array([[-1] * 5 + [1] * 5, [-1] * 5 + [1] * 5], np.float32)
    metrics = gh1.switch_metrics(pred, target)
    assert metrics['switch_f1'] == pytest.approx(2 / 3)
    assert metrics['switch_event_f1'] == pytest.approx(2 / 3)
    assert metrics['false_switch_rate'] == pytest.approx(1.)
    assert metrics['switch_time_error_steps'] == pytest.approx(0.)


def test_joint_objective_adds_lambda_times_bce():
    retrieval = torch.tensor(2., requires_grad=True)
    logits = torch.zeros(2, 10, requires_grad=True)
    target = torch.ones(2, 10)
    total = gh1.retrieval_plus_gripper_loss(retrieval, logits, target, gripper_weight=.5)
    torch.testing.assert_close(total, retrieval + .5 * np.log(2))
    total.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    with pytest.raises(ValueError):
        gh1.retrieval_plus_gripper_loss(retrieval, logits, target, gripper_weight=-1)


def test_dev_selection_uses_only_dev_gripper_mse():
    selected = gh1.dev_select([
        {'name': 'a', 'dev_gripper_mse': .03},
        {'name': 'b', 'dev_gripper_mse': .02},
    ])
    assert selected['name'] == 'b'
    with pytest.raises(ValueError, match='Dev metrics only'):
        gh1.dev_select([{'name': 'leaky', 'dev_gripper_mse': .01, 'eval_mse': 0.}])


def test_l90x_manifest_uses_la2_task_exclusion_contract():
    gh1.assert_l90x_manifest({'tasks': [1, 2], 'episodes': [{'task_id': 1}]})
    with pytest.raises(AssertionError):
        gh1.assert_l90x_manifest({'tasks': [6], 'episodes': [{'task_id': 1}]})


def test_training_rows_must_be_memory_and_disjoint_from_eval_demos():
    rows = {'task': np.array([0, 0, 1]), 'episode': np.array([1, 2, 3]),
            'memory': np.array([True, True, False])}
    gh1.assert_training_rows(rows, np.array([0]), eval_pairs={(0, 2)})
    with pytest.raises(AssertionError, match='memory'):
        gh1.assert_training_rows(rows, np.array([2]))
    with pytest.raises(AssertionError, match='evaluation demos'):
        gh1.assert_training_rows(rows, np.array([1]), eval_pairs={(0, 2)})


def test_key2_saved_prediction_alignment_uses_target_actions_field():
    target = np.zeros((2, 10, 7), np.float32)
    key = np.array([[1, 2, 3], [4, 5, 6]], np.int32)
    archive = {'target_actions': target.copy(), 'key': key.copy(),
               'blend_k1': np.zeros_like(target)}
    gh1.assert_saved_prediction_alignment(archive, target, key, source='p5 learned')
    with pytest.raises(AssertionError, match='do not align'):
        gh1.assert_saved_prediction_alignment(archive, target + 1, key, source='p5 learned')


def test_l90x_augmentation_concatenates_only_after_manifest_gate():
    hist = lambda n: {'history_action': np.zeros((n, 10), np.float32), 'history_valid': np.ones((n, 10), bool)}
    train = {'latent_tokens': np.zeros((2, 5, 4), np.float32),
             'gripper_state': np.zeros((2, 2), np.float32), 'target': -np.ones((2, 10), np.float32), **hist(2)}
    extra = {'latent_tokens': np.ones((1, 5, 4), np.float32),
             'gripper_state': np.ones((1, 2), np.float32), 'target': np.ones((1, 10), np.float32), **hist(1)}
    got = gh1.append_l90x_training_data(train, extra, {'tasks': [1, 2]})
    assert got['latent_tokens'].shape == (3, 5, 4)
    assert got['history_action'].shape == (3, 10)
    with pytest.raises(AssertionError):
        gh1.append_l90x_training_data(train, extra, {'tasks': [6]})


def test_frozen_key2_e_encoding_separates_latent_state_and_labels(tmp_path):
    from skill_moe.skillnet.examples.memory_pilot.unseen_libero import key_shared_latent as key2
    torch.manual_seed(9)
    model = key2._new_model('e_vlm_h')
    checkpoint = tmp_path / 'e.pt'
    torch.save(model.state_dict(), checkpoint)
    part = {'key': np.array([[1, 2, 3]], np.int32), 'tokens': np.zeros((1, 4, 2048), np.float32),
            'groups': np.array([[0, 1, 2, 3]], np.int8), 'history': np.zeros((1, 10, 7), np.float32),
            'history_mask': np.zeros((1, 10), bool), 'state': np.zeros((1, 8), np.float32),
            'actions': -np.ones((1, 10, 7), np.float32), 'task': np.array([1]), 'episode': np.array([2])}
    got = gh1.encode_frozen_key2_e(checkpoint, {'train': part}, device='cpu', batch_size=1)['train']
    assert got['latent_tokens'].shape == (1, 5, 256)
    assert got['gripper_state'].shape == (1, 2)
    assert got['target'].shape == (1, 10)
    assert np.all(got['target'] == -1)
    gh1.validate_model_inputs({k: got[k] for k in ('latent_tokens', 'gripper_state')})


def test_fit_head_selects_checkpoint_with_dev_only_and_keeps_eval_outside_fit():
    rng = np.random.default_rng(2)
    def part(n):
        x = rng.normal(size=(n, 3, 6)).astype(np.float32)
        state = rng.normal(size=(n, 2)).astype(np.float32)
        target = np.repeat(np.where(x[:, 0, 0, None] > 0, 1., -1.).astype(np.float32), 10, axis=1)
        return {'latent_tokens': x, 'gripper_state': state, 'target': target}
    train, dev, evaluation = part(12), part(4), part(3)
    model, record = gh1.fit_head(gh1.LatentGripperMLP(6, hidden_dim=8), train, dev,
                                 config=gh1.HeadConfig('tiny', epochs=2, learning_rate=.01), num_threads=1)
    with torch.no_grad():
        logits = model(torch.tensor(evaluation['latent_tokens']), torch.tensor(evaluation['gripper_state']))['logits']
    assert logits.shape == (3, 10)
    assert np.isfinite(record['dev_bce_at_selected_checkpoint'])
    assert 'evaluation' not in record


def test_dev_grid_scores_configs_without_eval_and_bootstraps_pi05_pairs():
    rng = np.random.default_rng(8)
    def part(n, *, eval_rows=False):
        x = rng.normal(size=(n, 3, 5)).astype(np.float32)
        state = rng.normal(size=(n, 2)).astype(np.float32)
        target = np.repeat(np.where(state[:, 0, None] > 0, 1., -1.).astype(np.float32), 10, axis=1)
        result = {'latent_tokens': x, 'gripper_state': state, 'target': target}
        if eval_rows:
            result.update(task=np.array([0] * n), episode=np.arange(n), hard=np.arange(n) % 2 == 0)
        return result
    train, dev, evaluation = part(12), part(4), part(4, eval_rows=True)
    model, selected, records = gh1.fit_dev_grid(train, dev, kinds=('mlp',), hidden_dims=(8,),
                                                learning_rates=(.01,), epochs_grid=(1,), num_threads=1)
    logits = gh1.predict(model, evaluation)
    metrics = gh1.evaluate_split(logits, evaluation, pi05_actions=np.ones((4, 10), np.float32),
                                 bootstrap_reps=40)
    assert len(records) == 1 and selected['kind'] == 'mlp'
    assert gh1.gripper_contribution_mse(np.where(gh1.predict(model, dev) >= 0, 1., -1.), dev['target']).mean() == pytest.approx(selected['dev_gripper_mse'])
    assert metrics['n_all'] == 4 and metrics['n_hard'] == 2
    assert metrics['pi05']['paired_head_minus_pi05']['ci95']
