import numpy as np
import pytest
import torch

from skill_moe.skillnet.examples.memory_pilot.unseen_libero import latent_action_units_la2 as la2


def test_l90x_manifest_rejects_dev_eval_tasks():
    la2.assert_l90x_manifest({'tasks': [1, 2], 'dev_tasks': [6, 7], 'eval_tasks': [0, 10]})
    with pytest.raises(AssertionError):
        la2.assert_l90x_manifest({'tasks': [6], 'episodes': [{'task_id': 6}], 'dev_tasks': [6], 'eval_tasks': []})


def test_task_names_prevent_cross_namespace_numeric_false_exclusion():
    candidates = {'task': np.array([6, 2]), 'episode': np.array([3, 3])}
    q = {'task': np.array([6]), 'episode': np.array([3]), 'domain': np.array([1])}
    # A LIBERO-90 task ID 6 with no matching instruction is not LIBERO-40 task 6.
    allowed = la2._allowed_matrix(q, candidates, {6: 'other task'}, {6: 'libero90 task'})
    assert allowed.tolist() == [[True, True]]


def test_matching_instruction_excludes_only_same_demo():
    candidates = {'task': np.array([6, 6, 2]), 'episode': np.array([3, 4, 3])}
    q = {'task': np.array([6]), 'episode': np.array([3]), 'domain': np.array([1])}
    allowed = la2._allowed_matrix(q, candidates, {6: 'pick up object'}, {6: 'pick up object'})
    assert allowed.tolist() == [[False, True, True]]


def test_lib40_query_excludes_same_task_episode():
    candidates = {'task': np.array([4, 4, 5]), 'episode': np.array([8, 9, 8])}
    q = {'task': np.array([4]), 'episode': np.array([8]), 'domain': np.array([0])}
    allowed = la2._allowed_matrix(q, candidates, {}, {})
    assert allowed.tolist() == [[False, True, True]]


def test_teacher_errors_are_chunked_pairwise_arm6_mse():
    q = {'actions': np.zeros((2, 10, 7), np.float32)}
    c = np.ones((3, 10, 7), np.float32)
    got = la2._teacher_errors(q, c, chunk=1)
    assert got.shape == (2, 3)
    assert np.allclose(got, 60 / 70)


def test_composite_query_and_unit_encoders_return_normalized_concatenated_codes():
    torch.manual_seed(3)
    qnet = la2.CompositeQueryEncoder(input_dim=14, dim=8)
    knet = la2.CompositeUnitEncoder(action_dim=60, dim=8)
    q = qnet(torch.randn(3, 14), torch.randn(3, 2048))
    k = knet(torch.randn(5, 60), torch.randn(5, 2048))
    assert q.shape == (3, 16) and k.shape == (5, 16)
    assert torch.allclose(q.norm(dim=-1), torch.ones(3), atol=1e-5)
    assert torch.allclose(k.norm(dim=-1), torch.ones(5), atol=1e-5)


def test_rerank_mask_limits_candidate_set_exactly():
    base = np.array([[8., 7., 6., 5., 4.]])
    top = np.zeros_like(base, dtype=bool); top[:, :2] = True
    latent = np.array([[0., 1., 100., 99., 98.]])
    ranked = np.where(top, latent, -np.inf)
    assert int(ranked.argmax(1)[0]) == 1
    assert np.all(~np.isfinite(ranked[~top]))


def test_fixed_la1_grid_and_la2_extension_values():
    assert la2.LRS == (1e-4, 3e-4)
    assert la2.MAX_EPOCHS == (16, 32, 64)
