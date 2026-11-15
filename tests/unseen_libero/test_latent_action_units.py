import numpy as np
import pytest
import torch

from skill_moe.skillnet.examples.memory_pilot.unseen_libero.latent_action_units import (
    ActionVQVAE,
    QueryEncoder,
    UnitKeyEncoder,
    choose_dev,
    _fit_code_head,
    _predict_latents,
    _score_and_decode,
    train_vqvae,
    action_error,
    exclude_self_episode,
    listwise_loss,
    pooled_tokens,
    ranking_metrics,
    select_blend_temperatures,
    topk_blend,
    vqvae_loss,
)


def test_pooling_ignores_padding_and_returns_expected_shape():
    tok = np.array([[[1., 2.], [3., 4.], [900., 900.]]], dtype=np.float16)
    groups = np.array([[0, 1, -1]], dtype=np.int8)
    pooled = pooled_tokens(tok, groups)
    np.testing.assert_allclose(pooled, [[2., 3.]])
    assert pooled.shape == (1, 2)


def test_vqvae_encode_decode_shape_and_code_range():
    torch.manual_seed(4)
    model = ActionVQVAE(codes=8, latent_dim=4)
    x = torch.randn(3, 10, 6)
    ids, z, q, rec = model(x)
    assert ids.shape == (3,)
    assert z.shape == q.shape == (3, 4)
    assert rec.shape == (3, 10, 6)
    assert int(ids.max()) < 8
    assert model.decode_code(ids).shape == rec.shape
    assert torch.isfinite(vqvae_loss(rec, x, q, z))


def test_soft_listwise_loss_uses_target_temperature_only_and_exclusion():
    scores = torch.tensor([[0.2, 0.1, 8.0]], requires_grad=True)
    errors = torch.tensor([[0.0, 0.02, 0.0]])
    allowed = torch.tensor([[True, True, False]])
    loss = listwise_loss(scores, errors, allowed, tau=0.01)
    loss.backward()
    assert scores.grad[0, 2].item() == 0.0
    assert torch.isfinite(scores.grad).all()


def test_query_and_unit_encoders_share_latent_size():
    q = QueryEncoder(17, latent_dim=12)(torch.randn(4, 17))
    k = UnitKeyEncoder(23, latent_dim=12)(torch.randn(5, 23))
    assert q.shape == (4, 12) and k.shape == (5, 12)
    torch.testing.assert_close(q.norm(dim=-1), torch.ones(4), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(k.norm(dim=-1), torch.ones(5), atol=1e-6, rtol=1e-6)


def test_self_episode_mask_is_exact():
    allowed = exclude_self_episode([1, 1], [10, 11], [1, 1, 2], [10, 10, 10])
    np.testing.assert_array_equal(allowed, [[False, False, True], [True, True, True]])


def test_dev_selection_uses_only_dev_measure_and_stable_tie_break():
    rows = [{'config_id':'z','dev_top1_arm6_mse':0.2},
            {'config_id':'a','dev_top1_arm6_mse':0.2},
            {'config_id':'b','dev_top1_arm6_mse':0.3}]
    assert choose_dev(rows)['config_id'] == 'a'


def test_topk_blend_is_action_space_and_rank1_is_exact_candidate():
    scores = np.array([[3.0, 1.0, -1.0]], np.float32)
    actions = np.arange(3 * 10 * 7, dtype=np.float32).reshape(3, 10, 7)
    pred1, ix1 = topk_blend(scores, actions, 1)
    np.testing.assert_array_equal(ix1, [[0]])
    np.testing.assert_array_equal(pred1[0], actions[0])
    pred3, ix3 = topk_blend(scores, actions, 3)
    assert pred3.shape == (1, 10, 7)
    assert set(ix3[0].tolist()) == {0, 1, 2}


def test_ranking_metrics_reports_requested_rank_fields():
    scores = np.array([[0.9, 0.1, -0.3], [0.1, 0.4, 0.3]])
    errors = np.array([[0., .03, .1], [.1, .02, .04]])
    m = ranking_metrics(scores, errors, tau=.01)
    assert m['recall_at_k']['1'] == 1.0
    assert 'spearman_score_neg_error' in m and 'kl_teacher_to_prediction' in m
    assert set(m['recall_at_k']) == {'1', '5', '20'}
    assert 'ndcg_at_20' in m and 'mrr' in m and 'key2_recovery' in m


def test_ranking_metrics_ignores_excluded_candidates_without_nan():
    scores = np.array([[0.8, -np.inf, 0.1], [0.2, 0.9, -np.inf]])
    errors = np.array([[0.01, 0.0, 0.03], [0.02, 0.04, 0.0]])
    metrics = ranking_metrics(scores, errors)
    assert np.isfinite(metrics['spearman_score_neg_error'])
    assert np.isfinite(metrics['H_pq_cross_entropy'])
    assert np.isfinite(metrics['kl_teacher_to_prediction'])


def test_topk_temperature_is_selected_from_dev_targets_only():
    scores=np.array([[2.,1.,0.],[0.,1.,2.]],np.float32)
    actions=np.zeros((3,10,7),np.float32)
    actions[0,:,:6]=1.;actions[1,:,:6]=.2;actions[2,:,:6]=.1
    targets=np.stack([actions[0],actions[2]])
    got=select_blend_temperatures(scores,actions,targets,temperatures=(.01,.1,1.))
    assert set(got)=={'1','5','20'}
    assert got['1']['temperature']==1.
    for k in ('5','20'):
        assert got[k]['temperature'] in (.01,.1,1.)
        assert got[k]['dev_mse_gripper_truth']>=0.


def test_arm6_error_denominator_matches_key2_metric():
    target = np.zeros((2, 10, 7), np.float32)
    pred = np.ones_like(target)
    assert np.allclose(action_error(pred, target), 6.0 / 7.0)


def test_cpu_training_smoke_vq_and_shared_query_exit():
    torch.set_num_threads(1)
    rng = np.random.default_rng(9)
    actions = rng.normal(size=(16, 10, 7)).astype(np.float32)
    obs = rng.normal(size=(16, 13)).astype(np.float32)
    vq, info = train_vqvae(actions, 8, epochs=1, learning_rate=1e-3, device='cpu')
    with torch.no_grad():
        labels, _, _, decoded = vq(torch.as_tensor(actions[..., :6]))
    assert decoded.shape == (16, 10, 6)
    assert 1 <= info['used_codes'] <= 8
    qnet = QueryEncoder(obs.shape[-1], latent_dim=128)
    head = _fit_code_head(qnet, obs, labels.numpy(), epochs=1, learning_rate=1e-3,
                          device='cpu', seed=12)
    q = _predict_latents(qnet, obs[:3], 'cpu')
    assert q.shape == (3, 128)
    assert head(qnet(torch.as_tensor(obs[:3]))).shape == (3, 8)
