import numpy as np
import torch

from skill_moe.skillnet.examples.memory_pilot.unseen_libero import key_attn_diag as diag
from skill_moe.skillnet.examples.memory_pilot.unseen_libero import key_shared_latent as key2


def _part(n=3):
    rng = np.random.default_rng(7)
    groups = np.array([[0, 1, 2, -1]] * n, np.int8)
    return {
        'tokens': rng.normal(size=(n, 4, 2048)).astype(np.float32),
        'groups': groups,
        'history': rng.normal(size=(n, 10, 7)).astype(np.float32),
        'history_mask': np.array([[0] * 7 + [1] * 3] * n, bool),
        'state': rng.normal(size=(n, 8)).astype(np.float32),
        'key': np.arange(n * 3).reshape(n, 3),
    }


def test_observer_keeps_original_forward_exact_and_attention_normalized():
    torch.manual_seed(4)
    model = key2._new_model('e_vlm_h').eval()
    part = _part()
    with torch.no_grad():
        original = key2._model_batch_encode(model, 'e_vlm_h', part, 'cpu')
        weights = diag.cross_attention_weights(model, part, 'cpu')
        observed = key2._model_batch_encode(model, 'e_vlm_h', part, 'cpu')
    assert torch.equal(original, observed)
    assert weights.shape == (3, 8, 5, 4 + 10 + 1)
    assert torch.allclose(weights.sum(-1), torch.ones_like(weights.sum(-1)), atol=1e-6)
    grouped, counts = diag.aggregate_attention(weights, part, True, False, True, True)
    assert grouped.shape == (3, 8, 5, 5)
    assert torch.allclose(grouped.sum(-1), torch.ones_like(grouped.sum(-1)), atol=1e-6)
    assert counts[0].tolist() == [2, 1, 0, 3, 1]


def test_observer_attention_matches_torch_multihead_attention_weights():
    torch.manual_seed(5)
    model = key2._new_model('e_vlm_h').eval()
    part = _part()
    memory, padding, _ = diag.shared_memory(model, part, 'cpu')
    q = model.cross.q_norm(model.latent.expand(len(memory), -1, -1))
    kv = model.cross.kv_norm(memory)
    with torch.no_grad():
        _, expected = model.cross.attn(q, kv, kv, key_padding_mask=padding,
                                      need_weights=True, average_attn_weights=False)
        observed = diag.cross_attention_weights(model, part, 'cpu')
    assert torch.allclose(observed, expected, atol=1e-6, rtol=1e-6)


def test_attention_mass_per_token_uses_mean_valid_token_count():
    mass = np.array([[[0.4, 0.3, 0.2, 0.1, 0.0]]], np.float64)
    mean_count = np.array([2.0, 3.0, 0.0, 5.0, 1.0])
    per_token = diag.per_input_token_mass(mass, mean_count)
    np.testing.assert_allclose(per_token, [[[0.2, 0.1, 0.2, 0.02, 0.0]]])


def test_ablation_zero_and_swap_are_deterministic_and_do_not_mutate_source():
    part = _part()
    before = part['tokens'].copy()
    zero = diag.apply_query_ablation(part, 'vlm', 'zero', np.random.default_rng(1))
    swap1 = diag.apply_query_ablation(part, 'vlm', 'swap', np.random.default_rng(8))
    swap2 = diag.apply_query_ablation(part, 'vlm', 'swap', np.random.default_rng(8))
    assert not np.any(zero['tokens'])
    assert np.array_equal(swap1['tokens'], swap2['tokens'])
    assert np.array_equal(swap1['groups'], part['groups'])
    assert np.array_equal(part['tokens'], before)


def test_latent_collapse_reports_valid_values():
    rng = np.random.default_rng(3)
    z = rng.normal(size=(12, 5, 32)).astype(np.float32)
    z /= np.linalg.norm(z, axis=-1, keepdims=True)
    result = diag.latent_collapse(z)
    assert -1 <= result['mean_pairwise_cosine'] <= 1
    assert 1 <= result['effective_rank_mean'] <= 5
    assert len(result['singular_values_mean']) == 5

