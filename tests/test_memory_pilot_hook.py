import numpy as np

from openpi.models.pi0_moe_skill import _apply_skill_embed_override
from openpi.models.gemma_moe_skill import _selected_expert
import jax.numpy as jnp
import jax
import flax.linen as nn


def test_disabled_skill_embedding_hook_preserves_original_array_exactly():
    original = np.arange(3 * 4 * 64, dtype=np.float32).reshape(3, 4, 64)
    result = _apply_skill_embed_override(original, None)
    assert result is original
    np.testing.assert_array_equal(result, original)


def test_skill_embedding_hook_uses_explicit_override():
    original = np.zeros((1, 4, 64), dtype=np.float32)
    replacement = np.ones_like(original)
    assert _apply_skill_embed_override(original, replacement) is replacement


def test_captured_top1_matches_dispatched_expert_when_router_logits_tie():
    # An all-masked skill set gives the bias-free router equal probabilities.
    # Actual dispatch uses argsort(...)[..., -1], not argmax.
    probs = jnp.full((1, 2, 4), 0.25)
    dispatched = jnp.argsort(probs, axis=-1)[..., -1:]
    captured = np.asarray(_selected_expert(dispatched))
    np.testing.assert_array_equal(captured, np.asarray(dispatched[..., 0]))
    assert np.all(captured == 3)
    assert not np.any(captured == np.asarray(jnp.argmax(probs, axis=-1)))


def test_moe_valid_skill_path_matches_legacy_precision_and_norms_match_dispatch(monkeypatch):
    from openpi.models import gemma_moe_skill as gemma

    x = jax.random.normal(jax.random.key(4), (2, 64, 16), dtype=jnp.bfloat16)
    skills = jax.random.normal(jax.random.key(5), (2, 4, 8), dtype=jnp.bfloat16)
    mask = jnp.array([[True, True, True, False], [True, True, True, True]])
    module = gemma.MoEFeedForward(num_experts=4, hidden_dim=32, output_dim=16)
    variables = module.init(jax.random.key(6), x, skills, mask)
    output, captured = module.apply(variables, x, skills, mask, capture_intermediates=True, mutable=["intermediates"])
    y, _, selected, selected_norm, shared_norm = output
    intermediates = captured["intermediates"]
    experts = np.stack([np.asarray(intermediates[f"expert_{i}"]["__call__"][0], dtype=np.float32)
                        for i in range(4)], axis=-2)
    shared = np.asarray(intermediates["shared_expert"]["__call__"][0], dtype=np.float32)
    # Output minus shared is exactly the dispatched (router-weighted) expert.
    # Addition rounds in bfloat16, so compare norms within that rounding error.
    np.testing.assert_allclose(np.asarray(shared_norm), np.linalg.norm(shared, axis=-1), rtol=1e-6)
    contribution = np.asarray(y, dtype=np.float32) - shared
    np.testing.assert_allclose(np.asarray(selected_norm), np.linalg.norm(contribution, axis=-1), rtol=.05, atol=.02)
    chosen = np.take_along_axis(experts, np.asarray(selected)[..., None, None], axis=-2)[..., 0, :]
    # The contribution must be collinear with the selected expert (positive router gate).
    assert np.all(np.sum(chosen * contribution, axis=-1) >= 0)

    def legacy_attention(scores, valid):
        return nn.softmax(jnp.where(valid[:, None, None, :], scores, -1e9), axis=-1).astype(x.dtype)

    monkeypatch.setattr(gemma, "_masked_skill_softmax", legacy_attention)
    legacy = module.apply(variables, x, skills, mask)
    for actual, expected in zip(output, legacy):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_transformer_capture_preserves_numerical_output():
    from openpi.models import gemma_moe_skill as gemma

    configs = [gemma.Config(width=8, depth=2, mlp_dim=16, num_heads=2, num_kv_heads=1, head_dim=4),
               gemma.Config(width=8, depth=2, mlp_dim=16, num_heads=2, num_kv_heads=1, head_dim=4,
                            moe=True, expert_num=4)]
    module = gemma.Module(configs=configs, embed_dtype="bfloat16", skill_embed_dim=8)
    embedded = [jnp.ones((1, 2, 8)), jnp.ones((1, 3, 8))]
    kwargs = {"skill_embeds": jnp.arange(32, dtype=jnp.float32).reshape(1, 4, 8),
              "skill_mask": jnp.array([[True, True, False, False]]),
              "positions": jnp.arange(5)[None, :], "mask": jnp.ones((1, 5, 5), dtype=bool)}
    # This model defines an init convenience method for NNX; initialize the
    # __call__ path through Linen's base implementation in this isolated test.
    variables = nn.Module.init(module, jax.random.key(0), embedded, **kwargs)
    plain = module.apply(variables, embedded, **kwargs)
    captured = module.apply(variables, embedded, capture_auxiliary=True, **kwargs)
    for a, b in zip(jax.tree.leaves(plain), jax.tree.leaves(captured[:3])):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    features, routes, selected_norm, shared_norm = captured[3]
    assert features.shape == (2, 1, 3, 8)
    assert routes.shape == selected_norm.shape == shared_norm.shape == (2, 1, 3)
