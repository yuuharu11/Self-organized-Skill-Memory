import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from skill_moe.skillnet.examples.memory_pilot.injection.adapter import MemoryInjection, add_memory_residual


def _adapter(mode, *, dz=7, width=12):
    return MemoryInjection(
        mode=mode,
        dz=dz,
        width=width,
        hidden_dim=9,
        attention_dim=6,
        rngs=nnx.Rngs(0),
    )


def test_pooled_modes_are_exact_zero_and_disabled_path_is_exact_identity():
    c = jnp.arange(21, dtype=jnp.float32).reshape(3, 7) / 10
    base = jnp.arange(3 * 4 * 12, dtype=jnp.float32).reshape(3, 4, 12)
    for mode in ("action_tokens", "adarms_cond"):
        adapter = _adapter(mode)
        residual = adapter(c=c, horizon=4) if mode == "action_tokens" else adapter(c=c)
        expected_shape = (3, 4, 12) if mode == "action_tokens" else (3, 12)
        np.testing.assert_array_equal(np.asarray(residual), np.zeros(expected_shape, np.float32))
        applied = add_memory_residual(base, residual if mode == "action_tokens" else residual[:, None, :], enabled=True)
        np.testing.assert_array_equal(np.asarray(applied), np.asarray(base))
        disabled = add_memory_residual(base, jnp.ones_like(base), enabled=False)
        assert disabled is base
        np.testing.assert_array_equal(np.asarray(disabled), np.asarray(base))


def test_zero_residual_preserves_bfloat16_activations_and_adarms_condition():
    adapter = _adapter("adarms_cond")
    cond = (jnp.arange(24, dtype=jnp.float32).reshape(2, 12) / 7).astype(jnp.bfloat16)
    residual = adapter(c=jnp.ones((2, 7), jnp.float32))
    applied = add_memory_residual(cond, residual, enabled=True)
    assert applied.dtype == cond.dtype
    np.testing.assert_array_equal(np.asarray(applied), np.asarray(cond))


def test_pooled_zero_projection_receives_gradient_at_initialization():
    adapter = _adapter("action_tokens")
    c = jnp.arange(14, dtype=jnp.float32).reshape(2, 7) / 10 + 0.2
    target = jnp.arange(24, dtype=jnp.float32).reshape(2, 12) / 10 + 0.1

    def loss(module):
        return jnp.sum(module(c=c, horizon=2) * target)

    grads = nnx.grad(loss)(adapter)
    assert np.any(np.asarray(grads.condition_out.kernel.value) != 0)
    assert np.any(np.asarray(grads.condition_out.bias.value) != 0)
    assert not np.any(np.asarray(grads.condition_in.kernel.value) != 0)

    # After the zero output layer moves, gradients reach the preceding layer.
    adapter.condition_out.kernel.value = jnp.full_like(adapter.condition_out.kernel.value, 0.01)
    grads_after_output_update = nnx.grad(loss)(adapter)
    assert np.any(np.asarray(grads_after_output_update.condition_in.kernel.value) != 0)


def test_candidate_attention_zero_gate_is_exact_and_learns_from_gradient():
    adapter = _adapter("candidate_attention")
    actions = jnp.arange(2 * 3 * 12, dtype=jnp.float32).reshape(2, 3, 12) / 10 + 0.2
    candidates = jnp.arange(2 * 4 * 7, dtype=jnp.float32).reshape(2, 4, 7) / 10 + 0.3
    mask = jnp.asarray([[True, True, False, False], [True, False, False, False]])
    residual = adapter(action_tokens=actions, candidates=candidates, candidate_mask=mask)
    np.testing.assert_array_equal(np.asarray(residual), np.zeros((2, 3, 12), np.float32))

    target = jnp.arange(2 * 3 * 12, dtype=jnp.float32).reshape(2, 3, 12) / 10 + 0.4
    grads = nnx.grad(lambda module: jnp.sum(module(action_tokens=actions, candidates=candidates, candidate_mask=mask) * target))(adapter)
    assert float(jnp.abs(grads.gate.value)) > 0
    assert not np.any(np.asarray(grads.value.kernel.value) != 0)

    # With a nonzero gate, candidate projections start receiving gradients.
    adapter.gate.value = jnp.asarray(0.1, jnp.float32)
    grads_after_gate_update = nnx.grad(
        lambda module: jnp.sum(module(action_tokens=actions, candidates=candidates,
                                     candidate_mask=mask) * target)
    )(adapter)
    assert np.any(np.asarray(grads_after_gate_update.value.kernel.value) != 0)
    assert np.any(np.asarray(grads_after_gate_update.query.kernel.value) != 0)
    assert np.any(np.asarray(grads_after_gate_update.key.kernel.value) != 0)


def test_all_masked_candidates_return_exact_zero():
    adapter = _adapter("candidate_attention")
    actions = jnp.ones((2, 3, 12), jnp.float32)
    candidates = jnp.ones((2, 4, 7), jnp.float32)
    mask = jnp.zeros((2, 4), jnp.bool_)
    result = adapter(action_tokens=actions, candidates=candidates, candidate_mask=mask)
    np.testing.assert_array_equal(np.asarray(result), np.zeros((2, 3, 12), np.float32))

    # Empty memory stays neutral after training has changed the output bias/gate.
    adapter.context_out.bias.value = jnp.ones_like(adapter.context_out.bias.value)
    adapter.gate.value = jnp.asarray(0.7, jnp.float32)
    trained_result = adapter(action_tokens=actions, candidates=candidates, candidate_mask=mask)
    np.testing.assert_array_equal(np.asarray(trained_result), np.zeros((2, 3, 12), np.float32))


def test_memory_payload_survives_observation_conversion_and_preprocessing():
    from openpi.models import model as _model

    features = np.arange(14, dtype=np.float32).reshape(2, 7)
    candidates = np.arange(2 * 3 * 7, dtype=np.float32).reshape(2, 3, 7)
    candidate_mask = np.asarray([[True, False, False], [True, True, False]])
    data = {
        "image": {key: np.zeros((2, 224, 224, 3), np.uint8) for key in _model.IMAGE_KEYS},
        "image_mask": {key: np.ones((2,), np.bool_) for key in _model.IMAGE_KEYS},
        "state": np.zeros((2, 8), np.float32),
        "skills": np.zeros((2, 4), np.int32),
        "skill_mask": np.ones((2, 4), np.bool_),
        "memory_features": features,
        "memory_candidates": candidates,
        "memory_candidate_mask": candidate_mask,
        "memory_enabled": np.asarray(True),
    }
    obs = _model.Observation.from_dict(data)
    processed = _model.preprocess_observation(None, obs)
    np.testing.assert_array_equal(np.asarray(processed.memory_features), features)
    np.testing.assert_array_equal(np.asarray(processed.memory_candidates), candidates)
    np.testing.assert_array_equal(np.asarray(processed.memory_candidate_mask), candidate_mask)
    assert processed.memory_enabled is True


def test_suffix_memory_hook_is_exact_with_disabled_or_zero_bfloat16_residual():
    from types import SimpleNamespace

    from openpi.models.pi0_moe_skill import _apply_memory_condition

    adapter = _adapter("action_tokens", dz=7, width=12)
    actions = (jnp.arange(2 * 3 * 12, dtype=jnp.float32).reshape(2, 3, 12) / 9).astype(jnp.bfloat16)
    cond = (jnp.arange(24, dtype=jnp.float32).reshape(2, 12) / 7).astype(jnp.bfloat16)
    memory = jnp.ones((2, 7), jnp.float32)

    disabled_obs = SimpleNamespace(memory_enabled=False)
    disabled_actions, disabled_cond = _apply_memory_condition(actions, cond, disabled_obs, adapter)
    assert disabled_actions is actions
    assert disabled_cond is cond

    enabled_obs = SimpleNamespace(memory_enabled=True, memory_features=memory)
    zero_actions, zero_cond = _apply_memory_condition(actions, cond, enabled_obs, adapter)
    assert zero_actions.dtype == actions.dtype
    assert zero_cond is cond
    np.testing.assert_array_equal(np.asarray(zero_actions), np.asarray(actions))


def test_suffix_hook_supports_adarms_and_candidate_routes_in_bfloat16():
    from types import SimpleNamespace

    from openpi.models.pi0_moe_skill import _apply_memory_condition

    batch, horizon, width, dz = 2, 3, 12, 7
    actions = jnp.ones((batch, horizon, width), dtype=jnp.bfloat16)
    cond = jnp.full((batch, width), 0.375, dtype=jnp.bfloat16)
    features = jnp.arange(batch * dz, dtype=jnp.float32).reshape(batch, dz)

    adarms_adapter = _adapter("adarms_cond", dz=dz, width=width)
    adarms_obs = SimpleNamespace(memory_enabled=True, memory_features=features)
    same_actions, same_cond = _apply_memory_condition(actions, cond, adarms_obs, adarms_adapter)
    assert same_actions is actions
    assert same_cond.dtype == jnp.bfloat16
    np.testing.assert_array_equal(np.asarray(same_cond), np.asarray(cond))

    candidate_adapter = _adapter("candidate_attention", dz=dz, width=width)
    candidates = jnp.arange(batch * 4 * dz, dtype=jnp.float32).reshape(batch, 4, dz)
    mask = jnp.asarray([[True, False, False, False], [False, False, False, False]])
    candidate_obs = SimpleNamespace(memory_enabled=True, memory_candidates=candidates, memory_candidate_mask=mask)
    candidate_actions, candidate_cond = _apply_memory_condition(actions, cond, candidate_obs, candidate_adapter)
    assert candidate_actions.dtype == jnp.bfloat16
    assert candidate_cond is cond
    np.testing.assert_array_equal(np.asarray(candidate_actions), np.asarray(actions))


def test_verification_compares_raw_parameter_bytes():
    from skill_moe.skillnet.examples.memory_pilot.injection.verify_checkpoint import _bitwise_unchanged

    before = {("base", "weight"): np.asarray([0.0, 1.0], dtype=np.float32)}
    same = {("base", "weight"): np.asarray([0.0, 1.0], dtype=np.float32)}
    signed_zero_changed = {("base", "weight"): np.asarray([-0.0, 1.0], dtype=np.float32)}
    assert _bitwise_unchanged(before, same) == (True, 0)
    assert _bitwise_unchanged(before, signed_zero_changed) == (False, 1)
