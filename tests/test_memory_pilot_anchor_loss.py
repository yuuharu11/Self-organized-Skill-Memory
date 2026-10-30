from __future__ import annotations

from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "skill_moe" / "skillnet" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from openpi.models.pi0_anchor_flow import (  # noqa: E402
    Pi0AnchorFlow,
    anchor_flow_velocity_target,
    integrate_velocity,
    mix_anchor_with_noise,
    sample_flow_times,
)
from openpi.models.pi0_moe_skill import Pi0 as SkillNetPi0  # noqa: E402


def test_ideal_anchor_velocity_euler_reaches_action_for_any_anchor_and_steps():
    rng = np.random.default_rng(2026)
    batch, horizon, action_dim = 4, 10, 32
    actions = jnp.asarray(rng.normal(size=(batch, horizon, action_dim)), jnp.float32)
    anchors = jnp.asarray(rng.normal(size=(batch, horizon, action_dim)), jnp.float32)
    noise = jnp.asarray(rng.normal(size=(batch, horizon, action_dim)), jnp.float32)
    t0 = jnp.asarray([0.13, 0.37, 0.71, 1.0], jnp.float32)
    x_start = mix_anchor_with_noise(anchors, noise, t0)

    for steps in (1, 2, 3, 7, 10, 31):
        def ideal_velocity(x_t, time):
            return (x_t - actions) / time[:, None, None]

        result = integrate_velocity(ideal_velocity, x_start, t0, steps)
        np.testing.assert_allclose(np.asarray(result), np.asarray(actions), rtol=2e-5, atol=2e-5)


def test_ordinary_rows_match_previous_velocity_and_loss_exactly():
    rng = np.random.default_rng(3)
    actions = jnp.asarray(rng.normal(size=(3, 10, 32)), jnp.float32)
    noise = jnp.asarray(rng.normal(size=(3, 10, 32)), jnp.float32)
    time = jnp.asarray([0.02, 0.5, 0.999], jnp.float32)
    target = anchor_flow_velocity_target(noise, actions, actions, time)
    old_target = noise - actions
    np.testing.assert_array_equal(np.asarray(target), np.asarray(old_target))

    # Reconstruct the exact old and new MSE reductions used by the training loss.
    predicted = jnp.asarray(rng.normal(size=(3, 10, 32)), jnp.float32)
    old_loss = jnp.mean(jnp.square(predicted - old_target), axis=-1)
    new_loss = jnp.mean(jnp.square(predicted - target), axis=-1)
    np.testing.assert_array_equal(np.asarray(new_loss), np.asarray(old_loss))


def test_anchor_target_is_xt_minus_action_over_time():
    rng = np.random.default_rng(11)
    actions = jnp.asarray(rng.normal(size=(2, 10, 32)), jnp.float32)
    anchors = jnp.asarray(rng.normal(size=(2, 10, 32)), jnp.float32)
    noise = jnp.asarray(rng.normal(size=(2, 10, 32)), jnp.float32)
    time = jnp.asarray([0.23, 0.82], jnp.float32)
    x_t = (1 - time[:, None, None]) * anchors + time[:, None, None] * noise
    target = anchor_flow_velocity_target(noise, actions, anchors, time)
    expected = (x_t - actions) / time[:, None, None]
    np.testing.assert_allclose(np.asarray(target), np.asarray(expected), rtol=2e-6, atol=2e-6)


def test_only_anchor_rows_get_t_minimum_and_ordinary_beta_schedule_is_unchanged():
    rng = jax.random.key(9)
    mask = jnp.asarray([True, False, True, False])
    times = sample_flow_times(rng, (4,), mask, anchor_t_min=0.05)
    beta = jax.random.beta(rng, 1.5, 1, (4,))
    old_ordinary = beta * 0.999 + 0.001
    expected_anchors = 0.05 + beta * 0.95
    values = np.asarray(times)
    np.testing.assert_array_equal(values[~np.asarray(mask)], np.asarray(old_ordinary)[~np.asarray(mask)])
    np.testing.assert_array_equal(values[np.asarray(mask)], np.asarray(expected_anchors)[np.asarray(mask)])
    assert np.all(values[np.asarray(mask)] >= 0.05)
    assert np.all(values[np.asarray(mask)] <= 1.0)


def test_direct_anchor_path_at_t0_one_delegates_to_skillnet_sampler(monkeypatch):
    calls = {}

    def parent_sampler(self, rng, observation, *, num_steps=10, noise=None, **kwargs):
        del self, rng, kwargs
        calls.update(observation=observation, num_steps=num_steps, noise=noise)
        return noise + jnp.asarray(0.125, noise.dtype)

    monkeypatch.setattr(SkillNetPi0, "sample_actions", parent_sampler)
    monkeypatch.setattr(
        Pi0AnchorFlow,
        "_integrate_from_anchor",
        lambda self, observation, *, x_src, t0, num_steps=10: x_src - 1,
    )
    model = object.__new__(Pi0AnchorFlow)
    observation = object()
    noise = jnp.arange(2 * 10 * 32, dtype=jnp.float32).reshape(2, 10, 32)
    for t0 in (1.0, jnp.asarray(1.0), jnp.ones((2,), jnp.float32)):
        result = Pi0AnchorFlow.sample_actions_from_anchor(
            model, observation, x_src=noise, t0=t0, num_steps=6
        )
        np.testing.assert_array_equal(np.asarray(result), np.asarray(noise + 0.125))
        assert calls["observation"] is observation
        assert calls["num_steps"] == 6
        np.testing.assert_array_equal(np.asarray(calls["noise"]), np.asarray(noise))


def test_design_documents_corrected_anchor_flow_target():
    design = (REPO / "skill_moe" / "skillnet" / "examples" / "memory_pilot" /
              "anchor_flow" / "DESIGN.md").read_text(encoding="utf-8")
    assert "target_v=(x_t−A)/t = ε−A + ((1−t)/t)(a−A)" in design
    assert "For ordinary rows, `a=A`, so this reduces exactly to OpenPI's `ε−A` target." in design
    assert "ideal velocity above reaches `A` for any starting anchor and step count" in design
