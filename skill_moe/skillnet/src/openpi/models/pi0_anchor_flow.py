"""Anchor-start flow matching for the LIBERO-90 π0.5 action expert.

This module is separate from the released Pi0/SkillNet implementations.  The
default path delegates to the inherited sampler exactly; callers opt into an
anchor by passing an already decoded, normalized action chunk and ``t0``.
"""
from __future__ import annotations

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models.pi0_moe_skill import Pi0 as _SkillNetPi0
from openpi.models.pi0_moe_skill import make_attn_mask
from openpi.shared import array_typing as at


def integrate_velocity(denoise_fn, x_src: at.Float[at.Array, "b h d"],
                       t0: at.Float[at.Array, ""] | at.Float[at.Array, "b"],
                       num_steps: int) -> at.Float[at.Array, "b h d"]:
    """Euler integrate dx/dt=-v from t0 to 0, with per-example start times."""
    if num_steps < 1:
        raise ValueError("num_steps must be positive")
    if x_src.ndim != 3:
        raise ValueError(f"x_src must have [batch,horizon,action_dim], got {x_src.shape}")
    starts = jnp.asarray(t0, dtype=jnp.float32)
    if starts.ndim == 0:
        starts = jnp.broadcast_to(starts, (x_src.shape[0],))
    if starts.shape != (x_src.shape[0],):
        raise ValueError(f"t0 must be scalar or [batch], got {starts.shape}")
    dt = starts / num_steps

    def body(i, x):
        time = starts * (1.0 - i / num_steps)
        velocity = denoise_fn(x, time)
        return x - dt[:, None, None] * velocity

    return jax.lax.fori_loop(0, num_steps, body, x_src)


def mix_anchor_with_noise(anchor_action, noise, t0):
    """Construct (★): x_t0=(1-t0)*a_k+t0*epsilon in normalized action space."""
    anchor_action = jnp.asarray(anchor_action)
    noise = jnp.asarray(noise, dtype=anchor_action.dtype)
    if anchor_action.shape != noise.shape or anchor_action.ndim != 3:
        raise ValueError("anchor_action and noise must share [batch,horizon,action_dim] shape")
    starts = jnp.asarray(t0, dtype=anchor_action.dtype)
    if starts.ndim == 0:
        starts = jnp.broadcast_to(starts, (anchor_action.shape[0],))
    if starts.shape != (anchor_action.shape[0],):
        raise ValueError("t0 must be scalar or [batch]")
    return (1.0 - starts[:, None, None]) * anchor_action + starts[:, None, None] * noise


class Pi0AnchorFlow(_SkillNetPi0):
    """SkillNet LIBERO-90 π0.5 backbone with an optional anchor-start path.

    ``x_src`` is in the same normalized 10x32 action space used by the trained
    action expert. ``t0=1`` delegates to the original sampler, preserving its
    output exactly. For ``t0<1`` the flow integrates from x_src at t0 to zero.
    The encoder, decoder, and M6 memory are external and are not model weights.
    """

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        x_src: at.Float[at.Array, "b ah ad"] | None = None,
        t0: at.Float[at.Array, ""] | at.Float[at.Array, "b"] = 1.0,
    ) -> _model.Actions:
        if num_steps < 1:
            raise ValueError("num_steps must be positive")
        if x_src is None:
            x_src = noise
        if x_src is None:
            # Preserve the upstream random draw and all sampler details.
            return super().sample_actions(rng, observation, num_steps=num_steps, noise=noise)
        expected = (observation.state.shape[0], self.action_horizon, self.action_dim)
        if x_src.shape != expected:
            raise ValueError(f"x_src shape {x_src.shape} != expected {expected}")
        # Exact compatibility at the ordinary noise start, including the
        # checkpoint's normal skill-conditioning route.
        if isinstance(t0, (int, float)):
            if float(t0) == 1.0:
                return super().sample_actions(rng, observation, num_steps=num_steps, noise=x_src)
            return self.sample_actions_from_anchor(observation, x_src=x_src, t0=t0, num_steps=num_steps)
        return jax.lax.cond(
            jnp.all(jnp.asarray(t0) == 1.0),
            lambda _: super(Pi0AnchorFlow, self).sample_actions(
                rng, observation, num_steps=num_steps, noise=x_src
            ),
            lambda _: self.sample_actions_from_anchor(observation, x_src=x_src, t0=t0, num_steps=num_steps),
            operand=None,
        )

    def sample_actions_from_anchor(self, observation, *, x_src, t0, num_steps=10):
        """Anchor integration without changing weights or the VLM prefix."""
        observation = _model.preprocess_observation(None, observation, train=False)
        expected = (observation.state.shape[0], self.action_horizon, self.action_dim)
        if x_src.shape != expected:
            raise ValueError(f"x_src shape {x_src.shape} != expected {expected}")
        starts = jnp.asarray(t0, dtype=jnp.float32)
        if starts.ndim == 0:
            starts = jnp.broadcast_to(starts, (expected[0],))
        if starts.shape != (expected[0],):
            raise ValueError(f"t0 must be scalar or [batch], got {starts.shape}")

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=-1) - 1
        # Preserve the pretrained SkillNet conditioning when supplied; the
        # anchor itself is selected only from the visual/language/state key.
        skill_embeds = None
        if observation.skills is not None:
            skill_embeds = self.PaliGemma.llm(observation.skills, method="skill_embed")
            if self.positional_encoding:
                length = observation.skills.shape[1]
                pos = jnp.broadcast_to(jnp.arange(length)[None, :], (expected[0], length))
                skill_embeds = jnp.concatenate(
                    [skill_embeds, self.PaliGemma.llm(pos, method="position_embed")], axis=-1
                )
        _, kv_cache, _ = self.PaliGemma.llm(
            [prefix_tokens, None], skill_embeds=skill_embeds,
            skill_mask=observation.skill_mask, mask=prefix_attn, positions=positions,
        )

        def denoise(x, time):
            suffix, suffix_mask, suffix_ar, adarms_cond = self.embed_suffix(
                observation, x, time
            )
            suffix_attn = make_attn_mask(suffix_mask, suffix_ar)
            prefix_to_suffix = einops.repeat(prefix_mask, "b p -> b s p", s=suffix.shape[1])
            mask = jnp.concatenate([prefix_to_suffix, suffix_attn], axis=-1)
            pos = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            (_, suffix_out), _, _ = self.PaliGemma.llm(
                [None, suffix], skill_embeds=skill_embeds,
                skill_mask=observation.skill_mask, mask=mask, positions=pos,
                kv_cache=kv_cache, adarms_cond=[None, adarms_cond],
            )
            return self.action_out_proj(suffix_out[:, -self.action_horizon :])

        return integrate_velocity(denoise, x_src, starts, num_steps)

    def encode_prefix_key(self, observation: _model.Observation) -> jax.Array:
        """Mean pooled final VLM prefix state (images + prompt + tokenized state)."""
        observation = _model.preprocess_observation(None, observation, train=False)
        tokens, valid, ar = self.embed_prefix(observation)
        mask = make_attn_mask(valid, ar)
        positions = jnp.cumsum(valid, axis=-1) - 1
        (hidden, _), _, _ = self.PaliGemma.llm(
            [tokens, None], skill_embeds=None, skill_mask=None,
            mask=mask, positions=positions,
        )
        weights = valid.astype(hidden.dtype)[..., None]
        pooled = jnp.sum(hidden * weights, axis=1) / jnp.maximum(jnp.sum(weights, axis=1), 1.0)
        return pooled / jnp.maximum(jnp.linalg.norm(pooled, axis=-1, keepdims=True), 1e-8)

    def compute_anchor_loss(self, rng, observation, actions, anchor_actions, *, train=False):
        """Train ordinary flow and anchor-corrupted inputs against target action A.

        For ordinary rows anchor_actions == actions, recovering OpenPI's
        x_t=(1-t)A+t*epsilon and target epsilon-A. For retrieved/target anchors,
        only the input path changes to x_t=(1-t)a+t*epsilon; the target stays
        epsilon-A so the action expert learns to correct an imperfect anchor.
        """
        if anchor_actions.shape != actions.shape:
            raise ValueError(f"anchor_actions shape {anchor_actions.shape} != actions shape {actions.shape}")
        prep_rng, time_rng = jax.random.split(rng)
        observation = _model.preprocess_observation(prep_rng if train else None, observation, train=train)
        batch = actions.shape[:-2]
        time = jax.random.beta(time_rng, 1.5, 1, batch) * 0.999 + 0.001
        noise_rng = jax.random.fold_in(time_rng, 1)
        noise = jax.random.normal(noise_rng, actions.shape, dtype=actions.dtype)
        te = time[..., None, None]
        x_t = (1.0 - te) * anchor_actions + te * noise
        target = noise - actions
        prefix, prefix_mask, prefix_ar = self.embed_prefix(observation)
        suffix, suffix_mask, suffix_ar, adarms_cond = self.embed_suffix(observation, x_t, time)
        in_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        suffix_ar = jnp.broadcast_to(suffix_ar, (prefix.shape[0], suffix.shape[1]))
        ar_mask = jnp.concatenate([prefix_ar, suffix_ar], axis=1)
        attn = make_attn_mask(in_mask, ar_mask)
        pos = jnp.cumsum(in_mask, axis=1) - 1
        skills = None
        if observation.skills is not None:
            skills = self.PaliGemma.llm(observation.skills, method="skill_embed")
        (_, suffix_out), _, balance = self.PaliGemma.llm(
            [prefix, suffix], skill_embeds=skills, skill_mask=observation.skill_mask,
            mask=attn, positions=pos, adarms_cond=[None, adarms_cond],
        )
        pred = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        return jnp.mean(jnp.square(pred - target), axis=-1), balance

    @override
    def compute_loss(self, rng, observation, actions, *, train=False):
        """Use packed anchor actions when present; otherwise retain inherited loss."""
        if observation.memory_features is None:
            return super().compute_loss(rng, observation, actions, train=train)
        flat_dim = self.action_horizon * self.action_dim
        if observation.memory_features.shape[-1] != flat_dim:
            raise ValueError(
                f"memory_features must contain flattened anchor actions of width {flat_dim}, "
                f"got {observation.memory_features.shape}"
            )
        anchor_actions = jnp.reshape(observation.memory_features, actions.shape)
        return self.compute_anchor_loss(rng, observation, actions, anchor_actions, train=train)


