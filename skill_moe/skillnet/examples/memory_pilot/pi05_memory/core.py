"""Dense pi0.5 memory bridge: token retrieval and an independent action expert.

No SkillNet weights, task labels, or evaluation rollouts enter this model.
The unmodified pretrained checkpoint remains the fallback policy.
"""
from __future__ import annotations

import numpy as np


def normalize_tokens(x):
    x = np.asarray(x, np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def late_interaction(query, candidates, query_groups, candidate_groups):
    """Reference exact MaxSim, equal mean over available camera/text groups.

    Invalid/padded token group is -1. No observation keys are averaged.
    """
    q, keys = normalize_tokens(query), normalize_tokens(candidates)
    qg, kg = np.asarray(query_groups), np.asarray(candidate_groups)
    scores = []
    for k, groups in zip(keys, kg, strict=True):
        terms = []
        for group in np.unique(qg[qg >= 0]):
            left, right = q[qg == group], k[groups == group]
            if not len(right):
                raise ValueError("candidate is missing a query modality")
            terms.append((left @ right.T).max(axis=1).mean())
        if not terms:
            raise ValueError("empty query")
        scores.append(np.mean(terms))
    return np.asarray(scores, np.float32)


def bridge_batch(actions, drafts, noise, t0, progress):
    """NumPy reference: time decreases from t0 to zero, dx/dt is positive."""
    a, m, eps = map(np.asarray, (actions, drafts, noise))
    starts = np.asarray(t0).reshape(-1, 1, 1)
    u = np.asarray(progress).reshape(-1, 1, 1)
    if np.any((starts <= 0) | (starts > 1)) or np.any((u < 0) | (u > 1)):
        raise ValueError("invalid bridge time")
    source = (1 - starts) * m + starts * eps
    return (1 - u) * a + u * source, (source - a) / starts, (starts * u)[:, 0, 0]


def exclude_episode(scores, memory_episodes, query_episode):
    scores = np.asarray(scores).copy()
    scores[np.asarray(memory_episodes) == query_episode] = -np.inf
    if not np.isfinite(scores).any():
        raise ValueError("no cross-episode memory candidate")
    return scores


def _disable_memory_model_subtask_by_default(model_class):
    """Mark a MemoryPi05 subclass as task-only without changing other callers."""
    model_class.subtask_generation_default = False
    return model_class


def make_model(*, abstract=False, action_horizon=10, discrete_state_input=False, max_token_len=200,
               robocasa=False):
    """Delayed imports keep pure retrieval/bridge tests CPU-only."""
    import jax
    import jax.numpy as jnp
    import flax.nnx as nnx
    from openpi.models import model as model_api
    from openpi.models.pi0 import Pi0, make_attn_mask
    from openpi.models.pi0_config import Pi0Config

    class MemoryPi05(Pi0):
        def prefix_features(self, observation):
            obs = model_api.preprocess_observation(None, observation, train=False)
            tokens, valid, ar = self.embed_prefix(obs)
            (hidden, _), _ = self.PaliGemma.llm(
                [tokens, None], mask=make_attn_mask(valid, ar),
                positions=jnp.cumsum(valid, axis=1) - 1)
            # Two real 16x16 cameras; absent right camera is not a search key.
            selected = jnp.concatenate([jnp.arange(512),
                                       jnp.arange(768, hidden.shape[1])])
            hidden = hidden[:, selected].astype(jnp.float32)
            mask = valid[:, selected]
            groups = jnp.broadcast_to(jnp.concatenate([
                jnp.zeros(256, jnp.int8), jnp.ones(256, jnp.int8),
                jnp.full(observation.tokenized_prompt.shape[1], 2, jnp.int8)]), mask.shape)
            text_valid = observation.tokenized_prompt > 2
            mask = mask.at[:, 512:].set(mask[:, 512:] & text_valid)
            groups = jnp.where(mask, groups, -1)
            hidden = hidden / jnp.maximum(jnp.linalg.norm(hidden, axis=-1, keepdims=True), 1e-8)
            return hidden, groups

        def velocity(self, obs, x, time, *, prefix_context=None):
            if prefix_context is None:
                prefix, mask, ar = self.embed_prefix(obs)
                _, cache = self.PaliGemma.llm(
                    [prefix, None], mask=make_attn_mask(mask, ar),
                    positions=jnp.cumsum(mask, axis=1) - 1)
            else:
                mask, cache = prefix_context
            suffix, smask, sar, cond = self.embed_suffix(obs, x, time)
            attn = jnp.concatenate([
                jnp.broadcast_to(mask[:, None, :], (len(x), suffix.shape[1], mask.shape[1])),
                make_attn_mask(smask, sar)], axis=-1)
            pos = jnp.sum(mask, axis=1)[:, None] + jnp.cumsum(smask, axis=1) - 1
            (_, out), _ = self.PaliGemma.llm(
                [None, suffix], mask=attn, positions=pos, kv_cache=cache,
                adarms_cond=[None, cond])
            return self.action_out_proj(out[:, -self.action_horizon:])

        def encode_prefix_context(self, observation):
            """Encode the image/language prefix once for paired action paths."""
            obs = model_api.preprocess_observation(None, observation, train=False)
            prefix, mask, ar = self.embed_prefix(obs)
            _, cache = self.PaliGemma.llm(
                [prefix, None], mask=make_attn_mask(mask, ar),
                positions=jnp.cumsum(mask, axis=1) - 1)
            return mask, cache

        def sample_actions_with_context(self, rng, observation, prefix_context, *, num_steps=10, noise=None):
            """Ordinary pretrained π0.5 flow using a caller-provided prefix KV cache."""
            obs = model_api.preprocess_observation(None, observation, train=False)
            batch_size = observation.state.shape[0]
            eps = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim)) if noise is None else noise

            def body(i, value):
                time = jnp.full((batch_size,), 1.0 - i / num_steps, jnp.float32)
                return value - self.velocity(obs, value, time, prefix_context=prefix_context) / num_steps

            return jax.lax.fori_loop(0, num_steps, body, eps)

        def sample_bridge(self, rng, observation, drafts, starts, *, num_steps=10, noise=None, prefix_context=None):
            obs = model_api.preprocess_observation(None, observation, train=False)
            if prefix_context is None:
                prefix, mask, ar = self.embed_prefix(obs)
                _, cache = self.PaliGemma.llm(
                    [prefix, None], mask=make_attn_mask(mask, ar),
                    positions=jnp.cumsum(mask, axis=1) - 1)
                prefix_context = (mask, cache)
            eps = jax.random.normal(rng, drafts.shape) if noise is None else noise
            starts = jnp.asarray(starts).reshape(-1)
            x = (1 - starts[:, None, None]) * drafts + starts[:, None, None] * eps
            def body(i, value):
                t = starts * (1 - i / num_steps)
                return value - starts[:, None, None] / num_steps * self.velocity(
                    obs, value, t, prefix_context=prefix_context)
            return jax.lax.fori_loop(0, num_steps, body, x)

    if robocasa:
        # Keep LIBERO's existing policy default untouched. This RoboCasa-only
        # subclass matches the released checkpoint's task-only prompt behavior.
        _disable_memory_model_subtask_by_default(MemoryPi05)

    # Defaults preserve existing LIBERO memory-bridge behavior. RoboCasa's
    # released checkpoint uses a 50-step horizon and discrete state tokens.
    config = Pi0Config(pi05=True, action_horizon=action_horizon, action_dim=32,
                      max_token_len=max_token_len, discrete_state_input=discrete_state_input)
    create = lambda: MemoryPi05(config, rngs=nnx.Rngs(17))
    return (nnx.eval_shape(create) if abstract else create()), config
