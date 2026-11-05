"""Optional OpenPI π0.5 adapter. All robotics dependencies import lazily."""
from __future__ import annotations

from pathlib import Path

import numpy as np


DEFAULT_TRAINABLE_PATTERN = r".*(llm.*_1|action_in_proj|time_mlp|action_out_proj).*"


def make_model(*, abstract=False, seed: int = 17):
    """Create the dense 10×32 π0.5 MemoryPi05 model used by the bridge."""
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
            selected = jnp.concatenate([jnp.arange(512), jnp.arange(768, hidden.shape[1])])
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
                _, cache = self.PaliGemma.llm([prefix, None], mask=make_attn_mask(mask, ar),
                    positions=jnp.cumsum(mask, axis=1) - 1)
            else:
                mask, cache = prefix_context
            suffix, smask, sar, cond = self.embed_suffix(obs, x, time)
            attn = jnp.concatenate([
                jnp.broadcast_to(mask[:, None, :], (len(x), suffix.shape[1], mask.shape[1])),
                make_attn_mask(smask, sar)], axis=-1)
            pos = jnp.sum(mask, axis=1)[:, None] + jnp.cumsum(smask, axis=1) - 1
            (_, out), _ = self.PaliGemma.llm([None, suffix], mask=attn, positions=pos,
                kv_cache=cache, adarms_cond=[None, cond])
            return self.action_out_proj(out[:, -self.action_horizon:])

        def sample_bridge(self, rng, observation, drafts, starts, *, num_steps=10, noise=None):
            obs = model_api.preprocess_observation(None, observation, train=False)
            prefix, mask, ar = self.embed_prefix(obs)
            _, cache = self.PaliGemma.llm([prefix, None], mask=make_attn_mask(mask, ar),
                positions=jnp.cumsum(mask, axis=1) - 1)
            eps = jax.random.normal(rng, drafts.shape) if noise is None else noise
            starts = jnp.asarray(starts).reshape(-1)
            x = (1 - starts[:, None, None]) * drafts + starts[:, None, None] * eps

            def body(i, value):
                t = starts * (1 - i / num_steps)
                return value - starts[:, None, None] / num_steps * self.velocity(
                    obs, value, t, prefix_context=(mask, cache))
            return jax.lax.fori_loop(0, num_steps, body, x)

    config = Pi0Config(pi05=True, action_horizon=10, action_dim=32,
                       max_token_len=200, discrete_state_input=False)
    create = lambda: MemoryPi05(config, rngs=nnx.Rngs(seed))
    return (nnx.eval_shape(create) if abstract else create()), config


def load_policy(checkpoint_path, *, model_factory=None):
    """Load complete base weights and fail on any tree/shape mismatch."""
    import jax
    import flax.nnx as nnx
    from openpi.models.model import restore_params
    factory = model_factory or make_model
    model, config = factory(abstract=True)
    state = nnx.state(model)
    loaded = restore_params(Path(checkpoint_path) / "params", restore_type=np.ndarray)
    expected = state.to_pure_dict()
    ea, et = jax.tree.flatten(expected)
    la, lt = jax.tree.flatten(loaded)
    if et != lt or any(x.shape != y.shape for x, y in zip(ea, la, strict=True)):
        raise ValueError("base checkpoint tree/shape mismatch")
    state.replace_by_pure_dict(jax.tree.map(jax.numpy.asarray, loaded))
    nnx.update(model, state)
    return model, config


def trainable_filter(pattern: str = DEFAULT_TRAINABLE_PATTERN):
    """Return the same action-expert parameter filter as the reference runs."""
    import flax.nnx as nnx
    from openpi.shared.nnx_utils import PathRegex
    return nnx.All(nnx.Param, PathRegex(pattern))


def load_expert(model, checkpoint_path, *, pattern: str = DEFAULT_TRAINABLE_PATTERN):
    """Overlay partial action-expert weights on a loaded dense policy."""
    import jax
    import jax.numpy as jnp
    import flax.nnx as nnx
    import orbax.checkpoint as ocp
    filt = trainable_filter(pattern)
    with ocp.PyTreeCheckpointer() as ckpt:
        partial = ckpt.restore(checkpoint_path)
    state = nnx.state(model, filt)
    state.replace_by_pure_dict(jax.tree.map(jnp.asarray, partial))
    nnx.update(model, state)
    return model


def make_bridge_step(model_filter, optimizer):
    """Build the exact bridge training update used for the saved Expert."""
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import optax

    @nnx.jit
    def step(model, opt_state, observation, actions, draft, rng):
        r1, r2, r3 = jax.random.split(rng, 3)
        t0 = jax.random.uniform(r1, (actions.shape[0],), minval=.2, maxval=.6)
        progress = jax.random.uniform(r2, (actions.shape[0],), minval=.01, maxval=1.)
        eps = jax.random.normal(r3, actions.shape)
        start = t0[:, None, None]
        source = (1 - start) * draft + start * eps
        x = (1 - progress[:, None, None]) * actions + progress[:, None, None] * source
        target = (source - actions) / start

        def loss_fn(current):
            velocity = current.velocity(observation, x, t0 * progress)
            return jnp.mean(jnp.square(velocity[:, :, :7] - target[:, :, :7]))

        loss, grads = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, model_filter))(model)
        params = nnx.state(model, model_filter)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        nnx.update(model, optax.apply_updates(params, updates))
        return loss, opt_state
    return step


def train_expert(model, batches, *, steps: int, seed: int, checkpoint_callback=None,
                 save_every: int = 500, learning_rate: float = 2.5e-5,
                 pattern: str = DEFAULT_TRAINABLE_PATTERN):
    """Minimal NNX Expert loop over `(obs, actions, draft)` batches.

    The caller owns data order, validation/checkpoint selection, and storage.
    This keeps data paths and task exclusions outside the portable package.
    """
    import flax.nnx as nnx
    import jax
    import optax
    filt = trainable_filter(pattern)
    params = nnx.state(model, filt)
    optimizer = optax.chain(optax.clip_by_global_norm(1.), optax.adamw(
        learning_rate, b1=.9, b2=.95, weight_decay=1e-5))
    opt_state = optimizer.init(params)
    step = make_bridge_step(filt, optimizer)
    rng = jax.random.key(seed)
    logs = []
    iterator = iter(batches)
    for i in range(steps):
        try:
            obs, actions, draft = next(iterator)
        except StopIteration as exc:
            raise ValueError("batch iterator ended before requested update count") from exc
        rng, sub = jax.random.split(rng)
        loss, opt_state = step(model, opt_state, obs, actions, draft, sub)
        rec = {"step": i + 1, "loss": float(loss)}
        logs.append(rec)
        if checkpoint_callback and ((i + 1) % save_every == 0 or i + 1 == steps):
            checkpoint_callback(model, i + 1, rec)
    return model, logs
