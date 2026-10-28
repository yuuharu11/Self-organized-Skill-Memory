"""Per-chunk continuous skill latent z: posterior encoder, observation prior, and injection projections.

All modules run in float32 and are independent of the diffusion noise / flow-matching time.
See examples/emergent/DESIGN.md.
"""

import flax.nnx as nnx
import jax
import jax.numpy as jnp


def _split_gaussian(h, z_dim, logvar_min, logvar_max):
    mean, logvar = h[..., :z_dim], h[..., z_dim:]
    return mean, jnp.clip(logvar, logvar_min, logvar_max)


class PosteriorEncoder(nnx.Module):
    """q(z | clean action chunk, proprio). The chunk is flattened so the temporal order is explicit."""

    def __init__(self, *, action_horizon, action_dim, state_dim, z_dim, hidden, logvar_min, logvar_max, rngs):
        self.z_dim = z_dim
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max
        self.fc1 = nnx.Linear(action_horizon * action_dim + state_dim, hidden, rngs=rngs)
        self.fc2 = nnx.Linear(hidden, hidden, rngs=rngs)
        self.out = nnx.Linear(hidden, 2 * z_dim, rngs=rngs)

    def __call__(self, actions, state):
        x = jnp.concatenate(
            [actions.reshape(actions.shape[0], -1).astype(jnp.float32), state.astype(jnp.float32)], axis=-1
        )
        h = nnx.swish(self.fc1(x))
        h = nnx.swish(self.fc2(h))
        return _split_gaussian(self.out(h), self.z_dim, self.logvar_min, self.logvar_max)


class ObservationPrior(nnx.Module):
    """p(z | observation) from pooled VLM (prefix) output tokens [+ proprio]."""

    def __init__(self, *, vlm_width, state_dim, use_state, z_dim, hidden, logvar_min, logvar_max, rngs):
        self.z_dim = z_dim
        self.use_state = use_state
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max
        self.norm = nnx.LayerNorm(vlm_width, rngs=rngs)
        self.fc1 = nnx.Linear(vlm_width + (state_dim if use_state else 0), hidden, rngs=rngs)
        self.fc2 = nnx.Linear(hidden, hidden, rngs=rngs)
        self.out = nnx.Linear(hidden, 2 * z_dim, rngs=rngs)

    def __call__(self, pooled_vlm, state):
        x = self.norm(pooled_vlm.astype(jnp.float32))
        if self.use_state:
            x = jnp.concatenate([x, state.astype(jnp.float32)], axis=-1)
        h = nnx.swish(self.fc1(x))
        h = nnx.swish(self.fc2(h))
        return _split_gaussian(self.out(h), self.z_dim, self.logvar_min, self.logvar_max)


class ZRouterTokens(nnx.Module):
    """z -> [B, M, d_key] tokens used as keys/values of the MoE router cross-attention (in place of skill IDs)."""

    def __init__(self, *, z_dim, num_tokens, key_dim, rngs):
        self.num_tokens = num_tokens
        self.key_dim = key_dim
        self.proj = nnx.Linear(z_dim, num_tokens * key_dim, rngs=rngs)

    def __call__(self, z):
        return self.proj(z.astype(jnp.float32)).reshape(z.shape[0], self.num_tokens, self.key_dim)


class ZToken(nnx.Module):
    """z -> one action-expert token [B, 1, width]."""

    def __init__(self, *, z_dim, hidden, width, rngs):
        self.fc1 = nnx.Linear(z_dim, hidden, rngs=rngs)
        self.fc2 = nnx.Linear(hidden, width, rngs=rngs)

    def __call__(self, z):
        return self.fc2(nnx.swish(self.fc1(z.astype(jnp.float32))))[:, None, :]


class SkillLatent(nnx.Module):
    """Container so all z parameters share the `skill_latent/` path prefix (used by weight-loader regexes)."""

    def __init__(self, *, encoder, prior, router_tokens, token):
        self.encoder = encoder
        self.prior = prior
        self.router_tokens = router_tokens
        self.token = token


def gaussian_kl(q_mean, q_logvar, p_mean, p_logvar):
    """Per-dimension KL(N(q) || N(p)) for diagonal Gaussians, float32, shape [..., z]."""
    q_mean, q_logvar, p_mean, p_logvar = (x.astype(jnp.float32) for x in (q_mean, q_logvar, p_mean, p_logvar))
    return 0.5 * (p_logvar - q_logvar + (jnp.exp(q_logvar) + jnp.square(q_mean - p_mean)) * jnp.exp(-p_logvar) - 1.0)


def kl_objective(q_mean, q_logvar, p_mean, p_logvar, *, free_bits=0.0, balance=None):
    """Returns (kl_loss, kl_raw): kl_raw is the batch mean of the summed KL; kl_loss adds free bits / balancing."""
    sg = jax.lax.stop_gradient
    kl_raw_dims = gaussian_kl(q_mean, q_logvar, p_mean, p_logvar)
    if balance is None:
        kl_dims = kl_raw_dims
    else:
        kl_dims = balance * gaussian_kl(sg(q_mean), sg(q_logvar), p_mean, p_logvar) + (1.0 - balance) * gaussian_kl(
            q_mean, q_logvar, sg(p_mean), sg(p_logvar)
        )
    per_dim = jnp.mean(kl_dims, axis=0)
    if free_bits > 0:
        per_dim = jnp.maximum(per_dim, free_bits)
    return jnp.sum(per_dim), jnp.mean(jnp.sum(kl_raw_dims, axis=-1))


def reparameterize(rng, mean, logvar):
    return mean + jnp.exp(0.5 * logvar) * jax.random.normal(rng, mean.shape, dtype=mean.dtype)


def masked_mean(x, mask):
    """x: [B, S, D], mask: [B, S] -> [B, D] (float32). All-masked rows return zeros."""
    w = mask.astype(jnp.float32)[..., None]
    return jnp.sum(x.astype(jnp.float32) * w, axis=1) / jnp.maximum(jnp.sum(w, axis=1), 1.0)
