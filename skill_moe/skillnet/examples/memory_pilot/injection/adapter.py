"""Small, zero-start memory-to-action conditioning modules for the memory pilot.

These modules intentionally produce residuals only. The caller decides where a
residual is added and can skip the adapter entirely when memory is disabled.
"""
from __future__ import annotations

from typing import Literal

from flax import nnx
import jax
import jax.numpy as jnp

InjectionMode = Literal["action_tokens", "adarms_cond", "candidate_attention"]


class MemoryInjection(nnx.Module):
    """Project pooled memory or attend to memory candidates.

    Args:
        mode: ``action_tokens`` or ``adarms_cond`` maps a pooled ``c: [B, Dz]``
            through a two-layer MLP and returns ``[B, width]``. The integration
            point broadcasts action-token residuals over the horizon. For
            ``candidate_attention``, pass action tokens ``[B, H, width]``,
            candidate vectors ``[B, K, Dz]``, and a boolean mask ``[B, K]``; the
            result is ``[B, H, width]``.
        dz: Dimension of the memory representation.
        width: Action-expert width (1024 for the published SkillNet config).
        hidden_dim: Hidden width of the pooled-vector MLP.
        attention_dim: Key/query/value width for candidate attention.
        rngs: NNX random streams.

    The pooled-vector MLP's output projection is initialized to exactly zero.
    Candidate attention uses a zero-initialized scalar residual gate. Thus all
    modes return exactly zero at initialization while retaining a gradient on
    their final zero-initialized projection/gate.
    """

    def __init__(
        self,
        *,
        mode: InjectionMode,
        dz: int,
        width: int = 1024,
        hidden_dim: int = 256,
        attention_dim: int = 128,
        rngs: nnx.Rngs,
    ):
        if mode not in ("action_tokens", "adarms_cond", "candidate_attention"):
            raise ValueError(f"Unknown memory injection mode: {mode}")
        if min(dz, width, hidden_dim, attention_dim) <= 0:
            raise ValueError("All adapter dimensions must be positive")
        self.mode = mode
        self.dz = dz
        self.width = width
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim

        if mode in ("action_tokens", "adarms_cond"):
            self.condition_in = nnx.Linear(dz, hidden_dim, rngs=rngs)
            self.condition_out = nnx.Linear(
                hidden_dim,
                width,
                rngs=rngs,
                kernel_init=nnx.initializers.zeros,
                bias_init=nnx.initializers.zeros,
            )
        else:
            self.query = nnx.Linear(width, attention_dim, rngs=rngs)
            self.key = nnx.Linear(dz, attention_dim, rngs=rngs)
            self.value = nnx.Linear(dz, attention_dim, rngs=rngs)
            self.context_out = nnx.Linear(attention_dim, width, rngs=rngs)
            self.gate = nnx.Param(jnp.zeros((), dtype=jnp.float32))

    def __call__(
        self,
        *,
        c: jax.Array | None = None,
        action_tokens: jax.Array | None = None,
        candidates: jax.Array | None = None,
        candidate_mask: jax.Array | None = None,
        horizon: int | None = None,
    ) -> jax.Array:
        """Return a residual, without modifying the base model activations."""
        if self.mode in ("action_tokens", "adarms_cond"):
            if c is None:
                raise ValueError(f"mode={self.mode!r} requires pooled memory c [B, Dz]")
            if c.ndim != 2 or c.shape[-1] != self.dz:
                raise ValueError(f"c must have shape [B, {self.dz}], got {c.shape}")
            residual = self.condition_out(nnx.gelu(self.condition_in(c)))
            if self.mode == "action_tokens":
                if horizon is None or horizon <= 0:
                    raise ValueError("action_tokens mode requires a positive horizon")
                return jnp.broadcast_to(residual[:, None, :], (c.shape[0], horizon, self.width))
            return residual

        if action_tokens is None or candidates is None or candidate_mask is None:
            raise ValueError("candidate_attention requires action_tokens, candidates, and candidate_mask")
        if action_tokens.ndim != 3 or action_tokens.shape[-1] != self.width:
            raise ValueError(f"action_tokens must have shape [B, H, {self.width}], got {action_tokens.shape}")
        if candidates.ndim != 3 or candidates.shape[-1] != self.dz:
            raise ValueError(f"candidates must have shape [B, K, {self.dz}], got {candidates.shape}")
        if candidate_mask.shape != candidates.shape[:2]:
            raise ValueError(f"candidate_mask must have shape {candidates.shape[:2]}, got {candidate_mask.shape}")
        if candidates.shape[0] != action_tokens.shape[0]:
            raise ValueError("action_tokens and candidates must share their batch dimension")

        query = self.query(action_tokens)  # [B, H, A]
        key = self.key(candidates)  # [B, K, A]
        value = self.value(candidates)  # [B, K, A]
        scores = jnp.einsum("bha,bka->bhk", query, key) * (self.attention_dim**-0.5)
        mask = candidate_mask.astype(jnp.bool_)
        scores = jnp.where(mask[:, None, :], scores, jnp.finfo(scores.dtype).min)
        weights = jax.nn.softmax(scores, axis=-1)
        context = jnp.einsum("bhk,bka->bha", weights, value)
        # An all-masked row must mean "no memory", not uniform attention.
        has_candidate = jnp.any(mask, axis=-1)[:, None, None]
        context = jnp.where(has_candidate, context, 0)
        residual = jnp.tanh(self.gate.value) * self.context_out(context)
        # context_out has a trainable bias; keep an empty set neutral even after
        # training has moved that bias away from zero.
        return jnp.where(has_candidate, residual, 0)


def add_memory_residual(base: jax.Array, residual: jax.Array | None, *, enabled: bool) -> jax.Array:
    """Apply an adapter residual, preserving the exact base value when disabled."""
    if not enabled or residual is None:
        return base
    # Preserve the model activation dtype (notably bfloat16) at the injection
    # boundary. Adding a zero fp32 residual directly would promote the base.
    residual = jnp.asarray(residual, dtype=base.dtype)
    return base + residual
