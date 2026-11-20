"""RES1 residual refinement primitives.

The pretrained K1 velocity field is evaluated on ``draft + residual_state``.  Its
six arm channels are trained against the flow velocity of ``target - draft``;
gripper channels remain owned by the KQ2 decoder.  This module is deliberately
independent of the running KQ2 experiment and can be smoke-tested on CPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class SavedContext:
    """Per-unit, modality-aligned context. Tensors are [units,tokens,dim]."""
    h: torch.Tensor
    state: torch.Tensor
    vlm: torch.Tensor | None = None
    vlm_mask: torch.Tensor | None = None

    def validate(self) -> None:
        n = self.h.shape[0]
        if self.h.ndim != 3 or self.h.shape[1] != 1:
            raise ValueError("h must be [N,1,D]")
        if self.state.ndim != 2 or self.state.shape[0] != n:
            raise ValueError("state must be raw [N,D]")
        if (self.vlm is None) != (self.vlm_mask is None):
            raise ValueError("VLM tokens and mask must be both present or both absent")
        if self.vlm is not None and (self.vlm.ndim != 3 or self.vlm.shape[0] != n or
                                     self.vlm_mask.shape != self.vlm.shape[:2]):
            raise ValueError("VLM context dimensions do not align")


def residual_target(actions: torch.Tensor, draft: torch.Tensor) -> torch.Tensor:
    """Return arm-only residual [B,T,6]; never includes gripper or padded channels."""
    if actions.shape != draft.shape or actions.ndim != 3 or actions.shape[-1] < 6:
        raise ValueError("actions and draft must have matching [B,T,D>=6] shape")
    return actions[..., :6].float() - draft[..., :6].float()


def sample_residual_flow(actions: torch.Tensor, draft: torch.Tensor, noise: torch.Tensor,
                         progress: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Flow path from white noise (u=1) to residual target (u=0).

    Returns residual state and its velocity d/du = noise - residual. Inference
    integrates from u=1 down to 0 by subtracting the learned velocity.
    """
    target = residual_target(actions, draft)
    if noise.shape != target.shape:
        raise ValueError("noise must match the arm residual shape")
    u = progress.reshape(-1, 1, 1).to(target).clamp(0, 1)
    x = (1 - u) * target + u * noise.float()
    return x, noise.float() - target


def residual_flow_loss(velocity_fn: Callable, observation, actions: torch.Tensor,
                       draft: torch.Tensor, noise: torch.Tensor, progress: torch.Tensor,
                       *, saved_context=None, current_context=None) -> torch.Tensor:
    """FM loss for residuals; velocity_fn(obs, draft_plus_residual, u, contexts).

    K1 receives the reconstructed action-space point, retaining the original
    action expert as initialization while the supervision is residual-only.
    """
    x, target_v = sample_residual_flow(actions, draft, noise, progress)
    arm_point = draft[..., :6].float() + x
    pred = velocity_fn(observation, arm_point, progress, saved_context, current_context)
    if pred.shape != target_v.shape:
        raise ValueError(f"velocity shape {tuple(pred.shape)} != residual target {tuple(target_v.shape)}")
    return F.mse_loss(pred.float(), target_v)


def residual_regression_loss(prediction: torch.Tensor, actions: torch.Tensor,
                             draft: torch.Tensor) -> torch.Tensor:
    """Direct-residual regression comparison, arm channels only."""
    return F.mse_loss(prediction.float(), residual_target(actions, draft))


@torch.no_grad()
def integrate_residual(velocity_fn: Callable, observation, draft: torch.Tensor,
                       noise: torch.Tensor, *, steps: int = 10,
                       saved_context=None, current_context=None) -> torch.Tensor:
    """Integrate residual flow from white noise and return ``draft + residual``."""
    if steps < 1:
        raise ValueError("steps must be positive")
    if noise.shape != draft[..., :6].shape:
        raise ValueError("noise must be [B,T,6]")
    residual = noise.float().clone()
    for i in range(steps):
        u = torch.full((len(draft),), 1. - i / steps, device=draft.device, dtype=torch.float32)
        point = draft[..., :6].float() + residual
        v = velocity_fn(observation, point, u, saved_context, current_context)
        residual = residual - v.float() / steps
    return draft[..., :6].float() + residual


class DeltaContextCrossAttention(nn.Module):
    """One cross-attention layer over saved/current/delta context tokens.

    Gate starts at exactly zero (Flamingo-style tanh gate), so adding this module
    initially leaves the base Expert token stream unchanged. ``expert_tokens``
    must be the K1 suffix hidden states, not noisy action vectors.
    """
    def __init__(self, dim: int = 256, heads: int = 4):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must divide heads")
        self.q_norm = nn.LayerNorm(dim)
        self.k_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.out_norm = nn.LayerNorm(dim)
        self.gate = nn.Parameter(torch.zeros(()))

    @staticmethod
    def make_memory(saved: torch.Tensor, current: torch.Tensor,
                    valid: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        if saved.shape != current.shape or saved.ndim != 3:
            raise ValueError("saved/current contexts must have identical [B,N,D] shapes")
        memory = torch.cat((saved, current, current - saved), dim=1)
        if valid is None:
            return memory, None
        valid = valid.bool()
        if valid.shape != saved.shape[:2]:
            raise ValueError("context mask must be [B,N]")
        return memory, ~torch.cat((valid, valid, valid), dim=1)

    def forward(self, expert_tokens: torch.Tensor, saved: torch.Tensor,
                current: torch.Tensor, valid: torch.Tensor | None = None,
                *, return_attention: bool = False):
        if expert_tokens.ndim != 3 or expert_tokens.shape[-1] != saved.shape[-1]:
            raise ValueError("Expert suffix tokens and context width must match")
        memory, pad = self.make_memory(saved, current, valid)
        cross, weights = self.attn(self.q_norm(expert_tokens), self.k_norm(memory),
                                   self.k_norm(memory), key_padding_mask=pad,
                                   need_weights=return_attention, average_attn_weights=False)
        result = expert_tokens + torch.tanh(self.gate) * self.out_norm(cross)
        return (result, weights) if return_attention else result


def residual_ratio(residual: np.ndarray, draft: np.ndarray, eps: float = 1e-8) -> dict[str, float]:
    """Per-row arm norm ratio distribution, guarding near-zero drafts."""
    r = np.asarray(residual, np.float64)[..., :6]
    d = np.asarray(draft, np.float64)[..., :6]
    if r.shape != d.shape or r.ndim < 2:
        raise ValueError("residual/draft shape mismatch")
    rn = np.linalg.norm(r.reshape(len(r), -1), axis=1)
    dn = np.linalg.norm(d.reshape(len(d), -1), axis=1)
    q = np.quantile(rn / np.maximum(dn, eps), [.05, .25, .5, .75, .95])
    return {k: float(v) for k, v in zip(("p05", "p25", "median", "p75", "p95"), q)}


def draw_rank_samples(rng: np.random.Generator, ranks: np.ndarray, n: int,
                      *, own_task_fraction: float, uniform_fraction: float,
                      n_candidates: int = 4516) -> tuple[np.ndarray, np.ndarray]:
    """Sample stored rank-law ranks and indicate which are own-task-excluded.

    The empirical rank law is supplied from KQ2/EX3 retrieval diagnostics. The
    own-task-excluded proportion is a Dev-selectable mixture to expose the
    cross-domain draft/error distribution during training.
    """
    if not 0 <= own_task_fraction <= 1 or not 0 <= uniform_fraction <= 1:
        raise ValueError("fractions must be in [0,1]")
    if own_task_fraction + uniform_fraction > 1:
        raise ValueError("mixture fractions sum to >1")
    base = np.asarray(ranks, np.int64)
    if base.ndim != 1 or not len(base) or np.any(base < 1):
        raise ValueError("rank law must be positive, nonempty 1-D")
    mode = rng.random(n)
    excluded = mode < own_task_fraction
    uniform = (mode >= own_task_fraction) & (mode < own_task_fraction + uniform_fraction)
    out = rng.choice(base, size=n, replace=True)
    out[uniform] = rng.integers(1, n_candidates + 1, size=int(uniform.sum()))
    return np.minimum(out, n_candidates), excluded

