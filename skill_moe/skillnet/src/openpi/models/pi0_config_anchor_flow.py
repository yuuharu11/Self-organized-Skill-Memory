"""Configuration for the opt-in S2 anchor-start π0.5 model."""
from __future__ import annotations

from typing import TYPE_CHECKING

import flax.nnx as nnx
from typing_extensions import override

from openpi.models import pi0_config_moe_skill as _skill_config
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0_anchor_flow import Pi0AnchorFlow


class Pi0AnchorFlowConfig(_skill_config.Pi0Config):
    """Checkpoint-compatible LIBERO-90 MoE π0.5 shape (H=10, D=32)."""

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0AnchorFlow":
        from openpi.models.pi0_anchor_flow import Pi0AnchorFlow

        return Pi0AnchorFlow(self, rngs=nnx.Rngs(rng))


def action_expert_trainable_filter():
    """Train only the action-expert tower, time MLPs and action projections."""
    return nnx.All(
        nnx.Param,
        nnx_utils.PathRegex(r".*(llm.*_1|action_in_proj|time_mlp|action_out_proj).*"),
    )


def action_expert_freeze_filter():
    return nnx.Not(action_expert_trainable_filter())
