"""Optimus-R-style configuration on the existing SkillNet π0.5 backbone."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import flax.nnx as nnx
from typing_extensions import override

from openpi.models import pi0_config_moe_skill as _skill_config
from openpi.shared import array_typing as at

if TYPE_CHECKING:
    from openpi.models.pi0_optimus_r import Pi0OptimusR


@dataclass(frozen=True)
class Pi0OptimusRConfig(_skill_config.Pi0Config):
    """LIBERO-90 SkillNet π0.5 shape plus Stage-A Optimus-R modules."""

    memory_token_count: int = 4
    query_dim: int = 256
    skill_dim: int = 256
    skill_token_count: int = 4
    interface_hidden_dim: int = 512
    query_contrastive_temperature: float = 0.07
    fm_loss_weight: float = 1.0
    query_loss_weight: float = 0.1
    skill_reconstruction_weight: float = 1.0
    trainable_vlm_layers: int = 4

    def __post_init__(self):
        super().__post_init__()
        if self.memory_token_count < 1 or self.skill_token_count < 1:
            raise ValueError("memory and skill token counts must be positive")
        if min(self.query_dim, self.skill_dim, self.interface_hidden_dim) < 1:
            raise ValueError("Optimus-R latent and hidden dimensions must be positive")
        if self.query_contrastive_temperature <= 0:
            raise ValueError("query contrastive temperature must be positive")

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0OptimusR":
        from openpi.models.pi0_optimus_r import Pi0OptimusR

        return Pi0OptimusR(self, rngs=nnx.Rngs(rng))
