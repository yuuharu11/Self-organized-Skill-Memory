"""Configuration for the opt-in DMP-routed π0.5 model."""
from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Literal

import flax.nnx as nnx
from typing_extensions import override

from openpi.models import pi0_config_moe_skill as _base
from openpi.shared import array_typing as at

if TYPE_CHECKING:
    from openpi.models.pi0_dmp_route import Pi0DmpRoute


InjectionMode = Literal["both", "router_only", "adarms_only", "none"]


@dataclasses.dataclass(frozen=True)
class Pi0DmpRouteConfig(_base.Pi0Config):
    """S3 config. All learned tensor shapes are independent of dictionary size."""

    enable_primitive_router: bool = True
    shape_basis_n: int = 16
    state_history_len: int = 5
    router_hidden_dim: int = 512
    injection_mode: InjectionMode = "both"
    bridge_t0: float = 0.4
    router_ce_weight: float = 1.0
    router_reg_weight: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        if self.shape_basis_n <= 0 or self.state_history_len <= 0:
            raise ValueError("shape_basis_n and state_history_len must be positive")
        if self.router_hidden_dim <= 0:
            raise ValueError("router_hidden_dim must be positive")
        if self.injection_mode not in ("both", "router_only", "adarms_only", "none"):
            raise ValueError(f"unsupported injection_mode: {self.injection_mode}")
        if not 0.0 < self.bridge_t0 <= 1.0:
            raise ValueError("bridge_t0 must be in (0, 1]")
        if self.pi05 and self.action_dim < 7:
            raise ValueError("DMP LIBERO actions require at least seven action dimensions")
        if not self.pi05 or self.action_horizon != 10 or self.action_dim != 32:
            raise ValueError("S3 currently targets the released LIBERO-90 π0.5 shape (H=10,D=32)")

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0DmpRoute":
        from openpi.models.pi0_dmp_route import Pi0DmpRoute

        return Pi0DmpRoute(self, rngs=nnx.Rngs(rng))
