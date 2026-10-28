"""Config for the emergent-skill model: Pi0 (MoE) conditioned on a per-chunk continuous skill latent z.

See examples/emergent/DESIGN.md. With ``z_dim == 0`` the model is exactly the existing
``pi0_moe_skill.Pi0`` (same parameters, same computation).
"""

import dataclasses
from typing import TYPE_CHECKING, Literal

import flax.nnx as nnx
from typing_extensions import override

from openpi.models import pi0_config_moe_skill
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0_skill_latent import Pi0SkillLatent

ZInject = Literal["router_token", "router", "token"]


@dataclasses.dataclass(frozen=True)
class Pi0SkillLatentConfig(pi0_config_moe_skill.Pi0Config):
    # Dimension of z. 0 disables the latent path entirely (plain SkillNet Pi0).
    z_dim: int = 32
    # Where z enters the action expert:
    #   "router_token": MoE router (instead of / in addition to skill IDs) + one extra suffix token (main variant)
    #   "router":       MoE router only (same path as SkillNet skill IDs; comparison)
    #   "token":        extra suffix token only (ablation)
    z_inject: ZInject = "router_token"
    # True: keep the skill-ID embeddings and append z tokens to the router's cross-attention keys
    # (existing checkpoint + z, comparison). False: no skill-ID input at all (main, label-free).
    use_skill_ids: bool = False
    # Number of key/value tokens z is projected to for the router cross-attention.
    z_router_tokens: int = 4

    # Weight of the MoE load-balancing loss (base SkillNet uses 1.0; 0.0 = reconstruction + KL only).
    balance_loss_weight: float = 1.0
    # Information bottleneck: loss = flow_matching + balance_loss_weight * balance + kl_beta * KL(q || p).
    # KL is summed over z dims and averaged over the batch; the flow-matching term is a per-element mean.
    kl_beta: float = 0.01
    # Per-dimension free bits (nats) on the batch-mean KL. 0 disables.
    kl_free_bits: float = 0.0
    # None: plain KL. alpha in [0, 1]: alpha * KL(sg(q) || p) + (1 - alpha) * KL(q || sg(p)).
    kl_balance: float | None = None
    logvar_min: float = -10.0
    logvar_max: float = 4.0

    # Posterior encoder q(z | clean action chunk, proprio): flatten-MLP hidden size.
    enc_hidden: int = 512
    # Prior p(z | observation): pooled VLM (prefix) output tokens [+ proprio] -> MLP.
    prior_hidden: int = 512
    # Pooling of the *normal-input* prefix outputs (image + task language attend bidirectionally, so even "image"
    # carries language information; the language-free memory key is a separate pass, see memory_key_features).
    # "image": masked mean over image-token outputs only; "all": over all valid prefix outputs (image + language).
    prior_pool: Literal["image", "all"] = "image"
    prior_use_state: bool = True
    # Stop the KL gradient from reshaping the VLM features that the prior (and the memory key) reads.
    prior_stop_gradient: bool = True
    # At inference without an external z: use the prior mean, or sample from the prior.
    z_inference: Literal["prior_mean", "prior_sample"] = "prior_mean"
    # Hidden size of the z -> token MLP (None: action-expert width).
    z_token_hidden: int | None = None

    # Freeze the VLM (SigLIP + PaliGemma LLM weights). Action expert, skill embedders, and z modules stay trainable.
    freeze_paligemma: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.z_dim < 0:
            raise ValueError("z_dim must be >= 0")
        if self.z_dim == 0 and not self.use_skill_ids:
            raise ValueError("z_dim=0 disables z; that is only meaningful with use_skill_ids=True (plain SkillNet)")
        if self.z_inject not in ("router_token", "router", "token"):
            raise ValueError(f"Unknown z_inject: {self.z_inject!r}")
        if self.kl_balance is not None and not 0.0 <= self.kl_balance <= 1.0:
            raise ValueError("kl_balance must be in [0, 1]")

    @property
    def z_enabled(self) -> bool:
        return self.z_dim > 0

    @property
    def z_in_router(self) -> bool:
        return self.z_enabled and self.z_inject in ("router_token", "router")

    @property
    def z_in_token(self) -> bool:
        return self.z_enabled and self.z_inject in ("router_token", "token")

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0SkillLatent":
        from openpi.models.pi0_skill_latent import Pi0SkillLatent

        return Pi0SkillLatent(self, rngs=nnx.Rngs(rng))

    @override
    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        if not self.freeze_paligemma:
            return super().get_freeze_filter()
        if "lora" in self.paligemma_variant or "lora" in self.action_expert_variant:
            raise NotImplementedError("freeze_paligemma with LoRA variants is not supported")
        return nnx.All(
            nnx_utils.PathRegex(".*PaliGemma.*"),
            # Action-expert weights carry an "_1" suffix (see gemma_moe_skill._name).
            nnx.Not(nnx_utils.PathRegex(".*llm.*_1.*")),
        )
