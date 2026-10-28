"""Small amortized posterior and segment-routed expert decoders."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class FiLMExpert(nn.Module):
    def __init__(self, state_dim: int, latent_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.base = nn.Linear(state_dim + 1, hidden_dim)
        self.film = nn.Linear(latent_dim, hidden_dim * 2)
        self.hidden = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor, phase: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # state: (B,H,Ds), phase: (B,H,1), z: (B,Dz)
        base = F.gelu(self.base(torch.cat((state, phase), dim=-1)))
        scale, bias = self.film(z).chunk(2, dim=-1)
        hidden = F.gelu(base * (1.0 + scale[:, None, :]) + bias[:, None, :])
        hidden = F.gelu(self.hidden(hidden))
        return self.output(hidden)


class PrimitiveMoE(nn.Module):
    """VAE-style q(z|segment) plus one gate per segment and FiLM action experts."""

    def __init__(self, *, horizon: int, state_dim: int, action_dim: int, motion_dim: int,
                 n_experts: int, latent_dim: int = 8, hidden_dim: int = 64,
                 shared_expert: bool = True, routing: str = "top1"):
        super().__init__()
        if horizon <= 0 or min(state_dim, action_dim, motion_dim, n_experts, latent_dim, hidden_dim) <= 0:
            raise ValueError("model dimensions and n_experts must be positive")
        if routing not in {"top1", "soft"}:
            raise ValueError("routing must be top1 or soft")
        self.horizon = horizon
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.motion_dim = motion_dim
        self.n_experts = n_experts
        self.shared_expert_enabled = shared_expert
        self.routing = routing
        encoder_input = horizon * (state_dim + action_dim + motion_dim)
        self.encoder = nn.Sequential(nn.Linear(encoder_input, hidden_dim), nn.GELU(),
                                     nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.posterior_mean = nn.Linear(hidden_dim, latent_dim)
        self.posterior_logvar = nn.Linear(hidden_dim, latent_dim)
        self.router = nn.Linear(latent_dim, n_experts)
        self.experts = nn.ModuleList([
            FiLMExpert(state_dim, latent_dim, action_dim, hidden_dim) for _ in range(n_experts)
        ])
        self.shared_expert = FiLMExpert(state_dim, latent_dim, action_dim, hidden_dim) if shared_expert else None

    def posterior(self, states: torch.Tensor, actions: torch.Tensor, motion: torch.Tensor):
        batch = len(states)
        context = torch.cat((states.flatten(1), actions.flatten(1), motion.flatten(1)), dim=1)
        encoded = self.encoder(context)
        mean = self.posterior_mean(encoded)
        logvar = self.posterior_logvar(encoded).clamp(-8.0, 5.0)
        return mean, logvar

    def route_weights(self, probabilities: torch.Tensor, *, hard: bool | None = None,
                      ablate_expert: int | None = None) -> torch.Tensor:
        if hard is None:
            hard = self.routing == "top1"
        probs = probabilities
        if ablate_expert is not None:
            if not 0 <= ablate_expert < self.n_experts:
                raise ValueError("ablate_expert is out of range")
            keep = torch.ones(self.n_experts, dtype=probs.dtype, device=probs.device)
            keep[ablate_expert] = 0
            probs = probs * keep[None, :]
            total = probs.sum(dim=-1, keepdim=True)
            probs = torch.where(total > 0, probs / total.clamp_min(1e-12), torch.zeros_like(probs))
            # If every routed expert is removed (N=1), the shared path or zero output remains.
            hard = self.routing == "top1"
        if not hard:
            return probs
        indices = probs.argmax(dim=-1)
        onehot = F.one_hot(indices, num_classes=self.n_experts).to(probs.dtype)
        onehot = torch.where((probs.sum(dim=-1, keepdim=True) > 0), onehot, torch.zeros_like(onehot))
        if self.training:
            onehot = onehot - probs.detach() + probs
        return onehot

    def decode(self, states: torch.Tensor, z: torch.Tensor, probabilities: torch.Tensor,
               *, ablate_expert: int | None = None, hard: bool | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        batch, horizon, _ = states.shape
        weights = self.route_weights(probabilities, hard=hard, ablate_expert=ablate_expert)
        phase = torch.linspace(0, 1, horizon, device=states.device, dtype=states.dtype)
        phase = phase.view(1, horizon, 1).expand(batch, -1, -1)
        outputs = torch.stack([expert(states, phase, z) for expert in self.experts], dim=1)
        prediction = (outputs * weights[:, :, None, None]).sum(dim=1)
        if self.shared_expert is not None:
            prediction = prediction + self.shared_expert(states, phase, z)
        return prediction, weights

    def forward(self, states: torch.Tensor, actions: torch.Tensor, motion: torch.Tensor,
                *, sample_latent: bool = True, ablate_expert: int | None = None,
                hard: bool | None = None, z_override: torch.Tensor | None = None):
        mean, logvar = self.posterior(states, actions, motion)
        if z_override is not None:
            z = z_override
        elif sample_latent:
            z = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)
        else:
            z = mean
        probabilities = torch.softmax(self.router(z), dim=-1)
        prediction, weights = self.decode(states[:, :1].expand(-1, self.horizon, -1), z,
                                          probabilities, ablate_expert=ablate_expert, hard=hard)
        return {"prediction": prediction, "mean": mean, "logvar": logvar,
                "probabilities": probabilities, "weights": weights, "z": z}
