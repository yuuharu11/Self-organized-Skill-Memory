"""Small conditional Gaussian bottleneck model used by the light experiment."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class BottleneckModel(nn.Module):
    def __init__(self, action_shape: tuple[int, int], state_dim: int, feature_dim: int, z_dim: int = 32,
                 width: int = 256):
        super().__init__()
        self.action_shape = action_shape
        self.z_dim = z_dim
        self.encoder = nn.Sequential(nn.Linear(action_shape[0] * action_shape[1] + state_dim, width), nn.GELU(),
                                     nn.Linear(width, width), nn.GELU(), nn.Linear(width, 2 * z_dim))
        self.prior = nn.Sequential(nn.Linear(feature_dim + state_dim, width), nn.GELU(),
                                   nn.Linear(width, width), nn.GELU(), nn.Linear(width, 2 * z_dim))
        self.decoder = nn.Sequential(nn.Linear(z_dim + feature_dim + state_dim, width), nn.GELU(),
                                     nn.Linear(width, width), nn.GELU(),
                                     nn.Linear(width, action_shape[0] * action_shape[1]))

    @staticmethod
    def _dist(out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, logvar = out.chunk(2, dim=-1)
        return mean, logvar.clamp(-8.0, 5.0)

    def encode(self, actions: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._dist(self.encoder(torch.cat([actions.flatten(1), state], dim=-1)))

    def predict_prior(self, feature: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._dist(self.prior(torch.cat([feature, state], dim=-1)))

    def decode(self, z: torch.Tensor, feature: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat([z, feature, state], dim=-1)).reshape(-1, *self.action_shape)

    def forward(self, actions: torch.Tensor, state: torch.Tensor, feature: torch.Tensor,
                *, sample: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        qm, qlv = self.encode(actions, state)
        pm, plv = self.predict_prior(feature, state)
        z = qm + torch.randn_like(qm) * torch.exp(0.5 * qlv) if sample else qm
        return self.decode(z, feature, state), qm, qlv, pm, plv


def gaussian_kl(qm: torch.Tensor, qlv: torch.Tensor, pm: torch.Tensor, plv: torch.Tensor) -> torch.Tensor:
    """Analytic KL(q||p), averaged over batch and summed over latent dimensions."""
    return 0.5 * (plv - qlv + (torch.exp(qlv) + (qm - pm).square()) / torch.exp(plv) - 1).sum(-1).mean()


def standardize(train: np.ndarray, *values: np.ndarray) -> tuple[np.ndarray, ...]:
    mean = np.asarray(train, dtype=np.float32).mean(axis=0)
    scale = np.asarray(train, dtype=np.float32).std(axis=0)
    scale[scale < 1e-6] = 1.0
    return tuple((np.asarray(v, dtype=np.float32) - mean) / scale for v in values)
