"""Model and teacher helpers for KEYAB (no data loading or GPU dependency)."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from key_motion import CrossOnlyBlock, _sinusoidal_positions


class KeyABEncoder(nn.Module):
    """Single learned query reads a padded action segment.

    Inputs are the same normalized arm6 sequence with a validity mask. The
    optional absolute sinusoid and optional input self-attention isolate the
    two architectural factors while retaining the X1 cross-attention readout.
    """
    def __init__(self, *, position: str = 'absolute', self_attention: bool = False,
                 dim: int = 256, heads: int = 8):
        super().__init__()
        if position not in ('absolute', 'none'):
            raise ValueError(position)
        self.position = position
        self.action_projection = nn.Linear(6, dim)
        self.input_self = nn.MultiheadAttention(dim, heads, batch_first=True) if self_attention else None
        self.input_norm = nn.LayerNorm(dim) if self_attention else None
        self.work = nn.Parameter(torch.randn(1, 1, dim) * .02)
        self.cross = CrossOnlyBlock(dim, heads)
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, actions: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        if actions.ndim != 3 or actions.shape[-1] != 6 or valid.shape != actions.shape[:2]:
            raise ValueError('expected action [B,T,6] and mask [B,T]')
        valid = valid.bool()
        if not valid.any(1).all():
            raise ValueError('each action sequence needs a valid timestep')
        x = torch.where(valid[..., None], actions.float(), torch.zeros_like(actions.float()))
        mem = self.action_projection(x)
        if self.position == 'absolute':
            mem = mem + _sinusoidal_positions(x.shape[1], mem.shape[-1], x.device, mem.dtype)[None]
        pad = ~valid
        if self.input_self is not None:
            z = self.input_norm(mem)
            sa, _ = self.input_self(z, z, z, key_padding_mask=pad, need_weights=False)
            mem = mem + sa
        q = self.work.expand(len(x), -1, -1)
        q, _ = self.cross(q, mem, pad)
        return F.normalize(self.out_norm(q[:, 0]), dim=-1)


def resample_sequence(x, length: int = 16):
    """Deterministically linearly resample each valid sequence to fixed length."""
    if x.ndim != 3 or x.shape[-1] != 6:
        raise ValueError('expected [B,T,6]')
    out=[]
    for row in x:
        src=torch.linspace(0, row.shape[0]-1, row.shape[0], device=row.device)
        dst=torch.linspace(0, row.shape[0]-1, length, device=row.device)
        lo=dst.floor().long();hi=dst.ceil().long();frac=(dst-lo).unsqueeze(-1)
        out.append(row[lo]*(1-frac)+row[hi]*frac)
    return torch.stack(out)


def pairwise_resampled_mse(query, candidates, length: int = 16):
    """Compare complete units after both are resampled to the same length."""
    q=resample_sequence(query, length).flatten(1).double()
    c=resample_sequence(candidates, length).flatten(1).double()
    qn=(q*q).sum(1);cn=(c*c).sum(1)
    return (qn[:,None]+cn[None]-2*q@c.T).clamp_min(0)/(length*6)

