"""Runtime helper for the RCX5 future gripper-sign classifier."""
from __future__ import annotations

import numpy as np


def t0_from_error(error, quantiles, low=.1, high=.9):
    """Same empirical CDF map used to sample training-time t0 from bridge error."""
    q = np.asarray(quantiles, np.float64)
    if q.shape != (101,) or np.any(np.diff(q) < 0):
        raise ValueError('expected the saved 101 nondecreasing t0_map_quantiles')
    cdf = np.interp(np.asarray(error, np.float64), q, np.linspace(0., 1., 101))
    return (float(low) + (float(high)-float(low))*cdf).astype(np.float32)


def make_gripper_model(input_dim: int, hidden: int):
    import torch
    from torch import nn
    return nn.Sequential(nn.Linear(input_dim, hidden), nn.GELU(), nn.LayerNorm(hidden),
                         nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 50))


class GripperClassifier:
    def __init__(self, checkpoint):
        import torch
        self.torch = torch
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        self.center = np.asarray(saved['center'], np.float32)
        self.scale = np.asarray(saved['scale'], np.float32)
        self.fixed_tail = np.asarray(saved['fixed_tail'], np.float32)
        self.model = make_gripper_model(int(saved['input_dim']), int(saved['hidden']))
        self.model.load_state_dict(saved['state']); self.model.eval()

    def predict(self, pooled_vlm, history10x7, state16):
        """Return exactly one ±1 gripper command per future control step."""
        x = np.concatenate([np.asarray(pooled_vlm, np.float32).reshape(-1),
                            np.asarray(history10x7, np.float32).reshape(-1),
                            np.asarray(state16, np.float32).reshape(-1)])
        if x.shape != self.center.shape:
            raise ValueError(f'gripper classifier expected {self.center.shape}, got {x.shape}')
        x = ((x - self.center) / self.scale).astype(np.float32)
        with self.torch.no_grad():
            logits = self.model(self.torch.from_numpy(x[None])).numpy()[0]
        return np.where(logits >= 0, 1., -1.).astype(np.float32)
