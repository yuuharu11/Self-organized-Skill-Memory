"""C pressure: yaw- and displacement-normalized end-effector translation.

Coordinate check: ``convert_libero_wrapper.py`` copies each processed RLDS
``step['action']`` directly. OpenVLA's official LIBERO transform builds that
7-vector as ``world_vector(3), rotation_delta(3), gripper(1)``; the cached
state is xyz, axis-angle xyz, and two gripper qpos (see light/extract.py).
The source action-field mapping is in OpenVLA's
``prismatic/vla/datasets/rlds/oxe/transforms.py`` (``libero_dataset_transform``).
Therefore this transform rotates only the world-frame translation deltas into
the start-yaw frame and scales their net displacement. It leaves angular
commands uninterpreted and carries them, scale, start yaw and gripper values
in the residual. Decoding reverses the rigid planar transform exactly.

The wrapper layers on any method with the study interface. Its inner method
quantizes only the canonical xyz trajectory; a small supervised prior predicts
the removed yaw/scale and the preserved rotation/gripper sequence from (f,s).
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def _is_torch(x):
    return torch.is_tensor(x)


def _yaw_from_state(state):
    """Extract ZYX yaw from LIBERO's axis-angle state representation."""
    rv = state[..., 3:6]
    if _is_torch(rv):
        theta = torch.linalg.vector_norm(rv, dim=-1, keepdim=True)
        half = theta * 0.5
        scale = torch.where(theta > 1e-8, torch.sin(half) / theta.clamp_min(1e-8), torch.full_like(theta, 0.5))
        q = rv * scale
        w = torch.cos(half)[..., 0]
        x, y, z = q.unbind(-1)
        return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))
    rv = np.asarray(rv)
    theta = np.linalg.norm(rv, axis=-1, keepdims=True)
    half = theta * 0.5
    scale = np.divide(np.sin(half), theta, out=np.full_like(theta, 0.5, dtype=np.result_type(rv, np.float32)), where=theta > 1e-8)
    q = rv * scale
    w = np.cos(half)[..., 0]
    x, y, z = np.moveaxis(q, -1, 0)
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def canonicalize(actions, state, *, eps: float = 1e-6):
    """Return normalized local-frame xyz for base and removed geometry for residual.

    ``actions[..., :3]`` is world-vector translation. The scale is the norm of
    the chunk's net displacement; near-zero net motion uses unit scale so loops
    remain finite and round-trip without amplification. Angular deltas and
    gripper command are preserved verbatim in the residual.
    """
    if actions.shape[-1] != 7 or state.shape[-1] < 6:
        raise ValueError("Expected seven-dimensional LIBERO actions and state with xyz+axis-angle")
    xyz = actions[..., :3]
    yaw = _yaw_from_state(state)
    if _is_torch(actions):
        c, si = torch.cos(yaw)[..., None], torch.sin(yaw)[..., None]
        local = torch.stack([c * xyz[..., 0] + si * xyz[..., 1], -si * xyz[..., 0] + c * xyz[..., 1], xyz[..., 2]], dim=-1)
        net = local.sum(dim=-2)
        scale = torch.linalg.vector_norm(net, dim=-1)
        scale = torch.where(scale > eps, scale, torch.ones_like(scale))
        base = local / scale[..., None, None]
        geometry = {"yaw": yaw, "scale": scale, "rotation": actions[..., 3:6], "gripper": actions[..., 6:7]}
    else:
        actions, state = np.asarray(actions), np.asarray(state)
        c, si = np.cos(yaw)[..., None], np.sin(yaw)[..., None]
        local = np.stack([c * xyz[..., 0] + si * xyz[..., 1], -si * xyz[..., 0] + c * xyz[..., 1], xyz[..., 2]], axis=-1)
        net = local.sum(axis=-2)
        scale = np.linalg.norm(net, axis=-1)
        scale = np.where(scale > eps, scale, 1.0)
        base = local / scale[..., None, None]
        geometry = {"yaw": yaw, "scale": scale, "rotation": actions[..., 3:6], "gripper": actions[..., 6:7]}
    return base, geometry


def decanonicalize(base_xyz, geometry):
    """Reconstruct the original seven-dimensional delta-action chunk."""
    yaw, scale = geometry["yaw"], geometry["scale"]
    if _is_torch(base_xyz):
        yaw, scale = torch.as_tensor(yaw, device=base_xyz.device), torch.as_tensor(scale, device=base_xyz.device)
        c, si = torch.cos(yaw)[..., None], torch.sin(yaw)[..., None]
        local = base_xyz * scale[..., None, None]
        xyz = torch.stack([c * local[..., 0] - si * local[..., 1], si * local[..., 0] + c * local[..., 1], local[..., 2]], dim=-1)
        return torch.cat([xyz, torch.as_tensor(geometry["rotation"], device=base_xyz.device),
                          torch.as_tensor(geometry["gripper"], device=base_xyz.device)], dim=-1)
    base_xyz = np.asarray(base_xyz)
    yaw, scale = np.asarray(yaw), np.asarray(scale)
    c, si = np.cos(yaw)[..., None], np.sin(yaw)[..., None]
    local = base_xyz * scale[..., None, None]
    xyz = np.stack([c * local[..., 0] - si * local[..., 1], si * local[..., 0] + c * local[..., 1], local[..., 2]], axis=-1)
    return np.concatenate([xyz, np.asarray(geometry["rotation"]), np.asarray(geometry["gripper"])], axis=-1)


def _get(data: Any, *keys):
    if isinstance(data, Mapping):
        for key in keys:
            if key in data:
                return data[key]
        raise KeyError(f"Expected one of {keys} in training data")
    if isinstance(data, (tuple, list)) and len(data) >= 3:
        return data[{"actions": 0, "state": 1, "feature": 2}[keys[0]]]
    raise TypeError("train must be a mapping or (actions, state, feature) tuple")


class CanonicalizedMethod:
    """Wrap an inner skill method with the C transform and geometry prior."""

    def __init__(self, method, *, epochs: int = 20, batch_size: int = 256,
                 learning_rate: float = 1e-3, hidden_dim: int = 128, device: str = "cpu", seed: int = 0,
                 decoder_obs: bool = False):
        self.method = method
        self.epochs, self.batch_size, self.learning_rate = epochs, batch_size, learning_rate
        self.hidden_dim, self.device, self.seed = hidden_dim, torch.device(device), seed
        self.decoder_obs = bool(decoder_obs)
        self.geometry_prior: nn.Module | None = None
        self.feature_mean = self.feature_scale = self.state_mean = self.state_scale = None
        self.action_horizon = None

    def fit(self, train):
        actions = torch.as_tensor(_get(train, "actions", "A"), dtype=torch.float32, device=self.device)
        state = torch.as_tensor(_get(train, "state", "s"), dtype=torch.float32, device=self.device)
        feature = torch.as_tensor(_get(train, "feature", "features", "f"), dtype=torch.float32, device=self.device)
        base_actions, geometry = canonicalize(actions, state)
        self.action_horizon = actions.shape[1]
        inner_train = dict(train) if isinstance(train, Mapping) else {}
        inner_feature = feature if self.decoder_obs else feature.new_zeros((len(feature), 0))
        inner_train.update({"actions": base_actions, "state": state, "feature": inner_feature})
        self.method.fit(inner_train)

        self.feature_mean, self.feature_scale = self._norm_stats(feature)
        self.state_mean, self.state_scale = self._norm_stats(state)
        target = self._geometry_target(geometry)
        self.geometry_prior = nn.Sequential(nn.Linear(feature.shape[-1] + state.shape[-1], self.hidden_dim), nn.GELU(),
                                            nn.Linear(self.hidden_dim, self.hidden_dim), nn.GELU(),
                                            nn.Linear(self.hidden_dim, target.shape[-1])).to(self.device)
        fn, sn = self._norm_feature(feature), self._norm_state(state)
        ds = torch.utils.data.TensorDataset(torch.cat([fn, sn], dim=-1), target)
        loader = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=True,
                                             generator=torch.Generator().manual_seed(self.seed))
        opt = torch.optim.AdamW(self.geometry_prior.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        for _ in range(self.epochs):
            for x, y in loader:
                loss = F.mse_loss(self.geometry_prior(x), y)
                opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(self.geometry_prior.parameters(), 1.0); opt.step()
        self.geometry_prior.eval()

    @staticmethod
    def _norm_stats(x):
        mean, scale = x.mean(0), x.std(0, unbiased=False)
        return mean, torch.where(scale < 1e-6, torch.ones_like(scale), scale)

    def _norm_feature(self, f):
        return (torch.as_tensor(f, dtype=torch.float32, device=self.device) - self.feature_mean) / self.feature_scale

    def _norm_state(self, s):
        return (torch.as_tensor(s, dtype=torch.float32, device=self.device) - self.state_mean) / self.state_scale

    @staticmethod
    def _geometry_target(geometry):
        scale = torch.as_tensor(geometry["scale"]).reshape(len(geometry["scale"]), 1).clamp_min(1e-6).log()
        rotation = geometry["rotation"].reshape(len(scale), -1)
        gripper = geometry["gripper"].reshape(len(scale), -1)
        return torch.cat([scale, rotation, gripper], dim=-1)

    def encode(self, A, s):
        base_actions, geometry = canonicalize(torch.as_tensor(A, dtype=torch.float32, device=self.device),
                                              torch.as_tensor(s, dtype=torch.float32, device=self.device))
        b, inner_residual = self._encode_inner(base_actions, s)
        return b, {"inner": inner_residual, "geometry": geometry}

    def decode(self, b, r, f, s):
        canonical_xyz = self._decode_inner(b, r["inner"], f, s)
        return decanonicalize(canonical_xyz, r["geometry"])

    def _encode_inner(self, actions, state):
        """Keep the EMA-VQ M2 posterior residual differentiable for S training."""
        method = self.method
        if all(hasattr(method, name) for name in ("encoder", "codebook", "am", "sm", "embedding_dim", "residual_dim")):
            device = method.device
            a = actions.to(device).reshape(len(actions), -1)
            s = torch.as_tensor(state, dtype=torch.float32, device=device)
            an = (a - torch.as_tensor(method.am, device=device)) / torch.as_tensor(method.ass, device=device)
            sn = (s - torch.as_tensor(method.sm, device=device)) / torch.as_tensor(method.ss, device=device)
            enc = method.encoder(torch.cat([an, sn], dim=-1))
            if callable(getattr(method, "_residual_stats", None)):
                z = enc[:, :method.embedding_dim]
                residual, _ = method._residual_stats(enc[:, method.embedding_dim:])
            else:
                z, residual = enc.split([method.embedding_dim, method.residual_dim], dim=-1)
            ids = torch.cdist(z, method.codebook).argmin(dim=-1)
            return ids, residual
        return method.encode(actions, state)

    def _decode_inner(self, b, r, f, s):
        """Use a differentiable M2 decoder; D1 fits it without an observation input."""
        method = self.method
        if all(hasattr(method, name) for name in ("decoder", "codebook", "am", "fm", "sm", "action_shape")):
            device = method.device
            base = b
            if torch.is_tensor(base) and base.is_floating_point() and base.ndim == 2 and base.shape[-1] == method.codebook.shape[-1]:
                base_embedding = base.to(device=device, dtype=torch.float32)
            else:
                if not torch.is_tensor(base) and not np.issubdtype(np.asarray(base).dtype, np.integer):
                    base = method.prior_base(base)
                ids = torch.as_tensor(base, dtype=torch.long, device=device).reshape(-1)
                base_embedding = method.codebook[ids]
            residual = torch.as_tensor(r, dtype=torch.float32, device=device)
            state = torch.as_tensor(s, dtype=torch.float32, device=device)
            if self.decoder_obs:
                feature = torch.as_tensor(f, dtype=torch.float32, device=device)
                fn = (feature - torch.as_tensor(method.fm, device=device)) / torch.as_tensor(method.fs, device=device)
            else:
                fn = torch.zeros((len(state), 0), dtype=torch.float32, device=device)
            sn = (state - torch.as_tensor(method.sm, device=device)) / torch.as_tensor(method.ss, device=device)
            pred = method.decoder(torch.cat([base_embedding, residual, fn, sn], dim=-1))
            pred = pred * torch.as_tensor(method.ass, device=device) + torch.as_tensor(method.am, device=device)
            return pred.reshape(len(pred), *method.action_shape)
        return method.decode(b, r, f, s)

    def prior(self, f, s):
        if self.geometry_prior is None:
            raise RuntimeError("fit(train) must be called before prior(f,s)")
        inner_feature = f if self.decoder_obs else np.zeros((len(f), 0), dtype=np.float32)
        base_distribution, inner_residual = self.method.prior(inner_feature, s)
        f_t, s_t = torch.as_tensor(f, dtype=torch.float32, device=self.device), torch.as_tensor(s, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            pred = self.geometry_prior(torch.cat([self._norm_feature(f_t), self._norm_state(s_t)], dim=-1))
        horizon = self.action_horizon
        scale = pred[:, :1].clamp(-8, 8).exp()[:, 0]
        rotation = pred[:, 1:1 + horizon * 3].reshape(-1, horizon, 3)
        gripper = pred[:, 1 + horizon * 3:].reshape(-1, horizon, 1)
        yaw = _yaw_from_state(s_t)
        residual = {"inner": inner_residual, "geometry": {"yaw": yaw, "scale": scale,
                    "rotation": rotation, "gripper": gripper}}
        return base_distribution, residual

    def memory_add(self, k, b, meta):
        return self.method.memory_add(k, b, meta)

    def memory_query(self, k, **kwargs):
        return self.method.memory_query(k, **kwargs)

    def describe(self):
        return {"method": "C canonicalized wrapper", "inner": self.method.describe(),
                "decoder_obs": self.decoder_obs,
                "base_input": "yaw-rotated, net-displacement-normalized xyz deltas",
                "residual": "start yaw, net displacement norm, rotation_delta sequence, gripper sequence",
                "geometry_prior_updates": "gradient regression during fit on object/goal rows",
                "memory_add_updates": "delegated to inner method; no geometry-prior or inner-parameter update"}

