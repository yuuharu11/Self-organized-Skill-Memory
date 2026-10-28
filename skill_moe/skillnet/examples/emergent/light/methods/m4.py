"""M4: DP-means action prototypes with ART-style non-gradient memory growth.

The class intentionally implements the light-study duck-typed method API and
does not inherit from a shared method base class. Inputs/outputs crossing the
API boundary are NumPy arrays; fitting and the small conditional networks use
PyTorch. The default device is CPU to keep comparison runs from claiming a GPU.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ._common import validation_split, update_early_stopping


def _np(value: Any, *, dtype=np.float32) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _field(data: Any, names: tuple[str, ...], *, required: bool = True):
    if isinstance(data, Mapping):
        for name in names:
            if name in data:
                return data[name]
    elif isinstance(data, (list, tuple)) and data and isinstance(data[0], Mapping):
        values = []
        for row in data:
            found = next((row[n] for n in names if n in row), None)
            if found is None:
                if required:
                    raise KeyError(f"training rows need one of {names}")
                return None
            values.append(found)
        return values
    if required:
        raise KeyError(f"training data need one of {names}")
    return None


class _PrototypeNet(nn.Module):
    def __init__(self, action_dim: int, state_dim: int, feature_dim: int, base_dim: int, residual_dim: int,
                 width: int, decoder_obs: bool):
        super().__init__()
        self.base_dim = base_dim
        self.residual_dim = residual_dim
        self.encoder = nn.Sequential(nn.Linear(action_dim + state_dim, width), nn.GELU(),
                                     nn.Linear(width, width), nn.GELU(), nn.Linear(width, base_dim + residual_dim))
        self.decoder_obs = bool(decoder_obs)
        decoder_feature_dim = feature_dim if self.decoder_obs else 0
        self.decoder = nn.Sequential(nn.Linear(base_dim + residual_dim + decoder_feature_dim + state_dim, width), nn.GELU(),
                                     nn.Linear(width, width), nn.GELU(), nn.Linear(width, action_dim))

    def encode(self, actions: torch.Tensor, state: torch.Tensor):
        base, residual = self.encoder(torch.cat((actions, state), dim=-1)).split(
            (self.base_dim, self.residual_dim), dim=-1
        )
        return F.normalize(base, dim=-1, eps=1e-8), residual

    def decode(self, base: torch.Tensor, residual: torch.Tensor, feature: torch.Tensor, state: torch.Tensor):
        parts = [base, residual]
        if self.decoder_obs:
            parts.append(feature)
        parts.append(state)
        return self.decoder(torch.cat(parts, dim=-1))


class M4:
    """Growing prototype base; `update_prototypes=False` is the frozen ART condition."""

    DP_MEANS_LAMBDA_POINTS = (0.5, 0.75, 1.0)
    VIGILANCE_POINTS = (0.70, 0.85, 0.95)
    RESIDUAL_BETA_POINTS = (0.01, 0.1)

    VARIANTS = ("add_only", "adaptive", "merge")

    def __init__(self, *, dpmeans_lambda: float = 0.75, vigilance: float = 0.85,
                 update_prototypes: bool | None = None, variant: str | None = None,
                 merge_threshold: float | None = None, merge_distance_quantile: float = 0.25,
                 merge_action_mse_tolerance: float = 0.01,
                 novelty_distance_quantile: float = 0.95, residual_beta: float = 0.1,
                 decoder_obs: bool = False, base_dim: int = 12, residual_dim: int = 4,
                 width: int = 96, epochs: int = 20, batch_size: int = 128,
                 lr: float = 1e-3, max_prototypes: int = 256, update_rate: float = 0.1,
                 retrieval_temperature: float = 0.1, seed: int = 0, device: str = "cpu"):
        if dpmeans_lambda <= 0:
            raise ValueError("dpmeans_lambda must be positive")
        if not 0.0 <= vigilance <= 1.0:
            raise ValueError("vigilance must be in [0, 1]")
        if variant is not None and variant not in self.VARIANTS:
            raise ValueError(f"variant must be one of {self.VARIANTS}")
        if variant is None:
            variant = "adaptive" if update_prototypes is not False else "add_only"
        variant_updates = variant in ("adaptive", "merge")
        if update_prototypes is not None and bool(update_prototypes) != variant_updates:
            raise ValueError("update_prototypes conflicts with variant")
        if merge_threshold is not None and merge_threshold <= 0:
            raise ValueError("merge_threshold must be positive")
        if merge_action_mse_tolerance < 0:
            raise ValueError("merge_action_mse_tolerance must be nonnegative")
        if not 0 < merge_distance_quantile <= 1 or not 0 < novelty_distance_quantile <= 1:
            raise ValueError("distance quantiles must be in (0, 1]")
        if residual_beta < 0:
            raise ValueError("residual_beta must be nonnegative")
        if variant_updates and not 0.0 < update_rate <= 1.0:
            raise ValueError("update_rate must be in (0, 1]")
        self.dpmeans_lambda = float(dpmeans_lambda)
        self.vigilance = float(vigilance)
        self.variant = variant
        self.update_prototypes = variant_updates
        self.merge_threshold = None if merge_threshold is None else float(merge_threshold)
        self.merge_distance_quantile = float(merge_distance_quantile)
        self.merge_action_mse_tolerance = float(merge_action_mse_tolerance)
        self.novelty_distance_quantile = float(novelty_distance_quantile)
        self.residual_beta = float(residual_beta)
        self.decoder_obs = bool(decoder_obs)
        self.novelty_distance_threshold: float | None = None
        self.merge_threshold_calibrated = merge_threshold is None
        self.base_dim, self.residual_dim = int(base_dim), int(residual_dim)
        self.width, self.epochs, self.batch_size = int(width), int(epochs), int(batch_size)
        self.lr, self.max_prototypes = float(lr), int(max_prototypes)
        self.update_rate, self.retrieval_temperature = float(update_rate), float(retrieval_temperature)
        self.seed, self.device = int(seed), torch.device(device)
        self.net: _PrototypeNet | None = None
        self.prior_net: nn.Module | None = None
        self._prior_prototype_count = 0
        self._prior_code_map = np.empty((0,), dtype=np.int64)
        self.prototypes = np.empty((0, self.base_dim), np.float32)
        self.prototype_counts: list[int] = []
        self.memory: list[dict[str, Any]] = []
        self.merge_history: list[dict[str, Any]] = []
        self.resonance_update_count = 0
        self.resonance_total_movement = 0.0
        self.resonance_max_movement = 0.0
        self._merge_eval: dict[str, np.ndarray] | None = None
        self.normalizers: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.action_shape: tuple[int, ...] | None = None
        self._fitted = False

    def _require_fit(self):
        if not self._fitted:
            raise RuntimeError("call fit(train) before using this method")

    @staticmethod
    def _fit_norm(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale[scale < 1e-6] = 1.0
        return mean.astype(np.float32), scale.astype(np.float32)

    def _normalise(self, name: str, x: np.ndarray) -> np.ndarray:
        mean, scale = self.normalizers[name]
        return ((x - mean) / scale).astype(np.float32)

    @staticmethod
    def _dpmeans(values: np.ndarray, threshold: float, cap: int) -> tuple[np.ndarray, np.ndarray]:
        centers: list[np.ndarray] = []
        counts: list[int] = []
        labels = np.empty(len(values), np.int64)
        for i, value in enumerate(values):
            if not centers:
                centers.append(value.copy())
                counts.append(1)
                labels[i] = 0
                continue
            distances = np.linalg.norm(np.asarray(centers) - value[None, :], axis=1)
            closest = int(np.argmin(distances))
            if distances[closest] > threshold and len(centers) < cap:
                centers.append(value.copy())
                counts.append(1)
                labels[i] = len(centers) - 1
            else:
                labels[i] = closest
                counts[closest] += 1
                centers[closest] += (value - centers[closest]) / counts[closest]
                centers[closest] /= max(np.linalg.norm(centers[closest]), 1e-8)
        return np.asarray(centers, dtype=np.float32), labels

    def fit(self, train) -> None:
        """Fit only the supplied train split (the harness supplies object/goal rows)."""
        actions = _np(_field(train, ("A", "actions", "action")))
        if actions.ndim < 2:
            raise ValueError("actions must have batch and action dimensions")
        self.action_shape = tuple(actions.shape[1:])
        action_flat = actions.reshape(len(actions), -1)
        state_value = _field(train, ("s", "state", "proprio"), required=False)
        feature_value = _field(train, ("f", "feature", "features"), required=False)
        states = np.zeros((len(actions), 0), np.float32) if state_value is None else _np(state_value).reshape(len(actions), -1)
        features = np.zeros((len(actions), 0), np.float32) if feature_value is None else _np(feature_value).reshape(len(actions), -1)
        if len(states) != len(actions) or len(features) != len(actions):
            raise ValueError("actions, state and feature batch lengths must match")
        for name, array in (("a", action_flat), ("s", states), ("f", features)):
            self.normalizers[name] = self._fit_norm(array) if array.shape[1] else (array.mean(0), np.ones(0, np.float32))

        torch.manual_seed(self.seed)
        self.net = _PrototypeNet(action_flat.shape[1], states.shape[1], features.shape[1], self.base_dim,
                                 self.residual_dim, self.width, self.decoder_obs).to(self.device)
        a = torch.as_tensor(self._normalise("a", action_flat), device=self.device)
        s = torch.as_tensor(self._normalise("s", states), device=self.device)
        f = torch.as_tensor(self._normalise("f", features), device=self.device)
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-5)
        rng = np.random.default_rng(self.seed)
        train_ix,val_ix=validation_split(len(a),self.seed); train_t=torch.as_tensor(train_ix,device=self.device); val=torch.as_tensor(val_ix,device=self.device)
        best=float("inf"); stale=0; self.epochs_run=0
        self.net.train()
        for epoch in range(self.epochs):
            order = rng.permutation(train_ix)
            for start in range(0, len(train_ix), self.batch_size):
                idx = torch.as_tensor(order[start:start + self.batch_size], device=self.device)
                base, residual = self.net.encode(a[idx], s[idx])
                pred = self.net.decode(base, residual, f[idx], s[idx])
                residual_kl = 0.5 * residual.square().sum(dim=-1).mean()
                loss = F.mse_loss(pred, a[idx]) + self.residual_beta * residual_kl
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                optimizer.step()
            self.epochs_run=epoch+1
            if len(val):
                self.net.eval()
                with torch.no_grad():
                    vb,vr=self.net.encode(a[val],s[val]); vp=self.net.decode(vb,vr,f[val],s[val])
                    val_loss=F.mse_loss(vp,a[val]).item()
                self.net.train(); best,stale,stop=update_early_stopping(val_loss,best,stale,patience=3)
                if stop: break

        self.net.eval()
        with torch.no_grad():
            encoded, residual = self.net.encode(a[train_t], s[train_t])
        prototypes, _ = self._dpmeans(encoded.cpu().numpy(), self.dpmeans_lambda, self.max_prototypes)
        self.prototypes = prototypes
        self.prototype_counts = [0] * len(prototypes)
        distances = np.linalg.norm(encoded.cpu().numpy()[:, None, :] - prototypes[None, :, :], axis=-1)
        labels = distances.argmin(axis=1)
        nearest_train_distances = distances[np.arange(len(distances)), labels]
        self.novelty_distance_threshold = float(np.quantile(nearest_train_distances, self.novelty_distance_quantile))
        prototype_pairs = np.linalg.norm(prototypes[:, None, :] - prototypes[None, :, :], axis=-1)
        upper = prototype_pairs[np.triu_indices(len(prototypes), k=1)]
        if self.merge_threshold is None:
            self.merge_threshold = float(np.quantile(upper, self.merge_distance_quantile)) if len(upper) else 0.0
        self.merge_threshold_calibrated = True
        self.prototype_counts = np.bincount(labels, minlength=len(prototypes)).astype(int).tolist()
        # Align the action decoder to the quantized base actually returned by
        # encode() and memory_query(), rather than only to pre-quantization z.
        prototype_t = torch.as_tensor(prototypes[labels], device=self.device)
        with torch.no_grad():
            _, residual_all = self.net.encode(a, s)
        decoder_optimizer = torch.optim.AdamW(self.net.decoder.parameters(), lr=self.lr, weight_decay=1e-5)
        self.net.decoder.train()
        best=float("inf"); stale=0; self.decoder_epochs_run=0
        for epoch in range(max(1, self.epochs // 2)):
            order = rng.permutation(len(train_ix))
            for start in range(0, len(train_ix), self.batch_size):
                idx = torch.as_tensor(order[start:start + self.batch_size], device=self.device)
                gi=train_t[idx]
                pred = self.net.decode(prototype_t[idx], residual[idx], f[gi], s[gi])
                residual_kl = 0.5 * residual_all[gi].square().sum(dim=-1).mean()
                loss = F.mse_loss(pred, a[gi]) + self.residual_beta * residual_kl
                decoder_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.decoder.parameters(), 1.0)
                decoder_optimizer.step()
            self.decoder_epochs_run=epoch+1
            if len(val):
                with torch.no_grad():
                    val_encoded,val_residual=self.net.encode(a[val],s[val])
                    proto=torch.as_tensor(prototypes,device=self.device)
                    val_proto=proto[torch.linalg.vector_norm(val_encoded[:,None,:]-proto[None,:,:],dim=-1).argmin(1)]
                    val_pred=self.net.decode(val_proto,val_residual,f[val],s[val]); val_loss=F.mse_loss(val_pred,a[val]).item()
                best,stale,stop=update_early_stopping(val_loss,best,stale,patience=3)
                if stop: break
        self.net.decoder.eval()
        # Keep a frozen object/goal evaluation bank so explicit merge() can
        # reject representative changes that materially worsen action MSE.
        with torch.no_grad():
            eval_base = prototypes[labels]
            eval_pred = self.net.decode(torch.as_tensor(eval_base, device=self.device), residual,
                                        f[train_t], s[train_t])
            eval_pred_raw = eval_pred.cpu().numpy() * self.normalizers["a"][1] + self.normalizers["a"][0]
            eval_actions_raw = a[train_t].cpu().numpy() * self.normalizers["a"][1] + self.normalizers["a"][0]
        self._merge_eval = {
            "codes": labels.astype(np.int64),
            "actions": eval_actions_raw.astype(np.float32),
            "residual": residual.cpu().numpy().astype(np.float32),
            "state": s[train_t].cpu().numpy().astype(np.float32),
            "features": f[train_t].cpu().numpy().astype(np.float32),
            "baseline_mse": np.mean((eval_pred_raw - eval_actions_raw) ** 2, axis=-1).astype(np.float32),
        }
        # Prior returns a categorical distribution over the training prototypes,
        # plus the continuous residual. Its final dimension is fixed after fit.
        self.prior_net = nn.Sequential(
            nn.Linear(features.shape[1] + states.shape[1], self.width), nn.GELU(),
            nn.Linear(self.width, self.width), nn.GELU(),
            nn.Linear(self.width, len(prototypes) + self.residual_dim),
        ).to(self.device)
        self._prior_prototype_count = len(prototypes)
        self._prior_code_map = np.arange(len(prototypes), dtype=np.int64)
        opt_prior = torch.optim.AdamW(self.prior_net.parameters(), lr=self.lr, weight_decay=1e-5)
        label_t = torch.as_tensor(labels, dtype=torch.long, device=self.device)
        residual_t = residual.detach()
        self.prior_net.train()
        best=float("inf"); stale=0; self.prior_epochs_run=0
        for epoch in range(max(1, self.epochs // 2)):
            order = rng.permutation(len(train_ix))
            for start in range(0, len(train_ix), self.batch_size):
                idx = torch.as_tensor(order[start:start + self.batch_size], device=self.device)
                gi=train_t[idx]
                out = self.prior_net(torch.cat((f[gi], s[gi]), dim=-1))
                logits, r_pred = out[:, :len(prototypes)], out[:, len(prototypes):]
                loss = F.cross_entropy(logits, label_t[idx]) + 0.25 * F.mse_loss(r_pred, residual_t[idx])
                opt_prior.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.prior_net.parameters(), 1.0)
                opt_prior.step()
            self.prior_epochs_run=epoch+1
            if len(val):
                with torch.no_grad():
                    out=self.prior_net(torch.cat((f[val],s[val]),dim=-1)); logits,rpred=out[:,:len(prototypes)],out[:,len(prototypes):]
                    vencoded,vres=self.net.encode(a[val],s[val]); vproto=proto[torch.linalg.vector_norm(vencoded[:,None,:]-proto[None,:,:],dim=-1).argmin(1)]
                    vlabs=torch.linalg.vector_norm(vencoded[:,None,:]-proto[None,:,:],dim=-1).argmin(1)
                    val_loss=(F.cross_entropy(logits,vlabs)+.25*F.mse_loss(rpred,vres)).item()
                best,stale,stop=update_early_stopping(val_loss,best,stale,patience=3)
                if stop: break
        self.prior_net.eval()
        self._fitted = True

    def _encoded(self, actions, state) -> tuple[np.ndarray, np.ndarray]:
        self._require_fit()
        actions_np = _np(actions)
        one = actions_np.ndim == len(self.action_shape or ())
        if one:
            actions_np = actions_np[None, ...]
        flat = actions_np.reshape(len(actions_np), -1)
        state_np = np.zeros((len(flat), 0), np.float32) if state is None else _np(state).reshape(len(flat), -1)
        with torch.no_grad():
            base, residual = self.net.encode(
                torch.as_tensor(self._normalise("a", flat), device=self.device),
                torch.as_tensor(self._normalise("s", state_np), device=self.device),
            )
        return base.cpu().numpy(), residual.cpu().numpy()

    def encode(self, A, s):
        base, residual = self._encoded(A, s)
        codes = np.linalg.norm(base[:, None, :] - self.prototypes[None, :, :], axis=-1).argmin(axis=1)
        # Keep the continuous base embedding alongside its discrete code. The
        # embedding is what ART compares to existing prototypes on memory_add.
        return {"codes": codes.astype(np.int64), "embedding": base.astype(np.float32)}, residual.astype(np.float32)

    def _base_array(self, b) -> np.ndarray:
        if isinstance(b, Mapping):
            if "embedding" in b:
                return _np(b["embedding"])
            if "base" in b:
                return _np(b["base"])
            if "codes" in b:
                codes = np.asarray(b["codes"], dtype=np.int64).reshape(-1)
                return self.prototypes[codes]
        arr = np.asarray(b)
        if np.issubdtype(arr.dtype, np.integer):
            return self.prototypes[arr.astype(np.int64).reshape(-1)]
        return _np(arr)

    def decode(self, b, r, f, s):
        self._require_fit()
        base = self._base_array(b)
        residual = _np(r)
        one = base.ndim == 1
        if one:
            base, residual = base[None, :], residual.reshape(1, -1)
        features = np.zeros((len(base), 0), np.float32) if f is None else _np(f).reshape(len(base), -1)
        states = np.zeros((len(base), 0), np.float32) if s is None else _np(s).reshape(len(base), -1)
        with torch.no_grad():
            pred = self.net.decode(
                torch.as_tensor(base, device=self.device), torch.as_tensor(residual, device=self.device),
                torch.as_tensor(self._normalise("f", features), device=self.device),
                torch.as_tensor(self._normalise("s", states), device=self.device),
            ).cpu().numpy()
        mean, scale = self.normalizers["a"]
        pred = (pred * scale + mean).reshape(len(base), *(self.action_shape or ()))
        return pred[0] if one else pred

    def prior(self, f, s):
        self._require_fit()
        features = _np(f)
        states = _np(s)
        if features.ndim == 1:
            features = features[None, :]
        if states.ndim == 1:
            states = states[None, :]
        with torch.no_grad():
            out = self.prior_net(torch.cat((
                torch.as_tensor(self._normalise("f", features), device=self.device),
                torch.as_tensor(self._normalise("s", states), device=self.device),
            ), dim=-1))
        logits, residual = out[:, :self._prior_prototype_count], out[:, self._prior_prototype_count:]
        old_probs = torch.softmax(logits, dim=-1).cpu().numpy()
        probs = np.zeros((len(old_probs), len(self.prototypes)), dtype=np.float32)
        if self._prior_code_map.size:
            for old_id, new_id in enumerate(self._prior_code_map):
                probs[:, new_id] += old_probs[:, old_id]
        expected_base = probs @ self.prototypes
        return {"probs": probs.astype(np.float32), "base": expected_base.astype(np.float32)}, residual.cpu().numpy()

    def memory_add(self, k, b, meta=None):
        self._require_fit()
        key = _np(k).reshape(-1)
        embeddings = self._base_array(b)
        if len(embeddings) != 1 or embeddings.ndim != 2:
            raise ValueError("memory_add accepts one sample at a time")
        embedding = embeddings[0] / max(np.linalg.norm(embeddings[0]), 1e-8)
        distances = np.linalg.norm(self.prototypes - embedding[None, :], axis=-1)
        nearest = int(np.argmin(distances))
        similarity = self.prototypes @ embedding
        distance = float(distances[nearest])
        threshold = float(self.novelty_distance_threshold or 0.0)
        is_new_skill = distance > threshold
        movement = 0.0
        if not is_new_skill:
            if self.update_prototypes:
                # Exponential moving average: use the configured rate on
                # every resonance. Dividing by historical fit count made
                # adaptation effectively invisible for frequently seen units.
                rate = self.update_rate
                before = self.prototypes[nearest].copy()
                updated = (1 - rate) * self.prototypes[nearest] + rate * embedding
                self.prototypes[nearest] = updated / max(np.linalg.norm(updated), 1e-8)
                self.prototype_counts[nearest] += 1
                movement = float(np.linalg.norm(self.prototypes[nearest] - before))
                self.resonance_update_count += 1
                self.resonance_total_movement += movement
                self.resonance_max_movement = max(self.resonance_max_movement, movement)
            reason = "resonance_updated" if self.update_prototypes else "resonance_frozen"
            added_prototype = False
        else:
            if len(self.prototypes) >= self.max_prototypes:
                reason = "vigilance_miss_capacity_reached"
                added_prototype = False
                nearest = int(np.argmax(similarity))
            else:
                nearest = len(self.prototypes)
                self.prototypes = np.concatenate((self.prototypes, embedding[None, :]), axis=0)
                self.prototype_counts.append(1)
                reason = "new_prototype"
                added_prototype = True
        self.memory.append({"key": key.copy(), "code": nearest, "meta": meta})
        return {"added": True, "added_prototype": added_prototype, "code": nearest, "reason": reason,
                "base_distance": distance, "novelty_distance_threshold": threshold,
                "prototype_movement_l2": movement,
                "resonance_update_count": self.resonance_update_count}

    def memory_merge(self) -> dict[str, Any]:
        """Merge only nearby prototypes whose representative change preserves actions."""
        self._require_fit()
        if self.variant != "merge":
            raise RuntimeError("memory_merge is available only for variant='merge'")
        old = self.prototypes.copy()
        old_count = len(old)
        counts = np.asarray([max(int(c), 1) for c in self.prototype_counts], dtype=np.float64)
        distances = np.linalg.norm(old[:, None, :] - old[None, :, :], axis=-1)
        candidates = sorted((float(distances[i, j]), i, j)
                            for i in range(old_count) for j in range(i + 1, old_count)
                            if distances[i, j] <= self.merge_threshold)
        groups = [[i] for i in range(old_count)]
        rejected_pairs: list[dict[str, float | int]] = []
        accepted_pairs: list[dict[str, float | int]] = []

        def representative(group: list[int]) -> np.ndarray:
            value = np.average(old[group], axis=0, weights=counts[group]).astype(np.float32)
            return value / max(float(np.linalg.norm(value)), 1e-8)

        def merge_delta(group: list[int]) -> float:
            bank = self._merge_eval
            if bank is not None:
                rows = np.isin(bank["codes"], group)
                if rows.any():
                    with torch.no_grad():
                        device = self.device
                        old_base = torch.as_tensor(old[bank["codes"][rows]], device=device)
                        new_base = torch.as_tensor(np.repeat(representative(group)[None, :], int(rows.sum()), axis=0), device=device)
                        residual = torch.as_tensor(bank["residual"][rows], device=device)
                        state = torch.as_tensor(bank["state"][rows], device=device)
                        feature = torch.as_tensor(bank["features"][rows], device=device)
                        before = self.net.decode(old_base, residual, feature, state)
                        after = self.net.decode(new_base, residual, feature, state)
                        scale = torch.as_tensor(self.normalizers["a"][1], device=device)
                        target = torch.as_tensor(bank["actions"][rows], device=device)
                        before_mse = ((before * scale + torch.as_tensor(self.normalizers["a"][0], device=device) - target) ** 2).mean(dim=-1)
                        after_mse = ((after * scale + torch.as_tensor(self.normalizers["a"][0], device=device) - target) ** 2).mean(dim=-1)
                    return float((after_mse - before_mse).mean().cpu())
            # Newly added units may not occur in the fit bank. Compare their
            # decoded action signatures using the same residual/state context.
            if self._merge_eval is None:
                return 0.0
            bank = self._merge_eval
            residual = np.repeat(bank["residual"].mean(axis=0, keepdims=True), len(group), axis=0)
            state = np.repeat(bank["state"].mean(axis=0, keepdims=True), len(group), axis=0)
            feature = np.repeat(bank["features"].mean(axis=0, keepdims=True), len(group), axis=0)
            with torch.no_grad():
                outputs = self.net.decode(torch.as_tensor(old[group], device=self.device),
                                          torch.as_tensor(residual, device=self.device),
                                          torch.as_tensor(feature, device=self.device),
                                          torch.as_tensor(state, device=self.device)).cpu().numpy()
            return float(np.mean(np.sum((outputs - outputs.mean(axis=0, keepdims=True)) ** 2, axis=-1)))

        for distance, left, right in candidates:
            gl = next(group for group in groups if left in group)
            gr = next(group for group in groups if right in group)
            if gl is gr:
                continue
            combined = sorted(gl + gr)
            delta = merge_delta(combined)
            pair = {"left": left, "right": right, "base_distance": distance, "action_mse_delta": delta}
            if delta <= self.merge_action_mse_tolerance:
                groups.remove(gl)
                groups.remove(gr)
                groups.append(combined)
                groups.sort(key=lambda group: group[0])
                accepted_pairs.append(pair)
            else:
                rejected_pairs.append(pair)

        old_to_new = np.empty(old_count, dtype=np.int64)
        new_prototypes, new_counts = [], []
        for new_id, group in enumerate(groups):
            old_to_new[group] = new_id
            new_prototypes.append(representative(group))
            new_counts.append(int(sum(self.prototype_counts[i] for i in group)))
        pre_merge_eval_mse = post_merge_eval_mse = None
        if self._merge_eval is not None:
            bank = self._merge_eval
            rows = bank["codes"]
            with torch.no_grad():
                device = self.device
                before_pred = self.net.decode(torch.as_tensor(old[rows], device=device),
                                              torch.as_tensor(bank["residual"], device=device),
                                              torch.as_tensor(bank["features"], device=device),
                                              torch.as_tensor(bank["state"], device=device)).cpu().numpy()
                new_base = np.asarray(new_prototypes, dtype=np.float32)[old_to_new[rows]]
                pred = self.net.decode(torch.as_tensor(new_base, device=self.device),
                                       torch.as_tensor(bank["residual"], device=self.device),
                                       torch.as_tensor(bank["features"], device=self.device),
                                       torch.as_tensor(bank["state"], device=self.device)).cpu().numpy()
            before_pred = before_pred * self.normalizers["a"][1] + self.normalizers["a"][0]
            pred = pred * self.normalizers["a"][1] + self.normalizers["a"][0]
            pre_merge_eval_mse = float(np.mean((before_pred - bank["actions"]) ** 2))
            post_merge_eval_mse = float(np.mean((pred - bank["actions"]) ** 2))
        accepted_base_distances = [float(pair["base_distance"]) for pair in accepted_pairs]
        accepted_action_deltas = [float(pair["action_mse_delta"]) for pair in accepted_pairs]

        def distribution(values: list[float]) -> dict[str, float | int | None]:
            if not values:
                return {"count": 0, "min": None, "median": None, "p95": None, "max": None}
            return {"count": len(values), "min": min(values), "median": float(np.median(values)),
                    "p95": float(np.quantile(values, 0.95)), "max": max(values)}
        self.prototypes = np.asarray(new_prototypes, dtype=np.float32).reshape(-1, self.base_dim)
        self.prototype_counts = new_counts
        for item in self.memory:
            item["code"] = int(old_to_new[item["code"]])
        if self._prior_code_map.size:
            self._prior_code_map = old_to_new[self._prior_code_map]
        report = {
            "pass": len(self.merge_history) + 1,
            "threshold": self.merge_threshold,
            "action_mse_tolerance": self.merge_action_mse_tolerance,
            "pre_merge_count": old_count,
            "post_merge_count": len(self.prototypes),
            "pre_merge_ids": list(range(old_count)),
            "post_merge_ids": list(range(len(self.prototypes))),
            "old_to_new": old_to_new.tolist(),
            "merged_groups": groups,
            "merged_pair_base_distances": accepted_base_distances,
            "merged_pair_base_distance_distribution": distribution(accepted_base_distances),
            "merged_pair_action_mse_deltas": accepted_action_deltas,
            "merged_pair_action_mse_delta_distribution": distribution(accepted_action_deltas),
            "pre_merge_eval_mse": pre_merge_eval_mse,
            "post_merge_eval_mse": post_merge_eval_mse,
            "merge_eval_mse_delta": (None if pre_merge_eval_mse is None or post_merge_eval_mse is None
                                      else post_merge_eval_mse - pre_merge_eval_mse),
            "rejected_merge_pairs": rejected_pairs,
            "representatives_are_count_weighted": True,
            "memory_representatives_changed": bool(any(len(group) > 1 for group in groups)),
            "old_ids_changed": bool(any(old_to_new[i] != i for i in range(old_count))),
        }
        self.merge_history.append(report)
        return report

    def memory_query(self, k, *, top_k: int = 1):
        self._require_fit()
        if not self.memory:
            return {"bases": np.empty((0, self.base_dim), np.float32), "codes": np.empty((0,), np.int64),
                    "weights": np.empty((0,), np.float32)}
        query = _np(k).reshape(-1)
        keys = np.stack([item["key"] for item in self.memory])
        if query.shape != (keys.shape[1],):
            raise ValueError(f"query key shape {query.shape} != memory key shape {(keys.shape[1],)}")
        qn = query / max(np.linalg.norm(query), 1e-8)
        kn = keys / np.maximum(np.linalg.norm(keys, axis=1, keepdims=True), 1e-8)
        sims = kn @ qn
        ix = np.argsort(-sims)[:max(1, int(top_k))]
        logits = (sims[ix] - sims[ix].max()) / max(self.retrieval_temperature, 1e-6)
        weights = np.exp(logits); weights /= weights.sum()
        codes = np.asarray([self.memory[i]["code"] for i in ix], dtype=np.int64)
        return {"bases": self.prototypes[codes].copy(), "codes": codes, "weights": weights.astype(np.float32)}

    def describe(self) -> dict[str, Any]:
        return {
            "method": "M4: DP-means prototypes + ART vigilance memory growth",
            "variant": self.variant,
            "variants": list(self.VARIANTS),
            "base": f"unit-normalized {self.base_dim}-D action encoder embedding clustered by DP-means; decode uses prototype + residual",
            "residual_dim": self.residual_dim,
            "dpmeans_lambda": self.dpmeans_lambda,
            "dpmeans_lambda_sweep": list(self.DP_MEANS_LAMBDA_POINTS),
            "vigilance": self.vigilance,
            "vigilance_sweep": list(self.VIGILANCE_POINTS),
            "prototype_count": int(len(self.prototypes)),
            "prototype_capacity": self.max_prototypes,
            "memory_update": "non-gradient ART resonance; update moving average" if self.update_prototypes else "non-gradient ART resonance; prototypes frozen",
            "update_rate": self.update_rate if self.update_prototypes else None,
            "resonance_updates": {
                "count": self.resonance_update_count,
                "total_prototype_movement_l2": self.resonance_total_movement,
                "mean_movement_l2": (self.resonance_total_movement / self.resonance_update_count
                                      if self.resonance_update_count else 0.0),
                "max_movement_l2": self.resonance_max_movement,
                "update_rule": "fixed-rate normalized exponential moving average",
            },
            "memory_growth": "create a new unit when nearest Euclidean base distance exceeds the train-calibrated quantile threshold",
            "legacy_vigilance_parameter": self.vigilance,
            "decoder_obs": self.decoder_obs,
            "residual_kl": {"beta": self.residual_beta, "sweep": list(self.RESIDUAL_BETA_POINTS),
                            "prior": "standard normal; 0.5 * ||r||^2"},
            "unit_novelty": {"distance": "Euclidean distance in normalized base space",
                             "quantile": self.novelty_distance_quantile,
                             "calibrated_threshold": self.novelty_distance_threshold},
            "merge": {
                "threshold": self.merge_threshold,
                "action_mse_tolerance": self.merge_action_mse_tolerance,
                "distance_quantile": self.merge_distance_quantile,
                "threshold_calibrated_from_training_units": self.merge_threshold_calibrated,
                "trigger": "explicit memory_merge() once after the spatial store pass" if self.variant == "merge" else "disabled",
                "representative": "count-weighted mean, renormalized; memory representative values change and old IDs change",
                "history": list(self.merge_history),
                "prior_old_to_current": self._prior_code_map.tolist(),
            },
            "training_data": "fit consumes only the caller-provided rows; comparison harness must pass object/goal train rows",
        }
