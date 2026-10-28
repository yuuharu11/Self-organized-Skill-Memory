"""M5: sparse compositional bases with an incrementally extended dictionary."""
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


class _SparseNet(nn.Module):
    def __init__(self, action_dim: int, state_dim: int, feature_dim: int, base_dim: int, residual_dim: int,
                 n_atoms: int, width: int, decoder_obs: bool):
        super().__init__()
        self.base_dim, self.residual_dim = base_dim, residual_dim
        self.encoder = nn.Sequential(nn.Linear(action_dim + state_dim, width), nn.GELU(),
                                     nn.Linear(width, width), nn.GELU(), nn.Linear(width, base_dim + residual_dim))
        dictionary = F.normalize(torch.randn(n_atoms, base_dim), dim=-1)
        self.dictionary = nn.Parameter(dictionary)
        self.decoder_obs = bool(decoder_obs)
        decoder_feature_dim = feature_dim if self.decoder_obs else 0
        self.decoder = nn.Sequential(nn.Linear(base_dim + residual_dim + decoder_feature_dim + state_dim, width), nn.GELU(),
                                     nn.Linear(width, width), nn.GELU(), nn.Linear(width, action_dim))
        self.prior = nn.Sequential(nn.Linear(feature_dim + state_dim, width), nn.GELU(),
                                   nn.Linear(width, width), nn.GELU(), nn.Linear(width, n_atoms + residual_dim))

    def sparse_coefficients(self, scores: torch.Tensor, top_k: int) -> torch.Tensor:
        top_k = min(max(1, top_k), scores.shape[-1])
        values, indices = torch.topk(scores, top_k, dim=-1)
        weights = torch.softmax(values, dim=-1)
        result = torch.zeros_like(scores)
        return result.scatter(-1, indices, weights)

    def encode(self, actions: torch.Tensor, state: torch.Tensor, top_k: int):
        raw_base, residual = self.encoder(torch.cat((actions, state), dim=-1)).split(
            (self.base_dim, self.residual_dim), dim=-1
        )
        raw_base = F.normalize(raw_base, dim=-1, eps=1e-8)
        dictionary = F.normalize(self.dictionary, dim=-1, eps=1e-8)
        alpha = self.sparse_coefficients(raw_base @ dictionary.T, top_k)
        reconstructed = alpha @ dictionary
        innovation = raw_base - reconstructed
        return alpha, residual, innovation, raw_base, dictionary

    def decode(self, base: torch.Tensor, residual: torch.Tensor, feature: torch.Tensor, state: torch.Tensor):
        parts = [base, residual]
        if self.decoder_obs:
            parts.append(feature)
        parts.append(state)
        return self.decoder(torch.cat(parts, dim=-1))


class M5:
    """Sparse atom-combination base; atom creation is a non-gradient memory update."""

    INITIAL_ATOM_POINTS = (8, 16, 32)
    TOP_K_POINTS = (2, 4, 6)
    COHERENCE_WEIGHT_POINTS = (0.05, 0.2)
    RESIDUAL_BETA_POINTS = (0.01, 0.1)
    VARIANTS = ("fixed", "growing")

    def __init__(self, *, n_atoms: int = 16, top_k: int = 4, base_dim: int = 12, residual_dim: int = 4,
                 width: int = 96, epochs: int = 20, batch_size: int = 128, lr: float = 1e-3,
                 sparsity_weight: float = 0.01, coherence_weight: float = 0.05,
                 novelty_threshold: float | None = None, novelty_quantile: float = 0.95,
                 residual_beta: float = 0.1, decoder_obs: bool = False,
                 novelty_batch: int = 4, max_atoms: int = 64,
                 variant: str = "growing", retrieval_temperature: float = 0.1,
                 seed: int = 0, device: str = "cpu"):
        if n_atoms < 1 or top_k < 1 or top_k > n_atoms:
            raise ValueError("require 1 <= top_k <= n_atoms")
        if novelty_batch < 1 or max_atoms < n_atoms:
            raise ValueError("invalid novelty_batch or max_atoms")
        if novelty_threshold is not None and novelty_threshold < 0:
            raise ValueError("novelty_threshold must be nonnegative")
        if not 0 < novelty_quantile <= 1 or residual_beta < 0:
            raise ValueError("invalid novelty_quantile or residual_beta")
        if variant not in self.VARIANTS:
            raise ValueError(f"variant must be one of {self.VARIANTS}")
        self.n_atoms, self.top_k = int(n_atoms), int(top_k)
        self.variant = variant
        self.base_dim, self.residual_dim = int(base_dim), int(residual_dim)
        self.width, self.epochs, self.batch_size = int(width), int(epochs), int(batch_size)
        self.lr, self.sparsity_weight, self.coherence_weight = float(lr), float(sparsity_weight), float(coherence_weight)
        self.novelty_threshold = None if novelty_threshold is None else float(novelty_threshold)
        self.novelty_threshold_calibrated = novelty_threshold is None
        self.novelty_quantile, self.residual_beta = float(novelty_quantile), float(residual_beta)
        self.decoder_obs = bool(decoder_obs)
        self.max_atoms, self.novelty_batch = int(max_atoms), int(novelty_batch)
        self.retrieval_temperature, self.seed = float(retrieval_temperature), int(seed)
        self.device = torch.device(device)
        self.net: _SparseNet | None = None
        self.action_shape: tuple[int, ...] | None = None
        self.normalizers: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.memory: list[dict[str, Any]] = []
        self.pending_innovations: list[np.ndarray] = []
        self.atom_additions = 0
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
    def _coherence(dictionary: torch.Tensor) -> torch.Tensor:
        atoms = F.normalize(dictionary, dim=-1, eps=1e-8)
        gram = atoms @ atoms.T
        offdiag = gram - torch.diag_embed(torch.diagonal(gram))
        return offdiag.square().sum() / max(1, atoms.shape[0] * (atoms.shape[0] - 1))

    @staticmethod
    def _coherence_report(dictionary: torch.Tensor) -> dict[str, float]:
        atoms = F.normalize(dictionary.detach(), dim=-1, eps=1e-8)
        gram = atoms @ atoms.T
        if len(atoms) < 2:
            return {"mean_squared_mutual_coherence": 0.0, "max_abs_mutual_coherence": 0.0}
        upper = gram[torch.triu(torch.ones_like(gram, dtype=torch.bool), diagonal=1)]
        return {"mean_squared_mutual_coherence": float(upper.square().mean().cpu()),
                "max_abs_mutual_coherence": float(upper.abs().max().cpu())}

    def fit(self, train) -> None:
        """Learn a sparse action dictionary from only the supplied object/goal rows."""
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
        self.net = _SparseNet(action_flat.shape[1], states.shape[1], features.shape[1], self.base_dim,
                              self.residual_dim, self.n_atoms, self.width, self.decoder_obs).to(self.device)
        a = torch.as_tensor(self._normalise("a", action_flat), device=self.device)
        s = torch.as_tensor(self._normalise("s", states), device=self.device)
        f = torch.as_tensor(self._normalise("f", features), device=self.device)
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-5)
        rng = np.random.default_rng(self.seed)
        train_ix,val_ix=validation_split(len(a),self.seed); val=torch.as_tensor(val_ix,device=self.device)
        best=float("inf"); stale=0; self.epochs_run=0
        self.net.train()
        for epoch in range(self.epochs):
            order = rng.permutation(train_ix)
            for start in range(0, len(train_ix), self.batch_size):
                idx = torch.as_tensor(order[start:start + self.batch_size], device=self.device)
                alpha, residual, _, _, dictionary = self.net.encode(a[idx], s[idx], self.top_k)
                base = alpha @ dictionary
                pred = self.net.decode(base, residual, f[idx], s[idx])
                entropy = -(alpha.clamp_min(1e-8) * alpha.clamp_min(1e-8).log()).sum(-1).mean()
                coherence = self._coherence(self.net.dictionary)
                residual_kl = 0.5 * residual.square().sum(dim=-1).mean()
                loss = (F.mse_loss(pred, a[idx]) + self.sparsity_weight * entropy
                        + self.coherence_weight * coherence + self.residual_beta * residual_kl)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                optimizer.step()
                with torch.no_grad():
                    self.net.dictionary.copy_(F.normalize(self.net.dictionary, dim=-1, eps=1e-8))
            self.epochs_run=epoch+1
            if len(val):
                with torch.no_grad():
                    alpha_v,res_v,_,_,dict_v=self.net.encode(a[val],s[val],self.top_k)
                    pred_v=self.net.decode(alpha_v@dict_v,res_v,f[val],s[val]); val_loss=F.mse_loss(pred_v,a[val]).item()
                best,stale,stop=update_early_stopping(val_loss,best,stale,patience=3)
                if stop: break
        self.net.eval()
        with torch.no_grad():
            _, _, innovations, _, _ = self.net.encode(a[train_ix], s[train_ix], self.top_k)
        train_novelty = torch.linalg.vector_norm(innovations, dim=-1).cpu().numpy()
        if self.novelty_threshold is None:
            self.novelty_threshold = float(np.quantile(train_novelty, self.novelty_quantile))
        self._fitted = True

    def _encode(self, actions, state):
        self._require_fit()
        actions_np = _np(actions)
        one = actions_np.ndim == len(self.action_shape or ())
        if one:
            actions_np = actions_np[None, ...]
        flat = actions_np.reshape(len(actions_np), -1)
        states = np.zeros((len(flat), 0), np.float32) if state is None else _np(state).reshape(len(flat), -1)
        with torch.no_grad():
            alpha, residual, innovation, raw_base, _ = self.net.encode(
                torch.as_tensor(self._normalise("a", flat), device=self.device),
                torch.as_tensor(self._normalise("s", states), device=self.device), self.top_k,
            )
        return ({"alpha": alpha.cpu().numpy().astype(np.float32),
                 "innovation": innovation.cpu().numpy().astype(np.float32),
                 "raw_base": raw_base.cpu().numpy().astype(np.float32)},
                residual.cpu().numpy().astype(np.float32), one)

    def encode(self, A, s):
        base, residual, _ = self._encode(A, s)
        return base, residual

    def _alpha(self, b) -> np.ndarray:
        value = b["alpha"] if isinstance(b, Mapping) else b
        alpha = _np(value)
        if alpha.ndim == 1:
            alpha = alpha[None, :]
        n_atoms = self.net.dictionary.shape[0]
        if alpha.shape[1] > n_atoms:
            raise ValueError("base coefficient vector has more atoms than the current dictionary")
        if alpha.shape[1] < n_atoms:
            alpha = np.pad(alpha, ((0, 0), (0, n_atoms - alpha.shape[1])))
        return alpha

    def decode(self, b, r, f, s):
        self._require_fit()
        alpha = self._alpha(b)
        residual = _np(r)
        if residual.ndim == 1:
            residual = residual[None, :]
        features = np.zeros((len(alpha), 0), np.float32) if f is None else _np(f).reshape(len(alpha), -1)
        states = np.zeros((len(alpha), 0), np.float32) if s is None else _np(s).reshape(len(alpha), -1)
        with torch.no_grad():
            dictionary = F.normalize(self.net.dictionary, dim=-1, eps=1e-8)
            base = torch.as_tensor(alpha, device=self.device) @ dictionary
            pred = self.net.decode(
                base, torch.as_tensor(residual, device=self.device),
                torch.as_tensor(self._normalise("f", features), device=self.device),
                torch.as_tensor(self._normalise("s", states), device=self.device),
            ).cpu().numpy()
        mean, scale = self.normalizers["a"]
        pred = (pred * scale + mean).reshape(len(alpha), *(self.action_shape or ()))
        return pred

    def _sparse_numpy(self, scores: np.ndarray) -> np.ndarray:
        top_k = min(self.top_k, scores.shape[-1])
        ix = np.argpartition(scores, -top_k, axis=-1)[:, -top_k:]
        values = np.take_along_axis(scores, ix, axis=-1)
        values = np.exp(values - values.max(axis=-1, keepdims=True))
        values /= values.sum(axis=-1, keepdims=True)
        alpha = np.zeros_like(scores, dtype=np.float32)
        np.put_along_axis(alpha, ix, values, axis=-1)
        return alpha

    def prior(self, f, s):
        self._require_fit()
        features, states = _np(f), _np(s)
        if features.ndim == 1:
            features = features[None, :]
        if states.ndim == 1:
            states = states[None, :]
        with torch.no_grad():
            out = self.net.prior(torch.cat((
                torch.as_tensor(self._normalise("f", features), device=self.device),
                torch.as_tensor(self._normalise("s", states), device=self.device),
            ), dim=-1))
            logits, residual = out[:, :self.n_atoms], out[:, self.n_atoms:]
            dictionary = F.normalize(self.net.dictionary, dim=-1, eps=1e-8)
            scores = logits.cpu().numpy()
            if dictionary.shape[0] > scores.shape[1]:
                scores = np.pad(scores, ((0, 0), (0, dictionary.shape[0] - scores.shape[1])))
            alpha = self._sparse_numpy(scores)
        return {"alpha": alpha, "probabilities": alpha.copy()}, residual.cpu().numpy()

    def _add_atom(self, innovations: list[np.ndarray]) -> None:
        if self.net.dictionary.shape[0] >= self.max_atoms:
            return
        matrix = np.stack(innovations).astype(np.float32)
        centered = matrix - matrix.mean(axis=0, keepdims=True)
        # A principal direction captures the common novel component in this
        # small residual buffer. Orient it toward the mean for reproducibility.
        if len(matrix) > 1 and np.linalg.norm(centered) > 1e-7:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            atom = vh[0]
            if np.dot(atom, matrix.mean(axis=0)) < 0:
                atom = -atom
        else:
            atom = matrix.mean(axis=0)
        atom = atom / max(np.linalg.norm(atom), 1e-8)
        with torch.no_grad():
            # Preserve fitted atoms bit-for-bit during the non-gradient growth
            # operation; only the newly initialized atom is normalized.
            old = self.net.dictionary.detach().clone()
            new_dictionary = torch.cat((old, torch.as_tensor(atom[None, :], device=self.device)), dim=0)
            self.net.dictionary = nn.Parameter(new_dictionary)
            # Keep the fixed learned prior head unchanged; prior() zero-pads
            # score for newly grown atoms until a later refit.
        self.atom_additions += 1

    def memory_add(self, k, b, meta=None):
        self._require_fit()
        if not isinstance(b, Mapping) or "alpha" not in b:
            raise ValueError("M5 memory_add expects the base mapping returned by encode/prior")
        alpha = self._alpha(b)
        if len(alpha) != 1:
            raise ValueError("memory_add accepts one sample at a time")
        innovation = _np(b.get("innovation", np.zeros((1, self.base_dim), np.float32))).reshape(1, -1)[0]
        reconstruction_error = float(np.linalg.norm(innovation))
        if meta is not None and isinstance(meta, Mapping):
            reconstruction_error = max(reconstruction_error, float(meta.get("reconstruction_error", 0.0)))
        reason = "represented_by_existing_atoms"
        added_atom = False
        if self.variant == "growing" and reconstruction_error > self.novelty_threshold and self.net.dictionary.shape[0] < self.max_atoms:
            self.pending_innovations.append(innovation.copy())
            if len(self.pending_innovations) >= self.novelty_batch:
                self._add_atom(self.pending_innovations[-self.novelty_batch:])
                self.pending_innovations = []
                reason, added_atom = "new_atom_from_residual_pca", True
            else:
                reason = "novel_residual_accumulating"
        item = {"key": _np(k).reshape(-1).copy(), "alpha": alpha[0].copy(), "meta": meta}
        self.memory.append(item)
        if self.variant == "fixed" and reconstruction_error > self.novelty_threshold:
            reason = "fixed_dictionary_novelty_not_added"
        return {"added": True, "added_atom": added_atom, "atom_count": int(self.net.dictionary.shape[0]),
                "reconstruction_error": reconstruction_error, "pending_novelty": len(self.pending_innovations),
                "reason": reason}

    def memory_query(self, k, *, top_k: int = 1):
        self._require_fit()
        n_atoms = self.net.dictionary.shape[0]
        if not self.memory:
            return {"alphas": np.empty((0, n_atoms), np.float32), "weights": np.empty((0,), np.float32)}
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
        alphas = np.stack([self.memory[i]["alpha"] for i in ix])
        if alphas.shape[1] < n_atoms:
            alphas = np.pad(alphas, ((0, 0), (0, n_atoms - alphas.shape[1])))
        return {"alphas": alphas.astype(np.float32), "weights": weights.astype(np.float32)}

    def describe(self) -> dict[str, Any]:
        return {
            "method": "M5: sparse compositional dictionary with residual-triggered atom growth",
            "variant": self.variant,
            "variants": list(self.VARIANTS),
            "base": f"nonnegative top-{self.top_k} coefficients over {int(self.net.dictionary.shape[0]) if self.net else self.n_atoms} normalized {self.base_dim}-D atoms",
            "residual_dim": self.residual_dim,
            "initial_atom_sweep": list(self.INITIAL_ATOM_POINTS),
            "sparsity_sweep_top_k": list(self.TOP_K_POINTS),
            "initial_atoms": self.n_atoms,
            "top_k": self.top_k,
            "decoder_obs": self.decoder_obs,
            "residual_kl": {"beta": self.residual_beta, "sweep": list(self.RESIDUAL_BETA_POINTS),
                            "prior": "standard normal; 0.5 * ||r||^2"},
            "sparsity_penalty": "coefficient entropy",
            "sparsity_weight": self.sparsity_weight,
            "mutual_coherence_penalty": self.coherence_weight,
            "mutual_coherence_weight_sweep": list(self.COHERENCE_WEIGHT_POINTS),
            "dictionary_coherence": (self._coherence_report(self.net.dictionary) if self.net else None),
            "atom_growth": {"enabled": self.variant == "growing", "error_threshold": self.novelty_threshold,
                            "error_quantile": self.novelty_quantile,
                            "threshold_calibrated_from_training_bases": self._fitted and self.novelty_threshold_calibrated,
                            "accumulated_segments": self.novelty_batch,
                            "initializer": "first principal component of pending encoder residuals",
                            "maximum_atoms": self.max_atoms},
            "updates_without_gradients": ("memory addition accumulates residuals and may append one atom using PCA; no optimizer step"
                                          if self.variant == "growing" else "fixed dictionary; no atom or parameter updates during memory_add"),
            "composition": "skills are sparse mixtures of dictionary atoms, so different skills can reuse/share atoms rather than occupy disjoint codes",
            "training_data": "fit consumes only caller-provided object/goal rows; spatial rows are reserved for store/query by the harness",
        }
