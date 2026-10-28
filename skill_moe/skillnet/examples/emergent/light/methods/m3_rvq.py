"""Residual vector quantization (RVQ) skill representation.

The base is a pair of discrete codes: a coarse code quantizes an 8-D action
latent, then a fine code quantizes the remaining error. A 4-D continuous
residual carries per-window detail. ``memory_add`` only stores detached keys
and codes; codebooks are updated by gradients in ``fit``.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ._common import validation_split, update_early_stopping


def _field(data: Any, *names: str):
    if isinstance(data, Mapping):
        for name in names:
            if name in data:
                return data[name]
        raise KeyError(f"Expected one of {names} in training data")
    if isinstance(data, (tuple, list)) and len(data) >= 3:
        return data[{'actions': 0, 'state': 1, 'feature': 2}[names[0]]]
    raise TypeError("train must be a mapping or (actions, state, feature) tuple")


class _MLP(nn.Sequential):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, out_dim))


class ResidualVQ(nn.Module):
    """Two-level residual VQ + continuous residual with the study's duck-typed API.

    Defaults: 16 coarse x 16 fine codes, 8-D code vectors, 4-D continuous
    residual, AdamW lr=1e-3, commitment weight=.25, 30 epochs. 16x16 provides
    256 compositional base types while keeping each stage's codebook small;
    four continuous dimensions preserve local variation without making the
    memory key scene-specific.
    """

    def __init__(
        self,
        *,
        action_horizon: int = 25,
        action_dim: int = 7,
        state_dim: int = 8,
        feature_dim: int = 2048,
        coarse_codes: int = 16,
        fine_codes: int = 16,
        code_dim: int = 8,
        residual_dim: int = 4,
        hidden_dim: int = 128,
        epochs: int = 30,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        commitment_weight: float = 0.25,
        prior_weight: float = 1.0,
        residual_kl_weight: float = 0.1,
        base_use_weight: float = 10.0,
        base_use_margin: float = 0.5,
        decoder_obs: bool = False,
        key_novelty_quantile: float = 0.95,
        base_novelty_quantile: float = 0.95,
        device: str | torch.device = "cpu",
        seed: int = 0,
    ):
        super().__init__()
        if min(action_horizon, action_dim, state_dim, feature_dim, coarse_codes, fine_codes, code_dim, residual_dim) <= 0:
            raise ValueError("All dimensions and codebook sizes must be positive")
        if (residual_kl_weight < 0 or base_use_weight < 0 or base_use_margin < 0
                or not 0 < key_novelty_quantile < 1 or not 0 < base_novelty_quantile < 1):
            raise ValueError("KL weight must be nonnegative and novelty quantiles must be in (0, 1)")
        self.action_horizon, self.action_dim = action_horizon, action_dim
        self.state_dim, self.feature_dim = state_dim, feature_dim
        self.coarse_codes, self.fine_codes = coarse_codes, fine_codes
        self.code_dim, self.residual_dim = code_dim, residual_dim
        self.epochs, self.batch_size = epochs, batch_size
        self.learning_rate, self.commitment_weight = learning_rate, commitment_weight
        self.prior_weight, self.seed = prior_weight, seed
        self.residual_kl_weight = float(residual_kl_weight)
        self.base_use_weight = float(base_use_weight)
        self.base_use_margin = float(base_use_margin)
        self.decoder_obs = bool(decoder_obs)
        self.key_novelty_quantile = float(key_novelty_quantile)
        self.base_novelty_quantile = float(base_novelty_quantile)
        self._device = torch.device(device)
        self.encoder = _MLP(action_horizon * action_dim + state_dim, hidden_dim, code_dim + 2 * residual_dim)
        self.prior_net = _MLP(feature_dim + state_dim, hidden_dim, coarse_codes * fine_codes + residual_dim)
        decoder_input = 2 * code_dim + residual_dim + state_dim + (feature_dim if self.decoder_obs else 0)
        self.decoder = _MLP(decoder_input,
                            hidden_dim, action_horizon * action_dim)
        gen = torch.Generator().manual_seed(seed)
        self.coarse_book = nn.Embedding(coarse_codes, code_dim)
        self.fine_book = nn.Embedding(fine_codes, code_dim)
        with torch.no_grad():
            self.coarse_book.weight.normal_(std=0.2, generator=gen)
            self.fine_book.weight.normal_(std=0.1, generator=gen)
        self.register_buffer("action_mean", None)
        self.register_buffer("action_scale", None)
        self.register_buffer("state_mean", None)
        self.register_buffer("state_scale", None)
        self.register_buffer("feature_mean", None)
        self.register_buffer("feature_scale", None)
        self.memory: list[dict[str, Any]] = []
        self.tau: float | None = None
        self.d_ref: float | None = None
        self.base_novelty_threshold: float | None = None
        self.train_base_codes = np.empty((0, 2), dtype=np.int64)
        self.train_base_features = np.empty((0, 2 * code_dim), dtype=np.float32)
        self.to(self._device)

    def _tensor(self, value, *, dtype=torch.float32):
        return torch.as_tensor(value, dtype=dtype, device=self._device)

    @staticmethod
    def _standardizer(x: torch.Tensor):
        mean = x.mean(dim=0)
        scale = x.std(dim=0, unbiased=False)
        return mean, torch.where(scale < 1e-6, torch.ones_like(scale), scale)

    def _inputs(self, actions, state, feature):
        a, s, f = self._tensor(actions), self._tensor(state), self._tensor(feature)
        if a.ndim != 3 or a.shape[1:] != (self.action_horizon, self.action_dim):
            raise ValueError(f"actions must have shape [N,{self.action_horizon},{self.action_dim}], got {tuple(a.shape)}")
        if s.shape != (len(a), self.state_dim) or f.shape != (len(a), self.feature_dim):
            raise ValueError(f"state/feature shapes must be [N,{self.state_dim}] and [N,{self.feature_dim}]")
        return a, s, f

    def fit(self, train) -> None:
        """Fit encoder/codebooks/prior/decoder on supplied object/goal rows."""
        actions = _field(train, "actions", "A")
        state = _field(train, "state", "s")
        feature = _field(train, "feature", "features", "f")
        a, s, f = self._inputs(actions, state, feature)
        if len(a) < 2:
            raise ValueError("RVQ fitting needs at least two training rows")
        flat_a = a.flatten(1)
        self.action_mean, self.action_scale = self._standardizer(flat_a)
        self.state_mean, self.state_scale = self._standardizer(s)
        self.feature_mean, self.feature_scale = self._standardizer(f)
        an, sn, fn = self._norm_a(flat_a), self._norm_s(s), self._norm_f(f)
        seed_value=int(train.get("seed",self.seed)) if isinstance(train,Mapping) else self.seed
        train_ix,val_ix=validation_split(len(a),seed_value)
        ds = torch.utils.data.TensorDataset(an[train_ix], sn[train_ix], fn[train_ix])
        loader = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=True, generator=torch.Generator().manual_seed(self.seed))
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        self.train()
        val=torch.as_tensor(val_ix,device=self._device); best=float("inf"); stale=0
        patience=int(train.get("early_stopping_patience",3)) if isinstance(train,Mapping) else 3; self.epochs_run=0
        for epoch in range(self.epochs):
            for ab, sb, fb in loader:
                latent = self.encoder(torch.cat([ab, sb], dim=-1))
                z_e, r_mu, r_logvar = self._split_encoder(latent)
                r_logvar = r_logvar.clamp(-8.0, 8.0)
                r = r_mu + torch.randn_like(r_mu) * torch.exp(0.5 * r_logvar)
                coarse_id, q_c = self._quantize(z_e, self.coarse_book)
                fine_target = z_e - q_c.detach()
                fine_id, q_f = self._quantize(fine_target, self.fine_book)
                # Straight-through quantization keeps the decoder loss connected
                # to the action encoder while preserving hard codebook values.
                q_c_st = z_e + (q_c - z_e).detach()
                fine_source = z_e - q_c.detach()
                q_f_st = fine_source + (q_f - fine_source).detach()
                pred = self._decode_norm({"coarse_embedding": q_c_st, "fine_embedding": q_f_st}, r, fb, sb)
                recon = F.mse_loss(pred, ab)
                if self.base_use_weight:
                    # Explicitly prevent a posterior residual from becoming a
                    # bypass around the discrete base. A shuffled code pair
                    # must reconstruct its own row worse than the correct
                    # pair; this is label-free and uses only batch structure.
                    wrong = self._decode_norm({"coarse_embedding": q_c_st.roll(1, 0),
                                               "fine_embedding": q_f_st.roll(1, 0)}, r, fb, sb)
                    true_error = (pred - ab).square().mean(dim=-1)
                    wrong_error = (wrong - ab).square().mean(dim=-1)
                    base_use = F.relu(self.base_use_margin + true_error - wrong_error).mean()
                else:
                    base_use = recon.new_zeros(())
                codebook = F.mse_loss(q_c, z_e.detach()) + F.mse_loss(q_f, fine_target.detach())
                commitment = F.mse_loss(z_e, (q_c + q_f).detach())
                p = self.prior_net(torch.cat([fb, sb], dim=-1))
                base_logits, r_prior = self._split_prior(p)
                base_id = coarse_id * self.fine_codes + fine_id
                prior = F.cross_entropy(base_logits, base_id) + F.mse_loss(r_prior, r_mu.detach())
                residual_kl = 0.5 * (r_mu.square() + r_logvar.exp() - r_logvar - 1.0).sum(dim=-1).mean()
                loss = (recon + codebook + self.commitment_weight * commitment + self.prior_weight * prior
                        + self.residual_kl_weight * residual_kl + self.base_use_weight * base_use)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()
            self.epochs_run=epoch+1
            if len(val):
                with torch.no_grad():
                    latent=self.encoder(torch.cat([an[val],sn[val]],dim=-1)); z_e,r,_=self._split_encoder(latent)
                    _,qc=self._quantize(z_e,self.coarse_book); _,qf=self._quantize(z_e-qc,self.fine_book)
                    vp=self._decode_norm({"coarse_embedding":qc,"fine_embedding":qf},r,fn[val],sn[val])
                    val_loss=F.mse_loss(vp,an[val]).item()
                best,stale,stop=update_early_stopping(val_loss,best,stale,patience=patience)
                if stop: break
        self.eval()
        with torch.no_grad():
            latent = self.encoder(torch.cat([an, sn], dim=-1))
            z_e, _, _ = self._split_encoder(latent)
            c_id, c = self._quantize(z_e, self.coarse_book)
            f_id, _ = self._quantize(z_e - c, self.fine_book)
            codes = torch.stack([c_id, f_id], dim=-1).cpu().numpy().astype(np.int64)
            self.train_base_codes = np.unique(codes, axis=0)
            self.train_base_features = self.base_features(self.train_base_codes)
        if all(name in train for name in ("key", "episode_id", "init_state_id")):
            self.tau, self.d_ref = self._calibrate_key_novelty(
                train["key"], train["episode_id"], train["init_state_id"], self.key_novelty_quantile
            )
        else:
            # The representation API can be fit without memory metadata. In
            # that case additions remain conservative only when calibrated
            # metadata is supplied; do not silently invent a threshold.
            self.tau, self.d_ref = None, None
        self.base_novelty_threshold = self._calibrate_base_novelty(self.train_base_features, self.base_novelty_quantile)

    def _norm_a(self, a):
        return (a - self.action_mean) / self.action_scale

    def _norm_s(self, s):
        return (s - self.state_mean) / self.state_scale

    def _norm_f(self, f):
        return (f - self.feature_mean) / self.feature_scale

    def _quantize(self, z: torch.Tensor, book: nn.Embedding):
        dist = z.square().sum(-1, keepdim=True) - 2 * z @ book.weight.T + book.weight.square().sum(-1)[None, :]
        ids = dist.argmin(dim=-1)
        return ids, book(ids)

    def _split_prior(self, p):
        base, r = torch.split(p, [self.coarse_codes * self.fine_codes, self.residual_dim], dim=-1)
        return base, r

    def _split_encoder(self, latent):
        return torch.split(latent, [self.code_dim, self.residual_dim, self.residual_dim], dim=-1)

    @staticmethod
    def _calibrate_key_novelty(keys, episode_ids, init_state_ids, quantile):
        keys = np.asarray(keys, dtype=np.float32)
        episodes, initials = np.asarray(episode_ids).astype(str), np.asarray(init_state_ids).astype(str)
        if len(keys) < 2:
            return None, None
        keys = keys / np.maximum(np.linalg.norm(keys, axis=1, keepdims=True), 1e-8)
        if len(keys) > 4096:
            rng = np.random.default_rng(7081)
            ix = np.sort(rng.choice(len(keys), size=4096, replace=False))
            keys, episodes, initials = keys[ix], episodes[ix], initials[ix]
        nearest = np.full(len(keys), np.inf, dtype=np.float64)
        for lo in range(0, len(keys), 256):
            hi = min(len(keys), lo + 256)
            sim = keys[lo:hi] @ keys.T
            excluded = ((episodes[lo:hi, None] == episodes[None, :]) |
                        (initials[lo:hi, None] == initials[None, :]))
            sim[excluded] = -np.inf
            nearest[lo:hi] = 1.0 - sim.max(axis=1)
        finite = nearest[np.isfinite(nearest)]
        if not len(finite):
            return None, None
        return float(np.quantile(finite, quantile)), float(np.median(finite))

    @staticmethod
    def _calibrate_base_novelty(unit_features, quantile):
        values = np.asarray(unit_features, dtype=np.float32).reshape(len(unit_features), -1)
        if len(values) < 2:
            return None
        dist = np.linalg.norm(values[:, None, :] - values[None, :, :], axis=-1)
        np.fill_diagonal(dist, np.inf)
        nearest = dist.min(axis=1)
        finite = nearest[np.isfinite(nearest)]
        if not len(finite):
            return None
        threshold = float(np.quantile(finite, quantile))
        return threshold if threshold > 1e-8 else None

    def encode(self, A, s):
        codes, residual = self.encode_trainable(A, s)
        return codes.detach().cpu().numpy().astype(np.int64), residual.detach().cpu().numpy().astype(np.float32)

    def encode_trainable(self, A, s):
        """Differentiable posterior path for training-time pressures such as S."""
        a, state = self._tensor(A), self._tensor(s)
        flat = a.reshape(len(a), -1)
        latent = self.encoder(torch.cat([self._norm_a(flat), self._norm_s(state)], dim=-1))
        z, r, _ = self._split_encoder(latent)
        c_id, c = self._quantize(z, self.coarse_book)
        f_id, f = self._quantize(z - c, self.fine_book)
        return torch.stack([c_id, f_id], dim=-1), r

    def _embeddings(self, b):
        if isinstance(b, Mapping):  # Accept explicit distributions for direct API callers.
            if "coarse_probs" in b:
                c = self._tensor(b["coarse_probs"]) @ self.coarse_book.weight
            elif "coarse_embedding" in b:
                c = b["coarse_embedding"]
            else:
                c = self.coarse_book(self._tensor(b["coarse_id"], dtype=torch.long).reshape(-1))
            if "fine_probs" in b:
                f = self._tensor(b["fine_probs"]) @ self.fine_book.weight
            elif "fine_embedding" in b:
                f = b["fine_embedding"]
            elif b.get("fine_id") is not None:
                f = self.fine_book(self._tensor(b["fine_id"], dtype=torch.long).reshape(-1))
            else:
                f = torch.zeros_like(c)
            return c, f
        base = self._tensor(b)
        if base.shape[-1] == 2 * self.code_dim and base.dtype != torch.long:
            return base[..., :self.code_dim], base[..., self.code_dim:]
        if base.shape[-1] == self.coarse_codes + self.fine_codes and base.dtype != torch.long:
            return base[..., :self.coarse_codes] @ self.coarse_book.weight, base[..., self.coarse_codes:] @ self.fine_book.weight
        ids = base.to(torch.long).reshape(-1, 2)
        c = self.coarse_book(ids[:, 0])
        fine_ids = ids[:, 1]
        f = torch.zeros_like(c)
        active = fine_ids >= 0
        if bool(active.any()):
            f[active] = self.fine_book(fine_ids[active])
        return c, f

    def _decode_norm(self, b, r, f, s):
        c, fine = self._embeddings(b)
        parts = [c, fine, self._tensor(r)]
        if self.decoder_obs:
            parts.append(self._norm_f(self._tensor(f)))
        parts.append(self._norm_s(self._tensor(s)))
        return self.decoder(torch.cat(parts, dim=-1))

    def decode_trainable(self, b, r, f, s):
        raw_a, raw_s, raw_f = self._tensor(f), self._tensor(s), self._tensor(r)
        pred = self._decode_norm(b, raw_f, raw_a, raw_s)
        return ((pred * self.action_scale + self.action_mean)
                .reshape(-1, self.action_horizon, self.action_dim))

    def decode(self, b, r, f, s):
        with torch.no_grad():
            return self.decode_trainable(b, r, f, s).cpu().numpy().astype(np.float32)

    def prior(self, f, s):
        fn, sn = self._norm_f(self._tensor(f)), self._norm_s(self._tensor(s))
        logits, r = self._split_prior(self.prior_net(torch.cat([fn, sn], dim=-1)))
        joint = logits.softmax(-1)
        joint_grid = joint.reshape(len(joint), self.coarse_codes, self.fine_codes)
        coarse_probs = joint_grid.sum(dim=-1)
        fine_probs = joint_grid.sum(dim=-2)
        distribution = {"base_probs": joint.detach().cpu().numpy(),
                        "coarse_probs": coarse_probs.detach().cpu().numpy(),
                        "fine_probs": fine_probs.detach().cpu().numpy(),
                        "base_logits": logits.detach().cpu().numpy()}
        return distribution, r.detach().cpu().numpy().astype(np.float32)

    def prior_base(self, base_distribution):
        """Decode the joint prior's MAP coarse/fine pair as a valid RVQ unit."""
        if "base_probs" in base_distribution:
            ids = self._tensor(base_distribution["base_probs"]).argmax(dim=-1)
            c_ids = torch.div(ids, self.fine_codes, rounding_mode="floor")
            f_ids = ids.remainder(self.fine_codes)
            return torch.cat([self.coarse_book(c_ids), self.fine_book(f_ids)], dim=-1).detach().cpu().numpy().astype(np.float32)
        c = self._tensor(base_distribution["coarse_probs"]).argmax(dim=-1)
        fine = self._tensor(base_distribution["fine_probs"]).argmax(dim=-1)
        return torch.cat([self.coarse_book(c), self.fine_book(fine)], dim=-1).detach().cpu().numpy().astype(np.float32)

    def base_features(self, b):
        base = np.asarray(b)
        if np.issubdtype(base.dtype, np.integer):
            with torch.no_grad():
                c, f = self._embeddings(base)
                return torch.cat([c, f], dim=-1).cpu().numpy().astype(np.float32)
        if base.shape[-1] == 2 * self.code_dim:
            return base.astype(np.float32)
        raise ValueError(f"Expected paired code IDs or {2 * self.code_dim}-D base embeddings, got {base.shape}")

    def blend_base(self, candidates, weights):
        return np.asarray(weights, dtype=np.float32) @ self.base_features(candidates)

    @staticmethod
    def _numpy(value):
        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def memory_add(self, k, b, meta):
        try:
            key = self._numpy(k).astype(np.float32).reshape(-1)
            if isinstance(b, Mapping):
                c = int(self._numpy(b["coarse_id"]).reshape(-1)[0])
                fine = b.get("fine_id")
                f = -1 if fine is None else int(self._numpy(fine).reshape(-1)[0])
            else:
                codes = self._numpy(b).reshape(-1, 2)[0]
                c, f = int(codes[0]), int(codes[1])
            if not np.all(np.isfinite(key)) or not (0 <= c < self.coarse_codes) or not (-1 <= f < self.fine_codes):
                return False, "invalid_key_or_code"
            metadata = dict(meta or {})
            key_unit = key / max(float(np.linalg.norm(key)), 1e-8)
            base_feature = self.base_features(np.asarray([[c, f]], dtype=np.int64))[0]
            is_seed = bool(metadata.get("seed", False))

            key_distance = float("inf")
            eligible = [entry for entry in self.memory if (
                entry["meta"].get("episode_id") != metadata.get("episode_id")
                and entry["meta"].get("init_state_id") != metadata.get("init_state_id"))]
            if eligible:
                stored_keys = np.stack([entry["key"] for entry in eligible]).astype(np.float32)
                stored_keys /= np.maximum(np.linalg.norm(stored_keys, axis=1, keepdims=True), 1e-8)
                key_distance = float(1.0 - np.max(stored_keys @ key_unit))
            key_novel = is_seed or (self.tau is not None and key_distance > self.tau)

            known_units = self.train_base_features
            added_units = [self.base_features(np.asarray([[entry["coarse_id"], entry["fine_id"]]], dtype=np.int64))[0]
                           for entry in self.memory if entry.get("is_new_unit", False)]
            if added_units:
                known_units = np.concatenate([known_units, np.stack(added_units)], axis=0)
            base_distance = float(np.linalg.norm(known_units - base_feature[None, :], axis=1).min()) if len(known_units) else float("inf")
            unit_novel = (not is_seed and self.base_novelty_threshold is not None
                          and base_distance > self.base_novelty_threshold)

            # A novel context adds a key/base memory item; a novel motor unit
            # can also enter the library even when its scene key is familiar.
            if not (key_novel or unit_novel):
                return False, "known_key_and_base"
            self.memory.append({"key": key.copy(), "coarse_id": c, "fine_id": f,
                                "meta": metadata, "is_new_unit": bool(unit_novel),
                                "key_distance": key_distance, "base_distance": base_distance})
            if is_seed:
                return True, "seed"
            if unit_novel and key_novel:
                return True, "added_new_key_and_unit"
            if unit_novel:
                return True, "added_new_unit"
            return True, "added_new_key_existing_unit"
        except (KeyError, TypeError, ValueError) as exc:
            return False, f"invalid key/base: {exc}"

    def memory_query(self, k, meta=None, *, top_k: int = 5, level: str = "both", temperature: float = 0.1):
        """Retrieve full (coarse+fine) bases or group retrieval by coarse code only."""
        if level not in {"both", "full", "coarse"}:
            raise ValueError("level must be 'coarse' or 'both'")
        if not self.memory:
            return []
        query = self._numpy(k).astype(np.float32).reshape(-1)
        eligible = [m for m in self.memory if meta is None or (
            m["meta"].get("episode_id") != meta.get("episode_id")
            and m["meta"].get("init_state_id") != meta.get("init_state_id"))]
        if not eligible:
            return np.empty((0, 2), np.int64), np.empty((0,), np.float32), []
        keys = np.stack([m["key"] for m in eligible])
        if query.shape[-1] != keys.shape[-1]:
            raise ValueError(f"query dimension {query.shape[-1]} != stored key dimension {keys.shape[-1]}")
        qn = query / max(float(np.linalg.norm(query)), 1e-12)
        kn = keys / np.maximum(np.linalg.norm(keys, axis=-1, keepdims=True), 1e-12)
        similarity = kn @ qn
        logits = similarity / max(float(temperature), 1e-6)
        weights = np.exp(logits - logits.max()); weights /= weights.sum()
        if level == "coarse":
            grouped: dict[int, float] = {}
            for entry, weight in zip(eligible, weights):
                code = entry["coarse_id"]
                grouped[code] = grouped.get(code, 0.0) + float(weight)
            items = sorted(grouped.items(), key=lambda x: -x[1])[:top_k]
            candidates = np.asarray([[code, -1] for code, _ in items], dtype=np.int64)
            grouped_weights = np.asarray([weight for _, weight in items], dtype=np.float32)
            grouped_weights /= max(float(grouped_weights.sum()), 1e-12)
            return candidates, grouped_weights, []
        order = np.argsort(-weights)[:top_k]
        candidates = np.asarray([[eligible[i]["coarse_id"], eligible[i]["fine_id"]] for i in order], dtype=np.int64)
        selected_weights = weights[order].astype(np.float32)
        selected_weights /= max(float(selected_weights.sum()), 1e-12)
        chosen = [eligible[i] for i in order]
        return candidates, selected_weights, chosen

    def describe(self):
        return {"method": "M3 residual VQ", "coarse_codes": self.coarse_codes, "fine_codes": self.fine_codes,
                "compositional_base_capacity": self.coarse_codes * self.fine_codes,
                "code_dim_each": self.code_dim, "continuous_residual_dim": self.residual_dim,
                "decoder_obs": self.decoder_obs,
                "residual_kl_weight": self.residual_kl_weight,
                "base_use_margin_weight": self.base_use_weight,
                "base_use_margin": self.base_use_margin,
                "residual_prior": "standard normal diagonal Gaussian",
                "prior_base": "MAP valid joint coarse/fine code pair (avoids decoding an out-of-distribution soft embedding mean)",
                "key_novelty_threshold": {"metric": "cosine distance", "quantile": self.key_novelty_quantile, "value": self.tau},
                "base_novelty_threshold": {"metric": "Euclidean distance in concatenated codebook embedding", "quantile": self.base_novelty_quantile, "value": self.base_novelty_threshold},
                "fit_updates": "encoder, codebooks, joint prior and decoder by AdamW gradients",
                "memory_add_updates": "detached key/base records; new base units are flagged separately; no gradient or codebook update"}


class M3(ResidualVQ):
    """Named study variant with the default 16 x 16 residual codebooks."""


def build_variants():
    return {"M3_rvq16x16": M3(residual_kl_weight=0.1),
            "M3_rvq16x16_beta1.0": M3(residual_kl_weight=1.0)}

