"""M6: open-ended functional skill units with soft memory blending.

The encoder/prior/decoder are reused from M1 by default (or M4 for an
encoder ablation). Skill count is not configured: units are created only when
an unsupported functional region accumulates enough segments and episodes.
"""
from __future__ import annotations

from collections.abc import Mapping
import copy
import hashlib
from typing import Any

import numpy as np

from .m1 import M1
from .m4 import M4


_BACKBONE_CACHE: dict[str, tuple[Any, np.ndarray, np.ndarray]] = {}
_FUNCTIONAL_EMBEDDING_CACHES: dict[str, dict[bytes, np.ndarray]] = {}


def _training_cache_key(actions, states, features, episodes, settings) -> str:
    digest = hashlib.sha256(repr(settings).encode())
    for value in (actions, states, features, np.asarray(episodes).astype(str)):
        array = np.ascontiguousarray(value)
        digest.update(str(array.shape).encode())
        digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _field(data: Mapping[str, Any], names: tuple[str, ...], *, required: bool = True):
    for name in names:
        if name in data:
            return data[name]
    if required:
        raise KeyError(f"training data need one of {names}")
    return None


def _array(value: Any, *, dtype=np.float32) -> np.ndarray:
    return np.asarray(value, dtype=dtype)


class M6:
    """Growing functional regions; all retrievals are weighted blends."""

    KAPPA_POINTS = (0.05, 0.1, 0.2)
    MIN_SUPPORT_POINTS = (5, 10, 20)
    MIN_EPISODES = 3
    PENDING_BATCH_SIZE = 200
    PROBE_SIZE = 64
    RANDOM_DISTANCE_PAIRS = 64

    def __init__(self, *, independence_weight: float = 0.01, kappa: float = 0.1,
                 min_support: int = 10, min_episodes: int = 3, encoder: str = "m1",
                 width: int = 128, epochs: int = 10, batch_size: int = 128,
                 pending_batch_size: int = 200, pending_max_age: int = 1000,
                 update_centers_on_add: bool = False, topk: int = 5,
                 temperature: float = 0.5, seed: int = 0, device: str = "cpu",
                 base_dim: int = 12, residual_dim: int = 4):
        if independence_weight not in (0.01, 0.1):
            raise ValueError("M6 reuses the M1 encoder with independence_weight 0.01 or 0.1")
        if kappa <= 0 or min_support < 1 or min_episodes < 1:
            raise ValueError("kappa and support thresholds must be positive")
        if encoder not in ("m1", "m4"):
            raise ValueError("encoder must be 'm1' or 'm4'")
        if pending_batch_size < 1 or pending_max_age < 1 or topk < 1 or temperature <= 0:
            raise ValueError("invalid pending or retrieval settings")
        self.independence_weight = float(independence_weight)
        if int(base_dim) < 1 or int(residual_dim) < 0:
            raise ValueError("base_dim must be positive and residual_dim non-negative")
        self.requested_base_dim, self.requested_residual_dim = int(base_dim), int(residual_dim)
        self.kappa, self.min_support, self.min_episodes = float(kappa), int(min_support), int(min_episodes)
        self.encoder_kind, self.width, self.epochs, self.batch_size = encoder, int(width), int(epochs), int(batch_size)
        self.pending_batch_size, self.pending_max_age = int(pending_batch_size), int(pending_max_age)
        self.update_centers_on_add = bool(update_centers_on_add)
        self.topk, self.temperature, self.seed, self.device = int(topk), float(temperature), int(seed), device

        self.backbone: M1 | M4 | None = None
        self.memory: list[dict[str, Any]] = []
        self._memory_index_cache: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._memory_index_keys: np.ndarray | None = None
        self._memory_index_episodes: np.ndarray | None = None
        self._memory_index_initials: np.ndarray | None = None
        self._memory_index_size = 0
        self._memory_index_capacity = 0
        self._memory_index_list_id: int | None = None
        self._centers: list[np.ndarray] = []
        self.prototypes = np.empty((0, 12), np.float32)
        self.support_counts: list[int] = []
        self.episode_sets: list[set[str]] = []
        self.pending: list[dict[str, Any]] = []
        self.pending_age: dict[int, int] = {}
        self._pending_since_cluster = 0
        self._next_pending_id = 0
        self._last_pending_id: int | None = None
        self._data_seen = 0
        self._key_tau: float | None = None
        self._key_d_ref: float | None = None
        self._functional_probe: dict[str, np.ndarray] | None = None
        self._functional_embedding_cache: dict[bytes, np.ndarray] = {}
        self.functional_median: float | None = None
        self.epsilon: float | None = None
        self.unit_count_history: list[dict[str, int]] = []
        # Row-aligned audit trail for units registered from the pending buffer.
        # The harness consumes these events to distinguish a registered skill
        # from an item that was merely admitted to episodic memory.
        self.unit_registration_events: list[dict[str, Any]] = []
        self.registered_count = 0
        self.merged_count = 0
        self.pending_expired_count = 0
        self.memory_added_count = 0
        self.memory_rejected_count = 0
        self.action_shape: tuple[int, ...] | None = None
        self.base_dim = 12
        self.residual_dim = 4
        self._fitted = False

    def _require_fit(self):
        if not self._fitted:
            raise RuntimeError("call fit(train) before using this method")

    @staticmethod
    def _backbone_base(base) -> np.ndarray:
        if isinstance(base, Mapping):
            for key in ("embedding", "base"):
                if key in base:
                    return _array(base[key])
            if "probs" in base and "base" in base:
                return _array(base["base"])
        return _array(base)

    def _decode(self, base: np.ndarray, residual: np.ndarray, feature: np.ndarray, state: np.ndarray) -> np.ndarray:
        return self.backbone.decode(base, residual, feature, state)

    def _functional_distance(self, left: np.ndarray, right: np.ndarray) -> float:
        delta = self._functional_embedding(left) - self._functional_embedding(right)
        value = float(np.dot(delta, delta))
        return value if np.isfinite(value) else float("inf")

    def _functional_embedding(self, base: np.ndarray) -> np.ndarray:
        """Cached decoder signature whose squared L2 distance is probe MSE."""
        value = _array(base).reshape(-1)
        key = np.ascontiguousarray(value).tobytes()
        cached = self._functional_embedding_cache.get(key)
        if cached is not None:
            return cached
        probe = self._functional_probe
        n_probe = len(probe["state"])
        repeated = np.repeat(value[None, :], n_probe, axis=0)
        decoded = np.asarray(self._decode(repeated, probe["residual"],
                                          probe["feature"], probe["state"]), dtype=np.float32)
        embedding = (decoded.reshape(-1) / np.sqrt(max(decoded.size, 1))).astype(np.float32)
        self._functional_embedding_cache[key] = embedding
        return embedding

    def _functional_distances(self, base: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        """Batch the fixed-probe decodes for the ten L2-nearest candidates."""
        if not len(candidates):
            return np.empty((0,), np.float64)
        left = self._functional_embedding(base)
        right = np.stack([self._functional_embedding(candidate) for candidate in candidates])
        delta = right - left[None, :]
        return np.sum(np.square(delta, dtype=np.float64), axis=1)

    def _nearest(self, base: np.ndarray, centers: list[np.ndarray] | None = None):
        centers = self._centers if centers is None else centers
        if not centers:
            return None, float("inf")
        matrix = np.asarray(centers, dtype=np.float32)
        l2 = np.linalg.norm(matrix - base[None, :], axis=-1)
        ix = np.argsort(l2, kind="stable")[:min(10, len(l2))]
        distances = self._functional_distances(base, matrix[ix])
        local = min(range(len(ix)), key=lambda i: (distances[i], int(ix[i])))
        return int(ix[local]), float(distances[local])

    def _sync_prototypes(self):
        dim = self.base_dim
        self.prototypes = (np.asarray(self._centers, dtype=np.float32).reshape(-1, dim)
                           if self._centers else np.empty((0, dim), np.float32))

    def _record_units(self):
        self._sync_prototypes()
        self.unit_count_history.append({"data_seen": int(self._data_seen), "unit_count": len(self._centers),
                                        "pending_count": len(self.pending), "registered_total": self.registered_count,
                                        "merged_total": self.merged_count,
                                        "expired_total": self.pending_expired_count})

    def _assign(self, base: np.ndarray, episode_id: str, *, training: bool) -> tuple[int | None, str, float]:
        unit_id, distance = self._nearest(base)
        self._last_pending_id = None
        if unit_id is not None and distance <= self.epsilon:
            old_count = self.support_counts[unit_id]
            new_count = old_count + 1
            if training or self.update_centers_on_add:
                self._centers[unit_id] = (self._centers[unit_id] * old_count + base) / new_count
            self.support_counts[unit_id] = new_count
            self.episode_sets[unit_id].add(episode_id)
            self._sync_prototypes()
            return unit_id, "assigned_existing_unit", distance
        pending_id = self._next_pending_id
        self._next_pending_id += 1
        self.pending.append({"id": pending_id, "base": base.copy(), "episode_id": episode_id})
        self._last_pending_id = pending_id
        self.pending_age[pending_id] = 0
        self._pending_since_cluster += 1
        return None, "pending_functionally_novel", distance

    def _cluster_pending(self, *, force: bool = False) -> dict[str, int]:
        self._require_fit()
        if not self.pending or (not force and self._pending_since_cluster < self.pending_batch_size):
            return {"registered": 0, "merged": 0, "pending": len(self.pending), "expired": 0}
        self._pending_since_cluster = 0
        # DP-means has no prescribed K. The functional radius sets the split.
        clusters: list[dict[str, Any]] = []
        for row in self.pending:
            base = row["base"]
            if not clusters:
                clusters.append({"center": base.copy(), "bases": [base], "episodes": {row["episode_id"]},
                                 "rows": [row]})
                continue
            centers = [cluster["center"] for cluster in clusters]
            nearest, distance = self._nearest(base, centers)
            if nearest is not None and distance <= self.epsilon:
                cluster = clusters[nearest]
                cluster["bases"].append(base)
                cluster["center"] = np.mean(cluster["bases"], axis=0).astype(np.float32)
                cluster["episodes"].add(row["episode_id"])
                cluster["rows"].append(row)
            else:
                clusters.append({"center": base.copy(), "bases": [base], "episodes": {row["episode_id"]},
                                 "rows": [row]})

        registered_ids: dict[int, int] = {}
        registered = merged = 0
        for cluster in clusters:
            count, episode_count = len(cluster["rows"]), len(cluster["episodes"])
            if count < self.min_support or episode_count < self.min_episodes:
                continue
            center = np.asarray(cluster["center"], dtype=np.float32)
            existing, distance = self._nearest(center)
            if existing is not None and distance <= self.epsilon:
                if getattr(self, "_fit_in_progress", False) or self.update_centers_on_add:
                    old_count = self.support_counts[existing]
                    total = old_count + count
                    self._centers[existing] = (self._centers[existing] * old_count + center * count) / total
                self.support_counts[existing] += count
                self.episode_sets[existing].update(cluster["episodes"])
                merged += 1
                self.merged_count += 1
            else:
                self._centers.append(center.copy())
                self.support_counts.append(count)
                self.episode_sets.append(set(cluster["episodes"]))
                existing = len(self._centers) - 1
                registered += 1
                self.registered_count += 1
                for row in cluster["rows"]:
                    event = {"unit_id": int(existing), "pending_id": int(row["id"]),
                             "base": np.asarray(row["base"], dtype=np.float32).copy(),
                             "skill_unit_added": True}
                    if row.get("meta"):
                        event.update(row["meta"])
                    self.unit_registration_events.append(event)
            registered_ids.update({row["id"]: int(existing) for row in cluster["rows"]})

        retained = []
        expired = 0
        for row in self.pending:
            if row["id"] in registered_ids:
                memory_index = row.get("memory_index")
                if memory_index is not None and memory_index < len(self.memory):
                    self.memory[memory_index]["unit_id"] = registered_ids[row["id"]]
                self.pending_age.pop(row["id"], None)
                continue
            pid = row["id"]
            self.pending_age[pid] = self.pending_age.get(pid, 0) + self.pending_batch_size
            if self.pending_age[pid] >= self.pending_max_age:
                self.pending_age.pop(pid, None)
                expired += 1
            else:
                retained.append(row)
        self.pending = retained
        self.pending_expired_count += expired
        self._sync_prototypes()
        self._record_units()
        return {"registered": registered, "merged": merged, "pending": len(self.pending), "expired": expired}

    def fit(self, train) -> None:
        """Fit M1/M4 and stream training bases in a fixed seeded order."""
        actions = _array(_field(train, ("actions", "A", "action")))
        states = _array(_field(train, ("state", "s", "proprio")))
        features = _array(_field(train, ("feature", "features", "f")))
        n = len(actions)
        if len(states) != n or len(features) != n:
            raise ValueError("actions, state and feature lengths must match")
        keys_value = _field(train, ("key", "keys"), required=False)
        keys = np.zeros((n, 1), np.float32) if keys_value is None else _array(keys_value).reshape(n, -1)
        episodes_value = _field(train, ("episode_id", "episode_ids"), required=False)
        episodes = np.asarray([f"row-{i}" for i in range(n)] if episodes_value is None else episodes_value).astype(str)
        initial_value = _field(train, ("init_state_id", "initial_state_id"), required=False)
        initials = np.asarray([f"init-{i}" for i in range(n)] if initial_value is None else initial_value).astype(str)
        seed_value = int(train.get("seed", self.seed)) if isinstance(train, Mapping) else self.seed
        self.action_shape = tuple(actions.shape[1:])
        self.base_dim = 12
        self._centers, self.support_counts, self.episode_sets = [], [], []
        self._functional_embedding_cache = {}
        self.pending, self.pending_age, self.memory, self.unit_count_history = [], {}, [], []
        self._memory_index_cache = None
        self._memory_index_keys = self._memory_index_episodes = self._memory_index_initials = None
        self._memory_index_size = self._memory_index_capacity = 0
        self._memory_index_list_id = None
        self._pending_since_cluster = self._next_pending_id = self._data_seen = 0
        self._last_pending_id = None
        self.registered_count = self.merged_count = self.pending_expired_count = 0
        self.unit_registration_events = []
        if self.encoder_kind == "m1":
            self.backbone = M1(independence_weight=self.independence_weight, width=self.width,
                               epochs=self.epochs, batch_size=self.batch_size,
                               seed_value=seed_value, device=self.device,
                               base_dim=self.requested_base_dim, residual_dim=self.requested_residual_dim)
        else:
            self.backbone = M4(epochs=self.epochs, batch_size=self.batch_size, width=self.width,
                               seed=seed_value, device=self.device)
        backbone_train = {"actions": actions, "state": states, "feature": features, "key": keys,
                          "episode_id": episodes, "init_state_id": initials, "seed": seed_value,
                          "epochs": self.epochs, "batch_size": self.batch_size}
        cache_key = _training_cache_key(actions, states, features, episodes,
            (self.encoder_kind, self.independence_weight, self.width, self.epochs,
             self.batch_size, self.requested_base_dim, self.requested_residual_dim, seed_value))
        cached = _BACKBONE_CACHE.get(cache_key)
        if cached is None:
            self.backbone.fit(backbone_train)
            base, _ = self.backbone.encode(actions, states)
            base = self._backbone_base(base)
            _, prior_residual = self.backbone.prior(features, states)
            _BACKBONE_CACHE[cache_key] = (copy.deepcopy(self.backbone), np.asarray(base).copy(),
                                          np.asarray(prior_residual).copy())
        else:
            cached_backbone, cached_base, cached_prior_residual = cached
            self.backbone = copy.deepcopy(cached_backbone)
            base = np.asarray(cached_base).copy()
            prior_residual = np.asarray(cached_prior_residual).copy()
        self.base_dim = int(base.shape[-1])
        self.residual_dim = int(prior_residual.shape[-1])
        rng = np.random.default_rng(seed_value + 6046)
        probe_ix = rng.choice(n, size=self.PROBE_SIZE, replace=n < self.PROBE_SIZE)
        self._functional_probe = {"residual": _array(prior_residual)[probe_ix], "state": states[probe_ix],
                                  "feature": features[probe_ix]}
        probe_digest = hashlib.sha256()
        for value in (self._functional_probe["residual"], self._functional_probe["state"],
                      self._functional_probe["feature"]):
            probe_digest.update(np.ascontiguousarray(value).view(np.uint8))
        functional_cache_key = f"{cache_key}:{probe_digest.hexdigest()}"
        self._functional_embedding_cache = _FUNCTIONAL_EMBEDDING_CACHES.setdefault(
            functional_cache_key, {})
        pair_ix = rng.integers(0, n, size=(self.RANDOM_DISTANCE_PAIRS, 2))
        random_distances = [self._functional_distance(base[i], base[j]) for i, j in pair_ix]
        self.functional_median = float(np.median(random_distances))
        self.epsilon = max(self.kappa * self.functional_median, 1e-12)
        self._key_tau, self._key_d_ref = None, None
        try:
            from ._common import calibrate
            self._key_tau, self._key_d_ref = calibrate(keys, episodes, initials, quantile=0.95)
        except (ValueError, IndexError):
            pass
        self._fitted = True
        self._fit_in_progress = True
        order = rng.permutation(n)
        for index in order:
            self._data_seen += 1
            self._assign(base[index], episodes[index], training=True)
            self._cluster_pending()
        self._cluster_pending(force=True)
        self._fit_in_progress = False
        self._record_units()

    def encode(self, A, s):
        self._require_fit()
        base, residual = self.backbone.encode(A, s)
        return self._backbone_base(base).astype(np.float32), _array(residual).astype(np.float32)

    def decode(self, b, r, f, s):
        self._require_fit()
        base, residual, feature, state = map(_array, (self._backbone_base(b), r, f, s))
        if base.ndim == 1:
            base = base[None, :]
        if residual.ndim == 1:
            residual = residual[None, :]
        if feature.ndim == 1:
            feature = feature[None, :]
        if state.ndim == 1:
            state = state[None, :]
        return self._decode(base, residual, feature, state)

    def prior(self, f, s):
        self._require_fit()
        base, residual = self.backbone.prior(f, s)
        return self._backbone_base(base).astype(np.float32), _array(residual).astype(np.float32)

    def _new_episode(self, meta: Mapping[str, Any] | None) -> str:
        meta = meta or {}
        return str(meta.get("episode_id", f"memory-row-{self._data_seen}"))

    def _memory_index(self):
        """Return a cached, incrementally grown array index for episodic keys."""
        count = len(self.memory)
        rebuild = (self._memory_index_list_id != id(self.memory)
                   or self._memory_index_size > count
                   or self._memory_index_keys is None)
        if rebuild:
            dim = int(np.asarray(self.memory[0]["key"]).size) if count else 0
            capacity = max(16, count * 2)
            self._memory_index_keys = np.empty((capacity, dim), dtype=np.float32)
            self._memory_index_episodes = np.empty((capacity,), dtype=object)
            self._memory_index_initials = np.empty((capacity,), dtype=object)
            self._memory_index_size = 0
            self._memory_index_capacity = capacity
            self._memory_index_list_id = id(self.memory)
        if count > self._memory_index_capacity:
            capacity = max(count, self._memory_index_capacity * 2)
            keys = np.empty((capacity, self._memory_index_keys.shape[1]), dtype=np.float32)
            episodes = np.empty((capacity,), dtype=object)
            initials = np.empty((capacity,), dtype=object)
            keys[:self._memory_index_size] = self._memory_index_keys[:self._memory_index_size]
            episodes[:self._memory_index_size] = self._memory_index_episodes[:self._memory_index_size]
            initials[:self._memory_index_size] = self._memory_index_initials[:self._memory_index_size]
            self._memory_index_keys, self._memory_index_episodes, self._memory_index_initials = keys, episodes, initials
            self._memory_index_capacity = capacity
        while self._memory_index_size < count:
            i = self._memory_index_size
            item = self.memory[i]
            self._memory_index_keys[i] = np.asarray(item["key"], dtype=np.float32).reshape(-1)
            meta = item.get("meta") or {}
            self._memory_index_episodes[i] = meta.get("episode_id")
            self._memory_index_initials[i] = meta.get("init_state_id")
            self._memory_index_size += 1
        self._memory_index_cache = (self._memory_index_keys[:count],
                                    self._memory_index_episodes[:count],
                                    self._memory_index_initials[:count])
        return self._memory_index_cache

    def memory_add(self, k, b, meta=None):
        self._require_fit()
        key = _array(k).reshape(-1)
        base = self._backbone_base(b).reshape(-1)
        if not np.all(np.isfinite(key)) or not np.all(np.isfinite(base)):
            return {"added": False, "reason": "nonfinite", "unit_id": None,
                    "assignment": "rejected_nonfinite", "pending_count": len(self.pending)}
        metadata = dict(meta or {})
        episode_id = self._new_episode(metadata)
        self._data_seen += 1
        if metadata.get("seed", False):
            unit_id, distance = self._nearest(base)
            assignment = "seed_existing_unit" if unit_id is not None and distance <= self.epsilon else "seed_unassigned"
        else:
            unit_id, assignment, distance = self._assign(base, episode_id, training=False)
            if self._last_pending_id is not None:
                for row in self.pending:
                    if row["id"] == self._last_pending_id:
                        row["meta"] = metadata.copy()
                        break

        if metadata.get("seed", False) or not self.memory:
            accepted, reason = True, "seed" if metadata.get("seed", False) else "added_empty"
        else:
            stored_keys, episodes, initials = self._memory_index()
            eligible = ((episodes != metadata.get("episode_id"))
                        & (initials != metadata.get("init_state_id")))
            if eligible.any():
                delta = stored_keys[eligible] - key[None, :]
                distance_key = float(np.sqrt(np.maximum((delta * delta).sum(axis=1), 0.0).min()))
            else:
                distance_key = float("inf")
            if self._key_tau is not None and distance_key <= self._key_tau:
                accepted, reason = False, "near_existing_key"
            else:
                accepted, reason = True, "added_novel_key"
        if accepted:
            self.memory.append({"key": key.copy(), "base": base.copy(), "unit_id": unit_id,
                                "meta": metadata, "assignment": assignment})
            self._memory_index()
            if self._last_pending_id is not None:
                for row in self.pending:
                    if row["id"] == self._last_pending_id:
                        row["memory_index"] = len(self.memory) - 1
                        break
            self.memory_added_count += 1
        else:
            self.memory_rejected_count += 1
        cluster_result = self._cluster_pending()
        if accepted and unit_id is None and self.memory and self.memory[-1].get("unit_id") is not None:
            unit_id = self.memory[-1]["unit_id"]
        self._record_units()
        return {"added": accepted, "reason": reason, "assignment": assignment,
                "unit_id": unit_id, "base_functional_distance": distance,
                "epsilon": self.epsilon, "pending_count": len(self.pending),
                "unit_count": len(self._centers), "registered_total": self.registered_count,
                "merged_total": self.merged_count, "expired_total": self.pending_expired_count,
                "cluster_result": cluster_result}

    def blend_base(self, candidates, weights):
        """Default harness blend: blend stored continuous bases (item blend)."""
        return _array(weights) @ _array(candidates)

    def memory_query(self, k, meta=None, *, top_k: int | None = None):
        self._require_fit()
        if not self.memory:
            empty = np.empty((0, self.base_dim), np.float32)
            return {"bases": empty, "weights": np.empty((0,), np.float32),
                    "item_blend": np.zeros((self.base_dim,), np.float32),
                    "skill_blend": np.zeros((self.base_dim,), np.float32),
                    "unit_weights": {}, "unit_ids": []}
        query = _array(k).reshape(-1)
        metadata = dict(meta or {})
        key_matrix, episode_ids, init_ids = self._memory_index()
        eligible_mask = ((episode_ids != metadata.get("episode_id"))
                         & (init_ids != metadata.get("init_state_id")))
        eligible_indices = np.flatnonzero(eligible_mask)
        if not len(eligible_indices):
            return {"bases": np.empty((0, self.base_dim), np.float32), "weights": np.empty((0,), np.float32),
                    "item_blend": np.zeros((self.base_dim,), np.float32),
                    "skill_blend": np.zeros((self.base_dim,), np.float32), "unit_weights": {}, "unit_ids": []}
        keys = key_matrix[eligible_indices]
        if query.shape != (keys.shape[1],):
            raise ValueError(f"query key shape {query.shape} != memory key shape {(keys.shape[1],)}")
        distances = np.linalg.norm(keys - query[None, :], axis=-1)
        indices = np.argsort(distances, kind="stable")[:min(int(top_k or self.topk), len(eligible_indices))]
        chosen_indices = eligible_indices[indices]
        chosen = [self.memory[int(i)] for i in chosen_indices]
        selected_distances = distances[indices]
        weights = np.exp(-selected_distances / max(self.temperature, 1e-12))
        weights /= max(float(weights.sum()), 1e-12)
        bases = np.stack([item["base"] for item in chosen]).astype(np.float32)
        item_blend = weights @ bases
        unit_weights: dict[int, float] = {}
        for item, alpha in zip(chosen, weights):
            if item["unit_id"] is not None:
                unit_weights[int(item["unit_id"])] = unit_weights.get(int(item["unit_id"]), 0.0) + float(alpha)
        skill_blend = np.zeros((self.base_dim,), np.float32)
        total_skill_weight = sum(unit_weights.values())
        if total_skill_weight:
            for unit_id, weight in unit_weights.items():
                skill_blend += weight * self._centers[unit_id]
        return {"bases": bases, "weights": weights.astype(np.float32),
                "item_blend": item_blend.astype(np.float32),
                "skill_blend": skill_blend.astype(np.float32),
                "unit_weights": unit_weights, "unit_ids": [item["unit_id"] for item in chosen],
                "distances": selected_distances.astype(np.float32)}

    def describe(self):
        self._require_fit()
        return {"method": "M6: open-ended functional regions with soft memory blending",
                "encoder": self.encoder_kind,
                "base_dim": self.requested_base_dim, "residual_dim": self.requested_residual_dim,
                "encoder_objective": ("M1 full fit reused; HSIC independence weight "
                                      f"{self.independence_weight}" if self.encoder_kind == "m1"
                                      else "M4 full encoder/prior/decoder fit reused"),
                "unit_count_is_fixed": False,
                "functional_distance": "mean squared decoded-action difference over fixed prior-residual/state probes",
                "functional_median_random_base_pairs": self.functional_median,
                "kappa": self.kappa, "kappa_sweep": list(self.KAPPA_POINTS), "epsilon": self.epsilon,
                "distance_prefilter": "top 10 by base L2, then functional distance",
                "min_support": self.min_support, "min_support_sweep": list(self.MIN_SUPPORT_POINTS),
                "min_episodes": self.min_episodes,
                "pending_batch_size": self.pending_batch_size,
                "functional_probe_count": int(self.PROBE_SIZE),
                "random_base_pair_count": int(self.RANDOM_DISTANCE_PAIRS),
                "pending_count": len(self.pending), "pending_expired_total": self.pending_expired_count,
                "pending_expiry": f"unassigned rows age by {self.pending_batch_size} per clustering pass; expire at {self.pending_max_age}; final fit remainder stays pending",
                "registered_units": self.registered_count, "merged_into_existing_units": self.merged_count,
                "memory_items_added": self.memory_added_count, "memory_items_rejected_by_key": self.memory_rejected_count,
                "unit_count_history": list(self.unit_count_history),
                "units": [{"unit_id": i, "support_count": self.support_counts[i],
                           "episode_count": len(self.episode_sets[i])} for i in range(len(self._centers))],
                "retrieval": "softmax(-key_distance / temperature); exposes item and skill blends",
                "center_updates": {"during_fit": "support-weighted arithmetic mean",
                                   "during_memory_add": "frozen by default" if not self.update_centers_on_add
                                   else "support-weighted arithmetic mean"},
                "updates_without_gradient": "unit registration/merge and center means only; backbone weights frozen after fit"}


def build_variants():
    variants = {}
    for hsic in (0.01, 0.1):
        for kappa in M6.KAPPA_POINTS:
            for minimum in M6.MIN_SUPPORT_POINTS:
                name = f"M6_m1_hsic{hsic:g}_k{kappa:g}_m{minimum}"
                variants[name] = M6(independence_weight=hsic, kappa=kappa, min_support=minimum)
    for kappa in M6.KAPPA_POINTS:
        for minimum in M6.MIN_SUPPORT_POINTS:
            name = f"M6_m4_k{kappa:g}_m{minimum}"
            variants[name] = M6(encoder="m4", kappa=kappa, min_support=minimum)
    return variants
