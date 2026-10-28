"""Skill memory for z: fixed-PCA keys, search-then-add with a relative threshold, top-k blend.

* Key = [pooled VLM image output tokens, proprio]. Two image pools exist (SCHEMA.md):
  ``vlm_img_nolang`` (VLM run without language input; main version) and ``vlm_img_lang``
  (normal input; image tokens attend to language in the bidirectional prefix, so language
  leaks into them; comparison version). `include_language` selects which. Each block is
  standardized with reference statistics and divided by sqrt(block dim) times a block
  weight so neither block dominates by dimension count; then a PCA fitted **once** on
  reference data. `KeyEncoder` is frozen after `fit` and saved with the memory.
* Distances are Euclidean in that fixed PCA space. `MemoryStore.search` re-standardizes
  keys on the current eligible entries, which would move the metric as memory grows and
  make the threshold inconsistent, so this wrapper ranks on its own and uses
  `MemoryStore` for entries, metadata validation (`n_executed > 0`), usage counts and
  save/load. Exclusion reasons match `MemoryStore.search`.
* Add rule: retrieve; if the nearest eligible distance > tau, the row is novel and is
  added (only if its chunk and the whole z interval were executed). tau = quantile q of
  the within-memory nearest-neighbour distances (each entry to its nearest entry from a
  different episode, the same exclusion used for queries), never an absolute value.
  ``threshold_mode="fixed"`` computes tau once from the seed memory (`calibrate`);
  ``"adaptive"`` recomputes it from the current memory before each decision. Note that
  adaptive tau ratchets upward because only far-apart rows get added.
* Retrieval returns the softmax(-d / (temperature * d_ref)) blend of the top-k z, where
  d_ref = median within-memory NN distance at calibration (temperature is relative).
* Every decision is appended to `log` (and optionally JSONL): added?, reason, nearest
  distance, tau, quantile, candidates, distances, weights.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from skill_moe.skillnet.examples.memory_pilot.memory import MemoryMetadata, MemoryStore


VLM_BLOCKS = {False: "vlm_img_nolang", True: "vlm_img_lang"}

# --------------------------------------------------------------------------
# Keys
# --------------------------------------------------------------------------

@dataclass
class KeyEncoder:
    blocks: tuple[str, ...]
    block_weights: tuple[float, ...]
    means: dict[str, np.ndarray]
    scales: dict[str, np.ndarray]
    pca_mean: np.ndarray
    components: np.ndarray        # (P, D_total)
    explained_variance_ratio: np.ndarray
    include_language: bool

    @classmethod
    def fit(cls, reference: Mapping[str, np.ndarray], *, include_language: bool = False,
            n_components: int | float = 32, block_weights: Mapping[str, float] | None = None) -> "KeyEncoder":
        """`reference`: {"vlm_img_nolang": (N,Dv), "vlm_img_lang": (N,Dv), "proprio": (N,Ds)}.

        The language version uses `vlm_img_lang` instead of `vlm_img_nolang`. `n_components` is
        a count, or a float in (0, 1) for the explained-variance fraction.
        """
        vlm = VLM_BLOCKS[include_language]
        blocks = tuple(b for b in (vlm, "proprio") if b in reference)
        if vlm not in blocks:
            raise ValueError(f"reference data has no {vlm} block")
        weights = {b: 1.0 for b in blocks} | dict(block_weights or {})
        means, scales, parts = {}, {}, []
        for b in blocks:
            x = np.asarray(reference[b], dtype=np.float64)
            means[b], scales[b] = x.mean(0), x.std(0)
            scales[b][scales[b] < 1e-8] = 1.0
            parts.append((x - means[b]) / scales[b] * (weights[b] / np.sqrt(x.shape[1])))
        x = np.concatenate(parts, axis=1)
        mean = x.mean(0)
        _, s, vt = np.linalg.svd(x - mean, full_matrices=False)
        ratio = s ** 2 / max((s ** 2).sum(), 1e-12)
        if isinstance(n_components, float) and 0 < n_components < 1:
            p = int(np.searchsorted(np.cumsum(ratio), n_components) + 1)
        else:
            p = int(n_components)
        p = max(1, min(p, len(s)))
        return cls(blocks, tuple(weights[b] for b in blocks), means, scales, mean, vt[:p], ratio[:p],
                   include_language)

    def transform(self, data: Mapping[str, np.ndarray]) -> np.ndarray:
        parts = []
        for b, w in zip(self.blocks, self.block_weights):
            x = np.atleast_2d(np.asarray(data[b], dtype=np.float64))
            parts.append((x - self.means[b]) / self.scales[b] * (w / np.sqrt(x.shape[1])))
        return (np.concatenate(parts, axis=1) - self.pca_mean) @ self.components.T

    @property
    def fingerprint(self) -> str:
        h = hashlib.sha256()
        for arr in (self.pca_mean, self.components):
            h.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
        h.update(json.dumps([self.blocks, self.block_weights, self.include_language]).encode())
        return h.hexdigest()[:16]

    def save(self, path: str | Path) -> None:
        payload = {"pca_mean": self.pca_mean, "components": self.components,
                   "explained_variance_ratio": self.explained_variance_ratio,
                   "meta": np.asarray(json.dumps({"blocks": self.blocks, "block_weights": self.block_weights,
                                                  "include_language": self.include_language}))}
        for b in self.blocks:
            payload[f"mean__{b}"], payload[f"scale__{b}"] = self.means[b], self.scales[b]
        with Path(path).open("xb") as f:
            np.savez(f, **payload)

    @classmethod
    def load(cls, path: str | Path) -> "KeyEncoder":
        with np.load(path, allow_pickle=False) as d:
            meta = json.loads(str(d["meta"].item()))
            blocks = tuple(meta["blocks"])
            return cls(blocks, tuple(meta["block_weights"]), {b: d[f"mean__{b}"] for b in blocks},
                       {b: d[f"scale__{b}"] for b in blocks}, d["pca_mean"], d["components"],
                       d["explained_variance_ratio"], bool(meta["include_language"]))


# --------------------------------------------------------------------------
# Memory
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class MemoryConfig:
    k: int = 5
    temperature: float = 0.5           # relative to d_ref
    threshold_quantile: float = 0.9
    threshold_mode: str = "fixed"      # "fixed" (calibrate once) | "adaptive"
    min_entries_for_threshold: int = 10
    exclude_same_episode: bool = True  # for queries and for the within-memory NN distribution
    exclude_same_init_state: bool = False
    require_full_interval: bool = True

    def __post_init__(self):
        if self.threshold_mode not in ("fixed", "adaptive"):
            raise ValueError("threshold_mode must be 'fixed' or 'adaptive'")
        if not 0 < self.threshold_quantile < 1:
            raise ValueError("threshold_quantile must be in (0, 1)")
        if self.k < 1 or self.temperature <= 0:
            raise ValueError("k must be >= 1 and temperature > 0")


@dataclass(frozen=True)
class Retrieval:
    z: np.ndarray | None
    candidate_ids: tuple[str, ...]
    weights: np.ndarray
    distances: np.ndarray
    nearest_distance: float
    tau: float | None
    novel: bool


class EmergentMemory:
    def __init__(self, encoder: KeyEncoder, config: MemoryConfig = MemoryConfig(), *,
                 horizon: int, representation_version: str = "emergent-z-v1",
                 log_path: str | Path | None = None, store: MemoryStore | None = None):
        self.encoder, self.config, self.horizon = encoder, config, int(horizon)
        self.representation_version = f"{representation_version}|key={encoder.fingerprint}"
        self.store = store if store is not None else MemoryStore()
        self.log: list[dict[str, Any]] = []
        self.log_path = Path(log_path) if log_path is not None else None
        self.tau: float | None = None
        self.d_ref: float | None = None

    # ---- internals -------------------------------------------------------
    def _sync(self) -> None:
        """Vectorized views of the store (rebuilt only when entries were appended)."""
        if getattr(self, "_synced", -1) == len(self.store):
            return
        entries = self.store.entries
        p = self.encoder.components.shape[0]
        self._keys = (np.stack([e.key for e in entries]).astype(np.float64) if entries else np.zeros((0, p)))
        self._ep = np.array([e.metadata.episode_id for e in entries], dtype=object)
        self._st = np.array([e.metadata.init_state_id for e in entries], dtype=object)
        self._ver = np.array([e.metadata.representation_version for e in entries], dtype=object)
        self._synced = len(entries)

    def _matrix(self) -> np.ndarray:
        self._sync()
        return self._keys

    def _eligible(self, episode_id: str | None, init_state_id: str | None) -> np.ndarray:
        self._sync()
        ok = self._ver == self.representation_version
        if self.config.exclude_same_episode and episode_id is not None:
            ok &= self._ep != episode_id
        if self.config.exclude_same_init_state and init_state_id is not None:
            ok &= self._st != init_state_id
        return ok.astype(bool)

    def nn_distances(self) -> np.ndarray:
        """Each entry's distance to its nearest *eligible* other entry (same exclusions as queries)."""
        keys = self._matrix()
        if len(keys) < 2:
            return np.zeros(0)
        d = np.sqrt(np.maximum((keys ** 2).sum(1)[:, None] - 2 * keys @ keys.T + (keys ** 2).sum(1)[None], 0))
        ep, st = self._ep.astype(str), self._st.astype(str)
        invalid = np.eye(len(keys), dtype=bool)
        if self.config.exclude_same_episode:
            invalid |= ep[:, None] == ep[None]
        if self.config.exclude_same_init_state:
            invalid |= st[:, None] == st[None]
        d = np.where(invalid, np.inf, d).min(axis=1)
        return d[np.isfinite(d)]

    def calibrate(self) -> float | None:
        nn = self.nn_distances()
        if len(nn) < self.config.min_entries_for_threshold:
            self.tau = None
            return None
        self.tau = float(np.quantile(nn, self.config.threshold_quantile))
        self.d_ref = float(np.median(nn))
        return self.tau

    def _current_tau(self) -> float | None:
        if self.config.threshold_mode == "adaptive":
            return self.calibrate()
        return self.tau

    def _emit(self, event: dict[str, Any]) -> None:
        self.log.append(event)
        if self.log_path is not None:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")

    def encode(self, key_inputs: Mapping[str, np.ndarray]) -> np.ndarray:
        return self.encoder.transform(key_inputs)[0]

    # ---- public API ------------------------------------------------------
    def add(self, key: np.ndarray, z: np.ndarray, *, episode_id: str, init_state_id: str, task: str = "",
            suite: str = "", step: int, segment_id: str = "", interval_len: int | None = None) -> str:
        """Unconditional add of an executed interval (seeding). Key must already be PCA-encoded."""
        n = int(self.horizon if interval_len is None else interval_len)
        meta = MemoryMetadata(episode_id=episode_id, init_state_id=init_state_id, task=task,
                              segment_id=segment_id or f"{episode_id}@{step}", start_step=int(step),
                              end_step=int(step) + n, n_executed=n,
                              representation_version=self.representation_version, suite=suite,
                              value_source="z_post_mean")
        return self.store.append(meta, key, z)

    def retrieve(self, key: np.ndarray, *, episode_id: str | None = None,
                 init_state_id: str | None = None, count_usage: bool = True) -> Retrieval:
        tau = self._current_tau()
        keys = self._matrix()
        ok = self._eligible(episode_id, init_state_id)
        idx = np.flatnonzero(ok)
        if not len(idx):
            return Retrieval(None, (), np.zeros(0), np.zeros(0), float("inf"), tau, True)
        d = np.linalg.norm(keys[idx] - np.asarray(key, dtype=np.float64)[None], axis=1)
        order = np.argsort(d, kind="stable")[:self.config.k]
        chosen, dist = idx[order], d[order]
        scale = self.config.temperature * (self.d_ref if self.d_ref else max(float(dist[0]), 1e-12))
        w = np.exp(-(dist - dist[0]) / max(scale, 1e-12))
        w = w / w.sum()
        values = np.stack([self.store.entries[i].value for i in chosen]).astype(np.float64)
        if count_usage:
            for i in chosen:
                self.store.entries[i].usage_count += 1
        novel = tau is None or float(dist[0]) > tau
        return Retrieval((w @ values).astype(np.float32),
                         tuple(self.store.entries[i].metadata.entry_id for i in chosen),
                         w, dist, float(dist[0]), tau, novel)

    def observe(self, key: np.ndarray, z: np.ndarray | None, *, executed: bool, interval_len: int,
                episode_id: str, init_state_id: str, task: str = "", suite: str = "", step: int,
                extra_log: Mapping[str, Any] | None = None) -> tuple[Retrieval, str | None]:
        """Search, then add the executed posterior z if nothing is closer than tau."""
        r = self.retrieve(key, episode_id=episode_id, init_state_id=init_state_id)
        entry_id, reason = None, None
        # Eligibility first so the log says why an unexecuted row was never a candidate.
        if not executed:
            reason = "not_executed"
        elif self.config.require_full_interval and interval_len != self.horizon:
            reason = "interval_truncated"
        elif z is None or not np.all(np.isfinite(z)):
            reason = "z_missing"
        elif not r.novel:
            reason = "near_existing"
        elif r.tau is None:
            reason = "added_no_threshold_yet"
        else:
            reason = "added_novel"
        if reason.startswith("added"):
            entry_id = self.add(key, z, episode_id=episode_id, init_state_id=init_state_id, task=task,
                                suite=suite, step=step, interval_len=interval_len)
        self._emit({"episode_id": episode_id, "step": int(step), "task": task, "added": entry_id is not None,
                    "reason": reason, "entry_id": entry_id, "nearest_distance": r.nearest_distance,
                    "tau": r.tau, "quantile": self.config.threshold_quantile,
                    "threshold_mode": self.config.threshold_mode, "memory_size": len(self.store),
                    "candidate_ids": list(r.candidate_ids), "distances": r.distances.tolist(),
                    "weights": r.weights.tolist(), **dict(extra_log or {})})
        return r, entry_id

    # ---- persistence -----------------------------------------------------
    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        self.store.save(directory / "store.npz")
        self.encoder.save(directory / "key_encoder.npz")
        (directory / "memory.json").write_text(json.dumps({
            "config": asdict(self.config), "horizon": self.horizon, "tau": self.tau, "d_ref": self.d_ref,
            "representation_version": self.representation_version}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path) -> "EmergentMemory":
        directory = Path(directory)
        meta = json.loads((directory / "memory.json").read_text(encoding="utf-8"))
        encoder = KeyEncoder.load(directory / "key_encoder.npz")
        mem = cls(encoder, MemoryConfig(**meta["config"]), horizon=meta["horizon"],
                  store=MemoryStore.load(directory / "store.npz"))
        mem.representation_version = meta["representation_version"]
        mem.tau, mem.d_ref = meta["tau"], meta["d_ref"]
        return mem
