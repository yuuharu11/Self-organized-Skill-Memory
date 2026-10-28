"""NumPy-only memory storage and retrieval for executed motion segments.

The library has no model dependency. Keys (retrieval features) and values (motion
representations) are stored separately so a caller can choose different encoders.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def namespace_init_state_id(suite: Any, task: Any, init_state_id: Any) -> str:
    """Return a stable ID for state IDs that may be local to each suite/task."""
    if init_state_id is None:
        raise ValueError("init_state_id is required")
    # Match g2_retrieval.load_run's qualified group key exactly.
    return json.dumps([str(suite or ""), str(task), str(init_state_id)])


def _canonical_init_state_id(value: str) -> str:
    try:
        parts = json.loads(value)
    except (TypeError, ValueError) as error:
        raise ValueError("use namespace_init_state_id for initial-state IDs") from error
    if not isinstance(parts, list) or len(parts) != 3 or not all(isinstance(part, str) for part in parts):
        raise ValueError("use namespace_init_state_id for initial-state IDs")
    return json.dumps(parts)


def _resample_sequence(sequence: Any, length: int) -> np.ndarray:
    """Linearly resample a nonempty (time, channels) sequence to fixed length."""
    values = np.asarray(sequence, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("executed segment sequence must have shape (T, D), with T,D > 0")
    if length <= 0:
        raise ValueError("resampled value length must be positive")
    if not np.all(np.isfinite(values)):
        raise ValueError("executed segment sequence must be finite")
    if values.shape[0] == length:
        return np.array(values, copy=True)
    if values.shape[0] == 1:
        return np.repeat(values, length, axis=0)
    source_time = np.linspace(0.0, 1.0, values.shape[0])
    target_time = np.linspace(0.0, 1.0, length)
    return np.stack([np.interp(target_time, source_time, values[:, channel])
                     for channel in range(values.shape[1])], axis=1).astype(np.float32)


def _resample_motion(sequence: np.ndarray, length: int) -> np.ndarray:
    """Resample observed motion, interpolating rotations on SO(3).

    Linear interpolation of shortest rotation vectors would turn +179 to -179
    degrees into a fictitious zero rotation at the midpoint.
    """
    result = _resample_sequence(sequence, length)
    if len(sequence) == 1 or len(sequence) == length:
        return result
    vectors = np.asarray(sequence[:, 3:6], dtype=np.float64)
    angles = np.linalg.norm(vectors, axis=1, keepdims=True)
    factors = np.divide(np.sin(angles / 2), angles, out=np.full_like(angles, .5), where=angles > 1e-12)
    quats = np.concatenate((vectors * factors, np.cos(angles / 2)), axis=1)
    times = np.linspace(0, len(sequence) - 1, length)
    left = np.floor(times).astype(int)
    right = np.minimum(left + 1, len(sequence) - 1)
    alpha = (times - left)[:, None]
    a, b = quats[left], quats[right]
    dot = np.sum(a * b, axis=1, keepdims=True)
    b = np.where(dot < 0, -b, b)
    dot = np.clip(np.abs(dot), 0., 1.)
    theta = np.arccos(dot)
    # The near-identity limit uses normalized linear interpolation.
    sin_theta = np.sin(theta)
    wa = np.divide(np.sin((1 - alpha) * theta), sin_theta,
                   out=1 - alpha, where=sin_theta > 1e-8)
    wb = np.divide(np.sin(alpha * theta), sin_theta,
                   out=alpha.copy(), where=sin_theta > 1e-8)
    result[:, 3:6] = _relative_rotation_vectors(np.array([0., 0., 0., 1.]), wa * a + wb * b)
    return result


def _environment_step_offset(manifest: dict[str, Any], episode: dict[str, Any],
                             arrays: dict[str, np.ndarray]) -> int:
    """Map policy `step` values to env_* row indices using recorder/G2 rules."""
    offset = episode.get("env_step_offset", manifest.get("env_step_offset"))
    if offset is None and manifest.get("format_version") == 1:
        total_steps = episode.get("steps")
        if not isinstance(total_steps, int) or "env_eef_pos" not in arrays:
            raise ValueError("format v1 env trajectory needs integer episode steps metadata")
        offset = total_steps + 1 - len(arrays["env_eef_pos"])
        if offset not in (0, 1):
            raise ValueError("cannot establish format v1 environment time origin")
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError("env_trajectory requires a nonnegative env_step_offset or inferable format v1 metadata")
    return offset


def _normalize_quaternions(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if not np.all(np.isfinite(q)) or np.any(norm < 1e-12):
        raise ValueError("env_eef_quat must contain finite nonzero xyzw quaternions")
    return q / norm


def _relative_rotation_vectors(q_start: np.ndarray, q_future: np.ndarray) -> np.ndarray:
    """Compute shortest relative rotations as axis-angle vectors (quaternion xyzw)."""
    q_start = _normalize_quaternions(np.asarray(q_start, dtype=np.float64).reshape(1, 4))[0]
    q_future = _normalize_quaternions(np.asarray(q_future, dtype=np.float64))
    inverse = np.concatenate((-q_start[:3], q_start[3:]))
    x1, y1, z1, w1 = np.broadcast_to(inverse, q_future.shape).T
    x2, y2, z2, w2 = q_future.T
    relative = np.stack((w1*x2 + x1*w2 + y1*z2 - z1*y2,
                         w1*y2 - x1*z2 + y1*w2 + z1*x2,
                         w1*z2 + x1*y2 - y1*x2 + z1*w2,
                         w1*w2 - x1*x2 - y1*y2 - z1*z2), axis=1)
    relative = _normalize_quaternions(relative)
    relative = np.where(relative[:, 3:4] < 0, -relative, relative)
    vector = relative[:, :3]
    vector_norm = np.linalg.norm(vector, axis=1, keepdims=True)
    angle = 2 * np.arctan2(vector_norm, np.clip(relative[:, 3:4], -1.0, 1.0))
    factor = np.divide(angle, vector_norm, out=np.zeros_like(angle), where=vector_norm > 1e-12)
    return (vector * factor).astype(np.float32)


def _recorded_motion_segment(arrays: dict[str, np.ndarray], *, step: int, n_executed: int,
                             env_step_offset: int) -> np.ndarray:
    """Recover executed relative pose/gripper changes, one row per environment action."""
    pos = np.asarray(arrays["env_eef_pos"], dtype=np.float64)
    quat = np.asarray(arrays["env_eef_quat"], dtype=np.float64)
    grip = np.asarray(arrays["env_gripper"], dtype=np.float64)
    if grip.ndim == 1:
        grip = grip[:, None]
    if pos.ndim != 2 or pos.shape[1] != 3 or quat.ndim != 2 or quat.shape[1] != 4:
        raise ValueError("env_eef_pos and env_eef_quat must have shapes (T,3) and (T,4)")
    if grip.ndim != 2 or grip.shape[1] == 0 or len(pos) != len(quat) or len(pos) != len(grip):
        raise ValueError("env_eef_pos, env_eef_quat, and env_gripper must have matching lengths")
    if not all(np.all(np.isfinite(array)) for array in (pos, quat, grip)):
        raise ValueError("environment trajectory arrays must be finite")
    start_index = step - env_step_offset
    final_index = start_index + n_executed
    if start_index < 0 or final_index >= len(pos):
        raise ValueError("env_* arrays do not contain the segment start and all executed observations")
    sample_slice = slice(start_index + 1, final_index + 1)
    relative_pos = pos[sample_slice] - pos[start_index]
    relative_rotvec = _relative_rotation_vectors(quat[start_index], quat[sample_slice])
    relative_gripper = grip[sample_slice] - grip[start_index]
    return np.concatenate((relative_pos, relative_rotvec, relative_gripper), axis=1).astype(np.float32)


@dataclass(frozen=True)
class MemoryMetadata:
    episode_id: str
    init_state_id: str
    task: str
    segment_id: str
    start_step: int
    end_step: int  # exclusive
    n_executed: int
    representation_version: str
    suite: str = ""
    entry_id: str = ""
    value_source: str = "feature"
    value_length: int | None = None

    def __post_init__(self) -> None:
        for name in ("episode_id", "init_state_id", "segment_id", "representation_version"):
            if not str(getattr(self, name)):
                raise ValueError(f"{name} must be non-empty")
        if isinstance(self.start_step, bool) or isinstance(self.end_step, bool):
            raise ValueError("step bounds must be integers")
        if int(self.start_step) != self.start_step or int(self.end_step) != self.end_step:
            raise ValueError("step bounds must be integers")
        if self.start_step < 0:
            raise ValueError("start_step must be nonnegative")
        if isinstance(self.n_executed, bool) or int(self.n_executed) != self.n_executed or self.n_executed <= 0:
            raise ValueError("n_executed must be a positive integer; unexecuted chunks cannot be stored")
        if self.end_step <= self.start_step:
            raise ValueError("end_step is exclusive and must be greater than start_step")
        if self.end_step - self.start_step != self.n_executed:
            raise ValueError("n_executed must equal end_step - start_step")
        if not self.value_source:
            raise ValueError("value_source must be non-empty")
        if self.value_length is not None and (isinstance(self.value_length, bool) or
                                               int(self.value_length) != self.value_length or self.value_length <= 0):
            raise ValueError("value_length must be a positive integer when specified")
        object.__setattr__(self, "init_state_id", _canonical_init_state_id(self.init_state_id))


@dataclass
class MemoryEntry:
    metadata: MemoryMetadata
    key: np.ndarray
    value: np.ndarray
    usage_count: int = 0


@dataclass(frozen=True)
class SearchResult:
    mode: str
    candidate_ids: tuple[str, ...]
    weights: np.ndarray
    composite_value: np.ndarray | None
    candidate_values: np.ndarray
    distances: np.ndarray
    eligible_candidate_ids: tuple[str, ...]
    excluded: tuple[dict[str, str], ...]

    @property
    def z_collection(self) -> np.ndarray:
        """Values for callers that want to consume retrieved candidates separately."""
        return self.candidate_values


class MemoryStore:
    """Append-only collection of executed segments with deterministic NumPy retrieval."""

    MODES = {"none", "random", "top1", "topk_softmax", "topk_uniform", "random_k"}

    def __init__(self, entries: Iterable[MemoryEntry] = (), *, seed: int = 0,
                 search_log_path: str | Path | None = None):
        self.entries: list[MemoryEntry] = []
        self._entry_ids: set[str] = set()
        self._rng = np.random.default_rng(seed)
        self._tie_rng = np.random.default_rng(seed + 1)
        self.search_log_path = Path(search_log_path) if search_log_path is not None else None
        self.search_log: list[dict[str, Any]] = []
        for entry in entries:
            self.append(entry.metadata, entry.key, entry.value, usage_count=entry.usage_count)

    def __len__(self) -> int:
        return len(self.entries)

    @staticmethod
    def _array(value: Any, label: str) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        if array.size == 0 or not np.all(np.isfinite(array)):
            raise ValueError(f"{label} must be non-empty and finite")
        return np.array(array, copy=True)

    def append(self, metadata: MemoryMetadata, key: Any, value: Any, *, usage_count: int = 0) -> str:
        """Append one already-executed segment and return its unique entry ID."""
        if not isinstance(metadata, MemoryMetadata):
            raise TypeError("metadata must be a MemoryMetadata instance")
        k = self._array(key, "key")
        z = self._array(value, "value")
        if k.ndim != 1 or z.ndim == 0:
            raise ValueError("key must be a vector; value must have at least one dimension")
        if metadata.value_length is not None and (z.ndim < 2 or z.shape[0] != metadata.value_length):
            raise ValueError("value first dimension must match metadata value_length")
        if isinstance(usage_count, bool) or int(usage_count) != usage_count or usage_count < 0:
            raise ValueError("usage_count must be a nonnegative integer")
        entry_id = metadata.entry_id or uuid.uuid4().hex
        if entry_id in self._entry_ids:
            raise ValueError(f"duplicate memory entry ID: {entry_id}")
        metadata = MemoryMetadata(**{**asdict(metadata), "entry_id": entry_id})
        self.entries.append(MemoryEntry(metadata, k, z, int(usage_count)))
        self._entry_ids.add(entry_id)
        return entry_id

    @classmethod
    def from_recording(cls, run_dir: str | Path, *, key_feature: str, value_feature: str | None = None,
                       value_source: str = "feature", value_length: int | None = None,
                       representation_version: str, seed: int = 0,
                       search_log_path: str | Path | None = None) -> "MemoryStore":
        """Build memory entries from a saved rollout run and its `feat__*` arrays.

        `key_feature` and (for `value_source="feature"`) `value_feature` are NPZ
        keys, e.g. `feat__L17_mean_t_last`; the same feature may be selected for both.
        Executed values may instead come from `pred_chunk[:n_executed]` or the environment
        trajectory recovered from `env_*` and `step`. Values are resampled to a
        fixed `value_length` (inferred from manifest `replan_steps`, otherwise max
        `n_executed` across the entire run). Observed rotations use SLERP.
        A zero-execution row rejects the import.
        """
        run_dir = Path(run_dir)
        if value_feature == "pred_chunk":
            raise ValueError("pred_chunk includes unexecuted predictions; choose an executed-segment representation")
        if value_source not in {"feature", "pred_chunk_executed", "env_trajectory"}:
            raise ValueError("value_source must be 'feature', 'pred_chunk_executed', or 'env_trajectory'")
        if value_source == "feature" and not value_feature:
            raise ValueError("value_feature is required when value_source='feature'")
        if value_source != "feature" and value_feature is not None:
            raise ValueError("value_feature is only valid when value_source='feature'")
        if value_length is not None and (isinstance(value_length, bool) or int(value_length) != value_length or value_length <= 0):
            raise ValueError("value_length must be a positive integer")
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        episodes = manifest.get("episodes", [])
        if isinstance(episodes, dict):
            episodes = [dict(v, episode_id=k) if isinstance(v, dict) else {"episode_id": k}
                        for k, v in episodes.items()]
        if not isinstance(episodes, list):
            raise ValueError("manifest episodes must be a list or object")
        store = cls(seed=seed, search_log_path=search_log_path)
        pending_entries = []
        seen_episodes = set()
        seen_files = set()
        for episode in episodes:
            if not isinstance(episode, dict):
                raise ValueError("each manifest episode must be an object")
            episode_id = str(episode.get("episode_id", episode.get("id", "")))
            if not episode_id:
                raise ValueError("manifest episode missing episode_id")
            if episode_id in seen_episodes:
                raise ValueError("manifest contains duplicate episode IDs")
            seen_episodes.add(episode_id)
            state_id = episode.get("init_state_id", episode.get("initial_state_id"))
            if state_id is None:
                raise ValueError(f"{episode_id}: manifest episode missing init_state_id")
            suite = episode.get("suite", manifest.get("suite", ""))
            task = episode.get("task_id", episode.get("task", ""))
            npz_path = run_dir / "episodes" / f"{episode_id}.npz"
            if not npz_path.exists():
                alternate = episode.get("file", episode.get("path"))
                if alternate:
                    npz_path = run_dir / str(alternate)
            stat = npz_path.stat()
            identity = (stat.st_dev, stat.st_ino)
            if identity in seen_files:
                raise ValueError("multiple episode IDs refer to the same episode file")
            seen_files.add(identity)
            with np.load(npz_path, allow_pickle=False) as data:
                source_key = {"pred_chunk_executed": "pred_chunk"}.get(value_source)
                required = ["step", "n_executed", "segment_id", key_feature]
                if value_source == "feature":
                    required.append(str(value_feature))
                elif source_key:
                    required.append(source_key)
                else:
                    required.extend(("env_eef_pos", "env_eef_quat", "env_gripper"))
                missing = [name for name in required if name not in data]
                if missing:
                    raise ValueError(f"{episode_id}: missing NPZ arrays: {missing}")
                steps = np.asarray(data["step"])
                executed = np.asarray(data["n_executed"])
                segment_ids = np.asarray(data["segment_id"])
                keys = np.asarray(data[key_feature])
                values = np.asarray(data[value_feature]) if value_source == "feature" else None
                pred_chunks = np.asarray(data["pred_chunk"]) if value_source == "pred_chunk_executed" else None
                env_arrays = ({name: np.asarray(data[name]) for name in ("env_eef_pos", "env_eef_quat", "env_gripper")}
                              if value_source == "env_trajectory" else None)
                lengths = {len(steps), len(executed), len(segment_ids), len(keys)}
                if values is not None:
                    lengths.add(len(values))
                if pred_chunks is not None:
                    lengths.add(len(pred_chunks))
                if len(lengths) != 1:
                    raise ValueError(f"{episode_id}: inference arrays have inconsistent row counts")
                if (steps.ndim != 1 or executed.ndim != 1 or segment_ids.ndim != 1
                        or not np.issubdtype(steps.dtype, np.integer)
                        or np.any(steps < 0) or np.any(steps[1:] <= steps[:-1])):
                    raise ValueError(f"{episode_id}: step, n_executed and segment_id must be vectors")
                if keys.ndim != 2 or (values is not None and values.ndim != 2):
                    raise ValueError(f"{episode_id}: key and value features must have shape (N, D)")
                for index, (start, count) in enumerate(zip(steps, executed)):
                    if isinstance(count, (bool, np.bool_)) or int(count) < 0 or int(count) != count:
                        raise ValueError(f"{episode_id}: n_executed must contain nonnegative integers")
                    if count == 0:
                        raise ValueError(f"{episode_id}: refusing unexecuted segment at row {index}")
                    start, count = int(start), int(count)
                    if index + 1 < len(steps) and start + count > int(steps[index + 1]):
                        raise ValueError(f"{episode_id}: executed segment overlaps the next inference")
                    total_steps = episode.get("steps")
                    if total_steps is not None and start + count > total_steps:
                        raise ValueError(f"{episode_id}: executed segment exceeds episode steps")
                    if value_source == "feature":
                        segment_value = np.asarray(values[index])
                        row_value = segment_value
                    elif value_source == "pred_chunk_executed":
                        if pred_chunks.ndim != 3:
                            raise ValueError(f"{episode_id}: pred_chunk must have shape (N, H, Da)")
                        if count > pred_chunks.shape[1]:
                            raise ValueError(f"{episode_id}: n_executed exceeds pred_chunk horizon at row {index}")
                        executed_prefix = pred_chunks[index, :count]
                        row_value = executed_prefix
                    else:
                        trajectory = _recorded_motion_segment(
                            env_arrays, step=start, n_executed=count,
                            env_step_offset=_environment_step_offset(manifest, episode, env_arrays),
                        )
                        row_value = trajectory
                    metadata = MemoryMetadata(
                        episode_id=episode_id,
                        init_state_id=namespace_init_state_id(suite, task, state_id),
                        task=str(episode.get("task", task)),
                        segment_id=str(segment_ids[index]),
                        start_step=start,
                        end_step=start + count,
                        n_executed=count,
                        representation_version=representation_version,
                        suite=str(suite),
                        value_source=value_source,
                        value_length=(count if value_source != "feature" else None),
                    )
                    pending_entries.append((metadata, keys[index], row_value))
        if value_source != "feature":
            inferred_length = value_length if value_length is not None else manifest.get("replan_steps")
            if inferred_length is None:
                inferred_length = max((meta.n_executed for meta, _, _ in pending_entries), default=0)
            if (isinstance(inferred_length, bool) or int(inferred_length) != inferred_length
                    or inferred_length <= 0):
                raise ValueError("fixed value length must be a positive integer")
        for metadata, key, raw_value in pending_entries:
            if value_source != "feature":
                metadata = replace(metadata, value_length=int(inferred_length))
                resample = _resample_motion if value_source == "env_trajectory" else _resample_sequence
                raw_value = resample(raw_value, int(inferred_length))
            store.append(metadata, key, raw_value)
        return store

    def _write_log(self, event: dict[str, Any]) -> None:
        self.search_log.append(event)
        if self.search_log_path is not None:
            self.search_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.search_log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")

    def search(self, query_key: Any, *, mode: str = "top1", k: int = 1, temperature: float = 1.0,
               random_k_weighting: str = "uniform",
               episode_id: str | None = None, init_state_id: str | None = None,
               exclude_episode_ids: Iterable[str] = (), exclude_init_state_ids: Iterable[str] = (),
               exclude_same_episode: bool = True, exclude_same_init_state: bool = True,
               representation_version: str | None = None) -> SearchResult:
        """Search by L2 distance with episode/state and optional version exclusions.

        `init_state_id` and exclusion IDs must use the same namespaced form as the
        stored metadata; `namespace_init_state_id` constructs it. Default guards
        require both query IDs. `random` chooses one eligible candidate. `random_k` chooses
        up to k candidates uniformly without replacement; its weights are uniform, or the
        top-k softmax weight sequence is assigned in random draw order (`topk_softmax`).
        `none` returns no candidates and never increments usage.
        """
        if mode not in self.MODES:
            raise ValueError(f"unknown retrieval mode {mode!r}; expected one of {sorted(self.MODES)}")
        if mode != "none":
            if exclude_same_episode and not episode_id:
                raise ValueError("episode_id is required when same-episode exclusion is enabled")
            if exclude_same_init_state and not init_state_id:
                raise ValueError("init_state_id is required when same-initial-state exclusion is enabled")
        if init_state_id is not None:
            init_state_id = _canonical_init_state_id(init_state_id)
        if isinstance(k, bool) or int(k) != k or k <= 0:
            raise ValueError("k must be a positive integer")
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be positive and finite")
        if random_k_weighting not in {"uniform", "topk_softmax"}:
            raise ValueError("random_k_weighting must be 'uniform' or 'topk_softmax'")
        query = self._array(query_key, "query_key")
        if query.ndim != 1:
            raise ValueError("query_key must be a vector")
        excluded_episode_set = {str(x) for x in exclude_episode_ids}
        excluded_init_set = {_canonical_init_state_id(x) for x in exclude_init_state_ids}
        excluded: list[dict[str, str]] = []
        eligible: list[MemoryEntry] = []
        for entry in self.entries:
            meta = entry.metadata
            reason = None
            if exclude_same_episode and episode_id is not None and meta.episode_id == episode_id:
                reason = "same_episode"
            elif meta.episode_id in excluded_episode_set:
                reason = "excluded_episode_id"
            elif exclude_same_init_state and init_state_id is not None and meta.init_state_id == init_state_id:
                reason = "same_init_state"
            elif meta.init_state_id in excluded_init_set:
                reason = "excluded_init_state_id"
            elif representation_version is not None and meta.representation_version != representation_version:
                reason = "representation_version_mismatch"
            elif entry.key.shape != query.shape:
                reason = "key_shape_mismatch"
            if reason is None:
                eligible.append(entry)
            else:
                excluded.append({"candidate_id": meta.entry_id, "reason": reason})

        if eligible:
            # Fit feature scaling on stored candidate keys only. Never use query/evaluation
            # keys to estimate statistics; constant dimensions retain unit scale.
            key_matrix = np.stack([entry.key.astype(np.float64) for entry in eligible])
            mean = key_matrix.mean(axis=0)
            scale = key_matrix.std(axis=0)
            scale[scale < 1e-8] = 1.0
            normalized_keys = (key_matrix - mean) / scale
            normalized_query = (query.astype(np.float64) - mean) / scale
            distances = np.linalg.norm(normalized_keys - normalized_query, axis=1)
        else:
            distances = np.empty((0,), dtype=np.float64)
        selected: list[int] = []
        if mode == "random" and eligible:
            selected = [int(self._rng.integers(len(eligible)))]
        elif mode == "random_k" and eligible:
            selected = self._rng.choice(len(eligible), size=min(int(k), len(eligible)), replace=False).tolist()
        elif mode == "top1" and eligible:
            selected = [int(self._tie_rng.choice(np.flatnonzero(distances == distances.min())))]
        elif mode in ("topk_softmax", "topk_uniform") and eligible:
            shuffled = self._tie_rng.permutation(len(eligible))
            selected = shuffled[np.argsort(distances[shuffled], kind="stable")][:min(int(k), len(eligible))].tolist()
        weights = np.empty((0,), dtype=np.float32)
        chosen: list[MemoryEntry] = [eligible[i] for i in selected]
        selected_distances = distances[selected] if selected else np.empty((0,), dtype=np.float64)
        if chosen:
            if mode == "topk_softmax":
                logits = -(selected_distances - selected_distances.min()) / float(temperature)
                raw = np.exp(logits)
                weights = (raw / raw.sum()).astype(np.float32)
            elif mode == "random_k" and random_k_weighting == "topk_softmax":
                nearest = np.argsort(distances, kind="stable")[:len(chosen)]
                topk_distances = distances[nearest]
                logits = -(topk_distances - topk_distances.min()) / float(temperature)
                raw = np.exp(logits)
                # Random selection order assigns the nearest-rank weights to unrelated entries.
                weights = (raw / raw.sum()).astype(np.float32)
            else:
                weights = np.full((len(chosen),), 1.0 / len(chosen), dtype=np.float32)
            z_shape = chosen[0].value.shape
            if any(entry.value.shape != z_shape for entry in chosen):
                raise ValueError("retrieved values have inconsistent shapes; filter by representation_version")
            candidate_values = np.stack([entry.value for entry in chosen]).astype(np.float32, copy=False)
            composite = np.tensordot(weights, candidate_values, axes=(0, 0)).astype(np.float32)
            for entry in chosen:
                entry.usage_count += 1
        else:
            candidate_values = np.empty((0,), dtype=np.float32)
            composite = None
        result = SearchResult(mode, tuple(entry.metadata.entry_id for entry in chosen), weights,
                              composite, candidate_values, selected_distances.astype(np.float32),
                              tuple(entry.metadata.entry_id for entry in eligible), tuple(excluded))
        self._write_log({
            "mode": mode,
            "query_episode_id": episode_id,
            "query_init_state_id": init_state_id,
            "representation_version": representation_version,
            "k": int(k),
            "temperature": float(temperature),
            "random_k_weighting": random_k_weighting,
            "candidate_ids": list(result.candidate_ids),
            "eligible_candidate_ids": list(result.eligible_candidate_ids),
            "weights": result.weights.tolist(),
            "distances": result.distances.tolist(),
            "excluded": list(result.excluded),
        })
        return result

    def save(self, path: str | Path, *, overwrite: bool = False) -> None:
        """Save the full store in a safe, pickle-free NPZ container."""
        path = Path(path)
        if path.exists() and not overwrite:
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        keys = [entry.key for entry in self.entries]
        values = [entry.value for entry in self.entries]
        key_shapes = [list(value.shape) for value in keys]
        value_shapes = [list(value.shape) for value in values]
        metadata = [asdict(entry.metadata) for entry in self.entries]
        metadata_json = json.dumps({"format_version": 2, "metadata": metadata,
                                    "key_shapes": key_shapes, "value_shapes": value_shapes,
                                    "usage_counts": [entry.usage_count for entry in self.entries],
                                    "search_log": self.search_log}, ensure_ascii=False)
        payload: dict[str, Any] = {"metadata_json": np.asarray(metadata_json)}
        for i, array in enumerate(keys):
            payload[f"key_{i}"] = array
        for i, array in enumerate(values):
            payload[f"value_{i}"] = array
        with path.open("wb" if overwrite else "xb") as f:
            np.savez_compressed(f, **payload)

    @classmethod
    def load(cls, path: str | Path, *, seed: int = 0,
             search_log_path: str | Path | None = None) -> "MemoryStore":
        with np.load(path, allow_pickle=False) as data:
            raw = json.loads(str(data["metadata_json"].item()))
            if raw.get("format_version") != 2:
                raise ValueError(f"unsupported memory format version: {raw.get('format_version')}")
            entries = []
            for i, metadata in enumerate(raw["metadata"]):
                entries.append(MemoryEntry(
                    MemoryMetadata(**metadata),
                    np.array(data[f"key_{i}"], copy=True),
                    np.array(data[f"value_{i}"], copy=True),
                    int(raw["usage_counts"][i]),
                ))
            store = cls(entries, seed=seed, search_log_path=search_log_path)
            store.search_log = raw.get("search_log", [])
        return store
