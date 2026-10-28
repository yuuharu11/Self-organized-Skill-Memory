"""Episode-safe extraction of executed motion segments for primitive learning."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from skill_moe.skillnet.examples.memory_pilot.memory.store import (
    _environment_step_offset,
    _recorded_motion_segment,
    _resample_motion,
    _resample_sequence,
    namespace_init_state_id,
)


@dataclass(frozen=True)
class PrimitiveDataset:
    """Tensors contain only state/action/motion; labels are carried separately for evaluation."""

    states: np.ndarray  # (segments, H, state_dim), current proprio repeated over phase
    actions: np.ndarray  # (segments, H, action_dim), actual executed action prefix
    motion: np.ndarray  # (segments, H, 7+), start-relative eef trajectory
    episode_ids: np.ndarray
    init_state_ids: np.ndarray
    task_labels: np.ndarray
    segment_ids: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.actions)
        if self.states.ndim != 3 or self.actions.ndim != 3 or self.motion.ndim != 3:
            raise ValueError("states/actions/motion must be rank-3 arrays")
        if len(self.states) != n or len(self.motion) != n:
            raise ValueError("dataset arrays have different segment counts")
        if any(len(x) != n for x in (self.episode_ids, self.init_state_ids,
                                     self.task_labels, self.segment_ids)):
            raise ValueError("dataset labels have different segment counts")
        if self.states.shape[1] != self.actions.shape[1] or self.motion.shape[1] != self.actions.shape[1]:
            raise ValueError("dataset arrays must have a common fixed horizon")
        if not all(np.all(np.isfinite(x)) for x in (self.states, self.actions, self.motion)):
            raise ValueError("dataset values must be finite")

    @property
    def n_segments(self) -> int:
        return int(len(self.actions))

    @property
    def horizon(self) -> int:
        return int(self.actions.shape[1])


def _manifest_episodes(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    episodes = manifest.get("episodes", [])
    if isinstance(episodes, dict):
        return [dict(v, episode_id=k) if isinstance(v, dict) else {"episode_id": k}
                for k, v in episodes.items()]
    if not isinstance(episodes, list):
        raise ValueError("manifest episodes must be a list or object")
    return episodes


def load_recording(run_dir: str | Path, *, horizon: int = 10) -> PrimitiveDataset:
    """Read recorded episodes. Task names and skill IDs never enter model input arrays.

    Actions are taken from each `pred_chunk[:n_executed]`. The motion channels are
    reconstructed from the observed environment trajectory and step alignment. Each
    executed segment is resampled to `horizon`; task/episode IDs remain evaluation labels.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    states, actions, motions = [], [], []
    episode_ids, init_state_ids, tasks, segment_ids = [], [], [], []
    seen_episode_ids = set()
    seen_files = set()
    for episode in _manifest_episodes(manifest):
        if not isinstance(episode, dict):
            raise ValueError("each manifest episode must be an object")
        episode_id = str(episode.get("episode_id", episode.get("id", "")))
        if not episode_id:
            raise ValueError("episode missing episode_id")
        if episode_id in seen_episode_ids:
            raise ValueError(f"duplicate episode ID: {episode_id}")
        seen_episode_ids.add(episode_id)
        task = str(episode.get("task", episode.get("task_id", "unknown")))
        task_key = episode.get("task_id", episode.get("task", ""))
        suite_key = episode.get("suite", manifest.get("suite", ""))
        init_state_id = episode.get("init_state_id", episode.get("initial_state_id"))
        if init_state_id is None:
            raise ValueError(f"{episode_id}: manifest episode missing init_state_id")
        # Match G2's suite/task-local initial-state namespace.
        suite_part = "" if suite_key is None else str(suite_key)
        task_part = "" if task_key is None else str(task_key)
        init_group = namespace_init_state_id(suite_part, task_part, init_state_id)
        task = json.dumps([suite_part, task_part])
        npz_path = run_dir / "episodes" / f"{episode_id}.npz"
        if not npz_path.exists() and episode.get("file", episode.get("path")):
            npz_path = run_dir / str(episode.get("file", episode.get("path")))
        stat = npz_path.stat()
        file_id = (stat.st_dev, stat.st_ino)
        if file_id in seen_files:
            raise ValueError("multiple episode IDs refer to the same episode file")
        seen_files.add(file_id)
        with np.load(npz_path, allow_pickle=False) as data:
            required = ["step", "proprio", "pred_chunk", "n_executed", "segment_id",
                        "env_eef_pos", "env_eef_quat", "env_gripper"]
            missing = [name for name in required if name not in data]
            if missing:
                raise ValueError(f"{episode_id}: missing arrays {missing}")
            step = np.asarray(data["step"])
            proprio = np.asarray(data["proprio"], dtype=np.float32)
            chunks = np.asarray(data["pred_chunk"], dtype=np.float32)
            n_executed = np.asarray(data["n_executed"])
            segment = np.asarray(data["segment_id"])
            env = {name: np.asarray(data[name]) for name in
                   ("env_eef_pos", "env_eef_quat", "env_gripper")}
            row_lengths = {len(step), len(proprio), len(chunks), len(n_executed), len(segment)}
            if len(row_lengths) != 1:
                raise ValueError(f"{episode_id}: inference arrays have inconsistent row counts")
            if (step.ndim != 1 or not np.issubdtype(step.dtype, np.integer)
                    or np.any(step < 0) or np.any(step[1:] <= step[:-1])):
                raise ValueError(f"{episode_id}: step must be increasing nonnegative integers")
            if proprio.ndim != 2 or chunks.ndim != 3:
                raise ValueError(f"{episode_id}: proprio and pred_chunk must have ranks 2 and 3")
            offset = _environment_step_offset(manifest, episode, env)
            for row, (start, count) in enumerate(zip(step, n_executed)):
                if isinstance(count, (bool, np.bool_)) or int(count) != count or count <= 0:
                    raise ValueError(f"{episode_id}: refusing non-executed or invalid segment at row {row}")
                start, count = int(start), int(count)
                if count > chunks.shape[1]:
                    raise ValueError(f"{episode_id}: n_executed exceeds predicted chunk horizon")
                if row + 1 < len(step) and start + count > int(step[row + 1]):
                    raise ValueError(f"{episode_id}: executed segments overlap")
                total_steps = episode.get("steps")
                if total_steps is not None and start + count > total_steps:
                    raise ValueError(f"{episode_id}: executed segment exceeds episode steps")
                raw_actions = chunks[row, :count]
                raw_motion = _recorded_motion_segment(env, step=start, n_executed=count,
                                                      env_step_offset=offset)
                actions.append(_resample_sequence(raw_actions, horizon))
                motions.append(_resample_motion(raw_motion, horizon))
                # This same observed start state is the decoder's current-state condition.
                states.append(np.broadcast_to(proprio[row], (horizon, proprio.shape[1])).copy())
                episode_ids.append(episode_id)
                init_state_ids.append(init_group)
                tasks.append(task)
                segment_ids.append(str(segment[row]))
    if not actions:
        raise ValueError("recording contains no executed segments")
    return PrimitiveDataset(
        np.asarray(states, dtype=np.float32), np.asarray(actions, dtype=np.float32),
        np.asarray(motions, dtype=np.float32), np.asarray(episode_ids), np.asarray(init_state_ids),
        np.asarray(tasks),
        np.asarray(segment_ids),
    )


def split_by_episode(dataset: PrimitiveDataset, *, eval_fraction: float = 0.25,
                     seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Split whole initial-state groups; episodes/windows never cross train/eval."""
    if not 0 < eval_fraction < 1:
        raise ValueError("eval_fraction must be strictly between 0 and 1")
    episode_to_init: dict[str, str] = {}
    for episode_id, init_id in zip(dataset.episode_ids, dataset.init_state_ids, strict=True):
        previous = episode_to_init.setdefault(str(episode_id), str(init_id))
        if previous != str(init_id):
            raise ValueError(f"episode {episode_id} maps to multiple initial-state groups")
    groups = np.unique(dataset.init_state_ids)
    if len(groups) < 2:
        raise ValueError("at least two distinct initial-state groups are needed for evaluation")
    rng = np.random.default_rng(seed)
    shuffled = groups[rng.permutation(len(groups))]
    n_eval = min(len(groups) - 1, max(1, int(round(len(groups) * eval_fraction))))
    eval_groups = set(shuffled[:n_eval].tolist())
    eval_mask = np.asarray([group in eval_groups for group in dataset.init_state_ids])
    train_indices = np.flatnonzero(~eval_mask)
    eval_indices = np.flatnonzero(eval_mask)
    if set(dataset.episode_ids[train_indices]) & set(dataset.episode_ids[eval_indices]):
        raise AssertionError("episode leakage in primitive train/eval split")
    if set(dataset.init_state_ids[train_indices]) & set(dataset.init_state_ids[eval_indices]):
        raise AssertionError("initial-state leakage in primitive train/eval split")
    return train_indices, eval_indices


def make_synthetic_dataset(*, n_episodes: int = 8, segments_per_episode: int = 8,
                           horizon: int = 10, state_dim: int = 6, action_dim: int = 7,
                           seed: int = 0) -> PrimitiveDataset:
    """Create unlabeled-by-input motion families for fast CPU smoke tests."""
    if n_episodes < 2 or segments_per_episode < 1 or horizon < 2:
        raise ValueError("synthetic dataset needs >=2 episodes, >=1 segment, and horizon >=2")
    rng = np.random.default_rng(seed)
    states, actions, motions, eps, init_ids, tasks, segs = [], [], [], [], [], [], []
    for episode_index in range(n_episodes):
        task = f"task-{episode_index % 3}"
        for segment_index in range(segments_per_episode):
            family = (episode_index + segment_index) % 3
            phase = np.linspace(0, 1, horizon, dtype=np.float32)
            action = np.zeros((horizon, action_dim), dtype=np.float32)
            motion = np.zeros((horizon, 7), dtype=np.float32)
            direction = np.eye(3, dtype=np.float32)[family]
            action[:, :min(3, action_dim)] = direction[:min(3, action_dim)] * (0.1 + phase[:, None] * .1)
            if action_dim >= 7:
                action[:, 3:6] = (family - 1) * phase[:, None] * .05
                action[:, 6] = (1 if family == 2 else -1 if family == 1 else 0) * phase
            motion[:, :3] = direction * phase[:, None]
            motion[:, 3 + family % 3] = phase * (family - 1) * .2
            motion[:, 6] = action[:, 6] if action_dim >= 7 else 0
            state = rng.normal(size=state_dim).astype(np.float32) * .1
            states.append(np.broadcast_to(state, (horizon, state_dim)).copy())
            actions.append(action)
            motions.append(motion)
            eps.append(f"episode-{episode_index}")
            init_ids.append(f"synthetic::task-{episode_index % 3}::{episode_index}")
            tasks.append(task)
            segs.append(str(segment_index))
    return PrimitiveDataset(np.asarray(states), np.asarray(actions), np.asarray(motions),
                            np.asarray(eps), np.asarray(init_ids), np.asarray(tasks), np.asarray(segs))
