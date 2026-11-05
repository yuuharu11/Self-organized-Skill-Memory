"""Dataset-row-indexed action memories and fixed-window segmentation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def cut_fixed_windows(actions: np.ndarray, *, horizon: int = 10, stride: int = 10,
                      include_tail: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Cut `[time,action_dim]` episodes into windows and return start offsets.

    The default drops an incomplete tail, matching fixed 10-step memory rows.
    """
    a = np.asarray(actions)
    if a.ndim != 2 or horizon <= 0 or stride <= 0:
        raise ValueError("expected [time,dim] actions and positive horizon/stride")
    if include_tail:
        starts = np.arange(0, len(a) - horizon + 1, stride, dtype=np.int64)
        if len(a) >= horizon and (not len(starts) or starts[-1] + horizon < len(a)):
            starts = np.r_[starts, len(a) - horizon].astype(np.int64)
    else:
        starts = np.arange(0, len(a) - horizon + 1, stride, dtype=np.int64)
    if not len(starts):
        return np.empty((0, horizon, a.shape[-1]), a.dtype), starts
    windows = np.stack([a[s:s + horizon] for s in starts])
    return windows, starts


@dataclass
class MotionMemory:
    """Memory values are raw actions; identifiers always refer to dataset rows."""

    row_ids: np.ndarray
    actions: np.ndarray
    tasks: np.ndarray
    episodes: np.ndarray

    def __post_init__(self):
        self.row_ids = np.asarray(self.row_ids, np.int64)
        self.actions = np.asarray(self.actions, np.float32)
        self.tasks = np.asarray(self.tasks, np.int32)
        self.episodes = np.asarray(self.episodes, np.int64)
        n = len(self.row_ids)
        if self.actions.ndim != 3 or any(len(x) != n for x in (self.actions, self.tasks, self.episodes)):
            raise ValueError("memory arrays must align; actions are [N,H,D]")
        if len(np.unique(self.row_ids)) != n:
            raise ValueError("dataset row IDs must be unique")

    def exclude_tasks(self, task_ids) -> "MotionMemory":
        keep = ~np.isin(self.tasks, np.asarray(list(task_ids), np.int32))
        return MotionMemory(self.row_ids[keep], self.actions[keep], self.tasks[keep], self.episodes[keep])

    def exclude_episodes(self, task_ids, episode_ids=None) -> "MotionMemory":
        """Filter `(task,episode)` pairs; a scalar episode applies to each task only."""
        if episode_ids is None:
            pairs = set((int(t), int(e)) for t, e in task_ids)
            drop = np.asarray([(int(t), int(e)) in pairs for t, e in zip(self.tasks, self.episodes)])
        else:
            pairs = set(zip(np.asarray(task_ids).tolist(), np.asarray(episode_ids).tolist()))
            drop = np.asarray([(int(t), int(e)) in pairs for t, e in zip(self.tasks, self.episodes)])
        keep = ~drop
        return MotionMemory(self.row_ids[keep], self.actions[keep], self.tasks[keep], self.episodes[keep])

    def drafts(self, dataset_actions: np.ndarray, dataset_row_ids: np.ndarray) -> np.ndarray:
        """Fetch action windows by dataset row ID, never by memory-local offset."""
        ids = np.asarray(dataset_row_ids, np.int64)
        if ids.size and not np.isin(ids, self.row_ids).all():
            raise KeyError("requested dataset row ID is not present in this memory")
        if ids.size and (ids.min() < 0 or ids.max() >= len(dataset_actions)):
            raise IndexError("dataset row ID is outside action array")
        return np.asarray(dataset_actions[ids], np.float32)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(path)
        np.savez_compressed(path, row_ids=self.row_ids, actions=self.actions,
                            tasks=self.tasks, episodes=self.episodes)

    @classmethod
    def load(cls, path: str | Path) -> "MotionMemory":
        with np.load(path, allow_pickle=False) as z:
            return cls(z["row_ids"], z["actions"], z["tasks"], z["episodes"])

    @classmethod
    def from_dataset_rows(cls, row_ids, actions, tasks, episodes) -> "MotionMemory":
        ids = np.asarray(row_ids, np.int64)
        return cls(ids, np.asarray(actions)[ids], np.asarray(tasks)[ids], np.asarray(episodes)[ids])
