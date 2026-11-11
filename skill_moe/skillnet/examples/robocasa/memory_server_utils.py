"""Small, dependency-light helpers for the RoboCasa memory server."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def replace_gripper_only(bridge_actions, policy_actions):
    """Replace fork-order action channel 6 while preserving every other value."""
    bridge = np.asarray(bridge_actions, np.float32)
    policy = np.asarray(policy_actions, np.float32)
    if bridge.ndim != 2 or bridge.shape[1] != 12:
        raise ValueError(f"bridge actions must be [T,12], got {bridge.shape}")
    if policy.shape != bridge.shape:
        raise ValueError(f"policy actions must match {bridge.shape}, got {policy.shape}")
    out = bridge.copy()
    out[:, 6] = policy[:, 6]
    return out


class ClientHistory:
    """Last executed actions for one websocket client/episode.

    Rows are left-zero-padded. The mask is explicit even though the current
    P5 adapter represents missing history with zero rows.
    """

    def __init__(self, horizon: int = 10):
        self.horizon = int(horizon)
        self.episode_id = None
        self.rows = np.zeros((self.horizon, 12), np.float32)
        self.mask = np.zeros((self.horizon,), bool)
        self.pending = np.zeros((0, 12), np.float32)
        self.calls = 0

    def begin_request(self, episode_id, episode_start: bool):
        episode_id = str(episode_id)
        if episode_start or self.episode_id != episode_id:
            self.episode_id = episode_id
            self.rows.fill(0)
            self.mask.fill(False)
            self.pending = np.zeros((0, 12), np.float32)
            self.calls = 0
        elif len(self.pending):
            self.append_executed(self.pending)
            self.pending = np.zeros((0, 12), np.float32)
        return self.rows.copy(), self.mask.copy(), self.calls

    def append_executed(self, actions):
        actions = np.asarray(actions, np.float32)
        if actions.ndim != 2 or actions.shape[1] != 12:
            raise ValueError(f"executed actions must be [T,12], got {actions.shape}")
        if not len(actions):
            return
        joined = np.concatenate([self.rows[self.mask], actions], axis=0)[-self.horizon :]
        self.rows.fill(0)
        self.mask.fill(False)
        self.rows[-len(joined) :] = joined
        self.mask[-len(joined) :] = True

    def set_pending(self, actions):
        actions = np.asarray(actions, np.float32)
        if actions.ndim != 2 or actions.shape[1] != 12:
            raise ValueError(f"pending actions must be [T,12], got {actions.shape}")
        self.pending = actions.copy()
        self.calls += 1


def fit_scale(candidate, reference, *, mode: str) -> np.ndarray:
    """Scale only EEF translation/rotation magnitudes; preserve directions.

    Inputs are physical fork-order action chunks [T,12]. ``saved`` preserves
    the stored magnitude; ``s-shared`` fits one nonnegative least-squares
    multiplier over all six EEF channels; ``s-sep`` fits position and rotation
    independently. Gripper, base and control mode are untouched.
    """
    candidate = np.asarray(candidate, np.float32)
    reference = np.asarray(reference, np.float32)
    if candidate.ndim != 2 or candidate.shape[1] != 12:
        raise ValueError(f"candidate must be [T,12], got {candidate.shape}")
    if reference.ndim != 2 or reference.shape[1] < 6 or not len(reference):
        raise ValueError(f"reference must be [T,>=6], got {reference.shape}")
    if mode == "saved":
        return candidate.copy()
    if mode not in {"s-shared", "s-sep"}:
        raise ValueError(f"unknown scale mode: {mode}")
    q = reference[:, :12] if reference.shape[1] >= 12 else reference
    if len(q) != len(candidate):
        positions = np.linspace(0, len(q) - 1, len(candidate))
        lo = np.floor(positions).astype(int)
        hi = np.minimum(lo + 1, len(q) - 1)
        frac = (positions - lo)[:, None]
        q = q[lo] * (1.0 - frac) + q[hi] * frac
    out = candidate.copy()
    groups = (slice(0, 6),) if mode == "s-shared" else (slice(0, 3), slice(3, 6))
    for group in groups:
        x = candidate[:, group].astype(np.float64).reshape(-1)
        y = q[:, group].astype(np.float64).reshape(-1)
        den = float(np.dot(x, x))
        if den > 1e-12:
            out[:, group] *= max(0.0, float(np.dot(x, y)) / den)
    return out


def completed_trials_from_shard(path):
    """Read complete task/trial identities for safe resume."""
    path = Path(path)
    if not path.exists():
        return set(), []
    rows = json.loads(path.read_text())
    if not isinstance(rows, list):
        raise ValueError(f"worker shard must contain a JSON list: {path}")
    complete = {(str(r["task"]), int(r["trial"])) for r in rows if r.get("status") == "complete"}
    return complete, rows


def merge_worker_shards(directory):
    """Merge all worker shards by task/trial, including across worker counts."""
    directory = Path(directory)
    records = {}
    for shard in sorted(directory.glob("worker-*.json")):
        _, rows = completed_trials_from_shard(shard)
        for row in rows:
            key = (str(row["task"]), int(row["trial"]))
            old = records.get(key)
            if old is not None and old != row:
                raise ValueError(f"conflicting records for {key} across worker shards")
            records[key] = row
    return list(records.values())
