"""Load and join cached LIBERO-10 light features with existing suite caches.

The cache files follow ``light.extract``: one NPZ per episode, sampled every
five environment steps, with float16 frozen-VLM features and float32 state and
action trajectories.  Camera-language features are optional for older caches;
their values are NaN-filled and accompanied by a per-modality availability
mask when caches are joined.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def _root(path: str | Path) -> Path:
    path = Path(path)
    return path / "feature-cache" if (path / "feature-cache").is_dir() else path


def _axis_angle_to_xyzw(rv: np.ndarray) -> np.ndarray:
    rv = np.asarray(rv, dtype=np.float64)
    angle = np.linalg.norm(rv, axis=-1, keepdims=True)
    half = angle * 0.5
    factor = np.divide(np.sin(half), angle, out=np.full_like(angle, 0.5), where=angle > 1e-12)
    return np.concatenate([rv * factor, np.cos(half)], axis=-1)


def initial_state_id(state: np.ndarray, task_index: int) -> str:
    """Stable, task-scoped identity shared by extraction and cache joining."""
    first = np.round(np.asarray(state, dtype=np.float32).reshape(-1), 4).astype("<f4")
    digest = hashlib.blake2s(first.tobytes(), digest_size=8).hexdigest()
    return f"task{int(task_index):02d}-{digest}"


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    av, aw, bv, bw = a[..., :3], a[..., 3:4], b[..., :3], b[..., 3:4]
    return np.concatenate([aw * bv + bw * av + np.cross(av, bv), aw * bw - np.sum(av * bv, axis=-1, keepdims=True)], axis=-1)


def _motion(state: np.ndarray, steps: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Build G2-compatible relative xyz/quaternion/gripper future trajectories."""
    state = np.asarray(state, dtype=np.float32).reshape(len(state), -1)
    steps = np.asarray(steps, dtype=np.int64)
    pos = state[:, :3]
    quat = _axis_angle_to_xyzw(state[:, 3:6])
    grip = state[:, 6:8]
    out = np.full((len(steps), horizon, 7 + grip.shape[1]), np.nan, np.float32)
    mask = np.zeros((len(steps), horizon), bool)
    base = int(np.min(steps))
    for row, step in enumerate(steps):
        i = int(step) - base
        future = i + np.arange(1, horizon + 1)
        if i < 0 or future[-1] >= len(state):
            continue
        q0 = quat[i]
        q0_inv = np.concatenate([-q0[:3], q0[3:]])
        relative = _quat_mul(q0_inv, quat[future])
        relative /= np.maximum(np.linalg.norm(relative, axis=-1, keepdims=True), 1e-12)
        out[row] = np.concatenate([pos[future] - pos[i], relative, grip[future] - grip[i]], axis=-1)
        mask[row] = True
    return out, mask


def _action_chunks(action: np.ndarray, steps: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    action = np.asarray(action, dtype=np.float32)
    steps = np.asarray(steps, dtype=np.int64)
    out = np.zeros((len(steps), horizon, action.shape[-1]), np.float32)
    mask = np.zeros((len(steps), horizon), bool)
    base = int(np.min(steps))
    for row, step in enumerate(steps):
        start = int(step) - base
        n = min(horizon, max(0, len(action) - start))
        if n:
            out[row, :n] = action[start:start + n]
            mask[row, :n] = True
    return out, mask


def _load_camera_rows(camera_dir: Path | None, episode_id: str, steps: np.ndarray, keys: tuple[str, ...]):
    if camera_dir is None:
        return {}
    path = camera_dir / "episodes" / f"{episode_id}.npz"
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=False) as data:
        lookup = {int(step): i for i, step in enumerate(data["step"])}
        indices = [lookup.get(int(step), -1) for step in steps]
        if any(i < 0 for i in indices):
            raise ValueError(f"camera cache {path} misses sampled steps")
        return {key: np.asarray(data[key][indices]) for key in keys if key in data}


def read_cache(cache_dir: str | Path, *, horizon: int = 25,
               camera_cache: str | Path | None = None,
               language_camera_cache: str | Path | None = None,
               split: str | None = None) -> dict[str, Any]:
    """Read a feature cache into aligned row arrays; optionally align camera caches."""
    root = _root(cache_dir)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    feature_cam = Path(camera_cache) if camera_cache is not None else None
    lang_cam = Path(language_camera_cache) if language_camera_cache is not None else None
    rows: dict[str, list] = {k: [] for k in (
        "episode_id", "init_state_id", "split", "suite", "task", "task_index", "step", "f", "k",
        "proprio", "action_chunk", "action_mask", "motion", "motion_mask", "base", "wrist",
        "base_lang", "wrist_lang")}
    camera_names = ("base", "wrist", "base_lang", "wrist_lang")
    camera_available: dict[str, list[np.ndarray]] = {name: [] for name in camera_names}
    for entry in manifest.get("episodes", []):
        if split is not None and entry.get("split") != split:
            continue
        episode_id = str(entry.get("episode_id", entry.get("id")))
        ep_path = Path(entry.get("cache", root / "episodes" / f"{episode_id}.npz"))
        if not ep_path.exists():
            ep_path = root / "episodes" / f"{episode_id}.npz"
        with np.load(ep_path, allow_pickle=False) as data:
            steps = np.asarray(data["step"], dtype=np.int32)
            env_steps = np.asarray(data["env_step"], dtype=np.int32) if "env_step" in data else steps
            action = np.asarray(data["action"], dtype=np.float32)
            state = np.asarray(data["env_state"], dtype=np.float32) if "env_state" in data else np.asarray(data["proprio"], dtype=np.float32)
            sampled_prop = np.asarray(data["proprio"], dtype=np.float32)
            f = np.asarray(data["f"])
            k = np.asarray(data["k"])
            if f.dtype != np.float16 or k.dtype != np.float16:
                raise ValueError(f"{ep_path}: f/k must be float16")
            if f.ndim != 2 or k.shape != f.shape or len(f) != len(steps):
                raise ValueError(f"{ep_path}: f/k/step shapes are inconsistent")
            chunks, action_mask = _action_chunks(action, steps, horizon)
            motion, motion_mask = _motion(state, steps, horizon)
            init_id = str(entry.get("init_state_id", ""))
            if not init_id:
                init_id = initial_state_id(state[0], int(entry.get("task_index", -1)))
            base_rows = _load_camera_rows(feature_cam, episode_id, steps, ("base", "wrist"))
            lang_rows = _load_camera_rows(lang_cam, episode_id, steps, ("base_lang", "wrist_lang"))
            split_name = str(entry.get("split", "existing"))
            values = {
                "episode_id": np.full(len(steps), episode_id, dtype=object),
                "init_state_id": np.full(len(steps), init_id, dtype=object),
                "split": np.full(len(steps), split_name, dtype=object),
                "suite": np.full(len(steps), str(entry.get("suite", manifest.get("suite", ""))), dtype=object),
                "task": np.full(len(steps), str(entry.get("task", "")), dtype=object),
                "task_index": np.full(len(steps), int(entry.get("task_index", -1)), dtype=np.int32),
                "step": steps,
                "f": f, "k": k, "proprio": sampled_prop,
                "action_chunk": chunks, "action_mask": action_mask,
                "motion": motion, "motion_mask": motion_mask,
                "base": base_rows.get("base"), "wrist": base_rows.get("wrist"),
                "base_lang": lang_rows.get("base_lang"), "wrist_lang": lang_rows.get("wrist_lang"),
            }
            for name, value in values.items():
                if name in camera_names:
                    if value is None:
                        rows[name].append(None)
                        camera_available[name].append(np.zeros(len(steps), dtype=bool))
                    else:
                        valid = np.isfinite(np.asarray(value).reshape(len(steps), -1)).all(axis=1)
                        rows[name].append(value)
                        camera_available[name].append(valid)
                    continue
                if value is not None:
                    rows[name].append(value)
    if not rows["step"]:
        raise ValueError(f"no rows selected from {root}")
    # Keep camera modalities aligned when a particular episode or camera cache
    # is absent. Missing values are NaN-filled; per-row masks carry validity.
    for name in camera_names:
        parts = rows[name]
        present = [part for part in parts if part is not None]
        if present:
            shape = np.asarray(present[0]).shape[1:]
            rows[name] = [
                np.full((len(camera_available[name][i]), *shape), np.nan, dtype=np.float32)
                if part is None else part
                for i, part in enumerate(parts)
            ]
        else:
            rows[name] = []
    out = {k: np.concatenate(v, axis=0) for k, v in rows.items() if v and v[0] is not None}
    for name in camera_names:
        out[f"has_{name}"] = np.concatenate(camera_available[name], axis=0)
    out["manifest"] = manifest
    return out


def join_existing_and_libero10(existing_cache: str | Path, libero10_root: str | Path, *, horizon: int = 25,
                               existing_camera_cache: str | Path | None = None) -> dict[str, Any]:
    """Concatenate existing suites with LIBERO-10, preserving missing camera-language masks."""
    new_root = Path(libero10_root)
    existing = read_cache(existing_cache, horizon=horizon, camera_cache=existing_camera_cache)
    fresh = read_cache(new_root / "feature-cache", horizon=horizon,
                       camera_cache=new_root / "camera-features",
                       language_camera_cache=new_root / "camera-language-features")
    metadata = {"manifest": [existing.pop("manifest"), fresh.pop("manifest")]}
    names = set(existing) | set(fresh)
    n_old, n_new = len(existing["step"]), len(fresh["step"])
    result = {}
    for name in names:
        a, b = existing.get(name), fresh.get(name)
        if a is None and b is None:
            continue
        if a is None or b is None:
            ref = b if a is None else a
            shape, dtype = ref.shape[1:], ref.dtype
            missing = np.full((n_old if a is None else n_new, *shape), np.nan, dtype=np.float32)
            a = missing if a is None else a
            b = missing if b is None else b
        result[name] = np.concatenate([a, b], axis=0)
    for name in ("base", "wrist", "base_lang", "wrist_lang"):
        result[f"has_{name}"] = np.concatenate([existing[f"has_{name}"], fresh[f"has_{name}"]])
    result["manifest"] = metadata
    return result

