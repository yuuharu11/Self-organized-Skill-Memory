"""S2 LIBERO-90 cache reader and Optimus-R's task/progress metadata adapter."""
from __future__ import annotations

import hashlib
from pathlib import Path

import jax.numpy as jnp
import numpy as np


def load_cache(path: str | Path) -> dict[str, np.ndarray]:
    """Load the Stage-A fields from the existing S2 export without rewriting it."""
    with np.load(path, allow_pickle=True) as archive:
        data = {name: archive[name] for name in archive.files}
    required = {"state", "actions", "skills", "skill_mask", "tokenized_prompt", "tokenized_prompt_mask",
                "image_base", "image_left_wrist", "episode_id", "instruction", "source_file", "frame",
                "suite", "sampling_stride", "normalization_sha256", "checkpoint"}
    missing = sorted(required - data.keys())
    if missing:
        raise ValueError(f"S2 LIBERO-90 export is missing Optimus-R fields: {missing}")
    if not np.all(data["suite"].astype(str) == "libero90"):
        raise ValueError("Optimus-R Stage A accepts only LIBERO-90 source rows")
    if "image_mask_right_wrist_0_rgb" in data and np.any(data["image_mask_right_wrist_0_rgb"]):
        raise ValueError("this π0.5 LIBERO-90 export has no right-wrist camera; reject valid-view rows")
    if data["actions"].ndim != 3 or data["actions"].shape[1:] != (10, 32):
        raise ValueError(f"expected normalized [N,10,32] actions, got {data['actions'].shape}")
    n = len(data["actions"])
    for name in required - {"suite", "sampling_stride", "normalization_sha256", "checkpoint"}:
        if len(data[name]) != n:
            raise ValueError(f"{name} row count differs from actions")
    return data


def official_task_ids(source_files: np.ndarray) -> np.ndarray:
    """Map each exported raw task filename to the official LIBERO-90 task order."""
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()["libero_90"]()
    filename_to_task = {f"{name}_demo.hdf5": task_id
                        for task_id, name in enumerate(suite.get_task_names())}
    missing = sorted(set(Path(str(item)).name for item in source_files) - filename_to_task.keys())
    if missing:
        raise ValueError(f"cached source files do not map to official LIBERO-90 task IDs: {missing[:5]}")
    return np.asarray([filename_to_task[Path(str(item)).name] for item in source_files], np.int32)


def contrastive_metadata(data: dict[str, np.ndarray], *, task_ids: np.ndarray | None = None):
    """Return task IDs, approximate progress, episode IDs, and within-episode row time.

    The S2 cache retains every tenth environment step, not the complete episode
    length. Progress is therefore normalized over retained complete H=10 rows.
    The temporal-positive window is six retained samples, matching the paper's
    index window in the available cache's sample units.
    """
    n = len(data["actions"])
    task_ids = official_task_ids(data["source_file"]) if task_ids is None else np.asarray(task_ids, np.int32)
    if task_ids.shape != (n,):
        raise ValueError("task_ids must have shape [N]")
    episode_labels = data["episode_id"].astype(str)
    episode_names, episode_ids = np.unique(episode_labels, return_inverse=True)
    frame = data["frame"].astype(np.int64)
    stride = int(np.asarray(data["sampling_stride"]).item())
    progress = np.zeros(n, np.float32)
    sample_indices = np.zeros(n, np.int32)
    for episode_number in range(len(episode_names)):
        rows = np.flatnonzero(episode_ids == episode_number)
        order = rows[np.argsort(frame[rows], kind="stable")]
        episode_frames = frame[order]
        sample_indices[order] = np.arange(len(order), dtype=np.int32)
        duration_estimate = max(int(episode_frames[-1]) + stride, stride)
        progress[order] = np.clip(episode_frames / duration_estimate, 0.0, 1.0)
    return task_ids, progress, episode_ids.astype(np.int32), sample_indices


def observation_batch(rows: dict[str, np.ndarray]):
    """Convert one selected row batch into the checkpoint-compatible observation."""
    from openpi.models import model as _model

    def image(value):
        value = jnp.asarray(value)
        if value.dtype == jnp.uint8:
            return value.astype(jnp.float32) / 255.0 * 2.0 - 1.0
        return value.astype(jnp.float32)

    base = image(rows["image_base"])
    wrist = image(rows["image_left_wrist"])
    zero_image = jnp.zeros_like(base)
    batch = base.shape[0]
    return _model.Observation(
        images={"base_0_rgb": base, "left_wrist_0_rgb": wrist, "right_wrist_0_rgb": zero_image},
        image_masks={"base_0_rgb": jnp.ones((batch,), jnp.bool_),
                     "left_wrist_0_rgb": jnp.ones((batch,), jnp.bool_),
                     "right_wrist_0_rgb": jnp.zeros((batch,), jnp.bool_)},
        state=jnp.asarray(rows["state"], jnp.float32),
        skills=jnp.asarray(rows["skills"], jnp.int32),
        skill_mask=jnp.asarray(rows["skill_mask"], jnp.bool_),
        tokenized_prompt=jnp.asarray(rows["tokenized_prompt"], jnp.int32),
        tokenized_prompt_mask=jnp.asarray(rows["tokenized_prompt_mask"], jnp.bool_),
    )


def data_fingerprint(data: dict[str, np.ndarray]) -> str:
    """Stable provenance digest from cache identity and row metadata, not image bytes."""
    digest = hashlib.sha256()
    for key in ("source_dataset", "checkpoint", "normalization_sha256", "sampling_stride"):
        if key in data:
            digest.update(str(np.asarray(data[key]).item()).encode())
    digest.update(np.ascontiguousarray(data["episode_id"]).view(np.uint8))
    digest.update(np.ascontiguousarray(data["frame"]).view(np.uint8))
    return digest.hexdigest()
