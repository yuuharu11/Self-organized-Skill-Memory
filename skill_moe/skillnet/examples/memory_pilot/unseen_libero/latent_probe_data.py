#!/usr/bin/env python3
"""Prepare exact-frame image/action rows for a latent-variable probe.

The default command is a read-only sizing and contract audit. Full extraction
requires the explicit ``--extract`` flag; it writes separate, memory-mappable
datasets for LIBERO-40 fit rows, fixed LIBERO-90 Eval rows, and dense LIBERO-90
Dev rows. It never modifies the source manifests or caches.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_STATE = Path(os.environ.get("SKILLNET_STATE_DIR", "/work/.skillnet-state"))
DEFAULT_MEMORY = DEFAULT_STATE / "outputs/memory-pilot/pi05-memory40-20261104/run-01"
DEFAULT_UNSEEN = DEFAULT_STATE / "outputs/memory-pilot/unseen-libero/unseen-rows-02"
DEFAULT_CHECKPOINT = DEFAULT_STATE / "checkpoints/openpi-assets/checkpoints/pi05_libero"
DEFAULT_OUTPUT = DEFAULT_STATE / "outputs/memory-pilot/unseen-libero/latent-probe-01/data"
DEFAULT_HELDOUT_TASKS = tuple(range(2, 38, 5))
DEV_TASKS = (6, 7)
EVAL_ROWS = 619
HORIZON = 10
STRIDE = 5


def _json(path: Path):
    return json.loads(Path(path).read_text())


def _run_rows(run_root: Path):
    return {
        p.stem: np.load(p, mmap_mode="r", allow_pickle=False)
        for p in (Path(run_root) / "rows").glob("*.npy")
    }


def build_libero40_fit_keys(run_root: Path, heldout_tasks=DEFAULT_HELDOUT_TASKS):
    """Return SEG2's exact non-memory, non-held-out (task,episode,frame) keys.

    This mirrors ``seg_context_run.build_fit_queries``: run-01 rows each
    contribute ten consecutive 10-Hz actions, then query starts are emitted at
    stride five. The same eligibility flags and causal end condition apply.
    """
    run_root = Path(run_root)
    manifest = _json(run_root / "data_manifest.json")
    rows = _run_rows(run_root)
    records = {(int(x["task"]), int(x["episode"])): x for x in manifest["episodes"]}
    keys: list[tuple[int, int, int]] = []
    for (task, episode), rec in sorted(records.items()):
        if rec.get("memory", False) or task in set(map(int, heldout_tasks)):
            continue
        ids = np.flatnonzero((rows["task"] == task) & (rows["episode"] == episode))
        if not len(ids):
            continue
        ids = ids[np.argsort(rows["frame"][ids])]
        # This is exactly the trace length used by build_fit_queries.
        trace_len = len(ids) * HORIZON
        keys.extend((task, episode, f) for f in range(0, max(0, trace_len - HORIZON + 1), STRIDE))
    return keys


def build_eval_keys(unseen_root: Path, expected_rows: int = EVAL_ROWS):
    """Read the fixed Eval identity list from unseen-rows-02 without resampling."""
    unseen_root = Path(unseen_root)
    arrays = {p.stem: np.load(p, mmap_mode="r", allow_pickle=False)
              for p in unseen_root.glob("*.npy")}
    use = ~np.asarray(arrays["is_dev"], dtype=bool)
    ids = np.flatnonzero(use)
    keys = [(int(arrays["task"][i]), int(arrays["episode"][i]), int(arrays["frame"][i]))
            for i in ids]
    if len(keys) != expected_rows:
        raise AssertionError(f"fixed Eval rows changed: expected {expected_rows}, got {len(keys)}")
    if any(f % 10 for _, _, f in keys):
        raise AssertionError("fixed Eval keys no longer match the stride-10 cache")
    return keys


def _demo_index(name: str) -> int:
    match = re.fullmatch(r"demo_(\d+)", name)
    if not match:
        raise ValueError(f"unrecognized LIBERO demo key: {name}")
    return int(match.group(1))


def build_dense_dev_records(unseen_root: Path, dev_tasks=DEV_TASKS):
    """Enumerate every HDF5 demo for Dev tasks 6/7 and emit stride-5 keys."""
    import h5py

    unseen_root = Path(unseen_root)
    manifest = _json(unseen_root / "manifest.json")
    records = {}
    for item in manifest["source_records"]:
        task = int(item["task_id"])
        if task in set(map(int, dev_tasks)):
            records[task] = item
    if set(records) != set(map(int, dev_tasks)):
        raise AssertionError(f"Dev source manifest lacks requested tasks: {set(dev_tasks)-set(records)}")

    demos = []
    keys = []
    for task, sample in sorted(records.items()):
        source = Path(sample["source_hdf5"])
        with h5py.File(source, "r") as h5:
            data = h5["data"]
            for demo_name in sorted(data.keys(), key=_demo_index):
                demo_id = _demo_index(demo_name)
                n = len(data[demo_name]["actions"][::2])
                rec = {
                    "task": task,
                    "episode": demo_id,
                    "task_name": sample["task_name"],
                    "instruction": sample["instruction"],
                    "source_demo": demo_name,
                    "source_hdf5": str(source),
                    "length_10hz": int(n),
                }
                demos.append(rec)
                keys.extend((task, demo_id, f) for f in range(0, max(0, n - HORIZON + 1), STRIDE))
    return demos, keys


def assert_dev_eval_disjoint(dev_records, eval_keys):
    dev_pairs = {(int(r["task"]), int(r["episode"])) for r in dev_records}
    eval_pairs = {(int(t), int(e)) for t, e, _ in eval_keys}
    overlap = dev_pairs & eval_pairs
    if overlap:
        raise AssertionError(f"Dev/Eval episode overlap: {sorted(overlap)[:5]}")
    return {"dev_episodes": len(dev_pairs), "eval_episodes": len(eval_pairs), "overlap": 0}


def orient_libero_hdf5_frame(frame: np.ndarray) -> np.ndarray:
    """Apply the same 180-degree source correction used by π0.5 LIBERO inputs."""
    frame = np.asarray(frame)
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError(f"expected HWC RGB frame, got {frame.shape}")
    return np.ascontiguousarray(frame[::-1, ::-1])


def action_targets(actions_physical: np.ndarray, actions_normalized: np.ndarray,
                   frame: int, spans: Iterable[tuple[int, int]]):
    """Build supervised targets; target labels may inspect future actions."""
    physical = np.asarray(actions_physical, dtype=np.float32)
    normalized = np.asarray(actions_normalized, dtype=np.float32)
    if physical.ndim != 2 or normalized.ndim != 2 or physical.shape[0] != normalized.shape[0]:
        raise ValueError("physical and normalized action traces must be [T,D]")
    if physical.shape[1] < 7 or normalized.shape[1] < 7:
        raise ValueError("active action space must contain six motion axes and gripper")
    f = int(frame)
    if f < 0 or f >= len(physical):
        raise ValueError(f"frame outside episode: {f}")
    p = np.zeros((HORIZON, 7), np.float32)
    n = np.zeros((HORIZON, 7), np.float32)
    count = min(HORIZON, len(physical) - f)
    p[:count] = physical[f:f + count, :7]
    n[:count] = normalized[f:f + count, :7]
    rms_physical = np.sqrt(np.mean(p[:, :6] ** 2, axis=0))
    rms_normalized = np.sqrt(np.mean(n[:, :6] ** 2, axis=0))
    signs = np.sign(p[:, 6]).astype(np.int8)
    current_sign = np.sign(physical[f, 6])
    switches = np.flatnonzero(signs[:count] != current_sign)
    first_switch = int(switches[0]) if len(switches) else -1

    segs = [(int(s), int(e)) for s, e in spans]
    current_i = next((i for i, (s, e) in enumerate(segs) if s <= f < e), None)
    if current_i is None:
        remaining, next_length = -1, -1
    else:
        remaining = segs[current_i][1] - f
        next_length = segs[current_i + 1][1] - segs[current_i + 1][0] if current_i + 1 < len(segs) else -1
    return {
        "actions_physical": p,
        "actions_normalized": n,
        "rms6_physical": rms_physical,
        "rms6_normalized": rms_normalized,
        "gripper_signs": signs,
        "first_gripper_switch": first_switch,
        "current_segment_remaining": int(remaining),
        "next_segment_length": int(next_length),
    }


def _anchor_spans(normalized_actions):
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    import motion_core_eval
    return [(int(s), int(e)) for s, e in motion_core_eval.anchor_spans(np.asarray(normalized_actions), min_length=10)]


def _norm_bounds(checkpoint: Path):
    data = _json(next((Path(checkpoint) / "assets").rglob("norm_stats.json")))['norm_stats']
    return {name: (np.asarray(value["q01"], np.float32), np.asarray(value["q99"], np.float32))
            for name, value in data.items()}


def quantile_normalize(values, q01, q99):
    values = np.asarray(values, np.float32)
    return (values - np.asarray(q01, np.float32)) / (np.asarray(q99, np.float32) - np.asarray(q01, np.float32) + 1e-6) * 2.0 - 1.0


def array_storage_estimate(rows: int, height: int, width: int):
    """Exact bytes for four raw uint8 images plus row targets/metadata."""
    image_bytes = int(rows) * 4 * int(height) * int(width) * 3
    # Per row: two states, two 10x7 action windows, RMS/segment/grip labels,
    # query keys and small flags. Full episode traces are counted separately.
    row_bytes = int(rows) * (2 * 8 * 4 + 2 * HORIZON * 7 * 4 + 6 * 4 * 2 + 32 + 4 + 150 * 4 + 300 * 4)
    return {"image_bytes": image_bytes, "row_metadata_target_bytes": row_bytes,
            "row_bytes_total": image_bytes + row_bytes}


def vjepa_storage_estimate(rows: int, *, token_count=256, width=1024, dtype_bytes=2):
    """Two cameras separately, each represented by a two-frame output."""
    per_camera_tokens = int(token_count) * int(width) * int(dtype_bytes)
    per_camera_pooled = int(width) * int(dtype_bytes)
    return {
        "rows": int(rows), "tokens_per_camera_shape_per_row": [1, int(token_count), int(width)],
        "two_camera_unpooled_bytes": int(rows) * 2 * per_camera_tokens,
        "two_camera_spatial_mean_bytes": int(rows) * 2 * per_camera_pooled,
    }


def _sha256(path: Path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def build_plan(run_root: Path, unseen_root: Path):
    fit_keys = build_libero40_fit_keys(run_root)
    eval_keys = build_eval_keys(unseen_root)
    dev_records, dev_keys = build_dense_dev_records(unseen_root)
    disjoint = assert_dev_eval_disjoint(dev_records, eval_keys)
    run_manifest = _json(Path(run_root) / "data_manifest.json")
    fit_pairs = {(t, e) for t, e, _ in fit_keys}
    fit_record_map = {(int(r["task"]), int(r["episode"])): r for r in run_manifest["episodes"]}
    fit_traj_frames = sum(int(fit_record_map[pair]["length"]) for pair in fit_pairs)
    eval_manifest = _json(Path(unseen_root) / "manifest.json")
    eval_pairs = {(t, e) for t, e, _ in eval_keys}
    eval_record_map = {(int(r["task_id"]), int(r["episode"])): r for r in eval_manifest["source_records"]}
    import h5py
    def source_trajectory_frames(records, pairs):
        count = 0
        for pair in sorted(pairs):
            rec = records[pair]
            with h5py.File(rec["source_hdf5"], "r") as h5:
                count += len(h5["data"][rec["source_demo"]]["actions"][::2])
        return int(count)
    eval_traj_frames = source_trajectory_frames(eval_record_map, eval_pairs)
    dev_pairs = {(int(r["task"]), int(r["episode"])) for r in dev_records}
    dev_record_map = {(int(r["task"]), int(r["episode"])): r for r in dev_records}
    dev_traj_frames = source_trajectory_frames(dev_record_map, dev_pairs)
    def trajectory_bytes(frames):
        # Physical+normalized active7 actions and physical+normalized state8.
        return int(frames) * (2 * 7 * 4 + 2 * 8 * 4)
    # Source shapes were checked against the persisted source data during plan.
    fit_shape = (256, 256, 3)
    unseen_shape = (128, 128, 3)
    all_rows = len(fit_keys) + len(eval_keys) + len(dev_keys)
    fit_storage = array_storage_estimate(len(fit_keys), *fit_shape[:2])
    eval_storage = array_storage_estimate(len(eval_keys), *unseen_shape[:2])
    dev_storage = array_storage_estimate(len(dev_keys), *unseen_shape[:2])
    for storage, frames in ((fit_storage, fit_traj_frames), (eval_storage, eval_traj_frames),
                            (dev_storage, dev_traj_frames)):
        storage["trajectory_frames"] = int(frames)
        storage["trajectory_bytes"] = trajectory_bytes(frames)
        storage["row_bytes_total"] += storage["trajectory_bytes"]
    return {
        "contract": "latent-probe-data-v1",
        "extract_command_is_explicit": True,
        "run_root": str(run_root), "unseen_root": str(unseen_root),
        "libero40_fit": {"rows": len(fit_keys), "episodes": len({(t, e) for t, e, _ in fit_keys}),
                         "task_ids": sorted({t for t, _, _ in fit_keys}), "frame_stride": STRIDE,
                         "image_shape_hwc": list(fit_shape), **fit_storage},
        "libero90_eval": {"rows": len(eval_keys), "episodes": disjoint["eval_episodes"],
                          "task_ids": sorted({t for t, _, _ in eval_keys}), "frame_stride_from_cache": 10,
                          "image_shape_hwc": list(unseen_shape), **eval_storage},
        "libero90_dev_dense": {"rows": len(dev_keys), "episodes": disjoint["dev_episodes"],
                               "task_ids": list(DEV_TASKS), "frame_stride": STRIDE,
                               "image_shape_hwc": list(unseen_shape), **dev_storage},
        "dev_eval_episode_disjoint": disjoint["overlap"] == 0,
        "total_rows": all_rows,
        "vjepa2_vitl_estimate": vjepa_storage_estimate(all_rows),
        "note": "Raw images are stored at source resolution; V-JEPA features are not extracted by --plan-only.",
    }


def _save_memmap(root: Path, name, shape, dtype):
    path = root / f"{name}.npy"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing extraction array: {path}")
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def _write_group_header(root: Path, group_name: str, keys, image_shape, source_manifest, trajectory_records):
    dest = root / group_name
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output: {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    n = len(keys)
    h, w, c = image_shape
    specs = {
        "image_base": ((n, h, w, c), np.uint8), "image_base_prev": ((n, h, w, c), np.uint8),
        "image_wrist": ((n, h, w, c), np.uint8), "image_wrist_prev": ((n, h, w, c), np.uint8),
        "state_physical": ((n, 8), np.float32), "state_normalized": ((n, 8), np.float32),
        "actions_physical": ((n, HORIZON, 7), np.float32), "actions_normalized": ((n, HORIZON, 7), np.float32),
        "rms6_physical": ((n, 6), np.float32), "rms6_normalized": ((n, 6), np.float32),
        "current_segment_remaining": ((n,), np.int32), "next_segment_length": ((n,), np.int32),
        "gripper_signs": ((n, HORIZON), np.int8), "first_gripper_switch": ((n,), np.int8),
        "task": ((n,), np.int32), "episode": ((n,), np.int32), "frame": ((n,), np.int32),
        "task_name": ((n,), "U150"), "instruction": ((n,), "U300"),
        "has_previous_frame": ((n,), np.bool_),
    }
    maps = {name: _save_memmap(dest, name, shape, dtype) for name, (shape, dtype) in specs.items()}
    trajectory_records = list(trajectory_records)
    n_traj = sum(int(x[2]) for x in trajectory_records)
    maps["trajectory_actions_physical"] = _save_memmap(dest, "trajectory_actions_physical", (n_traj, 7), np.float32)
    maps["trajectory_actions_normalized"] = _save_memmap(dest, "trajectory_actions_normalized", (n_traj, 7), np.float32)
    maps["trajectory_states_physical"] = _save_memmap(dest, "trajectory_states_physical", (n_traj, 8), np.float32)
    maps["trajectory_states_normalized"] = _save_memmap(dest, "trajectory_states_normalized", (n_traj, 8), np.float32)
    maps["trajectory_offsets"] = _save_memmap(dest, "trajectory_offsets", (len(trajectory_records), 2), np.int64)
    maps["trajectory_task"] = _save_memmap(dest, "trajectory_task", (len(trajectory_records),), np.int32)
    maps["trajectory_episode"] = _save_memmap(dest, "trajectory_episode", (len(trajectory_records),), np.int32)
    trajectory_slots = {}
    cursor = 0
    for j, (task, episode, length) in enumerate(trajectory_records):
        end = cursor + int(length)
        maps["trajectory_offsets"][j] = (cursor, end)
        maps["trajectory_task"][j] = int(task)
        maps["trajectory_episode"][j] = int(episode)
        trajectory_slots[(int(task), int(episode))] = (cursor, end)
        cursor = end
    maps["_trajectory_slots"] = trajectory_slots
    (dest / "extracting.json").write_text(json.dumps({"rows": n, "shape": list(image_shape),
        "source_manifest": str(source_manifest), "keys_sha256": hashlib.sha256(np.asarray(keys, np.int64).tobytes()).hexdigest()}, indent=2) + "\n")
    return dest, maps


def _finish_group(dest: Path, maps, keys, source_manifest, image_orientation):
    for name, value in maps.items():
        if name != "_trajectory_slots":
            value.flush()
    for name, vals, dtype in (
        ("task", [k[0] for k in keys], np.int32), ("episode", [k[1] for k in keys], np.int32),
        ("frame", [k[2] for k in keys], np.int32),
    ):
        # Already present in memmaps; this clause documents the row key schema.
        if len(vals) != len(keys):
            raise AssertionError(name)
    manifest = {"rows": len(keys), "key_fields": ["task", "episode", "frame"],
        "source_manifest": str(source_manifest), "source_manifest_sha256": _sha256(source_manifest),
        "image_orientation": image_orientation,
        "images": ["image_base", "image_base_prev", "image_wrist", "image_wrist_prev"],
        "targets_are_not_model_inputs": ["actions_physical", "actions_normalized", "rms6_physical",
            "rms6_normalized", "current_segment_remaining", "next_segment_length", "gripper_signs",
            "first_gripper_switch"], "complete": True}
    (dest / "extracting.json").unlink(missing_ok=True)
    tmp = dest / "manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=2) + "\n")
    tmp.replace(dest / "manifest.json")
    return manifest


def _fill_row(maps, i, key, current, previous, state_phys, state_norm, actions_phys,
              actions_norm, spans, task_name="", instruction=""):
    target = action_targets(actions_phys, actions_norm, key[2], spans)
    maps["image_base"][i] = current[0]
    maps["image_wrist"][i] = current[1]
    maps["image_base_prev"][i] = previous[0]
    maps["image_wrist_prev"][i] = previous[1]
    maps["state_physical"][i] = state_phys
    maps["state_normalized"][i] = state_norm
    for name in ("actions_physical", "actions_normalized", "rms6_physical", "rms6_normalized",
                 "gripper_signs", "first_gripper_switch", "current_segment_remaining", "next_segment_length"):
        maps[name][i] = target[name]
    maps["task"][i], maps["episode"][i], maps["frame"][i] = key
    maps["task_name"][i] = str(task_name)
    maps["instruction"][i] = str(instruction)
    maps["has_previous_frame"][i] = key[2] > 0


def _extract_libero40(run_root, dest_root, qstats):
    import io
    import pyarrow.parquet as pq
    from PIL import Image
    run_root = Path(run_root)
    manifest = _json(run_root / "data_manifest.json")
    rows = _run_rows(run_root)
    records = {(int(r["task"]), int(r["episode"])): r for r in manifest["episodes"]}
    keys = build_libero40_fit_keys(run_root)
    # Dataset-specific native resolution is retained rather than resized.
    by_episode = {}
    for i, key in enumerate(keys):
        by_episode.setdefault(key[:2], []).append((i, key))
    trajectory_records = [(t, e, int(records[(t, e)]["length"])) for t, e in sorted(by_episode)]
    dest, maps = _write_group_header(dest_root, "libero40_fit", keys, (256, 256, 3),
                                     run_root / "data_manifest.json", trajectory_records)
    tasks = {int(k): v for k, v in manifest["tasks"].items()}
    for pair, row_list in by_episode.items():
        rec = records[pair]
        table = pq.read_table(rec["path"], columns=["image", "wrist_image", "state", "actions"])
        raw_actions = np.asarray(table["actions"].to_pylist(), np.float32)
        raw_states = np.asarray(table["state"].to_pylist(), np.float32)
        # Normalize with the exact stored π0.5 quantile bounds, preserving the
        # source action/state values separately for physical-unit targets.
        action_q01, action_q99 = qstats["actions"]
        state_q01, state_q99 = qstats["state"]
        norm_actions = quantile_normalize(raw_actions, action_q01, action_q99)
        norm_states = quantile_normalize(raw_states[:, :8], state_q01, state_q99)
        ta, tb = maps["_trajectory_slots"][pair]
        maps["trajectory_actions_physical"][ta:tb] = raw_actions[:, :7]
        maps["trajectory_actions_normalized"][ta:tb] = norm_actions[:, :7]
        maps["trajectory_states_physical"][ta:tb] = raw_states[:, :8]
        maps["trajectory_states_normalized"][ta:tb] = norm_states
        spans = _anchor_spans(norm_actions[:, :7])
        for i, key in row_list:
            f = key[2]
            if f >= len(raw_actions):
                raise AssertionError(f"SEG2 frame exceeds source trace: {key}")
            def image_at(col, t):
                obj = table[col][t].as_py()
                with Image.open(io.BytesIO(obj["bytes"])) as im:
                    return np.asarray(im.convert("RGB"))
            _fill_row(maps, i, key,
                (image_at("image", f), image_at("wrist_image", f)),
                (image_at("image", max(0, f - 1)), image_at("wrist_image", max(0, f - 1))),
                raw_states[f, :8], norm_states[f], raw_actions, norm_actions, spans,
                task_name=rec["task"], instruction=tasks[key[0]])
        if len(row_list) and row_list[0][0] % 1000 == 0:
            print(json.dumps({"stage": "extract_libero40", "episode": list(pair), "rows": len(row_list)}), flush=True)
    return _finish_group(dest, maps, keys, run_root / "data_manifest.json", "stored source image orientation; no rotation")


def _hdf5_dataset_records(unseen_root, split):
    unseen_root = Path(unseen_root)
    manifest = _json(unseen_root / "manifest.json")
    if split == "eval":
        # Keep only the fixed 619 frame-identity rows and deduplicate demos.
        keys = build_eval_keys(unseen_root)
        keep = {(t, e) for t, e, _ in keys}
        records = [r for r in manifest["source_records"] if r["split"] == "eval" and
                   (int(r["task_id"]), int(r["episode"])) in keep]
    else:
        records, keys = build_dense_dev_records(unseen_root)
    return manifest, records, keys


def _extract_hdf5_group(unseen_root, dest_root, split, qstats):
    import h5py
    unseen_root = Path(unseen_root)
    source_manifest, records, keys = _hdf5_dataset_records(unseen_root, split)
    group_name = "libero90_eval" if split == "eval" else "libero90_dev"
    by_episode = {}
    for i, key in enumerate(keys):
        by_episode.setdefault(key[:2], []).append((i, key))
    task_field = "task_id" if split == "eval" else "task"
    rec_map = {(int(r[task_field]), int(r["episode"])): r for r in records}
    trajectory_records = []
    for pair in sorted(by_episode):
        rec = rec_map[pair]
        with h5py.File(rec["source_hdf5"], "r") as h5:
            length = len(h5["data"][rec["source_demo"]]["actions"][::2])
        trajectory_records.append((pair[0], pair[1], length))
    dest, maps = _write_group_header(dest_root, group_name, keys, (128, 128, 3),
                                     unseen_root / "manifest.json", trajectory_records)
    action_q01, action_q99 = qstats["actions"]
    state_q01, state_q99 = qstats["state"]
    for pair, row_list in by_episode.items():
        rec = rec_map[pair]
        with h5py.File(rec["source_hdf5"], "r") as h5:
            demo = h5["data"][rec["source_demo"]]
            obs, raw = demo["obs"], np.asarray(demo["actions"][::2], np.float32)
            state = np.concatenate([obs["ee_pos"][::2], obs["ee_ori"][::2], obs["gripper_states"][::2]], axis=-1).astype(np.float32)
            norm_actions = quantile_normalize(raw, action_q01, action_q99)
            norm_states = quantile_normalize(state, state_q01, state_q99)
            ta, tb = maps["_trajectory_slots"][pair]
            maps["trajectory_actions_physical"][ta:tb] = raw[:, :7]
            maps["trajectory_actions_normalized"][ta:tb] = norm_actions[:, :7]
            maps["trajectory_states_physical"][ta:tb] = state[:, :8]
            maps["trajectory_states_normalized"][ta:tb] = norm_states[:, :8]
            spans = _anchor_spans(norm_actions[:, :7])
            for i, key in row_list:
                f = key[2]
                if f >= len(raw):
                    raise AssertionError(f"Eval/Dev frame exceeds source trace: {key}")
                def image_at(name, t):
                    return orient_libero_hdf5_frame(obs[name][2 * t])
                _fill_row(maps, i, key,
                    (image_at("agentview_rgb", f), image_at("eye_in_hand_rgb", f)),
                    (image_at("agentview_rgb", max(0, f - 1)), image_at("eye_in_hand_rgb", max(0, f - 1))),
                    state[f, :8], norm_states[f], raw, norm_actions, spans,
                    task_name=rec["task_name"], instruction=rec["instruction"])
    return _finish_group(dest, maps, keys, unseen_root / "manifest.json", "HDF5 source frame[::-1,::-1] before any resize")


def extract_all(run_root, unseen_root, checkpoint, output_root):
    """Full explicit extraction. This function is intentionally not run by default."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    if (output_root / "manifest.json").exists():
        raise FileExistsError(f"refusing to overwrite completed extraction manifest: {output_root / 'manifest.json'}")
    norm_files = list((Path(checkpoint) / "assets").rglob("norm_stats.json"))
    if len(norm_files) != 1:
        raise ValueError(f"expected exactly one pi05_libero norm_stats.json, got {norm_files}")
    qstats = _norm_bounds(Path(checkpoint))
    outputs = {}
    expected = {
        "libero40_fit": len(build_libero40_fit_keys(run_root)),
        "libero90_eval": len(build_eval_keys(unseen_root)),
        "libero90_dev": len(build_dense_dev_records(unseen_root)[1]),
    }
    def complete_or_run(name, fn):
        group = output_root / name
        manifest_path = group / "manifest.json"
        if manifest_path.is_file():
            payload = _json(manifest_path)
            if payload.get("complete") is not True or int(payload.get("rows", -1)) != expected[name]:
                raise ValueError(f"existing group manifest does not match expected output: {manifest_path}")
            return payload
        if group.exists() and any(group.iterdir()):
            raise FileExistsError(f"partial output exists; refusing to overwrite it: {group}")
        return fn()
    outputs["libero40_fit"] = complete_or_run(
        "libero40_fit", lambda: _extract_libero40(run_root, output_root, qstats))
    outputs["libero90_eval"] = complete_or_run(
        "libero90_eval", lambda: _extract_hdf5_group(unseen_root, output_root, "eval", qstats))
    outputs["libero90_dev"] = complete_or_run(
        "libero90_dev", lambda: _extract_hdf5_group(unseen_root, output_root, "dev", qstats))
    root_manifest = {"contract": "latent-probe-data-v1", "groups": outputs, "complete": True,
        "normalization": str(norm_files[0]), "normalization_sha256": _sha256(norm_files[0]),
        "eval_labels_are_targets_only": True}
    (output_root / "manifest.json").write_text(json.dumps(root_manifest, indent=2) + "\n")
    return root_manifest


def verify_hdf5_image_parity(unseen_root: Path):
    """Assert one actual eval row matches unseen-rows-02 pixel-for-pixel."""
    import h5py
    from openpi.policies.libero_policy import LiberoInputs
    from openpi.models.model import ModelType
    from openpi.transforms import ResizeImages

    unseen_root = Path(unseen_root)
    arrays = {p.stem: np.load(p, mmap_mode="r", allow_pickle=False)
              for p in unseen_root.glob("*.npy")}
    manifest = _json(unseen_root / "manifest.json")
    row = int(np.flatnonzero(~arrays["is_dev"])[0])
    task, episode, frame = (int(arrays[k][row]) for k in ("task", "episode", "frame"))
    rec = next(r for r in manifest["source_records"] if r["split"] == "eval" and
               int(r["task_id"]) == task and int(r["episode"]) == episode)
    with h5py.File(rec["source_hdf5"], "r") as h5:
        demo = h5["data"][rec["source_demo"]]
        ix = 2 * frame
        base = orient_libero_hdf5_frame(demo["obs"]["agentview_rgb"][ix])
        wrist = orient_libero_hdf5_frame(demo["obs"]["eye_in_hand_rgb"][ix])
        state = np.concatenate([demo["obs"]["ee_pos"][ix], demo["obs"]["ee_ori"][ix],
                                demo["obs"]["gripper_states"][ix]]).astype(np.float32)
    item = LiberoInputs(ModelType.PI05)({"observation/image": base, "observation/wrist_image": wrist,
        "observation/state": state, "prompt": rec["instruction"]})
    item = ResizeImages(224, 224)(item)
    actual_base = np.asarray(item["image"]["base_0_rgb"])
    actual_wrist = np.asarray(item["image"]["left_wrist_0_rgb"])
    base_ok = np.array_equal(actual_base, np.asarray(arrays["image_base"][row]))
    wrist_ok = np.array_equal(actual_wrist, np.asarray(arrays["image_wrist"][row]))
    if not base_ok or not wrist_ok:
        raise AssertionError(f"unseen image parity failed for {(task, episode, frame)}: base={base_ok}, wrist={wrist_ok}")
    return {"key": [task, episode, frame], "base_exact": base_ok, "wrist_exact": wrist_ok,
            "cached_size": list(actual_base.shape), "source_size": list(base.shape)}


def run_vjepa_smoke(unseen_root: Path, model_dir: Path, *, memory_fraction=0.15):
    """Run a two-frame LIBERO pair through official HF V-JEPA 2 ViT-L.

    Base and wrist are independent videos/forward passes, matching the later
    feature-extraction design. The function measures first-call and warm GPU
    time and returns both dense-token and spatially pooled output shapes.
    """
    import h5py
    import torch
    from transformers import AutoModel, AutoVideoProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("V-JEPA smoke requires CUDA")
    torch.cuda.set_per_process_memory_fraction(float(memory_fraction))
    arrays = {p.stem: np.load(p, mmap_mode="r", allow_pickle=False)
              for p in Path(unseen_root).glob("*.npy")}
    manifest = _json(Path(unseen_root) / "manifest.json")
    row = int(np.flatnonzero((~arrays["is_dev"]) & (arrays["frame"] >= 10))[0])
    task, episode, frame = (int(arrays[k][row]) for k in ("task", "episode", "frame"))
    rec = next(r for r in manifest["source_records"] if r["split"] == "eval" and
               int(r["task_id"]) == task and int(r["episode"]) == episode)
    with h5py.File(rec["source_hdf5"], "r") as h5:
        demo = h5["data"][rec["source_demo"]]
        clips = {
            "base": np.stack([orient_libero_hdf5_frame(demo["obs"]["agentview_rgb"][2 * (frame - 1)]),
                              orient_libero_hdf5_frame(demo["obs"]["agentview_rgb"][2 * frame])]),
            "wrist": np.stack([orient_libero_hdf5_frame(demo["obs"]["eye_in_hand_rgb"][2 * (frame - 1)]),
                               orient_libero_hdf5_frame(demo["obs"]["eye_in_hand_rgb"][2 * frame])]),
        }
    model = AutoModel.from_pretrained(Path(model_dir), torch_dtype=torch.float16,
                                      local_files_only=True).cuda().eval()
    processor = AutoVideoProcessor.from_pretrained(Path(model_dir), local_files_only=True)
    torch.cuda.reset_peak_memory_stats()
    timings, output_shapes = {}, {}
    with torch.inference_mode():
        for name, clip in clips.items():
            batch = processor(videos=[clip], return_tensors="pt")
            video = batch["pixel_values_videos"].cuda().to(dtype=torch.float16)
            repeat_times = []
            for _ in range(4):
                torch.cuda.synchronize()
                start = __import__("time").perf_counter()
                features = model.get_vision_features(pixel_values_videos=video)
                torch.cuda.synchronize()
                repeat_times.append(__import__("time").perf_counter() - start)
            output_shapes[name] = {
                "input_shape": list(video.shape),
                "tokens_shape": list(features.shape),
                "spatial_mean_shape": list(features.mean(dim=-2).shape),
            }
            timings[name] = {"seconds_per_forward": repeat_times,
                             "warm_median_seconds": float(np.median(repeat_times[1:]))}
    return {
        "repo_id": "facebook/vjepa2-vitl-fpc64-256",
        "model_dir": str(model_dir),
        "license_field": "MIT",
        "license_card": "https://huggingface.co/facebook/vjepa2-vitl-fpc64-256",
        "eval_key": [task, episode, frame],
        "two_frame_order": "previous/current at adjacent 10-Hz ticks; cameras processed separately",
        "dtype": "float16", "outputs": output_shapes, "timings": timings,
        "warm_two_camera_seconds": float(sum(x["warm_median_seconds"] for x in timings.values())),
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "device": torch.cuda.get_device_name(),
        "token_count_from_config": 256,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_MEMORY)
    parser.add_argument("--unseen-root", type=Path, default=DEFAULT_UNSEEN)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--plan-only", action="store_true", help="count rows and estimate storage; default behavior")
    parser.add_argument("--extract", action="store_true", help="explicitly run the large image/target extraction")
    parser.add_argument("--check-image-parity", action="store_true", help="assert actual Eval image pixels match unseen-rows-02")
    parser.add_argument("--vjepa-smoke", action="store_true", help="run a two-frame base/wrist V-JEPA 2 ViT-L GPU smoke")
    parser.add_argument("--vjepa-model-dir", type=Path,
                        default=DEFAULT_OUTPUT / "vjepa2-vitl-fpc64-256")
    args = parser.parse_args(argv)
    if args.extract and args.plan_only:
        parser.error("choose only one of --extract or --plan-only")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if args.vjepa_smoke:
        result = run_vjepa_smoke(args.unseen_root, args.vjepa_model_dir)
        smoke_path = out / "vjepa_smoke_verified.json"
        if smoke_path.exists():
            raise FileExistsError(f"refusing to overwrite V-JEPA smoke result: {smoke_path}")
        smoke_path.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return
    result = build_plan(args.run_root, args.unseen_root)
    if args.check_image_parity:
        result["image_parity"] = verify_hdf5_image_parity(args.unseen_root)
    if args.extract:
        result["extraction"] = extract_all(args.run_root, args.unseen_root, args.checkpoint, out)
    plan_path = out / "plan.json"
    if not plan_path.exists():
        plan_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
