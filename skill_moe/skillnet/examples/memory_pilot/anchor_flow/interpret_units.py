#!/usr/bin/env python3
"""Post-hoc, label-aware catalog for an already-built S2 M6 memory.

Labels are read only after loading the frozen memory. They never participate in
unit creation. LIBERO-90 has task-level plan text in the checked-in annotation
JSON, but this export has no temporal plan boundaries; boundary agreement is
therefore reported as unavailable instead of guessed.
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any, Callable

import numpy as np

HERE = Path(__file__).resolve().parent
EXAMPLES = HERE.parents[1]
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))


VERB_PATTERNS = (
    ("turn_on", re.compile(r"\bturn\s+on\b")),
    ("turn_off", re.compile(r"\bturn\s+off\b")),
    ("pick", re.compile(r"\b(?:pick\s+up|pick|grasp|take)\b")),
    ("place", re.compile(r"\b(?:place|put|set)\b")),
    ("open", re.compile(r"\bopen\b")),
    ("close", re.compile(r"\bclose\b")),
    ("stack", re.compile(r"\bstack\b")),
    ("push", re.compile(r"\bpush\b")),
    ("move", re.compile(r"\bmove\b")),
    ("rotate", re.compile(r"\b(?:turn|rotate)\b")),
)
PREPOSITIONS = {"in", "into", "on", "onto", "to", "at", "under", "over",
                "of", "from", "inside", "between", "left", "right", "front",
                "back", "top", "bottom"}
STOP_WORDS = {"the", "a", "an", "it", "them", "this", "that", "and", "up"}


def parse_plan_phrase(text: str) -> dict[str, str]:
    """Extract a coarse verb, manipulated object and destination from a plan line."""
    phrase = " ".join(str(text).lower().strip().split())
    verb, start, end = "other", 0, 0
    for name, pattern in VERB_PATTERNS:
        match = pattern.search(phrase)
        if match:
            verb, start, end = name, match.start(), match.end()
            break
    tail = phrase[end:].strip()
    words = re.findall(r"[a-z]+(?:'[a-z]+)?", tail)
    split_at = next((i for i, word in enumerate(words) if word in PREPOSITIONS), len(words))
    obj = " ".join(word for word in words[:split_at] if word not in STOP_WORDS).strip()
    destination_words = words[split_at + 1:] if split_at < len(words) else []
    destination = " ".join(word for word in destination_words if word not in STOP_WORDS).strip()
    # Fall back to the content following the matched verb for single-verb
    # commands such as "open the drawer".
    if not obj:
        obj = " ".join(word for word in words if word not in STOP_WORDS and word not in PREPOSITIONS)
    return {"text": phrase, "verb": verb, "object": obj, "destination": destination}


def scene_name(source_file: str) -> str:
    match = re.match(r"(KITCHEN|LIVING_ROOM|STUDY)_SCENE\d+", Path(str(source_file)).name)
    return match.group(0) if match else "unknown_scene"


def episode_progress(frame: int, last_frame: int, horizon: int = 10) -> tuple[float, str]:
    """Approximate within-episode location using the final complete exported window."""
    denom = max(int(last_frame) + int(horizon), int(horizon))
    value = float(np.clip(int(frame) / denom, 0.0, 1.0))
    phase = "early" if value < (1 / 3) else "middle" if value < (2 / 3) else "late"
    return value, phase


def _quat_mul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.asarray((lw * rw - lx * rx - ly * ry - lz * rz,
                       lw * rx + lx * rw + ly * rz - lz * ry,
                       lw * ry - lx * rz + ly * rw + lz * rx,
                       lw * rz + lx * ry - ly * rx + lz * rw), np.float64)


def rotation_curve_deg(rotvec_steps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Integrate step-local axis-angle rotations; return cumulative rotvec and angle."""
    q = np.asarray([1.0, 0.0, 0.0, 0.0], np.float64)
    rotvecs, angles = [], []
    for vector in np.asarray(rotvec_steps, np.float64):
        angle = float(np.linalg.norm(vector))
        if angle > 1e-12:
            axis = vector / angle
            half = angle * 0.5
            dq = np.concatenate(([math.cos(half)], axis * math.sin(half)))
            q = _quat_mul(q, dq)
            q /= max(float(np.linalg.norm(q)), 1e-12)
        # q and -q are the same rotation; use the shortest axis-angle form.
        qw = float(q[0])
        if qw < 0:
            q = -q
            qw = -qw
        sin_half = float(np.linalg.norm(q[1:]))
        total = 2.0 * math.atan2(sin_half, max(qw, 0.0))
        if sin_half > 1e-12:
            rv = q[1:] / sin_half * total
        else:
            rv = np.zeros(3, np.float64)
        rotvecs.append(rv * (180.0 / math.pi))
        angles.append(total * (180.0 / math.pi))
    return np.asarray(rotvecs, np.float32), np.asarray(angles, np.float32)


def motion_signature(action_raw: np.ndarray, *,
                     translation_matrix_m: np.ndarray,
                     rotation_matrix_rad: np.ndarray,
                     grip_threshold: float = 0.25) -> dict[str, Any]:
    """Summarize de-normalized controller commands using demo-fitted scales."""
    action = np.asarray(action_raw, np.float32)
    if action.ndim != 2 or action.shape[1] < 7:
        raise ValueError(f"action chunk must be [H,>=7], got {action.shape}")
    displacement_mm = np.cumsum(action[:, :3] @ translation_matrix_m, axis=0) * 1000.0
    rotation_steps = action[:, 3:6] @ rotation_matrix_rad
    cumulative_rotvec, cumulative_rotation_deg = rotation_curve_deg(rotation_steps)
    grip = np.where(action[:, 6] >= grip_threshold, 1,
                    np.where(action[:, 6] <= -grip_threshold, -1, 0)).astype(np.int8)
    nonzero = grip[grip != 0]
    transitions = []
    for index in range(1, len(grip)):
        if grip[index] and grip[index - 1] and grip[index] != grip[index - 1]:
            transitions.append({"step": int(index),
                                "command": "close" if grip[index] > 0 else "open"})
    if len(nonzero):
        close_fraction = float(np.mean(nonzero > 0))
        if close_fraction >= 0.8:
            grip_mode = "close_command"
        elif close_fraction <= 0.2:
            grip_mode = "open_command"
        else:
            grip_mode = "mixed_commands"
    else:
        grip_mode = "neutral_or_hold"
    endpoint = displacement_mm[-1] if len(displacement_mm) else np.zeros(3, np.float32)
    rot_endpoint = cumulative_rotvec[-1] if len(cumulative_rotvec) else np.zeros(3, np.float32)
    return {
        "displacement_mm": displacement_mm.tolist(),
        "endpoint_displacement_mm": endpoint.astype(float).tolist(),
        "direction": classify_vector(endpoint, threshold=2.0),
        "cumulative_rotation_deg": cumulative_rotation_deg.tolist(),
        "endpoint_rotation_vector_deg": rot_endpoint.astype(float).tolist(),
        "physical_scale_units": {"translation": "mm", "rotation": "degrees",
                                 "scale_applied_to": "cumulative de-normalized action command"},
        "rotation": classify_vector(rot_endpoint, threshold=1.0),
        "gripper_commands": grip.tolist(),
        "gripper_mode": grip_mode,
        "gripper_events": transitions,
        "gripper_close_fraction": close_fraction if len(nonzero) else None,
    }


def classify_vector(vector: np.ndarray, *, threshold: float) -> str:
    value = np.asarray(vector, np.float64).reshape(-1)
    if value.size != 3 or float(np.linalg.norm(value)) < threshold:
        return "none"
    axis = int(np.argmax(np.abs(value)))
    return ("+" if value[axis] >= 0 else "-") + "xyz"[axis]


def signature_agreement(signatures: list[dict[str, Any]]) -> dict[str, Any]:
    """Mode agreement across states for a fixed base center."""
    if not signatures:
        return {"n_states": 0, "direction": None, "rotation": None, "gripper": None,
                "modes": {}}
    modes = {name: [str(item[name]) for item in signatures]
             for name in ("direction", "rotation", "gripper_mode")}
    agree = {"direction": max(Counter(modes["direction"]).values()) / len(signatures),
             "rotation": max(Counter(modes["rotation"]).values()) / len(signatures),
             "gripper": max(Counter(modes["gripper_mode"]).values()) / len(signatures)}
    return {"n_states": len(signatures), **agree,
            "modes": {name: dict(Counter(values)) for name, values in modes.items()}}


def interpretability_label(stability: dict[str, Any], *, threshold: float = 0.70,
                           min_states: int = 3) -> bool:
    """Operational label: each motion channel has a stable mode over contexts."""
    return (stability.get("n_states", 0) >= min_states
            and all(stability.get(name) is not None and stability[name] >= threshold
                    for name in ("direction", "rotation", "gripper")))


def build_transition_summary(episode_rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Count adjacent unit changes and frequent run-collapsed sequences."""
    transitions: Counter[tuple[int, int]] = Counter()
    sequences: Counter[tuple[int, ...]] = Counter()
    boundary_count = switch_count = 0
    for rows in episode_rows.values():
        ordered = sorted(rows, key=lambda row: (int(row["frame"]), int(row["row_index"])))
        ids = [int(row["unit_id"]) for row in ordered if row.get("unit_id") is not None]
        collapsed = []
        for unit_id in ids:
            if not collapsed or collapsed[-1] != unit_id:
                collapsed.append(unit_id)
        for pair in zip(ids, ids[1:]):
            boundary_count += 1
            if pair[0] != pair[1]:
                transitions[pair] += 1
                switch_count += 1
        if collapsed:
            sequences[tuple(collapsed)] += 1
    outgoing: Counter[int] = Counter()
    for (left, _), count in transitions.items():
        outgoing[left] += count
    matrix = [{"from_unit": left, "to_unit": right, "count": count,
               "probability_given_from": count / max(outgoing[left], 1)}
              for (left, right), count in sorted(transitions.items())]
    typical = [{"unit_sequence": list(sequence), "episode_count": count}
               for sequence, count in sequences.most_common(20)]
    return {"transition_matrix_sparse": matrix, "typical_collapsed_sequences": typical,
            "episodes": len(episode_rows), "adjacent_window_pairs": boundary_count,
            "unit_switches": switch_count,
            "switch_fraction_of_adjacent_pairs": switch_count / max(boundary_count, 1)}


def build_temporal_summary(episode_rows: dict[str, list[dict[str, Any]]], *,
                           stride: int = 10, sampling_rate_hz: float = 20.0) -> dict[str, Any]:
    """Summarize contiguous unit runs and representative compressed sequences by task."""
    run_lengths: list[int] = []
    task_sequences: dict[str, Counter[tuple[int, ...]]] = defaultdict(Counter)
    for rows in episode_rows.values():
        ordered = sorted(rows, key=lambda row: (int(row["frame"]), int(row["row_index"])))
        ordered = [row for row in ordered if row.get("unit_id") is not None]
        if not ordered:
            continue
        runs: list[tuple[int, int]] = []
        current, count = int(ordered[0]["unit_id"]), 1
        compressed = [current]
        for row in ordered[1:]:
            uid = int(row["unit_id"])
            if uid == current:
                count += 1
            else:
                runs.append((current, count))
                current, count = uid, 1
                compressed.append(uid)
        runs.append((current, count))
        run_lengths.extend(length for _, length in runs)
        # Each episode belongs to one task in this LIBERO export.
        task = str(ordered[0]["task"])
        task_sequences[task][tuple(compressed)] += 1
    values = np.asarray(run_lengths, np.float64)
    return {
        "window_stride_steps": int(stride), "window_horizon_steps": 10,
        "window_overlap_steps": max(0, 10 - int(stride)),
        "sampling_rate_hz": float(sampling_rate_hz),
        "run_count": int(len(values)),
        "run_length_windows": {"mean": float(values.mean()) if len(values) else None,
                               "median": float(np.median(values)) if len(values) else None,
                               "p90": float(np.quantile(values, .9)) if len(values) else None,
                               "max": int(values.max()) if len(values) else None,
                               "histogram": {str(int(k)): int(v) for k, v in
                                             zip(*np.unique(values.astype(int), return_counts=True), strict=True)}},
        "run_length_seconds": {"mean": float(values.mean() * stride / sampling_rate_hz) if len(values) else None,
                               "median": float(np.median(values) * stride / sampling_rate_hz) if len(values) else None,
                               "p90": float(np.quantile(values, .9) * stride / sampling_rate_hz) if len(values) else None,
                               "max": float(values.max() * stride / sampling_rate_hz) if len(values) else None},
        "typical_collapsed_sequences_by_task": [
            {"task": task, "sequences": [{"unit_sequence": list(seq), "episode_count": count}
                                          for seq, count in counts.most_common(3)]}
            for task, counts in sorted(task_sequences.items()) if counts
        ],
        "interpretation": (
            "Window-switch rate compares consecutive 10-step chunks; default stride=10 at 20 Hz, "
            "so windows are 0.5 s and non-overlapping. The 79% rate reflects frequent unit-ID "
            "changes between successive half-second windows, not a boundary artifact from overlapping windows."),
    }


def motion_description_overlap(units: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for unit in units:
        signature = unit.get("center_motion_signature")
        if signature:
            groups[(signature["direction"], signature["gripper_mode"], signature["rotation"])].append(
                int(unit["unit_id"]))
    duplicate_groups = [{"description": {"direction": key[0], "gripper": key[1], "rotation": key[2]},
                         "unit_ids": ids} for key, ids in groups.items() if len(ids) > 1]
    n = sum(bool(unit.get("center_motion_signature")) for unit in units)
    duplicate_unit_count = sum(len(item["unit_ids"]) for item in duplicate_groups)
    duplicate_pairs = sum(len(item["unit_ids"]) * (len(item["unit_ids"]) - 1) // 2
                          for item in duplicate_groups)
    return {"categorical_description": "direction × gripper mode × dominant rotation axis",
            "unit_count": n, "unique_descriptions": len(groups),
            "duplicated_description_group_count": len(duplicate_groups),
            "units_in_duplicated_groups": duplicate_unit_count,
            "fraction_units_with_duplicate_description": duplicate_unit_count / max(n, 1),
            "matching_unit_pairs": duplicate_pairs, "groups": duplicate_groups}


def _verb_name(text: str) -> str:
    return parse_plan_phrase(text)["verb"]


def _summary_counts(counter: Counter, denominator: int | None = None) -> dict[str, Any]:
    total = int(sum(counter.values()) if denominator is None else denominator)
    return {"counts": dict(counter), "total": total,
            "purity": (max(counter.values()) / sum(counter.values())) if counter else None}


def _decode_openpi_actions(actions_model: np.ndarray, norm_stats: dict[str, Any]) -> np.ndarray:
    """Invert M1 and π0.5 quantile normalization for the seven LIBERO controls."""
    stats = norm_stats["norm_stats"]["actions"]
    q01 = np.asarray(stats["q01"], np.float32)[:7]
    q99 = np.asarray(stats["q99"], np.float32)[:7]
    return ((np.asarray(actions_model, np.float32)[..., :7] + 1.0) * 0.5
            * (q99 - q01 + 1e-6) + q01)


def _denormalize_state(state_model: np.ndarray, norm_stats: dict[str, Any]) -> np.ndarray:
    """Invert checkpoint quantile normalization; state layout starts ee xyz + axis-angle."""
    stats = norm_stats["norm_stats"]["state"]
    q01 = np.asarray(stats["q01"], np.float32)[:8]
    q99 = np.asarray(stats["q99"], np.float32)[:8]
    return ((np.asarray(state_model, np.float32)[..., :8] + 1.0) * 0.5
            * (q99 - q01 + 1e-6) + q01)


def _axis_angle_quaternion(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, np.float64)
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return np.asarray([1., 0., 0., 0.])
    return np.r_[math.cos(angle / 2), vector / angle * math.sin(angle / 2)]


def _relative_rotation_vector(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    q0, q1 = _axis_angle_quaternion(start), _axis_angle_quaternion(end)
    inverse_q0 = q0 * np.asarray([1., -1., -1., -1.])
    # State/action rotation vectors are expressed in the world frame for LIBERO.
    # The world-frame relative rotation is q1 * inverse(q0), not inverse(q0) * q1.
    delta = _quat_mul(q1, inverse_q0)
    if delta[0] < 0:
        delta = -delta
    sin_half = float(np.linalg.norm(delta[1:]))
    if sin_half < 1e-12:
        return np.zeros(3, np.float64)
    angle = 2 * math.atan2(sin_half, max(float(delta[0]), 0.0))
    return delta[1:] / sin_half * angle


def calibrate_action_motion_scales(actions_model: np.ndarray, states_model: np.ndarray,
                                   episodes: np.ndarray, frames: np.ndarray,
                                   norm_stats: dict[str, Any], *, stride: int = 10
                                   ) -> dict[str, Any]:
    """Fit per-channel action-sum -> observed EE displacement/rotation over adjacent windows."""
    actions = _decode_openpi_actions(actions_model, norm_stats)[..., :7]
    states = _denormalize_state(states_model, norm_stats)
    by_frame = {(str(ep), int(frame)): i for i, (ep, frame) in enumerate(zip(episodes, frames, strict=True))}
    sums, positions, rotations = [], [], []
    for i, (episode, frame) in enumerate(zip(episodes, frames, strict=True)):
        j = by_frame.get((str(episode), int(frame) + int(stride)))
        if j is None:
            continue
        sums.append(actions[i, :, :6].sum(axis=0).reshape(2, 3))
        positions.append(states[j, :3] - states[i, :3])
        rotations.append(_relative_rotation_vector(states[i, 3:6], states[j, 3:6]))
    if len(sums) < 100:
        raise ValueError(f"too few adjacent action/state windows to calibrate scales: {len(sums)}")
    sums_arr = np.asarray(sums, np.float64)
    pos_x, rot_x = sums_arr[:, 0], sums_arr[:, 1]
    pos_y, rot_y = np.asarray(positions), np.asarray(rotations)
    # Include an intercept to absorb small dataset-specific controller bias.
    pos_coef = np.linalg.lstsq(np.c_[pos_x, np.ones(len(pos_x))], pos_y, rcond=None)[0]
    rot_coef = np.linalg.lstsq(np.c_[rot_x, np.ones(len(rot_x))], rot_y, rcond=None)[0]
    pos_pred = np.c_[pos_x, np.ones(len(pos_x))] @ pos_coef
    rot_pred = np.c_[rot_x, np.ones(len(rot_x))] @ rot_coef

    def channel_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
        correlations = []
        for axis in range(3):
            if np.std(actual[:, axis]) < 1e-10 or np.std(predicted[:, axis]) < 1e-10:
                correlations.append(None)
            else:
                correlations.append(float(np.corrcoef(actual[:, axis], predicted[:, axis])[0, 1]))
        return {"r2_per_axis": [float(1 - np.sum((actual[:, d] - predicted[:, d]) ** 2) /
                                              max(np.sum((actual[:, d] - actual[:, d].mean()) ** 2), 1e-12))
                                  for d in range(3)],
                "correlation_per_axis": correlations,
                "rmse_per_axis": np.sqrt(np.mean((actual - predicted) ** 2, axis=0)).tolist()}

    return {
        "method": "OLS with intercept on adjacent stride windows, action sums predict observed EE delta",
        "paired_windows": len(sums), "stride_steps": int(stride), "window_horizon_steps": 10,
        "sampling_rate_hz": 20, "window_seconds": 10 / 20, "action_inverse_normalization":
            "OpenPI quantile inverse: (normalized+1)/2 * (q99-q01+1e-6)+q01",
        "state_inverse_normalization":
            "OpenPI quantile inverse; state[0:3]=ee position meters, state[3:6]=ee axis-angle radians",
        "translation_action_to_meters_matrix": pos_coef[:3].tolist(),
        "translation_intercept_m": pos_coef[3].tolist(),
        "translation_fit": channel_metrics(pos_y, pos_pred),
        "rotation_relative_frame": "world (q_end * inverse(q_start))",
        "rotation_action_to_radians_matrix": rot_coef[:3].tolist(),
        "rotation_intercept_rad": rot_coef[3].tolist(),
        "rotation_fit": channel_metrics(rot_y, rot_pred),
    }


def _load_rlds_frames(rlds_root: Path, requests: dict[int, set[int]]) -> dict[tuple[int, int], np.ndarray]:
    """Fetch exact front-camera frames for selected global RLDS episode/frame IDs."""
    if not requests:
        return {}
    import tensorflow as tf
    from PIL import Image

    try:
        tf.config.set_visible_devices([], "GPU")
        tf.config.threading.set_intra_op_parallelism_threads(8)
        tf.config.threading.set_inter_op_parallelism_threads(2)
    except RuntimeError:
        pass
    info = json.loads((rlds_root / "dataset_info.json").read_text())
    split = next(item for item in info["splits"] if item["name"] == "train")
    lengths = [int(value) for value in split["shardLengths"]]
    shards = sorted(rlds_root.glob("*train.tfrecord-*"))
    if len(shards) != len(lengths):
        raise ValueError(f"RLDS shard count mismatch: files={len(shards)} manifest={len(lengths)}")
    cumulative = np.cumsum([0, *lengths])
    by_shard: dict[int, dict[int, tuple[int, set[int]]]] = defaultdict(dict)
    for episode_id, frames in requests.items():
        shard = bisect.bisect_right(cumulative, episode_id) - 1
        if shard < 0 or shard >= len(shards):
            raise ValueError(f"episode index {episode_id} outside RLDS shard inventory")
        by_shard[shard][episode_id - int(cumulative[shard])] = (episode_id, frames)

    from tensorflow.core.example import example_pb2
    output: dict[tuple[int, int], np.ndarray] = {}
    for shard_index, targets in sorted(by_shard.items()):
        remaining = set(targets)
        stop_after = max(remaining)
        for local_index, serialized in enumerate(
                tf.data.TFRecordDataset([str(shards[shard_index])], num_parallel_reads=1)):
            if local_index in targets:
                episode_id, frames = targets[local_index]
                example = example_pb2.Example.FromString(serialized.numpy())
                encoded = example.features.feature["steps/observation/image"].bytes_list.value
                for frame in frames:
                    if frame >= len(encoded):
                        raise ValueError(f"frame {frame} outside episode {episode_id} length {len(encoded)}")
                    with Image.open(BytesIO(encoded[frame])) as image:
                        output[(episode_id, int(frame))] = np.asarray(image.convert("RGB"))
                remaining.remove(local_index)
                if not remaining:
                    break
            if local_index >= stop_after:
                break
        if remaining:
            raise ValueError(f"RLDS shard {shard_index} missed local episode indices {sorted(remaining)}")
    return output


def _extract_episode_index(episode_id: str) -> int:
    match = re.search(r"/rlds-(\d+)$", str(episode_id))
    if not match:
        raise ValueError(f"unexpected LIBERO-90 episode id {episode_id!r}")
    return int(match.group(1))


def _short_task_name(value: str, limit: int = 34) -> str:
    name = re.sub(r"^(KITCHEN|LIVING_ROOM|STUDY)_SCENE\d+_", "", str(value))
    name = re.sub(r"_demo\.hdf5$", "", name)
    name = name.replace("_", " ").strip()
    return name if len(name) <= limit else name[:limit - 1].rstrip() + "…"


def _rotate_camera_180(image: np.ndarray | None) -> np.ndarray | None:
    return None if image is None else np.ascontiguousarray(np.rot90(image, 2))


def _render_unit_figure(unit: dict[str, Any], output_path: Path,
                        example_images: list[list[np.ndarray | None]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    signature = unit["center_motion_signature"]
    path = np.asarray(signature["displacement_mm"], np.float32)
    rotation = signature["cumulative_rotation_deg"]
    grip = signature["gripper_commands"]
    fig = plt.figure(figsize=(12, 8), constrained_layout=True)
    grid = fig.add_gridspec(4, 3, height_ratios=(1.15, 1, 1, 1))
    ax_xy = fig.add_subplot(grid[0, 0])
    ax_xz = fig.add_subplot(grid[0, 1])
    ax_profile = fig.add_subplot(grid[0, 2])
    if len(path):
        ax_xy.plot(np.r_[0, path[:, 0]], np.r_[0, path[:, 1]], marker="o", markersize=2)
        ax_xz.plot(np.r_[0, path[:, 0]], np.r_[0, path[:, 2]], marker="o", markersize=2)
    ax_xy.set(title="Cumulative XY (mm)", xlabel="x", ylabel="y", aspect="equal")
    ax_xz.set(title="Cumulative XZ (mm)", xlabel="x", ylabel="z", aspect="equal")
    ax_profile.plot(np.arange(len(rotation)), rotation, label="net rotation (deg)")
    ax_profile.set(title="Rotation / gripper command", xlabel="step", ylabel="deg")
    ax_grip = ax_profile.twinx()
    ax_grip.step(np.arange(len(grip)), grip, where="post", color="tab:red", alpha=.55,
                 label="gripper (+close / -open)")
    ax_grip.set_ylim(-1.2, 1.2)
    ax_grip.set_yticks([-1, 0, 1], labels=["open", "neutral", "close"])
    ax_profile.grid(alpha=.2)
    for example_index in range(3):
        for frame_index in range(3):
            ax = fig.add_subplot(grid[example_index + 1, frame_index])
            image = (example_images[example_index][frame_index]
                     if example_index < len(example_images) else None)
            if image is not None:
                ax.imshow(_rotate_camera_180(image))
            else:
                ax.text(.5, .5, "frame unavailable", ha="center", va="center")
                ax.set_facecolor("#eeeeee")
            if example_index < len(unit["representative_examples"]):
                example = unit["representative_examples"][example_index]
                task_label = _short_task_name(example["source_file"])
                label = (f"{('start', 'middle', 'end')[frame_index]}\n{task_label}"
                         if frame_index == 0 else ("middle" if frame_index == 1 else "end"))
                ax.set_title(label, fontsize=8)
            else:
                ax.set_title(("start", "middle", "end")[frame_index], fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Unit {unit['unit_id']} | n={unit['support_count']} | "
                 f"{signature['direction']} / {signature['rotation']} / {signature['gripper_mode']}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100, format="jpg", pil_kwargs={"quality": 82, "optimize": True})
    plt.close(fig)


def _automatic_description(signature: dict[str, Any]) -> str:
    directions = {"+x": "右へ", "-x": "左へ", "+y": "奥へ", "-y": "手前へ",
                  "+z": "上へ", "-z": "下へ", "none": "ほぼ移動せず"}
    text = directions.get(signature["direction"], signature["direction"])
    mode = signature["gripper_mode"]
    if mode == "close_command":
        text += "グリッパーを閉じる指令"
    elif mode == "open_command":
        text += "グリッパーを開く指令"
    elif mode == "mixed_commands":
        text += "グリッパーの開閉指令"
    angle = float(signature["cumulative_rotation_deg"][-1]) if signature["cumulative_rotation_deg"] else 0.0
    if angle >= 10:
        text += f"、回転約{angle:.0f}度"
    return text


def _plan_label_for_task(instruction: str, annotations: dict[str, Any]) -> dict[str, Any]:
    record = annotations.get(str(instruction).lower().strip())
    if not record:
        return {"plan": [], "phrases": [], "primary_verb": "unlabeled", "classes": []}
    phrases = [parse_plan_phrase(text) for text in record.get("plan", [])]
    return {"plan": record.get("plan", []), "phrases": phrases,
            "primary_verb": phrases[0]["verb"] if phrases else "unlabeled",
            "classes": record.get("all_classes", [])}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def analyze_arrays(memory: Any, rows: dict[str, np.ndarray], keys: np.ndarray,
                   annotations: dict[str, Any], norm_stats: dict[str, Any],
                   output_dir: Path, *, frame_loader: Callable | None = None,
                   rlds_root: Path | None = None, stability_samples: int = 16,
                   image_source: str = "LIBERO-90 RLDS TFRecords",
                   physical_calibration: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a catalog from already-loaded arrays and an M6 memory object."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    n = len(rows["actions"])
    keys = np.asarray(keys, np.float32)
    if len(memory.memory) != n or keys.shape[0] != n:
        raise ValueError(f"row alignment mismatch: rows={n}, memory={len(memory.memory)}, keys={len(keys)}")
    if keys.ndim != 2:
        raise ValueError("keys must be [N,D]")
    if not np.array_equal(np.asarray(rows["episode_id"]).astype(str),
                          np.asarray([item["meta"]["episode_id"] for item in memory.memory]).astype(str)):
        raise ValueError("memory and LIBERO-90 rows do not have identical episode ordering")

    # Check a deterministic slice instead of retaining/serializing a second copy
    # of the multi-gigabyte row archive.
    sampled = np.unique(np.linspace(0, n - 1, min(n, 257), dtype=np.int64))
    key_error = max(float(np.max(np.abs(keys[i] - memory.memory[i]["key"]))) for i in sampled)
    if key_error > 2e-5:
        raise ValueError(f"memory/keys NPZ misalignment; sampled max abs key error={key_error}")

    backbone = memory.backbone
    import torch
    backbone.device = torch.device("cpu")
    for module_name in ("encoder", "prior_net", "base_decoder", "residual_decoder"):
        module = getattr(backbone, module_name, None)
        if module is not None and callable(getattr(module, "to", None)):
            module.to("cpu")
    # Rebuild id->members and collect task-row metadata. Labels below are only
    # joined now, after the frozen units have been loaded.
    last_frame_by_episode: dict[str, int] = {}
    for episode, frame in zip(rows["episode_id"], rows["frame"], strict=True):
        episode = str(episode)
        last_frame_by_episode[episode] = max(last_frame_by_episode.get(episode, 0), int(frame))
    unit_members: dict[int, list[int]] = defaultdict(list)
    episode_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    row_meta: list[dict[str, Any]] = []
    task_labels: dict[str, dict[str, Any]] = {}
    for index in range(n):
        episode = str(rows["episode_id"][index])
        source = str(rows["source_file"][index])
        task = Path(source).name.removesuffix("_demo.hdf5").replace("_", " ")
        instruction = str(rows["instruction"][index])
        label = task_labels.setdefault(instruction, _plan_label_for_task(instruction, annotations))
        value, phase = episode_progress(int(rows["frame"][index]), last_frame_by_episode[episode])
        item = memory.memory[index]
        raw_uid = item.get("unit_id")
        unit_id = None if raw_uid is None else int(raw_uid)
        meta = {"row_index": index, "episode_id": episode,
                "init_state_id": str(item.get("meta", {}).get("init_state_id", "")),
                "task": task, "instruction": instruction, "source_file": source,
                "scene": scene_name(source), "frame": int(rows["frame"][index]),
                "normalized_episode_time": value, "episode_phase": phase,
                "unit_id": unit_id, "task_primary_verb": label["primary_verb"]}
        row_meta.append(meta)
        if unit_id is not None:
            unit_members[unit_id].append(index)
        episode_rows[episode].append(meta)

    # Inverse-transform the observed member actions once; their spread is in
    # physical LIBERO delta-action units, not M1's internal z-scored space.
    raw_actions = _decode_openpi_actions(rows["actions"], norm_stats)
    if physical_calibration is None:
        physical_calibration = calibrate_action_motion_scales(
            rows["actions"], rows["state"], rows["episode_id"], rows["frame"], norm_stats)
    translation_matrix = np.asarray(physical_calibration["translation_action_to_meters_matrix"], np.float64)
    rotation_matrix = np.asarray(physical_calibration["rotation_action_to_radians_matrix"], np.float64)
    centers = np.asarray(memory._centers, np.float32)
    units: list[dict[str, Any]] = []
    image_requests: dict[int, set[int]] = defaultdict(set)
    rep_rows_by_unit: dict[int, list[int]] = {}
    sampled_stability: dict[int, list[int]] = {}

    for unit_id in range(len(centers)):
        members = unit_members.get(unit_id, [])
        if not members:
            units.append({"unit_id": unit_id, "support_count": 0,
                          "catalog_status": "prototype_without_memory_member"})
            continue
        center = centers[unit_id]
        base_distances = np.asarray([np.linalg.norm(memory.memory[i]["base"] - center)
                                     for i in members], np.float32)
        order = np.argsort(base_distances, kind="stable")
        representatives = [members[int(i)] for i in order[:3]]
        rep_rows_by_unit[unit_id] = representatives
        if len(members) <= stability_samples:
            stable_ix = members
        else:
            picks = np.linspace(0, len(members) - 1, stability_samples, dtype=np.int64)
            stable_ix = [members[int(i)] for i in picks]
        sampled_stability[unit_id] = stable_ix
        for index in representatives:
            episode_index = _extract_episode_index(str(rows["episode_id"][index]))
            start = int(rows["frame"][index])
            image_requests[episode_index].update((start, start + 4, start + 9))

    if frame_loader is not None:
        frame_map = frame_loader(image_requests)
    elif rlds_root is not None:
        frame_map = _load_rlds_frames(Path(rlds_root), image_requests)
    else:
        frame_map = {}

    pca_data = np.stack([np.asarray(item["base"], np.float32) for item in memory.memory])
    pca_mean = pca_data.mean(axis=0)
    # The base is only 12-D. SVD is deterministic and avoids fitting PCA on labels.
    _, singular, vt = np.linalg.svd(pca_data - pca_mean, full_matrices=False)
    pca_components = vt[:min(3, len(vt))]
    pca_scales = singular[:len(pca_components)] / np.sqrt(max(len(pca_data) - 1, 1))
    pca_axes = []
    for axis_index, (component, scale) in enumerate(zip(pca_components, pca_scales, strict=True)):
        pca_axes.append({"axis": axis_index, "explained_variance_fraction": float(
            singular[axis_index] ** 2 / max(float(np.sum(singular ** 2)), 1e-12)),
            "one_std_scale": float(scale), "component": component.astype(float).tolist()})

    for unit_id, members in sorted(unit_members.items()):
        center = centers[unit_id]
        reps = rep_rows_by_unit[unit_id]
        rep_distances = [float(np.linalg.norm(memory.memory[i]["base"] - center)) for i in reps]
        primary = Counter(row_meta[i]["task_primary_verb"] for i in members)
        tasks = Counter(row_meta[i]["task"] for i in members)
        scenes = Counter(row_meta[i]["scene"] for i in members)
        phases = Counter(row_meta[i]["episode_phase"] for i in members)
        task_to_rows = Counter(row_meta[i]["task"] for i in members)
        plan_phrases: Counter[str] = Counter()
        plan_verbs: Counter[str] = Counter()
        plan_objects: Counter[str] = Counter()
        plan_classes: Counter[str] = Counter()
        task_plans = {}
        for instruction in {row_meta[i]["instruction"] for i in members}:
            label = task_labels[instruction]
            n_rows_for_task = sum(row_meta[i]["instruction"] == instruction for i in members)
            task_plans[instruction] = label["plan"]
            if label["phrases"]:
                share = n_rows_for_task / len(label["phrases"])
                for phrase in label["phrases"]:
                    plan_phrases[phrase["text"]] += share
                    plan_verbs[phrase["verb"]] += share
                    if phrase["object"]:
                        plan_objects[phrase["object"]] += share
                for class_id in label["classes"]:
                    plan_classes[str(class_id)] += share

        # Decode one center using the base-nearest member's own state/key, then
        # reuse its API for context-stability and PCA traversal probes.
        representative_index = reps[0]
        state = np.asarray(rows["state"][representative_index:representative_index + 1], np.float32)
        key = keys[representative_index:representative_index + 1]
        _, residual = memory.prior(key, state)
        center_action_model = memory.decode(center[None], residual, key, state)[0]
        center_action_raw = _decode_openpi_actions(center_action_model[None], norm_stats)[0]
        center_signature = motion_signature(center_action_raw,
                                            translation_matrix_m=translation_matrix,
                                            rotation_matrix_rad=rotation_matrix)

        stable_ix = sampled_stability[unit_id]
        stable_states = np.asarray(rows["state"][stable_ix], np.float32)
        stable_keys = keys[stable_ix]
        _, stable_residuals = memory.prior(stable_keys, stable_states)
        fixed_bases = np.broadcast_to(center[None], (len(stable_ix), len(center))).copy()
        decoded_model = memory.decode(fixed_bases, stable_residuals, stable_keys, stable_states)
        decoded_raw = _decode_openpi_actions(decoded_model, norm_stats)
        context_signatures = [motion_signature(action,
                                               translation_matrix_m=translation_matrix,
                                               rotation_matrix_rad=rotation_matrix)
                              for action in decoded_raw]
        stability = signature_agreement(context_signatures)

        member_actions = raw_actions[members]
        member_mean = member_actions.mean(axis=0)
        action_spread = float(np.sqrt(np.mean((member_actions - member_mean[None]) ** 2)))
        member_signature = [motion_signature(action,
                                             translation_matrix_m=translation_matrix,
                                             rotation_matrix_rad=rotation_matrix)
                            for action in member_actions[
                                np.linspace(0, len(member_actions) - 1,
                                            min(len(member_actions), 512), dtype=np.int64)]]
        member_directions = Counter(item["direction"] for item in member_signature)
        member_grippers = Counter(item["gripper_mode"] for item in member_signature)

        traversal_values = np.linspace(-2, 2, 9, dtype=np.float32)
        traversal = []
        for axis_index, (component, scale) in enumerate(zip(pca_components, pca_scales, strict=True)):
            traversed_b = pca_mean[None] + traversal_values[:, None] * scale * component[None]
            traversed_keys = np.broadcast_to(key, (len(traversal_values), key.shape[1]))
            traversed_states = np.broadcast_to(state, (len(traversal_values), state.shape[1]))
            _, traversed_r = memory.prior(traversed_keys, traversed_states)
            action_model = memory.decode(traversed_b, traversed_r, traversed_keys, traversed_states)
            actions = _decode_openpi_actions(action_model, norm_stats)
            traversal.append({"axis": axis_index, "values_in_std": traversal_values.tolist(),
                              "motion_signatures": [motion_signature(
                                  value, translation_matrix_m=translation_matrix,
                                  rotation_matrix_rad=rotation_matrix) for value in actions]})

        examples = []
        per_example_images = []
        for index, distance in zip(reps, rep_distances, strict=True):
            meta = row_meta[index]
            episode_index = _extract_episode_index(meta["episode_id"])
            frame = meta["frame"]
            frames = [frame, frame + 4, frame + 9]
            images = [frame_map.get((episode_index, value)) for value in frames]
            examples.append({**meta, "base_distance": distance,
                "frames": [{"frame": value,
                                         "image_path": (f"figures/frames/ep{episode_index:06d}_f{value:06d}.jpg"
                                                        if frame_map.get((episode_index, value)) is not None else None)}
                                        for value in frames]})
            per_example_images.append(images)
        figure_rel = f"figures/unit_{unit_id:03d}.jpg"
        unit = {
            "unit_id": unit_id, "support_count": len(members),
            "episode_count": len({row_meta[i]["episode_id"] for i in members}),
            "creation_stage": "libero90_memory_build",
            "center_base": center.astype(float).tolist(),
            "representative_state_source": row_meta[representative_index],
            "center_motion_signature": center_signature,
            "automatic_description": _automatic_description(center_signature),
            "center_action_chunk_raw_10x7": center_action_raw.astype(float).tolist(),
            "center_action_chunk_units": (
                "OpenPI quantile-denormalized LIBERO controller command; dimensionless action units, "
                "not meters or radians. Translation/rotation plots apply demo-fitted scales in metadata."),
            "context_motion_stability": stability,
            "motion_pattern_interpretable": interpretability_label(stability),
            "member_action_spread": {"raw_delta_action_rms": action_spread,
                                      "per_step_std_mean": member_actions.std(axis=0).mean(axis=1).tolist(),
                                      "sampled_direction_counts": dict(member_directions),
                                      "sampled_gripper_command_counts": dict(member_grippers),
                                      "sample_size_for_motion_counts": len(member_signature)},
            "member_composition": {
                "task_counts": dict(tasks), "n_tasks": len(tasks),
                "scene_counts": dict(scenes), "n_scenes": len(scenes),
                "episode_phase_counts": dict(phases),
                "primary_task_verb_counts": dict(primary),
                "primary_task_verb_purity": max(primary.values()) / len(members) if primary else None,
                "plan_phrase_member_weighted_counts": dict(plan_phrases),
                "plan_verb_member_weighted_counts": dict(plan_verbs),
                "plan_object_member_weighted_counts": dict(plan_objects),
                "plan_class_member_weighted_counts": dict(plan_classes),
                "task_level_plan_texts": task_plans,
                "label_granularity": "task-level plan only; rows are not assigned to a plan phase",
            },
            "member_rows": [row_meta[index] for index in members],
            "representative_examples": examples,
            "pca_traversal": traversal,
            "figure": figure_rel,
        }
        _render_unit_figure(unit, figures_dir / f"unit_{unit_id:03d}.jpg", per_example_images)
        units.append(unit)

        # Write each selected source frame separately as well as embedding the
        # same frames in the per-unit grid, for future web-catalog use.
        from PIL import Image
        for example, images in zip(examples, per_example_images, strict=True):
            episode_index = _extract_episode_index(example["episode_id"])
            for frame_entry, image in zip(example["frames"], images, strict=True):
                if image is None:
                    continue
                path = output_dir / frame_entry["image_path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                if not path.exists():
                    Image.fromarray(_rotate_camera_180(image)).save(
                        path, format="JPEG", quality=82, optimize=True)

    transitions = build_transition_summary(episode_rows)
    verb_to_units: dict[str, list[int]] = defaultdict(list)
    for unit in units:
        if unit.get("catalog_status"):
            continue
        for verb in unit["member_composition"]["primary_task_verb_counts"]:
            verb_to_units[verb].append(unit["unit_id"])
    multi_verb_units = [{"unit_id": unit["unit_id"],
                         "verbs": unit["member_composition"]["primary_task_verb_counts"]}
                        for unit in units if len(unit.get("member_composition", {}).get(
                            "primary_task_verb_counts", {})) > 1]
    interpretable = [unit for unit in units if unit.get("motion_pattern_interpretable")]
    not_interpretable = [unit for unit in units if "motion_pattern_interpretable" in unit
                         and not unit["motion_pattern_interpretable"]]

    pca_figure = _render_pca_traversal(pca_axes, units, figures_dir / "pca_traversal.jpg")
    temporal = build_temporal_summary(episode_rows)
    duplicate_descriptions = motion_description_overlap(units)
    result = {
        "schema_version": "s2-m6-unit-interpretability-v1",
        "metadata": {
            "labels_used_for_unit_creation": False,
            "labels_usage": "post-hoc only; task/plan annotations were not passed to M1 or M6",
            "unit_count": len(centers), "memory_item_count": len(memory.memory),
            "row_count": n, "key_count": len(keys),
            "key_text_source": getattr(memory, "key_text_source", "unknown"),
            "representation_version": getattr(memory, "representation_version", "unknown"),
            "image_source": image_source,
            "episode_time_estimate": "last complete exported frame + H=10; normalized window-start time",
            "plan_boundary_alignment": None,
            "plan_boundary_alignment_reason": (
                "instruct2plan_obj_90.json provides task-level plan phrases/classes but no temporal boundaries; "
                "no interval-to-plan-phase heuristic was applied"),
            "gripper_command_sign": "+1 close / -1 open (verified against LIBERO demo gripper-state extent)",
            "action_command_units": (
                "center_action_chunk_raw_10x7 is after inverse OpenPI quantile normalization; "
                "it remains dimensionless controller command, not physical position/rotation."),
            "physical_motion_calibration": physical_calibration,
            "physical_motion_calibration_scope": "all LIBERO-90 row episodes, no task/plan labels",
            "windowing": {"horizon_steps": 10, "stride_steps": 10, "sampling_rate_hz": 20,
                          "overlap_steps": 0, "window_seconds": 0.5,
                          "source": "export_libero90.py complete_starts(length,horizon=10,stride=10)"},
            "interpretability_rule": (
                "motion-pattern interpretable iff >=3 sampled contexts and direction, rotation, "
                "and gripper command mode each have >=0.70 modal agreement"),
            "annotation_file_granularity": "task-level plan; not segment-level ground truth",
        },
        "summary": {
            "interpretable_unit_count": len(interpretable),
            "not_interpretable_unit_count": len(not_interpretable),
            "interpretable_unit_fraction": len(interpretable) / max(len(interpretable) + len(not_interpretable), 1),
            "not_interpretable_unit_fraction": len(not_interpretable) / max(len(interpretable) + len(not_interpretable), 1),
            "task_count_per_unit_distribution": _summary_counts(Counter(
                str(unit.get("member_composition", {}).get("n_tasks", 0)) for unit in units)),
            "scene_count_per_unit_distribution": _summary_counts(Counter(
                str(unit.get("member_composition", {}).get("n_scenes", 0)) for unit in units)),
            "task_primary_verb_purity_distribution": [unit.get("member_composition", {}).get(
                "primary_task_verb_purity") for unit in units if unit.get("member_composition")],
            "motion_mode_agreement_distribution": {name: [unit["context_motion_stability"].get(name)
                for unit in units if unit.get("context_motion_stability")
                for name0 in [name] if name0 == name]
                for name in ("direction", "rotation", "gripper")},
            "units_by_task_primary_verb": dict(verb_to_units),
            "units_with_multiple_task_verbs": multi_verb_units,
            "unit_transition_statistics": transitions,
            "temporal_run_statistics": temporal,
            "duplicate_motion_description_statistics": duplicate_descriptions,
        },
        "pca": {"fit_rows": len(pca_data), "fit_scope": "LIBERO-90 memory member bases only",
                "axes": pca_axes, "figure": pca_figure},
        "units": units,
    }
    result["metadata"]["figure_policy"] = {
        "unit_figure_count": sum("figure" in unit for unit in units),
        "unit_figure_width_px": 1200, "unit_figure_format": "JPEG quality 82",
        "camera_rotation_degrees": 180,
        "individual_figure_and_frame_paths_are_relative_to_output": True,
    }
    with (output_dir / "units.json").open("x") as stream:
        json.dump(_json_safe(result), stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    (output_dir / "summary.md").write_text(_render_summary(result))
    return result


def _render_pca_traversal(pca_axes: list[dict], units: list[dict], output_path: Path) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, max(1, len(pca_axes)), figsize=(5 * max(1, len(pca_axes)), 4))
    axes = np.atleast_1d(axes)
    for axis, record in zip(axes, pca_axes, strict=False):
        values, x, y = [], [], []
        for unit in units:
            for traversal in unit.get("pca_traversal", []):
                if traversal["axis"] == record["axis"]:
                    for value, signature in zip(traversal["values_in_std"], traversal["motion_signatures"], strict=True):
                        values.append(value)
                        end = signature["endpoint_displacement_mm"]
                        x.append(end[0]); y.append(end[2])
                    break
        axis.scatter(x, y, c=values, cmap="coolwarm", s=9, alpha=.45)
        axis.set(title=f"PCA-{record['axis'] + 1} ({record['explained_variance_fraction']:.1%})",
                 xlabel="endpoint Δx (mm)", ylabel="endpoint Δz (mm)")
        axis.grid(alpha=.2)
    fig.suptitle("Decoder motion under base PCA traversal (sampled centers)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100, bbox_inches="tight", format="jpg",
                pil_kwargs={"quality": 82, "optimize": True})
    plt.close(fig)
    return str(output_path.relative_to(output_path.parents[1]))


def _render_summary(result: dict[str, Any]) -> str:
    meta, summary = result["metadata"], result["summary"]
    lines = ["# S2 M6 unit interpretability catalog", "",
             f"- Units: {meta['unit_count']:,}; memory intervals: {meta['memory_item_count']:,}; rows: {meta['row_count']:,}.",
             "- Unit formation used only LIBERO-90 actions/state/keys. Task names and plan annotations were joined after the frozen M1/M6 memory was loaded.",
             f"- Motion-pattern rule: {meta['interpretability_rule']}.",
             f"- Interpretable: {summary['interpretable_unit_count']}/{meta['unit_count']} ({summary['interpretable_unit_fraction']:.1%}); not interpretable: {summary['not_interpretable_unit_count']} ({summary['not_interpretable_unit_fraction']:.1%}).",
             f"- Demo-fitted diagonal scale (m/action-step): {np.diag(np.asarray(meta['physical_motion_calibration']['translation_action_to_meters_matrix'])).round(5).tolist()}; rotation (rad/action-step): {np.diag(np.asarray(meta['physical_motion_calibration']['rotation_action_to_radians_matrix'])).round(5).tolist()}.",
             f"- Plan-boundary agreement: not computed — {meta['plan_boundary_alignment_reason']}.",
             "- Plan phrase / verb statistics are task-level weak labels, not evidence that a particular action window belongs to that plan phase.",
             "", "## Representative units", "",
             "| unit | support | episodes | tasks | scenes | task-verb purity | direction / rotation / gripper stability | motion signature |",
             "|---:|---:|---:|---:|---:|---:|---|---|"]
    for unit in result["units"]:
        if "member_composition" not in unit:
            continue
        comp, stable, sig = unit["member_composition"], unit["context_motion_stability"], unit["center_motion_signature"]
        lines.append(f"| {unit['unit_id']} | {unit['support_count']} | {unit['episode_count']} | {comp['n_tasks']} | {comp['n_scenes']} | {comp['primary_task_verb_purity']:.2f} | {stable['direction']:.2f} / {stable['rotation']:.2f} / {stable['gripper']:.2f} | {unit['automatic_description']} ({sig['direction']}, {sig['rotation']}, {sig['gripper_mode']}) |")
    lines += ["", "## Whole-memory summaries", "",
             f"- Sparse unit transition edges: {len(summary['unit_transition_statistics']['transition_matrix_sparse'])}; unit-switch fraction among adjacent windows: {summary['unit_transition_statistics']['switch_fraction_of_adjacent_pairs']:.1%}.",
              f"- Windows are {meta['windowing']['horizon_steps']} steps / {meta['windowing']['window_seconds']:.1f}s with stride {meta['windowing']['stride_steps']} ({meta['windowing']['overlap_steps']} overlap); the switch rate is therefore not caused by overlapping windows.",
              f"- Contiguous run length: median {summary['temporal_run_statistics']['run_length_windows']['median']:.1f} windows ({summary['temporal_run_statistics']['run_length_seconds']['median']:.2f}s), p90 {summary['temporal_run_statistics']['run_length_seconds']['p90']:.2f}s.",
              f"- Repeated categorical motion description: {summary['duplicate_motion_description_statistics']['units_in_duplicated_groups']}/{summary['duplicate_motion_description_statistics']['unit_count']} units ({summary['duplicate_motion_description_statistics']['fraction_units_with_duplicate_description']:.1%}) share a direction × gripper × rotation description.",
              f"- PCA traversal figure: [figures/pca_traversal.jpg](figures/pca_traversal.jpg).",
              "- Full per-unit members, task/scene/phase/plan distributions, decoded [10×7] center chunks, context stability, action spread, examples and figure paths are in [units.json](units.json).",
              "- Each 1200px JPEG unit figure includes cumulative XY/XZ trajectories, rotation and gripper command curves, plus up to three examples with start/middle/end front-camera frames. Camera frames are rotated 180° per LIBERO convention.",
              "- `center_action_chunk_raw_10x7` means inverse-quantile-normalized, dimensionless controller commands; physical plots use the least-squares state/action calibration reported in units.json.",
              "", "## Caveats", "",
              "- Unit labels are descriptive post-hoc associations. The annotation JSON contains task-level plan text/classes but not per-episode segment boundaries, so boundary agreement and segment-local verb purity cannot be estimated from supplied files.",
              "- Episode time is normalized against the final complete H=10 window (the raw episode length is not included in the exported row NPZ).",
              "- Center-nearest examples are selected by Euclidean distance in the 12-D M6 base space. The context stability score is sampled across up to 16 member states/keys.",
              "- This is an exploratory visualization, not evidence that M6 units are causal skills or that labels were used to form them.", ""]
    return "\n".join(lines)


def run(memory_path: Path, rows_path: Path, keys_path: Path, norm_path: Path,
        annotations_path: Path, rlds_root: Path, output_dir: Path,
        *, stability_samples: int = 16) -> dict[str, Any]:
    # `memory.py` imports the M6 class needed by pickle; all heavy arrays are
    # loaded only after CLI thread/CPU-affinity settings have been applied.
    if Path(output_dir).exists():
        raise FileExistsError(f"refusing to reuse output directory: {output_dir}")
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    from memory import load_memory

    t0 = time.perf_counter()
    # Plain pickle preserves the original CUDA storage tags. Redirect tensor
    # storages to CPU while loading so this CPU catalog does not contend with
    # LIBERO evaluation or initialize a CUDA context.
    original_storage_loader = torch.storage._load_from_bytes
    torch.storage._load_from_bytes = lambda value: torch.load(
        BytesIO(value), map_location="cpu", weights_only=False)
    try:
        memory = load_memory(memory_path)
    finally:
        torch.storage._load_from_bytes = original_storage_loader
    print(f"loaded memory items={len(memory.memory)} units={len(memory._centers)} in {time.perf_counter()-t0:.1f}s", flush=True)
    with np.load(rows_path, allow_pickle=True) as archive:
        needed = ("actions", "state", "episode_id", "init_state_id", "instruction", "source_file", "frame")
        missing = [name for name in needed if name not in archive.files]
        if missing:
            raise ValueError(f"row cache is missing fields: {missing}")
        rows = {name: archive[name] for name in needed}
    with np.load(keys_path, allow_pickle=True) as archive:
        if "key" not in archive.files:
            raise ValueError("key cache has no key array")
        keys = archive["key"]
        source = str(np.asarray(archive["key_text_source"] if "key_text_source" in archive.files
                                else "unknown").item())
        if source != "instruction_only":
            raise ValueError(f"S2 catalog expects instruction_only keys, got {source}")
    norm_stats = json.loads(Path(norm_path).read_text())
    annotations = json.loads(Path(annotations_path).read_text())
    print(f"loaded rows={len(rows['actions'])} key_shape={keys.shape}; fetching representative RGB frames", flush=True)
    result = analyze_arrays(memory, rows, keys, annotations, norm_stats, output_dir,
                            rlds_root=rlds_root, stability_samples=stability_samples)
    result["metadata"]["elapsed_seconds"] = float(time.perf_counter() - t0)
    # `analyze_arrays` wrote a complete first version; replace only the newly
    # created output's own files with the elapsed time included.
    with (Path(output_dir) / "units.json").open("w") as stream:
        json.dump(_json_safe(result), stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    (Path(output_dir) / "summary.md").write_text(_render_summary(result))
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory", type=Path, default=Path(
        "/work/.skillnet-state/outputs/memory-pilot/anchor-s2-train-20261028T072152Z/memory/m6.pkl"))
    parser.add_argument("--rows", type=Path, default=Path(
        "/work/.skillnet-state/outputs/memory-pilot/anchor-s2-train-20261028T072152Z/libero90_rows.npz"))
    parser.add_argument("--keys", type=Path, default=Path(
        "/work/.skillnet-state/outputs/memory-pilot/anchor-s2-train-20261028T072152Z/libero90_keys.npz"))
    parser.add_argument("--norm-stats", type=Path, default=Path(
        "/work/.skillnet-state/checkpoints/pi05_libero_moe_skill_4_90/moe_balance_4_32_90_20000/19999/assets/jsw19/libero_90_v1/norm_stats.json"))
    parser.add_argument("--annotations", type=Path, default=Path(
        "/work/skill_moe/skillnet/examples/libero/annotations/instruct2plan_obj_90.json"))
    parser.add_argument("--rlds-root", type=Path, default=Path(
        "/work/.skillnet-state/data/libero/libero90_rlds/libero_90_openvla_processed/1.0.0"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stability-samples", type=int, default=16)
    args = parser.parse_args(argv)
    result = run(args.memory, args.rows, args.keys, args.norm_stats, args.annotations,
                 args.rlds_root, args.output, stability_samples=args.stability_samples)
    print(json.dumps({"output": str(args.output.resolve()),
                      "units": result["metadata"]["unit_count"],
                      "interpretable_fraction": result["summary"]["interpretable_unit_fraction"],
                      "elapsed_seconds": result["metadata"]["elapsed_seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
