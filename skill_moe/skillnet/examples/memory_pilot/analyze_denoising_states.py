"""Probe when LIBERO motion modes become stable during SkillNet denoising.

Inference only. Demo observations are loaded from the local unsegmented
LIBERO-40 LeRobot data; no optimizer or simulator is used.
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import itertools
import json
import pathlib
import sys
import uuid

import jax
import jax.numpy as jnp
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[4]
STATE = pathlib.Path(__import__("os").environ.get("SKILLNET_STATE_DIR", ROOT / ".skillnet-state"))
DATA = pathlib.Path(__import__("os").environ.get(
    "SKILLNET_LIBERO40_DATA", STATE / "data/lerobot/emergent/libero_40_unsegmented"))
CHECKPOINT = pathlib.Path(__import__("os").environ.get(
    "SKILLNET_MEMORY_PILOT_CHECKPOINT",
    STATE / "checkpoints/pi05_libero_moe_skill_4_40/moe_add_balance_4_128_30000/29999"))
TASKS = {
    "goal": tuple(range(10, 20)),
    "spatial": tuple(range(30, 40)),
}
TRACE_LAYERS = (9, 17)
MOTION_NAMES = ("direction", "rotation", "gripper")

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "skill_moe/skillnet"))
from openpi import transforms
from openpi.models import model as model_lib
from openpi.policies.policy_config import create_trained_policy
from openpi.training import config_moe_skill
from skill_moe.skillnet.examples.emergent.eval.metrics import action_motion_labels


def _new_run_dir() -> pathlib.Path:
    root = STATE / "outputs" / "memory-pilot"
    root.mkdir(parents=True, exist_ok=True)
    while True:
        path = root / f"denoise-{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:6]}"
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue


def _rows_for_tasks(segments_per_task: int, seed: int):
    tasks = [json.loads(line) for line in (DATA / "meta/tasks.jsonl").read_text().splitlines()]
    description = {int(row["task_index"]): row["task"] for row in tasks}
    episodes = [json.loads(line) for line in (DATA / "meta/episodes.jsonl").read_text().splitlines()]
    by_description: dict[str, list[dict]] = {}
    for row in episodes:
        values = row.get("tasks", [])
        if values:
            by_description.setdefault(str(values[0]), []).append(row)

    rng = np.random.default_rng(seed)
    selected = []
    for suite, task_ids in TASKS.items():
        for task_id in task_ids:
            candidates = sorted(by_description.get(description[task_id], []), key=lambda r: int(r["episode_index"]))
            if not candidates:
                raise ValueError(f"no episodes found for {suite} task_index={task_id}")
            # Spread requested windows over up to four demos. This provides
            # episode-held-out evaluation while sampling distinct times.
            n_episodes = min(4, len(candidates), segments_per_task)
            picked_ix = np.sort(rng.choice(len(candidates), size=n_episodes, replace=False))
            picked = [candidates[int(i)] for i in picked_ix]
            quotas = np.full(n_episodes, segments_per_task // n_episodes, dtype=int)
            quotas[: segments_per_task % n_episodes] += 1
            for ep, quota in zip(picked, quotas, strict=True):
                episode_index = int(ep["episode_index"])
                path = DATA / "data" / f"chunk-{episode_index // 1000:03d}" / f"episode_{episode_index:06d}.parquet"
                table = pq.read_table(path, columns=["image", "wrist_image", "state", "eval_all_classes",
                                                      "task_index", "frame_index", "timestamp"])
                count = table.num_rows
                # Stay away from both episode boundaries; frame spacing makes
                # the sampled windows less redundant than adjacent rows.
                positions = np.linspace(0.12, 0.88, int(quota))
                row_indices = np.unique(np.rint(positions * max(count - 1, 0)).astype(int))
                if len(row_indices) < quota:
                    available = np.setdiff1d(np.arange(count), row_indices)
                    row_indices = np.sort(np.concatenate([row_indices, available[: quota - len(row_indices)]]))
                for row_index in row_indices[:quota]:
                    row = table.slice(int(row_index), 1).to_pylist()[0]
                    if int(row["task_index"]) != task_id:
                        raise ValueError(f"task mismatch in {path}: expected {task_id}, got {row['task_index']}")
                    selected.append({
                        "suite": suite, "task_index": task_id, "task": description[task_id],
                        "episode_index": episode_index, "frame_row": int(row_index),
                        "frame_index": int(row["frame_index"]), "timestamp": float(row["timestamp"]),
                        "state": np.asarray(row["state"], np.float32),
                        "skills": str(row["eval_all_classes"]),
                        "image": np.asarray(Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB"), np.uint8),
                        "wrist_image": np.asarray(Image.open(io.BytesIO(row["wrist_image"]["bytes"])).convert("RGB"), np.uint8),
                    })
    return selected


def _as_batched_observation(policy, row, n_noise: int):
    raw = {
        "observation/image": row["image"],
        "observation/wrist_image": row["wrist_image"],
        "observation/state": row["state"],
        "prompt": row["task"],
        "skills": row["skills"],
    }
    transformed = policy._input_transform(raw)
    batched = jax.tree.map(
        lambda value: jnp.broadcast_to(jnp.asarray(value)[None, ...], (n_noise,) + np.shape(value)), transformed
    )
    return model_lib.Observation.from_dict(batched), batched["state"]


def _policy_actions(policy, actions: np.ndarray, normalized_state) -> np.ndarray:
    unnormalize = next(transform for transform in policy._output_transform.transforms
                       if isinstance(transform, transforms.Unnormalize))
    transformed = unnormalize({"state": normalized_state, "actions": jnp.asarray(actions)})
    return np.asarray(transformed["actions"][..., :7], dtype=np.float32)


def _labels(actions: np.ndarray) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    masks = np.ones(actions.shape[:-1], dtype=bool)
    result = action_motion_labels(actions, masks)
    return np.stack([result[name] for name in MOTION_NAMES], axis=-1).astype(np.int16)


def _episode_split(episode_ids: np.ndarray, task_ids: np.ndarray, seed: int):
    """Hold out one complete demonstration per task, with remaining demos for train."""
    rng = np.random.default_rng(seed)
    train, test = [], []
    for task in np.unique(task_ids):
        groups = np.unique(episode_ids[task_ids == task])
        if len(groups) < 2:
            continue
        held = groups[int(rng.integers(len(groups)))]
        test.extend(np.flatnonzero((task_ids == task) & (episode_ids == held)).tolist())
        train.extend(np.flatnonzero((task_ids == task) & (episode_ids != held)).tolist())
    return np.asarray(train, dtype=int), np.asarray(test, dtype=int)


def _probe_accuracies(x, targets, train_ix, test_ix, *, seed: int):
    """Fit closed-form ridge classifiers, a small CPU-only linear probe."""
    train_x = np.asarray(x[train_ix], dtype=np.float64)
    test_x = np.asarray(x[test_ix], dtype=np.float64)
    mean, scale = train_x.mean(axis=0), train_x.std(axis=0)
    scale[scale < 1e-6] = 1.0
    train_x, test_x = (train_x - mean) / scale, (test_x - mean) / scale
    # Random projection keeps the exact ridge solve small for 1024-D hidden
    # states. For action chunks (70-D) retain all input dimensions.
    if train_x.shape[1] > 128:
        rng = np.random.default_rng(seed)
        projection = rng.standard_normal((train_x.shape[1], 128)) / np.sqrt(128.0)
        train_x, test_x = train_x @ projection, test_x @ projection
    train_x = np.concatenate([train_x, np.ones((len(train_x), 1))], axis=1)
    test_x = np.concatenate([test_x, np.ones((len(test_x), 1))], axis=1)
    gram = train_x.T @ train_x
    gram.flat[::len(gram) + 1] += 1.0
    results = {}
    for name, y in targets.items():
        train_y = np.asarray(y[train_ix])
        classes = np.unique(train_y)
        if len(classes) < 2 or not len(test_ix):
            results[name] = None
            continue
        one_hot = (train_y[:, None] == classes[None, :]).astype(np.float64)
        weights = np.linalg.solve(gram, train_x.T @ one_hot)
        pred = classes[np.argmax(test_x @ weights, axis=1)]
        test_y = np.asarray(y[test_ix])
        results[name] = {"accuracy": float(np.mean(pred == test_y)),
                         "majority_baseline": float(np.max(np.unique(test_y, return_counts=True)[1]) / len(test_y))}
    return results


def _mean_pair_distance(features, labels, tasks, segments, *, mode: str, seed: int, max_pairs: int = 30000):
    """Sample eligible pairs while excluding replicas of the same observation."""
    rng = np.random.default_rng(seed)
    chunks = []
    per_cell = max(1, max_pairs // 400)
    if mode == "same_motion_different_task":
        first_key, second_key = labels, tasks
    else:
        first_key, second_key = tasks, labels
    for first in np.unique(first_key):
        first_rows = np.flatnonzero(first_key == first)
        # Each cell is defined by (motion, task) or (task, motion). Sample
        # from two distinct secondary groups within the same first group.
        secondary_values = np.unique(second_key[first_rows])
        if len(secondary_values) < 2:
            continue
        candidates = []
        for a, b in itertools.combinations(secondary_values, 2):
            left = np.flatnonzero((first_key == first) & (second_key == a))
            right = np.flatnonzero((first_key == first) & (second_key == b))
            if not len(left) or not len(right):
                continue
            n = min(per_cell, len(left) * len(right))
            li = rng.choice(left, n, replace=n > len(left))
            ri = rng.choice(right, n, replace=n > len(right))
            keep = segments[li] != segments[ri]
            if np.any(keep):
                candidates.append(np.stack([li[keep], ri[keep]], axis=1))
        if candidates:
            chunks.extend(candidates)
    if not chunks:
        return None, 0
    pairs = np.concatenate(chunks, axis=0)
    if len(pairs) > max_pairs:
        pairs = pairs[rng.choice(len(pairs), max_pairs, replace=False)]
    dist = []
    for start in range(0, len(pairs), 512):
        pair = pairs[start:start + 512]
        dist.extend(np.linalg.norm(features[pair[:, 0]] - features[pair[:, 1]], axis=1).tolist())
    return float(np.mean(dist)), len(dist)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segments-per-task", type=int, default=20)
    ap.add_argument("--noises", type=int, default=8)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20261027)
    args = ap.parse_args()
    if args.steps != 10:
        raise ValueError("the released SkillNet analysis requires the normal 10 denoising steps")
    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)
    if not DATA.exists():
        raise FileNotFoundError(DATA)

    run_dir = _new_run_dir()
    manifest = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "checkpoint": str(CHECKPOINT), "config": "pi05_libero_moe_skill_4_40",
        "dataset": str(DATA), "seed": args.seed, "segments_per_task": args.segments_per_task,
        "num_noises": args.noises, "num_denoising_steps": args.steps,
        "task_indices": {name: list(ids) for name, ids in TASKS.items()},
        "trace_layers": list(TRACE_LAYERS),
        "trace_layer_aggregation": "mean over action suffix tokens after each transformer block",
        "x_t": "first seven model action dimensions in normalized flow space; padded action dims omitted",
        "x0_hat": "Unnormalize transformed full action horizon, first seven LIBERO action dimensions in dataset command units",
        "episode_split": "one complete demonstration held out per task for linear probes",
        "linear_probe": "CPU ridge one-vs-rest regression; hidden features use seeded 128-D random projection",
        "records_completed": 0,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    config = config_moe_skill.get_config("pi05_libero_moe_skill_4_40")
    policy = create_trained_policy(config, CHECKPOINT, repack_transforms=transforms.Group(),
                                   sample_kwargs={"num_steps": args.steps})
    rows = _rows_for_tasks(args.segments_per_task, args.seed)
    print(f"run={run_dir} intervals={len(rows)}; loading selected demonstrations", flush=True)

    x_trace, x0_trace, hidden_trace, final_actions, metadata, noise_seeds = [], [], [], [], [], []
    for row_number, row in enumerate(rows):
        seed_seq = np.random.SeedSequence([args.seed, row["task_index"], row["episode_index"], row["frame_index"]])
        noise_seed = int(seed_seq.generate_state(1, dtype=np.uint32)[0])
        noise_seeds.append(noise_seed)
        noise = np.random.default_rng(noise_seed).standard_normal(
            (args.noises, config.model.action_horizon, config.model.action_dim)
        ).astype(np.float32)
        observation, normalized_state = _as_batched_observation(policy, row, args.noises)
        output = policy._sample_actions(
            policy._rng, observation, num_steps=args.steps, noise=jnp.asarray(noise),
            capture_denoising_trace=True,
        )
        final, xt, x0, hidden, times = jax.tree.map(lambda value: np.asarray(value), output)
        # Materialize each denoising estimate in the same physical command
        # space used by the evaluation-time action-motion label function.
        x0_physical = np.stack([_policy_actions(policy, x0[t], normalized_state) for t in range(args.steps)])
        final_physical = _policy_actions(policy, final, normalized_state)
        x_trace.append(xt[..., :7].astype(np.float16))
        x0_trace.append(x0_physical.astype(np.float16))
        hidden_trace.append(np.transpose(hidden, (0, 2, 1, 3)).astype(np.float16))  # [time,noise,layer,width]
        final_actions.append(final_physical.astype(np.float32))
        metadata.append({k: row[k] for k in ("suite", "task_index", "task", "episode_index", "frame_index", "timestamp")})
        if (row_number + 1) % 10 == 0 or row_number + 1 == len(rows):
            manifest["records_completed"] = row_number + 1
            (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
            print(f"recorded {row_number + 1}/{len(rows)} intervals", flush=True)

    # Per-call model outputs are [time,noise,...]; reorder to interval/noise/time.
    x_trace = np.transpose(np.stack(x_trace), (0, 2, 1, 3, 4)).astype(np.float16)
    x0_trace = np.transpose(np.stack(x0_trace), (0, 2, 1, 3, 4)).astype(np.float16)
    hidden_trace = np.transpose(np.stack(hidden_trace), (0, 2, 1, 3, 4)).astype(np.float16)
    final_actions = np.stack(final_actions)
    metadata = np.asarray(metadata, dtype=object)
    metadata_json = np.asarray([json.dumps(item, ensure_ascii=False) for item in metadata])
    labels_npz = run_dir / "denoising_traces.npz"
    np.savez_compressed(labels_npz, x_t=x_trace, x0_hat=x0_trace, hidden_mean=hidden_trace,
                        final_actions=final_actions, task_index=np.asarray([item["task_index"] for item in metadata], np.int32),
                        episode_index=np.asarray([item["episode_index"] for item in metadata], np.int64),
                        segment_id=np.arange(len(rows), dtype=np.int32), noise_seed=np.asarray(noise_seeds, np.uint32),
                        timesteps=np.asarray(times, np.float32), metadata_json=metadata_json)
    final_labels = _labels(final_actions.reshape(-1, *final_actions.shape[-2:])).reshape(
        len(rows), args.noises, len(MOTION_NAMES)
    )
    x0_labels = np.stack([
        _labels(x0_trace[:, :, t].reshape(-1, *x0_trace.shape[-2:])).reshape(len(rows), args.noises, len(MOTION_NAMES))
        for t in range(args.steps)
    ])
    tasks = np.asarray([item["task_index"] for item in metadata], np.int32)
    episodes = np.asarray([item["episode_index"] for item in metadata], np.int64)
    segments = np.arange(len(rows), dtype=np.int32)
    np.savez_compressed(run_dir / "motion_labels.npz", final_motion_labels=final_labels,
                        x0_motion_labels=x0_labels, task_index=tasks, episode_index=episodes,
                        segment_id=segments, timesteps=np.asarray(times, np.float32),
                        metadata_json=metadata_json)

    # Expand intervals into noise replicas; grouping remains at source-demo level.
    m, n_noise = len(rows), args.noises
    episode_samples = np.repeat(episodes, n_noise)
    task_samples = np.repeat(tasks, n_noise)
    segment_samples = np.repeat(segments, n_noise)
    task_train, task_test = _episode_split(episode_samples, task_samples, args.seed + 1)
    probe_rows = []
    match_rows = []
    distance_rows = []
    multimodal = np.any(np.ptp(final_labels, axis=1) > 0, axis=1)
    labels_flat_final = final_labels.reshape(m * n_noise, len(MOTION_NAMES))
    multi_samples = np.repeat(multimodal, n_noise)

    for ti, time_value in enumerate(np.asarray(times)):
        labels_at_time = x0_labels[ti].reshape(m * n_noise, len(MOTION_NAMES))
        for j, name in enumerate(MOTION_NAMES):
            match_rows.append({"t": float(time_value), "metric": f"match_{name}_all",
                               "value": float(np.mean(labels_at_time[:, j] == labels_flat_final[:, j]))})
            if multi_samples.any():
                match_rows.append({"t": float(time_value), "metric": f"match_{name}_multimodal",
                                   "value": float(np.mean(labels_at_time[multi_samples, j] == labels_flat_final[multi_samples, j]))})
        joint_final = np.asarray(["|".join(map(str, row)) for row in labels_flat_final])
        joint_at_time = np.asarray(["|".join(map(str, row)) for row in labels_at_time])
        match_rows.append({"t": float(time_value), "metric": "match_joint_all",
                           "value": float(np.mean(joint_at_time == joint_final))})
        if multi_samples.any():
            match_rows.append({"t": float(time_value), "metric": "match_joint_multimodal",
                               "value": float(np.mean(joint_at_time[multi_samples] == joint_final[multi_samples]))})

        reps = {"x0_hat": x0_trace[:, :, ti].reshape(m * n_noise, -1).astype(np.float32),
                f"L{TRACE_LAYERS[0]:02d}": hidden_trace[:, :, ti, 0].reshape(m * n_noise, -1).astype(np.float32),
                f"L{TRACE_LAYERS[1]:02d}": hidden_trace[:, :, ti, 1].reshape(m * n_noise, -1).astype(np.float32)}
        for rep_name, features in reps.items():
            accuracies = _probe_accuracies(features,
                                           {"task": task_samples, **{name: labels_at_time[:, j]
                                                                    for j, name in enumerate(MOTION_NAMES)}},
                                           task_train, task_test, seed=args.seed + ti)
            for target_name, acc in accuracies.items():
                probe_rows.append({"t": float(time_value), "representation": rep_name,
                                   "target": target_name, **(acc or {"accuracy": None, "majority_baseline": None})})
            # Standardized distances keep scale from dominating the ratio.
            distance_mean = features.mean(axis=0, keepdims=True)
            distance_scale = features.std(axis=0, keepdims=True)
            distance_scale[distance_scale < 1e-6] = 1.0
            z = ((features - distance_mean) / distance_scale).astype(np.float32)
            for name in MOTION_NAMES:
                label = labels_at_time[:, MOTION_NAMES.index(name)]
                same_motion, n_same = _mean_pair_distance(z, label, task_samples, segment_samples,
                                                           mode="same_motion_different_task", seed=args.seed + ti)
                same_task, n_task = _mean_pair_distance(z, label, task_samples, segment_samples,
                                                        mode="same_task_different_motion", seed=args.seed + ti + 1)
                distance_rows.append({"t": float(time_value), "representation": rep_name,
                                      "motion_label": name, "same_motion_different_task_distance": same_motion,
                                      "same_task_different_motion_distance": same_task,
                                      "distance_ratio": (same_motion / same_task if same_motion is not None and same_task else None),
                                      "n_same_motion_pairs": n_same, "n_same_task_pairs": n_task})

    results = {"timesteps": np.asarray(times).tolist(), "motion_match": match_rows,
               "linear_probes": probe_rows, "scene_invariance": distance_rows,
               "heldout_episodes": sorted(np.unique(episode_samples[task_test]).astype(int).tolist()),
               "train_episodes": int(len(np.unique(episode_samples[task_train]))),
               "test_episodes": int(len(np.unique(episode_samples[task_test]))),
               "multimodal_intervals": int(multimodal.sum()), "intervals": int(m),
               "notes": ["x0_hat labels use existing action_motion_labels on denormalized first-seven action commands.",
                         "Probe and distance noise replicas are grouped by source episode for train/test assignment.",
                         "Scene-distance pairs exclude different noises from the same sampled observation."]}
    (run_dir / "metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    _plot(run_dir / "denoising_curves.png", results)
    (run_dir / "report.md").write_text(_report(run_dir, results, args))
    print(f"completed: {run_dir}", flush=True)


def _plot(path: pathlib.Path, results: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times = results["timesteps"]
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.4))
    for metric, label in (("match_joint_all", "all intervals"), ("match_joint_multimodal", "multimodal intervals")):
        vals = {row["t"]: row["value"] for row in results["motion_match"] if row["metric"] == metric}
        if vals:
            axes[0].plot(times, [vals.get(t, np.nan) for t in times], marker="o", label=label)
    axes[0].set_title("1. x̂₀(t) label matches final output")
    axes[0].set_ylim(0, 1.02); axes[0].set_ylabel("fraction"); axes[0].legend()
    for rep in ("x0_hat", f"L{TRACE_LAYERS[0]:02d}", f"L{TRACE_LAYERS[1]:02d}"):
        for target, style in (("task", "-"), ("direction", "--"), ("rotation", ":"), ("gripper", "-.")):
            subset = [r for r in results["linear_probes"] if r["representation"] == rep and r["target"] == target]
            axes[1].plot([r["t"] for r in subset], [np.nan if r["accuracy"] is None else r["accuracy"] for r in subset],
                         linestyle=style, marker=".", label=f"{rep}/{target}")
    axes[1].set_title("2. Episode-held-out linear probe accuracy"); axes[1].set_ylabel("accuracy"); axes[1].legend(fontsize=7, ncol=2)
    for rep in ("x0_hat", f"L{TRACE_LAYERS[0]:02d}", f"L{TRACE_LAYERS[1]:02d}"):
        for motion in MOTION_NAMES:
            subset = [r for r in results["scene_invariance"] if r["representation"] == rep and r["motion_label"] == motion]
            axes[2].plot([r["t"] for r in subset], [np.nan if r["distance_ratio"] is None else r["distance_ratio"] for r in subset],
                         marker=".", label=f"{rep}/{motion}")
    axes[2].axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    axes[2].set_title("3. same-motion/different-task ÷ same-task/different-motion")
    axes[2].set_ylabel("standardized distance ratio"); axes[2].legend(fontsize=7, ncol=2)
    for ax in axes:
        ax.set_xlabel("denoising input t (1 → 0)")
        ax.invert_xaxis(); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def _report(run_dir: pathlib.Path, results: dict, args) -> str:
    times = results["timesteps"]
    def curve_value(metric, index):
        rows = [row for row in results["motion_match"] if row["metric"] == metric]
        return min(rows, key=lambda row: abs(row["t"] - times[index]))["value"]

    task_probe = {(row["representation"], row["t"]): row["accuracy"]
                  for row in results["linear_probes"] if row["target"] == "task"}
    def distance_mean(rep, index):
        t = times[index]
        values = [row["distance_ratio"] for row in results["scene_invariance"]
                  if row["representation"] == rep and abs(row["t"] - t) < 1e-5
                  and row["distance_ratio"] is not None]
        return float(np.mean(values)) if values else float("nan")

    probes = [r for r in results["linear_probes"] if r["target"] == "task"]
    lines = [
        "# LIBERO denoising-state probe", "", f"Output: `{run_dir}`",
        f"Checkpoint: `{CHECKPOINT}`", f"Dataset: `{DATA}`",
        f"Intervals: {results['intervals']} ({args.segments_per_task} per task × 20 tasks), noises/interval: {args.noises}; 10 denoising steps.",
        f"Multimodal final-label intervals: {results['multimodal_intervals']}/{results['intervals']}.",
        f"Episode split: train {results['train_episodes']} demos, test {results['test_episodes']} demos; one held-out demo per task.", "",
        "## Findings", "",
        f"- Joint label agreement at t={times[0]:.1f} → {times[-1]:.1f}: all intervals {curve_value('match_joint_all', 0):.3f} → {curve_value('match_joint_all', -1):.3f}; multimodal intervals {curve_value('match_joint_multimodal', 0):.3f} → {curve_value('match_joint_multimodal', -1):.3f}.",
        f"- Before the terminal Euler step, multimodal agreement at t={times[-2]:.1f} is {curve_value('match_joint_multimodal', -2):.3f}.",
        f"- Task-probe accuracy at t={times[0]:.1f} → {times[-1]:.1f}: x̂₀ {task_probe.get(('x0_hat', times[0]), float('nan')):.3f} → {task_probe.get(('x0_hat', times[-1]), float('nan')):.3f}; L09 {task_probe.get((f'L{TRACE_LAYERS[0]:02d}', times[0]), float('nan')):.3f} → {task_probe.get((f'L{TRACE_LAYERS[0]:02d}', times[-1]), float('nan')):.3f}; L17 {task_probe.get((f'L{TRACE_LAYERS[1]:02d}', times[0]), float('nan')):.3f} → {task_probe.get((f'L{TRACE_LAYERS[1]:02d}', times[-1]), float('nan')):.3f}.",
        f"- Motion probes: L17 direction {next(r['accuracy'] for r in results['linear_probes'] if r['representation'] == f'L{TRACE_LAYERS[1]:02d}' and r['target'] == 'direction' and abs(r['t']-times[0])<1e-5):.3f} → {next(r['accuracy'] for r in results['linear_probes'] if r['representation'] == f'L{TRACE_LAYERS[1]:02d}' and r['target'] == 'direction' and abs(r['t']-times[-1])<1e-5):.3f}; L09 rotation {next(r['accuracy'] for r in results['linear_probes'] if r['representation'] == f'L{TRACE_LAYERS[0]:02d}' and r['target'] == 'rotation' and abs(r['t']-times[0])<1e-5):.3f} → {next(r['accuracy'] for r in results['linear_probes'] if r['representation'] == f'L{TRACE_LAYERS[0]:02d}' and r['target'] == 'rotation' and abs(r['t']-times[-1])<1e-5):.3f}.",
        f"- Mean scene-distance ratio (direction/rotation/gripper): x̂₀ {distance_mean('x0_hat', 0):.3f} → {distance_mean('x0_hat', -1):.3f}; L09 {distance_mean(f'L{TRACE_LAYERS[0]:02d}', 0):.3f} → {distance_mean(f'L{TRACE_LAYERS[0]:02d}', -1):.3f}; L17 {distance_mean(f'L{TRACE_LAYERS[1]:02d}', 0):.3f} → {distance_mean(f'L{TRACE_LAYERS[1]:02d}', -1):.3f}. Ratios below 1 favor cross-task same-motion proximity.",
        "- Full curves and per-layer/per-label values are in `metrics.json`; plot is `denoising_curves.png`.", "",
        "## Interpretation limits", "",
        "- x̂₀ motion labels are derived from the model's own action estimates; their agreement measures mode stability, not correctness against the demo action.",
        "- At t=0.1, x̂₀(t)=x_t−t·v_t equals the next/final Euler state because Δt=−0.1; its 1.0 agreement is therefore an identity, not an independent decision signal.",
        "- Motion labels use the existing direction/rotation/gripper bins. Scene invariance uses standardized Euclidean distances and excludes noise replicas of the same observation.",
        "- The probe is a single fixed episode-held-out split; hidden vectors are projected by a seeded random 128-D linear map before ridge fitting. The 8 noises for one interval are correlated and do not count as independent scenes.",
        "- `x_t` stores only the first seven normalized model-action dimensions; padded action dimensions are omitted. Hidden states are suffix-token means from zero-based layers 9 and 17.", "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
