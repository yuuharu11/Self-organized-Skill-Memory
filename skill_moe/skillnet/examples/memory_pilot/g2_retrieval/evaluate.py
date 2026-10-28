"""CPU-only evaluation of whether saved policy features retrieve similar motion."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EvaluationConfig:
    horizon: int = 8
    query_fraction: float = 0.25
    seed: int = 0
    neighbor_exclusion_steps: int = 8
    k_values: tuple[int, ...] = (1, 5)
    bootstrap_samples: int = 10000


@dataclass
class Episode:
    episode_id: str
    init_state_id: str
    arrays: dict[str, np.ndarray]
    env_step_offset: int = 0


def _episode_entries(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    entries = manifest.get("episodes", [])
    if isinstance(entries, dict):
        return [dict(value, episode_id=key) if isinstance(value, dict) else {"episode_id": key}
                for key, value in entries.items()]
    if not isinstance(entries, list):
        raise ValueError("manifest 'episodes' must be a list or object")
    return entries


def load_run(run_dir: str | Path) -> tuple[dict[str, Any], list[Episode]]:
    """Load manifest and per-episode NPZ files without loading object arrays."""
    run_dir = Path(run_dir)
    with (run_dir / "manifest.json").open(encoding="utf-8") as f:
        manifest = json.load(f)
    entries = _episode_entries(manifest)
    episodes: list[Episode] = []
    seen_files: set[tuple[int, int]] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("each manifest episode entry must be an object")
        eid = str(entry.get("episode_id", entry.get("id", "")))
        if not eid:
            raise ValueError("manifest episode entry is missing episode_id")
        path = run_dir / "episodes" / f"{eid}.npz"
        if not path.exists():
            # Allow an explicitly named relative path while keeping episodes/ the default.
            named = entry.get("file", entry.get("path"))
            path = run_dir / str(named) if named else path
        if not path.exists():
            raise FileNotFoundError(f"episode file not found for {eid}: {path}")
        identity = (path.stat().st_dev, path.stat().st_ino)
        if identity in seen_files:
            raise ValueError("multiple episode IDs refer to the same episode file")
        seen_files.add(identity)
        with np.load(path, allow_pickle=False) as z:
            arrays = {key: np.asarray(z[key]) for key in z.files}
        init_id = entry.get("init_state_id", entry.get("initial_state_id"))
        if init_id is None:
            raise ValueError(f"manifest episode {eid} is missing init_state_id")
        # Recorder IDs are commonly local to a task (for example init_state_id=0
        # for each LIBERO task), so namespace them by suite and task when available.
        suite = entry.get("suite", manifest.get("suite"))
        task = entry.get("task_id", entry.get("task", ""))
        qualified_id = json.dumps([str(suite or ""), str(task), str(init_id)])
        offset = entry.get("env_step_offset", manifest.get("env_step_offset"))
        if offset is None and manifest.get("format_version") == 1:
            # The v1 recorder saves post-action observations only: T executed
            # steps give T env rows, whose first observation is at step 1.
            # A recorder including the reset observation gives T+1 rows.
            total_steps = entry.get("steps")
            if not isinstance(total_steps, int) or "env_eef_pos" not in arrays:
                raise ValueError(f"{eid}: v1 needs steps metadata or explicit env_step_offset")
            offset = total_steps + 1 - len(arrays["env_eef_pos"])
            if offset not in (0, 1):
                raise ValueError(f"{eid}: cannot establish environment time origin")
        offset = 0 if offset is None else offset
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("env_step_offset must be a nonnegative integer")
        episodes.append(Episode(eid, qualified_id, arrays, offset))
    if len({ep.episode_id for ep in episodes}) != len(episodes):
        raise ValueError("manifest contains duplicate episode IDs")
    return manifest, episodes


def split_by_init_state(episodes: list[Episode], query_fraction: float, seed: int) -> tuple[set[str], set[str]]:
    """Split whole initial-state groups; return query-side and store-side episode IDs."""
    if not 0 < query_fraction < 1:
        raise ValueError("query_fraction must be strictly between 0 and 1")
    groups: dict[str, list[str]] = {}
    for ep in episodes:
        groups.setdefault(ep.init_state_id, []).append(ep.episode_id)
    init_ids = sorted(groups)
    if len(init_ids) < 2:
        raise ValueError("at least two distinct init_state_id groups are required for leakage-safe split")
    rng = np.random.default_rng(seed)
    order = list(np.asarray(init_ids)[rng.permutation(len(init_ids))])
    n_query = min(len(init_ids) - 1, max(1, int(round(len(init_ids) * query_fraction))))
    query_init = set(order[:n_query])
    query_ids = {eid for init_id in query_init for eid in groups[init_id]}
    store_ids = {ep.episode_id for ep in episodes} - query_ids
    if query_ids & store_ids:
        raise AssertionError("episode leakage: query and store episode IDs overlap")
    query_inits = {ep.init_state_id for ep in episodes if ep.episode_id in query_ids}
    store_inits = {ep.init_state_id for ep in episodes if ep.episode_id in store_ids}
    if query_inits & store_inits:
        raise AssertionError("initial-state leakage: query and store groups overlap")
    return query_ids, store_ids


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if not np.all(np.isfinite(q)) or np.any(norm < 1e-12) or not np.all(np.isfinite(norm)):
        raise ValueError("quaternions must be finite and nonzero")
    return q / np.maximum(norm, 1e-12)


def _rotation_angle(q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
    """Shortest relative rotation angle in radians; quaternions use xyzw order."""
    a, b = _quat_normalize(q0), _quat_normalize(q1)
    dot = np.abs(np.sum(a * b, axis=-1))
    return 2.0 * np.arccos(np.clip(dot, 0.0, 1.0))


def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product for xyzw quaternions."""
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack((aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz), axis=-1)


def _valid_future(arrays: dict[str, np.ndarray], step: int, horizon: int) -> bool:
    required = ("env_eef_pos", "env_eef_quat", "env_gripper")
    if any(key not in arrays for key in required):
        missing = [key for key in required if key not in arrays]
        raise ValueError(f"episode is missing environment motion arrays: {missing}")
    n = len(arrays["env_eef_pos"])
    return 0 <= step < n and step + horizon < n


def _motion_target(arrays: dict[str, np.ndarray], step: int, horizon: int) -> np.ndarray:
    """Relative position, rotation, and gripper trajectories over H future steps."""
    pos = np.asarray(arrays["env_eef_pos"], dtype=np.float64)
    quat = np.asarray(arrays["env_eef_quat"], dtype=np.float64)
    grip = np.asarray(arrays["env_gripper"], dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError("env_eef_pos must have shape (T, 3)")
    if quat.ndim != 2 or quat.shape[1] != 4:
        raise ValueError("env_eef_quat must have shape (T, 4), in xyzw order")
    if grip.ndim == 1:
        grip = grip[:, None]
    if grip.ndim != 2 or grip.shape[1] == 0 or len(grip) != len(pos) or len(quat) != len(pos):
        raise ValueError("env_eef_pos, env_eef_quat and env_gripper must have matching time lengths")
    if not all(np.all(np.isfinite(x)) for x in (pos, quat, grip)):
        raise ValueError("environment motion arrays must be finite")
    offsets = np.arange(1, horizon + 1)
    future = step + offsets
    dpos = pos[future] - pos[step]
    # q0^-1 * qfuture expresses each change in the starting end-effector frame.
    q0 = _quat_normalize(quat[step])
    q0_inv = np.concatenate((-q0[:3], q0[3:]))
    drot = _quat_normalize(_quat_multiply(q0_inv, _quat_normalize(quat[future])))
    dgrip = grip[future] - grip[step]
    return np.concatenate([dpos, drot, dgrip], axis=1)


def _outcome_distance(a: np.ndarray, b: np.ndarray, horizon: int) -> dict[str, float]:
    diff = a - b
    gripper_diff = diff[:, 7:]
    return {
        "eef_relative_position": float(np.linalg.norm(diff[:, :3], axis=1).mean()),
        # Compare rotations on SO(3), not just magnitudes: +theta and -theta
        # differ, while q and -q represent the same rotation.
        "eef_relative_rotation_rad": float(_rotation_angle(a[:, 3:7], b[:, 3:7]).mean()),
        "gripper_relative": float(np.abs(gripper_diff).mean()),
    }


def _feature_matrix(ep: Episode, key: str) -> np.ndarray:
    values = np.asarray(ep.arrays[key], dtype=np.float64)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError(f"{key} must have shape (N, D) with D > 0")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{key} must contain only finite values")
    return values


def _stack_rows(episodes: list[Episode], ids: set[str], key: str, horizon: int):
    rows: list[dict[str, Any]] = []
    for ep in episodes:
        if ep.episode_id not in ids:
            continue
        if "step" not in ep.arrays:
            raise ValueError(f"episode {ep.episode_id} is missing step")
        steps = np.asarray(ep.arrays["step"])
        if (steps.ndim != 1 or not np.issubdtype(steps.dtype, np.integer)
                or np.any(steps < 0) or np.any(steps[1:] <= steps[:-1])):
            raise ValueError(f"{ep.episode_id}: step must be increasing nonnegative integers")
        feat = _feature_matrix(ep, key)
        if len(feat) != len(steps):
            raise ValueError(f"{key} and step lengths differ in {ep.episode_id}")
        for i, step in enumerate(steps):
            env_index = int(step) - ep.env_step_offset
            if env_index < 0:
                raise ValueError(f"{ep.episode_id}: query observation precedes environment records")
            if _valid_future(ep.arrays, env_index, horizon):
                rows.append({"episode_id": ep.episode_id, "init_state_id": ep.init_state_id,
                             "step": int(step), "feature": feat[i],
                             "target": _motion_target(ep.arrays, env_index, horizon)})
    return rows


def _distances(query: np.ndarray, store: np.ndarray) -> np.ndarray:
    # Scale each dimension using store-only statistics to avoid unit domination.
    mean = store.mean(axis=0)
    scale = store.std(axis=0)
    scale[scale < 1e-8] = 1.0
    distances = np.linalg.norm((query[None, :] - mean) / scale - (store - mean) / scale, axis=1)
    if not np.all(np.isfinite(distances)):
        raise ValueError("retrieval distances are not finite")
    return distances


def _evaluate_ranked(query_rows, store_rows, order_fn, k_values, horizon, rng=None):
    sums = {str(k): {m: 0.0 for m in ("eef_relative_position", "eef_relative_rotation_rad", "gripper_relative")}
            for k in k_values}
    n = 0
    per_query = []
    for q in query_rows:
        candidates = []
        for s in store_rows:
            # Guard against same-episode and adjacent-window contamination even if split code changes.
            if s["episode_id"] == q["episode_id"]:
                continue
            if s["init_state_id"] == q["init_state_id"]:
                continue
            if s["episode_id"] == q["episode_id"] and abs(s["step"] - q["step"]) <= horizon:
                continue
            candidates.append(s)
        if not candidates:
            continue
        order, distances = order_fn(q, candidates, rng)
        query_scores = []
        for k in k_values:
            effective_k = min(k, len(order))
            boundary = distances[order[effective_k - 1]]
            closer = np.flatnonzero(distances < boundary)
            tied = np.flatnonzero(distances == boundary)
            chosen = np.concatenate((closer, tied))
            weights = np.concatenate((np.ones(len(closer)),
                                      np.full(len(tied), (effective_k - len(closer)) / len(tied))))
            # Distances to the query's future motion are averaged over retrieved windows.
            # At the k boundary, integrate uniform tie breaking rather than favoring
            # manifest order. The random baseline uses an all-tied ranking.
            dists = [_outcome_distance(q["target"], candidates[i]["target"], horizon) for i in chosen]
            scores = []
            for metric in sums[str(k)]:
                score = float(np.dot(weights, [d[metric] for d in dists]) / effective_k)
                sums[str(k)][metric] += score
                scores.append(score)
            query_scores.append(scores)
        per_query.append(query_scores)
        n += 1
    if n == 0:
        raise ValueError("no valid query rows have eligible store neighbors")
    return ({k: {metric: value / n for metric, value in vals.items()} for k, vals in sums.items()},
            n, np.asarray(per_query))


def _feature_order(key: str):
    def order(q, candidates, _rng):
        matrix = np.stack([c["feature"] for c in candidates])
        distances = _distances(q["feature"], matrix)
        return np.argsort(distances, kind="stable"), distances
    return order


def _random_order(q, candidates, rng):
    # Expected error of uniform sampling without replacement, for every k.
    return np.arange(len(candidates)), np.zeros(len(candidates))


def _oracle_order(q, candidates, _rng):
    # A separate oracle ordering per motion component is computed in _evaluate_oracle below.
    return np.arange(len(candidates))


def _evaluate_oracle(query_rows, store_rows, k_values, horizon):
    metrics = ("eef_relative_position", "eef_relative_rotation_rad", "gripper_relative")
    results = {m: {str(k): 0.0 for k in k_values} for m in metrics}
    n = 0
    for q in query_rows:
        candidates = [s for s in store_rows if s["episode_id"] != q["episode_id"] and
                      s["init_state_id"] != q["init_state_id"]]
        if not candidates:
            continue
        per = [_outcome_distance(q["target"], s["target"], horizon) for s in candidates]
        for metric in metrics:
            order = np.argsort([x[metric] for x in per], kind="stable")
            for k in k_values:
                chosen = order[:min(k, len(order))]
                results[metric][str(k)] += float(np.mean([per[i][metric] for i in chosen]))
        n += 1
    if n == 0:
        raise ValueError("no valid query rows have oracle candidates")
    return {m: {k: val / n for k, val in by_k.items()} for m, by_k in results.items()}


def _compare_to_proprio(query_rows, scores, config):
    """Paired, whole-initial-state bootstrap, conditional on this retrieval store.

    Keep all overlapping windows and repeat episodes of a sampled initial state
    together. Recompute the row-weighted mean, matching the reported metric.
    Bonferroni intervals cover all tested features, k values and components.
    """
    metrics = ("eef_relative_position", "eef_relative_rotation_rad", "gripper_relative")
    names = [name for name in scores if name != "proprio"]
    deltas = np.stack([scores["proprio"] - scores[name] for name in names], axis=1)
    deltas = deltas.reshape(len(query_rows), -1)
    groups, group_index = np.unique([q["init_state_id"] for q in query_rows], return_inverse=True)
    n_groups = len(groups)
    estimates = deltas.mean(axis=0)
    counts = np.bincount(group_index)
    group_sums = np.zeros((n_groups, deltas.shape[1]))
    np.add.at(group_sums, group_index, deltas)
    # Ten groups is a conservative minimum, not a guarantee of asymptotic accuracy.
    # With one/few groups a row bootstrap would give spurious precision.
    minimum_groups = 10
    intervals = [None] * deltas.shape[1]
    reasons = ["too_few_query_init_states"] * deltas.shape[1]
    tail = 0.05 / (2 * deltas.shape[1])
    # At least ten resamples in each adjusted tail; avoid unresolvable percentiles
    # when many layers/aggregation methods are compared.
    n_resamples = max(config.bootstrap_samples, int(np.ceil(10 / tail)))
    if n_groups >= minimum_groups:
        rng = np.random.default_rng(config.seed + 2)
        boot = np.empty((n_resamples, deltas.shape[1]))
        for start in range(0, n_resamples, 256):
            indices = rng.integers(n_groups, size=(min(256, n_resamples - start), n_groups))
            boot[start:start + len(indices)] = group_sums[indices].sum(axis=1) / counts[indices].sum(axis=1)[:, None]
        bounds = np.quantile(boot, [tail, 1 - tail], axis=0)
        for j in range(deltas.shape[1]):
            # A constant observed nonzero effect has a degenerate bootstrap:
            # do not present its point mass as certainty of superiority.
            group_means = group_sums[:, j] / counts
            if np.ptp(group_means) <= 1e-12 and np.any(deltas[:, j] != 0):
                reasons[j] = "no_between_group_variation"
            else:
                intervals[j] = [float(bounds[0, j]), float(bounds[1, j])]
                reasons[j] = None
    comparisons = {}
    j = 0
    for name in names:
        by_k = {}
        clear = True
        for k in config.k_values:
            by_metric = {}
            for metric in metrics:
                by_metric[metric] = {"improvement": float(estimates[j]),
                                     "ci95_familywise": intervals[j], "unavailable_reason": reasons[j]}
                j += 1
            lower = [value["ci95_familywise"][0] if value["ci95_familywise"] is not None else None
                     for value in by_metric.values()]
            # Conservative G2 rule: a clear gain in at least one component,
            # with no component showing possible degradation, at every tested k.
            clear &= all(x is not None and x >= 0 for x in lower) and any(x is not None and x > 0 for x in lower)
            by_k[str(k)] = by_metric
        comparisons[name] = {"by_k": by_k, "decision": "clear_improvement" if clear else "inconclusive"}
    return {
        "method": "paired_initial_state_cluster_percentile_bootstrap_bonferroni",
        "confidence_level": 0.95, "n_query_init_states": n_groups,
        "minimum_query_init_states": minimum_groups,
        "bootstrap_samples": n_resamples if n_groups >= minimum_groups else 0,
        "n_comparisons": deltas.shape[1], "comparisons": comparisons,
        "decision": "clear_improvement" if any(v["decision"] == "clear_improvement" for v in comparisons.values()) else "inconclusive",
        "scope": "Conditional on this fixed store and split; query windows are row-weighted. "
                 "Store/split variation and additional runs selected after seeing scores are not covered.",
    }


def evaluate_run(run_dir: str | Path, config: EvaluationConfig = EvaluationConfig()) -> dict[str, Any]:
    if not isinstance(config.horizon, int) or config.horizon < 1:
        raise ValueError("horizon must be at least 1")
    if not config.k_values or any(not isinstance(k, int) or k < 1 for k in config.k_values):
        raise ValueError("k_values must contain positive integers")
    if len(set(config.k_values)) != len(config.k_values):
        raise ValueError("k_values must not contain duplicates")
    if not isinstance(config.bootstrap_samples, int) or config.bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be a positive integer")
    manifest, episodes = load_run(run_dir)
    query_ids, store_ids = split_by_init_state(episodes, config.query_fraction, config.seed)
    # Leakage invariant is checked independently at the row level below.
    if query_ids & store_ids:
        raise AssertionError("episode leakage detected")
    if ({e.init_state_id for e in episodes if e.episode_id in query_ids}
            & {e.init_state_id for e in episodes if e.episode_id in store_ids}):
        raise AssertionError("initial-state leakage detected")
    feature_keys = sorted({key for ep in episodes for key in ep.arrays if key.startswith("feat__")})
    if not feature_keys:
        raise ValueError("no feat__* arrays found")
    results: dict[str, Any] = {
        "config": asdict(config),
        "split": {"query_episode_ids": sorted(query_ids), "store_episode_ids": sorted(store_ids),
                  "query_init_state_ids": sorted({e.init_state_id for e in episodes if e.episode_id in query_ids}),
                  "store_init_state_ids": sorted({e.init_state_id for e in episodes if e.episode_id in store_ids})},
        "features": {},
    }
    key_list = feature_keys + ["proprio"]
    scores = {}
    row_ids = None
    for key in key_list:
        if any(key not in ep.arrays for ep in episodes):
            raise ValueError(f"{key} missing from one or more episodes; identical coverage is required")
        if len({_feature_matrix(ep, key).shape[1] for ep in episodes}) != 1:
            raise ValueError(f"{key} dimensions differ between episodes")
        qrows = _stack_rows(episodes, query_ids, key, config.horizon)
        srows = _stack_rows(episodes, store_ids, key, config.horizon)
        if not qrows or not srows:
            raise ValueError(f"feature {key}: query or store side has no valid future rows")
        current_row_ids = ([(r["episode_id"], r["step"]) for r in qrows],
                           [(r["episode_id"], r["step"]) for r in srows])
        if row_ids is not None and current_row_ids != row_ids:
            raise AssertionError("feature and proprio query/store rows differ")
        row_ids = current_row_ids
        rng = np.random.default_rng(config.seed)
        metrics, n, per_query = _evaluate_ranked(qrows, srows, _feature_order(key), config.k_values,
                                                 config.horizon, rng)
        scores[key] = per_query
        results["features"][key] = {"n_queries": n, "metrics": metrics,
                                   "effective_k": {str(k): min(k, len(srows)) for k in config.k_values}}
    results["uncertainty"] = _compare_to_proprio(qrows, scores, config)
    results["data_counts"] = {
        "query_windows": len(qrows), "store_windows": len(srows),
        "query_episodes": len({r["episode_id"] for r in qrows}),
        "store_episodes": len({r["episode_id"] for r in srows}),
    }
    # Exact expectation of uniform random retrieval on the same rows.
    qrows = _stack_rows(episodes, query_ids, "proprio", config.horizon)
    srows = _stack_rows(episodes, store_ids, "proprio", config.horizon)
    random_metrics, random_n, _ = _evaluate_ranked(qrows, srows, _random_order, config.k_values,
                                                   config.horizon)
    results["random"] = {"n_queries": random_n, "metrics": random_metrics,
                         "method": "exact_uniform_expectation"}
    results["oracle_upper_bound"] = _evaluate_oracle(qrows, srows, config.k_values, config.horizon)
    results["manifest_summary"] = {k: manifest.get(k) for k in ("suite", "checkpoint", "feature_metadata", "hidden_features") if k in manifest}
    results["environment_time_offsets"] = {e.episode_id: e.env_step_offset for e in episodes}
    return results


def write_report(results: dict[str, Any], path: str | Path) -> None:
    uncertainty = results["uncertainty"]
    counts = results["data_counts"]
    lines = ["# G2 retrieval evaluation", "", "Lower distance is better.", "",
             f"H={results['config']['horizon']}, seed={results['config']['seed']}", "",
             f"G2 evidence: **{uncertainty['decision']}**. "
             f"Query initial-state groups: {uncertainty['n_query_init_states']} "
             f"(minimum for intervals: {uncertainty['minimum_query_init_states']}).", "",
             f"Queries: {counts['query_windows']} windows / {counts['query_episodes']} episodes; "
             f"store: {counts['store_windows']} windows / {counts['store_episodes']} episodes.", "",
             "| Retrieval | k | Relative position (m) | Relative rotation (rad) | Gripper |",
             "|---|---:|---:|---:|---:|"]
    for name, value in results["features"].items():
        for k, metrics in value["metrics"].items():
            lines.append(f"| {name} | {k} | {metrics['eef_relative_position']:.6g} | "
                         f"{metrics['eef_relative_rotation_rad']:.6g} | {metrics['gripper_relative']:.6g} |")
    for name in ("random",):
        for k, metrics in results[name]["metrics"].items():
            lines.append(f"| {name} | {k} | {metrics['eef_relative_position']:.6g} | "
                         f"{metrics['eef_relative_rotation_rad']:.6g} | {metrics['gripper_relative']:.6g} |")
    for k in results["oracle_upper_bound"]["eef_relative_position"]:
        oracle = results["oracle_upper_bound"]
        lines.append(f"| oracle (position) | {k} | {oracle['eef_relative_position'][k]:.6g} | — | — |")
        lines.append(f"| oracle (rotation) | {k} | — | {oracle['eef_relative_rotation_rad'][k]:.6g} | — |")
        lines.append(f"| oracle (gripper) | {k} | — | — | {oracle['gripper_relative'][k]:.6g} |")
    lines += ["", "Oracle is selected independently for each metric and is an optimistic ceiling.",
              "Random scores are exact expectations, and kNN boundary ties are averaged uniformly.",
              "Split IDs, effective k values, feature metadata and environment time offsets are recorded in `results.json`.",
              "", "## Paired improvement over proprio", "",
              "Positive improvement means smaller error than proprio. Intervals are 95% familywise "
              "Bonferroni-adjusted percentile intervals over all listed features, k values and components. "
              "Whole initial-state groups are resampled; overlapping windows are not independent samples.", "",
              "| Feature | k | Component | Improvement | Interval |",
              "|---|---:|---|---:|---|"]
    for name, comparison in uncertainty["comparisons"].items():
        for k, by_metric in comparison["by_k"].items():
            for metric, value in by_metric.items():
                ci = value["ci95_familywise"]
                label = f"[{ci[0]:.6g}, {ci[1]:.6g}]" if ci is not None else value["unavailable_reason"]
                lines.append(f"| {name} | {k} | {metric} | {value['improvement']:.6g} | {label} |")
    lines += ["", "Clear improvement requires a strictly positive lower bound in at least one component "
              "and nonnegative lower bounds in all components, at every tested k for the same feature. "
              "An inconclusive result does not establish that internal features have no value.",
              "Fewer than ten query groups or a degenerate nonzero group effect cannot establish superiority. "
              "Ten groups is only a minimum; small-sample bootstrap coverage remains approximate.",
              uncertainty["scope"],
              "Feature provenance (action-expert suffix layer, diffusion time and pooling), the fixed checkpoint, "
              "and representative task coverage must also be checked before making the G2 deployment decision.", ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--query-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=10000,
                        help="minimum cluster resamples; increased for multiple-comparison tail resolution")
    parser.add_argument("--output-dir", type=Path,
                        help="new, non-existing output directory; defaults under .skillnet-state/outputs/memory-pilot")
    args = parser.parse_args()
    result = evaluate_run(args.run_dir, EvaluationConfig(args.horizon, args.query_fraction, args.seed,
                                                         bootstrap_samples=args.bootstrap_samples))
    if args.output_dir:
        out = args.output_dir
    else:
        import os
        from datetime import datetime, timezone

        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{os.getpid()}"
        out = Path("/work/.skillnet-state/outputs/memory-pilot") / f"g2-analysis-{run_id}"
    out.mkdir(parents=True, exist_ok=False)
    (out / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_report(result, out / "report.md")


if __name__ == "__main__":
    main()
