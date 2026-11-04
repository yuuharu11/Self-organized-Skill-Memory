"""NumPy-only data loading and metrics shared by S2 offline experiments.

No entry-point imports, model loading, or experiment execution.
"""
from __future__ import annotations

import numpy as np


ROW_FIELDS = (
    "state", "actions", "episode_id", "init_state_id", "image_base", "image_left_wrist",
    "tokenized_prompt", "tokenized_prompt_mask", "source_file", "frame",
)


ENDPOINT_SCALES_M = np.asarray([0.0111, 0.0122, 0.0110], np.float32)


def read_npz(path, fields):
    """Load only requested fields and report missing cache columns together."""
    with np.load(path, allow_pickle=True) as archive:
        missing = set(fields) - set(archive.files)
        if missing:
            raise ValueError(f"{path} is missing fields: {sorted(missing)}")
        return {name: archive[name] for name in fields}


def normalize_keys(x):
    """L2-normalize the final axis in float32, leaving zero vectors finite."""
    x = np.asarray(x, np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def retrieval_metrics(query_keys, memory_keys, entries, query_episode, query_init, oracle_units, majority_unit,
                      *, temperature):
    memory_keys = normalize_keys(memory_keys)
    query_keys = normalize_keys(query_keys)
    entry_episode = np.asarray([str(row["meta"]["episode_id"]) for row in entries])
    entry_init = np.asarray([str(row["meta"]["init_state_id"]) for row in entries])
    entry_units = [row.get("unit_id") for row in entries]
    top_indices, top_sims, top_weights = [], [], []
    top1_correct, top5_correct = [], []
    for i, query in enumerate(query_keys):
        similarities = memory_keys @ query
        eligible = (entry_episode != str(query_episode[i])) & (entry_init != str(query_init[i]))
        similarities[~eligible] = -np.inf
        order = np.argsort(-similarities, kind="stable")[:min(5, int(eligible.sum()))]
        sims = similarities[order]
        distances = np.sqrt(np.maximum(2.0 - 2.0 * sims, 0.0))
        logits = -distances / max(float(temperature), 1e-8)
        weights = np.exp(logits - logits.max())
        weights /= weights.sum()
        target = oracle_units[i]
        units = [None if entry_units[j] is None else int(entry_units[j]) for j in order]
        top_indices.append(order)
        top_sims.append(sims)
        top_weights.append(weights.astype(np.float32))
        top1_correct.append(target is not None and units[0] == int(target))
        top5_correct.append(target is not None and int(target) in units)
    return {
        "top_indices": top_indices, "top_sims": top_sims, "top_weights": top_weights,
        "top1_correct": np.asarray(top1_correct, bool), "top5_correct": np.asarray(top5_correct, bool),
        "majority_correct": np.asarray([u is not None and int(u) == majority_unit for u in oracle_units], bool),
    }


def quantile_summary(values):
    x = np.asarray(values, np.float64)
    return {f"q{p}": float(np.quantile(x, p / 100)) for p in (10, 25, 50, 75, 90)}


def denoising_steps(t0):
    """Historical offline Euler schedule; calibrated serving validates t0 separately."""
    return max(2, round(10 * float(t0)))


def row_errors(pred, target):
    """Per-row normalized MSE (7 active dims) and endpoint mm (key_check_extension)."""
    err = np.asarray(pred, np.float32)[..., :7] - np.asarray(target, np.float32)[..., :7]
    mse = np.mean(np.square(err), axis=(-2, -1))
    mm = np.linalg.norm(np.sum(err[..., :3] * ENDPOINT_SCALES_M, axis=-2), axis=-1) * 1000.0
    return mse, mm


def cluster_bootstrap(values, clusters, *, reps=4000, seed=0):
    """Mean and 95% percentile CI, resampling episodes with replacement."""
    v = np.asarray(values, np.float64)
    c = np.asarray(clusters).astype(str)
    uniq, inv = np.unique(c, return_inverse=True)
    sums = np.bincount(inv, weights=v, minlength=len(uniq))
    cnts = np.bincount(inv, minlength=len(uniq)).astype(np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(uniq), size=(reps, len(uniq)))
    boot = sums[draws].sum(1) / cnts[draws].sum(1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return float(v.mean()), float(lo), float(hi)
