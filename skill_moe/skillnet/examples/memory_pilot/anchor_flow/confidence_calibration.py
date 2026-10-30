"""LIBERO-90 validation-only confidence calibration and inference schedule."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def denoising_steps(t0: float) -> int:
    if not np.isfinite(t0) or not 0 <= t0 <= 1:
        raise ValueError("t0 must be finite and in [0,1]")
    return max(2, round(10 * float(t0)))


def fit_monotone(confidence, errors, grid=(.2, .4, .6, .8, 1.), bins=4):
    """Exact minimum-cost nonincreasing t0 across confidence quantile bins.

    Same dynamic program as fair_compare.fit_monotone_g. Error is per query,
    already averaged over paired noise/component seeds and the seven active
    action dimensions (padding must not dilute the loss).
    """
    confidence = np.asarray(confidence, np.float64)
    errors = np.asarray(errors, np.float64)
    grid = np.asarray(grid, np.float64)
    if confidence.ndim != 1 or not len(confidence) or errors.shape != (len(grid), len(confidence)):
        raise ValueError("expected confidence [N] and errors [len(grid),N]")
    if not np.isfinite(confidence).all() or not np.isfinite(errors).all():
        raise ValueError("calibration inputs must be finite")
    if not np.all((grid >= .2) & (grid <= 1)):
        raise ValueError("calibration t0 grid must lie in [.2,1]")
    edges = np.unique(np.quantile(confidence, np.linspace(0, 1, bins + 1)[1:-1]))
    which = np.searchsorted(edges, confidence, side="right")
    order = np.argsort(-grid)
    cost = np.asarray([errors[:, which == b].sum(1) for b in range(len(edges) + 1)])[:, order]
    dp = cost.copy()
    prev = np.zeros(cost.shape, int)
    for b in range(1, len(cost)):
        for j in range(len(grid)):
            prev[b, j] = np.argmin(dp[b - 1, :j + 1])
            dp[b, j] += dp[b - 1, prev[b, j]]
    j = int(np.argmin(dp[-1])); chosen = []
    for b in range(len(cost) - 1, -1, -1):
        chosen.append(float(grid[order[j]])); j = int(prev[b, j])
    return {"edges": edges.tolist(), "t0": chosen[::-1],
            "counts": np.bincount(which, minlength=len(cost)).tolist(),
            "per_bin_unconstrained_best": [float(grid[np.argmin(errors[:, which == b].sum(1))])
                                           for b in range(len(cost))],
            "validation_mse": float(dp[-1].min() / len(confidence))}


def confidence_value(metric, similarity, entropy):
    if metric == "max_cosine_similarity":
        return float(similarity)
    if metric == "one_minus_normalized_entropy":
        return 1. - float(entropy)
    raise ValueError(f"unknown calibration confidence: {metric}")


def calibrated_t0(table, similarity, entropy):
    value = confidence_value(table["metric"], similarity, entropy)
    index = int(np.searchsorted(table["edges"], value, side="right"))
    return float(table["t0"][index]), index, value


def load_calibration(path):
    table = json.loads(Path(path).read_text())
    if table.get("suite") != "libero90" or table.get("query_text_source") != "instruction_only":
        raise ValueError("calibration must use LIBERO-90 instruction_only validation data")
    edges = np.asarray(table["edges"], float); times = np.asarray(table["t0"], float)
    if (not np.isfinite(edges).all() or not np.isfinite(times).all()
            or len(times) != len(edges) + 1 or np.any(np.diff(edges) <= 0)
            or np.any(np.diff(times) > 0) or np.any(times < .2) or np.any(times > 1)
            or not np.isfinite(table["temperature"]) or table["temperature"] <= 0):
        raise ValueError("invalid calibrated monotone schedule/temperature")
    confidence_value(table["metric"], 0., 0.)
    return table
