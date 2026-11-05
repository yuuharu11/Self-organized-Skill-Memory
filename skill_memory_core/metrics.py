"""Small NumPy evaluation utilities; bootstrap units are whole episodes."""
from __future__ import annotations

import math
import numpy as np


def active_mse(left, right, *, active_dim: int = 7) -> np.ndarray:
    """Per-row MSE over action steps and the first `active_dim` channels."""
    a, b = np.asarray(left, np.float32), np.asarray(right, np.float32)
    if a.shape != b.shape or a.ndim < 2 or not 0 < active_dim <= a.shape[-1]:
        raise ValueError("action arrays must have identical [...,H,D] shapes")
    return np.mean((a[..., :active_dim] - b[..., :active_dim]) ** 2, axis=(-1, -2))


def nearest_oracle(query_actions, memory_actions, *, active_dim: int = 7, k: int = 1):
    """Brute-force nearest memory rows, returning local indices and distances."""
    q = np.asarray(query_actions, np.float32)[..., :active_dim].reshape(len(query_actions), -1)
    m = np.asarray(memory_actions, np.float32)[..., :active_dim].reshape(len(memory_actions), -1)
    if not len(m) or k <= 0:
        raise ValueError("need memory rows and positive k")
    out_i = np.empty((len(q), min(k, len(m))), np.int64)
    out_d = np.empty(out_i.shape, np.float32)
    for st in range(0, len(q), 128):
        d = np.mean((q[st:st + 128, None] - m[None]) ** 2, axis=-1)
        ix = np.argsort(d, axis=1, kind="stable")[:, :out_i.shape[1]]
        out_i[st:st + len(ix)] = ix
        out_d[st:st + len(ix)] = np.take_along_axis(d, ix, axis=1)
    return out_i, out_d


def episode_bootstrap(values, episode_ids, *, reps: int = 5000, seed: int = 4101,
                      confidence: float = .95):
    """Bootstrap row means by resampling unique episode clusters."""
    x, ep = np.asarray(values, np.float64).reshape(-1), np.asarray(episode_ids)
    if len(x) != len(ep) or not len(x):
        raise ValueError("values and episode IDs must align and be nonempty")
    unique = np.unique(ep)
    groups = {u: np.flatnonzero(ep == u) for u in unique}
    rng = np.random.default_rng(seed)
    means = np.empty(reps, np.float64)
    for i in range(reps):
        draw = rng.choice(unique, len(unique), replace=True)
        ix = np.concatenate([groups[u] for u in draw])
        means[i] = x[ix].mean()
    alpha = 1. - confidence
    return {"mean": float(x.mean()), "ci": [float(np.quantile(means, alpha / 2)),
            float(np.quantile(means, 1 - alpha / 2))], "episodes": int(len(unique)),
            "rows": int(len(x)), "reps": int(reps)}


def paired_episode_bootstrap(left, right, episode_ids, *, reps: int = 5000,
                             seed: int = 4101, confidence: float = .95):
    a, b = np.asarray(left, np.float64), np.asarray(right, np.float64)
    if a.shape != b.shape:
        raise ValueError("paired arrays must have the same shape")
    return episode_bootstrap(a - b, episode_ids, reps=reps, seed=seed, confidence=confidence)


def wilson_interval(successes: int, trials: int, *, z: float = 1.959963984540054):
    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("invalid binomial counts")
    p = successes / trials
    den = 1 + z * z / trials
    mid = (p + z * z / (2 * trials)) / den
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / den
    return p, (mid - half, mid + half)


def exact_mcnemar(b: int, c: int):
    """Two-sided exact binomial McNemar p-value for discordant pairs."""
    n = int(b) + int(c)
    if min(b, c) < 0:
        raise ValueError("discordant counts must be nonnegative")
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / (2 ** n)
    return min(1., 2 * tail)


def output_sensitivity(output_a, output_b, draft_a, draft_b, *, eps: float = 1e-8):
    ya, yb = np.asarray(output_a, np.float32), np.asarray(output_b, np.float32)
    da, db = np.asarray(draft_a, np.float32), np.asarray(draft_b, np.float32)
    if ya.shape != yb.shape or da.shape != db.shape or len(ya) != len(da):
        raise ValueError("paired output/draft tensors must align")
    axes = tuple(range(1, ya.ndim))
    yd = np.sqrt(np.mean((ya - yb) ** 2, axis=axes))
    dd = np.sqrt(np.mean((da - db) ** 2, axis=tuple(range(1, da.ndim))))
    ratio = yd / np.maximum(dd, eps)
    return {"row_output_distance": yd, "row_draft_distance": dd, "row_ratio": ratio,
            "mean_output_distance": float(yd.mean()), "mean_draft_distance": float(dd.mean()),
            "mean_ratio": float(ratio.mean())}
