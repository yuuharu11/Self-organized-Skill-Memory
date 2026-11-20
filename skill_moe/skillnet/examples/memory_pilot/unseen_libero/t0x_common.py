"""Shared helpers for T0X: confidence plus a nonzero adaptation radius."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[4]
STATE_DIR = Path(os.environ.get("SKILLNET_STATE_DIR", REPO_ROOT / ".skillnet-state")).expanduser()
ROOT = STATE_DIR / "outputs/memory-pilot/unseen-libero"
OUT = ROOT / "t0x-01"
JST = timezone(timedelta(hours=9))


def timestamp() -> str:
    return datetime.now(JST).isoformat(timespec="seconds")


def progress(stage: str, message: str, **fields) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    row = {"jst": timestamp(), "stage": stage, "message": message, **fields}
    with (OUT / "progress.log").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=float) + "\n")
    print(json.dumps(row, ensure_ascii=False, default=float), flush=True)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=float) + "\n", encoding="utf-8")
    tmp.replace(path)


def t0_add(ehat, b: float):
    """User-specified map: sqrt(.15^2 + c*ê), reaching .35 at ê=b."""
    if not np.isfinite(b) or b <= 0:
        raise ValueError(f"fallback threshold b must be finite and positive, got {b}")
    c = (.35**2 - .15**2) / b
    e = np.maximum(np.asarray(ehat, dtype=np.float64), 0.)
    return np.clip(np.sqrt(.15**2 + c * e), .15, .35)


def calibrate_m_radius(ehat, m, b: float) -> dict:
    """Dev-only threshold calibration for the T4 Expert-displacement radius."""
    e = np.asarray(ehat, dtype=np.float64).reshape(-1)
    displacement = np.asarray(m, dtype=np.float64).reshape(-1)
    if e.shape != displacement.shape or not len(e):
        raise ValueError(f"Dev ehat/m shape mismatch or empty: {e.shape} vs {displacement.shape}")
    if not np.isfinite(e).all() or not np.isfinite(displacement).all() or np.any(displacement < 0):
        raise ValueError("Dev ehat and m must be finite; displacement must be nonnegative")
    if not np.isfinite(b) or b <= 0:
        raise ValueError(f"fallback threshold b must be finite and positive, got {b}")
    p_b = float(np.mean(e > b))
    quantile = 1.0 - p_b
    m_b = float(np.quantile(displacement, quantile))
    if not np.isfinite(m_b) or m_b <= 0:
        raise ValueError(f"invalid Dev m_b quantile {m_b}; cannot calibrate radius")
    return {"b": float(b), "p_b": p_b, "m_b": m_b, "m_quantile": quantile,
            "dev_rows": int(len(e)), "probe_t0": 0.30,
            "definition": "p_b=mean(dev_ehat>b); m_b=quantile(dev_m,1-p_b); m_prime=b*(m/m_b)^2"}


def t0_from_m(m, calibration: dict):
    """Map a single-seed Expert displacement to [.15,.35] using the Dev m_b anchor."""
    b = float(calibration["b"])
    m_b = float(calibration["m_b"])
    if not np.isfinite(m_b) or m_b <= 0:
        raise ValueError(f"m_b must be finite and positive, got {m_b}")
    m_arr = np.maximum(np.asarray(m, dtype=np.float64), 0.0)
    m_prime = b * np.square(m_arr / m_b)
    c_m = (.35**2 - .15**2) / b
    return np.clip(np.sqrt(.15**2 + c_m * m_prime), .15, .35)


def t0_from_disagreement(disagreement, calibration: dict):
    """Map top-k motion disagreement to [.15,.35] using only the Dev D_b anchor."""
    b, d_b = float(calibration["b"]), float(calibration["D_b"])
    if not np.isfinite(b) or b <= 0 or not np.isfinite(d_b) or d_b <= 0:
        raise ValueError("invalid disagreement calibration")
    d = np.maximum(np.asarray(disagreement, dtype=np.float64), 0.)
    d_prime = b * d / d_b
    c = (.35**2 - .15**2) / b
    return np.clip(np.sqrt(.15**2 + c * d_prime), .15, .35)


def topk_disagreement(candidate_actions, scores, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Softmax-weighted arm6 MSE to the top-1 stored action, within selected top-k."""
    a = np.asarray(candidate_actions, dtype=np.float64)
    s = np.asarray(scores, dtype=np.float64)
    if a.ndim != 4 or a.shape[-1] < 6 or s.ndim != 2 or a.shape[:2] != s.shape:
        raise ValueError(f"candidate action/score shape mismatch: {a.shape} vs {s.shape}")
    if not len(a) or not a.shape[1] or not np.isfinite(a[..., :6]).all() or not np.isfinite(s).all():
        raise ValueError("candidate actions and scores must be finite and nonempty")
    k = min(int(k), s.shape[1])
    if k < 1:
        raise ValueError("k must be positive")
    a, s = a[:, :k, :, :6], s[:, :k]
    p = np.exp(s - s.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    err = np.square(a - a[:, :1]).mean(axis=(2, 3))
    return np.sum(p * err, axis=1), p


def calibrate_disagreement(disagreement, ehat, b: float) -> dict:
    """Choose Dev D_b for the same strict upper-tail fallback mass as ehat>b."""
    d = np.asarray(disagreement, dtype=np.float64).reshape(-1)
    e = np.asarray(ehat, dtype=np.float64).reshape(-1)
    if d.shape != e.shape or not len(d) or not np.isfinite(d).all() or not np.isfinite(e).all():
        raise ValueError(f"invalid Dev disagreement/ehat arrays: {d.shape}, {e.shape}")
    if not np.isfinite(b) or b <= 0 or np.any(d < 0):
        raise ValueError("b must be positive and disagreement must be nonnegative")
    p_b = float(np.mean(e > b))
    q = 1. - p_b
    d_b = float(np.quantile(d, q, method="inverted_cdf"))
    if not np.isfinite(d_b) or d_b <= 0:
        raise ValueError(f"invalid Dev disagreement threshold D_b={d_b}")
    return {"b": float(b), "p_b": p_b, "D_b": d_b, "quantile": q, "dev_rows": int(len(d)),
            "formula": "D'=b*D/D_b; t0=clip(sqrt(.15^2+((.35^2-.15^2)/b)*D'),.15,.35); fallback iff D>D_b"}


def quantile_map(value, reference_x, reference_y):
    """Map an online scalar through the Dev empirical CDF of x into Dev values of y."""
    x = np.sort(np.asarray(reference_x, np.float64).reshape(-1))
    y = np.asarray(reference_y, np.float64).reshape(-1)
    if not len(x) or not len(y) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('quantile map calibration must be finite and nonempty')
    q = np.interp(np.asarray(value, np.float64), x, np.linspace(0., 1., len(x)))
    return np.quantile(y, np.clip(q, 0., 1.))


def mcnemar_exact(left, right):
    """Exact two-sided McNemar test using the binomial distribution."""
    import math
    a, b = np.asarray(left, bool), np.asarray(right, bool)
    if a.shape != b.shape:
        raise ValueError(f"paired arrays differ: {a.shape} vs {b.shape}")
    b_only = int((b & ~a).sum()); a_only = int((a & ~b).sum()); n = a_only + b_only
    p = 1. if n == 0 else min(1., 2. * sum(math.comb(n, i) for i in range(min(a_only, b_only) + 1)) / 2**n)
    return {"b_only": b_only, "a_only": a_only, "p_exact": p}

