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

