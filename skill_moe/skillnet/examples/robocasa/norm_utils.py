"""Small pure helpers for magnitude fitting in RoboCasa memory drafts."""

from __future__ import annotations

import numpy as np


def norm_ratio_scale(draft, reference, length, *, lower=0.25, upper=4.0, eps=1e-8):
    """Match magnitudes over a selected unit's valid interval, independent of angle.

    Inputs are time-major physical action components. Only the first ``length``
    rows participate; padded/repeated tail actions cannot alter the fit.
    """
    x = np.asarray(draft, np.float64)
    y = np.asarray(reference, np.float64)
    n = min(max(int(length), 0), len(x), len(y))
    if n == 0:
        raise ValueError("norm-ratio scale needs a non-empty selected-unit interval")
    ratio = float(np.linalg.norm(y[:n]) / max(float(np.linalg.norm(x[:n])), eps))
    return float(np.clip(ratio, lower, upper))
