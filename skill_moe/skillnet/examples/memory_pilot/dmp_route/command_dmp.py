"""Action-command-space S1-v3 DMP adapter used by S3."""
from __future__ import annotations

import numpy as np


def command_draft(dmp_module, library, route, y0, *, q01, q99, horizon=10, dt=0.05,
                  shape_mode="learned"):
    """Expand one hard route to normalized LIBERO commands [B,H,7].

    S1-v3 shapes and g are in raw OSC command units. The six continuous
    dimensions are expanded directly in command space; no pose-to-command
    ridge conversion is used. The gripper sign is a discrete per-primitive
    attribute and is repeated across the chunk.
    """
    ids = np.asarray(route.prim_id).reshape(-1).astype(np.int64)
    goals = np.asarray(route.goal, np.float32).reshape((-1, 6))
    taus = np.asarray(route.tau, np.float32).reshape(-1)
    phases = np.asarray(route.phase_s, np.float32).reshape(-1)
    y0 = np.asarray(y0, np.float32)
    coefficients = np.asarray(library["coef"], np.float32)
    action_scale = np.asarray(library["action_scale"], np.float32)
    grip = np.asarray(library["grip"], np.float32).reshape(-1)
    q01 = np.asarray(q01, np.float32).reshape(-1)[:7]
    q99 = np.asarray(q99, np.float32).reshape(-1)[:7]
    if coefficients.ndim != 3 or coefficients.shape[1] != 6 or action_scale.shape != (6,):
        raise ValueError("S1-v3c library expects coef [K,6,N] and action_scale [6]")
    if y0.shape != (len(ids), 6):
        raise ValueError("y0 must be the previous raw 6-D command for every query")
    if len(grip) != len(coefficients) or np.any(ids < 0) or np.any(ids >= len(coefficients)):
        raise ValueError("invalid primitive IDs or grip dictionary")
    if q01.shape != (7,) or q99.shape != (7,) or np.any(q99 <= q01):
        raise ValueError("seven valid action quantile bounds are required")
    if shape_mode not in ("learned", "straight"):
        raise ValueError("shape_mode must be learned or straight")
    raw = np.empty((len(ids), horizon, 7), np.float32)
    for i, pid in enumerate(ids):
        weights = coefficients[pid]
        if shape_mode == "straight":
            weights = np.zeros_like(weights)
        commands = dmp_module.rollout(
            weights, y0[i], goals[i], taus[i], horizon,
            library={"action_scale": action_scale}, phase_s=float(phases[i]), dt=dt,
        )
        raw[i, :, :6] = commands
        raw[i, :, 6] = grip[pid]
    normalized = np.empty_like(raw)
    normalized[..., :6] = ((raw[..., :6] - q01[None, None, :6]) /
                           (q99[None, None, :6] - q01[None, None, :6] + 1e-6) * 2.0 - 1.0)
    normalized[..., 6] = (raw[..., 6] - q01[None, None, 6]) / (q99[None, None, 6] - q01[None, None, 6] + 1e-6) * 2.0 - 1.0
    return normalized.astype(np.float32)
