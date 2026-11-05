"""Pure flow-bridge equations compatible with NumPy, JAX arrays and Torch."""
from __future__ import annotations

import numpy as np


def bridge_batch(actions, drafts, noise, t0, progress):
    """Return bridge state, target velocity, and model time.

    `source=(1-t0)*draft+t0*noise`, `x=(1-u)*A+u*source`,
    `v_target=(source-A)/t0`, `t=t0*u`. Inputs may be array-library tensors.
    """
    starts = t0.reshape((-1, 1, 1)) if hasattr(t0, "reshape") else np.asarray(t0).reshape(-1, 1, 1)
    u = progress.reshape((-1, 1, 1)) if hasattr(progress, "reshape") else np.asarray(progress).reshape(-1, 1, 1)
    # Validate eager NumPy/list inputs. Avoid materializing device tensors or
    # JAX tracers, for which np.asarray/bool would break tracing/device use.
    if isinstance(t0, (np.ndarray, list, tuple, float, int)):
        if np.any(np.asarray(starts) <= 0) or np.any(np.asarray(starts) > 1):
            raise ValueError("t0 must lie in (0,1]")
    if isinstance(progress, (np.ndarray, list, tuple, float, int)):
        if np.any(np.asarray(u) < 0) or np.any(np.asarray(u) > 1):
            raise ValueError("progress must lie in [0,1]")
    source = (1 - starts) * drafts + starts * noise
    x = (1 - u) * actions + u * source
    target = (source - actions) / starts
    return x, target, (starts * u).reshape(-1)


def bridge_start(drafts, noise, t0):
    """Initial state for inference at `t0`."""
    t = t0.reshape((-1, 1, 1)) if hasattr(t0, "reshape") else np.asarray(t0).reshape(-1, 1, 1)
    return (1 - t) * drafts + t * noise


def integrate_euler(velocity, x, t0, *, steps: int):
    """Integrate `dx/dt=v` backward from t0 to zero with fixed Euler steps.

    `velocity(x,t)` receives batched model time. The bridge uses `dt=t0/steps`.
    """
    if steps <= 0:
        raise ValueError("steps must be positive")
    start = t0 if hasattr(t0, "reshape") else np.asarray(t0)
    for i in range(steps):
        t = start * (1.0 - i / steps)
        x = x - (start / steps).reshape((-1, 1, 1)) * velocity(x, t)
    return x
