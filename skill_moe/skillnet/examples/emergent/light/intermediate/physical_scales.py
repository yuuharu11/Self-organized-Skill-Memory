"""Fit action-to-motion scales on training episodes for offline evaluation.

This module needs only NumPy at import time; SciPy is loaded when fitting.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def fit_physical_scales(cache_dir, episode_ids, *, horizon=25, max_episodes=400,
                        seed=0, stride=5):
    """Least-squares map from summed action commands to measured end-effector motion.

    Position: Delta p (m, world) over H steps ~ k_pos * sum_t a[:3].
    Rotation: rotvec(R_{t+H} R_t^-1) (rad, world) ~ k_rot * sum_t a[3:6].
    Fitted per axis on training-suite episodes' env_state; recorded in results.
    """
    from scipy.spatial.transform import Rotation
    manifest = json.loads((Path(cache_dir) / "manifest.json").read_text())
    wanted = set(map(str, episode_ids))
    entries = [e for e in manifest["episodes"] if str(e["episode_id"]) in wanted]
    rng = np.random.default_rng(seed)
    if len(entries) > max_episodes:
        entries = [entries[i] for i in np.sort(rng.choice(len(entries), max_episodes, replace=False))]
    xp, yp, xr, yr = [], [], [], []
    for entry in entries:
        with np.load(entry["cache"], allow_pickle=False) as cache:
            action = np.asarray(cache["action"], np.float32)
            state = np.asarray(cache["env_state"], np.float32)
        for t in range(0, min(len(action), len(state)) - horizon - 1, stride):
            xp.append(action[t:t + horizon, :3].sum(0))
            yp.append(state[t + horizon, :3] - state[t, :3])
            xr.append(action[t:t + horizon, 3:6].sum(0))
            rel = Rotation.from_rotvec(state[t + horizon, 3:6]) * Rotation.from_rotvec(state[t, 3:6]).inv()
            yr.append(rel.as_rotvec())
    xp, yp, xr, yr = map(lambda v: np.asarray(v, np.float64), (xp, yp, xr, yr))
    k_pos = (xp * yp).sum(0) / np.maximum((xp * xp).sum(0), 1e-12)
    k_rot = (xr * yr).sum(0) / np.maximum((xr * xr).sum(0), 1e-12)
    corr = lambda x, y: [float(np.corrcoef(x[:, d], y[:, d])[0, 1]) for d in range(3)]
    return {"pos_m_per_action_unit_step": k_pos.astype(np.float32).tolist(),
            "rot_rad_per_action_unit_step": k_rot.astype(np.float32).tolist(),
            "pos_fit_correlation": corr(xp, yp), "rot_fit_correlation": corr(xr, yr),
            "windows": int(len(xp)), "episodes": int(len(entries))}
