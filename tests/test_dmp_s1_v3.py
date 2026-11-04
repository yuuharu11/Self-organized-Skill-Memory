import importlib.util
from pathlib import Path

import numpy as np


MODULE = Path("/work/.skillnet-state/outputs/memory-pilot/dmp-route-20261102/s1-v3/dmp.py")
SPEC = importlib.util.spec_from_file_location("dmp_s1_v3", MODULE)
dmp = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dmp)


def sample_curve(n=41):
    t = np.linspace(0.0, 1.0, n)
    y = np.stack([0.1*t, 0.03*np.sin(np.pi*t), 0.02*t*t,
                  0.2*t, 0.05*np.sin(2*np.pi*t), -0.1*t], axis=-1)
    return y.astype(np.float32)


def test_rollout_prefix_does_not_depend_on_requested_length():
    y = sample_curve()
    w, g, tau = dmp.fit(y)
    a = dmp.rollout(w, y[0], g, tau, 41)
    b = dmp.rollout(w, y[0], g, tau, 80)
    assert np.array_equal(a[:20], b[:20])


def test_rollout_reaches_goal_at_tau():
    y = sample_curve()
    w, g, tau = dmp.fit(y)
    out = dmp.rollout(w, y[0], g, tau, len(y))
    assert np.max(np.abs(out[-1, :3] - (y[0, :3] + g[:3]))) <= 1e-3
    assert np.max(np.abs(out[-1, 3:] - (y[0, 3:] + g[3:]))) <= 1e-2


def test_midphase_rollout_reaches_goal():
    y = sample_curve()
    w, g, tau = dmp.fit(y)
    out = dmp.rollout(w, y[20], g, tau, 30, phase_s=0.5)
    assert np.max(np.abs(out[-1, :3] - (y[20, :3] + g[:3]))) <= 1e-3
    assert np.max(np.abs(out[-1, 3:] - (y[20, 3:] + g[3:]))) <= 1e-2


def test_fit_rollout_reconstructs_curve():
    y = sample_curve()
    w, g, tau = dmp.fit(y)
    out = dmp.rollout(w, y[0], g, tau, len(y))
    assert np.mean((out - y) ** 2) < 1e-5

