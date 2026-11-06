import importlib.util
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parents[1] / "skill_moe/skillnet/examples/memory_pilot/pi05_memory/output_by_draft/run.py"
spec = importlib.util.spec_from_file_location("output_by_draft_run", CODE)
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)


def test_direction_cos_uses_summed_xyz():
    a = np.zeros((1, 10, 7)); a[0, :, 0] = 1.
    assert run.direction_cos(a, a) == 1.
    assert run.direction_cos(a, -a) == -1.
    b = np.zeros((1, 10, 7)); b[0, :, 1] = 1.
    assert run.direction_cos(a, b) == 0.


def test_row_mse_ignores_inactive_dimensions():
    a = np.zeros((1, 10, 32)); b = a.copy(); b[:, :, 7:] = 5.
    assert run.row_mse(a, b)[0] == 0.


def test_bins_split_points_and_skip_sparse_ranges():
    draft = np.r_[np.full(25, .001), np.full(30, .03), np.full(3, 1.)]
    gen = {t: draft * 2 for t in run.T0S}
    bins = run.bin_by_draft(draft, np.ones_like(draft), gen)
    assert [b["points"] for b in bins] == [25, 30]
    assert bins[0]["t0=0.2"] == .002
    assert bins[0]["below_normal_t0=0.2"] == 1.


def test_bootstrap_is_paired_over_episodes():
    r = run.bootstrap(np.r_[np.full(10, 1.), np.full(10, 1.)], np.r_[np.zeros(10), np.ones(10)], reps=50)
    assert r["mean"] == 1. and r["ci95"] == [1., 1.]
