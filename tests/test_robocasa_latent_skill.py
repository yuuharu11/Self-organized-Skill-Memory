from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
MOD = ROOT / "skill_moe/skillnet/examples/memory_pilot/robocasa"
sys.path.insert(0, str(MOD))
import latent_skill as ls  # noqa: E402


def test_dct3_keeps_natural_length_and_is_exact_for_short_units():
    rng = np.random.default_rng(0)
    arm = rng.normal(size=(17, 6))
    coeff, dec = ls.dct3_natural(arm)
    assert coeff.shape == (3, 6) and dec.shape == (17, 6)
    short = rng.normal(size=(3, 6))
    _, dec_short = ls.dct3_natural(short)
    np.testing.assert_allclose(dec_short, short, atol=1e-5)
    # A unit that is a pure DCT-3 signal is reconstructed exactly.
    _, again = ls.dct3_natural(dec)
    np.testing.assert_allclose(again, dec, atol=1e-5)


def test_window_from_unit_holds_last_step():
    unit = np.arange(6 * 7, dtype=np.float32).reshape(6, 7)
    w = ls.window_from_unit(np.r_[unit, np.full((4, 7), 99.)], 6)
    np.testing.assert_array_equal(w[:6], unit)
    np.testing.assert_array_equal(w[6:], np.repeat(unit[-1:], 4, 0))
    with pytest.raises(ValueError):
        ls.window_from_unit(unit, 0)


def test_arm6_matrix_matches_rowwise_and_ignores_gripper():
    rng = np.random.default_rng(1)
    q = rng.normal(size=(5, 10, 7)).astype(np.float32)
    c = rng.normal(size=(8, 10, 7)).astype(np.float32)
    m = ls.arm6_mse_matrix(q, c)
    for i in range(5):
        np.testing.assert_allclose(m[i], ls.arm6_mse(np.repeat(q[i:i + 1], 8, 0), c), rtol=1e-5)
    c2 = c.copy(); c2[..., 6] += 10
    np.testing.assert_allclose(ls.arm6_mse_matrix(q, c2), m, rtol=1e-6)
    # Denominator is 10*7 over the six arm axes.
    np.testing.assert_allclose(ls.arm6_mse(np.ones((1, 10, 7)), np.zeros((1, 10, 7))), [60 / 70])


def test_rows_are_causal_and_targets_are_future():
    T = 37
    pol = np.arange(T * 12, dtype=np.float32).reshape(T, 12)
    st = np.arange(T * 16, dtype=np.float32).reshape(T, 16)
    rows = ls.episode_rows(pol, st)
    assert [r[0] for r in rows] == [0, 10, 20]
    f, hist, mask, state, target = rows[1]
    np.testing.assert_array_equal(hist, pol[0:10, :7]); assert mask.all()
    np.testing.assert_array_equal(state, st[10]); np.testing.assert_array_equal(target, pol[10:20, :7])
    f0, h0, m0, _, _ = rows[0]
    assert not m0.any() and not h0.any()


def test_hold_last_uses_only_valid_history():
    rows = {"history": np.ones((2, 10, 7), np.float32), "history_mask": np.array([[True] * 10, [False] * 10])}
    out = ls.hold_last(rows)
    assert out.shape == (2, 10, 7)
    np.testing.assert_array_equal(out[0], 1.); np.testing.assert_array_equal(out[1], 0.)


def test_heldout_dev_eval_split_is_disjoint_and_deterministic():
    held = np.arange(20) * 3
    d, e = ls.split_heldout(held, 2)
    d2, e2 = ls.split_heldout(held, 2)
    np.testing.assert_array_equal(d, d2); np.testing.assert_array_equal(e, e2)
    assert len(d) == ls.DEV_PER_TASK and not np.intersect1d(d, e).size
    np.testing.assert_array_equal(np.sort(np.r_[d, e]), held)


def test_refuses_to_write_into_data_mount():
    with pytest.raises(PermissionError):
        ls._guard_output(Path("/work/data/anything"))
    ls._guard_output(Path("/work/.skillnet-state/outputs/x"))


def test_bootstrap_brackets_mean():
    v = np.r_[np.zeros(50), np.ones(50)]
    task = np.zeros(100, int); ep = np.repeat(np.arange(10), 10)
    r = ls.episode_bootstrap(v, task, ep, reps=500)
    assert r["lo"] <= r["mean"] <= r["hi"]
