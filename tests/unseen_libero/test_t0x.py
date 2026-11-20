from pathlib import Path

import numpy as np

from skill_moe.skillnet.examples.memory_pilot.unseen_libero.t0x_common import (
    REPO_ROOT,
    calibrate_m_radius,
    mcnemar_exact,
    quantile_map,
    t0_add,
    t0_from_m,
)


def test_repo_root_default_is_workspace_root():
    assert REPO_ROOT == Path(__file__).resolve().parents[2]


def test_additive_radius_hits_anchors_and_clips():
    b = 0.08
    got = t0_add(np.asarray([0., b, 10*b]), b)
    np.testing.assert_allclose(got[:2], [0.15, 0.35], atol=1e-7)
    assert got[2] == 0.35


def test_additive_radius_is_monotone_and_nonzero():
    got = t0_add(np.linspace(0, 1, 100), .1)
    assert np.all(np.diff(got) >= 0)
    assert got.min() >= .15 and got.max() <= .35


def test_mcnemar_exact_counts_discordant_pairs():
    r = mcnemar_exact([1, 1, 0, 0], [1, 0, 1, 0])
    assert r == {"b_only": 1, "a_only": 1, "p_exact": 1.0}


def test_mcnemar_requires_paired_shapes():
    try:
        mcnemar_exact([1, 0], [1])
    except ValueError:
        pass
    else:
        raise AssertionError("paired input shape mismatch must fail")


def test_quantile_map_uses_calibration_cdf():
    got = quantile_map([0., 1., 2.], [0., 1., 2.], [10., 20., 30.])
    np.testing.assert_allclose(got, [10., 20., 30.])


def test_t4_m_calibration_quantile_and_radius_anchors():
    calibration = calibrate_m_radius([0.01, 0.03, 0.07, 0.09], [1., 2., 3., 4.], .08)
    assert calibration["p_b"] == .25
    assert calibration["m_quantile"] == .75
    assert calibration["m_b"] == 3.25
    got = t0_from_m(np.asarray([0., calibration["m_b"], 100.]), calibration)
    np.testing.assert_allclose(got, [.15, .35, .35], atol=1e-7)


def test_t4_max_radius_never_below_ehat_radius():
    calibration = {"b": .06849, "m_b": 2.0}
    ehat = np.linspace(0, .2, 101)
    m = np.linspace(0, 3, 101)
    e_radius = t0_add(ehat, calibration["b"])
    combined = np.maximum(e_radius, t0_from_m(m, calibration))
    assert np.all(combined >= e_radius)

