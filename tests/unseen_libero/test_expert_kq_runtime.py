import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'skill_moe/skillnet/examples/memory_pilot/unseen_libero'))
import expert_kq_runtime as ekq


def _rows(n=4, m=12, seed=0):
    rng = np.random.default_rng(seed)
    units = rng.normal(size=(m, 10, 7)).astype(np.float32)
    units[..., 6] = np.where(rng.random((m, 10)) > .5, 1., -1.)
    hist = rng.normal(size=(n, 10, 7)).astype(np.float32)
    mask = np.ones((n, 10), bool)
    mask[0] = False; hist[0] = 0
    mask[1, :7] = False; hist[1, :7] = 0
    hist[..., 6] = np.where(hist[..., 6] > 0, 1., -1.) * mask
    state = rng.normal(size=(n, 8)).astype(np.float32)
    scores = rng.normal(size=(n, m)) * 5
    return units, hist, mask, state, scores


def test_score_summary_matches_ex2_entropy_and_order():
    import expert_ex2 as ex2
    units, hist, mask, state, scores = _rows()
    s = ekq.score_summary(scores, k=5)
    assert np.array_equal(s['ids'][:, 0], scores.argmax(1))
    assert np.all(np.diff(s['vals'], axis=1) <= 0)
    d = units[s['ids'][:, 0]]
    old = ex2.confidence_features(d, d, scores)
    np.testing.assert_allclose(s['entropy'], old[:, 2], rtol=1e-5)
    np.testing.assert_allclose(s['vals'][:, 0] - s['vals'][:, 1], old[:, 1], rtol=1e-5)


def test_score_summary_masked_entropy_uses_finite_count():
    sc = np.zeros((1, 6)); sc[0, 3:] = -np.inf
    s = ekq.score_summary(sc, k=3)
    assert abs(float(s['entropy'][0]) - 1.0) < 1e-6
    assert set(s['ids'][0].tolist()) == {0, 1, 2}


def test_confidence_features_shape_and_no_history_row():
    units, hist, mask, state, scores = _rows()
    s = ekq.score_summary(scores)
    f = ekq.confidence_features(s['ids'], s['vals'], s['entropy'], s['top1_prob'], units, hist, mask, state, np.ones(4))
    assert f.shape == (4, len(ekq.CONF_FEATURES)) and np.isfinite(f).all()
    assert f[0, ekq.CONF_FEATURES.index('history_valid_frac')] == 0
    assert f[1, ekq.CONF_FEATURES.index('history_valid_frac')] == pytest.approx(.3)
    # identical top-k drafts -> minimal dispersion
    s2 = dict(s); s2['ids'] = np.repeat(s['ids'][:, :1], 5, 1)
    f2 = ekq.confidence_features(s2['ids'], s['vals'][:, :5], s['entropy'], s['top1_prob'], units, hist, mask, state, np.ones(4))
    assert np.allclose(f2[:, ekq.CONF_FEATURES.index('log_topk_pair_mse')], np.log(1e-6), atol=1e-4)


def test_gripper_inputs_dim():
    units, hist, mask, state, scores = _rows()
    s = ekq.score_summary(scores)
    x = ekq.gripper_inputs(s['ids'], s['vals'], units, hist, mask, state)
    assert x.shape == (4, ekq.GRIP_INPUT_DIM)


def test_hold_last_and_flip_rule():
    hist = np.zeros((2, 10, 7), np.float32); mask = np.zeros((2, 10), bool)
    hist[1, 9, 6] = 1.; mask[1, 9] = True
    hl = ekq.hold_last(hist, mask)
    assert np.all(hl[0] == ekq.DEFAULT_START_GRIP) and np.all(hl[1] == 1)
    ug = np.array([[-1] * 4 + [1] * 6, [1] * 10], np.float32)
    fr = ekq.unit_flip_rule(hist, mask, ug)
    assert np.all(fr[0, :4] == -1) and np.all(fr[0, 4:] == 1) and np.all(fr[1] == 1)


def test_switch_metrics_timing_missed_false():
    hist = np.zeros((3, 10, 7), np.float32); mask = np.ones((3, 10), bool); hist[:, 9, 6] = -1
    tgt = np.full((3, 10), -1.); tgt[0, 3:] = 1; tgt[1, 6:] = 1
    pred = np.full((3, 10), -1.); pred[0, 5:] = 1; pred[2, 2:] = 1
    m = ekq.switch_metrics(pred, tgt, hist, mask)
    assert m['gt_switch_rows'] == 2
    assert m['missed_switch_rate'] == .5
    assert m['false_switch_rate'] == 1.0
    assert m['timing_error_mean_steps'] == 2


def test_select_t0_and_interp():
    t = ekq.select_t0(np.array([0., .04, 1.]), 2., .05, .6)
    np.testing.assert_allclose(t, [.05, .4, .6], rtol=1e-6)
    curve = np.array([[0., 1., 2.]]); grid = (.1, .2, .4)
    assert ekq.interp_curve(curve, grid, np.array([.3]))[0] == pytest.approx(1.5)
    assert ekq.interp_curve(curve, grid, np.array([.9]))[0] == pytest.approx(2.)


def test_heads_roundtrip():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(64, len(ekq.CONF_FEATURES))).astype(np.float32)
    y = x[:, 0] * .5 - 3
    m = ekq.fit_conf_head(x, y, 'linear', steps=300, lr=1e-2)
    p = ekq.predict_conf(m, x)
    assert p.shape == (64,) and np.corrcoef(np.log(p), y)[0, 1] > .9
    gx = rng.normal(size=(64, ekq.GRIP_INPUT_DIM)).astype(np.float32)
    gy = np.where(gx[:, :10] > 0, 1., -1.)
    gm = ekq.fit_grip_head(gx, gy, hidden=32, steps=300)
    assert set(np.unique(ekq.predict_grip(gm, gx))) <= {-1., 1.}


def test_metric_conventions():
    a = np.zeros((1, 10, 7)); b = np.zeros((1, 10, 7)); b[..., :6] = 1; b[..., 6] = 2
    assert ekq.arm6_mse(a, b)[0] == pytest.approx(60 / 70)
    assert ekq.grip_mse(a[..., 6], b[..., 6])[0] == pytest.approx(40 / 70)


def test_grip_from_prob_threshold_around_hold_last():
    p_close = np.array([[.1, .55, .95] + [.5] * 7, [.1, .45, .95] + [.5] * 7])
    g0 = np.array([-1., 1.])
    out = ekq.grip_from_prob(p_close, g0, .6)
    assert out[0, :3].tolist() == [-1, -1, 1]      # open -> close only when P(close) > .6
    assert out[1, :3].tolist() == [-1, 1, 1]       # closed -> open only where P(open) > .6 (.9 yes, .55 no)
