import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'skill_moe/skillnet/examples/memory_pilot/unseen_libero'))
import expert_ex3 as ex3


def test_candidate_mse_matches_direct():
    rng = np.random.default_rng(0)
    y = rng.normal(size=(4, 10, 7)); u = rng.normal(size=(9, 10, 7))
    d = ex3.candidate_mse(y, u)
    direct = np.square(y[:, None, :, :6] - u[None, :, :, :6]).sum((-1, -2)) / 70.
    assert np.allclose(d, direct)
    assert np.allclose(ex3.arm6(u[[2]], y[[0]]), direct[0, 2])


def test_true_rank_respects_eligibility():
    m = np.array([[.1, .5, .2, .05], [.3, .2, .1, .4]])
    el = np.array([[True, True, True, False], [True, True, True, True]])
    r = ex3.true_rank(m, np.array([2, 2]), el)
    assert r.tolist() == [2, 1]          # unit 3 (.05) is ineligible for row 0
    try:
        ex3.true_rank(m, np.array([3, 0]), el)
    except ValueError:
        pass
    else:
        raise AssertionError('ineligible pick must raise')


def test_hold_last_gripper():
    g = ex3.hold_last_gripper(np.array([.7, -.2, 0., .5]), np.array([True, True, True, False]))
    assert g.shape == (4, 10)
    assert g[:, 0].tolist() == [1., -1., -1., -1.]
    assert (g == g[:, :1]).all()


def test_sample_ranks_law_and_clip():
    rng = np.random.default_rng(1)
    measured = np.full(1000, 300)
    r, near = ex3.sample_ranks(rng, 5000, measured, np.full(5000, 200), near_fraction=.2)
    assert abs(near.mean() - .2) < .03
    assert (r[~near] == 200).all()                   # clipped to eligible count
    assert r[near].min() >= 1 and r[near].max() <= 50
    r2, near2 = ex3.sample_ranks(np.random.default_rng(1), 5000, np.full(10, 3), np.full(5000, 4000), near_fraction=0., rank_scale=5)
    assert (r2 == 15).all() and not near2.any()


def test_select_t0_and_interp():
    t = ex3.select_t0(np.array([0., .01, 1.]), 2.)
    assert np.allclose(t, [.05, .2, .6])
    cur = np.array([[1., 2., 3.], [0., 0., 6.]])
    v = ex3.interp_curve(cur, (.1, .2, .4), np.array([.15, .3]))
    assert np.allclose(v, [1.5, 3.])


def test_switch_metrics():
    tgt = np.array([[-1] * 3 + [1] * 7, [-1] * 10, [1] * 10], float)
    pred = np.array([[-1] * 4 + [1] * 6, [-1] * 9 + [1], [1] * 10], float)
    m = ex3.switch_metrics(pred, tgt, np.array([-1., -1., 1.]), np.array([True, True, True]))
    assert m['gt_switch_rows'] == 1 and m['missed_switch_rate'] == 0.
    assert m['timing_error_mean_steps'] == 1.
    assert m['false_switch_rate'] == .5


def test_mcnemar():
    a = np.array([1, 1, 0, 0, 1], bool); b = np.array([1, 0, 1, 1, 1], bool)
    r = ex3.mcnemar_exact(a, b)
    assert r['b_only'] == 2 and r['a_only'] == 1 and r['p'] == 1.0
