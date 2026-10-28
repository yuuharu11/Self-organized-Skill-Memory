import numpy as np
import pytest
import torch

from skill_moe.skillnet.examples.emergent.light.intermediate import anchor_flow as af
from skill_moe.skillnet.examples.emergent.light.intermediate import fair_compare as fc
from skill_moe.skillnet.examples.emergent.light.intermediate.experiment import FlowActionHead

H, D = 25, 7


def _setup(n=9, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = FlowActionHead(6, 4, H * D).eval()
    norm = {"f_mean": np.zeros(6, np.float32), "f_scale": np.ones(6, np.float32),
            "s_mean": np.zeros(4, np.float32), "s_scale": np.ones(4, np.float32),
            "a_mean": rng.normal(size=H * D).astype(np.float32) * 0.1,
            "a_scale": np.full(H * D, 0.5, np.float32), "action_shape": (H, D)}
    rows = {"f": rng.normal(size=(n, 6)).astype(np.float32), "s": rng.normal(size=(n, 4)).astype(np.float32),
            "a": rng.normal(size=(n, H, D)).astype(np.float32)}
    eps = rng.standard_normal((fc.SAMPLES, n, H * D)).astype(np.float32)
    return model, norm, rows, eps


def test_t0_one_n10_matches_ordinary_policy_exactly():
    model, norm, rows, eps = _setup()
    ref = af._rollout_flow(model, norm, rows, None, eps, np.ones(len(rows["a"]), np.float32),
                           steps=10, device="cpu")
    mine_noise = fc.rollout(model, norm, rows["f"], rows["s"], None, eps, np.ones(len(rows["a"])), device="cpu")
    anchor = np.random.default_rng(5).normal(size=rows["a"].shape).astype(np.float32) * 3
    mine_anchor = fc.rollout(model, norm, rows["f"], rows["s"], anchor, eps, np.ones(len(rows["a"])), device="cpu")
    assert fc.steps_for(1.0) == 10
    np.testing.assert_allclose(mine_noise, ref, rtol=0, atol=1e-6)
    # At t0 = 1 the anchor coefficient is zero: identical to the ordinary policy.
    np.testing.assert_allclose(mine_anchor, ref, rtol=0, atol=1e-6)


def test_per_row_t0_and_steps_match_individual_rollouts():
    model, norm, rows, eps = _setup()
    anchor = np.random.default_rng(1).normal(size=rows["a"].shape).astype(np.float32)
    t0 = np.asarray([0.1, 0.2, 0.4, 0.6, 1.0, 0.3, 0.8, 0.5, 0.7], np.float32)
    batch = fc.rollout(model, norm, rows["f"], rows["s"], anchor, eps, t0, device="cpu")
    for i in range(len(t0)):
        one = {k: v[i:i + 1] for k, v in rows.items()}
        src = af._standardize(anchor[i:i + 1].reshape(1, -1), norm["a_mean"], norm["a_scale"])
        steps = int(fc.steps_for(t0[i]))
        ref = af._rollout_flow(model, norm, one, np.broadcast_to(src[None], (fc.SAMPLES, 1, H * D)),
                               eps[:, i:i + 1], t0[i:i + 1], steps=steps, device="cpu")
        np.testing.assert_allclose(batch[:, i:i + 1], ref, rtol=0, atol=1e-5)
    # Step size equals the ordinary 0.1 on the grid.
    assert np.allclose(t0 / fc.steps_for(t0), 0.1)


def test_noise_is_paired_and_methods_differ_only_by_start():
    model, norm, rows, eps = _setup()
    a1 = np.zeros(rows["a"].shape, np.float32)
    t0 = np.full(len(rows["a"]), 0.4, np.float32)
    p1 = fc.rollout(model, norm, rows["f"], rows["s"], a1, eps, t0, device="cpu")
    p2 = fc.rollout(model, norm, rows["f"], rows["s"], a1, eps.copy(), t0, device="cpu")
    np.testing.assert_array_equal(p1, p2)
    # Changing only one row's noise changes only that row's output.
    eps3 = eps.copy(); eps3[:, 2] += 1.0
    p3 = fc.rollout(model, norm, rows["f"], rows["s"], a1, eps3, t0, device="cpu")
    changed = np.abs(p3 - p1).reshape(fc.SAMPLES, len(t0), -1).max(axis=(0, 2)) > 0
    assert changed.tolist() == [i == 2 for i in range(len(t0))]


def test_calibration_rejects_new_suite_and_eval_episodes():
    rows = {"task_index": np.asarray([20, 21, 25]), "episode_id": np.asarray(["e1", "e2", "e3"])}
    fc.assert_calibration_rows(rows, "A", forbidden_episodes=("x",))
    bad = {"task_index": np.asarray([20, 12]), "episode_id": np.asarray(["e1", "e2"])}
    with pytest.raises(ValueError):
        fc.assert_calibration_rows(bad, "A")        # goal task in Split A calibration
    bad_b = {"task_index": np.asarray([31]), "episode_id": np.asarray(["e1"])}
    with pytest.raises(ValueError):
        fc.assert_calibration_rows(bad_b, "B")      # spatial task in Split B calibration
    with pytest.raises(ValueError):
        fc.assert_calibration_rows(rows, "A", forbidden_episodes=("e2",))


def test_monotone_g_is_non_increasing_and_optimal():
    rng = np.random.default_rng(0)
    conf = rng.uniform(size=400)
    grid = np.asarray(fc.T0_GRID)
    # Ideal t0 decreases with confidence, plus noise in a middle bin.
    ideal = np.clip(1.0 - conf, 0.1, 1.0)
    err = (grid[:, None] - ideal[None]) ** 2 + rng.normal(scale=0.01, size=(len(grid), 400))
    g = fc.fit_monotone_g(conf, err)
    assert all(a >= b for a, b in zip(g["t0"], g["t0"][1:]))
    # Brute-force check on 3 bins.
    g3 = fc.fit_monotone_g(conf, err, bins=3)
    which = np.searchsorted(g3["edges"], conf, side="right")
    cost = np.stack([err[:, which == b].sum(1) for b in range(3)])
    best = min(cost[0, i] + cost[1, j] + cost[2, k] for i in range(10) for j in range(10) for k in range(10)
               if grid[i] >= grid[j] >= grid[k])
    got = sum(cost[b, list(grid).index(g3["t0"][b])] for b in range(3))
    assert np.isclose(got, best)
    assert np.allclose(fc.apply_g(g3, conf), np.asarray(g3["t0"])[which])


def test_derangement_has_no_fixed_points():
    rng = np.random.default_rng(0)
    for k in (2, 3, 17, 64):
        p = fc.derangement(k, rng)
        assert sorted(p.tolist()) == list(range(k)) and not np.any(p == np.arange(k))


def test_row_metrics_zero_for_identical_prediction():
    _, _, rows, _ = _setup()
    target = rows["a"]
    pred = np.stack([target] * fc.SAMPLES)
    scales = {"pos_m_per_action_unit_step": [0.01] * 3, "rot_rad_per_action_unit_step": [0.1] * 3}
    m = fc.row_metrics(pred, target, pred, scales, device="cpu")
    for k in ("mse", "best8", "pos_term_mm", "dn", "spread", "spread_pos_term_mm"):
        assert np.allclose(m[k], 0, atol=1e-6)
    assert np.allclose(m["grip_agree"], 1)
