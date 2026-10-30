import numpy as np

from skill_moe.skillnet.examples.emergent.light.intermediate.skill_v2 import (
    TrajectorySkillMemory,
    _confidence_for,
    _monotone_bins,
    _rep,
    apply_confidence_map,
    constrain_spans,
    derive_length_limits,
    decode_member_anchor,
    functional_positive_ids,
    fit_multi_positive_selector,
    _candidate_choice,
    _fit_repair_lambda,
    _rep,
    _v22_schedule,
    SkillUnit,
    trajectory_representation,
    tau_from_confidence,
)


def test_trajectory_is_resampled_and_relative_to_initial_motion():
    a = np.zeros((7, 7), np.float32)
    a[:, 0] = np.arange(7) + 5
    a[:, 1] = np.arange(7) * 2 - 8
    a[:, 6] = [1, 1, 1, -1, -1, -1, -1]
    curve, log_duration = trajectory_representation(a, 0.5)
    assert curve.shape == (32, 7)
    assert np.allclose(curve[0, :6], 0)
    assert np.isclose(curve[-1, 0], 6)
    assert np.allclose(curve[:, 6], np.interp(np.linspace(0, 1, 32),
                                               np.linspace(0, 1, 7), a[:, 6]))
    assert np.isclose(np.exp(log_duration), 3.5)


def test_duration_limits_are_derived_and_constraints_split_merge_without_losing_rows():
    rows = {"episode_id": np.asarray(["e1"] * 10 + ["e2"] * 15),
            "frame": np.r_[np.arange(10), np.arange(15)],
            "a": np.zeros((25, 1, 7), np.float32)}
    rows["a"][:, 0, 6] = np.r_[np.ones(5), -np.ones(5), np.ones(8), -np.ones(7)]
    limits = derive_length_limits(rows)
    assert limits["minimum"] >= 2
    assert limits["maximum"] >= limits["minimum"]
    spans = constrain_spans([(0, 2), (2, 3), (3, 19)], 2, 6)
    assert spans[0][0] == 0 and spans[-1][1] == 19
    assert all(b - a <= 6 for a, b in spans)
    assert sum(b - a for a, b in spans) == 19


def _segment(i, episode, curve):
    return {"segment_id": i, "episode_id": episode, "curve": curve.astype(np.float32),
            "length": 8, "length_seconds": 4.0, "key": np.asarray([1., 0.], np.float32),
            "task_index": i % 2, "skill_class": i % 3}


def test_open_ended_memory_forms_supported_units_and_leaves_singleton_outlier_unregistered():
    rng = np.random.default_rng(1)
    segments = []
    for center in (0.0, 1.0):
        group_id = 0 if center == 0 else 1
        for i in range(12):
            curve = np.full((32, 7), center, np.float32) + rng.normal(0, .002, (32, 7))
            segments.append(_segment(len(segments), f"ep-{group_id}-{i % 3}", curve))
    segments.append(_segment(24, "outlier", np.full((32, 7), 8.0, np.float32)))
    memory = TrajectorySkillMemory(.1, kappa=.1, min_support=10, min_episodes=3).fit(segments, seed=3)
    assert len(memory.units) == 2
    assert len(memory.assignments) == 24
    assert memory.expired == 1
    for unit in memory.units:
        assert unit.support >= 10
        assert len(unit.episodes) >= 3
        assert unit.variance.shape == (32, 7)
        assert unit.medoid.shape == unit.mean.shape == (32, 7)


def test_confidence_map_is_monotone_and_uses_same_selected_key_score_for_wrong_choice():
    scores = np.asarray([.8, .6, -.1], np.float32)
    c_top, sim_top, margin_top = _confidence_for(scores, 0)
    c_wrong, sim_wrong, margin_wrong = _confidence_for(scores, 2)
    assert sim_top == scores[0] and margin_top > 0
    assert sim_wrong == scores[2] and margin_wrong < 0
    assert c_wrong < c_top
    assert np.all(np.diff(tau_from_confidence(np.linspace(0, 1, 20))) <= 1e-7)
    conf = np.linspace(0, 1, 40)
    errors = np.stack([np.square(conf - tau) for tau in (.9, .6, .3, .1)])
    calibration = _monotone_bins(conf, errors, tau_grid=(.9, .6, .3, .1), bins=4)
    assert np.all(np.diff(calibration["tau"]) <= 1e-8)
    tau, steps = apply_confidence_map(calibration, np.asarray([0., 1.], np.float32))
    assert tau[0] >= tau[1]
    assert steps[0] >= steps[1] >= 2


def test_functional_positive_set_uses_epsilon_and_includes_assigned_unit():
    rng = np.random.default_rng(5)
    segments = []
    for center, group in ((0.0, 0), (1.5, 1)):
        for i in range(12):
            curve = np.full((32, 7), center, np.float32) + rng.normal(0, .001, (32, 7))
            segments.append(_segment(len(segments), f"ep-{group}-{i % 3}", curve))
    memory = TrajectorySkillMemory(.1, kappa=.1, min_support=10, min_episodes=3).fit(segments, seed=9)
    assert len(memory.units) == 2
    near = segments[0]
    far = dict(near, segment_id=999, curve=np.full((32, 7), 12.0, np.float32))
    near_ids = functional_positive_ids(near, memory)
    far_ids = functional_positive_ids(far, memory)
    assert len(near_ids) == 1
    assert far_ids == []


def test_multi_positive_selector_trains_and_returns_topk_candidates():
    rng = np.random.default_rng(6)
    segments = []
    for center, group in ((0.0, 0), (1.2, 1)):
        for i in range(12):
            curve = np.full((32, 7), center, np.float32) + rng.normal(0, .003, (32, 7))
            seg = _segment(len(segments), f"ep-{group}-{i % 3}", curve)
            seg["key"] = np.eye(2, dtype=np.float32)[group]
            segments.append(seg)
    memory = TrajectorySkillMemory(.1, kappa=1000, min_support=10, min_episodes=3).fit(segments, seed=1)
    model, metrics = fit_multi_positive_selector(segments, memory, seed=4, epochs=3)
    assert metrics["segments"] == 24
    assert metrics["loss_final"] < metrics["loss_start"]
    assert 0 <= metrics["training_top5_positive_recall"] <= 1


def test_state_dependent_anchor_decodes_member_base_at_query_observation():
    class DummyM1:
        residual_dim = 2
        def decode(self, b, r, f, s):
            assert b.shape == (1, 2)
            assert r.shape == (1, 2)
            return np.full((1, 3, 7), float(b[0, 0] + f[0, 0] + s[0, 0]), np.float32)

    member = {"base_seq": np.asarray([[0., 2.], [4., 6.]], np.float32)}
    out = decode_member_anchor(DummyM1(), member, .25,
                               np.asarray([10., 0.]), np.asarray([1., 0.]))
    assert out.shape == (3, 7)
    assert np.allclose(out, 12.)  # base first coordinate 1 + current f 10 + state 1


def test_denoising_repair_ranks_candidate_and_phase_and_calibrates_lambda():
    similarity = np.asarray([[.9, .6], [.8, .5]], np.float32)
    correction = np.asarray([[[.8, .7], [.1, .05]],
                             [[.9, .8], [.2, .04]]], np.float32)
    k0, p0, _ = _candidate_choice(similarity, correction, 0.0)
    k1, p1, _ = _candidate_choice(similarity, correction, 2.0)
    assert np.array_equal(k0, [0, 0])
    assert np.array_equal(k1, [1, 1])
    assert np.array_equal(p1, [1, 1])
    candidate_mse = np.asarray([[[.8, .7], [.1, .05]],
                                [[.9, .8], [.2, .04]]], np.float32)
    cal = _fit_repair_lambda(similarity, correction, candidate_mse,
                             lambdas=(0., 1., 2.))
    assert cal["selected_lambda"] == 2.0


def test_v22_fixed_policy_reselects_only_after_the_selected_skill_runs_its_length():
    curve0 = np.zeros((32, 7), np.float32)
    curve1 = np.ones((32, 7), np.float32) * 3
    units = []
    for uid, curve in enumerate((curve0, curve1)):
        unit = SkillUnit(uid, curve, curve.copy(), np.zeros_like(curve), 2.,
                         np.asarray([1., 0.], np.float32))
        unit.mean_rep = _rep(curve, 2.)
        units.append(unit)
    memory = TrajectorySkillMemory(.1)
    memory.units = units
    segment = {"segment_id": 0, "rows": np.arange(5), "curve": curve0,
               "length": 2, "episode_id": "e", "task_index": 0}
    sim = np.asarray([[1., 0.], [0., 1.], [0., 1.], [1., 0.], [1., 0.]], np.float32)
    bundle = {"similarity": sim, "correction": np.zeros((5, 2, 2), np.float32),
              "skill_ids": np.tile(np.asarray([0, 1]), (5, 1)),
              "phases": np.asarray([0., .5], np.float32)}
    rows = {"a": np.zeros((5, 1, 25, 7), np.float32)}
    fixed = _v22_schedule(rows, [segment], memory, bundle, lam=0., k=2, fixed=True)
    chunked = _v22_schedule(rows, [segment], memory, bundle, lam=0., k=2, fixed=False)
    assert fixed["skill_id"].tolist() == [0, 0, 1, 1, 0]
    assert chunked["skill_id"].tolist() == [0, 1, 1, 0, 0]
