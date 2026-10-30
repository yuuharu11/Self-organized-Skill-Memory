import numpy as np

from skill_moe.skillnet.examples.emergent.light.intermediate.skill_v3_competition import (
    CompetitiveExperts,
    canonicalize_v4,
    restore_v4_geometry,
)


def test_v4_canonical_geometry_roundtrip_preserves_direction_distance_and_curve():
    curve = np.zeros((12, 7), np.float32)
    curve[:, 0] = np.linspace(0, 0.2, len(curve))
    curve[:, 1] = np.linspace(0, 0.1, len(curve))
    curve[:, 6] = np.linspace(-1, 1, len(curve))
    canonical, direction, distance = canonicalize_v4(curve)
    restored = restore_v4_geometry(canonical, direction, distance)
    assert np.isclose(distance, np.sqrt(.2**2 + .1**2), atol=1e-6)
    assert np.allclose(direction, np.asarray([2, 1, 0]) / np.sqrt(5), atol=1e-6)
    assert np.allclose(restored[:, :3], curve[:, :3], atol=1e-6)
    assert np.allclose(restored[:, 6], curve[:, 6])


def test_competitive_experts_fit_router_without_using_labels():
    nseg, per = 12, 4
    n = nseg * per
    rows = {
        "k": np.zeros((n, 2), np.float32),
        "s": np.zeros((n, 1), np.float32),
        "a": np.zeros((n, 1, 7), np.float32),
    }
    segments = []
    rng = np.random.default_rng(7)
    for i in range(nseg):
        group = i % 2
        ix = np.arange(i * per, (i + 1) * per)
        rows["k"][ix, group] = 1
        rows["a"][ix, 0, 0] = group * 2 + rng.normal(0, .01, per)
        curve = np.zeros((32, 7), np.float32)
        curve[:, 0] = group
        segments.append({"segment_id": i, "rows": ix, "episode_id": f"ep-{i//2}",
                         "task_index": group, "key": np.eye(2, dtype=np.float32)[group],
                         "curve": curve, "length": per, "skill_class": -1})
    experts = CompetitiveExperts(2, 1, 7, min_support=2, min_episodes=2,
                                 max_rounds=2, device="cpu")
    experts.fit(rows, segments, seed=3, epochs=2).fit_router(rows, segments, seed=5, epochs=3)
    pred, scores = experts.predict(rows, np.zeros(n, np.float32))
    assert pred.shape == (n, len(experts.experts), 7)
    assert scores.shape == (n, len(experts.experts))
    assert np.isfinite(pred).all() and np.isfinite(scores).all()
    assert experts.assignments.shape == (nseg,)
    assert np.all((experts.assignments >= 0) & (experts.assignments < len(experts.experts)))


def test_soft_em_responsibilities_are_normalized_and_threshold_parameters_are_explicit():
    nseg, per = 12, 4
    n = nseg * per
    rows = {"k": np.zeros((n, 2), np.float32), "s": np.zeros((n, 1), np.float32),
            "a": np.zeros((n, 1, 7), np.float32)}
    segments = []
    for i in range(nseg):
        group = i % 2
        ix = np.arange(i * per, (i + 1) * per)
        rows["k"][ix, group] = 1
        rows["a"][ix, 0, 0] = group * 1.5
        curve = np.zeros((32, 7), np.float32); curve[:, 0] = group
        segments.append({"segment_id": i, "rows": ix, "episode_id": f"episode-{i//2}",
                         "task_index": group, "key": np.eye(2, dtype=np.float32)[group],
                         "curve": curve, "length": per, "skill_class": -1})
    experts = CompetitiveExperts(2, 1, 7, q=.7, min_support=2, min_episodes=2,
                                 temperature=.05, max_rounds=2, device="cpu")
    experts.fit(rows, segments, seed=11, epochs=2)
    assert experts.q == .7
    assert experts.min_support == 2 and experts.min_episodes == 2
    assert np.allclose(experts.responsibilities.sum(axis=1), 1.)
    assert experts.temperature == .05
