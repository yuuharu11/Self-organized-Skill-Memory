import numpy as np
import json

from skill_moe.skillnet.examples.emergent.light.intermediate.hier_skills import (
    HierSkillMemory, make_boundaries, segment_representation,
    _load_partial_resume,
)


def test_boundary_modes_and_minimum_segment_length():
    n = 8
    grip = np.asarray([False, True, False, True, False, False, True])
    dist = np.asarray([0.1, 0.1, 0.9, 0.1, 0.1, 0.9, 0.1])
    g = make_boundaries(n, mode="gripper", grip_switch=grip)
    f = make_boundaries(n, mode="functional", functional_distance=dist,
                        functional_threshold=0.5)
    both = make_boundaries(n, mode="both", grip_switch=grip,
                           functional_distance=dist, functional_threshold=0.5)
    assert all(b - a >= 2 for spans in (g, f, both) for a, b in spans)
    assert both[0][0] == 0 and both[-1][1] == n
    assert make_boundaries(5, mode="gripper", grip_switch=[True, False, False, False]) == [(0, 5)]


def test_segment_representation_resamples_and_encodes_length():
    bases = np.arange(5 * 3, dtype=np.float32).reshape(5, 3)
    curve, length = segment_representation(bases, 0.25)
    assert curve.shape == (4, 3)
    assert np.allclose(curve[[0, -1]], bases[[0, -1]])
    assert length.shape == (1,)
    assert np.isclose(np.exp(length[0]), 1.25)


def _segment(i, episode):
    return {"local_segment": i, "episode_id": episode,
            "base_curve": np.full((4, 3), i, np.float32),
            "key": np.asarray([float(i % 2), 1.0], np.float32),
            "bases": np.full((2, 3), i, np.float32), "keys": np.ones((2, 2), np.float32)}


def test_unbounded_m6_style_clustering_registers_supported_groups_not_single_outlier():
    segments, vectors = [], []
    rng = np.random.default_rng(4)
    sid = 0
    for center in (np.asarray([0.0, 0.0]), np.asarray([1.0, 1.0])):
        for ep in range(3):
            for _ in range(4):
                segments.append(_segment(sid, f"ep-{ep}")); sid += 1
                vectors.append((center + rng.normal(0, 0.01, 2)).astype(np.float32))
    segments.append(_segment(sid, "outlier-episode")); vectors.append(np.asarray([8.0, -8.0], np.float32))
    memory = HierSkillMemory(0.01, min_segments=10, min_episodes=3, pending_batch=200)
    memory.fit(segments, np.stack(vectors), seed=0)
    assert len(memory.units) == 2
    assert memory.describe()["covered_segments"] == 24
    assert memory.describe()["pending_expired"] == 1
    assert len(memory.top1(np.asarray([1.0, 0.0]))[0].members) >= 10


def test_nearby_post_training_segment_updates_center_without_gradient():
    memory = HierSkillMemory(0.02, min_segments=2, min_episodes=1)
    s0 = _segment(0, "ep-a")
    memory.add(s0, np.asarray([0.0, 0.0], np.float32))
    s1 = _segment(1, "ep-b")
    memory.add(s1, np.asarray([0.01, 0.0], np.float32))
    assert len(memory.units) == 0
    memory.flush(force=True)
    assert len(memory.units) == 1
    unit = memory.units[0]
    previous = unit.functional_center.copy()
    s2 = _segment(2, "ep-c")
    uid, state = memory.add(s2, np.asarray([0.015, 0.0], np.float32), allow_pending=False)
    assert state == "assigned"
    assert uid == unit.skill_id
    assert unit.support == 3
    assert not np.array_equal(unit.functional_center, previous)


def test_resume_reads_completed_runs_and_checks_configuration(tmp_path):
    partial = {"seeds": [0, 1, 2], "splits": ["A", "B"], "smoke": False,
               "runs": [{"split": "A", "seed": 0}]}
    (tmp_path / "results.partial.json").write_text(json.dumps(partial))
    assert _load_partial_resume(tmp_path, (0, 1, 2), ("A", "B"), False) == partial["runs"]
    try:
        _load_partial_resume(tmp_path, (0,), ("A", "B"), False)
    except ValueError:
        pass
    else:
        raise AssertionError("resume must reject a mismatching run configuration")
