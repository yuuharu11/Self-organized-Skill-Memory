"""Shared offline metrics must retain the original evaluation conventions."""
import numpy as np
import pytest

from skill_moe.skillnet.examples.memory_pilot.anchor_flow.offline_common import (
    cluster_bootstrap,
    normalize_keys,
    read_npz,
    retrieval_metrics,
    row_errors,
)
from skill_moe.skillnet.examples.memory_pilot.anchor_flow.offline_retrieval import (
    behavior_neighbors,
    macro_recall,
    sample_task_queries,
    topk_cosine,
)


def test_action_error_excludes_padding_and_endpoint_uses_net_displacement():
    prediction = np.zeros((2, 2, 32), np.float32)
    target = np.zeros_like(prediction)
    prediction[..., 7:] = 1e6
    prediction[0, :, 0] = [1, -1]
    prediction[1, :, 0] = [1, 1]
    mse, mm = row_errors(prediction, target)
    np.testing.assert_allclose(mse, [1 / 7, 1 / 7])
    np.testing.assert_allclose(mm, [0, 22.2], atol=1e-5)


@pytest.mark.parametrize("seed", [0, 42])
def test_bootstrap_draws_whole_episodes_with_unequal_lengths(seed):
    # Two episode draws yield either 0, 25, or 100. Resampling individual rows
    # would give a narrower upper interval, and averaging episode means gives 50.
    values = np.array([0., 0., 0., 100.])
    episodes = np.array(["long", "long", "long", "short"])
    result = cluster_bootstrap(values, episodes, reps=2000, seed=seed)
    assert result == (25., 0., 100.)
    assert cluster_bootstrap(np.repeat(values, 3), np.repeat(episodes, 3),
                             reps=2000, seed=seed) == result


def test_npz_requires_metadata_and_only_loads_requested_columns(tmp_path):
    path = tmp_path / "rows.npz"
    np.savez(path, actions=np.ones((2, 10, 32)), episode_id=["a", "b"], extra=[99])
    rows = read_npz(path, ("actions", "episode_id"))
    assert set(rows) == {"actions", "episode_id"}
    np.testing.assert_array_equal(rows["episode_id"], ["a", "b"])
    with pytest.raises(ValueError, match="missing fields.*init_state_id.*state"):
        read_npz(path, ("actions", "state", "init_state_id"))


def test_normalization_handles_zero_keys_and_preserves_float32():
    keys = np.array([[0, 0], [3, 4]], np.float64)
    normalized = normalize_keys(keys)
    assert normalized.dtype == np.float32
    np.testing.assert_array_equal(normalized[0], [0, 0])
    np.testing.assert_allclose(normalized[1], [.6, .8])
    np.testing.assert_array_equal(keys, [[0, 0], [3, 4]])


def test_query_sampling_keeps_task_episode_plan_and_even_frame_coverage():
    tasks = np.repeat(["task-a", "task-b", "task-c"], 30)
    episodes = np.repeat([f"episode-{i}" for i in range(9)], 10)
    rows, selected, plan = sample_task_queries(
        tasks, episodes, 21, n_tasks=2, episodes_per_task=2, rows_per_episode=3)
    assert len(selected) == 2
    assert set(tasks[rows]) == set(selected) == set(plan)
    assert len(rows) == 12
    for task, chosen in plan.items():
        assert len(chosen) == 2
        for episode in chosen:
            original = np.flatnonzero((tasks == task) & (episodes == episode))
            actual = rows[episodes[rows] == episode]
            assert len(actual) == 3
            assert actual[0] == original[0] and actual[-1] == original[-1]
    repeated, selected_again, plan_again = sample_task_queries(
        tasks, episodes, 21, n_tasks=2, episodes_per_task=2, rows_per_episode=3)
    np.testing.assert_array_equal(rows, repeated)
    assert (selected, plan) == (selected_again, plan_again)


def test_behavior_neighbors_exclude_both_episode_and_initial_state():
    bases = np.arange(5, dtype=np.float32)[:, None]
    episodes = np.array(["a", "a", "b", "c", "d"])
    initials = np.array(["i0", "i1", "i0", "i2", "i3"])
    neighbors = behavior_neighbors(bases, episodes, initials, k=3, query_k=2, workers=1)
    for i, candidates in enumerate(neighbors):
        assert np.all(episodes[candidates] != episodes[i])
        assert np.all(initials[candidates] != initials[i])
    # Row zero has only two eligible examples, repeated to fill three positives.
    assert set(neighbors[0]) == {3, 4}


def test_cosine_ties_use_row_order_after_both_exclusions():
    keys = np.tile([1., 0.], (6, 1))
    episodes = np.array([f"e{i}" for i in range(6)])
    initials = np.array([f"i{i}" for i in range(6)])
    rows, scores = topk_cosine(keys, [[1., 0.]], episodes, initials,
                               np.array(["e0"]), np.array(["i1"]), k=3)
    np.testing.assert_array_equal(rows, [[2, 3, 4]])
    np.testing.assert_array_equal(scores, [[1., 1., 1.]])


def test_macro_recall_weights_units_equally():
    metrics = macro_recall([0, 0, 0, 1], [[0, 1]] * 4, majority_unit=0)
    assert metrics["hit1"] == .75
    assert metrics["macro_recall1"] == .5
    assert metrics["hit5"] == metrics["macro_recall5"] == 1.


def test_key_pooling_diagnostic_retains_distance_based_weights():
    entries = [dict(unit_id=i, meta=dict(episode_id=f"e{i}", init_state_id=f"i{i}"))
               for i in range(3)]
    metrics = retrieval_metrics([[1., 0.]], [[1., 0.], [0., 1.], [-1., 0.]],
                                entries, ["e0"], ["new"], [1], 0, temperature=.5)
    np.testing.assert_array_equal(metrics["top_indices"][0], [1, 2])
    expected = np.exp(-np.array([np.sqrt(2.), 2.]) / .5)
    np.testing.assert_allclose(metrics["top_weights"][0], expected / expected.sum(), rtol=1e-6)
