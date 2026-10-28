from __future__ import annotations

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from skill_moe.skillnet.examples.emergent.light.intermediate.experiment import (
    align_segment_labels,
    dct_matrix,
    load_rows,
    low_frequency_intermediate,
    make_units,
    make_memory_payload,
    nearest_units,
    retrieve_top1_same_class,
    split_split_a,
    unit_match_metrics,
    _query_features,
)
from skill_moe.skillnet.examples.emergent.light.intermediate.abstracted_m1 import (
    action_motion_labels,
    pairwise_label_agreement,
    white_noise_mix,
)
from skill_moe.skillnet.examples.emergent.light.intermediate.compare_splits import (
    _same_class_alternatives,
    xhat0_from_velocity,
)


def test_dct_low_frequency_intermediate_selects_only_signal_dominant_coefficients():
    matrix = dct_matrix(4)
    coefficients = np.zeros((12, 4, 1), np.float32)
    coefficients[:, 0, 0] = np.linspace(-3.0, 3.0, len(coefficients))
    coefficients[:, 1, 0] = np.linspace(-0.05, 0.05, len(coefficients))
    actions = np.einsum("kt,nkd->ntd", matrix, coefficients)
    filtered, keep, variance = low_frequency_intermediate(actions, actions, t0=0.5)
    assert keep[:, 0].tolist() == [True, False, False, False]
    assert variance[0, 0] > 1.0
    np.testing.assert_allclose(filtered, np.einsum("kt,nkd->ntd", matrix, coefficients * keep[None]), atol=1e-6)


def test_intermediate_v_uses_clean_target_and_noise_mixture_at_t0():
    target = np.asarray([[1.0, -2.0]], np.float32)
    noise = np.asarray([[3.0, 4.0]], np.float32)
    t0 = 0.6
    x_t0 = t0 * noise + (1.0 - t0) * target
    np.testing.assert_allclose(x_t0, [[2.2, 1.6]], atol=1e-6)


def test_annotation_alignment_uses_cached_window_start_frames():
    classes, segments = align_segment_labels(
        [0, 5, 10], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [4, 4, 4, 4, 4, 7, 7, 7, 7, 7, 9],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2],
    )
    np.testing.assert_array_equal(classes, [4, 7, 9])
    np.testing.assert_array_equal(segments, [0, 1, 2])


def test_load_rows_joins_cache_and_parquet_with_matching_row_counts(tmp_path):
    cache_dir = tmp_path / "cache"
    episode_dir = cache_dir / "episodes"
    episode_dir.mkdir(parents=True)
    dataset_dir = tmp_path / "dataset"
    parquet_dir = dataset_dir / "data" / "chunk-000"
    parquet_dir.mkdir(parents=True)
    cache_path = episode_dir / "libero40-000007.npz"
    np.savez(cache_path, step=np.asarray([0, 5, 10], np.int32),
             env_step=np.arange(20, dtype=np.int32), action=np.zeros((20, 7), np.float32),
             env_state=np.zeros((20, 8), np.float32), f=np.ones((3, 4), np.float16),
             k=np.ones((3, 5), np.float16), proprio=np.ones((3, 8), np.float32),
             task_index=np.asarray(20, np.int32))
    table = pa.table({"frame_index": np.arange(20, dtype=np.int64),
                      "eval_skill_class": np.repeat([2, 3], 10),
                      "eval_skill_segment": np.repeat([0, 1], 10)})
    pq.write_table(table, parquet_dir / "episode_000007.parquet")
    (cache_dir / "manifest.json").write_text(json.dumps({"episodes": [{
        "episode_id": "libero40-000007", "cache": str(cache_path)}]}))

    rows = load_rows(cache_dir, dataset_dir, max_horizon=25)
    # start 0 and 5 have at least ten future frames; frame 10 is too close to
    # the tail for the mandatory H=10 support condition.
    assert len(rows["f"]) == len(rows["skill_class"]) == len(rows["a"]) == 2
    np.testing.assert_array_equal(rows["frame"], [0, 5])
    np.testing.assert_array_equal(rows["skill_class"], [2, 2])
    np.testing.assert_array_equal(rows["skill_segment"], [0, 0])
    assert rows["a"].shape == (2, 25, 7)


def test_split_a_and_memory_payload_exclude_goal_rows():
    rows = {"task_index": np.asarray([10, 20, 35]), "episode_id": np.asarray(["g", "o", "s"])}
    train, goal = split_split_a(rows)
    np.testing.assert_array_equal(train["task_index"], [20, 35])
    np.testing.assert_array_equal(goal["task_index"], [10])
    payload = make_memory_payload(train, np.zeros((2, 3)), np.asarray([0, 1]))
    assert set(payload["rows"]["episode_id"]) == {"o", "s"}
    try:
        make_memory_payload(rows, np.zeros((3, 3)), np.asarray([0, 1, 2]))
    except ValueError as exc:
        assert "goal rows" in str(exc)
    else:
        raise AssertionError("held-out goal rows were accepted into memory")


def test_open_ended_units_recover_cluster_count_and_keep_singleton_outlier_pending():
    rng = np.random.default_rng(81)
    centers = np.asarray([[0.0, 0.0], [5.0, 0.0], [0.0, 5.0]], np.float32)
    values, episodes = [], []
    for cluster, center in enumerate(centers):
        for i in range(60):
            values.append(center + rng.normal(0, 0.025, size=2))
            episodes.append(f"cluster-{cluster}-episode-{i // 10}")
    values.append(np.asarray([50.0, 50.0], np.float32))
    episodes.append("one-off-outlier")
    units = make_units(np.asarray(values, np.float32), np.asarray(episodes),
                       kappa=0.1, min_support=10, min_episodes=3,
                       pending_batch=30, pending_max_age=120, seed=81)
    assert len(units["centers"]) == 3
    assert units["labels"][-1] == -1
    assert len(units["pending_indices"]) >= 1
    assert units["registered"] == 3
    assert all(len(ids) >= 3 for ids in units["episodes"])
    assert units["history"][-1]["data_seen"] == len(values)


def test_nearest_units_handles_unformed_skill_vocabulary():
    ids, distances = nearest_units(np.ones((3, 4), np.float32), np.empty((0, 4), np.float32))
    np.testing.assert_array_equal(ids, [-1, -1, -1])
    assert np.isinf(distances).all()


def test_unit_match_is_retrieved_item_id_vs_gold_value_nearest_center():
    centers = np.asarray([[0.0, 0.0], [10.0, 0.0]], np.float32)
    gold = np.asarray([[0.0, 0.0], [10.0, 0.0], [1.0, 0.0]], np.float32)
    gold_ids, distances, covered, match, assigned_fraction, conditional_match = unit_match_metrics(
        gold, centers, np.asarray([0, 0, -1]), epsilon=1.0)
    np.testing.assert_array_equal(gold_ids, [0, 1, 0])
    np.testing.assert_allclose(distances, [0.0, 0.0, 1.0])
    np.testing.assert_array_equal(covered, [True, True, True])
    assert match == 1.0 / 3.0
    assert assigned_fraction == 2.0 / 3.0
    assert conditional_match == 0.5


def test_same_class_filter_retrieves_within_class_and_marks_global_fallback():
    q = np.asarray([[0.0], [4.0]], np.float32)
    values = np.asarray([[10.0], [20.0]], np.float32)
    units = np.asarray([1, 2])
    classes = np.asarray([0, 1])
    query = np.asarray([[3.8], [0.1]], np.float32)
    query_classes = np.asarray([1, 2])
    got, got_units, indices, fallback = retrieve_top1_same_class(
        q, values, units, query, classes, query_classes)
    np.testing.assert_array_equal(indices, [1, -1])
    np.testing.assert_allclose(got[0, 0], 20.0)
    assert np.isnan(got[1, 0])
    np.testing.assert_array_equal(got_units, [2, -1])
    np.testing.assert_array_equal(fallback, [False, True])


def test_concat_subtask_embedding_has_same_per_row_norm_as_normalized_key():
    rows = {"k": np.asarray([[3.0, 4.0], [0.0, 2.0]], np.float32),
            "s": np.zeros((2, 1), np.float32)}
    classes = np.eye(2, dtype=np.float32)
    combined = _query_features(rows, classes, "concat", np.zeros(3), np.ones(3))
    np.testing.assert_allclose(np.linalg.norm(combined[:, :3], axis=1),
                               np.linalg.norm(combined[:, 3:], axis=1), atol=1e-6)


def test_white_noise_midpoint_mixture_uses_t0_as_noise_weight():
    x_mid = np.asarray([[2.0, -1.0]], np.float32)
    noise = np.asarray([[4.0, 3.0]], np.float32)
    np.testing.assert_allclose(white_noise_mix(x_mid, noise, 0.0), x_mid)
    np.testing.assert_allclose(white_noise_mix(x_mid, noise, 0.2), [[2.4, -0.2]])


def test_motion_labels_and_residual_draw_agreement_are_axis_and_sign_sensitive():
    actions = np.zeros((2, 4, 7), np.float32)
    actions[0, :, 0] = 1.0       # +x
    actions[0, :, 4] = -0.5      # -y rotation
    actions[0, :, 6] = -1.0      # negative gripper command
    actions[1, :, 5] = 0.5       # +z rotation
    actions[1, :, 6] = 1.0
    labels = action_motion_labels(actions)
    np.testing.assert_array_equal(labels, [[1, 4, 1], [0, 5, 2]])
    draws = np.stack([labels, labels.copy(), labels.copy()])
    assert pairwise_label_agreement(draws) == {"direction": 1.0, "rotation": 1.0, "gripper": 1.0}
    draws[1, 0, 0] = 2
    assert pairwise_label_agreement(draws)["direction"] < 1.0


def test_xhat0_formula_matches_flow_velocity_convention():
    x_t = np.asarray([[1.0, -2.0]], np.float32)
    velocity = np.asarray([[0.5, 3.0]], np.float32)
    np.testing.assert_allclose(xhat0_from_velocity(x_t, velocity, 0.4), [[0.8, -3.2]])


def test_shuffle_control_uses_a_different_interval_from_the_same_class():
    classes = np.asarray([0, 0, 0, 1, 1])
    selected = np.asarray([0, 2, 3, 4])
    query = np.asarray([0, 0, 1, 1])
    alternatives = _same_class_alternatives(classes, query, selected, seed=9)
    assert np.all(alternatives >= 0)
    assert np.all(classes[alternatives] == query)
    assert np.all(alternatives != selected)
