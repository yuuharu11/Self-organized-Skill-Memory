import json
from pathlib import Path

import numpy as np
import pytest

from skill_moe.skillnet.examples.memory_pilot.g2_retrieval.evaluate import (
    Episode,
    EvaluationConfig,
    _compare_to_proprio,
    _distances,
    _evaluate_ranked,
    _feature_order,
    _motion_target,
    _outcome_distance,
    _random_order,
    _stack_rows,
    evaluate_run,
    load_run,
    split_by_init_state,
    write_report,
)


def _make_run(root: Path, n_episodes: int = 12):
    (root / "episodes").mkdir(parents=True)
    manifest = {"suite": "synthetic", "episodes": []}
    # Each initial state supplies a unique, constant velocity; features encode it.
    for i in range(n_episodes):
        eid = f"ep{i:02d}"
        manifest["episodes"].append({"episode_id": eid, "init_state_id": f"init{i:02d}", "task": "move"})
        t = np.arange(16, dtype=np.float32)
        velocity = np.array([i + 1, 1, 0], dtype=np.float32) / 100
        pos = np.stack([t * velocity[0], t * velocity[1], t * velocity[2]], axis=1)
        quat = np.zeros((len(t), 4), dtype=np.float32)
        quat[:, 3] = 1
        grip = np.zeros((len(t), 1), dtype=np.float32)
        step = np.arange(0, 8, dtype=np.int32)
        # Feature/proprio proximity tracks the episode's motion signature.
        feature = np.tile(velocity, (len(step), 1))
        np.savez_compressed(root / "episodes" / f"{eid}.npz", step=step,
                            proprio=feature, **{"feat__motion": feature},
                            env_eef_pos=pos, env_eef_quat=quat, env_gripper=grip)
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_split_keeps_init_state_groups_and_episodes_disjoint():
    episodes = [Episode("a", "same", {}), Episode("b", "same", {}),
                Episode("c", "other", {}), Episode("d", "third", {})]
    query, store = split_by_init_state(episodes, 0.34, 11)
    assert query.isdisjoint(store)
    query_init = {e.init_state_id for e in episodes if e.episode_id in query}
    store_init = {e.init_state_id for e in episodes if e.episode_id in store}
    assert query_init.isdisjoint(store_init)
    assert ("a" in query) == ("b" in query)


def test_split_rejects_single_init_state_to_prevent_leakage():
    with pytest.raises(ValueError, match="two distinct init_state_id"):
        split_by_init_state([Episode("a", "only", {}), Episode("b", "only", {})], 0.5, 0)


def test_task_local_init_state_ids_are_namespaced(tmp_path):
    _make_run(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for i, entry in enumerate(manifest["episodes"]):
        entry["init_state_id"] = 0
        entry["task_id"] = i
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _, episodes = load_run(tmp_path)
    assert len({episode.init_state_id for episode in episodes}) == len(episodes)


def test_motion_target_uses_future_relative_pose_and_gripper():
    arrays = {
        "env_eef_pos": np.array([[0, 0, 0], [1, 0, 0], [3, 0, 0]], dtype=float),
        "env_eef_quat": np.array([[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=float),
        "env_gripper": np.array([[0], [0.2], [0.5]], dtype=float),
    }
    target = _motion_target(arrays, 0, 2)
    np.testing.assert_allclose(target[:, 0], [1, 3])
    np.testing.assert_allclose(target[:, 3:7], [[0, 0, 0, 1], [0, 0, 1, 0]], atol=1e-6)
    np.testing.assert_allclose(target[:, 7], [0.2, 0.5])


def test_rotation_metric_compares_orientation_and_ignores_quaternion_sign():
    # Same angle but opposite direction must differ; q and -q are equivalent.
    plus_90 = np.array([[0, 0, np.sqrt(0.5), np.sqrt(0.5)]])
    minus_90 = np.array([[0, 0, -np.sqrt(0.5), np.sqrt(0.5)]])
    same_rotation_other_sign = -plus_90
    grip = np.zeros((1, 1))
    base = np.zeros((1, 3))
    a = np.concatenate([base, plus_90, grip], axis=1)
    b = np.concatenate([base, minus_90, grip], axis=1)
    c = np.concatenate([base, same_rotation_other_sign, grip], axis=1)
    assert _outcome_distance(a, b, 1)["eef_relative_rotation_rad"] == pytest.approx(np.pi)
    assert _outcome_distance(a, c, 1)["eef_relative_rotation_rad"] == pytest.approx(0)


def test_synthetic_run_computes_feature_proprio_random_and_oracle(tmp_path):
    _make_run(tmp_path)
    result = evaluate_run(tmp_path, EvaluationConfig(horizon=3, query_fraction=0.25, seed=4))
    assert "feat__motion" in result["features"]
    assert "proprio" in result["features"]
    assert "random" in result and "oracle_upper_bound" in result
    assert set(result["features"]["feat__motion"]["metrics"]) == {"1", "5"}
    assert result["features"]["feat__motion"]["n_queries"] > 0
    assert set(result["split"]["query_init_state_ids"]).isdisjoint(
        result["split"]["store_init_state_ids"])


def test_evaluator_fails_if_split_leaks_episode(monkeypatch, tmp_path):
    _make_run(tmp_path)
    import skill_moe.skillnet.examples.memory_pilot.g2_retrieval.evaluate as module

    monkeypatch.setattr(module, "split_by_init_state", lambda *_: ({"ep00"}, {"ep00", "ep01"}))
    with pytest.raises(AssertionError, match="episode leakage"):
        module.evaluate_run(tmp_path, EvaluationConfig(horizon=3))


def _rewrite_episode(root, eid, update):
    path = root / "episodes" / f"{eid}.npz"
    with np.load(path, allow_pickle=False) as z:
        arrays = {key: z[key] for key in z.files}
    update(arrays)
    np.savez_compressed(path, **arrays)


def test_initial_state_leak_is_rejected_independently_of_splitter(monkeypatch, tmp_path):
    _make_run(tmp_path)
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["episodes"][1]["init_state_id"] = manifest["episodes"][0]["init_state_id"]
    path.write_text(json.dumps(manifest))
    import skill_moe.skillnet.examples.memory_pilot.g2_retrieval.evaluate as module

    monkeypatch.setattr(module, "split_by_init_state", lambda *_: ({"ep00"}, {"ep01"}))
    with pytest.raises(AssertionError, match="initial-state leakage"):
        evaluate_run(tmp_path, EvaluationConfig(horizon=3))


def test_duplicate_episode_file_cannot_cross_split(tmp_path):
    _make_run(tmp_path)
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["episodes"].append({"episode_id": "alias", "init_state_id": "alias",
                                 "file": "episodes/ep00.npz"})
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="same episode file"):
        load_run(tmp_path)


@pytest.mark.parametrize("include_reset", [False, True])
def test_v1_recording_time_origin_and_last_complete_future(tmp_path, include_reset):
    _make_run(tmp_path)
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["format_version"] = 1
    for entry in manifest["episodes"]:
        entry["steps"] = 16
        def update(arrays):
            # Accelerating motion makes a one-step shift observable. At the
            # last query step (14), exactly two future steps remain.
            times = np.arange(0 if include_reset else 1, 17)
            arrays["env_eef_pos"] = np.column_stack((times ** 2, times * 0, times * 0))
            arrays["env_eef_quat"] = np.tile([0., 0., 0., 1.], (len(times), 1))
            arrays["env_gripper"] = np.zeros((len(times), 1))
            arrays["step"] = np.arange(8, 16, dtype=np.int32)
        _rewrite_episode(tmp_path, entry["episode_id"], update)
    path.write_text(json.dumps(manifest))
    _, episodes = load_run(tmp_path)
    episode = episodes[0]
    assert episode.env_step_offset == (0 if include_reset else 1)
    rows = _stack_rows(episodes, {episode.episode_id}, "proprio", 2)
    assert [r["step"] for r in rows] == list(range(8, 15))
    np.testing.assert_allclose(rows[0]["target"][:, 0], [17, 36])
    np.testing.assert_allclose(rows[-1]["target"][:, 0], [29, 60])


def test_explicit_time_origin_overrides_v1_inference(tmp_path):
    _make_run(tmp_path)
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest.update(format_version=1, env_step_offset=0)
    path.write_text(json.dumps(manifest))
    _, episodes = load_run(tmp_path)
    assert {e.env_step_offset for e in episodes} == {0}


def test_ambiguous_v1_time_origin_is_rejected(tmp_path):
    _make_run(tmp_path)
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["format_version"] = 1
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="steps metadata or explicit env_step_offset"):
        load_run(tmp_path)


@pytest.mark.parametrize("key", ["feat__motion", "proprio"])
def test_partial_feature_coverage_is_rejected(tmp_path, key):
    _make_run(tmp_path)
    _rewrite_episode(tmp_path, "ep00", lambda arrays: arrays.pop(key))
    with pytest.raises(ValueError, match="missing from one or more episodes"):
        evaluate_run(tmp_path, EvaluationConfig(horizon=3))


@pytest.mark.parametrize("key", ["feat__motion", "proprio", "env_eef_pos", "env_gripper"])
@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_nonfinite_data_cannot_produce_a_verdict(tmp_path, key, value):
    _make_run(tmp_path)
    def update(arrays):
        arrays[key][0, 0] = value
    _rewrite_episode(tmp_path, "ep00", update)
    with pytest.raises(ValueError, match="finite"):
        evaluate_run(tmp_path, EvaluationConfig(horizon=3))


def test_invalid_quaternion_cannot_be_treated_as_a_real_rotation(tmp_path):
    _make_run(tmp_path)
    def update(arrays):
        arrays["env_eef_quat"][0] = 0
    _rewrite_episode(tmp_path, "ep00", update)
    with pytest.raises(ValueError, match="quaternions must be finite and nonzero"):
        evaluate_run(tmp_path, EvaluationConfig(horizon=3))


@pytest.mark.parametrize("key", ["feat__motion", "proprio"])
def test_inconsistent_feature_dimensions_are_rejected(tmp_path, key):
    _make_run(tmp_path)
    def update(arrays):
        arrays[key] = arrays[key][:, :1]
    _rewrite_episode(tmp_path, "ep00", update)
    with pytest.raises(ValueError, match="dimensions differ"):
        evaluate_run(tmp_path, EvaluationConfig(horizon=3))


def test_fractional_steps_are_rejected_instead_of_truncated(tmp_path):
    _make_run(tmp_path)
    def update(arrays):
        arrays["step"] = arrays["step"].astype(float) + .5
    _rewrite_episode(tmp_path, "ep00", update)
    with pytest.raises(ValueError, match="step must be increasing nonnegative integers"):
        evaluate_run(tmp_path, EvaluationConfig(horizon=3))


def _row(eid, feature, position):
    return {"episode_id": eid, "init_state_id": eid, "step": 0,
            "feature": np.array([feature], dtype=float),
            "target": np.array([[position, 0., 0., 0., 0., 0., 1., 0.]])}


def test_knn_ties_and_random_expectation_do_not_depend_on_file_order():
    queries = [_row("query", 0, 0)]
    # One strictly closer neighbor, then a tie straddling k=2.
    store = [_row("a", 0, 0), _row("b", -1, 2), _row("c", 1, 10)]
    forward, _, _ = _evaluate_ranked(queries, store, _feature_order("x"), (1, 2, 5), 1)
    reverse, _, _ = _evaluate_ranked(queries, store[::-1], _feature_order("x"), (1, 2, 5), 1)
    assert forward == reverse
    assert forward["1"]["eef_relative_position"] == 0
    assert forward["2"]["eef_relative_position"] == pytest.approx(3)
    assert forward["5"]["eef_relative_position"] == pytest.approx(4)
    random, _, _ = _evaluate_ranked(queries, store, _random_order, (1, 2, 5), 1)
    assert [random[str(k)]["eef_relative_position"] for k in (1, 2, 5)] == [4, 4, 4]


def test_standardization_is_fitted_only_on_store():
    # Adding the outlying query to the fit would strongly shrink dimension 2.
    store = np.array([[0., 0.], [2., 2.]])
    np.testing.assert_allclose(_distances(np.array([1., 101.]), store),
                               [np.sqrt(1 + 101 ** 2), np.sqrt(1 + 99 ** 2)])


def _comparison_data(n_groups=12, repetitions=1, constant_effect=False):
    rows = [{"init_state_id": f"group-{group:02d}"}
            for group in range(n_groups) for _ in range(repetitions)]
    group = np.repeat(np.arange(n_groups, dtype=float), repetitions)
    proprio = np.zeros((len(rows), 1, 3))
    feature = np.zeros_like(proprio)
    # Shared query difficulty must cancel in a paired comparison.
    feature[:, 0, 0] = 1000 * group
    proprio[:, 0, 0] = feature[:, 0, 0] + (1 if constant_effect else 1 + group)
    return rows, {"proprio": proprio, "feat__motion": feature}


@pytest.mark.parametrize("n_groups", [1, 2, 9])
def test_many_overlapping_queries_cannot_replace_independent_init_states(n_groups):
    rows, scores = _comparison_data(n_groups, repetitions=200)
    result = _compare_to_proprio(rows, scores, EvaluationConfig(k_values=(1,)))
    value = result["comparisons"]["feat__motion"]["by_k"]["1"]["eef_relative_position"]
    assert result["n_query_init_states"] == n_groups
    assert result["decision"] == "inconclusive"
    assert result["bootstrap_samples"] == 0
    assert value["ci95_familywise"] is None
    assert value["unavailable_reason"] == "too_few_query_init_states"


def test_paired_cluster_intervals_ignore_duplication_of_overlapping_windows():
    config = EvaluationConfig(k_values=(1,))
    rows, scores = _comparison_data()
    original = _compare_to_proprio(rows, scores, config)
    rows, scores = _comparison_data(repetitions=20)
    duplicated = _compare_to_proprio(rows, scores, config)
    assert duplicated == original
    position = original["comparisons"]["feat__motion"]["by_k"]["1"]["eef_relative_position"]
    assert position["improvement"] == pytest.approx(6.5)
    assert 1 < position["ci95_familywise"][0] < 6.5 < position["ci95_familywise"][1] < 12
    assert original["decision"] == "clear_improvement"


def test_degenerate_nonzero_group_effect_cannot_claim_certainty():
    rows, scores = _comparison_data(constant_effect=True)
    result = _compare_to_proprio(rows, scores, EvaluationConfig(k_values=(1,)))
    position = result["comparisons"]["feat__motion"]["by_k"]["1"]["eef_relative_position"]
    assert position["improvement"] == 1
    assert position["ci95_familywise"] is None
    assert position["unavailable_reason"] == "no_between_group_variation"
    assert result["decision"] == "inconclusive"


def test_multiple_feature_comparisons_widen_intervals():
    rows, scores = _comparison_data()
    config = EvaluationConfig(k_values=(1,))
    single = _compare_to_proprio(rows, scores, config)
    scores["feat__other"] = scores["feat__motion"].copy()
    multiple = _compare_to_proprio(rows, scores, config)
    single_ci = single["comparisons"]["feat__motion"]["by_k"]["1"]["eef_relative_position"]["ci95_familywise"]
    multiple_ci = multiple["comparisons"]["feat__motion"]["by_k"]["1"]["eef_relative_position"]["ci95_familywise"]
    assert single["n_comparisons"] == 3
    assert multiple["n_comparisons"] == 6
    assert multiple_ci[0] <= single_ci[0]
    assert multiple_ci[1] >= single_ci[1]


def test_gain_in_position_cannot_hide_rotation_regression():
    rows, scores = _comparison_data()
    scores["feat__motion"][:, 0, 1] = np.arange(len(rows)) + 1
    result = _compare_to_proprio(rows, scores, EvaluationConfig(k_values=(1,)))
    assert result["decision"] == "inconclusive"


def test_real_retrieval_detects_signal_without_claiming_equal_features_are_better(tmp_path):
    _make_run(tmp_path, n_episodes=24)
    config = EvaluationConfig(horizon=3, query_fraction=.5, seed=4)
    equal = evaluate_run(tmp_path, config)
    assert equal["features"]["feat__motion"]["metrics"] == equal["features"]["proprio"]["metrics"]
    assert equal["uncertainty"]["decision"] == "inconclusive"
    for path in (tmp_path / "episodes").glob("*.npz"):
        def update(arrays):
            arrays["proprio"][:] = 0
        _rewrite_episode(tmp_path, path.stem, update)
    signal = evaluate_run(tmp_path, config)
    assert signal["uncertainty"]["decision"] == "clear_improvement"
    assert signal["features"]["feat__motion"]["n_queries"] == signal["features"]["proprio"]["n_queries"]
    # The constant-proprio baseline is an all-tied ranking, equal to random in expectation.
    assert signal["features"]["proprio"]["metrics"] == signal["random"]["metrics"]
    assert evaluate_run(tmp_path, config) == signal
    json.dumps(signal, allow_nan=False)
    report = tmp_path / "report.md"
    write_report(signal, report)
    text = report.read_text()
    assert "clear_improvement" in text
    assert "Bonferroni" in text
    assert "Conditional on this fixed store and split" in text
