import json

import numpy as np
import pytest

from skill_moe.skillnet.examples.memory_pilot.memory import (
    MemoryMetadata,
    MemoryStore,
    namespace_init_state_id,
)


def _meta(eid="e1", *, init="i1", suite="suite", task="task", segment="s1", start=0, count=2,
          version="v1"):
    return MemoryMetadata(
        episode_id=eid,
        init_state_id=namespace_init_state_id(suite, task, init),
        task=task,
        segment_id=segment,
        start_step=start,
        end_step=start + count,
        n_executed=count,
        representation_version=version,
        suite=suite,
    )


def _store():
    store = MemoryStore(seed=5)
    ids = [
        store.append(_meta("same-episode", init="1", segment="a"), [0.0, 0.0], [1.0, 0.0]),
        store.append(_meta("other-episode", init="1", segment="b"), [0.5, 0.0], [0.0, 1.0]),
        store.append(_meta("eligible", init="2", segment="c"), [1.0, 0.0], [2.0, 2.0]),
    ]
    return store, ids


def test_rejects_unexecuted_or_inconsistent_segment():
    with pytest.raises(ValueError, match="n_executed must be a positive"):
        MemoryMetadata("ep", "state", "task", "seg", 2, 2, 0, "v1")
    with pytest.raises(ValueError, match="must equal"):
        MemoryMetadata("ep", "state", "task", "seg", 2, 5, 2, "v1")


def test_default_leakage_exclusions_and_exclusion_log():
    store, ids = _store()
    result = store.search([0.0, 0.0], mode="top1", episode_id="same-episode",
                          init_state_id=namespace_init_state_id("suite", "task", "1"))
    # Same episode and the other episode from the same initial state are both excluded.
    assert result.candidate_ids == (ids[2],)
    assert {item["reason"] for item in result.excluded} == {"same_episode", "same_init_state"}
    assert store.entries[2].usage_count == 1


def test_topk_weighting_returns_composite_and_individual_values():
    store, ids = _store()
    result = store.search([0.0, 0.0], mode="topk_uniform", k=2,
                          exclude_same_episode=False, exclude_same_init_state=False)
    assert result.candidate_ids == (ids[0], ids[1])
    np.testing.assert_allclose(result.weights, [0.5, 0.5])
    np.testing.assert_allclose(result.composite_value, [0.5, 0.5])
    np.testing.assert_allclose(result.z_collection, [[1, 0], [0, 1]])

    soft = store.search([0.0, 0.0], mode="topk_softmax", k=2, temperature=0.1,
                        exclude_same_episode=False, exclude_same_init_state=False)
    assert soft.weights[0] > soft.weights[1]
    np.testing.assert_allclose(soft.weights.sum(), 1.0)


def test_all_search_modes_and_empty_none_result():
    store, ids = _store()
    random_result = store.search([1.0, 0.0], mode="random", episode_id="same-episode",
                                 init_state_id=namespace_init_state_id("suite", "task", "1"))
    assert random_result.candidate_ids == (ids[2],)
    query = {"episode_id": "query", "init_state_id": namespace_init_state_id("suite", "task", "query")}
    assert len(store.search([0.0, 0.0], mode="top1", **query).candidate_ids) == 1
    assert len(store.search([0.0, 0.0], mode="topk_uniform", k=20, **query).candidate_ids) == 3
    none = store.search([0.0, 0.0], mode="none")
    assert none.candidate_ids == ()
    assert none.composite_value is None
    assert none.candidate_values.size == 0
    with pytest.raises(ValueError, match="unknown retrieval mode"):
        store.search([0.0, 0.0], mode="bad")


def test_append_save_load_usage_and_search_log(tmp_path):
    log_path = tmp_path / "searches.jsonl"
    store, ids = _store()
    query = {"episode_id": "query", "init_state_id": namespace_init_state_id("suite", "task", "query")}
    store.search([0.0, 0.0], mode="top1", **query)
    assert store.search_log[-1]["candidate_ids"]
    assert "excluded" in store.search_log[-1]
    store.search_log_path = log_path
    store.search([0.0, 0.0], mode="top1", **query)
    assert json.loads(log_path.read_text().splitlines()[0])["mode"] == "top1"

    path = tmp_path / "memory.npz"
    store.save(path)
    with pytest.raises(FileExistsError):
        store.save(path)
    restored = MemoryStore.load(path)
    assert len(restored) == len(store)
    assert restored.entries[0].metadata.entry_id == ids[0]
    assert restored.entries[0].metadata.representation_version == "v1"
    assert restored.entries[0].metadata.init_state_id == namespace_init_state_id("suite", "task", "1")
    np.testing.assert_array_equal(restored.entries[0].key, store.entries[0].key)
    np.testing.assert_array_equal(restored.entries[0].value, store.entries[0].value)
    assert restored.entries[0].usage_count == store.entries[0].usage_count
    assert len(restored.search_log) == len(store.search_log)
    restored.append(_meta("extra", init="3"), [9, 9], [4, 4])
    assert len(restored) == 4


def test_builds_from_recording_skips_zero_execution_and_allows_same_feature(tmp_path):
    (tmp_path / "episodes").mkdir()
    manifest = {
        "format_version": 1,
        "suite": "libero_spatial",
        "episodes": [{"episode_id": "ep", "task_id": 3, "task": "move cube",
                      "init_state_id": 0, "steps": 7}],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    np.savez(tmp_path / "episodes" / "ep.npz",
             step=np.array([1, 3, 5]), n_executed=np.array([2, 0, 2]), segment_id=np.array([10, 11, 12]),
             **{"feat__same": np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)})
    with pytest.raises(ValueError, match="refusing unexecuted segment"):
        MemoryStore.from_recording(tmp_path, key_feature="feat__same", value_feature="feat__same",
                                   representation_version="same-v1")
    np.savez(tmp_path / "episodes" / "ep.npz",
             step=np.array([1, 5]), n_executed=np.array([2, 2]), segment_id=np.array([10, 12]),
             **{"feat__same": np.array([[1, 2], [5, 6]], dtype=np.float32)})
    store = MemoryStore.from_recording(tmp_path, key_feature="feat__same", value_feature="feat__same",
                                       representation_version="same-v1")
    assert len(store) == 2
    entry = store.entries[0]
    assert entry.metadata.segment_id == "10"
    assert entry.metadata.start_step == 1 and entry.metadata.end_step == 3
    assert entry.metadata.n_executed == 2
    assert entry.metadata.init_state_id == namespace_init_state_id("libero_spatial", 3, 0)
    np.testing.assert_array_equal(entry.key, entry.value)


@pytest.mark.parametrize("task", [0, "task", "name::with::separator"])
def test_namespaced_init_id_matches_actual_g2_loader(tmp_path, task):
    from skill_moe.skillnet.examples.memory_pilot.g2_retrieval.evaluate import load_run

    manifest = {"suite": "suite", "episodes": [{"episode_id": "ep", "task_id": task, "init_state_id": 4}]}
    _write_run_with_episode(tmp_path, manifest,
        {"step": np.array([0]), "n_executed": np.array([1]), "segment_id": np.array([0]),
         "feat__key": np.ones((1, 1))})
    _, episodes = load_run(tmp_path)
    assert namespace_init_state_id("suite", task, 4) == episodes[0].init_state_id
    store = MemoryStore.from_recording(tmp_path, key_feature="feat__key", value_feature="feat__key",
                                       representation_version="v1")
    result = store.search([1.], episode_id="other-episode", init_state_id=episodes[0].init_state_id)
    assert result.candidate_ids == ()
    assert result.excluded[0]["reason"] == "same_init_state"


def test_distance_standardization_uses_only_stored_candidate_keys():
    store = MemoryStore()
    a_id = store.append(_meta("a", init="a"), [0, 0], [1])
    b_id = store.append(_meta("b", init="b"), [2, 10], [2])
    result = store.search([1000, 5], mode="topk_uniform", k=2, exclude_same_episode=False,
                          exclude_same_init_state=False)
    # Store-only means [1, 5], stds [1, 5]. A query outlier is transformed with
    # those store statistics; it does not move the normalization center/scale.
    np.testing.assert_allclose(result.distances, [np.sqrt(998**2 + 1**2),
                                                  np.sqrt(1000**2 + 1**2)])
    assert result.candidate_ids == (b_id, a_id)


def test_from_recording_refuses_untrimmed_prediction_chunk_as_value(tmp_path):
    (tmp_path / "episodes").mkdir()
    (tmp_path / "manifest.json").write_text(json.dumps({"episodes": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="pred_chunk includes unexecuted predictions"):
        MemoryStore.from_recording(tmp_path, key_feature="feat__x", value_feature="pred_chunk",
                                   representation_version="v1")


def _write_run_with_episode(tmp_path, manifest, arrays):
    (tmp_path / "episodes").mkdir(exist_ok=True)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    np.savez(tmp_path / "episodes" / "ep.npz", **arrays)


def test_pred_chunk_value_uses_only_executed_prefix_and_resamples(tmp_path):
    _write_run_with_episode(tmp_path,
        {"format_version": 1, "suite": "suite", "replan_steps": 3,
         "episodes": [{"episode_id": "ep", "task_id": 2, "task": "move", "init_state_id": 1}]},
        {"step": np.array([2]), "n_executed": np.array([2]), "segment_id": np.array([7]),
         "feat__key": np.array([[1., 2.]], dtype=np.float32),
         "pred_chunk": np.array([[[10., 20.], [30., 40.], [999., 999.]]], dtype=np.float32)})
    store = MemoryStore.from_recording(tmp_path, key_feature="feat__key", value_source="pred_chunk_executed",
                                       representation_version="action-v1")
    np.testing.assert_allclose(store.entries[0].value, [[10, 20], [20, 30], [30, 40]])
    assert store.entries[0].metadata.value_source == "pred_chunk_executed"
    assert store.entries[0].metadata.value_length == 3


def test_environment_value_recovers_executed_relative_trajectory(tmp_path):
    identity = np.array([0, 0, 0, 1], dtype=np.float32)
    quat = np.array([identity, identity,
                     [0, 0, np.sqrt(0.5), np.sqrt(0.5)],
                     [0, 0, 1, 0]], dtype=np.float32)
    _write_run_with_episode(tmp_path,
        {"format_version": 1, "suite": "suite", "replan_steps": 2,
         # len(env_*) = steps gives G2's v1 offset of 1.
         "episodes": [{"episode_id": "ep", "task_id": 2, "task": "move", "init_state_id": 1,
                       "steps": 4}]},
        {"step": np.array([2]), "n_executed": np.array([2]), "segment_id": np.array([7]),
         "feat__key": np.array([[1., 2.]], dtype=np.float32),
         "env_eef_pos": np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [4, 0, 0]], dtype=np.float32),
         "env_eef_quat": quat,
         "env_gripper": np.array([[0.0], [0.1], [0.4], [0.8]], dtype=np.float32)})
    store = MemoryStore.from_recording(tmp_path, key_feature="feat__key", value_source="env_trajectory",
                                       representation_version="trajectory-v1")
    # Per point: world-frame position delta (3), start-relative rotation vector (3),
    # gripper delta (1). The row at env index 0 is before the policy call's start.
    np.testing.assert_allclose(store.entries[0].value[:, 0], [1, 3])
    np.testing.assert_allclose(store.entries[0].value[:, 5], [np.pi / 2, np.pi], atol=1e-6)
    np.testing.assert_allclose(store.entries[0].value[:, 6], [0.3, 0.7])
    assert store.entries[0].metadata.value_source == "env_trajectory"


def test_environment_value_requires_time_origin_and_complete_executed_steps(tmp_path):
    manifest = {"suite": "suite", "replan_steps": 1,
                "episodes": [{"episode_id": "ep", "task": "move", "init_state_id": 1}]}
    arrays = {"step": np.array([0]), "n_executed": np.array([1]), "segment_id": np.array([7]),
              "feat__key": np.array([[1.]], dtype=np.float32),
              "env_eef_pos": np.zeros((1, 3)), "env_eef_quat": np.array([[0, 0, 0, 1.]]),
              "env_gripper": np.zeros((1, 1))}
    _write_run_with_episode(tmp_path, manifest, arrays)
    with pytest.raises(ValueError, match="env_trajectory requires"):
        MemoryStore.from_recording(tmp_path, key_feature="feat__key", value_source="env_trajectory",
                                   representation_version="trajectory-v1")

    manifest["env_step_offset"] = 0
    _write_run_with_episode(tmp_path, manifest, arrays)
    with pytest.raises(ValueError, match="do not contain the segment start"):
        MemoryStore.from_recording(tmp_path, key_feature="feat__key", value_source="env_trajectory",
                                   representation_version="trajectory-v1")


@pytest.mark.parametrize("mode", ["random", "random_k", "top1", "topk_uniform", "topk_softmax"])
@pytest.mark.parametrize("missing", ["episode_id", "init_state_id"])
def test_missing_query_metadata_cannot_silently_disable_exclusion(mode, missing):
    store, _ = _store()
    query = {"episode_id": "query", "init_state_id": namespace_init_state_id("suite", "task", "query")}
    query.pop(missing)
    with pytest.raises(ValueError, match=f"{missing} is required"):
        store.search([0., 0.], mode=mode, **query)


def test_bare_initial_state_ids_are_rejected_and_json_spacing_is_canonicalized():
    store, ids = _store()
    with pytest.raises(ValueError, match="namespace_init_state_id"):
        store.search([0., 0.], episode_id="same-episode", init_state_id="1")
    result = store.search([0., 0.], episode_id="same-episode", init_state_id='["suite","task","1"]')
    assert result.candidate_ids == (ids[2],)
    with pytest.raises(ValueError, match="namespace_init_state_id"):
        store.search([0., 0.], episode_id="same-episode", init_state_id='["suite","task","1"]',
                     exclude_init_state_ids=["2"])


def test_save_refuses_overwrite_even_if_the_file_appears_after_exists_check(tmp_path, monkeypatch):
    from pathlib import Path

    store, _ = _store()
    path = tmp_path / "memory.npz"
    original = b"existing memory must be preserved"
    path.write_bytes(original)
    with monkeypatch.context() as patch:
        real_exists = Path.exists
        patch.setattr(Path, "exists", lambda value: False if value == path else real_exists(value))
        with pytest.raises(FileExistsError):
            store.save(path)
    assert path.read_bytes() == original


def test_all_modes_share_exclusions_and_heldout_keys_do_not_affect_scaling():
    store = MemoryStore(seed=3)
    a = store.append(_meta("a", init="a"), [0, 0], [1])
    b = store.append(_meta("b", init="b"), [2, 10], [2])
    excluded_ids = {
        store.append(_meta("query", init="different", start=100), [1e7, 1e7], [99]),
        store.append(_meta("same-state", init="query"), [-1e7, 1e7], [99]),
        store.append(_meta("heldout-episode", init="heldout-episode"), [1e7, -1e7], [99]),
        store.append(_meta("heldout-state", init="heldout-state"), [-1e7, -1e7], [99]),
        store.append(_meta("other-version", init="other-version", version="v2"), [1e7, 0], [99]),
    }
    query = {"episode_id": "query", "init_state_id": namespace_init_state_id("suite", "task", "query"),
             "exclude_episode_ids": ["heldout-episode"],
             "exclude_init_state_ids": [namespace_init_state_id("suite", "task", "heldout-state")],
             "representation_version": "v1"}
    expected = {a: np.sqrt(1000**2 + 1), b: np.sqrt(998**2 + 1)}
    for mode in ("random", "random_k", "top1", "topk_uniform", "topk_softmax"):
        result = store.search([1000, 5], mode=mode, k=20, **query)
        assert {e["candidate_id"] for e in result.excluded} == excluded_ids
        assert set(result.candidate_ids) <= {a, b}
        for eid, distance in zip(result.candidate_ids, result.distances):
            assert distance == pytest.approx(expected[eid])
    random_ids = {store.search([1000, 5], mode="random", **query).candidate_ids[0] for _ in range(50)}
    assert random_ids == {a, b}


def test_random_and_top1_use_the_same_eligible_candidates_and_exclusions():
    store, _ = _store()
    query = {"episode_id": "query", "init_state_id": namespace_init_state_id("suite", "task", "query"),
             "exclude_episode_ids": ["same-episode"],
             "exclude_init_state_ids": [namespace_init_state_id("suite", "task", "2")]}
    top1 = store.search([0.2, 0.0], mode="top1", **query)
    random_one = store.search([0.2, 0.0], mode="random", **query)
    assert top1.eligible_candidate_ids == random_one.eligible_candidate_ids == (store.entries[1].metadata.entry_id,)
    assert top1.excluded == random_one.excluded
    assert len(top1.candidate_ids) == len(random_one.candidate_ids) == 1


@pytest.mark.parametrize("weighting", ["uniform", "topk_softmax"])
def test_random_k_matches_topk_candidate_pool_and_exclusions(weighting):
    store, _ = _store()
    store.append(_meta("heldout", init="heldout"), [100., 100.], [99., 99.])
    query = {"episode_id": "query", "init_state_id": namespace_init_state_id("suite", "task", "query"),
             "exclude_episode_ids": ["heldout"],
             "exclude_init_state_ids": [namespace_init_state_id("suite", "task", "missing-state")]}
    softmax = store.search([0.2, 0.0], mode="topk_softmax", k=2, temperature=0.7, **query)
    uniform = store.search([0.2, 0.0], mode="topk_uniform", k=2, **query)
    random_k = store.search([0.2, 0.0], mode="random_k", k=2, temperature=0.7,
                            random_k_weighting=weighting, **query)
    assert random_k.eligible_candidate_ids == softmax.eligible_candidate_ids == uniform.eligible_candidate_ids
    assert random_k.excluded == softmax.excluded == uniform.excluded
    assert len(random_k.candidate_ids) == len(softmax.candidate_ids) == len(uniform.candidate_ids) == 2
    assert len(set(random_k.candidate_ids)) == 2
    if weighting == "uniform":
        np.testing.assert_allclose(random_k.weights, [0.5, 0.5])
    else:
        # The nearest top-k weights are reused, but assigned in random-candidate order.
        np.testing.assert_allclose(np.sort(random_k.weights), np.sort(softmax.weights))


def test_random_k_samples_without_replacement_and_respects_k():
    store, _ = _store()
    query = {"episode_id": "query", "init_state_id": namespace_init_state_id("suite", "task", "query")}
    result = store.search([0., 0.], mode="random_k", k=20, **query)
    assert len(result.candidate_ids) == len(result.eligible_candidate_ids) == 3
    assert len(set(result.candidate_ids)) == 3
    assert result.weights.shape == (3,)
    with pytest.raises(ValueError, match="random_k_weighting"):
        store.search([0., 0.], mode="random_k", random_k_weighting="distance", **query)


def test_equal_keys_do_not_always_prefer_first_stored_value():
    store = MemoryStore(seed=11)
    ids = [store.append(_meta(f"e{i}", init=str(i)), [0.], [i]) for i in range(4)]
    query = {"episode_id": "query", "init_state_id": namespace_init_state_id("suite", "task", "query")}
    for mode in ("top1", "topk_uniform", "topk_softmax"):
        selected = {store.search([0.], mode=mode, k=1, **query).candidate_ids[0] for _ in range(100)}
        assert selected == set(ids)


def test_random_draws_are_not_changed_by_interleaved_knn_ties():
    stores = [MemoryStore(seed=7), MemoryStore(seed=7)]
    for store in stores:
        for i in range(4):
            store.append(_meta(f"e{i}", init=str(i)), [0.], [i])
    query = {"episode_id": "query", "init_state_id": namespace_init_state_id("suite", "task", "query")}
    for _ in range(20):
        stores[1].search([0.], mode="topk_uniform", k=2, **query)
        a, b = [store.search([0.], mode="random", **query) for store in stores]
        np.testing.assert_array_equal(a.composite_value, b.composite_value)


@pytest.mark.parametrize("source", ["feature", "pred_chunk_executed", "env_trajectory"])
def test_executed_windows_cannot_overlap_the_next_inference(tmp_path, source):
    _write_run_with_episode(tmp_path,
        {"env_step_offset": 0, "replan_steps": 3,
         "episodes": [{"episode_id": "ep", "task": "move", "init_state_id": 1, "steps": 6}]},
        {"step": np.array([0, 2]), "n_executed": np.array([3, 2]), "segment_id": np.array([0, 1]),
         "feat__key": np.ones((2, 1)), "pred_chunk": np.zeros((2, 4, 1)),
         "env_eef_pos": np.zeros((7, 3)), "env_eef_quat": np.tile([0., 0., 0., 1.], (7, 1)),
         "env_gripper": np.zeros((7, 1))})
    kwargs = {"value_feature": "feat__key"} if source == "feature" else {}
    with pytest.raises(ValueError, match="overlaps the next inference"):
        MemoryStore.from_recording(tmp_path, key_feature="feat__key", value_source=source,
                                   representation_version="v1", **kwargs)


def test_execution_count_cannot_exceed_episode_end(tmp_path):
    _write_run_with_episode(tmp_path,
        {"episodes": [{"episode_id": "ep", "task": "move", "init_state_id": 1, "steps": 2}]},
        {"step": np.array([1]), "n_executed": np.array([2]), "segment_id": np.array([0]),
         "feat__key": np.ones((1, 1)), "pred_chunk": np.zeros((1, 4, 1))})
    with pytest.raises(ValueError, match="exceeds episode steps"):
        MemoryStore.from_recording(tmp_path, key_feature="feat__key", value_source="pred_chunk_executed",
                                   representation_version="v1")


def test_predicted_tails_and_neighboring_chunks_do_not_enter_values(tmp_path):
    _write_run_with_episode(tmp_path,
        {"replan_steps": 3, "episodes": [{"episode_id": "ep", "task": "move", "init_state_id": 1}]},
        {"step": np.array([0, 2]), "n_executed": np.array([2, 2]), "segment_id": np.array([0, 1]),
         "feat__key": np.ones((2, 1)),
         "pred_chunk": np.array([[[0.], [2.], [np.nan]], [[100.], [102.], [np.inf]]])})
    store = MemoryStore.from_recording(tmp_path, key_feature="feat__key", value_source="pred_chunk_executed",
                                       representation_version="v1")
    np.testing.assert_allclose(store.entries[0].value[:, 0], [0, 1, 2])
    np.testing.assert_allclose(store.entries[1].value[:, 0], [100, 101, 102])


def test_fallback_length_is_global_and_original_samples_are_resampled_once(tmp_path):
    (tmp_path / "episodes").mkdir()
    manifest = {"episodes": []}
    for eid, values in (("a", [0., 100., 0.]), ("b", [1., 2., 3., 4., 5.])):
        manifest["episodes"].append({"episode_id": eid, "init_state_id": eid, "task": "move"})
        np.savez(tmp_path / "episodes" / f"{eid}.npz", step=np.array([0]),
                 n_executed=np.array([len(values)]), segment_id=np.array([0]),
                 **{"feat__key": np.ones((1, 1)), "pred_chunk": np.array(values)[None, :, None]})
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    store = MemoryStore.from_recording(tmp_path, key_feature="feat__key", value_source="pred_chunk_executed",
                                       representation_version="v1")
    assert [e.metadata.value_length for e in store.entries] == [5, 5]
    np.testing.assert_allclose(store.entries[0].value[:, 0], [0, 50, 100, 50, 0])
    result = store.search([1.], mode="topk_uniform", k=2, episode_id="query",
                          init_state_id=namespace_init_state_id("", "move", "query"))
    assert result.candidate_values.shape == (2, 5, 1)


def test_rotation_resampling_crosses_pi_without_inventing_zero_rotation(tmp_path):
    angles = np.deg2rad([0., 170., 190.])
    quats = np.column_stack((angles * 0, angles * 0, np.sin(angles / 2), np.cos(angles / 2)))
    _write_run_with_episode(tmp_path,
        {"env_step_offset": 0, "replan_steps": 3,
         "episodes": [{"episode_id": "ep", "task": "move", "init_state_id": 1}]},
        {"step": np.array([0]), "n_executed": np.array([2]), "segment_id": np.array([0]),
         "feat__key": np.ones((1, 1)), "env_eef_pos": np.zeros((3, 3)),
         "env_eef_quat": quats, "env_gripper": np.zeros((3, 1))})
    store = MemoryStore.from_recording(tmp_path, key_feature="feat__key", value_source="env_trajectory",
                                       representation_version="v1")
    np.testing.assert_allclose(np.abs(store.entries[0].value[:, 5]), np.deg2rad([170, 180, 170]), atol=1e-6)


def test_prediction_sequence_cannot_bypass_executed_extraction_as_feature(tmp_path):
    _write_run_with_episode(tmp_path,
        {"episodes": [{"episode_id": "ep", "task": "move", "init_state_id": 1}]},
        {"step": np.array([0]), "n_executed": np.array([1]), "segment_id": np.array([0]),
         "feat__key": np.ones((1, 1)), "feat__untrimmed_sequence": np.ones((1, 3, 2))})
    with pytest.raises(ValueError, match="features must have shape"):
        MemoryStore.from_recording(tmp_path, key_feature="feat__key", value_feature="feat__untrimmed_sequence",
                                   representation_version="v1")


def test_alias_episode_cannot_reintroduce_a_query_recording(tmp_path):
    _write_run_with_episode(tmp_path,
        {"episodes": [{"episode_id": "ep", "init_state_id": "first"},
                      {"episode_id": "alias", "init_state_id": "second", "file": "episodes/ep.npz"}]},
        {"step": np.array([0]), "n_executed": np.array([1]), "segment_id": np.array([0]),
         "feat__key": np.ones((1, 1))})
    with pytest.raises(ValueError, match="same episode file"):
        MemoryStore.from_recording(tmp_path, key_feature="feat__key", value_feature="feat__key",
                                   representation_version="v1")


def test_legacy_store_is_rejected_instead_of_silently_mismatching_initial_state_ids(tmp_path):
    path = tmp_path / "legacy.npz"
    np.savez(path, metadata_json=np.asarray(json.dumps({"format_version": 1})))
    with pytest.raises(ValueError, match="unsupported memory format version"):
        MemoryStore.load(path)
