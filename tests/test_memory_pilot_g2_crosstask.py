import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from skill_moe.skillnet.examples.memory_pilot.g2_retrieval import crosstask
from skill_moe.skillnet.examples.memory_pilot.g2_retrieval.crosstask import (
    AMBIGUOUS,
    CrossTaskConfig,
    condition_masks,
    evaluate_crosstask,
    gripper_events,
    subtask_labels,
    write_report,
)
from skill_moe.skillnet.examples.memory_pilot.g2_retrieval.evaluate import EvaluationConfig, evaluate_run


def _closed(pattern: str) -> np.ndarray:
    return np.array([c == "C" for c in pattern])


def _labels(plan, pattern, debounce=2):
    return "".join("?" if x == AMBIGUOUS else str(x) for x in subtask_labels(plan, _closed(pattern), debounce))


def test_pick_place_boundaries_follow_completed_grasp_and_release():
    # Close at 3 (complete at 5), open at 7 (complete at 9).
    assert _labels([0, 1], "oooCCCCoooo") == "00000111111"
    assert _labels([0, 1, 0, 1], "ooCCooCCoo") == "0000110011"


def test_debounce_ignores_single_step_flicker():
    assert gripper_events(_closed("ooCoooCCCoo"), 2) == [(6, "close"), (9, "open")]


def test_single_skill_plan_labels_whole_episode():
    assert _labels([2], "oooCCoo") == "2222222"


def test_leading_non_grasp_skill_uses_its_own_grasp_or_is_ambiguous():
    # Turn knob grasps once more than the remaining picks need.
    assert _labels([4, 0, 1], "oCCooooCCCoo") == "444440000111"
    # Drawer opening without a grasp: boundary with the pick is unobservable.
    assert _labels([3, 0, 1], "ooooooCCCoo") == "????????111"


def test_trailing_skill_after_place_and_extra_regrasp_is_ambiguous():
    assert _labels([0, 1, 3], "ooCCCooooo") == "0000111333"
    assert _labels([0, 1], "ooCCooCCCo") == "000011????"


def _make_run(root: Path, n_tasks=4, n_inits=3, task_signal=1.0, plan=(0, 1), same_motion=()):
    """Each task has its own velocity; pick moves +v, place moves -v. Features encode task and time.

    Tasks listed in ``same_motion`` share task 0's velocity (same motion, different task).
    """
    (root / "episodes").mkdir(parents=True)
    manifest = {"suite": "synthetic", "env_step_offset": 0, "episodes": []}
    total, close_at, stride = 24, 12, 4
    for t in range(n_tasks):
        u = 0 if t in same_motion else t
        velocity = np.array([u + 1, 0.5 * u, 1.0]) / 100
        for i in range(n_inits):
            eid = f"t{t}_i{i}"
            manifest["episodes"].append({"episode_id": eid, "task": f"task {t}", "task_id": t,
                                         "init_state_id": i, "suite": "synthetic"})
            sign = np.where(np.arange(total + 1) < close_at + 2, 1.0, -1.0)
            pos = np.cumsum(sign[:, None] * velocity[None], axis=0)
            quat = np.tile([0.0, 0.0, 0.0, 1.0], (total + 1, 1))
            grip = (np.arange(total + 1) >= close_at).astype(float)[:, None]
            step = np.arange(0, total, stride, dtype=np.int32)
            chunk = np.zeros((len(step), stride, 7))
            for r, s in enumerate(step):
                chunk[r, :, 6] = np.where(np.arange(s, s + stride) >= close_at, 1.0, -1.0)
            phase = (step >= close_at + 2).astype(float)
            feat = np.column_stack([task_signal * np.eye(n_tasks)[np.full(len(step), t)], step / total])
            proprio = np.column_stack([pos[step], phase])
            skills = np.tile(np.array(list(plan) + [-1] * (4 - len(plan))) + 1, (len(step), 1))
            mask = np.tile(np.arange(4) < len(plan), (len(step), 1))
            np.savez_compressed(root / "episodes" / f"{eid}.npz", step=step, proprio=proprio,
                                feat__task=feat, env_eef_pos=pos, env_eef_quat=quat, env_gripper=grip,
                                pred_chunk=chunk, n_executed=np.full(len(step), stride, dtype=np.int32),
                                skills=skills.astype(np.int32), skill_mask=mask)
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


CONFIG = CrossTaskConfig(base=EvaluationConfig(horizon=2, query_fraction=0.25, seed=1, bootstrap_samples=200),
                         recall_k_values=(1, 3), gripper_debounce_steps=2)


def test_constraints_are_enforced_end_to_end(tmp_path):
    _make_run(tmp_path)
    result = evaluate_crosstask(tmp_path, CONFIG)
    item1 = result["item1_constrained_retrieval"]
    # Same task (other initial state) has identical motion and features: perfect unconstrained retrieval.
    assert item1["a_unconstrained"]["methods"]["feat__task"]["1"]["eef_relative_position"] == pytest.approx(0)
    # Excluding the same task or the same subtask removes every zero-error candidate.
    for condition in ("b_other_task", "c_other_subtask", "d_other_task_same_subtask"):
        value = item1[condition]
        assert value["oracle_upper_bound"]["1"]["eef_relative_position"] > 1e-4
        assert value["methods"]["feat__task"]["1"]["eef_relative_position"] > 1e-4
    sources = result["item2_neighbor_sources"]["by_method"]["feat__task"]["1"]
    assert sources["same_task_same_subtask"] == pytest.approx(1)
    assert result["data_counts"]["query_windows_dropped_ambiguous_subtask"] == 0
    write_report(result, tmp_path / "report.md")
    assert "b_other_task" in (tmp_path / "report.md").read_text()


def test_condition_masks_exclude_leaks_and_violations():
    q = {"episode_id": np.array(["e0", "e1"]), "init_state_id": np.array(["g0", "g1"]),
         "task": np.array(["A", "B"]), "subtask": np.array([0, AMBIGUOUS])}
    s = {"episode_id": np.array(["e0", "e2", "e3", "e4", "e5"]),
         "init_state_id": np.array(["g0", "g1", "g2", "g3", "g4"]),
         "task": np.array(["A", "A", "A", "B", "B"]), "subtask": np.array([0, 0, 1, 0, AMBIGUOUS])}
    masks = condition_masks(q, s)
    for mask in masks.values():
        assert not mask[0, 0]  # same episode / initial state
        assert not mask[1, 1]  # same initial state
    a, b, c, d = (masks[k] for k in ("a_unconstrained", "b_other_task", "c_other_subtask",
                                     "d_other_task_same_subtask"))
    assert np.all(b <= a) and np.all(c <= a) and np.all(d <= b)
    assert not np.any(b & (q["task"][:, None] == s["task"][None]))
    assert not np.any(c & (q["subtask"][:, None] == s["subtask"][None]))
    assert not c[1].any() and not d[1].any()  # ambiguous query has no subtask conditions
    assert not c[:, 4].any() and not d[:, 4].any()  # ambiguous store rows are excluded
    assert c[0].tolist() == [False, False, True, False, False]
    assert d[0].tolist() == [False, False, False, True, False]


def test_split_leakage_is_rejected(monkeypatch, tmp_path):
    _make_run(tmp_path)
    monkeypatch.setattr(crosstask, "split_by_init_state", lambda *_: ({"t0_i0"}, {"t0_i0", "t1_i0"}))
    with pytest.raises(AssertionError, match="episode leakage"):
        evaluate_crosstask(tmp_path, CONFIG)
    monkeypatch.setattr(crosstask, "split_by_init_state", lambda *_: ({"t0_i0"}, {"t1_i0"}))
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["episodes"][3]["task_id"] = 0  # t1_i0 now shares the qualified initial-state ID of t0_i0
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(AssertionError, match="initial-state leakage"):
        evaluate_crosstask(tmp_path, CONFIG)


def test_split_groups_are_disjoint_and_queries_never_see_their_group(tmp_path):
    _make_run(tmp_path)
    result = evaluate_crosstask(tmp_path, CONFIG)
    assert set(result["split"]["query_init_state_ids"]).isdisjoint(result["split"]["store_init_state_ids"])
    assert set(result["split"]["query_episode_ids"]).isdisjoint(result["split"]["store_episode_ids"])


def test_unconstrained_condition_matches_base_g2_evaluator(tmp_path):
    _make_run(tmp_path, task_signal=0.3)
    result = evaluate_crosstask(tmp_path, CONFIG)
    base = evaluate_run(tmp_path, CONFIG.base)
    a = result["item1_constrained_retrieval"]["a_unconstrained"]
    for key in ("feat__task", "proprio"):
        for k, metrics in base["features"][key]["metrics"].items():
            for metric, value in metrics.items():
                assert a["methods"][key][k][metric] == pytest.approx(value, abs=1e-12)
    for metric, value in base["random"]["metrics"]["1"].items():
        assert a["random"][metric] == pytest.approx(value, abs=1e-12)
    for metric, by_k in base["oracle_upper_bound"].items():
        for k, value in by_k.items():
            assert a["oracle_upper_bound"][k][metric] == pytest.approx(value, abs=1e-12)


def test_cross_task_recall_counts_only_other_task_relevant_rows(tmp_path):
    _make_run(tmp_path, same_motion=(1,))
    result = evaluate_crosstask(tmp_path, CONFIG)
    thresholds = result["item3_cross_task_motion_recall"]["thresholds"]
    assert all(entry["queries_with_relevant"] > 0 for entry in thresholds.values())
    for entry in thresholds.values():
        for pool in entry["pools"].values():
            for by_k in pool.values():
                for v in by_k.values():
                    assert 0 <= v["recall"] <= 1 + 1e-12 and 0 <= v["precision"] <= 1 + 1e-12
        # The oracle ranking inside the other-task pool hits relevant rows first.
        oracle = entry["pools"]["other_task_pool"]["oracle_motion_combined"]["k=1"]
        assert oracle["precision"] == pytest.approx(1)


def test_plan_must_be_constant_within_episode(tmp_path):
    _make_run(tmp_path)
    path = tmp_path / "episodes" / "t0_i0.npz"
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    arrays["skills"][2, 0] = 3
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="change within an episode"):
        evaluate_crosstask(tmp_path, CONFIG)


def test_no_cross_task_relevant_rows_is_reported_not_nan(tmp_path):
    _make_run(tmp_path)  # every task has a unique motion; tau = 0 admits exact matches only
    result = evaluate_crosstask(tmp_path, replace(CONFIG, near_quantiles=(0.0,), near_oracle_k=1))
    for entry in result["item3_cross_task_motion_recall"]["thresholds"].values():
        assert entry["queries_with_relevant"] == 0 and entry["pools"] == {}


def test_same_task_share_counts_rows_with_unknown_subtask(monkeypatch, tmp_path):
    _make_run(tmp_path)
    # Pretend every store subtask is undetermined: task identity must still be counted.
    monkeypatch.setattr(crosstask, "_source_category",
                        lambda q, s: np.full((len(q["task"]), len(s["task"])), 4))
    item2 = evaluate_crosstask(tmp_path, CONFIG)["item2_neighbor_sources"]
    assert item2["same_task_share"]["feat__task"]["1"] == pytest.approx(1)
    excess = {(e["method"], e["k"], e["quantity"]): e["estimate"] for e in item2["excess_over_chance"]}
    chance = item2["same_task_share"]["chance (candidate share)"]
    assert excess[("feat__task", "1", "same_task_minus_chance")] == pytest.approx(1 - chance)
