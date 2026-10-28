import numpy as np
import torch

from skill_moe.skillnet.examples.emergent.light.extract import READ_COLUMNS, SUITE_BY_TASK
from skill_moe.skillnet.examples.emergent.light.model import BottleneckModel, gaussian_kl
from skill_moe.skillnet.examples.emergent.light.run import _prepare, _write_eval_input
from skill_moe.skillnet.examples.emergent.light.skill_bench import (
    _growth_state, _restore_growth_state, _seed_sequence_stage, _sequence_unit_events,
    _should_merge_sequence_stage,
)
from pathlib import Path
import json


def test_frozen_extractor_selects_only_unannotated_model_columns():
    assert "eval_skill_class" not in READ_COLUMNS
    assert "eval_skill_segment" not in READ_COLUMNS
    assert "eval_all_classes" not in READ_COLUMNS
    assert set(SUITE_BY_TASK) == set(range(10, 40))


def test_task_text_matches_corrected_goal_and_spatial_suite_mapping():
    tasks_path = Path("/work/.skillnet-state/data/lerobot/emergent/libero_40_unsegmented/meta/tasks.jsonl")
    if not tasks_path.exists():
        return
    tasks = {int(x["task_index"]): x["task"] for x in map(json.loads, tasks_path.read_text().splitlines())}
    assert tasks[16] == "turn on the stove"
    assert SUITE_BY_TASK[16] == "libero_goal"
    assert "pick up the black bowl" in tasks[34]
    assert SUITE_BY_TASK[34] == "libero_spatial"


def test_model_q_prior_decoder_and_analytic_kl_shapes():
    torch.manual_seed(0)
    model = BottleneckModel((10, 7), state_dim=8, feature_dim=64, z_dim=32)
    a, s, f = torch.randn(4, 70), torch.randn(4, 8), torch.randn(4, 64)
    pred, qm, qlv, pm, plv = model(a, s, f)
    assert pred.shape == (4, 10, 7)
    assert qm.shape == qlv.shape == pm.shape == plv.shape == (4, 32)
    assert torch.isfinite(gaussian_kl(qm, qlv, pm, plv))


def test_outer_split_keeps_each_episode_wholly_on_one_side():
    records = []
    for task_idx in range(3):
        for ep_idx in range(10):
            records.append({"episode_id": f"e{task_idx}-{ep_idx}", "task": f"task{task_idx}",
                            "cache_arrays": {"step": np.array([0, 5]), "f": np.zeros((2, 4)),
                                             "k": np.zeros((2, 4)), "proprio": np.zeros((2, 3))},
                            "windows": [np.zeros((5, 7)), np.zeros((5, 7))]})
    train, held, train_eps, held_eps = _prepare(records, 5)
    assert {x["episode_id"] for x in train_eps}.isdisjoint({x["episode_id"] for x in held_eps})
    assert len(held_eps) == 6
    assert len(train) == 48 and len(held) == 12


def test_eval_adapter_uses_task_proxy_without_annotation_source(tmp_path):
    ep = {"episode_id": "ep-1", "task": "task text", "task_index": 12, "suite": "libero_spatial",
          "cache_arrays": {"step": np.array([0]), "proprio": np.zeros((1, 8), np.float32),
                           "action": np.zeros((3, 7), np.float32), "env_state": np.zeros((3, 8), np.float32),
                           "env_step": np.arange(3), "k": np.zeros((1, 4), np.float32),
                           "f": np.zeros((1, 4), np.float32)}}
    row = {"ep": ep, "j": 0, "step": 0, "a": np.ones((2, 7), np.float32)}
    preds = {"prior_mean": np.zeros((1, 32), np.float32), "prior_logvar": np.zeros((1, 32), np.float32),
             "post_mean": np.zeros((1, 32), np.float32), "post_logvar": np.zeros((1, 32), np.float32),
             "untrained_prior_mean": np.zeros((1, 32), np.float32)}
    _write_eval_input(tmp_path / "run", [ep], [row], preds, 2, {"checkpoint": "test", "seed": 0,
                                                                "training_episodes": [], "beta": .01, "epochs": 1})
    with np.load(tmp_path / "run/episodes/ep-1.npz", allow_pickle=False) as data:
        assert data["skills"].tolist() == [[1, 2]]
        assert data["skill_mask"].tolist() == [[True, True]]
        assert data["n_executed"].tolist() == [2]


def test_sequential_training_rows_are_seeded_and_unit_events_stay_row_aligned():
    class FakeMethod:
        def __init__(self):
            self.memory = []

        def memory_add(self, key, base, meta):
            self.memory.append({"key": np.array(key), "base": np.array(base), "meta": dict(meta)})
            return True, "seed"

    table = {"step": np.array([0, 5]), "episode_id": np.array(["ep-a", "ep-b"]),
             "init_state_id": np.array(["init-a", "init-b"]), "task": np.array(["task", "task"]),
             "suite": np.array(["libero_spatial", "libero_spatial"]),
             "key": np.zeros((2, 3), np.float32), "base": np.zeros((2, 2), np.float32)}
    method = FakeMethod()
    accepted, _ = _seed_sequence_stage(method, table, np.array([0, 1]), "after_object+spatial")
    assert accepted.tolist() == [0, 1]
    assert all(entry["meta"]["seed"] is True for entry in method.memory)
    assert all(entry["meta"]["skill_unit_added"] is False for entry in method.memory)

    # Same step IDs in different episodes cannot collide; only the second row
    # is marked when its own persisted entry says that a unit was added.
    method.memory.append({"key": np.ones(3), "base": np.ones(2),
                          "meta": {"suite": "libero_spatial", "episode_id": "ep-b",
                                   "init_state_id": "init-b", "segment_id": "ep-b@5",
                                   "skill_unit_added": True}})
    flags, entries = _sequence_unit_events(method, table, 2)
    assert flags.tolist() == [False, True]
    assert sorted(entries) == [1]


def test_growth_snapshot_restores_train_only_unit_centers():
    class FakeMethod:
        pass

    method = FakeMethod()
    method._d3_unit_centers = np.array([[0.0, 1.0], [2.0, 3.0]], np.float32)
    train_only = _growth_state(method)
    method._d3_unit_centers = np.array([[99.0, 99.0]], np.float32)
    _restore_growth_state(method, train_only)
    np.testing.assert_array_equal(method._d3_unit_centers, [[0.0, 1.0], [2.0, 3.0]])


def test_merge_variant_does_not_merge_the_training_seed_stage():
    class MergeMethod:
        variant = "merge"

        def memory_merge(self):
            return {}

    method = MergeMethod()
    assert _should_merge_sequence_stage(method, 0) is False
    assert _should_merge_sequence_stage(method, 1) is True
