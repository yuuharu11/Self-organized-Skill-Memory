import json

import numpy as np
import pytest
import torch

from skill_moe.skillnet.examples.memory_pilot.primitives.data import (
    PrimitiveDataset,
    load_recording,
    make_synthetic_dataset,
    split_by_episode,
)
from skill_moe.skillnet.examples.memory_pilot.primitives.experiment import (
    ExperimentConfig,
    mutual_information_bits,
    run_experiment,
    run_grid,
    write_report,
)
from skill_moe.skillnet.examples.memory_pilot.primitives.model import PrimitiveMoE


def _make_small_recording(run_dir):
    (run_dir / "episodes").mkdir()
    manifest = {"format_version": 1, "suite": "synthetic", "replan_steps": 2, "episodes": []}
    for episode_index in range(4):
        episode_id = f"ep-{episode_index}"
        task = f"task-{episode_index % 2}"
        manifest["episodes"].append({"episode_id": episode_id, "task": task, "task_id": episode_index % 2,
                                     "init_state_id": episode_index, "steps": 7})
        identity = np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (7, 1))
        positions = np.zeros((7, 3), dtype=np.float32)
        positions[:, 0] = np.arange(7, dtype=np.float32) * (episode_index + 1) / 10
        action = np.zeros((2, 3, 2), dtype=np.float32)
        action[0, :2, 0] = [episode_index, episode_index + 1]
        action[0, 2] = 999  # predicted-only tail must not enter dataset
        action[1, :, 1] = [1, 2, 3]
        np.savez(run_dir / "episodes" / f"{episode_id}.npz",
                 step=np.array([1, 4]), proprio=np.array([[episode_index, 0.], [0., episode_index]], np.float32),
                 pred_chunk=action, n_executed=np.array([2, 2]), segment_id=np.array([0, 1]),
                 env_eef_pos=positions, env_eef_quat=identity,
                 env_gripper=np.zeros((7, 1), dtype=np.float32), skills=np.full((2, 1), 999))
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_recording_loader_uses_executed_segments_without_task_or_skill_inputs(tmp_path):
    _make_small_recording(tmp_path)
    dataset = load_recording(tmp_path, horizon=4)
    assert dataset.states.shape == (8, 4, 2)
    assert dataset.actions.shape == (8, 4, 2)
    assert dataset.motion.shape == (8, 4, 7)
    assert dataset.init_state_ids[0] == json.dumps(["synthetic", "0", "0"])
    np.testing.assert_array_equal(dataset.states[0], np.repeat([[0.0, 0.0]], 4, axis=0))
    np.testing.assert_allclose(dataset.actions[0, :, 0], [0, 1 / 3, 2 / 3, 1])
    assert not np.any(dataset.actions == 999)
    assert set(dataset.task_labels) == {json.dumps(["synthetic", str(i)]) for i in (0, 1)}


def test_episode_split_keeps_all_windows_together():
    dataset = make_synthetic_dataset(n_episodes=8, segments_per_episode=3, horizon=4, seed=3)
    train, evaluation = split_by_episode(dataset, eval_fraction=0.25, seed=12)
    assert set(dataset.episode_ids[train]).isdisjoint(set(dataset.episode_ids[evaluation]))
    assert len(train) + len(evaluation) == dataset.n_segments


def test_split_keeps_repeated_initial_state_groups_together():
    dataset = make_synthetic_dataset(n_episodes=8, segments_per_episode=2, horizon=4, seed=3)
    init_ids = dataset.init_state_ids.copy()
    # Episodes 0 and 3 are separate rollouts from the same task-local initial state.
    init_ids[dataset.episode_ids == "episode-3"] = init_ids[0]
    dataset = PrimitiveDataset(dataset.states, dataset.actions, dataset.motion,
                               dataset.episode_ids, init_ids, dataset.task_labels,
                               dataset.segment_ids)
    train, evaluation = split_by_episode(dataset, eval_fraction=0.25, seed=12)
    assert set(init_ids[train]).isdisjoint(set(init_ids[evaluation]))
    assert set(dataset.episode_ids[train]).isdisjoint(set(dataset.episode_ids[evaluation]))


def test_loader_rejects_overlapping_executed_windows(tmp_path):
    _make_small_recording(tmp_path)
    path = tmp_path / "episodes" / "ep-0.npz"
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    arrays["step"] = np.array([1, 2], dtype=np.int32)
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="executed segments overlap"):
        load_recording(tmp_path, horizon=4)


def test_router_is_constant_per_segment_for_top1_and_soft():
    for routing in ("top1", "soft"):
        model = PrimitiveMoE(horizon=5, state_dim=3, action_dim=2, motion_dim=7,
                             n_experts=4, shared_expert=True, routing=routing)
        states = torch.zeros((6, 5, 3))
        actions = torch.zeros((6, 5, 2))
        motion = torch.zeros((6, 5, 7))
        result = model(states, actions, motion, sample_latent=False)
        assert result["weights"].shape == (6, 4)  # one route vector per segment, not per phase
        assert result["prediction"].shape == (6, 5, 2)
        torch.testing.assert_close(result["weights"].sum(dim=-1), torch.ones(6))


def test_expert_ablation_reroutes_after_removal():
    model = PrimitiveMoE(horizon=2, state_dim=2, action_dim=1, motion_dim=7,
                         n_experts=3, shared_expert=False, routing="top1")
    probabilities = torch.tensor([[0.7, 0.2, 0.1]])
    weights = model.route_weights(probabilities, ablate_expert=0)
    torch.testing.assert_close(weights, torch.tensor([[0.0, 1.0, 0.0]]))


def test_smoke_training_reports_mi_z_checks_and_expert_ablation(tmp_path):
    dataset = make_synthetic_dataset(n_episodes=6, segments_per_episode=3, horizon=5, seed=4)
    config = ExperimentConfig(n_experts=2, shared_expert=True, routing="top1", seed=2,
                              hidden_dim=16, latent_dim=3, epochs=2, batch_size=8,
                              beta_kl=1e-3, load_balance_weight=0.02)
    result = run_experiment(dataset, config, eval_fraction=1 / 3, split_seed=7)
    assert result["n_train_segments"] > 0 and result["n_eval_segments"] > 0
    assert set(result["eval_episode_ids"]).isdisjoint(result["train_episode_ids"])
    assert result["loss"]["beta_kl"] == 1e-3
    assert result["loss"]["load_balance_weight"] == 0.02
    assert result["eval_mean_kl_nats"] >= 0
    assert result["eval_mean_kl_per_latent_dim_nats"] >= 0
    assert result["mi_n_samples"] == result["n_eval_segments"]
    assert result["mi_null_permutations"] == config.mi_permutations
    assert set(result["mutual_information_bits"]) == {"direction", "rotation", "gripper", "task"}
    assert "shuffle_null_mean_bits" in result["mutual_information_bits"]["task"]
    assert "raw_bits" in result["random_init_mutual_information"]["task"]
    assert "renormalize" in result["expert_ablation_routing"]
    assert set(result["expert_ablation"]) == {"0", "1"}
    assert "by_motion_category" in result["expert_ablation"]["0"]
    assert len(result["tasks_per_expert"]) == 2
    assert np.isfinite(result["z_shuffle_mse"])
    assert np.isfinite(result["z_zero_mse"])
    cross = result["cross_reconstruction"]
    assert cross["own_z_mse"] == result["eval_reconstruction_mse"]
    assert cross["n_queries"] == result["n_eval_segments"]
    assert np.isfinite(cross["retrieved_z_mse"]) and np.isfinite(cross["random_z_mse"])
    for q, near, random_indices in zip(cross["query_dataset_indices"], cross["retrieved_dataset_indices"],
                                       cross["random_dataset_indices"]):
        for donor in [near, *random_indices]:
            assert dataset.episode_ids[q] != dataset.episode_ids[donor]
            assert dataset.init_state_ids[q] != dataset.init_state_ids[donor]
            assert dataset.episode_ids[donor] in result["train_episode_ids"]
    report_dir = write_report([result], tmp_path / "report", source="synthetic")
    report_text = (report_dir / "report.md").read_text(encoding="utf-8")
    assert "Eval KL (nats)" in report_text
    assert "Shuffle-null mean D/R/G/T" in report_text
    assert "renormalizes remaining probabilities" in report_text
    assert "Own z MSE | Retrieved z MSE | Random z MSE" in report_text
    assert "KL (nats) | Query init groups" in report_text
    assert "oracle transfer diagnostic" in report_text
    assert "not a cluster-aware significance test" in report_text


def test_mutual_information_known_perfect_and_independent_cases():
    labels = np.array(["a", "a", "b", "b"])
    assert mutual_information_bits(labels, labels) == 1.0
    assert mutual_information_bits(labels, np.array([0, 1, 0, 1])) == 0.0


def test_grid_can_sweep_beta_and_latent_dimension():
    dataset = make_synthetic_dataset(n_episodes=4, segments_per_episode=2, horizon=3, seed=8)
    rows = run_grid(dataset, seeds=(0,), n_experts=(1,), shared_options=(False,),
                    routings=("soft",), epochs=1, batch_size=4, eval_fraction=0.25,
                    beta_kl_values=(0.0, 0.01), latent_dims=(2, 4), mi_permutations=5)
    assert len(rows) == 4
    assert {(row["config"]["beta_kl"], row["config"]["latent_dim"]) for row in rows} == {
        (0.0, 2), (0.0, 4), (0.01, 2), (0.01, 4)
    }
    first = rows[0]["cross_reconstruction"]
    for row in rows[1:]:
        cross = row["cross_reconstruction"]
        assert cross["query_dataset_indices"] == first["query_dataset_indices"]
        assert cross["retrieved_dataset_indices"] == first["retrieved_dataset_indices"]
        assert cross["random_dataset_indices"] == first["random_dataset_indices"]
    json.dumps(rows, allow_nan=False)
