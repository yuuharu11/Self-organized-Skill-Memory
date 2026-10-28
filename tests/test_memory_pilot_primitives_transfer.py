from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from skill_moe.skillnet.examples.memory_pilot.primitives import experiment
from skill_moe.skillnet.examples.memory_pilot.primitives.data import PrimitiveDataset, load_recording, make_synthetic_dataset
from skill_moe.skillnet.examples.memory_pilot.primitives.experiment import ExperimentConfig, run_experiment


def _dataset(positions, *, rotations=None):
    n = len(positions)
    motion = np.zeros((n, 2, 7), dtype=np.float32)
    motion[..., 0] = np.asarray(positions)[:, None]
    if rotations is not None:
        motion[..., 5] = np.asarray(rotations)[:, None]
    return PrimitiveDataset(np.zeros((n, 2, 1), np.float32), np.zeros((n, 2, 1), np.float32), motion,
                            np.asarray([f"ep{i}" for i in range(n)]), np.asarray([f"init{i}" for i in range(n)]),
                            np.asarray(["task"] * n), np.asarray(["segment"] * n))


def test_motion_donors_exclude_episode_and_initial_state_and_share_random_pool():
    data = _dataset([0, 0, 2, 0, 10])
    data = replace(data, episode_ids=np.array(["query", "same-state", "near", "query", "far"]),
                   init_state_ids=np.array(["query-init", "query-init", "near", "query-init", "far"]))
    train, query = np.array([0, 1, 2, 4]), np.array([3])
    nearest, random, diagnostics = experiment._motion_neighbor_indices(data, train, query, seed=7, random_draws=100)
    assert train[nearest].tolist() == [2]
    assert set(train[random].reshape(-1)) == {2, 4}
    assert diagnostics["eligible_donor_counts"] == [2]
    with pytest.raises(ValueError, match="no disjoint training donor"):
        experiment._motion_neighbor_indices(data, np.array([0, 1]), query, seed=7, random_draws=2)


def test_motion_distance_handles_rotation_branch_and_fits_only_training_scales():
    data = _dataset([0, 0, 0], rotations=[-np.pi + .01, 0, np.pi - .01])
    train, query = np.array([0, 1]), np.array([2])
    nearest, _, before = experiment._motion_neighbor_indices(data, train, query, seed=0, random_draws=2)
    assert nearest.tolist() == [0]
    changed = data.motion.copy()
    changed[2, :, 0] = 1e6
    _, _, after = experiment._motion_neighbor_indices(replace(data, motion=changed), train, query,
                                                      seed=0, random_draws=2)
    assert before["motion_component_scales"] == after["motion_component_scales"]


def test_cross_reconstruction_uses_donor_code_and_query_start_state_without_query_posterior():
    data = _dataset([0, 10, 0])
    data = replace(data, states=np.broadcast_to(np.array([1., 2., 10.])[:, None, None], (3, 2, 1)).copy(),
                   actions=np.broadcast_to(np.array([3., 9., 13.])[:, None, None], (3, 2, 1)).copy())
    class Model:
        horizon = 2
        routing = "top1"
        def posterior(self, states, actions, motion):
            assert torch.all(states < 5), "evaluated query must not supply its own posterior"
            return actions[:, 0], torch.zeros_like(actions[:, 0])
        def router(self, z):
            return torch.zeros_like(z)
        def decode(self, states, z, probabilities, hard):
            return states + z[:, None], probabilities
    train, query = np.array([0, 1]), np.array([2])
    tensors = {key: torch.as_tensor(getattr(data, key), dtype=torch.float32) for key in ("states", "actions", "motion")}
    result = experiment._cross_reconstruction(Model(), data, train, query,
        {key: value[train] for key, value in tensors.items()}, {key: value[query] for key, value in tensors.items()},
        {"action_mean": np.zeros(1), "action_scale": np.ones(1)}, seed=0, random_draws=32, own_mse=0.)
    assert result["retrieved_z_mse"] == 0
    assert result["random_z_mse"] > 0
    donors = np.asarray(result["random_dataset_indices"])
    assert set(donors.reshape(-1)) == {0, 1}
    assert result["random_z_mse"] == pytest.approx(np.mean(np.where(donors == 0, 0., 36.)))


def test_soft_information_uses_routed_mass_instead_of_argmax_only():
    weights = np.array([[.9, .1], [.6, .4]])
    labels = np.array(["a", "b"])
    assert experiment.mutual_information_bits(weights.argmax(axis=1), labels) == 0
    assert experiment._routing_information_bits(weights, labels) > 0
    assert experiment._routing_information_bits(np.full((2, 2), .5), labels) == 0
    assert experiment._routing_information_bits(np.eye(2), labels) == pytest.approx(1.)


def test_task_motion_confounding_is_unidentifiable_without_crossed_coverage():
    mass = np.array([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])
    task = np.array(["a", "a", "b", "b"])
    motion = np.array(["x", "x", "y", "y"])
    assert experiment._routing_information_bits(mass, motion) == 1
    result = experiment._conditional_routing_information(mass, motion, task)
    assert not result["identifiable"]
    crossed_motion = np.array(["x", "y", "x", "y"])
    crossed_mass = np.array([[1., 0.], [0., 1.], [1., 0.], [0., 1.]])
    within_task = experiment._conditional_routing_information(crossed_mass, crossed_motion, task)
    task_given_motion = experiment._conditional_routing_information(crossed_mass, task, crossed_motion)
    assert within_task["identifiable"] and within_task["bits"] == 1
    assert task_given_motion["identifiable"] and task_given_motion["bits"] == 0


def test_category_ablation_is_paired_even_when_full_model_category_errors_differ(monkeypatch):
    data = make_synthetic_dataset(n_episodes=6, segments_per_episode=3, horizon=3)
    original = experiment._predict
    def unchanged_ablation(model, batch, **kwargs):
        kwargs.pop("ablate_expert", None)
        return original(model, batch, **kwargs)
    monkeypatch.setattr(experiment, "_predict", unchanged_ablation)
    row = run_experiment(data, ExperimentConfig(n_experts=2, routing="soft", hidden_dim=8, latent_dim=2,
                                               epochs=1, mi_permutations=3, cross_random_draws=2))
    full_errors = []
    for ablation in row["expert_ablation"].values():
        for categories in ablation["by_motion_category"].values():
            for category in categories.values():
                assert category["delta_from_full"] == pytest.approx(0.)
                full_errors.append(category["full_mse"])
    assert np.ptp(full_errors) > 1e-4
    assert all(weight > 0 for weight in row["expert_usage_fraction"])
    assert sum(row["expert_usage_fraction"]) == pytest.approx(1.)


def test_recording_alias_cannot_leak_the_same_episode_across_initial_states(tmp_path):
    (tmp_path / "episodes").mkdir()
    (tmp_path / "manifest.json").write_text(json.dumps({"episodes": [
        {"episode_id": "ep", "init_state_id": "one"},
        {"episode_id": "alias", "init_state_id": "two", "file": "episodes/ep.npz"}], "env_step_offset": 0}))
    np.savez(tmp_path / "episodes" / "ep.npz", step=np.array([0]), proprio=np.zeros((1, 1)),
             pred_chunk=np.zeros((1, 1, 1)), n_executed=np.array([1]), segment_id=np.array([0]),
             env_eef_pos=np.zeros((2, 3)), env_eef_quat=np.array([[0, 0, 0, 1.]] * 2), env_gripper=np.zeros((2, 1)))
    with pytest.raises(ValueError, match="same episode file"):
        load_recording(tmp_path)


def test_gripper_category_uses_two_finger_gap_change():
    motion = np.zeros((3, 2, 8), dtype=np.float32)
    motion[0, -1, 6:8] = [.008, -.008]   # opening: gap +0.016, each finger below 0.02
    motion[1, -1, 6:8] = [-.008, .008]   # closing
    motion[2, -1, 6:8] = [.008, .008]    # whole-hand drift, gap unchanged
    assert experiment._motion_categories(motion)["gripper"].tolist() == [1, -1, 0]
