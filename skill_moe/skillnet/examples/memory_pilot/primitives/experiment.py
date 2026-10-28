"""Train/evaluate the small unlabeled motion-primitive MoE on CPU or GPU."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from .data import PrimitiveDataset, split_by_episode
from .model import PrimitiveMoE


@dataclass(frozen=True)
class ExperimentConfig:
    n_experts: int = 4
    shared_expert: bool = True
    routing: str = "top1"
    seed: int = 0
    latent_dim: int = 8
    hidden_dim: int = 64
    epochs: int = 60
    batch_size: int = 32
    learning_rate: float = 1e-3
    beta_kl: float = 1e-3
    load_balance_weight: float = 0.03
    mi_permutations: int = 100
    cross_random_draws: int = 8

    def __post_init__(self):
        if self.n_experts not in {1, 2, 4, 8, 16}:
            raise ValueError("n_experts must be one of 1, 2, 4, 8, 16")
        if self.routing not in {"top1", "soft"}:
            raise ValueError("routing must be top1 or soft")
        if self.epochs <= 0 or self.batch_size <= 0 or self.learning_rate <= 0:
            raise ValueError("epochs, batch_size, and learning_rate must be positive")
        if self.latent_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("latent_dim and hidden_dim must be positive")
        if (not np.isfinite(self.beta_kl) or not np.isfinite(self.load_balance_weight)
                or self.beta_kl < 0 or self.load_balance_weight < 0):
            raise ValueError("loss weights must be nonnegative")
        if self.mi_permutations <= 0:
            raise ValueError("mi_permutations must be positive")
        if (not isinstance(self.cross_random_draws, int) or isinstance(self.cross_random_draws, bool)
                or self.cross_random_draws < 1):
            raise ValueError("cross_random_draws must be a positive integer")


def mutual_information_bits(x: Any, y: Any) -> float:
    """Empirical discrete MI in bits, without an optional sklearn dependency."""
    x = np.asarray(x).astype(str).reshape(-1)
    y = np.asarray(y).astype(str).reshape(-1)
    if len(x) != len(y) or len(x) == 0:
        raise ValueError("MI inputs must be nonempty and have equal lengths")
    _, xi = np.unique(x, return_inverse=True)
    _, yi = np.unique(y, return_inverse=True)
    joint = np.zeros((xi.max() + 1, yi.max() + 1), dtype=np.float64)
    np.add.at(joint, (xi, yi), 1)
    pxy = joint / len(x)
    px, py = pxy.sum(axis=1, keepdims=True), pxy.sum(axis=0, keepdims=True)
    expected = px * py
    mask = pxy > 0
    return float(np.sum(pxy[mask] * np.log2(pxy[mask] / expected[mask])))


def _entropy_bits(labels: Any) -> float:
    labels = np.asarray(labels).astype(str).reshape(-1)
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log2(probs)))


def _routing_information_bits(weights: Any, labels: Any) -> float:
    """MI of actual routed expert mass, reducing to ordinary MI for hard routes."""
    weights = np.asarray(weights)
    if weights.ndim == 1:
        return mutual_information_bits(weights, labels)
    labels = np.asarray(labels).astype(str).reshape(-1)
    mass = np.asarray(weights, dtype=np.float64)
    if (mass.ndim != 2 or len(mass) != len(labels) or not len(labels)
            or not np.all(np.isfinite(mass)) or np.any(mass < 0) or np.any(mass.sum(axis=1) <= 0)):
        raise ValueError("routing mass must be finite, nonnegative and nonempty")
    mass = mass / mass.sum(axis=1, keepdims=True)
    _, label_index = np.unique(labels, return_inverse=True)
    joint = np.zeros((mass.shape[1], int(label_index.max()) + 1))
    for label in range(joint.shape[1]):
        joint[:, label] = mass[label_index == label].sum(axis=0) / len(labels)
    expected = joint.sum(axis=1, keepdims=True) * joint.sum(axis=0, keepdims=True)
    nonzero = joint > 0
    return max(0., float(np.sum(joint[nonzero] * np.log2(joint[nonzero] / expected[nonzero]))))


def _conditional_routing_information(weights, labels, conditions, *, seed=0, n_permutations=100):
    """Distinguish within-task motion association from task/motion confounding."""
    labels, conditions = np.asarray(labels).astype(str), np.asarray(conditions).astype(str)
    information, covered, varying = 0., 0, 0
    masks = [conditions == condition for condition in np.unique(conditions)]
    for mask in masks:
        information += float(mask.mean()) * _routing_information_bits(weights[mask], labels[mask])
        if len(np.unique(labels[mask])) > 1:
            covered += int(mask.sum())
            varying += 1
    rng = np.random.default_rng(seed)
    null = np.asarray([sum(float(mask.mean()) * _routing_information_bits(weights[mask], rng.permutation(labels[mask]))
                           for mask in masks) for _ in range(n_permutations)])
    return {"bits": information, "shuffle_null_mean_bits": float(null.mean()),
            "shuffle_adjusted_bits": information - float(null.mean()),
            "shuffle_null_p95_bits": float(np.quantile(null, .95)), "n_varying_strata": varying,
            "fraction_segments_in_varying_strata": covered / len(labels), "identifiable": varying > 0}


def _mi_with_shuffle_null(assignments: Any, labels: Any, *, seed: int,
                          n_permutations: int) -> dict[str, float]:
    """Report plug-in MI beside a label-permutation null with fixed marginals."""
    assignments = np.asarray(assignments)
    labels = np.asarray(labels).astype(str).reshape(-1)
    observed = _routing_information_bits(assignments, labels)
    rng = np.random.default_rng(seed)
    null = np.asarray([
        _routing_information_bits(assignments, rng.permutation(labels))
        for _ in range(n_permutations)
    ])
    entropy = _entropy_bits(labels)
    label_values, label_counts = np.unique(labels, return_counts=True)
    return {
        "n_samples": int(len(labels)),
        "label_counts": {str(label): int(count) for label, count in zip(label_values, label_counts, strict=True)},
        "raw_bits": observed,
        "label_entropy_bits": entropy,
        "normalized_by_label_entropy": observed / entropy if entropy > 0 else 0.0,
        "shuffle_null_mean_bits": float(null.mean()),
        "shuffle_null_std_bits": float(null.std()),
        "shuffle_null_p95_bits": float(np.quantile(null, 0.95)),
        "shuffle_adjusted_bits": float(observed - null.mean()),
    }


def _motion_categories(motion: np.ndarray) -> dict[str, np.ndarray]:
    """Coarse, label-free outcome categories used only for held-out analysis."""
    final = motion[:, -1]
    displacement = final[:, :3]
    magnitude = np.linalg.norm(displacement, axis=1)
    direction = np.zeros(len(motion), dtype=np.int64)  # stationary
    moving = magnitude >= 1e-5
    if np.any(moving):
        axis = np.argmax(np.abs(displacement[moving]), axis=1)
        sign = displacement[moving, axis] >= 0
        direction[moving] = 1 + 2 * axis + (~sign).astype(np.int64)
    rotation = final[:, 3:6]
    rotation_mag = np.linalg.norm(rotation, axis=1)
    rotation_class = np.zeros(len(motion), dtype=np.int64)
    turning = rotation_mag >= 1e-3
    if np.any(turning):
        axis = np.argmax(np.abs(rotation[turning]), axis=1)
        sign = rotation[turning, axis] >= 0
        rotation_class[turning] = 1 + 2 * axis + (~sign).astype(np.int64)
    # LIBERO/Panda `robot0_gripper_qpos` is [left(+open), right(-open)]: classify the
    # finger-gap change, not one finger alone (a single finger at 0.02 missed most events).
    if final.shape[1] > 7:
        grip = final[:, 6] - final[:, 7]
    else:
        grip = final[:, 6] if final.shape[1] > 6 else np.zeros(len(motion))
    grip_class = np.where(grip > 0.01, 1, np.where(grip < -0.01, -1, 0))
    return {"direction": direction, "rotation": rotation_class, "gripper": grip_class}


def _standardize(train: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train.reshape(-1, train.shape[-1]).mean(axis=0)
    scale = train.reshape(-1, train.shape[-1]).std(axis=0)
    scale[scale < 1e-6] = 1.0
    return ((values - mean) / scale).astype(np.float32), mean.astype(np.float32), scale.astype(np.float32)


def _prepare(dataset: PrimitiveDataset, train_idx: np.ndarray, eval_idx: np.ndarray):
    train_states, state_mean, state_scale = _standardize(dataset.states[train_idx], dataset.states)
    train_actions, action_mean, action_scale = _standardize(dataset.actions[train_idx], dataset.actions)
    train_motion, motion_mean, motion_scale = _standardize(dataset.motion[train_idx], dataset.motion)
    train = {"states": train_states[train_idx], "actions": train_actions[train_idx], "motion": train_motion[train_idx]}
    evaluation = {"states": train_states[eval_idx], "actions": train_actions[eval_idx], "motion": train_motion[eval_idx]}
    scales = {"action_mean": action_mean, "action_scale": action_scale}
    return train, evaluation, scales


def _to_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.float32, device=device)


def _predict(model: PrimitiveMoE, batch: dict[str, torch.Tensor], *, z: torch.Tensor | None = None,
             ablate_expert: int | None = None):
    states, actions, motion = batch["states"], batch["actions"], batch["motion"]
    with torch.no_grad():
        mean, _ = model.posterior(states, actions, motion)
        latent = mean if z is None else z
        probs = torch.softmax(model.router(latent), dim=-1)
        prediction, weights = model.decode(states[:, :1].expand(-1, model.horizon, -1), latent,
                                           probs, ablate_expert=ablate_expert,
                                           hard=model.routing == "top1")
    return prediction, mean, probs, weights


def _raw_action_mse(prediction: torch.Tensor, target: torch.Tensor, scales: dict[str, np.ndarray]) -> float:
    mean = torch.as_tensor(scales["action_mean"], device=prediction.device)
    scale = torch.as_tensor(scales["action_scale"], device=prediction.device)
    pred_raw = prediction * scale + mean
    target_raw = target * scale + mean
    return float(torch.mean((pred_raw - target_raw) ** 2).item())


def _action_component_mse(prediction: torch.Tensor, target: torch.Tensor,
                          scales: dict[str, np.ndarray]) -> dict[str, float]:
    mean = torch.as_tensor(scales["action_mean"], device=prediction.device)
    scale = torch.as_tensor(scales["action_scale"], device=prediction.device)
    delta = (prediction * scale + mean) - (target * scale + mean)
    dim = delta.shape[-1]
    groups = {"translation": slice(0, min(3, dim))}
    if dim >= 6:
        groups["rotation"] = slice(3, 6)
    if dim >= 7:
        groups["gripper"] = slice(6, 7)
    if dim > 7:
        groups["other"] = slice(7, dim)
    return {name: float(torch.mean(delta[..., index] ** 2).item()) for name, index in groups.items()}


def _loss(model: PrimitiveMoE, batch: dict[str, torch.Tensor], config: ExperimentConfig):
    result = model(batch["states"], batch["actions"], batch["motion"], sample_latent=True)
    recon = F.mse_loss(result["prediction"], batch["actions"])
    kl = _kl_nats(result["mean"], result["logvar"])
    avg_gate = result["probabilities"].mean(dim=0)
    balance = model.n_experts * torch.sum(avg_gate.square()) - 1.0
    total = recon + config.beta_kl * kl + config.load_balance_weight * balance
    return total, {"reconstruction": recon.detach(), "kl": kl.detach(), "load_balance": balance.detach()}


def _kl_nats(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Mean per-segment KL(q(z|x)||N(0,I)), in nats."""
    return -0.5 * torch.mean(torch.sum(1 + logvar - mean.square() - logvar.exp(), dim=-1))


def _batch_dict(data: dict[str, np.ndarray], indices: np.ndarray, device: torch.device):
    return {name: _to_tensor(values[indices], device) for name, values in data.items()}


def _motion_neighbor_indices(dataset, train_idx, eval_idx, *, seed, random_draws):
    """Oracle future-motion retrieval; no evaluated action enters the candidate z.

    Component units are scaled using training motion only. Rotations are compared
    on SO(3), including the +/-pi branch and equivalent quaternion signs.
    """
    train_motion = np.asarray(dataset.motion[train_idx], dtype=np.float64)
    scales = [max(float(np.sqrt(np.mean(np.sum(train_motion[..., sl] ** 2, axis=-1)))), 1e-6)
              for sl in (slice(0, 3), slice(3, 6), slice(6, None))]
    def quaternions(rotvec):
        angle = np.linalg.norm(rotvec, axis=-1, keepdims=True)
        factor = np.divide(np.sin(angle / 2), angle, out=np.full_like(angle, .5), where=angle > 1e-12)
        return np.concatenate((rotvec * factor, np.cos(angle / 2)), axis=-1)
    train_quat = quaternions(train_motion[..., 3:6])
    nearest, random, counts, nearest_distance, random_distance = [], [], [], [], []
    tie_rng = np.random.default_rng(seed)
    random_rng = np.random.default_rng(seed + 1)
    for query_index in eval_idx:
        eligible = np.flatnonzero((dataset.episode_ids[train_idx] != dataset.episode_ids[query_index])
                                  & (dataset.init_state_ids[train_idx] != dataset.init_state_ids[query_index]))
        if not len(eligible):
            raise ValueError("cross reconstruction has no disjoint training donor for a query")
        query = np.asarray(dataset.motion[query_index], dtype=np.float64)
        candidate = train_motion[eligible]
        pos = np.linalg.norm(candidate[..., :3] - query[None, ..., :3], axis=-1).mean(axis=-1) / scales[0]
        dot = np.abs(np.sum(train_quat[eligible] * quaternions(query[..., 3:6])[None], axis=-1))
        dot = np.where(dot > 1 - 1e-12, 1., np.clip(dot, 0., 1.))
        rot = (2 * np.arccos(dot)).mean(axis=-1) / scales[1]
        grip = np.linalg.norm(candidate[..., 6:] - query[None, ..., 6:], axis=-1).mean(axis=-1) / scales[2]
        distance = pos + rot + grip
        chosen = int(tie_rng.choice(np.flatnonzero(np.isclose(distance, distance.min(), rtol=1e-12, atol=1e-12))))
        draws = random_rng.integers(len(eligible), size=random_draws)
        nearest.append(int(eligible[chosen]))
        random.append(eligible[draws])
        counts.append(len(eligible))
        nearest_distance.append(float(distance[chosen]))
        random_distance.append(float(distance[draws].mean()))
    return np.asarray(nearest), np.asarray(random), {
        "motion_component_scales": dict(zip(("position", "rotation", "gripper"), scales)),
        "eligible_donor_counts": counts,
        "retrieved_motion_distance_mean": float(np.mean(nearest_distance)),
        "random_motion_distance_mean": float(np.mean(random_distance)),
    }


def _cross_reconstruction(model, dataset, train_idx, eval_idx, train_tensors, eval_tensors,
                          scales, *, seed, random_draws, own_mse):
    nearest, random, diagnostics = _motion_neighbor_indices(dataset, train_idx, eval_idx,
                                                            seed=seed, random_draws=random_draws)
    # Encode donor segments only. Decoder state and phase belong to the query;
    # both its gate and expert FiLM conditioning receive the donor's z.
    with torch.no_grad():
        donor_z, _ = model.posterior(train_tensors["states"], train_tensors["actions"], train_tensors["motion"])
        def error(indices):
            total = 0.
            for start in range(0, len(eval_idx), 256):
                end = min(start + 256, len(eval_idx))
                z = donor_z[torch.as_tensor(indices[start:end], device=donor_z.device)]
                probabilities = torch.softmax(model.router(z), dim=-1)
                states = eval_tensors["states"][start:end, :1].expand(-1, model.horizon, -1)
                pred, _ = model.decode(states, z, probabilities, hard=model.routing == "top1")
                total += (end - start) * _raw_action_mse(pred, eval_tensors["actions"][start:end], scales)
            return total / len(eval_idx)
        retrieved_mse = error(nearest)
        random_errors = [error(random[:, draw]) for draw in range(random_draws)]
    return {
        "selection": "oracle observed future-motion nearest neighbor; training donors only",
        "latent": "posterior mean; donor router and FiLM z, query start state and phase",
        "own_z_mse": own_mse, "retrieved_z_mse": retrieved_mse,
        "random_z_mse": float(np.mean(random_errors)),
        "retrieved_minus_own_mse": retrieved_mse - own_mse,
        "random_minus_own_mse": float(np.mean(random_errors)) - own_mse,
        "random_draw_mses": random_errors, "random_draws": random_draws,
        "query_dataset_indices": eval_idx.tolist(),
        "retrieved_dataset_indices": train_idx[nearest].tolist(),
        "random_dataset_indices": train_idx[random].tolist(),
        "n_queries": len(eval_idx),
        "n_query_init_states": len(set(dataset.init_state_ids[eval_idx].tolist())),
        **diagnostics,
    }


def run_experiment(dataset: PrimitiveDataset, config: ExperimentConfig, *, eval_fraction: float = 0.25,
                   split_seed: int = 0, device_name: str = "cpu") -> dict[str, Any]:
    """Fit one configuration and report held-out reconstruction, MI and ablations."""
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.set_num_threads(1)
    device = torch.device(device_name)
    train_idx, eval_idx = split_by_episode(dataset, eval_fraction=eval_fraction, seed=split_seed)
    train, evaluation, scales = _prepare(dataset, train_idx, eval_idx)
    model = PrimitiveMoE(horizon=dataset.horizon, state_dim=dataset.states.shape[-1],
                         action_dim=dataset.actions.shape[-1], motion_dim=dataset.motion.shape[-1],
                         n_experts=config.n_experts, latent_dim=config.latent_dim,
                         hidden_dim=config.hidden_dim, shared_expert=config.shared_expert,
                         routing=config.routing).to(device)
    # Same held-out samples and label cardinalities, before any gradient updates.
    model.eval()
    eval_tensors = {key: _to_tensor(value, device) for key, value in evaluation.items()}
    _, _, _, random_init_weights = _predict(model, eval_tensors)
    random_init_weights = random_init_weights.cpu().numpy()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    train_tensors = {key: _to_tensor(value, device) for key, value in train.items()}
    rng = np.random.default_rng(config.seed)
    model.train()
    for _epoch in range(config.epochs):
        order = rng.permutation(len(train_idx))
        for start in range(0, len(order), config.batch_size):
            indexes = order[start:start + config.batch_size]
            batch = {key: values[indexes] for key, values in train_tensors.items()}
            optimizer.zero_grad(set_to_none=True)
            total, _ = _loss(model, batch, config)
            total.backward()
            optimizer.step()

    model.eval()
    baseline, z_mean, probabilities, route_weights = _predict(model, eval_tensors)
    target = eval_tensors["actions"]
    base_mse = _raw_action_mse(baseline, target, scales)
    base_components = _action_component_mse(baseline, target, scales)
    assignments = probabilities.argmax(dim=-1).cpu().numpy()
    routed_mass = route_weights.cpu().numpy()
    eval_raw_motion = dataset.motion[eval_idx]
    categories = _motion_categories(eval_raw_motion)
    task_labels = dataset.task_labels[eval_idx]
    mi_labels = {**categories, "task": task_labels}
    mi = {name: _mi_with_shuffle_null(routed_mass, label, seed=config.seed + i,
                                      n_permutations=config.mi_permutations)
          for i, (name, label) in enumerate(mi_labels.items())}
    random_init_mi = {name: _mi_with_shuffle_null(random_init_weights, label,
                                                  seed=config.seed + 101 + i,
                                                  n_permutations=config.mi_permutations)
                      for i, (name, label) in enumerate(mi_labels.items())}
    with torch.no_grad():
        eval_mean, eval_logvar = model.posterior(eval_tensors["states"], eval_tensors["actions"],
                                                 eval_tensors["motion"])
        eval_kl_nats = float(_kl_nats(eval_mean, eval_logvar).item())
        eval_kl_per_dim_nats = eval_kl_nats / config.latent_dim
    joint_motion_labels = np.asarray([json.dumps([int(categories[name][i]) for name in ("direction", "rotation", "gripper")])
                                      for i in range(len(eval_idx))])
    conditional_mi = {
        "motion_given_task": _conditional_routing_information(routed_mass, joint_motion_labels, task_labels,
                                                               seed=config.seed + 201, n_permutations=config.mi_permutations),
        "task_given_motion": _conditional_routing_information(routed_mass, task_labels, joint_motion_labels,
                                                               seed=config.seed + 202, n_permutations=config.mi_permutations),
    }
    task_sharing, task_router_mass = {}, {}
    for expert_id in range(config.n_experts):
        used = routed_mass[:, expert_id] > 0
        task_sharing[str(expert_id)] = sorted(set(task_labels[used].tolist()))
        task_router_mass[str(expert_id)] = {str(task): float(routed_mass[task_labels == task, expert_id].mean())
                                             for task in np.unique(task_labels)}
    usage = routed_mass.mean(axis=0)

    # Latent ablations preserve the held-out query states/actions and only alter z.
    perm = torch.as_tensor(np.random.default_rng(config.seed + 91).permutation(len(z_mean)),
                           device=device, dtype=torch.long)
    shuffled_pred, _, _, _ = _predict(model, eval_tensors, z=z_mean[perm])
    zero_pred, _, _, _ = _predict(model, eval_tensors, z=torch.zeros_like(z_mean))
    z_shuffle_mse = _raw_action_mse(shuffled_pred, target, scales)
    z_zero_mse = _raw_action_mse(zero_pred, target, scales)
    cross = _cross_reconstruction(model, dataset, train_idx, eval_idx, train_tensors, eval_tensors,
                                 scales, seed=split_seed + 2025, random_draws=config.cross_random_draws,
                                 own_mse=base_mse)

    category_lists = {name: labels.astype(str) for name, labels in categories.items()}
    ablations = {}
    for expert_id in range(config.n_experts):
        ablated_pred, _, _, _ = _predict(model, eval_tensors, ablate_expert=expert_id)
        overall = _raw_action_mse(ablated_pred, target, scales)
        group_mse = {}
        for feature, labels in category_lists.items():
            feature_metrics = {}
            for category in np.unique(labels):
                mask = np.flatnonzero(labels == category)
                index = torch.as_tensor(mask, device=device, dtype=torch.long)
                value = _raw_action_mse(ablated_pred[index], target[index], scales)
                full_value = _raw_action_mse(baseline[index], target[index], scales)
                feature_metrics[str(category)] = {"mse": value, "full_mse": full_value,
                                                  "n_segments": len(mask), "delta_from_full": value - full_value}
            group_mse[feature] = feature_metrics
        ablations[str(expert_id)] = {"mse": overall, "delta_from_full": overall - base_mse,
                                     "by_motion_category": group_mse}
    return {
        "config": asdict(config),
        "loss": {"objective": "action_mse + beta_kl * KL(q(z|segment)||N(0,I)) + load_balance_weight * (N*sum(mean_gate^2)-1)",
                 "beta_kl": config.beta_kl, "load_balance_weight": config.load_balance_weight},
        "n_train_segments": int(len(train_idx)), "n_eval_segments": int(len(eval_idx)),
        "train_episode_ids": sorted(set(dataset.episode_ids[train_idx].tolist())),
        "eval_episode_ids": sorted(set(dataset.episode_ids[eval_idx].tolist())),
        "train_init_state_ids": sorted(set(dataset.init_state_ids[train_idx].tolist())),
        "eval_init_state_ids": sorted(set(dataset.init_state_ids[eval_idx].tolist())),
        "eval_reconstruction_mse": base_mse, "eval_action_component_mse": base_components,
        "eval_mean_kl_nats": eval_kl_nats,
        "eval_mean_kl_per_latent_dim_nats": eval_kl_per_dim_nats,
        "expert_usage_fraction": usage.tolist(), "mean_router_probability": probabilities.mean(0).cpu().tolist(),
        "argmax_expert_usage_fraction": (np.bincount(assignments, minlength=config.n_experts) / len(assignments)).tolist(),
        "mutual_information_bits": mi, "random_init_mutual_information": random_init_mi,
        "conditional_routing_mi": conditional_mi,
        "mi_assignment": "actual routed mass; hard one-hot or soft probability",
        "mi_null_scope": "per-segment label permutation: cardinality diagnostic, not cluster-aware significance",
        "mi_n_samples": int(len(assignments)), "mi_null_permutations": config.mi_permutations,
        "tasks_per_expert": task_sharing, "mean_routed_mass_by_task": task_router_mass,
        "n_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "z_shuffle_mse": z_shuffle_mse, "z_zero_mse": z_zero_mse,
        "cross_reconstruction": cross,
        "expert_ablation_routing": "renormalize remaining expert probabilities and reroute",
        "expert_ablation": ablations,
    }


def run_grid(dataset: PrimitiveDataset, *, seeds: tuple[int, ...] = (0, 1, 2),
             n_experts: tuple[int, ...] = (1, 2, 4, 8, 16),
             shared_options: tuple[bool, ...] = (False, True),
             routings: tuple[str, ...] = ("top1", "soft"), epochs: int = 60,
             batch_size: int = 32, eval_fraction: float = 0.25,
             device_name: str = "cpu", beta_kl_values: tuple[float, ...] = (1e-3,),
             latent_dims: tuple[int, ...] = (8,), mi_permutations: int = 100,
             cross_random_draws: int = 8) -> list[dict[str, Any]]:
    """Run the requested factor grid; train/eval episodes are fixed across variants."""
    rows = []
    for seed in seeds:
        for shared in shared_options:
            for count in n_experts:
                for routing in routings:
                    for beta_kl in beta_kl_values:
                        for latent_dim in latent_dims:
                            config = ExperimentConfig(n_experts=count, shared_expert=shared, routing=routing,
                                                      seed=seed, latent_dim=latent_dim, epochs=epochs,
                                                      batch_size=batch_size, beta_kl=beta_kl,
                                                      mi_permutations=mi_permutations,
                                                      cross_random_draws=cross_random_draws)
                            rows.append(run_experiment(dataset, config, eval_fraction=eval_fraction,
                                                       split_seed=0, device_name=device_name))
    return rows


def write_report(rows: list[dict[str, Any]], output_dir: str | Path, *, source: str) -> Path:
    """Write a unique experiment bundle under the memory-pilot output root."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "results.json").write_text(json.dumps({"source": source, "runs": rows}, indent=2),
                                               encoding="utf-8")
    lines = ["# Emergent motion primitive quick experiment", "",
             f"Source: `{source}`. Episode and namespaced initial-state groups are disjoint between train/eval; task labels are held-out analysis only.",
             "", "Objective: action MSE + beta-weighted Gaussian KL + expert load-balancing penalty.",
             "MI uses actual routed mass, including soft routes; cells show raw bits (shuffle-adjusted bits; MI/label entropy). Baseline vectors use direction/rotation/gripper/task order.", "",
             "| N | Shared | Routing | β | Dz | Seed | Eval action MSE | Eval KL (nats) | MI direction | MI rotation | MI gripper | MI task | Random-init MI D/R/G/T | Shuffle-null mean D/R/G/T | MI samples | z shuffle | z zero |",
             "|---:|:---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|"]
    for row in rows:
        cfg, mi = row["config"], row["mutual_information_bits"]
        mi_names = ("direction", "rotation", "gripper", "task")
        random_init_values = ",".join(
            f"{row['random_init_mutual_information'][name]['raw_bits']:.3f}" for name in mi_names
        )
        null_values = ",".join(f"{mi[name]['shuffle_null_mean_bits']:.3f}" for name in mi_names)
        mi_cell = lambda name: (f"{mi[name]['raw_bits']:.3f} "
                                f"({mi[name]['shuffle_adjusted_bits']:.3f}; "
                                f"{mi[name]['normalized_by_label_entropy']:.3f})")
        lines.append(f"| {cfg['n_experts']} | {cfg['shared_expert']} | {cfg['routing']} | "
                     f"{cfg['beta_kl']:.3g} | {cfg['latent_dim']} | {cfg['seed']} | "
                     f"{row['eval_reconstruction_mse']:.6g} | "
                     f"{row['eval_mean_kl_nats']:.4g} | "
                     f"{mi_cell('direction')} | {mi_cell('rotation')} | "
                     f"{mi_cell('gripper')} | {mi_cell('task')} | "
                     f"{random_init_values} | {null_values} | {row['mi_n_samples']} | "
                     f"{row['z_shuffle_mse']:.6g} | {row['z_zero_mse']:.6g} |")
    lines += ["", "## Cross reconstruction by beta", "",
              "All errors use held-out query actions in original command units (MSE). "
              "Own z and donor z are posterior means; each donor comes only from the training split, "
              "with both episode and initial-state IDs different from the query. "
              "Random draws use exactly the same eligible pool. Donors are fixed across model/beta/seed variants.", "",
              "| β | N | Shared | Routing | Dz | Seed | Own z MSE | Retrieved z MSE | Random z MSE | Retrieved − own | KL (nats) | Query init groups |",
              "|---:|---:|:---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in sorted(rows, key=lambda row: (row["config"]["n_experts"], row["config"]["shared_expert"],
                                            row["config"]["routing"], row["config"]["latent_dim"],
                                            row["config"]["seed"], row["config"]["beta_kl"])):
        cfg, cross = row["config"], row["cross_reconstruction"]
        lines.append(f"| {cfg['beta_kl']:.4g} | {cfg['n_experts']} | {cfg['shared_expert']} | {cfg['routing']} | "
                     f"{cfg['latent_dim']} | {cfg['seed']} | {cross['own_z_mse']:.6g} | {cross['retrieved_z_mse']:.6g} | "
                     f"{cross['random_z_mse']:.6g} | {cross['retrieved_minus_own_mse']:.6g} | "
                     f"{row['eval_mean_kl_nats']:.6g} | {cross['n_query_init_states']} |")
    lines += ["", "Retrieval uses observed future eef motion: this is an oracle transfer diagnostic, not deployable retrieval. "
              "Its position/rotation/gripper scales are fitted on training motion only. Random errors average "
              "the configured number of uniform donor draws; draw errors and exact donor indices are in results.json.",
              "A small retrieved-minus-own gap supports transfer only if random/zero z controls are worse. "
              "If all z conditions are similar, the decoder may simply ignore z. Low Gaussian KL does not bound "
              "the information in deterministic posterior means, and this diagnostic cannot prove a primitive has emerged.",
              "", "## Task versus motion association", "",
              "| N | Shared | Routing | β | Dz | Seed | I(expert; motion ∣ task), bits | I(expert; task ∣ motion), bits |",
              "|---:|:---:|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        cfg, conditional = row["config"], row["conditional_routing_mi"]
        cell = lambda key: (f"{conditional[key]['bits']:.6g} ({conditional[key]['shuffle_adjusted_bits']:.6g})"
                            if conditional[key]["identifiable"] else "unidentifiable")
        lines.append(f"| {cfg['n_experts']} | {cfg['shared_expert']} | {cfg['routing']} | {cfg['beta_kl']:.3g} | "
                     f"{cfg['latent_dim']} | {cfg['seed']} | {cell('motion_given_task')} | {cell('task_given_motion')} |")
    lines += ["", "Conditional MI cells show raw bits (within-stratum shuffle-adjusted bits). "
              "Motion here is the joint coarse direction/rotation/gripper category. Stratum coverage is in results.json; "
              "without crossed task/motion coverage, lack of task dependence is unidentifiable. "
              "Marginal MI comparisons alone cannot distinguish task-dependent from motion-dependent routing.",
              "", "Expert usage, tasks-per-expert, mean router mass by task, parameter counts and paired per-expert ablation deltas are in `results.json`.",
              "Ablation removes the named expert, renormalizes remaining probabilities, then reroutes (top-1 or soft per configuration).",
              "MI is empirical plug-in MI; `results.json` also contains each label entropy, normalized MI, null standard deviation/p95, random-initialization MI, sample count, and permutation count.",
              "Ablation category deltas subtract the full model's MSE on the same category, not the global MSE. "
              "For soft routing every positive-weight expert counts as used; mean mass by task is more informative than task counts.",
              "The per-segment shuffle null is a cardinality diagnostic, not a cluster-aware significance test: "
              "segments within an initial state are correlated. Three model seeds on one split do not measure split uncertainty. "
              "Shared/N variants also change parameter count. This is descriptive decision support, not a formal emergence verdict.", ""]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return output_dir


def unique_output_dir(root: str | Path = "/work/.skillnet-state/outputs/memory-pilot") -> Path:
    base = Path(root)
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for _ in range(20):
        candidate = base / f"primitives-{stamp}-{uuid.uuid4().hex[:6]}"
        if not candidate.exists():
            return candidate
    raise RuntimeError("unable to allocate a unique primitives output directory")
