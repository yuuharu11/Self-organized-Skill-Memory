#!/usr/bin/env python3
"""Train a small retrieval projection and compare it with centered/oracle keys.

The policy/M1 weights remain frozen. Flow sampling and bootstrap helpers are
shared with anchor_v2_offline.py.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
EXAMPLES = HERE.parents[1]
for path in (str(HERE), str(EXAMPLES)):
    if path not in sys.path:
        sys.path.insert(0, path)

from key_check import DEFAULT_CHECKPOINT, DEFAULT_HELDOUT, DEFAULT_MEMORY, DEFAULT_ROWS, ROW_FIELDS, _read_npz
from research_paths import REPO_ROOT as RESEARCH_REPO_ROOT, STATE_DIR as RESEARCH_STATE_DIR
from memory import load_memory

DEFAULT_QUERY_KEYS = str(Path(DEFAULT_HELDOUT).with_name("query-keys.npy"))
DEFAULT_OUTPUT_ROOT = str(RESEARCH_STATE_DIR / 'outputs/memory-pilot')
ACTIVE_T0 = (0.2, 0.4, 0.6, 1.0)
ANCHOR_NAMES = ("Ap", "R")
TAU = 0.05
ENDPOINT_SCALE_M = np.asarray([0.0111, 0.0122, 0.0110], np.float32)


def _load_offline_helpers():
    """Reuse the offline sampler metrics through normal module imports."""
    from anchor_v2_offline import cluster_bootstrap, denoising_steps, row_errors
    return denoising_steps, row_errors, cluster_bootstrap


def _normalize(x):
    x = np.asarray(x, np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def _source_task(source_file):
    return str(source_file)


def _sample_task_queries(source_files, episode_ids, seed, n_tasks=10, episodes_per_task=2, rows_per_episode=8):
    rng = np.random.default_rng(seed)
    tasks = sorted(set(map(str, source_files)))
    selected_tasks = sorted(rng.choice(tasks, size=min(n_tasks, len(tasks)), replace=False).tolist())
    selected_rows, selected_episode_ids = [], []
    task_reports = {}
    source_files = np.asarray(source_files).astype(str)
    episode_ids = np.asarray(episode_ids).astype(str)
    for task in selected_tasks:
        task_rows = np.flatnonzero(source_files == task)
        eps = sorted(set(episode_ids[task_rows]))
        chosen_eps = sorted(rng.choice(eps, size=min(episodes_per_task, len(eps)), replace=False).tolist())
        task_reports[task] = chosen_eps
        for episode in chosen_eps:
            rows = task_rows[episode_ids[task_rows] == episode]
            if len(rows) > rows_per_episode:
                take = np.linspace(0, len(rows) - 1, rows_per_episode).round().astype(int)
                rows = rows[take]
            selected_rows.extend(map(int, rows))
            selected_episode_ids.extend([episode] * len(rows))
    return np.asarray(selected_rows, np.int64), selected_tasks, task_reports


def _behavior_neighbors(bases, episodes, initials, *, k=8, query_k=48, workers=8):
    from scipy.spatial import cKDTree

    tree = cKDTree(np.asarray(bases, np.float32))
    distances, indices = tree.query(bases, k=min(query_k, len(bases)), workers=workers)
    if indices.ndim == 1:
        indices, distances = indices[:, None], distances[:, None]
    positives = np.empty((len(bases), k), np.int64)
    for row in range(len(bases)):
        candidates = [int(j) for j in indices[row]
                      if int(j) != row and episodes[int(j)] != episodes[row]
                      and initials[int(j)] != initials[row]]
        if len(candidates) < k:
            # Keep the closest available different-demo examples; self/init are always excluded.
            candidates = [int(j) for j in np.argsort(np.linalg.norm(bases - bases[row], axis=1))
                          if int(j) != row and episodes[int(j)] != episodes[row]
                          and initials[int(j)] != initials[row]]
        if not candidates:
            raise ValueError(f"no different-demo positive for row {row}")
        fill = (candidates * ((k + len(candidates) - 1) // len(candidates)))[:k]
        positives[row] = fill
    return positives


def _fit_projector(keys, states, bases, episodes, initials, *, seed, steps=600, batch_size=256,
                   learning_rate=1e-3, output_path=None):
    import torch
    from torch import nn
    import torch.nn.functional as F

    torch.set_num_threads(8)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    keys = _normalize(keys - np.asarray(keys, np.float32).mean(axis=0, keepdims=True))
    states = _normalize(states)
    inputs = np.concatenate([keys, states], axis=1).astype(np.float32)
    positives = _behavior_neighbors(bases, episodes, initials)
    x = torch.from_numpy(inputs)
    positive_np = positives

    class Projector(nn.Module):
        def __init__(self, in_dim):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(in_dim, 256), nn.GELU(), nn.Linear(256, 128))

        def forward(self, value):
            return F.normalize(self.net(value), dim=-1)

    model = Projector(inputs.shape[1]).cpu()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    history = []
    model.train()
    for step in range(steps):
        anchor_ids = rng.integers(0, len(inputs), size=batch_size)
        positive_ids = np.asarray([rng.choice(positive_np[i]) for i in anchor_ids], np.int64)
        anchor = model(x[torch.from_numpy(anchor_ids)])
        candidate = model(x[torch.from_numpy(positive_ids)])
        logits = (anchor @ candidate.T) / 0.07
        posmask_np = np.any(positive_np[anchor_ids, :, None] == positive_ids[None, None, :], axis=1)
        posmask = torch.from_numpy(posmask_np)
        posmask[torch.arange(batch_size), torch.arange(batch_size)] = True
        log_denom = torch.logsumexp(logits, dim=1)
        log_positive = torch.logsumexp(logits.masked_fill(~posmask, -1e9), dim=1)
        loss = (log_denom - log_positive).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if (step + 1) % 50 == 0 or step == 0:
            history.append({"step": step + 1, "multi_positive_infonce": float(loss.detach())})
            if (step + 1) % 100 == 0 or step == 0:
                print(f"projector step={step + 1}/{steps} loss={history[-1]['multi_positive_infonce']:.4f}", flush=True)
    model.eval()
    if output_path is not None:
        torch.save({"state_dict": model.state_dict(), "input_dim": inputs.shape[1],
                    "output_dim": 128, "seed": seed}, output_path)
    return None, {"steps": steps, "batch_size": batch_size, "learning_rate": learning_rate,
                       "positive_rule": "nearest 8 M6 base vectors by Euclidean L2; exclude same episode and init_state",
                       "negative_rule": "other in-batch positive-candidate vectors",
                       "temperature": 0.07, "history": history,
                       "final_logged_loss": history[-1]["multi_positive_infonce"]}


def _fit_projector_memory(all_keys, all_states, all_bases, all_eps, all_initials, active_ids,
                          *, seed, steps, batch_size, output_path=None):
    local_keys = all_keys[active_ids]
    local_states = all_states[active_ids]
    local_bases = all_bases[active_ids]
    local_eps = all_eps[active_ids]
    local_initials = all_initials[active_ids]
    projected, history = _fit_projector(local_keys, local_states, local_bases, local_eps, local_initials,
                                        seed=seed, steps=steps, batch_size=batch_size,
                                        output_path=output_path)
    return projected, history


def _project_queries(model_path, mem_keys, mem_state, q_keys, q_state, *, seed):
    # Rebuild the small CPU MLP from its saved state. Inputs are centered by the
    # active memory rows only, exactly matching training.
    import torch
    from torch import nn
    import torch.nn.functional as F

    class Projector(nn.Module):
        def __init__(self, in_dim):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(in_dim, 256), nn.GELU(), nn.Linear(256, 128))

        def forward(self, value):
            return F.normalize(self.net(value), dim=-1)

    saved = torch.load(model_path, map_location="cpu", weights_only=False)
    model = Projector(saved["input_dim"])
    model.load_state_dict(saved["state_dict"])
    model.eval()
    key_mean = np.asarray(mem_keys, np.float32).mean(axis=0, keepdims=True)
    train_x = np.concatenate([_normalize(np.asarray(mem_keys, np.float32) - key_mean), _normalize(mem_state)], axis=1)
    query_x = np.concatenate([_normalize(np.asarray(q_keys, np.float32) - key_mean), _normalize(q_state)], axis=1)
    with torch.no_grad():
        mem_out = model(torch.from_numpy(train_x)).numpy().astype(np.float32)
        query_out = model(torch.from_numpy(query_x)).numpy().astype(np.float32)
    return mem_out, query_out, key_mean


def _topk_cosine(memory_keys, query_keys, mem_eps, mem_initials, q_eps, q_initials, *, k=5):
    memory_keys = _normalize(memory_keys)
    query_keys = _normalize(query_keys)
    ids = np.empty((len(query_keys), k), np.int64)
    sims = np.empty((len(query_keys), k), np.float32)
    for i, query in enumerate(query_keys):
        scores = memory_keys @ query
        eligible = (mem_eps != q_eps[i]) & (mem_initials != q_initials[i])
        scores[~eligible] = -np.inf
        selected = np.argsort(-scores, kind="stable")[:k]
        ids[i], sims[i] = selected, scores[selected]
    return ids, sims


def _topk_distance(memory_values, query_values, mem_eps, mem_initials, q_eps, q_initials, *, k=5):
    ids = np.empty((len(query_values), k), np.int64)
    distances = np.empty((len(query_values), k), np.float32)
    for i, query in enumerate(query_values):
        d = np.linalg.norm(memory_values - query[None], axis=1)
        eligible = (mem_eps != q_eps[i]) & (mem_initials != q_initials[i])
        d[~eligible] = np.inf
        selected = np.argsort(d, kind="stable")[:k]
        ids[i], distances[i] = selected, d[selected]
    return ids, distances


def _weights_from_similarity(sims):
    logits = np.asarray(sims, np.float64) / TAU
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    return (weights / weights.sum(axis=1, keepdims=True)).astype(np.float32)


def _weights_from_distance(distances):
    # Oracle list is ordered by latent distance; deterministic top-1 ceiling.
    weights = np.zeros_like(distances, np.float32)
    weights[:, 0] = 1.0
    return weights


def _macro_recall(true_unit, candidate_units, majority_unit):
    true_unit = np.asarray(true_unit, np.int64)
    candidate_units = np.asarray(candidate_units, np.int64)
    classes = np.unique(true_unit[true_unit >= 0])
    hit1 = candidate_units[:, 0] == true_unit
    hit5 = np.any(candidate_units == true_unit[:, None], axis=1)
    per1, per5 = {}, {}
    for unit in classes:
        mask = true_unit == unit
        per1[str(int(unit))] = float(hit1[mask].mean())
        per5[str(int(unit))] = float(hit5[mask].mean())
    majority_macro = float(np.mean(classes == int(majority_unit))) if len(classes) else 0.0
    return {"hit1": float(hit1.mean()), "hit5": float(hit5.mean()),
            "macro_recall1": float(np.mean(list(per1.values()))) if per1 else 0.0,
            "macro_recall5": float(np.mean(list(per5.values()))) if per5 else 0.0,
            "per_unit_recall1": per1, "per_unit_recall5": per5,
            "majority_macro_recall1_and_5": majority_macro}


def _functional_target_units(memory, bases, active_unit_ids, active_unit_centers):
    output = []
    for base in bases:
        local_id, _ = memory._nearest(base, centers=list(active_unit_centers))
        output.append(-1 if local_id is None else int(active_unit_ids[local_id]))
    return np.asarray(output, np.int64)


def _prepare_predictions(memory, active_entry_ids, q_state, q_key, query_actions, query_eps,
                         query_initials, query_bases, unit_targets, retrieval_kind,
                         retrieved_ids, retrieved_sims_or_distances, active_units,
                         active_unit_centers, all_arrays):
    mem_base, mem_raw = all_arrays["bases"][active_entry_ids], all_arrays["actions"][active_entry_ids]
    mem_eps, mem_initials = all_arrays["episodes"][active_entry_ids], all_arrays["initials"][active_entry_ids]
    mem_unit = all_arrays["units"][active_entry_ids]
    n = len(query_actions)
    if retrieval_kind == "oracle_latent":
        weights = _weights_from_distance(retrieved_sims_or_distances)
    else:
        weights = _weights_from_similarity(retrieved_sims_or_distances)
    q_prior_key = _normalize(q_key)
    _, q_residual = memory.prior(q_prior_key, q_state)
    flat_ids = retrieved_ids.reshape(-1)
    selected_bases = mem_base[flat_ids]
    selected_raw = mem_raw[flat_ids]
    repeated_res = np.repeat(np.asarray(q_residual, np.float32), 5, axis=0)
    repeated_key = np.repeat(q_prior_key, 5, axis=0)
    repeated_state = np.repeat(q_state, 5, axis=0)
    decoded_each = np.asarray(memory.decode(selected_bases, repeated_res, repeated_key, repeated_state), np.float32).reshape(n, 5, 10, 32)
    ap = np.einsum("nk,nkhd->nhd", weights, decoded_each)
    r = np.einsum("nk,nkhd->nhd", weights, selected_raw.reshape(n, 5, 10, 32))
    candidates_unit = mem_unit[retrieved_ids]
    unit_metrics = _macro_recall(unit_targets, candidates_unit, Counter(mem_unit[mem_unit >= 0]).most_common(1)[0][0])
    return {"Ap": ap, "R": r}, unit_metrics, weights


def _make_unit_centers(memory, active_entry_ids, active_unit_ids, all_arrays, *, use_original=False):
    units = np.asarray(sorted(unit for unit in set(map(int, active_unit_ids)) if unit >= 0), np.int64)
    centers = []
    for unit in units:
        bases = all_arrays["bases"][active_entry_ids[all_arrays["units"][active_entry_ids] == unit]]
        if use_original and len(bases) == sum(int(x) == unit for x in all_arrays["units"]):
            centers.append(np.asarray(memory._centers[unit], np.float32))
        else:
            centers.append(np.mean(bases, axis=0).astype(np.float32))
    return units, np.stack(centers)


def _metric_summary(pred, actions, row_errors):
    mse, mm = row_errors(pred, actions)
    return {"mse": float(np.mean(mse)), "endpoint_mm": float(np.mean(mm))}


def _evaluate_condition(name, query, memory, active_entry_ids, all_arrays, query_data,
                        model, config, output_path, *, seeds, batch_size, row_errors,
                        denoising_steps, cluster_bootstrap, task_holdout=False):
    import jax
    import jax.numpy as jnp
    from openpi.shared import nnx_utils
    from train import observation_batch

    n = len(query_data["actions"])
    q_eps = query_data["episodes"].astype(str)
    q_init = query_data["initials"].astype(str)
    q_state = np.asarray(query_data["state"], np.float32)
    q_actions = np.asarray(query_data["actions"], np.float32)
    q_bases = np.asarray(query_data["bases"], np.float32)
    mem_eps = all_arrays["episodes"][active_entry_ids]
    mem_init = all_arrays["initials"][active_entry_ids]
    mem_bases = all_arrays["bases"][active_entry_ids]
    mem_raw = all_arrays["actions"][active_entry_ids]
    mem_units = all_arrays["units"][active_entry_ids]
    unit_ids, unit_centers = _make_unit_centers(memory, active_entry_ids, mem_units, all_arrays,
                                                use_original=len(active_entry_ids) == len(memory.memory))
    true_units = _functional_target_units(memory, q_bases, unit_ids, unit_centers)
    majority_unit = int(Counter(mem_units[mem_units >= 0]).most_common(1)[0][0])

    # Baseline keys use only the currently available memory; learned keys share
    # a training mean and state normalization from that same memory partition.
    raw_mem_keys = all_arrays["keys"][active_entry_ids]
    centered_memory = _normalize(raw_mem_keys - raw_mem_keys.mean(axis=0, keepdims=True))
    centered_query = _normalize(np.asarray(query_data["centered_query"], np.float32)
                                - raw_mem_keys.mean(axis=0, keepdims=True))
    learned_memory, learned_query = query["learned_memory"], query["learned_query"]
    key_inputs = {"centered_no_training": (centered_memory, centered_query),
                  "learned_projection": (learned_memory, learned_query)}

    retrievals, anchors, key_metrics = {}, {}, {}
    for key_type, (key_memory, key_query) in key_inputs.items():
        ids, sims = _topk_cosine(key_memory, key_query, mem_eps, mem_init, q_eps, q_init)
        retrievals[key_type] = {"ids": ids, "sims": sims}
        key_metrics[key_type] = _macro_recall(true_units, mem_units[ids], majority_unit)
        key_metrics[key_type]["score_kind"] = "learned cosine score" if key_type == "learned_projection" else "cosine"

    # Two oracle rankings: nearest behavior latent for Ap and nearest raw action for R.
    latent_ids, latent_d = _topk_distance(mem_bases, q_bases, mem_eps, mem_init, q_eps, q_init)
    action_ids, action_d = _topk_distance(mem_raw.reshape(len(mem_raw), -1), q_actions.reshape(n, -1),
                                           mem_eps, mem_init, q_eps, q_init)
    retrievals["oracle"] = {"ids": latent_ids, "sims": latent_d}
    oracle_unit_metrics = _macro_recall(true_units, mem_units[latent_ids], majority_unit)
    action_oracle_unit_metrics = _macro_recall(true_units, mem_units[action_ids], majority_unit)
    key_metrics["oracle_latent"] = oracle_unit_metrics
    key_metrics["oracle_action"] = action_oracle_unit_metrics

    for key_type in ("centered_no_training", "learned_projection"):
        predictions, _, _ = _prepare_predictions(
            memory, active_entry_ids, q_state, query_data["original_key"], q_actions,
            q_eps, q_init, q_bases, true_units, key_type, retrievals[key_type]["ids"],
            retrievals[key_type]["sims"], unit_ids, unit_centers, all_arrays,
        )
        anchors[key_type] = predictions
    # The latent oracle is a one-best base for Ap; action oracle is one-best raw chunk for R.
    latent_weights = _weights_from_distance(latent_d)
    action_weights = _weights_from_distance(action_d)
    prior_key = _normalize(query_data["original_key"])
    _, prior_res = memory.prior(prior_key, q_state)
    latent_b = mem_bases[latent_ids[:, 0]]
    oracle_ap = np.asarray(memory.decode(latent_b, prior_res, prior_key, q_state), np.float32).reshape(n, 10, 32)
    oracle_r = mem_raw[action_ids[:, 0]]
    anchors["oracle"] = {"Ap": oracle_ap, "R": oracle_r}

    anchor_only = {}
    for key_type, pair in anchors.items():
        anchor_only[key_type] = {kind: _metric_summary(pair[kind], q_actions, row_errors) for kind in ANCHOR_NAMES}

    sample = nnx_utils.module_jit(model.sample_actions, static_argnames=("num_steps", "t0"))
    obs_all = observation_batch(query_data["observation"], config)
    errors = {}
    rng_noise = [np.random.default_rng(20261029 + i).standard_normal((n, 10, 32)).astype(np.float32)
                 for i in range(seeds)]
    base_mse = np.zeros((seeds, n), np.float32)
    base_mm = np.zeros((seeds, n), np.float32)
    for si in range(seeds):
        for lo in range(0, n, batch_size):
            hi = min(lo + batch_size, n)
            obs = jax.tree.map(lambda value: value[lo:hi], obs_all)
            eps = jnp.asarray(rng_noise[si][lo:hi])
            pred = sample(jax.random.key(0), obs, num_steps=10, noise=eps, x_src=eps, t0=1.0)
            base_mse[si, lo:hi], base_mm[si, lo:hi] = row_errors(np.asarray(pred), q_actions[lo:hi])
    for key_type, pair in anchors.items():
        for anchor_name in ANCHOR_NAMES:
            for t0 in ACTIVE_T0:
                if t0 == 1.0:
                    # The exact same baseline prediction is repeated for pairing.
                    errors[(key_type, anchor_name, t0)] = (base_mse.copy(), base_mm.copy())
                    continue
                mse = np.zeros((seeds, n), np.float32)
                mm = np.zeros((seeds, n), np.float32)
                anchor = np.asarray(pair[anchor_name], np.float32)
                for si in range(seeds):
                    for lo in range(0, n, batch_size):
                        hi = min(lo + batch_size, n)
                        obs = jax.tree.map(lambda value: value[lo:hi], obs_all)
                        eps = jnp.asarray(rng_noise[si][lo:hi])
                        x_src = (1.0 - t0) * jnp.asarray(anchor[lo:hi]) + t0 * eps
                        pred = sample(jax.random.key(0), obs, num_steps=denoising_steps(t0),
                                      noise=eps, x_src=x_src, t0=float(t0))
                        mse[si, lo:hi], mm[si, lo:hi] = row_errors(np.asarray(pred), q_actions[lo:hi])
                errors[(key_type, anchor_name, t0)] = (mse, mm)

    splits = {"all": np.ones(n, bool)}
    if "partition" in query_data:
        parts = np.asarray(query_data["partition"]).astype(str)
        splits.update({"fit": parts == "fit", "select": parts == "select"})
    out_grid = {}
    baseline = {split: {"mse": float(base_mse.mean(axis=0)[mask].mean()),
                        "endpoint_mm": float(base_mm.mean(axis=0)[mask].mean()),
                        "rows": int(mask.sum()), "episodes": int(len(set(q_eps[mask])))}
                for split, mask in splits.items()}
    for key_type in anchors:
        out_grid[key_type] = {}
        for anchor_name in ANCHOR_NAMES:
            out_grid[key_type][anchor_name] = {}
            for t0 in ACTIVE_T0:
                row_mse, row_mm = errors[(key_type, anchor_name, t0)]
                mean_mse = row_mse.mean(axis=0)
                mean_mm = row_mm.mean(axis=0)
                out_grid[key_type][anchor_name][str(t0)] = {}
                for split, mask in splits.items():
                    d_mse = mean_mse - base_mse.mean(axis=0)
                    d_mm = mean_mm - base_mm.mean(axis=0)
                    dm, dlo, dhi = cluster_bootstrap(d_mse[mask], q_eps[mask], reps=2000,
                                                      seed=711 + int(t0 * 100) + len(split))
                    dmm, mmlo, mmhi = cluster_bootstrap(d_mm[mask], q_eps[mask], reps=2000,
                                                        seed=991 + int(t0 * 100) + len(split))
                    out_grid[key_type][anchor_name][str(t0)][split] = {
                        "mse": float(mean_mse[mask].mean()), "endpoint_mm": float(mean_mm[mask].mean()),
                        "delta_mse_vs_t0_1": dm, "delta_mse_ci95": [dlo, dhi],
                        "delta_mm_vs_t0_1": dmm, "delta_mm_ci95": [mmlo, mmhi],
                        "rows": int(mask.sum()), "episodes": int(len(set(q_eps[mask]))),
                    }

    return {"rows": n, "episodes": int(len(set(q_eps))), "task_holdout": task_holdout,
            "majority_unit": majority_unit, "correct_units_present": int(len(set(true_units[true_units >= 0]))),
            "retrieval": key_metrics, "anchor_only": anchor_only, "baseline_t0_1": baseline,
            "denoising": out_grid, "oracle_anchor_definitions": {
                "Ap": "nearest eligible base to b_gt in 12-D Euclidean latent space, Dec with query prior residual",
                "R": "nearest eligible normalized action chunk in 10x32 Euclidean space",
            }}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heldout", default=DEFAULT_HELDOUT)
    ap.add_argument("--query-keys", default=DEFAULT_QUERY_KEYS)
    ap.add_argument("--rows", default=DEFAULT_ROWS)
    ap.add_argument("--memory", default=DEFAULT_MEMORY)
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--output", required=True)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--task-episodes", type=int, default=2)
    ap.add_argument("--task-rows-per-episode", type=int, default=8)
    ap.add_argument("--eval-batch-size", type=int, default=4)
    ap.add_argument("--gpu-fraction", type=float, default=0.08)
    ap.add_argument("--resume", action="store_true", help="reuse projector checkpoints in an existing incomplete output run")
    args = ap.parse_args()
    if args.steps < 1 or args.seeds < 1:
        raise ValueError("steps and seeds must be positive")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", str(args.gpu_fraction))
    out = Path(args.output)
    if args.resume:
        if not out.is_dir() or (out / "results.json").exists():
            raise FileExistsError("resume requires an existing incomplete run with no results.json")
    else:
        out.mkdir(parents=True, exist_ok=False)
    start_time = time.perf_counter()
    _, row_errors, cluster_bootstrap = _load_offline_helpers()

    import torch
    torch.set_num_threads(8)
    from memory import load_memory
    memory = load_memory(args.memory)
    held = _read_npz(args.heldout, ROW_FIELDS + ("partition", "skills", "skill_mask", "source_file"))
    q_keys = np.load(args.query_keys).astype(np.float32)
    with np.load(args.rows, allow_pickle=True) as z:
        row_fields = ("actions", "state", "episode_id", "init_state_id", "source_file", "frame")
        rows = {key: z[key] for key in row_fields}
    if len(memory.memory) != len(rows["actions"]):
        raise ValueError("M6 memory and LIBERO-90 cache rows are not aligned")
    if q_keys.shape[0] != len(held["actions"]):
        raise ValueError("heldout query keys do not align with observations")
    mem_episode = np.asarray([str(item["meta"]["episode_id"]) for item in memory.memory])
    mem_initial = np.asarray([str(item["meta"]["init_state_id"]) for item in memory.memory])
    if not set(np.asarray(held["episode_id"]).astype(str)).isdisjoint(mem_episode):
        raise ValueError("heldout episodes overlap memory")
    for i in range(len(memory.memory)):
        if mem_episode[i] != str(rows["episode_id"][i]) or mem_initial[i] != str(rows["init_state_id"][i]):
            raise ValueError(f"M6/cache row order mismatch at {i}")

    keys = _normalize(np.stack([np.asarray(item["key"], np.float32).reshape(-1) for item in memory.memory]))
    bases = np.stack([np.asarray(item["base"], np.float32).reshape(-1) for item in memory.memory])
    actions = np.stack([np.asarray(item["meta"]["raw_action"], np.float32) for item in memory.memory])
    units = np.asarray([-1 if item.get("unit_id") is None else int(item["unit_id"]) for item in memory.memory], np.int64)
    states = np.asarray(rows["state"], np.float32)
    episodes = np.asarray(rows["episode_id"]).astype(str)
    initials = np.asarray(rows["init_state_id"]).astype(str)
    source_files = np.asarray(rows["source_file"]).astype(str)
    all_arrays = {"keys": keys, "bases": bases, "actions": actions, "episodes": episodes,
                  "initials": initials, "units": units, "source_files": source_files}
    print(f"loaded {len(keys)} memory rows; raw task files={len(set(source_files))}", flush=True)

    # Task-level split: select 10 of the 89 cached task files, and use only 80-task
    # rows for both projector updates and candidate memory.
    task_query_ids, heldout_tasks, task_episode_plan = _sample_task_queries(
        source_files, episodes, 20261030, episodes_per_task=args.task_episodes,
        rows_per_episode=args.task_rows_per_episode)
    task_memory_ids = np.flatnonzero(~np.isin(source_files, heldout_tasks))
    if set(source_files[task_memory_ids]) & set(heldout_tasks):
        raise AssertionError("task holdout leaked into memory")

    # Train both projections on memory rows only. Condition (i) uses all LIBERO-90;
    # condition (ii) fits a distinct P on the remaining 80 task files.
    training_started = time.perf_counter()
    projector_paths = {name: out / f"{name}.pt" for name in ("projector_all90", "projector_task80")}
    training_report_path = out / "projector_training.json"
    if args.resume:
        if not all(path.exists() for path in projector_paths.values()):
            raise FileNotFoundError("resume run is missing projector checkpoint(s)")
        training_report = json.loads(training_report_path.read_text())
        train_all, train_80 = training_report["all90"], training_report["task80"]
        train_seconds = float(training_report["elapsed_seconds"])
    else:
        p_all, train_all = _fit_projector_memory(keys, states, bases, episodes, initials,
            np.arange(len(keys)), seed=20261030, steps=args.steps, batch_size=args.batch_size,
            output_path=projector_paths["projector_all90"])
        p_80, train_80 = _fit_projector_memory(keys, states, bases, episodes, initials,
            task_memory_ids, seed=20261031, steps=args.steps, batch_size=args.batch_size,
            output_path=projector_paths["projector_task80"])
        train_seconds = time.perf_counter() - training_started
        training_report = {"all90": train_all, "task80": train_80, "elapsed_seconds": train_seconds}
        training_report_path.write_text(json.dumps(training_report, indent=2) + "\n")

    # Compute projections for heldout rows from the same training means as their memory.
    def project_queries_from_saved(model_path, active_ids, q_key, q_state):
        mem_out, q_out, _ = _project_queries(model_path, keys[active_ids], states[active_ids], q_key, q_state, seed=0)
        return mem_out, q_out

    held_eps = np.asarray(held["episode_id"]).astype(str)
    held_init = np.asarray(held["init_state_id"]).astype(str)
    projection_cache_path = out / "projection_cache.npz"
    if args.resume and projection_cache_path.exists():
        with np.load(projection_cache_path, allow_pickle=False) as cached:
            full_mem_out, held_out = cached["full_mem_out"], cached["held_out"]
            task_mem_out, task_query_out = cached["task_mem_out"], cached["task_query_out"]
    else:
        full_mem_out, held_out = project_queries_from_saved(projector_paths["projector_all90"],
            np.arange(len(keys)), q_keys, np.asarray(held["state"], np.float32))
        task_mem_out, task_query_out, _ = _project_queries(projector_paths["projector_task80"],
            keys[task_memory_ids], states[task_memory_ids], keys[task_query_ids], states[task_query_ids], seed=0)
        np.savez_compressed(projection_cache_path, full_mem_out=full_mem_out, held_out=held_out,
                            task_mem_out=task_mem_out, task_query_out=task_query_out)
    all_query = {"actions": np.asarray(held["actions"], np.float32), "state": np.asarray(held["state"], np.float32),
                 "episodes": held_eps, "initials": held_init, "bases": np.asarray(memory.encode(held["actions"], held["state"])[0], np.float32),
                 "original_key": q_keys, "centered_query": q_keys, "learned_memory": full_mem_out,
                 "learned_query": held_out, "partition": np.asarray(held["partition"]).astype(str),
                 "observation": {key: held[key] for key in ("image_base", "image_left_wrist", "state", "tokenized_prompt",
                                                             "tokenized_prompt_mask", "skills", "skill_mask")}}

    # Load only the task-heldout observation rows needed for frozen-policy denoising.
    task_observation_cache = out / "task_observations.npz"
    if args.resume and task_observation_cache.exists():
        with np.load(task_observation_cache, allow_pickle=False) as cached:
            task_observation = {key: cached[key] for key in cached.files}
    else:
        with np.load(args.rows, allow_pickle=True) as z:
            task_observation = {"image_base": z["image_base"][task_query_ids],
                                "image_left_wrist": z["image_left_wrist"][task_query_ids],
                                "state": states[task_query_ids],
                                "tokenized_prompt": z["tokenized_prompt"][task_query_ids],
                                "tokenized_prompt_mask": z["tokenized_prompt_mask"][task_query_ids],
                                "skills": z["skills"][task_query_ids],
                                "skill_mask": z["skill_mask"][task_query_ids]}
        np.savez_compressed(task_observation_cache, **task_observation)
    task_query = {"actions": actions[task_query_ids], "state": states[task_query_ids],
                  "episodes": episodes[task_query_ids], "initials": initials[task_query_ids],
                  "bases": bases[task_query_ids], "original_key": keys[task_query_ids],
                  "centered_query": keys[task_query_ids], "learned_memory": task_mem_out,
                  "learned_query": task_query_out,
                  "observation": task_observation}

    import jax
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"offline denoising requires GPU; backend={jax.default_backend()}")
    from smoke import load_model
    model, config = load_model(args.checkpoint)
    all_ids = np.arange(len(memory.memory), dtype=np.int64)
    all_fit = _evaluate_condition("heldout_known_tasks", all_query, memory, all_ids, all_arrays,
                                  all_query, model, config, out, seeds=args.seeds, batch_size=args.eval_batch_size,
                                  row_errors=row_errors, denoising_steps=_load_offline_helpers()[0],
                                  cluster_bootstrap=cluster_bootstrap)
    task_result = _evaluate_condition("heldout_10_tasks", task_query, memory, task_memory_ids, all_arrays,
                                      task_query, model, config, out, seeds=args.seeds, batch_size=args.eval_batch_size,
                                      row_errors=row_errors, denoising_steps=_load_offline_helpers()[0],
                                      cluster_bootstrap=cluster_bootstrap, task_holdout=True)
    result = {
        "heldout_known_tasks": all_fit,
        "heldout_10_tasks": task_result,
        "training": {"all90": train_all, "task80": train_80, "elapsed_seconds": train_seconds,
                     "method": "CPU PyTorch MLP: concat centered/L2 VLM key + separately L2-normalized proprio; 256-GELU-128, normalized output; multi-positive InfoNCE."},
        "task_holdout": {"cached_tasks": int(len(set(source_files))), "heldout_tasks": heldout_tasks,
                         "memory_task_count": int(len(set(source_files[task_memory_ids]))),
                         "episode_plan": task_episode_plan, "sampled_rows": int(len(task_query_ids)),
                         "sampling": f"{args.task_episodes} episodes/task, at most {args.task_rows_per_episode} evenly spaced cached rows/episode",
                         "seed": 20261030},
        "retrieval": {"centered_no_training": "cosine of memory-mean-centered L2 VLM key",
                      "learned_projection": "trained P(key,state), cosine top-5",
                      "oracle": "O1 nearest eligible M6 base for Ap; O2 nearest eligible normalized action chunk for R",
                      "tau": TAU, "exclude": "same episode_id OR init_state_id"},
        "data": {"memory_entries": len(memory.memory), "heldout_observations": int(len(held["actions"])),
                 "known_task_episode_overlap": 0, "image_pool_available_for_full_memory": False,
                 "image_pool_note": "image-only VLM averages are not stored for all memory rows; this run uses cached full-token mean plus normalized state.",
                 "task_holdout_scope": "The P training set and candidate memory exclude the 10 selected tasks. Frozen policy and M1 codec/prior are the existing LIBERO-90 fit; task-holdout estimates isolate retrieval-key generalization, not a fully unseen action codec."},
        "denoising": {"t0": list(ACTIVE_T0), "seeds": args.seeds,
                      "noise": "shared per-query Gaussian noise across keys/anchors/t0, seeds 20261029 onward",
                      "sampler": "same sample_actions, denoising_steps, row_errors and episode bootstrap helper definitions as anchor_v2_offline.py (loaded without executing main)",
                      "ci": "paired t0 minus t0=1, bootstrap by episode, 2000 replicates, 95% percentile"},
        "elapsed_seconds": time.perf_counter() - start_time,
    }
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    (out / "run_config.json").write_text(json.dumps({"args": vars(args), "memory": args.memory,
        "heldout": args.heldout, "rows": args.rows, "checkpoint": args.checkpoint}, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
