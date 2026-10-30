#!/usr/bin/env python3
"""Small held-out comparison of frozen π0.5 prefix pooling choices."""
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
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

from research_paths import REPO_ROOT as RESEARCH_REPO_ROOT, STATE_DIR as RESEARCH_STATE_DIR
DEFAULT_HELDOUT = str(RESEARCH_STATE_DIR / 'outputs/memory-pilot/anchor-s2-calibrated-resume-20261028T215014Z/calibration/fit-v2/heldout-observations.npz')
DEFAULT_ROWS = str(RESEARCH_STATE_DIR / 'outputs/memory-pilot/anchor-s2-train-20261028T072152Z/libero90_rows.npz')
DEFAULT_MEMORY = str(RESEARCH_STATE_DIR / 'outputs/memory-pilot/anchor-s2-calibrated-resume-20261028T215014Z/memory/initial_libero90.pkl')
DEFAULT_CHECKPOINT = str(RESEARCH_STATE_DIR / 'checkpoints/pi05_libero_moe_skill_4_90/moe_balance_4_32_90_20000/19999/params')
ROW_FIELDS = (
    "state", "actions", "episode_id", "init_state_id", "image_base", "image_left_wrist",
    "tokenized_prompt", "tokenized_prompt_mask", "source_file", "frame",
)


def _read_npz(path, fields):
    with np.load(path, allow_pickle=True) as archive:
        missing = set(fields) - set(archive.files)
        if missing:
            raise ValueError(f"{path} is missing fields: {sorted(missing)}")
        return {name: archive[name] for name in fields}


def _normalize(x):
    x = np.asarray(x, np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def _whitener(memory_full_keys, query_full_keys, *, shrinkage=0.1, variance_fraction=0.95):
    """Regularized full-covariance ZCA whitening via the sample-space SVD."""
    x = np.asarray(memory_full_keys, np.float64)
    mean = x.mean(axis=0)
    centered = x - mean
    _, singular, vectors = np.linalg.svd(centered, full_matrices=False)
    eigenvalues = singular**2 / max(len(x) - 1, 1)
    cumulative = np.cumsum(eigenvalues) / max(float(eigenvalues.sum()), 1e-12)
    rank = min(len(eigenvalues), int(np.searchsorted(cumulative, variance_fraction) + 1))
    eigenvalues = eigenvalues[:rank]
    vectors = vectors[:rank]
    # Shrink toward isotropic covariance. This is important because a 512-row
    # subset has rank at most 511 while the PaliGemma width is 2048.
    ridge = shrinkage * float(np.sum(eigenvalues)) / x.shape[1]
    ridge = max(ridge, 1e-8)

    def apply(values):
        z = np.asarray(values, np.float64) - mean
        projected = z @ vectors.T
        scales = 1.0 / np.sqrt((1.0 - shrinkage) * eigenvalues + ridge) - 1.0 / np.sqrt(ridge)
        whitened = z / np.sqrt(ridge) + (projected * scales) @ vectors
        return _normalize(whitened.astype(np.float32))

    return apply(memory_full_keys), apply(query_full_keys), {
        "method": "regularized ZCA; 95% sample PCA subspace plus isotropic shrinkage",
        "sample_rank": int(rank), "variance_fraction": float(cumulative[rank - 1]),
        "shrinkage": shrinkage, "ridge": ridge,
    }


def _retrieval_metrics(query_keys, memory_keys, entries, query_episode, query_init, oracle_units, majority_unit,
                       *, temperature):
    memory_keys = _normalize(memory_keys)
    query_keys = _normalize(query_keys)
    entry_episode = np.asarray([str(row["meta"]["episode_id"]) for row in entries])
    entry_init = np.asarray([str(row["meta"]["init_state_id"]) for row in entries])
    entry_units = [row.get("unit_id") for row in entries]
    top_indices, top_sims, top_weights = [], [], []
    top1_correct, top5_correct = [], []
    for i, query in enumerate(query_keys):
        similarities = memory_keys @ query
        eligible = (entry_episode != str(query_episode[i])) & (entry_init != str(query_init[i]))
        similarities[~eligible] = -np.inf
        order = np.argsort(-similarities, kind="stable")[:min(5, int(eligible.sum()))]
        sims = similarities[order]
        distances = np.sqrt(np.maximum(2.0 - 2.0 * sims, 0.0))
        logits = -distances / max(float(temperature), 1e-8)
        weights = np.exp(logits - logits.max())
        weights /= weights.sum()
        target = oracle_units[i]
        units = [None if entry_units[j] is None else int(entry_units[j]) for j in order]
        top_indices.append(order)
        top_sims.append(sims)
        top_weights.append(weights.astype(np.float32))
        top1_correct.append(target is not None and units[0] == int(target))
        top5_correct.append(target is not None and int(target) in units)
    return {
        "top_indices": top_indices, "top_sims": top_sims, "top_weights": top_weights,
        "top1_correct": np.asarray(top1_correct, bool), "top5_correct": np.asarray(top5_correct, bool),
        "majority_correct": np.asarray([u is not None and int(u) == majority_unit for u in oracle_units], bool),
    }


def _quantile_summary(values):
    x = np.asarray(values, np.float64)
    return {f"q{p}": float(np.quantile(x, p / 100)) for p in (10, 25, 50, 75, 90)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--heldout", default=DEFAULT_HELDOUT)
    parser.add_argument("--libero90-rows", default=DEFAULT_ROWS)
    parser.add_argument("--memory", default=DEFAULT_MEMORY)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", required=True)
    parser.add_argument("--memory-rows", type=int, default=512, help="uniform subset of M6 entries")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--memory-fraction", type=float, default=0.18)
    args = parser.parse_args()
    if args.memory_rows < 5 or args.batch_size < 1:
        raise ValueError("memory-rows must be >=5 and batch-size must be positive")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", str(args.memory_fraction))

    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import torch
    torch.set_num_threads(8)
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"key check expects a GPU for the frozen VLM, got {jax.default_backend()}")

    from openpi.models import model as _model
    from openpi.models.pi0_moe_skill import make_attn_mask
    from openpi.shared import nnx_utils
    from memory import load_memory, validate_key_text_source
    from smoke import load_model
    from train import observation_batch

    started = time.perf_counter()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    held = _read_npz(args.heldout, ROW_FIELDS + ("partition", "suite"))
    rows = _read_npz(args.libero90_rows, ROW_FIELDS + ("suite", "key_text_source", "normalization_sha256"))
    if not np.all(np.asarray(rows["suite"]).astype(str) == "libero90"):
        raise ValueError("memory rows must be LIBERO-90 only")
    if str(np.asarray(rows["key_text_source"]).item()) != "instruction_only":
        raise ValueError("this small check expects the current instruction_only memory")
    memory = load_memory(args.memory)
    if len(memory.memory) != len(rows["actions"]):
        raise ValueError(f"M6 rows ({len(memory.memory)}) do not align with source rows ({len(rows['actions'])})")
    if validate_key_text_source({"key_text_source": np.asarray("instruction_only")}) != memory.key_text_source:
        raise ValueError("held-out queries and memory have different text provenance")
    held_episodes = set(np.asarray(held["episode_id"]).astype(str))
    all_memory_episodes = set(str(row["meta"]["episode_id"]) for row in memory.memory)
    overlap = sorted(held_episodes & all_memory_episodes)
    if overlap:
        raise ValueError(f"held-out episode leakage into M6 memory: {len(overlap)} IDs, e.g. {overlap[:3]}")

    rng = np.random.default_rng(args.seed)
    count = min(args.memory_rows, len(memory.memory))
    memory_indices = np.sort(rng.choice(len(memory.memory), size=count, replace=False))
    # M6 was seeded in source-row order. Verify that sampled memory entries
    # really correspond to these source observations before recomputing keys.
    for index in memory_indices[::max(1, count // 64)]:
        item = memory.memory[int(index)]
        if str(item["meta"]["episode_id"]) != str(rows["episode_id"][index]):
            raise ValueError(f"M6/source episode order mismatch at row {index}")
        if str(item["meta"]["init_state_id"]) != str(rows["init_state_id"][index]):
            raise ValueError(f"M6/source initial-state order mismatch at row {index}")
        if not np.allclose(item["meta"]["raw_action"], rows["actions"][index], atol=2e-6):
            raise ValueError(f"M6/source raw-action mismatch at row {index}")
    if str(np.asarray(rows["normalization_sha256"]).item()) != memory.normalization_sha256:
        raise ValueError("LIBERO-90 rows and M6 memory have different action normalization")

    model, config = load_model(args.checkpoint)

    class PrefixPools(nnx.Module):
        def __init__(self, frozen_model):
            self.model = frozen_model

        def encode(self, obs):
            obs = _model.preprocess_observation(None, obs, train=False)
            token_list, mask_list = [], []
            image_count = 0
            for name, image in obs.images.items():
                image_tokens, _ = self.model.PaliGemma.img(image, train=False)
                token_list.append(image_tokens)
                image_mask = jnp.repeat(obs.image_masks[name][:, None], image_tokens.shape[1], axis=1)
                mask_list.append(image_mask)
                image_count += image_tokens.shape[1]
            if obs.tokenized_prompt is not None:
                text_tokens = self.model.PaliGemma.llm(obs.tokenized_prompt, method="embed")
                token_list.append(text_tokens)
                mask_list.append(obs.tokenized_prompt_mask)
            prefix = jnp.concatenate(token_list, axis=1)
            valid = jnp.concatenate(mask_list, axis=1)
            attn = make_attn_mask(valid, jnp.zeros_like(valid))
            positions = jnp.cumsum(valid, axis=-1) - 1
            (hidden, _), _, _ = self.model.PaliGemma.llm(
                [prefix, None], skill_embeds=None, skill_mask=None, mask=attn, positions=positions
            )
            image_mask = jnp.arange(valid.shape[1])[None, :] < image_count
            image_mask = jnp.broadcast_to(image_mask, valid.shape) & valid
            text_mask = valid & ~image_mask
            # Drop PaliGemma BOS/EOS from the instruction-only pool; retain
            # them in the full-pool and last-valid-token candidates.
            text_positions = jnp.arange(obs.tokenized_prompt.shape[1])[None, :]
            prompt_valid = obs.tokenized_prompt_mask
            first = jnp.argmax(prompt_valid, axis=1)[:, None]
            last = jnp.maximum(jnp.sum(prompt_valid, axis=1) - 1, 0)[:, None]
            token_ids = obs.tokenized_prompt
            is_bos = (text_positions == first) & (token_ids == 2)
            is_eos = (text_positions == last) & (token_ids == 1)
            content_prompt_mask = prompt_valid & ~is_bos & ~is_eos
            content_mask = jnp.concatenate(
                [jnp.zeros((valid.shape[0], image_count), jnp.bool_), content_prompt_mask], axis=1
            )

            def pool(mask):
                weights = mask.astype(hidden.dtype)[..., None]
                pooled = jnp.sum(hidden * weights, axis=1) / jnp.maximum(jnp.sum(weights, axis=1), 1.0)
                return pooled / jnp.maximum(jnp.linalg.norm(pooled, axis=-1, keepdims=True), 1e-8)

            last_index = jnp.maximum(jnp.sum(valid, axis=1) - 1, 0)
            last_hidden = jnp.take_along_axis(hidden, last_index[:, None, None], axis=1)[:, 0]
            last_hidden = last_hidden / jnp.maximum(jnp.linalg.norm(last_hidden, axis=-1, keepdims=True), 1e-8)
            return jnp.stack([pool(valid), pool(image_mask), pool(text_mask & content_mask), last_hidden], axis=1)

    encode_pools = nnx_utils.module_jit(PrefixPools(model).encode)
    selected_memory = [memory.memory[int(i)] for i in memory_indices]
    n_query = len(held["actions"])
    joined = {}
    for name in ROW_FIELDS:
        joined[name] = np.concatenate([rows[name][memory_indices], held[name]], axis=0)
    obs = observation_batch(joined, config)
    pool_chunks = []
    for start in range(0, len(joined["actions"]), args.batch_size):
        stop = min(start + args.batch_size, len(joined["actions"]))
        batch = jax.tree.map(lambda value: value[start:stop], obs)
        features = encode_pools(batch)
        jax.block_until_ready(features)
        pool_chunks.append(np.asarray(features, np.float32))
        print(f"frozen VLM prefix rows: {stop}/{len(joined['actions'])}", flush=True)
    pooled = np.concatenate(pool_chunks, axis=0)
    memory_pools, query_pools = pooled[:count], pooled[count:]
    # In the current π0.5 config, state is a separate proprio field, not a
    # PaliGemma prefix token: pi05=True and TokenizePrompt(discrete_state_input=False).
    state_token_available = False
    whiten_memory, whiten_query, whiten_info = _whitener(memory_pools[:, 0], query_pools[:, 0])
    key_names = ["full_mean", "image_mean", "instruction_mean", "last_valid", "full_mean_whitened"]
    memory_keys = {
        "full_mean": memory_pools[:, 0], "image_mean": memory_pools[:, 1],
        "instruction_mean": memory_pools[:, 2], "last_valid": memory_pools[:, 3],
        "full_mean_whitened": whiten_memory,
    }
    query_keys = {
        "full_mean": query_pools[:, 0], "image_mean": query_pools[:, 1],
        "instruction_mean": query_pools[:, 2], "last_valid": query_pools[:, 3],
        "full_mean_whitened": whiten_query,
    }
    original_memory_keys = _normalize(np.stack([row["key"] for row in selected_memory]))
    normalized_recomputed_full = _normalize(memory_keys["full_mean"])
    stored_full_cos = np.sum(original_memory_keys * normalized_recomputed_full, axis=1)

    query_actions = np.asarray(held["actions"], np.float32)
    query_state = np.asarray(held["state"], np.float32)
    query_episode = np.asarray(held["episode_id"]).astype(str)
    query_init = np.asarray(held["init_state_id"]).astype(str)
    encoded_base, _ = memory.encode(query_actions, query_state)
    oracle_units, oracle_distances = [], []
    for base in encoded_base:
        unit, distance = memory._nearest(base)
        oracle_units.append(unit)
        oracle_distances.append(distance)
    majority_unit = Counter(int(row["unit_id"]) for row in memory.memory if row.get("unit_id") is not None).most_common(1)[0][0]

    # Isolate retrieval-key effects: the trained M1 prior/decoder still sees
    # the original full-mean feature distribution for every candidate method.
    full_query_key = query_keys["full_mean"]
    _, prior_residual = memory.prior(full_query_key, query_state)
    unit_members: dict[int, list[dict]] = defaultdict(list)
    for row in memory.memory:
        if row.get("unit_id") is not None:
            unit_members[int(row["unit_id"])].append(row)
    query_key_results, mse_results = {}, {}
    zero_actions = np.zeros_like(query_actions)
    mean_actions = np.broadcast_to(query_actions.mean(axis=0, keepdims=True), query_actions.shape)
    zero_mse = np.mean(np.square(zero_actions - query_actions), axis=(1, 2))
    mean_mse = np.mean(np.square(mean_actions - query_actions), axis=(1, 2))

    for name in key_names:
        result = _retrieval_metrics(
            query_keys[name], memory_keys[name], selected_memory, query_episode, query_init, oracle_units,
            majority_unit, temperature=memory.temperature,
        )
        query_key_results[name] = result
        pred_center = np.zeros_like(query_actions)
        pred_blend = np.zeros_like(query_actions)
        pred_sample = np.zeros_like(query_actions)
        pred_oracle = np.zeros_like(query_actions)
        action_valid = np.zeros((n_query,), bool)
        oracle_valid = np.asarray([u is not None for u in oracle_units], bool)
        sample_rng = np.random.default_rng(args.seed + 1)
        for i, (indices, weights) in enumerate(zip(result["top_indices"], result["top_weights"], strict=True)):
            units = defaultdict(float)
            entry_bases = []
            for index, weight in zip(indices, weights, strict=True):
                row = selected_memory[int(index)]
                entry_bases.append(np.asarray(row["base"], np.float32))
                unit = row.get("unit_id")
                if unit is not None:
                    units[int(unit)] += float(weight)
            if units:
                selected_unit = max(units, key=lambda unit: (units[unit], -unit))
                action_valid[i] = True
                candidates = unit_members[selected_unit]
                eligible = [row for row in candidates
                            if str(row["meta"].get("episode_id")) != query_episode[i]
                            and str(row["meta"].get("init_state_id")) != query_init[i]]
                bases = np.stack([row["base"] for row in eligible]).astype(np.float32)
                if len(eligible) == len(candidates):
                    center = np.asarray(memory._centers[selected_unit], np.float32)
                else:
                    center = bases.mean(axis=0)
                sigma = float(np.sqrt(np.mean(np.square(bases - center[None, :]))))
                pred_center[i] = memory.decode(
                    center[None], prior_residual[i:i + 1], full_query_key[i:i + 1], query_state[i:i + 1]
                )[0]
                sampled = center + sigma * sample_rng.normal(size=center.shape).astype(np.float32)
                pred_sample[i] = memory.decode(
                    sampled[None], prior_residual[i:i + 1], full_query_key[i:i + 1], query_state[i:i + 1]
                )[0]
            blend = np.asarray(weights, np.float32) @ np.stack(entry_bases)
            pred_blend[i] = memory.decode(
                blend[None], prior_residual[i:i + 1], full_query_key[i:i + 1], query_state[i:i + 1]
            )[0]
            if oracle_units[i] is not None:
                true_unit = int(oracle_units[i])
                pred_oracle[i] = memory.decode(
                    np.asarray(memory._centers[true_unit], np.float32)[None], prior_residual[i:i + 1],
                    full_query_key[i:i + 1], query_state[i:i + 1],
                )[0]
        mse_results[name] = {
            "rows": n_query, "retrieved_unit_rows": int(action_valid.sum()),
            "oracle_unit_rows": int(oracle_valid.sum()),
            "zero_action_mse": float(zero_mse.mean()), "heldout_mean_action_mse": float(mean_mse.mean()),
            "retrieved_unit_center_mse": float(np.mean(np.mean(np.square(pred_center[action_valid] - query_actions[action_valid]), axis=(1, 2)))) if action_valid.any() else None,
            "retrieved_top_entries_weighted_base_mse": float(np.mean(np.mean(np.square(pred_blend - query_actions), axis=(1, 2)))),
            "retrieved_unit_gaussian_sample_mse": float(np.mean(np.mean(np.square(pred_sample[action_valid] - query_actions[action_valid]), axis=(1, 2)))) if action_valid.any() else None,
            "oracle_correct_unit_center_mse": float(np.mean(np.mean(np.square(pred_oracle[oracle_valid] - query_actions[oracle_valid]), axis=(1, 2)))) if oracle_valid.any() else None,
        }

    summary = {}
    for name in key_names:
        result = query_key_results[name]
        sims = np.asarray([row[0] for row in result["top_sims"]], np.float64)
        gaps = np.asarray([row[0] - row[-1] for row in result["top_sims"]], np.float64)
        valid_targets = np.asarray([u is not None for u in oracle_units], bool)
        top1 = float(result["top1_correct"][valid_targets].mean()) if valid_targets.any() else 0.0
        top5 = float(result["top5_correct"][valid_targets].mean()) if valid_targets.any() else 0.0
        majority = float(result["majority_correct"][valid_targets].mean()) if valid_targets.any() else 0.0
        summary[name] = {
            "top1_cosine_quantiles": _quantile_summary(sims),
            "top1_minus_top5_cosine_gap_quantiles": _quantile_summary(gaps),
            "correct_unit_top1": top1, "correct_unit_top5": top5,
            "majority_unit_baseline_top1": majority,
            "top1_minus_majority": top1 - majority, "top5_minus_majority": top5 - majority,
            "anchor_mse": mse_results[name],
        }

    report = {
        "heldout": str(Path(args.heldout).resolve()), "memory": str(Path(args.memory).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()), "rows_path": str(Path(args.libero90_rows).resolve()),
        "heldout_rows": int(n_query), "heldout_episodes": int(len(held_episodes)),
        "memory_entries_total": len(memory.memory), "memory_entries_reencoded_uniform_subset": int(count),
        "heldout_episode_overlap_with_full_memory": len(overlap),
        "source_row_alignment_verified_for": int(len(range(0, count, max(1, count // 64)))),
        "subset_seed": args.seed, "heldout_partition_values": sorted(set(np.asarray(held["partition"]).astype(str))),
        "query_text_source": "instruction_only", "state_token_candidate_available": state_token_available,
        "state_token_unavailable_reason": "pi05=True and TokenizePrompt(discrete_state_input=False): VLM prefix is image+prompt; proprio is a separate field, not a prefix token.",
        "pooling_features": "single frozen VLM forward per row; all/image/instruction/last-valid pooled from identical final hidden states; each L2 normalized",
        "whitening": whiten_info,
        "full_mean_recomputed_vs_stored_memory_key_cosine": _quantile_summary(stored_full_cos),
        "full_mean_recomputed_l2_norms_before_fp32_renormalization": _quantile_summary(
            np.linalg.norm(memory_pools[:, 0], axis=1)
        ),
        "m6": {"units": len(memory._centers), "temperature": float(memory.temperature),
               "majority_unit": int(majority_unit), "majority_unit_count": int(Counter(
                   int(row["unit_id"]) for row in memory.memory if row.get("unit_id") is not None
               )[majority_unit]), "oracle_assignment_distance_quantiles": _quantile_summary(oracle_distances)},
        "feature_results": summary,
        "decoder_prior_control": "all anchor Dec/prior inputs use the original full-mean query key to isolate retrieval key changes; retrieved bases/unit IDs vary by candidate key",
        "action_mse_space": "normalized LIBERO action space [10,32], per-row mean squared error",
        "elapsed_seconds": time.perf_counter() - started,
        "partial_memory_note": "Only the uniform subset is re-encoded/re-keyed; full M6 unit centers and assignments are retained for oracle labels and majority baseline.",
    }
    (out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
