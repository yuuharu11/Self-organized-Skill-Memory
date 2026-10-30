#!/usr/bin/env python3
"""Centered and state-augmented S2 key check; appends a separate result file."""
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

from key_check import DEFAULT_CHECKPOINT, DEFAULT_HELDOUT, DEFAULT_MEMORY, DEFAULT_ROWS, ROW_FIELDS
from key_check import _normalize, _read_npz, _retrieval_metrics, _quantile_summary


def _summary_for(name, query_keys, memory_keys, entries, query_episode, query_init,
                 oracle_units, majority_unit, temperature):
    result = _retrieval_metrics(query_keys, memory_keys, entries, query_episode, query_init,
                               oracle_units, majority_unit, temperature=temperature)
    sims = np.asarray([row[0] for row in result["top_sims"]], np.float64)
    gaps = np.asarray([row[0] - row[-1] for row in result["top_sims"]], np.float64)
    valid = np.asarray([unit is not None for unit in oracle_units], bool)
    top1 = float(result["top1_correct"][valid].mean())
    top5 = float(result["top5_correct"][valid].mean())
    majority = float(result["majority_correct"][valid].mean())
    true_units = np.asarray(oracle_units, dtype=object)[valid].astype(int)
    class_units = np.unique(true_units)
    top1_unit = np.asarray([
        None if entries[int(indices[0])].get("unit_id") is None else int(entries[int(indices[0])]["unit_id"])
        for indices in result["top_indices"]
    ], dtype=object)[valid]
    top5_units = [
        {int(entries[int(index)]["unit_id"]) for index in indices
         if entries[int(index)].get("unit_id") is not None}
        for indices in result["top_indices"]
    ]
    per_unit_top1, per_unit_top5 = {}, {}
    for unit in class_units:
        class_mask = true_units == unit
        per_unit_top1[str(int(unit))] = float(np.mean(top1_unit[class_mask] == unit))
        valid_indices = np.flatnonzero(valid)
        class_rows = valid_indices[class_mask]
        per_unit_top5[str(int(unit))] = float(np.mean([int(unit) in top5_units[i] for i in class_rows]))
    macro_top1 = float(np.mean(list(per_unit_top1.values()))) if per_unit_top1 else 0.0
    macro_top5 = float(np.mean(list(per_unit_top5.values()))) if per_unit_top5 else 0.0
    macro_majority = float(np.mean(class_units == majority_unit)) if len(class_units) else 0.0
    return result, {
        "top1_cosine_quantiles_q5_q50_q95": {
            f"q{p}": float(np.quantile(sims, p / 100)) for p in (5, 50, 95)
        },
        "top1_minus_top5_cosine_gap_quantiles_q5_q50_q95": {
            f"q{p}": float(np.quantile(gaps, p / 100)) for p in (5, 50, 95)
        },
        "correct_unit_top1": top1, "correct_unit_top5": top5,
        "macro_recall_top1": macro_top1, "macro_recall_top5": macro_top5,
        "macro_recall_per_unit_top1": per_unit_top1,
        "macro_recall_per_unit_top5": per_unit_top5,
        "majority_unit_macro_recall_top1_and_top5": macro_majority,
        "majority_unit_baseline": majority,
        "top1_minus_majority": top1 - majority,
        "top5_minus_majority": top5 - majority,
    }


def _decode_error_metrics(prediction, target, translation_scales_m):
    # LIBERO uses the first seven channels; the remaining 25 are model padding.
    error = np.asarray(prediction, np.float32)[..., :7] - np.asarray(target, np.float32)[..., :7]
    terminal_mm = np.linalg.norm(np.sum(error[:, :, :3] * translation_scales_m[None, None, :], axis=1), axis=-1) * 1000.0
    return {"normalized_action_mse": float(np.mean(np.square(error))),
            "endpoint_offset_mm_mean": float(terminal_mm.mean()),
            "endpoint_offset_mm_q50": float(np.quantile(terminal_mm, 0.5)),
            "endpoint_offset_mm_q90": float(np.quantile(terminal_mm, 0.9))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--heldout", default=DEFAULT_HELDOUT)
    parser.add_argument("--libero90-rows", default=DEFAULT_ROWS)
    parser.add_argument("--memory", default=DEFAULT_MEMORY)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", required=True, help="existing key-check run directory")
    parser.add_argument("--memory-rows", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--memory-fraction", type=float, default=0.12)
    args = parser.parse_args()
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", str(args.memory_fraction))
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import torch
    torch.set_num_threads(8)
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"expected shared GPU for frozen VLM; backend={jax.default_backend()}")
    from openpi.models import model as _model
    from openpi.models.pi0_moe_skill import make_attn_mask
    from openpi.shared import nnx_utils
    from memory import load_memory, validate_key_text_source
    from smoke import load_model
    from train import observation_batch

    start_time = time.perf_counter()
    out = Path(args.output_dir)
    result_file = out / "extension-v3.json"
    if not out.is_dir():
        raise FileNotFoundError(f"expected existing run directory: {out}")
    if result_file.exists():
        raise FileExistsError(f"refusing to overwrite {result_file}")
    held = _read_npz(args.heldout, ROW_FIELDS + ("partition", "suite"))
    rows = _read_npz(args.libero90_rows, ROW_FIELDS + ("suite", "key_text_source", "normalization_sha256"))
    if not np.all(np.asarray(rows["suite"]).astype(str) == "libero90"):
        raise ValueError("only LIBERO-90 memory rows are accepted")
    if str(np.asarray(rows["key_text_source"]).item()) != "instruction_only":
        raise ValueError("expected instruction_only memory keys")
    memory = load_memory(args.memory)
    if len(memory.memory) != len(rows["actions"]):
        raise ValueError("M6 memory and source rows do not align")
    if validate_key_text_source({"key_text_source": np.asarray("instruction_only")}) != memory.key_text_source:
        raise ValueError("heldout and memory key text sources mismatch")
    held_eps = set(np.asarray(held["episode_id"]).astype(str))
    overlap = held_eps & {str(row["meta"]["episode_id"]) for row in memory.memory}
    if overlap:
        raise ValueError(f"heldout episode leakage: {len(overlap)} IDs")

    rng = np.random.default_rng(args.seed)
    count = min(args.memory_rows, len(memory.memory))
    memory_indices = np.sort(rng.choice(len(memory.memory), size=count, replace=False))
    for index in memory_indices[::max(1, count // 64)]:
        item = memory.memory[int(index)]
        if (str(item["meta"]["episode_id"]) != str(rows["episode_id"][index])
                or str(item["meta"]["init_state_id"]) != str(rows["init_state_id"][index])
                or not np.allclose(item["meta"]["raw_action"], rows["actions"][index], atol=2e-6)):
            raise ValueError(f"sampled M6/source row mismatch at {index}")
    if str(np.asarray(rows["normalization_sha256"]).item()) != memory.normalization_sha256:
        raise ValueError("action normalization mismatch")

    model, config = load_model(args.checkpoint)

    class PrefixPools(nnx.Module):
        def __init__(self, frozen_model):
            self.model = frozen_model

        def encode(self, obs):
            obs = _model.preprocess_observation(None, obs, train=False)
            token_list, mask_list = [], []
            image_count = 0
            for name, image in obs.images.items():
                tokens, _ = self.model.PaliGemma.img(image, train=False)
                token_list.append(tokens)
                mask_list.append(jnp.repeat(obs.image_masks[name][:, None], tokens.shape[1], axis=1))
                image_count += tokens.shape[1]
            text_tokens = self.model.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            token_list.append(text_tokens)
            mask_list.append(obs.tokenized_prompt_mask)
            prefix, valid = jnp.concatenate(token_list, axis=1), jnp.concatenate(mask_list, axis=1)
            attn = make_attn_mask(valid, jnp.zeros_like(valid))
            positions = jnp.cumsum(valid, axis=-1) - 1
            (hidden, _), _, _ = self.model.PaliGemma.llm(
                [prefix, None], skill_embeds=None, skill_mask=None, mask=attn, positions=positions
            )
            image_mask = (jnp.arange(valid.shape[1])[None, :] < image_count) & valid
            text_positions = jnp.arange(obs.tokenized_prompt.shape[1])[None, :]
            prompt_valid = obs.tokenized_prompt_mask
            first = jnp.argmax(prompt_valid, axis=1)[:, None]
            last_prompt = jnp.maximum(jnp.sum(prompt_valid, axis=1) - 1, 0)[:, None]
            is_bos = (text_positions == first) & (obs.tokenized_prompt == 2)
            is_eos = (text_positions == last_prompt) & (obs.tokenized_prompt == 1)
            content = prompt_valid & ~is_bos & ~is_eos
            content_mask = jnp.concatenate([jnp.zeros((valid.shape[0], image_count), bool), content], axis=1)

            def pool(mask):
                weights = mask.astype(hidden.dtype)[..., None]
                value = jnp.sum(hidden * weights, axis=1) / jnp.maximum(jnp.sum(weights, axis=1), 1.0)
                return value / jnp.maximum(jnp.linalg.norm(value, axis=-1, keepdims=True), 1e-8)

            last_index = jnp.maximum(jnp.sum(valid, axis=1) - 1, 0)
            final_token = jnp.take_along_axis(hidden, last_index[:, None, None], axis=1)[:, 0]
            final_token = final_token / jnp.maximum(jnp.linalg.norm(final_token, axis=-1, keepdims=True), 1e-8)
            return jnp.stack([pool(valid), pool(image_mask), pool(valid & ~image_mask & content_mask), final_token], axis=1)

    encode_pools = nnx_utils.module_jit(PrefixPools(model).encode)
    joined = {name: np.concatenate([rows[name][memory_indices], held[name]], axis=0) for name in ROW_FIELDS}
    obs = observation_batch(joined, config)
    pooled_batches = []
    for begin in range(0, len(joined["actions"]), args.batch_size):
        end = min(begin + args.batch_size, len(joined["actions"]))
        features = encode_pools(jax.tree.map(lambda value: value[begin:end], obs))
        jax.block_until_ready(features)
        pooled_batches.append(np.asarray(features, np.float32))
        print(f"frozen VLM prefix rows: {end}/{len(joined['actions'])}", flush=True)
    pooled = np.concatenate(pooled_batches, axis=0)
    mem_pools, query_pools = pooled[:count], pooled[count:]
    # Center each token-selection variant using memory keys only, never heldout queries.
    names = {"full_mean_centered": 0, "image_mean_centered": 1,
             "instruction_mean_centered": 2, "last_valid_centered": 3}
    memory_vlm, query_vlm, center_means = {}, {}, {}
    for name, column in names.items():
        center = mem_pools[:, column].mean(axis=0, keepdims=True)
        center_means[name] = center.astype(np.float32)
        memory_vlm[name] = _normalize(mem_pools[:, column] - center)
        query_vlm[name] = _normalize(query_pools[:, column] - center)
    state_memory = _normalize(np.asarray(rows["state"][memory_indices], np.float32))
    state_query = _normalize(np.asarray(held["state"], np.float32))
    selected_memory = [memory.memory[int(i)] for i in memory_indices]
    query_actions = np.asarray(held["actions"], np.float32)
    query_state = np.asarray(held["state"], np.float32)
    query_episode = np.asarray(held["episode_id"]).astype(str)
    query_init = np.asarray(held["init_state_id"]).astype(str)
    encoded_base, encoded_residual = memory.encode(query_actions, query_state)
    oracle_pairs = [memory._nearest(base) for base in encoded_base]
    oracle_units = [unit for unit, _ in oracle_pairs]
    oracle_distances = [distance for _, distance in oracle_pairs]
    majority_unit = Counter(int(row["unit_id"]) for row in memory.memory if row.get("unit_id") is not None).most_common(1)[0][0]
    majority_rate = float(np.mean([unit == majority_unit for unit in oracle_units]))

    # Select the strongest centered b-e query key on fit episodes only; report
    # select episodes separately to avoid evaluating that choice on its selector.
    fit_mask = np.asarray(held["partition"]).astype(str) == "fit"
    fit_indices = np.flatnonzero(fit_mask)
    selection_scores = {}
    for candidate in ("image_mean_centered", "instruction_mean_centered", "last_valid_centered"):
        fit_result = _retrieval_metrics(
            query_vlm[candidate][fit_mask], memory_vlm[candidate], selected_memory,
            query_episode[fit_mask], query_init[fit_mask], [oracle_units[i] for i in fit_indices],
            majority_unit, temperature=memory.temperature,
        )
        valid_fit = np.asarray([oracle_units[i] is not None for i in fit_indices], bool)
        selection_scores[candidate] = {
            "fit_top1": float(fit_result["top1_correct"][valid_fit].mean()),
            "fit_top5": float(fit_result["top5_correct"][valid_fit].mean()),
        }
    best_centered_vlm_key = max(
        selection_scores, key=lambda item: (selection_scores[item]["fit_top1"], selection_scores[item]["fit_top5"])
    )
    combined = {}
    # sqrt weights make w equal to the state block's fraction of squared norm.
    for weight in (0.25, 0.5, 0.75):
        name = f"best_centered_plus_state_w{weight:.2f}"
        memory_vlm_block = np.sqrt(1.0 - weight) * memory_vlm[best_centered_vlm_key]
        query_vlm_block = np.sqrt(1.0 - weight) * query_vlm[best_centered_vlm_key]
        memory_state_block = np.sqrt(weight) * state_memory
        query_state_block = np.sqrt(weight) * state_query
        memory_vlm[name] = _normalize(np.concatenate([memory_vlm_block, memory_state_block], axis=1))
        query_vlm[name] = _normalize(np.concatenate([query_vlm_block, query_state_block], axis=1))
        combined[name] = {"state_norm_weight": weight, "text_key_parent": best_centered_vlm_key}

    # Decoder/prior conditioning stays on the original, uncentered full-token key,
    # isolating only the retrieval key change, as in the initial experiment.
    full_query = _normalize(query_pools[:, 0])
    _, prior_residual = memory.prior(full_query, query_state)
    encoded_reconstruction = memory.decode(encoded_base, encoded_residual, full_query, query_state)
    endpoint_scales_m = np.asarray([0.0111, 0.0122, 0.0110], np.float32)
    recon_metrics = {"self_encode_decode_oracle": _decode_error_metrics(encoded_reconstruction, query_actions, endpoint_scales_m)}
    memory_bases = np.stack([np.asarray(row["base"], np.float32) for row in memory.memory])
    memory_episodes = np.asarray([str(row["meta"].get("episode_id")) for row in memory.memory])
    memory_initials = np.asarray([str(row["meta"].get("init_state_id")) for row in memory.memory])
    nearest_base, nearest_row_indices = [], []
    for i, base in enumerate(encoded_base):
        eligible = (memory_episodes != query_episode[i]) & (memory_initials != query_init[i])
        distances = np.linalg.norm(memory_bases - np.asarray(base, np.float32)[None], axis=1)
        distances[~eligible] = np.inf
        nearest_index = int(np.argmin(distances))
        nearest_row_indices.append(nearest_index)
        nearest_base.append(memory_bases[nearest_index])
    nearest_base = np.stack(nearest_base)
    nearest_reconstruction = memory.decode(nearest_base, prior_residual, full_query, query_state)
    nearest_distance = np.linalg.norm(nearest_base - np.asarray(encoded_base, np.float32), axis=1)
    nearest_units = [memory.memory[index].get("unit_id") for index in nearest_row_indices]
    recon_metrics["nearest_other_memory_base_oracle"] = _decode_error_metrics(
        nearest_reconstruction, query_actions, endpoint_scales_m
    )
    mean_action = np.broadcast_to(query_actions.mean(axis=0, keepdims=True), query_actions.shape)
    recon_metrics["heldout_mean_action_baseline"] = _decode_error_metrics(mean_action, query_actions, endpoint_scales_m)
    recon_metrics["zero_action_baseline"] = _decode_error_metrics(np.zeros_like(query_actions), query_actions, endpoint_scales_m)
    unit_members: dict[int, list[dict]] = defaultdict(list)
    for item in memory.memory:
        if item.get("unit_id") is not None:
            unit_members[int(item["unit_id"])].append(item)

    retrieval_summaries, decoder_summaries = {}, {}
    for key_name, query_key in query_vlm.items():
        result, summary = _summary_for(key_name, query_key, memory_vlm[key_name], selected_memory,
                                       query_episode, query_init, oracle_units, majority_unit, memory.temperature)
        select_mask = np.asarray(held["partition"]).astype(str) == "select"
        _, select_summary = _summary_for(
            key_name, query_key[select_mask], memory_vlm[key_name], selected_memory,
            query_episode[select_mask], query_init[select_mask],
            [oracle_units[i] for i in np.flatnonzero(select_mask)], majority_unit, memory.temperature,
        )
        summary["select_partition_only"] = select_summary
        retrieval_summaries[key_name] = summary
        predictions = {kind: np.zeros_like(query_actions) for kind in ("unit_center", "top5_weighted_base", "unit_gaussian_sample", "correct_unit_center")}
        for i, (indices, weights) in enumerate(zip(result["top_indices"], result["top_weights"], strict=True)):
            weighted_units = defaultdict(float)
            bases = []
            for index, weight in zip(indices, weights, strict=True):
                item = selected_memory[int(index)]
                bases.append(np.asarray(item["base"], np.float32))
                if item.get("unit_id") is not None:
                    weighted_units[int(item["unit_id"])] += float(weight)
            if not weighted_units:
                continue
            selected_unit = max(weighted_units, key=lambda unit: (weighted_units[unit], -unit))

            def get_center(unit):
                all_rows = unit_members[unit]
                eligible = [row for row in all_rows if str(row["meta"].get("episode_id")) != query_episode[i]
                            and str(row["meta"].get("init_state_id")) != query_init[i]]
                values = np.stack([row["base"] for row in eligible]).astype(np.float32)
                center = (np.asarray(memory._centers[unit], np.float32) if len(eligible) == len(all_rows)
                          else values.mean(axis=0))
                return center, values

            center, member_bases = get_center(selected_unit)
            sigma = float(np.sqrt(np.mean(np.square(member_bases - center[None]))))
            sample_rng = np.random.default_rng(args.seed + i)
            sample = center + sigma * sample_rng.normal(size=center.shape).astype(np.float32)
            blend = np.asarray(weights, np.float32) @ np.stack(bases)
            for kind, base in (("unit_center", center), ("unit_gaussian_sample", sample), ("top5_weighted_base", blend)):
                predictions[kind][i] = memory.decode(base[None], prior_residual[i:i+1], full_query[i:i+1], query_state[i:i+1])[0]
            correct_unit = int(oracle_units[i])
            correct_center, _ = get_center(correct_unit)
            predictions["correct_unit_center"][i] = memory.decode(
                correct_center[None], prior_residual[i:i+1], full_query[i:i+1], query_state[i:i+1]
            )[0]
        decoder_summaries[key_name] = {
            kind: _decode_error_metrics(pred, query_actions, endpoint_scales_m)
            for kind, pred in predictions.items()
        }

    report = {
        "heldout": str(Path(args.heldout).resolve()), "memory": str(Path(args.memory).resolve()),
        "heldout_rows": len(query_actions), "heldout_episodes": len(held_eps),
        "heldout_episode_overlap_with_full_memory": len(overlap),
        "memory_entries_total": len(memory.memory), "memory_entries_rekeyed_uniform_subset": count,
        "uniform_subset_seed": args.seed, "m6_units": len(memory._centers),
        "majority_unit": int(majority_unit), "majority_unit_query_rate": majority_rate,
        "correct_units_in_heldout": int(len(set(int(unit) for unit in oracle_units if unit is not None))),
        "macro_recall_definition": "unweighted mean of per-correct-unit hit rates over units present in the scored partition; majority baseline always predicts the global majority unit.",
        "nearest_memory_base_oracle": {
            "distance": "Euclidean L2 in 12-D M6 base space",
            "exclude_same_episode_and_init_state": True,
            "mean_distance": float(np.mean(nearest_distance)),
            "mean_distance_q50": float(np.quantile(nearest_distance, 0.5)),
            "selected_same_correct_unit_rate": float(np.mean(np.asarray(nearest_units, dtype=object) == np.asarray(oracle_units, dtype=object))),
        },
        "state_token_key_available": False,
        "state_token_unavailable_reason": "π0.5 prefix contains image and prompt; robot proprioception is a separate vector.",
        "centering": "Per-pooling memory-key mean only, subtracted from memory and heldout query vectors before L2 normalization.",
        "combined_key": "L2-normalize centered VLM key and state independently, concatenate sqrt(1-w)*VLM and sqrt(w)*state blocks, then L2-normalize; w is state squared-norm share.",
        "combined_key_variants": combined,
        "best_b_to_e_centered_key_for_combination": best_centered_vlm_key,
        "centered_key_selection_fit_partition_scores": selection_scores,
        "centered_key_selection_partition": "fit (128 rows / 32 episodes); select partition metrics are reported separately.",
        "decoder_prior_control": "All Dec/prior calls use original uncentered full-token query key; only retrieval key changes.",
        "endpoint_offset_method": "Mean and quantiles of ||sum_t((pred-target)[:3] * [0.0111,0.0122,0.0110] m/action-unit/step)||*1000 at horizon end; factors copied from roundtrip-20261028T094621Z fit (env_state calibration). This is command-space estimate, not observed physical trajectory.",
        "normalized_action_mse_space": "first 7 active LIBERO action channels over the 10-step chunk; padded channels are excluded.",
        "decoder_upper_bound_conditions": {
            "A": "Dec(Enc action base, Enc action residual, original query key, query state).",
            "B": "Find nearest eligible other-demo memory base in Euclidean 12-D latent space; decode it with this query's predicted prior residual, original query key, and state.",
            "C": "Decode the M6 center of the correct unit with this query's predicted prior residual, original query key, and state.",
        },
        "oracle_autoencoder_and_shared_baselines": recon_metrics,
        "retrieval_results": retrieval_summaries,
        "decoded_action_results": decoder_summaries,
        "elapsed_seconds": time.perf_counter() - start_time,
        "partial_memory_note": "Only a uniform 512-entry subset was re-keyed; full M6 unit assignments/centers support oracle labels and majority baseline.",
    }
    with result_file.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
