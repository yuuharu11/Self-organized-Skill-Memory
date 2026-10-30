"""Websocket policy implementation for S2 LIBERO case execution."""
from __future__ import annotations

import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models import tokenizer as _tokenizer
from openpi.policies import policy as _policy
from openpi.shared import nnx_utils
from openpi.transforms import TokenizeSubtaskInference

from memory import (load_memory, query_raw_action, retrieval_confidence, retrieval_records,
                    sample_memory_anchor, sample_optimus_anchor, shuffle_memory_values,
                    prepare_online_query, prepare_centered_online_query,
                    anchor_v2_components, anchor_v2_t0)
from openpi.models.pi0_anchor_flow import mix_anchor_with_noise
from confidence_calibration import calibrated_t0, denoising_steps, load_calibration


def sample_anchor_v2_actions(sample_fn, rng, observation, *, noise, anchor, t0, num_steps):
    """Use the exact legacy t0=1 sampler call, otherwise blend a draft."""
    if float(t0) == 1.0:
        return sample_fn(rng, observation, num_steps=10, noise=noise,
                         x_src=noise, t0=1.0)
    x_src = mix_anchor_with_noise(jnp.asarray(anchor)[None], noise, float(t0))
    return sample_fn(rng, observation, num_steps=num_steps, x_src=x_src,
                     t0=jnp.asarray([float(t0)]))


def integer_object_ids(transformed):
    """The public transform gives an empty object annotation a float dtype."""
    if 'objects' in transformed:
        values = np.asarray(transformed['objects'])
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            raise ValueError('object IDs must be finite integers')
        transformed = {**transformed, 'objects': values.astype(np.int32)}
    return transformed


class S2WebsocketPolicy(_policy.Policy):
    """One loaded S2 checkpoint serves all paired cases and worker clients."""

    def __init__(self, model, *, input_transforms, output_transforms, memory_dir: str,
                 normalization_hash: str, generator_model=None, generator_checkpoint: str = "",
                 seed: int = 0,
                 absolute_similarity_low: float = 0.2, absolute_similarity_high: float = 0.8,
                 fixed_t0: float = 0.5, generated_subtask_tokens: int = 32,
                 calibration_path: str = "", anchor_v2_source: str = "dec_each",
                 anchor_v2_tau: float = 0.05, anchor_v2_t0_mode: str = "fixed",
                 anchor_v2_fixed_t0: float = 0.5, anchor_v2_calibration_path: str = "",
                 anchor_v2_gate_threshold: float | None = None):
        # The parent Policy currently tests the key by Python truthiness; let
        # it create its default key and use deterministic per-request keys below.
        super().__init__(model)
        self._input_transform = _transforms.compose(input_transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._memory_dir = Path(memory_dir)
        self._normalization_hash = normalization_hash
        self._generator_model = generator_model
        self._generator_checkpoint = str(Path(generator_checkpoint).resolve()) if generator_checkpoint else ""
        self._encode_key = nnx_utils.module_jit(model.encode_prefix_key)
        self._encode_key_and_prefix_cache = nnx_utils.module_jit(model.encode_prefix_key_and_cache)
        self._sample_actions_with_prefix_cache = nnx_utils.module_jit(
            model.sample_actions_from_prefix_cache, static_argnames=("num_steps",)
        )
        self._generate_subtask = (nnx_utils.module_jit(generator_model.generate_subtask,
                              static_argnames=("max_tokens",)) if generator_model is not None else None)
        self._seed = int(seed)
        self._query_index = 0
        self._memory_cache: dict[tuple[str, str], object] = {}
        self._t0_history: list[float] = []
        self._absolute_similarity_low = absolute_similarity_low
        self._absolute_similarity_high = absolute_similarity_high
        self._fixed_t0 = fixed_t0
        self._max_generated_tokens = generated_subtask_tokens
        self._calibration = load_calibration(calibration_path) if calibration_path else None
        if anchor_v2_source not in ("dec_latent", "dec_each", "raw"):
            raise ValueError("anchor_v2_source must be dec_latent, dec_each, or raw")
        if anchor_v2_tau <= 0 or anchor_v2_t0_mode not in ("calibration", "fixed", "one"):
            raise ValueError("invalid anchor_v2 temperature or t0 mode")
        if not 0.2 <= anchor_v2_fixed_t0 <= 1.0:
            raise ValueError("anchor_v2_fixed_t0 must be in [.2,1]")
        self._anchor_v2_source = anchor_v2_source
        self._anchor_v2_tau = float(anchor_v2_tau)
        self._anchor_v2_t0_mode = anchor_v2_t0_mode
        self._anchor_v2_fixed_t0 = float(anchor_v2_fixed_t0)
        self._anchor_v2_table = None
        self._anchor_v2_gate_threshold = anchor_v2_gate_threshold
        if anchor_v2_calibration_path:
            import json
            self._anchor_v2_table = json.loads(Path(anchor_v2_calibration_path).read_text())
            edges = np.asarray(self._anchor_v2_table.get("edges",
                self._anchor_v2_table.get("top1_similarity_edges", [])), np.float64)
            values = np.asarray(self._anchor_v2_table.get("t0",
                self._anchor_v2_table.get("t0_by_bin", [])), np.float64)
            if (len(values) != len(edges) + 1 or not np.isfinite(edges).all()
                    or not np.isfinite(values).all() or np.any(np.diff(edges) <= 0)
                    or np.any(values < .2) or np.any(values > 1.0)):
                raise ValueError("invalid anchor_v2 calibration table")
            if self._anchor_v2_gate_threshold is None:
                self._anchor_v2_gate_threshold = self._anchor_v2_table.get("gate_threshold",
                    self._anchor_v2_table.get("similarity_threshold",
                    self._anchor_v2_table.get("holdout_similarity_median")))
                if self._anchor_v2_gate_threshold is None and "holdout_top1_similarities" in self._anchor_v2_table:
                    self._anchor_v2_gate_threshold = float(np.median(
                        np.asarray(self._anchor_v2_table["holdout_top1_similarities"], np.float64)))
        if self._anchor_v2_t0_mode == "calibration" and self._anchor_v2_table is None:
            raise ValueError("anchor_v2 calibration mode needs --anchor-v2-calibration")
        self._anchor_v2_t0_by_episode: dict[str, list[float]] = {}
        self._subtask_tokenizer = _tokenizer.PaligemmaTokenizer(max_len=model.max_token_len)
        self._subtask_prefix = TokenizeSubtaskInference(self._subtask_tokenizer)

    def _memory_for(self, snapshot: str, condition: str):
        path = Path(snapshot)
        if not path.is_absolute():
            path = self._memory_dir / (snapshot if snapshot.endswith(".pkl") else f"{snapshot}.pkl")
        cache_key = (str(path.resolve()), condition)
        if cache_key not in self._memory_cache:
            memory = load_memory(path)
            if self._calibration is not None:
                if self._calibration['normalization_sha256'] != self._normalization_hash:
                    raise ValueError("calibration normalization differs from checkpoint")
                memory.temperature = self._calibration['temperature']
            if memory.normalization_sha256 != self._normalization_hash:
                raise ValueError("S2 memory normalization does not match the server checkpoint")
            if condition == "anchor_memory_shuffled":
                memory = shuffle_memory_values(memory, seed=self._seed)
            self._memory_cache[cache_key] = memory
        return self._memory_cache[cache_key]

    def _generated_query_observation(self, inputs: dict, instruction: str):
        if self._generator_model is None:
            raise ValueError("pi05_generated_subtask requested, but the fixed pi05 generator is not loaded")
        prefix = self._subtask_prefix({"prompt": instruction})
        generation_inputs = dict(inputs)
        generation_inputs["tokenized_prompt"] = jnp.asarray(prefix["tokenized_prompt"])[None]
        generation_inputs["tokenized_prompt_mask"] = jnp.asarray(prefix["tokenized_prompt_mask"])[None]
        generation_observation = _model.Observation.from_dict(generation_inputs)
        tokens = self._generate_subtask(generation_observation, max_tokens=self._max_generated_tokens)
        text = self._subtask_tokenizer.detokenize(np.asarray(tokens[0], dtype=np.int32))
        # Re-tokenize the generated text with the current normalized pi0.5
        # discrete state so keys have the same image/text/state prefix contract.
        key_tokens, key_mask = self._subtask_tokenizer.tokenize(
            f"{instruction}. Subtask: {text}", np.asarray(inputs["state"][0], np.float32)
        )
        key_inputs = dict(inputs)
        key_inputs["tokenized_prompt"] = jnp.asarray(key_tokens)[None]
        key_inputs["tokenized_prompt_mask"] = jnp.asarray(key_mask)[None]
        key_observation = _model.Observation.from_dict(key_inputs)
        return key_observation, text

    @staticmethod
    def _key_only_observation(observation):
        """Drop annotations and memory payloads unused by the VLM key.

        Inference may carry variable-length annotated skills. Leaving those
        arrays in a jitted method's argument tree causes shape-specialized
        recompilation even though encode_prefix_key never reads them.
        """
        return observation.replace(
            skills=None, skill_mask=None, skill_embed_override=None, objects=None,
            memory_features=None, memory_candidates=None, memory_candidate_mask=None,
            memory_enabled=False, skill_z=None,
        )

    def infer(self, observation: dict) -> dict:
        started = time.perf_counter()
        raw = dict(observation)
        condition = str(raw.pop("s2_condition", "anchor_entropy_current_memory"))
        snapshot = str(raw.pop("s2_memory_snapshot", "initial_libero90"))
        text_source = str(raw.pop("s2_query_text_source", "instruction_only"))
        episode_id = str(raw.pop("s2_episode_id", ""))
        init_state_id = str(raw.pop("s2_init_state_id", ""))
        query_id = int(raw.pop("s2_query_id", self._query_index))
        request_seed = int(raw.pop("s2_rng_seed", self._seed + query_id))
        self._query_index += 1

        instruction = str(raw.get("prompt", ""))
        transformed = integer_object_ids(self._input_transform(raw))
        inputs = jax.tree.map(lambda value: jnp.asarray(value)[None, ...], transformed)
        action_observation = _model.Observation.from_dict(inputs)
        transformed_at = time.perf_counter()
        key = None
        prefix_context = None
        generated_text = None
        if condition != "anchor_disabled_t0_1":
            key_observation = action_observation
            if text_source == "pi05_generated_subtask":
                key_observation, generated_text = self._generated_query_observation(inputs, instruction)
            elif text_source != "instruction_only":
                raise ValueError(f"unsupported S2 query text source: {text_source}")
            if (condition == "anchor_v2" and text_source == "instruction_only"
                    and key_observation.skills is None):
                encoded_key, prefix_context = self._encode_key_and_prefix_cache(key_observation)
                key = np.asarray(encoded_key[0], dtype=np.float32)
            else:
                key = np.asarray(self._encode_key(
                    self._key_only_observation(key_observation)
                )[0], dtype=np.float32)
        elif text_source not in ("instruction_only", "pi05_generated_subtask"):
            raise ValueError(f"unsupported S2 query text source: {text_source}")
        encoded_at = time.perf_counter()
        retrieved_at = encoded_at

        rng = jax.random.key(request_seed)
        rng, action_rng = jax.random.split(rng)
        debug = {"query_id": query_id, "condition": condition, "query_text_source": text_source,
                 "generated_subtask": generated_text, "memory_snapshot": snapshot,
                 "selected_unit_id": None, "selected_task": None, "t0": 1.0,
                 "retrieval": [], "n": 10, "num_steps": 10,
                 "normalized_retrieval_entropy": None, "max_cosine_similarity": None,
                 "prefix_cache_reused": prefix_context is not None}
        if condition == "anchor_disabled_t0_1":
            noise = jax.random.normal(action_rng, (1, self._model.action_horizon, self._model.action_dim))
            actions = self._sample_actions(action_rng, action_observation, num_steps=10,
                                           noise=noise, x_src=noise, t0=1.0)
        else:
            memory = self._memory_for(snapshot, condition)
            if memory.key_text_source != text_source:
                raise ValueError(f"memory key source {memory.key_text_source} != query source {text_source}")
            if text_source == "pi05_generated_subtask":
                memory_generator = getattr(memory, "key_text_generator_checkpoint", "")
                if memory_generator != self._generator_checkpoint:
                    raise ValueError(
                        "generated-subtask memory and query generator checkpoints differ: "
                        f"{memory_generator!r} != {self._generator_checkpoint!r}"
                    )
            meta = {"episode_id": episode_id, "init_state_id": init_state_id}
            assert key is not None
            if condition == "anchor_v2":
                meta = prepare_centered_online_query(memory, key, meta, top_k=5,
                                                     temperature=self._anchor_v2_tau)
                max_similarity = float(meta["_anchor_v2_similarities"][0])
                anchor, records = anchor_v2_components(memory, key,
                    np.asarray(action_observation.state[0]), meta, source=self._anchor_v2_source)
                debug.update(max_cosine_similarity=max_similarity,
                             top1_minus_top5_similarity=float(meta["_anchor_v2_similarities"][0]
                                                              - meta["_anchor_v2_similarities"][-1]),
                             retrieval=records, anchor_source=self._anchor_v2_source,
                             retrieval_temperature=self._anchor_v2_tau)
                gate_threshold = self._anchor_v2_gate_threshold
                t0, index, gate_threshold = anchor_v2_t0(max_similarity,
                    mode=self._anchor_v2_t0_mode, fixed_t0=self._anchor_v2_fixed_t0,
                    table=self._anchor_v2_table, gate_threshold=gate_threshold)
                if index is not None:
                    debug["calibration_bin"] = index
                if gate_threshold is not None:
                    debug["gate_threshold"] = float(gate_threshold)
                num_steps = denoising_steps(t0)
                trial_t0 = self._anchor_v2_t0_by_episode.setdefault(episode_id or f"query-{query_id}", [])
                trial_t0.append(float(t0))
                debug["t0_trial_mean"] = float(np.mean(trial_t0))
                debug["t0_trial_min"] = float(np.min(trial_t0))
            else:
                meta = prepare_online_query(memory, key, meta)
                max_similarity, entropy = retrieval_confidence(memory, key, meta=meta)
                records = retrieval_records(memory, key, meta=meta)
                debug["max_cosine_similarity"] = max_similarity
                debug["normalized_retrieval_entropy"] = entropy
                debug["retrieval"] = records
            selection_seed = request_seed ^ 0x51A2
            if condition == "anchor_v2":
                selected = records[0] if records else {"unit_id": None, "task": None}
            elif condition == "rts_raw_chunk_entropy":
                anchor, selected = query_raw_action(memory, key, meta=meta, return_details=True)
            elif condition == "optimus_gaussian_entropy":
                anchor = sample_optimus_anchor(memory, key, np.asarray(action_observation.state[0]),
                                               seed=selection_seed, meta=meta)
                selected = {"unit_id": None, "task": "moment_matched", "weight": 1.0}
            else:
                anchor, selected = sample_memory_anchor(
                    memory, key, np.asarray(action_observation.state[0]), seed=selection_seed,
                    meta=meta, return_details=True
                )
            calibrated_condition = condition in ("anchor_entropy_current_memory", "anchor_memory_shuffled")
            if condition == "anchor_v2":
                pass
            elif calibrated_condition and self._calibration is not None:
                t0, bin_id, confidence = calibrated_t0(self._calibration, max_similarity, entropy)
                debug.update(calibration_bin=bin_id, calibrated_confidence=confidence,
                             confidence_metric=self._calibration['metric'],
                             retrieval_temperature=self._calibration['temperature'])
            elif condition == "anchor_absolute_similarity":
                confidence = np.clip((max_similarity - self._absolute_similarity_low)
                                     / max(self._absolute_similarity_high - self._absolute_similarity_low, 1e-8),
                                     0.0, 1.0)
                t0 = 1.0 - 0.8 * float(confidence)
            elif condition == "anchor_fixed_t0":
                t0 = self._fixed_t0
            elif condition == "anchor_t0_shuffled":
                calculated = 0.2 + 0.8 * entropy
                t0 = float(np.random.default_rng(selection_seed).choice(self._t0_history)) if self._t0_history else 0.6
                self._t0_history.append(calculated)
            else:
                t0 = 0.2 + 0.8 * entropy
            if condition == "anchor_absolute_similarity" and max_similarity < self._absolute_similarity_low:
                t0 = 1.0
            t0 = float(np.clip(t0, 0.2, 1.0))
            if condition != "anchor_v2":
                num_steps = denoising_steps(t0) if calibrated_condition else 10
            noise = jax.random.normal(action_rng, (1, self._model.action_horizon, self._model.action_dim))
            retrieved_at = time.perf_counter()
            if condition == "anchor_v2":
                # At t0=1 this helper makes exactly the same sampler call as
                # anchor_disabled_t0_1; no draft value participates.
                if prefix_context is not None and t0 != 1.0:
                    x_src = mix_anchor_with_noise(jnp.asarray(anchor)[None], noise, t0)
                    actions = self._sample_actions_with_prefix_cache(
                        action_rng, action_observation, prefix_cache=prefix_context,
                        num_steps=num_steps, noise=noise, x_src=x_src, t0=t0,
                    )
                else:
                    actions = sample_anchor_v2_actions(self._sample_actions, action_rng,
                        action_observation, noise=noise, anchor=anchor, t0=t0,
                        num_steps=num_steps)
            else:
                x_t0 = mix_anchor_with_noise(jnp.asarray(anchor)[None], noise, t0)
                actions = self._sample_actions(action_rng, action_observation, num_steps=num_steps,
                                               x_src=x_t0, t0=jnp.asarray([t0]))
            debug.update({"selected_unit_id": selected.get("unit_id"),
                          "selected_task": selected.get("task"), "t0": t0,
                          "n": num_steps, "num_steps": num_steps,
                          "euler_step_size": t0 / num_steps})
        output = self._output_transform({"state": inputs["state"], "actions": actions})
        output = jax.tree.map(lambda value: np.asarray(value[0]), output)
        finished = time.perf_counter()
        debug['timing_seconds'] = dict(transform=transformed_at-started,
            encode_key=encoded_at-transformed_at, retrieval_and_anchor=retrieved_at-encoded_at,
            sample_and_output=finished-retrieved_at, total=finished-started)
        output["s2_debug"] = debug
        return output
