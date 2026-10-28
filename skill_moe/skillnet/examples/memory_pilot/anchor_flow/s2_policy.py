"""Websocket policy implementation for S2 LIBERO case execution."""
from __future__ import annotations

import os
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
                    sample_memory_anchor, sample_optimus_anchor, shuffle_memory_values)
from openpi.models.pi0_anchor_flow import mix_anchor_with_noise


class S2WebsocketPolicy(_policy.Policy):
    """One loaded S2 checkpoint serves all paired cases and worker clients."""

    def __init__(self, model, *, input_transforms, output_transforms, memory_dir: str,
                 normalization_hash: str, generator_model=None, generator_checkpoint: str = "",
                 seed: int = 0,
                 absolute_similarity_low: float = 0.2, absolute_similarity_high: float = 0.8,
                 fixed_t0: float = 0.5, generated_subtask_tokens: int = 32):
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
        self._subtask_tokenizer = _tokenizer.PaligemmaTokenizer(max_len=model.max_token_len)
        self._subtask_prefix = TokenizeSubtaskInference(self._subtask_tokenizer)

    def _memory_for(self, snapshot: str, condition: str):
        path = Path(snapshot)
        if not path.is_absolute():
            path = self._memory_dir / (snapshot if snapshot.endswith(".pkl") else f"{snapshot}.pkl")
        cache_key = (str(path.resolve()), condition)
        if cache_key not in self._memory_cache:
            memory = load_memory(path)
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

    def infer(self, observation: dict) -> dict:
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
        transformed = self._input_transform(raw)
        inputs = jax.tree.map(lambda value: jnp.asarray(value)[None, ...], transformed)
        action_observation = _model.Observation.from_dict(inputs)
        key = None
        generated_text = None
        if condition != "anchor_disabled_t0_1":
            key_observation = action_observation
            if text_source == "pi05_generated_subtask":
                key_observation, generated_text = self._generated_query_observation(inputs, instruction)
            elif text_source != "instruction_only":
                raise ValueError(f"unsupported S2 query text source: {text_source}")
            key = np.asarray(self._encode_key(key_observation)[0], dtype=np.float32)
        elif text_source not in ("instruction_only", "pi05_generated_subtask"):
            raise ValueError(f"unsupported S2 query text source: {text_source}")

        rng = jax.random.key(request_seed)
        rng, action_rng = jax.random.split(rng)
        debug = {"query_id": query_id, "condition": condition, "query_text_source": text_source,
                 "generated_subtask": generated_text, "memory_snapshot": snapshot,
                 "selected_unit_id": None, "selected_task": None, "t0": 1.0,
                 "retrieval": []}
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
            max_similarity, entropy = retrieval_confidence(memory, key, meta=meta)
            records = retrieval_records(memory, key, meta=meta)
            debug["max_cosine_similarity"] = max_similarity
            debug["normalized_retrieval_entropy"] = entropy
            debug["retrieval"] = records
            selection_seed = request_seed ^ 0x51A2
            if condition == "rts_raw_chunk_entropy":
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
            if condition == "anchor_absolute_similarity":
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
            noise = jax.random.normal(action_rng, (1, self._model.action_horizon, self._model.action_dim))
            x_t0 = mix_anchor_with_noise(jnp.asarray(anchor)[None], noise, t0)
            actions = self._sample_actions(action_rng, action_observation, num_steps=10,
                                           x_src=x_t0, t0=jnp.asarray([t0]))
            debug.update({"selected_unit_id": selected.get("unit_id"),
                          "selected_task": selected.get("task"), "t0": t0})
        output = self._output_transform({"state": inputs["state"], "actions": actions})
        output = jax.tree.map(lambda value: np.asarray(value[0]), output)
        output["s2_debug"] = debug
        return output
