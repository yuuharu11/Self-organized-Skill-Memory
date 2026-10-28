#!/usr/bin/env python3
"""Compute the normalized VLM prefix key for a prepared action-row NPZ."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model
from openpi.models import pi0_config as _plain_pi0_config
from openpi.models import tokenizer as _tokenizer
from openpi.shared import nnx_utils
from openpi.transforms import TokenizeSubtaskInference
from memory import validate_key_text_source
from smoke import DEFAULT_CHECKPOINT, load_model
from train import observation_batch
from serve_s2_policy import load_generator, DEFAULT_PI05_GENERATOR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-npz", required=True, help="Prepared π0.5 [N] observation/action rows")
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--query-text-source", choices=("instruction_only", "pi05_generated_subtask"),
                        help="must match the source declared in the input cache")
    parser.add_argument("--generator-checkpoint", default=DEFAULT_PI05_GENERATOR)
    parser.add_argument("--generated-subtask-tokens", type=int, default=32)
    args = parser.parse_args()
    output = Path(args.output_npz)
    if output.exists():
        raise FileExistsError(output)
    if jax.default_backend() != "gpu":
        raise RuntimeError("VLM prefix extraction expects a GPU")
    with np.load(args.input_npz, allow_pickle=True) as archive:
        data = {name: archive[name] for name in archive.files}
    declared_source = validate_key_text_source(data)
    if 'suite' not in data or not np.all(np.asarray(data['suite']).astype(str)=='libero90'):
        raise ValueError('training key extraction accepts LIBERO-90 only')
    key_text_source = args.query_text_source or declared_source
    if key_text_source != declared_source:
        raise ValueError(f"input cache declares {declared_source}, requested {key_text_source}")
    if data.get("actions", np.empty((0,))).shape[1:] != (10, 32):
        raise ValueError("input actions must be normalized LIBERO [N,10,32]")
    norm_hash = str(np.asarray(data.get("normalization_sha256", "")).item())
    if len(norm_hash) != 64:
        raise ValueError("input needs normalization_sha256 for the LIBERO-90 action/state stats")
    required = ("image_base", "image_left_wrist", "state", "tokenized_prompt", "tokenized_prompt_mask")
    missing = [name for name in required if name not in data]
    if missing:
        raise ValueError(f"input rows missing fields: {missing}")
    if "image_right_wrist" not in data:
        data["image_right_wrist"] = np.zeros_like(data["image_base"])
    if "image_mask_base_0_rgb" not in data:
        data["image_mask_base_0_rgb"] = np.ones((len(data["actions"]),), bool)
    if "image_mask_left_wrist_0_rgb" not in data:
        data["image_mask_left_wrist_0_rgb"] = np.ones((len(data["actions"]),), bool)
    if "image_mask_right_wrist_0_rgb" not in data:
        data["image_mask_right_wrist_0_rgb"] = np.zeros((len(data["actions"]),), bool)

    model, config = load_model(args.checkpoint)
    encode = nnx_utils.module_jit(model.encode_prefix_key)
    generator = None
    generator_tokenizer = None
    prefix_transform = None
    if key_text_source == "pi05_generated_subtask":
        if "prompt" not in data:
            raise ValueError("generated-subtask extraction requires original prompt text in each row")
        generator = load_generator(args.generator_checkpoint, seed=991)
        generator_tokenizer = _tokenizer.PaligemmaTokenizer(max_len=config.max_token_len)
        prefix_transform = TokenizeSubtaskInference(generator_tokenizer)
        generate = nnx_utils.module_jit(generator.generate_subtask, static_argnames=("max_tokens",))
        generated_text = []
    result = []
    batch_size = args.batch_size
    for start in range(0, len(data["actions"]), batch_size):
        stop = min(start + batch_size, len(data["actions"]))
        sl = slice(start, stop)
        rows = {
            **{name:np.asarray(value)[sl] for name,value in data.items()
               if isinstance(value,np.ndarray) and value.ndim>0 and len(value)==len(data['actions'])},
        }
        if generator is not None:
            prefixes = [prefix_transform({"prompt": str(prompt)}) for prompt in rows["prompt"]]
            rows["tokenized_prompt"] = np.stack([item["tokenized_prompt"] for item in prefixes])
            rows["tokenized_prompt_mask"] = np.stack([item["tokenized_prompt_mask"] for item in prefixes])
            generator_obs = observation_batch(rows, config)
            generated_tokens = generate(generator_obs, max_tokens=args.generated_subtask_tokens)
            batch_text = [generator_tokenizer.detokenize(row) for row in np.asarray(generated_tokens)]
            generated_text.extend(batch_text)
            keyed_prompts = [f"{prompt}. Subtask: {text}" for prompt, text in zip(rows["prompt"], batch_text, strict=True)]
            keyed_tokens = [generator_tokenizer.tokenize(prompt, state)
                            for prompt, state in zip(keyed_prompts, rows["state"], strict=True)]
            rows["tokenized_prompt"] = np.stack([item[0] for item in keyed_tokens])
            rows["tokenized_prompt_mask"] = np.stack([item[1] for item in keyed_tokens])
        obs = observation_batch(rows, config)
        batch_key = encode(obs)
        jax.block_until_ready(batch_key)
        result.append(np.asarray(batch_key, np.float32))
        print(f"prefix keys: {stop}/{len(data['actions'])}", flush=True)
    data["key"] = np.concatenate(result, axis=0)
    data["key_checkpoint"] = np.asarray(str(Path(args.checkpoint).resolve()))
    data["key_l2_normalized"] = np.asarray(True)
    data["key_text_source"] = np.asarray(key_text_source)
    if key_text_source == "pi05_generated_subtask":
        data["key_text_generator"] = np.asarray("pi05")
        data["key_text_generator_checkpoint"] = np.asarray(str(Path(args.generator_checkpoint).resolve()))
        data["generated_subtask"] = np.asarray(generated_text, dtype=str)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **data)
    (output.with_suffix(".json")).write_text(json.dumps({
        "rows": len(data["key"]), "key_shape": list(data["key"].shape), "key_unit_norm_mean": float(
            np.mean(np.linalg.norm(data["key"], axis=-1))), "checkpoint": str(Path(args.checkpoint).resolve()),
        "normalization_sha256": norm_hash, "key_text_source": key_text_source,
        "key_text_generator": str(np.asarray(data.get("key_text_generator", "")).item()) or None,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
