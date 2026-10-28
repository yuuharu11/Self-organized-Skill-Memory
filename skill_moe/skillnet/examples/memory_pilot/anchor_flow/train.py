#!/usr/bin/env python3
"""Fine-tune the S2 action expert from a prepared, normalized LIBERO-90 cache.

The cache is an explicit preprocessing boundary so neither this script nor the
model config can silently fall back to LIBERO-40. See DESIGN.md for fields.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp

from openpi.models import model as _model
from openpi.models.pi0_config_anchor_flow import Pi0AnchorFlowConfig, action_expert_trainable_filter
from openpi.training.weight_loaders import CheckpointWeightLoader
from smoke import DEFAULT_CHECKPOINT, _make_train_step, load_model
from memory import load_memory, validate_key_text_source


def batch_indices(permutation,step,batch_size):
    if not len(permutation):
        raise ValueError("empty training data")
    return permutation[np.arange(step*batch_size,(step+1)*batch_size)%len(permutation)]


def image_input(value):
    value=jnp.asarray(value)
    return value.astype(jnp.float32)/255.*2.-1. if value.dtype == jnp.uint8 else value.astype(jnp.float32)


def observation_batch(rows, config):
    images = {
        "base_0_rgb": image_input(rows["image_base"]),
        "left_wrist_0_rgb": image_input(rows["image_left_wrist"]),
        "right_wrist_0_rgb": image_input(rows.get("image_right_wrist",np.zeros_like(rows['image_base']))),
    }
    image_masks = {name: jnp.asarray(rows.get("image_mask_" + name, np.full(len(next(iter(images.values()))),name!='right_wrist_0_rgb',bool)))
                   for name in images}
    return _model.Observation(
        images=images,
        image_masks=image_masks,
        state=jnp.asarray(rows["state"], jnp.float32),
        tokenized_prompt=jnp.asarray(rows["tokenized_prompt"], jnp.int32),
        tokenized_prompt_mask=jnp.asarray(rows["tokenized_prompt_mask"], jnp.bool_),
        skills=jnp.asarray(rows["skills"], jnp.int32) if "skills" in rows else None,
        skill_mask=jnp.asarray(rows["skill_mask"], jnp.bool_) if "skill_mask" in rows else None,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero90-cache", required=True)
    parser.add_argument("--memory", required=True, help="M6 memory built from the same LIBERO-90 split")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2.5e-5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if jax.default_backend() != "gpu":
        raise RuntimeError("S2 training expects a GPU")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    with np.load(args.libero90_cache, allow_pickle=True) as archive:
        data = {name: archive[name] for name in archive.files}
    needed = {"actions", "anchor_actions", "state", "tokenized_prompt", "tokenized_prompt_mask",
              "image_base", "image_left_wrist"}
    if not needed.issubset(data):
        raise ValueError(f"training cache is missing {sorted(needed - data.keys())}")
    if "suite" not in data or not np.all(np.asarray(data["suite"]).astype(str) == "libero90"):
        raise ValueError("training cache must identify every row as LIBERO-90")
    if data["actions"].shape[1:] != (10, 32) or data["anchor_actions"].shape != data["actions"].shape:
        raise ValueError("actions and anchor_actions must share the normalized [N,10,32] shape")
    memory = load_memory(args.memory)
    if len(memory.memory) == 0 or memory.action_shape != (10, 32):
        raise ValueError("M6 memory is empty or was fit in the wrong action space")
    norm_hash = str(np.asarray(data.get("normalization_sha256", "")).item())
    if norm_hash != memory.normalization_sha256:
        raise ValueError("training cache normalization differs from the LIBERO-90 memory codec")
    if validate_key_text_source(data) != memory.key_text_source:
        raise ValueError("training cache key text differs from the LIBERO-90 memory key source")
    if str(np.asarray(data.get('key_checkpoint','')).item()) != str(Path(args.checkpoint).resolve()):
        raise ValueError('training cache prefix keys came from a different checkpoint')

    model, config = load_model(args.checkpoint)
    trainable = action_expert_trainable_filter()
    tx = optax.adamw(args.learning_rate, weight_decay=1e-5, b1=0.9, b2=0.95)
    opt_state = tx.init(nnx.state(model, trainable))
    train_step = _make_train_step(tx)
    rng = jax.random.key(args.seed)
    rng, permutation_rng = jax.random.split(rng)
    permutation = np.random.default_rng(args.seed).permutation(len(data["actions"]))
    logs = []
    started = time.perf_counter()
    previous_logged=started
    for step in range(args.steps):
        indices = batch_indices(permutation,step,args.batch_size)
        rows = {name: np.asarray(value)[indices] for name, value in data.items()
                if isinstance(value, np.ndarray) and value.ndim>0 and len(value) == len(data["actions"])}
        obs = observation_batch(rows, config)
        actions = jnp.asarray(rows["actions"], jnp.float32)
        anchor_actions = jnp.asarray(rows["anchor_actions"], jnp.float32)
        rng, step_rng = jax.random.split(rng)
        loss, (flow, balance), opt_state = train_step(step_rng, model, opt_state, obs, actions, anchor_actions)
        jax.block_until_ready(loss)
        if step % 100 == 0 or step == args.steps - 1:
            now=time.perf_counter()
            record = {"step": step + 1, "loss": float(loss), "flow_loss": float(flow),
                      "balance_loss": float(balance), "elapsed_seconds": now - started,
                      "seconds_per_step_since_log":(now-previous_logged)/(1 if step==0 else min(100,step)),
                      "utc":__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()}
            print(record, flush=True)
            logs.append(record)
            previous_logged=now
    params = nnx.state(model).to_pure_dict()
    checkpoint = output / "params"
    ocp.PyTreeCheckpointer().save(checkpoint, params)
    (output / "train_metrics.json").write_text(json.dumps({
        "checkpoint_init": str(Path(args.checkpoint).resolve()), "dataset": str(Path(args.libero90_cache).resolve()),
        "memory": str(Path(args.memory).resolve()), "steps": args.steps, "batch_size": args.batch_size,
        "learning_rate": args.learning_rate, "trainable_filter": "action expert + router + time MLP + projections",
        "memory_items": len(memory.memory), "memory_representation": memory.representation_version,
        "logs": logs, "elapsed_seconds": time.perf_counter() - started,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
