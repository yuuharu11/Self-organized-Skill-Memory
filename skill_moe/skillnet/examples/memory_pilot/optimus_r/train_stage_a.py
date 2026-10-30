#!/usr/bin/env python3
"""Train Stage A on the existing LIBERO-90-only S2 export."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import time

import numpy as np


import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research_paths import REPO_ROOT as RESEARCH_REPO_ROOT, STATE_DIR as RESEARCH_STATE_DIR
DEFAULT_CHECKPOINT = str(RESEARCH_STATE_DIR / 'checkpoints/pi05_libero_moe_skill_4_90/moe_balance_4_32_90_20000/19999/params')
DEFAULT_CACHE = str(RESEARCH_STATE_DIR / 'outputs/memory-pilot/anchor-s2-train-20261028T072152Z/libero90_rows.npz')
HERE = Path(__file__).resolve().parent
EXAMPLES = HERE.parents[1]


def _joined_path(path):
    return "/".join(str(part) for part in path)


def stage_a_trainable_filter():
    """New interface + action expert + visual-language transformer layers.

    The documented "last layers" count is missing from v1; the local default
    trains the last four PaliGemma visual-language blocks, action expert, and
    all new Optimus-R modules.  The gradient mask below freezes earlier VLM
    blocks inside scanned parameter stacks.
    """
    import flax.nnx as nnx
    from openpi.shared import nnx_utils

    return nnx.All(nnx.Param, nnx_utils.PathRegex(
        r".*(memory_seed_tokens|optimus_state_token|query_pool|skill_pool|query_encoder|skill_encoder|skill_decoder|"
        r"skill_token_projector|llm.*_1|llm.*_0|action_in_proj|time_mlp|action_out_proj).*"
    ))


def mask_earlier_vlm_layers(gradients, *, total_layers: int, trainable_layers: int):
    """Zero gradients for all but the last configured VLM layers."""
    if trainable_layers < 1 or trainable_layers > total_layers:
        raise ValueError("trainable_vlm_layers must be between one and total VLM depth")

    def apply(path, variable_state):
        joined = _joined_path(path)
        value = variable_state.value
        if "llm" in joined and "layers" in joined and re.search(r"_0(?:/|$)", joined):
            if value.ndim and value.shape[0] == total_layers:
                mask = jnp.zeros((total_layers,), value.dtype).at[-trainable_layers:].set(1)
                value = value * mask.reshape((total_layers,) + (1,) * (value.ndim - 1))
                return variable_state.replace(value=value)
        return variable_state

    import jax.numpy as jnp
    return gradients.map(apply)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero90-cache", default=DEFAULT_CACHE)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--learning-rate-new", type=float, default=2e-4)
    parser.add_argument("--learning-rate-backbone", type=float, default=2e-5)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--memory-fraction", type=float, default=0.35)
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        raise ValueError("steps and batch size must be positive")
    if not 0.05 <= args.memory_fraction <= 0.9:
        raise ValueError("memory fraction must be in [0.05, 0.9]")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", str(args.memory_fraction))
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(args.threads))

    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import optax
    import orbax.checkpoint as ocp
    from openpi.models.pi0_optimus_r_config import Pi0OptimusRConfig
    from openpi.training.weight_loaders import PartialCheckpointWeightLoader

    if jax.default_backend() != "gpu":
        raise RuntimeError(f"Stage A expects a GPU, got {jax.default_backend()}")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    from data import contrastive_metadata, data_fingerprint, load_cache, observation_batch

    data = load_cache(args.libero90_cache)
    task_ids, progress, episode_ids, sample_indices = contrastive_metadata(data)
    config = Pi0OptimusRConfig(
        pi05=True, action_horizon=10, action_dim=32, max_token_len=200,
        discrete_state_input=False, paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m_moe_4", skill_num=6, skill_embed_dim=64,
    )
    model = config.create(jax.random.key(args.seed))
    state = nnx.state(model)
    loaded = PartialCheckpointWeightLoader(args.checkpoint, missing_regex=".*").load(state.to_pure_dict())
    state.replace_by_pure_dict(loaded)
    nnx.update(model, state)
    trainable = stage_a_trainable_filter()
    trainable_state = nnx.state(model, trainable)
    params_paths = [_joined_path(path) for path in trainable_state.flat_state()]
    vlm_depth = __import__("openpi.models.gemma_moe_skill", fromlist=["get_config"]).get_config(
        config.paligemma_variant
    ).depth
    if not any("optimus" in path or any(token in path for token in ("memory_seed_tokens", "query_pool", "skill_pool"))
               for path in params_paths):
        raise RuntimeError("new Optimus-R parameters were not included in the trainable parameter filter")

    # Paper v1 omits optimizer details. Use Adam without decoupled decay so
    # masked early-layer gradients remain exactly zero; use its two reported LR scales.
    tx = optax.scale_by_adam()
    opt_state = tx.init(trainable_state)

    @nnx.jit
    def train_step(rng, module, optimizer_state, obs, actions, batch_tasks, batch_progress,
                   batch_episodes, batch_times):
        def loss_fn(current_model):
            return current_model.compute_stage_a_loss(
                rng, obs, actions, task_ids=batch_tasks, progress=batch_progress,
                episode_ids=batch_episodes, sample_indices=batch_times, train=True,
            )

        diff = nnx.DiffState(0, trainable)
        (loss, metrics), grads = nnx.value_and_grad(loss_fn, argnums=diff, has_aux=True)(module)
        grads = mask_earlier_vlm_layers(
            grads, total_layers=vlm_depth, trainable_layers=config.trainable_vlm_layers
        )
        params = nnx.state(module, trainable)
        updates, optimizer_state = tx.update(grads, optimizer_state, params)

        def scale_update(path, update):
            joined = _joined_path(path)
            rate = args.learning_rate_backbone if "llm" in joined else args.learning_rate_new
            return update.replace(value=-rate * update.value)

        updates = updates.map(scale_update)
        nnx.update(module, optax.apply_updates(params, updates))
        return loss, metrics, optimizer_state

    rng = jax.random.key(args.seed + 1)
    permutation = np.random.default_rng(args.seed).permutation(len(data["actions"]))
    records = []
    start_time = time.perf_counter()
    previous_log = start_time
    for step in range(args.steps):
        indices = permutation[(np.arange(step * args.batch_size, (step + 1) * args.batch_size) % len(permutation))]
        rows = {name: value[indices] for name, value in data.items()
                if isinstance(value, np.ndarray) and value.ndim and len(value) == len(data["actions"])}
        obs = observation_batch(rows)
        actions = jnp.asarray(rows["actions"], jnp.float32)
        rng, step_rng = jax.random.split(rng)
        update_start = time.perf_counter()
        loss, metrics, opt_state = train_step(
            step_rng, model, opt_state, obs, actions, jnp.asarray(task_ids[indices]),
            jnp.asarray(progress[indices]), jnp.asarray(episode_ids[indices]),
            jnp.asarray(sample_indices[indices]),
        )
        jax.block_until_ready(loss)
        elapsed = time.perf_counter() - update_start
        record = {"step": step + 1, "loss": float(loss),
                  **{key: float(value) for key, value in metrics.items()},
                  "update_seconds_including_first_compile": elapsed}
        records.append(record)
        if (step + 1) % 5 == 0 or step + 1 == args.steps:
            now = time.perf_counter()
            print({**record, "seconds_per_update_since_log": (now - previous_log) / min(5, step + 1)}, flush=True)
            previous_log = now

    checkpoint = out / "params"
    ocp.PyTreeCheckpointer().save(checkpoint, nnx.state(model).to_pure_dict())
    try:
        memory_stats = jax.devices()[0].memory_stats() or {}
    except (AttributeError, RuntimeError):
        memory_stats = {}
    (out / "train_metrics.json").write_text(json.dumps({
        "stage": "A", "steps": args.steps, "batch_size": args.batch_size,
        "dataset": str(Path(args.libero90_cache).resolve()), "data_fingerprint": data_fingerprint(data),
        "source_rows": len(data["actions"]), "source_tasks": int(len(np.unique(task_ids))),
        "checkpoint_init": str(Path(args.checkpoint).resolve()), "output_checkpoint": str(checkpoint.resolve()),
        "memory_fraction": args.memory_fraction, "trainable_vlm_layers": config.trainable_vlm_layers,
        "learning_rate_new": args.learning_rate_new, "learning_rate_backbone": args.learning_rate_backbone,
        "loss_weights": {"flow": config.fm_loss_weight, "query_nce": config.query_loss_weight,
                         "skill_huber": config.skill_reconstruction_weight},
        "optimizer": "Adam, no weight decay (paper details unspecified)",
        "gpu_memory_stats": {key: int(value) for key, value in memory_stats.items()
                             if isinstance(value, (int, np.integer)) and "bytes" in key},
        "elapsed_seconds": time.perf_counter() - start_time, "steps": records,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
