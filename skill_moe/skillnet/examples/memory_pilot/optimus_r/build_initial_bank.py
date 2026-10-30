#!/usr/bin/env python3
"""Encode Stage-A LIBERO-90 rows and build the fixed 128-entry initial bank."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
EXAMPLES = HERE.parents[1]
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from bank import build_initial_bank
from train_stage_a import DEFAULT_CACHE, DEFAULT_CHECKPOINT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero90-cache", default=DEFAULT_CACHE)
    parser.add_argument("--checkpoint", required=True, help="Stage-A checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-rows", type=int, default=0, help="nonzero only for a smoke bank")
    parser.add_argument("--memory-fraction", type=float, default=0.35)
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_rows < 0:
        raise ValueError("invalid batch size or max_rows")
    import os
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", str(args.memory_fraction))
    import flax.nnx as nnx
    import jax
    import orbax.checkpoint as ocp
    from openpi.models.pi0_optimus_r_config import Pi0OptimusRConfig
    from openpi.shared import nnx_utils
    from openpi.training.weight_loaders import PartialCheckpointWeightLoader
    from data import data_fingerprint, load_cache, observation_batch, official_task_ids

    if jax.default_backend() != "gpu":
        raise RuntimeError(f"initial bank feature extraction expects a GPU, got {jax.default_backend()}")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    data = load_cache(args.libero90_cache)
    count = len(data["actions"]) if not args.max_rows else min(args.max_rows, len(data["actions"]))
    config = Pi0OptimusRConfig(
        pi05=True, action_horizon=10, action_dim=32, max_token_len=200,
        discrete_state_input=False, paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m_moe_4", skill_num=6, skill_embed_dim=64,
    )
    model = config.create(jax.random.key(args.seed))
    state = nnx.state(model)
    try:
        loaded = PartialCheckpointWeightLoader(args.checkpoint, missing_regex=".*").load(state.to_pure_dict())
    except KeyError:
        # Stage-A checkpoints are saved as a bare NNX parameter state (unlike
        # openpi training checkpoints, which wrap it under `params`). Restore
        # those against the initialized model tree so the sharding is explicit.
        import jax.sharding

        target = state.to_pure_dict()
        mesh = jax.sharding.Mesh(jax.devices(), ("x",))
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        restore_args = jax.tree.map(
            lambda _: ocp.ArrayRestoreArgs(sharding=sharding, restore_type=np.ndarray), target
        )
        with ocp.PyTreeCheckpointer() as checkpointer:
            loaded = checkpointer.restore(
                args.checkpoint,
                ocp.args.PyTreeRestore(item=target, restore_args=restore_args),
            )
    state.replace_by_pure_dict(loaded)
    nnx.update(model, state)
    encode = nnx_utils.module_jit(model.encode_memory, static_argnames=("train",))
    query = np.empty((count, config.query_dim), np.float32)
    skill = np.empty((count, config.skill_dim), np.float32)
    started = time.perf_counter()
    for start in range(0, count, args.batch_size):
        stop = min(start + args.batch_size, count)
        rows = {name: value[start:stop] for name, value in data.items()
                if isinstance(value, np.ndarray) and value.ndim and len(value) == len(data["actions"])}
        q_batch, s_batch = encode(observation_batch(rows), train=False)
        q_batch, s_batch = jax.device_get((q_batch, s_batch))
        query[start:stop] = np.asarray(q_batch, np.float32)
        skill[start:stop] = np.asarray(s_batch, np.float32)
        if stop % (args.batch_size * 100) == 0 or stop == count:
            print({"encoded": stop, "rows": count, "elapsed_seconds": time.perf_counter() - started}, flush=True)

    feature_extraction_seconds = time.perf_counter() - started

    feature_path = output / "stage_a_latents.npz"
    with feature_path.open("xb") as stream:
        np.savez_compressed(stream, query=query, skill=skill, episode_id=data["episode_id"][:count],
                            task_id=official_task_ids(data["source_file"][:count]),
                            frame=data["frame"][:count], source_file=data["source_file"][:count])
    bank_started = time.perf_counter()
    bank = build_initial_bank(query, skill, bank_size=128, query_seed_clusters=64, seed=args.seed)
    bank_clustering_seconds = time.perf_counter() - bank_started
    bank_path = output / "initial_bank_128.npz"
    with bank_path.open("xb") as stream:
        np.savez_compressed(stream, query=bank.query, skill=bank.skill, support=bank.support,
                            exemplars=bank.exemplars, exemplar_mask=bank.exemplar_mask)
    report = {**bank.metadata, "input_cache": str(Path(args.libero90_cache).resolve()),
              "input_checkpoint": str(Path(args.checkpoint).resolve()),
              "source_data_fingerprint": data_fingerprint(data), "encoded_rows": count,
              "query_shape": list(bank.query.shape), "skill_shape": list(bank.skill.shape),
              "support_min": int(bank.support.min()), "support_max": int(bank.support.max()),
              "feature_extraction_seconds": feature_extraction_seconds,
              "bank_clustering_seconds": bank_clustering_seconds,
              "features": str(feature_path.resolve()), "bank": str(bank_path.resolve()),
              "smoke_subset": bool(args.max_rows)}
    with (output / "bank_report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
