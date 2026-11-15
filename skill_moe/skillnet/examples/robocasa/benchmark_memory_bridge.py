#!/usr/bin/env python3
"""Parity and steady-state latency check for RM-D2's cached bridge server."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--memory-dir", type=Path, required=True)
    p.add_argument("--expert-dir", type=Path, required=True)
    p.add_argument("--prepared-dir", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=5)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)

    from serve_memory_policy import MemoryBridgePolicy
    import expert as expert_api
    import jax
    import jax.numpy as jnp
    from experiment import _load_cached_episode, state_storage_to_policy

    server_args = SimpleNamespace(
        checkpoint=args.checkpoint, memory_dir=args.memory_dir, expert_dir=args.expert_dir,
        weights="expert", scale_mode="s-sep", t0=.1, integral_steps=10,
        replan_steps=5, seed=7, port=0, query_log=None,
    )
    policy = MemoryBridgePolicy(server_args)
    # The prepared-memory manifest owns `demos`; expert-01/manifest.json is
    # only a compact training provenance record and intentionally has no rows.
    manifest = json.loads((args.memory_dir / "manifest.json").read_text())
    info = next(row for row in manifest["demos"] if row["split"] == "memory")
    start = 100
    observation, _, _, raw_state = expert_api._build_obs_and_target(
        info, start, args.memory_dir, args.checkpoint
    )
    raw_action, _, _ = _load_cached_episode(args.memory_dir, info["task_id"], info["episode"])
    history = np.zeros((10, 12), np.float32)
    past = np.asarray(raw_action[:start], np.float32)[-10:]
    if len(past):
        history[-len(past):] = expert_api.ex.action_storage_to_policy(past)
    state = state_storage_to_policy(np.asarray(raw_state, np.float32))

    # Deterministic scale reference from the base-checkpoint expert, sharing a
    # single image/language KV cache with the bridge path.
    prefix_context = policy.prefix_context_fn(observation)
    dim = int(policy.base_model.action_dim)
    noise_ref = jax.random.normal(jax.random.key(4112), (1, 50, dim))
    reference_norm = policy.base_reference_fn(
        jax.random.key(4113), observation, prefix_context, num_steps=10, noise=noise_ref
    )
    mean = np.asarray(policy.action_stats.mean, np.float32)
    std = np.asarray(policy.action_stats.std, np.float32)
    reference_phys = np.asarray(reference_norm, np.float32)[0] * (std[None, :] + 1e-6) + mean[None, :]
    scale_reference = policy.set_reference(reference_phys[:, :6])

    started = time.perf_counter()
    legacy = expert_api.serve_bridge_action(
        policy.bridge_model, observation, history, state, policy.pool, policy.p5_models,
        policy.action_stats, t0=.1, seed=4110, num_steps=10, device=policy.p5_device,
        prefix_context=None, scale_reference=scale_reference,
    )
    legacy_seconds = time.perf_counter() - started
    # First call may JIT the cached fixed-shape executable.
    started = time.perf_counter()
    fast_first = expert_api.serve_bridge_action(
        policy.bridge_model, observation, history, state, policy.pool, policy.p5_models,
        policy.action_stats, t0=.1, seed=4110, num_steps=10, device=policy.p5_device,
        compiled_bridge_fn=policy.bridge_fn, prefix_context=prefix_context,
        scale_reference=scale_reference,
    )
    first_seconds = time.perf_counter() - started
    max_abs = float(np.max(np.abs(np.asarray(legacy) - np.asarray(fast_first))))
    if max_abs > 1e-5:
        raise AssertionError(f"old/new bridge outputs differ by {max_abs}")

    timings = []
    outputs = [np.asarray(fast_first, np.float32)]
    for _ in range(int(args.repeats)):
        started = time.perf_counter()
        value = expert_api.serve_bridge_action(
            policy.bridge_model, observation, history, state, policy.pool, policy.p5_models,
            policy.action_stats, t0=.1, seed=4110, num_steps=10, device=policy.p5_device,
            compiled_bridge_fn=policy.bridge_fn, prefix_context=prefix_context,
            scale_reference=scale_reference,
        )
        timings.append(time.perf_counter() - started)
        outputs.append(np.asarray(value, np.float32))
        if np.max(np.abs(np.asarray(legacy) - np.asarray(value))) > 1e-5:
            raise AssertionError("steady output changed across cached calls")
    result = {
        "observation": {"task": info["task"], "episode": int(info["episode"]), "start": start},
        "shape": list(np.asarray(legacy).shape),
        "scale": "s-sep; pretrained reference used only for scale",
        "t0": .1,
        "p5_device": policy.p5_device,
        "legacy_one_call_seconds": legacy_seconds,
        "cached_first_call_seconds": first_seconds,
        "cached_steady_median_seconds": float(np.median(timings)),
        "cached_steady_p90_seconds": float(np.quantile(timings, .9)),
        "steady_repeats": len(timings),
        "max_abs_output_difference": max_abs,
        "all_output_parity": True,
    }
    (args.out / "speed.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(args.out / "parity_actions.npz", legacy=legacy, fast=fast_first, repeats=np.stack(outputs))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

