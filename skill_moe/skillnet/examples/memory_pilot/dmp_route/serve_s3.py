#!/usr/bin/env python3
"""Serve a trained S3 DMP-routed π0.5 checkpoint over the LIBERO websocket API."""
from __future__ import annotations

import argparse
import importlib.util
import logging
import os
from pathlib import Path
import sys

import flax.nnx as nnx
import jax

from openpi.models.pi0_config_dmp_route import Pi0DmpRouteConfig
from openpi.policies import policy_config as _policy_config
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.training import checkpoints as _checkpoints
from openpi.training import config_moe_skill as _train_configs
from openpi.training.weight_loaders import CheckpointWeightLoader

from s3_policy import DmpRoutePolicy, load_prompt_map
from train_utils import load_s1_library


def import_s1_dmp(path: str | Path):
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location("dmp_route_s1_dmp", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import S1 DMP implementation from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _server_transforms(data_config, norm_stats, *, inputs):
    from openpi import transforms as _transforms
    if inputs:
        return [
            _transforms.InjectDefaultPrompt(None),
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ]
    return [
        *data_config.model_transforms.outputs,
        _transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.data_transforms.outputs,
    ]


def main():
    import torch

    thread_count = int(os.environ.get("OMP_NUM_THREADS", "8"))
    torch.set_num_threads(thread_count)
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="trained S3 checkpoint/snapshot")
    state = Path(os.environ.get("SKILLNET_STATE_DIR",
                 Path(__file__).resolve().parents[5] / ".skillnet-state")).expanduser()
    parser.add_argument("--s1-dir", default=str(state / "outputs/memory-pilot/dmp-route-20261102/s1-v3c"))
    parser.add_argument("--condition", choices=("G0", "G1"), default="G1")
    parser.add_argument("--base-config", default="pi05_libero_moe_skill_4_90")
    parser.add_argument("--prompt-level", choices=("L0", "L3"), default="L0")
    parser.add_argument("--prompts-json", default=str(state / "outputs/memory-pilot/abstract-prompt-20261101/prompts.json"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--bridge-t0", type=float, default=0.4)
    parser.add_argument("--denoise-steps", type=int, default=2)
    parser.add_argument("--injection-mode", choices=("both", "router_only", "adarms_only"), default="both")
    parser.add_argument("--shape-mode", choices=("learned", "straight"), default="learned")
    parser.add_argument("--primitive-mode", choices=("learned", "shuffle"), default="learned")
    parser.add_argument("--position-m-per-command", type=float, default=0.05)
    parser.add_argument("--rotation-rad-per-command", type=float, default=0.5)
    args = parser.parse_args()
    if not 0 < args.port < 65536:
        raise ValueError("port must be 1..65535")

    s1_dir = Path(args.s1_dir)
    if not (s1_dir / "DONE").exists():
        raise FileNotFoundError(f"S1-v3c interface is not complete: {s1_dir}")
    library = load_s1_library(s1_dir / "library.npz")
    dmp_module = import_s1_dmp(s1_dir / "dmp.py")

    train_config = _train_configs.get_config(args.base_config)
    base = train_config.model
    config = Pi0DmpRouteConfig(
        dtype=base.dtype,
        paligemma_variant=base.paligemma_variant,
        action_expert_variant=base.action_expert_variant,
        action_dim=32,
        action_horizon=10,
        max_token_len=base.max_token_len,
        pi05=True,
        # Match the public SkillNet-90 training config; a discrete-state
        # mismatch changes the prefix presented to the learned pointer.
        discrete_state_input=base.discrete_state_input,
        skill_num=base.skill_num,
        skill_embed_dim=base.skill_embed_dim,
        positional_encoding=base.positional_encoding,
        t_coding=base.t_coding,
        enable_primitive_router=(args.condition == "G1"),
        shape_basis_n=int(library["shapes"].shape[-1]),
        injection_mode=(args.injection_mode if args.condition == "G1" else "none"),
        bridge_t0=args.bridge_t0,
    )
    model = config.create(jax.random.key(args.seed))
    state = nnx.state(model)
    loaded = CheckpointWeightLoader(args.checkpoint).load(state.to_pure_dict())
    state.replace_by_pure_dict(loaded)
    nnx.update(model, state)

    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    stats = _checkpoints.load_norm_stats(Path(args.checkpoint).parent / "assets", data_config.asset_id)
    # S3 fine-tune snapshots may not carry their own assets; use the public
    # SkillNet-90 normalization bundle from the train config's asset directory.
    if stats is None:
        stats = _checkpoints.load_norm_stats(train_config.assets_dirs, data_config.asset_id)
    if stats is None or "actions" not in stats or stats["actions"].q01 is None:
        raise ValueError("LIBERO-90 action quantile statistics are required")
    policy = DmpRoutePolicy(
        model,
        input_transforms=_server_transforms(data_config, stats, inputs=True),
        output_transforms=_server_transforms(data_config, stats, inputs=False),
        dmp_module=dmp_module,
        library=library,
        action_q01=stats["actions"].q01,
        action_q99=stats["actions"].q99,
        prompt_map=load_prompt_map(args.prompts_json),
        prompt_level=args.prompt_level,
        bridge_t0=args.bridge_t0,
        num_steps=args.denoise_steps,
        injection_mode=(args.injection_mode if args.condition == "G1" else "none"),
        position_m_per_command=args.position_m_per_command,
        rotation_rad_per_command=args.rotation_rad_per_command,
        shape_mode=args.shape_mode,
        primitive_mode=args.primitive_mode,
        seed=args.seed,
    )
    server = WebsocketPolicyServer(
        policy,
        host=args.host,
        port=args.port,
        metadata={
            "model": f"skillnet-pi05-libero90-dmp-route-s3-{args.condition.lower()}",
            "condition": args.condition,
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "num_primitives": int(library["shapes"].shape[0]),
            "shape_mode": args.shape_mode,
            "primitive_mode": args.primitive_mode,
            "prompt_level": args.prompt_level,
            "bridge_t0": args.bridge_t0,
            "denoise_steps": args.denoise_steps,
        },
    )
    logging.info("serving S3 on %s:%d; primitive K=%d", args.host, args.port, len(library["shapes"]))
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
