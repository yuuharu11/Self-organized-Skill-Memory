#!/usr/bin/env python3
"""Serve one S2 checkpoint to parallel LIBERO websocket clients."""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import socket

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.models import pi0_config as _plain_pi0_config
from openpi.models import model as _model
from openpi.models.pi0_anchor_flow import Pi0AnchorFlow
from openpi.models import tokenizer as _tokenizer
from openpi.policies import policy_config as _policy_config
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.shared import nnx_utils
from openpi.training import checkpoints as _checkpoints
from openpi.training import config_moe_skill as _config
from openpi.training.weight_loaders import CheckpointWeightLoader

from s2_policy import S2WebsocketPolicy
from smoke import DEFAULT_CHECKPOINT, load_model
from research_paths import STATE_DIR

DEFAULT_PI05_GENERATOR = str(STATE_DIR / "checkpoints/openpi-assets/checkpoints/pi05_base/params")


def load_generator(checkpoint: str, seed: int):
    config = _plain_pi0_config.Pi0Config(
        pi05=True, action_horizon=10, action_dim=32,
        paligemma_variant="gemma_2b", action_expert_variant="gemma_300m",
    )
    model = config.create(jax.random.key(seed))
    state = nnx.state(model)
    loaded = CheckpointWeightLoader(checkpoint).load(state.to_pure_dict())
    state.replace_by_pure_dict(loaded)
    nnx.update(model, state)
    return model


def main():
    import torch
    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help="same S2 fine-tuned snapshot for every evaluation condition")
    parser.add_argument("--memory-dir", default="",
                        help="directory containing initial_libero90.pkl and after_XX_adds.pkl snapshots")
    parser.add_argument("--normalization-sha256", default="",
                        help="required when any anchor memory condition is used")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--query-text-source", choices=("instruction_only", "pi05_generated_subtask"),
                        default="instruction_only")
    parser.add_argument("--generator-checkpoint", default=DEFAULT_PI05_GENERATOR)
    parser.add_argument("--absolute-similarity-low", type=float, default=0.2)
    parser.add_argument("--absolute-similarity-high", type=float, default=0.8)
    parser.add_argument("--fixed-t0", type=float, default=0.5)
    parser.add_argument("--generated-subtask-tokens", type=int, default=32)
    parser.add_argument("--calibration", default="", help="LIBERO-90-only calibrated temperature and monotone t0 table")
    parser.add_argument("--anchor-source", "--anchor-v2-source", dest="anchor_v2_source",
                        choices=("dec_latent", "dec_each", "raw"), default="dec_each")
    parser.add_argument("--anchor-v2-tau", type=float, default=0.05)
    parser.add_argument("--anchor-v2-t0-mode", choices=("calibration", "fixed", "one"), default="fixed")
    parser.add_argument("--anchor-v2-fixed-t0", type=float, default=0.5)
    parser.add_argument("--anchor-v2-calibration", default="",
                        help="JSON with similarity edges, per-bin t0, and holdout gate_threshold")
    parser.add_argument("--anchor-v2-gate-threshold", type=float, default=None,
                        help="top-1 similarity <= threshold falls back to t0=1; default can come from calibration JSON")
    args = parser.parse_args()
    if not 0 < args.port < 65536:
        raise ValueError("port must be in [1,65535]")
    if args.absolute_similarity_high <= args.absolute_similarity_low:
        raise ValueError("absolute similarity high threshold must exceed low threshold")

    model, model_config = load_model(args.checkpoint)
    train_config = _config.get_config("pi05_libero_moe_skill_4_90")
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    base_checkpoint = Path(DEFAULT_CHECKPOINT).parent
    norm_stats = _checkpoints.load_norm_stats(base_checkpoint / "assets", data_config.asset_id)
    inputs = [
        *_transforms_for_server(data_config, norm_stats, inputs=True),
    ]
    outputs = _transforms_for_server(data_config, norm_stats, inputs=False)
    generator = load_generator(args.generator_checkpoint, args.seed + 1) if \
        args.query_text_source == "pi05_generated_subtask" else None
    policy = S2WebsocketPolicy(
        model, input_transforms=inputs, output_transforms=outputs, memory_dir=args.memory_dir,
        normalization_hash=args.normalization_sha256, generator_model=generator,
        generator_checkpoint=args.generator_checkpoint, seed=args.seed,
        absolute_similarity_low=args.absolute_similarity_low,
        absolute_similarity_high=args.absolute_similarity_high, fixed_t0=args.fixed_t0,
        generated_subtask_tokens=args.generated_subtask_tokens,
        calibration_path=args.calibration,
        anchor_v2_source=args.anchor_v2_source, anchor_v2_tau=args.anchor_v2_tau,
        anchor_v2_t0_mode=args.anchor_v2_t0_mode, anchor_v2_fixed_t0=args.anchor_v2_fixed_t0,
        anchor_v2_calibration_path=args.anchor_v2_calibration,
        anchor_v2_gate_threshold=args.anchor_v2_gate_threshold,
    )
    server = WebsocketPolicyServer(policy, host=args.host, port=args.port,
        metadata={"model": "pi0.5-libero90-s2", "checkpoint": str(Path(args.checkpoint).resolve()),
                  "action_horizon": model_config.action_horizon, "action_dim": model_config.action_dim,
                  "query_text_source": args.query_text_source,
                  "normalization_sha256": args.normalization_sha256})
    logging.info("serving S2 on %s:%d", args.host, args.port)
    server.serve_forever()


def _transforms_for_server(data_config, norm_stats, *, inputs: bool):
    from openpi import transforms as _transforms
    if inputs:
        return [
            _transforms.InjectDefaultPrompt(None), *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ]
    return [
        *data_config.model_transforms.outputs,
        _transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.data_transforms.outputs,
    ]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    from openpi import transforms as _transforms
    main()
