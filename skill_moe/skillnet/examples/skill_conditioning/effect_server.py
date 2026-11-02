"""Unchanged public action sampler with a separate, read-only prefix encoder.

Feature extraction never adds an auxiliary output to the action sampling JIT.
The caller supplies the original task prompt for monitoring after any update.
"""
import argparse
import logging
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch
import websockets.asyncio.server

from smoke import DEFAULT_CHECKPOINT, load_model
from s2_policy import S2WebsocketPolicy, integer_object_ids
from serve_s2_policy import _transforms_for_server
from openpi.models import model as model_lib
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.training import checkpoints, config_moe_skill


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    model, _ = load_model(DEFAULT_CHECKPOINT)
    train = config_moe_skill.get_config("pi05_libero_moe_skill_4_90")
    data = train.data.create(train.assets_dirs, train.model)
    norm = checkpoints.load_norm_stats(Path(DEFAULT_CHECKPOINT).parent / "assets", data.asset_id)
    base = S2WebsocketPolicy(model, input_transforms=_transforms_for_server(data, norm, inputs=True),
                             output_transforms=_transforms_for_server(data, norm, inputs=False),
                             memory_dir="", normalization_hash="", seed=7)

    class Public:
        def infer(self, observation):
            raw = dict(observation)
            feature_prompt = raw.pop("monitor_prompt", None)
            feature_only = raw.pop("monitor_only", False)
            raw.pop("s2_arm", None)
            raw["s2_condition"] = "anchor_disabled_t0_1"
            out = {} if feature_only else base.infer(raw)
            if feature_prompt is not None:
                start = time.perf_counter()
                feature_raw = dict(raw, prompt=str(feature_prompt))
                transformed = integer_object_ids(base._input_transform(feature_raw))
                inputs = jax.tree.map(lambda value: jnp.asarray(value)[None, ...], transformed)
                obs = model_lib.Observation.from_dict(inputs)
                out["monitor_feature"] = np.asarray(base._encode_key(base._key_only_observation(obs))[0], np.float32)
                out["monitor_seconds"] = time.perf_counter() - start
            return out

    original_serve = websockets.asyncio.server.serve
    def serve(*a, **kw):
        kw.update(ping_interval=None, ping_timeout=None, open_timeout=180)
        return original_serve(*a, **kw)
    websockets.asyncio.server.serve = serve
    logging.basicConfig(level=logging.INFO)
    print("MODEL_READY", args.port, flush=True)
    WebsocketPolicyServer(Public(), host="127.0.0.1", port=args.port,
                          metadata={"model": "frozen public n10; separate prefix monitoring",
                                    "checkpoint": DEFAULT_CHECKPOINT}).serve_forever()


if __name__ == "__main__":
    main()
