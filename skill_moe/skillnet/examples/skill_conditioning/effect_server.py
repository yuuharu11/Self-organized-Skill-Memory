"""Unchanged public action sampler with a separate, read-only prefix encoder.

Feature extraction never adds an auxiliary output to the action sampling JIT.
The caller supplies the original task prompt for monitoring after any update.
"""
import argparse
import json
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
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help="Full params directory; defaults to the original public checkpoint")
    parser.add_argument("--weight-audit-file", type=Path,
                        help="Optional expected parameter digests, checked after checkpoint reload")
    args = parser.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    model, _ = load_model(args.checkpoint)
    if args.weight_audit_file:
        import flax.nnx as nnx
        from smoke import parameter_digest
        from openpi.models.pi0_config_anchor_flow import action_expert_trainable_filter
        expected = json.loads(args.weight_audit_file.read_text())
        trainable = action_expert_trainable_filter()
        actual_frozen = parameter_digest(model, nnx.All(nnx.Param, nnx.Not(trainable)))
        actual_trainable = parameter_digest(model, trainable)
        if actual_frozen != expected.get("frozen_after", expected["frozen_before"]):
            raise ValueError("reloaded frozen parameter digest mismatch")
        if actual_trainable != expected.get("trainable_after", expected.get("trainable_before")):
            raise ValueError("reloaded action expert digest mismatch")
        audit = dict(checkpoint=args.checkpoint, frozen=actual_frozen, trainable=actual_trainable,
                     matches_training=True)
        args.weight_audit_file.with_name(f"server_load_audit_{args.port}.json").write_text(json.dumps(audit, indent=2))
    train = config_moe_skill.get_config("pi05_libero_moe_skill_4_90")
    data = train.data.create(train.assets_dirs, train.model)
    norm = checkpoints.load_norm_stats(Path(args.checkpoint).parent / "assets", data.asset_id)
    base = S2WebsocketPolicy(model, input_transforms=_transforms_for_server(data, norm, inputs=True),
                             output_transforms=_transforms_for_server(data, norm, inputs=False),
                             memory_dir="", normalization_hash="", seed=7)

    class Public:
        def infer(self, observation):
            raw = dict(observation)
            audit_tokens = raw.pop("diagnostic_tokens", False)
            tokens_only = raw.pop("diagnostic_tokens_only", False)
            feature_prompt = raw.pop("monitor_prompt", None)
            feature_only = raw.pop("monitor_only", False)
            raw.pop("s2_arm", None)
            raw["s2_condition"] = "anchor_disabled_t0_1"
            out = {} if feature_only or tokens_only else base.infer(raw)
            if audit_tokens or tokens_only:
                if train.model.discrete_state_input:
                    raise ValueError("token audit currently requires the frozen checkpoint's text-only tokenization")
                transformed = integer_object_ids(base._input_transform(dict(raw)))
                tokens = np.asarray(transformed["tokenized_prompt"], np.int32)
                mask = np.asarray(transformed["tokenized_prompt_mask"], bool)
                tokenizer = base._subtask_tokenizer._tokenizer
                clean = str(raw["prompt"]).strip().replace("_", " ").replace("\n", " ")
                full = tokenizer.encode(clean, add_bos=True) + tokenizer.encode("\n")
                expected = np.asarray(full[:len(tokens)], np.int32)
                exact = np.array_equal(tokens[:len(expected)], expected)
                out["token_audit"] = dict(ids=tokens.tolist(), mask=mask.tolist(),
                    decoded=tokenizer.decode(tokens[mask].tolist()), length=int(mask.sum()),
                    untruncated_length=len(full), limit=len(tokens), truncated=len(full)>len(tokens),
                    exact_encoding=bool(exact), discrete_state_input=False)
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
                                    "checkpoint": args.checkpoint}).serve_forever()


if __name__ == "__main__":
    main()
