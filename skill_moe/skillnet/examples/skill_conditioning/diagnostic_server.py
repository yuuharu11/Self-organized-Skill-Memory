"""Frozen SkillNet server returning final-denoising routing diagnostics.

No training or changed policy equations. Expert IDs are not semantic skill IDs.
Only action-token diagnostics are returned; prefix/state tokens are excluded.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi import transforms
from openpi.models import model as model_api
from openpi.models.pi0_config_moe_skill import Pi0Config
from openpi.models.pi0_moe_skill import Pi0
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.shared import nnx_utils
from openpi.training import checkpoints, config_moe_skill
from openpi.training.weight_loaders import CheckpointWeightLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "memory_pilot/anchor_flow"))
from s2_policy import integer_object_ids
from serve_s2_policy import _transforms_for_server


class DiagnosticPi0(Pi0):
    def diagnose(self, rng, observation, noise):
        actions, _, routes, selected_norm, shared_norm = self.sample_actions(
            rng, observation, num_steps=10, noise=noise, capture_auxiliary=True)
        h = self.action_horizon
        return actions, routes[:, :, -h:], selected_norm[:, :, -h:], shared_norm[:, :, -h:]


class DiagnosticPolicy:
    def __init__(self, checkpoint):
        cfg = Pi0Config(pi05=True, action_horizon=10, action_dim=32, max_token_len=200,
                        discrete_state_input=False, skill_num=6, skill_embed_dim=64)
        self.model = DiagnosticPi0(cfg, rngs=nnx.Rngs(jax.random.key(17)))
        state = nnx.state(self.model)
        loaded = CheckpointWeightLoader(checkpoint).load(state.to_pure_dict())
        state.replace_by_pure_dict(loaded)
        nnx.update(self.model, state)
        train = config_moe_skill.get_config("pi05_libero_moe_skill_4_90")
        data = train.data.create(train.assets_dirs, train.model)
        stats = checkpoints.load_norm_stats(Path(checkpoint).parent / "assets", data.asset_id)
        self.inputs = transforms.compose(_transforms_for_server(data, stats, inputs=True))
        self.outputs = transforms.compose(_transforms_for_server(data, stats, inputs=False))
        self.sample = nnx_utils.module_jit(self.model.diagnose)

    def infer(self, observation):
        raw = dict(observation)
        seed = int(raw.pop("s2_rng_seed"))
        no_skills = bool(raw.pop("diagnostic_no_skills", False))
        values = integer_object_ids(self.inputs(raw))
        if no_skills:
            values["skill_mask"] = np.zeros_like(values["skill_mask"], dtype=bool)
        batched = jax.tree.map(lambda x: jnp.asarray(x)[None], values)
        obs = model_api.Observation.from_dict(batched)
        _, key = jax.random.split(jax.random.key(seed))
        noise = jax.random.normal(key, (1, 10, 32))
        actions, routes, selected, shared = self.sample(key, obs, noise)
        result = self.outputs({"state": batched["state"], "actions": actions})
        result = jax.tree.map(lambda x: np.asarray(x[0]), result)
        result["diagnostics"] = {
            "router_top1": np.asarray(routes[:, 0], np.int32),
            "selected_contribution_norm": np.asarray(selected[:, 0], np.float32),
            "shared_output_norm": np.asarray(shared[:, 0], np.float32),
            "skills_after_transform": np.asarray(values["skills"], np.int32),
            "skill_mask": np.asarray(values["skill_mask"], bool),
            "scope": "18 layers x 10 action tokens; final denoising iteration only",
        }
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    policy = DiagnosticPolicy(args.checkpoint)
    # Compilation can exceed default websocket keepalive timeouts.
    import websockets.asyncio.server
    serve = websockets.asyncio.server.serve
    def without_keepalive(*a, **kw):
        kw.update(ping_interval=None, ping_timeout=None, open_timeout=300)
        return serve(*a, **kw)
    websockets.asyncio.server.serve = without_keepalive
    print("DIAGNOSTIC_MODEL_READY", args.port, flush=True)
    WebsocketPolicyServer(policy, host="127.0.0.1", port=args.port,
                          metadata={"model": "frozen SkillNet diagnostic", "checkpoint": args.checkpoint}).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
