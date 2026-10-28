#!/usr/bin/env python3
"""Short, in-process LIBERO closed-loop smoke for the S2 checkpoint path."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from skillnet_client import image_tools

from openpi.models import model as _model
from openpi.policies import libero_policy as _libero_policy
from openpi.policies import policy as _policy
from openpi.training import checkpoints as _checkpoints
from openpi.training import config_moe_skill as _config
from smoke import DEFAULT_CHECKPOINT, load_model
from memory import (load_memory, query_raw_action,
                    retrieval_confidence, sample_memory_anchor, sample_optimus_anchor,
                    shuffle_memory_values)
from openpi.models.pi0_anchor_flow import mix_anchor_with_noise

ANNOTATIONS = Path(__file__).resolve().parents[2] / "libero" / "annotations" / "libero_skill_obj_annotations.json"


class _AnchorPolicy(_policy.Policy):
    def __init__(self, model, memory=None, *, condition="anchor_disabled_t0_1", seed=0,
                 absolute_similarity_low=0.2, absolute_similarity_high=0.8, fixed_t0=0.5):
        super().__init__(model)
        self._anchor_memory = memory
        self.condition = condition
        self.last_t0 = 1.0
        self._rng_np = np.random.default_rng(seed)
        self._t0_history = []
        self.absolute_similarity_low = absolute_similarity_low
        self.absolute_similarity_high = absolute_similarity_high
        self.fixed_t0 = fixed_t0

    def infer(self, obs, *, noise=None):
        inputs = self._input_transform(jax.tree.map(lambda x: x, obs))
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], inputs)
        self._rng, rng = jax.random.split(self._rng)
        observation = _model.Observation.from_dict(inputs)
        if self._anchor_memory is None or self.condition == "anchor_disabled_t0_1":
            if noise is None:
                noise = jax.random.normal(rng, (1, self._model.action_horizon, self._model.action_dim))
            actions = self._model.sample_actions(rng, observation, num_steps=1, x_src=noise, t0=1.0)
            self.last_t0 = 1.0
        else:
            key = self._model.encode_prefix_key(observation)[0]
            key_np, state_np = np.asarray(key), np.asarray(observation.state[0])
            max_similarity, normalized_entropy = retrieval_confidence(self._anchor_memory, key_np)
            if self.condition == "rts_raw_chunk_entropy":
                anchor, _ = query_raw_action(self._anchor_memory, key_np)
            elif self.condition == "optimus_gaussian_entropy":
                anchor = sample_optimus_anchor(
                    self._anchor_memory, key_np, state_np, seed=int(self._rng_np.integers(2**31))
                )
            else:
                anchor = sample_memory_anchor(
                    self._anchor_memory, key_np, state_np, seed=int(self._rng_np.integers(2**31))
                )
            if self.condition == "anchor_absolute_similarity":
                # Pilot thresholds are explicit CLI values and must be
                # calibrated once on LIBERO-90 before the production run.
                confidence = np.clip((max_similarity - self.absolute_similarity_low)
                                     / max(self.absolute_similarity_high - self.absolute_similarity_low, 1e-8), 0, 1)
                t0 = 1.0 - 0.8 * float(confidence)
            elif self.condition == "anchor_fixed_t0":
                t0 = self.fixed_t0
            elif self.condition == "anchor_t0_shuffled":
                calculated = 0.2 + 0.8 * normalized_entropy
                t0 = (float(self._rng_np.choice(self._t0_history)) if self._t0_history else 0.6)
                self._t0_history.append(calculated)
            else:
                t0 = 0.2 + 0.8 * normalized_entropy
            if self.condition != "anchor_absolute_similarity" or max_similarity < self.absolute_similarity_low:
                t0 = float(np.clip(t0, 0.2, 1.0))
            self.last_t0 = t0
            epsilon = jax.random.normal(rng, (1, self._model.action_horizon, self._model.action_dim))
            source = mix_anchor_with_noise(jnp.asarray(anchor)[None], epsilon, t0)
            actions = self._model.sample_actions_from_anchor(
                observation, x_src=source, t0=jnp.asarray([t0]), num_steps=1
            )
        outputs = {"state": inputs["state"], "actions": actions}
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        return self._output_transform(outputs)


def _quat2axisangle(quat):
    quat = np.asarray(quat).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(max(1.0 - quat[3] * quat[3], 0.0))
    if den < 1e-8:
        return np.zeros(3, np.float32)
    return (quat[:3] * (2.0 * np.arccos(quat[3])) / den).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--max-env-steps", type=int, default=24)
    parser.add_argument("--memory", help="optional S2 M6 memory; omitted uses exact t0=1 fallback")
    parser.add_argument("--condition", default="anchor_disabled_t0_1")
    parser.add_argument("--absolute-similarity-low", type=float, default=0.2)
    parser.add_argument("--absolute-similarity-high", type=float, default=0.8)
    parser.add_argument("--fixed-t0", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.trials < 1 or args.max_env_steps < 1:
        raise ValueError("trials and max-env-steps must be positive")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    model, model_config = load_model(args.checkpoint)
    training_config = _config.get_config("pi05_libero_moe_skill_4_90")
    data_config = training_config.data.create(training_config.assets_dirs, training_config.model)
    checkpoint_dir = Path(args.checkpoint).parent
    norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)
    memory = load_memory(args.memory) if args.memory else None
    if memory is not None and memory.key_text_source != "instruction_only":
        raise ValueError("rollout_smoke currently uses instruction-only query text; memory keys must match")
    if args.condition == "anchor_memory_shuffled" and memory is not None:
        memory = shuffle_memory_values(memory, seed=args.seed)
    policy = _AnchorPolicy(
        model, memory, condition=args.condition, seed=args.seed,
        absolute_similarity_low=args.absolute_similarity_low,
        absolute_similarity_high=args.absolute_similarity_high, fixed_t0=args.fixed_t0,
    )
    # Rebuild the same input/output transforms as create_trained_policy. The
    # local policy module is used only to reuse checkpoint normalization and
    # standard LIBERO image/state preprocessing.
    import openpi.transforms as _transforms
    policy._input_transform = _transforms.compose([
        _transforms.InjectDefaultPrompt(None),
        *data_config.data_transforms.inputs,
        _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.model_transforms.inputs,
    ])
    policy._output_transform = _transforms.compose([
        *data_config.model_transforms.outputs,
        _transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.data_transforms.outputs,
    ])

    suite = benchmark.get_benchmark_dict()["libero_10"]()
    if not 0 <= args.task_id < suite.n_tasks:
        raise ValueError(f"task-id must be in [0,{suite.n_tasks})")
    task = suite.get_task(args.task_id)
    annotations = json.loads(ANNOTATIONS.read_text())
    if str(task.language) not in annotations:
        raise KeyError(f"No SkillNet skill annotation for LIBERO-10 task {task.language!r}")
    task_annotation = annotations[str(task.language)]
    initial_states = suite.get_task_init_states(args.task_id)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    env.seed(7)
    records = []
    started = time.perf_counter()
    dummy = [0.0] * 6 + [-1.0]
    try:
        for trial in range(args.trials):
            env.reset()
            obs = env.set_init_state(initial_states[trial % len(initial_states)])
            for _ in range(10):
                obs, _, _, _ = env.step(dummy)
            done = False
            steps = 0
            while not done and steps < args.max_env_steps:
                image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                image = image_tools.convert_to_uint8(image_tools.resize_with_pad(image, 224, 224))
                wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, 224, 224))
                element = {
                    "observation/image": image,
                    "observation/wrist_image": wrist,
                    "observation/state": np.concatenate((obs["robot0_eef_pos"],
                        _quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])),
                    "prompt": str(task.language),
                    # The selected LIBERO-90 SkillNet checkpoint requires the
                    # same skill/object conditioning used by its regular policy.
                    "skills": str(task_annotation["all_classes"]),
                    "objects": str(task_annotation["objects"]),
                }
                chunk = policy.infer(element)["actions"]
                for action in chunk[: min(5, args.max_env_steps - steps)]:
                    obs, _, done, _ = env.step(np.asarray(action).tolist())
                    steps += 1
                    if done or steps >= args.max_env_steps:
                        break
            records.append({"task_id": args.task_id, "task": str(task.language), "trial": trial,
                            "env_steps": steps, "success": bool(done), "t0_last": policy.last_t0,
                            "anchor_condition": "M6 mixture" if memory is not None else "ordinary-noise compatibility fallback"})
    finally:
        env.close()
    result = {"checkpoint": str(Path(args.checkpoint).resolve()), "suite": "libero_10",
              "model": {"action_horizon": model_config.action_horizon, "action_dim": model_config.action_dim},
              "query_text_source": "instruction_only",
              "skill_annotation": "same task-level annotation as standard SkillNet evaluator; identical for all conditions and not used in encode_prefix_key",
              "trials": records, "elapsed_seconds": time.perf_counter() - started,
              "anchor_enabled": memory is not None and args.condition != "anchor_disabled_t0_1",
              "condition": args.condition, "short_horizon_smoke_only": True}
    (out / "rollout.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
