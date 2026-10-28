"""GPU verification for default-step identity and adapter-only updates.

Run after ``source env/activate.sh``. This loads the fixed LIBERO-40 checkpoint
and one saved rollout observation; it does not start a simulator or a server.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import inspect
import json
import pathlib
import uuid

from flax import nnx, traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi import transforms
from openpi.models import model as model_lib
from openpi.policies.policy import Policy
from openpi.training import checkpoints, config_moe_skill
try:
    from .adapter import MemoryInjection
except ImportError:  # Executed as a standalone verification script.
    from adapter import MemoryInjection

ROOT = pathlib.Path("/work")
STATE = ROOT / ".skillnet-state"
DEFAULT_CHECKPOINT = STATE / "checkpoints/pi05_libero_moe_skill_4_40/moe_add_balance_4_128_30000/29999"
DEFAULT_ROLLOUT = STATE / "outputs/memory-pilot/record-20261025T130411-ff650e"


def _make_dir() -> pathlib.Path:
    parent = STATE / "outputs" / "memory-pilot"
    parent.mkdir(parents=True, exist_ok=True)
    path = parent / f"review-checkpoint-{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"
    path.mkdir(exist_ok=False)
    return path


def _raw_observation(archive, index: int) -> dict:
    ids = (archive["skills"][index][archive["skill_mask"][index]] - 1).astype(np.int32).tolist()
    return {
        "observation/image": archive["img_base"][index],
        "observation/wrist_image": archive["img_wrist"][index],
        "observation/state": archive["proprio"][index],
        "prompt": str(archive["prompt"][index]),
        "skills": str(ids),
    }


def _base_state(model):
    state = nnx.state(model, nnx.Param).to_pure_dict()
    flat = traverse_util.flatten_dict(state)
    return {path: value for path, value in flat.items() if "memory_injection" not in path}


def _bitwise_unchanged(before, after) -> tuple[bool, int]:
    if before.keys() != after.keys():
        return False, len(set(before) ^ set(after))
    changed = 0
    for path, value_before in before.items():
        left = np.asarray(value_before)
        right = np.asarray(after[path])
        if left.dtype != right.dtype or left.shape != right.shape or not np.array_equal(left.view(np.uint8), right.view(np.uint8)):
            changed += 1
    return changed == 0, changed


def main(checkpoint: pathlib.Path = DEFAULT_CHECKPOINT, rollout_dir: pathlib.Path = DEFAULT_ROLLOUT) -> pathlib.Path:
    run_dir = _make_dir()
    manifest = json.loads((rollout_dir / "manifest.json").read_text())
    episode = manifest["episodes"][0]
    archive = np.load(rollout_dir / "episodes" / f"{episode['episode_id']}.npz")
    # Exercise a few saved observations without changing the recording.
    observation_count = min(2, len(archive["step"]))

    train_config = dataclasses.replace(
        config_moe_skill.get_config("pi05_libero_moe_skill_4_40"), exp_name=checkpoint.parent.name
    )
    model = train_config.model.load(model_lib.restore_params(checkpoint / "params", dtype=jnp.bfloat16))
    adapter = MemoryInjection(mode="action_tokens", dz=16, width=1024, hidden_dim=32, rngs=nnx.Rngs(20261025))
    model.attach_memory_injection(adapter)

    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    norm_stats = checkpoints.load_norm_stats(checkpoint / "assets", data_config.asset_id)
    policy = Policy(
        model,
        transforms=[
            transforms.InjectDefaultPrompt(None),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        # No override: the model signature's public default num_steps=10 is used.
    )
    num_steps_default = inspect.signature(model.sample_actions).parameters["num_steps"].default
    if num_steps_default != 10:
        raise AssertionError(f"expected public default num_steps=10, found {num_steps_default}")

    exact_matches = []
    noise = np.random.default_rng(20261025).standard_normal((10, train_config.model.action_dim)).astype(np.float32)
    for i in range(observation_count):
        raw = _raw_observation(archive, i)
        baseline = policy.infer(raw, noise=noise)["actions"]
        with_zero_adapter = policy.infer(
            {**raw, "memory_features": np.linspace(-1, 1, 16, dtype=np.float32), "memory_enabled": True},
            noise=noise,
        )["actions"]
        exact_matches.append(bool(np.array_equal(baseline, with_zero_adapter)))
        if not exact_matches[-1]:
            raise AssertionError(f"zero-init mismatch at saved observation {i}")

    # Hold references to every non-adapter parameter leaf. The optimizer below
    # is constructed over `adapter` only; compare raw bytes for all frozen leaves.
    base_before = _base_state(model)
    condition = jnp.linspace(-1, 1, 16, dtype=jnp.float32)[None, :]
    target = jnp.full((1, 1, 1024), 0.05, dtype=jnp.float32)
    loss_fn = lambda module: jnp.mean(jnp.square(module(c=condition, horizon=1) - target))
    initial_loss = float(loss_fn(adapter))
    optimizer = nnx.Optimizer(adapter, optax.adam(learning_rate=0.02), wrt=nnx.Param)
    losses = []
    for _ in range(5):
        optimizer.update(nnx.grad(loss_fn)(adapter))
        losses.append(float(loss_fn(adapter)))
    if not losses[-1] < initial_loss:
        raise AssertionError(f"adapter loss did not fall below initial loss: {initial_loss} -> {losses[-1]}")
    unchanged, changed_count = _bitwise_unchanged(base_before, _base_state(model))
    if not unchanged:
        raise AssertionError(f"{changed_count} non-adapter parameter leaves changed")

    result = {
        "checkpoint": str(checkpoint),
        "rollout_dir": str(rollout_dir),
        "episode_id": episode["episode_id"],
        "observations_checked": observation_count,
        "default_num_steps": int(num_steps_default),
        "zero_init_exact_matches": exact_matches,
        "adapter_training_steps": len(losses),
        "adapter_initial_loss": initial_loss,
        "adapter_step_losses": losses,
        "base_parameter_leaves_checked": len(base_before),
        "base_parameter_leaves_changed": changed_count,
        "base_parameter_bitwise_unchanged": unchanged,
    }
    (run_dir / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(run_dir)
    return run_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=pathlib.Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--rollout-dir", type=pathlib.Path, default=DEFAULT_ROLLOUT)
    args = parser.parse_args()
    main(args.checkpoint, args.rollout_dir)
