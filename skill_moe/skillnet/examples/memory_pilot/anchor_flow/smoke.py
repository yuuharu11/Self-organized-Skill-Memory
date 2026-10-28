#!/usr/bin/env python3
"""Checkpoint-backed S2 smoke: compatibility, anchored sampling, and 2 updates."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import time

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi.models.pi0_anchor_flow import Pi0AnchorFlow, mix_anchor_with_noise
from openpi.models.pi0_config_anchor_flow import Pi0AnchorFlowConfig, action_expert_trainable_filter
from openpi.training.weight_loaders import CheckpointWeightLoader
from openpi.models import model as _model

DEFAULT_CHECKPOINT = "/work/.skillnet-state/checkpoints/pi05_libero_moe_skill_4_90/moe_balance_4_32_90_20000/19999/params"


def parameter_digest(model,filter):
    digest=hashlib.sha256()
    for path,value in sorted(nnx.state(model,filter).flat_state().items()):
        digest.update(str(path).encode());digest.update(np.asarray(value.value).tobytes())
    return digest.hexdigest()


def observation(config, batch=1):
    images = {name: jnp.zeros((batch, 224, 224, 3), jnp.float32) for name in
              ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")}
    image_masks = {name: jnp.ones((batch,), jnp.bool_) for name in images}
    prompt = jnp.zeros((batch, config.max_token_len), jnp.int32).at[:, :5].set(
        jnp.asarray([2, 100, 101, 102, 1], jnp.int32)
    )
    prompt_mask = jnp.arange(config.max_token_len)[None, :] < 5
    return _model.Observation(
        images=images,
        image_masks=image_masks,
        state=jnp.zeros((batch, 32), jnp.float32),
        tokenized_prompt=prompt,
        tokenized_prompt_mask=jnp.broadcast_to(prompt_mask, prompt.shape),
        skills=jnp.zeros((batch, 1), jnp.int32),
        skill_mask=jnp.ones((batch, 1), jnp.bool_),
    )


def load_model(checkpoint: str) -> tuple[Pi0AnchorFlow, Pi0AnchorFlowConfig]:
    config = Pi0AnchorFlowConfig(
        pi05=True, action_horizon=10, action_dim=32, max_token_len=200,
        discrete_state_input=False, paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m_moe_4", skill_num=6, skill_embed_dim=64,
    )
    model = config.create(jax.random.key(17))
    state = nnx.state(model)
    loaded = CheckpointWeightLoader(checkpoint).load(state.to_pure_dict())
    state.replace_by_pure_dict(loaded)
    nnx.update(model, state)
    return model, config


def _make_train_step(tx):
    trainable = action_expert_trainable_filter()

    @nnx.jit
    def step(rng, model, opt_state, obs, actions, anchor_actions):
        def loss_fn(current_model, step_rng):
            loss, balance = current_model.compute_anchor_loss(step_rng, obs, actions, anchor_actions, train=True)
            return jnp.mean(loss) + balance, (jnp.mean(loss), balance)

        diff = nnx.DiffState(0, trainable)
        (loss, aux), grads = nnx.value_and_grad(loss_fn, argnums=diff, has_aux=True)(model, rng)
        params = nnx.state(model, trainable)
        updates, opt_state = tx.update(grads, opt_state, params)
        nnx.update(model, optax.apply_updates(params, updates))
        return loss, aux, opt_state

    return step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        raise ValueError("steps and batch-size must be positive")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"S2 checkpoint smoke requires GPU, got {jax.default_backend()}")

    model, config = load_model(args.checkpoint)
    obs = observation(config, batch=args.batch_size)
    rng = jax.random.key(27)
    noise = jax.random.normal(rng, (args.batch_size, 10, 32))
    before = model.sample_actions(rng, obs, num_steps=2, noise=noise)
    compat = model.sample_actions(rng, obs, num_steps=2, x_src=noise, t0=1.0)
    jax.block_until_ready((before, compat))
    max_abs = float(jnp.max(jnp.abs(before - compat)))
    if not np.array_equal(np.asarray(before), np.asarray(compat)):
        raise AssertionError(f"t0=1 is not exactly compatible, max_abs={max_abs}")
    anchor_action = jnp.ones_like(noise) * 0.1
    anchor_t0 = jnp.full((args.batch_size,), 0.5)
    anchor_start = mix_anchor_with_noise(anchor_action, noise, anchor_t0)
    anchored = model.sample_actions_from_anchor(
        obs, x_src=anchor_start, t0=anchor_t0, num_steps=2
    )
    jax.block_until_ready(anchored)
    if anchored.shape != (args.batch_size, 10, 32):
        raise AssertionError(f"unexpected anchor output shape {anchored.shape}")

    trainable = action_expert_trainable_filter()
    frozen=nnx.All(nnx.Param,nnx.Not(trainable))
    frozen_before=parameter_digest(model,frozen)
    trainable_before=parameter_digest(model,trainable)
    tx = optax.adamw(1e-6)
    opt_state = tx.init(nnx.state(model, trainable))
    train_step = _make_train_step(tx)
    actions = jnp.zeros((args.batch_size, 10, 32), jnp.float32)
    rows = []
    started = time.perf_counter()
    step_seconds = []
    for index in range(args.steps):
        rng, step_rng = jax.random.split(rng)
        anchor_actions = jnp.ones_like(actions) * 0.1
        step_started = time.perf_counter()
        loss, (flow_loss, balance_loss), opt_state = train_step(
            step_rng, model, opt_state, obs, actions, anchor_actions
        )
        jax.block_until_ready(loss)
        step_seconds.append(time.perf_counter() - step_started)
        rows.append({"step": index + 1, "loss": float(loss), "flow_loss": float(flow_loss),
                     "balance_loss": float(balance_loss)})
    elapsed = time.perf_counter() - started
    frozen_after=parameter_digest(model,frozen)
    trainable_after=parameter_digest(model,trainable)
    if frozen_before != frozen_after or trainable_before == trainable_after:
        raise AssertionError('frozen parameters changed or action-expert parameters did not update')
    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "model_config": {"pi05": True, "action_horizon": 10, "action_dim": 32,
                         "paligemma_variant": "gemma_2b", "action_expert_variant": "gemma_300m_moe_4"},
        "backend": jax.default_backend(),
        "batch_size": args.batch_size,
        "t0_1_max_abs": max_abs,
        "anchor_shape": list(anchored.shape),
        "anchor_start_shape": list(anchor_start.shape),
        "anchor_start_formula_max_abs": float(jnp.max(jnp.abs(
            anchor_start - ((1.0 - anchor_t0[:, None, None]) * anchor_action
                            + anchor_t0[:, None, None] * noise)))),
        "training_steps": rows,
        "training_seconds_including_first_compile": elapsed,
        "per_step_seconds_including_first_compile": step_seconds,
        "steady_step_seconds_median": float(np.median(step_seconds[1:])) if len(step_seconds) > 1 else None,
        "seconds_per_step_amortized": elapsed / args.steps,
        "synthetic_data_only": True,
        "frozen_parameters_identical":frozen_before==frozen_after,
        "action_parameters_changed":trainable_before!=trainable_after,
    }
    (out / "smoke.json").write_text(__import__("json").dumps(result, indent=2) + "\n")
    print(result)


if __name__ == "__main__":
    main()
