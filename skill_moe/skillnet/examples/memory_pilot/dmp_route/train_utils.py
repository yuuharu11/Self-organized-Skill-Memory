"""S3 trainable filters and S1 batch validation helpers."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def load_s1_library(path: str | Path):
    """Read an S1-v3c coefficient library without imposing a dictionary K."""
    with np.load(path, allow_pickle=False) as data:
        if "grip" not in data or not ("coef" in data or "shapes" in data):
            raise ValueError("S1 library requires coef [K,6,N] (or legacy shapes) and grip [K]")
        result = {key: np.asarray(data[key]) for key in data.files}
    if "coef" in result:
        coeff = np.asarray(result["coef"], np.float32)
        if str(np.asarray(result.get("contract_version", ""))) != "dmp-v3c-raw-action-coef16-v1":
            raise ValueError("DMP S3 training requires the explicit s1-v3c library contract")
        # The pointer's w input is the canonical normalized coefficient shape,
        # never the visualization-only sampled curve.
        result["shapes"] = coeff
    shapes = np.asarray(result["shapes"], np.float32)
    grip = np.asarray(result["grip"]).reshape(-1)
    if shapes.ndim != 3 or shapes.shape[1] != 6 or shapes.shape[0] != len(grip):
        raise ValueError(f"invalid S1 library coefficient/grip arrays: {shapes.shape}, {grip.shape}")
    result["shapes"] = shapes
    result["grip"] = grip
    return result


def validate_s1_frames(path: str | Path, num_primitives: int):
    """Validate the self-decomposition targets; no task/subtask labels read."""
    required = {"episode_id", "task_id", "t", "prim_id", "seg_start", "seg_end", "g", "tau", "phase_s", "split"}
    with np.load(path, allow_pickle=False) as z:
        missing = required - set(z.files)
        if missing:
            raise ValueError(f"S1 frames missing keys: {sorted(missing)}")
        data = {key: np.asarray(z[key]) for key in required}
    count = len(data["prim_id"])
    for key in required:
        if len(data[key]) != count:
            raise ValueError(f"S1 frame field {key} row count differs")
    if data["g"].shape != (count, 6):
        raise ValueError(f"S1 g must be [N,6], got {data['g'].shape}")
    if np.any((data["prim_id"] < 0) | (data["prim_id"] >= num_primitives)):
        raise ValueError("S1 prim_id references an absent dictionary item")
    if np.any(data["tau"] <= 0) or np.any((data["phase_s"] < 0) | (data["phase_s"] > 1)):
        raise ValueError("S1 tau/phase_s values are invalid")
    return data


def join_s1_targets(rows, frames, *, allowed_splits=("train",)):
    """Join action rows to S1 frame targets by exact episode/frame identity.

    Human task/subtask annotations are intentionally not read. Split-restricted
    joins require an explicit split column and return only allowed
    self-discovery splits; callers should pass ``("train",)`` during S3 fitting.
    """
    row_episode = np.asarray(rows["episode_id"]).astype(str)
    row_t = np.asarray(rows["t"], np.int64)
    frame_episode = np.asarray(frames["episode_id"]).astype(str)
    frame_t = np.asarray(frames["t"], np.int64)
    allowed = set(map(str, allowed_splits))
    if "split" not in frames:
        raise ValueError("S1 frames need an explicit split column for split-restricted training")
    keep = np.isin(np.asarray(frames["split"]).astype(str), list(allowed))
    # S1 should provide one label per (episode,t); fail on ambiguity rather
    # than silently choosing a route target.
    index = {}
    for i in np.flatnonzero(keep):
        key = (frame_episode[i], int(frame_t[i]))
        if key in index:
            raise ValueError(f"duplicate S1 frame target for {key}")
        index[key] = int(i)
    row_indices, frame_indices = [], []
    for i, key in enumerate(zip(row_episode, row_t, strict=True)):
        j = index.get((key[0], int(key[1])))
        if j is not None:
            row_indices.append(i)
            frame_indices.append(j)
    return np.asarray(row_indices, np.int64), np.asarray(frame_indices, np.int64)


def make_s3_train_step(optimizer, *, use_primitives: bool, microbatch_size: int = 16):
    """Build a filtered NNX update for matched G0 or primitive-routed G1.

    The caller initializes the optimizer from the selected model filter and
    supplies fixed-order batches. This helper intentionally has no task or
    subtask labels in its signature.
    """
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import optax

    from openpi.models.pi0_dmp_route import dmp_g0_trainable_filter, dmp_trainable_filter

    trainable = dmp_trainable_filter() if use_primitives else dmp_g0_trainable_filter()

    if microbatch_size <= 0:
        raise ValueError("microbatch_size must be positive")

    @nnx.jit
    def microbatch_grad(rng, model, observation, actions, draft, primitive_shapes, targets):
        def loss_fn(current_model, step_rng):
            if use_primitives:
                loss, metrics = current_model.compute_s3_loss(
                    step_rng,
                    observation,
                    actions,
                    dmp_draft=draft,
                    primitive_shapes=primitive_shapes,
                    state_history=targets["state_history"],
                    target_prim_id=targets["prim_id"],
                    target_goal=targets["g"],
                    target_tau=targets["tau"],
                    target_phase_s=targets["phase_s"],
                    use_primitives=True,
                    train=True,
                )
            else:
                loss, metrics = current_model.compute_s3_loss(
                    step_rng, observation, actions, use_primitives=False, train=True
                )
            return jnp.mean(loss), metrics

        diff = nnx.DiffState(0, trainable)
        (loss, metrics), grads = nnx.value_and_grad(loss_fn, argnums=diff, has_aux=True)(
            model, rng
        )
        return loss, metrics, grads

    @nnx.jit
    def apply_update(model, opt_state, grads):
        params = nnx.state(model, trainable)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        nnx.update(model, optax.apply_updates(params, updates))
        return opt_state

    def step(rng, model, opt_state, observation, actions, draft, primitive_shapes, targets):
        """Accumulate gradients over microbatches, then perform one batch-32 update.

        This keeps the logical batch, RNG row ordering, and optimizer update count
        unchanged while bounding activation memory for the shared GPU.
        """
        batch = int(actions.shape[0])
        if batch % microbatch_size:
            raise ValueError(f"batch {batch} must be divisible by microbatch_size {microbatch_size}")
        total_loss = None
        total_metrics = None
        grad_sum = None
        n = batch // microbatch_size
        for i in range(n):
            lo, hi = i * microbatch_size, (i + 1) * microbatch_size
            obs_i = jax.tree.map(lambda x: x[lo:hi], observation)
            actions_i = actions[lo:hi]
            draft_i = draft[lo:hi]
            targets_i = jax.tree.map(lambda x: x[lo:hi], targets)
            loss_i, metrics_i, grads_i = microbatch_grad(
                jax.random.fold_in(rng, i), model, obs_i, actions_i, draft_i,
                primitive_shapes, targets_i,
            )
            if grad_sum is None:
                grad_sum = jax.tree.map(lambda x: x / n, grads_i)
                total_loss = loss_i / n
                total_metrics = jax.tree.map(lambda x: x / n, metrics_i)
            else:
                grad_sum = jax.tree.map(lambda a, b: a + b / n, grad_sum, grads_i)
                total_loss = total_loss + loss_i / n
                total_metrics = jax.tree.map(lambda a, b: a + b / n, total_metrics, metrics_i)
        opt_state = apply_update(model, opt_state, grad_sum)
        return total_loss, total_metrics, opt_state

    return step


def low_memory_adamw(learning_rate, *, b1=0.9, b2=0.95, eps=1e-8,
                    weight_decay=1e-10, moment_dtype="bfloat16"):
    """AdamW with both moments stored compactly for the shared-GPU run.

    Adam's coefficients, decoupled weight decay, clipping (applied by caller),
    and update schedule stay unchanged; only accumulator storage precision is
    reduced. Parameters and gradients remain float32.
    """
    import jax
    import jax.numpy as jnp
    import optax
    from typing import NamedTuple

    dtype = jnp.dtype(moment_dtype)

    class _State(NamedTuple):
        count: jax.Array
        mu: object
        nu: object

    def init_fn(params):
        zeros = lambda p: jnp.zeros_like(p, dtype=dtype)
        return _State(jnp.zeros([], jnp.int32), jax.tree.map(zeros, params),
                      jax.tree.map(zeros, params))

    def update_fn(grads, state, params=None):
        if params is None:
            raise ValueError("AdamW requires params for decoupled weight decay")
        count = state.count + jnp.asarray(1, jnp.int32)
        mu = jax.tree.map(lambda m, g: (b1 * m.astype(jnp.float32) + (1. - b1) * g).astype(dtype),
                          state.mu, grads)
        nu = jax.tree.map(lambda v, g: (b2 * v.astype(jnp.float32) + (1. - b2) * jnp.square(g)).astype(dtype),
                          state.nu, grads)
        b1_corr = 1. - jnp.power(jnp.asarray(b1, jnp.float32), count)
        b2_corr = 1. - jnp.power(jnp.asarray(b2, jnp.float32), count)
        lr = jnp.asarray(learning_rate(state.count), jnp.float32)
        updates = jax.tree.map(
            lambda m, v, p: -lr * ((m.astype(jnp.float32) / b1_corr) /
                                   (jnp.sqrt(v.astype(jnp.float32) / b2_corr) + eps) +
                                   weight_decay * p.astype(jnp.float32)),
            mu, nu, params,
        )
        return updates, _State(count, mu, nu)

    return optax.GradientTransformation(init_fn, update_fn)


def state_history_windows(states, episode_ids, times, *, history_len=5):
    """Build causal, same-episode state windows ending at each row.

    Rows are sorted by caller order only within the explicit `(episode,t)` key;
    missing earlier frames are left-padded by the earliest available state.
    """
    states = np.asarray(states, np.float32)
    episode_ids = np.asarray(episode_ids).astype(str)
    times = np.asarray(times, np.int64)
    if states.ndim != 2 or len(states) != len(episode_ids) or len(states) != len(times):
        raise ValueError("states, episode_ids and times must have equal row counts")
    if history_len < 1:
        raise ValueError("history_len must be positive")
    by_episode = {}
    for i, (episode, t) in enumerate(zip(episode_ids, times, strict=True)):
        by_episode.setdefault(episode, []).append((int(t), i))
    result = np.empty((len(states), history_len, states.shape[-1]), np.float32)
    for entries in by_episode.values():
        entries.sort()
        history = []
        for _, i in entries:
            history.append(states[i])
            window = history[-history_len:]
            if len(window) < history_len:
                window = [window[0]] * (history_len - len(window)) + window
            result[i] = np.stack(window)
    return result


def initialize_from_skillnet90(model, checkpoint: str | Path):
    """Restore the public SkillNet-90 weights while retaining new S3 params."""
    import flax.nnx as nnx
    from openpi.training.weight_loaders import PartialCheckpointWeightLoader

    state = nnx.state(model)
    # S3-only router/condition leaves do not exist in the released checkpoint;
    # preserve their seeded initialization and load every compatible base leaf.
    loaded = PartialCheckpointWeightLoader(str(checkpoint), missing_regex=".*").load(state.to_pure_dict())
    state.replace_by_pure_dict(loaded)
    nnx.update(model, state)
    return model
