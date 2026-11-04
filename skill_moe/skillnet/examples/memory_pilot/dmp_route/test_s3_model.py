"""CPU-only synthetic checks for the DMP-routed S3 path."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model
from openpi.models.pi0_config_dmp_route import Pi0DmpRouteConfig
from openpi.models.pi0_dmp_route import (
    DmpConditionEncoder,
    DmpShapeEncoder,
    DmpRouteState,
    DmpRoutePrediction,
    DmpRouterHead,
    bridge_source,
    dmp_trainable_filter,
    shuffle_route_primitives,
    score_shape_dictionary,
    decode_route,
)

HERE = Path(__file__).resolve().parent
S1_DIR = Path("/work/.skillnet-state/outputs/memory-pilot/dmp-route-20261102/s1")
sys.path.insert(0, str(HERE))


def _load_dmp():
    spec = importlib.util.spec_from_file_location("s3_test_dmp", S1_DIR / "dmp.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _synthetic_library(k=2, n=16):
    phase = np.linspace(0, 1, n, dtype=np.float32)
    shapes = np.zeros((k, 6, n), np.float32)
    shapes[:, 0] = phase
    shapes[:, 3] = phase
    if k > 1:
        shapes[1, 1] = 0.35 * np.sin(np.pi * phase)
        shapes[1, 4] = 0.15 * np.sin(np.pi * phase)
    return {"shapes": shapes, "grip": np.asarray([-1, 1][:k], np.int8), "alpha": np.asarray([4.0])}


def test_bridge_layout_and_values():
    draft = jnp.arange(2 * 10 * 7, dtype=jnp.float32).reshape((2, 10, 7)) / 100
    noise = jnp.ones((2, 10, 32), jnp.float32)
    x = bridge_source(draft, noise, t0=0.4)
    assert x.shape == (2, 10, 32)
    np.testing.assert_allclose(x[..., :7], 0.6 * np.asarray(draft) + 0.4, atol=1e-6)
    np.testing.assert_allclose(x[..., 7:14], draft, atol=0)
    np.testing.assert_allclose(x[..., 14:], 0, atol=0)


def test_s1_rollout_and_action_adapter_are_single_primitive():
    from bridge import build_dmp_draft, straight_minimum_jerk_shape

    dmp = _load_dmp()
    library = _synthetic_library()
    dmp.configure_library(library)
    route0 = DmpRouteState(jnp.asarray([0]), jnp.zeros((1, 4)), jnp.asarray([[0.1, 0, 0, 0.2, 0, 0]]),
                           jnp.asarray([0.5]), jnp.asarray([0.2]))
    route1 = route0._replace(prim_id=jnp.asarray([1]), shape_embedding=jnp.ones((1, 4)))
    q01, q99 = np.full(7, -1, np.float32), np.full(7, 1, np.float32)
    a0 = build_dmp_draft(dmp, library, route0, np.zeros((1, 6), np.float32), q01=q01, q99=q99)
    a1 = build_dmp_draft(dmp, library, route1, np.zeros((1, 6), np.float32), q01=q01, q99=q99)
    straight = np.broadcast_to(straight_minimum_jerk_shape(16)[None], (1, 6, 16))
    straight_action = build_dmp_draft(
        dmp, library, route0, np.zeros((1, 6), np.float32), q01=q01, q99=q99,
        shape_override=straight,
    )
    assert a0.shape == a1.shape == (1, 10, 7)
    assert np.isfinite(a0).all() and np.isfinite(a1).all()
    # Different trajectory shape and grip are retained as one hard-selected draft.
    assert not np.allclose(a0, a1)
    assert not np.allclose(a0, straight_action)
    # A straight ablation must not invent rotation when the predicted rotation
    # goal is zero; S1's loop-shape fallback otherwise expands a canonical +x
    # rotational endpoint into a nonzero 0.1 rad rotation.
    no_rotation = route0._replace(goal=jnp.asarray([[.1, 0, 0, 0, 0, 0]], jnp.float32))
    straight_no_rotation = build_dmp_draft(
        dmp, library, no_rotation, np.zeros((1, 6), np.float32), q01=q01, q99=q99,
        shape_override=straight,
    )
    np.testing.assert_allclose(straight_no_rotation[..., 3:6], 0.0, atol=2e-6)


def test_pose_adapter_uses_osc_world_rotation_delta():
    from bridge import dmp_pose_to_normalized_actions
    from scipy.spatial.transform import Rotation

    y0 = np.asarray([[0, 0, 0, np.pi / 2, 0, 0]], np.float32)
    r0 = Rotation.from_rotvec(y0[0, 3:])
    delta = Rotation.from_rotvec([0, 0.1, 0])
    poses = []
    for _ in range(10):
        r0 = delta * r0  # robosuite OSC_POSE left-multiplies the world delta
        poses.append([0, 0, 0, *r0.as_rotvec()])
    actions = dmp_pose_to_normalized_actions(
        np.asarray([poses], np.float32), y0, [-1],
        q01=np.full(7, -1, np.float32), q99=np.full(7, 1, np.float32),
    )
    raw = (actions + 1.0) * 0.5 * (2.0 + 1e-6) - 1.0
    np.testing.assert_allclose(raw[0, :, 3:6], np.broadcast_to([0, 0.2, 0], (10, 3)), atol=2e-5)


def test_straight_shape_and_state_history_are_causal():
    from bridge import straight_minimum_jerk_shape
    from train_utils import state_history_windows

    shape = straight_minimum_jerk_shape(16)
    np.testing.assert_allclose(shape[:, 0], 0, atol=1e-7)
    np.testing.assert_allclose(shape[:, -1], [1, 0, 0, 1, 0, 0], atol=1e-7)
    assert np.all(np.diff(shape[0]) >= -1e-7)
    states = np.arange(5, dtype=np.float32)[:, None]
    windows = state_history_windows(states, ["a", "a", "a", "b", "a"], [0, 1, 2, 0, 4], history_len=3)
    np.testing.assert_array_equal(windows[2, :, 0], [0, 1, 2])
    # Episode b cannot leak into episode a; sorting uses episode-local time.
    np.testing.assert_array_equal(windows[4, :, 0], [1, 2, 4])


def test_content_pointer_is_variable_k_and_has_finite_gradients():
    router = DmpRouterHead(prefix_width=12, state_width=8, state_history_len=3,
                           hidden_dim=16, shape_width=7, rngs=nnx.Rngs(3))
    encoder = DmpShapeEncoder(basis_n=16, hidden_dim=16, output_dim=7, rngs=nnx.Rngs(30))
    hidden = jax.random.normal(jax.random.key(4), (2, 7, 12))
    mask = jnp.asarray([[1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1]], bool)
    history = jax.random.normal(jax.random.key(40), (2, 3, 8))
    query, goal, log_tau, phase = router(hidden, mask, history)
    assert query.shape == (2, 7) and goal.shape == (2, 6)
    assert log_tau.shape == phase.shape == (2,)
    shapes = jax.random.normal(jax.random.key(41), (3, 6, 16))
    emb = encoder(shapes)
    assert score_shape_dictionary(query, emb).shape == (2, 3)
    prediction = DmpRoutePrediction(score_shape_dictionary(query, emb), goal, log_tau, phase, query, emb)
    decoded = decode_route(prediction)
    np.testing.assert_array_equal(np.asarray(decoded.prim_id), np.asarray(jnp.argmax(prediction.prim_logits, axis=-1)))
    np.testing.assert_allclose(np.asarray(decoded.shape_embedding), np.asarray(emb)[np.asarray(decoded.prim_id)])
    # Changing K changes only the supplied dictionary rows, not the learned
    # query or encoder parameter shapes.
    params_before = jax.tree.map(lambda x: x.shape, nnx.state(encoder).to_pure_dict())
    assert score_shape_dictionary(query, encoder(shapes[:2])).shape == (2, 2)
    assert jax.tree.map(lambda x: x.shape, nnx.state(encoder).to_pure_dict()) == params_before
    grads = nnx.grad(lambda module: jnp.sum(module(hidden, mask, history)[0] ** 2) +
                     jnp.sum(module(hidden, mask, history)[1] ** 2))(router)
    leaves = jax.tree.leaves(grads.to_pure_dict())
    assert leaves and all(np.isfinite(np.asarray(x)).all() for x in leaves)
    route = DmpRouteState(jnp.asarray([0, 1, 2]), emb, jnp.zeros((3, 6)), jnp.ones(3), jnp.zeros(3))
    shuffled = shuffle_route_primitives(route, jax.random.key(5), emb)
    assert shuffled.prim_id.shape == route.prim_id.shape
    assert sorted(np.asarray(shuffled.prim_id).tolist()) == [0, 1, 2]
    np.testing.assert_allclose(np.asarray(shuffled.shape_embedding), np.asarray(emb)[np.asarray(shuffled.prim_id)])


def test_log_tau_supervision_decodes_to_the_same_duration():
    duration = 0.5
    emb = jnp.ones((1, 4), jnp.float32)
    prediction = DmpRoutePrediction(
        jnp.zeros((1, 1), jnp.float32), jnp.zeros((1, 6), jnp.float32),
        jnp.asarray([np.log(duration)], jnp.float32), jnp.zeros((1,), jnp.float32),
        jnp.zeros((1, 4), jnp.float32), emb,
    )
    decoded = decode_route(prediction)
    np.testing.assert_allclose(np.asarray(decoded.tau), [duration], rtol=1e-6, atol=1e-7)


def test_s1_training_join_requires_and_applies_split_labels(tmp_path):
    from train_utils import join_s1_targets, validate_s1_frames

    rows = {"episode_id": np.asarray(["a", "a"]), "t": np.asarray([0, 1])}
    frames = {
        "episode_id": np.asarray(["a", "a"]), "t": np.asarray([0, 1]),
        "split": np.asarray(["train", "development"]),
    }
    row_ix, frame_ix = join_s1_targets(rows, frames, allowed_splits=("train",))
    np.testing.assert_array_equal(row_ix, [0])
    np.testing.assert_array_equal(frame_ix, [0])
    with np.testing.assert_raises_regex(ValueError, "explicit split column"):
        join_s1_targets(rows, {k: v for k, v in frames.items() if k != "split"}, allowed_splits=("train",))

    serialized = {
        "episode_id": frames["episode_id"], "task_id": np.asarray([0, 0]),
        "t": frames["t"], "prim_id": np.asarray([0, 0]),
        "seg_start": np.asarray([0, 0]), "seg_end": np.asarray([1, 1]),
        "g": np.zeros((2, 6), np.float32), "tau": np.ones(2, np.float32),
        "phase_s": np.zeros(2, np.float32), "split": frames["split"],
    }
    valid_path = tmp_path / "frames-with-split.npz"
    missing_path = tmp_path / "frames-without-split.npz"
    np.savez(valid_path, **serialized)
    validated = validate_s1_frames(valid_path, num_primitives=1)
    assert "split" in validated
    np.savez(missing_path, **{k: v for k, v in serialized.items() if k != "split"})
    with np.testing.assert_raises_regex(ValueError, "split"):
        validate_s1_frames(missing_path, num_primitives=1)


def _tiny_observation(batch=1):
    images = {name: jnp.zeros((batch, 224, 224, 3), jnp.float32) for name in
              ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")}
    image_masks = {name: jnp.asarray([name != "right_wrist_0_rgb"] * batch, bool) for name in images}
    tokens = jnp.zeros((batch, 12), jnp.int32).at[:, :4].set(jnp.asarray([2, 32, 33, 1]))
    token_mask = jnp.arange(12)[None, :] < 4
    return _model.Observation(
        images=images,
        image_masks=image_masks,
        state=jnp.zeros((batch, 32), jnp.float32),
        tokenized_prompt=tokens,
        tokenized_prompt_mask=jnp.broadcast_to(token_mask, tokens.shape),
    )


def test_primitive_swap_changes_action_output():
    config = Pi0DmpRouteConfig(
        dtype="float32", paligemma_variant="dummy_moe_4", action_expert_variant="dummy_moe_4",
        vision_variant="mu/32", pi05=True, action_horizon=10, action_dim=32,
        max_token_len=12, skill_num=1, skill_embed_dim=16, enable_primitive_router=True,
        shape_basis_n=16, state_history_len=5,
        router_hidden_dim=32, bridge_t0=0.4,
    )
    model = config.create(jax.random.key(11))
    trainable_names = [str(path) for path in nnx.state(model, dmp_trainable_filter()).flat_state()]
    assert any("dmp_router" in path for path in trainable_names)
    assert any("dmp_condition" in path for path in trainable_names)
    assert any("_1" in path for path in trainable_names)
    assert not any("PaliGemma/llm/layers" in path and "_1" not in path for path in trainable_names)
    # The public adaRMS gate projections are zero-initialized. Give this
    # synthetic untrained model a small nonzero gate, as a fine-tuned checkpoint
    # has, so the test checks end-to-end condition influence rather than the
    # intentional initialization-time residual shutoff.
    state = nnx.state(model)
    pure = state.to_pure_dict()
    for norm_name in ("pre_attention_norm_1", "pre_ffw_norm_1"):
        kernel = pure["PaliGemma"]["llm"]["layers"][norm_name]["Dense_0"]["kernel"]
        width = kernel.shape[-1] // 3
        pure["PaliGemma"]["llm"]["layers"][norm_name]["Dense_0"]["kernel"] = kernel.at[..., 2 * width :].set(0.02)
    state.replace_by_pure_dict(pure)
    nnx.update(model, state)
    obs = _tiny_observation()
    primitive_shapes = jnp.asarray(_synthetic_library()["shapes"])
    embeddings = model.encode_primitive_shapes(primitive_shapes)
    route0 = DmpRouteState(jnp.asarray([0]), embeddings[:1], jnp.asarray([[.1, 0, 0, 0, 0, 0]]),
                           jnp.asarray([.5]), jnp.asarray([.2]))
    route1 = route0._replace(prim_id=jnp.asarray([1]), shape_embedding=embeddings[1:2])
    router_token0 = model.dmp_condition.router_tokens(route0)
    router_token1 = model.dmp_condition.router_tokens(route1)
    ada0 = model.dmp_condition.adarms_delta(route0)
    ada1 = model.dmp_condition.adarms_delta(route1)
    assert np.max(np.abs(np.asarray(router_token0 - router_token1))) > 1e-7
    assert np.max(np.abs(np.asarray(ada0 - ada1))) > 1e-7
    draft = jnp.zeros((1, 10, 7), jnp.float32).at[..., 0].set(.1)
    noise = jnp.zeros((1, 10, 32), jnp.float32)
    source = bridge_source(draft, noise, t0=0.4)
    pred0, bal0 = model._bridge_prediction(obs, source, jnp.asarray([0.4]), route0, injection_mode="both")
    pred1, bal1 = model._bridge_prediction(obs, source, jnp.asarray([0.4]), route1, injection_mode="both")
    pred_diff = float(jnp.max(jnp.abs(pred0 - pred1)))
    balance_diff = float(jnp.max(jnp.abs(bal0 - bal1)))
    assert pred_diff > 1e-7, (pred_diff, balance_diff)
    y0 = model.sample_actions_s3(jax.random.key(1), obs, draft=draft, noise=noise,
                                 route=route0, num_steps=2)
    y1 = model.sample_actions_s3(jax.random.key(1), obs, draft=draft, noise=noise,
                                 route=route1, num_steps=2)
    y0, y1 = jax.device_get((y0, y1))
    assert y0.shape == y1.shape == (1, 10, 32)
    assert np.isfinite(y0).all() and np.isfinite(y1).all()
    assert np.max(np.abs(np.asarray(y0) - np.asarray(y1))) > 1e-7
    np.testing.assert_array_equal(np.asarray(y0)[..., 7:], 0)
    loss, metrics = model.compute_s3_loss(
        jax.random.key(22), obs, jnp.zeros((1, 10, 32), jnp.float32),
        dmp_draft=draft,
        primitive_shapes=primitive_shapes,
        state_history=jnp.zeros((1, 5, 32), jnp.float32),
        target_prim_id=route0.prim_id,
        target_goal=route0.goal,
        target_tau=route0.tau,
        target_phase_s=route0.phase_s,
        use_primitives=True,
        train=False,
    )
    loss, metrics = jax.device_get((loss, metrics))
    assert np.isfinite(np.asarray(loss)).all()
    assert np.isfinite(np.asarray(metrics["flow"])).all()
    assert np.isfinite(np.asarray(metrics["router_ce"])).all()
    diff = nnx.DiffState(0, dmp_trainable_filter())

    def trainable_loss(current):
        value, _ = current.compute_s3_loss(
            jax.random.key(23), obs, jnp.zeros((1, 10, 32), jnp.float32),
            dmp_draft=draft,
            primitive_shapes=primitive_shapes,
            state_history=jnp.zeros((1, 5, 32), jnp.float32),
            target_prim_id=route0.prim_id,
            target_goal=route0.goal,
            target_tau=route0.tau,
            target_phase_s=route0.phase_s,
            use_primitives=True,
            train=False,
        )
        return jnp.mean(value)

    grad_loss, grads = nnx.value_and_grad(trainable_loss, argnums=diff)(model)
    grad_values = jax.tree.leaves(grads.to_pure_dict())
    assert np.isfinite(np.asarray(grad_loss)).all()
    assert grad_values and all(np.isfinite(np.asarray(g)).all() for g in grad_values)
    assert any(np.any(np.asarray(g) != 0) for g in grad_values)


def test_matched_g0_uses_zero_draft_bridge_without_router():
    config = Pi0DmpRouteConfig(
        dtype="float32", paligemma_variant="dummy_moe_4", action_expert_variant="dummy_moe_4",
        vision_variant="mu/32", pi05=True, action_horizon=10, action_dim=32,
        max_token_len=12, skill_num=1, skill_embed_dim=16, enable_primitive_router=False,
        shape_basis_n=16, state_history_len=5,
        router_hidden_dim=32, bridge_t0=0.4, injection_mode="none",
    )
    model = config.create(jax.random.key(31))
    obs = _tiny_observation()
    noise = jax.random.normal(jax.random.key(32), (1, 10, 32))
    output = model.sample_actions_s3(
        jax.random.key(33), obs, draft=None, noise=noise, route=None,
        t0=0.4, num_steps=2, injection_mode="none",
    )
    output = jax.device_get(output)
    assert output.shape == (1, 10, 32)
    assert np.isfinite(output).all()
    np.testing.assert_array_equal(np.asarray(output)[..., 7:], 0)
    loss, metrics = model.compute_s3_loss(
        jax.random.key(34), obs, jnp.zeros((1, 10, 32), jnp.float32),
        use_primitives=False, train=False,
    )
    loss, metrics = jax.device_get((loss, metrics))
    assert np.isfinite(np.asarray(loss)).all()
    np.testing.assert_allclose(metrics["bridge_t0"], 0.4, atol=1e-7)
