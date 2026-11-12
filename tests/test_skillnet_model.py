"""SkillNet model pieces in openpi: skill conditioning, anchor flow, subtask tokens, serving."""
from __future__ import annotations

import asyncio
import threading

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import websockets.asyncio.server
from openpi.models.gemma_moe_skill import MoEFeedForward, _selected_expert
from openpi.models.pi0_anchor_flow import (
    Pi0AnchorFlow,
    anchor_flow_velocity_target,
    integrate_velocity,
    mix_anchor_with_noise,
    sample_flow_times,
)
from openpi.models.pi0_moe_skill import Pi0 as SkillNetPi0
from openpi.models.pi0_moe_skill import (
    _apply_skill_embed_override,
    subtask_generated_lengths,
)
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.training.config_moe_skill import subtask_task_indices_for_split
from openpi.transforms import (
    AttachLiberoSubtask,
    TokenizeSubtaskInference,
    TokenizeSubtaskTraining,
)
from skillnet_client.websocket_client_policy import WebsocketClientPolicy

# ── Skill-embedding override and expert selection ─────────────────────────────────────────────────

def test_disabled_skill_embedding_hook_preserves_original_array_exactly():
    original = np.arange(3 * 4 * 64, dtype=np.float32).reshape(3, 4, 64)
    result = _apply_skill_embed_override(original, None)
    assert result is original
    np.testing.assert_array_equal(result, original)


def test_skill_embedding_hook_uses_explicit_override():
    original = np.zeros((1, 4, 64), dtype=np.float32)
    replacement = np.ones_like(original)
    assert _apply_skill_embed_override(original, replacement) is replacement


def test_captured_top1_matches_dispatched_expert_when_router_logits_tie():
    # An all-masked skill set gives the bias-free router equal probabilities.
    # Actual dispatch uses argsort(...)[..., -1], not argmax.
    probs = jnp.full((1, 2, 4), 0.25)
    dispatched = jnp.argsort(probs, axis=-1)[..., -1:]
    captured = np.asarray(_selected_expert(dispatched))
    np.testing.assert_array_equal(captured, np.asarray(dispatched[..., 0]))
    assert np.all(captured == 3)
    assert not np.any(captured == np.asarray(jnp.argmax(probs, axis=-1)))


def test_moe_valid_skill_path_matches_legacy_precision_and_norms_match_dispatch(monkeypatch):
    from openpi.models import gemma_moe_skill as gemma

    x = jax.random.normal(jax.random.key(4), (2, 64, 16), dtype=jnp.bfloat16)
    skills = jax.random.normal(jax.random.key(5), (2, 4, 8), dtype=jnp.bfloat16)
    mask = jnp.array([[True, True, True, False], [True, True, True, True]])
    module = gemma.MoEFeedForward(num_experts=4, hidden_dim=32, output_dim=16)
    variables = module.init(jax.random.key(6), x, skills, mask)
    output, captured = module.apply(variables, x, skills, mask, capture_intermediates=True, mutable=["intermediates"])
    y, _, selected, selected_norm, shared_norm = output
    intermediates = captured["intermediates"]
    experts = np.stack([np.asarray(intermediates[f"expert_{i}"]["__call__"][0], dtype=np.float32)
                        for i in range(4)], axis=-2)
    shared = np.asarray(intermediates["shared_expert"]["__call__"][0], dtype=np.float32)
    # Output minus shared is exactly the dispatched (router-weighted) expert.
    # Addition rounds in bfloat16, so compare norms within that rounding error.
    np.testing.assert_allclose(np.asarray(shared_norm), np.linalg.norm(shared, axis=-1), rtol=1e-6)
    contribution = np.asarray(y, dtype=np.float32) - shared
    np.testing.assert_allclose(np.asarray(selected_norm), np.linalg.norm(contribution, axis=-1), rtol=.05, atol=.02)
    chosen = np.take_along_axis(experts, np.asarray(selected)[..., None, None], axis=-2)[..., 0, :]
    # The contribution must be collinear with the selected expert (positive router gate).
    assert np.all(np.sum(chosen * contribution, axis=-1) >= 0)

    def legacy_attention(scores, valid):
        return nn.softmax(jnp.where(valid[:, None, None, :], scores, -1e9), axis=-1).astype(x.dtype)

    monkeypatch.setattr(gemma, "_masked_skill_softmax", legacy_attention)
    legacy = module.apply(variables, x, skills, mask)
    for actual, expected in zip(output, legacy):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_transformer_capture_preserves_numerical_output():
    from openpi.models import gemma_moe_skill as gemma

    configs = [gemma.Config(width=8, depth=2, mlp_dim=16, num_heads=2, num_kv_heads=1, head_dim=4),
               gemma.Config(width=8, depth=2, mlp_dim=16, num_heads=2, num_kv_heads=1, head_dim=4,
                            moe=True, expert_num=4)]
    module = gemma.Module(configs=configs, embed_dtype="bfloat16", skill_embed_dim=8)
    embedded = [jnp.ones((1, 2, 8)), jnp.ones((1, 3, 8))]
    kwargs = {"skill_embeds": jnp.arange(32, dtype=jnp.float32).reshape(1, 4, 8),
              "skill_mask": jnp.array([[True, True, False, False]]),
              "positions": jnp.arange(5)[None, :], "mask": jnp.ones((1, 5, 5), dtype=bool)}
    # This model defines an init convenience method for NNX; initialize the
    # __call__ path through Linen's base implementation in this isolated test.
    variables = nn.Module.init(module, jax.random.key(0), embedded, **kwargs)
    plain = module.apply(variables, embedded, **kwargs)
    captured = module.apply(variables, embedded, capture_auxiliary=True, **kwargs)
    for a, b in zip(jax.tree.leaves(plain), jax.tree.leaves(captured[:3])):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    features, routes, selected_norm, shared_norm = captured[3]
    assert features.shape == (2, 1, 3, 8)
    assert routes.shape == selected_norm.shape == shared_norm.shape == (2, 1, 3)


# ── Skill conditioning reaches only the MoE router — CPU-sized intervention on the actual skill cross-attention module. ───

def test_skill_order_without_positions_is_invariant_but_content_can_matter():
    rng = np.random.default_rng(20261102)
    x = jnp.asarray(rng.normal(size=(1, 7, 8)), jnp.float32)
    skills = jnp.asarray(rng.normal(size=(1, 5, 8)), jnp.float32)
    mask = jnp.asarray([[True, True, True, False, False]])
    module = MoEFeedForward(num_experts=4, hidden_dim=16, output_dim=8)
    weights = module.init(jax.random.key(7), x, skills, mask)
    original = module.apply(weights, x, skills, mask)
    permutation = jnp.asarray([2, 0, 4, 1, 3])
    reordered = module.apply(weights, x, skills[:, permutation], mask[:, permutation])
    changed = module.apply(weights, x, skills.at[:, 0].set(-3 * skills[:, 0]), mask)
    np.testing.assert_allclose(original[0], reordered[0], rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(original[2], reordered[2])
    assert float(jnp.max(jnp.abs(original[0] - changed[0]))) > 1e-5


# ── Anchor-flow loss and integration ──────────────────────────────────────────────────────────────

def test_ideal_anchor_velocity_euler_reaches_action_for_any_anchor_and_steps():
    rng = np.random.default_rng(2026)
    batch, horizon, action_dim = 4, 10, 32
    actions = jnp.asarray(rng.normal(size=(batch, horizon, action_dim)), jnp.float32)
    anchors = jnp.asarray(rng.normal(size=(batch, horizon, action_dim)), jnp.float32)
    noise = jnp.asarray(rng.normal(size=(batch, horizon, action_dim)), jnp.float32)
    t0 = jnp.asarray([0.13, 0.37, 0.71, 1.0], jnp.float32)
    x_start = mix_anchor_with_noise(anchors, noise, t0)

    for steps in (1, 2, 3, 7, 10, 31):
        def ideal_velocity(x_t, time):
            return (x_t - actions) / time[:, None, None]

        result = integrate_velocity(ideal_velocity, x_start, t0, steps)
        np.testing.assert_allclose(np.asarray(result), np.asarray(actions), rtol=2e-5, atol=2e-5)


def test_ordinary_rows_match_previous_velocity_and_loss_exactly():
    rng = np.random.default_rng(3)
    actions = jnp.asarray(rng.normal(size=(3, 10, 32)), jnp.float32)
    noise = jnp.asarray(rng.normal(size=(3, 10, 32)), jnp.float32)
    time = jnp.asarray([0.02, 0.5, 0.999], jnp.float32)
    target = anchor_flow_velocity_target(noise, actions, actions, time)
    old_target = noise - actions
    np.testing.assert_array_equal(np.asarray(target), np.asarray(old_target))

    # Reconstruct the exact old and new MSE reductions used by the training loss.
    predicted = jnp.asarray(rng.normal(size=(3, 10, 32)), jnp.float32)
    old_loss = jnp.mean(jnp.square(predicted - old_target), axis=-1)
    new_loss = jnp.mean(jnp.square(predicted - target), axis=-1)
    np.testing.assert_array_equal(np.asarray(new_loss), np.asarray(old_loss))


def test_anchor_target_is_xt_minus_action_over_time():
    rng = np.random.default_rng(11)
    actions = jnp.asarray(rng.normal(size=(2, 10, 32)), jnp.float32)
    anchors = jnp.asarray(rng.normal(size=(2, 10, 32)), jnp.float32)
    noise = jnp.asarray(rng.normal(size=(2, 10, 32)), jnp.float32)
    time = jnp.asarray([0.23, 0.82], jnp.float32)
    x_t = (1 - time[:, None, None]) * anchors + time[:, None, None] * noise
    target = anchor_flow_velocity_target(noise, actions, anchors, time)
    expected = (x_t - actions) / time[:, None, None]
    np.testing.assert_allclose(np.asarray(target), np.asarray(expected), rtol=2e-6, atol=2e-6)


def test_only_anchor_rows_get_t_minimum_and_ordinary_beta_schedule_is_unchanged():
    rng = jax.random.key(9)
    mask = jnp.asarray([True, False, True, False])
    times = sample_flow_times(rng, (4,), mask, anchor_t_min=0.05)
    beta = jax.random.beta(rng, 1.5, 1, (4,))
    old_ordinary = beta * 0.999 + 0.001
    expected_anchors = 0.05 + beta * 0.95
    values = np.asarray(times)
    np.testing.assert_array_equal(values[~np.asarray(mask)], np.asarray(old_ordinary)[~np.asarray(mask)])
    np.testing.assert_array_equal(values[np.asarray(mask)], np.asarray(expected_anchors)[np.asarray(mask)])
    assert np.all(values[np.asarray(mask)] >= 0.05)
    assert np.all(values[np.asarray(mask)] <= 1.0)


def test_direct_anchor_path_at_t0_one_delegates_to_skillnet_sampler(monkeypatch):
    calls = {}

    def parent_sampler(self, rng, observation, *, num_steps=10, noise=None, **kwargs):
        del self, rng, kwargs
        calls.update(observation=observation, num_steps=num_steps, noise=noise)
        return noise + jnp.asarray(0.125, noise.dtype)

    monkeypatch.setattr(SkillNetPi0, "sample_actions", parent_sampler)
    monkeypatch.setattr(
        Pi0AnchorFlow,
        "_integrate_from_anchor",
        lambda self, observation, *, x_src, t0, num_steps=10: x_src - 1,
    )
    model = object.__new__(Pi0AnchorFlow)
    observation = object()
    noise = jnp.arange(2 * 10 * 32, dtype=jnp.float32).reshape(2, 10, 32)
    for t0 in (1.0, jnp.asarray(1.0), jnp.ones((2,), jnp.float32)):
        result = Pi0AnchorFlow.sample_actions_from_anchor(
            model, observation, x_src=noise, t0=t0, num_steps=6
        )
        np.testing.assert_array_equal(np.asarray(result), np.asarray(noise + 0.125))
        assert calls["observation"] is observation
        assert calls["num_steps"] == 6
        np.testing.assert_array_equal(np.asarray(calls["noise"]), np.asarray(noise))


# ── Subtask tokens ────────────────────────────────────────────────────────────────────────────────

def test_libero_subtask_transform_preserves_rows_and_maps_no_segment():
    transform = AttachLiberoSubtask({"move the object": ["pick the cup", "place the cup"]})
    rows = [
        {"prompt": "move the object", "eval_skill_segment": i, "row_id": i}
        for i in (-1, 0, 1)
    ]
    output = [transform(row) for row in rows]
    assert len(output) == len(rows)
    assert [row["row_id"] for row in output] == [-1, 0, 1]
    assert [row["subtask"] for row in output] == ["none", "pick the cup", "place the cup"]


def test_subtask_prefix_and_training_target_format_and_eos():
    tokenizer = PaligemmaTokenizer(max_len=96)
    prompt = "pick the black bowl"
    inference = TokenizeSubtaskInference(tokenizer)({"prompt": prompt})
    prefix_len = int(inference["tokenized_prompt_mask"].sum())
    prefix = tokenizer._tokenizer.decode(inference["tokenized_prompt"][:prefix_len].tolist())
    assert prefix.endswith("Task: pick the black bowl. Subtask: ")

    target = "pick the bowl"
    train = TokenizeSubtaskTraining(tokenizer)({"prompt": prompt, "subtask": target})
    ids = train["tokenized_prompt"]
    mask = train["tokenized_prompt_mask"]
    loss_mask = train["token_loss_mask"]
    assert int(mask.sum()) == int(loss_mask.sum()) + prefix_len
    assert ids[int(mask.sum()) - 1] == 1
    decoded = tokenizer._tokenizer.decode(ids[prefix_len : int(mask.sum()) - 1].tolist())
    assert decoded.strip() == target


def test_decoder_eos_is_included_and_absent_eos_uses_max_length():
    tokens = jnp.asarray([[7, 8, 1, 9, 0], [4, 5, 6, 0, 0]], dtype=jnp.int32)
    np.testing.assert_array_equal(np.asarray(subtask_generated_lengths(tokens)), [3, 5])
    tokenizer = PaligemmaTokenizer(max_len=96)
    encoded = tokenizer._tokenizer.encode("pick the cup")
    assert tokenizer.detokenize(encoded + [1, 17, 0]) == "pick the cup"


def test_research_split_task_indices_exclude_libero10():
    split_a = subtask_task_indices_for_split("A")
    split_b = subtask_task_indices_for_split("B")
    assert split_a == tuple(range(20, 30)) + tuple(range(30, 40))
    assert split_b == tuple(range(20, 30)) + tuple(range(10, 20))
    assert not set(range(10)).intersection(split_a)
    assert not set(range(10)).intersection(split_b)


# ── Websocket policy server — Exercise request/response alignment with the real SkillNet websocket client. ───

class EchoPolicy:
    def infer(self, observation):
        return {"actions": np.array([[observation["request_id"]]], dtype=np.float32)}


def test_each_request_receives_its_own_action():
    loop = asyncio.new_event_loop()
    ready = threading.Event()
    state = {}

    async def start():
        policy_server = WebsocketPolicyServer(EchoPolicy())
        state["server"] = await websockets.asyncio.server.serve(
            policy_server._handler, "127.0.0.1", 0, ping_interval=None
        )
        state["port"] = state["server"].sockets[0].getsockname()[1]
        ready.set()

    def run():
        asyncio.set_event_loop(loop)
        loop.run_until_complete(start())
        loop.run_forever()
        loop.close()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    client = None
    try:
        assert ready.wait(10), "Websocket test server failed to start"
        client = WebsocketClientPolicy("127.0.0.1", state["port"])
        for request_id in range(8):
            response = client.infer({"request_id": request_id})
            assert response["actions"][0, 0] == request_id
    finally:
        if client is not None:
            client._ws.close()

        async def stop():
            state["server"].close()
            await state["server"].wait_closed()

        if ready.is_set():
            asyncio.run_coroutine_threadsafe(stop(), loop).result(timeout=10)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=10)
        assert not thread.is_alive()
