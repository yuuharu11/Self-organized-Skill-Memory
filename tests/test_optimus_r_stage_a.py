from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

EXAMPLES = Path(__file__).resolve().parents[1] / "skill_moe" / "skillnet" / "examples"
OPTIMUS_R = EXAMPLES / "memory_pilot" / "optimus_r"
for _path in (str(EXAMPLES), str(OPTIMUS_R)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from bank import build_initial_bank, retrieve_numpy
from data import official_task_ids
from openpi.models.pi0_optimus_r import Pi0OptimusR, query_contrastive_loss
from openpi.models.pi0_optimus_r_config import Pi0OptimusRConfig
from train_stage_a import mask_earlier_vlm_layers


def _tiny_model(batch=2):
    config = Pi0OptimusRConfig(
        pi05=True, action_dim=8, action_horizon=2, max_token_len=32,
        discrete_state_input=False, paligemma_variant="dummy",
        action_expert_variant="dummy_moe_4", vision_variant="mu/32",
        memory_token_count=2, query_dim=8, skill_dim=8, skill_token_count=2,
        interface_hidden_dim=16,
    )
    model = config.create(jax.random.key(3))
    images = {name: jnp.zeros((batch, 224, 224, 3), jnp.float32) for name in (
        "base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")}
    masks = {name: jnp.ones((batch,), jnp.bool_) for name in images}
    masks["right_wrist_0_rgb"] = jnp.zeros((batch,), jnp.bool_)
    prompt = jnp.zeros((batch, 32), jnp.int32).at[:, :4].set(jnp.asarray([2, 10, 11, 1]))
    prompt_mask = jnp.arange(32)[None] < 4
    observation = __import__("openpi.models.model", fromlist=["Observation"]).Observation(
        images=images, image_masks=masks, state=jnp.zeros((batch, 8), jnp.float32),
        skills=jnp.zeros((batch, 1), jnp.int32), skill_mask=jnp.ones((batch, 1), jnp.bool_),
        tokenized_prompt=prompt, tokenized_prompt_mask=jnp.broadcast_to(prompt_mask, prompt.shape),
    )
    return model, observation


def test_fixed_bank_is_128_and_top2_softmax_is_normalized():
    rng = np.random.default_rng(4)
    query = rng.normal(size=(384, 16)).astype(np.float32)
    skill = rng.normal(size=(384, 8)).astype(np.float32)
    bank = build_initial_bank(query, skill, bank_size=128, query_seed_clusters=64, seed=5)
    assert bank.query.shape == (128, 16)
    assert bank.skill.shape == (128, 8)
    assert int(bank.support.sum()) == 384
    assert bank.exemplars.shape == (128, 3)
    merged, indices, weights, _ = retrieve_numpy(query[:2], bank.query, bank.skill, top_k=2)
    assert merged.shape == (2, 8)
    assert indices.shape == weights.shape == (2, 2)
    np.testing.assert_allclose(weights.sum(axis=-1), 1.0)


def test_query_nce_uses_task_progress_and_nearby_trajectory_pairs():
    query = jnp.asarray([[1., 0.], [0.9, 0.1], [-1., 0.], [-0.9, -0.1]])
    loss = query_contrastive_loss(
        query, jnp.asarray([0, 0, 1, 1]), jnp.asarray([0., 0.05, 0., 0.05]),
        jnp.asarray([0, 0, 1, 1]), jnp.asarray([0, 1, 0, 1]), temperature=0.07,
    )
    assert np.isfinite(float(loss))
    no_positive = query_contrastive_loss(
        query, jnp.arange(4), jnp.zeros((4,)), jnp.arange(4), jnp.zeros((4,)), temperature=0.07,
    )
    assert float(no_positive) == 0.0


def test_official_task_id_uses_source_filename_when_instruction_is_ambiguous():
    ids = official_task_ids(np.asarray([
        "KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet_demo.hdf5",
        "KITCHEN_SCENE2_open_the_top_drawer_of_the_cabinet_demo.hdf5",
    ]))
    assert ids.shape == (2,)
    assert ids[0] != ids[1]


def test_memory_tokens_state_and_skill_tokens_join_flow_prefix():
    model, observation = _tiny_model()
    prefix, mask, _ = model._prefix_with_memory(observation)
    conditioned = observation.replace(memory_features=jnp.ones((2, 8), jnp.float32))
    prefix_with_skill, skill_mask, _ = model._prefix_with_memory(conditioned)
    assert prefix_with_skill.shape[1] == prefix.shape[1] + 2
    assert skill_mask.shape == mask.shape[:-1] + (prefix_with_skill.shape[1],)
    query, skill = model.encode_memory(observation)
    assert query.shape == skill.shape == (2, 8)


def test_stage_a_loss_and_bank_condition_path_run_on_tiny_model():
    model, observation = _tiny_model(batch=2)
    actions = jnp.zeros((2, 2, 8), jnp.float32)
    loss, metrics = model.compute_stage_a_loss(
        jax.random.key(9), observation, actions,
        task_ids=jnp.asarray([0, 0]), progress=jnp.asarray([0., 0.05]),
        episode_ids=jnp.asarray([0, 1]), sample_indices=jnp.asarray([0, 0]), train=False,
    )
    assert np.isfinite(float(loss))
    assert set(metrics) >= {"flow", "query_nce", "skill_huber"}
    conditioned, indices, weights, _ = model.retrieve_skill_condition(
        observation, jnp.eye(8), jnp.ones((8, 8)), top_k=2, temperature=0.1,
    )
    assert conditioned.memory_features.shape == (2, 8)
    assert indices.shape == weights.shape == (2, 2)


def test_trainable_layer_gradient_mask_only_keeps_last_vlm_layers():
    import flax.nnx as nnx

    state = nnx.State.from_flat_path({
        ("PaliGemma", "llm", "layers", "dense_0", "kernel"): nnx.VariableState(
            nnx.Param, jnp.ones((6, 2, 2))
        ),
    })
    masked = mask_earlier_vlm_layers(state, total_layers=6, trainable_layers=2)
    value = next(iter(masked.flat_state().values())).value
    assert np.all(np.asarray(value[:4]) == 0)
    assert np.all(np.asarray(value[4:]) == 1)
