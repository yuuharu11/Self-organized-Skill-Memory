"""Optimus-R baseline (stage A) and the public LIBERO evaluation protocol."""
from __future__ import annotations

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from openpi.models.pi0_optimus_r import query_contrastive_loss
from openpi.models.pi0_optimus_r_config import Pi0OptimusRConfig

from tests.support import (
    EXAMPLES,
    MEMORY_PILOT,
    add_path,
)

add_path(EXAMPLES)
add_path(MEMORY_PILOT / "optimus_r")
# optimus_r/data.py shares the top-level name "data" with shape_skills/data.py.
# Drop any cached copy before and after importing so test order does not matter.
sys.modules.pop("data", None)
from bank import build_initial_bank, retrieve_numpy  # noqa: E402
from train_stage_a import mask_earlier_vlm_layers  # noqa: E402

from data import official_task_ids  # noqa: E402

sys.modules.pop("data", None)
add_path(MEMORY_PILOT / "anchor_flow")
from run_case import (  # noqa: E402
    environment_horizon,
    environment_seed,
    episode_status,
    initial_state_index,
    suite_control_limit,
)

# ── Optimus-R stage A ─────────────────────────────────────────────────────────────────────────────

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
    pytest.importorskip('libero.libero.benchmark')
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


# ── Control budgets and initial-state selection of the public evaluator — Control budgets and initial-state selection of the public Skill evaluator. ───

@pytest.mark.parametrize('suite,limit', [('libero_skill_obj',1600),('libero_skill',1600),
    ('libero_10',520),('libero_90',400),('libero_spatial',220),('libero_object',280),('libero_goal',300)])
def test_public_control_budgets(suite,limit):
    assert suite_control_limit(suite)==limit


def test_initial_states_are_public_prefix_not_wrapped():
    assert [initial_state_index(i,50) for i in range(20)]==list(range(20))
    with pytest.raises(IndexError):initial_state_index(50,50)
    with pytest.raises(IndexError):initial_state_index(-1,50)


def test_skill_environment_seed_matches_public_fixed_seed():
    assert [environment_seed('libero_skill_obj',7,i) for i in range(20)]==[7]*20
    assert environment_seed('libero_10',7,1)==8  # retain non-Skill paired protocol


def test_horizon_covers_full_control_and_stabilization():
    assert environment_horizon(1600,10)==1611
    assert environment_horizon(520,10)==1000
    assert environment_horizon(400,10)==1000


@pytest.mark.parametrize('success,returned_done,internal_done,expected',[
    (False,False,False,(False,False)),
    (False,False,True,(False,True)),  # LIBERO masks the robosuite timeout
    (False,True,False,(False,True)),
    (True,True,False,(True,True)),
])
def test_timeout_is_not_success(success,returned_done,internal_done,expected):
    from types import SimpleNamespace
    env=SimpleNamespace(env=SimpleNamespace(done=internal_done),check_success=lambda:success)
    assert episode_status(env,returned_done)==expected
