from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

EXAMPLES = Path(__file__).resolve().parents[1] / "skill_moe" / "skillnet" / "examples"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

from memory_pilot.anchor_flow.memory import (backbone_digest, fit_memory, query_raw_action,
                                              sample_memory_anchor, sample_optimus_anchor,
                                              shuffle_memory_values, validate_key_text_source)
from memory_pilot.anchor_flow.add_task_memory import append_task
from memory_pilot.anchor_flow.eval_series_a import make_cases
from openpi.models.pi0_anchor_flow import integrate_velocity, mix_anchor_with_noise


def test_t0_one_integrates_same_euler_path_as_ordinary_noise_start():
    x_noise = jnp.arange(2 * 3 * 4, dtype=jnp.float32).reshape(2, 3, 4) / 10

    def flow(x, t):
        del t
        return jnp.ones_like(x) * 0.25

    ordinary = jax.lax.fori_loop(
        0, 5, lambda i, x: x - (1.0 / 5) * flow(x, 1.0 - i / 5), x_noise
    )
    anchored = integrate_velocity(flow, x_noise, 1.0, 5)
    np.testing.assert_array_equal(np.asarray(anchored), np.asarray(ordinary))


def test_anchor_start_has_pi05_chunk_shape_and_follows_star():
    anchor = jnp.ones((4, 10, 32), jnp.float32)
    noise = jnp.zeros((4, 10, 32), jnp.float32)
    starts = jnp.asarray([0.0, 0.25, 0.75, 1.0])
    samples = mix_anchor_with_noise(anchor, noise, starts)
    assert samples.shape == (4, 10, 32)
    np.testing.assert_array_equal(np.asarray(samples[:, 0, 0]), np.asarray([1.0, 0.75, 0.25, 0.0]))
    assert np.all(np.isfinite(np.asarray(samples)))


def test_m6_add_is_append_only_and_does_not_change_codec_weights():
    rng = np.random.default_rng(5)
    n = 36
    actions = rng.normal(size=(n, 10, 32)).astype(np.float32)
    states = rng.normal(size=(n, 32)).astype(np.float32)
    keys = rng.normal(size=(n, 8)).astype(np.float32)
    episodes = np.asarray([f"episode-{i // 3}" for i in range(n)])
    initials = np.asarray([f"init-{i // 3}" for i in range(n)])
    memory = fit_memory({
        "actions": actions, "state": states, "key": keys,
        "episode_id": episodes, "init_state_id": initials,
        "suite": np.asarray(["libero90"] * n),
        "normalization_sha256": np.asarray("a" * 64),
        "key_text_source": np.asarray("instruction_only"),
    }, seed=1, epochs=1)
    before = backbone_digest(memory)
    anchor = sample_memory_anchor(memory, keys[0], states[0], seed=7)
    assert anchor.shape == (10, 32)
    assert np.all(np.isfinite(anchor))
    raw_anchor, similarity = query_raw_action(memory, keys[0])
    assert raw_anchor.shape == (10, 32)
    assert np.isfinite(similarity)
    moment_anchor = sample_optimus_anchor(memory, keys[0], states[0], seed=8)
    assert moment_anchor.shape == (10, 32)
    shuffled = shuffle_memory_values(memory, seed=9)
    assert len(shuffled.memory) == len(memory.memory)
    assert backbone_digest(shuffled) == backbone_digest(memory)
    weights_before = np.asarray([np.asarray(row["key"]).copy() for row in memory.memory])
    if memory.memory:
        memory.memory_add(keys[0], memory.memory[0]["base"], {
            "episode_id": "later-episode", "init_state_id": "later-init", "task": "series-a"
        })
    assert backbone_digest(memory) == before
    assert len(memory.memory) >= len(weights_before)
    np.testing.assert_array_equal(
        np.asarray([np.asarray(row["key"]) for row in memory.memory[:len(weights_before)]]),
        weights_before,
    )


def test_key_text_source_requires_instruction_or_pi05_generated_subtask():
    assert validate_key_text_source({"key_text_source": np.asarray("instruction_only")}) == "instruction_only"
    assert validate_key_text_source({"key_text_source": np.asarray("pi05_generated_subtask"),
                                     "key_text_generator": np.asarray("pi05"),
                                     "key_text_generator_checkpoint": np.asarray("/model/params")}) == "pi05_generated_subtask"
    for invalid in (
        {},
        {"key_text_source": np.asarray("instruct2plan_annotation")},
        {"key_text_source": np.asarray(["instruction_only", "pi05_generated_subtask"]),
         "key_text_generator": np.asarray("pi05")},
        {"key_text_source": np.asarray("pi05_generated_subtask"),
         "key_text_generator": np.asarray("pi05")},
    ):
        try:
            validate_key_text_source(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted unsafe/ambiguous key source: {invalid}")


def test_task_addition_rejects_unapproved_suite_before_mutating_memory():
    n = 1
    data = {
        "actions": np.zeros((n, 10, 32), np.float32),
        "state": np.zeros((n, 32), np.float32),
        "key": np.zeros((n, 8), np.float32),
        "episode_id": np.asarray(["ep"]),
        "init_state_id": np.asarray(["init"]),
        "task": np.asarray(["task"]),
        "suite": np.asarray(["libero40"]),
        "normalization_sha256": np.asarray("a" * 64),
        "key_text_source": np.asarray("instruction_only"),
    }
    try:
        append_task(None, data)
    except ValueError as exc:
        assert "only LIBERO-10 or LIBERO-Skill" in str(exc)
    else:
        raise AssertionError("LIBERO-40 was accepted as a post-fit memory addition")


def test_minimal_s2_gate_matrix_has_paired_controls_and_fewer_cases():
    checkpoint = "/tmp/fixed-finetuned-s2/params"
    cases = make_cases(20, minimal=True, checkpoint=checkpoint, seed=31)
    assert len(cases) == 92
    assert len(cases) * 20 == 1840
    assert all(row["checkpoint"] == checkpoint and row["seed"] == 31 for row in cases)
    assert all(row["query_text_source"] == "instruction_only" for row in cases)
    retention = [row for row in cases if row["suite"] == "libero_90"]
    assert len(retention) == 20
    new_rows = [row for row in cases if row["suite"] != "libero_90"]
    assert len(new_rows) == 72
    assert {row["condition"] for row in new_rows} == {
        "anchor_entropy_current_memory", "anchor_disabled_t0_1",
        "anchor_memory_shuffled", "anchor_before_task_add",
    }
    assert all(row["memory_snapshot"] == "initial_libero90" for row in retention)


def test_training_batch_wraps_without_empty_rows_after_first_epoch():
    path=EXAMPLES/'memory_pilot/anchor_flow'
    sys.path.insert(0,str(path))
    from train import batch_indices,observation_batch
    order=np.array([4,2,1,0,3])
    for step in range(20):
        got=batch_indices(order,step,3)
        np.testing.assert_array_equal(got,order[np.arange(step*3,(step+1)*3)%5])
        assert len(got)==3
    row={'image_base':np.full((2,4,4,3),255,np.uint8),
         'image_left_wrist':np.zeros((2,4,4,3),np.uint8),
         'state':np.zeros((2,32),np.float32),'tokenized_prompt':np.zeros((2,200),np.int32),
         'tokenized_prompt_mask':np.zeros((2,200),bool)}
    obs=observation_batch(row,None)
    np.testing.assert_allclose(np.asarray(obs.images['base_0_rgb']).min(), 1.0, atol=2e-7)
    assert np.asarray(obs.images['left_wrist_0_rgb']).max()==-1
    assert not np.asarray(obs.image_masks['right_wrist_0_rgb']).any()


def test_excluded_episode_or_init_cannot_reenter_through_unit_center_or_sigma():
    from memory_pilot.anchor_flow.memory import _component_values
    class Memory:
        _centers=[np.array([1000.,1000.])]
        memory=[{'unit_id':0,'base':np.array([1000.,1000.]),'meta':{'episode_id':'same','init_state_id':'a'}},
                {'unit_id':0,'base':np.array([500.,500.]),'meta':{'episode_id':'other','init_state_id':'a'}},
                {'unit_id':0,'base':np.array([2.,4.]),'meta':{'episode_id':'good','init_state_id':'b'}}]
    center,sigma=_component_values(Memory(),0,None,{'episode_id':'same','init_state_id':'a'})
    np.testing.assert_array_equal(center,[2,4]);assert sigma==0


def test_export_uses_only_complete_nonoverlapping_chunks():
    path=EXAMPLES/'memory_pilot/anchor_flow';sys.path.insert(0,str(path))
    from export_libero90 import complete_starts
    np.testing.assert_array_equal(complete_starts(29),[0,10])
    assert not len(complete_starts(9))


def test_export_rejects_missing_declared_rlds_shards(tmp_path):
    import json
    path=EXAMPLES/'memory_pilot/anchor_flow';sys.path.insert(0,str(path))
    from export_libero90 import source_inventory
    (tmp_path/'dataset_info.json').write_text(json.dumps({'splits':[{
        'name':'train','shardLengths':['3','4']}]}))
    (tmp_path/'liber_o90-train.tfrecord-00000-of-00002').touch()
    try:source_inventory(tmp_path)
    except ValueError:pass
    else:raise AssertionError('accepted missing RLDS shard')
    (tmp_path/'liber_o90-train.tfrecord-00001-of-00002').touch()
    files,n=source_inventory(tmp_path)
    assert len(files)==2 and n==7


def test_prepared_cosine_search_matches_exclusion_and_stable_ties():
    path=EXAMPLES/'memory_pilot/anchor_flow';sys.path.insert(0,str(path))
    from prepare_sources import training_neighbors
    from memory_pilot.anchor_flow.memory import _retrieved_components
    class Memory:
        temperature=.1
        memory=[{'key':np.array(k,np.float32),'meta':{'episode_id':e,'init_state_id':i}}
                for k,e,i in [([1,0],'same','x'),([1,0],'other','same-init'),
                              ([1,0],'b','b'),([1,0],'c','c'),([0,1],'d','d')]]
    memory=Memory();queries=np.array([[1,0],[0,1]],np.float32)
    episodes=np.array(['same','query2']);initials=np.array(['same-init','z'])
    got=training_neighbors(memory,queries,episodes,initials,top_k=3,block=1)
    np.testing.assert_array_equal(got[0],[2,3,4])
    for q,ep,init,ix in zip(queries,episodes,initials,got):
        meta={'episode_id':ep,'init_state_id':init}
        a,wa=_retrieved_components(memory,q,top_k=3,meta=meta)
        b,wb=_retrieved_components(memory,q,top_k=3,meta={**meta,'_prepared_candidates':ix})
        assert [r['meta']['episode_id'] for r in a]==[r['meta']['episode_id'] for r in b]
        np.testing.assert_allclose(wa,wb)
