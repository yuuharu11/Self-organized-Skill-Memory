import numpy as np
import pytest

from skill_moe.skillnet.examples.memory_pilot.unseen_libero.expert_ex2 import (
    confidence_features,
    geometric_rank_sample,
    rank_stored_actions,
    join_reference_by_key,
    select_dev_policy,
    single_float,
    select_t0,
)


def test_geometric_rank_favors_near_units_and_marks_uniform_branch():
    err = np.tile(np.arange(1, 51, dtype=np.float32), (5000, 1))
    rank, random_mask = geometric_rank_sample(err, np.random.default_rng(14), p=.2, random_fraction=.1)
    assert rank.shape == random_mask.shape == (5000,)
    assert random_mask.mean() == pytest.approx(.1, abs=.02)
    assert np.all(rank[random_mask] == -1)
    assert np.median(rank[~random_mask]) < 5


def test_t0_increases_with_error_and_clips():
    got = select_t0(np.asarray([0., .01, .04, 1.]), 2.)
    assert np.all(np.diff(got) >= 0)
    assert got[0] == pytest.approx(.05)
    assert got[-1] == pytest.approx(.6)


def test_single_float_handles_squeezed_or_batched_outputs():
    assert single_float(np.asarray(0.25)) == pytest.approx(.25)
    assert single_float(np.asarray([0.25], np.float32)) == pytest.approx(.25)
    with pytest.raises(ValueError):
        single_float(np.asarray([.1, .2]), name='t0')


def test_confidence_features_accept_active7_draft_and_full_pi05_but_no_labels():
    draft = np.ones((3, 10, 7), np.float32)
    normal = np.ones((3, 10, 32), np.float32) * .5
    scores = np.asarray([[2., 1., 0.]] * 3)
    x = confidence_features(draft, normal, scores)
    assert x.shape == (3, 4)
    assert np.isfinite(x).all()
    assert np.allclose(x[0], x[1])
    with pytest.raises(ValueError):
        confidence_features(draft, normal, np.ones((3, 1)))


def test_dev_policy_selects_lowest_mse_and_never_uses_eval():
    baseline = np.asarray([.1, .2])
    selected = select_dev_policy({(.5, None): [.11, .18], (1., .2): [.08, .21]}, baseline)
    assert selected['selected']['scale'] == .5
    assert selected['uses_eval'] is False


def test_training_rank_excludes_every_same_task_episode():
    target=np.zeros((1,10,7),np.float32)
    candidates=np.zeros((3,10,7),np.float32)
    candidates[0,:,:6]=.001  # best-looking action, but from the query episode
    candidates[1,:,:6]=.01   # same task, different episode: also excluded
    candidates[2,:,:6]=.02   # another task
    ids,errors=rank_stored_actions(target,candidates,np.asarray([4,4,5]),query_task=np.asarray([4]),
        candidate_episode=np.asarray([8,9,1]),query_episode=np.asarray([8]),k=1)
    assert ids.tolist()==[[2]]
    assert errors[0,0] > 0


def test_rank_candidate_indices_are_bank_row_ids_and_no_future_input_is_accepted():
    target=np.zeros((1,10,7),np.float32);candidates=np.zeros((2,10,7),np.float32)
    candidates[1,:,:6]=.1
    ids,_=rank_stored_actions(target,candidates,np.asarray([0,1]),query_task=np.asarray([0]),k=1)
    assert ids.tolist()==[[1]]
    # The predictor API has no target/future-action parameter; only action
    # chunks already observed/produced and retrieval scores can be supplied.
    import inspect
    assert set(inspect.signature(confidence_features).parameters)=={'draft','pi05','scores'}


def test_zero_t0_floor_is_supported_and_dev_selection_records_it():
    assert select_t0(np.asarray([0.]),1.,low=0.).tolist()==[0.]
    chosen=select_dev_policy({(1.,None,.05):[.2],(1.,None,0.):[.1]},np.asarray([.3]))
    assert chosen['selected']['high_confidence_t0_floor']==0.


def test_pi05_reference_joins_by_episode_frame_identity_not_cache_row_id():
    keys=np.asarray([[0,0,0],[1,3,10],[2,8,20]],np.int64)
    order=np.asarray([2,0,1])
    reference_keys=keys[order]
    target=np.arange(3*10*7,dtype=np.float32).reshape(3,10,7)
    reference_target=target[order]
    actions=reference_target+.25
    got=join_reference_by_key(keys,target,reference_keys,reference_target,actions)
    np.testing.assert_array_equal(got,target+.25)
    with pytest.raises(AssertionError):
        join_reference_by_key(keys,target,reference_keys+100,reference_target,actions)

