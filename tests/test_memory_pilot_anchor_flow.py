import numpy as np
import pytest

from skill_moe.skillnet.examples.emergent.light.intermediate.anchor_flow import (
    _gripper_timing_tv, _rollout_flow, entropy_t0, join_predicted_subtask_rows,
    attach_subtask_source, moment_match_bases, parse_source_mix, predicted_subtask_file,
    similarity_t0, softmax_cosine,
)


def test_source_mix_is_normalized_and_rejects_invalid_values():
    assert parse_source_mix("2,1,1") == (0.5, 0.25, 0.25)
    with pytest.raises(ValueError):
        parse_source_mix("1,0")
    with pytest.raises(ValueError):
        parse_source_mix("-1,1,1")


def test_entropy_mapping_is_monotone_and_bounded():
    values = entropy_t0(np.asarray([0.0, np.log(2) / 2, np.log(2)]), 2, 0.2, 0.8)
    assert np.allclose(values, [0.2, 0.5, 0.8])
    assert np.all(np.diff(values) >= 0)


def test_cosine_weights_sum_to_one_and_prefer_near_key():
    weights = softmax_cosine([1.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], 0.1)
    assert np.isclose(weights.sum(), 1.0)
    assert weights[0] > 0.99


def test_gripper_timing_tv_is_zero_for_identical_trajectories():
    action = np.zeros((3, 2, 5, 7), np.float32)
    action[:, :, 2, 6] = 0.8
    assert _gripper_timing_tv(action, action[0]) == {"positive": 0.0, "negative": 0.0}


def test_absolute_similarity_maps_high_confidence_to_less_noise():
    t0 = similarity_t0(np.asarray([0.1, 0.5, 0.9]), 0.1, 0.9, 0.2, 0.8)
    assert np.allclose(t0, [0.8, 0.5, 0.2])
    assert np.all(np.diff(t0) <= 0)


def test_moment_matching_keeps_between_and_within_unit_variance():
    centers = np.asarray([[0.0, 0.0], [2.0, 0.0]], np.float32)
    mean, variance = moment_match_bases(centers, np.asarray([0.5, 0.5]), [0.5, 0.5])
    assert np.allclose(mean, [1.0, 0.0])
    assert np.allclose(variance, [1.25, 0.25])


def test_t0_one_is_exactly_the_noise_only_start():
    class ZeroFlow:
        def __call__(self, x, t, f, s):
            return .1*x+t[:,None]+f+s

    rows = {"a": np.zeros((2, 3), np.float32),
            "f": np.zeros((2, 1), np.float32),
            "s": np.zeros((2, 1), np.float32)}
    norm = {"a_mean": np.zeros(3, np.float32), "a_scale": np.ones(3, np.float32),
            "f_mean": np.zeros(1, np.float32), "f_scale": np.ones(1, np.float32),
            "s_mean": np.zeros(1, np.float32), "s_scale": np.ones(1, np.float32),
            "action_shape": (3,)}
    noise = np.arange(6, dtype=np.float32).reshape(1, 2, 3)
    source = np.full_like(noise, 100.0)
    ordinary = _rollout_flow(ZeroFlow(), norm, rows, None, noise, [1.0],
                             steps=4, device="cpu")
    anchored = _rollout_flow(ZeroFlow(), norm, rows, source, noise, [1.0],
                             steps=4, device="cpu")
    assert np.array_equal(ordinary, anchored)


def test_predicted_subtask_join_preserves_rows_and_matches_episode_step(tmp_path):
    path = tmp_path / "cache_aligned_subtasks.jsonl"
    path.write_text(
        '{"episode_id":"ep-a","step":0,"subtask":"pick cup"}\n'
        '{"episode_id":"ep-a","step":5,"subtask":"place cup"}\n')
    rows = {"episode_id": np.asarray(["ep-a", "ep-a"]),
            "frame": np.asarray([0, 5]), "a": np.zeros((2, 1))}
    joined = join_predicted_subtask_rows(rows, path)
    assert len(joined["subtask_text"]) == len(rows["episode_id"])
    assert joined["subtask_text"].tolist() == ["pick cup", "place cup"]


def test_predicted_subtask_join_fails_if_any_cache_row_is_missing(tmp_path):
    path = tmp_path / "cache_aligned_subtasks.jsonl"
    path.write_text('{"episode_id":"ep-a","step":0,"subtask":"pick cup"}\n')
    rows = {"episode_id": np.asarray(["ep-a", "ep-a"]), "frame": np.asarray([0, 5])}
    with pytest.raises(ValueError, match="missed 1/2 rows"):
        join_predicted_subtask_rows(rows, path)


def test_split_specific_predicted_files_use_held_out_suite_checkpoints():
    root = "/work/.skillnet-state/outputs/memory-pilot/subtask-pi05-20261027T165005Z"
    file_a = predicted_subtask_file(root, "A")
    file_b = predicted_subtask_file(root, "B")
    assert "split-a" in str(file_a)
    assert "split-b" in str(file_b)


def test_none_and_instruction_sources_have_expected_text(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "manifest.json").write_text(
        '{"episodes":[{"episode_id":"ep-a","task":"open the drawer"}]}'
    )
    rows = {"episode_id": np.asarray(["ep-a"]), "frame": np.asarray([0])}
    none = attach_subtask_source(rows, "none", split="A", cache_dir=cache,
                                annotations_path="unused", predicted_root="unused")
    instruction = attach_subtask_source(rows, "instruction", split="A", cache_dir=cache,
                                        annotations_path="unused", predicted_root="unused")
    assert none["subtask_text"].tolist() == [""]
    assert instruction["subtask_text"].tolist() == ["open the drawer"]


def test_unknown_motion_classes_are_not_removed_from_s1_evaluation():
    from skill_moe.skillnet.examples.emergent.light.intermediate.anchor_flow import _prepare_split_rows
    rows={'task_index':np.array([20,30,10,12]),'skill_class':np.array([0,0,0,99]),
          'valid_h25':np.ones(4,bool),'episode_id':np.array(['a','b','c','d']),
          'a':np.zeros((4,25,7),np.float32)}
    train,test,excluded=_prepare_split_rows(rows,'A',smoke=False,seed=0,
                                smoke_train_episodes=3,smoke_eval_episodes=3)
    assert excluded == 0 and set(test['skill_class']) == {0,99}


def test_leave_recording_out_removes_values_keys_and_variance_without_fallback():
    from skill_moe.skillnet.examples.emergent.light.intermediate.anchor_flow import UnitBank,_raw_top1
    class Units:prototypes=np.array([[1234.,1234.]],np.float32)
    bank=UnitBank(Units(),1.)
    bank.add(0,[1000,1000],[100,100],'same')
    bank.add(0,[2,4],[1,0],'other')
    ids,centers,keys,sigmas,eps=bank.centers_keys_sigmas(['same'])
    np.testing.assert_array_equal(centers,[[2,4]])
    np.testing.assert_array_equal(keys,[[1,0]])
    assert eps == [{'other'}] and sigmas[0] == np.float32(.03)
    assert not len(bank.centers_keys_sigmas(['same','other'])[0])
    with pytest.raises(ValueError,match='cross-episode'):
        _raw_top1(np.array([1.,0.]),np.array([[1.,0.]]),np.zeros((1,25,7)),['same'],'same',None)


def test_training_anchor_excludes_all_same_task_recordings(monkeypatch):
    from skill_moe.skillnet.examples.emergent.light.intermediate.anchor_flow import UnitBank,_make_flow_sources
    class Units:prototypes=np.zeros((1,2),np.float32)
    class Decoder:
        residual_dim=0
        action_shape=(1,2)
        def decode(self,b,r,f,s):return b[:,None]
    bank=UnitBank(Units(),1.)
    bank.add(0,[100,100],[1,0],'a');bank.add(0,[200,200],[1,0],'near_a')
    bank.add(0,[2,4],[1,0],'other')
    rows={'task_index':np.array([20,20,30]),'episode_id':np.array(['a','near_a','other']),
          'f':np.zeros((3,1)), 's':np.zeros((3,1))}
    sources=_make_flow_sources(Decoder(),bank,rows,np.zeros((3,2)),np.array([[1,0]]*3),.1,
                          {'a_mean':np.zeros(2),'a_scale':np.ones(2)},seed=0)
    assert np.max(np.abs(sources[0][:2]))<5  # neither own nor neighboring recording leaks
    assert sources[0][2].min()>140


def test_anchor_loss_targets_clean_action_on_augmented_path():
    import torch
    from skill_moe.skillnet.examples.emergent.light.intermediate.anchor_flow import _flow_loss
    a=torch.tensor([[2.]])
    eps=torch.tensor([[4.]])
    t=torch.tensor([.5])
    xt=torch.tensor([[7.]])  # off-path anchor=10
    exact=(xt-a)/t[:,None]
    assert _flow_loss(exact,xt,t,eps,a,torch.tensor([0])).item() == 0
    assert _flow_loss(eps-a,xt,t,eps,a,torch.tensor([0])).item() > 0
    assert _flow_loss(eps-a,xt,t,eps,a,torch.tensor([2])).item() == 0


def test_sigma_is_applied_in_base_space_before_nonlinear_decode():
    from skill_moe.skillnet.examples.emergent.light.intermediate.anchor_flow import _draw_decoded_samples
    class Decoder:
        residual_dim=0
        def decode(self,b,r,f,s):return (b*b)[:,None]
    rows={'f':np.zeros((1,1)), 's':np.zeros((1,1))}
    norm={'a_mean':np.zeros(1),'a_scale':np.array([2.])}
    got=_draw_decoded_samples(Decoder(),np.array([[2.]]),np.array([.5]),rows,norm,
                             np.random.default_rng(5),draws=8)
    z=np.random.default_rng(5).standard_normal((8,1,1)).astype(np.float32)
    np.testing.assert_allclose(got,(2+.5*z)**2/2,rtol=1e-6)


def test_gripper_timing_counts_switches_not_held_initial_sign_and_checks_initial_error():
    from skill_moe.skillnet.examples.emergent.light.intermediate.anchor_flow import _metrics
    target=np.zeros((1,5,7),np.float32);target[:,:,6]=1
    target[:,2:,6]=-1
    pred=np.tile(target[None],(8,1,1,1));pred[:,:,:3,6]=1
    pred[:,:,3:,6]=-1
    assert _gripper_timing_tv(pred,target)=={'positive':0.,'negative':1.}
    target[:,:,6]=1;pred[:,:,:,6]=-1
    assert _metrics(pred,target)['gripper_timing_tv']['initial_command_error'] == 1


def test_best_of_eight_minimizes_per_query_and_s1b_denominator_is_new_motion_only():
    from skill_moe.skillnet.examples.emergent.light.intermediate.anchor_flow import _sample_mse,_post_add_rate
    target=np.zeros((2,1,7),np.float32)
    pred=np.ones((8,2,1,7),np.float32)
    pred[0,0]=0;pred[1,1]=0
    score=_sample_mse(pred,target)
    assert score['best_of_8_mse']==0 and score['mse']>0
    assert _post_add_rate([0,0,1,1],[0,0,1,0]) == .5
    assert _post_add_rate([0],[1]) is None


def test_s1b_excludes_known_pick_place_intervals_inside_novel_tasks():
    from skill_moe.skillnet.examples.emergent.light.intermediate.anchor_flow import _novel_motion_mask
    rows={'task_index':np.array([12,12,15,10]),'skill_class':np.array([0,2,1,2])}
    np.testing.assert_array_equal(_novel_motion_mask(rows,'A',[0,1]),[False,True,False,False])
    assert not _novel_motion_mask(rows,'B',[0,1]).any()


def test_s1b_registers_after_three_supporting_episodes_and_retrieves_on_fourth(monkeypatch):
    from skill_moe.skillnet.examples.emergent.light.intermediate import anchor_flow as af
    from skill_moe.skillnet.examples.emergent.light.methods.m6 import M6
    class FixedDistance(M6):
        def _functional_embedding(self,b): return np.asarray(b,np.float32).reshape(-1)
    m=FixedDistance(min_support=10,min_episodes=3)
    m.base_dim=1;m._centers=[np.array([0.],np.float32)]
    m.support_counts=[10];m.episode_sets=[{'train'}];m.epsilon=.1;m._fitted=True
    m._sync_prototypes()
    bank=af.UnitBank(m,1.);bank.add(0,[0],[1,0],'train')
    episodes=np.repeat(['a','b','c','d'],4)
    rows={'episode_id':episodes,'frame':np.tile(np.arange(4),4),
          'a':np.zeros((16,25,7),np.float32)}
    original=af._raw_top1
    def checked_raw(q,keys,acts,eps,query_episode,prior_new):
        assert all(x['episode_id']!=query_episode for x in prior_new)
        return original(q,keys,acts,eps,query_episode,prior_new)
    monkeypatch.setattr(af,'_raw_top1',checked_raw)
    snapshots,order=af._m6_new_suite_stream(m,bank,rows,np.full((16,1),10,np.float32),
                    np.tile([0.,1.],(16,1)),temperature=.05,initial_unit_ids=[0],
                    train_keys=np.array([[1.,0.]],np.float32),train_actions=np.zeros((1,25,7)),
                    train_episodes=['train'],seed=2)
    assert len(m.prototypes)==2 and m.registered_count==1
    assert not any(s['selected_post_add'] for s in snapshots[:12])
    assert all(s['selected_post_add'] for s in snapshots[12:])
    assert af._post_add_rate(np.ones(16,bool),[s['selected_post_add'] for s in snapshots])==.25
    for start in range(0,16,4): assert len(set(episodes[order[start:start+4]]))==1
    assert m.pending_batch_size==200


def test_calibration_uses_real_flow_features_and_eight_component_draws(monkeypatch):
    from skill_moe.skillnet.examples.emergent.light.intermediate import anchor_flow as af
    rows={'episode_id':np.array(['v1','v2']),'a':np.zeros((2,25,7),np.float32),
          'f':np.full((2,3),7,np.float32),'k':np.zeros((2,2),np.float32),'s':np.zeros((2,8),np.float32)}
    monkeypatch.setattr(af,'_query_embedding',lambda *a,**k:np.ones((2,2),np.float32))
    class Bank:
        def centers_keys_sigmas(self):
            return np.arange(2),np.array([[0.],[10.]],np.float32),np.eye(2,dtype=np.float32),np.ones(2),[{'t'},{'t'}]
    def decoded(m,b,s,r,n,rng,*,draws):
        assert b.shape==(8,2,1) and draws==8
        assert len(np.unique(b))==2
        return np.zeros((8,2,175),np.float32)
    monkeypatch.setattr(af,'_draw_decoded_samples',decoded)
    def rollout(m,n,r,s,eps,t0,*,steps,device):
        assert (r['f']==7).all() and steps==10
        assert s.shape==eps.shape==(8,2,175)
        return np.zeros((8,2,25,7),np.float32)
    monkeypatch.setattr(af,'_rollout_flow',rollout)
    got=af._fit_t0_calibration(None,{'a_mean':np.zeros(175),'a_scale':np.ones(175)},None,None,
            Bank(),None,{'temperature':.1},rows,None,
            raw_memory=(np.eye(2),np.zeros((2,25,7)),np.array(['t1','t2'])),seed=0,device='cpu')
    assert '1.0' in got['fixed'] and got['validation_rows']==2
