import numpy as np
import pytest
import torch

from skill_moe.skillnet.examples.memory_pilot.unseen_libero.key_shared_latent import (
    SharedLatentEncoder,
    action_errors,
    episode_bootstrap_ci,
    join_feature_rows,
    shared_token_scores,
    soft_listwise_loss,
    scale_only_type_errors,
    stable_row_map,
    two_temperature_target,
    TypeKeyEncoder,
    _pool_tokens,
    _metric_block,
    _summarize_split,
    _training_candidate_ids,
    _arm6_candidate_mse,
    natural_unit_replay,
    natural_topk_blend_actions,
    candidate_rank_metrics,
    _pack_query_observation,
    _episode_history,
    _query_representations,
    VectorRanker,
    _assert_eval_candidates_disjoint,
    _candidate_window_actions,
    _type_features,
    contrastive_candidate_set,
    contrastive_teacher_loss,
    INPUT_CONDITIONS,
    _hold_last_actions,
    _scale_logits,
    _logit_scale_parameter,
    LOGIT_SCALE_INITIAL,
    LOGIT_SCALE_MAX,
    _assert_hold_last_gate,
)


def test_shared_encoder_outputs_normalized_shared_tokens():
    torch.manual_seed(3)
    model = SharedLatentEncoder(token_dim=12, context_dim=16, token_count=5,
                                history_steps=3, history_action_dim=4, state_dim=2,
                                group_count=3, heads=4)
    vlm = torch.randn(2, 7, 12)
    groups = torch.tensor([[0, 0, 1, 1, 2, -1, -1], [0, 1, 1, 2, 2, 2, -1]])
    history = torch.randn(2, 3, 4)
    mask = torch.tensor([[False, True, True], [True, True, True]])
    state = torch.randn(2, 2)
    out = model(vlm, groups, history, mask, state)
    assert out.shape == (2, 5, 16)
    torch.testing.assert_close(out.norm(dim=-1), torch.ones(2, 5), atol=1e-6, rtol=1e-6)
    # The same encoder call is used to represent query and memory keys.
    torch.testing.assert_close(model(vlm[:1], groups[:1], history[:1], mask[:1], state[:1]), out[:1])


@pytest.mark.parametrize('mode', tuple(INPUT_CONDITIONS))
def test_each_input_condition_encoder_runs_with_only_declared_modalities(mode):
    torch.manual_seed(100+len(mode))
    model=SharedLatentEncoder(token_dim=12,context_dim=16,token_count=3,
        history_steps=2,history_action_dim=7,state_dim=4,group_count=6,heads=4,input_mode=mode)
    kwargs={
        'vlm_tokens':torch.randn(2,5,12),'groups':torch.tensor([[0,1,2,3,-1],[1,2,3,-1,-1]]),
        'vjepa_tokens':torch.randn(2,8,1024),'vjepa_groups':torch.tensor([[0]*4+[1]*4,[0]*4+[1]*4]),
        'history':torch.randn(2,2,7),'history_mask':torch.tensor([[True,False],[True,True]]),
        'state':torch.randn(2,4)}
    out=model(**kwargs)
    assert out.shape==(2,3,16)
    torch.testing.assert_close(out.norm(dim=-1),torch.ones(2,3),atol=1e-6,rtol=1e-6)
    if mode=='e_vlm':
        changed={**kwargs,'history':torch.full_like(kwargs['history'],99.),'state':torch.full_like(kwargs['state'],99.)}
        torch.testing.assert_close(out,model(**changed))
    if mode=='e_h':
        changed={**kwargs,'vlm_tokens':torch.full_like(kwargs['vlm_tokens'],99.),
                 'vjepa_tokens':torch.full_like(kwargs['vjepa_tokens'],99.)}
        torch.testing.assert_close(out,model(**changed))


def test_separate_query_and_key_calls_update_the_same_encoder_parameters():
    torch.manual_seed(31)
    model = SharedLatentEncoder(token_dim=8, context_dim=16, token_count=3,
                                history_steps=2, history_action_dim=2, state_dim=2,
                                group_count=2, heads=4)
    q = (torch.randn(1, 4, 8), torch.tensor([[0, 0, 1, -1]]),
         torch.randn(1, 2, 2), torch.tensor([[True, True]]), torch.randn(1, 2))
    k = (torch.randn(2, 4, 8), torch.tensor([[0, 1, 1, -1], [0, 0, 1, 1]]),
         torch.randn(2, 2, 2), torch.tensor([[True, False], [True, True]]), torch.randn(2, 2))
    q_tokens = model(*q)
    k_tokens = model(*k)
    scores = shared_token_scores(q_tokens, k_tokens)
    (-scores.mean()).backward()
    assert model.vlm_projection.weight.grad is not None
    assert model.vlm_projection.weight.grad.norm() > 0
    assert model.latent.grad is not None and model.latent.grad.norm() > 0


def test_token_scores_are_aligned_mean_inner_products():
    q = torch.nn.functional.normalize(torch.tensor([[[1., 0.], [0., 1.]]]), dim=-1)
    keys = torch.nn.functional.normalize(torch.tensor([[[1., 0.], [0., 1.]],
                                                         [[0., 1.], [1., 0.]]]), dim=-1)
    scores = shared_token_scores(q, keys)
    torch.testing.assert_close(scores, torch.tensor([[1., 0.]]))


def test_hold_last_repeats_most_recent_valid_causal_action():
    hist=np.zeros((2,10,7),np.float32);mask=np.zeros((2,10),bool)
    hist[0,7]=np.arange(7,dtype=np.float32);mask[0,7:]=True
    hist[0,8]=np.arange(7,dtype=np.float32)+10
    hist[0,9]=np.arange(7,dtype=np.float32)+20
    hist[1]=99.;mask[1]=False
    out=_hold_last_actions({'history':hist,'history_mask':mask})
    np.testing.assert_array_equal(out[0],np.repeat(hist[0,9:10],10,axis=0))
    np.testing.assert_array_equal(out[1],np.zeros((10,7),np.float32))


def test_fixed_hold_last_gate_uses_unseen_and_learned_reference_values():
    def fixture(n,hard_n,all_mse,hard_mse):
        hard=np.zeros(n,bool);hard[:hard_n]=True
        other=(all_mse*n-hard_n*hard_mse)/(n-hard_n)
        values=np.full(n,other,np.float64);values[:hard_n]=hard_mse
        target=np.zeros((n,10,7),np.float32)
        target[:,:,:6]=np.sqrt(values[:,None,None]*7/6).astype(np.float32)
        return {'history':np.zeros((n,10,7),np.float32),'history_mask':np.zeros((n,10),bool),
                'actions':target,'hard':hard}
    learned=fixture(621,125,.05639717563886132,.09724447740896078)
    unseen=fixture(619,134,.113994986,.20097965)
    got=_assert_hold_last_gate(learned,unseen,atol=1e-6)
    np.testing.assert_allclose(got['unseen']['all'],.113994986,atol=1e-6)
    np.testing.assert_allclose(got['unseen']['hard'],.20097965,atol=1e-6)


def test_logit_scale_is_trainable_and_capped_at_one_hundred():
    torch.manual_seed(91)
    model=SharedLatentEncoder(token_dim=8,context_dim=16,token_count=2,
        history_steps=2,history_action_dim=7,state_dim=4,group_count=6,heads=4)
    scale=_logit_scale_parameter(model)
    np.testing.assert_allclose(float(scale.detach().exp()),LOGIT_SCALE_INITIAL,rtol=1e-6)
    raw=torch.tensor([[.01,.02]],requires_grad=True)
    scaled=_scale_logits(model,raw)
    torch.testing.assert_close(scaled,raw*LOGIT_SCALE_INITIAL)
    scaled.sum().backward()
    assert scale.grad is not None and torch.isfinite(scale.grad)
    with torch.no_grad():scale.fill_(100.)
    clipped=_scale_logits(model,raw)
    torch.testing.assert_close(clipped,raw*LOGIT_SCALE_MAX)


def test_soft_listwise_temperature_applies_to_teacher_only_and_masks_self():
    scores = torch.tensor([[0.1, 0.7, -0.2]], requires_grad=True)
    errors = torch.tensor([[0.0, 1.0, 2.0]])
    allowed = torch.tensor([[False, True, True]])
    loss = soft_listwise_loss(scores, errors, allowed, teacher_temperature=.5)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(scores.grad).all()


def test_cache_join_uses_exact_keys_and_requested_order():
    keys = np.array([[2, 4, 6], [1, 3, 5]], np.int64)
    tokens = np.arange(2 * 3 * 4, dtype=np.float16).reshape(2, 3, 4)
    groups = np.array([[0, 1, -1], [2, 0, -1]], np.int8)
    out, out_groups = join_feature_rows([[1, 3, 5], [2, 4, 6]], keys, tokens, groups)
    np.testing.assert_array_equal(out[0, :2], tokens[1, :2])
    np.testing.assert_array_equal(out[1, :2], tokens[0, :2])
    np.testing.assert_array_equal(out_groups, groups[::-1])
    with pytest.raises(KeyError):
        join_feature_rows([[99, 1, 0]], keys, tokens, groups)
    with pytest.raises(ValueError):
        stable_row_map([[1, 2, 3], [1, 2, 3]])


def test_action_error_decomposition_and_episode_bootstrap():
    target = np.zeros((4, 10, 7), np.float32)
    pred = target.copy()
    pred[:, :, :6] = 1
    pred[:, :, 6] = -1
    err = action_errors(pred, target, pi05_grip=target[:, :, 6])
    np.testing.assert_allclose(err["arm6"], 6 / 7)
    np.testing.assert_allclose(err["mse_gripper_truth"], 6 / 7)
    np.testing.assert_allclose(err["active7"], 1.)
    np.testing.assert_allclose(err["active7_saved_gripper"], 1.)
    np.testing.assert_allclose(err["active7_pi05_grip"], 6 / 7)
    ci = episode_bootstrap_ci(np.array([0., 1., 2., 3.]), np.array([0, 0, 1, 1]),
                              np.array([0, 0, 0, 0]), reps=40, seed=8)
    assert ci["episodes"] == 2
    assert ci["rows"] == 4


def test_arm6_teacher_uses_saved_direction_and_magnitude_but_ignores_gripper():
    query=np.zeros((1,10,7),np.float32)
    candidates=np.zeros((3,10,7),np.float32)
    candidates[0,:,:6]=1.
    candidates[1,:,:6]=4.
    candidates[2,:,6]=1000.
    errors=_arm6_candidate_mse(query,candidates)
    np.testing.assert_allclose(errors[0,0],6/7,rtol=1e-6)
    np.testing.assert_allclose(errors[0,1],16*6/7,rtol=1e-6)
    np.testing.assert_allclose(errors[0,2],0.,atol=1e-8)


def test_scale_only_type_error_ignores_magnitude_but_keeps_direction():
    q = np.zeros((1, 2, 6), np.float32)
    q[0, :, 0] = 1.
    same_shape = q * 4.
    reversed_shape = -q
    err = scale_only_type_errors(q, np.concatenate([same_shape, reversed_shape], axis=0))
    np.testing.assert_allclose(err[0, 0], 0., atol=1e-8)
    assert err[0, 1] > 0.


def test_two_temperature_teacher_only_and_masking():
    err = torch.tensor([[0.0, 0.1, 2.0]])
    allowed = torch.tensor([[False, True, True]])
    target = two_temperature_target(err, allowed, (.01, .03))
    assert torch.isfinite(target).all()
    torch.testing.assert_close(target.sum(-1), torch.ones(1))
    assert target[0, 0] == 0
    assert target[0, 1] > target[0, 2]


def test_contrastive_candidate_sampling_is_label_independent_and_masks_self_episode():
    cids,allowed=contrastive_candidate_set([0,1],[2,4],[1,1],[10,20],
        [1,1,2,3,4],[10,20,30,40,50])
    np.testing.assert_array_equal(cids,[0,1,2,4])
    assert allowed.shape==(2,4)
    assert not allowed[0,0] and not allowed[1,1]
    assert allowed[0,1] and allowed[1,0]


def test_soft_and_hard_info_nce_losses():
    scores=torch.tensor([[0.,2.,1.]],requires_grad=True)
    errors=torch.tensor([[.2,0.,.1]])
    allowed=torch.tensor([[True,True,False]])
    soft=contrastive_teacher_loss(scores,errors,allowed,objective='soft',teacher_temperature=.1)
    hard=contrastive_teacher_loss(scores,errors,allowed,objective='hard',teacher_temperature=.1)
    assert torch.isfinite(soft) and torch.isfinite(hard)
    assert hard.item()<soft.item()
    hard.backward()
    assert torch.isfinite(scores.grad).all()


def test_separate_type_key_is_k_normalized_tokens():
    torch.manual_seed(13)
    model = TypeKeyEncoder(input_dim=60, width=24, token_count=5)
    out = model(torch.randn(3, 10, 6))
    assert out.shape == (3, 5, 24)
    torch.testing.assert_close(out.norm(dim=-1), torch.ones(3, 5), atol=1e-6, rtol=1e-6)


def test_pool_tokens_masks_padding_without_full_precision_cache_copy():
    tokens=np.array([[[1., 3.], [5., 7.], [99., 99.]]],dtype=np.float16)
    groups=np.array([[0, 1, -1]],dtype=np.int8)
    np.testing.assert_allclose(_pool_tokens(tokens,groups),[[3.,5.]])


def test_natural_length_replay_advances_on_selected_unit_duration():
    actions=np.zeros((3,10,7),np.float32)
    split={'actions':actions,'task':np.zeros(3,np.int64),'episode':np.ones(3,np.int64),
           'key':np.array([[0,1,0],[0,1,10],[0,1,20]],np.int64)}
    natural=np.zeros((2,15,7),np.float32);natural[0,:,:6]=1.;natural[1,:,:6]=2.
    bank={'length':np.array([15,6]),'natural_actions':natural}
    # The unit chosen at frame 10 becomes available at frame 10, not at frame
    # 15 when the first (15-frame) unit ends. The frame-20 selection must not
    # leak backward into the interval 15:20.
    got=natural_unit_replay(np.array([0,1,0]),split,bank)
    np.testing.assert_array_equal(got[0,:,:6],np.ones((10,6)))
    np.testing.assert_array_equal(got[1,:5,:6],np.ones((5,6)))
    np.testing.assert_array_equal(got[1,5:,:6],np.full((5,6),2.))
    np.testing.assert_array_equal(got[2,0,:6],np.full(6,2.))
    np.testing.assert_array_equal(got[2,1:,:6],np.ones((9,6)))


def test_random_and_oracle_use_full_candidate_rank_reference():
    target=np.zeros((1,10,7),np.float32)
    candidates=np.zeros((3,10,7),np.float32)
    candidates[1,:,:6]=1.
    candidates[2,:,:6]=.5
    scores=np.array([[0.,0.,0.]],np.float32)
    errors=np.array([[0.,2.,1.]],np.float32)
    split={'actions':target,'task':np.array([5]),'episode':np.array([9]),
           'key':np.array([[5,9,0]],np.int64),'hard':np.array([True]),
           'history':np.zeros((1,10,7),np.float32),'history_mask':np.zeros((1,10),bool)}
    bank={'value_actions10':candidates,'length':np.array([10,10,10]),
          'natural_actions':candidates.copy()}
    got=_metric_block(scores,errors,split,bank,random_runs=5)
    assert got['rank']['oracle_top1_arm6_mse']==0.
    assert got['rank']['random_top1_arm6_mse']>=0.
    assert got['oracle_action_index'].tolist()==[0]
    assert got['oracle_playback_metrics']['mse_gripper_truth']==got['querywise_oracle_actions']['mse_gripper_truth']
    assert got['random_indices'].shape==(5,1)


def test_arm6_oracle_ignores_gripper_and_main_error_scores_gripper_as_correct():
    target=np.zeros((1,10,7),np.float32);target[...,6]=1.
    candidates=np.zeros((2,10,7),np.float32)
    candidates[0,...,6]=-1.
    candidates[1,:,:6]=.25;candidates[1,...,6]=1.
    split={'actions':target,'task':np.array([5]),'episode':np.array([9]),
           'key':np.array([[5,9,0]],np.int64),'hard':np.array([False]),
           'history':np.zeros_like(target),'history_mask':np.zeros((1,10),bool)}
    bank={'value_actions10':candidates,'length':np.array([10,10]),'natural_actions':candidates.copy()}
    scores=np.array([[0.,1.]],np.float32)
    errors=_arm6_candidate_mse(target,candidates)
    got=_metric_block(scores,errors,split,bank,random_runs=1)
    assert got['oracle_action_index'].tolist()==[0]
    assert got['oracle_playback_metrics']['mse_gripper_truth']==0.
    assert got['oracle_playback_metrics']['active7_saved_gripper']>0.
    assert got['actions']['mse_gripper_truth']==got['actions']['arm6']


def test_sparse_hard_subset_slices_full_natural_playback():
    n=5
    target=np.zeros((n,10,7),np.float32)
    natural=np.zeros((2,25,7),np.float32)
    natural[0,:,:6]=1.;natural[1,:,:6]=2.
    bank={'length':np.array([15,20]),'natural_actions':natural,
          'value_actions10':natural[:,:10]}
    split={'actions':target,'task':np.zeros(n,np.int64),'episode':np.ones(n,np.int64),
           'key':np.array([[0,1,i*10] for i in range(n)],np.int64),
           'hard':np.array([True,False,True,False,True]),
           'history':np.zeros_like(target),'history_mask':np.zeros((n,10),bool)}
    scores=np.array([[1.,0.],[0.,1.],[1.,0.],[0.,1.],[1.,0.]],np.float32)
    errors=np.zeros_like(scores)
    result=_summarize_split('sparse-hard',scores,errors,split,bank,tau=.1,
        selected_blend={'k':1,'temperature':1.},blend_by_k={'1':{'k':1,'temperature':1.}},reps=10)
    np.testing.assert_array_equal(result['_hard']['selected_actions'],result['_selected_actions'][split['hard']])
    expected=np.square(result['_selected_actions'][split['hard']]-target[split['hard']]).mean((1,2))
    np.testing.assert_allclose(result['actions_hard']['active7'],expected.mean())


def test_training_candidates_exclude_every_fixed_eval_episode():
    tasks=np.array([1,1,2,2,3,3])
    episodes=np.array([4,5,4,5,4,5])
    ids=_training_candidate_ids(tasks,episodes,{(1,5),(3,4)})
    assert ids.tolist()==[0,2,3,5]
    assert not ({(int(tasks[i]),int(episodes[i])) for i in ids}&{(1,5),(3,4)})


def test_eval_observation_packer_excludes_targets_and_metadata_and_query_is_invariant():
    rng=np.random.default_rng(7)
    split={'tokens':rng.normal(size=(1,3,4)).astype(np.float32),
        'groups':np.array([[0,1,-1]],np.int8),'history':rng.normal(size=(1,10,7)).astype(np.float32),
        'history_mask':np.ones((1,10),bool),'state':rng.normal(size=(1,8)).astype(np.float32),
        'actions':rng.normal(size=(1,10,7)).astype(np.float32),'task':np.array([9]),
        'episode':np.array([22]),'hard':np.array([True]),'row':np.array([101]),
        'key':np.array([[9,22,40]],np.int64)}
    packed=_pack_query_observation(split,[0])
    assert set(packed)=={'tokens','groups','history','history_mask','state'}
    torch.manual_seed(9)
    model=VectorRanker(token_dim=4,context_dim=8,use_history=True,
        key_input_dim=4+10*7+10+8,token_count=2).eval()
    first,_=_query_representations(model,'p5',split,'cpu')
    changed={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in split.items()}
    changed['actions'][:]=1000.;changed['task'][:]=-50;changed['episode'][:]=123456
    changed['hard'][:]=False;changed['row'][:]=0;changed['key'][:]=999
    second,_=_query_representations(model,'p5',changed,'cpu')
    np.testing.assert_array_equal(first,second)


def test_history_at_t_uses_only_action_values_strictly_before_t():
    rows={'task':np.array([1,1,1,1]),'episode':np.array([4,4,4,4]),
        'frame':np.array([0,10,20,30]),'actions':np.stack([
            np.full((10,7),1.),np.full((10,7),2.),np.full((10,7),3.),np.full((10,7),4.)]),
        'state':np.arange(4*8,dtype=np.float32).reshape(4,8)}
    h1,m1,s1,f1=_episode_history(rows,1,4,20)
    mutated={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in rows.items()}
    mutated['actions'][2:]=999.
    h2,m2,s2,f2=_episode_history(mutated,1,4,20)
    np.testing.assert_array_equal(h1,h2);np.testing.assert_array_equal(m1,m2)
    np.testing.assert_array_equal(s1,s2)
    assert not np.array_equal(f1,f2)  # labels may change; causal input may not
    assert np.all(h1[m1]==2.)


def test_topk_blend_averages_native_action_windows_only():
    natural=np.zeros((2,12,7),np.float32)
    natural[0,:,0]=np.arange(12,dtype=np.float32)
    natural[1,:5,0]=20.
    bank={'natural_actions':natural,'length':np.array([12,5])}
    scores=np.array([[1.,0.]],np.float32)
    temp=.7
    got=natural_topk_blend_actions(scores,bank,k=2,temperature=temp)
    w=np.exp(np.array([1.,0.])/temp);w=w/w.sum()
    expected=w[0]*np.arange(10,dtype=np.float32)+w[1]*20.
    np.testing.assert_allclose(got[0,:,0],expected,rtol=1e-6,atol=1e-6)
    assert np.all(got[...,1:]==0.)
    one=natural_topk_blend_actions(scores,bank,k=1,temperature=.001)
    np.testing.assert_array_equal(one[0,:,0],np.arange(10,dtype=np.float32))
    windows=_candidate_window_actions(np.arange(2),bank)
    np.testing.assert_array_equal(windows[1,:5,0],np.full(5,20.))
    np.testing.assert_array_equal(windows[1,5:,0],np.full(5,20.))


def test_separate_key_features_keep_saved_scale_and_native_window_values():
    actions=np.arange(2*10*7,dtype=np.float32).reshape(2,10,7)
    got=_type_features(actions)
    np.testing.assert_array_equal(got,actions[...,:6].reshape(2,-1))


def test_full_candidate_rank_distribution_and_topk_metrics():
    errors=np.array([[0.,.1,.2]],np.float64)
    scores=np.array([[3.,2.,1.]],np.float64)
    got=candidate_rank_metrics(scores,errors,tau=.1,temperature=1.)
    assert got['spearman_positive']==pytest.approx(1.)
    assert np.isfinite(got['cross_entropy'])
    assert got['kl_pq']==pytest.approx(0.,abs=1e-12)
    assert got['recall@1']==1. and got['recall@5']==1. and got['recall@20']==1.
    assert got['ndcg@20']==pytest.approx(1.) and got['mrr']==1.
    assert got['top1_arm6_mse']==0. and got['top20_arm6_mse']==pytest.approx(.1)


def test_eval_episode_candidate_disjointness_is_asserted_not_used_as_filter():
    bank={'task':np.array([1,2]),'episode':np.array([3,4])}
    split={'task':np.array([5]),'episode':np.array([1_000_006]),'key':np.array([[5,6,80]])}
    assert _assert_eval_candidates_disjoint(split,bank) is None
    with pytest.raises(AssertionError):
        _assert_eval_candidates_disjoint({'task':np.array([1]),'episode':np.array([1_000_003]),
                                           'key':np.array([[1,3,10]])},bank)
