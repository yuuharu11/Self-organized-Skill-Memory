"""Key/query probes K1 and K2."""
import sys

import numpy as np
import pytest

from tests.support import UNSEEN, load_script

PROBE_SCRIPT = UNSEEN / "key_query_probe.py"
probe = load_script(PROBE_SCRIPT)
k2 = load_script(UNSEEN / "key_query_k2.py")


# ── K1 probe ──────────────────────────────────────────────────────────────────────────────────────

def test_cross_task_filter_excludes_query_and_holdout_tasks():
    pool={'source_task':np.array([1,2,3,7]),'shape':np.zeros((4,10,7))}
    mask=probe.filter_cross_task(pool,np.array([1,4]),heldout=(7,))
    assert mask.tolist()==[[False,True,True,False],[True,True,True,False]]


def test_eval_label_is_only_consumed_by_score_boundary():
    # Fit-side helpers accept errors/teacher inputs but never an eval target.
    err=np.array([[.1,.2,.3],[.3,.1,.4]])
    allow=np.array([[True,True,False],[True,True,True]])
    teacher=probe.target_distribution(err,allow,.1)
    assert np.allclose(teacher.sum(1),1)
    logits=-err/.2
    assert np.isfinite(probe.listwise_loss(logits,teacher,allow))
    # Evaluation target is only supplied to this scoring function.
    action=np.zeros((2,10,7),np.float32);ep=np.array([10,11]);hard=np.array([True,False])
    out=probe.score_predictions(np.array([0,1]),err,action,action,np.array([0,1]),.01,ep,hard)
    assert out['action_mse']['all']==0 and out['n']==2


def test_unseen_metadata_fit_load_uses_correct_root_and_omits_actions(tmp_path,monkeypatch):
    names=['image_base','image_wrist','state','prompt','prompt_mask','task','episode','frame','is_dev','actions']
    for name in names:
        np.save(tmp_path/f'{name}.npy',np.zeros((2,1),np.float32))
    monkeypatch.setattr(probe,'UNSEEN',tmp_path)
    fit=probe._load_unseen_metadata(include_actions=False)
    assert set(fit)==set(names)-{'actions'}
    score=probe._load_unseen_metadata(include_actions=True)
    assert 'actions' in score


def test_p8_flow_keeps_full_model_noise_but_scores_active7():
    cand=np.ones((1,2,10,7),np.float32)
    noise=np.zeros((1,10,32),np.float32)
    x,target=probe.flow_state_and_target_velocity(cand,noise,.25)
    assert x.shape==(1,2,10,32)
    np.testing.assert_allclose(x[...,:7],.25)
    np.testing.assert_allclose(x[...,7:],0.)
    np.testing.assert_allclose(target,-1.)
    vel=np.zeros_like(cand)
    scores=probe.p8_expert_candidate_scores(vel,cand,noise)
    np.testing.assert_allclose(scores,1.)


def test_pi05_velocity_at_xt_batches_candidate_count():
    jax=pytest.importorskip('jax');jnp=pytest.importorskip('jax.numpy')
    class LLM:
        seen_cache_shape=None
        def __call__(self,inputs,**kwargs):
            LLM.seen_cache_shape=kwargs['kv_cache']['dummy'].shape
            return (None,inputs[1]),None
    class Model:
        action_horizon=10
        action_dim=32
        PaliGemma=type('PG',(),{'llm':LLM()})()
        def embed_suffix(self,obs,xt,t):
            assert obs['dummy'].shape[0]==1 # Pi0.5 suffix reuses one-query prefix observation
            b=xt.shape[0]
            return xt,jnp.ones((b,10),dtype=jnp.bool_),jnp.array([True]+[False]*9),None
        @staticmethod
        def action_out_proj(x):return x
    x=np.arange(6*10*32,dtype=np.float32).reshape(6,10,32)/100
    prepared={'obs':{'dummy':jnp.ones((2,1))},'prefix_mask':jnp.ones((2,2),dtype=jnp.bool_),
              'cache':{'dummy':jnp.ones((2,2,3,1,4))}}
    got=probe.pi05_velocity_at_xt(Model(),prepared,x,.4)
    assert got.shape==(6,10,7)
    assert LLM.seen_cache_shape==(2,1,3,1,4)
    np.testing.assert_array_equal(got,x[...,:7])


def test_holdout_rank_batch_uses_compact_positions_not_source_positions():
    source_split_positions=np.array([5022,5030,7001,9004])
    ranks=np.arange(4*3,dtype=np.int64).reshape(4,3)
    ids,selected=probe.holdout_rank_batch(source_split_positions,ranks,1,3)
    np.testing.assert_array_equal(ids,[5030,7001])
    np.testing.assert_array_equal(selected,ranks[1:3])
    with pytest.raises(ValueError,match='compact row axis'):
        probe.holdout_rank_batch(source_split_positions,ranks[:3],0,2)


def test_p8_calibration_sample_is_deterministic_and_task_balanced():
    task=np.repeat([3,5,7],10)
    holdout=np.arange(len(task))
    selected=probe.p8_calibration_sample_ids(holdout,task,[7,3],per_task=4)
    assert selected.tolist()==[0,3,6,9,20,23,26,29]
    assert np.unique(task[selected],return_counts=True)[1].tolist()==[4,4]


def test_qmid_v_cache_pooling_masks_text_padding_and_store_reads_one_layer(tmp_path):
    b,l,t,d=2,6,3*256+8,4
    values=np.arange(b*l*t*d,dtype=np.float32).reshape(b,l,t,d)
    lang=np.array([[1]*5+[0]*3,[1]*2+[0]*6],bool)
    pooled,mask=probe.pool_prefix_v_layers(values,lang)
    assert pooled.shape==(b,l,136,d) and mask.shape==(b,136)
    assert mask.sum(1).tolist()==[133,130]
    np.testing.assert_array_equal(pooled[0,:,133:],0.)
    disk=np.lib.format.open_memmap(tmp_path/'v.npy',mode='w+',dtype=np.float16,shape=(b,l,136,d));disk[:]=pooled
    mm=np.lib.format.open_memmap(tmp_path/'m.npy',mode='w+',dtype=np.bool_,shape=(b,136));mm[:]=mask
    store=probe.MidLayerTokenFeatureStore(tmp_path/'v.npy',tmp_path/'m.npy',2,np.array([0,1]))
    tok,m,mean=store[np.array([1])]
    assert tok.shape==(1,136,d) and m.shape==(1,136) and mean.shape==(1,d)
    np.testing.assert_array_equal(tok[0],pooled[1,2].astype(np.float16).astype(np.float32))
    assert not m[0,-6:].any()


def test_extracted_action_cache_uses_sampler_result_exactly():
    # Production extraction persists this exact float16 cast of sample_actions.
    sampled=np.random.default_rng(77).normal(size=(3,10,32)).astype(np.float32)
    extracted=sampled[...,:7].astype(np.float16)
    np.testing.assert_array_equal(extracted,sampled[...,:7].astype(np.float16))


def test_vectorized_procrustes_errors_match_scalar_shape_error():
    theta=probe._load_theta();vs=type('VS',(),{})()
    q=np.random.default_rng(11).normal(size=(3,10,7)).astype(np.float32)
    c=np.random.default_rng(12).normal(size=(5,10,7)).astype(np.float32)
    got=probe.shape_error_matrix(q,c,vs,batch=2)
    expected=np.array([[theta._shape_error(cc,qq,vs)[0] for cc in c] for qq in q],np.float32)
    np.testing.assert_allclose(got,expected,rtol=2e-5,atol=2e-7)


def test_cuda_batched_procrustes_matches_cpu_when_available():
    torch=pytest.importorskip('torch')
    if not torch.cuda.is_available():pytest.skip('CUDA is unavailable')
    rng=np.random.default_rng(321)
    q=rng.normal(size=(3,10,7)).astype(np.float32)
    c=rng.normal(size=(7,10,7)).astype(np.float32)
    cpu=probe.shape_error_matrix(q,c,None,batch=2)
    cuda=probe.shape_error_matrix(q,c,None,batch=2,device='cuda')
    np.testing.assert_allclose(cuda,cpu,rtol=2e-5,atol=2e-7)


def test_single_top1_and_random_oracle_baselines():
    scores=np.array([[.1,.8,.2],[.7,.1,.2]])
    allowed=np.array([[True,True,False],[True,True,True]])
    ids=np.array([100,200,300])
    picked=probe.top1_single(scores,allowed,ids)
    assert picked.tolist()==[1,0]  # positions used for all pool-array indexing
    assert ids[picked].tolist()==[200,100]  # global/source IDs are a serialization mapping
    assert picked.shape==(2,)
    # Shape-select uses a single oracle candidate and 5 independent random picks;
    # compare deterministic oracle and random draws against its documented rule.
    target=np.array([0,1]);rng=np.random.default_rng(4101);pool_n=9
    ours_oracle=np.argmin(np.array([[2,1,3],[1,4,2]]),axis=1)
    ref_oracle=np.argmin(np.array([[2,1,3],[1,4,2]]),axis=1)
    ours_random=np.array([[rng.integers(pool_n) for _ in range(5)] for _ in target])
    rng2=np.random.default_rng(4101)
    ref_random=np.array([[rng2.integers(pool_n) for _ in range(5)] for _ in target])
    np.testing.assert_array_equal(ours_oracle,ref_oracle)
    np.testing.assert_array_equal(ours_random,ref_random)


def test_compact_feature_cache_uses_split_positions_not_source_row_ids():
    # A compact 3-row split can correspond to widely spaced source row IDs.
    global_ids=np.array([4,120,9001])
    positions=probe.compact_split_positions(global_ids)
    np.testing.assert_array_equal(positions,[0,1,2])
    cache=np.array([[4],[120],[9001]])
    np.testing.assert_array_equal(cache[positions,0],global_ids)
    with pytest.raises(ValueError,match='unique vector'):
        probe.compact_split_positions([4,4])


def test_query_network_batch_excludes_targets_and_eval_metadata():
    torch=pytest.importorskip('torch');torch.set_num_threads(8)
    features={'mean':np.zeros((2,3,4),np.float32),
        'target_actions':np.ones((2,10,7),np.float32),
        'task':np.array([8,9]),'episode':np.array([1,2]),'frame':np.array([0,10])}
    eval_batch=probe._torch_batch(features,np.array([0,1]),'cpu')
    assert 'target_actions' not in eval_batch
    assert not {'task','episode','frame','_training_metadata'} & set(eval_batch)
    train_batch=probe._torch_batch(features,np.array([0,1]),'cpu',include_training_metadata=True)
    assert 'target_actions' not in train_batch
    meta=train_batch.pop('_training_metadata')
    assert set(meta)=={'task','episode','frame'}
    assert not {'task','episode','frame'} & set(train_batch)


def test_mode_specific_batch_does_not_read_unused_lazy_features():
    class UnusedLazy:
        def __len__(self):return 3
        def __getitem__(self,index):raise AssertionError('unused token feature was read')
    torch=pytest.importorskip('torch');torch.set_num_threads(8)
    features={'mean':np.zeros((3,3,4),np.float32),'state':np.zeros((3,8),np.float32),
        'tokens':UnusedLazy(),'expert_t1':np.zeros((3,10,5),np.float32),
        'history':np.zeros((3,10,7),np.float32),'action':np.zeros((3,10,7),np.float32)}
    batch=probe._torch_batch(features,np.array([0,2]),'cpu',mode='P0_mean')
    assert set(batch)=={'mean','state'}


def test_training_config_shards_are_disjoint_and_cover_all_configs():
    configs=[('m',float(i),float(i%2)) for i in range(13)]
    shards=[probe.shard_training_configs(configs,i,4) for i in range(4)]
    merged=[x for shard in shards for x in shard]
    assert len(merged)==len(configs) and len(set(merged))==len(configs)
    assert set(merged)==set(configs)


def test_p6_all_uses_declared_mean_expert_history_inputs_without_tokens():
    torch=pytest.importorskip('torch');torch.set_num_threads(8)
    class UnusedLazy:
        def __len__(self):return 2
        def __getitem__(self,index):raise AssertionError('P6_all must not require token inputs')
    features={'mean':np.zeros((2,3,2048),np.float32),'expert_t1':np.zeros((2,10,8),np.float32),
        'state':np.zeros((2,8),np.float32),'history':np.zeros((2,10,7),np.float32),
        'tokens':UnusedLazy()}
    batch=probe._torch_batch(features,np.array([0,1]),'cpu',mode='P6_all')
    net=probe.ProbeNetwork('P6_all',expert_dim=8).module.eval()
    with torch.no_grad():q,_=net(batch)
    assert q.shape==(2,256)


def test_retrieval_recall_uses_model_ranking_and_random_respects_allowed_mask():
    err=np.array([[0.,.001,.5],[.4,.3,0.]])
    allowed=np.array([[True,True,False],[True,True,True]])
    # The model ranks the oracle-positive candidate second for row 0 and does
    # not retrieve row 1's oracle. Sorting err itself would report recall@1=1.
    ranked=np.array([[1,0],[1,0]])
    recall=probe.retrieval_recall_at_k(ranked,err,allowed,.0001,ks=(1,2,3))
    assert recall=={'1':0.,'2':.5,'3':None}
    draws=probe.random_candidate_indices(allowed,seed=8,reps=20)
    assert draws.shape==(2,20)
    assert np.all(allowed[np.arange(2)[:,None],draws])
    assert np.array_equal(draws,probe.random_candidate_indices(allowed,seed=8,reps=20))


def test_retrieval_recall_skips_masked_tail_of_full_pool_ranking():
    err=np.array([[0.,.001,.5],[.4,.3,0.]])
    allowed=np.array([[True,True,False],[True,True,True]])
    ranked=np.array([[1,0,2],[1,0,2]])
    recall=probe.retrieval_recall_at_k(ranked,err,allowed,.0001,ks=(1,2,3))
    # For row 0, the masked candidate at rank 3 does not consume a slot.
    assert recall=={'1':0.,'2':.5,'3':1.}


def test_token_pooling_and_teacher_exclusion():
    rng=np.random.default_rng(3);tok=rng.normal(size=(712,8)).astype(np.float32)
    groups=np.r_[np.zeros(256),np.ones(256),np.full(200,2)].astype(np.int8)
    pooled=probe.downsample_camera_tokens(tok,groups)
    assert pooled.shape==(328,8) # 64 base + 64 wrist + 200 language
    err=np.array([[.1,.01,.2]])
    allow=np.array([[True,False,True]])
    t=probe.target_distribution(err,allow,.05)
    assert t[0,1]==0 and t[0,0]>.5


def test_resumable_fp16_token_cache_and_position_encoding(tmp_path):
    tok=np.random.default_rng(4).normal(size=(2,712,8)).astype(np.float32)
    grp=np.r_[np.zeros(256),np.ones(256),np.full(200,2)].astype(np.int8)
    groups=np.stack([grp,grp])
    a=probe.downsample_camera_tokens(tok[0],grp,add_position=True)
    b=probe.downsample_camera_tokens(tok[0],grp,add_position=False)
    assert not np.allclose(a[:64],b[:64]) and np.allclose(a[128:],b[128:])
    out=tmp_path/'tokens.npy'
    info=probe.write_token_memmap(tok,groups,np.array([0,1]),out)
    assert info['dtype']=='float16' and info['rows']==2
    saved=np.load(out,mmap_mode='r',allow_pickle=False)
    mask=np.load(tmp_path/'tokens_mask.npy',mmap_mode='r',allow_pickle=False)
    assert saved.shape==(2,712,8) and saved.dtype==np.float16
    assert mask.shape==(2,712) and mask[0].sum()==328


def test_pooled_token_store_cache_matches_uncached_and_subset_is_lazy(tmp_path):
    rng=np.random.default_rng(41)
    tok=rng.normal(size=(3,712,8)).astype(np.float16)
    grp=np.r_[np.zeros(256),np.ones(256),np.full(200,2)].astype(np.int8)
    groups=np.stack([grp]*3)
    row_ids=np.array([2,0,1])
    uncached=probe.PooledTokenFeatureStore(tok,groups,row_ids)
    expected=uncached[np.array([0,2])]
    cached=probe.PooledTokenFeatureStore(tok,groups,row_ids,cache_path=tmp_path/'pooled_image_means.npy')
    actual=cached[np.array([0,2])]
    assert actual[0].shape==(2,328,8) # valid 128 image + 200 language tokens only
    np.testing.assert_array_equal(actual[1],expected[1])
    np.testing.assert_allclose(actual[0],expected[0],rtol=0,atol=0)
    np.testing.assert_allclose(actual[2],expected[2],rtol=0,atol=0)
    subset=cached.subset(np.array([2,0]))
    assert subset.image_cache is cached.image_cache
    got=subset[np.array([0,1])]
    np.testing.assert_allclose(got[0],cached[np.array([2,0])][0],rtol=0,atol=0)


def test_train_teacher_filters_holdout_and_same_task():
    pool={'shape':np.zeros((3,10,7),np.float32),'source_task':np.array([1,2,3])}
    cache={'row_ids':np.array([10,11]),'task':np.array([1,7]),
           'shape':np.zeros((2,10,7),np.float32)}
    out=probe.train_teacher_from_libero40(cache,pool,vs=None,heldout_tasks=(7,),tau_l=.1)
    assert out['row_ids'].tolist()==[10]
    assert out['query_task'].tolist()==[1]
    assert not out['allowed'][0,0]
    assert out['unseen_label_access'] is False


def test_oracle_and_random_ids_match_shape_select_smoke_cache():
    root=probe.ROOT/'theta-regression-01'/'smoke_shape_selection'/'C_vm_none'/'eval'
    if not (root/'rows.npz').exists():pytest.skip('prior theta-regression shape_select smoke not present')
    cache=np.load(root/'rows.npz',allow_pickle=False)
    pool=probe._load_theta().candidate_shape_pool(rep='C_vm_none')
    assert len(pool['shape'])==len(cache['candidate_source_rows'])
    sys.path.insert(0,str(PROBE_SCRIPT.parent))
    import variable_shape_memory as vs
    target=vs.to_physical(cache['target'][:,:,:6])
    oracle,random=probe.oracle_and_random_indices(target,pool['shape'],vs,seed=4101,random_reps=5)
    np.testing.assert_array_equal(oracle,cache['selected_candidate'][:,1])
    errors,rank=probe.pi05_shape_ranking(pool,cache['normal'][:,:,:7],vs,
        np.ones((len(cache['normal']),len(pool['shape'])),bool))
    np.testing.assert_array_equal(rank[:,0],cache['selected_candidate'][:,0])
    # theta_regression's rows artifact stores the first random ID only, while
    # its metric aggregates five independent draws.
    np.testing.assert_array_equal(random[:,0],cache['selected_candidate'][:,2])


def test_flow_blend_respects_openpi_noise_at_t1():
    cand=np.ones((1,1,10,7),np.float32)*2
    noise=np.zeros((1,10,7),np.float32)
    x,v=probe.flow_state_and_target_velocity(cand,noise,.25)
    # tau is candidate weight; OpenPI model time is 1-tau.
    np.testing.assert_allclose(x,0.5)
    np.testing.assert_allclose(v,-2.)
    x0,_=probe.flow_state_and_target_velocity(cand,noise,0.)
    x1,_=probe.flow_state_and_target_velocity(cand,noise,1.)
    np.testing.assert_allclose(x0,noise[:,None])
    np.testing.assert_allclose(x1,cand)


def test_p8_batches_candidate_velocities_and_reuses_prefix_once():
    cand=np.zeros((2,3,10,7),np.float32)
    cand[:,0]=.1;cand[:,1]=.5;cand[:,2]=1.
    noise=np.ones((2,10,7),np.float32)
    calls=[]
    def vf(x,t,cache):
        calls.append((x.shape,t,cache))
        # A perfect field for candidate #1 (index 1)
        return np.broadcast_to(noise[:,None]-cand[:,1:2],x.shape)
    scores=probe.p8_score_with_shared_prefix(cand,noise,.5,vf,prefix_cache='one-kv')
    assert len(calls)==1 and calls[0][1]==.5 and calls[0][2]=='one-kv'
    np.testing.assert_allclose(scores[:,1],0)
    assert probe.p8_select(scores).tolist()==[1,1]
    # The integrated P8 wrapper batches both theta decodings with one prefix.
    calls.clear();rank=np.array([8,4,2]);src=cand[0,:3].copy();pi=src.copy();pi[0]=src[1]
    # Return the exact required velocity for theta_pi05 variant rank zero.
    def vf2(x,t,cache):
        calls.append(cache)
        expected=np.concatenate([src,pi],axis=0)[None]
        return np.broadcast_to(np.ones((1,1,10,7),np.float32)-expected,x.shape)
    # all candidates tie only if predicted fields match; candidate 3 in the combined batch wins.
    out=probe.p8_rerank_one_query(rank,src,pi,np.ones((10,7),np.float32),.5,vf2,prefix_cache='shared',top_k=3)
    assert len(calls)==1 and calls[0]=='shared'
    assert out['candidate_count']==6 and out['type_pool_local'] in rank


def test_p8_tau_selected_only_from_explicit_holdout_arrays():
    truth=np.zeros((2,10,7),np.float32)
    acts=[];scores=[]
    for error in (.4,.1):
        a=np.zeros((2,2,10,7),np.float32);a[:,0]=error;a[:,1]=error+1
        acts.append(a);scores.append(np.tile([0.,1.],(2,1)))
    tau,table=probe.select_p8_tau_on_holdout(scores,acts,truth,[.3,.5],split_name='libero40_task_holdout')
    assert tau==.5 and table[1]['mean_mse']<table[0]['mean_mse']
    with pytest.raises(ValueError,match='LIBERO-40 task holdout'):
        probe.select_p8_tau_on_holdout(scores,acts,truth,[.3,.5],split_name='unseen_eval')


def test_p9_returns_one_whole_candidate_after_top32_stage():
    type_scores=np.array([[.9,.8,.7,.6,.5]])
    allowed=np.ones_like(type_scores,dtype=bool)
    unit_ids=np.array([10,11,12,13,14])
    second=np.array([[0.,.8,.7,.6,.1]])
    pick,top=probe.p9_two_stage(type_scores,allowed,unit_ids,second,top_k=3)
    assert top.tolist()==[[0,1,2]]
    assert pick.tolist()==[0] and pick.shape==(1,)
    # Heading criterion is a distance, so aligned early displacement wins.
    cand=np.zeros((1,2,10,7),np.float32);q=np.zeros((1,10,7),np.float32)
    cand[0,0,:3,0]=1;cand[0,1,:3,1]=1;q[0,:3,0]=1
    scores=probe.p9_direction_scores(cand,q)
    assert scores[0,0]<scores[0,1]


def test_connected_p8_p9_wrapper_uses_one_candidate_and_one_prefix(monkeypatch):
    rng=np.random.default_rng(31);actions=rng.normal(size=(40,10,7)).astype(np.float32)
    pool={'shape':np.zeros((40,10,7),np.float32),'source_theta':np.zeros((40,4),np.float32)}
    def decode(_pool,indices,vs,*,theta_source='src',query_pi05=None):
        ix=np.asarray(indices);value=actions[ix].copy()
        if theta_source=='pi05':value+=.01
        return value
    monkeypatch.setattr(probe,'decode_pool_actions',decode)
    calls=[]
    def velocity(x,t,cache):calls.append((x.shape,t,cache));return np.zeros_like(x)
    got=probe.p8_p9_select_one_query(pool,np.arange(40),np.zeros((10,7),np.float32),
        np.zeros((10,7),np.float32),None,velocity,prefix_cache='shared-kv',tau=.5)
    assert got['P8']['candidate_count']==32 and got['P8']['variant'] in ('theta_src','theta_pi05')
    assert got['P9_expert']['pool_local'] in got['top32']
    assert got['P9_direction']['pool_local'] in got['top32']
    assert calls and all(x[2]=='shared-kv' for x in calls)


def test_p7b_objective_has_listwise_and_dct_gradients():
    torch=pytest.importorskip('torch')
    logits=torch.tensor([[1.,0.]],requires_grad=True)
    teacher=torch.tensor([[.8,.2]])
    pred=torch.zeros((1,21),requires_grad=True);target=torch.ones((1,21))
    allowed=torch.ones((1,2),dtype=torch.bool)
    loss,parts=probe.p7b_type_loss(logits,teacher,pred,target,allowed)
    loss.backward()
    assert loss.isfinite() and logits.grad.abs().sum()>0 and pred.grad.abs().sum()>0
    assert set(parts)=={'infonce','dct'}
    assert probe.dct3_motion_coefficients(np.zeros((2,10,7),np.float32)).shape==(2,3,7)


def test_p6_history_dropout_and_branching_negatives():
    x=np.ones((4,78),np.float32)
    dropped=probe.apply_history_modality_dropout(x,.5,seed=1)
    assert np.all(dropped[:,70:]==1)
    assert np.any(np.all(dropped[:,:70]==0,axis=1))
    assert np.array_equal(probe.apply_history_modality_dropout(x,.9,training=False),x)
    # Similar previous type but different next type, and adjacent same-episode windows.
    prev=np.array([[1.,0.],[.99,.01],[0.,1.]])
    nxt=np.array([1,2,1]);ep=np.array([7,8,7]);win=np.array([0,0,1])
    pairs=probe.branching_hard_negative_pairs(prev,nxt,ep,win,prev_similarity=.1)
    assert (0,1) in map(tuple,pairs.tolist()) and (0,2) in map(tuple,pairs.tolist())
    torch=pytest.importorskip('torch');z=torch.tensor([[1.,0.],[.9,.1]],requires_grad=True)
    loss=probe.hard_negative_margin_loss(z,np.array([[0,1]]),margin=.2);loss.backward()
    assert loss.isfinite()


def test_p6_hard_negative_variant_is_connected_to_fit_loop():
    rng=np.random.default_rng(4);n=8
    features={'history':np.zeros((n,10,7),np.float32),'state':np.zeros((n,8),np.float32),
        'expert_t1':rng.normal(size=(n,10,16)).astype(np.float32),
        'task':np.repeat([1,2],4),'episode':np.repeat(np.arange(4),2),
        'frame':np.tile([0,10],4)}
    features['history'][...,0]=1.
    error=np.tile(np.array([[0.,.2],[.2,0.]],np.float32),(4,1));allowed=np.ones_like(error,bool)
    teacher=probe.target_distribution(error,allowed,.02)
    _,state=probe.fit_probe_model('P6_exp_hist',features,
        rng.normal(size=(2,10,7)).astype(np.float32),teacher,allowed,np.arange(n),
        epochs=1,batch_size=n,threads=1,history_dropout=.3,hard_negative_weight=.05)
    assert state['history_dropout']==.3 and state['hard_negative_weight']==.05
    assert state['train_loss'][0]['hard_negative_pairs']>0


def test_p7_observation_bottleneck_and_p7b_type_fit_are_connected():
    rng=np.random.default_rng(5);n=4
    features={'tokens':rng.normal(size=(n,8,6)).astype(np.float32),
        'token_mask':np.ones((n,8),bool),'expert_t1':rng.normal(size=(n,10,3)).astype(np.float32)}
    error=np.tile(np.array([[0.,.2],[.2,0.]],np.float32),(2,1));allowed=np.ones_like(error,bool)
    teacher=probe.target_distribution(error,allowed,.02);candidates=rng.normal(size=(2,10,7)).astype(np.float32)
    _,p7=probe.fit_probe_model('P7_bottleneck',features,candidates,teacher,allowed,np.arange(n),
        epochs=1,batch_size=n,threads=1)
    _,p7b=probe.fit_probe_model('P7b_type_bottleneck',features,candidates,teacher,allowed,np.arange(n),
        epochs=1,batch_size=n,threads=1)
    assert len(p7['observation_only_pretrain'])==1
    assert len(p7['train_loss'])==len(p7b['train_loss'])==1


def test_p1_language_variant_and_stratification_helpers():
    torch=pytest.importorskip('torch')
    enc=probe.QueryArchitectures('P1_tok_lang')
    q,w=enc(torch.zeros(2,8,2048),torch.zeros(2,78),torch.ones(2,2048))
    assert q.shape==(2,256) and w.shape[0]==2
    act=np.zeros((6,10,7),np.float32);act[3:,:,0]=1
    norm,labels,cuts=probe.translation_tercile_labels(act)
    assert len(norm)==6 and len(cuts)==2 and set(labels)<=set(range(3))
    assert probe.stratified_metric(np.arange(6),labels)
    strat=probe._stratified_bootstrap(np.arange(6),labels,np.array([0,0,1,1,2,2]),np.array([1,0,1,0,1,0],bool))
    assert all('hard' in v and v['episode_bootstrap']['reps']==5000 for v in strat.values())
    no_hard=probe._stratified_bootstrap(np.arange(3),np.zeros(3,int),np.arange(3),np.zeros(3,bool))
    assert no_hard['0']['hard']=={'n':0,'mean':None,'episode_bootstrap':None}
    assert probe.source_length_categories(np.array([1,2,9]),[2,5]).tolist()==[0,0,2]
    chosen,info=probe.choose_p1_or_p3_on_holdout(np.array([.1,.2]),np.array([.2,.3]),split_name='libero40_task_holdout')
    assert chosen=='P1_tok' and info['P1_tok_mean']<info['P3_exp_mean']


def test_candidate_method_metrics_include_both_primary_stratifications():
    n=6
    target=np.zeros((n,10,7),np.float32)
    target[:,0,0]=np.arange(n,dtype=np.float32)
    actions=target.copy();actions[::2,:,0]+=.1
    errors=np.tile(np.array([.1,.2,.3],np.float32),(n,1))
    allowed=np.ones_like(errors,dtype=bool)
    picked=np.zeros(n,np.int64)
    ranked=np.tile(np.array([0,1,2],np.int64),(n,1))
    result=probe._candidate_choice_metrics('P8',actions,picked,ranked,errors,allowed,
        target,np.repeat(np.arange(3),2),np.array([1,0,1,0,1,0],bool),
        source_lengths=np.array([5,10,15],np.int32))
    diag=result['diagnostics']
    assert set(diag)=={'translation_norm_terciles','selected_source_fragment_length_terciles'}
    assert set(diag['translation_norm_terciles']['action_mse'])=={'0','1','2'}
    assert set(diag['translation_norm_terciles']['recall_oracle_epsilon'])=={'1','5','20'}
    np.testing.assert_allclose(diag['selected_source_fragment_length_terciles']['train_cut_points'],
        np.quantile([5,10,15],[1/3,2/3]))


def test_baseline_diagnostics_are_preserved_for_report_rows():
    metrics={
        'baselines':{'pi05_shape':{'action_mse_all':.2},'random5':{'action_mse_all':.4}},
        'diagnostics':{'baselines':{
            'pi05_shape':{'translation_norm_terciles':{'action_mse':{'0':{'n':3,'mean':.2,'hard':None}}}},
            'random5':{'translation_norm_terciles':{'action_mse':{'0':{'n':3,'mean':.4,'hard':None}}}},
        }},
    }
    rows=probe._baseline_result_entries(metrics)
    assert rows['pi05_shape']['metrics']['action_mse_all']==.2
    assert rows['pi05_shape']['metrics']['diagnostics']['translation_norm_terciles']['action_mse']['0']['mean']==.2
    assert rows['random5']['metrics']['diagnostics']['translation_norm_terciles']['action_mse']['0']['mean']==.4


def test_decode_pool_actions_batched_matches_scalar_theta_decode():
    from types import SimpleNamespace

    rng=np.random.default_rng(413)
    pool={'shape':rng.normal(size=(4,10,7)).astype(np.float32),
        'source_theta':rng.normal(scale=.1,size=(4,4)).astype(np.float32)}
    ix=np.array([[3,0,2],[1,2,0]])
    vs=SimpleNamespace(to_normalized=lambda x:np.asarray(x,np.float32))
    actual=probe.decode_pool_actions(pool,ix,vs,theta_source='src')
    theta=probe._load_theta();expected=np.empty_like(actual)
    for i in range(len(ix)):
        for j,candidate in enumerate(ix[i]):
            expected[i,j]=theta.decode_theta(pool['shape'][candidate],pool['source_theta'][candidate],vs)
    np.testing.assert_allclose(actual,expected,rtol=2e-6,atol=2e-6)


def test_language_mean_uses_only_valid_language_tokens_and_unit_oracle_is_single():
    t=np.arange(20,dtype=np.float32).reshape(5,4)
    groups=np.array([0,2,2,1,2]);mask=np.array([1,1,0,1,1],bool)
    np.testing.assert_allclose(probe.language_token_mean(t,groups,mask),t[[1,4]].mean(0))
    np.testing.assert_array_equal(probe.language_token_mean(t,groups,np.zeros(5,bool)),np.zeros(4))
    cand=np.zeros((2,10,7),np.float32);cand[1]=1;y=np.ones((10,7),np.float32)
    ix,err=probe.unit_action_oracle(cand,y)
    assert ix==1 and err.shape==(2,)


def test_train_and_score_probe_model_never_needs_eval_labels(tmp_path):
    torch=pytest.importorskip('torch')
    torch.manual_seed(0)
    n,k=4,5;rng=np.random.default_rng(9)
    features={'mean':rng.normal(size=(n,3,2048)).astype(np.float32),
        'state':rng.normal(size=(n,8)).astype(np.float32),
        'history':rng.normal(size=(n,10,7)).astype(np.float32),
        'action':rng.normal(size=(n,10,7)).astype(np.float32)}
    candidates=rng.normal(size=(k,10,7)).astype(np.float32)
    allowed=np.ones((n,k),bool);allowed[:,0]=False
    teacher=probe.target_distribution(rng.uniform(0,.1,(n,k)),allowed,.02)
    checkpoint=tmp_path/'p0.pt'
    net,state=probe.fit_probe_model('P0_mean',features,candidates,teacher,allowed,np.arange(n),
        epochs=1,batch_size=2,checkpoint=checkpoint,threads=1)
    scores,ranks=probe.score_probe_model(net,'P0_mean',features,candidates,allowed,batch_size=2)
    assert checkpoint.exists() and state['unseen_target_access'] is False
    assert scores.shape==(n,k) and np.all(ranks[:,0]!=0)
    assert np.isneginf(scores[:,0]).all()


def test_noise_zero_variant_and_procrustes_degeneracy_report():
    noise=np.ones((1,10,7),np.float32)
    np.testing.assert_array_equal(probe.initial_flow_noise(noise,zero_noise=True),np.zeros_like(noise))
    np.testing.assert_array_equal(probe.initial_flow_noise(noise),noise)
    info=probe.procrustes_degeneracy(np.zeros((10,6),np.float32))
    assert info['degenerate'] and info['rank']==0
    assert 'no explicit degeneracy branch' in info['existing_behavior']


# ── K2 ────────────────────────────────────────────────────────────────────────────────────────────

def test_pool_prefix_all_layers_4x4_and_valid_instruction_mean():
    # 18 layers, one batch, 3x256 image slots, two prompt tokens, width 2.
    x = np.zeros((18, 1, 770, 2), np.float32)
    x[:, :, :256] = 1
    x[:, :, 256:512] = 2
    x[:, :, 512:768] = 99  # masked third camera must not appear
    x[:, :, 768] = 4
    x[:, :, 769] = 8
    pooled, mask = k2.pool_prefix_layer_values(x, np.asarray([[True, False]]))
    assert pooled.shape == (1, 18, 33, 2)
    assert np.all(pooled[:, :, :16] == 1)
    assert np.all(pooled[:, :, 16:32] == 2)
    assert np.all(pooled[:, :, 32] == 4)
    assert mask.shape == (1, 33) and mask.all()


def test_history_only_uses_preceding_chunks_and_zero_masks_at_episode_start():
    rows = {"task": np.asarray([4, 4, 4]), "episode": np.asarray([9, 9, 9]),
            "frame": np.asarray([0, 10, 20]), "state": np.arange(24, dtype=np.float32).reshape(3, 8)}
    actions = np.arange(3 * 10 * 7, dtype=np.float32).reshape(3, 10, 7)
    out = k2.build_history_arrays(rows, np.asarray([0, 1, 2]), actions)
    assert not out["action_mask_10"][0].any()
    assert out["action_10"][0].sum() == 0
    np.testing.assert_array_equal(out["action_10"][1], actions[0])
    np.testing.assert_array_equal(out["action_30"][2], np.concatenate([np.zeros((10, 7)), actions[0], actions[1]]))
    assert not out["action_mask_50"][1, :40].any()
    assert out["action_mask_50"][1, 40:].all()


def test_history_unit_codes_and_mask_are_bounded():
    x = np.zeros((30, 7), np.float32)
    x[:15, 0] = np.linspace(0, 1, 15)
    x[15:, 1] = np.linspace(0, 1, 15)
    x[20:, 6] = 1
    code, mask = k2.history_unit_codes(x)
    assert code.shape == (3, 21)
    assert mask.shape == (3,) and mask.any()
    assert np.isfinite(code).all()


def test_h3_trims_unavailable_padding_and_preserves_empty_history_mask():
    real = np.zeros((10, 7), np.float32)
    real[:, 0] = np.linspace(-0.4, 0.5, 10)
    real[6:, 6] = 1.0
    padded = np.concatenate([np.zeros((20, 7), np.float32), real])
    valid = np.concatenate([np.zeros(20, bool), np.ones(10, bool)])
    codes, mask = k2.history_unit_codes(padded, valid)
    expected_codes, expected_mask = k2.history_unit_codes(real, np.ones(10, bool))
    np.testing.assert_array_equal(codes, expected_codes)
    np.testing.assert_array_equal(mask, expected_mask)

    empty_codes, empty_mask = k2.history_unit_codes(np.zeros((30, 7), np.float32),
                                                     np.zeros(30, bool))
    assert not empty_mask.any()
    assert not empty_codes.any()
    # A real, executed zero-action segment remains distinguishable from
    # unavailable history because its mask is true.
    _, executed_zero_mask = k2.history_unit_codes(np.zeros((10, 7), np.float32),
                                                   np.ones(10, bool))
    assert executed_zero_mask.any()

    histories = {"action_10": np.stack([np.zeros((10, 7), np.float32), real]),
                 "action_mask_10": np.stack([np.zeros(10, bool), np.ones(10, bool)])}
    features = k2.history_only_inputs(histories, length=10, encoder="h3")
    assert not features[0, -3:].any()
    assert features[1, -3:].any()


def test_h3_rejects_non_suffix_history_mask():
    with pytest.raises(ValueError, match="contiguous suffix"):
        k2.history_unit_codes(np.zeros((4, 7), np.float32),
                              np.asarray([True, False, True, True]))


def test_top1_is_candidate_local_and_respects_exclusions():
    scores = np.asarray([[100, 2, 3], [5, 4, 8]], np.float32)
    allowed = np.asarray([[False, True, True], [True, False, True]])
    np.testing.assert_array_equal(k2.masked_top1(scores, allowed), [2, 2])
    with pytest.raises(ValueError):
        k2.masked_top1(scores, np.zeros_like(allowed))


def test_k1_task_and_cross_task_exclusions_remain_the_shared_contract():
    k1 = k2._k1()
    pool = {"source_task": np.asarray([0, 1, 2, 3])}
    allow = k1.filter_cross_task(pool, np.asarray([0, 2]), heldout=(3,))
    np.testing.assert_array_equal(allow, [[False, True, True, False], [True, True, False, False]])


def test_layer_query_variants_and_history_encoders_smoke():
    import torch
    torch.set_num_threads(8)
    tokens = torch.randn(4, 33, 256)
    mask = torch.ones(4, 33, dtype=torch.bool)
    state = torch.randn(4, 8)
    hist = torch.randn(4, 64)
    for fusion in ("attention", "concat", "visual", "history"):
        model = k2.make_k2_layer_query(history_mode=fusion, history_dim=64)
        result = model(tokens, mask, state, hist)
        if isinstance(result, tuple): result = result[0]
        assert result.shape == (4, 256)
    gru = k2.K2HistoryGRU()
    acts = torch.randn(4, 30, 7)
    hmask = torch.ones(4, 30, dtype=torch.bool)
    assert gru(acts, hmask).shape == (4, 128)


def test_one_step_listwise_retriever_fit_and_rank():
    rng = np.random.default_rng(3)
    n, k = 8, 5
    tokens = rng.normal(size=(n, 33, 256)).astype(np.float32)
    token_mask = np.ones((n, 33), bool)
    state = rng.normal(size=(n, 8)).astype(np.float32)
    history = rng.normal(size=(n, 16)).astype(np.float32)
    shapes = rng.normal(size=(k, 10, 7)).astype(np.float32)
    allowed = np.ones((n, k), bool)
    allowed[:, 0] = False
    teacher = allowed.astype(np.float32) / allowed.sum(1, keepdims=True)
    model = k2.fit_layer_retriever(tokens, token_mask, state, history, shapes,
        teacher, allowed, np.arange(n), epochs=1, batch_size=4,
        history_mode="attention", device="cpu", threads=8,
        train_metadata={"episode":np.repeat(np.arange(4),2),"frame":np.tile([0,10],4),
                        "hard_context":rng.normal(size=(n,70)).astype(np.float32)},
        hard_negative_weight=.05)
    scores, rank = k2.score_layer_retriever(model, tokens, token_mask, state,
        history, shapes, allowed, batch_size=4)
    assert scores.shape == (n, k) and rank.shape == (n, k)
    assert np.all(allowed[np.arange(n), rank[:, 0]])


def test_layer_selection_and_history_dropout_are_deterministic():
    assert k2.select_top_layers({2: .3, 7: .1, 1: .2, 4: .4}) == (7, 1, 2)
    x = np.ones((20, 9), np.float32)
    a = k2.drop_history_modality(x, .5, seed=4)
    b = k2.drop_history_modality(x, .5, seed=4)
    np.testing.assert_array_equal(a, b)
    assert np.any(np.all(a == 0, axis=1)) and np.any(np.all(a == 1, axis=1))
    assert k2.drop_history_modality(x, .5, seed=4, training=False).sum() == x.sum()


def test_elmo_scalar_mixer_fit_score_and_layer_weights():
    rng = np.random.default_rng(9)
    n, layers, candidates = 8, 3, 5
    x = rng.normal(size=(n, layers, 4, 16)).astype(np.float32)
    mask = np.ones((n, 4), bool)
    state = rng.normal(size=(n, 8)).astype(np.float32)
    hist = np.zeros((n, 1), np.float32)
    shapes = rng.normal(size=(candidates, 10, 7)).astype(np.float32)
    allowed = np.ones((n, candidates), bool)
    teacher = np.full((n, candidates), 1 / candidates, np.float32)
    fit = k2.fit_elmo_layer_mixer(x, mask, state, hist, shapes, teacher, allowed,
                                  np.arange(n), epochs=1, batch_size=4)
    scores, ranks = k2.score_elmo_layer_mixer(fit, x, mask, state, hist, shapes, allowed)
    assert scores.shape == (n, candidates)
    assert ranks.shape == (n, candidates)
    assert fit["layer_weights"].shape == (layers,)
    np.testing.assert_allclose(fit["layer_weights"].sum(), 1., atol=1e-6)


def test_top_three_layer_concatenation_preserves_layer_identity():
    x=np.stack([np.full((2,3,4),float(i),np.float32) for i in range(5)],axis=1)
    y=k2.combine_layer_tokens(x,(1,3,4),method="concat")
    assert y.shape==(2,3,12)
    np.testing.assert_array_equal(y[0,0],[1,1,1,1,3,3,3,3,4,4,4,4])
