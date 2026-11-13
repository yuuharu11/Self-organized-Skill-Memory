import numpy as np
import pytest
import torch

from skill_moe.skillnet.examples.memory_pilot.unseen_libero import seg_context_retrieval as sc


def _trace(n=48):
    a=np.zeros((n,7),np.float32)
    a[:,0]=np.linspace(0,1,n);a[:,6]=-1
    a[20:,6]=1
    return a


def test_causal_segmenter_uses_only_prefix_and_returns_completed_units():
    a=_trace(60)
    states=np.zeros((len(a),8),np.float32)
    before=sc.causal_history_at(a,states,35,min_length=10)
    altered=a.copy();altered[35:,0]+=100;altered[35:,6]*=-1
    after=sc.causal_history_at(altered,states,35,min_length=10)
    assert before[2]==after[2] and np.array_equal(before[0],after[0])
    assert before and before[2][1] == 20
    assert all(e-s>=10 for s,e in sc.causal_completed_segments(a[:35],min_length=10))


def test_segment_bank_has_data_determined_count_and_roundtrips_shape():
    a=_trace(60)
    bank=sc.encode_segment_bank({(3,4):a},min_length=10)
    assert len(bank['length'])>=2
    assert bank['coefficients'].shape[1:]==(3,6)
    decoded=sc.segment_value_decode(bank['coefficients'][0],bank['magnitude'][0],
                                    int(bank['length'][0]),out_length=10)
    assert decoded.shape==(10,6) and np.isfinite(decoded).all()


def test_memory_builder_filters_holdout_and_emits_absolute_segment_frames():
    trace=_trace(40)
    rows={'task':np.array([0,0,0,0,2,2,2,2]),'episode':np.array([1]*4+[2]*4),
          'frame':np.array([0,10,20,30]*2),'actions':np.stack([trace[i:i+10] for i in (0,10,20,30)]*2).astype(np.float32)}
    bank=sc.build_memory_segments(rows,np.arange(8),min_length=10)
    assert len(bank['length'])>=1 and set(bank['task'].tolist())=={0}
    assert np.all(bank['source'][:,0]==0) and np.all(bank['source'][:,2]>=0)


def test_same_encoder_is_used_for_query_and_memory_and_mask_excludes_same_episode():
    torch.set_num_threads(2)
    net=sc.SharedContextEncoder(sc.SegmentConfig(token_dim=12,width=16,heads=4))
    tokens=torch.randn(2,5,12);mask=torch.ones(2,5,dtype=torch.bool)
    ha=torch.randn(2,3,7);hs=torch.randn(2,3,8);state=torch.randn(2,8)
    q=net(tokens,mask,ha,hs,state);k=net(tokens,mask,ha,hs,state)
    assert q.shape==(2,16) and torch.allclose(q,k)
    allowed=sc.episode_exclusion_mask([1],[10],[1,2],[10,11],same_task=False)
    assert allowed.tolist()==[[False,True]]


def test_gru_padding_is_masked_not_learned_as_zero_valued_history():
    torch.set_num_threads(2)
    net=sc.SharedContextEncoder(sc.SegmentConfig(token_dim=6,width=8,heads=2)).eval()
    tok=torch.randn(1,4,6);tm=torch.ones(1,4,dtype=torch.bool);state=torch.randn(1,8)
    a=np.ones((3,7),np.float32);s=np.ones((3,8),np.float32)
    a5,s5,m5=sc.pack_histories([(a,s)],max_length=5)
    a10,s10,m10=sc.pack_histories([(a,s)],max_length=10)
    with torch.no_grad():
        x=net(tok,tm,torch.tensor(a5),torch.tensor(s5),state,torch.tensor(m5))
        y=net(tok,tm,torch.tensor(a10),torch.tensor(s10),state,torch.tensor(m10))
    assert torch.allclose(x,y,atol=1e-6)


def test_task_and_loop_unit_exclusions_are_independent_options():
    m=sc.episode_exclusion_mask([1],[3],[1,2,2],[4,4,5],same_task=True,
                                previous_unit=[7],memory_unit=[7,8,7],exclude_previous_unit=True)
    assert m.tolist()==[[False,True,False]]
    n=sc.episode_exclusion_mask([1],[3],[1,2],[4,4],same_task=False)
    assert n.tolist()==[[True,True]]


def test_history_corruption_respects_configured_mixture():
    torch.set_num_threads(2)
    g=torch.Generator().manual_seed(44)
    a=torch.ones(4000,3,7);s=torch.ones(4000,3,8)
    _,_,m=sc.corrupt_history(a,s,generator=g)
    freq=torch.bincount(m,minlength=4).float()/len(m)
    assert torch.allclose(freq,torch.tensor([.25,.25,.15,.35]),atol=.025)
    assert (m==0).any() and (m==1).any() and (m==2).any() and (m==3).any()


def test_pose_scale_fit_recovers_synthetic_rotation_and_magnitude():
    from scipy.spatial.transform import Rotation
    rng=np.random.default_rng(8)
    shape=np.zeros((12,6),np.float32)
    for j in range(6):shape[2*j,j]=.1;shape[2*j+1,j]=-.1
    expected_rotation=Rotation.from_euler('xyz',[.12,-.08,.05]).as_matrix()
    target=np.concatenate([shape[:,:3]@expected_rotation.T,
                           shape[:,3:6]@expected_rotation.T],axis=1)*1.7
    theta=sc.fit_pose_scale(shape,target)
    assert theta[3]==pytest.approx(np.log(1.7),abs=1e-4)
    recovered=np.concatenate([shape[:,:3]@Rotation.from_rotvec(theta[:3]).as_matrix().T,
                              shape[:,3:6]@Rotation.from_rotvec(theta[:3]).as_matrix().T],axis=1)*np.exp(theta[3])
    assert np.allclose(recovered,target,atol=5e-4)


def test_pose_head_initializes_at_identity_and_loss_is_finite():
    head=sc.PoseScaleHead(width=16,coeff_dim=18)
    q=torch.randn(2,16);k=torch.randn(2,16);c=torch.randn(2,3,6)
    pred=head(q,k,c)
    assert torch.equal(pred,torch.zeros_like(pred))
    loss=sc.pose_scale_loss(pred,torch.tensor([[.1,0.,0.,.2],[0.,.1,0.,-.1]]))
    assert torch.isfinite(loss) and loss>0


def test_boundary_blend_and_continuity_term():
    source=np.ones((5,7),np.float32);previous=np.zeros(7,np.float32)
    out=sc.blend_boundary(source,previous,steps=2)
    assert np.allclose(out[:2,0],[1/3,2/3])
    assert np.allclose(out[2:],1)
    bonus=sc.continuity_bonus(torch.tensor([[1.,0.],[2.,0.]]),torch.tensor([0.,0.]))
    assert bonus[0]>bonus[1]


def test_fallback_until_a_completed_historical_segment_and_online_offline_draft_match():
    # Initial online calls use pi05; synthetic shared decoder is then used by both paths.
    class DummyEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__();self.p=torch.nn.Parameter(torch.zeros(()))
    calls=[]
    policy=sc.SegmentContextPolicy(DummyEncoder(),{},prefix_fn=lambda obs:(np.zeros((2,3)),np.ones(2)),
        pi05_fn=lambda obs:(calls.append('fallback') or np.zeros((1,7),np.float32)),
        decode_fn=lambda q,i,b:np.ones((10,7),np.float32),config=sc.SegmentConfig(token_dim=3,width=4,heads=1))
    act,meta=policy.act(None,np.zeros(8),task_id=1,episode_id=2)
    assert meta['fallback'] and calls==['fallback'] and act.shape==(1,7)

    c=np.arange(18,dtype=np.float32).reshape(3,6)*.01;m=.7;length=18
    pose=np.array([.02,-.01,.03,np.log(1.2)],np.float32);grip=np.linspace(-1,1,10,dtype=np.float32)
    offline=sc.offline_draft(c,m,length,pose,grip_actions=grip)[0]
    online=sc.draft_one_unit(c,m,length,pose,grip)
    assert np.array_equal(offline,online)

    class FixedPose(torch.nn.Module):
        def forward(self,q,k,coeff):return torch.tensor([[.02,-.01,.03,np.log(1.2)]],device=q.device)
    encoder=sc.SharedContextEncoder(sc.SegmentConfig(token_dim=3,width=4,heads=1))
    bank={'keys':torch.ones(1,4),'episode':np.array([99]),'task':np.array([99]),'unit':np.array([3]),
          'coefficients':c[None],'magnitude':np.array([m]),'length':np.array([length])}
    policy=sc.SegmentContextPolicy(encoder,bank,prefix_fn=lambda obs:(np.zeros((2,3)),np.ones(2)),
        pi05_fn=lambda obs:np.column_stack([np.zeros((10,6),np.float32),grip]),
        config=sc.SegmentConfig(token_dim=3,width=4,heads=1),pose_head=FixedPose())
    executed=_trace(21);states=np.zeros((21,8),np.float32)
    policy.observe_executed(executed,states)
    server_action,meta=policy.act(None,np.zeros(8),task_id=1,episode_id=2)
    expected=sc.draft_one_unit(c,m,length,pose,grip)
    assert not meta['fallback'] and meta['unit_id']==3
    assert np.allclose(server_action,expected,atol=1e-6)


def test_retrieval_scores_mask_same_episode_and_same_task():
    q=torch.tensor([[1.,0.]]);mem=torch.tensor([[.9,.1],[.8,.2],[0.,1.]])
    score,mask=sc.retrieval_scores(q,mem,query_episode=[1],memory_episode=[1,2,3],
        query_task=[1],memory_task=[1,1,2],exclude_same_task=True)
    assert mask.tolist()==[[False,False,True]]
    assert int(sc.select_memory(score,mask)[0])==2


def test_training_objective_runs_with_shared_encoder_and_pose_supervision():
    torch.set_num_threads(2)
    model=sc.SegmentContextModel(sc.SegmentConfig(token_dim=12,width=16,heads=4))
    b=3
    batch={'query_tokens':torch.randn(b,5,12),'query_token_mask':torch.ones(b,5,dtype=torch.bool),
        'memory_tokens':torch.randn(b,5,12),'memory_token_mask':torch.ones(b,5,dtype=torch.bool),
        'history_actions':torch.randn(b,4,7),'history_states':torch.randn(b,4,8),
        'memory_history_actions':torch.randn(b,4,7),'memory_history_states':torch.randn(b,4,8),
        'query_state':torch.randn(b,8),'memory_state':torch.randn(b,8),
        'query_episode':torch.arange(b),'memory_episode':torch.arange(b)+10,
        'query_task':torch.arange(b),'memory_task':torch.arange(b)+10,
        'type_error':torch.rand(b,b),'action_distance':torch.rand(b,b),
        'positive_memory_key':torch.randn(b,16),'positive_coefficients':torch.randn(b,3,6),
        'pose_target':torch.zeros(b,4)}
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3)
    result=sc.training_objective(model.encoder,model.pose_head,batch,generator=torch.Generator().manual_seed(4))
    result['loss'].backward();opt.step()
    assert result['loss'].isfinite() and result['query_keys'].shape==(b,16)


def test_selection_is_dev_only():
    choose=lambda c:{'type_error':float(c),'spearman':.5}
    assert sc.select_dev_configuration([.2,.1],choose,split_name='dev')==.1
    with pytest.raises(ValueError,match='Dev'):
        sc.select_dev_configuration([.1],choose,split_name='eval')


def test_resumable_extractor_and_cache_audit(tmp_path):
    from skill_moe.skillnet.examples.memory_pilot.unseen_libero import seg_context_extract as ex
    wanted=ex.required_query_keys([1,1],[2,2],[0,10],stride=5,segment_starts=[(1,2,7)])
    assert wanted==[(1,2,0),(1,2,5),(1,2,7),(1,2,10)]
    out=tmp_path/'features'
    def observe(key):return np.asarray(key)
    def encode(batch):
        return np.zeros((len(batch),2,3),np.float32),np.zeros((len(batch),2),np.int8)
    first=ex.extract_prefix_memmap(wanted,observe,encode,out,token_shape=(2,3),batch_size=2)
    second=ex.extract_prefix_memmap(wanted,observe,encode,out,token_shape=(2,3),batch_size=2)
    assert first==second and first['rows']==4
    assert np.load(out/'tokens.npy').shape==(4,2,3)


def test_prefix_stride_and_eval_mse_axes():
    assert sc.stride_query_frames(21,stride=5).tolist()==[0,5,10,15,20]
    scores=np.array([[.9,.1],[.1,.9]])
    type_error=np.array([[.01,.2],[.2,.01]])
    action_error=np.ones((2,10,7),np.float32);action_error[0,:5]=0;action_error[1,5:]=0
    out=sc.evaluate_retrieval(scores,type_error,action_error,row_episode=np.array([1,2]),hard=[True,False])
    assert out['all']['recall_at_1']==1
    assert out['all']['mse_first5_x6_7']==pytest.approx(3/7)
    assert out['hard']['rows']==1


def test_heldout_and_memory_rows_are_removed_from_fit_mask():
    task=np.array([0,2,7,37,39]);memory=np.array([False,False,False,False,True])
    assert sc.training_row_mask(task,memory).tolist()==[True,False,False,False,False]
    assert sc.memory_row_mask(task,memory).tolist()==[False,False,False,False,True]
    assert sc.prefix_token_mask(np.array([-1,0,1,2])).tolist()==[False,True,True,True]


def test_s_shared_pair_error_and_action_soft_targets():
    q=np.zeros((1,10,6),np.float32);m=np.ones((2,10,6),np.float32)
    d=sc.s_shared_type_error(q,m,normalized_to_physical=lambda x:x)
    assert d.shape==(1,2) and np.isfinite(d).all()
    action=sc.future_action_mse(np.zeros((1,10,7)),np.ones((2,10,7)))
    target=sc.soft_action_targets(torch.tensor(d,dtype=torch.float32),torch.tensor(action),
        torch.ones((1,2),dtype=torch.bool),tau_type=.03,tau_action=.03)
    assert target.shape==(1,2) and torch.allclose(target.sum(-1),torch.ones(1))
