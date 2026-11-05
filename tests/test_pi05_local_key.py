import importlib.util
from pathlib import Path

import numpy as np

SCRIPT=Path(__file__).resolve().parents[1] / 'skill_moe/skillnet/examples/memory_pilot/pi05_memory/local_key/run.py'
SPEC=importlib.util.spec_from_file_location('pi05_local_key_testmod',SCRIPT)
mod=importlib.util.module_from_spec(SPEC);assert SPEC.loader is not None;SPEC.loader.exec_module(mod)


def test_project_and_patch_grid_mapping():
    t=np.eye(4,dtype=np.float32)
    assert np.allclose(mod.project_world_to_pixel([128.,128.,1.],t),[111.125,111.125])
    ix=mod.patch_ids_for_point([112.,112.],3)
    assert len(ix)==9 and ix.tolist()==[119,120,121,135,136,137,151,152,153]
    assert len(mod.patch_ids_for_point([0.,0.],5))==9


def test_state_standardization_is_fit_on_training_only():
    train=np.asarray([[0.,2.],[2.,4.]],np.float32);other=np.asarray([[1.,3.],[100.,100.]],np.float32)
    tr,ot,spec=mod._standardize(train,other)
    assert np.allclose(tr.mean(0),0.,atol=1e-6)
    assert np.allclose(spec['mean'],[1.,3.])
    assert ot[0].tolist()==[0.,0.] and ot[1,0]>90


def test_pairwise_and_candidate_action_mse():
    q=np.zeros((2,10,7),np.float32);m=np.ones((3,10,7),np.float32)
    assert np.allclose(mod._row_mse(q,m),1.)
    c=np.stack([np.zeros((2,10,7)),np.ones((2,10,7))],axis=1).astype(np.float32)
    assert np.allclose(mod._candidate_mse(q,c),[[0.,1.],[0.,1.]])
    m=mod._metrics('test',np.zeros((2,64),np.int64),np.ones((2,64),np.float32),
                   np.ones((2,64),np.float32),None,np.asarray([0,1]),{'episode':np.asarray([3,4])})
    assert np.isnan(m['recall8']).all()


def test_recall8_counts_only_first_eight_ranked_candidates():
    cand=np.tile(np.arange(64,dtype=np.int64),(1,1))
    scores=np.arange(64,dtype=np.float32)[None,:]
    # Relevant items exist only at ranks 9..12. They belong in the denominator
    # but cannot count as retrieved by Recall@8.
    candidate_mse=np.ones((1,64),np.float32)
    candidate_mse[0,8:12]=0.
    all_mse=np.zeros((1,4),np.float32)
    got=mod._metrics('recall',cand,scores,candidate_mse,all_mse,np.asarray([100]),{'episode':np.asarray([7])})
    assert got['query_positive']==[True]
    assert got['recall8']==[0.]


def test_retrieve_maps_pool_position_back_to_dataset_row_id():
    # Memory pool order is [row 4, row 1], deliberately different from dataset order.
    rows={'task':np.asarray([0,1,9,2,3]),'episode':np.asarray([0,1,9,2,3])}
    actions=np.zeros((5,10,7),np.float32)
    actions[4]=.25
    actions[1]=1.
    actions[0]=-.75  # the old bug accidentally compared against this dataset row
    qfeat=np.asarray([[1.,0.]],np.float32)
    mfeat=np.asarray([[1.,0.],[0.,1.]],np.float32)
    qids=np.asarray([2]);mids=np.asarray([4,1])
    cand,score,mse=mod._retrieve(qfeat,mfeat,qids,mids,rows,actions,k=2,exclusion='task')
    assert cand.tolist()==[[4,1]]
    assert np.allclose(mse[0,0],mod._row_mse(actions[[2]],actions[[4]])[0,0])
    assert not np.isclose(mse[0,0],mod._row_mse(actions[[2]],actions[[0]])[0,0])


def test_key_blocks_are_independently_normalized_and_weighted():
    v=np.asarray([[3.,4.]],np.float32);s=np.asarray([[0.,2.]],np.float32)
    h=np.asarray([[1.,0.]],np.float32)
    out=mod._join_key_blocks(v,s,h,(.5,.25,.25))
    assert out.shape==(1,6) and np.isclose(np.linalg.norm(out),1.)
    assert np.allclose(out[0,:2],np.asarray([.6,.8],np.float32)*np.sqrt(.5))
    assert np.allclose(out[0,2:4],np.asarray([0.,1.],np.float32)*.5)
    assert np.allclose(out[0,4:6],np.asarray([1.,0.],np.float32)*.5)


def test_episode_bootstrap_is_paired_and_reproducible():
    d=np.asarray([-1.,-2.,1.,2.]);ep=np.asarray([10,10,20,20])
    a=mod._episode_bootstrap(d,ep,reps=100);b=mod._episode_bootstrap(d,ep,reps=100)
    assert a==b and a['episodes']==2 and a['rows']==4


def test_long_leakage_assertion_uses_task_ids():
    rows={'task':np.asarray([10,11,0,1])}
    mod._assert_no_long(np.asarray([0,1]),np.asarray([1]),rows)
    try:mod._assert_no_long(np.asarray([0,2]),np.asarray([1]),rows)
    except AssertionError:pass
    else:raise AssertionError('Long row must be rejected from fitting')


def test_direct_mlp_small_cpu_fit_and_prediction():
    rng=np.random.default_rng(3);x=rng.normal(size=(24,6)).astype(np.float32);y=np.repeat(x[:,:1,None],70,axis=2).reshape(24,10,7)
    model,meta=mod._fit_direct(x[:18],y[:18],x[18:],y[18:],threads=1)
    pred=mod._predict_direct(model,x[18:],meta['mean'],meta['std'])
    assert pred.shape==(6,10,7) and np.isfinite(pred).all() and meta['epochs']>0


def test_contrastive_projection_uses_cross_task_action_positives():
    rng=np.random.default_rng(9);x=rng.normal(size=(32,12)).astype(np.float32)
    a=np.zeros((32,10,7),np.float32);a[16:]=.01
    task=np.repeat(np.arange(4),8);devx=rng.normal(size=(12,12)).astype(np.float32)
    deva=np.zeros((12,10,7),np.float32);devtask=np.repeat(np.arange(3),4)
    model,meta=mod._fit_projection(x,a,task,devx,deva,devtask,threads=1)
    out=mod._project_features(model,x,meta['feature_mean'],meta['feature_std'])
    assert out.shape==(32,128) and np.isfinite(out).all() and meta['tau'] in (.005,.01,.02)
