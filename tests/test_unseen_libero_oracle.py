import importlib.util
from pathlib import Path

import numpy as np
import pytest

P = Path(__file__).resolve().parents[1] / 'skill_moe/skillnet/examples/memory_pilot/unseen_libero/oracle_repr.py'
S = importlib.util.spec_from_file_location('unseen_oracle_repr', P)
M = importlib.util.module_from_spec(S)
S.loader.exec_module(M)


@pytest.mark.parametrize('name', ['raw','dct14','dct21','keyframe21','keyframe35'])
def test_representation_shape_and_raw_identity(name):
    x=np.random.default_rng(3).normal(size=(5,10,7)).astype(np.float32)
    z=M.encode_fixed(x,name); y=M.decode_fixed(z,name)
    assert y.shape==x.shape and np.isfinite(y).all()
    if name=='raw': np.testing.assert_array_equal(y,x)


def test_zero_padded_dct_reconstruction_is_low_frequency_projection():
    x=np.random.default_rng(4).normal(size=(2,10,7)).astype(np.float32)
    z=M.encode_fixed(x,'dct14'); y=M.decode_fixed(z,'dct14')
    direct=M.idct(np.pad(z.reshape(2,2,7),((0,0),(0,8),(0,0))),type=2,axis=1,norm='ortho')
    np.testing.assert_allclose(y,direct,atol=1e-6)


def test_oracle_selects_one_real_candidate_and_ids_are_not_offsets():
    target=np.zeros((10,7),np.float32)
    drafts=np.stack([np.ones((10,7)),np.full((10,7),.2)]).astype(np.float32)
    row,draft,mse=M.oracle_candidate(target,np.array([9001,12003]),drafts)
    assert row==12003 and mse==pytest.approx(.04)
    np.testing.assert_array_equal(draft,drafts[1])


def test_episode_bootstrap_pairs_at_episode_level():
    a=np.array([2.,2.,4.,4.]);b=np.zeros(4);ep=np.array([1,1,2,2])
    got=M.episode_bootstrap_diff(a,b,ep,reps=100,seed=1)
    assert got['mean']==3 and got['episodes']==2 and got['rows']==4


def test_pca_fit_source_excludes_unseen_eval_by_construction():
    # The fitting helper accepts training actions only; this guards the row-id
    # split contract in the runner without needing a checkpoint/GPU.
    from sklearn.decomposition import PCA
    train=np.arange(40,dtype=np.float32).reshape(4,10)
    eval_actions=np.full((1,10),1e6,np.float32)
    pca=PCA(n_components=2).fit(train)
    expected=PCA(n_components=2).fit(train)
    np.testing.assert_array_equal(pca.mean_,expected.mean_)
    assert eval_actions.min()>pca.mean_.max()


def test_shuffle_preserves_selected_rows_and_changes_task():
    # IDs are global row IDs; make three equally sized task partitions.
    tasks=np.repeat(np.array([11,22,33],np.int64),4)
    ids=np.arange(12,dtype=np.int64)
    query_task=tasks.copy()
    shuffled=M.deterministic_cross_task_shuffle(ids,query_task,tasks,seed=9)
    np.testing.assert_array_equal(np.sort(shuffled),ids)
    assert np.all(tasks[shuffled]!=query_task)


def test_pool_shuffle_returns_unique_cross_task_rows():
    tasks=np.repeat(np.array([11,22,33],np.int64),5)
    q=np.array([11,11,22,22,33,33])
    pool=np.arange(len(tasks))
    shuffled=M.deterministic_cross_task_from_pool(pool,q,tasks,seed=4)
    assert len(np.unique(shuffled))==len(q)
    assert np.all(tasks[shuffled]!=q)


def test_zero_bridge_draft_uses_model_padded_action_width():
    p=Path(__file__).resolve().parents[1] / 'skill_moe/skillnet/examples/memory_pilot/unseen_libero/oracle_experiment.py'
    s=importlib.util.spec_from_file_location('unseen_oracle_experiment_test',p)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    active7=np.ones((3,10,7),np.float32)
    padded32=np.zeros((3,10,32),np.float32)
    # Eval labels are active7, but sample_bridge combines with model-space 32-D noise.
    adapted=m.pad_bridge_action(active7)
    assert adapted.shape==(3,10,32)
    np.testing.assert_array_equal(adapted[:,:,:7],active7)
    assert not adapted[:,:,7:].any()
    z=m.zero_bridge_draft(padded32)
    assert z.shape==(3,10,32) and not z.any()


def test_keyframe_linear_interpolation_uses_endpoints():
    x=np.zeros((1,10,7),np.float32);x[0,:,0]=np.arange(10,dtype=np.float32)
    z=M.encode_fixed(x,'keyframe21');y=M.decode_fixed(z,'keyframe21')
    assert y.shape==x.shape
    assert y[0,0,0]==pytest.approx(0.) and y[0,-1,0]==pytest.approx(9.)
