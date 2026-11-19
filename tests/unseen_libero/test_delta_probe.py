import importlib.util
from pathlib import Path

import numpy as np

MODULE = Path(__file__).resolve().parents[2] / 'skill_moe/skillnet/examples/memory_pilot/unseen_libero/delta_probe.py'
spec = importlib.util.spec_from_file_location('delta_probe', MODULE)
dp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dp)


def _part(n=4, offset=0.):
    rng=np.random.default_rng(11)
    return {
        'key':np.stack([np.zeros(n,int),np.arange(n),np.zeros(n,int)],axis=1),
        'tokens':(rng.normal(size=(n,4,8))+offset).astype(np.float32),
        'groups':np.tile(np.array([[0,1,0,1]],np.int8),(n,1)),
        'history':rng.normal(size=(n,10,7)).astype(np.float32)+offset,
        'history_mask':np.ones((n,10),bool),
        'state':rng.normal(size=(n,8)).astype(np.float32)+offset,
        'actions':rng.normal(size=(n,10,7)).astype(np.float32),
    }


def test_delta_features_use_selected_start_observation_and_draft():
    q=_part(); b=_part(6,offset=2.)
    ids=np.array([4,1,5,0])
    acts=np.zeros((6,10,7),np.float32)
    parts,draft=dp.draft_features(q,b,ids,acts)
    assert parts['N'].shape[0]==4 and draft.shape==(4,10,6)
    pooledq=dp.key2._pool_tokens(q['tokens'],q['groups'])
    pooledb=dp.key2._pool_tokens(b['tokens'][ids],b['groups'][ids])
    # The delta blocks are appended after the shared current-observation+draft base.
    base_dim=8+pooledq.shape[1]+80+60
    np.testing.assert_allclose(parts['delta_state'][:,base_dim:],q['state']-b['state'][ids])
    np.testing.assert_allclose(parts['delta_vlm'][:,base_dim:],pooledq-pooledb)
    histq=np.concatenate([q['history'].reshape(4,-1),q['history_mask'].astype(np.float32)],1)
    histb=np.concatenate([b['history'][ids].reshape(4,-1),b['history_mask'][ids].astype(np.float32)],1)
    np.testing.assert_allclose(parts['delta_history'][:,base_dim:],histq-histb)
    assert parts['delta_all'].shape[1]==base_dim+8+pooledq.shape[1]+80


def test_residual_is_target_minus_single_selected_draft_arm6():
    y=np.ones((2,10,7),np.float32);d=np.full((2,10,6),.25,np.float32)
    r=dp.action_residual(y,d)
    assert r.shape==(2,10,6)
    np.testing.assert_allclose(r,.75)


def test_ridge_recovers_linear_residual_and_switch_targets():
    rng=np.random.default_rng(5);x=rng.normal(size=(80,5)).astype(np.float32)
    w=rng.normal(size=(5,3)).astype(np.float32);y=x@w
    model=dp.ridge_fit(x,y,1e-6)
    np.testing.assert_allclose(dp.ridge_predict(model,x),y,atol=1e-4)
    a=np.zeros((2,10,7),np.float32);a[0,:4,6]=-1;a[0,4:,6]=1;a[1,:,6]=1
    sw=dp._switch_labels(a)
    assert sw[0,4]==1 and not sw[1].any()

