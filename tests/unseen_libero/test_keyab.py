import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

ROOT=Path('/work/skill_moe/skillnet/examples/memory_pilot/unseen_libero')
sys.path.insert(0,str(ROOT))
from keyab_core import KeyABEncoder, pairwise_resampled_mse
from keyab import raw_teacher_features, teacher_tau_from_memory


def test_padding_contents_do_not_affect_key():
    torch.manual_seed(1)
    model=KeyABEncoder(position='absolute',self_attention=True).eval()
    a=torch.randn(2,12,6);mask=torch.arange(12)[None,:]<torch.tensor([[5],[8]])
    b=a.clone();b[~mask]=1e4
    with torch.no_grad():x=model(a,mask);y=model(b,mask)
    torch.testing.assert_close(x,y,rtol=0,atol=0)


def test_position_and_self_attention_are_independent_switches():
    a=KeyABEncoder(position='absolute',self_attention=False)
    b=KeyABEncoder(position='none',self_attention=False)
    c=KeyABEncoder(position='absolute',self_attention=True)
    assert a.position=='absolute' and a.input_self is None
    assert b.position=='none' and b.input_self is None
    assert c.position=='absolute' and c.input_self is not None
    x=torch.randn(3,7,6);m=torch.ones(3,7,dtype=torch.bool)
    assert a(x,m).shape==(3,256)


def test_whole_unit_teacher_resamples_both_sides_to_equal_length():
    q=torch.arange(2*5*6,dtype=torch.float32).reshape(2,5,6)
    c=q.clone();e=pairwise_resampled_mse(q,c,16)
    assert e.shape==(2,2)
    torch.testing.assert_close(e.diag(),torch.zeros(2,dtype=e.dtype),atol=1e-5,rtol=0)
    assert torch.all(e>=0)


def test_source_manifest_shape_contract():
    source=np.array([[1,2,3,7],[1,2,7,11]])
    meta=np.array([[1,2,3,4,0],[1,2,7,4,1]])
    rec=np.column_stack([meta[:,0],meta[:,1],meta[:,2],meta[:,2]+meta[:,3]])
    np.testing.assert_array_equal(source,rec)


def test_raw_dct_teacher_normalizes_unit_length_and_retains_low_coefficients():
    raw=np.zeros((2,12,6),np.float32)
    raw[0,:4]=2.;raw[1,:8]=2.
    feat=raw_teacher_features(raw,np.array([4,8]),'raw_dct3')
    assert feat.shape==(2,18)
    np.testing.assert_allclose(feat[0],feat[1],atol=1e-6,rtol=0)


def test_raw_window_teacher_holds_last_value_for_short_units():
    raw=np.zeros((1,12,6),np.float32)
    raw[0,:3]=np.array([[1.],[2.],[3.]],np.float32)
    feat=raw_teacher_features(raw,np.array([3]),'raw_window10').reshape(10,6)
    np.testing.assert_array_equal(feat[:,0],np.array([1,2,3,3,3,3,3,3,3,3],np.float32))


def test_teacher_tau_uses_positive_different_episode_pair_distribution():
    x=np.array([[0.,0.],[1.,0.],[0.,2.],[3.,0.]],np.float32)
    tasks=np.array([0,0,1,1]);episodes=np.array([0,1,0,1])
    a=teacher_tau_from_memory(x,tasks,episodes,seed=5,pairs=5000)
    b=teacher_tau_from_memory(x,tasks,episodes,seed=5,pairs=5000)
    assert a['tau_values']==b['tau_values']
    assert a['tau_values'][0]>0 and a['tau_values'][1]>=a['tau_values'][0]
    assert a['quantiles']==[.25,.5]

