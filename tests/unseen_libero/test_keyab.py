import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

ROOT=Path('/work/skill_moe/skillnet/examples/memory_pilot/unseen_libero')
sys.path.insert(0,str(ROOT))
from keyab_core import KeyABEncoder, pairwise_resampled_mse


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

