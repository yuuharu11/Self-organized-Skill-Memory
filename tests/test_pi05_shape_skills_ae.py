import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch

SCRIPT = Path(__file__).resolve().parents[1] / 'skill_moe/skillnet/examples/memory_pilot/pi05_memory/shape_skills/ae.py'
spec = importlib.util.spec_from_file_location('shape_skills_ae_test', SCRIPT)
ae = importlib.util.module_from_spec(spec); sys.modules[spec.name] = ae; spec.loader.exec_module(ae)


def test_masked_mse_ignores_padding_values():
    torch.set_num_threads(1)
    y=torch.tensor([[[1.,2.],[3.,4.]]]);p=y.clone();m=torch.tensor([[True,True]])
    expected=ae.masked_mse(p,y,m)
    yp=torch.cat([y,torch.full((1,4,2),100.)],1)
    pp=torch.cat([p,torch.full((1,4,2),-50.)],1)
    mp=torch.tensor([[True,True,False,False,False,False]])
    torch.testing.assert_close(expected,ae.masked_mse(pp,yp,mp))


def test_pad_batch_uses_global_lmax_and_zero_suffix_mask():
    x=[np.ones((3,7),np.float32),np.full((5,7),2.,np.float32)]
    y=[v.copy() for v in x]
    xb,yb,phase,mask,length=ae._pad_batch(x,y,np.zeros(7),np.ones(7),np.zeros(7),np.ones(7),8,'cpu')
    assert tuple(xb.shape)==(2,8,7)
    assert mask.tolist()==[[True,True,True,False,False,False,False,False],
                           [True,True,True,True,True,False,False,False]]
    assert torch.count_nonzero(xb[~mask])==0 and torch.count_nonzero(yb[~mask])==0
    assert length.shape==(2,1) and phase.shape==(2,8)


def test_encoder_latent_is_independent_of_arbitrary_padding_values():
    torch.set_num_threads(1);torch.manual_seed(2)
    _,_,_,Model=ae._torch_classes();model=Model(latent_dim=8).eval()
    x=torch.randn(2,7,7);phase=torch.zeros(2,7);mask=torch.tensor([[1,1,1,0,0,0,0],[1,1,1,1,1,0,0]],dtype=torch.bool)
    lengths=torch.tensor([[.4],[.7]])
    x2=x.clone();x2[~mask]=987.
    with torch.no_grad():z0=model.encode(x,phase,mask,lengths);z1=model.encode(x2,phase,mask,lengths)
    torch.testing.assert_close(z0,z1)


def test_length_is_an_explicit_encoder_and_decoder_input():
    torch.set_num_threads(1);torch.manual_seed(3)
    _,_,_,Model=ae._torch_classes();model=Model(latent_dim=8).eval()
    x=torch.zeros(1,5,7);phase=torch.linspace(0,1,5)[None];mask=torch.ones(1,5,dtype=torch.bool)
    zshort=model.encode(x,phase,mask,torch.tensor([[.2]]))
    zlong=model.encode(x,phase,mask,torch.tensor([[.9]]))
    assert zshort.shape==(1,8) and not torch.allclose(zshort,zlong)
    assert model.decode(zshort,phase,torch.tensor([[.2]])).shape==(1,5,7)


def test_physical_augmentation_rotates_position_and_rotation_but_not_gripper():
    x=np.arange(70,dtype=np.float32).reshape(10,7)
    y,pose=ae.augment_physical(x,np.random.default_rng(4),scale_range=(1.2,1.2))
    np.testing.assert_allclose(y[:,6],x[:,6])
    np.testing.assert_allclose(pose['rotation'].T@pose['rotation'],np.eye(3),atol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(y[:,:3]),np.linalg.norm(x[:,:3])*1.2,rtol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(y[:,3:6]),np.linalg.norm(x[:,3:6])*1.2,rtol=1e-5)


def test_canonical_shape_keeps_gripper_as_seventh_channel():
    x=np.zeros((13,7),np.float32)
    x[:,0]=.01;x[:,3]=.02;x[:,6]=np.linspace(-1,1,13)
    ref=np.cumsum(np.tile([1.,0.,0.],(32,1)),axis=0)
    y,pose=ae.canonicalize_7(x,ref)
    assert y.shape==(13,7)
    np.testing.assert_allclose(y[:,6],x[:,6])
    assert pose['rotation'].shape==(3,3)


def test_lmax_is_ceiling_of_nonlong_p99_statistic():
    l,p=ae.choose_lmax([10,20,30,40,100],percentile=.8)
    np.testing.assert_allclose(p,52.)
    assert l==52

