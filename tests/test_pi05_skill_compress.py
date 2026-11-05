import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = Path('/work/skill_moe/skillnet/examples/memory_pilot/pi05_memory/skill_compress/run.py')
SPEC = importlib.util.spec_from_file_location('pi05_skill_compress_testmod', SCRIPT)
mod = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(mod)


def test_timewarp_identity_and_fit_recovery():
    x = np.linspace(-1, 1, 70, dtype=np.float32).reshape(10, 7)
    assert np.allclose(mod.timewarp(x, 1., 0.), x)
    target = mod.timewarp(x, 1.25, -1.)
    theta, pred = mod.fit_template(target, x, mode='timewarp')
    assert np.mean((pred-target)**2) < 1e-10
    assert np.isfinite(theta).all()


def test_axis_scale_least_squares_recovers_positive_scales():
    x = np.arange(70, dtype=np.float32).reshape(10,7)/70
    scales = np.asarray([.5,.8,1.1,1.2,.7,1.4,.9],np.float32)
    target=x*scales
    theta,pred=mod.fit_template(target,x,mode='axis')
    assert np.allclose(theta[2:],scales,atol=1e-5)
    assert np.allclose(pred,target,atol=1e-6)


def test_pose_rotation_identity_and_quantile_roundtrip():
    x=np.zeros((10,7),np.float32); x[:,0]=1.;x[:,3]=.1
    assert np.allclose(mod.pose_align_template(x,np.zeros(8),np.zeros(8)),x,atol=1e-6)
    q01=np.arange(7,dtype=np.float32)-4;q99=q01+2
    a=np.random.default_rng(2).normal(size=(3,10,7)).astype(np.float32)
    assert np.allclose(mod.denormalize_action(mod.normalize_action(a,q01,q99),q01,q99),a,atol=1e-5)


def test_rotvec_matrix_is_axis_angle_rodrigues():
    r=mod.rotvec_matrix(np.asarray([0.,0.,np.pi/2],np.float32))
    assert np.allclose(r@np.asarray([1.,0.,0.]),[0.,1.,0.],atol=1e-6)
    assert np.allclose(r.T@r,np.eye(3),atol=1e-6)


def test_pending_assignment_does_not_create_units():
    actions=np.asarray([[[0.]*7]*10,[[1.]*7]*10],np.float32)
    centers=np.asarray([actions[0]],np.float32)
    ids,pending=mod.assign_pending_nearest(np.asarray([0,-1]),actions,centers)
    assert ids.tolist()==[0,0]
    assert pending.tolist()==[False,True]
    assert len(centers)==1

