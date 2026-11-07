import importlib.util
from pathlib import Path
import numpy as np
import pytest

P=(Path(__file__).resolve().parents[1]/'skill_moe/skillnet/examples/memory_pilot/unseen_libero/theta_regression.py')
S=importlib.util.spec_from_file_location('unseen_theta_regression_test',P)
M=importlib.util.module_from_spec(S);S.loader.exec_module(M)


def test_grouped_prefix_features_keep_camera_and_language_separate():
    h=np.arange(7*4,dtype=np.float32).reshape(7,4);g=np.array([0,0,1,1,1,2,2])
    x=M.grouped_means(h,g)
    np.testing.assert_allclose(x[0],h[:2].mean(0));np.testing.assert_allclose(x[1],h[2:5].mean(0));np.testing.assert_allclose(x[2],h[5:].mean(0))
    assert x.shape==(3,4)


def test_rot6_representation_roundtrips_axis_angle():
    from scipy.spatial.transform import Rotation
    t=np.array([.2,-.1,.3]);v=M.rotation6(t)
    np.testing.assert_allclose(Rotation.from_matrix(M.rot6_matrix(v)).as_rotvec(),t,atol=1e-6)


def test_axis_angle_log_scale_decodes_in_canonical_physical_space():
    from scipy.spatial.transform import Rotation
    import importlib.util
    vs_path=P.parent/'variable_shape_memory.py'
    spec=importlib.util.spec_from_file_location('theta_test_variable_shape_memory',vs_path)
    vs=importlib.util.module_from_spec(spec);spec.loader.exec_module(vs)
    shape=np.arange(70,dtype=np.float32).reshape(10,7)/50
    theta=np.r_[np.array([.2,-.1,.3],np.float32),np.log(1.4)]
    actual=M.decode_theta(shape,theta,vs)
    r=Rotation.from_rotvec(theta[:3]).as_matrix();scale=np.exp(theta[3])
    physical=np.concatenate([shape[:,:3]@r,shape[:,3:6]@r],axis=1)*scale
    expected=np.concatenate([vs.to_normalized(physical),shape[:,6:7]],axis=1)
    np.testing.assert_allclose(actual,expected,atol=1e-6)


def test_resampling_and_candidate_scoring_ignore_gripper_for_shape():
    a=np.zeros((4,7),np.float32);a[:,0]=np.arange(4);a[:,6]=1
    b=a.copy();b[:,6]=-1
    x=M._resample_window(a);y=M._resample_window(b)
    assert x.shape==(10,7)
    assert M._shape_error(x,y,None)[0]==pytest.approx(0.)


def test_random_candidate_baseline_averages_errors_not_actions():
    target=np.zeros((1,10,7),np.float32)
    replicas=np.stack([np.ones((10,7),np.float32),-np.ones((10,7),np.float32)],axis=0)[None]
    mean_error=M._replica_mean_errors(replicas,target)
    blended=np.mean(replicas,axis=1)
    blended_error=np.mean((blended-target)**2,axis=(1,2))
    np.testing.assert_allclose(mean_error,[1.])
    np.testing.assert_allclose(blended_error,[0.])


def test_training_cache_inventory_is_fixed_and_separate_from_eval_targets():
    assert M.SOURCE.name=='production'
    assert M.SPLITS==('train','dev','eval')
    assert M.CACHES==('C_vm_none','C_vm_08')
    assert len(M.VARIANTS)==5


def test_pose_diagnostics_compare_matching_representations():
    p=np.array([.2,-.1,.3,.4],np.float32)
    t=np.array([-.1,.2,.1,-.3],np.float32)
    for kind in ('aa','6d'):
        expected=np.mean((M._target_theta_repr(p,kind)-M._target_theta_repr(t,kind))**2)
        assert M._pose_parameter_error(p,t,kind)==pytest.approx(expected)
        assert M._pose_parameter_error(p,p,kind)==pytest.approx(0.)
