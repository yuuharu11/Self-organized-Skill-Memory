import numpy as np
from scipy.fft import dct
from skill_moe.skillnet.examples.memory_pilot.robocasa.units import cut

def test_units_preserve_robot_channels_and_segment_without_labels():
    x=np.zeros((50,12),np.float32); x[:25,5]=.1; x[25:,6]=.1; x[24:26,11]=1
    out=cut(x)
    assert out and all(len(u['arm_actions'][0])==7 for u in out)
    assert all(len(u['other_actions'][0])==5 for u in out)
    # A grip switch is a forced cut even when it creates a short interval.
    assert [(u['start'],u['end']) for u in out]==[(0,24),(24,26),(26,50)]

def test_base_motion_is_kept_separate_and_optional():
    x=np.zeros((30,12),np.float32); x[:,0]=.2
    unit=cut(x,base_motion_boundaries=True)[0]
    assert len(unit['other_actions'][0])==5

def test_gripper_switches_force_boundaries_even_for_short_terminal_units():
    x=np.zeros((24,12),np.float32)
    x[6:,11]=1;x[19:,11]=-1
    out=cut(x,min_length=10)
    assert [(u['start'],u['end']) for u in out]==[(0,6),(6,19),(19,24)]

def test_type_feature_is_orthonormal_dct_ii_not_real_fft():
    x=np.zeros((20,12),np.float32)
    x[:,5]=np.linspace(-1,1,20,dtype=np.float32)
    x[:,6]=np.sin(np.linspace(0,3,20,dtype=np.float32))
    unit=cut(x,min_length=10)[0]
    arm=np.asarray(unit['arm_actions'],np.float32)
    res=arm[np.linspace(0,len(arm)-1,10).round().astype(int)]
    expected=dct(res,type=2,norm='ortho',axis=0)[:6].reshape(-1)
    np.testing.assert_allclose(unit['type_dct'],expected,rtol=1e-6,atol=1e-6)
