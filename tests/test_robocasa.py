"""RoboCasa policy I/O, task-set coverage and unit cutting."""
import importlib.util
import json

import numpy as np
import pytest
from openpi.policies.robocasa_policy import RobocasaInputs, RobocasaOutputs
from openpi.training.data_loader import _validate_robocasa_task_coverage
from scipy.fft import dct

from skill_moe.skillnet.examples.memory_pilot.robocasa.units import cut
from tests.support import (
    REPO,
)

# ── Policy inputs and outputs ─────────────────────────────────────────────────────────────────────

def test_policy_output_strips_padding_from_policy_command_order():
    padded=np.arange(16,dtype=np.float32)[None,:]
    got=RobocasaOutputs()({'actions':padded})['actions']
    np.testing.assert_array_equal(got,np.arange(12,dtype=np.float32)[None,:])

def test_policy_input_reorders_training_actions_and_pads_after_reorder():
    action=np.arange(12,dtype=np.float32)
    inputs=RobocasaInputs(action_dim=16)
    got=inputs({'observation/image':np.zeros((8,8,3),np.uint8),
        'observation/right_image':np.zeros((8,8,3),np.uint8),
        'observation/wrist_image':np.zeros((8,8,3),np.uint8),
        'observation/state':np.zeros(16,np.float32),'actions':action})
    np.testing.assert_array_equal(got['actions'],[5,6,7,8,9,10,11,0,1,2,3,4,0,0,0,0])

def test_replay_reorders_dataset_actions_before_environment_conversion():
    path=REPO/'skill_moe/skillnet/examples/robocasa/replay_server.py'
    spec=importlib.util.spec_from_file_location('robocasa_replay_server',path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    np.testing.assert_array_equal(module.dataset_to_env_actions(np.arange(12)),[5,6,7,8,9,10,11,0,1,2,3,4])


def test_widen_degenerate_quantiles_only_touches_constant_channels():
    import numpy as np
    from openpi.shared import normalize
    from openpi.training.config_moe_skill import _widen_degenerate_quantiles

    stats = {"actions": normalize.NormStats(mean=np.zeros(3), std=np.ones(3),
                                            q01=np.array([-1.0, 0.0, -0.5]), q99=np.array([-1.0, 0.0, 0.5]))}
    out = _widen_degenerate_quantiles(stats)["actions"]
    np.testing.assert_allclose(out.q01, [-2.0, -1.0, -0.5])
    np.testing.assert_allclose(out.q99, [0.0, 1.0, 0.5])
    # Idempotent: already widened stats are left unchanged.
    again = _widen_degenerate_quantiles({"actions": out})["actions"]
    np.testing.assert_allclose(again.q01, out.q01)
    np.testing.assert_allclose(again.q99, out.q99)


# ── Task-set coverage ─────────────────────────────────────────────────────────────────────────────

def test_static_task_set_schema_is_independent_of_robocasa_import():
    p=REPO/'data_process/robocasa/task_sets.json'
    if not p.exists(): return
    d=json.loads(p.read_text()); sets=d['task_sets']
    assert d['robocasa_commit']=='3443cdf3dd20f9ae8ed41adba442059b506891d0'
    assert len(sets['atomic_seen'])==18 and len(sets['composite_seen'])==16 and len(sets['composite_unseen'])==16
    assert set(sets['target50'])==set(sets['atomic_seen']+sets['composite_seen']+sets['composite_unseen'])
    assert all(t in d['tasks'] and d['tasks'][t]['horizon']>0 for t in sets['target50'])
    assert all(d['tasks'][t]['dataset_path'].endswith('/lerobot') for t in sets['target50'])
    assert d['action_fields']['dataset_order']=={'base_motion':[0,4],'control_mode':[4,5],'end_effector_position':[5,8],'end_effector_rotation':[8,11],'gripper_close':[11,12]}

def test_atomic_training_rejects_partial_dataset_unless_explicit_smoke_override():
    with pytest.raises(FileNotFoundError,match='incomplete'):
        _validate_robocasa_task_coverage(['A','B'],['A'])
    assert _validate_robocasa_task_coverage(['A','B'],['A'],allow_partial=True)==['B']


# ── Unit cutting ──────────────────────────────────────────────────────────────────────────────────

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
