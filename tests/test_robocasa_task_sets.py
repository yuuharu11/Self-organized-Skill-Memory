import json, pathlib
import pytest
from openpi.training.data_loader import _validate_robocasa_task_coverage

def test_static_task_set_schema_is_independent_of_robocasa_import():
    p=pathlib.Path(__file__).parents[1]/'data_process/robocasa/task_sets.json'
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
