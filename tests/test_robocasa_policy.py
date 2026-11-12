"""RoboCasa policy checks added with the skill-memory work: official checkpoint order, multitask config, subtask defaults."""
import numpy as np
import pytest
from openpi.policies.robocasa_policy import RobocasaInputs, RobocasaOutputs, RobocasaDatasetOrder
import importlib.util
from pathlib import Path


def test_official_multitask_checkpoint_path_keeps_saved_action_order_and_three_cameras():
    action=np.arange(12,dtype=np.float32)
    got=RobocasaInputs(action_dim=32,reorder_actions=False)({
        'observation/image':np.zeros((8,8,3),np.uint8),
        'observation/right_image':np.zeros((8,8,3),np.uint8),
        'observation/wrist_image':np.zeros((8,8,3),np.uint8),
        'observation/state':np.arange(16,dtype=np.float32),'actions':action,
    })
    np.testing.assert_array_equal(got['actions'][:12],action)
    np.testing.assert_array_equal(got['actions'][12:],np.zeros(20))
    np.testing.assert_array_equal(got['state'][:16],np.arange(16))
    assert got['image_mask']=={'base_0_rgb':np.True_,'right_wrist_0_rgb':np.True_,'left_wrist_0_rgb':np.True_}

def test_dataset_only_transform_reorders_state_and_action_to_fork_model_order():
    row={'observation/state':np.arange(16,dtype=np.float32),'actions':np.arange(12,dtype=np.float32)}
    got=RobocasaDatasetOrder()(row)
    np.testing.assert_array_equal(got['observation/state'],[7,8,9,10,11,12,13,0,1,2,3,4,5,6,14,15])
    np.testing.assert_array_equal(got['actions'],[5,6,7,8,9,10,11,0,1,2,3,4])

def test_multitask_is_plain_pi05_horizon_50_and_not_moe():
    from openpi.training import config
    cfg=config.get_config('pi05_robocasa_multitask')
    assert cfg.model.pi05 and cfg.model.action_horizon==50 and cfg.model.action_dim==32
    assert cfg.model.__class__.__module__.endswith('pi0_config')
    assert cfg.model.discrete_state_input is True and cfg.model.max_token_len==200
    assert cfg.weight_loader.__class__.__name__=='CheckpointWeightLoader'
    try:
        data=cfg.data.create(cfg.assets_dirs,cfg.model)
    except FileNotFoundError:
        pytest.skip('multitask checkpoint assets not present')
    assert data.use_quantile_norm is False


def test_baseline_defaults_to_task_only_but_respects_explicit_subtask_setting(monkeypatch):
    spec=importlib.util.spec_from_file_location(
        'robocasa_serve_policy', Path(__file__).parents[1]/'skill_moe/skillnet/examples/robocasa/serve_policy.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    monkeypatch.delenv('SKILLNET_SUBTASK_GENERATE', raising=False)
    monkeypatch.delenv('OPENPI_SUBTASK_GENERATE', raising=False)
    module.configure_plain_pi05_inference()
    assert __import__('os').environ['SKILLNET_SUBTASK_GENERATE']=='0'
    monkeypatch.setenv('SKILLNET_SUBTASK_GENERATE','1')
    module.configure_plain_pi05_inference()
    assert __import__('os').environ['SKILLNET_SUBTASK_GENERATE']=='1'


def test_robocasa_subtask_default_helper_and_memory_model_are_scoped(monkeypatch):
    from openpi.policies.robocasa_policy import configure_robocasa_subtask_generation
    import importlib.util
    core_path = Path(__file__).parents[1]/'skill_moe/skillnet/examples/memory_pilot/pi05_memory/core.py'
    spec = importlib.util.spec_from_file_location('pi05_memory_core_for_test', core_path)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)

    monkeypatch.delenv('SKILLNET_SUBTASK_GENERATE', raising=False)
    monkeypatch.delenv('OPENPI_SUBTASK_GENERATE', raising=False)
    assert configure_robocasa_subtask_generation() is False
    assert __import__('os').environ['SKILLNET_SUBTASK_GENERATE']=='0'

    class PlainModel:
        pass
    class RoboCasaMemoryModel:
        pass
    core._disable_memory_model_subtask_by_default(RoboCasaMemoryModel)
    assert RoboCasaMemoryModel.subtask_generation_default is False
    assert not hasattr(PlainModel, 'subtask_generation_default')


def test_robocasa_default_preserves_explicit_openpi_override(monkeypatch):
    from openpi.policies.robocasa_policy import configure_robocasa_subtask_generation
    monkeypatch.delenv('SKILLNET_SUBTASK_GENERATE', raising=False)
    monkeypatch.setenv('OPENPI_SUBTASK_GENERATE','1')
    assert configure_robocasa_subtask_generation() is True
    assert 'SKILLNET_SUBTASK_GENERATE' not in __import__('os').environ

def test_checkpoint_smoke_compares_action_in_policy_order():
    from openpi.policies.robocasa_policy import dataset_action_to_policy
    raw=np.arange(12,dtype=np.float32)
    np.testing.assert_array_equal(dataset_action_to_policy(raw),[5,6,7,8,9,10,11,0,1,2,3,4])
