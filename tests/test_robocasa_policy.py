import numpy as np
from openpi.policies.robocasa_policy import RobocasaInputs, RobocasaOutputs
import importlib.util
from pathlib import Path

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
    path=Path(__file__).parents[1]/'skill_moe/skillnet/examples/robocasa/replay_server.py'
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
