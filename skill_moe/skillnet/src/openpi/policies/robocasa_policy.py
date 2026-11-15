"""Policy input/output transforms for RoboCasa Panda-Omron observations."""
import os
import dataclasses

import einops
import numpy as np
from openpi import transforms
from openpi.models import model as _model


def configure_robocasa_subtask_generation():
    """Disable SkillNet's optional generated-subtask hook by default.

    The official RoboCasa pi0.5 checkpoint was trained and evaluated with the
    task instruction only. An explicitly set SkillNet or OpenPI environment
    variable remains authoritative.
    """
    if "SKILLNET_SUBTASK_GENERATE" not in os.environ and "OPENPI_SUBTASK_GENERATE" not in os.environ:
        os.environ["SKILLNET_SUBTASK_GENERATE"] = "0"
    value = os.environ.get("SKILLNET_SUBTASK_GENERATE", os.environ.get("OPENPI_SUBTASK_GENERATE", "0"))
    return value.strip().lower() not in {"", "0", "false", "no"}


def _image(x):
    x=np.asarray(x)
    if x.ndim==3 and x.shape[0] in (3,4) and x.shape[-1] not in (3,4): x=einops.rearrange(x,'c h w -> h w c')
    if np.issubdtype(x.dtype,np.floating): x=(255*np.clip(x,0,1)).astype(np.uint8)
    return np.ascontiguousarray(x)

def dataset_action_to_policy(action):
    """Reorder RoboCasa LeRobot action channels into the model command order."""
    action=np.asarray(action)
    if action.shape[-1] != 12:
        raise ValueError(f'RoboCasa action must have 12 dimensions, got {action.shape}')
    # LeRobot: base(4), mode(1), EEF position(3), EEF rotation(3), gripper(1).
    # Policy: EEF position(3), EEF rotation(3), gripper(1), base(4), mode(1).
    return np.concatenate([action[...,5:8],action[...,8:11],action[...,11:12],action[...,0:5]],axis=-1)


def dataset_state_to_policy(state):
    """Map LeRobot state [base, base rot, relative EEF, EEF rot, gripper] to the fork order."""
    state=np.asarray(state)
    if state.shape[-1] != 16:
        raise ValueError(f'RoboCasa state must have 16 dimensions, got {state.shape}')
    return np.concatenate([state[...,7:10],state[...,10:14],state[...,0:3],state[...,3:7],state[...,14:16]],axis=-1)


@dataclasses.dataclass(frozen=True)
class RobocasaDatasetOrder(transforms.DataTransformFn):
    """Reorder raw dataset rows only; online policy observations already use fork order."""
    def __call__(self,data):
        out=dict(data)
        state_key='observation/state' if 'observation/state' in data else 'state'
        out[state_key]=dataset_state_to_policy(data[state_key])
        if 'actions' in data:
            out['actions']=dataset_action_to_policy(data['actions'])
        return out

@dataclasses.dataclass(frozen=True)
class RobocasaInputs(transforms.DataTransformFn):
    action_dim:int
    model_type:_model.ModelType=_model.ModelType.PI05
    # The official RoboCasa fork feeds saved LeRobot action order straight to
    # convert_action. Keep that exact behavior selectable for checkpoint parity;
    # the separate atomic fine-tuning experiment uses the corrected EEF-first order.
    reorder_actions: bool = True
    def __call__(self,data):
        cameras={
            'base_0_rgb':_image(data['observation/image']),
            'right_wrist_0_rgb':_image(data['observation/right_image']),
            'left_wrist_0_rgb':_image(data['observation/wrist_image']),
        }
        state=np.asarray(data['observation/state'],dtype=np.float32)
        if state.shape[-1] != 16: raise ValueError(f'RoboCasa policy expects state[16], got {state.shape}')
        out={'state':transforms.pad_to_dim(state,self.action_dim),'image':cameras,'image_mask':{k:np.True_ for k in cameras}}
        if 'actions' in data:
            policy_actions=(dataset_action_to_policy(data['actions']) if self.reorder_actions
                            else np.asarray(data['actions']))
            out['actions']=transforms.pad_to_dim(policy_actions,self.action_dim)
        if 'prompt' in data: out['prompt']=data['prompt']
        return out

@dataclasses.dataclass(frozen=True)
class RobocasaOutputs(transforms.DataTransformFn):
    def __call__(self,data):
        action=np.asarray(data['actions'])
        if action.shape[-1]<12: raise ValueError(f'Expected at least 12 action dimensions, got {action.shape}')
        # Dataset-only reordering occurs in RobocasaDatasetOrder, before this
        # model transform. Online outputs already have convert_action order.
        return {'actions':action[...,:12]}
