"""Policy input/output transforms for RoboCasa Panda-Omron observations."""
import dataclasses
import einops
import numpy as np
from openpi import transforms
from openpi.models import model as _model

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

@dataclasses.dataclass(frozen=True)
class RobocasaInputs(transforms.DataTransformFn):
    action_dim:int
    model_type:_model.ModelType=_model.ModelType.PI05
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
            policy_actions=dataset_action_to_policy(data['actions'])
            out['actions']=transforms.pad_to_dim(policy_actions,self.action_dim)
        if 'prompt' in data: out['prompt']=data['prompt']
        return out

@dataclasses.dataclass(frozen=True)
class RobocasaOutputs(transforms.DataTransformFn):
    def __call__(self,data):
        action=np.asarray(data['actions'])
        if action.shape[-1]<12: raise ValueError(f'Expected at least 12 action dimensions, got {action.shape}')
        # Training labels were reordered to policy-command order by
        # RobocasaInputs. Only strip model padding here; env_utils.convert_action
        # already consumes this EEF/gripper/base/mode order.
        return {'actions':action[...,:12]}
