"""Ordinary openpi training config entry point plus RoboCasa multitask baseline.

Research/MoE configurations remain in :mod:`config_moe_skill`; the RoboCasa
baseline is deliberately a dense, ordinary Pi0.5 configuration.
"""
from __future__ import annotations

import dataclasses
import os
import pathlib

from openpi.training.config_moe_skill import *  # noqa: F403
from openpi.training import config_moe_skill as _research
from openpi.models import pi0_config as _pi0_config
from openpi.training import weight_loaders as _weight_loaders
from openpi import transforms as _transforms
from openpi.shared import normalize as _normalize
from openpi.policies import robocasa_policy as _robocasa_policy

_STATE_DIR = pathlib.Path(os.environ.get("SKILLNET_STATE_DIR", ".skillnet-state"))
_CHECKPOINT = pathlib.Path(os.environ.get(
    "SKILLNET_ROBOCASA_CHECKPOINT",
    str(_STATE_DIR / "checkpoints/robocasa365/pi05_pretrain_human300/multitask_learning/75000"),
))


@dataclasses.dataclass(frozen=True)
class _RobocasaPretrainData(_research.DataConfigFactory):
    repo_id: str = "robocasa:pretrain_human300"

    def create(self, assets_dirs, model_config):
        # The released checkpoint stores norm_stats.json directly under assets/.
        # Its quantiles are null, so the only supported exact normalization is
        # the checkpoint's mean/std (z-score), not synthesized quantiles.
        stats_dir = pathlib.Path(os.environ.get("SKILLNET_ROBOCASA_CHECKPOINT", str(_CHECKPOINT))) / "assets"
        stats = _normalize.load(stats_dir)
        repack = _transforms.Group(inputs=[_transforms.RepackTransform({
            "observation/image": "observation.images.robot0_agentview_left",
            "observation/right_image": "observation.images.robot0_agentview_right",
            "observation/wrist_image": "observation.images.robot0_eye_in_hand",
            "observation/state": "observation.state",
            "actions": "action",
            "prompt": "task",
        }), _robocasa_policy.RobocasaDatasetOrder()])
        data = _transforms.Group(
            inputs=[_robocasa_policy.RobocasaInputs(
                action_dim=model_config.action_dim,
                model_type=model_config.model_type,
                reorder_actions=False,
            )],
            outputs=[_robocasa_policy.RobocasaOutputs()],
        )
        model_transforms = _research.ModelTransformFactory()(model_config)
        root = os.environ.get(
            "SKILLNET_ROBOCASA_DATA",
            str(pathlib.Path(os.environ.get("SKILLNET_STATE_DIR", ".skillnet-state")) / "data/robocasa"),
        )
        return _research.DataConfig(
            repo_id=self.repo_id, root=root, norm_stats=stats, use_quantile_norm=False,
            repack_transforms=repack, data_transforms=data,
            model_transforms=model_transforms, action_sequence_keys=("actions",),
        )


_ROBOKIT = _research.TrainConfig(
    name="pi05_robocasa_multitask",
    exp_name="pi05_robocasa_multitask",
    model=_pi0_config.Pi0Config(pi05=True, max_token_len=200, action_horizon=50),
    data=_RobocasaPretrainData(),
    weight_loader=_weight_loaders.CheckpointWeightLoader(_research.PI05_BASE_PARAMS),
    assets_base_dir=str(_CHECKPOINT / "assets"),
    checkpoint_base_dir=str(_CHECKPOINT.parent),
    batch_size=1,
    num_train_steps=1,
)


def get_config(config_name: str):
    if config_name == _ROBOKIT.name:
        return _research.validate_config(_ROBOKIT)
    return _research.get_config(config_name)


def cli():
    import tyro
    configs = dict(_research._CONFIGS_DICT)
    configs[_ROBOKIT.name] = _ROBOKIT
    return _research.validate_config(tyro.extras.overridable_config_cli({k: (k, v) for k, v in configs.items()}))
