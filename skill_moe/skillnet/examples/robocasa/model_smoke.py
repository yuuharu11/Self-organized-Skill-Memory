"""Wiring check for pi05_robocasa_atomic (run in the SkillNet Python 3.10 venv).

Loads one small batch from the locally downloaded RoboCasa data, builds plain
pi0.5 from pi05_base weights, and computes the flow-matching loss once. No
optimizer step and no checkpoint is written. With only the smoke task present,
set SKILLNET_ROBOCASA_ALLOW_PARTIAL=1.
"""
import argparse
import dataclasses

import jax
import numpy as np

from openpi.models import model as _model
from openpi.shared import download
from openpi.training import config_moe_skill as _config
from openpi.training import data_loader_skill as _data_loader


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="pi05_robocasa_atomic")
    p.add_argument("--batch-size", type=int, default=2)
    a = p.parse_args()
    config = dataclasses.replace(_config.get_config(a.config), batch_size=a.batch_size, num_workers=0)
    loader = _data_loader.create_data_loader(config, shuffle=True, num_batches=1)
    observation, actions = next(iter(loader))
    print("batch", {k: tuple(v.shape) for k, v in observation.images.items()}, "state", observation.state.shape,
          "actions", actions.shape)
    params_path = download.maybe_download(config.weight_loader.params_path)
    params = _model.restore_params(params_path, dtype=jax.numpy.bfloat16)
    model = config.model.load(params)
    loss, _ = model.compute_loss(jax.random.key(0), observation, actions, train=False)
    loss = np.asarray(loss)
    assert np.isfinite(loss).all(), loss
    print("model", type(model).__name__, "expert", config.model.action_expert_variant, "loss_mean", float(loss.mean()))
    print("MODEL SMOKE OK")


if __name__ == "__main__":
    main()
