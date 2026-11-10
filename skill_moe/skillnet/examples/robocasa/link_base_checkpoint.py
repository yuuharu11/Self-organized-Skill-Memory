"""Assemble a servable checkpoint dir from pi05_base weights + RoboCasa norm stats.

For the wiring check only (run in the SkillNet Python 3.10 venv): pi05_base has
not been fine-tuned on RoboCasa, so its success rate is not a baseline. The
directory holds symlinks only:

  <out>/params                       -> cached pi05_base/params (downloaded anonymously if absent)
  <out>/assets/robocasa:atomic_seen  -> $SKILLNET_STATE_DIR/assets/pi05_robocasa_atomic/robocasa:atomic_seen
"""
import argparse
import os
import pathlib

from openpi.shared import download
from openpi.training import config_moe_skill as _config


def main() -> None:
    state = pathlib.Path(os.environ.get("SKILLNET_STATE_DIR", ".skillnet-state"))
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="pi05_robocasa_atomic")
    p.add_argument("--output", type=pathlib.Path, default=state / "checkpoints" / "pi05_base_robocasa_wiring")
    a = p.parse_args()
    config = _config.get_config(a.config)
    data_config = config.data.create(config.assets_dirs, config.model)
    stats = config.assets_dirs / data_config.asset_id
    if not (stats / "norm_stats.json").is_file():
        raise SystemExit(f"Missing {stats}/norm_stats.json; run compute_norm_stats_moe_skill.py first")
    params = download.maybe_download(config.weight_loader.params_path)
    (a.output / "assets").mkdir(parents=True, exist_ok=True)
    for link, target in ((a.output / "params", params), (a.output / "assets" / data_config.asset_id, stats)):
        if link.is_symlink() or link.exists():
            if link.resolve() != pathlib.Path(target).resolve():
                raise SystemExit(f"{link} exists and points elsewhere; choose another --output")
            continue
        link.symlink_to(pathlib.Path(target).resolve(), target_is_directory=True)
    print(a.output)


if __name__ == "__main__":
    main()
