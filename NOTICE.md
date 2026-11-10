# Notices

This repository contains SkillNet modifications and experiment scripts derived
from Apache-2.0 upstream pi0.5/OpenPI components; public workflows are
documented as SkillNet.

## Upstream Projects

- OpenPI is licensed under Apache-2.0.
- RoboCasa policy transforms and evaluation-client wiring in `skill_moe/skillnet/src/openpi/policies/robocasa_policy.py` and `skill_moe/skillnet/examples/robocasa/` are adapted from the Apache-2.0 RoboCasa OpenPI fork at `https://github.com/robocasa-benchmark/openpi`, revision `5a6beda9ff99da30b4e1b59320f6a32971d7c397`.
- RoboCasa environment integration uses RoboCasa v1.0.1 and robosuite at the pinned revisions documented in `env/robocasa/`; their source licenses remain with those upstream projects.
- LIBERO code is licensed under MIT. The included LIBERO-Skill task files are provided for benchmark reproduction; external LIBERO datasets/assets must be obtained under their own terms.

## Data and Weights

This repository does not include model checkpoints, training datasets, videos, logs, or generated experiment artifacts. Configure checkpoint, dataset, cache, and output locations for your own machine before running training or evaluation.
