# Emergent motion primitive quick experiment

This is a small decision experiment, not a VLA and not a publication benchmark. It reads
recorded executed segments, trains a Gaussian posterior `q(z | state, action, eef motion)`,
and routes each complete segment to N small FiLM action decoders. Task strings and skill IDs
are retained only as held-out evaluation labels and are never passed to the model.

## Data and model

For each inference row, the loader uses `pred_chunk[:n_executed]` as the action target and
`env_*` plus `step` to recover relative end-effector position, rotation, and gripper motion.
The segment start's `proprio` is the current-state condition at every decoded phase. Action
and motion sequences are resampled to H points; no unexecuted predicted tail enters the
dataset. Suite/task-namespaced initial-state groups are split before fitting normalization
statistics, so episodes from one initial state and all of their windows stay on one side.
At least two distinct groups are required.

The objective is action MSE plus `beta_kl * KL(q(z|segment)||N(0,I))` and a small load
balancing penalty `lambda * (N * sum(mean_gate_probability^2) - 1)`. Defaults are
`beta_kl=0.001`, `latent_dim=8`, and `lambda=0.03`. The encoder sees the complete executed
action and observed motion segment while the decoder receives only the start state, phase,
and z. Consequently z can encode much or all of the target; this is a posterior reconstruction
probe, not evidence that a deployable primitive code can be inferred without the segment.
Use beta and latent-dimension sweeps and inspect held-out KL in nats to diagnose that channel.
Top-1 uses a straight-through hard route during training;
soft routing mixes experts. An optional shared expert is added to the routed prediction.
The grid covers N={1,2,4,8,16}, shared expert on/off, top-1/soft, and seeds 0,1,2.

Held-out outputs include action reconstruction MSE, actual routed expert mass, mean KL, and empirical MI
between routed expert mass and displacement direction / rotation / gripper / task labels.
Hard routes reduce to one-hot assignments; soft routes retain all expert weights rather than
silently replacing them with argmax. Argmax usage is kept separately as a diagnostic. Task
and initial-state labels are suite/task-qualified; duplicate source files are rejected.
Direction is binned by dominant signed XYZ axis, rotation by dominant signed axis, and gripper
by opening/closing/stationary thresholds (±0.01 on the two-finger gap change `q_left - q_right`
of `robot0_gripper_qpos`; a single motion gripper channel is used as-is). MI is reported in raw bits and relative to a
label-shuffle null that preserves class counts; random-initialization MI uses the same held-out
segments. Results include sample counts, null spread, label entropy, and normalized MI because
plug-in MI is positively biased at small sample sizes. This per-segment permutation null is
a cardinality diagnostic, not a significance test for correlated windows. Conditional MI
`I(expert; joint motion category | task)` and `I(expert; task | joint motion category)` are
reported with stratum coverage. If tasks and motions do not cross, these comparisons are
marked unidentifiable; comparing marginal MI alone cannot establish motion-based specialization.
Soft task-sharing counts include all positive-weight experts, so inspect mean mass by task.
Per-expert ablation removes an expert,
renormalizes remaining router probabilities, and reroutes (so it measures post-removal
rerouting, not a fixed-route knockout). Each category's error increase subtracts the full
model error on the same category. The experiment does not claim to predict future
environment state from actions.

## Cross reconstruction

For each evaluated segment, the decoder also reconstructs its actions with a posterior-mean
z encoded from a **training** segment in a different episode and initial-state group. The
query's starting state and phase stay fixed; donor z supplies both routing and FiLM input.
An oracle selects the nearest observed future eef trajectory: mean relative-position distance,
SO(3) rotation angle, and gripper distance, each divided by a training-only RMS motion scale.
No evaluated action is used to encode donor z. Query future motion is used for selection,
so this is an oracle transfer diagnostic, not an online retrieval policy.

Uniform random donors use exactly the same eligibility pool. Their MSE averages eight draws
per query by default (`--cross-random-draws`); indices and draw errors are saved. Selection
uses a split-specific seed independent of model seed, so donors stay fixed across beta/N/
shared/routing variants. One report table lists beta, own-z MSE, retrieved-z MSE, random-z
MSE, the retrieved-minus-own gap, and KL in nats for each configuration and model seed.

A small gap supports transfer only when random/zero-z controls are worse. If every condition
has similar error, the decoder may ignore z. KL regularizes the Gaussian posterior used in
training; it does not guarantee that deterministic posterior means cannot encode target
actions. Neither this oracle diagnostic nor MI alone proves primitive emergence. These are
descriptive results: model seeds do not measure split uncertainty, and changing shared/N
also changes parameter count (recorded in results.json).

## Run

Checker validation uses synthetic data. For a completed `record-*` run whose own
`report.md` marks it 有効, the full comparison grid is:

```bash
cd /work && source env/activate.sh
python -m skill_moe.skillnet.examples.memory_pilot.primitives.run_experiment \
  --run-dir /work/.skillnet-state/outputs/memory-pilot/record-<valid-run-id> \
  --horizon 10 --epochs 30 --batch-size 64 --beta-kl 0.0001 0.001 0.01 0.1 \
  --latent-dims 4 8 16 --device cpu
```

If there is a valid report under a `record-*` directory, `--run-dir` may be omitted and the
CLI selects the newest valid record. Outputs are created in a new
`/work/.skillnet-state/outputs/memory-pilot/primitives-<timestamp>-<id>/` directory with
`results.json` and `report.md`; existing output directories are never replaced.

Run a generated CPU smoke experiment from the repository root with
`python -m skill_moe.skillnet.examples.memory_pilot.primitives.run_experiment --synthetic`.
The automated test suite uses synthetic recorded NPZ files and a reduced training setting.
