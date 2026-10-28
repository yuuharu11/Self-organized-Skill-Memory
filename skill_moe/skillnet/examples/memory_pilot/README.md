# Memory pilot: rollout recording and G0/G1 replay

Run from `/work` after `source env/activate.sh`. The default checkpoint is
`.skillnet-state/checkpoints/pi05_libero_moe_skill_4_40/moe_add_balance_4_128_30000/29999`.
Set `SKILLNET_MEMORY_PILOT_CHECKPOINT` to select another checkpoint.

Record one spatial task (task 0, one initial state):

```bash
cd /work
source env/activate.sh
python skill_moe/skillnet/examples/memory_pilot/record_rollouts.py \
  --suite libero_spatial --task-ids 0 --trials 1
```

Record two tasks with two trials each:

```bash
python skill_moe/skillnet/examples/memory_pilot/record_rollouts.py \
  --suite libero_spatial --task-ids 0 1 --trials 2
```

Run deterministic offline G0/G1 replay for all saved inference observations:

```bash
python skill_moe/skillnet/examples/memory_pilot/analyze_offline.py \
  /work/.skillnet-state/outputs/memory-pilot/record-<run-id> --g0 --g1
```

For a short smoke analysis, add `--steps 2`. The recorder writes a unique
`record-*` directory; analysis writes a separate unique `analysis-*` directory.
Both contain machine-readable metadata/results and do not overwrite prior runs.
Each inference `step` is stored as the matching zero-based row in `env_*`; with
`env_step_offset: 1`, environment step 10 is row 9.
The recorder captures action-expert suffix features at layers 2, 9, and 17 by
averaging suffix tokens at the final diffusion step. It also retains per-layer
actual dispatched expert choices for G0 change-rate measurements (including
the dispatch path's deterministic tie behavior when all skills are masked).
The analysis also reports the gated selected-expert contribution norm divided
by the shared-expert output norm, by layer. These features are stored
as `feat__L02_mean_t_last`, `feat__L09_mean_t_last`, and
`feat__L17_mean_t_last`, which match the G2 reader schema.

G0 conditions are the recorded skill IDs, guaranteed-different IDs from the
suite's annotation pool, all skill tokens masked to zero attention contribution,
and a changed noise tensor. The same fixed noise is used for all skill conditions;
the changed-noise condition uses the same observation and an independent seed.
G1 interpolates one active post-embedding token at alpha 0, 0.25, 0.5, 0.75,
and 1, then compares the midpoint with an independently evaluated two-token
set condition. All conditions use the same saved observation and noise. The
report includes sample counts and descriptive distances; it does not make a
success-rate claim.
# π0.5 subtask generation

`subtask_pi05_eval.py` runs frame aligned greedy subtask decoding from the plain
`pi05_base` model or a fine tuned Split A/B checkpoint. Labels are attached in
memory from `eval_skill_segment`; `-1` means no segment and is labeled `none`.
The exact prefix is `Task: <instruction>. Subtask: `. Decoding uses a fixed KV
cache, emits at most 50 tokens by default, includes EOS (token 1) in the token
array, and zero pads the remainder; text decoding stops before EOS. When EOS is
absent, the text is truncated at the token cap.

The evaluator joins outputs to the existing feature cache using
`(episode_id, step)`. LIBERO-40 task indices are goal 10–19, object 20–29, and
spatial 30–39; indices 0–9 belong to LIBERO-10 and are excluded from both train
splits.
