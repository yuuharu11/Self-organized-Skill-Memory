# DMP route S3 model

This directory contains an opt-in S3 model class and helpers. It does not
modify the released `Pi0`/`Pi0Skill` sampling paths.

## Model and training API

- `Pi0DmpRouteConfig` contains no dictionary-sized parameters. It fixes the S1
  basis width and state-history length, not the number of primitives.
- `route_from_observation(observation, primitive_shapes, state_history)` uses
  a learned shape encoder `E(w)` and a query `q(prefix, state_history)`;
  scores are `q @ E(w).T` over the runtime dictionary. Inference is hard
  top-1, and the selected `E(w)` is passed to the generator. Adding a shape
  does not change any parameter shape (the S4 fixed-weight pointer path).
- `compute_s3_loss(..., use_primitives=True)` trains that self-discovery target
  with CE + parameter regression and the t0=.4 bridge velocity loss. The matched
  G0 comparison uses `use_primitives=False`, the same bridge/noise/time sample
  with A=0, and no route injection. `make_s3_train_step` supplies filters for
  equal-update G0/G1 runs. `initialize_from_skillnet90` retains random S3-only parameters while
  loading compatible weights from the public SkillNet-90 checkpoint.
- `injection_mode` supports `both`, `router_only`, `adarms_only`, and `none`.
  The `none` skill mask is all false, so G0 receives no valid primitive token.
- `shuffle_route_primitives` provides a key-controlled primitive permutation;
  regenerate the single draft and embedding from that ID.
- `dmp_shape_mode="straight"` replaces only the canonical shape with a
  minimum-jerk straight-to-goal path; predicted `g`, `tau`, `phase_s`, and the
  selected gripper remain unchanged. `dmp_primitive_mode="shuffle"` enables
  the existing shape-shuffle control.
- Inference creates one hard-selected DMP draft, keeps it in dimensions 7:14
  through every denoising call, zeros 14:32, and returns only active dimensions
  0:7. No candidate averaging or normal-policy fallback is present.

Train labels must be joined by `(episode_id, t)` using `join_s1_targets` with
`allowed_splits=("train",)`. Human task/subtask annotations are not read by the
router or DMP training API. The model's VLM prefix is frozen via stop-gradient;
the optimizer filter covers action expert, query encoder, shape encoder, and
route condition encoder. State histories are causal and same-episode; training
can build them with `state_history_windows`.
At inference, callers may pass a transformed `dmp_state_history`, or a stable
`dmp_history_id` plus `dmp_reset_history` to maintain causal history. If no
stream ID is provided, the policy repeats the current state for a cold start;
it never shares hidden history across concurrent clients.

## Inference server after a trained S3 checkpoint exists

From `/work/skill_moe/skillnet`:

```bash
source /work/env/activate.sh
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 \
python examples/memory_pilot/dmp_route/serve_s3.py \
  --checkpoint /path/to/s3/step/params --port <free-port> \
  --prompt-level L0 --s1-dir /work/.skillnet-state/outputs/memory-pilot/dmp-route-20261102/s1
```

The caller can override the default level per request with `dmp_prompt_level`
(`L0` or `L3`). L3 is looked up by the incoming original prompt in
`abstract-prompt-20261101/prompts.json`. The server is intentionally not started
by the model implementation task.
