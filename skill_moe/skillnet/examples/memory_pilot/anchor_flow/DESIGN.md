# S2: π0.5 action expert with an M6 anchor-start distribution

This is the full-model counterpart to S1. S1's `examples/emergent/light/intermediate/anchor_flow.py` is not imported or modified here. The S2 implementation is opt-in in `openpi.models.pi0_anchor_flow`; the released `pi0.py` and SkillNet model behavior remain unchanged.

## Checkpoint, action convention, and data boundary

- Initialize from `/work/.skillnet-state/checkpoints/pi05_libero_moe_skill_4_90/moe_balance_4_32_90_20000/19999/params` (the requested checkpoint directory's concrete run/step path). This is the LIBERO-90 checkpoint. Never initialize S2 from a LIBERO-40 checkpoint.
- Match the checkpoint config: π0.5, `gemma_2b` PaliGemma, four-expert `gemma_300m_moe_4` action expert, `action_horizon=10`, `action_dim=32`. LIBERO supplies 7 action channels; use the existing LIBERO input transform and checkpoint normalization, then pad the normalized action to 32 channels exactly as OpenPI does. The final 25 dimensions are the standard padded dimensions, not a second normalization space.
- Fit M1 Enc/Prior/Dec and M6 solely on LIBERO-90 demonstrations in this same normalized `[10,32]` space. Freeze them after fitting. Store their action normalization and representation version with the memory; reject a LIBERO-40 or H=25 codec. The retrieval key is the L2-normalized mean pooled final hidden state over valid π0.5 VLM prefix tokens (image, query text, and π0.5's tokenized robot state). Every input cache must declare one `key_text_source`: `instruction_only` or `pi05_generated_subtask`; generated keys also name `key_text_generator=pi05` and the exact `key_text_generator_checkpoint` used to generate text. Annotation-derived `instruct2plan` text is rejected. Memory construction, task additions, fine-tuning, and closed-loop queries must use the same text source. **The primary evaluation uses `instruction_only`:** at every action query it uses `task.language`, with no ground-truth subtask or segment boundary. The runner also supports `pi05_generated_subtask`, generated at each query from the current observation by a separate frozen, plain `pi05_base` model; only that generated text enters the query key, never annotation-derived labels. A generated-subtask memory must be built with the same generator version and key source. The existing generator was trained on LIBERO-40 Split A/B, not LIBERO-90. Its weights can be mechanically reused as a secondary text source, but cross-suite held-out exact match was only 0.276 for goal and 0.228 for spatial; instruction-only remains the main S2 condition. A LIBERO-90-specific generator is not a prerequisite for this run and would require compatible per-frame labels plus a separate train run. The checkpoint's separate SkillNet action-expert conditioning path is preserved for checkpoint-compatible behavior and held identical across S2 controls. The rollout smoke passes the same task-level skill annotation as the standard evaluator in every condition; this annotation only enters the unchanged SkillNet routing path, never the memory key.
- Available locally: LIBERO-90 RLDS TFRecords under `.skillnet-state/data/libero/libero90_rlds/libero_90_openvla_processed/1.0.0`; the 90 checkpoint and its `jsw19/libero_90_v1` normalization stats; LIBERO-10 demo HDF5 files under `/work/data/LIBERO/libero_10`. No converted, stable LIBERO-90 LeRobot directory was present during inspection (only staging directories); S2's `build_memory.py` therefore consumes an explicit normalized LIBERO-90 NPZ export. LIBERO-Skill `skill_obj_02`–`09` demonstration files/manifests were not found under `/work/data` or `.skillnet-state/data`; their memory-add portion cannot run until those demos are supplied. This is a data availability limitation, not an implicit substitute with LIBERO-40.

## Algorithm

For normalized expert action chunk `A ∈ R^(10×32)` and proprioceptive state `s`, fit a frozen M1 representation `(b,r)=Enc(A,s)`, `Â=Dec(b,r,s)`. Use M6's functional decoder distance and pending-cluster rule to form an open-ended set of units. A unit stores its center and decoder-space spread `(c_k, σ_k)`; episodic entries store the center/value together with `κ_k`, the VLM prefix key. Additions append keys/values or unit metadata only; the trainable model parameter digest must be identical before and after each add.

At observation `o`, calculate `q=normalize(mean_pool(VLM_prefix(o)))`, retrieve eligible M6 entries by cosine-softmax `w_k=softmax(q·κ_k/τ)` using M6's calibrated temperature, and sample one unit `k` (weights of entries assigned to the same unit are summed). Draw a base in M6's own latent space, not action space:

`b' ~ N(c_k, σ_k² I)`, `a_k=Dec(b', r_prior(q,s), q, s)`, then `x_src=(1-t0)*a_k+t0*ε`, `ε~N(0,I)`.

Here `σ_k` is the isotropic RMS spread of member codes around unit center `c_k`; it is measured in base-code coordinates and is never applied in action coordinates. Training retains the ordinary path `x_t=(1−t)A+tε`, and mixes in anchor-corrupted inputs `x_t=(1−t)a+tε`, always supervised against the action-flow target `ε−A`. The prepared rows are 1/3 ordinary, 60% M6 retrieval (excluding the same episode and initial state), and 1/15 target-derived anchor. The target-derived anchor is deliberately a small share so it does not overstate retrieval quality. All inference controls reuse the same checkpoint, training rows, optimizer updates, and step count:

For each training row the action-flow target is `target_v=ε−A`; the ordinary input is `(1−t)A+tε`, while an anchor row uses `(1−t)a+tε` with the identical target.

The VLM and vision tower stay frozen; only the action-expert layers, MoE router, timestep MLP, and action input/output projections are trainable. There is no auxiliary router/skill-label/contrastive loss. For inference use `x_src` at `t0(o)` and Euler-integrate `dx/dt=vθ(x_t,o,t)` while stepping time downward from `t0` to `0` (equivalently, `x_next=x+Δt·vθ` for `Δt<0`, matching OpenPI's sampler); `t0=1` with the same initial noise delegates to the existing sampler and is bitwise-equivalent at the API output. The controlled comparisons are: (a) M6 component sampling with entropy-based `t0`, (b) absolute max-cosine similarity, (c) fixed `t0=0.5`, (d) query-shuffled `t0` from previous retrievals, (e) shuffled key/value assignment, (f) pre-add memory, (g) RTS-style top-1 raw normalized action chunk with the same confidence-based `t0` and no critic, and (h) OptimusVLA-style moment-matched single Gaussian over retrieved decoded chunks. Similarity thresholds are explicit CLI parameters and must be calibrated on LIBERO-90 before evaluation; the current defaults are only smoke placeholders. `eval_series_a.py` emits these nine conditions for each series-A task and the entropy/disabled pair for retention checks.

Memory construction and retrieval use M6's functional unitizer and leave learned weights frozen. In the current wrapper, `M6.fit` owns the M1 codec; training/build scripts must pass only LIBERO-90 rows and persist the exact action statistics and checkpoint/config hashes. Episodic keys and values are added from the same interval, and evaluation passes episode/init-state metadata to exclude leakage.

## S2 training and evaluation plan (production runs intentionally not launched)

- Proposed fine-tuning budget: **20,000 steps, batch 32**, action expert only. The H200 synthetic batch-32 smoke measured 1.39 seconds for the post-compile update; that implies about **7.7 hours of step time**, or **8–10 hours** including data/validation/checkpoint overhead. This does not include M1/M6 fitting. No LIBERO-10 or LIBERO-Skill transitions are used for this fine-tune.
- Closed-loop series A: first report LIBERO-90 known-task retention. Then process LIBERO-10 in its official task order and LIBERO-Skill `skill_obj_02` through `skill_obj_09`, adding demonstrations to memory without optimizer updates. Compare the nine controls above for each series-A task; use the same fine-tuned action head, training data, update count, and evaluation initial states for every control. Use at least 20 episodes per task; log success, retrieved source task/weights, M6 key-only versus new-unit adds, memory growth, checkpoint hashes, and confidence intervals. Do not treat the missing LIBERO-Skill data as zero-shot results. All controls receive the same checkpoint path and per-task trial seed base; the runner must replay the same LIBERO initial-state index and policy-noise seed for each paired trial. SkillNet's task-level plan annotation is also held identical across conditions, as in the standard SkillNet evaluator, and is not the memory-query text.

### Minimal decision matrix

For the S2-a/b gates, run only the required controls first: the 10 retention tasks × `{anchor_entropy_current_memory, anchor_disabled_t0_1}` (20 task-condition cases), plus the 18 new tasks × `{anchor_entropy_current_memory, anchor_disabled_t0_1, anchor_memory_shuffled, anchor_before_task_add}` (72 cases). This is **92 cases × 20 trials = 1,840 episodes**, versus 182 × 20 = 3,640 in the full ablation plan. It directly covers retention, added-memory benefit against both no-anchor and pre-add memory, and the shuffled-memory negative control. Defer RTS, OptimusVLA, and `t0` ablations until the S2 gates pass. At a planning rate of 30–60 seconds per full LIBERO episode, this is about **15–31 serial environment-hours**; measure actual throughput on the first task and revise before launching the matrix. The planning rate is an estimate, not a measured full-episode benchmark.
- The S2 runner is implemented as `run_series_a.py` → `eval_series_a.py` → `run_case.py`, using the existing LIBERO suite/init-state APIs and a websocket policy service. One server loads the model on one GPU and uses an ephemeral loopback port; `--workers N` runs N simulator environments concurrently while websocket inference remains serialized through that shared model. Each trial appends success, seed, init-state index/hash, selected M6 unit/task, retrieval scores, and t0 to JSONL. Completed trials are skipped on resume; failed trials are retried. The production matrix remains deferred until the final-check snapshot is supplied.

## Commands

From `/work` after `source env/activate.sh`:

```bash
python skill_moe/skillnet/examples/memory_pilot/anchor_flow/extract_prefix_keys.py \
  --input-npz /path/to/libero90_normalized_action_and_observation_rows.npz \
  --output-npz /work/.skillnet-state/outputs/memory-pilot/anchor-flow-<run-id>/libero90_prefix_cache.npz
python skill_moe/skillnet/examples/memory_pilot/anchor_flow/build_memory.py \
  --libero90-npz /work/.skillnet-state/outputs/memory-pilot/anchor-flow-<run-id>/libero90_prefix_cache.npz \
  --output /work/.skillnet-state/outputs/memory-pilot/anchor-flow-<run-id>/memory/m6.pkl
python skill_moe/skillnet/examples/memory_pilot/anchor_flow/add_task_memory.py \
  --memory /work/.skillnet-state/outputs/memory-pilot/anchor-flow-<run-id>/memory/m6.pkl \
  --task-npz /path/to/libero10_task_rows.npz \
  --output-memory /work/.skillnet-state/outputs/memory-pilot/anchor-flow-<run-id>/memory/after_task_00.pkl \
  --report /work/.skillnet-state/outputs/memory-pilot/anchor-flow-<run-id>/memory/add_task_00.json
python skill_moe/skillnet/examples/memory_pilot/anchor_flow/prepare_sources.py \
  --libero90-npz /path/to/libero90_normalized_prefix_cache.npz \
  --memory /work/.skillnet-state/outputs/memory-pilot/anchor-flow-<run-id>/memory/m6.pkl \
  --output /work/.skillnet-state/outputs/memory-pilot/anchor-flow-<run-id>/train_cache.npz
python skill_moe/skillnet/examples/memory_pilot/anchor_flow/train.py \
  --libero90-cache /work/.skillnet-state/outputs/memory-pilot/anchor-flow-<run-id>/train_cache.npz \
  --memory /work/.skillnet-state/outputs/memory-pilot/anchor-flow-<run-id>/memory/m6.pkl \
  --output /work/.skillnet-state/outputs/memory-pilot/anchor-flow-<run-id>/finetune
python skill_moe/skillnet/examples/memory_pilot/anchor_flow/smoke.py \
  --output /work/.skillnet-state/outputs/memory-pilot/anchor-flow-<run-id>/smoke
python skill_moe/skillnet/examples/memory_pilot/anchor_flow/rollout_smoke.py \
  --output /work/.skillnet-state/outputs/memory-pilot/anchor-flow-<run-id>/rollout-smoke \
  --task-id 0 --trials 2 --max-env-steps 5
python skill_moe/skillnet/examples/memory_pilot/anchor_flow/eval_series_a.py \
  --output /work/.skillnet-state/outputs/memory-pilot/anchor-flow-<run-id>/series-a-plan
```

The NPZ schemas are documented in each CLI's validation and `memory.py`; they are cache/export boundaries, not an invitation to mix in LIBERO-40 data. `smoke.py` loads the 90 checkpoint, checks the original-vs-`t0=1` output, checks anchored integration shapes and performs a few action-expert optimizer steps on synthetic tensors. `rollout_smoke.py` reuses the installed LIBERO suite, checkpoint normalization, and existing image/state transforms for two short task-0 episodes. Full series A and fine-tuning remain deferred.


### Closed-loop runner commands

From `/work` after `source env/activate.sh`, with a matching memory directory and fine-tuned checkpoint:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python skill_moe/skillnet/examples/memory_pilot/anchor_flow/run_series_a.py \
  --output /work/.skillnet-state/outputs/memory-pilot/anchor-flow-<run-id>/series-a \
  --memory-dir /path/to/libero90_m6_memory --checkpoint /path/to/final-check/params \
  --minimal --workers 2 --trials 20
# Resume the same case matrix and output directory:
XLA_PYTHON_CLIENT_PREALLOCATE=false python skill_moe/skillnet/examples/memory_pilot/anchor_flow/run_series_a.py \
  --output /work/.skillnet-state/outputs/memory-pilot/anchor-flow-<run-id>/series-a \
  --memory-dir /path/to/libero90_m6_memory --checkpoint /path/to/final-check/params \
  --minimal --workers 2 --trials 20 --resume
```

Baseline timing smoke (LIBERO-90 task 0, `anchor_disabled_t0_1`, three paired initial states, two environment workers) completed 3/3 successes in 128.5 seconds wall time; individual episode times were 31.9, 60.5, and 64.6 seconds over 63–69 environment steps. Applying the observed throughput directly to 1,840 episodes gives **21.9 hours**. This is a rough projection from one short task and includes first-case compilation; task complexity varies, so use it as planning data rather than a guaranteed upper bound. Trial rows and the timing calculation are in `/work/.skillnet-state/outputs/memory-pilot/anchor-flow-20261028T090000Z/base-timing/`.

`--query-text-source instruction_only` is the default. `pi05_generated_subtask` is an explicit alternative and requires a memory built with generated-subtask keys and the identical generator checkpoint; it loads the fixed plain π0.5 base generator at each server start. The available LIBERO-40-trained generator had unseen-suite exact match 0.276 (goal) and 0.228 (spatial), so it is not substituted for the primary instruction-only condition. A 6,000-step generator adaptation is estimated at 15–20 GPU minutes from the previous Split A/B runs, excluding preparation and validation; it is optional for this pilot.
