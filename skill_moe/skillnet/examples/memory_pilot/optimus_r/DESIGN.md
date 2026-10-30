# Optimus-R Stage A: independent openpi reimplementation

**This is an independent reimplementation based on the paper, not the authors' code or a claim of exact reproduction.** Optimus-R code/checkpoints were not located in the research review. Existing `pi0*.py` and the S2 anchor-flow path are not changed; this path subclasses the SkillNet π0.5 model and adds separate parameters.

## Scope and local data

- Initialize from `.skillnet-state/checkpoints/pi05_libero_moe_skill_4_90/moe_balance_4_32_90_20000/19999/params`. No LIBERO-40 checkpoint/data is accepted.
- Reuse `.skillnet-state/outputs/memory-pilot/anchor-s2-train-20261028T072152Z/libero90_rows.npz`. It has 55,660 complete nonoverlapping H=10 rows, 3,959 episodes and 89 tasks, with normalized actions `[N,10,32]`, normalized state `[N,32]`, base/wrist images, instructions, episode IDs, source task files, and sampled frame offsets. The checker notes one missing LIBERO-90 task file. The cache is sufficient to train and build a bank without recomputing image features.
- The export does not explicitly store official task IDs or true episode lengths. Task ID is recovered by the exact `source_file` task filename via the local official LIBERO-90 suite. Instruction lookup was rejected because several scenes reuse identical instruction strings. Progress uses `frame/(last retained frame + sampling stride)` and is an approximation because only stride-10 complete chunks were retained. Within-trajectory window `w=6` uses retained-row indices (up to 60 environment frames). This preserves the paper's pair rule in the available sample units but is not frame-exact.
- The S2 export has an all-false right-wrist mask; this new path omits those masked image tokens (identical attention semantics, one fewer redundant SigLIP pass). The checkpoint-compatible S2 tokenizer has `discrete_state_input=False`; its prompt keys contain image and instruction, not state tokens. To honor Optimus-R's `O_t=(image,state)` input, this reimplementation adds one learned projection of the normalized 32-D state to the prefix. That state token is new and Stage-A trainable. The unchanged SkillNet task-skill IDs still feed its existing MoE route; original Optimus-R uses no such SkillNet routing labels, so this remains a stated backbone-specific difference.

## Stage A algorithm and fixed bank

1. Append four learned memory seed tokens after the regular image/instruction prefix and the new normalized-state token. Run the normal SkillNet PaliGemma stream and take the final-normalized outputs at those four positions as `M_t`. No future action or denoising suffix is read by the memory encoders.
2. Use independent attention poolers for query and skill, each with learned `u_b` and full `W_b`, then independent `2048→512→256` GELU encoders. L2-normalize query vectors; keep skill vectors unnormalized. The widths and MLP depth are implementation choices because the paper does not specify them.
3. Decode the skill vector with a `256→512→320` MLP to the normalized `[10,32]` action chunk. Train with Huber delta 1.0. Query NCE uses temperature 0.07. Positives are exactly the union of same task with progress distance ≤0.1 and same episode with retained-row distance ≤6; self-pairs are removed and anchors with no positive are skipped. When the batch has no valid positive, query loss is zero.
4. The joint loss is `L = L_flow + 0.1 L_query_NCE + 1.0 L_skill_Huber`. Task ID and progress are used for the paper's query contrastive objective; task IDs are not added to the query itself. Stage A trains all new modules, the action expert, and the last four PaliGemma layers. Four is a chosen completion of the paper's unspecified “last layers”. New modules use LR `2e-4`, VLM/action backbone LR `2e-5`. Adam without weight decay is used because optimizer details are absent; earlier VLM layers are gradient-masked to zero.
5. Project the current or retrieved 256-D skill vector to four 2048-D prefix tokens and append those tokens before the flow action suffix. This is a distinct prefix-token route; it does not inject the value into the existing MoE `skill_embeds` router. At inference, encode the current query, cosine-rank all bank keys, select top-2, softmax their scores at fixed `τ_mem=0.1`, average the two 256-D values, project to tokens, and run the usual flow policy. `τ_mem` is not reported by the paper and is a fixed Stage-A completion, not tuned on target tasks.
6. Build the fixed 128-entry initial bank after Stage A. Use deterministic cosine k-means for 64 coarse query clusters, then repeatedly bisect the cluster with largest skill-space variance along its dominant skill-variance axis until there are exactly 128 entries. Each stores normalized mean query, mean skill value, support count, and up to three latent exemplar pairs. Clustering algorithm, split rule, and 64→128 schedule are not specified in the paper; this is the documented completion. No online add/reuse, trainable memory residuals, Stage B/C, adapter, or target data are in this first tranche.

The original paper's general setting reports batch 256 on 8×A800 and 30k steps, but does not identify Stage-A-only step count or a one-H200 batch. The implementation defaults to 30k updates and batch 32 on one H200 as the intended S2 plan, not a matched global-batch reproduction. Batch 32 was not validated in this shared-GPU smoke; only batch 1 training and batch 4 bank extraction were measured. Target LIBERO data are not used.

## Files and commands

- Model/config: `src/openpi/models/pi0_optimus_r.py`, `pi0_optimus_r_config.py`.
- Stage A: `train_stage_a.py` consumes the S2 cache directly and writes a new checkpoint directory.
- Bank: `build_initial_bank.py` extracts Stage-A query/skill vectors and writes a fixed bank. It performs feature extraction on the GPU; the 128-cluster procedure itself uses CPU arrays.
- `bank.py` contains the deterministic prototype builder and retrieval utilities.

From `/work`, commands use the shared-resource envelope below. Pick a never-used output directory for each run. The `--steps 20 --batch-size 1` command is a short Stage-A smoke; production training is intentionally not launched in this task.

```bash
cd "$SKILLNET_REPO_ROOT" && source env/activate.sh
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.35
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
nice -n 10 taskset -c 8-47 python skill_moe/skillnet/examples/memory_pilot/optimus_r/train_stage_a.py \
  --libero90-cache .skillnet-state/outputs/memory-pilot/anchor-s2-train-20261028T072152Z/libero90_rows.npz \
  --output .skillnet-state/outputs/memory-pilot/optimus-r-<run-id>/stage-a-smoke \
  --steps 20 --batch-size 1 --memory-fraction 0.35
nice -n 10 taskset -c 8-47 python skill_moe/skillnet/examples/memory_pilot/optimus_r/build_initial_bank.py \
  --libero90-cache .skillnet-state/outputs/memory-pilot/anchor-s2-train-20261028T072152Z/libero90_rows.npz \
  --checkpoint /path/to/stage-a/params \
  --output .skillnet-state/outputs/memory-pilot/optimus-r-<run-id>/initial-bank
```

The bank extractor accepts `--max-rows` only for a labelled smoke bank; production bank construction uses all allowed LIBERO-90 rows. Both scripts use exclusive-create output directories and never replace existing runs.

## Smoke measurements (2026-10-28)

- Stage-A training smoke: `--steps 20 --batch-size 1 --memory-fraction 0.42` completed and saved a checkpoint. The first update (including compilation) took 13.27 s; the median of updates 2–20 was 0.185 s/update (mean 0.237 s). Peak JAX allocation was 52.66 GiB under a 58.72 GiB pool limit. The 105.1 s total wall time includes checkpoint serialization; the update-only 30k projection is 1.54 GPU-hours at batch 1, excluding startup, validation, and checkpoint saves.
- The batch-1 smoke has no valid in-batch query positives (`valid_query_anchors=0`, `query_nce=0`), so it verifies the model/optimizer/flow and skill reconstruction execution, not effective NCE learning. Contrastive-positive rules and the nonzero NCE path are covered by CPU unit tests. A batch-2 GPU attempt did not complete: allocation of an additional 4.5 GiB failed after total shared GPU use rose to about 130/144 GiB. Batch-32 runtime and peak memory remain unmeasured. Thus 1.54 h is a batch-1 update-count reference, not a batch-32 promise.
- Fixed-bank smoke: encoded 256 cached rows at batch 4 and produced exactly 128 query and skill prototypes (`[128,256]` each). Feature extraction took 116.83 s and CPU clustering/splitting took 0.10 s. Linear scaling to all 55,660 rows gives 7.06 GPU-hours for feature extraction; this rough estimate uses a small sample on a shared GPU and excludes model/checkpoint/cache loading. The combined A30k + full-bank reference is about 8.6 GPU-hours at the measured batch-1 training and batch-4 extraction settings. Full bank construction was not run.
- The first bank smoke exposed that Stage-A checkpoints are bare NNX states while the base SkillNet loader expects an openpi `params` wrapper. `build_initial_bank.py` now restores these Stage-A states against the initialized target tree. The successful batch-4 run used memory fraction 0.25 after shared GPU usage subsided; an earlier attempt at the same batch failed while the GPU was more heavily occupied.

Smoke artifacts are under `.skillnet-state/outputs/memory-pilot/optimus-r-stage-a-20261028T175200Z/`. Successful runs are `stage-a-smoke-cap42/` and `bank-smoke-256-batch4/`; the latter is a labeled 256-row smoke subset, not the production bank.
