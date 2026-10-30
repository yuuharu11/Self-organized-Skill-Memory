# Frozen-VLM light go/no-go

For a fresh machine, follow the [reproduction guide](../../../../../docs/reproduction.md)
before running these commands. The full-episode LIBERO-40 dataset and feature
caches used by the research are separate assets; their schema and availability
are described there. The standard skill-sliced `libero_40_v1` dataset has
different episode boundaries and is not a replacement for that input.

## Script guide

| Module / directory | Purpose |
| --- | --- |
| `extract.py` | Cache frozen VLM features for the light experiments |
| `run.py` | Train and evaluate the initial bottleneck model |
| `heldout.py` | Evaluate memory insertion on held-out spatial tasks |
| `diagnose_sharing.py` | Diagnose whether transfer is limited by values or keys |
| `compare_keys.py`, `compare_language_keys.py` | Compare camera and language key representations |
| `skill_bench.py` | Compare the M0–M6 skill representations in `methods/` |
| `libero10_features.py`, `extract_libero10_features.py` | Read and extract LIBERO-10 features |
| [`intermediate/`](intermediate/README.md) | Compare intermediate actions, retrieval, and anchor-flow denoising |
| `runtime.py` | Shared RNG initialization and CPU thread limit for the light runners |
| `methods/_common.py` | Shared training-input conversion, numerical helpers, and memory operations |

Run a script as `python -m skill_moe.skillnet.examples.emergent.light.<module>`
from the repository root. Existing module names and command-line options are
preserved; the table distinguishes the experiments that reuse the same helpers.

## Initial experiment

Run from the repository root after installing and activating the project environment.
Place the full-episode dataset and π0.5 base parameters at the paths below, or
pass your own paths explicitly:

```bash
source env/activate.sh
python -m skill_moe.skillnet.examples.emergent.light.run \
  --dataset "$HF_LEROBOT_HOME/emergent/libero_40_unsegmented" \
  --checkpoint "$SKILLNET_STATE_DIR/checkpoints/pi05_base/params" \
  --cache-dir "$SKILLNET_STATE_DIR/cache/emergent-light" \
  --output-root "$SKILLNET_STATE_DIR/outputs/memory-pilot" \
  --vision-batch 2 --train-batch 256 --epochs 20 --seed 0
```

The run creates a unique `emergent-light-*` directory under
`$SKILLNET_STATE_DIR/outputs/memory-pilot/`. A persistent feature cache can be
provided with `--cache-dir`; completed per-episode feature files are reused. Do not
point it at an existing output run. For a quick plumbing check, `--max-episodes 12`
extracts a small prefix (not a valid go/no-go data sample).

The extractor selects only `image`, `wrist_image`, `state`, `actions`, `episode_index`,
`task_index`, and `frame_index` from each Parquet file. It never reads skill ID or
annotation columns. Only task indices 10–39 are included (spatial/object/goal; task
indices 0–9 are excluded). At every fifth frame, the base and wrist images are each run
through the frozen π0.5 base PaliGemma. Language-conditioned and empty-language passes
are separate; image token outputs are mean pooled and cached as float16.

For each H in 25 and 10, full executed action windows are used for training and posterior
reference rows. H-tail windows are omitted from model fitting and counted. Episodes are
split per task into training and held-out evaluation before fitting (whole episodes stay
on one side). The existing evaluator then makes its store/query split by held-out
initial-state ID. For each β in `{0.001, 0.01, 0.1}`, the learned objective is normalized
action MSE plus β times analytic KL(q‖p). The model is a 2-layer MLP Enc/Prior/Dec with
z_dim=32 and AdamW; no VLA or production z model is loaded. Cross-reconstruction calls
the learned decoder. Existing evaluator criteria i–iii are used unchanged.

The older crosstask evaluator requires `skills` and `skill_mask` arrays even for its
different-task condition. Generated evaluation NPZs supply a synthetic `[pick, place]`
proxy plan, with transitions inferred only from the executed gripper open/close trace.
No source skill or annotation columns are read; this does not represent LIBERO's
annotated subtasks. Treat only the different-task condition as meaningful evidence.

If runtime is too long for the allotted window, preserve the feature cache and rerun with
the same command after extraction. Feature extraction is the main cost: roughly 35k
sampled frames × two image views × two prefix passes for the full 30-task data (the
current cache contains 34,919 sampled frames across 1,314 episodes).

## Held-out spatial task memory check

Reuse the completed feature cache from the first run and run:

```bash
source env/activate.sh
python -m skill_moe.skillnet.examples.emergent.light.heldout \
  --dataset "$HF_LEROBOT_HOME/emergent/libero_40_unsegmented" \
  --checkpoint "$SKILLNET_STATE_DIR/checkpoints/pi05_base/params" \
  --cache-dir "$SKILLNET_STATE_DIR/cache/emergent-light" \
  --output-root "$SKILLNET_STATE_DIR/outputs/memory-pilot" \
  --seed 0
```

This trains only on libero_object/libero_goal (20 tasks), fits action/state/feature
normalizers and fixed memory PCA on those rows, then splits whole spatial episodes by
task into memory store and query. The spatial store is streamed through the existing
relative-distance threshold without changing model weights. Report output is placed in
a new `emergent-light-heldout-*` directory.
