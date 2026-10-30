# S2 anchor-flow experiments

This directory contains the LIBERO-90 memory builder, the S2 policy server,
offline diagnostics, and experiment runners. Start with the repository's
reproduction guide for environment setup; [DESIGN.md](DESIGN.md) records the
algorithm and the corrected anchor-training target.

## Entry points

Run scripts from the repository root after activating the project environment.
Each entry point supports `--help`; loading help does not launch training or evaluation.

| Stage | Scripts | Role |
| --- | --- | --- |
| Prepare | `export_libero90.py`, `extract_prefix_keys.py`, `build_memory.py`, `prepare_sources.py` | Export normalized rows, extract VLM keys, fit M1/M6, and prepare training sources |
| Train / serve | `train.py`, `smoke.py`, `serve_s2_policy.py`, `s2_policy.py` | Corrected anchor loss, model smoke, and policy serving |
| Evaluate | `run_case.py`, `eval_series_a.py`, `run_series_a.py` | Paired trial execution and task-condition matrices |
| Calibrate | `calibrate_s2.py`, `confidence_calibration.py` | Episode-held-out calibration of confidence, start time, and step count |
| Inspect keys | `key_check.py`, `key_check_extension.py` | Key pooling, centering, rankings, and action-draft diagnostics |
| Compare anchors | `anchor_v2_offline.py`, `offline_retrieval.py` | Paired frozen-policy comparison of centered-key drafts; retrieval shared with normal module imports |
| Learn keys | `train_retrieval_key.py`, `retrieval_key_validation_curve.py`, `sweep_retrieval_epsilon.py` | Train a small key projection, inspect validation curves, and replay unit-formation thresholds |
| Longer motion | `overnight_prep.py`, `seg_skill_offline.py` | Oracle controls, gripper segmentation, and phase-aligned trajectory reuse |
| Adapt drafts | `scene_adapter.py` | Experimental small scene adapters evaluated offline |

Examples for inspecting available parameters:

```bash
python skill_moe/skillnet/examples/memory_pilot/anchor_flow/anchor_v2_offline.py --help
python skill_moe/skillnet/examples/memory_pilot/anchor_flow/train_retrieval_key.py --help
python skill_moe/skillnet/examples/memory_pilot/anchor_flow/retrieval_key_validation_curve.py --help
```

## Required assets

The research scripts and tests are versioned. Their large inputs and outputs
remain separate assets:

| Asset | Required contents |
| --- | --- |
| Public LIBERO-90 checkpoint | Compatible `params/` and action/state normalization in `assets/` |
| Normalized row export | LIBERO-90-only actions `[N,10,32]`, state, images, prompt tokens, episode/init IDs, source filename, and frame IDs |
| Held-out observations | Rows separated from the memory episodes, plus the calibration partition and routing inputs used by the selected diagnostic |
| Query keys | One key per held-out row in the same order and from the same checkpoint/text source |
| Frozen memory | M1/M6 codec, entries, keys, raw actions, and normalization metadata |
| Projection / prior-run outputs | Additional inputs for threshold sweeps, segment comparisons, and validation curves; see each script's options |

Use `SKILLNET_STATE_DIR` to relocate experiment state. The shared path settings
in `examples/research_paths.py` resolve from the checkout when the variable is
unset. Historical run-directory names remain as defaults so recorded runs can
still be identified. Supply the actual input paths explicitly on another machine.
Older preparation scripts may have additional defaults; check their arguments
before running them. A successful `--help` check does not establish that the
required experiment assets are available.

For an offline comparison, set the following variables to the corresponding
local assets and choose a new output directory:

```bash
python skill_moe/skillnet/examples/memory_pilot/anchor_flow/anchor_v2_offline.py \
  --heldout "$S2_HELDOUT" --query-keys "$S2_QUERY_KEYS" \
  --memory "$S2_MEMORY" --checkpoint "$S2_PARAMS" --output "$S2_RUN_DIR"
```

The offline helpers reject queries with fewer than five eligible memory rows
after episode/init exclusion. They never fill missing candidates with excluded
demonstrations. The policy is frozen for this comparison; this command does not
start simulator rollouts.

## Interpreting the records

The corrected training target is `(x_t - A) / t`. Earlier fine-tuned S2 results
that used `epsilon - A` on imperfect-anchor rows are invalidated in the research
notes. The corrected implementation is available for new experiments; those
old trained weights are not repaired by changing the source.

For the recorded October 30 results and the stopped, incomplete closed-loop
measurement, see the [results review](../../../../../docs/research/reviews/experiment_results_2026-10-30.md).
The proposed scene-adapter and Optimus-R implementations have separate scopes;
shipping their source does not establish improved success rates.
