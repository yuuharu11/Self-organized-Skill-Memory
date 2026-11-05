# PI05 memory score audit

All caches are read-only inputs from `pi05-memory40-20261104/run-01`. Outputs are
written only below `score-audit-01/`. No policy training or environment access is
used.

## Shared key format

Each `keys/<method>.npz` contains `query_rows[Q]`, `cand_rows[Q,K]` (original
global row IDs), `score[Q,K]` (descending, float32), `exclusion` (scalar string,
`task` or `episode`), and `description` (scalar string). Optional audit fields
are `score_group[Q,K,3]`, `token_maxsim[Q,K,712]` (float16; invalid query token
positions are NaN), `draft[Q,K,10,7]`, `dist[Q,K]`, `first_step_dist[Q,K]`,
`last_step_dist[Q,K]`, `cand_task[Q,K]`, and `cand_episode[Q,K]`. `score_group`
is ordered base/wrist/instruction. Each query row has `split`, `task`, and
`episode` in `query_meta.npz`; split comes from `run.calibration_episodes`.

`oracle.npz` stores `query_rows`, `memory_rows`, `action_mse[Q,M]` float16,
candidate task/episode metadata, and the three threshold inclusion masks for
0.005/0.01/0.02. Oracle distances use only normalized recorded active-7 action
chunks, never codec reconstruction.

## Reproduction

```bash
cd "$SKILLNET_REPO_ROOT" && source env/activate.sh
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
  taskset -c 8-47 nice -n 10 python skill_moe/skillnet/examples/memory_pilot/pi05_memory/score_audit/oracle.py --limit 64
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false taskset -c 8-47 nice -n 10 \
  python skill_moe/skillnet/examples/memory_pilot/pi05_memory/score_audit/make_vlm_keys.py --limit 64 --block 128
```

The full run omits `--limit`. It writes all keys and a common report via
`evaluate_keys.py`. The VLM cache uses exact grouped token MaxSim as in
`run.retrieve`: token-wise L2 normalization, candidate-token maximum, mean over
query tokens per modality, then mean over three modalities; candidates from the
query's own episode are excluded for the exact current baseline.

## Availability and costs

All VLM keys here are inference-time available from the same cached prefix
tokens. Group-only/aggregation/centering are deterministic postprocessing; no
labels or future actions enter them. Pooled keys are also inference-time
available. Exact token MaxSim keeps the 6,583 x 712 x 2,048 fp16 bank resident
(about 19.3 GiB) and runs chunked matrix products; the process caps its PyTorch
CUDA allocator to 24% of device memory. Centered/diagonal-whitened variants use
memory-only group statistics (whitening statistics are estimated from a fixed
sample of up to 256 memory rows); these add one transformed bank allocation
while constructing the index, within the same cap. The smoke report records
measured cost and full-run extrapolation before any full query run.
