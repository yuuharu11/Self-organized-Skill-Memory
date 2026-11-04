# b-prior bridge conditions

`bprior_direct` and `bprior_retrieve` use the fixed `best_bridge.json` expert
and the existing bridge with `t0=0.6`, `num_steps=10`, with no confidence gate
or fallback. The server never creates or loads the large VLM `TokenIndex`.

- `bprior_direct`: compute `(b_hat, r_hat) = codec.prior(pooled_live, state)`;
  decode `b_hat` with `r_hat` and bridge from that draft.
- `bprior_retrieve`: compute `b_hat` the same way, standardize it and the 6,583
  memory prior bases with memory-only mean/std, select the maximum cosine row,
  and decode that row's prior base with the current query's `r_hat`.

`serve.py` asserts that the mean of live prefix tokens with `groups >= 0`
matches the prepared `pooled.npy` feature for a cached dataset row. It also
compares `(b_hat, r_hat)` with a second `restore_codec(...).prior(...)` call,
which is the exact helper and input convention used by
`score_audit/keys.py::generate_codec`. These assertions are recorded in
`server_checks.json`. Prepared run-01 files are read-only.

## Current smoke (2 trials per condition)

The evaluator uses `run_case._run_trial` with the run-01 LIBERO-10 initial
states, task 0, initial-state indices 0 and 1, seed base 7, replan 5, and the
same 520-step cap. Results are append-only in `rollouts.jsonl`; `in_progress`,
`complete`, and `error` records are distinct. A rerun skips completed keys and
repeats incomplete/error keys while retaining earlier records.

```bash
cd /work
source env/activate.sh
tmux new-session -d -s bprior-long-01 \
  "cd /work && source env/activate.sh && OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 taskset -c 8-47 nice -n 10 python skill_moe/skillnet/examples/memory_pilot/pi05_memory/bprior/evaluate.py --port 8167 --task-ids 0 --trial-indices 0 1"
```

Attach with `tmux attach -t bprior-long-01`; logs are in
`/work/.skillnet-state/outputs/memory-pilot/pi05-memory40-20261104/bprior-long-01/server.log`.

## Exp full-run command

The code below resumes the smoke keys and schedules all 10 tasks by 10 public
initial states (200 condition trials total). Exp should use a new output
directory if retaining the smoke artifacts separately is preferred, updating
`OUTPUT` or passing `--output` explicitly. No evaluation result is overwritten
inside the run log; records are appended.

```bash
cd /work && source env/activate.sh
tmux new-session -d -s bprior-long-full \
  "cd /work && source env/activate.sh && OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 taskset -c 8-47 nice -n 10 python skill_moe/skillnet/examples/memory_pilot/pi05_memory/bprior/evaluate.py --output /work/.skillnet-state/outputs/memory-pilot/pi05-memory40-20261104/bprior-long-01 --port 8167 --task-ids 0 1 2 3 4 5 6 7 8 9 --trial-indices 0 1 2 3 4 5 6 7 8 9"
```

The server sets the allocator limits before JAX import. The single server
process restores the codec and prior-computes memory bases on CPU, keeping its
GPU use to the dense policy and bridge (well below the 50 GB cap in this smoke).
