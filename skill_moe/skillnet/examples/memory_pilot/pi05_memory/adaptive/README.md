# Adaptive noise and fallback: LIBERO-Long

The two conditions reuse `pi05_memory/serve.py` for the official π0.5 policy,
VLM token MaxSim Top-1 retrieval, memory draft, and the `best_bridge.json`
expert. Both use ten bridge steps. The only policy change is the fixed score
schedule below. A score is a retrieval similarity, **not** a probability that
the selected action is correct. The thresholds are the pre-fixed 20/60/80%
quantiles from run-01's LIBERO-Long memory rollouts and were not selected from
success outcomes.

| score | `adaptive` | `adaptive_fb` |
|---|---:|---:|
| `< 0.92968` | t0=0.6 | ordinary π0.5 fallback |
| `0.92968–<0.94267` | t0=0.6 | t0=0.6 |
| `0.94267–<0.95002` | t0=0.4 | t0=0.4 |
| `≥ 0.95002` | t0=0.2 | t0=0.2 |

The offline results showed t0=0.2 was usually worse than t0=0.6 and similarity
was correlated with draft error in the unexpected direction. This is a
closed-loop hypothesis check, not a performance-promising configuration.

`serve.py` makes the JAX random key directly from each request's
`s2_rng_seed`. `anchor_flow/run_case.py` derives that seed from the trial seed
and query step; no server-global random stream or request-arrival order is
used. The server writes a locked `action_trace.jsonl` with the score, selected
mode, t0/fallback, memory row, action hash, and preprocessing/VLM/search/draft/
denoise/output-transform times. The evaluator reports environment plus
websocket time as the per-trial wall time minus summed inference time.

## Smoke and serial/parallel determinism check

This runs task 0, initial states 0 and 1 for each condition serially, then
repeats those same two trials twice concurrently per server (4 simulator
workers/server). It compares every action chunk's SHA-256. It is a smoke only;
the duplicate parallel trials are not counted as additional evaluation rows.

```bash
cd "$SKILLNET_REPO_ROOT"
source env/activate.sh
OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 \
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.12 \
ADAPTIVE_TORCH_MEM_FRACTION=0.20 \
taskset -c 8-47 nice -n 10 python \
  skill_moe/skillnet/examples/memory_pilot/pi05_memory/adaptive/evaluate.py \
  --output .skillnet-state/outputs/memory-pilot/pi05-memory40-20261104/adaptive-long-01/smoke-run2 \
  --smoke --workers-per-server 4 --timeout 600
```

All records are append-only in `rollouts.jsonl`; each attempt gets a unique
directory. Completed keys are skipped on restart. An infrastructure error or
600-second timeout is recorded and retried once. The run writes
`parallel_consistency.json`, `closedloop_summary.json`, and `report.md`.

## Exp's full 100-trial-per-condition run

Use a separate output subdirectory so the smoke data remain intact. The
controller starts two policy servers on distinct free ports and runs at most
four simulator workers against each. Eight workers × two CPU threads run on
cores 20–35; the servers each use a disjoint six-core set (8–13 and 14–19),
and the controller uses at most three threads. The peak is 31 threads/cores.
JAX is capped at 12% of GPU memory and
the server's Torch allocator at 20%; the combined allocator budget is below
50 GB per policy server. The controller handles resume, per-trial timeouts,
one technical retry, matched-baseline summaries, and clean server shutdown.

```bash
cd "$SKILLNET_REPO_ROOT"
source env/activate.sh
tmux new-session -d -s adaptive-long-full \
  "cd "$SKILLNET_REPO_ROOT" && source env/activate.sh && OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.12 ADAPTIVE_TORCH_MEM_FRACTION=0.20 taskset -c 8-47 nice -n 10 python skill_moe/skillnet/examples/memory_pilot/pi05_memory/adaptive/evaluate.py --output .skillnet-state/outputs/memory-pilot/pi05-memory40-20261104/adaptive-long-01/full --workers-per-server 4 --timeout 600"
```

Inspect with `tmux attach -t adaptive-long-full`. The chosen ports are written
to `experiment.json`; specify `--ports PORT_ADAPTIVE PORT_FALLBACK` if local
port policy requires fixed ports. `nvidia-smi` should be checked before
starting, and the per-server peaks should remain under 50 GB.
