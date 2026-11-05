#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../../.."
source env/activate.sh
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export XLA_PYTHON_CLIENT_PREALLOCATE=false

PI05_ROOT="${SKILLNET_PI05_ROOT:-$SKILLNET_STATE_DIR/outputs/memory-pilot/pi05-memory40-20261104}"
RUN="$PI05_ROOT/run-01"
OUT="$PI05_ROOT/skill-expert-01"
CODE=skill_moe/skillnet/examples/memory_pilot/pi05_memory/skill_expert/run.py
EVAL=skill_moe/skillnet/examples/memory_pilot/pi05_memory/skill_expert/evaluate.py
CKPT="$SKILLNET_STATE_DIR"/checkpoints/openpi-assets/checkpoints/pi05_libero
EXPERT="$OUT/production-expert"

test -f "$OUT/draft_sources.npz"
test ! -e "$EXPERT/best_bridge.json"
mkdir -p "$EXPERT"

# The row sampler, seed, optimizer, and 4,000-update ceiling match run-01.
XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 timeout --signal=TERM --kill-after=60s 3h \
  taskset -c 8-15 nice -n 10 python -u "$CODE" train --run "$RUN" --output "$EXPERT" \
  --draft-dir "$OUT" --checkpoint "$CKPT" --seed 20261104 --steps 4000 --batch 8 --save-every 500 \
  > "$OUT/production-train.log" 2>&1

# Full 1,318-row paired offline evaluation; fixed noise shared by all conditions.
XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 timeout --signal=TERM --kill-after=60s 2h \
  taskset -c 8-15 nice -n 10 python -u "$CODE" offline --run "$RUN" --output "$OUT/offline-full" \
  --expert-dir "$EXPERT" --draft-dir "$OUT" --checkpoint "$CKPT" --batch 4 \
  > "$OUT/production-offline.log" 2>&1

# Confidence boundaries are derived only from fixed Dev rows, never success labels.
taskset -c 8-15 nice -n 10 python "$CODE" calibrate --run "$RUN" --output "$EXPERT" --draft-dir "$OUT"

# C/A/D use the same task/init pairs. Three small-memory policy servers run in
# parallel; each is capped to 10% JAX + 10% Torch memory and two CPU threads.
# evaluate.py allocates an available loopback port for each server.
mkdir -p "$OUT/production-logs"
pids=()
for cond in C A D; do
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.10 \
    timeout --signal=TERM --kill-after=60s 3h taskset -c 32-39 nice -n 10 \
    python -u "$EVAL" --output "$OUT/closedloop-$cond" --expert "$EXPERT" --run "$RUN" \
      --checkpoint "$CKPT" --condition "$cond" --tasks {0..9} --trials {0..9} \
      --workers 2 --server-threads 2 --timeout 600 --mem-fraction .10 --torch-fraction .10 \
      > "$OUT/production-logs/closedloop-$cond.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
if [[ "$status" != 0 ]]; then exit "$status"; fi

# Optional B (C schedule without the low-confidence fallback), after C/A/D.
if [[ "${RUN_B:-0}" == 1 ]]; then
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.10 \
    timeout --signal=TERM --kill-after=60s 3h taskset -c 32-39 nice -n 10 \
    python -u "$EVAL" --output "$OUT/closedloop-B" --expert "$EXPERT" --run "$RUN" \
      --checkpoint "$CKPT" --condition B --tasks {0..9} --trials {0..9} \
      --workers 2 --server-threads 2 --timeout 600 --mem-fraction .10 --torch-fraction .10 \
      > "$OUT/production-logs/closedloop-B.log" 2>&1
fi
