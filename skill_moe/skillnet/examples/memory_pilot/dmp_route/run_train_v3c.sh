#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
OUT="${SKILLNET_STATE_DIR:-$REPO_ROOT/.skillnet-state}/outputs/memory-pilot/dmp-route-20261102/s3-train-v3c"
LOG="$OUT/train_runner.log"
THREADS=4
MICROBATCH=8

run_condition() {
  local condition="$1"
  local ok=0
  local attempt
  for attempt in 10 11 12; do
    local attempt_dir="$OUT/attempts/${condition}-attempt-${attempt}"
    if [[ -e "$attempt_dir/metrics.jsonl" ]]; then
      printf 'utc=%s event=attempt_path_occupied condition=%s attempt=%s\n' \
        "$(date -u +%FT%TZ)" "$condition" "$attempt" >> "$LOG"
      continue
    fi
    printf 'utc=%s event=train_start condition=%s attempt=%s logical_batch=32 microbatch=%s\n' \
      "$(date -u +%FT%TZ)" "$condition" "$attempt" "$MICROBATCH" >> "$LOG"
    set +e
    timeout --signal=TERM --kill-after=120s 21600s \
      nice -n 10 taskset -c 14-47 env \
      OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" \
      XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 \
      XLA_PYTHON_CLIENT_ALLOCATOR=platform \
      python -u examples/memory_pilot/dmp_route/train_s3.py \
        --condition "$condition" --attempt "$attempt" --steps 1500 \
        --threads "$THREADS" --microbatch-size "$MICROBATCH" >> "$LOG" 2>&1
    local status=$?
    set -e
    if [[ $status -eq 0 ]] && grep -q '"event": "complete"' "$attempt_dir/metrics.jsonl"; then
      printf 'utc=%s event=train_complete condition=%s attempt=%s\n' \
        "$(date -u +%FT%TZ)" "$condition" "$attempt" >> "$LOG"
      ok=1
      break
    fi
    printf 'utc=%s event=train_retry condition=%s attempt=%s status=%s\n' \
      "$(date -u +%FT%TZ)" "$condition" "$attempt" "$status" >> "$LOG"
  done
  [[ $ok -eq 1 ]]
}

cd "$REPO_ROOT"
source env/activate.sh >/dev/null
cd "$REPO_ROOT/skill_moe/skillnet"
run_condition G1-dmp
run_condition G0
printf 'utc=%s event=both_training_complete\n' "$(date -u +%FT%TZ)" >> "$LOG"
