#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WT="$(cd "$HERE/../../../../" && pwd)"
STATE="${SKILLNET_STATE_DIR:-/work/.skillnet-state}"
OUT="${1:?usage: run_memonly_speed_nohist.sh ATTEMPT_DIR NH_CHECKPOINT PROGRESS_FILE}"
NH="${2:?missing NH checkpoint}"
PROGRESS="${3:?missing progress path}"
SKILLPY="$STATE/envs/skillnet/bin/python"; ROBOPY="$STATE/envs/robocasa/bin/python"
CHECKPOINT="$STATE/checkpoints/robocasa365/pi05_pretrain_human300/multitask_learning/75000"
[[ -d "$OUT" && ! -e "$OUT/nohist" ]] || { echo "Expected attempt dir and absent nohist result directory" >&2; exit 2; }
export SKILLNET_STATE_DIR="$STATE" SKILLNET_ROBOCASA_CHECKPOINT="$CHECKPOINT"
export SKILLNET_SUBTASK_GENERATE=0 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.20
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$WT/skill_moe/skillnet/src:$WT/skill_moe/skillnet/packages/skillnet-client/src:$WT/skill_moe/skillnet/examples/robocasa:$WT/skill_moe/skillnet/examples/memory_pilot/robocasa${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUT/nohist/eval"
taskset -c 8 nice -n 10 "$SKILLPY" "$HERE/serve_memonly_policy.py" --host 127.0.0.1 --port 8891 \
  --magnitude-checkpoint "$NH" --magnitude-multiplier 1.0 --query-log "$OUT/nohist/queries.jsonl" \
  > "$OUT/nohist/server.log" 2>&1 & SERVER=$!
cleanup() { kill -TERM "$SERVER" 2>/dev/null || true; wait "$SERVER" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
ready=0
for _ in $(seq 1 600); do grep -q 'server listening' "$OUT/nohist/server.log" && { ready=1; break; }; kill -0 "$SERVER" 2>/dev/null || { tail -80 "$OUT/nohist/server.log" >&2; exit 1; }; sleep 1; done
(( ready == 1 )) || { echo 'NH server startup timeout' >&2; exit 1; }
export SKILLNET_ROBOCASA_VENV="$STATE/envs/robocasa"; source "$WT/env/robocasa/activate.sh"
export SKILLNET_SUBTASK_GENERATE=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$WT/skill_moe/skillnet/packages/skillnet-client/src:$WT/skill_moe/skillnet/examples/robocasa:$PYTHONPATH"
taskset -c 12-23 nice -n 10 timeout --signal=TERM 80m "$ROBOPY" "$HERE/main.py" \
  --host 127.0.0.1 --port 8891 --tasks PickPlaceCounterToCabinet OpenDrawer CloseFridge \
  --split pretrain --num-trials 12 --replan-steps 5 --seed 7 --seed-task-stride 50 \
  --seed-task-indices 0 2 3 --resize 224 --workers 12 --output "$OUT/nohist/eval" \
  > "$OUT/nohist/eval.log" 2>&1
echo "evaluation_complete=$(date -u +%FT%TZ)" >> "$OUT/eval-status.txt"
"$SKILLPY" "$HERE/summarize_memonly_speed.py" "$OUT"
echo "complete=$(date -u +%FT%TZ)" > "$OUT/DONE"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$ROBOPY" "$HERE/progress_memonly_speed.py" "$OUT" "$PROGRESS"
