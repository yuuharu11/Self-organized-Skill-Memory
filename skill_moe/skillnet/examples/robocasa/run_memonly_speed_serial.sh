#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WT="$(cd "$HERE/../../../../" && pwd)"
STATE="${SKILLNET_STATE_DIR:-/work/.skillnet-state}"
OUT="${1:?usage: run_memonly_speed_serial.sh OUTPUT_DIR NH_CHECKPOINT PROGRESS_FILE}"
NH="${2:?missing NH checkpoint}"
PROGRESS="${3:?missing progress path}"
[[ ! -e "$OUT" ]] || { echo "Refusing existing output: $OUT" >&2; exit 2; }
SKILLPY="$STATE/envs/skillnet/bin/python"; ROBOPY="$STATE/envs/robocasa/bin/python"
CHECKPOINT="$STATE/checkpoints/robocasa365/pi05_pretrain_human300/multitask_learning/75000"
BASEMAG="$STATE/outputs/robocasa-memory/key-lite-01/magnitude/magnitude_head.pt"
TASKS="PickPlaceCounterToCabinet OpenDrawer CloseFridge"; PORT=8891
mkdir -p "$OUT" "$(dirname "$PROGRESS")"; OUT="$(realpath "$OUT")"
export SKILLNET_STATE_DIR="$STATE" SKILLNET_ROBOCASA_CHECKPOINT="$CHECKPOINT"
export SKILLNET_SUBTASK_GENERATE=0 XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$WT/skill_moe/skillnet/src:$WT/skill_moe/skillnet/packages/skillnet-client/src:$WT/skill_moe/skillnet/examples/robocasa:$WT/skill_moe/skillnet/examples/memory_pilot/robocasa${PYTHONPATH:+:$PYTHONPATH}"
printf '%s\n' 'conditions=x1p5,x2p0,nohist' "tasks=$TASKS" 'trials=12 seed=7 task_indices=0,2,3 task_stride=50 split=pretrain replan=5' 'GPU fallback: simultaneous .15 servers and a single .15 server OOMed with shared GPU occupancy; run serially at .20, 12 workers per condition.' "NH checkpoint=$NH; trained on 11314 memory rows, 9126 fit / 2188 Dev; same labels, history-free magnitude input." 'action expert calls=0; subtask generation disabled' > "$OUT/run_config.txt"
printf '%s\n' 'Fixed 36 trials per condition, seed 7+canonical_task_index*50+trial. X1.5/X2.0 are learned scale times multiplier then clip [.25,4]. NH scale head uses context+state+unit metadata only; retrieval, memory and gripper are unchanged.' 'The requested three-server run at fraction .15 failed RESOURCE_EXHAUSTED, and the single-server .15 fallback also failed while another process used the shared GPU. To finish within 1.5h without touching other jobs, this run serializes the three conditions at .20 (one server at a time), with 12 workers each.' > "$OUT/plan.txt"
mkdir -p "$OUT/nohist-training"; cp "$(dirname "$NH")/training.json" "$OUT/nohist-training/training.json"
export SKILLNET_ROBOCASA_VENV="$STATE/envs/robocasa"; source "$WT/env/robocasa/activate.sh"
export SKILLNET_SUBTASK_GENERATE=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.20 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$WT/skill_moe/skillnet/packages/skillnet-client/src:$WT/skill_moe/skillnet/examples/robocasa:$PYTHONPATH"
echo "evaluation_started=$(date -u +%FT%TZ)" > "$OUT/eval-status.txt"
"$ROBOPY" "$HERE/progress_memonly_speed.py" "$OUT" "$PROGRESS" >> "$PROGRESS" 2>&1
(while sleep 600; do "$ROBOPY" "$HERE/progress_memonly_speed.py" "$OUT" "$PROGRESS" >> "$PROGRESS" 2>&1 || true; done) & WATCH=$!
SERVER=""
cleanup() { [[ -n "$WATCH" ]] && kill "$WATCH" 2>/dev/null || true; if [[ -n "$SERVER" ]]; then kill -TERM "$SERVER" 2>/dev/null || true; wait "$SERVER" 2>/dev/null || true; fi; }
trap cleanup EXIT INT TERM
runcond() {
  local cond="$1" mult="$2" mag="$3"; mkdir -p "$OUT/$cond/eval"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.20 taskset -c 8 nice -n 10 "$SKILLPY" "$HERE/serve_memonly_policy.py" --host 127.0.0.1 --port "$PORT" --magnitude-checkpoint "$mag" --magnitude-multiplier "$mult" --query-log "$OUT/$cond/queries.jsonl" > "$OUT/$cond/server.log" 2>&1 & SERVER=$!
  local ready=0
  for _ in $(seq 1 600); do grep -q 'server listening' "$OUT/$cond/server.log" && { ready=1; break; }; kill -0 "$SERVER" 2>/dev/null || { tail -80 "$OUT/$cond/server.log" >&2; return 1; }; sleep 1; done
  (( ready == 1 )) || { echo "server startup timeout: $cond" >&2; return 1; }
  if ! setsid bash -c 'set -euo pipefail; label="$1"; port="$2"; py="$3"; main="$4"; out="$5"; tasks="$6"; read -r -a ts <<< "$tasks"; taskset -c 12-23 nice -n 10 timeout --signal=TERM 80m "$py" "$main" --host 127.0.0.1 --port "$port" --tasks "${ts[@]}" --split pretrain --num-trials 12 --replan-steps 5 --seed 7 --seed-task-stride 50 --seed-task-indices 0 2 3 --resize 224 --workers 12 --output "$out/$label/eval" > "$out/$label/eval.log" 2>&1' _ "$cond" "$PORT" "$ROBOPY" "$HERE/main.py" "$OUT" "$TASKS"; then echo "condition_failed=$cond at=$(date -u +%FT%TZ)" >> "$OUT/eval-status.txt"; return 1; fi
  kill -TERM "$SERVER" 2>/dev/null || true; wait "$SERVER" 2>/dev/null || true; SERVER=""
  "$ROBOPY" "$HERE/progress_memonly_speed.py" "$OUT" "$PROGRESS" >> "$PROGRESS" 2>&1
}
runcond x1p5 1.5 "$BASEMAG"
runcond x2p0 2.0 "$BASEMAG"
runcond nohist 1.0 "$NH"
kill "$WATCH" 2>/dev/null || true; WATCH=""
echo "evaluation_complete=$(date -u +%FT%TZ)" >> "$OUT/eval-status.txt"
"$SKILLPY" "$HERE/summarize_memonly_speed.py" "$OUT"
echo "complete=$(date -u +%FT%TZ)" > "$OUT/DONE"
PARENT_REPORT="$(dirname "$OUT")/report.md"; [[ -e "$PARENT_REPORT" ]] || cp "$OUT/report.md" "$PARENT_REPORT"
