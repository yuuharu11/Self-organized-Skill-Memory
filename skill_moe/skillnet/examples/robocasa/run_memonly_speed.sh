#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WT="$(cd "$HERE/../../../../" && pwd)"
STATE="${SKILLNET_STATE_DIR:-/work/.skillnet-state}"
OUT="${1:?usage: run_memonly_speed.sh OUTPUT_DIR}"
(( $# == 1 )) || exit 2
[[ ! -e "$OUT" ]] || { echo "Refusing existing output: $OUT" >&2; exit 2; }
SKILLPY="$STATE/envs/skillnet/bin/python"; ROBOPY="$STATE/envs/robocasa/bin/python"
CHECKPOINT="$STATE/checkpoints/robocasa365/pi05_pretrain_human300/multitask_learning/75000"
BASEMAG="$STATE/outputs/robocasa-memory/key-lite-01/magnitude/magnitude_head.pt"
TASKS=(PickPlaceCounterToCabinet OpenDrawer CloseFridge)
mkdir -p "$OUT"; OUT="$(realpath "$OUT")"
PROGRESS="${MEMONLY_SPEED_PROGRESS:-$OUT/progress.md}"
mkdir -p "$(dirname "$PROGRESS")"
[[ -x "$SKILLPY" && -x "$ROBOPY" ]] || { echo "Missing venv" >&2; exit 2; }
for port in 8891 8892 8893; do
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$SKILLPY" - "$port" <<'PY'
import socket, sys
s=socket.socket(); s.settimeout(.2)
try: s.connect(("127.0.0.1", int(sys.argv[1])))
except OSError: pass
else: raise SystemExit(f"Port occupied: {sys.argv[1]}")
finally: s.close()
PY
done
export SKILLNET_STATE_DIR="$STATE" SKILLNET_ROBOCASA_CHECKPOINT="$CHECKPOINT"
export SKILLNET_SUBTASK_GENERATE=0 XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$WT/skill_moe/skillnet/src:$WT/skill_moe/skillnet/packages/skillnet-client/src:$WT/skill_moe/skillnet/examples/robocasa:$WT/skill_moe/skillnet/examples/memory_pilot/robocasa${PYTHONPATH:+:$PYTHONPATH}"
cat > "$OUT/run_config.txt" <<CFG
conditions=x1p5,x2p0,nohist
tasks=${TASKS[*]}
trials=12 seed=7 task_indices=0,2,3 task_stride=50 split=pretrain replan=5
workers=36 (12 per condition); worker cpus 12-47; server cpus 8-10
XLA_PYTHON_CLIENT_PREALLOCATE=false; server MEM_FRACTION=0.15 each
X1.5/X2.0 use existing history-aware magnitude times multiplier then clip [.25,4]. NH uses memory-side fit and omits action history from magnitude input.
retrieval/gripper/memory unchanged; action expert calls=0; subtask generation disabled
CFG
cat > "$OUT/plan.txt" <<'PLAN'
Question: test whether increasing magnitude alone recovers slow memory-only behavior, and whether a history-free magnitude model avoids a small-scale feedback loop.
Fixed: three servers, each evaluates three tasks x 12 trials; seed=7+canonical_task_index*50+trial; split=pretrain; replan=5; 12 workers per condition (36 total).
Only magnitude differs: existing scale x1.5; existing scale x2.0; newly fit history-free magnitude MLP. Retrieval remains Context+history; gripper and memory units unchanged.
No π0.5 action head calls. Baseline action-norm comparison is made only if an existing action trace is available.
PLAN
export SKILLNET_SUBTASK_GENERATE=0 XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
taskset -c 24-31 nice -n 10 timeout --signal=TERM 8m "$SKILLPY" "$HERE/train_memonly_magnitude_nohist.py" --output "$OUT/nohist-training" > "$OUT/nohist-training.log" 2>&1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.15 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
ports=(8891 8892 8893); conds=(x1p5 x2p0 nohist); mult=(1.5 2.0 1.0); pids=(); clients=(); watcher=""
cleanup() { [[ -n "$watcher" ]] && kill "$watcher" 2>/dev/null || true; for p in "${clients[@]}"; do kill -TERM -- "-$p" 2>/dev/null || true; done; for p in "${clients[@]}"; do wait "$p" 2>/dev/null || true; done; for p in "${pids[@]}"; do kill -TERM "$p" 2>/dev/null || true; done; for p in "${pids[@]}"; do wait "$p" 2>/dev/null || true; done; }
trap cleanup EXIT INT TERM
for i in 0 1 2; do
  c="${conds[$i]}"; port="${ports[$i]}"; mkdir -p "$OUT/$c/eval"; mag="$BASEMAG"
  [[ "$c" != nohist ]] || mag="$OUT/nohist-training/magnitude_head_nohist.pt"
  taskset -c "$((8+i))" nice -n 10 "$SKILLPY" "$HERE/serve_memonly_policy.py" --host 127.0.0.1 --port "$port" --magnitude-checkpoint "$mag" --magnitude-multiplier "${mult[$i]}" --query-log "$OUT/$c/queries.jsonl" > "$OUT/$c/server.log" 2>&1 & pids+=("$!")
done
for i in 0 1 2; do
  ready=0
  for _ in $(seq 1 600); do grep -q 'server listening' "$OUT/${conds[$i]}/server.log" && { ready=1; break; }; kill -0 "${pids[$i]}" 2>/dev/null || { tail -80 "$OUT/${conds[$i]}/server.log" >&2; exit 1; }; sleep 1; done
  (( ready == 1 )) || { echo "Server startup timeout ${conds[$i]}" >&2; exit 1; }
done
export SKILLNET_ROBOCASA_VENV="$STATE/envs/robocasa"; source "$WT/env/robocasa/activate.sh"
export SKILLNET_SUBTASK_GENERATE=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$WT/skill_moe/skillnet/packages/skillnet-client/src:$WT/skill_moe/skillnet/examples/robocasa:$PYTHONPATH"
"$SKILLPY" "$HERE/progress_memonly_speed.py" "$OUT" >> "$PROGRESS" 2>&1
(while sleep 600; do "$ROBOPY" "$HERE/progress_memonly_speed.py" "$OUT" >> "$PROGRESS" 2>&1 || true; done) & watcher=$!
cpus=(12-23 24-35 36-47)
for i in 0 1 2; do
  c="${conds[$i]}"; port="${ports[$i]}"; cpu="${cpus[$i]}"
  setsid bash -c 'set -euo pipefail; label="$1"; port="$2"; cpus="$3"; out="$4"; main="$5"; tasks="$6"; read -r -a ts <<< "$tasks"; taskset -c "$cpus" nice -n 10 timeout --signal=TERM 80m "$main" --host 127.0.0.1 --port "$port" --tasks "${ts[@]}" --split pretrain --num-trials 12 --replan-steps 5 --seed 7 --seed-task-stride 50 --seed-task-indices 0 2 3 --resize 224 --workers 12 --output "$out/$label/eval" > "$out/$label/eval.log" 2>&1' _ "$c" "$port" "$cpu" "$OUT" "$ROBOPY" "${TASKS[*]}" & clients+=("$!")
done
failed=0; for p in "${clients[@]}"; do wait "$p" || failed=1; done; clients=()
[[ -n "$watcher" ]] && kill "$watcher" 2>/dev/null || true; watcher=""
"$ROBOPY" "$HERE/progress_memonly_speed.py" "$OUT" >> "$PROGRESS" 2>&1
if (( failed )); then echo "evaluation_partial=$(date -u +%FT%TZ)" > "$OUT/eval-status.txt"; "$SKILLPY" "$HERE/summarize_memonly_speed.py" "$OUT" || true; exit 1; fi
echo "evaluation_complete=$(date -u +%FT%TZ)" > "$OUT/eval-status.txt"
"$SKILLPY" "$HERE/summarize_memonly_speed.py" "$OUT"
echo "complete=$(date -u +%FT%TZ)" > "$OUT/DONE"
PARENT_REPORT="$(dirname "$OUT")/report.md"
if [[ ! -e "$PARENT_REPORT" ]]; then cp "$OUT/report.md" "$PARENT_REPORT"; fi
