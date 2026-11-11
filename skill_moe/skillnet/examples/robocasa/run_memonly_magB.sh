#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WT="$(cd "$HERE/../../../../" && pwd)"
STATE="${SKILLNET_STATE_DIR:-/work/.skillnet-state}"
OUT="${1:?usage: run_memonly_magB.sh OUTPUT_DIR}"
MAG="${2:-$STATE/outputs/robocasa-memory/key-lite-01/magnitude_v2/magnitude_head.pt}"
MULTIPLIER="${3:-1.0}"
[[ ! -e "$OUT" ]] || { echo "Refusing existing output: $OUT" >&2; exit 2; }
[[ -f "$MAG" ]] || { echo "Missing magnitude checkpoint: $MAG" >&2; exit 2; }
SKILLPY="$STATE/envs/skillnet/bin/python"
ROBOPY="$STATE/envs/robocasa/bin/python"
CHECKPOINT="$STATE/checkpoints/robocasa365/pi05_pretrain_human300/multitask_learning/75000"
mkdir -p "$OUT/eval"; OUT="$(realpath "$OUT")"
export SKILLNET_STATE_DIR="$STATE" SKILLNET_ROBOCASA_CHECKPOINT="$CHECKPOINT"
export SKILLNET_SUBTASK_GENERATE=0 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.20
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$WT/skill_moe/skillnet/src:$WT/skill_moe/skillnet/packages/skillnet-client/src:$HERE:$WT/skill_moe/skillnet/examples/memory_pilot/robocasa${PYTHONPATH:+:$PYTHONPATH}"
PORT="${PORT:-8894}"
"$SKILLPY" - "$PORT" <<'PY'
import socket,sys
s=socket.socket();s.settimeout(.2)
try:s.connect(("127.0.0.1",int(sys.argv[1])))
except OSError:pass
else:raise SystemExit(f"port occupied: {sys.argv[1]}")
finally:s.close()
PY
cat > "$OUT/run_config.txt" <<CFG
condition=history-free position norm-ratio magnitude head or specified magnitude ablation, single scale application
tasks=PickPlaceCounterToCabinet OpenDrawer CloseFridge; trials=12 each
seed=7+canonical_task_index*50+trial; task indices=0,2,3; split=pretrain; replan=5; workers=24
magnitude_checkpoint=$MAG
magnitude_multiplier=$MULTIPLIER
scale_target=clip(||truth_L_position|| / ||selected_unit_L_position||, .25, 4)
XLA_PYTHON_CLIENT_PREALLOCATE=false; XLA_PYTHON_CLIENT_MEM_FRACTION=0.20; SKILLNET_SUBTASK_GENERATE=0
CFG
echo "evaluation_started=$(date -u +%FT%TZ)" > "$OUT/eval-status.txt"
taskset -c 38 nice -n 10 "$SKILLPY" "$HERE/serve_memonly_policy.py" --host 127.0.0.1 --port "$PORT" \
  --magnitude-checkpoint "$MAG" --magnitude-multiplier "$MULTIPLIER" --query-log "$OUT/queries.jsonl" \
  > "$OUT/server.log" 2>&1 & SERVER=$!
cleanup() { kill -TERM "$SERVER" 2>/dev/null || true; wait "$SERVER" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
ready=0
for _ in $(seq 1 600); do
  grep -q 'server listening' "$OUT/server.log" && { ready=1; break; }
  kill -0 "$SERVER" 2>/dev/null || { tail -80 "$OUT/server.log" >&2; exit 1; }
  sleep 1
done
(( ready == 1 )) || { echo 'server startup timeout' >&2; exit 1; }
export SKILLNET_ROBOCASA_VENV="$STATE/envs/robocasa"
source "$WT/env/robocasa/activate.sh"
export SKILLNET_SUBTASK_GENERATE=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$WT/skill_moe/skillnet/packages/skillnet-client/src:$HERE:$PYTHONPATH"
taskset -c 8-31 nice -n 10 timeout --signal=TERM 48m "$ROBOPY" "$HERE/main.py" \
  --host 127.0.0.1 --port "$PORT" --tasks PickPlaceCounterToCabinet OpenDrawer CloseFridge \
  --split pretrain --num-trials 12 --replan-steps 5 --seed 7 --seed-task-stride 50 \
  --seed-task-indices 0 2 3 --resize 224 --workers 24 --output "$OUT/eval" \
  > "$OUT/eval.log" 2>&1
echo "evaluation_complete=$(date -u +%FT%TZ)" >> "$OUT/eval-status.txt"
