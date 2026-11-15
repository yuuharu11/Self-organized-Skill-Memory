#!/usr/bin/env bash
# One memory+Expert closed-loop condition. Usage: expert-closed-loop.sh OUTPUT CHECKPOINT TRIALS [--tasks TASK ...] [--indices N ...] [--fixgr]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; source "$HERE/common.sh"
OUT="${1:?output directory required}"; CK="${2:?expert checkpoint required}"; TRIALS="${3:-50}"; shift 3
TASKS=(PickPlaceCounterToCabinet PickPlaceSinkToCounter OpenDrawer CloseFridge TurnOnSinkFaucet TurnOffStove); INDICES=(0 1 2 3 4 5); PORT="${PORT:-8896}"; WORKERS="${WORKERS:-24}"; FIXGR=0
while (($#)); do
  case "$1" in
    --tasks) shift; TASKS=(); while (($#)) && [[ "$1" != --* ]]; do TASKS+=("$1"); shift; done ;;
    --indices) shift; INDICES=(); while (($#)) && [[ "$1" != --* ]]; do INDICES+=("$1"); shift; done ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --fixgr) FIXGR=1; shift ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
MAG="${MAGNITUDE_CHECKPOINT:-$R/key-lite-01/magnitude_v2/magnitude_head.pt}"
[[ ! -e "$OUT" ]] || { echo "Refusing existing output $OUT" >&2; exit 2; }; mkdir -p "$OUT/eval"
export SKILLNET_ROBOCASA_CHECKPOINT="${SKILLNET_ROBOCASA_CHECKPOINT:-$SKILLNET_STATE_DIR/checkpoints/robocasa365/pi05_pretrain_human300/multitask_learning/75000}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-.30}"
FIXGR_ARGS=(); if [[ "$FIXGR" == 1 ]]; then FIXGR_ARGS+=(--draft-unit-gripper --grip-after-bridge --scale-position-only); fi
taskset -c 8-15 nice -n 10 "$SKILL_PY" "$SKILL_ROOT/examples/robocasa/serve_memonly_policy.py" --host 127.0.0.1 --port "$PORT" --query-log "$OUT/queries.jsonl" --magnitude-checkpoint "$MAG" --bridge-checkpoint "$CK" --bridge-t0 .2 --bridge-steps 10 "${FIXGR_ARGS[@]}" > "$OUT/server.log" 2>&1 & SERVER=$!
cleanup(){ kill -TERM "$SERVER" 2>/dev/null || true; wait "$SERVER" 2>/dev/null || true; }; trap cleanup EXIT INT TERM
for i in $(seq 1 240); do grep -q 'serving rm-memory-bridge' "$OUT/server.log" && break; kill -0 "$SERVER" || { tail -80 "$OUT/server.log"; exit 1; }; sleep 2; done
source "$REPO_ROOT/env/robocasa/activate.sh"; export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PYTHONPATH="$SKILL_ROOT/packages/skillnet-client/src:$SKILL_ROOT/examples/robocasa${PYTHONPATH:+:$PYTHONPATH}" taskset -c 16-47 nice -n 10 python "$SKILL_ROOT/examples/robocasa/main.py" --host 127.0.0.1 --port "$PORT" --tasks "${TASKS[@]}" --split pretrain --num-trials "$TRIALS" --replan-steps 5 --seed 7 --seed-task-stride 50 --seed-task-indices "${INDICES[@]}" --resize 224 --workers "$WORKERS" --output "$OUT/eval"
