#!/usr/bin/env bash
# Reproduce one baseline run. Usage: baseline.sh OUT_DIR pretrain|target TRIALS WORKERS [TASK ...]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; source "$HERE/common.sh"
OUT="${1:?output directory required}"; SPLIT="${2:-pretrain}"; TRIALS="${3:-50}"; WORKERS="${4:-12}"; shift $(( $# >= 4 ? 4 : $# ));
if (($#)); then TASKS=("$@"); else TASKS=(PickPlaceCounterToCabinet PickPlaceSinkToCounter OpenDrawer CloseFridge TurnOnSinkFaucet TurnOffStove); fi
PORT="${PORT:-8766}"
[[ ! -e "$OUT" ]] || { echo "Refusing existing output $OUT" >&2; exit 2; }
mkdir -p "$OUT"
CK="${SKILLNET_ROBOCASA_CHECKPOINT:-$SKILLNET_STATE_DIR/checkpoints/robocasa365/pi05_pretrain_human300/multitask_learning/75000}"
export SKILLNET_ROBOCASA_CHECKPOINT="$CK" XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.20}"
taskset -c 8-15 nice -n 10 "$SKILL_PY" "$SKILL_ROOT/examples/robocasa/serve_policy.py" --checkpoint "$CK" --host 127.0.0.1 --port "$PORT" > "$OUT/server.log" 2>&1 & SERVER=$!
cleanup(){ kill -TERM "$SERVER" 2>/dev/null || true; wait "$SERVER" 2>/dev/null || true; }; trap cleanup EXIT INT TERM
for i in $(seq 1 300); do grep -q 'serving pi05_robocasa_multitask' "$OUT/server.log" && break; kill -0 "$SERVER" || { tail -80 "$OUT/server.log"; exit 1; }; sleep 2; done
source "$REPO_ROOT/env/robocasa/activate.sh"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PYTHONPATH="$SKILL_ROOT/packages/skillnet-client/src:$SKILL_ROOT/examples/robocasa${PYTHONPATH:+:$PYTHONPATH}" \
  taskset -c 16-47 nice -n 10 python "$SKILL_ROOT/examples/robocasa/main.py" --host 127.0.0.1 --port "$PORT" --tasks "${TASKS[@]}" --split "$SPLIT" --num-trials "$TRIALS" --replan-steps 5 --seed 7 --seed-task-stride 50 --seed-task-indices 0 1 2 3 4 5 --resize 224 --workers "$WORKERS" --output "$OUT/eval"
python "$SKILL_ROOT/examples/robocasa/get_eval_stats.py" "$OUT/eval/results.json"
