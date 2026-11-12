#!/usr/bin/env bash
# Memory-only 3-task development run (12 trials/task), with no action-expert calls.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; source "$HERE/common.sh"
OUT="${1:?new output directory required}"; PORT="${PORT:-8879}"; WORKERS="${WORKERS:-12}"
[[ ! -e "$OUT" ]] || { echo "Refusing existing output $OUT" >&2; exit 2; }; mkdir -p "$OUT/eval"
MAG="${MAGNITUDE_CHECKPOINT:-$R/key-lite-01/magnitude_v2/magnitude_head.pt}"
taskset -c 8-15 nice -n 10 "$SKILL_PY" "$SKILL_ROOT/examples/robocasa/serve_memonly_policy.py" --host 127.0.0.1 --port "$PORT" --magnitude-checkpoint "$MAG" --query-log "$OUT/queries.jsonl" > "$OUT/server.log" 2>&1 & SERVER=$!
cleanup(){ kill -TERM "$SERVER" 2>/dev/null || true; wait "$SERVER" 2>/dev/null || true; }; trap cleanup EXIT INT TERM
for i in $(seq 1 240); do grep -q 'serving rm-memory-only' "$OUT/server.log" && break; kill -0 "$SERVER" || { tail -80 "$OUT/server.log"; exit 1; }; sleep 2; done
source "$REPO_ROOT/env/robocasa/activate.sh"; export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PYTHONPATH="$SKILL_ROOT/packages/skillnet-client/src:$SKILL_ROOT/examples/robocasa${PYTHONPATH:+:$PYTHONPATH}" taskset -c 16-47 nice -n 10 python "$SKILL_ROOT/examples/robocasa/main.py" --host 127.0.0.1 --port "$PORT" --tasks PickPlaceCounterToCabinet OpenDrawer CloseFridge --split pretrain --num-trials 12 --replan-steps 5 --seed 7 --seed-task-stride 50 --seed-task-indices 0 2 3 --resize 224 --workers "$WORKERS" --output "$OUT/eval"
