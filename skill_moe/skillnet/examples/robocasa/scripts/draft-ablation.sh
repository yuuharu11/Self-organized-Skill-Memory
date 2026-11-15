#!/usr/bin/env bash
# Reproduce the logged ablations: offline, offline with base bridge, random, static.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; source "$HERE/common.sh"
OUT="${1:?new output root required}"; CK="${BRIDGE_CHECKPOINT:-$R/expert-02/train/attempt-04/expert/expert_001000.msgpack}"
PORT="${PORT:-8897}"; [[ ! -e "$OUT" ]] || { echo "Refusing existing output $OUT" >&2; exit 2; }
mkdir -p "$OUT"
run_one() {
  local name="$1" mode="$2" trials="$3" workers="$4" bridge="$5"
  local d="$OUT/$name"; local -a args=(--host 127.0.0.1 --port "$PORT" --query-log "$d/queries.jsonl"
    --bridge-t0 .2 --bridge-steps 10 --magnitude-checkpoint "$R/key-lite-01/magnitude/magnitude_head.pt"
    --ablation-mode "$mode" --ablation-log "$d/ablation.jsonl" --ablation-queries 40)
  [[ ! -e "$d" ]] || { echo "Refusing existing output $d" >&2; return 2; }
  mkdir -p "$d/eval"
  if [[ "$bridge" == expert ]]; then args+=(--bridge-checkpoint "$CK"); else args+=(--bridge-use-base-expert); fi
  taskset -c 8-15 nice -n 10 "$SKILL_PY" "$SKILL_ROOT/examples/memory_pilot/robocasa/draft_ablation.py" "${args[@]}" > "$d/server.log" 2>&1 &
  local server=$!; local ready=0
  cleanup_server() { kill -TERM "$server" 2>/dev/null || true; wait "$server" 2>/dev/null || true; }
  trap cleanup_server EXIT INT TERM
  for _ in $(seq 1 240); do
    if grep -q 'serving rm-memory-bridge' "$d/server.log" && timeout 1 bash -c "</dev/tcp/127.0.0.1/$PORT" 2>/dev/null; then ready=1; break; fi
    if ! kill -0 "$server" 2>/dev/null; then tail -80 "$d/server.log"; return 1; fi
    sleep 2
  done
  if [[ $ready != 1 ]]; then tail -80 "$d/server.log"; return 1; fi
  source "$REPO_ROOT/env/robocasa/activate.sh"
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  PYTHONPATH="$SKILL_ROOT/packages/skillnet-client/src:$SKILL_ROOT/examples/robocasa${PYTHONPATH:+:$PYTHONPATH}" \
    taskset -c 16-39 nice -n 10 "$ROBO_PY" "$SKILL_ROOT/examples/robocasa/main.py" --host 127.0.0.1 --port "$PORT" \
      --tasks PickPlaceCounterToCabinet OpenDrawer CloseFridge --split pretrain --num-trials "$trials" --replan-steps 5 \
      --seed 7 --seed-task-stride 50 --seed-task-indices 0 2 3 --resize 224 --workers "$workers" --output "$d/eval" \
      --save-video --video-trials 2 > "$d/client.log" 2>&1
  cleanup_server
  trap - EXIT INT TERM
}
run_one offline offline 3 12 expert
run_one offline_base offline 3 12 base
run_one random random 12 16 expert
run_one static static 12 16 expert
