#!/usr/bin/env bash
# Run one rerank-01 condition after training. Usage: ... OUTPUT rerank|top1 TRIALS
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; source "$HERE/common.sh"
OUT="${1:?new output directory required}"; MODE="${2:-rerank}"; TRIALS="${3:-25}"; PORT="${PORT:-8904}"
[[ ! -e "$OUT" ]] || { echo "Refusing existing output $OUT" >&2; exit 2; }; mkdir -p "$OUT/eval"
SERVER_ARGS=(--host 127.0.0.1 --port "$PORT" --query-log "$OUT/queries.jsonl"
  --bridge-checkpoint "$R/expert-03/train/expert/expert_003000.msgpack" --bridge-t0 .2 --bridge-steps 10)
if [[ "$MODE" == rerank ]]; then
  SERVER_ARGS+=(--reranker-checkpoint "$R/rerank-01/ranker_k50.pt")
elif [[ "$MODE" != top1 ]]; then
  echo 'condition must be rerank or top1' >&2; exit 2
fi
taskset -c 8-15 nice -n 10 "$SKILL_PY" "$SKILL_ROOT/examples/robocasa/serve_rerank_policy.py" "${SERVER_ARGS[@]}" > "$OUT/server.log" 2>&1 & SERVER=$!
cleanup(){ kill -TERM "$SERVER" 2>/dev/null || true; wait "$SERVER" 2>/dev/null || true; }; trap cleanup EXIT INT TERM
for i in $(seq 1 240); do timeout 1 bash -c "</dev/tcp/127.0.0.1/$PORT" 2>/dev/null && break; kill -0 "$SERVER" || { tail -80 "$OUT/server.log"; exit 1; }; sleep 2; done
source "$REPO_ROOT/env/robocasa/activate.sh"; export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PYTHONPATH="$SKILL_ROOT/packages/skillnet-client/src:$SKILL_ROOT/examples/robocasa${PYTHONPATH:+:$PYTHONPATH}" taskset -c 16-39 nice -n 10 python "$SKILL_ROOT/examples/robocasa/main.py" --host 127.0.0.1 --port "$PORT" --tasks TurnOnSinkFaucet PickPlaceCounterToCabinet PickPlaceSinkToCounter --split pretrain --num-trials "$TRIALS" --replan-steps 5 --seed 7 --seed-task-stride 50 --seed-task-indices 4 0 1 --resize 224 --workers "${WORKERS:-24}" --output "$OUT/eval"
