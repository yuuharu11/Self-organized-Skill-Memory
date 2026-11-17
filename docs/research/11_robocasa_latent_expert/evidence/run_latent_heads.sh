#!/usr/bin/env bash
set -euo pipefail
STATE=/work/.skillnet-state
WT=${WT:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}   # リポジトリの根
ROOT=$WT/skill_moe/skillnet
OUT=$STATE/outputs/robocasa-memory/latent-skill-01
COND=$OUT/latent-heads/closed/$1
PORT=$2; TASKS=$3; NTRIALS=$4; INDICES=$5; WORKERS=${6:-8}
mkdir -p "$COND"
if [[ -e "$COND/eval/results.json" ]]; then echo "refusing to overwrite completed output: $COND/eval" >&2; exit 2; fi
export SKILLNET_STATE_DIR=$STATE SKILLNET_REPO_ROOT=$WT SKILLNET_SUBTASK_GENERATE=0
export SKILLNET_ROBOCASA_CHECKPOINT=$STATE/checkpoints/robocasa365/pi05_pretrain_human300/multitask_learning/75000
export SKILLNET_ROBOCASA_VENV=$STATE/envs/robocasa
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=.35
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
source "$WT/env/robocasa/activate.sh"
export PYTHONPATH="$ROOT/src:$ROOT/packages/skillnet-client/src:$ROOT/examples/robocasa:$ROOT/examples/memory_pilot/robocasa${PYTHONPATH:+:$PYTHONPATH}"
stamp=$(date -u +%Y%m%dT%H%M%SZ)
slog=$COND/server-$stamp.log; clog=$COND/client-$stamp.log
taskset -c 8-15 nice -n 10 "$STATE/envs/skillnet/bin/python" "$ROOT/examples/robocasa/serve_latent_policy.py" \
  --out "$OUT" --mode e_vlm_h --expert "$OUT/expert/train/key-agnostic-01-retry-fraction035/expert_005000.msgpack" \
  --t0-policy latent --latent-heads "$OUT/latent-heads/heads.pt" --latent-fallback-rate 20 \
  --cache-dir "$OUT/closed-loop" --key-device cuda --noise-seed 4101 --replan-steps 5 \
  --host 127.0.0.1 --port "$PORT" --query-log "$COND/queries.jsonl" >"$slog" 2>&1 &
spid=$!
cleanup(){ kill -TERM "$spid" 2>/dev/null || true; wait "$spid" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
ready=0
for _ in $(seq 1 360); do
  if grep -q 'serving latent-key-expert' "$slog"; then ready=1; break; fi
  kill -0 "$spid" 2>/dev/null || break
  sleep 2
done
if [[ $ready != 1 ]]; then cat "$slog"; exit 1; fi
taskset -c 16-39 nice -n 10 python "$ROOT/examples/robocasa/main.py" \
  --host 127.0.0.1 --port "$PORT" --tasks $TASKS --split pretrain --num-trials "$NTRIALS" \
  --replan-steps 5 --seed 7 --seed-task-stride 50 --seed-task-indices $INDICES \
  --resize 224 --workers "$WORKERS" --resume --output "$COND/eval" >"$clog" 2>&1
echo "$(date -u +%FT%TZ) DONE $1" >>"$OUT/latent-heads/closed/progress.log"
