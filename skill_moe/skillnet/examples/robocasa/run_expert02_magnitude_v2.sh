#!/usr/bin/env bash
set -euo pipefail

STATE=/work/.skillnet-state
WT=$STATE/worktrees/robocasa-memory
ROOT=$STATE/outputs/robocasa-memory/expert-02
OUT=$ROOT/closedloop-v2/run-01
CK=$ROOT/train/attempt-04/expert/expert_001000.msgpack
MAG=$STATE/outputs/robocasa-memory/key-lite-01/magnitude_v2/magnitude_head.pt
[[ -f "$CK" ]] || { echo "missing Expert checkpoint used by closedloop/attempt-01: $CK" >&2; exit 2; }
[[ -f "$MAG" ]] || { echo "missing magnitude_v2 checkpoint: $MAG" >&2; exit 2; }
[[ ! -e "$OUT" ]] || { echo "refusing existing output: $OUT" >&2; exit 2; }
mkdir -p "$OUT/eval"

source /work/env/activate.sh
export SKILLNET_STATE_DIR=$STATE
export SKILLNET_ROBOCASA_CHECKPOINT=$STATE/checkpoints/robocasa365/pi05_pretrain_human300/multitask_learning/75000
export SKILLNET_SUBTASK_GENERATE=0 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.30
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PYTHONPATH="$WT/skill_moe/skillnet/src:$WT/skill_moe/skillnet/packages/skillnet-client/src:$WT/skill_moe/skillnet/examples/robocasa:$WT/skill_moe/skillnet/examples/memory_pilot/robocasa${PYTHONPATH:+:$PYTHONPATH}"

setsid taskset -c 8-15 nice -n 10 timeout --signal=TERM 82m \
  /work/.skillnet-state/envs/skillnet/bin/python "$WT/skill_moe/skillnet/examples/robocasa/serve_memonly_policy.py" \
  --host 127.0.0.1 --port 8896 --query-log "$OUT/queries.jsonl" \
  --diagnostic-log "$OUT/expert-vs-pi05.jsonl" --diagnostic-reference-count 32 \
  --magnitude-checkpoint "$MAG" --bridge-checkpoint "$CK" --bridge-t0 .2 --bridge-steps 10 \
  > "$OUT/server.log" 2>&1 &
SERVER_PID=$!
printf '%s\n' "$SERVER_PID" > "$OUT/server.pid"
cleanup() { kill -TERM "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

READY=0
for i in $(seq 1 180); do
  if grep -q 'rm-memory-bridge' "$OUT/server.log" && timeout 1 bash -c '</dev/tcp/127.0.0.1/8896' 2>/dev/null; then READY=1; break; fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then tail -100 "$OUT/server.log"; exit 1; fi
  sleep 2
done
[[ "$READY" == 1 ]] || { echo 'server readiness timeout' >&2; tail -100 "$OUT/server.log"; exit 1; }

printf 'evaluation_started=%s\ntasks=PickPlaceCounterToCabinet,OpenDrawer,CloseFridge\ntrials_per_task=12\nseed=7\nseed_rule=7 + canonical_task_index*50 + trial\nseed_task_indices=0,2,3\nreplan_steps=5\nworkers=24\nbridge_t0=.2\nbridge_steps=10\nexpert_checkpoint=%s\nrequested_attempt03_path_exists=false\nmagnitude_checkpoint=%s\ndiagnostic_same_observation_samples=32\nXLA_PYTHON_CLIENT_PREALLOCATE=false\nXLA_PYTHON_CLIENT_MEM_FRACTION=0.30\nSKILLNET_SUBTASK_GENERATE=0\n' "$(date -u +%FT%TZ)" "$CK" "$MAG" > "$OUT/run_config.txt"

source "$WT/env/robocasa/activate.sh"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$WT/skill_moe/skillnet/packages/skillnet-client/src:$WT/skill_moe/skillnet/examples/robocasa${PYTHONPATH:+:$PYTHONPATH}"
setsid taskset -c 16-39 nice -n 10 timeout --signal=TERM 78m \
  /work/.skillnet-state/envs/robocasa/bin/python "$WT/skill_moe/skillnet/examples/robocasa/main.py" \
  --host 127.0.0.1 --port 8896 --tasks PickPlaceCounterToCabinet OpenDrawer CloseFridge \
  --split pretrain --num-trials 12 --replan-steps 5 --seed 7 --seed-task-stride 50 \
  --seed-task-indices 0 2 3 --resize 224 --workers 24 --save-video --video-trials 2 \
  --output "$OUT/eval" > "$OUT/client.log" 2>&1
printf 'evaluation_complete=%s\n' "$(date -u +%FT%TZ)" >> "$OUT/run_config.txt"
