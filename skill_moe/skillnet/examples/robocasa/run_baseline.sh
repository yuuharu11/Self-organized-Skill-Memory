#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../../../" && pwd)"
STATE="${SKILLNET_STATE_DIR:-/work/.skillnet-state}"
CHECKPOINT="${SKILLNET_ROBOCASA_CHECKPOINT:-$STATE/checkpoints/robocasa365/pi05_pretrain_human300/multitask_learning/75000}"
export SKILLNET_ROBOCASA_CHECKPOINT="$CHECKPOINT"
OUT="${1:-$STATE/outputs/robocasa-memory/baseline-01}"
PORT="${PORT:-8765}"
TASKS=(PickPlaceCounterToCabinet PickPlaceSinkToCounter OpenDrawer CloseFridge TurnOnSinkFaucet TurnOffStove)
if [[ -e "$OUT" ]]; then
  [[ -d "$OUT" ]] || { echo "Refusing to overwrite non-directory $OUT" >&2; exit 2; }
  [[ -z "$(find "$OUT" -mindepth 1 -maxdepth 1 -print -quit)" ]] || {
    echo "Refusing to overwrite non-empty output directory $OUT" >&2; exit 2;
  }
fi
if command -v ss >/dev/null 2>&1 && ss -ltn | grep -Eq ":${PORT}[[:space:]]"; then
  echo "Port $PORT is already occupied; set PORT to a free port" >&2
  exit 2
fi
mkdir -p "$OUT"
echo "checkpoint=$CHECKPOINT" > "$OUT/run_config.txt"
echo "tasks=${TASKS[*]} split=pretrain trials=50 replan=5 seed_rule=7+task_index*50+trial workers=4 subtask_generation=disabled" >> "$OUT/run_config.txt"
export SKILLNET_STATE_DIR="$STATE" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.35
export SKILLNET_SUBTASK_GENERATE=0
export PYTHONPATH="$ROOT/skill_moe/skillnet/src:$ROOT/skill_moe/skillnet/packages/skillnet-client/src${PYTHONPATH:+:$PYTHONPATH}"
source "$STATE/envs/skillnet/bin/activate"
taskset -c 8-47 nice -n 10 python "$ROOT/skill_moe/skillnet/examples/robocasa/serve_policy.py" \
  --checkpoint "$CHECKPOINT" --host 127.0.0.1 --port "$PORT" > "$OUT/policy-server.log" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$OUT/policy-server.pid"
cleanup() { kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT
for _ in $(seq 1 900); do
  kill -0 "$SERVER_PID" 2>/dev/null || { echo "policy server exited" >&2; tail -100 "$OUT/policy-server.log"; exit 3; }
  if grep -q 'serving pi05_robocasa_multitask' "$OUT/policy-server.log"; then break; fi
  sleep 1
done
grep -q 'serving pi05_robocasa_multitask' "$OUT/policy-server.log" || { echo "server startup timed out" >&2; exit 4; }
export SKILLNET_ROBOCASA_VENV="$STATE/envs/robocasa"
export PYTHONPATH="$ROOT/skill_moe/skillnet/packages/skillnet-client/src${PYTHONPATH:+:$PYTHONPATH}"
source "$ROOT/env/robocasa/activate.sh"
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
PILOT="$OUT/pilot"
taskset -c 8-47 nice -n 10 timeout 3h python "$ROOT/skill_moe/skillnet/examples/robocasa/main.py" \
  --host 127.0.0.1 --port "$PORT" --tasks OpenDrawer CloseFridge --split pretrain --num-trials 10 \
  --replan-steps 5 --workers 4 --seed 7 --output "$PILOT" > "$OUT/pilot-run.log" 2>&1
python "$ROOT/skill_moe/skillnet/examples/robocasa/get_eval_stats.py" "$PILOT/results.json" > "$OUT/pilot/stats.txt"
echo "pilot=OpenDrawer+CloseFridge, 10 each" >> "$OUT/run_config.txt"
# Gate the 300-rollout run on a non-near-zero pilot that is at least compatible
# with the reported atomic baseline at this small n. Keep the threshold explicit.
python - "$PILOT/results.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); p=d['per_task']; total=sum(x['successes'] for x in p.values())
print('pilot gate successes',total,'OpenDrawer',p['OpenDrawer']['successes'],'/10')
if total < 5 or p['OpenDrawer']['successes'] < 4:
    raise SystemExit('pilot too low: stopping before full baseline; inspect pilot and parity.md')
PY
mkdir -p "$OUT/eval"
taskset -c 8-47 nice -n 10 timeout 36h python "$ROOT/skill_moe/skillnet/examples/robocasa/main.py" \
  --host 127.0.0.1 --port "$PORT" --tasks "${TASKS[@]}" --split pretrain --num-trials 50 \
  --replan-steps 5 --workers 4 --seed 7 --output "$OUT/eval" > "$OUT/eval-run.log" 2>&1
python "$ROOT/skill_moe/skillnet/examples/robocasa/get_eval_stats.py" "$OUT/eval/results.json" \
  > "$OUT/eval/stats.txt"
echo "COMPLETE $(date -u +%FT%TZ)" > "$OUT/DONE"
