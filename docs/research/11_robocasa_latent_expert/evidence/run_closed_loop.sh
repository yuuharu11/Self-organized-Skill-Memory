#!/usr/bin/env bash
# Two-lane closed-loop runner (dist t0; arm from Expert, gripper/base/mode from π0.5). RCX2 6 tasks and the 2 unseen tasks.
# Usage: run_lanes.sh <lane-name> <server-cores> <client-cores> <port> <mode:end> [<mode:end> ...]
set -uo pipefail
STATE=/work/.skillnet-state
WT=${WT:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}   # リポジトリの根
ROOT=$WT/skill_moe/skillnet
OUT=${OUT:-$STATE/outputs/robocasa-memory/latent-skill-01/closed}   # 2タスクは .../unseen2/dist
EXPERT=$STATE/outputs/robocasa-memory/latent-skill-01/expert/train/key-agnostic-01-retry-fraction035/expert_005000.msgpack
TASKS=(${TASKS:-PickPlaceCounterToCabinet PickPlaceSinkToCounter OpenDrawer CloseFridge TurnOnSinkFaucet TurnOffStove})   # 2タスクは "OpenCabinet PickPlaceCounterToStove"
INDICES=${INDICES:-0 1 2 3 4 5}   # 2タスクは "6 7"
lane=$1; scores=$2; ccores=$3; port=$4; shift 4
export SKILLNET_STATE_DIR=$STATE SKILLNET_REPO_ROOT=$WT SKILLNET_SUBTASK_GENERATE=0
export SKILLNET_ROBOCASA_CHECKPOINT=$STATE/checkpoints/robocasa365/pi05_pretrain_human300/multitask_learning/75000
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=.35
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export SKILLNET_ROBOCASA_VENV=$STATE/envs/robocasa
source "$WT/env/robocasa/activate.sh"
export PYTHONPATH="$ROOT/src:$ROOT/packages/skillnet-client/src:$ROOT/examples/robocasa:$ROOT/examples/memory_pilot/robocasa${PYTHONPATH:+:$PYTHONPATH}"
S=$OUT/lanes.status
SERVER_PID=
cleanup(){ [[ -n "$SERVER_PID" ]] && { kill -TERM "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null; SERVER_PID=; }; }
trap 'rc=$?; cleanup; echo "$(date -u +%FT%TZ) lane=$lane EXIT rc=$rc" >> $S' EXIT
trap 'exit 143' TERM
for job in "$@"; do
  mode=${job%%:*}; end=${job##*:}; cond=$OUT/$mode; mkdir -p "$cond"
  exec 9>"$cond/.lock"; flock 9
  echo "$(date -u +%FT%TZ) lane=$lane start mode=$mode trials<$end" >> $S
  ts=$(date -u +%Y%m%dT%H%M%SZ); slog=$cond/server-lane$lane-$ts.log; clog=$cond/lane$lane-$ts.client.log
  taskset -c $scores nice -n 10 "$STATE/envs/skillnet/bin/python" "$ROOT/examples/robocasa/serve_latent_policy.py" \
    --out "$STATE/outputs/robocasa-memory/latent-skill-01" --mode "$mode" --expert "$EXPERT" --t0-policy dist \
    --selector "$STATE/outputs/robocasa-memory/latent-skill-01/closed-loop/selector_${mode}.json" \
    --cache-dir "$STATE/outputs/robocasa-memory/latent-skill-01/closed-loop" --key-device cuda --noise-seed 4101 \
    --replan-steps 5 --host 127.0.0.1 --port "$port" --query-log "$cond/queries.jsonl" > "$slog" 2>&1 &
  SERVER_PID=$!; ready=0
  for _ in $(seq 1 300); do grep -q 'serving latent-key-expert' "$slog" && { ready=1; break; }; kill -0 $SERVER_PID 2>/dev/null || break; sleep 2; done
  if [[ $ready != 1 ]]; then echo "$(date -u +%FT%TZ) lane=$lane FAILED server mode=$mode ($slog)" >> $S; exit 1; fi
  taskset -c $ccores nice -n 10 python "$ROOT/examples/robocasa/main.py" --host 127.0.0.1 --port "$port" --tasks "${TASKS[@]}" \
    --split pretrain --num-trials "$end" --replan-steps 5 --seed 7 --seed-task-stride 50 --seed-task-indices $INDICES \
    --resize 224 --workers 8 --resume --output "$cond/eval" > "$clog" 2>&1
  rc=$?; cleanup; flock -u 9
  if [[ $rc != 0 ]]; then echo "$(date -u +%FT%TZ) lane=$lane FAILED client rc=$rc mode=$mode ($clog)" >> $S; exit 1; fi
  python - "$cond/eval/results.json" "$mode" "$end" >> "$OUT/progress.md" <<'PY'
import json,sys,datetime
p,mode,end=sys.argv[1],sys.argv[2],int(sys.argv[3])
tasks=['PickPlaceCounterToCabinet','PickPlaceSinkToCounter','OpenDrawer','CloseFridge','TurnOnSinkFaucet','TurnOffStove']
R=json.load(open(p))['records']
for lo,hi in ((0,10),(10,30),(30,50)):
    if hi>end: break
    rec=[r for r in R if lo<=int(r['trial'])<hi]
    c=[sum(bool(r['success']) for r in rec if r['task']==t) for t in tasks]
    print(f"- {datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='minutes')} {mode} trial {lo}–{hi-1}: "+' / '.join(map(str,c))+f" = {sum(c)}/{len(rec)}")
PY
  echo "$(date -u +%FT%TZ) lane=$lane done mode=$mode trials<$end" >> $S
done
echo "$(date -u +%FT%TZ) lane=$lane ALL_DONE" >> $S
