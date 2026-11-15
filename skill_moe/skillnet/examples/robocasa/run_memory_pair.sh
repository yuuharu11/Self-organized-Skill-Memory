#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../../../" && pwd)"
STATE="${SKILLNET_STATE_DIR:-/work/.skillnet-state}"
CHECKPOINT="${SKILLNET_ROBOCASA_CHECKPOINT:-$STATE/checkpoints/robocasa365/pi05_pretrain_human300/multitask_learning/75000}"
MEMORY_DIR="${ROBOCASA_MEMORY_DIR:-$STATE/outputs/robocasa-memory/memory-01/policy-order-v2}"
EXPERT_DIR="${ROBOCASA_EXPERT_DIR:-$STATE/outputs/robocasa-memory/expert-01/expert}"
PORT_A="${PORT_A:-8767}"; PORT_B="${PORT_B:-8768}"; PORT_CONTROL="${PORT_CONTROL:-8769}"
TRIALS="${TRIALS:-50}"; SPLIT="${SPLIT:-pretrain}"; SEED="${SEED:-7}"; REPLAN="${REPLAN_STEPS:-5}"
SCALE_MODE="${SCALE_MODE:-s-sep}"; T0="${T0:-0.1}"; TIMEOUT="${TIMEOUT:-36h}"; RESUME="${RESUME:-0}"
OUT="${1:?usage: run_memory_pair.sh OUTPUT_DIR [--resume] [--workers N] [--trials N] [--tasks TASK ...]}"; shift
WORKERS=""
TASKS=(PickPlaceCounterToCabinet PickPlaceSinkToCounter OpenDrawer CloseFridge TurnOnSinkFaucet TurnOffStove)
while (($#)); do
  case "$1" in
    --resume) RESUME=1; shift ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --trials) TRIALS="$2"; shift 2 ;;
    --tasks) shift; TASKS=(); while (($#)) && [[ "$1" != --* ]]; do TASKS+=("$1"); shift; done ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ -e "$OUT" ]]; then
  [[ -d "$OUT" && "$RESUME" == 1 ]] || { echo "Refusing existing output without --resume: $OUT" >&2; exit 2; }
  [[ -f "$OUT/run_config.txt" ]] || { echo "Cannot resume without run_config.txt: $OUT" >&2; exit 2; }
else mkdir -p "$OUT"; fi
for port in "$PORT_A" "$PORT_B" "$PORT_CONTROL"; do
  ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq ":${port}$" && { echo "Port occupied: $port" >&2; exit 2; } || true
done
[[ -x "$STATE/envs/skillnet/bin/python" && -x "$STATE/envs/robocasa/bin/python" ]] || { echo "Missing SkillNet/RoboCasa venv" >&2; exit 2; }
[[ "$SCALE_MODE" == s-sep && "$T0" == 0.1 ]] || { echo "RM-D2 fixed setting: scale=s-sep t0=.1" >&2; exit 2; }
LOAD1="$(awk '{print $1}' /proc/loadavg)"
if [[ -z "$WORKERS" ]]; then
  LOAD_INT="$(awk -v x="$LOAD1" 'BEGIN { print int(x + 0.5) }')"
  if (( LOAD_INT < 12 )); then WORKERS=8; elif (( LOAD_INT < 20 )); then WORKERS=6; else WORKERS=4; fi
fi
(( WORKERS >= 1 && WORKERS <= 8 )) || { echo "workers must be in 1..8 per condition (3 conditions)" >&2; exit 2; }
if [[ "$RESUME" == 1 ]]; then
  grep -Fxq "tasks=${TASKS[*]}" "$OUT/run_config.txt" || { echo "resume task order differs" >&2; exit 2; }
  grep -Fxq "split=$SPLIT trials=$TRIALS replan=$REPLAN seed=$SEED" "$OUT/run_config.txt" || { echo "resume protocol differs" >&2; exit 2; }
  grep -Fxq "main=A-expert; grip=B-expert+pi05-gripper; control=pretrained" "$OUT/run_config.txt" || { echo "resume conditions differ" >&2; exit 2; }
else
  cat > "$OUT/run_config.txt" <<CFG
checkpoint=$CHECKPOINT
memory_dir=$MEMORY_DIR
expert_dir=$EXPERT_DIR
tasks=${TASKS[*]}
split=$SPLIT trials=$TRIALS replan=$REPLAN seed=$SEED
seed_rule=seed + task_index*num_trials + trial
workers_per_condition=$WORKERS loadavg_before=$LOAD1
main=A-expert; grip=B-expert+pi05-gripper; control=pretrained
scale=$SCALE_MODE t0=$T0 scale_reference=frozen_pretrained_pi05; position_and_rotation_fitted_separately
cpu_affinity=servers:8-13; workers:A14-23,B24-33,control34-43
threads=worker:1 each_server:2 upper_bound=$((3*WORKERS+6))
subtask_generation=disabled; baseline_server=not_launched
CFG
fi
mkdir -p "$OUT/main/eval" "$OUT/gripper/eval" "$OUT/control/eval"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ "$RESUME" == 1 ]]; then echo "resume_attempt=$RUN_ID workers=$WORKERS loadavg_before=$LOAD1" >> "$OUT/run_config.txt"; fi
export SKILLNET_STATE_DIR="$STATE" SKILLNET_ROBOCASA_CHECKPOINT="$CHECKPOINT"
export SKILLNET_SUBTASK_GENERATE=0 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.20
export TORCH_MEMORY_FRACTION="${TORCH_MEMORY_FRACTION:-0.03}" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH="$ROOT/skill_moe/skillnet/src:$ROOT/skill_moe/skillnet/packages/skillnet-client/src:$ROOT/skill_moe/skillnet/examples/memory_pilot/robocasa:$ROOT/skill_moe/skillnet/examples/memory_pilot/pi05_memory${PYTHONPATH:+:$PYTHONPATH}"
source "$STATE/envs/skillnet/bin/activate"
LA="$OUT/main-server.log"; LB="$OUT/gripper-server.log"; LC="$OUT/control-server.log"
taskset -c 8-9 nice -n 10 python "$HERE/serve_memory_policy.py" --checkpoint "$CHECKPOINT" --memory-dir "$MEMORY_DIR" --expert-dir "$EXPERT_DIR" --weights expert --scale-mode "$SCALE_MODE" --t0 "$T0" --replan-steps "$REPLAN" --seed "$SEED" --host 127.0.0.1 --port "$PORT_A" --query-log "$OUT/main/queries.jsonl" > "$LA" 2>&1 & PA=$!
taskset -c 10-11 nice -n 10 python "$HERE/serve_memory_policy.py" --checkpoint "$CHECKPOINT" --memory-dir "$MEMORY_DIR" --expert-dir "$EXPERT_DIR" --weights expert --replace-gripper --scale-mode "$SCALE_MODE" --t0 "$T0" --replan-steps "$REPLAN" --seed "$SEED" --host 127.0.0.1 --port "$PORT_B" --query-log "$OUT/gripper/queries.jsonl" > "$LB" 2>&1 & PB=$!
taskset -c 12-13 nice -n 10 python "$HERE/serve_memory_policy.py" --checkpoint "$CHECKPOINT" --memory-dir "$MEMORY_DIR" --expert-dir "$EXPERT_DIR" --weights pretrained --scale-mode "$SCALE_MODE" --t0 "$T0" --replan-steps "$REPLAN" --seed "$SEED" --host 127.0.0.1 --port "$PORT_CONTROL" --query-log "$OUT/control/queries.jsonl" > "$LC" 2>&1 & PC=$!
printf 'A=%s B=%s control=%s\n' "$PA" "$PB" "$PC" > "$OUT/server-pids-$RUN_ID.txt"
P1=""; P2=""; P3=""; WATCH_PID=""
PROGRESS_FILE="$(dirname "$OUT")/progress.md"
cleanup() {
  [[ -n "$WATCH_PID" ]] && kill "$WATCH_PID" 2>/dev/null || true
  for p in "$P1" "$P2" "$P3"; do [[ -n "$p" ]] && kill -- -"$p" 2>/dev/null || true; done
  kill "$PA" "$PB" "$PC" 2>/dev/null || true
  wait "$PA" "$PB" "$PC" 2>/dev/null || true
}
trap cleanup EXIT INT TERM
wait_log() {
  local pid="$1" log="$2" marker='"serving": "rm-d2-memory-bridge"'
  for _ in $(seq 1 1200); do
    kill -0 "$pid" 2>/dev/null || { tail -100 "$log" >&2; return 1; }
    [[ "$(ps -o stat= -p "$pid" 2>/dev/null || true)" == *Z* ]] && { tail -100 "$log" >&2; return 1; }
    grep -q "$marker" "$log" && return 0; sleep 1
  done
  tail -100 "$log" >&2; return 1
}
wait_log "$PA" "$LA"; wait_log "$PB" "$LB"; wait_log "$PC" "$LC"
export SKILLNET_ROBOCASA_VENV="$STATE/envs/robocasa"; source "$ROOT/env/robocasa/activate.sh"
export SKILLNET_SUBTASK_GENERATE=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$ROOT/skill_moe/skillnet/packages/skillnet-client/src:$PYTHONPATH"
write_progress() {
  "$STATE/envs/robocasa/bin/python" - "$OUT" "$PROGRESS_FILE" "${TASKS[*]}" "$TRIALS" <<'PY'
import datetime,json,pathlib,sys
out=pathlib.Path(sys.argv[1]); progress=pathlib.Path(sys.argv[2]); tasks=sys.argv[3].split(); trials=int(sys.argv[4])
lines=[f"\n## {datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')} UTC"]
for arm in ('main','gripper','control'):
 rows={}
 for path in sorted((out/arm/'eval/workers').glob('worker-*.json')):
  try: data=json.loads(path.read_text())
  except (OSError,json.JSONDecodeError): continue
  for row in data:
   if row.get('status')=='complete': rows[(row['task'],int(row['trial']))]=row
 lines.append(f"\n### {arm}")
 for task in tasks:
  subset=[r for (t,_),r in rows.items() if t==task]
  lines.append(f"- {task}: {sum(bool(r['success']) for r in subset)}/{len(subset)} success; {len(subset)}/{trials} complete")
with progress.open('a') as f: f.write('\n'.join(lines)+'\n')
PY
}
write_progress; (while sleep 1800; do write_progress; done) & WATCH_PID=$!
TASK_STRING="${TASKS[*]}"
run_arm() {
  local label="$1" port="$2" cpus="$3" resume_flag=0
  [[ "$RESUME" == 1 ]] && resume_flag=1
  exec setsid bash -c '
   set -euo pipefail
   label="$1";port="$2";cpus="$3";out="$4";run_id="$5";timeout_s="$6";here="$7";split="$8";trials="$9";replan="${10}";seed="${11}";workers="${12}";resume="${13}";task_string="${14}"
   read -r -a task_values <<< "$task_string"; task_args=(--tasks "${task_values[@]}"); common=(--split "$split" --num-trials "$trials" --replan-steps "$replan" --seed "$seed" --resize 224); resume_args=(); [[ "$resume" == 1 ]] && resume_args=(--resume)
   log="$out/$label/eval-$run_id.log"; [[ "$resume" == 1 ]] && log="$out/$label/eval.log"
   taskset -c "$cpus" nice -n 10 timeout "$timeout_s" python "$here/main.py" --host 127.0.0.1 --port "$port" "${task_args[@]}" "${common[@]}" --workers "$workers" --output "$out/$label/eval" "${resume_args[@]}" >> "$log" 2>&1
   python "$here/get_eval_stats.py" "$out/$label/eval/results.json" > "$out/$label/eval/stats.txt"
  ' _ "$label" "$port" "$cpus" "$OUT" "$RUN_ID" "$TIMEOUT" "$HERE" "$SPLIT" "$TRIALS" "$REPLAN" "$SEED" "$WORKERS" "$resume_flag" "$TASK_STRING"
}
run_arm main "$PORT_A" 14-23 & P1=$!
run_arm gripper "$PORT_B" 24-33 & P2=$!
run_arm control "$PORT_CONTROL" 34-43 & P3=$!
if wait "$P1"; then C1=0; else C1=$?; fi
if wait "$P2"; then C2=0; else C2=$?; fi
if wait "$P3"; then C3=0; else C3=$?; fi
[[ -n "$WATCH_PID" ]] && kill "$WATCH_PID" 2>/dev/null || true; wait "$WATCH_PID" 2>/dev/null || true; WATCH_PID=""
write_progress
if (( C1 || C2 || C3 )); then echo "partial run; resume with --resume; A=$C1 B=$C2 control=$C3" >&2; exit 1; fi
echo "COMPLETE $(date -u +%FT%TZ)" > "$OUT/DONE"
