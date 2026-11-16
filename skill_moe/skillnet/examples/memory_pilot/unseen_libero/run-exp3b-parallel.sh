#!/usr/bin/env bash
set -euo pipefail

cd /work
source /work/env/activate.sh

SCRIPT=/work/skill_moe/skillnet/examples/memory_pilot/unseen_libero/skill_expert_refine.py
L90X=/work/.skillnet-state/outputs/memory-pilot/unseen-libero/expert-refine-exp3b-01/l90x_rows
SELECTIONS=/work/.skillnet-state/outputs/memory-pilot/unseen-libero/expert-refine-exp3b-01/unit-selections-taskexclude-02
PROD=/work/.skillnet-state/outputs/memory-pilot/unseen-libero/expert-refine-exp3b-01/production
PREPARED="$PROD/prepared"
LOGS="$PROD/logs"
STEPS=${STEPS:-5000}
BATCH=${BATCH:-8}
THREADS=${THREADS:-8}

export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS
[[ -f "$L90X/manifest.json" && -f "$SELECTIONS/selection_manifest.json" ]] || {
  echo "Required L90X cache or task-excluded unit selections are missing" >&2; exit 2;
}
taskset -c 8-47 nice -n 10 python - "$L90X/manifest.json" <<'PY'
import json, sys
m=json.load(open(sys.argv[1]))
missing={int(x['task_id']) for x in m.get('missing_tasks', [])}
zero={int(k) for k,v in m.get('task_row_counts', {}).items() if int(v)==0}
if len(m.get('tasks', [])) != 78 or missing != {51} or zero != {51}:
    print('Expected the documented L90X cache with only source task 51 absent.', file=sys.stderr)
    print('Missing tasks: ' + json.dumps(m.get('missing_tasks', [])), file=sys.stderr)
    raise SystemExit(2)
PY
mkdir -p "$LOGS"
[[ -f "$PREPARED/prepare.json" ]] || {
  [[ ! -e "$PREPARED" ]] || { echo "Partial prepared cache exists; inspect it before continuing: $PREPARED" >&2; exit 2; }
  taskset -c 8-47 nice -n 10 env \
    XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.10 \
    python "$SCRIPT" prepare --out "$PREPARED" --l90x-dir "$L90X" \
      --selection-dir "$SELECTIONS" --threads "$THREADS" --mem-fraction 0.10 \
      > "$LOGS/prepare.log" 2>&1
}
[[ -f "$PREPARED/train_a.npz" && -f "$PREPARED/train_l90x.npz" && -f "$PREPARED/dev.npz" ]] || {
  echo "Prepared cache is incomplete: $PREPARED" >&2; exit 2;
}
[[ ! -e "$PROD/A" && ! -e "$PROD/B" ]] || {
  echo "A or B output already exists; choose a fresh production directory" >&2; exit 2;
}

taskset -c 8-27 nice -n 10 env \
  OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" \
  XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 \
  python "$SCRIPT" train --prepared-dir "$PREPARED" --out "$PROD/A" --condition A \
    --steps "$STEPS" --threads "$THREADS" --batch "$BATCH" --mem-fraction 0.30 \
    > "$LOGS/train_A.log" 2>&1 &
pid_a=$!
taskset -c 28-47 nice -n 10 env \
  OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" \
  XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 \
  python "$SCRIPT" train --prepared-dir "$PREPARED" --out "$PROD/B" --condition B \
    --steps "$STEPS" --threads "$THREADS" --batch "$BATCH" --mem-fraction 0.30 \
    > "$LOGS/train_B.log" 2>&1 &
pid_b=$!

status=0
wait "$pid_a" || status=1
wait "$pid_b" || status=1
if (( status )); then
  echo "At least one EXP3-B condition failed; inspect $LOGS" >&2
  exit 1
fi
echo "EXP3-B A and B complete: $PROD"
