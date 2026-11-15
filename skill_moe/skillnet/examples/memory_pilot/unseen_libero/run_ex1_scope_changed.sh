#!/usr/bin/env bash
set -euo pipefail

ROOT=/work/.skillnet-state/outputs/memory-pilot/unseen-libero/expert-ex1-01/production-03
RUN=$ROOT/scope-change-01
PREP=/work/.skillnet-state/outputs/memory-pilot/unseen-libero/expert-ex1-01/production-02/prepared
CODE=/work/skill_moe/skillnet/examples/memory_pilot/unseen_libero
mkdir -p "$RUN/logs" "$RUN/train"
source /work/env/activate.sh
cd /work/skill_moe/skillnet

wait_gpu() {
  local label="$1" stable=0 used
  while (( stable < 5 )); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
    printf '%s %s used=%s MiB stable=%s/5\n' "$(date -Is)" "$label" "$used" "$stable" >> "$RUN/logs/gpu-wait.log"
    if [[ "$used" =~ ^[0-9]+$ ]] && (( used < 80000 )); then stable=$((stable+1)); else stable=0; fi
    sleep 60
  done
}

train_one() {
  local cond="$1" frac out
  for frac in .20 .16 .12; do
    out="$RUN/train/${cond}-fraction-${frac#.}"
    if [[ -f "$out/train.json" ]]; then return 0; fi
    wait_gpu "train-${cond}-fraction-${frac}"
    printf '%s start %s fraction=%s\n' "$(date -Is)" "$cond" "$frac" >> "$RUN/logs/training.log"
    export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION="$frac"
    export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
    if nice -n 10 taskset -c 8-15 python "$CODE/skill_expert_refine.py" train \
      --prepared-dir "$PREP" --out "$out" --condition "$cond" --steps 5000 \
      --threads 8 --batch 8 --microbatch 1 --mem-fraction "$frac" \
      > "$RUN/logs/train-${cond}-${frac#.}.log" 2>&1; then
      printf '%s complete %s fraction=%s\n' "$(date -Is)" "$cond" "$frac" >> "$RUN/logs/training.log"
      return 0
    fi
    printf '%s retry %s fraction=%s rc=%s\n' "$(date -Is)" "$cond" "$frac" "$?" >> "$RUN/logs/training.log"
  done
  return 1
}

train_one A
train_one B
mkdir -p "$RUN/train-aggregate"
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python - "$RUN" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]); merged=None; sources={}
for cond in ('A','B'):
    candidates=sorted((root/'train').glob(f'{cond}-fraction-*/train.json'))
    if not candidates: raise FileNotFoundError(f'missing completed {cond} train.json')
    path=candidates[-1]; value=json.loads(path.read_text())
    if value.get('status')!='complete': raise RuntimeError(f'{cond} is not complete: {path}')
    sources[cond]=str(path)
    if merged is None: merged=value
    else: merged['conditions'].update(value['conditions'])
merged['status']='complete'; merged['artifact_dir']=str(root/'train-aggregate'); merged['scope_changed_sources']=sources
(root/'train-aggregate/train.json').write_text(json.dumps(merged,indent=2,ensure_ascii=False)+'\n')
PY

wait_gpu retrieval
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=.20 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
nice -n 10 taskset -c 8-15 python "$CODE/expert_ex1.py" production-eval-drafts \
  --out "$RUN/eval-drafts" --threads 8 --mem-fraction .20 --batch 8 > "$RUN/logs/retrieval.log" 2>&1
wait_gpu offline
nice -n 10 taskset -c 8-15 python "$CODE/expert_ex1.py" production-offline-eval \
  --train-dir "$RUN/train-aggregate" --drafts-dir "$RUN/eval-drafts" \
  --out "$RUN/offline" --threads 8 --mem-fraction .20 --batch 2 > "$RUN/logs/offline.log" 2>&1

wait_gpu closedloop-libero90
export XLA_PYTHON_CLIENT_MEM_FRACTION=.16 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 SKILLNET_SUBTASK_GENERATE=0
nice -n 10 taskset -c 16-39 python "$CODE/run_ex1_scope_changed.py" --suite libero90 \
  --train-dir "$RUN/train-aggregate" --candidate-cache "$RUN/eval-drafts/closedloop_p5_bank.npz" \
  --out "$RUN/closedloop-libero90" --workers 24 --fraction .16 --resume > "$RUN/logs/closedloop-libero90.log" 2>&1
wait_gpu closedloop-libero10
nice -n 10 taskset -c 16-39 python "$CODE/run_ex1_scope_changed.py" --suite libero10 \
  --train-dir "$RUN/train-aggregate" --candidate-cache "$RUN/eval-drafts/closedloop_p5_bank.npz" \
  --out "$RUN/closedloop-libero10" --workers 24 --fraction .16 --resume > "$RUN/logs/closedloop-libero10.log" 2>&1

OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python "$CODE/report_ex1_scope_changed.py" --run-dir "$RUN" --production-dir "$ROOT"
printf '%s\n' "complete $(date -Is)" > "$RUN/STATUS"
