#!/usr/bin/env bash
set -euo pipefail

STATE=${SKILLNET_STATE_DIR:-/work/.skillnet-state}
CODE=/work/skill_moe/skillnet/examples/memory_pilot/unseen_libero
OUT=${EXPERT_EX2_RUN_DIR:-$STATE/outputs/memory-pilot/unseen-libero/expert-ex2-01/production-05}
if [[ -e "$OUT/PRODUCTION_STARTED" && "${EXPERT_EX2_RESUME:-0}" != 1 ]]; then
  echo "refusing to reuse existing EX2 output directory: $OUT" >&2
  exit 2
fi
mkdir -p "$OUT/logs"
touch "$OUT/PRODUCTION_STARTED"
export SKILLNET_VENV=${SKILLNET_VENV:-$STATE/envs/skillnet}
export SKILLNET_STATE_DIR=$STATE
export SKILLNET_SUBTASK_GENERATE=0
if ! source /work/env/activate.sh; then
  # The shared activation script returns nonzero when optional LIBERO assets
  # are absent, even though the SkillNet environment itself is usable.
  [[ -x "$SKILLNET_VENV/bin/python" ]] || exit 1
  export PATH="$SKILLNET_VENV/bin:$PATH"
fi
cd /work
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=.20
export PYTHONPATH="/work/skill_moe/skillnet/src:/work/skill_moe/skillnet:/work/skill_moe/skillnet/packages/skillnet-client/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -f "$OUT/retrieval/retrieval_manifest.json" || ! -f "$OUT/dev-selection.npz" ||
      ! -f "$OUT/rank-samples/rank_sample_manifest.json" || ! -f "$OUT/confidence/confidence_fit.json" ]]; then
  nice -n 10 taskset -c 8-15 python -u "$CODE/expert_ex2.py" prepare \
    --output "$OUT" --threads 8 --memory-fraction .20 --device cuda:0 \
    >"$OUT/logs/prepare.log" 2>&1
fi

# Training and batched bridge evaluation need more XLA arena space than the
# small CPU/GPU cache-preparation phase. This remains below the shared H200
# capacity observed with other jobs resident.
export XLA_PYTHON_CLIENT_MEM_FRACTION=.72

for condition in A B; do
  if [[ -f "$OUT/train-$condition/train.json" ]] && \
     python -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('status') == 'complete' else 1)" "$OUT/train-$condition/train.json"; then
    continue
  fi
  nice -n 10 taskset -c 8-23 python -u "$CODE/expert_ex2.py" train \
    --output "$OUT" --condition "$condition" --updates 5000 --batch 8 --microbatch 1 \
    --eval-every 500 --threads 8 --memory-fraction .72 --train-scale 2.0 \
    >"$OUT/logs/train-${condition}.log" 2>&1
done

nice -n 10 taskset -c 8-15 python -u "$CODE/expert_ex2.py" aggregate --output "$OUT" \
  >"$OUT/logs/aggregate.log" 2>&1
nice -n 10 taskset -c 8-15 python -u "$CODE/expert_ex2.py" select \
  --retrieval "$OUT/retrieval" --train-dir "$OUT" --confidence "$OUT/confidence" \
  --output "$OUT/selected-schedule.json" --threads 8 --memory-fraction .25 \
  >"$OUT/logs/select.log" 2>&1
nice -n 10 taskset -c 8-15 python -u "$CODE/expert_ex2.py" offline \
  --retrieval "$OUT/retrieval" --train-dir "$OUT" --confidence "$OUT/confidence" \
  --schedule "$OUT/selected-schedule.json" --output "$OUT/offline" \
  --threads 8 --memory-fraction .25 --batch 4 \
  >"$OUT/logs/offline.log" 2>&1

export SKILLNET_SUBTASK_GENERATE=0
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  nice -n 10 taskset -c 8-47 python -u "$CODE/run_ex2_closed_loop.py" \
    --suite both --train-dir "$OUT" --retrieval-dir "$OUT/retrieval" \
    --candidate-cache "$STATE/outputs/memory-pilot/unseen-libero/expert-ex1-01/production-03/scope-change-01/eval-drafts/closedloop_p5_bank.npz" \
    --confidence "$OUT/confidence" --schedule "$OUT/selected-schedule.json" \
    --out "$OUT/closed-loop" --workers 24 --fraction .20 --resume \
    >"$OUT/logs/closed-loop.log" 2>&1

nice -n 10 taskset -c 8-15 python -u "$CODE/expert_ex2.py" finalize --run-dir "$OUT" \
  >"$OUT/logs/finalize.log" 2>&1
touch "$OUT/PRODUCTION_COMPLETE"
echo "EX2 production complete: $OUT"
