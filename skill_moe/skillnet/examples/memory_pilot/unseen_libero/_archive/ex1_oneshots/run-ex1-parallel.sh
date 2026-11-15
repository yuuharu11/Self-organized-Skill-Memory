#!/usr/bin/env bash
set -euo pipefail

# EX1 training stage only. PREPARED must contain EXP3 row caches augmented by
# expert_ex1.attach_training_mixture (actual KEY2-F P5 index cache required).
ROOT="${SKILLNET_STATE_DIR:?set SKILLNET_STATE_DIR}/outputs/memory-pilot/unseen-libero/expert-ex1-01"
PREPARED="${EX1_PREPARED:?set EX1_PREPARED to a new prepared cache}"
STEPS="${EX1_STEPS:-5000}"
THREADS="${EX1_THREADS:-8}"
BATCH="${EX1_BATCH:-8}"
MICROBATCH="${EX1_MICROBATCH:-2}"
MEM_FRACTION="${EX1_MEM_FRACTION:-0.20}"
DEST="${EX1_TRAIN_OUT:-$ROOT/production-01}"
LOGS="$DEST/logs"
SCRIPT="$(cd "$(dirname "$0")" && pwd)/skill_expert_refine.py"

[[ -f "$PREPARED/prepare.json" && -f "$PREPARED/train_a.npz" && -f "$PREPARED/train_l90x.npz" ]] || {
  echo "Incomplete EX1 prepared cache: $PREPARED" >&2; exit 2;
}
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS"
python - "$PREPARED" <<'PY'
import json,sys,numpy as np
from pathlib import Path
p=Path(sys.argv[1]); meta=json.loads((p/'prepare.json').read_text())
if meta.get('smoke') or meta.get('dev_draft_selection_method')!='key2f_retrieval':
 raise SystemExit('production training requires full rows and actual locked KEY2-F Dev retrieval')
if not (p/'dev_drafts.npz').is_file(): raise SystemExit('missing independent D1-D4 Dev draft tensors')
with np.load(p/'dev_drafts.npz',allow_pickle=False) as d:
 required={'D1_P5_top1','D2_P5_top20','D3_E_VLM_H_top1','D4_E_VJ_H_top20'}
 if set(d.files)!=required: raise SystemExit(f'Dev D1-D4 draft modes mismatch: {d.files}')
missing={int(x['task_id'] if isinstance(x,dict) else x)
         for x in meta.get('l90x_missing_tasks',[])}
if missing != {51}: raise SystemExit(f'expected only documented L90X task 51 missing: {sorted(missing)}')
for name in ('train_a','train_l90x'):
 with np.load(p/f'{name}.npz',allow_pickle=False) as x:
  if not {'nearest_top3','p5_top1','nearest_top3_unit_ids','p5_unit_ids'}<=set(x.files):
   raise SystemExit(f'{name} lacks actual EX1 draft-mixture arrays; run mixture preparation first')
PY
[[ ! -e "$DEST/A" && ! -e "$DEST/B" ]] || { echo "Output exists: $DEST" >&2; exit 2; }
mkdir -p "$LOGS"

taskset -c 8-23 nice -n 10 env XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION="$MEM_FRACTION" \
  python "$SCRIPT" train --prepared-dir "$PREPARED" --out "$DEST/A" --condition A \
  --steps "$STEPS" --threads "$THREADS" --batch "$BATCH" --microbatch "$MICROBATCH" \
  --mem-fraction "$MEM_FRACTION" >"$LOGS/A.log" 2>&1 & pid_a=$!
taskset -c 24-39 nice -n 10 env XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION="$MEM_FRACTION" \
  python "$SCRIPT" train --prepared-dir "$PREPARED" --out "$DEST/B" --condition B \
  --steps "$STEPS" --threads "$THREADS" --batch "$BATCH" --microbatch "$MICROBATCH" \
  --mem-fraction "$MEM_FRACTION" >"$LOGS/B.log" 2>&1 & pid_b=$!
status=0
wait "$pid_a" || status=1
wait "$pid_b" || status=1
(( status == 0 )) || { echo "EX1 A/B training failed; see $LOGS" >&2; exit 1; }
echo "EX1 A/B training complete: $DEST"
