#!/usr/bin/env bash
# RM-EXP3: corrected dynamic t0 drafts, Expert fit, fixed-grid validation and report.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; source "$HERE/common.sh"
ROOT="${ROBOCASA_EXPERT3_ROOT:-$R/expert-03}"
[[ ! -e "$ROOT/prep/drafts/memory_dynamic.npz" && ! -e "$ROOT/train/expert/train_summary.json" ]] || { echo 'Refusing to overwrite Expert-03 outputs.' >&2; exit 2; }
export ROBOCASA_EXPERT3_ROOT="$ROOT" ROBOCASA_EXPERT3_PREP="$ROOT/prep" ROBOCASA_EXPERT3_TRAIN="$ROOT/train"
skillrun "$SKILL_ROOT/examples/memory_pilot/robocasa/expert3_prep.py"
export XLA_PYTHON_CLIENT_MEM_FRACTION=.30
skillrun "$SKILL_ROOT/examples/memory_pilot/robocasa/expert3_train.py" --steps 3000 --microbatch 8 --effective-batch 32 --threads 8 --fraction .30
skillrun "$SKILL_ROOT/examples/memory_pilot/robocasa/expert3_validate.py" --threads 8 --fraction .30
skillrun "$SKILL_ROOT/examples/memory_pilot/robocasa/expert3_report.py"
