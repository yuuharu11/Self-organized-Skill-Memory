#!/usr/bin/env bash
# RM-EXP2 prep, 3000-step Expert training, and fixed validation.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; source "$HERE/common.sh"
ROOT="$R/expert-02"; PREP="${1:-$ROOT/prep/attempt-10}"; TRAIN="${2:-$ROOT/train/attempt-04}"
[[ ! -e "$PREP" && ! -e "$TRAIN" ]] || { echo 'Refusing existing stage output; choose a new directory.' >&2; exit 2; }
mkdir -p "$PREP" "$TRAIN"
export ROBOCASA_EXPERT2_PREP="$PREP" ROBOCASA_EXPERT2_OUT="$TRAIN"
export XLA_PYTHON_CLIENT_MEM_FRACTION=.30
skillrun "$SKILL_ROOT/examples/memory_pilot/robocasa/expert2_prep.py" --out "$PREP" --steps 200 --microbatch 4 --effective-batch 32 --device cuda
skillrun "$SKILL_ROOT/examples/memory_pilot/robocasa/expert2_train.py" --out "$TRAIN" --steps 3000 --microbatch 8 --effective-batch 32 --threads 8 --fraction .30
skillrun "$SKILL_ROOT/examples/memory_pilot/robocasa/expert2_validate.py" --out "$TRAIN" --threads 8 --fraction .30
