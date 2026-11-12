#!/usr/bin/env bash
# Train/evaluate Context gripper and MAG-v2. Optional old head is separate.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; source "$HERE/common.sh"
OUT="${1:-$R/gripper-01}"
skillrun "$SKILL_ROOT/examples/memory_pilot/robocasa/gripper.py" select --out "$OUT"
skillrun "$SKILL_ROOT/examples/memory_pilot/robocasa/gripper.py" normal --out "$OUT"
skillrun "$SKILL_ROOT/examples/memory_pilot/robocasa/gripper.py" evaluate --out "$OUT"
skillrun "$SKILL_ROOT/examples/memory_pilot/robocasa/gripper.py" report --out "$OUT"
skillrun "$SKILL_ROOT/examples/robocasa/train_memonly_magnitude_nohist.py" --output "$R/key-lite-01/magnitude" --max-epochs 150
skillrun "$SKILL_ROOT/examples/robocasa/train_memonly_magnitude_norm.py" --output "$R/key-lite-01/magnitude_v2" --max-epochs 150
