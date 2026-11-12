#!/usr/bin/env bash
# Build P5 history references and train/evaluate Context key. FEATURE=final|input.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; source "$HERE/common.sh"
OUT="${1:-$R/ctx-01}"; FEATURE="${FEATURE:-final}"
export ROBOCASA_CTX_OUT="$OUT"
skillrun "$SKILL_ROOT/examples/memory_pilot/robocasa/p5_baselines.py"
skillrun "$SKILL_ROOT/examples/memory_pilot/robocasa/context_key.py" extract --out "$OUT" --batch-size 4
skillrun "$SKILL_ROOT/examples/memory_pilot/robocasa/context_key.py" train --out "$OUT" --epochs 20 --batch-size 128 --device cuda --feature "$FEATURE"
skillrun "$SKILL_ROOT/examples/memory_pilot/robocasa/context_key.py" evaluate --out "$OUT" --device cuda --feature "$FEATURE"
skillrun "$SKILL_ROOT/examples/memory_pilot/robocasa/context_key.py" report --out "$OUT"
