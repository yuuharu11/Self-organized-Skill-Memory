#!/usr/bin/env bash
# Build stride-10 U memory and the frozen P5 reference artifacts.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; source "$HERE/common.sh"
OUT="${1:-$R/memory-01/policy-order-v2}"
mkdir -p "$(dirname "$OUT")"
skillrun "$SKILL_ROOT/examples/memory_pilot/robocasa/experiment.py" all --out "$OUT" --epochs 20 --batch 256 --device cuda --no-pi05
