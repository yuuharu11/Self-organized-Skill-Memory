#!/usr/bin/env bash
# Offline trained reranker then optional closed-loop runner.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; source "$HERE/common.sh"
export ROBOCASA_RERANK_OUT="${1:-$R/rerank-01}"
skillrun "$SKILL_ROOT/examples/memory_pilot/robocasa/rerank.py"
