#!/usr/bin/env bash
# Offline diagnosis drivers; outputs go to their canonical child directories.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; source "$HERE/common.sh"
export ROBOCASA_ACT_ORACLE_OUT="${ROBOCASA_ACT_ORACLE_OUT:-$R/expert-02/act-oracle}"
export ROBOCASA_GAP_OUT="${ROBOCASA_GAP_OUT:-$R/expert-02/gap}"
export ROBOCASA_OUTPUT_ROOT="$R"
skillrun "$SKILL_ROOT/examples/memory_pilot/robocasa/act_oracle.py"
skillrun "$SKILL_ROOT/examples/memory_pilot/robocasa/gap_offline_compare.py"
skillrun "$SKILL_ROOT/examples/memory_pilot/robocasa/gap_postprocess.py"
skillrun "$SKILL_ROOT/examples/memory_pilot/robocasa/gap_log_analysis.py"
skillrun "$HERE/key-diag.py" --out "$R/expert-02/key-diag"
