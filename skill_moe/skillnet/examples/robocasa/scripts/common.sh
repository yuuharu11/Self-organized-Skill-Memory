#!/usr/bin/env bash
# Shared path/environment setup for the research replay scripts.
set -euo pipefail
: "${SKILLNET_STATE_DIR:?Set SKILLNET_STATE_DIR to a writable artifact root}"
REPO_ROOT="${SKILLNET_REPO_ROOT:-$(git rev-parse --show-toplevel)}"
R="$SKILLNET_STATE_DIR/outputs/robocasa-memory"
SKILL_ROOT="$REPO_ROOT/skill_moe/skillnet"
SKILLNET_VENV="${SKILLNET_VENV:-$SKILLNET_STATE_DIR/envs/skillnet}"
SKILLNET_ROBOCASA_VENV="${SKILLNET_ROBOCASA_VENV:-$SKILLNET_STATE_DIR/envs/robocasa}"
SKILL_PY="$SKILLNET_VENV/bin/python"
ROBO_PY="$SKILLNET_ROBOCASA_VENV/bin/python"
export SKILLNET_VENV SKILLNET_ROBOCASA_VENV
export SKILLNET_REPO_ROOT="$REPO_ROOT"
export SKILLNET_SUBTASK_GENERATE=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$OMP_NUM_THREADS}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-.20}"
export PYTHONPATH="$SKILL_ROOT/src:$SKILL_ROOT/packages/skillnet-client/src:$SKILL_ROOT/examples/robocasa:$SKILL_ROOT/examples/memory_pilot/robocasa${PYTHONPATH:+:$PYTHONPATH}"
SKILLNET_CPUSET="${SKILLNET_CPUSET:-8-15}"
skillrun() { taskset -c "$SKILLNET_CPUSET" nice -n 10 "$SKILL_PY" "$@"; }
