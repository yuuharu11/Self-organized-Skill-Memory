#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  echo "Usage: bash env/eval_smoke.sh [evaluator arguments]"
  echo "Runs the released LIBERO-40 checkpoint on Spatial, one trial per task."
  echo "Overrides: SKILLNET_VENV, CKPT_DIR, TASK_SUITE, NUM_TRIALS, SKILLNET_RUN_DIR."
  exit 0
fi
source "${SCRIPT_DIR}/activate.sh"
cd "${SKILLNET_REPO_ROOT}/skill_moe/skillnet"
RUN_DIR="${SKILLNET_RUN_DIR:-${SKILLNET_STATE_DIR}/outputs/eval-smoke-$(date +%s)}"
export CONFIG_NAME=pi05_libero_moe_skill_4_40
export CKPT_DIR="${CKPT_DIR:-${SKILLNET_STATE_DIR}/checkpoints/pi05_libero_moe_skill_4_40/moe_add_balance_4_128_30000/29999}"
export TASK_SUITE="${TASK_SUITE:-libero_spatial}"
export SKILL_ANNOTATION_PATH=examples/libero/annotations/instruct2plan_40.json
export NUM_TRIALS="${NUM_TRIALS:-1}"
export ZERO_SHOT=0
export FAIL_FAST=1
export SERVER_WAIT_SECONDS="${SERVER_WAIT_SECONDS:-600}"
export VIDEO_OUT_PATH="${VIDEO_OUT_PATH:-${RUN_DIR}/videos}"
export SERVER_LOG_PATH="${SERVER_LOG_PATH:-${RUN_DIR}/server.log}"
mkdir -p "${RUN_DIR}"
bash examples/libero/run_eval_libero_skill_moe.sh "$@" 2>&1 | tee "${RUN_DIR}/evaluation.log"
