#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; source "$HERE/common.sh"
skillrun "$SKILL_ROOT/examples/memory_pilot/robocasa/memory18_prepare.py" all --batch-size "${BATCH_SIZE:-4}"
