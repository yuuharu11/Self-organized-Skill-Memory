#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; source "$HERE/common.sh"
exec taskset -c "$SKILLNET_CPUSET" nice -n 10 "$SKILL_PY" "$HERE/run_key_lite.py" "$@"
