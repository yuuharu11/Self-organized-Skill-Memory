#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
state="${SKILLNET_STATE_DIR:-${root}/.skillnet-state}"
source "${root}/env/robocasa/activate.sh"
dataset="${1:-${state}/data/robocasa/v1.0/target/atomic/OpenDrawer/20250816/lerobot}"
demo="${dataset}/extras/episode_000000"
[[ -f "${demo}/states.npz" ]] || { echo "Missing smoke demo ${demo}; run python data_process/robocasa/download.py --preset smoke" >&2; exit 2; }
threads="${ROBOCASA_CPU_THREADS:-4}"
run_id="$(date -u +%Y%m%dT%H%M%S)-$$"
port="${ROBOCASA_SMOKE_PORT:-}"
if [[ -z "${port}" ]]; then
  port="$(OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python - <<'PY'
import socket
s=socket.socket();s.bind(('127.0.0.1',0));print(s.getsockname()[1]);s.close()
PY
)"
fi
mkdir -p "${state}/logs"
server_log="${state}/logs/robocasa-replay-server-${run_id}.log"
server_python="${SKILLNET_VENV:-${root}/.venv}/bin/python"
[[ -x "${server_python}" ]] || server_python="$(command -v python)"
OMP_NUM_THREADS="${threads}" MKL_NUM_THREADS="${threads}" OPENBLAS_NUM_THREADS="${threads}" \
PYTHONPATH="${root}/skill_moe/skillnet/packages/skillnet-client/src:${PYTHONPATH:-}" \
  "${server_python}" "${root}/skill_moe/skillnet/examples/robocasa/replay_server.py" --dataset "${dataset}" --port "${port}" >"${server_log}" 2>&1 &
server_pid=$!
cleanup(){ kill "${server_pid}" 2>/dev/null || true; wait "${server_pid}" 2>/dev/null || true; }
trap cleanup EXIT
ready=0
for _ in $(seq 1 60); do
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    cat "${server_log}" >&2
    echo 'replay server exited before listening' >&2
    exit 1
  fi
  if grep -q 'replay server listening' "${server_log}"; then ready=1; break; fi
  sleep 1
done
if [[ "${ready}" != 1 ]]; then cat "${server_log}" >&2; echo "replay server did not listen on ${port}" >&2; exit 1; fi
OMP_NUM_THREADS="${threads}" MKL_NUM_THREADS="${threads}" OPENBLAS_NUM_THREADS="${threads}" \
python "${root}/skill_moe/skillnet/examples/robocasa/main.py" --tasks OpenDrawer --split target --num-trials 1 --host 127.0.0.1 --port "${port}" --replay-demo-dir "${demo}" --dataset-root "${dataset}" --output "${state}/outputs/robocasa-eval-smoke-${run_id}"
