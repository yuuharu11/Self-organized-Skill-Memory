#!/usr/bin/env bash
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
state="${SKILLNET_STATE_DIR:-${root}/.skillnet-state}"
venv="${SKILLNET_ROBOCASA_VENV:-${state}/envs/robocasa}"
[[ -x "${venv}/bin/python" ]] || { echo "Missing RoboCasa venv: ${venv}; run env/robocasa/setup.sh" >&2; return 2; }
source "${venv}/bin/activate"
export SKILLNET_STATE_DIR="${state}" MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}" EGL_DEVICE_ID="${EGL_DEVICE_ID:-0}"
export ROBOCASA_DATASET_BASE_PATH="${state}/data/robocasa"
# Headless EGL through the NVIDIA driver (no /dev/dri access or root needed).
# Leave an explicit setting alone; switch to OSMesa with MUJOCO_GL=osmesa.
if [[ "${MUJOCO_GL}" == egl && -z "${__EGL_VENDOR_LIBRARY_FILENAMES:-}" ]]; then
  export __EGL_VENDOR_LIBRARY_FILENAMES="${root}/env/egl_nvidia.json"
fi
