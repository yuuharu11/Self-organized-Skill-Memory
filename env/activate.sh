#!/usr/bin/env bash
# Source this file in the existing container: source env/activate.sh
_skillnet_activate() {
  local skillnet_root skillnet_venv skillnet_libero skillnet_egl
  skillnet_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || return
  skillnet_venv="${SKILLNET_VENV:-${VIRTUAL_ENV:-${skillnet_root}/.venv}}"
  if [[ ! -f "${skillnet_venv}/bin/activate" && -z "${SKILLNET_VENV:-}${VIRTUAL_ENV:-}" && -f /tmp/skillnet-validation/bin/activate ]]; then
    skillnet_venv=/tmp/skillnet-validation
  fi
  if [[ ! -f "${skillnet_venv}/bin/activate" ]]; then
    echo "Set SKILLNET_VENV to an installed Python 3.10 environment; see docs/development_environment.md." >&2
    return 2
  fi
  if [[ -v LEROBOT_HOME ]]; then
    echo "This LeRobot version requires HF_LEROBOT_HOME; unset LEROBOT_HOME first." >&2
    return 2
  fi
  source "${skillnet_venv}/bin/activate" || return
  export SKILLNET_REPO_ROOT="${skillnet_root}"
  export SKILLNET_STATE_DIR="${SKILLNET_STATE_DIR:-${skillnet_root}/.skillnet-state}"
  skillnet_libero="${SKILLNET_LIBERO_SOURCE:-}"
  if [[ -z "${skillnet_libero}" ]]; then
    skillnet_libero="$(python - <<'LIBERO_PY'
import importlib.metadata
import json
from urllib.parse import unquote, urlparse
try:
    metadata = importlib.metadata.distribution('libero').read_text('direct_url.json')
    url = urlparse(json.loads(metadata or '{}').get('url', ''))
    if url.scheme == 'file':
        print(unquote(url.path))
except (importlib.metadata.PackageNotFoundError, ValueError):
    pass
LIBERO_PY
)" || return
  fi
  export PYTHONPATH="${skillnet_root}/skill_moe/skillnet/src:${skillnet_root}/skill_moe/skillnet/packages/skillnet-client/src:${skillnet_libero}${PYTHONPATH:+:${PYTHONPATH}}"
  export HF_HOME="${HF_HOME:-${SKILLNET_STATE_DIR}/cache/huggingface}"
  export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${SKILLNET_STATE_DIR}/data/lerobot}"
  export SKILLNET_DATA_HOME="${SKILLNET_DATA_HOME:-${SKILLNET_STATE_DIR}/cache/skillnet}"
  export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${SKILLNET_STATE_DIR}/cache/libero}"
  export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-${SKILLNET_STATE_DIR}/cache/jax}"
  export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${SKILLNET_STATE_DIR}/cache/numba}"
  export MPLCONFIGDIR="${MPLCONFIGDIR:-${SKILLNET_STATE_DIR}/cache/matplotlib}"
  export MUJOCO_GL="${MUJOCO_GL:-egl}"
  export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
  export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
  export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
  export PYTHON_BIN=python
  skillnet_egl="${SKILLNET_EGL_LIBRARY_DIR:-}"
  if [[ -z "${skillnet_egl}" && -d /tmp/skillnet-render-libs/usr/lib/x86_64-linux-gnu ]]; then
    skillnet_egl=/tmp/skillnet-render-libs/usr/lib/x86_64-linux-gnu
  fi
  if [[ -n "${skillnet_egl}" ]]; then
    export LD_LIBRARY_PATH="${skillnet_egl}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  fi
  export __EGL_VENDOR_LIBRARY_FILENAMES="${__EGL_VENDOR_LIBRARY_FILENAMES:-${skillnet_root}/env/egl_nvidia.json}"
  mkdir -p "${HF_HOME}" "${HF_LEROBOT_HOME}" "${LIBERO_CONFIG_PATH}" || return
  python "${skillnet_root}/env/configure_libero.py" || return
  echo "SkillNet environment: ${VIRTUAL_ENV}"
}
_skillnet_activate
