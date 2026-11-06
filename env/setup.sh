#!/usr/bin/env bash
set -euo pipefail

# Prepare the pinned Linux/Python environment after cloning this repository.
# System packages, NVIDIA drivers, and large experiment datasets are managed by
# the user because they depend on the host and selected experiment.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
state_dir="${SKILLNET_STATE_DIR:-${repo_root}/.skillnet-state}"
venv_dir="${SKILLNET_VENV:-${repo_root}/.venv}"
libero_source="${SKILLNET_LIBERO_SOURCE:-${state_dir}/sources/LIBERO}"
libero_commit="8f1084e3132a39270c3a13ebe37270a43ece2a01"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install uv 0.8.22, then rerun: bash env/setup.sh" >&2
  exit 2
fi
if ! command -v git >/dev/null 2>&1; then
  echo "git is required." >&2
  exit 2
fi
if ! command -v python3.10 >/dev/null 2>&1; then
  echo "Python 3.10 is required (python3.10 was not found)." >&2
  exit 2
fi

mkdir -p "${state_dir}/sources"
if [[ ! -d "${libero_source}/.git" ]]; then
  if [[ -e "${libero_source}" ]]; then
    echo "LIBERO destination exists but is not a Git checkout: ${libero_source}" >&2
    exit 2
  fi
  git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "${libero_source}"
fi
current_commit="$(git -C "${libero_source}" rev-parse HEAD)"
if [[ "${current_commit}" != "${libero_commit}" ]]; then
  if [[ -n "$(git -C "${libero_source}" status --porcelain)" ]]; then
    echo "LIBERO checkout has local changes; refusing to switch its revision: ${libero_source}" >&2
    exit 2
  fi
  git -C "${libero_source}" checkout --detach "${libero_commit}"
fi

uv venv --python 3.10 "${venv_dir}"
(
  cd "${repo_root}/env"
  uv pip sync --python "${venv_dir}/bin/python" requirements.lock
)
uv pip install --python "${venv_dir}/bin/python" --no-deps -e "${libero_source}"

export SKILLNET_REPO_ROOT="${repo_root}"
export SKILLNET_STATE_DIR="${state_dir}"
export SKILLNET_VENV="${venv_dir}"
export SKILLNET_LIBERO_SOURCE="${libero_source}"
unset LEROBOT_HOME || true
source "${repo_root}/env/activate.sh"
python "${repo_root}/skill_moe/skillnet/examples/libero/install_libero_skill_assets.py" \
  --libero-root "${libero_source}/libero/libero" --install
uv pip check --python "${venv_dir}/bin/python"

cat <<EOF

Setup complete.
Next: source env/activate.sh
Quick validation: python env/smoke_test.py
Public checkpoint evaluation: python env/download_checkpoint.py 40 && bash env/eval_smoke.sh
Full experiment reproduction: see docs/reproduction.md
EOF
