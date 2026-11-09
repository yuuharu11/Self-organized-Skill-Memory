#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
state="${SKILLNET_STATE_DIR:-${root}/.skillnet-state}"
venv="${SKILLNET_ROBOCASA_VENV:-${state}/envs/robocasa}"
src="${state}/sources"
robocasa_rev=3443cdf3dd20f9ae8ed41adba442059b506891d0
robosuite_rev=5ce6643f3092639d08f7b0f90ed1c6a84f50552c
assets=0
if [[ "${1:-}" == "--assets" ]]; then assets=1; fi
mkdir -p "${state}" "${src}"
export MUJOCO_GL="${MUJOCO_GL:-egl}" PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
if ! command -v uv >/dev/null; then
  echo 'uv is required (tested with uv 0.8.22); install uv and rerun.' >&2; exit 2
fi
uv python install 3.11
for spec in "robocasa https://github.com/robocasa/robocasa.git ${robocasa_rev}" "robosuite https://github.com/ARISE-Initiative/robosuite.git ${robosuite_rev}"; do
  read -r name url rev <<<"${spec}"; dir="${src}/${name}"
  if [[ ! -d "${dir}/.git" ]]; then
    [[ ! -e "${dir}" ]] || { echo "Not a git checkout: ${dir}" >&2; exit 2; }
    git clone "${url}" "${dir}"
  fi
  current="$(git -C "${dir}" rev-parse HEAD)"
  if [[ "${current}" != "${rev}" ]]; then
    [[ -z "$(git -C "${dir}" status --porcelain)" ]] || { echo "Local changes; refusing checkout: ${dir}" >&2; exit 2; }
    git -C "${dir}" checkout --detach "${rev}"
  fi
done
[[ -x "${venv}/bin/python" ]] || uv venv --python 3.11 "${venv}"
lock_hash="$(sha256sum "${root}/env/robocasa/requirements.lock" | cut -d' ' -f1)"
ready="${venv}/.skillnet-robocasa-lock-sha256"
if [[ ! -f "${ready}" || "$(cat "${ready}")" != "${lock_hash}" ]]; then
  uv pip sync --python "${venv}/bin/python" "${root}/env/robocasa/requirements.lock"
  # Avoid an isolated build environment; the pinned environment already has
  # setuptools. This also keeps the editable install fully offline.
  uv pip install --python "${venv}/bin/python" --no-build-isolation --no-deps -e "${src}/robosuite" -e "${src}/robocasa"
  # This first-party client uses hatchling; allow uv to build it in an isolated
  # build environment while keeping its runtime requirements pinned in lock.
  uv pip install --python "${venv}/bin/python" --no-deps -e "${root}/skill_moe/skillnet/packages/skillnet-client"
  printf '%s\n' "${lock_hash}" >"${ready}"
fi
export SKILLNET_STATE_DIR="${state}" SKILLNET_ROBOCASA_VENV="${venv}"
if [[ ! -f "${src}/robocasa/robocasa/macros_private.py" ]]; then
  "${venv}/bin/python" -m robocasa.scripts.setup_macros
fi
if [[ ! -f "${src}/robosuite/robosuite/macros_private.py" ]]; then
  "${venv}/bin/python" -m robosuite.scripts.setup_macros
fi
"${venv}/bin/python" - <<'PY'
import os, pathlib, robocasa
p=pathlib.Path(robocasa.__file__).parent/'macros_private.py'
template=pathlib.Path(robocasa.__file__).parent/'macros.py'
text=template.read_text()
text=text.replace('DATASET_BASE_PATH = None', f"DATASET_BASE_PATH = {str(pathlib.Path(os.environ['SKILLNET_STATE_DIR'])/'data'/'robocasa')!r}")
p.write_text(text)
print(f'Configured {p}')
PY
mkdir -p "${state}/data/robocasa"
if [[ "${assets}" == "1" ]]; then
  if [[ ! -f "${state}/validation/robocasa_assets_complete" ]]; then
    mkdir -p "${state}/validation"
    (cd "${src}/robocasa" && printf 'y\n' | "${venv}/bin/python" -m robocasa.scripts.download_kitchen_assets --type all)
    touch "${state}/validation/robocasa_assets_complete"
  else
    echo 'Kitchen assets already present; skipping download.'
  fi
fi
"${venv}/bin/python" "${root}/data_process/robocasa/export_task_sets.py" --output "${root}/data_process/robocasa/task_sets.json"
echo "RoboCasa environment ready: ${venv}"
