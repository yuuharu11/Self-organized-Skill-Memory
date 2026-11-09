#!/usr/bin/env bash
# Check host prerequisites without root. Nothing is installed; for each missing
# item a user-space fix (no sudo) is printed. Exit code is the number of
# missing required items.
set -uo pipefail

missing=0
ok()   { printf '  [ok]   %s\n' "$1"; }
miss() { printf '  [MISS] %s\n         -> %s\n' "$1" "$2"; missing=$((missing + 1)); }
warn() { printf '  [warn] %s\n         -> %s\n' "$1" "$2"; }
lib()  { ldconfig -p 2>/dev/null | grep -q "$1" || compgen -G "/usr/lib*/**/$1*" >/dev/null 2>&1 || compgen -G "/usr/lib/x86_64-linux-gnu/$1*" >/dev/null 2>&1; }

echo "User-space tools (installable without sudo)"
if command -v uv >/dev/null 2>&1; then ok "uv $(uv --version | awk '{print $2}')"
else miss "uv" 'curl -LsSf https://astral.sh/uv/0.8.22/install.sh | sh && export PATH="$HOME/.local/bin:$PATH"'; fi
command -v git >/dev/null 2>&1 && ok "git" || miss "git" "ask the administrator, or install git via conda/micromamba into your home directory"
command -v python3.10 >/dev/null 2>&1 && ok "python3.10 (SkillNet venv)" || miss "python3.10" 'uv python install 3.10   # puts python3.10 in ~/.local/bin'
command -v python3.11 >/dev/null 2>&1 && ok "python3.11 (RoboCasa venv)" || warn "python3.11 not on PATH" 'env/robocasa/setup.sh runs "uv python install 3.11" itself'
if command -v cmake >/dev/null 2>&1; then
  major="$(cmake --version | awk 'NR==1{split($3,v,"."); print v[1]}')"
  [[ "${major}" == 3 ]] && ok "cmake $(cmake --version | awk 'NR==1{print $3}')" \
    || miss "cmake 3.x (found ${major}.x; egl-probe needs <4)" 'uv tool install "cmake<4" && export PATH="$HOME/.local/bin:$PATH"'
else
  miss "cmake 3.x (needed to build egl-probe in the SkillNet venv)" 'uv tool install "cmake<4" && export PATH="$HOME/.local/bin:$PATH"'
fi

echo "Provided by the system image (cannot be installed without admin; usually present on GPU hosts)"
for tool in gcc g++ make; do
  command -v "${tool}" >/dev/null 2>&1 && ok "${tool}" || miss "${tool}" "ask the administrator for build-essential, or use a conda/micromamba compiler toolchain in your home directory"
done
command -v nvidia-smi >/dev/null 2>&1 && ok "NVIDIA driver ($(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1))" \
  || miss "NVIDIA driver / nvidia-smi" "GPU hosts only; ask the administrator"
lib libGL.so.1  && ok "libGL.so.1"  || miss "libGL.so.1 (libglvnd)" "ask the administrator, or use a container image with libglvnd"
lib libEGL.so.1 && ok "libEGL.so.1" || miss "libEGL.so.1 (libglvnd)" "ask the administrator, or use a container image with libglvnd"
if lib libEGL_nvidia.so.0; then ok "libEGL_nvidia.so.0 (headless rendering via the NVIDIA EGL vendor file env/egl_nvidia.json)"
elif lib libOSMesa.so; then warn "no NVIDIA EGL; libOSMesa found" "export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa before rendering (slow, CPU)"
else miss "a headless renderer (NVIDIA EGL or OSMesa)" "install the NVIDIA driver's EGL library (admin), or mesalib from conda-forge into your home and add it to LD_LIBRARY_PATH"; fi
if command -v ffmpeg >/dev/null 2>&1; then ok "ffmpeg (optional)"; else ok "ffmpeg not installed (not needed: video decoding uses PyAV's bundled FFmpeg)"; fi

echo "Disk"
state="${SKILLNET_STATE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.skillnet-state}"
mkdir -p "${state}" 2>/dev/null && printf '  SKILLNET_STATE_DIR=%s, free: %s\n' "${state}" "$(df -h "${state}" | awk 'NR==2{print $4}')"
echo "  needs: ~10 GB kitchen assets, ~1 GB smoke data, ~15 GB pi05_base weights; full atomic data is much larger"

echo
if [[ "${missing}" == 0 ]]; then echo "All required prerequisites found."; else echo "${missing} required item(s) missing."; fi
exit "${missing}"
