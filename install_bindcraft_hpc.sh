#!/usr/bin/env bash
# install_bindcraft_hpc.sh
set -euo pipefail

# --------- settings you can tweak ---------
INSTALL_ROOT="${HOME}/apps"
BINDCRAFT_TAG="v1.5.2"        # latest release as of Jun 30, 2025
PKG_MGR="mamba"               # use mamba if available (faster), else conda
# ------------------------------------------

mkdir -p "${INSTALL_ROOT}"
INSTALL_DIR="${INSTALL_ROOT}/BindCraft"
LOG_DIR="${INSTALL_DIR}/_install_logs"

# 1) Ensure mamba/conda present (install Mambaforge locally if missing)
if ! command -v mamba >/dev/null 2>&1 && ! command -v conda >/dev/null 2>&1 ; then
  echo "[*] Installing Mambaforge under ${HOME}/mambaforge ..."
  cd "${HOME}"
  curl -L -o Mambaforge.sh \
    https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh
  bash Mambaforge.sh -b -p "${HOME}/mambaforge"
  rm -f Mambaforge.sh
  # shell init (non-interactive safe)
  source "${HOME}/mambaforge/etc/profile.d/conda.sh"
  CONDA_BASE="$(conda info --base)"
else
  if command -v mamba >/dev/null 2>&1; then
    source "$(mamba info --base)/etc/profile.d/conda.sh"
    CONDA_BASE="$(conda info --base)"
  else
    # conda only
    source "$(conda info --base)/etc/profile.d/conda.sh"
    CONDA_BASE="$(conda info --base)"
    PKG_MGR="conda"
  fi
fi

mkdir -p "${CONDA_BASE}/envs"
export CONDA_ENVS_PATH="${CONDA_BASE}/envs"

echo "[*] Using package manager: ${PKG_MGR}"

# 2) Detect CUDA version compatible with drivers to pass to the installer (fallback to 12.1)
CUDA_VER_DEFAULT="12.1"
CUDA_VER="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9.]*\).*/\1/p' | head -n1 || true)"
CUDA_VER="${CUDA_VER:-$CUDA_VER_DEFAULT}"
echo "[*] Will install for CUDA ${CUDA_VER}"

# 3) Clone BindCraft at a tagged release (reproducible) or update if exists
if [ -d "${INSTALL_DIR}/.git" ]; then
  echo "[*] Found existing repo at ${INSTALL_DIR}; updating..."
  git -C "${INSTALL_DIR}" fetch --tags
  git -C "${INSTALL_DIR}" checkout "${BINDCRAFT_TAG}"
  git -C "${INSTALL_DIR}" pull --ff-only || true
else
  echo "[*] Cloning BindCraft into ${INSTALL_DIR} ..."
  git clone --branch "${BINDCRAFT_TAG}" --depth 1 https://github.com/martinpacesa/BindCraft "${INSTALL_DIR}"
fi

mkdir -p "${LOG_DIR}"

# 4) Run official installer (creates conda env 'BindCraft' and downloads AF2 weights ~5.3 GB)
cd "${INSTALL_DIR}"
echo "[*] Running BindCraft installer (this creates env 'BindCraft' and pulls model weights)..."
# examples from official README: bash install_bindcraft.sh --cuda '12.4' --pkg_manager 'conda'
# We pass the detected CUDA version and chosen pkg manager.
bash install_bindcraft.sh --cuda "${CUDA_VER}" --pkg_manager "${PKG_MGR}" | tee "${LOG_DIR}/install.log"

echo
echo "[✓] BindCraft install finished."
echo "    Repo: ${INSTALL_DIR}"
echo "    Conda env: BindCraft"
echo "    Install logs: ${LOG_DIR}/install.log"
