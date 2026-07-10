#!/usr/bin/env bash
set -euo pipefail

EXPECTED_ENV="phystwin"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
export BOBA_BATCHED_ROOT="${BOBA_BATCHED_ROOT:-/home/yihan/Research/Boba_Latest}"

die() {
  printf '[Demo2 launcher] ERROR: %s\n' "$*" >&2
  exit 1
}

active_env_name="${CONDA_DEFAULT_ENV:-}"
active_env_name="${active_env_name##*/}"
if [[ "${active_env_name}" != "${EXPECTED_ENV}" ]]; then
  die "activate ${EXPECTED_ENV} before starting Demo 2 (active: ${CONDA_DEFAULT_ENV:-none})."
fi
if [[ -z "${CONDA_PREFIX:-}" || ! -x "${CONDA_PREFIX}/bin/python" ]]; then
  die "CONDA_PREFIX does not identify the active ${EXPECTED_ENV} environment."
fi
if [[ ! -d "${BOBA_BATCHED_ROOT}" ]]; then
  die "BOBA_BATCHED_ROOT is not a directory: ${BOBA_BATCHED_ROOT}"
fi

resolved_cuda_home="${CUDA_HOME:-}"
if [[ -z "${resolved_cuda_home}" || ! -x "${resolved_cuda_home}/bin/nvcc" ]]; then
  if command -v nvcc >/dev/null 2>&1; then
    resolved_nvcc="$(readlink -f "$(command -v nvcc)")"
    resolved_cuda_home="$(cd "$(dirname "${resolved_nvcc}")/.." && pwd -P)"
  elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
    resolved_cuda_home="/usr/local/cuda"
  else
    die "CUDA_HOME is unset and nvcc was not found."
  fi
fi

export CUDA_HOME="${resolved_cuda_home}"
export PYTHONNOUSERSITE=1
export PATH="${CONDA_PREFIX}/bin:${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

cd "${REPO_ROOT}"
exec "${CONDA_PREFIX}/bin/python" -u "${REPO_ROOT}/demos/demo2_server.py" "$@"
