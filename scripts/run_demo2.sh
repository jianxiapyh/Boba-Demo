#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"

die() {
  printf '[Demo2 launcher] ERROR: %s\n' "$*" >&2
  exit 1
}

active_env_name="${CONDA_DEFAULT_ENV:-}"
active_env_name="${active_env_name##*/}"
if [[ "${active_env_name}" != "phystwin-cu132" ]]; then
  die "activate phystwin-cu132 before starting Demo 2 (active: ${CONDA_DEFAULT_ENV:-none})."
fi
if [[ -z "${CONDA_PREFIX:-}" || ! -x "${CONDA_PREFIX}/bin/python" ]]; then
  die "CONDA_PREFIX does not identify the active ${active_env_name} environment."
fi
resolved_cuda_home="${CUDA_HOME:-}"
if [[ -z "${resolved_cuda_home}" || ! -x "${resolved_cuda_home}/bin/nvcc" ]]; then
  if [[ -x "${CONDA_PREFIX}/bin/nvcc" ]]; then
    resolved_cuda_home="${CONDA_PREFIX}"
  elif command -v nvcc >/dev/null 2>&1; then
    resolved_nvcc="$(readlink -f "$(command -v nvcc)")"
    resolved_cuda_home="$(cd "$(dirname "${resolved_nvcc}")/.." && pwd -P)"
  elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
    resolved_cuda_home="/usr/local/cuda"
  else
    die "CUDA_HOME is unset and nvcc was not found."
  fi
fi

export CUDA_HOME="${resolved_cuda_home}"
# Conda CUDA toolkits keep headers under a target directory. PyTorch's
# extension builder needs this path when compiling gsplat's C++ sources.
cuda_target_include="${CUDA_HOME}/targets/$(uname -m)-linux/include"
if [[ -z "${CUDA_INC_PATH:-}" && -f "${cuda_target_include}/cuda_runtime.h" ]]; then
  export CUDA_INC_PATH="${cuda_target_include}"
fi
export PYTHONNOUSERSITE=1
export PATH="${CONDA_PREFIX}/bin:${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

cd "${REPO_ROOT}"
exec "${CONDA_PREFIX}/bin/python" -u "${REPO_ROOT}/demos/demo2_server.py" "$@"
