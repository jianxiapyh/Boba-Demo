#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
EXPECTED_CONDA_ENV="boba"
ACTIVE_CONDA_ENV="${CONDA_DEFAULT_ENV:-}"
PYCUDA_VERSION="2026.1"

if [[ -z "${ACTIVE_CONDA_ENV}" && -n "${CONDA_PREFIX:-}" ]]; then
  ACTIVE_CONDA_ENV="$(basename "${CONDA_PREFIX}")"
fi

if [[ "${ACTIVE_CONDA_ENV}" != "${EXPECTED_CONDA_ENV}" ]]; then
  echo "Activate the '${EXPECTED_CONDA_ENV}' conda environment before running this installer." >&2
  echo "Active environment: ${ACTIVE_CONDA_ENV:-<none>}" >&2
  exit 1
fi

if [[ -z "${CONDA_PREFIX:-}" || ! -d "${CONDA_PREFIX}" ]]; then
  echo "CONDA_PREFIX is not set for the active '${EXPECTED_CONDA_ENV}' environment." >&2
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found on PATH. Run this from a shell with conda initialized." >&2
  exit 1
fi

cd "${REPO_ROOT}"

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export BOBA_GSPLAT_SOURCE_ROOT="${BOBA_GSPLAT_SOURCE_ROOT:-${WORKSPACE_ROOT}/Boba/gaussian_splatting/submodules/gsplat}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"

if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
  echo "Expected CUDA compiler at: ${CUDA_HOME}/bin/nvcc" >&2
  exit 1
fi

if [[ ! -d "${BOBA_GSPLAT_SOURCE_ROOT}" ]]; then
  echo "Expected vendored gsplat source at: ${BOBA_GSPLAT_SOURCE_ROOT}" >&2
  echo "Set BOBA_GSPLAT_SOURCE_ROOT to the sibling Boba gsplat source root before running this installer." >&2
  exit 1
fi

conda install -y numpy=1.26.4 opencv libstdcxx-ng libgcc-ng libgl-devel glew git-lfs

# Keep numpy on the runtime-tested 1.26 baseline even if pip-installed packages
# previously upgraded it inside the shared boba environment.
python -m pip install --force-reinstall numpy==1.26.4

python -m pip install --upgrade \
  "open3d==0.19.0" \
  trimesh \
  pyrender \
  "pyglet<2" \
  rtree \
  einops \
  warp-lang \
  termcolor \
  imageio \
  glfw \
  kornia \
  plyfile

python -m pip uninstall -y gsplat || true
BUILD_NO_CUDA=1 python -m pip install -e "${BOBA_GSPLAT_SOURCE_ROOT}"

python -m pip uninstall -y pycuda || true
pycuda_build_root="$(mktemp -d)"
trap 'rm -rf "${pycuda_build_root}"' EXIT
python -m pip download --no-binary pycuda "pycuda==${PYCUDA_VERSION}" -d "${pycuda_build_root}"
tar -xf "${pycuda_build_root}/pycuda-${PYCUDA_VERSION}.tar.gz" -C "${pycuda_build_root}"
pushd "${pycuda_build_root}/pycuda-${PYCUDA_VERSION}" >/dev/null
rm -f siteconf.py
python configure.py \
  --cuda-root="${CUDA_HOME}" \
  --cuda-enable-gl \
  --cuda-inc-dir="${CUDA_HOME}/include,${CONDA_PREFIX}/include" \
  --cxxflags=-I"${CUDA_HOME}/include" \
  --ldflags=-L"${CUDA_HOME}/lib64"
python -m pip install --no-build-isolation --no-deps .
popd >/dev/null

python - <<'PY'
import pycuda.gl
print("Verified pycuda.gl")
PY

pushd gaussian_splatting/submodules/diff-gaussian-rasterization >/dev/null
python setup.py build_ext --inplace
python -m pip install --no-build-isolation -e .
popd >/dev/null

pushd gaussian_splatting/submodules/simple-knn >/dev/null
python setup.py build_ext --inplace
python -m pip install --no-build-isolation -e .
popd >/dev/null

compgen -G "${REPO_ROOT}/gaussian_splatting/submodules/diff-gaussian-rasterization/diff_gaussian_rasterization/_C*.so" >/dev/null || {
  echo "diff-gaussian-rasterization CUDA extension was not built." >&2
  exit 1
}

compgen -G "${REPO_ROOT}/gaussian_splatting/submodules/simple-knn/simple_knn/_C*.so" >/dev/null || {
  echo "simple-knn CUDA extension was not built." >&2
  exit 1
}

env -u LD_LIBRARY_PATH git -C "${REPO_ROOT}" lfs install --local
python tools/fetch_demo_case_assets.py --all
env -u LD_LIBRARY_PATH bash "${REPO_ROOT}/linux_pose_probe/check_boba_immersive_bridge_deps.sh"

echo "RTX6000 Boba-Demo environment install complete for '${EXPECTED_CONDA_ENV}'."
