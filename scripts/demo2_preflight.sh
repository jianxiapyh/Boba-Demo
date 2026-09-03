#!/usr/bin/env bash
set -euo pipefail

EXPECTED_ENV="phystwin"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
failures=0

ok() {
  printf '[Demo2 preflight] OK: %s\n' "$*"
}

warn() {
  printf '[Demo2 preflight] WARNING: %s\n' "$*" >&2
}

fail() {
  printf '[Demo2 preflight] FAILED: %s\n' "$*" >&2
  failures=$((failures + 1))
}

active_env_name="${CONDA_DEFAULT_ENV:-}"
active_env_name="${active_env_name##*/}"
if [[ "${active_env_name}" == "${EXPECTED_ENV}" ]]; then
  ok "active Conda environment is ${EXPECTED_ENV}"
else
  fail "activate ${EXPECTED_ENV} first (active: ${CONDA_DEFAULT_ENV:-none})"
fi

if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PYTHON="${CONDA_PREFIX}/bin/python"
  python_prefix_name="$("${PYTHON}" -c 'import pathlib, sys; print(pathlib.Path(sys.prefix).resolve().name)' 2>/dev/null || true)"
  if [[ "${python_prefix_name}" == "${EXPECTED_ENV}" ]]; then
    ok "Python resolves inside ${EXPECTED_ENV}: ${PYTHON}"
  else
    fail "${PYTHON} resolves to environment ${python_prefix_name:-unknown}, not ${EXPECTED_ENV}"
  fi
else
  fail "CONDA_PREFIX does not contain an executable Python"
  printf '[Demo2 preflight] Stop: activate phystwin and rerun this preflight.\n' >&2
  exit 1
fi

export PYTHONNOUSERSITE=1

for runtime_marker in \
  "interactive_playground.py" \
  "qqtt/engine/trainer_warp.py" \
  "gaussian_splatting/submodules/gsplat/gsplat/__init__.py"; do
  if [[ -f "${REPO_ROOT}/${runtime_marker}" ]]; then
    ok "bundled Boba runtime exists: ${runtime_marker}"
  else
    fail "bundled Boba runtime file is missing: ${runtime_marker}"
  fi
done

if command -v nvidia-smi >/dev/null 2>&1; then
  if gpu_report="$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>&1)"; then
    ok "NVIDIA driver sees GPU(s)"
    printf '%s\n' "${gpu_report}" | sed 's/^/[Demo2 preflight]   /'
  else
    fail "nvidia-smi could not query the GPU: ${gpu_report}"
  fi
else
  fail "nvidia-smi is not on PATH"
fi

resolved_cuda_home="${CUDA_HOME:-}"
if [[ -z "${resolved_cuda_home}" || ! -x "${resolved_cuda_home}/bin/nvcc" ]]; then
  if command -v nvcc >/dev/null 2>&1; then
    resolved_nvcc="$(readlink -f "$(command -v nvcc)")"
    resolved_cuda_home="$(cd "$(dirname "${resolved_nvcc}")/.." && pwd -P)"
  elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
    resolved_cuda_home="/usr/local/cuda"
  fi
fi
if [[ -n "${resolved_cuda_home}" && -x "${resolved_cuda_home}/bin/nvcc" ]]; then
  export CUDA_HOME="${resolved_cuda_home}"
  ok "CUDA toolkit: ${CUDA_HOME}"
  "${CUDA_HOME}/bin/nvcc" --version | tail -n 1 | sed 's/^/[Demo2 preflight]   /'
else
  fail "a CUDA toolkit with nvcc is required; set CUDA_HOME"
fi

printf '[Demo2 preflight] Core package versions:\n'
if ! "${PYTHON}" <<'PY'
from importlib import metadata

missing = []
for name in (
    "torch", "torchvision", "torchaudio", "numpy", "scipy", "warp-lang",
    "pycuda", "pytorch3d", "open3d", "PyOpenGL", "glfw", "kornia", "Pillow",
):
    try:
        version = metadata.version(name)
    except metadata.PackageNotFoundError:
        version = "NOT_INSTALLED"
        missing.append(name)
    print(f"[Demo2 preflight]   {name}: {version}")
if missing:
    raise SystemExit(f"required Boba distributions are missing: {', '.join(missing)}")
PY
then
  fail "could not inspect core Python package versions"
fi

if cuda_python_report="$(${PYTHON} <<'PY' 2>&1
import torch

if not torch.cuda.is_available():
    raise RuntimeError("torch.cuda.is_available() is false")
print(f"PyTorch {torch.__version__}; CUDA build {torch.version.cuda}")
for index in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(index)
    print(f"cuda:{index}: {props.name}; {props.total_memory / (1024 ** 3):.1f} GiB")
_ = torch.empty(1, device="cuda")
torch.cuda.synchronize()
PY
)"; then
  ok "PyTorch CUDA allocation succeeds"
  printf '%s\n' "${cuda_python_report}" | sed 's/^/[Demo2 preflight]   /'
else
  fail "PyTorch CUDA check failed: ${cuda_python_report}"
fi

if pycuda_report="$(${PYTHON} <<'PY' 2>&1
import pycuda
import pycuda.driver as driver
import pycuda.gl  # Verifies that PyCUDA was built with OpenGL interoperability.

driver.init()
if driver.Device.count() < 1:
    raise RuntimeError("PyCUDA sees no CUDA devices")
print(f"PyCUDA {getattr(pycuda, 'VERSION_TEXT', 'unknown')}; devices={driver.Device.count()}")
PY
)"; then
  ok "pycuda.gl imports and PyCUDA sees a device"
  printf '%s\n' "${pycuda_report}" | sed 's/^/[Demo2 preflight]   /'
else
  fail "PyCUDA/OpenGL interoperability check failed: ${pycuda_report}"
fi

gsplat_pythonpath="${REPO_ROOT}/gaussian_splatting/submodules/gsplat:${REPO_ROOT}"
if [[ -n "${PYTHONPATH:-}" ]]; then
  gsplat_pythonpath="${gsplat_pythonpath}:${PYTHONPATH}"
fi
if gsplat_report="$(PYTHONPATH="${gsplat_pythonpath}" "${PYTHON}" <<'PY' 2>&1
from gaussian_splatting._gsplat_vendor import gsplat, rasterization_shared_template

if not callable(rasterization_shared_template):
    raise RuntimeError("gsplat.rasterization_shared_template is not callable")
print(f"gsplat source: {gsplat.__file__}")
print(f"gsplat version: {gsplat.__version__}")
print("gsplat.rasterization_shared_template is available")
PY
)"; then
  ok "vendored gsplat shared-template renderer is available"
  printf '%s\n' "${gsplat_report}" | sed 's/^/[Demo2 preflight]   /'
else
  fail "gsplat shared-template check failed: ${gsplat_report}"
fi

if [[ -z "${DISPLAY:-}" ]]; then
  fail "DISPLAY is unset; Demo 2 needs an X11/OpenGL desktop session"
else
  ok "DISPLAY is ${DISPLAY}"
  if command -v glxinfo >/dev/null 2>&1; then
    if glx_report="$(glxinfo -B 2>&1)"; then
      ok "OpenGL context query succeeds"
      printf '%s\n' "${glx_report}" \
        | sed -n '/OpenGL vendor string:/p;/OpenGL renderer string:/p;/OpenGL core profile version string:/p' \
        | sed 's/^/[Demo2 preflight]   /'
    else
      fail "glxinfo could not open an OpenGL context: ${glx_report}"
    fi
  else
    warn "glxinfo is unavailable; using a hidden GLFW context for the OpenGL check"
    if glfw_report="$(${PYTHON} <<'PY' 2>&1
import glfw
from OpenGL import GL

if not glfw.init():
    raise RuntimeError("glfw.init() failed")
window = None
try:
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    window = glfw.create_window(64, 64, "Demo2 preflight", None, None)
    if window is None:
        raise RuntimeError("glfw.create_window() failed")
    glfw.make_context_current(window)
    renderer = GL.glGetString(GL.GL_RENDERER)
    version = GL.glGetString(GL.GL_VERSION)
    print(f"renderer={renderer.decode() if renderer else 'unknown'}")
    print(f"version={version.decode() if version else 'unknown'}")
finally:
    if window is not None:
        glfw.destroy_window(window)
    glfw.terminate()
PY
)"; then
      ok "hidden GLFW/OpenGL context creation succeeds"
      printf '%s\n' "${glfw_report}" | sed 's/^/[Demo2 preflight]   /'
    else
      fail "hidden GLFW/OpenGL context check failed: ${glfw_report}"
    fi
  fi
fi

if extras_report="$(${PYTHON} <<'PY' 2>&1
import flask
import flask_sock
import qrcode
import simple_websocket
from PIL import Image
import shutil
import sys
from importlib import metadata
from pathlib import Path

ninja = shutil.which("ninja")
if ninja is None:
    raise RuntimeError("ninja is not on PATH")
try:
    Path(ninja).resolve().relative_to(Path(sys.prefix).resolve())
except ValueError as exc:
    raise RuntimeError(f"ninja resolves outside phystwin: {ninja}") from exc
print(f"Flask={metadata.version('Flask')}")
print(
    f"Flask-Sock={metadata.version('Flask-Sock')}; "
    f"simple-websocket={metadata.version('simple-websocket')}"
)
print(f"qrcode={metadata.version('qrcode')}; Pillow={Image.__version__}")
print(f"ninja={metadata.version('ninja')} ({ninja})")
PY
)"; then
  ok "Demo 2 web/QR additions are installed"
  printf '%s\n' "${extras_report}" | sed 's/^/[Demo2 preflight]   /'
else
  fail "Demo 2 additions are incomplete; run env_install/install_demo2_extras.sh: ${extras_report}"
fi

asset_validator="${REPO_ROOT}/tools/validate_demo2_assets.py"
if [[ -f "${asset_validator}" ]]; then
  for packaged_case in single_push_rope_4 double_stretch_sloth; do
    if asset_report="$(cd "${REPO_ROOT}" && "${PYTHON}" "${asset_validator}" --case "${packaged_case}" 2>&1)"; then
      ok "packaged ${packaged_case} assets validate"
      printf '%s\n' "${asset_report}" | sed 's/^/[Demo2 preflight]   /'
    else
      fail "packaged ${packaged_case} asset validation failed: ${asset_report}"
    fi
  done
else
  warn "asset validator is unavailable: ${asset_validator}"
  for packaged_case in single_push_rope_4 double_stretch_sloth; do
    if [[ -f "${REPO_ROOT}/assets/${packaged_case}/manifest.json" ]]; then
      ok "packaged ${packaged_case} manifest exists (full validation skipped)"
    else
      fail "packaged ${packaged_case} manifest is missing"
    fi
  done
fi

if (( failures > 0 )); then
  printf '[Demo2 preflight] %d check(s) failed. Fix them before starting Demo 2.\n' "${failures}" >&2
  exit 1
fi

printf '[Demo2 preflight] All checks passed. This verifies compatibility, not batch-100 GPU capacity.\n'
