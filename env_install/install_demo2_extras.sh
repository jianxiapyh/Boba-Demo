#!/usr/bin/env bash
set -euo pipefail

# Install only the small web/QR additions needed by Demo 2. The existing
# Boba-Batched phystwin environment remains the source of all core packages.

EXPECTED_ENV="phystwin"

die() {
  printf '[Demo2 extras] ERROR: %s\n' "$*" >&2
  exit 1
}

active_env_name="${CONDA_DEFAULT_ENV:-}"
active_env_name="${active_env_name##*/}"
if [[ "${active_env_name}" != "${EXPECTED_ENV}" ]]; then
  die "activate the ${EXPECTED_ENV} Conda environment before running this script (active: ${CONDA_DEFAULT_ENV:-none})."
fi

if [[ -z "${CONDA_PREFIX:-}" || ! -x "${CONDA_PREFIX}/bin/python" ]]; then
  die "CONDA_PREFIX does not identify an active Conda environment with Python."
fi

PYTHON="${CONDA_PREFIX}/bin/python"
export PYTHONNOUSERSITE=1

python_prefix_name="$("${PYTHON}" -c 'import pathlib, sys; print(pathlib.Path(sys.prefix).resolve().name)')"
if [[ "${python_prefix_name}" != "${EXPECTED_ENV}" ]]; then
  die "${PYTHON} belongs to ${python_prefix_name}, not ${EXPECTED_ENV}."
fi

tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/boba-demo2-extras.XXXXXX")"
trap 'rm -rf "${tmp_dir}"' EXIT
before_snapshot="${tmp_dir}/core-before.json"
after_snapshot="${tmp_dir}/core-after.json"
constraints_file="${tmp_dir}/installed-versions.txt"

snapshot_core_versions() {
  local destination="$1"
  "${PYTHON}" - "${destination}" <<'PY'
import json
import platform
import sys
from importlib import metadata
from pathlib import Path

distribution_names = (
    "torch",
    "torchvision",
    "torchaudio",
    "numpy",
    "scipy",
    "warp-lang",
    "pycuda",
    "gsplat",
    "pytorch3d",
    "open3d",
    "PyOpenGL",
    "glfw",
    "kornia",
    "Pillow",
)

snapshot = {
    "python": platform.python_version(),
    "python_executable": str(Path(sys.executable).resolve()),
}
for distribution_name in distribution_names:
    try:
        value = metadata.version(distribution_name)
    except metadata.PackageNotFoundError:
        value = "NOT_INSTALLED"
    snapshot[f"distribution:{distribution_name}"] = value

try:
    import torch

    snapshot["torch_cuda_build"] = torch.version.cuda or "NONE"
except Exception as exc:  # Keep the audit useful even for a broken environment.
    snapshot["torch_cuda_build"] = f"ERROR:{type(exc).__name__}:{exc}"

rendered = json.dumps(snapshot, indent=2, sort_keys=True)
Path(sys.argv[1]).write_text(rendered + "\n", encoding="utf-8")
print(rendered)
PY
}

write_installed_constraints() {
  "${PYTHON}" - "${constraints_file}" <<'PY'
import re
import sys
from importlib import metadata
from pathlib import Path

# Pin every distribution already present. pip may add missing web dependencies,
# but cannot upgrade, downgrade, or replace anything in the working Boba env.
installed = {}
for distribution in metadata.distributions():
    name = distribution.metadata.get("Name")
    version = distribution.version
    if not name or not version:
        continue
    normalized = re.sub(r"[-_.]+", "-", name).lower()
    installed[normalized] = (name, version)

lines = [
    f"{name}==={version}"
    for name, version in (installed[key] for key in sorted(installed))
]
Path(sys.argv[1]).write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
}

detect_missing_extras() {
  "${PYTHON}" <<'PY'
import importlib
import shutil
import sys
from importlib import metadata
from pathlib import Path

def imports_cleanly(module_name):
    try:
        importlib.import_module(module_name)
    except Exception:
        return False
    return True

def distribution_exists(distribution_name):
    try:
        metadata.version(distribution_name)
    except metadata.PackageNotFoundError:
        return False
    return True

if not (distribution_exists("Flask") and imports_cleanly("flask")):
    print("Flask")
if not (
    distribution_exists("qrcode")
    and distribution_exists("Pillow")
    and imports_cleanly("qrcode")
    and imports_cleanly("PIL")
):
    print("qrcode[pil]")
ninja = shutil.which("ninja")
ninja_in_env = False
if ninja is not None:
    try:
        Path(ninja).resolve().relative_to(Path(sys.prefix).resolve())
        ninja_in_env = True
    except ValueError:
        pass
if not (distribution_exists("ninja") and ninja_in_env):
    print("ninja")
PY
}

verify_extras() {
  "${PYTHON}" <<'PY'
import importlib
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path

for module_name in ("flask", "qrcode", "PIL"):
    importlib.import_module(module_name)

ninja = shutil.which("ninja")
if ninja is None:
    raise RuntimeError("ninja is not on PATH after installation")
try:
    Path(ninja).resolve().relative_to(Path(sys.prefix).resolve())
except ValueError as exc:
    raise RuntimeError(f"ninja resolves outside the active environment: {ninja}") from exc

print("[Demo2 extras] Installed web-demo versions:")
for distribution_name in ("Flask", "qrcode", "Pillow", "ninja"):
    try:
        version = metadata.version(distribution_name)
    except metadata.PackageNotFoundError:
        version = "provided outside pip"
    print(f"  {distribution_name}: {version}")
print(f"  ninja executable: {ninja} ({subprocess.check_output([ninja, '--version'], text=True).strip()})")
PY
}

printf '[Demo2 extras] Core package versions before installation:\n'
snapshot_core_versions "${before_snapshot}"
write_installed_constraints

mapfile -t missing_extras < <(detect_missing_extras)
if (( ${#missing_extras[@]} == 0 )); then
  printf '[Demo2 extras] Flask, qrcode[pil], and Ninja are already available; nothing to install.\n'
else
  printf '[Demo2 extras] Installing missing additions only:'
  printf ' %q' "${missing_extras[@]}"
  printf '\n'
  if ! "${PYTHON}" -m pip install \
    --disable-pip-version-check \
    --upgrade-strategy only-if-needed \
    --constraint "${constraints_file}" \
    "${missing_extras[@]}"; then
    printf '[Demo2 extras] pip could not satisfy the additions without changing the existing environment.\n' >&2
    printf '[Demo2 extras] Resolve the reported web-package conflict manually; core packages were not intentionally modified.\n' >&2
    snapshot_core_versions "${after_snapshot}" >/dev/null
    if ! cmp -s "${before_snapshot}" "${after_snapshot}"; then
      printf '[Demo2 extras] WARNING: core package audit changed during the failed install:\n' >&2
      diff -u "${before_snapshot}" "${after_snapshot}" >&2 || true
    fi
    exit 1
  fi
fi

verify_extras
printf '[Demo2 extras] Core package versions after installation:\n'
snapshot_core_versions "${after_snapshot}"

if ! cmp -s "${before_snapshot}" "${after_snapshot}"; then
  printf '[Demo2 extras] ERROR: a core package changed; review the audit below.\n' >&2
  diff -u "${before_snapshot}" "${after_snapshot}" >&2 || true
  exit 1
fi

printf '[Demo2 extras] Success: core Boba package versions are unchanged.\n'
