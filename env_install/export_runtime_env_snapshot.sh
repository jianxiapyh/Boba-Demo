#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
DEFAULT_OUTPUT_REL="env_install/runtime_env_snapshot.md"
RUNTIME_ENV_NAME="${BOBA_SNAPSHOT_CONDA_ENV:-boba}"
BASELINE_SCRIPT="${BOBA_SNAPSHOT_BASELINE_SCRIPT:-${SCRIPT_DIR}/RTX6000_env_install.sh}"

usage() {
  cat <<'EOF'
Usage: export_runtime_env_snapshot.sh [output_path]

Generate a Markdown snapshot of the current Boba-Demo-upload runtime environment.
If output_path is relative, it is resolved relative to the Boba-Demo-upload repo root.
EOF
}

resolve_output_path() {
  local requested_path="$1"
  if [[ "${requested_path}" = /* ]]; then
    printf '%s\n' "${requested_path}"
    return
  fi
  printf '%s/%s\n' "${REPO_ROOT}" "${requested_path#./}"
}

capture_shell() {
  local name="$1"
  local command_text="$2"
  local output_file="${TMP_DIR}/${name}.txt"
  local status_file="${TMP_DIR}/${name}.status"

  if bash -lc "${command_text}" >"${output_file}" 2>&1; then
    printf 'ok\n' >"${status_file}"
  else
    printf 'fail\n' >"${status_file}"
  fi
}

capture_stdout_only() {
  local name="$1"
  local command_text="$2"
  local output_file="${TMP_DIR}/${name}.txt"
  local stderr_file="${TMP_DIR}/${name}.stderr.txt"
  local status_file="${TMP_DIR}/${name}.status"

  if bash -lc "${command_text}" >"${output_file}" 2>"${stderr_file}"; then
    printf 'ok\n' >"${status_file}"
  else
    printf 'fail\n' >"${status_file}"
  fi
}

if [[ $# -gt 1 ]]; then
  usage >&2
  exit 2
fi

OUTPUT_PATH="$(resolve_output_path "${1:-${DEFAULT_OUTPUT_REL}}")"
mkdir -p "$(dirname "${OUTPUT_PATH}")"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found in PATH; cannot capture the ${RUNTIME_ENV_NAME} runtime." >&2
  exit 1
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

export BOBA_SNAPSHOT_REPO_ROOT="${REPO_ROOT}"
export BOBA_SNAPSHOT_WORKSPACE_ROOT="${WORKSPACE_ROOT}"
export BOBA_SNAPSHOT_BASELINE_SCRIPT="${BASELINE_SCRIPT}"
export BOBA_SNAPSHOT_OUTPUT_PATH="${OUTPUT_PATH}"
export BOBA_SNAPSHOT_RUNTIME_ENV="${RUNTIME_ENV_NAME}"
export BOBA_SNAPSHOT_CAPTURED_AT="$(date --iso-8601=seconds)"

cat >"${TMP_DIR}/runtime_probe.py" <<'PY'
from __future__ import annotations

import contextlib
import importlib
import io
import json
import os
import platform
import sys
import traceback
import warnings
from pathlib import Path

repo_root = Path(os.environ["BOBA_SNAPSHOT_REPO_ROOT"]).resolve()
warnings.filterwarnings("ignore")


def package_entry(name: str, module_name: str, module, *, version: str | None = None) -> dict[str, str]:
    module_file = getattr(module, "__file__", None)
    if version is None:
        for attr_name in ("__version__", "VERSION_TEXT", "version"):
            attr_value = getattr(module, attr_name, None)
            if attr_value is None:
                continue
            version = str(attr_value() if callable(attr_value) else attr_value)
            break
    return {
        "status": "ok",
        "name": name,
        "module_name": module_name,
        "version": version or "unknown",
        "file": str(module_file) if module_file else "namespace/no __file__",
    }


def import_entry(name: str, module_name: str, *, version_override: str | None = None) -> dict[str, str]:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - diagnostic path
        return {
            "status": "error",
            "name": name,
            "module_name": module_name,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    return package_entry(name, module_name, module, version=version_override)


def reload_prefix(module_prefix: str) -> None:
    for module_name in list(sys.modules):
        if module_name == module_prefix or module_name.startswith(module_prefix + "."):
            del sys.modules[module_name]


result: dict[str, object] = {
    "env": {
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "sys_prefix": sys.prefix,
        "platform": platform.platform(),
    },
    "packages": {},
    "launch_imports": {},
    "cuda_probe": {},
    "notes": {},
}

packages: dict[str, dict[str, str]] = {}
packages["torch"] = import_entry("torch", "torch")
packages["torchvision"] = import_entry("torchvision", "torchvision")
packages["torchaudio"] = import_entry("torchaudio", "torchaudio")
packages["warp"] = import_entry("warp", "warp")
packages["pycuda"] = import_entry("pycuda", "pycuda")
packages["open3d"] = import_entry("open3d", "open3d")
packages["pyglet"] = import_entry("pyglet", "pyglet")
packages["opencv"] = import_entry("opencv", "cv2")
packages["numpy"] = import_entry("numpy", "numpy")
packages["pytorch3d"] = import_entry("pytorch3d", "pytorch3d")
packages["trimesh"] = import_entry("trimesh", "trimesh")
packages["pyrender"] = import_entry("pyrender", "pyrender")
packages["diff_gaussian_rasterization_default"] = import_entry(
    "diff_gaussian_rasterization_default",
    "diff_gaussian_rasterization",
)
packages["simple_knn_extension_default"] = import_entry(
    "simple_knn_extension_default",
    "simple_knn._C",
)
result["packages"] = packages

torch_module = None
if packages["torch"]["status"] == "ok":
    torch_module = importlib.import_module("torch")
    cuda_probe = {
        "torch_cuda_version": str(getattr(torch_module.version, "cuda", None)),
    }
    try:
        cuda_available = bool(torch_module.cuda.is_available())
        cuda_probe["cuda_available"] = cuda_available
        cuda_probe["device_count"] = int(torch_module.cuda.device_count())
        cuda_probe["device_0_name"] = (
            str(torch_module.cuda.get_device_name(0)) if cuda_available else "none"
        )
    except Exception as exc:  # pragma: no cover - diagnostic path
        cuda_probe["status"] = "error"
        cuda_probe["error"] = f"{type(exc).__name__}: {exc}"
        cuda_probe["traceback"] = traceback.format_exc()
    result["cuda_probe"] = cuda_probe
else:
    result["cuda_probe"] = {
        "status": "error",
        "error": "torch import failed; CUDA probe unavailable",
    }

sys.path.insert(0, str(repo_root))
import boba_quest_immersive as app  # noqa: E402

path_setup_stdout = io.StringIO()
with contextlib.redirect_stdout(path_setup_stdout):
    app.prioritize_conda_bin()
    app.prefer_system_ninja_binary()
    app.configure_local_python_paths()
result["notes"]["launcher_path_setup_stdout"] = path_setup_stdout.getvalue().strip()

from gaussian_splatting._gsplat_vendor import GSPLAT_SOURCE_ENV_VAR, import_gsplat  # noqa: E402

gsplat_module = import_gsplat()
result["launch_imports"]["gsplat"] = package_entry("gsplat", "gsplat", gsplat_module)
result["launch_imports"]["gsplat_source_env_var"] = GSPLAT_SOURCE_ENV_VAR
result["launch_imports"]["gsplat_source_root"] = os.environ.get(
    GSPLAT_SOURCE_ENV_VAR,
    str(
        Path(repo_root).resolve().parents[0]
        / "Boba"
        / "gaussian_splatting"
        / "submodules"
        / "gsplat"
    ),
)

reload_prefix("diff_gaussian_rasterization")
result["launch_imports"]["diff_gaussian_rasterization"] = import_entry(
    "diff_gaussian_rasterization_launch",
    "diff_gaussian_rasterization",
)

reload_prefix("simple_knn")
result["launch_imports"]["simple_knn_extension"] = import_entry(
    "simple_knn_extension_launch",
    "simple_knn._C",
)

print(json.dumps(result, indent=2, sort_keys=True))
PY

cat >"${TMP_DIR}/render_snapshot.py" <<'PY'
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

output_path = Path(os.environ["BOBA_SNAPSHOT_OUTPUT_PATH"]).resolve()
repo_root = Path(os.environ["BOBA_SNAPSHOT_REPO_ROOT"]).resolve()
workspace_root = Path(os.environ["BOBA_SNAPSHOT_WORKSPACE_ROOT"]).resolve()
baseline_script = Path(os.environ["BOBA_SNAPSHOT_BASELINE_SCRIPT"]).resolve()
runtime_env_name = os.environ["BOBA_SNAPSHOT_RUNTIME_ENV"]
captured_at = os.environ["BOBA_SNAPSHOT_CAPTURED_AT"]
tmp_dir = Path(sys.argv[1]).resolve()
runtime_probe = json.loads((tmp_dir / "runtime_probe.json").read_text(encoding="utf-8"))


def read_text(name: str) -> str:
    path = tmp_dir / f"{name}.txt"
    if not path.exists():
        return "unverified\n(no capture output was written)"
    text = path.read_text(encoding="utf-8", errors="replace").rstrip()
    return text or "(no output)"


def read_status(name: str) -> str:
    path = tmp_dir / f"{name}.status"
    if not path.exists():
        return "fail"
    return path.read_text(encoding="utf-8").strip() or "fail"


def git_info(path: Path) -> dict[str, str]:
    result = {"path": str(path), "head": "unverified", "dirty": "unverified"}
    if not path.exists():
        result["dirty"] = "missing"
        return result
    try:
        head = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
        result["head"] = head
    except subprocess.CalledProcessError as exc:
        result["head"] = f"unverified ({exc.output.strip() or exc})"
        return result

    try:
        status = subprocess.check_output(
            ["git", "-C", str(path), "status", "--short"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
        result["dirty"] = "dirty" if status else "clean"
    except subprocess.CalledProcessError as exc:
        result["dirty"] = f"unverified ({exc.output.strip() or exc})"
    return result


def code_block(text: str) -> str:
    return f"```text\n{text.rstrip()}\n```"


def md_escape(text: str) -> str:
    return text.replace("|", "\\|")


def package_version(key: str) -> str:
    entry = runtime_probe["packages"].get(key) or runtime_probe["launch_imports"].get(key)
    if not entry:
        return "unverified"
    if entry.get("status") != "ok":
        return f"unverified ({entry.get('error', 'import failed')})"
    return str(entry.get("version", "unknown"))


def package_file(key: str, *, launch: bool = False) -> str:
    section = runtime_probe["launch_imports"] if launch else runtime_probe["packages"]
    entry = section.get(key)
    if not entry:
        return "unverified"
    if entry.get("status") != "ok":
        return f"unverified ({entry.get('error', 'import failed')})"
    return str(entry.get("file", "unknown"))


def package_entry(section: str, key: str) -> dict[str, str]:
    return runtime_probe[section].get(key, {})


def parse_baseline_specs(script_path: Path) -> list[str]:
    specs: list[str] = []
    skip_next = {"--index-url", "-i", "--extra-index-url", "--find-links"}

    for raw_line in script_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        line = re.sub(r"\s+#.*$", "", line)
        if " install " not in f" {line} ":
            continue
        try:
            tokens = shlex.split(line)
        except ValueError:
            continue
        if "install" not in tokens:
            continue
        idx = tokens.index("install") + 1
        skip_value = False
        for token in tokens[idx:]:
            if skip_value:
                skip_value = False
                continue
            if token in skip_next:
                skip_value = True
                continue
            if token in {"||", "true"}:
                break
            if token.startswith("-"):
                continue
            if token == ".":
                continue
            specs.append(token)
    deduped: list[str] = []
    seen: set[str] = set()
    for spec in specs:
        if spec in seen:
            continue
        seen.add(spec)
        deduped.append(spec)
    return deduped


def extract_editable_lines(pip_freeze_text: str) -> dict[str, str]:
    editables: dict[str, str] = {}
    for line in pip_freeze_text.splitlines():
        if not line.startswith("-e "):
            continue
        match = re.search(r"#egg=([^&\s]+)", line)
        if match:
            editables[match.group(1)] = line
    return editables


def parse_pip_show_blocks(text: str) -> dict[str, dict[str, str]]:
    blocks: dict[str, dict[str, str]] = {}
    current: dict[str, str] = {}
    current_name: str | None = None
    for line in text.splitlines():
        if not line.strip():
            if current_name is not None:
                blocks[current_name] = current
            current_name = None
            current = {}
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key == "Name":
            if current_name is not None:
                blocks[current_name] = current
            current_name = value
            current = {"Name": value}
        elif current_name is not None:
            current[key] = value
    if current_name is not None:
        blocks[current_name] = current
    return blocks


repo_infos = {
    "Boba-Demo-upload": git_info(repo_root),
    "Boba": git_info(workspace_root / "Boba"),
}

pip_freeze_text = read_text("pip_freeze")
conda_list_text = read_text("conda_list")
pip_show_text = read_text("pip_show_selected")
editable_lines = extract_editable_lines(pip_freeze_text)
pip_show_blocks = parse_pip_show_blocks(pip_show_text)
baseline_specs = parse_baseline_specs(baseline_script)

launch_notes = runtime_probe.get("notes", {}).get("launcher_path_setup_stdout", "")
cuda_probe = runtime_probe.get("cuda_probe", {})
steamvr_root = Path.home() / ".local" / "share" / "Steam" / "steamapps" / "common" / "SteamVR"
steamvr_runtime_json = steamvr_root / "steamxr_linux64.json"
steamvr_lib_dir = steamvr_root / "bin" / "linux64"

summary_lines = [
    f"- Capture time: `{captured_at}`",
    f"- Conda env: `{runtime_probe['env'].get('conda_default_env') or runtime_env_name}`",
    f"- Python: `{runtime_probe['env'].get('python_version', 'unverified')}` at `{runtime_probe['env'].get('python_executable', 'unverified')}`",
    (
        "- Key repos: "
        f"`Boba-Demo-upload@{repo_infos['Boba-Demo-upload']['head']}` ({repo_infos['Boba-Demo-upload']['dirty']}), "
        f"`Boba@{repo_infos['Boba']['head']}` ({repo_infos['Boba']['dirty']})"
    ),
    (
        "- Key runtime packages: "
        f"`torch=={package_version('torch')}`, "
        f"`torchvision=={package_version('torchvision')}`, "
        f"`torchaudio=={package_version('torchaudio')}`, "
        f"`warp-lang=={package_version('warp')}`, "
        f"`pycuda=={package_version('pycuda')}`, "
        f"`open3d=={package_version('open3d')}`, "
        f"`pyglet=={package_version('pyglet')}`, "
        f"`numpy=={package_version('numpy')}`, "
        f"`opencv=={package_version('opencv')}`"
    ),
    (
        "- Native prereqs observed here: "
        f"`openxr={read_text('pkgconfig_openxr').splitlines()[0] if read_status('pkgconfig_openxr') == 'ok' else 'unverified'}`, "
        f"`jsoncpp={read_text('pkgconfig_jsoncpp').splitlines()[0] if read_status('pkgconfig_jsoncpp') == 'ok' else 'unverified'}`, "
        f"`glfw3={read_text('pkgconfig_glfw3').splitlines()[0] if read_status('pkgconfig_glfw3') == 'ok' else 'unverified'}`"
    ),
    (
        "- GPU probe: "
        f"`torch.cuda.is_available()={cuda_probe.get('cuda_available', 'unverified')}`; "
        f"`torch.version.cuda={cuda_probe.get('torch_cuda_version', 'unverified')}`; "
        f"`device_count={cuda_probe.get('device_count', 'unverified')}`; "
        f"`nvidia-smi={('ok' if read_status('nvidia_smi') == 'ok' else 'unverified')}`"
    ),
    (
        "- Caveat: `gsplat` resolves from "
        f"`{package_file('gsplat', launch=True)}` via the vendored sibling Boba path."
    ),
    (
        "- Caveat: `pytorch3d` currently resolves from "
        f"`{package_file('pytorch3d')}`, not from a Boba-Demo-upload-local clone."
    ),
    (
        "- Caveat: default env import for `diff_gaussian_rasterization` resolves to "
        f"`{package_file('diff_gaussian_rasterization_default')}`, while launch-path import switches to "
        f"`{package_file('diff_gaussian_rasterization', launch=True)}`."
    ),
    (
        "- Caveat: default env import for `simple_knn._C` is "
        f"`{package_entry('packages', 'simple_knn_extension_default').get('status', 'unverified')}`; "
        f"launch-path import resolves to `{package_file('simple_knn_extension', launch=True)}`."
    ),
]
if launch_notes:
    summary_lines.append(f"- Launcher path setup note: `{launch_notes}`")

paste_block = "\n".join(
    [
        "I am trying to match the Boba-Demo-upload runtime from another machine.",
        "Please compare this machine against the known-good snapshot below and tell me what is missing or mismatched.",
        "",
        f"Repo SHAs: Boba-Demo-upload@{repo_infos['Boba-Demo-upload']['head']}, "
        f"Boba@{repo_infos['Boba']['head']}",
        f"Expected conda env: {runtime_probe['env'].get('conda_default_env') or runtime_env_name}",
        (
            "Expected launch command: "
            f"conda run -n {runtime_env_name} env PYTHONNOUSERSITE=1 python boba_quest_immersive.py "
            "--case_name sloth --n_dup 0 --interactive_window_mode hidden"
        ),
        "",
        "Key Python packages:",
        f"- torch=={package_version('torch')}",
        f"- torchvision=={package_version('torchvision')}",
        f"- torchaudio=={package_version('torchaudio')}",
        f"- warp-lang=={package_version('warp')}",
        f"- pycuda=={package_version('pycuda')}",
        f"- open3d=={package_version('open3d')}",
        f"- pyglet=={package_version('pyglet')}",
        f"- numpy=={package_version('numpy')}",
        f"- opencv=={package_version('opencv')}",
        f"- pytorch3d=={package_version('pytorch3d')}",
        "",
        "Editable/native runtime paths:",
        f"- gsplat -> {package_file('gsplat', launch=True)}",
        f"- pytorch3d -> {package_file('pytorch3d')}",
        f"- diff_gaussian_rasterization default -> {package_file('diff_gaussian_rasterization_default')}",
        f"- diff_gaussian_rasterization launch -> {package_file('diff_gaussian_rasterization', launch=True)}",
        f"- simple_knn launch -> {package_file('simple_knn_extension', launch=True)}",
        "",
        "Native prerequisites:",
        f"- pkg-config openxr={read_text('pkgconfig_openxr').splitlines()[0] if read_status('pkgconfig_openxr') == 'ok' else 'unverified'}",
        f"- pkg-config jsoncpp={read_text('pkgconfig_jsoncpp').splitlines()[0] if read_status('pkgconfig_jsoncpp') == 'ok' else 'unverified'}",
        f"- pkg-config glfw3={read_text('pkgconfig_glfw3').splitlines()[0] if read_status('pkgconfig_glfw3') == 'ok' else 'unverified'}",
        f"- nvcc={read_text('nvcc').splitlines()[-1] if read_status('nvcc') == 'ok' else 'unverified'}",
        f"- g++={read_text('gpp').splitlines()[0] if read_status('gpp') == 'ok' else 'unverified'}",
        (
            f"- torch.cuda.is_available()={cuda_probe.get('cuda_available', 'unverified')}, "
            f"torch.version.cuda={cuda_probe.get('torch_cuda_version', 'unverified')}, "
            f"device_count={cuda_probe.get('device_count', 'unverified')}"
        ),
        f"- nvidia-smi={'available' if read_status('nvidia_smi') == 'ok' else 'unverified on source machine'}",
        "",
        "Please compare installed package versions, import paths, native libraries, and missing build artifacts.",
    ]
)

baseline_section_lines = [
    f"### Parsed package specs from `{baseline_script}`",
    code_block("\n".join(baseline_specs)),
    "### Important baseline-to-runtime notes",
    f"- Baseline pins `numpy==1.26.4`; import-time runtime observed `numpy=={package_version('numpy')}`.",
    f"- Baseline constrains `pyglet<2`; import-time runtime observed `pyglet=={package_version('pyglet')}`.",
    (
        "- Baseline installs Torch from the CUDA 12.8 wheel index; import-time runtime observed "
        f"`torch=={package_version('torch')}`, `torchvision=={package_version('torchvision')}`, "
        f"`torchaudio=={package_version('torchaudio')}`."
    ),
    (
        "- Baseline installs editable `gsplat` from `${BOBA_GSPLAT_SOURCE_ROOT}`; runtime resolves "
        f"`gsplat` from `{package_file('gsplat', launch=True)}`."
    ),
    (
        "- Baseline clones/builds `pytorch3d` inside the demo tree, but current runtime resolves "
        f"`pytorch3d` from `{package_file('pytorch3d')}`."
    ),
    (
        "- Baseline builds `diff_gaussian_rasterization` and `simple_knn` under `Boba-Demo-upload`; "
        "current package metadata still points at sibling repos, while launch-path imports can switch "
        "to local Boba-Demo-upload builds."
    ),
]

editable_rows = [
    (
        "gsplat",
        editable_lines.get("gsplat", "unverified"),
        pip_show_blocks.get("gsplat", {}).get("Editable project location", "unverified"),
        package_file("gsplat", launch=True),
        "Forced through `gaussian_splatting/_gsplat_vendor.py`.",
    ),
    (
        "pytorch3d",
        editable_lines.get("pytorch3d", "unverified"),
        pip_show_blocks.get("pytorch3d", {}).get("Editable project location", "unverified"),
        package_file("pytorch3d"),
        "Current runtime resolves from sibling `Boba/pytorch3d`.",
    ),
    (
        "diff_gaussian_rasterization",
        editable_lines.get("diff_gaussian_rasterization", "unverified"),
        pip_show_blocks.get("diff_gaussian_rasterization", {}).get("Editable project location", "unverified"),
        (
            f"default: {package_file('diff_gaussian_rasterization_default')}<br>"
            f"launch: {package_file('diff_gaussian_rasterization', launch=True)}"
        ),
        "Default env import hits sibling `Boba`; launch path can use local demo build.",
    ),
    (
        "simple_knn",
        editable_lines.get("simple_knn", "unverified"),
        pip_show_blocks.get("simple_knn", {}).get("Editable project location", "unverified"),
        (
            f"default: {package_entry('packages', 'simple_knn_extension_default').get('status', 'unverified')}<br>"
            f"launch: {package_file('simple_knn_extension', launch=True)}"
        ),
        "Local extension loads after `configure_local_python_paths()` and importing `torch` first.",
    ),
]

package_rows = [
    ("torch", package_version("torch"), package_file("torch")),
    ("torchvision", package_version("torchvision"), package_file("torchvision")),
    ("torchaudio", package_version("torchaudio"), package_file("torchaudio")),
    ("warp", package_version("warp"), package_file("warp")),
    ("pycuda", package_version("pycuda"), package_file("pycuda")),
    ("open3d", package_version("open3d"), package_file("open3d")),
    ("pyglet", package_version("pyglet"), package_file("pyglet")),
    ("opencv", package_version("opencv"), package_file("opencv")),
    ("numpy", package_version("numpy"), package_file("numpy")),
    ("pytorch3d", package_version("pytorch3d"), package_file("pytorch3d")),
    ("trimesh", package_version("trimesh"), package_file("trimesh")),
    ("pyrender", package_version("pyrender"), package_file("pyrender")),
    (
        "diff_gaussian_rasterization (default env import)",
        package_entry("packages", "diff_gaussian_rasterization_default").get("version", "unknown"),
        package_file("diff_gaussian_rasterization_default"),
    ),
    (
        "diff_gaussian_rasterization (launch-path import)",
        package_entry("launch_imports", "diff_gaussian_rasterization").get("version", "unknown"),
        package_file("diff_gaussian_rasterization", launch=True),
    ),
    (
        "gsplat (launch-path import)",
        package_entry("launch_imports", "gsplat").get("version", "unknown"),
        package_file("gsplat", launch=True),
    ),
]

repo_table_lines = [
    "| Repo | Path | HEAD | Status |",
    "| --- | --- | --- | --- |",
]
for repo_name, info in repo_infos.items():
    repo_table_lines.append(
        "| "
        + " | ".join(
            [
                md_escape(repo_name),
                md_escape(info["path"]),
                md_escape(info["head"]),
                md_escape(info["dirty"]),
            ]
        )
        + " |"
    )

package_table_lines = [
    "| Package | Version | Import path |",
    "| --- | --- | --- |",
]
for name, version, import_path in package_rows:
    package_table_lines.append(
        "| "
        + " | ".join([md_escape(name), md_escape(str(version)), md_escape(str(import_path))])
        + " |"
    )

editable_table_lines = [
    "| Package | `pip freeze` / metadata | Editable project location | Runtime resolution | Notes |",
    "| --- | --- | --- | --- | --- |",
]
for row in editable_rows:
    editable_table_lines.append("| " + " | ".join(md_escape(str(value)) for value in row) + " |")

native_section_lines = [
    f"- Expected launcher runtime JSON path: `{steamvr_runtime_json}` ({'present' if steamvr_runtime_json.exists() else 'missing'})",
    f"- Expected launcher SteamVR lib dir: `{steamvr_lib_dir}` ({'present' if steamvr_lib_dir.exists() else 'missing'})",
    f"- `pkg-config openxr`: `{read_text('pkgconfig_openxr').splitlines()[0] if read_status('pkgconfig_openxr') == 'ok' else 'unverified'}`",
    f"- `pkg-config jsoncpp`: `{read_text('pkgconfig_jsoncpp').splitlines()[0] if read_status('pkgconfig_jsoncpp') == 'ok' else 'unverified'}`",
    f"- `pkg-config glfw3`: `{read_text('pkgconfig_glfw3').splitlines()[0] if read_status('pkgconfig_glfw3') == 'ok' else 'unverified'}`",
    f"- `g++`: `{read_text('gpp').splitlines()[0] if read_status('gpp') == 'ok' else 'unverified'}`",
    f"- `nvcc`: `{read_text('nvcc').splitlines()[-1] if read_status('nvcc') == 'ok' else 'unverified'}`",
    (
        "- Torch CUDA probe: "
        f"`torch.cuda.is_available()={cuda_probe.get('cuda_available', 'unverified')}`, "
        f"`torch.version.cuda={cuda_probe.get('torch_cuda_version', 'unverified')}`, "
        f"`device_count={cuda_probe.get('device_count', 'unverified')}`, "
        f"`device_0_name={cuda_probe.get('device_0_name', 'unverified')}`"
    ),
    (
        "- `nvidia-smi`: "
        + ("captured successfully." if read_status("nvidia_smi") == "ok" else "unverified on this session; see raw output below.")
    ),
]

markdown_parts = [
    "# Boba-Demo-upload Runtime Environment Snapshot",
    "",
    f"Generated by `env_install/export_runtime_env_snapshot.sh` at `{captured_at}`.",
    "",
    "## Runtime Summary",
    "\n".join(summary_lines),
    "",
    "## Paste This To Codex On The New Machine",
    code_block(paste_block),
    "",
    "## Repo Context",
    "\n".join(repo_table_lines),
    "",
    "## Declared Install Baseline vs Observed Runtime",
    "\n".join(baseline_section_lines),
    "",
    "## Key Python Packages",
    "\n".join(package_table_lines),
    "",
    "## Editable and Local Native Builds",
    "\n".join(editable_table_lines),
    "",
    "## Native Linux/OpenXR/CUDA Prerequisites",
    "\n".join(native_section_lines),
    "",
    "### `nvidia-smi`",
    code_block(read_text("nvidia_smi")),
    "",
    "### `openxr_controller_stream` linkage",
    code_block(read_text("ldd_openxr_controller_stream")),
    "",
    "### `openxr_headset_pose_probe` linkage",
    code_block(read_text("ldd_openxr_headset_pose_probe")),
    "",
    "### `boba_immersive_demo` linkage",
    code_block(read_text("ldd_boba_immersive_demo")),
    "",
    "### Local `diff_gaussian_rasterization` linkage",
    code_block(read_text("ldd_diff_gaussian_local")),
    "",
    "### Local `simple_knn` linkage",
    code_block(read_text("ldd_simple_knn_local")),
    "",
    "## Raw pip show (selected packages)",
    code_block(pip_show_text),
    "",
    "## Raw conda list",
    code_block(conda_list_text),
    "",
    "## Raw pip freeze",
    code_block(pip_freeze_text),
]

output_path.write_text("\n".join(markdown_parts).rstrip() + "\n", encoding="utf-8")
PY

capture_stdout_only "runtime_probe" \
  "conda run -n ${RUNTIME_ENV_NAME} env PYTHONNOUSERSITE=1 python '${TMP_DIR}/runtime_probe.py'"
capture_shell "pip_freeze" \
  "conda run -n ${RUNTIME_ENV_NAME} env PYTHONNOUSERSITE=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1 python -m pip freeze"
capture_shell "conda_list" \
  "conda list -n ${RUNTIME_ENV_NAME}"
capture_shell "pip_show_selected" \
  "conda run -n ${RUNTIME_ENV_NAME} env PYTHONNOUSERSITE=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1 python -m pip show gsplat pytorch3d diff_gaussian_rasterization simple_knn warp-lang open3d torch torchvision torchaudio pycuda"
capture_shell "nvcc" "nvcc --version"
capture_shell "gpp" "g++ --version | head -n 1"
capture_shell "pkgconfig_openxr" "pkg-config --modversion openxr"
capture_shell "pkgconfig_jsoncpp" "pkg-config --modversion jsoncpp"
capture_shell "pkgconfig_glfw3" "pkg-config --modversion glfw3"
capture_shell "nvidia_smi" "nvidia-smi --query-gpu=name,driver_version,cuda_version --format=csv,noheader"
capture_shell "ldd_openxr_controller_stream" \
  "ldd '${REPO_ROOT}/linux_pose_probe/openxr_controller_stream'"
capture_shell "ldd_openxr_headset_pose_probe" \
  "ldd '${REPO_ROOT}/linux_pose_probe/openxr_headset_pose_probe'"
capture_shell "ldd_boba_immersive_demo" \
  "ldd '${REPO_ROOT}/linux_pose_probe/boba_immersive_demo'"
capture_shell "ldd_diff_gaussian_local" \
  "ldd '${REPO_ROOT}/gaussian_splatting/submodules/diff-gaussian-rasterization/diff_gaussian_rasterization/_C.cpython-310-x86_64-linux-gnu.so'"
capture_shell "ldd_simple_knn_local" \
  "ldd '${REPO_ROOT}/gaussian_splatting/submodules/simple-knn/simple_knn/_C.cpython-310-x86_64-linux-gnu.so'"

if [[ ! -s "${TMP_DIR}/runtime_probe.txt" ]] || [[ "$(cat "${TMP_DIR}/runtime_probe.status")" != "ok" ]]; then
  echo "Runtime probe failed; cannot render snapshot." >&2
  cat "${TMP_DIR}/runtime_probe.txt" >&2 || true
  cat "${TMP_DIR}/runtime_probe.stderr.txt" >&2 || true
  exit 1
fi

mv "${TMP_DIR}/runtime_probe.txt" "${TMP_DIR}/runtime_probe.json"

conda run -n "${RUNTIME_ENV_NAME}" env PYTHONNOUSERSITE=1 python "${TMP_DIR}/render_snapshot.py" "${TMP_DIR}"

echo "Wrote runtime snapshot to ${OUTPUT_PATH}"
