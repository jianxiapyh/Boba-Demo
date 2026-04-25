from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path


EXPECTED_CONDA_ENV = os.environ.get("BOBA_EXPECTED_CONDA_ENV", "boba")
GSPLAT_SOURCE_ENV_VAR = "BOBA_GSPLAT_SOURCE_ROOT"
_WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_GSPLAT_ROOT_CANDIDATES = (
    _WORKSPACE_ROOT / "Boba" / "gaussian_splatting" / "submodules" / "gsplat",
    _WORKSPACE_ROOT / "Boba_OpenSource" / "gaussian_splatting" / "submodules" / "gsplat",
)


def _default_gsplat_root() -> Path:
    for candidate in _DEFAULT_GSPLAT_ROOT_CANDIDATES:
        if candidate.is_dir():
            return candidate.resolve()
    return _DEFAULT_GSPLAT_ROOT_CANDIDATES[0].resolve()


DEFAULT_GSPLAT_ROOT = _default_gsplat_root()


def _active_env_name() -> str:
    env_name = os.environ.get("CONDA_DEFAULT_ENV")
    if env_name:
        return env_name
    return Path(sys.prefix).resolve().name


def _configured_gsplat_root() -> Path:
    configured_root = os.environ.get(GSPLAT_SOURCE_ENV_VAR)
    if configured_root:
        return Path(configured_root).expanduser().resolve()
    return DEFAULT_GSPLAT_ROOT.resolve()


def _expected_gsplat_package(root: Path) -> Path:
    return root / "gsplat"


def _install_hint(root: Path) -> str:
    return (
        "Install or expose the vendored gsplat from the sibling Boba checkout:\n"
        f"  conda run -n {EXPECTED_CONDA_ENV} env PYTHONNOUSERSITE=1 BUILD_NO_CUDA=1 "
        f"python -m pip install -e {root}\n"
        f"Or point {GSPLAT_SOURCE_ENV_VAR} at the gsplat source root."
    )


def _prioritize_conda_bin() -> None:
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return

    conda_bin = os.path.join(conda_prefix, "bin")
    if not os.path.isdir(conda_bin):
        return

    path_parts = [part for part in os.environ.get("PATH", "").split(os.pathsep) if part]
    path_parts = [part for part in path_parts if part != conda_bin]
    os.environ["PATH"] = os.pathsep.join([conda_bin] + path_parts)


def _bootstrap_gsplat_source(root: Path) -> None:
    if not root.is_dir():
        raise RuntimeError(
            "Boba Demo could not find the vendored gsplat source tree.\n"
            f"Expected path: {root}\n"
            f"{_install_hint(root)}"
        )

    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def validate_gsplat_runtime(gsplat_module) -> None:
    active_env = _active_env_name()
    if active_env != EXPECTED_CONDA_ENV:
        raise RuntimeError(
            "Boba Demo rendering only supports the vendored gsplat inside the "
            f"{EXPECTED_CONDA_ENV!r} conda environment. Active env: {active_env!r}, "
            f"sys.prefix: {Path(sys.prefix).resolve()}.\n"
            f"{_install_hint(_configured_gsplat_root())}"
        )

    module_file = getattr(gsplat_module, "__file__", None)
    if not module_file:
        raise RuntimeError(
            "Boba Demo imported gsplat, but the package does not expose __file__.\n"
            f"{_install_hint(_configured_gsplat_root())}"
        )

    module_path = Path(module_file).resolve()
    expected_package = _expected_gsplat_package(_configured_gsplat_root())
    if expected_package not in module_path.parents:
        raise RuntimeError(
            "Boba Demo resolved gsplat from an unexpected location.\n"
            f"Resolved path: {module_path}\n"
            f"Expected under: {expected_package}\n"
            f"{_install_hint(_configured_gsplat_root())}"
        )


def import_gsplat():
    _prioritize_conda_bin()
    gsplat_root = _configured_gsplat_root()
    _bootstrap_gsplat_source(gsplat_root)

    try:
        gsplat_module = importlib.import_module("gsplat")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Boba Demo could not import gsplat from the configured vendored path.\n"
            f"Expected source root: {gsplat_root}\n"
            f"{_install_hint(gsplat_root)}"
        ) from exc

    validate_gsplat_runtime(gsplat_module)
    return gsplat_module


gsplat = import_gsplat()
rasterization = gsplat.rasterization


__all__ = [
    "DEFAULT_GSPLAT_ROOT",
    "EXPECTED_CONDA_ENV",
    "GSPLAT_SOURCE_ENV_VAR",
    "gsplat",
    "import_gsplat",
    "rasterization",
    "validate_gsplat_runtime",
]
