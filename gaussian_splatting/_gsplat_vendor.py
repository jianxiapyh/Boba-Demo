from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path


EXPECTED_CONDA_ENV = "phystwin"
VENDORED_GSPLAT_ROOT = Path(__file__).resolve().parent / "submodules" / "gsplat"
VENDORED_GSPLAT_PACKAGE = VENDORED_GSPLAT_ROOT / "gsplat"


def _install_hint() -> str:
    return (
        "From the Boba_OpenSource repository root, run:\n"
        "  conda run -n phystwin env PYTHONNOUSERSITE=1 BUILD_NO_CUDA=1 "
        "python -m pip install -e ./gaussian_splatting/submodules/gsplat"
    )


def _active_env_name() -> str:
    env_name = os.environ.get("CONDA_DEFAULT_ENV")
    if env_name:
        return env_name
    return Path(sys.prefix).resolve().name


def validate_gsplat_runtime(gsplat_module) -> None:
    active_env = _active_env_name()
    if active_env != EXPECTED_CONDA_ENV:
        raise RuntimeError(
            "Boba rendering only supports the vendored gsplat installed inside the "
            f"{EXPECTED_CONDA_ENV!r} conda environment. Active env: {active_env!r}, "
            f"sys.prefix: {Path(sys.prefix).resolve()}.\n{_install_hint()}"
        )

    module_file = getattr(gsplat_module, "__file__", None)
    if not module_file:
        raise RuntimeError(
            "Boba imported gsplat, but the package does not expose __file__. "
            f"Unable to verify provenance.\n{_install_hint()}"
        )

    module_path = Path(module_file).resolve()
    if VENDORED_GSPLAT_PACKAGE not in module_path.parents:
        raise RuntimeError(
            "Boba resolved gsplat from an unexpected location.\n"
            f"Resolved path: {module_path}\n"
            f"Expected under: {VENDORED_GSPLAT_PACKAGE}\n"
            f"{_install_hint()}"
        )


def import_gsplat():
    active_env = _active_env_name()
    if active_env != EXPECTED_CONDA_ENV:
        raise RuntimeError(
            "Boba rendering only supports the vendored gsplat installed inside the "
            f"{EXPECTED_CONDA_ENV!r} conda environment. Active env: {active_env!r}, "
            f"sys.prefix: {Path(sys.prefix).resolve()}.\n{_install_hint()}"
        )

    try:
        gsplat_module = importlib.import_module("gsplat")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Boba could not import gsplat from the active phystwin environment.\n"
            f"{_install_hint()}"
        ) from exc

    validate_gsplat_runtime(gsplat_module)
    return gsplat_module


gsplat = import_gsplat()
rasterization = gsplat.rasterization
rasterization_shared_template = gsplat.rasterization_shared_template


__all__ = [
    "EXPECTED_CONDA_ENV",
    "VENDORED_GSPLAT_PACKAGE",
    "VENDORED_GSPLAT_ROOT",
    "gsplat",
    "import_gsplat",
    "rasterization",
    "rasterization_shared_template",
    "validate_gsplat_runtime",
]
