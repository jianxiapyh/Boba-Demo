from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path


EXPECTED_CONDA_ENV = "phystwin"
REPO_ROOT = Path(__file__).resolve().parents[1]
BOBA_BATCHED_ROOT_ENV_VAR = "BOBA_BATCHED_ROOT"
DEFAULT_BOBA_BATCHED_ROOT = Path("/home/yihan/Research/Boba_Latest")
BOBA_BATCHED_ROOT = Path(
    os.environ.get(BOBA_BATCHED_ROOT_ENV_VAR, DEFAULT_BOBA_BATCHED_ROOT)
).expanduser().resolve()
VENDORED_GSPLAT_ROOT = (
    BOBA_BATCHED_ROOT / "gaussian_splatting" / "submodules" / "gsplat"
)
VENDORED_GSPLAT_PACKAGE = VENDORED_GSPLAT_ROOT / "gsplat"


def _install_hint() -> str:
    return (
        "Prepare and validate Boba-Batched first, then point this demo at that checkout:\n"
        f"  export {BOBA_BATCHED_ROOT_ENV_VAR}={BOBA_BATCHED_ROOT}\n"
        "  conda activate phystwin\n"
        f"Expected gsplat source: {VENDORED_GSPLAT_ROOT}"
    )


def _active_env_name() -> str:
    env_name = os.environ.get("CONDA_DEFAULT_ENV")
    if env_name:
        return env_name
    return Path(sys.prefix).resolve().name


def _is_vendored_gsplat_module(gsplat_module) -> bool:
    module_file = getattr(gsplat_module, "__file__", None)
    if not module_file:
        return False

    module_path = Path(module_file).resolve()
    return VENDORED_GSPLAT_PACKAGE in module_path.parents


def _prepare_vendored_gsplat_import() -> None:
    if not VENDORED_GSPLAT_PACKAGE.is_dir():
        raise RuntimeError(
            "Boba Phone Demo could not find the Boba-Batched gsplat source tree.\n"
            f"{_install_hint()}"
        )
    vendored_root = str(VENDORED_GSPLAT_ROOT)
    if vendored_root not in sys.path:
        sys.path.insert(0, vendored_root)

    loaded_gsplat = sys.modules.get("gsplat")
    if loaded_gsplat is None or _is_vendored_gsplat_module(loaded_gsplat):
        return

    for module_name in list(sys.modules):
        if module_name == "gsplat" or module_name.startswith("gsplat."):
            del sys.modules[module_name]


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
    if not _is_vendored_gsplat_module(gsplat_module):
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

    _prepare_vendored_gsplat_import()

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
    "BOBA_BATCHED_ROOT",
    "BOBA_BATCHED_ROOT_ENV_VAR",
    "DEFAULT_BOBA_BATCHED_ROOT",
    "EXPECTED_CONDA_ENV",
    "VENDORED_GSPLAT_PACKAGE",
    "VENDORED_GSPLAT_ROOT",
    "gsplat",
    "import_gsplat",
    "rasterization",
    "rasterization_shared_template",
    "validate_gsplat_runtime",
]
