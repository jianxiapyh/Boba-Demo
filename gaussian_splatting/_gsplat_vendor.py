from __future__ import annotations

import importlib
import functools
import inspect
import os
import sys
from pathlib import Path
from typing import Any

import torch


EXPECTED_CONDA_ENV = "phystwin"
REPO_ROOT = Path(__file__).resolve().parents[1]
VENDORED_GSPLAT_ROOT = Path(__file__).resolve().parent / "submodules" / "gsplat"
VENDORED_GSPLAT_PACKAGE = VENDORED_GSPLAT_ROOT / "gsplat"

GSPLAT_VERSION = "1.5.3"
BOBA_BATCHED_SOURCE_COMMIT = "99e50055a60a4bc7e5022abba1a938bf386b273d"
UPSTREAM_GSPLAT_COMMIT = "937e29912570c372bed6747a5c9bf85fed877bae"
RASTERIZATION_ENTRYPOINT = "rasterization"
CUSTOM_API_MARKERS = (
    "rasterization_shared_template",
    "fully_fused_projection_shared_template",
    "fully_fused_projection_shared_template_sh_rgb",
)


def _active_env_name() -> str:
    env_name = os.environ.get("CONDA_DEFAULT_ENV")
    if env_name:
        return env_name
    return Path(sys.prefix).resolve().name


def _runtime_hint() -> str:
    return (
        f"Run the demo in the existing {EXPECTED_CONDA_ENV!r} environment, for example:\n"
        f"  conda run -n {EXPECTED_CONDA_ENV} env PYTHONNOUSERSITE=1 "
        "python boba_quest_immersive.py --case_name rope_game\n"
        "The gsplat source is committed in Boba-Demo and must not be replaced by "
        "an editable package from another checkout."
    )


def _is_vendored_gsplat_module(gsplat_module: Any) -> bool:
    module_file = getattr(gsplat_module, "__file__", None)
    if not module_file:
        return False

    module_path = Path(module_file).resolve()
    package_path = VENDORED_GSPLAT_PACKAGE.resolve()
    return module_path == package_path / "__init__.py" or package_path in module_path.parents


def _prepare_vendored_gsplat_import() -> None:
    package_init = VENDORED_GSPLAT_PACKAGE / "__init__.py"
    if not package_init.is_file():
        raise RuntimeError(
            "Boba Demo is missing its committed custom gsplat fork.\n"
            f"Expected package: {VENDORED_GSPLAT_PACKAGE}\n"
            "Restore the vendored source from the Boba-Demo checkout."
        )

    vendored_root = str(VENDORED_GSPLAT_ROOT.resolve())
    sys.path[:] = [
        path_entry
        for path_entry in sys.path
        if not path_entry or str(Path(path_entry).expanduser().resolve()) != vendored_root
    ]
    sys.path.insert(0, vendored_root)

    loaded_gsplat = sys.modules.get("gsplat")
    if loaded_gsplat is None or _is_vendored_gsplat_module(loaded_gsplat):
        return

    # An editable Boba-Batched or PyPI install may already have been imported by
    # another dependency. Purge that package tree before resolving our local fork.
    for module_name in tuple(sys.modules):
        if module_name == "gsplat" or module_name.startswith("gsplat."):
            del sys.modules[module_name]


def validate_gsplat_runtime(gsplat_module: Any) -> None:
    active_env = _active_env_name()
    if active_env != EXPECTED_CONDA_ENV:
        raise RuntimeError(
            "Boba Demo requires the working Boba-Batched conda environment. "
            f"Expected {EXPECTED_CONDA_ENV!r}, active {active_env!r}, "
            f"sys.prefix={Path(sys.prefix).resolve()}.\n{_runtime_hint()}"
        )

    module_file = getattr(gsplat_module, "__file__", None)
    if not module_file or not _is_vendored_gsplat_module(gsplat_module):
        raise RuntimeError(
            "Boba Demo resolved gsplat from an unexpected location.\n"
            f"Resolved path: {module_file or '<missing __file__>'}\n"
            f"Expected under: {VENDORED_GSPLAT_PACKAGE}\n{_runtime_hint()}"
        )

    actual_version = str(getattr(gsplat_module, "__version__", "unknown"))
    if actual_version != GSPLAT_VERSION:
        raise RuntimeError(
            "Boba Demo resolved the wrong gsplat version. "
            f"Expected {GSPLAT_VERSION}, got {actual_version} from {module_file}."
        )

    required_api = (RASTERIZATION_ENTRYPOINT, *CUSTOM_API_MARKERS)
    missing_markers = [
        marker for marker in required_api if not callable(getattr(gsplat_module, marker, None))
    ]
    if missing_markers:
        raise RuntimeError(
            "Boba Demo's vendored gsplat does not match the Boba-Batched custom fork. "
            f"Missing callable API marker(s): {', '.join(missing_markers)}.\n"
            f"Resolved path: {module_file}"
        )


def import_gsplat():
    if _active_env_name() != EXPECTED_CONDA_ENV:
        raise RuntimeError(
            "Boba Demo requires the working Boba-Batched conda environment. "
            f"Expected {EXPECTED_CONDA_ENV!r}, active {_active_env_name()!r}.\n"
            f"{_runtime_hint()}"
        )

    _prepare_vendored_gsplat_import()
    try:
        gsplat_module = importlib.import_module("gsplat")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Boba Demo could not import its committed custom gsplat fork.\n"
            f"Expected source root: {VENDORED_GSPLAT_ROOT}\n{_runtime_hint()}"
        ) from exc

    validate_gsplat_runtime(gsplat_module)
    return gsplat_module


def gsplat_provenance() -> dict[str, str]:
    return {
        "fork": "Boba-Batched custom gsplat",
        "version": GSPLAT_VERSION,
        "boba_batched_source_commit": BOBA_BATCHED_SOURCE_COMMIT,
        "upstream_commit": UPSTREAM_GSPLAT_COMMIT,
        "source_path": str(Path(gsplat.__file__).resolve()),
        "rasterization_entrypoint": RASTERIZATION_ENTRYPOINT,
        "custom_api_marker": "rasterization_shared_template",
    }


def log_gsplat_runtime() -> None:
    provenance = gsplat_provenance()
    print(
        "[Boba Demo] gsplat backend: "
        f"{provenance['fork']} v{provenance['version']} "
        f"source={provenance['source_path']} "
        f"entrypoint={provenance['rasterization_entrypoint']} "
        "stereo=two_camera_standard",
        flush=True,
    )


gsplat = import_gsplat()
rasterization_shared_template = gsplat.rasterization_shared_template


def _empty_rasterization_result(*args, **kwargs):
    bound = inspect.signature(gsplat.rasterization).bind_partial(*args, **kwargs)
    bound.apply_defaults()
    inputs = bound.arguments

    means = inputs["means"]
    colors = inputs["colors"]
    viewmats = inputs["viewmats"]
    backgrounds = inputs["backgrounds"]
    width = int(inputs["width"])
    height = int(inputs["height"])
    tile_size = int(inputs["tile_size"])
    render_mode = inputs["render_mode"]

    batch_shape = tuple(viewmats.shape[:-3])
    num_cameras = int(viewmats.shape[-3])
    if render_mode.startswith("RGB"):
        color_channels = 3 if inputs["sh_degree"] is not None else int(colors.shape[-1])
    else:
        color_channels = 0
    output_channels = color_channels + int(render_mode in {"RGB+D", "RGB+ED"})
    if render_mode in {"D", "ED"}:
        output_channels = 1

    render_colors = means.new_zeros(
        (*batch_shape, num_cameras, height, width, output_channels)
    )
    render_alphas = means.new_zeros((*batch_shape, num_cameras, height, width, 1))
    if backgrounds is not None and color_channels:
        render_colors[..., :color_channels] = backgrounds.to(
            device=means.device,
            dtype=means.dtype,
        )[..., None, None, :]

    if inputs["packed"]:
        projection_shape = (0,)
    else:
        projection_shape = (*batch_shape, num_cameras, 0)
    empty_long = torch.empty((0,), device=means.device, dtype=torch.long)
    info = {
        "camera_ids": empty_long if inputs["packed"] else None,
        "gaussian_ids": empty_long if inputs["packed"] else None,
        "radii": torch.empty(
            (*projection_shape, 2), device=means.device, dtype=torch.int32
        ),
        "means2d": means.new_empty((*projection_shape, 2)),
        "depths": means.new_empty(projection_shape),
        "conics": means.new_empty((*projection_shape, 3)),
        "compensations": None,
        "tile_width": (width + tile_size - 1) // tile_size,
        "tile_height": (height + tile_size - 1) // tile_size,
        "width": width,
        "height": height,
        "tile_size": tile_size,
        "n_cameras": num_cameras,
    }
    return render_colors, render_alphas, info


@functools.wraps(gsplat.rasterization)
def rasterization(*args, **kwargs):
    """Call the fork's standard API, guarding its zero-Gaussian native edge case."""
    means = kwargs.get("means", args[0] if args else None)
    if torch.is_tensor(means) and means.shape[-2] == 0:
        return _empty_rasterization_result(*args, **kwargs)
    return gsplat.rasterization(*args, **kwargs)


log_gsplat_runtime()


__all__ = [
    "BOBA_BATCHED_SOURCE_COMMIT",
    "CUSTOM_API_MARKERS",
    "EXPECTED_CONDA_ENV",
    "GSPLAT_VERSION",
    "RASTERIZATION_ENTRYPOINT",
    "REPO_ROOT",
    "UPSTREAM_GSPLAT_COMMIT",
    "VENDORED_GSPLAT_PACKAGE",
    "VENDORED_GSPLAT_ROOT",
    "gsplat",
    "gsplat_provenance",
    "import_gsplat",
    "log_gsplat_runtime",
    "rasterization",
    "rasterization_shared_template",
    "validate_gsplat_runtime",
]
