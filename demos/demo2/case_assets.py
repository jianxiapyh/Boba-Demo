"""Manifest-based asset resolution for the self-contained Demo 2 case."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_MANIFEST_KEYS = (
    "best_model",
    "calibrate",
    "config",
    "final_data",
    "gaussian_ply",
    "metadata",
    "optimal_params",
    "controller_bank",
    "background_image",
)


@dataclass(frozen=True)
class Demo2CaseAssets:
    """Resolved, validated paths for one packaged Demo 2 case."""

    case_name: str
    manifest_path: Path
    best_model: Path
    calibrate: Path
    config: Path
    final_data: Path
    gaussian_ply: Path
    metadata: Path
    optimal_params: Path
    controller_bank: Path
    background_image: Path


def _path_inside_repo(repo_root: Path, path: Path, *, key: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError(
            f"Demo 2 manifest key {key!r} resolves outside the repository: {resolved}"
        ) from exc
    return resolved


def resolve_demo2_case_assets(
    repo_root: str | Path,
    case_name: str,
) -> Demo2CaseAssets:
    """Resolve and validate ``assets/<case_name>/manifest.json``.

    ``config`` follows the existing Boba manifest convention and is relative to
    the repository root. All runtime payloads are relative to the case directory.
    """

    root = Path(repo_root).resolve()
    requested_case = str(case_name).strip()
    if not requested_case or Path(requested_case).name != requested_case:
        raise ValueError(f"Invalid Demo 2 case name: {case_name!r}")

    manifest_path = root / "assets" / requested_case / "manifest.json"
    if not manifest_path.is_file():
        available = sorted(
            path.parent.name for path in (root / "assets").glob("*/manifest.json")
        )
        available_text = ", ".join(available) if available else "none"
        raise FileNotFoundError(
            f"Demo 2 manifest for case {requested_case!r} was not found at "
            f"{manifest_path}. Packaged cases: {available_text}."
        )

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise TypeError(
            f"Demo 2 manifest must contain a JSON object: {manifest_path}"
        )

    missing_keys = [key for key in REQUIRED_MANIFEST_KEYS if not manifest.get(key)]
    if missing_keys:
        raise KeyError(
            f"Demo 2 manifest {manifest_path} is missing required keys: "
            + ", ".join(missing_keys)
        )

    case_dir = manifest_path.parent
    resolved_paths: dict[str, Path] = {}
    missing_files: list[tuple[str, Path]] = []
    for key in REQUIRED_MANIFEST_KEYS:
        base_dir = root if key == "config" else case_dir
        path = _path_inside_repo(root, base_dir / str(manifest[key]), key=key)
        resolved_paths[key] = path
        if not path.is_file():
            missing_files.append((key, path))

    if missing_files:
        details = ", ".join(f"{key}={path}" for key, path in missing_files)
        raise FileNotFoundError(
            f"Demo 2 case {requested_case!r} has missing packaged assets: {details}"
        )

    return Demo2CaseAssets(
        case_name=requested_case,
        manifest_path=manifest_path,
        **resolved_paths,
    )


def select_controller_bank(
    case_assets: Demo2CaseAssets,
    controller_pkl: str | Path | None,
) -> Path:
    """Use the packaged controller bank unless a development override is given."""

    selected = (
        Path(controller_pkl).expanduser().resolve()
        if controller_pkl is not None
        else case_assets.controller_bank
    )
    if not selected.is_file():
        source = "--controller_pkl override" if controller_pkl is not None else "manifest"
        raise FileNotFoundError(
            f"Demo 2 controller bank selected from {source} is missing: {selected}"
        )
    return selected


def load_demo2_case_config(
    case_assets: Demo2CaseAssets,
    cfg: Any,
    logger: Any,
) -> dict[str, Any]:
    """Load simulation and camera configuration from resolved manifest assets."""

    logger.info(f"Load configuration from: {case_assets.config}")
    cfg.load_from_yaml(str(case_assets.config))

    logger.info(f"Load optimal parameters from: {case_assets.optimal_params}")
    with case_assets.optimal_params.open("rb") as handle:
        optimal_params = pickle.load(handle)
    cfg.set_optimal_params(optimal_params)

    with case_assets.calibrate.open("rb") as handle:
        c2ws = pickle.load(handle)
    cfg.c2ws = np.asarray(c2ws)
    cfg.w2cs = np.asarray([np.linalg.inv(c2w) for c2w in c2ws])

    with case_assets.metadata.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    cfg.intrinsics = np.asarray(metadata["intrinsics"])
    cfg.WH = metadata["WH"]
    cfg.bg_img_path = str(case_assets.background_image)
    return metadata
