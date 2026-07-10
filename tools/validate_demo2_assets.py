#!/usr/bin/env python3
"""Validate the self-contained runtime assets for Boba Demo 2."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
import sys
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASE = "single_push_rope_4"
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
CASE_ASSET_KEYS = tuple(key for key in REQUIRED_MANIFEST_KEYS if key != "config")
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"
SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}\Z")


class AssetValidationError(RuntimeError):
    """A packaged Demo 2 asset failed validation."""


def _load_json(path: Path, description: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise AssetValidationError(f"Missing {description}: {path}") from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AssetValidationError(f"Unable to read {description} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AssetValidationError(f"{description.capitalize()} must contain a JSON object: {path}")
    return value


def _safe_relative_path(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise AssetValidationError(f"Manifest field {field!r} must be a non-empty path string")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise AssetValidationError(
            f"Manifest field {field!r} must be a relative path without '..': {value!r}"
        )
    return path


def _reject_lfs_pointer(path: Path) -> None:
    try:
        with path.open("rb") as handle:
            prefix = handle.read(256)
    except OSError as exc:
        raise AssetValidationError(f"Unable to read asset {path}: {exc}") from exc
    if prefix.startswith(LFS_POINTER_PREFIX):
        raise AssetValidationError(
            f"Asset is a Git LFS pointer rather than payload data: {path}. "
            "Fetch the real object before running Demo 2."
        )


def _resolve_manifest_paths(
    manifest: Mapping[str, Any], case_dir: Path, repo_root: Path
) -> dict[str, Path]:
    missing = [key for key in REQUIRED_MANIFEST_KEYS if key not in manifest]
    if missing:
        raise AssetValidationError(
            "Manifest is missing required field(s): " + ", ".join(missing)
        )

    resolved: dict[str, Path] = {}
    for key in REQUIRED_MANIFEST_KEYS:
        relative = _safe_relative_path(manifest[key], key)
        base = repo_root if key == "config" else case_dir
        path = base / relative
        if not path.is_file():
            raise AssetValidationError(f"Manifest field {key!r} points to a missing file: {path}")
        if path.stat().st_size == 0:
            raise AssetValidationError(f"Manifest field {key!r} points to an empty file: {path}")
        _reject_lfs_pointer(path)
        resolved[key] = path
    return resolved


def _read_ply_header(path: Path, maximum_header_bytes: int = 65536) -> list[str]:
    try:
        with path.open("rb") as handle:
            header_bytes = handle.read(maximum_header_bytes)
    except OSError as exc:
        raise AssetValidationError(f"Unable to read Gaussian PLY {path}: {exc}") from exc

    marker = b"end_header"
    marker_index = header_bytes.find(marker)
    if marker_index < 0:
        raise AssetValidationError(
            f"Gaussian PLY has no end_header marker in its first "
            f"{maximum_header_bytes} bytes: {path}"
        )
    line_end = header_bytes.find(b"\n", marker_index)
    if line_end < 0:
        line_end = marker_index + len(marker)
    try:
        header = header_bytes[:line_end].decode("ascii")
    except UnicodeDecodeError as exc:
        raise AssetValidationError(f"Gaussian PLY header is not ASCII: {path}") from exc
    return [line.strip() for line in header.splitlines() if line.strip()]


def _validate_ply(path: Path) -> int:
    lines = _read_ply_header(path)
    if not lines or lines[0] != "ply":
        raise AssetValidationError(f"Gaussian PLY does not begin with the 'ply' signature: {path}")
    formats = [line.split() for line in lines if line.startswith("format ")]
    if len(formats) != 1 or len(formats[0]) != 3 or formats[0][2] != "1.0":
        raise AssetValidationError(f"Gaussian PLY has an invalid or unsupported format line: {path}")
    if formats[0][1] not in {"ascii", "binary_little_endian"}:
        raise AssetValidationError(
            f"Gaussian PLY format must be ascii or binary_little_endian: {path}"
        )

    vertex_count = None
    for line in lines:
        parts = line.split()
        if len(parts) == 3 and parts[:2] == ["element", "vertex"]:
            try:
                vertex_count = int(parts[2])
            except ValueError as exc:
                raise AssetValidationError(f"Gaussian PLY has an invalid vertex count: {path}") from exc
            break
    if vertex_count is None or vertex_count < 1:
        raise AssetValidationError(f"Gaussian PLY must declare at least one vertex: {path}")

    properties = {
        parts[-1]
        for line in lines
        if (parts := line.split()) and parts[0] == "property" and len(parts) >= 3
    }
    required_properties = {
        "x",
        "y",
        "z",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        "opacity",
        "scale_0",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    }
    missing = sorted(required_properties - properties)
    if missing:
        raise AssetValidationError(
            "Gaussian PLY is missing required Gaussian vertex properties: "
            + ", ".join(missing)
        )
    return vertex_count


def _validate_controller_bank(path: Path, case_name: str, expected_count: int) -> int:
    try:
        with path.open("rb") as handle:
            bank = pickle.load(handle)
    except ModuleNotFoundError as exc:
        raise AssetValidationError(
            f"Unable to inspect controller bank because Python module {exc.name!r} is missing. "
            "Activate the Boba-Batched phystwin environment and retry."
        ) from exc
    except (OSError, pickle.UnpicklingError, EOFError, AttributeError, ValueError) as exc:
        raise AssetValidationError(f"Unable to load filtered controller bank {path}: {exc}") from exc

    if not isinstance(bank, dict):
        raise AssetValidationError("Filtered controller bank must contain a dictionary")
    if bank.get("case_name") != case_name:
        raise AssetValidationError(
            f"Filtered controller bank case_name={bank.get('case_name')!r}; "
            f"expected {case_name!r}"
        )

    metadata = bank.get("meta")
    if not isinstance(metadata, dict):
        raise AssetValidationError("Filtered controller bank is missing its 'meta' dictionary")
    if metadata.get("case_name") != case_name:
        raise AssetValidationError(
            f"Filtered controller bank meta.case_name={metadata.get('case_name')!r}; "
            f"expected {case_name!r}"
        )

    trajectories = bank.get("controller_points_group")
    try:
        trajectory_count = len(trajectories)
    except TypeError as exc:
        raise AssetValidationError(
            "Filtered controller bank is missing a sized 'controller_points_group'"
        ) from exc
    if trajectory_count != expected_count:
        raise AssetValidationError(
            f"Filtered controller bank contains {trajectory_count} trajectories; "
            f"expected exactly {expected_count}"
        )

    source_indices = bank.get("source_indices")
    if source_indices is not None and len(source_indices) != expected_count:
        raise AssetValidationError(
            f"Filtered controller bank has {len(source_indices)} source indices; "
            f"expected {expected_count}"
        )
    return trajectory_count


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_provenance(
    provenance_path: Path,
    case_name: str,
    manifest: Mapping[str, Any],
    resolved_paths: Mapping[str, Path],
) -> int:
    if not provenance_path.exists():
        return 0
    provenance = _load_json(provenance_path, "asset provenance")
    if provenance.get("schema_version") != 1:
        raise AssetValidationError(
            f"Unsupported asset provenance schema_version: "
            f"{provenance.get('schema_version')!r}"
        )
    if provenance.get("case_name") != case_name:
        raise AssetValidationError(
            f"Asset provenance case_name={provenance.get('case_name')!r}; "
            f"expected {case_name!r}"
        )
    files = provenance.get("files")
    if not isinstance(files, dict):
        raise AssetValidationError("Asset provenance must contain a 'files' object")

    checked = 0
    for key in CASE_ASSET_KEYS:
        packaged_name = manifest[key]
        record = files.get(packaged_name)
        if not isinstance(record, dict):
            raise AssetValidationError(
                f"Asset provenance has no record for packaged file {packaged_name!r}"
            )
        expected_size = record.get("size_bytes")
        if not isinstance(expected_size, int) or isinstance(expected_size, bool) or expected_size < 1:
            raise AssetValidationError(
                f"Asset provenance has an invalid size_bytes for {packaged_name!r}"
            )
        actual_size = resolved_paths[key].stat().st_size
        if actual_size != expected_size:
            raise AssetValidationError(
                f"Asset size mismatch for {packaged_name}: expected {expected_size}, "
                f"found {actual_size}"
            )

        expected_hash = record.get("sha256")
        if not isinstance(expected_hash, str) or not SHA256_PATTERN.fullmatch(expected_hash):
            raise AssetValidationError(
                f"Asset provenance has an invalid sha256 for {packaged_name!r}"
            )
        actual_hash = _sha256(resolved_paths[key])
        if actual_hash != expected_hash.lower():
            raise AssetValidationError(
                f"Asset SHA256 mismatch for {packaged_name}: expected "
                f"{expected_hash.lower()}, found {actual_hash}"
            )
        checked += 1
    return checked


def validate_case_assets(
    case_name: str = DEFAULT_CASE,
    assets_root: Path | str = REPO_ROOT / "assets",
    repo_root: Path | str = REPO_ROOT,
    expected_trajectories: int = 100,
) -> dict[str, int]:
    if not case_name or Path(case_name).name != case_name:
        raise AssetValidationError(f"Invalid case name: {case_name!r}")
    if expected_trajectories < 1:
        raise AssetValidationError("Expected trajectory count must be positive")

    assets_root = Path(assets_root).resolve()
    repo_root = Path(repo_root).resolve()
    case_dir = assets_root / case_name
    manifest_path = case_dir / "manifest.json"
    manifest = _load_json(manifest_path, "case manifest")
    resolved = _resolve_manifest_paths(manifest, case_dir, repo_root)

    metadata = _load_json(resolved["metadata"], "case metadata")
    if not metadata:
        raise AssetValidationError(f"Case metadata is empty: {resolved['metadata']}")

    vertex_count = _validate_ply(resolved["gaussian_ply"])
    trajectory_count = _validate_controller_bank(
        resolved["controller_bank"], case_name, expected_trajectories
    )
    provenance_count = _validate_provenance(
        case_dir / "asset_source.json", case_name, manifest, resolved
    )
    return {
        "assets": len(resolved),
        "gaussian_vertices": vertex_count,
        "trajectories": trajectory_count,
        "provenance_records": provenance_count,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default=DEFAULT_CASE, help="Case directory under assets/")
    parser.add_argument(
        "--assets-root",
        type=Path,
        default=REPO_ROOT / "assets",
        help="Asset root (default: repository assets directory)",
    )
    parser.add_argument(
        "--expected-trajectories",
        type=int,
        default=100,
        help="Required number of filtered controller trajectories (default: 100)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = validate_case_assets(
            case_name=args.case,
            assets_root=args.assets_root,
            repo_root=REPO_ROOT,
            expected_trajectories=args.expected_trajectories,
        )
    except AssetValidationError as exc:
        print(f"[Demo2 assets] ERROR: {exc}", file=sys.stderr)
        return 1

    provenance_note = (
        f", {report['provenance_records']} provenance hashes"
        if report["provenance_records"]
        else ", no asset_source.json"
    )
    print(
        f"[Demo2 assets] OK: {args.case}: {report['assets']} files, "
        f"{report['trajectories']} controller trajectories, "
        f"{report['gaussian_vertices']} Gaussian vertices{provenance_note}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
