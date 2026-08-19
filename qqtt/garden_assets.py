"""Mip-NeRF 360 Garden asset acquisition, preparation, and validation.

Only small manifests and deterministic processing code are tracked.  The
official Graphdeco source model and all generated runtime PLYs live below the
gitignored ``data/garden`` directory.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable

import numpy as np


GARDEN_PIPELINE_VERSION = 4
GARDEN_QUALITIES = ("auto", "full", "balanced", "performance")
GARDEN_MANIFEST_RELATIVE_PATH = Path("assets/scenes/garden/manifest.json")


class GardenAssetError(RuntimeError):
    pass


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise GardenAssetError(f"Unable to read Garden JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise GardenAssetError(f"Garden JSON must contain an object: {path}")
    return value


def load_garden_manifest(repo_root: str | Path) -> tuple[Path, dict]:
    repo_root = Path(repo_root).resolve()
    path = repo_root / GARDEN_MANIFEST_RELATIVE_PATH
    if not path.is_file():
        raise GardenAssetError(f"Garden scene manifest is missing: {path}")
    manifest = _load_json(path)
    if int(manifest.get("schema_version", -1)) != 1:
        raise GardenAssetError(
            f"Unsupported Garden manifest schema: {manifest.get('schema_version')}"
        )
    return path, manifest


def resolve_repo_path(repo_root: str | Path, relative_path: str | Path) -> Path:
    repo_root = Path(repo_root).resolve()
    path = (repo_root / Path(relative_path)).resolve()
    try:
        path.relative_to(repo_root)
    except ValueError as exc:
        raise GardenAssetError(f"Garden path escapes the repository: {path}") from exc
    return path


def normalize_garden_quality(quality: str) -> str:
    normalized = str(quality or "auto").strip().lower()
    if normalized not in GARDEN_QUALITIES:
        raise GardenAssetError(
            f"Unsupported Garden quality {quality!r}; expected one of {GARDEN_QUALITIES}."
        )
    return normalized


def garden_source_point_cloud_path(repo_root: str | Path, manifest: dict | None = None) -> Path:
    repo_root = Path(repo_root).resolve()
    if manifest is None:
        _, manifest = load_garden_manifest(repo_root)
    source_dir = resolve_repo_path(repo_root, manifest["local"]["source_dir"])
    member = Path(manifest["source"]["point_cloud_member"])
    return source_dir / member


def garden_source_cameras_path(repo_root: str | Path, manifest: dict | None = None) -> Path:
    repo_root = Path(repo_root).resolve()
    if manifest is None:
        _, manifest = load_garden_manifest(repo_root)
    source_dir = resolve_repo_path(repo_root, manifest["local"]["source_dir"])
    member = Path(manifest["source"]["cameras_member"])
    return source_dir / member


def garden_archive_path(repo_root: str | Path, manifest: dict | None = None) -> Path:
    repo_root = Path(repo_root).resolve()
    if manifest is None:
        _, manifest = load_garden_manifest(repo_root)
    download_dir = resolve_repo_path(repo_root, manifest["local"]["download_dir"])
    return download_dir / str(manifest["source"]["archive_filename"])


def garden_quality_paths(
    repo_root: str | Path,
    quality: str,
    manifest: dict | None = None,
) -> tuple[Path, Path]:
    quality = normalize_garden_quality(quality)
    if quality == "auto":
        raise GardenAssetError("Resolve auto quality before requesting a runtime path.")
    repo_root = Path(repo_root).resolve()
    if manifest is None:
        _, manifest = load_garden_manifest(repo_root)
    spec = manifest.get("qualities", {}).get(quality)
    if not isinstance(spec, dict):
        raise GardenAssetError(f"Garden manifest has no quality tier {quality!r}.")
    return (
        resolve_repo_path(repo_root, spec["ply"]),
        resolve_repo_path(repo_root, spec["metadata"]),
    )


def validate_garden_source(repo_root: str | Path) -> Path:
    _, manifest = load_garden_manifest(repo_root)
    source_path = garden_source_point_cloud_path(repo_root, manifest)
    if not source_path.is_file():
        raise GardenAssetError(
            "Garden source Gaussian is not installed. Run:\n"
            "  conda run -n phystwin-cu132 env PYTHONNOUSERSITE=1 "
            "python tools/fetch_demo_case_assets.py --scene garden --fetch"
        )
    with source_path.open("rb") as handle:
        if handle.read(4) != b"ply\n":
            raise GardenAssetError(f"Garden source is not a valid PLY: {source_path}")
    expected = str(manifest["source"].get("point_cloud_sha256") or "").strip().lower()
    if expected:
        actual = sha256_file(source_path)
        if actual != expected:
            raise GardenAssetError(
                f"Garden source checksum mismatch at {source_path}: {actual}; expected {expected}. "
                "Rerun the fetch command to restore the official model."
            )
    return source_path


def _calibration_hash(repo_root: Path, manifest: dict) -> str:
    calibration_path = resolve_repo_path(repo_root, manifest["calibration"])
    return sha256_file(calibration_path)


def validate_garden_quality(
    repo_root: str | Path,
    quality: str,
    *,
    verify_payload_hash: bool = False,
) -> tuple[Path, dict]:
    repo_root = Path(repo_root).resolve()
    _, manifest = load_garden_manifest(repo_root)
    ply_path, metadata_path = garden_quality_paths(repo_root, quality, manifest)
    if not ply_path.is_file() or not metadata_path.is_file():
        raise GardenAssetError(
            f"Garden {quality} runtime assets are missing. Run:\n"
            "  conda run -n phystwin-cu132 env PYTHONNOUSERSITE=1 "
            "python tools/fetch_demo_case_assets.py --scene garden --fetch"
        )
    with ply_path.open("rb") as handle:
        if handle.read(4) != b"ply\n":
            raise GardenAssetError(f"Garden runtime PLY has an invalid header: {ply_path}")
    metadata = _load_json(metadata_path)
    quality_spec = manifest["qualities"][str(quality)]
    checks = {
        "pipeline_version": GARDEN_PIPELINE_VERSION,
        "quality": str(quality),
        "retention": float(quality_spec["retention"]),
        "pruning_mode": str(quality_spec.get("pruning_mode", "opacity_area")),
        "calibration_sha256": _calibration_hash(repo_root, manifest),
        "source_sha256": str(manifest["source"]["point_cloud_sha256"]),
    }
    for key, expected in checks.items():
        actual = metadata.get(key)
        if actual != expected:
            raise GardenAssetError(
                f"Garden {quality} metadata is stale ({key}={actual!r}, expected {expected!r}). "
                "Rerun the Garden fetch command."
            )
    if int(metadata.get("gaussian_count", 0)) <= 0:
        raise GardenAssetError(f"Garden {quality} metadata has no Gaussian count.")
    validate_spatial_chunk_metadata(metadata, manifest)
    if verify_payload_hash:
        expected_ply_hash = str(metadata.get("ply_sha256") or "")
        actual_ply_hash = sha256_file(ply_path)
        if not expected_ply_hash or actual_ply_hash != expected_ply_hash:
            raise GardenAssetError(
                f"Garden {quality} runtime checksum mismatch at {ply_path}. "
                "Rerun the Garden fetch command."
            )
    return ply_path, metadata


def available_garden_qualities(repo_root: str | Path) -> list[str]:
    _, manifest = load_garden_manifest(repo_root)
    available = []
    for quality in manifest.get("quality_order", []):
        try:
            validate_garden_quality(repo_root, quality, verify_payload_hash=False)
        except GardenAssetError:
            continue
        available.append(str(quality))
    return available


def garden_profile_key(
    *,
    model_sha256: str,
    gpu_name: str,
    driver_version: str,
    renderer_revision: str,
    eye_resolution: int,
) -> str:
    payload = {
        "model_sha256": str(model_sha256),
        "gpu_name": str(gpu_name),
        "driver_version": str(driver_version),
        "renderer_revision": str(renderer_revision),
        "eye_resolution": int(eye_resolution),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resolve_garden_quality(
    repo_root: str | Path,
    requested_quality: str,
    *,
    profile_key: str | None = None,
    target_fps: float = 72.0,
) -> str:
    repo_root = Path(repo_root).resolve()
    requested = normalize_garden_quality(requested_quality)
    _, manifest = load_garden_manifest(repo_root)
    available = available_garden_qualities(repo_root)
    if not available:
        validate_garden_source(repo_root)
        raise GardenAssetError(
            "Garden source exists but no prepared runtime tier is valid. Rerun the Garden fetch command."
        )
    if requested != "auto":
        if requested not in available:
            validate_garden_quality(repo_root, requested)
        return requested

    if profile_key:
        cache_path = resolve_repo_path(repo_root, manifest["local"]["profile_cache"])
        if cache_path.is_file():
            cache = _load_json(cache_path)
            entries = cache.get("entries", {}).get(profile_key, {})
            for quality in manifest.get("quality_order", []):
                if quality not in available:
                    continue
                fps = entries.get(quality, {}).get("source_fps")
                if fps is not None and float(fps) >= float(target_fps):
                    return str(quality)
            profiled = [quality for quality in available if quality in entries]
            if profiled:
                return "performance" if "performance" in available else available[-1]

    fallback = str(manifest.get("default_uncalibrated_auto_quality", "balanced"))
    if fallback in available:
        return fallback
    return available[-1]


def record_garden_profile(
    repo_root: str | Path,
    *,
    profile_key: str,
    quality: str,
    source_fps: float,
    sample_count: int,
) -> Path:
    repo_root = Path(repo_root).resolve()
    _, manifest = load_garden_manifest(repo_root)
    cache_path = resolve_repo_path(repo_root, manifest["local"]["profile_cache"])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = {"schema_version": 1, "entries": {}}
    if cache_path.is_file():
        try:
            cache = _load_json(cache_path)
        except GardenAssetError:
            pass
    entries = cache.setdefault("entries", {})
    profile = entries.setdefault(str(profile_key), {})
    profile[str(quality)] = {
        "source_fps": float(source_fps),
        "sample_count": int(sample_count),
    }
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(cache, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, cache_path)
    return cache_path


def _download_with_resume(
    url: str,
    destination: Path,
    progress_callback: Callable[[int, int | None], None] | None = None,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    existing = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(str(url))
    if existing > 0:
        request.add_header("Range", f"bytes={existing}-")
    with urllib.request.urlopen(request) as response:  # noqa: S310 - manifest-owned HTTPS URL
        status = int(getattr(response, "status", 200) or 200)
        append = existing > 0 and status == 206
        if existing > 0 and not append:
            existing = 0
        content_length = response.headers.get("Content-Length")
        total = None if content_length is None else int(content_length) + existing
        mode = "ab" if append else "wb"
        downloaded = existing
        with partial.open(mode) as handle:
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                if progress_callback is not None:
                    progress_callback(downloaded, total)
    os.replace(partial, destination)
    return destination


def _extract_member(archive: Path, member_name: str, destination: Path) -> None:
    normalized_member = str(Path(member_name).as_posix()).lstrip("/")
    with zipfile.ZipFile(archive, "r") as bundle:
        names = {str(Path(name).as_posix()).lstrip("/"): name for name in bundle.namelist()}
        actual_member = names.get(normalized_member)
        if actual_member is None:
            suffix_matches = [name for normalized, name in names.items() if normalized.endswith(normalized_member)]
            if len(suffix_matches) != 1:
                raise GardenAssetError(
                    f"Official archive does not contain {member_name!r}; matches={suffix_matches}."
                )
            actual_member = suffix_matches[0]
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        with bundle.open(actual_member, "r") as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
        os.replace(temporary, destination)


def fetch_garden_source(
    repo_root: str | Path,
    *,
    archive_override: str | Path | None = None,
    progress_callback: Callable[[int, int | None], None] | None = None,
) -> Path:
    repo_root = Path(repo_root).resolve()
    _, manifest = load_garden_manifest(repo_root)
    source_path = garden_source_point_cloud_path(repo_root, manifest)
    expected_source_hash = str(manifest["source"]["point_cloud_sha256"]).lower()
    if source_path.is_file() and sha256_file(source_path) == expected_source_hash:
        return source_path

    archive = (
        Path(archive_override).expanduser().resolve()
        if archive_override is not None
        else garden_archive_path(repo_root, manifest)
    )
    if not archive.is_file():
        _download_with_resume(
            str(manifest["source"]["official_archive_url"]),
            archive,
            progress_callback=progress_callback,
        )
    expected_archive_hash = str(manifest["source"].get("archive_sha256") or "").lower()
    if expected_archive_hash:
        actual_archive_hash = sha256_file(archive)
        if actual_archive_hash != expected_archive_hash:
            raise GardenAssetError(
                f"Official Garden archive checksum mismatch: {actual_archive_hash}; "
                f"expected {expected_archive_hash}."
            )

    _extract_member(archive, manifest["source"]["point_cloud_member"], source_path)
    camera_path = garden_source_cameras_path(repo_root, manifest)
    try:
        _extract_member(archive, manifest["source"]["cameras_member"], camera_path)
    except GardenAssetError:
        # The runtime needs only the Gaussian PLY; cameras are retained when the
        # upstream bundle includes them because they are useful for calibration.
        pass
    actual_source_hash = sha256_file(source_path)
    if actual_source_hash != expected_source_hash:
        raise GardenAssetError(
            f"Extracted Garden PLY checksum mismatch: {actual_source_hash}; "
            f"expected {expected_source_hash}."
        )
    return source_path


def _matrix_to_quaternion_wxyz(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [0.25 * s, (matrix[2, 1] - matrix[1, 2]) / s, (matrix[0, 2] - matrix[2, 0]) / s, (matrix[1, 0] - matrix[0, 1]) / s],
            dtype=np.float64,
        )
    else:
        diagonal = np.diag(matrix)
        index = int(np.argmax(diagonal))
        if index == 0:
            s = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quat = np.array([ (matrix[2, 1] - matrix[1, 2]) / s, 0.25 * s, (matrix[0, 1] + matrix[1, 0]) / s, (matrix[0, 2] + matrix[2, 0]) / s ])
        elif index == 1:
            s = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quat = np.array([ (matrix[0, 2] - matrix[2, 0]) / s, (matrix[0, 1] + matrix[1, 0]) / s, 0.25 * s, (matrix[1, 2] + matrix[2, 1]) / s ])
        else:
            s = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quat = np.array([ (matrix[1, 0] - matrix[0, 1]) / s, (matrix[0, 2] + matrix[2, 0]) / s, (matrix[1, 2] + matrix[2, 1]) / s, 0.25 * s ])
    quat /= max(float(np.linalg.norm(quat)), 1.0e-12)
    if quat[0] < 0.0:
        quat = -quat
    return quat.astype(np.float32)


def _quaternion_multiply_wxyz(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lhs = np.asarray(lhs, dtype=np.float32).reshape(1, 4)
    rhs = np.asarray(rhs, dtype=np.float32).reshape(-1, 4)
    aw, ax, ay, az = [lhs[:, index] for index in range(4)]
    bw, bx, by, bz = [rhs[:, index] for index in range(4)]
    output = np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=1,
    )
    output /= np.maximum(np.linalg.norm(output, axis=1, keepdims=True), 1.0e-12)
    output[output[:, 0] < 0.0] *= -1.0
    return output.astype(np.float32)


def _real_sh_basis(directions: np.ndarray) -> np.ndarray:
    directions = np.asarray(directions, dtype=np.float64).reshape(-1, 3)
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1.0e-12)
    x, y, z = directions[:, 0], directions[:, 1], directions[:, 2]
    c0 = 0.28209479177387814
    c1 = 0.4886025119029199
    c2 = np.array([1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396])
    c3 = np.array([-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154, -0.4570457994644658, 1.445305721320277, -0.5900435899266435])
    return np.stack(
        [
            np.full_like(x, c0),
            -c1 * y,
            c1 * z,
            -c1 * x,
            c2[0] * x * y,
            c2[1] * y * z,
            c2[2] * (2.0 * z * z - x * x - y * y),
            c2[3] * x * z,
            c2[4] * (x * x - y * y),
            c3[0] * y * (3.0 * x * x - y * y),
            c3[1] * x * y * z,
            c3[2] * y * (4.0 * z * z - x * x - y * y),
            c3[3] * z * (2.0 * z * z - 3.0 * x * x - 3.0 * y * y),
            c3[4] * x * (4.0 * z * z - x * x - y * y),
            c3[5] * z * (x * x - y * y),
            c3[6] * x * (x * x - 3.0 * y * y),
        ],
        axis=1,
    )


def sh_rotation_matrix(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    # Fixed Fibonacci samples make the offline transform deterministic.
    sample_count = 256
    indices = np.arange(sample_count, dtype=np.float64) + 0.5
    z = 1.0 - 2.0 * indices / sample_count
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    phi = math.pi * (3.0 - math.sqrt(5.0)) * indices
    world = np.stack([radius * np.cos(phi), radius * np.sin(phi), z], axis=1)
    source = world @ rotation
    world_basis = _real_sh_basis(world)
    source_basis = _real_sh_basis(source)
    transform, _, _, _ = np.linalg.lstsq(world_basis, source_basis, rcond=None)
    return transform.astype(np.float32)


def _transform_vertex_payload(vertex: np.ndarray, calibration: dict) -> tuple[np.ndarray, np.ndarray]:
    transformed = vertex.copy()
    xyz = np.stack([vertex[axis] for axis in ("x", "y", "z")], axis=1).astype(np.float32)
    center = np.asarray(calibration["source_table_center"], dtype=np.float32).reshape(1, 3)
    rotation = np.asarray(calibration["source_to_canonical_rotation"], dtype=np.float32).reshape(3, 3)
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-4) or np.linalg.det(rotation) < 0.999:
        raise GardenAssetError("Garden source_to_canonical_rotation must be a proper rotation.")
    scale = float(calibration["meters_per_source_unit"])
    if not np.isfinite(scale) or scale <= 0.0:
        raise GardenAssetError("Garden meters_per_source_unit must be finite and positive.")
    canonical_xyz = (xyz - center) @ rotation.T * scale
    for index, axis in enumerate(("x", "y", "z")):
        transformed[axis] = canonical_xyz[:, index]

    scale_names = sorted(
        [name for name in vertex.dtype.names or () if name.startswith("scale_")],
        key=lambda name: int(name.split("_")[-1]),
    )
    log_scale_offset = float(math.log(scale))
    for name in scale_names:
        transformed[name] = np.asarray(vertex[name], dtype=np.float32) + log_scale_offset

    rotation_names = sorted(
        [name for name in vertex.dtype.names or () if name.startswith("rot_")],
        key=lambda name: int(name.split("_")[-1]),
    )
    if len(rotation_names) == 4:
        source_quats = np.stack([vertex[name] for name in rotation_names], axis=1)
        frame_quat = _matrix_to_quaternion_wxyz(rotation)
        canonical_quats = _quaternion_multiply_wxyz(frame_quat, source_quats)
        for index, name in enumerate(rotation_names):
            transformed[name] = canonical_quats[:, index]

    rest_names = sorted(
        [name for name in vertex.dtype.names or () if name.startswith("f_rest_")],
        key=lambda name: int(name.split("_")[-1]),
    )
    if len(rest_names) == 45:
        transform = sh_rotation_matrix(rotation)
        rest = np.stack([vertex[name] for name in rest_names], axis=1).astype(np.float32)
        rest = rest.reshape(-1, 3, 15)
        full = np.zeros((rest.shape[0], 3, 16), dtype=np.float32)
        for channel, dc_name in enumerate(("f_dc_0", "f_dc_1", "f_dc_2")):
            full[:, channel, 0] = np.asarray(vertex[dc_name], dtype=np.float32)
        full[:, :, 1:] = rest
        rotated = np.einsum("ij,ncj->nci", transform, full, optimize=True)
        rotated_rest = rotated[:, :, 1:].reshape(-1, 45)
        for index, name in enumerate(rest_names):
            transformed[name] = rotated_rest[:, index]
        for channel, dc_name in enumerate(("f_dc_0", "f_dc_1", "f_dc_2")):
            transformed[dc_name] = rotated[:, channel, 0]
    return transformed, canonical_xyz


def _deterministic_u01(indices: np.ndarray) -> np.ndarray:
    values = np.asarray(indices, dtype=np.uint64)
    values ^= values >> np.uint64(30)
    values *= np.uint64(0xBF58476D1CE4E5B9)
    values ^= values >> np.uint64(27)
    values *= np.uint64(0x94D049BB133111EB)
    values ^= values >> np.uint64(31)
    return (values.astype(np.float64) / float(2**64)).astype(np.float64)


def _make_tabletop_patch(vertex: np.ndarray, xyz: np.ndarray, calibration: dict) -> tuple[np.ndarray, np.ndarray]:
    spec = calibration["tabletop_patch"]
    radial = np.linalg.norm(xyz[:, :2], axis=1)
    ring_mask = (
        (radial >= float(spec["source_ring_inner_radius_m"]))
        & (radial <= float(spec["source_ring_outer_radius_m"]))
        & (xyz[:, 2] >= float(spec["surface_z_min_m"]))
        & (xyz[:, 2] <= float(spec["surface_z_max_m"]))
    )
    ring_indices = np.flatnonzero(ring_mask)
    if ring_indices.size == 0:
        raise GardenAssetError(
            "Garden tabletop calibration selected no source-ring Gaussians; calibration must be updated."
        )
    ring_vertex = vertex[ring_indices]
    ring_xyz = xyz[ring_indices]
    rin = float(spec["source_ring_inner_radius_m"])
    rout = float(spec["source_ring_outer_radius_m"])
    target_radius = float(spec["target_radius_m"])
    angular_copies = max(1, int(spec.get("angular_copies", 1)))
    angle = np.arctan2(ring_xyz[:, 1], ring_xyz[:, 0])
    normalized_radius = np.clip((np.linalg.norm(ring_xyz[:, :2], axis=1) - rin) / max(rout - rin, 1.0e-6), 0.0, 1.0)
    placement = str(spec.get("placement", "ring_remap")).strip().lower()
    dc_names = [name for name in ("f_dc_0", "f_dc_1", "f_dc_2") if name in ring_vertex.dtype.names]
    dc_median = {
        name: float(np.median(np.asarray(ring_vertex[name], dtype=np.float32)))
        for name in dc_names
    }
    total_patch_count = int(ring_vertex.shape[0]) * angular_copies
    patch_vertices = []
    patch_xyzs = []
    for copy_index in range(angular_copies):
        patched = ring_vertex.copy()
        if placement == "golden_disk":
            indices = (
                np.arange(ring_vertex.shape[0], dtype=np.float64)
                + copy_index * ring_vertex.shape[0]
            )
            new_radius = target_radius * np.sqrt((indices + 0.5) / total_patch_count)
            copy_angle = indices * (math.pi * (3.0 - math.sqrt(5.0)))
        elif placement == "ring_remap":
            copy_angle = angle + copy_index * (2.0 * math.pi / angular_copies)
            # Square-root radial distribution keeps approximate point density
            # uniform over the filled disk rather than concentrating at its center.
            new_radius = target_radius * np.sqrt(normalized_radius)
        else:
            raise GardenAssetError(f"Unsupported Garden tabletop patch placement: {placement!r}")
        new_xyz = ring_xyz.copy()
        new_xyz[:, 0] = new_radius * np.cos(copy_angle)
        new_xyz[:, 1] = new_radius * np.sin(copy_angle)
        if "surface_z_m" in spec:
            new_xyz[:, 2] = float(spec["surface_z_m"])
        for axis_index, axis_name in enumerate(("x", "y", "z")):
            patched[axis_name] = new_xyz[:, axis_index]
        scale_names = sorted(
            [name for name in patched.dtype.names or () if name.startswith("scale_")],
            key=lambda name: int(name.split("_")[-1]),
        )
        if len(scale_names) == 3:
            tangent_log_scale = math.log(float(spec["tangent_scale_m"]))
            normal_log_scale = math.log(float(spec["normal_scale_m"]))
            patched[scale_names[0]] = tangent_log_scale
            patched[scale_names[1]] = tangent_log_scale
            patched[scale_names[2]] = normal_log_scale
        rotation_names = sorted(
            [name for name in patched.dtype.names or () if name.startswith("rot_")],
            key=lambda name: int(name.split("_")[-1]),
        )
        if len(rotation_names) == 4:
            for rotation_index, name in enumerate(rotation_names):
                patched[name] = 1.0 if rotation_index == 0 else 0.0
        if "opacity_logit" in spec and "opacity" in (patched.dtype.names or ()):
            patched["opacity"] = float(spec["opacity_logit"])
        variation_scale = float(spec.get("dc_variation_scale", 1.0))
        variation_clip = float(spec.get("dc_variation_clip", math.inf))
        wood_spec = spec.get("procedural_wood")
        if wood_spec and len(dc_names) == 3:
            base_rgb_mode = str(
                wood_spec.get("base_rgb_mode", "constant")
            ).strip().lower()
            if base_rgb_mode == "source_ring_median":
                source_dc = np.stack(
                    [
                        np.asarray(ring_vertex[name], dtype=np.float32)
                        for name in dc_names
                    ],
                    axis=1,
                )
                source_rgb = np.clip(
                    source_dc * 0.28209479177387814 + 0.5,
                    0.02,
                    0.98,
                )
                base_rgb = np.median(source_rgb, axis=0, keepdims=True)
                base_rgb *= float(wood_spec.get("base_rgb_scale", 1.0))
            elif base_rgb_mode == "constant":
                base_rgb = np.asarray(
                    wood_spec["base_rgb"], dtype=np.float32
                ).reshape(1, 3)
            else:
                raise GardenAssetError(
                    "Unsupported procedural wood base_rgb_mode: "
                    f"{base_rgb_mode!r}."
                )
            base_rgb = np.clip(base_rgb, 0.02, 0.98)
            plank_count = max(3, int(wood_spec.get("plank_count", 28)))
            plank_period = 2.0 * math.pi / plank_count
            wrapped = np.mod(copy_angle + 0.5 * plank_period, plank_period) - 0.5 * plank_period
            seam_distance = np.abs(wrapped) * np.maximum(new_radius, 0.06)
            seam = seam_distance <= 0.5 * float(wood_spec.get("seam_width_m", 0.006))
            plank_index = np.floor((copy_angle + math.pi) / plank_period)
            plank_variation = float(wood_spec.get("plank_variation", 0.035))
            grain_variation = float(wood_spec.get("grain_variation", 0.018))
            grain_frequency = float(wood_spec.get("grain_frequency_per_m", 82.0))
            brightness = (
                1.0
                + plank_variation * np.sin(plank_index * 2.3999632297)
                + grain_variation * np.sin(new_radius * grain_frequency + plank_index * 0.73)
            )
            rgb = np.clip(base_rgb * brightness.reshape(-1, 1), 0.02, 0.98)
            rgb[seam] *= float(wood_spec.get("seam_darkening", 0.42))
            dc = (rgb - 0.5) / 0.28209479177387814
            for channel, name in enumerate(dc_names):
                patched[name] = dc[:, channel]
        else:
            for name in dc_names:
                source_value = np.asarray(ring_vertex[name], dtype=np.float32)
                delta = np.clip(
                    source_value - dc_median[name],
                    -variation_clip,
                    variation_clip,
                )
                patched[name] = dc_median[name] + variation_scale * delta
        if bool(spec.get("zero_directional_sh", True)):
            for name in patched.dtype.names or ():
                if name.startswith("f_rest_"):
                    patched[name] = 0.0
        patch_vertices.append(patched)
        patch_xyzs.append(new_xyz)
    return np.concatenate(patch_vertices), np.concatenate(patch_xyzs)


def _apply_tabletop_opacity_support(
    vertex: np.ndarray,
    xyz: np.ndarray,
    calibration: dict,
    *,
    eligible_mask: np.ndarray | None = None,
) -> int:
    """Raise low-alpha tabletop splats without adding renderable points."""

    spec = calibration.get("tabletop_opacity_support")
    if not spec:
        return 0
    if "opacity" not in (vertex.dtype.names or ()):
        raise GardenAssetError(
            "Garden tabletop opacity support requires an opacity PLY field."
        )
    xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    if xyz.shape[0] != vertex.shape[0]:
        raise GardenAssetError("Garden tabletop opacity support shape mismatch.")
    radius = np.linalg.norm(xyz[:, :2], axis=1)
    selected = (
        (radius <= float(spec["radius_m"]))
        & (xyz[:, 2] >= float(spec["z_min_m"]))
        & (xyz[:, 2] <= float(spec["z_max_m"]))
    )
    if eligible_mask is not None:
        eligible = np.asarray(eligible_mask, dtype=bool).reshape(-1)
        if eligible.shape[0] != vertex.shape[0]:
            raise GardenAssetError(
                "Garden tabletop opacity eligibility mask shape mismatch."
            )
        selected &= eligible
    floor = float(spec["opacity_floor_logit"])
    if not np.isfinite(floor):
        raise GardenAssetError("Garden tabletop opacity floor must be finite.")
    opacity = vertex["opacity"]
    changed = selected & (np.asarray(opacity, dtype=np.float32) < floor)
    opacity[changed] = floor
    return int(np.count_nonzero(changed))


def centerpiece_removal_mask(
    xyz: np.ndarray,
    calibration: dict,
    vertex: np.ndarray | None = None,
) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    spec = calibration["centerpiece_removal"]
    volumes = spec.get("volumes")
    if volumes is None:
        volumes = [spec]
    radial = np.linalg.norm(xyz[:, :2], axis=1)
    gaussian_extent = np.zeros(xyz.shape[0], dtype=np.float32)
    if vertex is not None:
        scale_names = [
            name for name in vertex.dtype.names or () if name.startswith("scale_")
        ]
        if scale_names:
            gaussian_extent = np.maximum.reduce(
                [np.exp(np.asarray(vertex[name], dtype=np.float32)) for name in scale_names]
            )
    selected = np.zeros(xyz.shape[0], dtype=bool)
    for volume in volumes:
        shape = str(volume.get("shape", "cylinder")).strip().lower()
        if shape != "cylinder":
            raise GardenAssetError(
                f"Unsupported Garden centerpiece removal shape: {shape!r}."
            )
        extent_sigma = float(volume.get("gaussian_extent_sigma", 0.0))
        effective_radius = radial - extent_sigma * gaussian_extent
        max_center_radius = float(volume.get("max_center_radius_m", math.inf))
        selected |= (
            (effective_radius <= float(volume["radius_m"]))
            & (radial <= max_center_radius)
            & (xyz[:, 2] >= float(volume["z_min_m"]))
            & (xyz[:, 2] <= float(volume["z_max_m"]))
        )
    return selected


def interaction_roi_mask(
    vertex: np.ndarray,
    xyz: np.ndarray,
    calibration: dict,
) -> np.ndarray:
    """Protect Gaussians whose calibrated support intersects the table ROI."""

    xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    roi = calibration["interaction_roi"]
    scale_names = [
        name for name in vertex.dtype.names or () if name.startswith("scale_")
    ]
    extent = np.zeros(xyz.shape[0], dtype=np.float32)
    if scale_names:
        extent = np.maximum.reduce(
            [np.exp(np.asarray(vertex[name], dtype=np.float32)) for name in scale_names]
        )
    support_radius = float(roi.get("gaussian_extent_sigma", 0.0)) * extent
    radial = np.linalg.norm(xyz[:, :2], axis=1)
    selected = (
        (radial - support_radius <= float(roi["radius_m"]))
        & (xyz[:, 2] + support_radius >= float(roi["z_min_m"]))
        & (xyz[:, 2] - support_radius <= float(roi["z_max_m"]))
    )
    max_center_radius = float(roi.get("max_center_radius_m", math.inf))
    if np.isfinite(max_center_radius):
        selected &= radial <= max_center_radius
    return selected


def gaussian_importance(vertex: np.ndarray, mode: str) -> np.ndarray:
    """Match GS-Playground top-k scores for a standard 3DGS PLY payload."""

    names = vertex.dtype.names or ()
    if "opacity" not in names:
        raise GardenAssetError("Garden pruning requires an opacity PLY field.")
    opacity_logit = np.asarray(vertex["opacity"], dtype=np.float32)
    opacity = 1.0 / (1.0 + np.exp(-np.clip(opacity_logit, -20.0, 20.0)))
    normalized_mode = str(mode).strip().lower()
    if normalized_mode == "opacity":
        return opacity

    scale_names = sorted(
        [name for name in names if name.startswith("scale_")],
        key=lambda name: int(name.split("_")[-1]),
    )
    if not scale_names:
        raise GardenAssetError(
            f"Garden {normalized_mode} pruning requires scale PLY fields."
        )
    log_scales = np.stack(
        [np.asarray(vertex[name], dtype=np.float32) for name in scale_names],
        axis=1,
    )
    scales = np.exp(np.clip(log_scales, -20.0, 6.0))
    if normalized_mode == "opacity_area":
        sorted_scales = np.sort(np.maximum(scales, 1.0e-8), axis=1)
        if sorted_scales.shape[1] == 1:
            area = sorted_scales[:, 0] * sorted_scales[:, 0]
        else:
            area = sorted_scales[:, -1] * sorted_scales[:, -2]
        return opacity * area
    if normalized_mode == "opacity_volume":
        return opacity * np.prod(np.maximum(scales, 1.0e-8), axis=1)
    raise GardenAssetError(
        "Unsupported Garden pruning mode "
        f"{mode!r}; expected opacity, opacity_area, or opacity_volume."
    )


def _prune_exterior_by_importance(
    vertex: np.ndarray,
    xyz: np.ndarray,
    roi_mask: np.ndarray,
    retention: float,
    *,
    mode: str = "opacity_area",
) -> np.ndarray:
    """Keep the protected ROI plus the highest-scoring exterior Gaussians."""

    if retention >= 1.0:
        return np.ones(vertex.shape[0], dtype=bool)
    exterior = np.flatnonzero(~roi_mask)
    keep = roi_mask.copy()
    if exterior.size == 0:
        return keep
    target = max(1, int(round(float(retention) * exterior.size)))
    target = min(target, exterior.size)
    score = np.asarray(gaussian_importance(vertex[exterior], mode), dtype=np.float64)
    score = np.where(np.isfinite(score), score, -np.inf)
    partition = np.argpartition(score, -target)[-target:]
    cutoff = float(np.min(score[partition]))
    above = np.flatnonzero(score > cutoff)
    tie_count = target - int(above.size)
    ties = np.flatnonzero(score == cutoff)
    # Original PLY order is the deterministic tie-breaker.
    selected_local = np.concatenate([above, ties[:tie_count]])
    keep[exterior[selected_local]] = True
    return keep


def normalize_spatial_chunk_config(config: dict) -> dict:
    """Validate and canonicalize the committed Garden chunking parameters."""

    if not isinstance(config, dict):
        raise GardenAssetError("Garden manifest spatial_chunks must be an object.")
    normalized = {
        "cell_size_m": float(config.get("cell_size_m", 0.0)),
        "gaussian_extent_sigma": float(
            config.get("gaussian_extent_sigma", 0.0)
        ),
        "frustum_padding_m": float(config.get("frustum_padding_m", 0.0)),
        "prefetch_margin_ratio": float(
            config.get("prefetch_margin_ratio", 0.0)
        ),
        "near_plane_m": float(config.get("near_plane_m", 0.0)),
    }
    if not all(np.isfinite(value) for value in normalized.values()):
        raise GardenAssetError("Garden spatial chunk parameters must be finite.")
    if normalized["cell_size_m"] <= 0.0:
        raise GardenAssetError("Garden spatial chunk cell_size_m must be positive.")
    if normalized["gaussian_extent_sigma"] < 3.0:
        raise GardenAssetError(
            "Garden spatial chunks require gaussian_extent_sigma >= 3 for conservative culling."
        )
    if normalized["frustum_padding_m"] < 0.0:
        raise GardenAssetError(
            "Garden spatial chunk frustum_padding_m must be non-negative."
        )
    if normalized["prefetch_margin_ratio"] < 0.0:
        raise GardenAssetError(
            "Garden spatial chunk prefetch_margin_ratio must be non-negative."
        )
    if normalized["near_plane_m"] <= 0.0:
        raise GardenAssetError(
            "Garden spatial chunk near_plane_m must be positive."
        )
    return normalized


def partition_gaussians_into_spatial_chunks(
    vertex: np.ndarray,
    xyz: np.ndarray,
    config: dict,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Reorder a Garden PLY into contiguous conservative spatial chunks.

    Chunk membership is based on Gaussian centers. Each recorded bound and
    sphere includes ``gaussian_extent_sigma`` times the largest principal
    Gaussian scale, so runtime frustum tests may accept extra work but cannot
    discard a splat merely because its center lies outside the image.
    """

    normalized = normalize_spatial_chunk_config(config)
    xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    if int(vertex.shape[0]) != int(xyz.shape[0]) or int(vertex.shape[0]) <= 0:
        raise GardenAssetError(
            "Garden spatial chunking requires matching non-empty vertex and xyz arrays."
        )
    cell_size = float(normalized["cell_size_m"])
    cell_keys = np.floor(xyz / cell_size).astype(np.int32)
    order = np.lexsort((cell_keys[:, 2], cell_keys[:, 1], cell_keys[:, 0]))
    ordered_vertex = vertex[order]
    ordered_xyz = xyz[order]
    ordered_keys = cell_keys[order]

    scale_names = sorted(
        [name for name in vertex.dtype.names or () if name.startswith("scale_")],
        key=lambda name: int(name.split("_")[-1]),
    )
    if not scale_names:
        raise GardenAssetError("Garden spatial chunking requires Gaussian scales.")
    max_scale = np.maximum.reduce(
        [
            np.exp(np.clip(np.asarray(vertex[name], dtype=np.float32), -20.0, 6.0))
            for name in scale_names
        ]
    )
    ordered_extent = (
        max_scale[order] * float(normalized["gaussian_extent_sigma"])
    ).astype(np.float32)

    key_change = np.any(ordered_keys[1:] != ordered_keys[:-1], axis=1)
    starts = np.concatenate(
        [np.array([0], dtype=np.int64), np.flatnonzero(key_change).astype(np.int64) + 1]
    )
    stops = np.concatenate(
        [starts[1:], np.array([ordered_xyz.shape[0]], dtype=np.int64)]
    )
    chunks: list[dict] = []
    for start_value, stop_value in zip(starts, stops):
        start = int(start_value)
        stop = int(stop_value)
        chunk_xyz = ordered_xyz[start:stop]
        chunk_extent = ordered_extent[start:stop]
        bounds_min = np.min(chunk_xyz - chunk_extent[:, None], axis=0)
        bounds_max = np.max(chunk_xyz + chunk_extent[:, None], axis=0)
        sphere_center = 0.5 * (bounds_min + bounds_max)
        sphere_radius = float(
            np.max(
                np.linalg.norm(chunk_xyz - sphere_center[None, :], axis=1)
                + chunk_extent
            )
        )
        chunks.append(
            {
                "cell": [int(value) for value in ordered_keys[start]],
                "start": start,
                "count": stop - start,
                "bounds_min": [float(value) for value in bounds_min],
                "bounds_max": [float(value) for value in bounds_max],
                "sphere_center": [float(value) for value in sphere_center],
                "sphere_radius": sphere_radius,
            }
        )
    return ordered_vertex, ordered_xyz, chunks


def validate_spatial_chunk_metadata(metadata: dict, manifest: dict) -> None:
    expected_config = normalize_spatial_chunk_config(
        manifest.get("spatial_chunks")
    )
    actual_config = metadata.get("spatial_chunk_config")
    if actual_config != expected_config:
        raise GardenAssetError(
            "Garden spatial chunk metadata is stale. Rerun the Garden fetch command."
        )
    chunks = metadata.get("spatial_chunks")
    if not isinstance(chunks, list) or not chunks:
        raise GardenAssetError("Garden runtime metadata contains no spatial chunks.")
    expected_start = 0
    for chunk_index, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            raise GardenAssetError(
                f"Garden spatial chunk {chunk_index} is not an object."
            )
        start = int(chunk.get("start", -1))
        count = int(chunk.get("count", 0))
        center = np.asarray(chunk.get("sphere_center"), dtype=np.float64)
        radius = float(chunk.get("sphere_radius", -1.0))
        if start != expected_start or count <= 0:
            raise GardenAssetError(
                f"Garden spatial chunk {chunk_index} has a non-contiguous range."
            )
        if center.shape != (3,) or not np.isfinite(center).all():
            raise GardenAssetError(
                f"Garden spatial chunk {chunk_index} has an invalid sphere center."
            )
        if not np.isfinite(radius) or radius < 0.0:
            raise GardenAssetError(
                f"Garden spatial chunk {chunk_index} has an invalid sphere radius."
            )
        expected_start += count
    if expected_start != int(metadata.get("gaussian_count", -1)):
        raise GardenAssetError(
            "Garden spatial chunks do not cover the complete runtime Gaussian payload."
        )


def prepare_garden_assets(repo_root: str | Path) -> dict[str, dict]:
    from plyfile import PlyData, PlyElement

    repo_root = Path(repo_root).resolve()
    _, manifest = load_garden_manifest(repo_root)
    source_path = validate_garden_source(repo_root)
    calibration_path = resolve_repo_path(repo_root, manifest["calibration"])
    calibration = _load_json(calibration_path)
    source_sha = str(manifest["source"]["point_cloud_sha256"])
    calibration_sha = sha256_file(calibration_path)
    spatial_chunk_config = normalize_spatial_chunk_config(
        manifest.get("spatial_chunks")
    )

    source_ply = PlyData.read(str(source_path))
    if not source_ply.elements or source_ply.elements[0].name != "vertex":
        raise GardenAssetError(f"Garden source has no vertex element: {source_path}")
    source_vertex = source_ply.elements[0].data
    transformed_vertex, transformed_xyz = _transform_vertex_payload(source_vertex, calibration)

    removal_mask = centerpiece_removal_mask(
        transformed_xyz,
        calibration,
        vertex=transformed_vertex,
    )
    patch_vertex, patch_xyz = _make_tabletop_patch(
        transformed_vertex,
        transformed_xyz,
        calibration,
    )
    tabletop_opacity_raised_count = _apply_tabletop_opacity_support(
        transformed_vertex,
        transformed_xyz,
        calibration,
        eligible_mask=~removal_mask,
    )
    cleaned_vertex = np.concatenate([transformed_vertex[~removal_mask], patch_vertex])
    cleaned_xyz = np.concatenate([transformed_xyz[~removal_mask], patch_xyz])

    roi_mask = interaction_roi_mask(cleaned_vertex, cleaned_xyz, calibration)

    outputs: dict[str, dict] = {}
    for quality in manifest["quality_order"]:
        spec = manifest["qualities"][quality]
        retention = float(spec["retention"])
        pruning_mode = str(spec.get("pruning_mode", "opacity_area"))
        keep_mask = _prune_exterior_by_importance(
            cleaned_vertex,
            cleaned_xyz,
            roi_mask,
            retention,
            mode=pruning_mode,
        )
        output_vertex = cleaned_vertex[keep_mask]
        output_xyz = cleaned_xyz[keep_mask]
        output_vertex, output_xyz, spatial_chunks = (
            partition_gaussians_into_spatial_chunks(
                output_vertex,
                output_xyz,
                spatial_chunk_config,
            )
        )
        output_path, metadata_path = garden_quality_paths(repo_root, quality, manifest)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_ply = output_path.with_suffix(output_path.suffix + ".tmp")
        PlyData(
            [PlyElement.describe(output_vertex, "vertex")],
            text=False,
            byte_order="<",
        ).write(str(temporary_ply))
        os.replace(temporary_ply, output_path)
        metadata = {
            "pipeline_version": GARDEN_PIPELINE_VERSION,
            "quality": quality,
            "retention": retention,
            "pruning_mode": pruning_mode,
            "source_sha256": source_sha,
            "calibration_sha256": calibration_sha,
            "source_gaussian_count": int(source_vertex.shape[0]),
            "removed_gaussian_count": int(np.count_nonzero(removal_mask)),
            "patch_gaussian_count": int(patch_vertex.shape[0]),
            "tabletop_opacity_raised_count": tabletop_opacity_raised_count,
            "cleaned_gaussian_count": int(cleaned_vertex.shape[0]),
            "interaction_roi_gaussian_count": int(np.count_nonzero(roi_mask)),
            "exterior_gaussian_count": int(np.count_nonzero(~roi_mask)),
            "exterior_kept_gaussian_count": int(np.count_nonzero(keep_mask & ~roi_mask)),
            "gaussian_count": int(output_vertex.shape[0]),
            "spatial_chunk_config": spatial_chunk_config,
            "spatial_chunk_count": int(len(spatial_chunks)),
            "spatial_chunks": spatial_chunks,
            "ply_sha256": sha256_file(output_path),
        }
        temporary_metadata = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
        with temporary_metadata.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_metadata, metadata_path)
        outputs[quality] = metadata
    return outputs


def fetch_and_prepare_garden(
    repo_root: str | Path,
    *,
    archive_override: str | Path | None = None,
    progress_callback: Callable[[int, int | None], None] | None = None,
) -> dict[str, dict]:
    _, manifest = load_garden_manifest(repo_root)
    cached: dict[str, dict] = {}
    try:
        for quality in manifest["quality_order"]:
            _, metadata = validate_garden_quality(
                repo_root,
                quality,
                verify_payload_hash=True,
            )
            cached[str(quality)] = metadata
    except GardenAssetError:
        cached.clear()
    if cached:
        return cached

    fetch_garden_source(
        repo_root,
        archive_override=archive_override,
        progress_callback=progress_callback,
    )
    return prepare_garden_assets(repo_root)


def validate_garden_runtime_selection(
    repo_root: str | Path,
    requested_quality: str,
) -> list[Path]:
    requested = normalize_garden_quality(requested_quality)
    _, manifest = load_garden_manifest(repo_root)
    manifest_path = resolve_repo_path(repo_root, GARDEN_MANIFEST_RELATIVE_PATH)
    calibration_path = resolve_repo_path(repo_root, manifest["calibration"])
    collision_path = resolve_repo_path(repo_root, manifest["collision_proxy"])
    license_path = resolve_repo_path(repo_root, manifest["license"])
    calibration = _load_json(calibration_path)
    collision = _load_json(collision_path)
    if int(calibration.get("schema_version", -1)) != 1:
        raise GardenAssetError("Unsupported Garden calibration schema.")
    if int(collision.get("schema_version", -1)) != 1:
        raise GardenAssetError("Unsupported Garden collision-proxy schema.")
    rotation = np.asarray(
        calibration.get("source_to_canonical_rotation"), dtype=np.float64
    )
    if (
        rotation.shape != (3, 3)
        or not np.isfinite(rotation).all()
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-4)
        or float(np.linalg.det(rotation)) < 0.999
    ):
        raise GardenAssetError(
            "Garden calibration source_to_canonical_rotation is invalid."
        )
    scale = float(calibration.get("meters_per_source_unit", 0.0))
    if not np.isfinite(scale) or scale <= 0.0:
        raise GardenAssetError("Garden calibration scale must be positive.")
    if not isinstance(collision.get("primitives"), list) or not collision["primitives"]:
        raise GardenAssetError("Garden collision proxy contains no primitives.")
    if not license_path.is_file() or license_path.stat().st_size <= 0:
        raise GardenAssetError(f"Garden asset license metadata is missing: {license_path}")
    paths = [manifest_path, calibration_path, collision_path, license_path]
    if requested == "auto":
        available = available_garden_qualities(repo_root)
        if not available:
            validate_garden_quality(repo_root, "balanced")
        for quality in available:
            ply_path, metadata_path = garden_quality_paths(repo_root, quality, manifest)
            paths.extend([ply_path, metadata_path])
    else:
        ply_path, _ = validate_garden_quality(repo_root, requested)
        _, metadata_path = garden_quality_paths(repo_root, requested, manifest)
        paths.extend([ply_path, metadata_path])
    return paths
