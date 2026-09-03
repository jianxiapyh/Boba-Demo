"""Decode bundled PlayCanvas SOG v2 Gaussian scenes.

The format is a stored ZIP containing ``meta.json`` and lossless WebP textures.
This module mirrors the SplatTransform v2 reader so Boba can consume compact
SOG assets directly without generating a much larger intermediate PLY.
"""

from __future__ import annotations

import io
import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np


class SogAssetError(RuntimeError):
    """Raised when a bundled SOG scene is missing or malformed."""


@dataclass
class DecodedSog:
    xyz: np.ndarray
    features_dc: np.ndarray
    features_rest: np.ndarray
    opacity: np.ndarray
    scaling: np.ndarray
    rotation: np.ndarray
    sh_degree: int
    metadata: dict[str, Any]


_SH_COEFFICIENTS_BY_BAND = (0, 3, 8, 15)
_QUATERNION_PACKED_COMPONENTS = (
    (1, 2, 3),
    (0, 2, 3),
    (0, 1, 3),
    (0, 1, 2),
)


def _load_metadata(bundle: zipfile.ZipFile, path: Path) -> dict[str, Any]:
    try:
        metadata = json.loads(bundle.read("meta.json"))
    except KeyError as exc:
        raise SogAssetError(f"SOG bundle has no meta.json: {path}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SogAssetError(f"SOG meta.json is invalid: {path}: {exc}") from exc
    if not isinstance(metadata, dict):
        raise SogAssetError(f"SOG meta.json must contain an object: {path}")
    if int(metadata.get("version", -1)) != 2:
        raise SogAssetError(
            f"Unsupported SOG version {metadata.get('version')!r}; Boba requires v2."
        )
    count = int(metadata.get("count", 0))
    if count <= 0:
        raise SogAssetError(f"SOG scene has no Gaussians: {path}")
    return metadata


def read_sog_metadata(path: str | Path) -> dict[str, Any]:
    """Read and validate the small metadata record without decoding textures."""

    sog_path = Path(path).resolve()
    if not sog_path.is_file():
        raise SogAssetError(f"SOG scene is missing: {sog_path}")
    if not zipfile.is_zipfile(sog_path):
        raise SogAssetError(f"SOG scene is not a ZIP bundle: {sog_path}")
    try:
        with zipfile.ZipFile(sog_path, "r") as bundle:
            return _load_metadata(bundle, sog_path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise SogAssetError(f"Unable to read SOG scene {sog_path}: {exc}") from exc


def _metadata_files(metadata: dict[str, Any], field: str, count: int) -> list[str]:
    record = metadata.get(field)
    if not isinstance(record, dict):
        raise SogAssetError(f"SOG metadata is missing the {field!r} record.")
    files = record.get("files")
    if not isinstance(files, list) or len(files) != int(count):
        raise SogAssetError(
            f"SOG {field!r} must reference exactly {int(count)} texture file(s)."
        )
    return [str(value) for value in files]


def _decode_rgba(
    bundle: zipfile.ZipFile,
    filename: str,
    *,
    minimum_pixels: int,
) -> tuple[np.ndarray, tuple[int, int]]:
    from PIL import Image

    try:
        payload = bundle.read(filename)
    except KeyError as exc:
        raise SogAssetError(f"SOG bundle is missing texture {filename!r}.") from exc
    try:
        with Image.open(io.BytesIO(payload)) as image:
            rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8).copy()
            dimensions = tuple(int(value) for value in image.size)
    except (OSError, ValueError) as exc:
        raise SogAssetError(f"Unable to decode SOG texture {filename!r}: {exc}") from exc
    pixels = rgba.reshape(-1, 4)
    if pixels.shape[0] < int(minimum_pixels):
        raise SogAssetError(
            f"SOG texture {filename!r} contains {pixels.shape[0]} pixels; "
            f"expected at least {int(minimum_pixels)}."
        )
    return pixels[: int(minimum_pixels)], dimensions


def _codebook(metadata: dict[str, Any], field: str) -> np.ndarray:
    record = metadata.get(field)
    values = record.get("codebook") if isinstance(record, dict) else None
    if not isinstance(values, list) or not values:
        raise SogAssetError(f"SOG {field!r} has no numeric codebook.")
    result = np.asarray(values, dtype=np.float32).reshape(-1)
    if not np.all(np.isfinite(result)):
        raise SogAssetError(f"SOG {field!r} codebook contains non-finite values.")
    return result


def _lookup_codebook(
    labels: np.ndarray,
    codebook: np.ndarray,
    *,
    field: str,
) -> np.ndarray:
    max_label = int(labels.max(initial=0))
    if max_label >= int(codebook.size):
        raise SogAssetError(
            f"SOG {field!r} label {max_label} exceeds its "
            f"{int(codebook.size)}-entry codebook."
        )
    return codebook[labels]


def _decode_positions(
    low: np.ndarray,
    high: np.ndarray,
    metadata: dict[str, Any],
) -> np.ndarray:
    means = metadata.get("means")
    mins = np.asarray(means.get("mins") if isinstance(means, dict) else None)
    maxs = np.asarray(means.get("maxs") if isinstance(means, dict) else None)
    if mins.shape != (3,) or maxs.shape != (3,):
        raise SogAssetError("SOG means mins/maxs must each contain three values.")
    if not np.all(np.isfinite(mins)) or not np.all(np.isfinite(maxs)):
        raise SogAssetError("SOG means mins/maxs contain non-finite values.")
    packed = low[:, :3].astype(np.uint16)
    packed |= high[:, :3].astype(np.uint16) << np.uint16(8)
    log_positions = mins + (maxs - mins) * (
        packed.astype(np.float64) / 65535.0
    )
    positions = np.sign(log_positions) * np.expm1(np.abs(log_positions))
    return np.ascontiguousarray(positions, dtype=np.float32)


def _decode_quaternions(packed: np.ndarray) -> np.ndarray:
    count = int(packed.shape[0])
    output = np.zeros((count, 4), dtype=np.float32)
    output[:, 0] = 1.0
    tags = packed[:, 3].astype(np.int16) - 252
    decoded = (
        (packed[:, :3].astype(np.float32) / 255.0 * 2.0 - 1.0)
        / math.sqrt(2.0)
    )
    for omitted, component_indices in enumerate(_QUATERNION_PACKED_COMPONENTS):
        rows = np.flatnonzero(tags == omitted)
        if rows.size == 0:
            continue
        output[rows] = 0.0
        output[rows[:, None], np.asarray(component_indices)[None, :]] = decoded[rows]
        squared = np.sum(decoded[rows] * decoded[rows], axis=1)
        output[rows, omitted] = np.sqrt(np.maximum(0.0, 1.0 - squared))
    return np.ascontiguousarray(output, dtype=np.float32)


def _decode_sh_rest(
    bundle: zipfile.ZipFile,
    metadata: dict[str, Any],
    count: int,
) -> tuple[np.ndarray, int]:
    record = metadata.get("shN")
    if not isinstance(record, dict):
        return np.zeros((count, 0, 3), dtype=np.float32), 0
    bands = int(record.get("bands", 0))
    if bands < 0 or bands >= len(_SH_COEFFICIENTS_BY_BAND):
        raise SogAssetError(f"Unsupported SOG spherical-harmonic band count: {bands}")
    coefficient_count = int(_SH_COEFFICIENTS_BY_BAND[bands])
    if coefficient_count == 0:
        return np.zeros((count, 0, 3), dtype=np.float32), 0
    palette_count = int(record.get("count", 0))
    if palette_count <= 0 or palette_count > 65536:
        raise SogAssetError(f"Invalid SOG SH palette size: {palette_count}")
    files = _metadata_files(metadata, "shN", 2)
    centroid_pixels_required = palette_count * coefficient_count
    centroids, centroid_dimensions = _decode_rgba(
        bundle,
        files[0],
        minimum_pixels=centroid_pixels_required,
    )
    expected_width = 64 * coefficient_count
    if int(centroid_dimensions[0]) != expected_width:
        raise SogAssetError(
            f"SOG SH centroid width is {centroid_dimensions[0]}; "
            f"expected {expected_width}."
        )
    labels_rgba, _ = _decode_rgba(
        bundle,
        files[1],
        minimum_pixels=count,
    )
    labels = labels_rgba[:, 0].astype(np.uint16)
    labels |= labels_rgba[:, 1].astype(np.uint16) << np.uint16(8)
    if int(labels.max(initial=0)) >= palette_count:
        raise SogAssetError("SOG SH label exceeds the declared palette size.")

    sh_codebook = _codebook(metadata, "shN")
    centroid_labels = centroids[:centroid_pixels_required, :3].reshape(
        palette_count,
        coefficient_count,
        3,
    )
    palette = _lookup_codebook(
        centroid_labels,
        sh_codebook,
        field="shN",
    )
    features_rest = palette[labels]
    return np.ascontiguousarray(features_rest, dtype=np.float32), bands


def decode_sog(
    path: str | Path,
    *,
    include_sh_rest: bool = True,
    progress_callback: Callable[[str], None] | None = None,
) -> DecodedSog:
    """Decode a bundled SOG v2 scene into Graphdeco Gaussian arrays."""

    sog_path = Path(path).resolve()
    if not sog_path.is_file():
        raise SogAssetError(f"SOG scene is missing: {sog_path}")
    try:
        with zipfile.ZipFile(sog_path, "r") as bundle:
            metadata = _load_metadata(bundle, sog_path)
            count = int(metadata["count"])

            def report(stage: str) -> None:
                if progress_callback is not None:
                    progress_callback(stage)

            report("positions")
            means_files = _metadata_files(metadata, "means", 2)
            means_low, dimensions = _decode_rgba(
                bundle,
                means_files[0],
                minimum_pixels=count,
            )
            means_high, high_dimensions = _decode_rgba(
                bundle,
                means_files[1],
                minimum_pixels=count,
            )
            if high_dimensions != dimensions:
                raise SogAssetError("SOG position textures have different dimensions.")
            xyz = _decode_positions(means_low, means_high, metadata)

            report("geometry")
            quaternion_files = _metadata_files(metadata, "quats", 1)
            quaternion_rgba, _ = _decode_rgba(
                bundle,
                quaternion_files[0],
                minimum_pixels=count,
            )
            rotation = _decode_quaternions(quaternion_rgba)

            scale_files = _metadata_files(metadata, "scales", 1)
            scale_rgba, _ = _decode_rgba(
                bundle,
                scale_files[0],
                minimum_pixels=count,
            )
            scaling = _lookup_codebook(
                scale_rgba[:, :3],
                _codebook(metadata, "scales"),
                field="scales",
            )
            scaling = np.ascontiguousarray(scaling, dtype=np.float32)

            report("appearance")
            sh0_files = _metadata_files(metadata, "sh0", 1)
            sh0_rgba, _ = _decode_rgba(
                bundle,
                sh0_files[0],
                minimum_pixels=count,
            )
            features_dc = _lookup_codebook(
                sh0_rgba[:, :3],
                _codebook(metadata, "sh0"),
                field="sh0",
            )
            features_dc = np.ascontiguousarray(
                features_dc[:, None, :],
                dtype=np.float32,
            )
            opacity_probability = np.clip(
                sh0_rgba[:, 3:4].astype(np.float32) / 255.0,
                1.0e-6,
                1.0 - 1.0e-6,
            )
            opacity = np.log(opacity_probability / (1.0 - opacity_probability))
            opacity = np.ascontiguousarray(opacity, dtype=np.float32)

            if include_sh_rest:
                report("directional_appearance")
                features_rest, sh_degree = _decode_sh_rest(
                    bundle,
                    metadata,
                    count,
                )
            else:
                sh_degree = int(
                    metadata.get("shN", {}).get("bands", 0)
                    if isinstance(metadata.get("shN"), dict)
                    else 0
                )
                coefficient_count = (
                    _SH_COEFFICIENTS_BY_BAND[sh_degree]
                    if 0 <= sh_degree < len(_SH_COEFFICIENTS_BY_BAND)
                    else 0
                )
                features_rest = np.zeros(
                    (count, coefficient_count, 3),
                    dtype=np.float32,
                )
            report("complete")
    except (OSError, zipfile.BadZipFile) as exc:
        raise SogAssetError(f"Unable to read SOG scene {sog_path}: {exc}") from exc

    return DecodedSog(
        xyz=xyz,
        features_dc=features_dc,
        features_rest=features_rest,
        opacity=opacity,
        scaling=scaling,
        rotation=rotation,
        sh_degree=sh_degree,
        metadata=metadata,
    )


def load_sog_gaussian_model(
    path: str | Path,
    *,
    device,
    progress_callback: Callable[[str], None] | None = None,
):
    """Decode a SOG archive and upload it as a runtime ``GaussianModel``."""

    import torch

    from gaussian_splatting.scene.gaussian_model import GaussianModel

    decoded = decode_sog(path, progress_callback=progress_callback)
    model = GaussianModel(sh_degree=decoded.sh_degree)
    model.active_sh_degree = decoded.sh_degree
    model.isotropic = False
    model._xyz = torch.as_tensor(decoded.xyz, device=device).contiguous()
    model._features_dc = torch.as_tensor(
        decoded.features_dc,
        device=device,
    ).contiguous()
    model._features_rest = torch.as_tensor(
        decoded.features_rest,
        device=device,
    ).contiguous()
    model._opacity = torch.as_tensor(decoded.opacity, device=device).contiguous()
    model._scaling = torch.as_tensor(decoded.scaling, device=device).contiguous()
    model._rotation = torch.as_tensor(decoded.rotation, device=device).contiguous()
    return model
