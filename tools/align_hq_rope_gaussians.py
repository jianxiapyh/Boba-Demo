#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import pickle
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

SUPPORT_BAND_QUANTILE = 0.995
SUPPORT_PLANE_TOLERANCE = 0.003
SPAN_ALIGNMENT_TOLERANCE = 1e-4


def _sigmoid(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _normalize(vector: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= eps:
        raise ValueError(f"Cannot normalize near-zero vector: {vector.tolist()}")
    return vector / norm


def _principal_axis(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(points.shape[0], 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    return _normalize(axis)


def _endpoints_along_axis(points: np.ndarray, axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    projections = points @ axis
    return points[int(np.argmin(projections))], points[int(np.argmax(projections))]


def _rotation_matrix_from_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = _normalize(source)
    target = _normalize(target)
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if dot >= 1.0 - 1e-8:
        return np.eye(3, dtype=np.float64)
    if dot <= -1.0 + 1e-8:
        fallback = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(source, fallback))) > 0.9:
            fallback = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = _normalize(np.cross(source, fallback))
        x, y, z = axis.tolist()
        return np.array(
            [
                [-1.0 + 2.0 * x * x, 2.0 * x * y, 2.0 * x * z],
                [2.0 * x * y, -1.0 + 2.0 * y * y, 2.0 * y * z],
                [2.0 * x * z, 2.0 * y * z, -1.0 + 2.0 * z * z],
            ],
            dtype=np.float64,
        )

    cross = np.cross(source, target)
    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + skew + skew @ skew * (1.0 / (1.0 + dot))


def _rotation_matrix_to_quaternion_wxyz(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.array(
            [
                0.25 * s,
                (rotation[2, 1] - rotation[1, 2]) / s,
                (rotation[0, 2] - rotation[2, 0]) / s,
                (rotation[1, 0] - rotation[0, 1]) / s,
            ],
            dtype=np.float64,
        )
    else:
        diagonal = np.diag(rotation)
        idx = int(np.argmax(diagonal))
        if idx == 0:
            s = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            quaternion = np.array(
                [
                    (rotation[2, 1] - rotation[1, 2]) / s,
                    0.25 * s,
                    (rotation[0, 1] + rotation[1, 0]) / s,
                    (rotation[0, 2] + rotation[2, 0]) / s,
                ],
                dtype=np.float64,
            )
        elif idx == 1:
            s = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            quaternion = np.array(
                [
                    (rotation[0, 2] - rotation[2, 0]) / s,
                    (rotation[0, 1] + rotation[1, 0]) / s,
                    0.25 * s,
                    (rotation[1, 2] + rotation[2, 1]) / s,
                ],
                dtype=np.float64,
            )
        else:
            s = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            quaternion = np.array(
                [
                    (rotation[1, 0] - rotation[0, 1]) / s,
                    (rotation[0, 2] + rotation[2, 0]) / s,
                    (rotation[1, 2] + rotation[2, 1]) / s,
                    0.25 * s,
                ],
                dtype=np.float64,
            )
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[0] < 0.0:
        quaternion = -quaternion
    return quaternion


def _quaternion_multiply_wxyz(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = [lhs[:, i] for i in range(4)]
    w2, x2, y2, z2 = [rhs[:, i] for i in range(4)]
    result = np.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        axis=1,
    )
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    result = result / np.clip(norms, 1e-12, None)
    negative_w = result[:, :1] < 0.0
    return np.where(negative_w, -result, result)


def _structured_names(vertex_data: np.ndarray, prefix: str) -> list[str]:
    return sorted(
        [name for name in vertex_data.dtype.names if name.startswith(prefix)],
        key=lambda name: int(name.split("_")[-1]),
    )


def _load_target_frame0_points(final_data_path: Path) -> np.ndarray:
    with open(final_data_path, "rb") as handle:
        data = pickle.load(handle)
    points = np.asarray(data["object_points"], dtype=np.float64)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(
            f"Expected object_points with shape (T, N, 3) in {final_data_path}, got {points.shape}."
        )
    return points[0]


def _load_vertex_data(ply_path: Path) -> tuple[PlyData, np.ndarray]:
    ply = PlyData.read(str(ply_path))
    if len(ply.elements) != 1 or ply.elements[0].name != "vertex":
        raise ValueError(f"Expected a single vertex element in {ply_path}.")
    return ply, np.array(ply.elements[0].data, copy=True)


def _extract_xyz(vertex_data: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            np.asarray(vertex_data["x"], dtype=np.float64),
            np.asarray(vertex_data["y"], dtype=np.float64),
            np.asarray(vertex_data["z"], dtype=np.float64),
        ],
        axis=1,
    )


def _extract_rotations(vertex_data: np.ndarray) -> tuple[np.ndarray, list[str]]:
    rotation_names = _structured_names(vertex_data, "rot")
    if len(rotation_names) != 4:
        raise ValueError(f"Expected 4 quaternion fields, found {rotation_names}.")
    quaternions = np.stack(
        [np.asarray(vertex_data[name], dtype=np.float64) for name in rotation_names],
        axis=1,
    )
    return quaternions, rotation_names


def _apply_similarity_transform(
    xyz: np.ndarray,
    source_midpoint: np.ndarray,
    target_midpoint: np.ndarray,
    rotation: np.ndarray,
    scale: float,
) -> np.ndarray:
    centered = xyz - source_midpoint.reshape(1, 3)
    return scale * (centered @ rotation.T) + target_midpoint.reshape(1, 3)


def _fit_similarity(source_points: np.ndarray, target_points: np.ndarray) -> dict:
    source_axis = _principal_axis(source_points)
    target_axis = _principal_axis(target_points)

    source_left, source_right = _endpoints_along_axis(source_points, source_axis)
    target_left, target_right = _endpoints_along_axis(target_points, target_axis)

    if float(np.dot(source_right - source_left, target_right - target_left)) < 0.0:
        source_left, source_right = source_right, source_left

    source_direction = _normalize(source_right - source_left)
    target_direction = _normalize(target_right - target_left)
    source_span = float(np.linalg.norm(source_right - source_left))
    target_span = float(np.linalg.norm(target_right - target_left))
    if source_span <= 1e-8 or target_span <= 1e-8:
        raise ValueError(
            f"Degenerate rope span during similarity fit: source_span={source_span}, target_span={target_span}."
        )

    rotation = _rotation_matrix_from_vectors(source_direction, target_direction)
    scale = target_span / source_span
    source_midpoint = 0.5 * (source_left + source_right)
    target_midpoint = 0.5 * (target_left + target_right)

    return {
        "rotation": rotation,
        "scale": float(scale),
        "source_midpoint": source_midpoint,
        "target_midpoint": target_midpoint,
        "source_endpoints": np.stack([source_left, source_right], axis=0),
        "target_endpoints": np.stack([target_left, target_right], axis=0),
    }


def _compute_alignment_summary(
    transformed_fit_points: np.ndarray,
    target_points: np.ndarray,
    fit: dict,
) -> dict:
    target_axis = _normalize(
        fit["target_endpoints"][1] - fit["target_endpoints"][0]
    )
    centered = transformed_fit_points - fit["target_midpoint"].reshape(1, 3)
    projections = centered @ target_axis
    line_projection = np.outer(projections, target_axis)
    line_residual = np.linalg.norm(centered - line_projection, axis=1)
    target_bounds = np.stack([target_points.min(axis=0), target_points.max(axis=0)], axis=0)
    transformed_bounds = np.stack(
        [transformed_fit_points.min(axis=0), transformed_fit_points.max(axis=0)],
        axis=0,
    )
    return {
        "fit_point_count": int(transformed_fit_points.shape[0]),
        "line_residual_mean": float(line_residual.mean()),
        "line_residual_max": float(line_residual.max()),
        "target_bounds": target_bounds.tolist(),
        "aligned_fit_bounds": transformed_bounds.tolist(),
        "source_span": float(
            np.linalg.norm(fit["source_endpoints"][1] - fit["source_endpoints"][0])
        ),
        "target_span": float(
            np.linalg.norm(fit["target_endpoints"][1] - fit["target_endpoints"][0])
        ),
    }


def _support_band_height(points: np.ndarray, quantile: float = SUPPORT_BAND_QUANTILE) -> float:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected (N, 3) points when computing support band, got {points.shape}.")
    clipped_quantile = float(np.clip(quantile, 0.5, 1.0))
    return float(np.quantile(points[:, 2], clipped_quantile))


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Align an HQ rope Gaussian PLY to the packaged final_data frame-0 rest shape."
    )
    parser.add_argument("--final-data", type=Path, required=True, help="packaged final_data.pkl path")
    parser.add_argument("--input-ply", type=Path, required=True, help="native input Gaussian PLY path")
    parser.add_argument("--output-ply", type=Path, required=True, help="aligned output Gaussian PLY path")
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="optional provenance/debug sidecar with fit metadata",
    )
    parser.add_argument(
        "--fit-opacity-threshold",
        type=float,
        default=0.1,
        help="sigmoid(opacity) threshold used only for the rope similarity fit subset",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    target_points = _load_target_frame0_points(args.final_data)
    ply, vertex_data = _load_vertex_data(args.input_ply)
    xyz = _extract_xyz(vertex_data)
    quaternions, rotation_names = _extract_rotations(vertex_data)
    scale_names = _structured_names(vertex_data, "scale")
    if not scale_names:
        raise ValueError(f"No scale_* fields found in {args.input_ply}.")

    opacity = np.asarray(vertex_data["opacity"], dtype=np.float64)
    fit_mask = _sigmoid(opacity) > float(args.fit_opacity_threshold)
    if int(fit_mask.sum()) < 16:
        fit_mask = np.ones_like(fit_mask, dtype=bool)

    fit = _fit_similarity(xyz[fit_mask], target_points)
    rotation = fit["rotation"]
    rotation_quaternion = _rotation_matrix_to_quaternion_wxyz(rotation).reshape(1, 4)
    transformed_xyz_stage1 = _apply_similarity_transform(
        xyz,
        source_midpoint=fit["source_midpoint"],
        target_midpoint=fit["target_midpoint"],
        rotation=rotation,
        scale=fit["scale"],
    )
    transformed_fit_points_stage1 = transformed_xyz_stage1[fit_mask]
    target_support_height = _support_band_height(target_points)
    stage1_support_height = _support_band_height(transformed_fit_points_stage1)
    residual_z_shift = target_support_height - stage1_support_height
    transformed_xyz = np.array(transformed_xyz_stage1, copy=True)
    transformed_xyz[:, 2] += residual_z_shift
    transformed_fit_points = transformed_xyz[fit_mask]
    final_support_height = _support_band_height(transformed_fit_points)
    transformed_normals = None
    if {"nx", "ny", "nz"}.issubset(vertex_data.dtype.names):
        normals = np.stack(
            [
                np.asarray(vertex_data["nx"], dtype=np.float64),
                np.asarray(vertex_data["ny"], dtype=np.float64),
                np.asarray(vertex_data["nz"], dtype=np.float64),
            ],
            axis=1,
        )
        transformed_normals = normals @ rotation.T

    transformed_quaternions = _quaternion_multiply_wxyz(
        np.repeat(rotation_quaternion, quaternions.shape[0], axis=0),
        quaternions,
    )

    vertex_data["x"] = transformed_xyz[:, 0]
    vertex_data["y"] = transformed_xyz[:, 1]
    vertex_data["z"] = transformed_xyz[:, 2]
    if transformed_normals is not None:
        vertex_data["nx"] = transformed_normals[:, 0]
        vertex_data["ny"] = transformed_normals[:, 1]
        vertex_data["nz"] = transformed_normals[:, 2]
    for idx, name in enumerate(rotation_names):
        vertex_data[name] = transformed_quaternions[:, idx]

    log_scale_delta = math.log(max(fit["scale"], 1e-12))
    for name in scale_names:
        vertex_data[name] = np.asarray(vertex_data[name], dtype=np.float64) + log_scale_delta

    args.output_ply.parent.mkdir(parents=True, exist_ok=True)
    element = PlyElement.describe(vertex_data, "vertex")
    PlyData(
        [element],
        text=ply.text,
        byte_order=ply.byte_order,
        comments=ply.comments,
        obj_info=ply.obj_info,
    ).write(str(args.output_ply))

    summary = {
        "final_data": str(args.final_data.resolve()),
        "input_ply": str(args.input_ply.resolve()),
        "output_ply": str(args.output_ply.resolve()),
        "fit_opacity_threshold": float(args.fit_opacity_threshold),
        "support_band_quantile": SUPPORT_BAND_QUANTILE,
        "support_plane_tolerance": SUPPORT_PLANE_TOLERANCE,
        "span_alignment_tolerance": SPAN_ALIGNMENT_TOLERANCE,
        "input_gaussian_count": int(xyz.shape[0]),
        "fit_gaussian_count": int(fit_mask.sum()),
        "scale_names": scale_names,
        "rotation_names": rotation_names,
        "uniform_scale": float(fit["scale"]),
        "rotation_matrix": rotation.tolist(),
        "rotation_quaternion_wxyz": rotation_quaternion[0].tolist(),
        "source_midpoint": fit["source_midpoint"].tolist(),
        "target_midpoint": fit["target_midpoint"].tolist(),
        "source_endpoints": fit["source_endpoints"].tolist(),
        "target_endpoints": fit["target_endpoints"].tolist(),
        "target_frame0_bounds": np.stack(
            [target_points.min(axis=0), target_points.max(axis=0)],
            axis=0,
        ).tolist(),
        "stage1_aligned_gaussian_bounds": np.stack(
            [transformed_xyz_stage1.min(axis=0), transformed_xyz_stage1.max(axis=0)],
            axis=0,
        ).tolist(),
        "aligned_gaussian_bounds": np.stack(
            [transformed_xyz.min(axis=0), transformed_xyz.max(axis=0)],
            axis=0,
        ).tolist(),
        "target_support_height": float(target_support_height),
        "stage1_support_height": float(stage1_support_height),
        "final_support_height": float(final_support_height),
        "residual_z_shift": float(residual_z_shift),
    }
    summary.update(_compute_alignment_summary(transformed_fit_points, target_points, fit))

    span_error = abs(summary["source_span"] * fit["scale"] - summary["target_span"])
    if span_error > SPAN_ALIGNMENT_TOLERANCE:
        raise ValueError(
            "Aligned rope span regressed beyond tolerance: "
            f"span_error={span_error:.6f} tolerance={SPAN_ALIGNMENT_TOLERANCE:.6f}."
        )

    support_plane_delta = final_support_height - target_support_height
    summary["support_plane_delta"] = float(support_plane_delta)
    if support_plane_delta > SUPPORT_PLANE_TOLERANCE:
        raise ValueError(
            "Aligned rope still sits below the target support plane: "
            f"support_plane_delta={support_plane_delta:.6f} "
            f"tolerance={SUPPORT_PLANE_TOLERANCE:.6f}."
        )

    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.summary_json, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

    print(
        "[hq_rope_align] "
        f"input={args.input_ply} output={args.output_ply} "
        f"fit_gaussians={int(fit_mask.sum())}/{int(xyz.shape[0])} "
        f"uniform_scale={fit['scale']:.6f} "
        f"source_span={summary['source_span']:.6f} "
        f"target_span={summary['target_span']:.6f} "
        f"target_support_height={target_support_height:.6f} "
        f"stage1_support_height={stage1_support_height:.6f} "
        f"final_support_height={final_support_height:.6f} "
        f"residual_z_shift={residual_z_shift:.6f} "
        f"line_residual_mean={summary['line_residual_mean']:.6f} "
        f"line_residual_max={summary['line_residual_max']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
