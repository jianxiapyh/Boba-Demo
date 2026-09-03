"""Runtime scene adapter for the Gaussian Mip-NeRF 360 Garden."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from .garden_assets import (
    GardenAssetError,
    garden_profile_key,
    load_garden_manifest,
    record_garden_profile,
    resolve_garden_quality,
    resolve_repo_path,
    sha256_file,
    sh_rotation_matrix,
    validate_garden_quality,
)
from .immersive_scene import SimpleLabLayout


GARDEN_SCENE_NAME = "garden"
GARDEN_TABLE_FORWARD_M = 0.78
GARDEN_TABLE_DOWN_M = 0.62
GARDEN_PROFILE_SAMPLE_COUNT = 120
GARDEN_PROFILE_TARGET_FPS = 72.0


def _normalize(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32).reshape(3)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm <= 1.0e-6:
        value = np.asarray(fallback, dtype=np.float32).reshape(3)
        norm = float(np.linalg.norm(value))
    return (value / max(norm, 1.0e-6)).astype(np.float32)


def gaussian_chunk_spheres_in_camera_frustum(
    sphere_centers_world: np.ndarray,
    sphere_radii: np.ndarray,
    w2c_cv: np.ndarray,
    intrinsic: np.ndarray,
    width: int,
    height: int,
    *,
    near_plane_m: float = 0.01,
    padding_m: float = 0.0,
    margin_ratio: float = 0.0,
) -> np.ndarray:
    """Conservatively test world-space chunk spheres against one CV frustum."""

    centers = np.asarray(sphere_centers_world, dtype=np.float32).reshape(-1, 3)
    radii = np.asarray(sphere_radii, dtype=np.float32).reshape(-1)
    if centers.shape[0] != radii.shape[0]:
        raise ValueError("Garden chunk center/radius count mismatch.")
    if centers.shape[0] == 0:
        return np.zeros((0,), dtype=bool)
    w2c = np.asarray(w2c_cv, dtype=np.float32).reshape(4, 4)
    camera_centers = centers @ w2c[:3, :3].T + w2c[:3, 3]
    k = np.asarray(intrinsic, dtype=np.float32).reshape(3, 3)
    fx = float(k[0, 0])
    fy = float(k[1, 1])
    cx = float(k[0, 2])
    cy = float(k[1, 2])
    if fx <= 0.0 or fy <= 0.0 or int(width) <= 0 or int(height) <= 0:
        raise ValueError("Garden chunk frustum requires valid camera intrinsics.")
    margin = max(0.0, float(margin_ratio))
    u_min = -margin * float(width)
    u_max = (1.0 + margin) * float(width)
    v_min = -margin * float(height)
    v_max = (1.0 + margin) * float(height)
    planes = np.array(
        [
            [fx, 0.0, cx - u_min],
            [-fx, 0.0, u_max - cx],
            [0.0, fy, cy - v_min],
            [0.0, -fy, v_max - cy],
        ],
        dtype=np.float32,
    )
    plane_norms = np.linalg.norm(planes, axis=1).clip(min=1.0e-8)
    signed_distances = camera_centers @ planes.T / plane_norms[None, :]
    effective_radii = np.maximum(radii, 0.0) + max(0.0, float(padding_m))
    side_visible = np.all(
        signed_distances >= -effective_radii[:, None],
        axis=1,
    )
    near_visible = (
        camera_centers[:, 2] + effective_radii
        >= max(float(near_plane_m), 1.0e-6)
    )
    return side_visible & near_visible


def _box_mesh(center: np.ndarray, size: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.asarray(center, dtype=np.float32)
    half = np.asarray(size, dtype=np.float32) * 0.5
    mins = center - half
    maxs = center + half
    vertices = np.array(
        [
            [mins[0], mins[1], mins[2]],
            [maxs[0], mins[1], mins[2]],
            [maxs[0], maxs[1], mins[2]],
            [mins[0], maxs[1], mins[2]],
            [mins[0], mins[1], maxs[2]],
            [maxs[0], mins[1], maxs[2]],
            [maxs[0], maxs[1], maxs[2]],
            [mins[0], maxs[1], maxs[2]],
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [
            [0, 2, 1], [0, 3, 2],
            [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4],
            [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6],
            [3, 0, 4], [3, 4, 7],
        ],
        dtype=np.int32,
    )
    return vertices, faces


def _cylinder_mesh(
    center: np.ndarray,
    radius: float,
    height: float,
    segments: int,
) -> tuple[np.ndarray, np.ndarray]:
    center = np.asarray(center, dtype=np.float32)
    radius = float(radius)
    height = float(height)
    segments = max(8, int(segments))
    angles = np.arange(segments, dtype=np.float32) * (2.0 * np.pi / segments)
    ring = np.stack([np.cos(angles) * radius, np.sin(angles) * radius], axis=1)
    z_min = float(center[2] - height * 0.5)
    z_max = float(center[2] + height * 0.5)
    bottom = np.column_stack(
        [ring[:, 0] + center[0], ring[:, 1] + center[1], np.full(segments, z_min)]
    )
    top = np.column_stack(
        [ring[:, 0] + center[0], ring[:, 1] + center[1], np.full(segments, z_max)]
    )
    vertices = np.concatenate(
        [
            bottom,
            top,
            np.array([[center[0], center[1], z_min], [center[0], center[1], z_max]]),
        ],
        axis=0,
    ).astype(np.float32)
    center_bottom = 2 * segments
    center_top = center_bottom + 1
    faces = []
    for index in range(segments):
        next_index = (index + 1) % segments
        faces.append([center_bottom, next_index, index])
        faces.append([center_top, segments + index, segments + next_index])
        faces.append([index, next_index, segments + next_index])
        faces.append([index, segments + next_index, segments + index])
    return vertices, np.asarray(faces, dtype=np.int32)


def load_garden_collision_proxy(repo_root: str | Path) -> tuple[dict, dict]:
    repo_root = Path(repo_root).resolve()
    _, manifest = load_garden_manifest(repo_root)
    proxy_path = resolve_repo_path(repo_root, manifest["collision_proxy"])
    try:
        with proxy_path.open("r", encoding="utf-8") as handle:
            proxy = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise GardenAssetError(f"Unable to read Garden collision proxy {proxy_path}: {exc}") from exc
    if int(proxy.get("schema_version", -1)) != 1:
        raise GardenAssetError("Unsupported Garden collision-proxy schema.")
    return manifest, proxy


def build_garden_collision_proxy_canonical(
    proxy: dict,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    vertices_parts = []
    faces_parts = []
    metadata = []
    vertex_offset = 0
    for primitive in proxy.get("primitives", []):
        kind = str(primitive.get("kind", "")).strip().lower()
        if kind == "box":
            vertices, faces = _box_mesh(primitive["center"], primitive["size"])
        elif kind == "cylinder":
            vertices, faces = _cylinder_mesh(
                primitive["center"],
                primitive["radius"],
                primitive["height"],
                primitive.get("segments", 32),
            )
        else:
            raise GardenAssetError(f"Unsupported Garden collision primitive: {kind!r}")
        faces_parts.append(faces + vertex_offset)
        vertices_parts.append(vertices)
        metadata.append(
            {
                "name": str(primitive.get("name", kind)),
                "kind": kind,
                "support": bool(primitive.get("support", False)),
                "vertex_start": int(vertex_offset),
                "vertex_count": int(vertices.shape[0]),
                "face_count": int(faces.shape[0]),
                "canonical_bounds_min": vertices.min(axis=0),
                "canonical_bounds_max": vertices.max(axis=0),
            }
        )
        vertex_offset += int(vertices.shape[0])
    if not vertices_parts:
        raise GardenAssetError("Garden collision proxy contains no primitives.")
    return (
        np.concatenate(vertices_parts, axis=0).astype(np.float32),
        np.concatenate(faces_parts, axis=0).astype(np.int32),
        metadata,
    )


def make_garden_layout(
    head_position: np.ndarray,
    forward_direction: np.ndarray,
    *,
    repo_root: str | Path,
    scene_up: np.ndarray | None = None,
) -> SimpleLabLayout:
    head_position = np.asarray(head_position, dtype=np.float32).reshape(3)
    scene_up = _normalize(
        np.array([0.0, 0.0, -1.0], dtype=np.float32) if scene_up is None else scene_up,
        np.array([0.0, 0.0, -1.0], dtype=np.float32),
    )
    forward = np.asarray(forward_direction, dtype=np.float32).reshape(3)
    forward = forward - float(np.dot(forward, scene_up)) * scene_up
    forward = _normalize(forward, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    right = _normalize(np.cross(scene_up, forward), np.array([1.0, 0.0, 0.0]))
    down = -scene_up
    rotation = np.stack([right, forward, down], axis=1).astype(np.float32)
    if float(np.linalg.det(rotation)) < 0.99:
        raise GardenAssetError("Garden layout basis is not right-handed.")
    table_center = (
        head_position
        + forward * GARDEN_TABLE_FORWARD_M
        + down * GARDEN_TABLE_DOWN_M
    ).astype(np.float32)

    manifest, proxy = load_garden_collision_proxy(repo_root)
    canonical_vertices, faces, metadata = build_garden_collision_proxy_canonical(proxy)
    world_vertices = canonical_vertices @ rotation.T + table_center.reshape(1, 3)
    runtime_surfaces = []
    runtime_boxes = []
    for primitive in proxy.get("primitives", []):
        name = str(primitive.get("name", ""))
        kind = str(primitive.get("kind", "")).strip().lower()
        if name == "round_tabletop" and kind == "cylinder":
            canonical_center = np.asarray(primitive["center"], dtype=np.float32).copy()
            canonical_center[2] -= 0.5 * float(primitive["height"])
            runtime_surfaces.append(
                {
                    "name": name,
                    # Keep the inexpensive analytic top contact, but retain the
                    # physical tabletop depth so hanging objects can also hit
                    # the rim and underside instead of slipping through the
                    # zero-thickness disk at the boundary.
                    "kind": "cylinder",
                    "center": (
                        canonical_center @ rotation.T + table_center
                    ).astype(np.float32),
                    "normal": scene_up.copy(),
                    "axis_u": right.copy(),
                    "axis_v": forward.copy(),
                    "extent_u": float(primitive["radius"]),
                    "extent_v": float(primitive["height"]),
                }
            )
        elif name == "stone_patio" and kind == "box":
            canonical_center = np.asarray(primitive["center"], dtype=np.float32).copy()
            size = np.asarray(primitive["size"], dtype=np.float32)
            canonical_center[2] -= 0.5 * float(size[2])
            runtime_surfaces.append(
                {
                    "name": name,
                    "kind": "rectangle",
                    "center": (
                        canonical_center @ rotation.T + table_center
                    ).astype(np.float32),
                    "normal": scene_up.copy(),
                    "axis_u": right.copy(),
                    "axis_v": forward.copy(),
                    "extent_u": 0.5 * float(size[0]),
                    "extent_v": 0.5 * float(size[1]),
                }
            )
    support_boxes = []
    table_bounds = None
    patio_top_center = None
    for entry in metadata:
        start = int(entry["vertex_start"])
        stop = start + int(entry["vertex_count"])
        primitive_vertices = world_vertices[start:stop]
        entry["world_bounds_min"] = primitive_vertices.min(axis=0).astype(np.float32)
        entry["world_bounds_max"] = primitive_vertices.max(axis=0).astype(np.float32)
        if entry["support"]:
            support_boxes.append([entry["world_bounds_min"], entry["world_bounds_max"]])
        elif entry["kind"] == "box":
            runtime_boxes.append(
                [entry["world_bounds_min"], entry["world_bounds_max"]]
            )
        if entry["name"] == "round_tabletop":
            table_bounds = np.stack(
                [entry["world_bounds_min"], entry["world_bounds_max"]], axis=0
            ).astype(np.float32)
        if entry["name"] == "stone_patio":
            canonical_min = np.asarray(entry["canonical_bounds_min"], dtype=np.float32)
            canonical_max = np.asarray(entry["canonical_bounds_max"], dtype=np.float32)
            canonical_top = np.array(
                [
                    0.5 * (canonical_min[0] + canonical_max[0]),
                    0.5 * (canonical_min[1] + canonical_max[1]),
                    canonical_min[2],
                ],
                dtype=np.float32,
            )
            patio_top_center = canonical_top @ rotation.T + table_center
    if table_bounds is None or patio_top_center is None:
        raise GardenAssetError("Garden proxy must define round_tabletop and stone_patio.")
    if [entry["name"] for entry in runtime_surfaces] != [
        "round_tabletop",
        "stone_patio",
    ]:
        raise GardenAssetError(
            "Garden runtime collision requires round_tabletop and stone_patio surfaces."
        )
    layout = SimpleLabLayout(
        table_top_center=table_center,
        table_size=np.array([1.56, 1.56, 0.065], dtype=np.float32),
        floor_z=float(patio_top_center[2]),
        room_half_extent=np.array([2.3, 1.9], dtype=np.float32),
        wall_height=0.0,
        scene_up=scene_up,
        room_center_xy=np.asarray(table_center[:2], dtype=np.float32),
        static_collider_boxes=None,
        static_collider_box_metadata=None,
        support_surface_boxes=np.asarray(support_boxes, dtype=np.float32),
        active_table_bounds=table_bounds,
        active_table_surface_center=table_center.copy(),
        smooth_tabletop_bounds=table_bounds.copy(),
        smooth_tabletop_patch_count=1,
    )
    # Runtime-only fields are intentionally attached to the shared layout type;
    # Lab code continues to see precisely its existing dataclass fields.
    layout.scene_name = GARDEN_SCENE_NAME
    layout.scene_forward = forward
    layout.scene_right = right
    layout.canonical_to_world_rotation = rotation
    layout.canonical_to_world_translation = table_center
    layout.static_collision_mesh_vertices = world_vertices.astype(np.float32)
    layout.static_collision_mesh_faces = faces.astype(np.int32)
    layout.static_collision_mesh_metadata = metadata
    layout.static_collision_mesh_contact = dict(proxy.get("contact", {}))
    # The closed mesh is retained for developer visualization. Runtime contact
    # uses an analytic finite cylinder and patio plane, avoiding per-particle
    # winding queries while preserving the tabletop rim/underside and patio
    # footprint.
    layout.static_collision_surfaces = runtime_surfaces
    layout.static_collision_boxes = np.asarray(runtime_boxes, dtype=np.float32)
    layout.garden_manifest = manifest
    return layout


def _matrix_to_quaternion_wxyz_torch(rotation, *, torch_module, device, dtype):
    matrix = np.asarray(rotation, dtype=np.float32).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        values = [
            0.25 * scale,
            (matrix[2, 1] - matrix[1, 2]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[1, 0] - matrix[0, 1]) / scale,
        ]
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            values = [(matrix[2, 1] - matrix[1, 2]) / scale, 0.25 * scale, (matrix[0, 1] + matrix[1, 0]) / scale, (matrix[0, 2] + matrix[2, 0]) / scale]
        elif index == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            values = [(matrix[0, 2] - matrix[2, 0]) / scale, (matrix[0, 1] + matrix[1, 0]) / scale, 0.25 * scale, (matrix[1, 2] + matrix[2, 1]) / scale]
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            values = [(matrix[1, 0] - matrix[0, 1]) / scale, (matrix[0, 2] + matrix[2, 0]) / scale, (matrix[1, 2] + matrix[2, 1]) / scale, 0.25 * scale]
    quaternion = torch_module.tensor(values, device=device, dtype=dtype)
    quaternion = quaternion / torch_module.linalg.norm(quaternion).clamp_min(1.0e-8)
    return quaternion


class GardenSceneRenderer:
    """Blank static adapter plus persistent combined Garden/object Gaussian model.

    Garden itself is rasterized together with the deformable object, so the
    conventional static-scene methods intentionally return a depthless dark
    frame.  The runtime can publish the combined result directly while keeping
    the established overlay/publish pipeline and exact Garden/object depth
    ordering from one gsplat call.
    """

    # This capability is intentionally exclusive to the combined Gaussian
    # Garden adapter. Mesh-backed scene renderers must continue through the
    # regular depth/alpha compositor.
    supports_direct_gaussian_output = True
    scene_runtime_name = GARDEN_SCENE_NAME
    scene_display_name = "Garden"

    def __init__(
        self,
        scene_assets_root: str | Path,
        width: int,
        height: int,
        *,
        repo_root: str | Path | None = None,
        garden_quality: str = "balanced",
        eye_resolution: int | None = None,
        **_kwargs,
    ):
        self.repo_root = (
            Path(repo_root).resolve()
            if repo_root is not None
            else Path(__file__).resolve().parents[1]
        )
        self.scene_assets_root = Path(scene_assets_root).resolve()
        self.width = int(width)
        self.height = int(height)
        self.requested_quality = str(garden_quality).strip().lower()
        self.eye_resolution = int(eye_resolution or width)
        self.layout: SimpleLabLayout | None = None
        self._blank_cache = {}
        self._static_gaussians = None
        self._render_storage = None
        self._render_storages = []
        self._combined_gaussians = None
        self._combined_models = []
        self._active_render_slot = 0
        self._chunk_rebuild_stream = None
        self._pending_chunk_rebuild = None
        self._static_count = 0
        self._active_static_count = 0
        self._dynamic_capacity = 0
        self._dynamic_count = 0
        self._dynamic_gaussians = None
        self._chunk_starts = np.zeros((0,), dtype=np.int64)
        self._chunk_counts = np.zeros((0,), dtype=np.int64)
        self._chunk_centers_world = np.zeros((0, 3), dtype=np.float32)
        self._chunk_radii = np.zeros((0,), dtype=np.float32)
        self._active_chunk_ids: tuple[int, ...] = ()
        self._chunk_selection_initialized = False
        self._chunk_rebuild_count = 0
        self._last_chunk_selection_debug: dict[str, Any] = {}
        self._profile_source_frame_seconds: list[float] = []
        self._profile_written = False
        self.manifest_path, self.manifest = load_garden_manifest(self.repo_root)
        self._profile_key = self._make_profile_key()
        self.quality = resolve_garden_quality(
            self.repo_root,
            self.requested_quality,
            profile_key=self._profile_key,
            target_fps=GARDEN_PROFILE_TARGET_FPS,
        )
        self.runtime_ply_path, self.runtime_metadata = validate_garden_quality(
            self.repo_root,
            self.quality,
            verify_payload_hash=False,
        )

    def _make_profile_key(self) -> str:
        try:
            import torch

            gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
        except Exception:
            gpu_name = "unknown"
        driver_path = Path("/proc/driver/nvidia/version")
        try:
            driver_version = driver_path.read_text(encoding="utf-8", errors="ignore").strip()
        except OSError:
            driver_version = "unknown"
        renderer_digest = hashlib.sha256()
        renderer_paths = (
            self.repo_root / "qqtt/garden_assets.py",
            self.repo_root / "qqtt/garden_scene.py",
            self.repo_root / "qqtt/engine/trainer_warp.py",
            self.repo_root / "qqtt/immersive_gaussian_fusion_triton.py",
            self.repo_root / "gaussian_splatting/gaussian_renderer/__init__.py",
            self.repo_root / "gaussian_splatting/_gsplat_vendor.py",
            self.repo_root
            / "gaussian_splatting/submodules/gsplat/BOBA_VENDOR_INFO.md",
        )
        for path in renderer_paths:
            renderer_digest.update(str(path.relative_to(self.repo_root)).encode("utf-8"))
            renderer_digest.update(
                (sha256_file(path) if path.is_file() else "missing").encode("ascii")
            )
        renderer_revision = renderer_digest.hexdigest()
        calibration_path = resolve_repo_path(
            self.repo_root,
            self.manifest["calibration"],
        )
        model_digest = hashlib.sha256()
        model_digest.update(
            str(self.manifest["source"]["point_cloud_sha256"]).encode("ascii")
        )
        model_digest.update(sha256_file(self.manifest_path).encode("ascii"))
        model_digest.update(sha256_file(calibration_path).encode("ascii"))
        model_sha = model_digest.hexdigest()
        return garden_profile_key(
            model_sha256=model_sha,
            gpu_name=gpu_name,
            driver_version=driver_version,
            renderer_revision=renderer_revision,
            eye_resolution=self.eye_resolution,
        )

    def scene_analysis_cache_debug(self) -> dict[str, Any]:
        return {
            "status": "prepared",
            "reason": f"garden_{self.quality}",
            "schema": 1,
            "input_hash": self.runtime_metadata.get("source_sha256"),
            "path": str(self.runtime_ply_path),
        }

    def set_layout(self, layout: SimpleLabLayout) -> None:
        if str(getattr(layout, "scene_name", "")) != self.scene_runtime_name:
            raise GardenAssetError(
                f"{self.scene_display_name} renderer received a layout for "
                f"{getattr(layout, 'scene_name', None)!r}."
            )
        self.layout = layout
        chunks = self.runtime_metadata.get("spatial_chunks")
        if not isinstance(chunks, list) or not chunks:
            raise GardenAssetError(
                f"{self.scene_display_name} runtime metadata contains no spatial chunks."
            )
        self._chunk_starts = np.asarray(
            [int(chunk["start"]) for chunk in chunks], dtype=np.int64
        )
        self._chunk_counts = np.asarray(
            [int(chunk["count"]) for chunk in chunks], dtype=np.int64
        )
        canonical_centers = np.asarray(
            [chunk["sphere_center"] for chunk in chunks], dtype=np.float32
        )
        rotation = np.asarray(
            layout.canonical_to_world_rotation, dtype=np.float32
        ).reshape(3, 3)
        translation = np.asarray(
            layout.canonical_to_world_translation, dtype=np.float32
        ).reshape(3)
        self._chunk_centers_world = (
            canonical_centers @ rotation.T + translation[None, :]
        ).astype(np.float32)
        self._chunk_radii = np.asarray(
            [float(chunk["sphere_radius"]) for chunk in chunks],
            dtype=np.float32,
        )

    def pyrender_readback_mode(self) -> str:
        return "gpu_native"

    def pyrender_readback_reason(self) -> None:
        return None

    def last_balanced_render_debug(self) -> dict[str, Any]:
        return {
            "request_kind": "garden_combined_gaussian",
            "effective_backend": "gpu",
            "quality": self.quality,
        }

    def _blank_frame(self, width: int, height: int):
        import torch

        key = (int(width), int(height), int(torch.cuda.current_device()))
        cached = self._blank_cache.get(key)
        if cached is None:
            color = torch.zeros((int(height), int(width), 4), dtype=torch.float32, device="cuda")
            color[..., 3] = 255.0
            depth = torch.zeros((int(height), int(width)), dtype=torch.float32, device="cuda")
            cached = (color, depth)
            self._blank_cache[key] = cached
        return cached

    def prepare_direct_gaussian_eye_output(
        self,
        gaussian_rgba,
        gaussian_depth,
        *,
        output_dtype=None,
    ):
        """Convert one combined static/object Gaussian render into an eye frame.

        gsplat RGB is already composited against the black render background.
        Multiplying it by its alpha again, as the Lab scene/object compositor
        does, both darkens partially covered scene surfaces and performs an
        unnecessary full-frame blend. The Gaussian image is the complete scene,
        so preserve its RGB and publish it as opaque.
        """

        import torch

        if not torch.is_tensor(gaussian_rgba):
            raise TypeError(
                f"{self.scene_display_name} direct output requires a torch "
                "Gaussian RGBA tensor."
            )
        if gaussian_rgba.ndim != 3 or int(gaussian_rgba.shape[0]) < 3:
            raise ValueError(
                f"{self.scene_display_name} direct output expects Gaussian color "
                "with shape [3|4, H, W]."
            )
        if output_dtype is None:
            output_dtype = torch.float32
        if output_dtype not in {torch.float32, torch.uint8}:
            raise ValueError(
                f"{self.scene_display_name} direct output supports only "
                "torch.float32 or torch.uint8 frames."
            )

        height = int(gaussian_rgba.shape[1])
        width = int(gaussian_rgba.shape[2])
        rgb_hwc = gaussian_rgba[:3].detach().permute(1, 2, 0)
        if output_dtype is torch.uint8:
            rgb_output = (
                rgb_hwc.mul(255.0)
                .round()
                .clamp_(0.0, 255.0)
                .to(dtype=torch.uint8)
            )
        else:
            rgb_output = rgb_hwc.mul(255.0).clamp_(0.0, 255.0)

        frame = torch.empty(
            (height, width, 4),
            dtype=output_dtype,
            device=gaussian_rgba.device,
        )
        frame[..., :3].copy_(rgb_output)
        frame[..., 3].fill_(255)

        depth = None
        if gaussian_depth is not None:
            if not torch.is_tensor(gaussian_depth):
                depth = torch.as_tensor(
                    gaussian_depth,
                    dtype=torch.float32,
                    device=gaussian_rgba.device,
                )
            else:
                depth = gaussian_depth.detach().to(
                    device=gaussian_rgba.device,
                    dtype=torch.float32,
                )
            depth = depth.squeeze()
            if tuple(depth.shape) != (height, width):
                raise ValueError(
                    f"{self.scene_display_name} direct output depth must match "
                    "the Gaussian image size."
                )
            depth = torch.nan_to_num(
                depth,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).clamp_min_(0.0).contiguous()

        metrics = {
            "compose_mode": "garden_direct_output",
            "direct_gaussian_output": True,
            "direct_gaussian_scene_name": self.scene_runtime_name,
            "compose_roi_ratio": 1.0,
            "visible_retention_ratio": 1.0,
            "scene_depth_invalid": False,
            "scene_depth_suppressed": False,
            "garden_direct_output": True,
        }
        return frame, depth, metrics

    def render_eye(self, _camera_pose_world, _intrinsic, width=None, height=None):
        return self._blank_frame(width or self.width, height or self.height)

    def render_background_eye(self, camera_pose_world, intrinsic, width=None, height=None):
        return self.render_eye(camera_pose_world, intrinsic, width=width, height=height)

    def render_table_eye(self, camera_pose_world, intrinsic, width=None, height=None):
        return self.render_eye(camera_pose_world, intrinsic, width=width, height=height)

    def _render_roi(self, roi_bounds, return_render_info=False, render_scale=1.0):
        x0, y0, x1, y1 = [int(value) for value in roi_bounds]
        width = max(1, int(round((x1 - x0) * float(render_scale))))
        height = max(1, int(round((y1 - y0) * float(render_scale))))
        output = self._blank_frame(width, height)
        if return_render_info:
            return output[0], output[1], {
                "roi_bounds": (x0, y0, x1, y1),
                "render_width": width,
                "render_height": height,
            }
        return output

    def render_eye_roi(self, _pose, _intrinsic, roi_bounds, render_scale=1.0, return_render_info=False):
        return self._render_roi(roi_bounds, return_render_info, render_scale)

    def render_background_eye_roi(self, _pose, _intrinsic, roi_bounds):
        return self._render_roi(roi_bounds)

    def render_table_eye_roi(self, _pose, _intrinsic, roi_bounds, render_scale=1.0, return_render_info=False):
        return self._render_roi(roi_bounds, return_render_info, render_scale)

    def warmup_balanced_runtime_paths(self, **_kwargs) -> list[str]:
        self._blank_frame(self.width, self.height)
        return [f"{self.scene_runtime_name}_combined_gaussian"]

    def table_world_bounds(self):
        if self.layout is None or self.layout.active_table_bounds is None:
            return None
        bounds = np.asarray(self.layout.active_table_bounds, dtype=np.float32)
        return bounds[0].copy(), bounds[1].copy()

    def wall_world_bounds(self, _wall_name):
        return None

    def table_alignment_debug(self) -> dict[str, Any] | None:
        if self.layout is None:
            return None
        active_surface_center = getattr(
            self.layout,
            "active_table_surface_center",
            None,
        )
        center = np.asarray(
            self.layout.table_top_center
            if active_surface_center is None
            else active_surface_center,
            dtype=np.float32,
        )
        scene_up = np.asarray(self.layout.scene_up, dtype=np.float32)
        surface_normal = np.asarray(
            getattr(self.layout, "ambulance_mattress_normal_world", scene_up),
            dtype=np.float32,
        )
        surface_normal_alignment = float(
            np.clip(np.dot(surface_normal, scene_up), -1.0, 1.0)
        )
        runtime_surface_kinds = {
            str(surface.get("kind", "unknown"))
            for surface in self.layout.static_collision_surfaces
        }
        has_runtime_heightfield = any(
            kind.startswith("heightfield") for kind in runtime_surface_kinds
        )
        collision_metadata = list(
            getattr(self.layout, "static_collision_mesh_metadata", [])
        )
        runtime_mattress_mesh_face_count = sum(
            int(entry.get("face_count", 0))
            for entry in collision_metadata
            if entry.get("kind") == "source_mesh_full_resolution_surface"
        )
        runtime_heightfield_face_count = int(
            len(getattr(self.layout, "ambulance_mattress_heightfield_faces", []))
        )
        runtime_detail_mesh_face_count = int(
            len(getattr(self.layout, "static_collision_detail_mesh_faces", []))
        )
        if (
            runtime_mattress_mesh_face_count > 0
            and runtime_detail_mesh_face_count > runtime_mattress_mesh_face_count
        ):
            runtime_collision_kind = (
                "full_mattress_mesh+compact_detail_mesh+finite_support_surfaces"
            )
        elif runtime_mattress_mesh_face_count > 0:
            runtime_collision_kind = "full_mattress_mesh+finite_support_surfaces"
        elif has_runtime_heightfield and runtime_detail_mesh_face_count > 0:
            runtime_collision_kind = (
                "heightfield+detail_mesh+finite_support_surfaces"
            )
        elif has_runtime_heightfield:
            runtime_collision_kind = "heightfield+finite_support_surfaces"
        elif runtime_detail_mesh_face_count > 0:
            runtime_collision_kind = "detail_mesh+finite_support_surfaces"
        else:
            runtime_collision_kind = "finite_support_surfaces"
        return {
            "asset_transform": (
                f"{self.scene_runtime_name}_canonical_to_head_aligned_world"
            ),
            "world_surface_normal": surface_normal.tolist(),
            "surface_normal_alignment": surface_normal_alignment,
            "world_surface_center": center.tolist(),
            "world_surface_plane_height": float(center[2]),
            "collider_top_plane_height": float(center[2]),
            "active_table_support_patch_count": int(
                getattr(self.layout, "smooth_tabletop_patch_count", 1)
            ),
            "support_slab_count": int(
                len(getattr(self.layout, "support_surface_boxes", []))
            ),
            "blocker_box_count": 0,
            "collider_box_count": 0,
            "collision_mesh_face_count": int(self.layout.static_collision_mesh_faces.shape[0]),
            "runtime_collision_kind": runtime_collision_kind,
            "runtime_collision_surface_count": len(
                self.layout.static_collision_surfaces
            ),
            "runtime_collision_heightfield_face_count": (
                runtime_heightfield_face_count
            ),
            "runtime_collision_mattress_mesh_face_count": (
                runtime_mattress_mesh_face_count
            ),
            "runtime_collision_detail_mesh_face_count": (
                runtime_detail_mesh_face_count
            ),
            "runtime_collision_detail_mesh_component_count": int(
                len(
                    getattr(
                        self.layout,
                        "static_collision_detail_mesh_component_bounds",
                        [],
                    )
                )
            ),
            "runtime_collision_detail_mesh_two_sided": bool(
                getattr(
                    self.layout,
                    "static_collision_detail_mesh_two_sided",
                    False,
                )
            ),
            "runtime_collision_detail_mesh_source_asset": str(
                getattr(
                    self.layout,
                    "static_collision_detail_mesh_source_asset",
                    "",
                )
            ),
            "table_render_component_ids": [],
            "background_excludes_active_table": False,
            "table_render_bounds_source": (
                f"{self.scene_runtime_name}_collision_proxy"
            ),
            "table_render_world_bounds": np.asarray(self.layout.active_table_bounds).tolist(),
        }

    def support_surface_entries_ref(self) -> list[dict[str, Any]]:
        if self.layout is None:
            return []
        output = []
        support_id = 1
        for entry in getattr(self.layout, "static_collision_mesh_metadata", []):
            if not bool(entry.get("support", False)):
                continue
            bounds_min = np.asarray(entry["world_bounds_min"], dtype=np.float32)
            bounds_max = np.asarray(entry["world_bounds_max"], dtype=np.float32)
            center = 0.5 * (bounds_min + bounds_max)
            entry_name = str(entry["name"])
            entry_name_lower = entry_name.lower()
            output.append(
                {
                    "support_id": support_id,
                    "component_id": None,
                    "kind": (
                        "floor"
                        if "floor" in entry_name_lower or "patio" in entry_name_lower
                        else "table"
                    ),
                    "name": entry_name,
                    "bounds_min": bounds_min,
                    "bounds_max": bounds_max,
                    "render_bounds_min": bounds_min,
                    "render_bounds_max": bounds_max,
                    "center": center,
                    "support_area": float(
                        max(bounds_max[0] - bounds_min[0], 0.0)
                        * max(bounds_max[1] - bounds_min[1], 0.0)
                    ),
                }
            )
            support_id += 1
        return output

    def focus_render_catalog_world_entries_ref(self):
        return []

    def focus_render_catalog_world_by_id_ref(self):
        return {}

    def focus_render_bvh_world_nodes(self):
        return []

    def focus_render_catalog_total_faces(self):
        return 0

    def focus_render_active_table_entry_ids(self):
        return []

    def _load_static_gaussian_model(self, device):
        from gaussian_splatting.scene.gaussian_model import GaussianModel

        static = GaussianModel(sh_degree=3)
        static.load_ply(str(self.runtime_ply_path))
        return static

    def bind_dynamic_gaussians(self, dynamic_gaussians):
        if self.layout is None:
            raise RuntimeError(
                f"{self.scene_display_name} layout must be set before binding Gaussians."
            )
        import torch
        from gaussian_splatting.rotation_utils import quaternion_multiply
        from gaussian_splatting.scene.gaussian_model import GaussianModel

        dynamic_count = int(dynamic_gaussians._xyz.shape[0])
        capacity = int(self.manifest.get("dynamic_gaussian_capacity", 262144))
        if dynamic_count > capacity:
            raise RuntimeError(
                f"{self.scene_display_name} dynamic Gaussian capacity {capacity} "
                f"is smaller than object count {dynamic_count}."
            )

        if self._combined_gaussians is None:
            static = self._load_static_gaussian_model(dynamic_gaussians._xyz.device)
            static.isotropic = False
            rotation_np = np.asarray(
                self.layout.canonical_to_world_rotation,
                dtype=np.float32,
            )
            translation = torch.as_tensor(
                self.layout.canonical_to_world_translation,
                dtype=static._xyz.dtype,
                device=static._xyz.device,
            )
            rotation = torch.as_tensor(
                rotation_np,
                dtype=static._xyz.dtype,
                device=static._xyz.device,
            )
            with torch.no_grad():
                static._xyz = static._xyz @ rotation.T + translation.unsqueeze(0)
                rotation_quaternion = _matrix_to_quaternion_wxyz_torch(
                    rotation_np,
                    torch_module=torch,
                    device=static._rotation.device,
                    dtype=static._rotation.dtype,
                )
                static._rotation = quaternion_multiply(
                    rotation_quaternion.unsqueeze(0),
                    static.get_rotation,
                )
                sh_transform = torch.as_tensor(
                    sh_rotation_matrix(rotation_np),
                    dtype=static._features_dc.dtype,
                    device=static._features_dc.device,
                )
                features = torch.cat([static._features_dc, static._features_rest], dim=1)
                features = torch.einsum("ij,njc->nic", sh_transform, features)
                static._features_dc = features[:, :1].contiguous()
                static._features_rest = features[:, 1:].contiguous()
                if static._scaling.shape[1] == 1:
                    static._scaling = static._scaling.repeat(1, 3).contiguous()

            self._static_count = int(static._xyz.shape[0])
            if int(np.sum(self._chunk_counts, dtype=np.int64)) != self._static_count:
                raise GardenAssetError(
                    f"{self.scene_display_name} spatial chunks do not match the "
                    "loaded static Gaussian count."
                )
            self._dynamic_capacity = capacity
            total = self._static_count + capacity
            def _allocate_render_slot():
                storage = GaussianModel(sh_degree=3)
                storage.active_sh_degree = 3
                storage.isotropic = False
                storage._xyz = torch.empty(
                    (total, 3), dtype=static._xyz.dtype, device=static._xyz.device
                )
                storage._features_dc = torch.empty(
                    (total, 1, 3),
                    dtype=static._features_dc.dtype,
                    device=static._features_dc.device,
                )
                storage._features_rest = torch.empty(
                    (total, 15, 3),
                    dtype=static._features_rest.dtype,
                    device=static._features_rest.device,
                )
                storage._opacity = torch.full(
                    (total, 1),
                    -100.0,
                    dtype=static._opacity.dtype,
                    device=static._opacity.device,
                )
                storage._scaling = torch.zeros(
                    (total, 3),
                    dtype=static._scaling.dtype,
                    device=static._scaling.device,
                )
                storage._rotation = torch.zeros(
                    (total, 4),
                    dtype=static._rotation.dtype,
                    device=static._rotation.device,
                )
                storage._rotation[:, 0] = 1.0
                model = GaussianModel(sh_degree=3)
                model.active_sh_degree = 3
                model.isotropic = False
                return storage, model

            storage, combined = _allocate_render_slot()
            standby_storage, standby_combined = _allocate_render_slot()
            with torch.no_grad():
                prefix = slice(0, self._static_count)
                storage._xyz[prefix].copy_(static._xyz)
                storage._features_dc[prefix].copy_(static._features_dc)
                storage._features_rest[prefix].copy_(static._features_rest)
                storage._opacity[prefix].copy_(static._opacity)
                storage._scaling[prefix].copy_(static._scaling)
                storage._rotation[prefix].copy_(static._rotation)
            self._static_gaussians = static
            self._render_storages = [storage, standby_storage]
            self._combined_models = [combined, standby_combined]
            self._active_render_slot = 0
            self._render_storage = storage
            self._combined_gaussians = combined
            self._chunk_rebuild_stream = torch.cuda.Stream(priority=0)
            self._pending_chunk_rebuild = None
            self._active_static_count = self._static_count
            self._active_chunk_ids = tuple(range(int(self._chunk_starts.size)))
            self._set_active_render_views(self._static_count)
            del features
            gc.collect()
            torch.cuda.empty_cache()

        self._dynamic_gaussians = dynamic_gaussians
        self._copy_dynamic_payload(dynamic_gaussians, include_appearance=True)
        return self._combined_gaussians

    def _set_active_render_views(self, total_count: int) -> None:
        total_count = int(total_count)
        storage = getattr(self, "_render_storage", None)
        target = getattr(self, "_combined_gaussians", None)
        if storage is None or target is None:
            return
        self._set_render_views(storage, target, total_count)

    @staticmethod
    def _set_render_views(storage, target, total_count: int) -> None:
        total_count = int(total_count)
        target._xyz = storage._xyz[:total_count]
        target._features_dc = storage._features_dc[:total_count]
        target._features_rest = storage._features_rest[:total_count]
        target._opacity = storage._opacity[:total_count]
        target._scaling = storage._scaling[:total_count]
        target._rotation = storage._rotation[:total_count]

    def _normalize_chunk_ids(self, chunk_ids) -> tuple[int, ...]:
        normalized = tuple(sorted(set(int(value) for value in chunk_ids)))
        chunk_count = int(self._chunk_starts.size)
        if any(value < 0 or value >= chunk_count for value in normalized):
            raise ValueError(
                f"{self.scene_display_name} spatial chunk selection is out of range."
            )
        return normalized

    def _gather_selected_static_chunks_into(
        self,
        chunk_ids: tuple[int, ...],
        target,
    ) -> int:
        import torch

        if self._static_gaussians is None or target is None:
            raise RuntimeError(
                f"{self.scene_display_name} static Gaussian storage is not initialized."
            )
        chunk_ids = self._normalize_chunk_ids(chunk_ids)
        chunk_count = int(self._chunk_starts.size)
        selected_count = int(
            sum(int(self._chunk_counts[value]) for value in chunk_ids)
        )
        source = self._static_gaussians
        with torch.no_grad():
            if len(chunk_ids) == chunk_count:
                copy_ranges = [(0, self._static_count, 0, self._static_count)]
            elif selected_count > 0:
                # Chunk records are contiguous in the cached PLY. Merge
                # neighboring selected IDs and copy those source ranges
                # directly. This avoids constructing and uploading a large
                # per-Gaussian gather-index tensor whenever the view changes.
                source_ranges: list[tuple[int, int]] = []
                run_first = chunk_ids[0]
                run_last = run_first
                for value in chunk_ids[1:]:
                    if value == run_last + 1:
                        run_last = value
                        continue
                    source_ranges.append(
                        (
                            int(self._chunk_starts[run_first]),
                            int(
                                self._chunk_starts[run_last]
                                + self._chunk_counts[run_last]
                            ),
                        )
                    )
                    run_first = value
                    run_last = value
                source_ranges.append(
                    (
                        int(self._chunk_starts[run_first]),
                        int(
                            self._chunk_starts[run_last]
                            + self._chunk_counts[run_last]
                        ),
                    )
                )
                copy_ranges = []
                destination_start = 0
                for source_start, source_stop in source_ranges:
                    destination_stop = destination_start + (
                        source_stop - source_start
                    )
                    copy_ranges.append(
                        (
                            source_start,
                            source_stop,
                            destination_start,
                            destination_stop,
                        )
                    )
                    destination_start = destination_stop
            else:
                copy_ranges = []
            for name in (
                "_xyz",
                "_features_dc",
                "_features_rest",
                "_opacity",
                "_scaling",
                "_rotation",
            ):
                source_tensor = getattr(source, name)
                target_tensor = getattr(target, name)
                for (
                    source_start,
                    source_stop,
                    destination_start,
                    destination_stop,
                ) in copy_ranges:
                    target_tensor[
                        destination_start:destination_stop
                    ].copy_(
                        source_tensor[source_start:source_stop]
                    )
        return selected_count

    def _copy_selected_static_chunks(self, chunk_ids: tuple[int, ...]) -> int:
        chunk_ids = self._normalize_chunk_ids(chunk_ids)
        selected_count = self._gather_selected_static_chunks_into(
            chunk_ids,
            self._render_storage,
        )
        self._active_static_count = selected_count
        self._active_chunk_ids = chunk_ids
        self._chunk_rebuild_count += 1
        if self._dynamic_gaussians is None:
            self._set_active_render_views(selected_count)
        else:
            self._copy_dynamic_payload(
                self._dynamic_gaussians,
                include_appearance=True,
            )
        return selected_count

    def _start_async_chunk_rebuild(self, chunk_ids: tuple[int, ...]) -> bool:
        import torch

        if self._pending_chunk_rebuild is not None:
            return False
        if len(self._render_storages) < 2 or self._chunk_rebuild_stream is None:
            return False
        chunk_ids = self._normalize_chunk_ids(chunk_ids)
        standby_slot = 1 - int(self._active_render_slot)
        standby_storage = self._render_storages[standby_slot]
        release_event = torch.cuda.Event()
        release_event.record(torch.cuda.current_stream())
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(self._chunk_rebuild_stream):
            self._chunk_rebuild_stream.wait_event(release_event)
            start_event.record(self._chunk_rebuild_stream)
            selected_count = self._gather_selected_static_chunks_into(
                chunk_ids,
                standby_storage,
            )
            end_event.record(self._chunk_rebuild_stream)
        self._pending_chunk_rebuild = {
            "slot": int(standby_slot),
            "chunk_ids": chunk_ids,
            "selected_count": int(selected_count),
            "release_event": release_event,
            "start_event": start_event,
            "end_event": end_event,
        }
        return True

    def _try_complete_async_chunk_rebuild(
        self,
        required_ids: tuple[int, ...],
        *,
        force: bool,
    ) -> tuple[bool, float]:
        import torch

        pending = self._pending_chunk_rebuild
        if pending is None:
            return False, 0.0
        pending_ids = set(pending["chunk_ids"])
        required_covered = bool(
            not pending.get("discard", False)
            and set(required_ids).issubset(pending_ids)
        )
        end_event = pending["end_event"]
        ready = bool(end_event.query())
        if not ready and force and required_covered:
            end_event.synchronize()
            ready = True
        if not ready:
            return False, 0.0
        elapsed_ms = float(pending["start_event"].elapsed_time(end_event))
        if not required_covered:
            self._pending_chunk_rebuild = None
            return False, elapsed_ms

        torch.cuda.current_stream().wait_event(end_event)
        slot = int(pending["slot"])
        self._active_render_slot = slot
        self._render_storage = self._render_storages[slot]
        self._combined_gaussians = self._combined_models[slot]
        self._active_static_count = int(pending["selected_count"])
        self._active_chunk_ids = tuple(pending["chunk_ids"])
        self._chunk_rebuild_count += 1
        self._pending_chunk_rebuild = None
        if self._dynamic_gaussians is None:
            self._set_active_render_views(self._active_static_count)
        else:
            self._copy_dynamic_payload(
                self._dynamic_gaussians,
                include_appearance=True,
            )
        return True, elapsed_ms

    def _copy_dynamic_payload(self, dynamic_gaussians, *, include_appearance: bool) -> None:
        import torch

        if self._combined_gaussians is None:
            raise RuntimeError(
                f"{self.scene_display_name} combined Gaussian model is not initialized."
            )
        count = int(dynamic_gaussians._xyz.shape[0])
        if count > self._dynamic_capacity:
            raise RuntimeError(
                f"Dynamic {self.scene_display_name} Gaussian capacity exceeded."
            )
        start = int(getattr(self, "_active_static_count", self._static_count))
        stop = start + count
        target = getattr(self, "_render_storage", None)
        if target is None:
            target = self._combined_gaussians
        with torch.no_grad():
            target._xyz[start:stop].copy_(dynamic_gaussians._xyz)
            target._rotation[start:stop].copy_(dynamic_gaussians.get_rotation)
            if include_appearance:
                target._opacity[start:stop].copy_(dynamic_gaussians._opacity)
                target._features_dc[start:stop].copy_(dynamic_gaussians._features_dc)
                target._features_rest[start:stop].copy_(dynamic_gaussians._features_rest)
                scaling = dynamic_gaussians._scaling
                if scaling.shape[1] == 1:
                    scaling = scaling.repeat(1, 3)
                target._scaling[start:stop].copy_(scaling)
        self._dynamic_count = count
        self._dynamic_gaussians = dynamic_gaussians
        self._set_active_render_views(stop)

    def sync_dynamic_gaussians(self, dynamic_gaussians):
        self._copy_dynamic_payload(dynamic_gaussians, include_appearance=False)
        return self._combined_gaussians

    @staticmethod
    def _camera_arrays_from_eye_state(eye_state: dict) -> tuple[np.ndarray, np.ndarray, int, int]:
        w2c = eye_state.get("w2c_cv_np")
        intrinsic = eye_state.get("intrinsic_np")
        width = eye_state.get("width")
        height = eye_state.get("height")
        view = eye_state.get("view")
        if w2c is None and view is not None:
            value = view.world_view_transform.transpose(0, 1)
            w2c = value.detach().cpu().numpy() if hasattr(value, "detach") else value
        if intrinsic is None and view is not None:
            value = view.K
            intrinsic = value.detach().cpu().numpy() if hasattr(value, "detach") else value
        if width is None and view is not None:
            width = int(view.image_width)
        if height is None and view is not None:
            height = int(view.image_height)
        return (
            np.asarray(w2c, dtype=np.float32).reshape(4, 4),
            np.asarray(intrinsic, dtype=np.float32).reshape(3, 3),
            int(width),
            int(height),
        )

    def _stereo_chunk_mask(
        self,
        left_eye_state: dict,
        right_eye_state: dict,
        *,
        margin_ratio: float,
    ) -> np.ndarray:
        config = self.runtime_metadata["spatial_chunk_config"]
        result = np.zeros((self._chunk_centers_world.shape[0],), dtype=bool)
        for eye_state in (left_eye_state, right_eye_state):
            w2c, intrinsic, width, height = self._camera_arrays_from_eye_state(
                eye_state
            )
            result |= gaussian_chunk_spheres_in_camera_frustum(
                self._chunk_centers_world,
                self._chunk_radii,
                w2c,
                intrinsic,
                width,
                height,
                near_plane_m=float(config["near_plane_m"]),
                padding_m=float(config["frustum_padding_m"]),
                margin_ratio=float(margin_ratio),
            )
        return result

    def select_stereo_frustum_gaussians(
        self,
        left_eye_state: dict,
        right_eye_state: dict,
    ):
        """Return scene/object Gaussians for the conservative stereo chunk union."""

        if self._combined_gaussians is None:
            raise RuntimeError(
                f"{self.scene_display_name} Gaussians must be bound before chunk "
                "selection."
            )
        required_mask = self._stereo_chunk_mask(
            left_eye_state,
            right_eye_state,
            margin_ratio=0.0,
        )
        required_ids = tuple(int(value) for value in np.flatnonzero(required_mask))
        first_selection = not self._chunk_selection_initialized
        required_set = set(required_ids)
        active_set = set(self._active_chunk_ids)
        active_covers_required = required_set.issubset(active_set)
        rebuilt, completed_rebuild_cuda_ms = (
            self._try_complete_async_chunk_rebuild(
                required_ids,
                force=not active_covers_required,
            )
        )
        active_set = set(self._active_chunk_ids)
        active_covers_required = required_set.issubset(active_set)
        prefetch_margin = float(
            self.runtime_metadata["spatial_chunk_config"][
                "prefetch_margin_ratio"
            ]
        )
        selected_mask = self._stereo_chunk_mask(
            left_eye_state,
            right_eye_state,
            margin_ratio=prefetch_margin,
        )
        selected_mask |= required_mask
        selected_ids = tuple(int(value) for value in np.flatnonzero(selected_mask))
        emergency_rebuild = False
        if not active_covers_required:
            pending = self._pending_chunk_rebuild
            if pending is not None:
                pending["discard"] = True
            self._copy_selected_static_chunks(selected_ids)
            rebuilt = True
            emergency_rebuild = True
            active_set = set(self._active_chunk_ids)

        rebuild_started = False
        # Rebuild for both entering and leaving chunks. The padded selection
        # normally absorbs small head motion, while this exact comparison
        # prevents old, now-offscreen chunks from remaining submitted after a
        # larger view change.
        needs_prefetched_selection = bool(
            first_selection or selected_ids != self._active_chunk_ids
        )
        if (
            needs_prefetched_selection
            and self._pending_chunk_rebuild is None
        ):
            rebuild_started = self._start_async_chunk_rebuild(selected_ids)
        self._chunk_selection_initialized = True
        total_chunks = max(1, int(self._chunk_starts.size))
        static_total = max(1, int(self._static_count))
        pending = self._pending_chunk_rebuild
        debug = {
            "rebuilt": bool(rebuilt),
            "rebuild_started": bool(rebuild_started),
            "rebuild_pending": bool(pending is not None),
            "emergency_rebuild": bool(emergency_rebuild),
            "completed_rebuild_cuda_ms": float(completed_rebuild_cuda_ms),
            "required_chunk_count": int(len(required_ids)),
            "selected_chunk_count": int(len(self._active_chunk_ids)),
            "total_chunk_count": int(self._chunk_starts.size),
            "selected_chunk_ratio": float(len(self._active_chunk_ids) / total_chunks),
            "selected_static_count": int(self._active_static_count),
            "total_static_count": int(self._static_count),
            "selected_static_ratio": float(self._active_static_count / static_total),
            "dynamic_count": int(self._dynamic_count),
            "render_gaussian_count": int(
                self._active_static_count + self._dynamic_count
            ),
            "rebuild_count": int(self._chunk_rebuild_count),
        }
        if pending is not None:
            debug["pending_selected_chunk_count"] = int(
                len(pending["chunk_ids"])
            )
            debug["pending_selected_static_count"] = int(
                pending["selected_count"]
            )
        self._last_chunk_selection_debug = debug
        if first_selection:
            print(
                f"[quest_display] {self.scene_runtime_name} stereo frustum chunks: "
                f"required={debug['required_chunk_count']} "
                f"active={debug['selected_chunk_count']}/"
                f"{debug['total_chunk_count']} "
                f"active_static_gaussians={debug['selected_static_count']}/"
                f"{debug['total_static_count']} "
                f"ratio={debug['selected_static_ratio']:.3f} "
                f"dynamic_gaussians={debug['dynamic_count']} "
                f"async_compaction_started={int(debug['rebuild_started'])}",
                flush=True,
            )
        return self._combined_gaussians, debug

    @property
    def combined_gaussians(self):
        return self._combined_gaussians

    def record_source_frame_seconds(self, duration_seconds: float) -> None:
        """Record measured source-frame cadence for ``--garden-quality auto``."""

        if self._profile_written:
            return
        duration = float(duration_seconds)
        if not np.isfinite(duration) or duration <= 0.0:
            return
        self._profile_source_frame_seconds.append(duration)
        if len(self._profile_source_frame_seconds) < GARDEN_PROFILE_SAMPLE_COUNT:
            return
        median_seconds = statistics.median(
            self._profile_source_frame_seconds[-GARDEN_PROFILE_SAMPLE_COUNT:]
        )
        source_fps = 1.0 / max(float(median_seconds), 1.0e-9)
        record_garden_profile(
            self.repo_root,
            profile_key=self._profile_key,
            quality=self.quality,
            source_fps=source_fps,
            sample_count=GARDEN_PROFILE_SAMPLE_COUNT,
        )
        self._profile_written = True
        print(
            "[quest_display] garden quality profile: "
            f"quality={self.quality} source_fps={source_fps:.2f} "
            f"target_fps={GARDEN_PROFILE_TARGET_FPS:.2f} samples={GARDEN_PROFILE_SAMPLE_COUNT}",
            flush=True,
        )

    def record_gaussian_render_seconds(self, duration_seconds: float) -> None:
        """Backward-compatible alias retained for local profiling tools."""

        self.record_source_frame_seconds(duration_seconds)

    def export_collision_proxy_obj(self, path: str | Path) -> Path:
        if self.layout is None:
            raise RuntimeError("Garden layout is not configured.")
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        vertices = np.asarray(self.layout.static_collision_mesh_vertices, dtype=np.float32)
        faces = np.asarray(self.layout.static_collision_mesh_faces, dtype=np.int32)
        frame_origin = np.asarray(
            self.layout.canonical_to_world_translation, dtype=np.float32
        )
        frame_rotation = np.asarray(
            self.layout.canonical_to_world_rotation, dtype=np.float32
        )
        frame_vertices = np.concatenate(
            [
                frame_origin.reshape(1, 3),
                frame_origin.reshape(1, 3) + frame_rotation.T * 0.25,
            ],
            axis=0,
        )
        temporary = output.with_suffix(output.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write("# Boba Garden world-space collision proxy\n")
            handle.write("o collision_proxy\n")
            for vertex in vertices:
                handle.write(f"v {vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f}\n")
            for face in faces:
                handle.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")
            handle.write("o placement_frame\n")
            handle.write("# axes: canonical +x right, +y forward, +z down\n")
            for vertex in frame_vertices:
                handle.write(f"v {vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f}\n")
            frame_start = int(vertices.shape[0]) + 1
            for axis_offset in (1, 2, 3):
                handle.write(f"l {frame_start} {frame_start + axis_offset}\n")
        os.replace(temporary, output)
        return output

    def delete(self) -> None:
        pending = getattr(self, "_pending_chunk_rebuild", None)
        if pending is not None:
            try:
                pending["end_event"].synchronize()
            except Exception:
                pass
        self._blank_cache.clear()
        self._static_gaussians = None
        self._render_storage = None
        self._render_storages = []
        self._combined_gaussians = None
        self._combined_models = []
        self._dynamic_gaussians = None
        self._pending_chunk_rebuild = None
        self._chunk_rebuild_stream = None
        self._dynamic_count = 0
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass


def direct_gaussian_scene_enabled(scene_name: str, scene_renderer: Any) -> bool:
    """Return whether the selected renderer owns the whole Gaussian scene."""

    normalized_scene_name = str(scene_name).strip().lower()
    renderer_scene_name = str(
        getattr(scene_renderer, "scene_runtime_name", "")
    ).strip().lower()
    return bool(
        normalized_scene_name
        and normalized_scene_name == renderer_scene_name
        and bool(getattr(scene_renderer, "supports_direct_gaussian_output", False))
    )


def garden_direct_output_enabled(scene_name: str, scene_renderer: Any) -> bool:
    """Backward-compatible Garden-specific direct-output capability check."""

    return bool(
        str(scene_name).strip().lower() == GARDEN_SCENE_NAME
        and direct_gaussian_scene_enabled(scene_name, scene_renderer)
    )
