from __future__ import annotations

import gzip
import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh
from trimesh.graph import connected_component_labels

from .pyrender_cuda_bridge import (
    PyrenderCudaInteropOffscreenRenderer,
    probe_pyrender_cuda_bridge_support,
)


MANIFEST_RELATIVE_PATH = Path("assets/scenes/ILLIXR_lab/manifest.json")
ILLIXR_SCENE_NAME = "ILLIXR_lab"
ILLIXR_BAKED_LIGHTING_MODE = "baked_texture_ambient"
SCENE_ANALYSIS_CACHE_SCHEMA = 1
DEFAULT_SCENE_ANALYSIS_CACHE_FILENAME = "scene_analysis_cache_v1.pkl.gz"

TARGET_TABLE_SIZE_X = 0.95
TARGET_TABLE_SIZE_Y = 0.68
TABLE_SUPPORT_HEIGHT_MIN = 0.12
TABLE_SUPPORT_AREA_MIN = 0.08
TABLE_SUPPORT_GAP_Y = 0.03
TABLE_PRIMARY_PATCH_MIN_SPAN = 0.35
TABLE_PRIMARY_PATCH_MIN_LONG_SPAN = 0.75
TABLE_COMPONENT_VERTICAL_REACH = 0.95
COLLIDER_COMPONENT_AREA_MIN = 0.30
COLLIDER_HORIZONTAL_AREA_MIN = 0.12
COLLIDER_VERTICAL_HEIGHT_MIN = 0.32
COLLIDER_MIN_THICKNESS = 0.035
FLOOR_COLLIDER_THICKNESS = 0.05
WALL_COLLIDER_THICKNESS = 0.08
COLLIDER_SUPPORT_PATCH_MIN_SPAN = 0.10
COLLIDER_BLOCKER_BAND_GAP_Y = 0.18
COLLIDER_BLOCKER_AREA_MIN = 0.02
COLLIDER_BLOCKER_SPAN_MIN = 0.05


@dataclass
class SceneColliderBox:
    mins: np.ndarray
    maxs: np.ndarray


@dataclass
class SimpleLabLayout:
    table_top_center: np.ndarray
    table_size: np.ndarray
    floor_z: float
    room_half_extent: np.ndarray
    wall_height: float
    scene_up: np.ndarray
    room_center_xy: np.ndarray | None = None
    static_collider_boxes: np.ndarray | None = None
    active_table_bounds: np.ndarray | None = None
    active_table_surface_center: np.ndarray | None = None

    @property
    def scene_down(self) -> np.ndarray:
        return -self.scene_up

    @property
    def table_top_z(self) -> float:
        return float(self.table_top_center[2])

    @property
    def table_box(self) -> SceneColliderBox:
        if self.active_table_bounds is not None:
            bounds = np.asarray(self.active_table_bounds, dtype=np.float32)
            return SceneColliderBox(
                mins=bounds[0].copy(),
                maxs=bounds[1].copy(),
            )
        table_half = self.table_size[:2] * 0.5
        mins = np.array(
            [
                self.table_top_center[0] - table_half[0],
                self.table_top_center[1] - table_half[1],
                self.table_top_z
                if float(self.scene_up[2]) < 0.0
                else self.table_top_z - self.table_size[2],
            ],
            dtype=np.float32,
        )
        maxs = np.array(
            [
                self.table_top_center[0] + table_half[0],
                self.table_top_center[1] + table_half[1],
                self.table_top_z + self.table_size[2]
                if float(self.scene_up[2]) < 0.0
                else self.table_top_z,
            ],
            dtype=np.float32,
        )
        return SceneColliderBox(mins=mins, maxs=maxs)

    @property
    def floor_box(self) -> SceneColliderBox:
        room_center_xy = (
            np.asarray(self.room_center_xy, dtype=np.float32)
            if self.room_center_xy is not None
            else np.asarray(self.table_top_center[:2], dtype=np.float32)
        )
        mins = np.array(
            [
                room_center_xy[0] - self.room_half_extent[0],
                room_center_xy[1] - self.room_half_extent[1],
                self.floor_z if float(self.scene_up[2]) < 0.0 else self.floor_z - 0.06,
            ],
            dtype=np.float32,
        )
        maxs = np.array(
            [
                room_center_xy[0] + self.room_half_extent[0],
                room_center_xy[1] + self.room_half_extent[1],
                self.floor_z + 0.06 if float(self.scene_up[2]) < 0.0 else self.floor_z,
            ],
            dtype=np.float32,
        )
        return SceneColliderBox(mins=mins, maxs=maxs)


def _normalize(vec: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm < eps:
        raise ValueError("Cannot normalize near-zero vector")
    return vec / norm


def _cluster_sorted_value_groups(values: np.ndarray, gap: float) -> list[np.ndarray]:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return []
    order = np.argsort(values)
    ordered_values = values[order]
    groups: list[np.ndarray] = []
    start = 0
    for idx in range(1, ordered_values.shape[0] + 1):
        if idx == ordered_values.shape[0] or abs(float(ordered_values[idx] - ordered_values[idx - 1])) > gap:
            groups.append(order[start:idx].copy())
            start = idx
    return groups


def _cluster_support_levels(
    heights: np.ndarray,
    areas: np.ndarray,
    bounds_mins: np.ndarray,
    bounds_maxs: np.ndarray,
    centers: np.ndarray,
    gap: float = TABLE_SUPPORT_GAP_Y,
) -> list[dict[str, Any]]:
    if heights.size == 0:
        return []
    order = np.argsort(heights)
    heights = heights[order]
    areas = areas[order]
    bounds_mins = bounds_mins[order]
    bounds_maxs = bounds_maxs[order]
    centers = centers[order]
    levels: list[dict[str, Any]] = []
    start = 0
    for idx in range(1, heights.shape[0] + 1):
        if idx == heights.shape[0] or abs(float(heights[idx] - heights[idx - 1])) > gap:
            seg = slice(start, idx)
            seg_areas = areas[seg]
            total_area = float(np.sum(seg_areas))
            if total_area > 1e-6:
                weights = seg_areas / total_area
                level_center = np.sum(centers[seg] * weights[:, None], axis=0)
                level_bounds_min = np.min(bounds_mins[seg], axis=0)
                level_bounds_max = np.max(bounds_maxs[seg], axis=0)
                levels.append(
                    {
                        "y": float(np.sum(heights[seg] * weights)),
                        "area": total_area,
                        "center": level_center.astype(np.float32),
                        "bounds_min": level_bounds_min.astype(np.float32),
                        "bounds_max": level_bounds_max.astype(np.float32),
                    }
                )
            start = idx
    levels.sort(key=lambda level: (level["area"], level["y"]), reverse=True)
    return levels


def _split_face_group_by_connectivity(
    face_indices: np.ndarray,
    face_adjacency: np.ndarray,
) -> list[np.ndarray]:
    face_indices = np.asarray(face_indices, dtype=np.int64)
    if face_indices.size == 0:
        return []
    if face_indices.size == 1:
        return [face_indices.copy()]

    face_lookup = {int(face_id): idx for idx, face_id in enumerate(face_indices.tolist())}
    parent = np.arange(face_indices.shape[0], dtype=np.int64)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for edge in np.asarray(face_adjacency, dtype=np.int64):
        local_a = face_lookup.get(int(edge[0]))
        local_b = face_lookup.get(int(edge[1]))
        if local_a is None or local_b is None:
            continue
        union(local_a, local_b)

    groups: dict[int, list[int]] = {}
    for local_idx, face_id in enumerate(face_indices.tolist()):
        root = find(int(local_idx))
        groups.setdefault(root, []).append(int(face_id))
    return [np.asarray(group, dtype=np.int64) for group in groups.values()]


def _build_support_patches(
    mesh: trimesh.Trimesh,
    face_indices: np.ndarray,
    face_centroids: np.ndarray,
    face_areas: np.ndarray,
    gap: float = TABLE_SUPPORT_GAP_Y,
) -> list[dict[str, Any]]:
    face_indices = np.asarray(face_indices, dtype=np.int64)
    if face_indices.size == 0:
        return []

    face_adjacency = np.asarray(mesh.face_adjacency, dtype=np.int64)
    level_groups = _cluster_sorted_value_groups(face_centroids[face_indices, 1], gap=gap)
    support_patches: list[dict[str, Any]] = []
    patch_index = 0
    for level_id, level_group in enumerate(level_groups):
        level_face_indices = face_indices[level_group]
        patch_groups = _split_face_group_by_connectivity(level_face_indices, face_adjacency)
        for patch_face_indices in patch_groups:
            patch_areas = np.asarray(face_areas[patch_face_indices], dtype=np.float32)
            total_area = float(np.sum(patch_areas))
            if total_area <= 1e-6:
                continue
            patch_centers = np.asarray(face_centroids[patch_face_indices], dtype=np.float32)
            weights = patch_areas / total_area
            patch_vertices = np.asarray(
                mesh.vertices[mesh.faces[patch_face_indices].reshape(-1)],
                dtype=np.float32,
            )
            patch_bounds_min = patch_vertices.min(axis=0)
            patch_bounds_max = patch_vertices.max(axis=0)
            patch_center = np.sum(patch_centers * weights[:, None], axis=0)
            support_patches.append(
                {
                    "patch_index": int(patch_index),
                    "level_id": int(level_id),
                    "face_count": int(patch_face_indices.shape[0]),
                    "face_indices": patch_face_indices.astype(np.int64),
                    "y": float(np.sum(patch_centers[:, 1] * weights)),
                    "area": total_area,
                    "center": patch_center.astype(np.float32),
                    "bounds_min": patch_bounds_min.astype(np.float32),
                    "bounds_max": patch_bounds_max.astype(np.float32),
                }
            )
            patch_index += 1
    support_patches.sort(key=lambda patch: (patch["area"], patch["y"]), reverse=True)
    return support_patches


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    ones = np.ones((points.shape[0], 1), dtype=np.float32)
    homogeneous = np.concatenate([points, ones], axis=1)
    transformed = (transform @ homogeneous.T).T
    return transformed[:, :3].astype(np.float32)


def _transform_bounds(
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    transform: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    bounds_min = np.asarray(bounds_min, dtype=np.float32)
    bounds_max = np.asarray(bounds_max, dtype=np.float32)
    corners = np.array(
        [
            [bounds_min[0], bounds_min[1], bounds_min[2]],
            [bounds_min[0], bounds_min[1], bounds_max[2]],
            [bounds_min[0], bounds_max[1], bounds_min[2]],
            [bounds_min[0], bounds_max[1], bounds_max[2]],
            [bounds_max[0], bounds_min[1], bounds_min[2]],
            [bounds_max[0], bounds_min[1], bounds_max[2]],
            [bounds_max[0], bounds_max[1], bounds_min[2]],
            [bounds_max[0], bounds_max[1], bounds_max[2]],
        ],
        dtype=np.float32,
    )
    transformed = _transform_points(corners, transform)
    return (
        transformed.min(axis=0).astype(np.float32),
        transformed.max(axis=0).astype(np.float32),
    )


def _expand_bounds_min_thickness(
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    min_thickness: float = COLLIDER_MIN_THICKNESS,
    support_axis: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    bounds_min = np.asarray(bounds_min, dtype=np.float32).copy()
    bounds_max = np.asarray(bounds_max, dtype=np.float32).copy()
    extents = bounds_max - bounds_min
    for axis in range(3):
        if extents[axis] >= min_thickness:
            continue
        if support_axis is not None and axis == int(support_axis):
            bounds_min[axis] = bounds_max[axis] - min_thickness
        else:
            pad = 0.5 * (min_thickness - extents[axis])
            bounds_min[axis] -= pad
            bounds_max[axis] += pad
    return bounds_min, bounds_max


def _support_patch_is_collision_usable(
    patch: dict[str, Any],
    floor_y: float,
) -> bool:
    if float(patch["y"]) <= float(floor_y + TABLE_SUPPORT_HEIGHT_MIN):
        return False
    if float(patch["area"]) < float(TABLE_SUPPORT_AREA_MIN):
        return False
    extents = np.asarray(patch["bounds_max"] - patch["bounds_min"], dtype=np.float32)
    return min(float(extents[0]), float(extents[2])) >= COLLIDER_SUPPORT_PATCH_MIN_SPAN


def _build_blocker_groups(
    mesh: trimesh.Trimesh,
    component_face_indices: np.ndarray,
    support_patches: list[dict[str, Any]],
    face_centroids: np.ndarray,
    face_areas: np.ndarray,
    gap: float = COLLIDER_BLOCKER_BAND_GAP_Y,
) -> list[dict[str, Any]]:
    component_face_indices = np.asarray(component_face_indices, dtype=np.int64)
    if component_face_indices.size == 0:
        return []

    if support_patches:
        support_face_indices = np.unique(
            np.concatenate(
                [
                    np.asarray(patch["face_indices"], dtype=np.int64)
                    for patch in support_patches
                    if np.asarray(patch["face_indices"], dtype=np.int64).size > 0
                ],
                axis=0,
            )
        )
        blocker_face_indices = np.setdiff1d(
            component_face_indices,
            support_face_indices,
            assume_unique=False,
        )
    else:
        blocker_face_indices = component_face_indices.copy()

    if blocker_face_indices.size == 0:
        return []

    face_adjacency = np.asarray(mesh.face_adjacency, dtype=np.int64)
    height_groups = _cluster_sorted_value_groups(
        face_centroids[blocker_face_indices, 1],
        gap=gap,
    )
    blocker_groups: list[dict[str, Any]] = []
    blocker_index = 0
    for height_group in height_groups:
        band_face_indices = blocker_face_indices[height_group]
        connected_groups = _split_face_group_by_connectivity(
            band_face_indices,
            face_adjacency,
        )
        for group_face_indices in connected_groups:
            group_areas = np.asarray(face_areas[group_face_indices], dtype=np.float32)
            total_area = float(np.sum(group_areas))
            if total_area <= 1e-6:
                continue
            group_vertices = np.asarray(
                mesh.vertices[mesh.faces[group_face_indices].reshape(-1)],
                dtype=np.float32,
            )
            bounds_min = group_vertices.min(axis=0)
            bounds_max = group_vertices.max(axis=0)
            extents = bounds_max - bounds_min
            if (
                total_area < COLLIDER_BLOCKER_AREA_MIN
                and float(max(extents[0], extents[1], extents[2])) < COLLIDER_BLOCKER_SPAN_MIN
            ):
                continue
            group_centers = np.asarray(face_centroids[group_face_indices], dtype=np.float32)
            weights = group_areas / total_area
            center = np.sum(group_centers * weights[:, None], axis=0)
            blocker_groups.append(
                {
                    "blocker_index": int(blocker_index),
                    "face_count": int(group_face_indices.shape[0]),
                    "face_indices": group_face_indices.astype(np.int64),
                    "area": total_area,
                    "center": center.astype(np.float32),
                    "bounds_min": bounds_min.astype(np.float32),
                    "bounds_max": bounds_max.astype(np.float32),
                    "y": float(np.sum(group_centers[:, 1] * weights)),
                }
            )
            blocker_index += 1
    blocker_groups.sort(key=lambda group: (group["area"], group["y"]), reverse=True)
    return blocker_groups


def illixr_lab_manifest_path(scene_assets_root: str | Path) -> Path:
    assets_root = Path(scene_assets_root).resolve()
    candidates = []
    if assets_root.name == ILLIXR_SCENE_NAME:
        candidates.append(assets_root / "manifest.json")
    candidates.extend(
        [
            assets_root / ILLIXR_SCENE_NAME / "manifest.json",
            assets_root / "scenes" / ILLIXR_SCENE_NAME / "manifest.json",
            assets_root / "assets" / "scenes" / ILLIXR_SCENE_NAME / "manifest.json",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if assets_root.name == ILLIXR_SCENE_NAME:
        return assets_root / "manifest.json"
    if assets_root.name == "scenes":
        return assets_root / ILLIXR_SCENE_NAME / "manifest.json"
    if assets_root.name == "assets":
        return assets_root / "scenes" / ILLIXR_SCENE_NAME / "manifest.json"
    return assets_root / ILLIXR_SCENE_NAME / "manifest.json"


def simple_lab_manifest_path(scene_assets_root: str | Path) -> Path:
    return illixr_lab_manifest_path(scene_assets_root)


def load_illixr_lab_manifest(scene_assets_root: str | Path) -> dict[str, Any]:
    manifest_path = illixr_lab_manifest_path(scene_assets_root)
    with open(manifest_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_simple_lab_manifest(scene_assets_root: str | Path) -> dict[str, Any]:
    return load_illixr_lab_manifest(scene_assets_root)


def ensure_illixr_lab_assets(scene_assets_root: str | Path) -> Path:
    assets_root = Path(scene_assets_root).resolve()
    manifest_path = illixr_lab_manifest_path(assets_root)
    scene_root = manifest_path.parent
    manifest = load_illixr_lab_manifest(assets_root)
    required_paths = [
        scene_root / manifest["scene_model"],
        scene_root / manifest["scene_material"],
    ]
    required_paths.extend(scene_root / texture for texture in manifest.get("textures", []))
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "The shipped ILLIXR_lab scene bundle is incomplete. Missing files:\n"
            + "\n".join(missing)
        )
    return scene_root


def ensure_simple_lab_assets(scene_assets_root: str | Path) -> Path:
    return ensure_illixr_lab_assets(scene_assets_root)


def make_illixr_lab_layout(
    head_position: np.ndarray,
    forward_direction: np.ndarray,
    scene_up: np.ndarray | None = None,
) -> SimpleLabLayout:
    head_position = np.asarray(head_position, dtype=np.float32)
    forward_direction = np.asarray(forward_direction, dtype=np.float32)
    if scene_up is None:
        scene_up = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    else:
        scene_up = np.asarray(scene_up, dtype=np.float32)
    scene_up = _normalize(scene_up)

    horizontal_forward = forward_direction - float(np.dot(forward_direction, scene_up)) * scene_up
    norm = float(np.linalg.norm(horizontal_forward))
    if norm < 1e-5:
        horizontal_forward = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        horizontal_forward = horizontal_forward - float(
            np.dot(horizontal_forward, scene_up)
        ) * scene_up
        fallback_norm = float(np.linalg.norm(horizontal_forward))
        if fallback_norm < 1e-5:
            horizontal_forward = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            horizontal_forward /= fallback_norm
    else:
        horizontal_forward /= norm

    scene_down = -scene_up
    table_top_center = head_position + horizontal_forward * 0.78 + scene_down * 0.62
    floor_point = head_position + scene_down * 1.35
    return SimpleLabLayout(
        table_top_center=table_top_center.astype(np.float32),
        table_size=np.array([TARGET_TABLE_SIZE_X, TARGET_TABLE_SIZE_Y, 0.40], dtype=np.float32),
        floor_z=float(floor_point[2]),
        room_half_extent=np.array([2.7, 2.7], dtype=np.float32),
        wall_height=2.8,
        scene_up=scene_up.astype(np.float32),
        room_center_xy=np.asarray(table_top_center[:2], dtype=np.float32).copy(),
        static_collider_boxes=None,
    )


def make_simple_lab_layout(
    head_position: np.ndarray,
    forward_direction: np.ndarray,
    scene_up: np.ndarray | None = None,
) -> SimpleLabLayout:
    return make_illixr_lab_layout(
        head_position,
        forward_direction,
        scene_up=scene_up,
    )


def _as_trimesh(mesh_or_scene: trimesh.Trimesh | trimesh.Scene) -> trimesh.Trimesh:
    if isinstance(mesh_or_scene, trimesh.Trimesh):
        return mesh_or_scene.copy()
    geometries = [geom.copy() for geom in mesh_or_scene.dump(concatenate=False)]
    if not geometries:
        raise ValueError("No geometry found in immersive scene asset.")
    return trimesh.util.concatenate(geometries)


class SimpleLabSceneRenderer:
    def __init__(
        self,
        scene_assets_root: str | Path,
        width: int,
        height: int,
        lighting_mode: str = "full",
        balanced_render_backend: str = "pyrender",
        scene_analysis_cache_mode: str = "auto",
    ):
        import pyrender

        self.width = int(width)
        self.height = int(height)
        self.requested_lighting_mode = str(lighting_mode)
        self.lighting_mode = self._resolve_lighting_mode(lighting_mode)
        self.balanced_render_backend = "pyrender"
        self._pyrender = pyrender
        self.scene_root = ensure_illixr_lab_assets(scene_assets_root)
        self.manifest = load_illixr_lab_manifest(scene_assets_root)
        self.layout: SimpleLabLayout | None = None
        self._scene_analysis_cache_mode = str(scene_analysis_cache_mode).strip().lower()
        if self._scene_analysis_cache_mode not in {"auto", "rebuild"}:
            raise ValueError(
                "scene_analysis_cache_mode must be one of {'auto', 'rebuild'}"
            )

        self._scene_clear_color = np.array([243, 244, 246, 255], dtype=np.uint8)
        self._table_clear_color = np.array([0, 0, 0, 0], dtype=np.uint8)
        self._ambient_light = self._ambient_light_for_mode(self.lighting_mode)
        self._layer_specs = {
            "full": {
                "bg_color": self._scene_clear_color,
                "table_role": "full",
                "background_role": "all",
            },
            "background": {
                "bg_color": self._scene_clear_color,
                "table_role": None,
                "background_role": "all",
            },
            "table": {
                "bg_color": self._table_clear_color,
                "table_role": "full",
                "background_role": None,
            },
            "balanced_far_front_back_walls": {
                "bg_color": self._scene_clear_color,
                "table_role": None,
                "background_role": "front_back_walls",
            },
            "balanced_left_wall": {
                "bg_color": self._scene_clear_color,
                "table_role": None,
                "background_role": "left_wall",
            },
            "balanced_right_wall": {
                "bg_color": self._scene_clear_color,
                "table_role": None,
                "background_role": "right_wall",
            },
            "balanced_near_floor": {
                "bg_color": self._scene_clear_color,
                "table_role": None,
                "background_role": "floor",
            },
        }
        self._eager_layer_names = {"background", "table"}
        self._layer_entries: dict[str, dict[str, Any]] = {}
        self._table_alignment_debug: dict[str, Any] | None = None
        self._table_world_bounds: tuple[np.ndarray, np.ndarray] | None = None
        self._wall_world_bounds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._scene_collider_boxes: np.ndarray | None = None
        self._pyrender_readback_mode = "cpu_fallback"
        self._pyrender_readback_reason: str | None = None
        self._pyrender_cuda_interop_supported = False
        (
            self._pyrender_cuda_interop_supported,
            self._pyrender_readback_reason,
        ) = probe_pyrender_cuda_bridge_support()
        self._pyrender_readback_mode = (
            "gl_cuda_interop"
            if self._pyrender_cuda_interop_supported
            else "cpu_fallback"
        )

        for layer_name in self._eager_layer_names:
            self._layer_entries[layer_name] = self._make_layer_entry(layer_name)

        self._asset_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self._scene_x_rotation_degrees = float(self.manifest.get("x_rotation_degrees", -90.0))
        self._target_table_size_xy = np.asarray(
            self.manifest.get("target_table_size_m", [TARGET_TABLE_SIZE_X, TARGET_TABLE_SIZE_Y]),
            dtype=np.float32,
        )
        self._scene_asset = None
        self._scene_analysis_cache_path = self._resolve_scene_analysis_cache_path()
        self._scene_analysis_cache_input_hash = self._compute_scene_analysis_cache_input_hash()
        self._scene_analysis_cache_debug = {
            "status": "miss",
            "reason": "not_attempted",
            "schema": int(SCENE_ANALYSIS_CACHE_SCHEMA),
            "input_hash": self._scene_analysis_cache_input_hash,
            "path": None
            if self._scene_analysis_cache_path is None
            else str(self._scene_analysis_cache_path),
        }
        cache_loaded = False
        if self._scene_analysis_cache_mode == "auto":
            cache_loaded = self._load_scene_analysis_cache()
        else:
            self._scene_analysis_cache_debug["reason"] = "forced_rebuild"
        if not cache_loaded:
            self._build_scene_analysis()
            self._scene_analysis_cache_debug.update(
                {
                    "status": "miss",
                    "reason": str(self._scene_analysis_cache_debug.get("reason", "rebuilt")),
                    "schema": int(SCENE_ANALYSIS_CACHE_SCHEMA),
                    "input_hash": self._scene_analysis_cache_input_hash,
                }
            )
        self._full_scene_mesh_world: trimesh.Trimesh | None = None
        self._background_mesh_world: trimesh.Trimesh | None = None
        self._table_mesh_world: trimesh.Trimesh | None = None
        self._floor_mesh_world: trimesh.Trimesh | None = None
        self._left_wall_mesh_world: trimesh.Trimesh | None = None
        self._right_wall_mesh_world: trimesh.Trimesh | None = None
        self._front_back_mesh_world: trimesh.Trimesh | None = None

    def _resolve_scene_analysis_cache_path(self) -> Path:
        relative_path = self.manifest.get(
            "scene_analysis_cache",
            DEFAULT_SCENE_ANALYSIS_CACHE_FILENAME,
        )
        return self.scene_root / str(relative_path)

    def _scene_analysis_cache_manifest_subset(self) -> dict[str, Any]:
        return {
            key: self.manifest.get(key)
            for key in (
                "version",
                "scene_preset",
                "scene_model",
                "scene_material",
                "textures",
                "asset_up_axis",
                "scene_up_axis",
                "x_rotation_degrees",
                "target_table_size_m",
                "floor_materials",
                "wall_materials",
                "furniture_materials",
            )
        }

    def _compute_scene_analysis_cache_input_hash(self) -> str:
        hasher = hashlib.sha256()
        hasher.update(
            json.dumps(
                self._scene_analysis_cache_manifest_subset(),
                sort_keys=True,
            ).encode("utf-8")
        )
        for manifest_key in ("scene_model", "scene_material"):
            rel_path = self.manifest.get(manifest_key)
            if rel_path is None:
                continue
            asset_path = self.scene_root / str(rel_path)
            hasher.update(manifest_key.encode("utf-8"))
            hasher.update(asset_path.read_bytes())
        texture_names = [
            str(name)
            for name in self.manifest.get("textures", [])
        ]
        hasher.update(json.dumps(texture_names, sort_keys=False).encode("utf-8"))
        return hasher.hexdigest()

    def scene_analysis_cache_debug(self) -> dict[str, Any]:
        return dict(self._scene_analysis_cache_debug)

    def _build_scene_analysis(self) -> None:
        self._scene_asset = trimesh.load(
            self.scene_root / self.manifest["scene_model"],
            force="scene",
            process=False,
        )
        self._floor_asset_mesh = self._concat_geometries(self.manifest.get("floor_materials", []))
        self._wall_asset_mesh = self._concat_geometries(self.manifest.get("wall_materials", []))
        self._furniture_asset_mesh = self._concat_geometries(
            self.manifest.get("furniture_materials", [])
        )
        self._full_asset_mesh = _as_trimesh(self._scene_asset)

        self._asset_floor_y = float(np.median(self._floor_asset_mesh.vertices[:, 1]))
        self._asset_room_bounds = self._full_asset_mesh.bounds.astype(np.float32)
        self._asset_room_center_xz = np.array(
            [
                0.5 * float(self._asset_room_bounds[0, 0] + self._asset_room_bounds[1, 0]),
                0.5 * float(self._asset_room_bounds[0, 2] + self._asset_room_bounds[1, 2]),
            ],
            dtype=np.float32,
        )

        self._wall_face_masks = self._split_wall_faces(self._wall_asset_mesh)
        self._furniture_component_labels = connected_component_labels(
            self._furniture_asset_mesh.face_adjacency,
            node_count=len(self._furniture_asset_mesh.faces),
        )
        self._furniture_component_records = self._build_component_records(
            self._furniture_asset_mesh,
            self._furniture_component_labels,
        )
        (
            self._table_component_ids,
            self._table_support_component_ids,
            self._asset_table_scale_reference_bounds,
            self._asset_table_scale_reference_center,
            self._asset_table_scale_reference_height,
            self._asset_startup_table_patch,
            self._asset_visible_tabletop_patches,
        ) = self._select_table_components(self._furniture_component_records)
        self._table_asset_mesh = self._slice_mesh_by_component_ids(
            self._furniture_asset_mesh,
            self._furniture_component_labels,
            self._table_component_ids,
        )
        self._startup_table_asset_mesh = self._slice_mesh_by_face_indices(
            self._furniture_asset_mesh,
            np.asarray(self._asset_startup_table_patch["face_indices"], dtype=np.int64),
        )
        self._visible_tabletop_asset_mesh = self._slice_mesh_by_face_indices(
            self._furniture_asset_mesh,
            np.concatenate(
                [
                    np.asarray(patch["face_indices"], dtype=np.int64)
                    for patch in self._asset_visible_tabletop_patches
                ],
                axis=0,
            ),
        )
        self._left_wall_asset_mesh = self._slice_mesh_by_face_mask(
            self._wall_asset_mesh,
            self._wall_face_masks["left"],
        )
        self._right_wall_asset_mesh = self._slice_mesh_by_face_mask(
            self._wall_asset_mesh,
            self._wall_face_masks["right"],
        )
        self._front_back_asset_mesh = self._slice_mesh_by_face_mask(
            self._wall_asset_mesh,
            self._wall_face_masks["front"] | self._wall_face_masks["back"],
        )
        self._collision_component_records = self._select_collision_component_records(
            self._furniture_component_records,
            self._table_component_ids,
        )

    def _export_scene_analysis_payload(self) -> dict[str, Any]:
        return {
            "floor_asset_mesh": self._floor_asset_mesh.copy(),
            "wall_asset_mesh": self._wall_asset_mesh.copy(),
            "furniture_asset_mesh": self._furniture_asset_mesh.copy(),
            "asset_floor_y": float(self._asset_floor_y),
            "asset_room_bounds": np.array(self._asset_room_bounds, copy=True),
            "asset_room_center_xz": np.array(self._asset_room_center_xz, copy=True),
            "wall_face_masks": {
                key: np.array(value, copy=True)
                for key, value in self._wall_face_masks.items()
            },
            "furniture_component_records": self._furniture_component_records,
            "table_component_ids": list(self._table_component_ids),
            "table_support_component_ids": list(self._table_support_component_ids),
            "asset_table_scale_reference_bounds": (
                np.array(self._asset_table_scale_reference_bounds[0], copy=True),
                np.array(self._asset_table_scale_reference_bounds[1], copy=True),
            ),
            "asset_table_scale_reference_center": np.array(
                self._asset_table_scale_reference_center,
                copy=True,
            ),
            "asset_table_scale_reference_height": float(
                self._asset_table_scale_reference_height
            ),
            "asset_startup_table_patch": self._asset_startup_table_patch,
            "asset_visible_tabletop_patches": self._asset_visible_tabletop_patches,
        }

    def _apply_scene_analysis_payload(self, payload: dict[str, Any]) -> None:
        self._scene_asset = None
        self._floor_asset_mesh = payload["floor_asset_mesh"]
        self._wall_asset_mesh = payload["wall_asset_mesh"]
        self._furniture_asset_mesh = payload["furniture_asset_mesh"]
        self._full_asset_mesh = trimesh.util.concatenate(
            [
                self._floor_asset_mesh.copy(),
                self._wall_asset_mesh.copy(),
                self._furniture_asset_mesh.copy(),
            ]
        )
        self._asset_floor_y = float(payload["asset_floor_y"])
        self._asset_room_bounds = np.asarray(
            payload["asset_room_bounds"],
            dtype=np.float32,
        )
        self._asset_room_center_xz = np.asarray(
            payload["asset_room_center_xz"],
            dtype=np.float32,
        )
        self._wall_face_masks = {
            str(key): np.asarray(value, dtype=bool)
            for key, value in payload["wall_face_masks"].items()
        }
        self._furniture_component_labels = None
        self._furniture_component_records = payload["furniture_component_records"]
        self._table_component_ids = [int(v) for v in payload["table_component_ids"]]
        self._table_support_component_ids = [
            int(v) for v in payload["table_support_component_ids"]
        ]
        self._asset_table_scale_reference_bounds = (
            np.asarray(
                payload["asset_table_scale_reference_bounds"][0],
                dtype=np.float32,
            ),
            np.asarray(
                payload["asset_table_scale_reference_bounds"][1],
                dtype=np.float32,
            ),
        )
        self._asset_table_scale_reference_center = np.asarray(
            payload["asset_table_scale_reference_center"],
            dtype=np.float32,
        )
        self._asset_table_scale_reference_height = float(
            payload["asset_table_scale_reference_height"]
        )
        self._asset_startup_table_patch = payload["asset_startup_table_patch"]
        self._asset_visible_tabletop_patches = payload["asset_visible_tabletop_patches"]
        table_face_indices = np.concatenate(
            [
                np.asarray(record["face_indices"], dtype=np.int64)
                for record in self._furniture_component_records
                if int(record["id"]) in set(self._table_component_ids)
            ],
            axis=0,
        )
        self._table_asset_mesh = self._slice_mesh_by_face_indices(
            self._furniture_asset_mesh,
            table_face_indices,
        )
        self._startup_table_asset_mesh = self._slice_mesh_by_face_indices(
            self._furniture_asset_mesh,
            np.asarray(self._asset_startup_table_patch["face_indices"], dtype=np.int64),
        )
        self._visible_tabletop_asset_mesh = self._slice_mesh_by_face_indices(
            self._furniture_asset_mesh,
            np.concatenate(
                [
                    np.asarray(patch["face_indices"], dtype=np.int64)
                    for patch in self._asset_visible_tabletop_patches
                ],
                axis=0,
            ),
        )
        self._left_wall_asset_mesh = self._slice_mesh_by_face_mask(
            self._wall_asset_mesh,
            self._wall_face_masks["left"],
        )
        self._right_wall_asset_mesh = self._slice_mesh_by_face_mask(
            self._wall_asset_mesh,
            self._wall_face_masks["right"],
        )
        self._front_back_asset_mesh = self._slice_mesh_by_face_mask(
            self._wall_asset_mesh,
            self._wall_face_masks["front"] | self._wall_face_masks["back"],
        )
        self._collision_component_records = self._select_collision_component_records(
            self._furniture_component_records,
            self._table_component_ids,
        )

    def _load_scene_analysis_cache(self) -> bool:
        cache_path = self._scene_analysis_cache_path
        expected_manifest_schema = int(
            self.manifest.get(
                "scene_analysis_cache_schema",
                SCENE_ANALYSIS_CACHE_SCHEMA,
            )
        )
        if expected_manifest_schema != int(SCENE_ANALYSIS_CACHE_SCHEMA):
            self._scene_analysis_cache_debug["reason"] = "manifest_cache_schema_mismatch"
            return False
        if cache_path is None:
            self._scene_analysis_cache_debug["reason"] = "cache_path_unavailable"
            return False
        if not cache_path.exists():
            self._scene_analysis_cache_debug["reason"] = "cache_file_missing"
            return False
        try:
            with gzip.open(cache_path, "rb") as handle:
                payload = pickle.load(handle)
        except Exception as exc:
            self._scene_analysis_cache_debug["reason"] = (
                f"cache_load_error:{exc.__class__.__name__}"
            )
            return False
        if int(payload.get("schema", -1)) != int(SCENE_ANALYSIS_CACHE_SCHEMA):
            self._scene_analysis_cache_debug["reason"] = "cache_schema_mismatch"
            return False
        if str(payload.get("input_hash", "")) != self._scene_analysis_cache_input_hash:
            self._scene_analysis_cache_debug["reason"] = "cache_input_hash_mismatch"
            return False
        if int(payload.get("manifest_version", -1)) != int(self.manifest.get("version", -1)):
            self._scene_analysis_cache_debug["reason"] = "cache_manifest_version_mismatch"
            return False
        analysis = payload.get("analysis")
        if not isinstance(analysis, dict):
            self._scene_analysis_cache_debug["reason"] = "cache_payload_invalid"
            return False
        self._apply_scene_analysis_payload(analysis)
        self._scene_analysis_cache_debug.update(
            {
                "status": "hit",
                "reason": "cache_loaded",
                "schema": int(payload.get("schema", SCENE_ANALYSIS_CACHE_SCHEMA)),
                "input_hash": str(payload.get("input_hash", "")),
            }
        )
        return True

    def write_scene_analysis_cache(
        self,
        output_path: str | Path | None = None,
    ) -> Path:
        cache_path = (
            Path(output_path)
            if output_path is not None
            else self._scene_analysis_cache_path
        )
        if cache_path is None:
            raise RuntimeError("Scene analysis cache path is unavailable.")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": int(SCENE_ANALYSIS_CACHE_SCHEMA),
            "manifest_version": int(self.manifest.get("version", -1)),
            "input_hash": self._scene_analysis_cache_input_hash,
            "analysis": self._export_scene_analysis_payload(),
        }
        with gzip.open(cache_path, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return cache_path

    def _concat_geometries(self, names: list[str]) -> trimesh.Trimesh:
        geometries: list[trimesh.Trimesh] = []
        for name in names:
            geom = self._scene_asset.geometry.get(name)
            if geom is not None:
                geometries.append(geom.copy())
        if not geometries:
            raise ValueError(f"Could not resolve scene geometry group: {names}")
        return trimesh.util.concatenate(geometries)

    def _resolve_lighting_mode(self, lighting_mode: str) -> str:
        requested = str(lighting_mode)
        if requested == ILLIXR_BAKED_LIGHTING_MODE:
            return requested
        # The vendored room already contains a baked look in its textures, so
        # favor a bright ambient presentation over extra relighting.
        return ILLIXR_BAKED_LIGHTING_MODE

    def _ambient_light_for_mode(self, lighting_mode: str) -> np.ndarray:
        if str(lighting_mode) == ILLIXR_BAKED_LIGHTING_MODE:
            return np.array([0.98, 0.98, 0.98], dtype=np.float32)
        if str(lighting_mode) == "simple":
            return np.array([0.40, 0.40, 0.40], dtype=np.float32)
        return np.array([0.22, 0.22, 0.22], dtype=np.float32)

    def _make_layer_entry(self, layer_name: str) -> dict[str, Any]:
        layer_spec = self._layer_specs[layer_name]
        scene = self._pyrender.Scene(
            bg_color=np.array(layer_spec["bg_color"], copy=True),
            ambient_light=np.array(self._ambient_light, copy=True),
        )
        camera = self._pyrender.IntrinsicsCamera(
            fx=1.0,
            fy=1.0,
            cx=float(self.width) * 0.5,
            cy=float(self.height) * 0.5,
            znear=0.02,
            zfar=100.0,
        )
        camera_node = scene.add(camera, pose=np.eye(4, dtype=np.float32))
        self._setup_lights(scene)
        return {
            "scene": scene,
            "camera": camera,
            "camera_node": camera_node,
            "renderer": None,
            "table_node": None,
            "floor_node": None,
            "wall_nodes": [],
            "background_nodes": [],
            "table_role": layer_spec["table_role"],
            "background_role": layer_spec["background_role"],
        }

    def _ensure_layer_entry(self, layer_name: str) -> dict[str, Any]:
        entry = self._layer_entries.get(layer_name)
        if entry is not None:
            return entry
        entry = self._make_layer_entry(layer_name)
        self._layer_entries[layer_name] = entry
        if self.layout is not None:
            self._rebuild_scene_nodes()
        return entry

    def delete(self) -> None:
        for entry in self._layer_entries.values():
            renderer = entry.get("renderer")
            if renderer is not None:
                renderer.delete()
                entry["renderer"] = None

    def pyrender_readback_mode(self) -> str:
        return str(self._pyrender_readback_mode)

    def pyrender_readback_reason(self) -> str | None:
        if self._pyrender_readback_reason is None:
            return None
        return str(self._pyrender_readback_reason)

    def uses_gpu_balanced_table_renderer(self) -> bool:
        return False

    def uses_gpu_balanced_plane_renderer(self) -> bool:
        return False

    def uses_gpu_balanced_side_wall_renderer(self) -> bool:
        return False

    def table_alignment_debug(self) -> dict[str, Any] | None:
        if self._table_alignment_debug is None:
            return None
        return dict(self._table_alignment_debug)

    def table_world_bounds(self) -> tuple[np.ndarray, np.ndarray] | None:
        if self._table_world_bounds is None:
            return None
        return (
            np.array(self._table_world_bounds[0], copy=True),
            np.array(self._table_world_bounds[1], copy=True),
        )

    def wall_world_bounds(self, wall_name: str) -> tuple[np.ndarray, np.ndarray] | None:
        wall_key = str(wall_name)
        if wall_key not in self._wall_world_bounds:
            return None
        bounds_min, bounds_max = self._wall_world_bounds[wall_key]
        return np.array(bounds_min, copy=True), np.array(bounds_max, copy=True)

    def static_collider_boxes(self) -> np.ndarray | None:
        if self._scene_collider_boxes is None:
            return None
        return np.array(self._scene_collider_boxes, copy=True)

    def set_layout(self, layout: SimpleLabLayout) -> None:
        self.layout = layout
        self._rebuild_scene_nodes()

    def render_eye(
        self,
        camera_pose_world: np.ndarray,
        intrinsic: np.ndarray,
        width: int | None = None,
        height: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._render_layer_eye("full", camera_pose_world, intrinsic, width=width, height=height)

    def render_background_eye(
        self,
        camera_pose_world: np.ndarray,
        intrinsic: np.ndarray,
        width: int | None = None,
        height: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._render_layer_eye("background", camera_pose_world, intrinsic, width=width, height=height)

    def render_background_eye_roi(
        self,
        camera_pose_world: np.ndarray,
        full_intrinsic: np.ndarray,
        roi_bounds: tuple[int, int, int, int],
    ) -> tuple[np.ndarray, np.ndarray] | tuple[torch.Tensor, torch.Tensor]:
        return self._render_layer_eye_roi("background", camera_pose_world, full_intrinsic, roi_bounds)

    def render_balanced_far_front_back_walls_eye(
        self,
        camera_pose_world: np.ndarray,
        intrinsic: np.ndarray,
        width: int | None = None,
        height: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[torch.Tensor, torch.Tensor]:
        return self._render_balanced_layer(
            "balanced_far_front_back_walls",
            camera_pose_world,
            intrinsic,
            width=width,
            height=height,
        )

    def render_balanced_far_walls_eye(
        self,
        camera_pose_world: np.ndarray,
        intrinsic: np.ndarray,
        width: int | None = None,
        height: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.render_balanced_far_front_back_walls_eye(
            camera_pose_world,
            intrinsic,
            width=width,
            height=height,
        )

    def render_balanced_left_wall_eye(
        self,
        camera_pose_world: np.ndarray,
        intrinsic: np.ndarray,
        width: int | None = None,
        height: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[torch.Tensor, torch.Tensor]:
        return self._render_balanced_layer(
            "balanced_left_wall",
            camera_pose_world,
            intrinsic,
            width=width,
            height=height,
        )

    def render_balanced_left_wall_eye_roi(
        self,
        camera_pose_world: np.ndarray,
        full_intrinsic: np.ndarray,
        roi_bounds: tuple[int, int, int, int],
    ) -> tuple[np.ndarray, np.ndarray] | tuple[torch.Tensor, torch.Tensor]:
        return self._render_balanced_layer_roi(
            "balanced_left_wall",
            camera_pose_world,
            full_intrinsic,
            roi_bounds,
        )

    def render_balanced_right_wall_eye(
        self,
        camera_pose_world: np.ndarray,
        intrinsic: np.ndarray,
        width: int | None = None,
        height: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[torch.Tensor, torch.Tensor]:
        return self._render_balanced_layer(
            "balanced_right_wall",
            camera_pose_world,
            intrinsic,
            width=width,
            height=height,
        )

    def render_balanced_right_wall_eye_roi(
        self,
        camera_pose_world: np.ndarray,
        full_intrinsic: np.ndarray,
        roi_bounds: tuple[int, int, int, int],
    ) -> tuple[np.ndarray, np.ndarray] | tuple[torch.Tensor, torch.Tensor]:
        return self._render_balanced_layer_roi(
            "balanced_right_wall",
            camera_pose_world,
            full_intrinsic,
            roi_bounds,
        )

    def render_balanced_near_floor_eye(
        self,
        camera_pose_world: np.ndarray,
        intrinsic: np.ndarray,
        width: int | None = None,
        height: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[torch.Tensor, torch.Tensor]:
        return self._render_balanced_layer(
            "balanced_near_floor",
            camera_pose_world,
            intrinsic,
            width=width,
            height=height,
        )

    def render_table_eye(
        self,
        camera_pose_world: np.ndarray,
        intrinsic: np.ndarray,
        width: int | None = None,
        height: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[torch.Tensor, torch.Tensor]:
        return self._render_layer_eye("table", camera_pose_world, intrinsic, width=width, height=height)

    def render_table_eye_roi(
        self,
        camera_pose_world: np.ndarray,
        full_intrinsic: np.ndarray,
        roi_bounds: tuple[int, int, int, int],
        render_scale: float = 1.0,
        return_render_info: bool = False,
    ) -> (
        tuple[np.ndarray, np.ndarray]
        | tuple[np.ndarray, np.ndarray, dict[str, Any]]
        | tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, dict[str, Any]]
    ):
        return self._render_layer_eye_roi(
            "table",
            camera_pose_world,
            full_intrinsic,
            roi_bounds,
            render_scale=render_scale,
            return_render_info=return_render_info,
        )

    def _render_balanced_layer(
        self,
        layer_name: str,
        camera_pose_world: np.ndarray,
        intrinsic: np.ndarray,
        width: int | None = None,
        height: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[torch.Tensor, torch.Tensor]:
        return self._render_layer_eye(layer_name, camera_pose_world, intrinsic, width=width, height=height)

    def _render_balanced_layer_roi(
        self,
        layer_name: str,
        camera_pose_world: np.ndarray,
        full_intrinsic: np.ndarray,
        roi_bounds: tuple[int, int, int, int],
    ) -> tuple[np.ndarray, np.ndarray] | tuple[torch.Tensor, torch.Tensor]:
        return self._render_layer_eye_roi(layer_name, camera_pose_world, full_intrinsic, roi_bounds)

    def _should_use_gpu_balanced_layer(self, layer_name: str) -> bool:
        _ = layer_name
        return False

    def _resolve_roi_render_params(
        self,
        full_intrinsic: np.ndarray,
        roi_bounds: tuple[int, int, int, int],
        render_scale: float = 1.0,
    ) -> tuple[np.ndarray, int, int, dict[str, Any]]:
        x0, y0, x1, y1 = [int(v) for v in roi_bounds]
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"Invalid ROI bounds: {roi_bounds}")
        roi_width = x1 - x0
        roi_height = y1 - y0
        render_scale = max(float(render_scale), 1.0)
        roi_intrinsic = np.array(full_intrinsic, dtype=np.float32, copy=True)
        roi_intrinsic[0, 2] -= float(x0)
        roi_intrinsic[1, 2] -= float(y0)
        render_width = roi_width
        render_height = roi_height
        if render_scale > 1.0:
            roi_intrinsic[0, 0] *= render_scale
            roi_intrinsic[1, 1] *= render_scale
            roi_intrinsic[0, 2] *= render_scale
            roi_intrinsic[1, 2] *= render_scale
            render_width = max(4, int(np.ceil((float(roi_width) * render_scale) / 4.0) * 4))
            render_height = max(4, int(np.ceil((float(roi_height) * render_scale) / 4.0) * 4))
        return (
            roi_intrinsic,
            render_width,
            render_height,
            {
                "roi_width": int(roi_width),
                "roi_height": int(roi_height),
                "render_width": int(render_width),
                "render_height": int(render_height),
                "render_scale": float(render_scale),
            },
        )

    def _setup_lights(self, scene) -> None:
        if self.lighting_mode == ILLIXR_BAKED_LIGHTING_MODE:
            return
        key_light = self._pyrender.DirectionalLight(
            color=np.ones(3, dtype=np.float32),
            intensity=3.5,
        )
        scene.add(key_light, pose=trimesh.transformations.euler_matrix(-0.7, 0.35, 0.0))
        if self.lighting_mode == "full":
            fill_light = self._pyrender.DirectionalLight(
                color=np.array([0.86, 0.90, 1.0], dtype=np.float32),
                intensity=1.4,
            )
            point_light = self._pyrender.PointLight(
                color=np.ones(3, dtype=np.float32),
                intensity=22.0,
            )
            scene.add(fill_light, pose=trimesh.transformations.euler_matrix(-1.1, -0.4, 0.0))
            scene.add(point_light, pose=trimesh.transformations.translation_matrix([0.0, 0.0, -0.2]))

    def _slice_mesh_by_face_mask(self, mesh: trimesh.Trimesh, face_mask: np.ndarray) -> trimesh.Trimesh:
        face_mask = np.asarray(face_mask, dtype=bool)
        if face_mask.size == 0 or not np.any(face_mask):
            return trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=np.int64), process=False)
        return mesh.submesh([face_mask], append=True, only_watertight=False)

    def _slice_mesh_by_component_ids(
        self,
        mesh: trimesh.Trimesh,
        labels: np.ndarray,
        component_ids: list[int],
    ) -> trimesh.Trimesh:
        if not component_ids:
            return trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=np.int64), process=False)
        face_mask = np.isin(labels, np.asarray(component_ids, dtype=np.int64))
        return self._slice_mesh_by_face_mask(mesh, face_mask)

    def _slice_mesh_by_face_indices(
        self,
        mesh: trimesh.Trimesh,
        face_indices: np.ndarray,
    ) -> trimesh.Trimesh:
        face_indices = np.asarray(face_indices, dtype=np.int64)
        if face_indices.size == 0:
            return trimesh.Trimesh(
                vertices=np.zeros((0, 3)),
                faces=np.zeros((0, 3), dtype=np.int64),
                process=False,
            )
        face_mask = np.zeros((mesh.faces.shape[0],), dtype=bool)
        face_mask[face_indices] = True
        return self._slice_mesh_by_face_mask(mesh, face_mask)

    def _split_wall_faces(self, mesh: trimesh.Trimesh) -> dict[str, np.ndarray]:
        face_centroids = np.asarray(mesh.triangles_center, dtype=np.float32)
        bounds = mesh.bounds.astype(np.float32)
        dist_left = np.abs(face_centroids[:, 0] - bounds[0, 0])
        dist_right = np.abs(bounds[1, 0] - face_centroids[:, 0])
        dist_back = np.abs(face_centroids[:, 2] - bounds[0, 2])
        dist_front = np.abs(bounds[1, 2] - face_centroids[:, 2])
        distances = np.stack([dist_left, dist_right, dist_back, dist_front], axis=1)
        face_labels = np.argmin(distances, axis=1)
        return {
            "left": face_labels == 0,
            "right": face_labels == 1,
            "back": face_labels == 2,
            "front": face_labels == 3,
        }

    def _build_component_records(
        self,
        mesh: trimesh.Trimesh,
        labels: np.ndarray,
    ) -> list[dict[str, Any]]:
        face_normals = np.asarray(mesh.face_normals, dtype=np.float32)
        face_centroids = np.asarray(mesh.triangles_center, dtype=np.float32)
        face_areas = np.asarray(mesh.area_faces, dtype=np.float32)
        horizontal_face_mask = (face_normals @ self._asset_up) >= 0.90
        component_records: list[dict[str, Any]] = []
        for component_id in np.unique(labels):
            component_face_mask = labels == component_id
            component_face_indices = np.nonzero(component_face_mask)[0].astype(np.int64)
            area = float(np.sum(face_areas[component_face_mask]))
            if area <= 1e-6:
                continue
            component_faces = mesh.faces[component_face_mask]
            vertex_ids = np.unique(component_faces.reshape(-1))
            component_vertices = np.asarray(mesh.vertices[vertex_ids], dtype=np.float32)
            bounds_min = component_vertices.min(axis=0)
            bounds_max = component_vertices.max(axis=0)
            horizontal_mask = component_face_mask & horizontal_face_mask
            support_patches: list[dict[str, Any]] = []
            support_levels: list[dict[str, Any]] = []
            blocker_groups: list[dict[str, Any]] = []
            if np.any(horizontal_mask):
                support_face_indices = np.nonzero(horizontal_mask)[0]
                support_patches = _build_support_patches(
                    mesh,
                    support_face_indices,
                    face_centroids,
                    face_areas,
                )
                if support_patches:
                    patch_heights = np.asarray([patch["y"] for patch in support_patches], dtype=np.float32)
                    patch_areas = np.asarray([patch["area"] for patch in support_patches], dtype=np.float32)
                    patch_bounds_min = np.stack(
                        [np.asarray(patch["bounds_min"], dtype=np.float32) for patch in support_patches],
                        axis=0,
                    )
                    patch_bounds_max = np.stack(
                        [np.asarray(patch["bounds_max"], dtype=np.float32) for patch in support_patches],
                        axis=0,
                    )
                    patch_centers = np.stack(
                        [np.asarray(patch["center"], dtype=np.float32) for patch in support_patches],
                        axis=0,
                    )
                    support_levels = _cluster_support_levels(
                        patch_heights,
                        patch_areas,
                        patch_bounds_min,
                        patch_bounds_max,
                        patch_centers,
                    )
            blocker_groups = _build_blocker_groups(
                mesh,
                component_face_indices,
                support_patches,
                face_centroids,
                face_areas,
            )
            support_top_level = support_levels[0] if support_levels else None
            component_records.append(
                {
                    "id": int(component_id),
                    "area": area,
                    "face_indices": component_face_indices,
                    "bounds_min": bounds_min.astype(np.float32),
                    "bounds_max": bounds_max.astype(np.float32),
                    "center": (0.5 * (bounds_min + bounds_max)).astype(np.float32),
                    "extents": (bounds_max - bounds_min).astype(np.float32),
                    "horizontal_area": float(
                        np.sum(
                            [
                                patch["area"]
                                for patch in support_patches
                                if patch["y"] >= (self._asset_floor_y + TABLE_SUPPORT_HEIGHT_MIN)
                            ]
                        )
                    ),
                    "support_patches": support_patches,
                    "support_levels": support_levels,
                    "support_top_level": support_top_level,
                    "blocker_groups": blocker_groups,
                }
            )
        return component_records

    def _select_table_components(
        self,
        component_records: list[dict[str, Any]],
    ) -> tuple[
        list[int],
        list[int],
        tuple[np.ndarray, np.ndarray],
        np.ndarray,
        float,
        dict[str, Any],
        list[dict[str, Any]],
    ]:
        room_center_xz = self._asset_room_center_xz
        candidates: list[tuple[float, dict[str, Any]]] = []
        for record in component_records:
            level = record["support_top_level"]
            if level is None:
                continue
            if level["y"] <= (self._asset_floor_y + TABLE_SUPPORT_HEIGHT_MIN):
                continue
            support_bounds = level["bounds_max"] - level["bounds_min"]
            if min(float(support_bounds[0]), float(support_bounds[2])) < 0.10:
                continue
            dist = float(np.linalg.norm(level["center"][[0, 2]] - room_center_xz))
            score = float(level["area"]) / (0.15 + dist * dist)
            candidates.append((score, record))
        if not candidates:
            raise RuntimeError("Could not identify a central table support surface in ILLIXR_lab.")
        candidates.sort(key=lambda item: item[0], reverse=True)
        primary_record = candidates[0][1]
        primary_level = primary_record["support_top_level"]
        assert primary_level is not None
        primary_center_xz = primary_level["center"][[0, 2]]
        primary_y = float(primary_level["y"])

        support_component_ids: list[int] = []
        for record in component_records:
            level = record["support_top_level"]
            if level is None:
                continue
            dist = float(np.linalg.norm(level["center"][[0, 2]] - primary_center_xz))
            if dist > 1.4:
                continue
            if abs(float(level["y"]) - primary_y) > 0.08:
                continue
            if float(level["area"]) < TABLE_SUPPORT_AREA_MIN:
                continue
            support_component_ids.append(int(record["id"]))
        if not support_component_ids:
            support_component_ids = [int(primary_record["id"])]

        scale_reference_bounds_min = np.min(
            [
                next(
                    level["bounds_min"]
                    for level in record["support_levels"]
                    if abs(float(level["y"]) - primary_y) <= 0.08
                )
                for record in component_records
                if int(record["id"]) in support_component_ids
            ],
            axis=0,
        ).astype(np.float32)
        scale_reference_bounds_max = np.max(
            [
                next(
                    level["bounds_max"]
                    for level in record["support_levels"]
                    if abs(float(level["y"]) - primary_y) <= 0.08
                )
                for record in component_records
                if int(record["id"]) in support_component_ids
            ],
            axis=0,
        ).astype(np.float32)
        scale_reference_center = np.array(
            [
                0.5 * float(scale_reference_bounds_min[0] + scale_reference_bounds_max[0]),
                primary_y,
                0.5 * float(scale_reference_bounds_min[2] + scale_reference_bounds_max[2]),
            ],
            dtype=np.float32,
        )

        support_expand = np.array([0.18, 0.02, 0.18], dtype=np.float32)
        expanded_min = scale_reference_bounds_min.copy()
        expanded_max = scale_reference_bounds_max.copy()
        expanded_min[[0, 2]] -= support_expand[[0, 2]]
        expanded_max[[0, 2]] += support_expand[[0, 2]]
        table_component_ids = set(support_component_ids)
        for record in component_records:
            if int(record["id"]) in table_component_ids:
                continue
            bounds_min = record["bounds_min"]
            bounds_max = record["bounds_max"]
            overlaps_xz = not (
                float(bounds_max[0]) < float(expanded_min[0])
                or float(bounds_min[0]) > float(expanded_max[0])
                or float(bounds_max[2]) < float(expanded_min[2])
                or float(bounds_min[2]) > float(expanded_max[2])
            )
            if not overlaps_xz:
                continue
            if float(bounds_min[1]) < float(self._asset_floor_y - 0.02):
                continue
            if float(bounds_max[1]) > float(primary_y + 0.14):
                continue
            table_component_ids.add(int(record["id"]))

        startup_patch_candidates: list[tuple[float, float, float, dict[str, Any]]] = []
        startup_bounds_min = scale_reference_bounds_min.copy()
        startup_bounds_max = scale_reference_bounds_max.copy()
        startup_bounds_min[[0, 2]] -= np.array([0.10, 0.10], dtype=np.float32)
        startup_bounds_max[[0, 2]] += np.array([0.10, 0.10], dtype=np.float32)
        for record in component_records:
            if int(record["id"]) not in set(support_component_ids):
                continue
            for patch in record.get("support_patches", []):
                if patch["y"] <= (self._asset_floor_y + TABLE_SUPPORT_HEIGHT_MIN):
                    continue
                if float(patch["area"]) < TABLE_SUPPORT_AREA_MIN:
                    continue
                patch_extents = np.asarray(
                    patch["bounds_max"] - patch["bounds_min"],
                    dtype=np.float32,
                )
                if min(float(patch_extents[0]), float(patch_extents[2])) < 0.10:
                    continue
                overlaps_xz = not (
                    float(patch["bounds_max"][0]) < float(startup_bounds_min[0])
                    or float(patch["bounds_min"][0]) > float(startup_bounds_max[0])
                    or float(patch["bounds_max"][2]) < float(startup_bounds_min[2])
                    or float(patch["bounds_min"][2]) > float(startup_bounds_max[2])
                )
                if not overlaps_xz:
                    continue
                dist = float(
                    np.linalg.norm(
                        np.asarray(patch["center"], dtype=np.float32)[[0, 2]]
                        - scale_reference_center[[0, 2]]
                    )
                )
                startup_patch_candidates.append(
                    (
                        float(patch["y"]),
                        float(patch["area"]),
                        -dist,
                        patch,
                    )
                )

        if not startup_patch_candidates:
            raise RuntimeError(
                "Could not identify a connected startup table patch in ILLIXR_lab."
            )

        highest_patch_y = max(candidate[0] for candidate in startup_patch_candidates)
        highest_patch_candidates = [
            candidate
            for candidate in startup_patch_candidates
            if candidate[0] >= (highest_patch_y - TABLE_SUPPORT_GAP_Y)
        ]
        highest_patch_candidates.sort(key=lambda item: (item[1], item[2]), reverse=True)
        startup_patch = highest_patch_candidates[0][3]
        visible_tabletop_patches = [candidate[3] for candidate in highest_patch_candidates]

        return (
            sorted(table_component_ids),
            sorted(set(support_component_ids)),
            (scale_reference_bounds_min, scale_reference_bounds_max),
            scale_reference_center,
            primary_y,
            startup_patch,
            visible_tabletop_patches,
        )

    def _select_collision_component_records(
        self,
        component_records: list[dict[str, Any]],
        table_component_ids: list[int],
    ) -> list[dict[str, Any]]:
        table_component_id_set = set(table_component_ids)
        retained: list[dict[str, Any]] = []
        for record in component_records:
            extents = np.asarray(record["extents"], dtype=np.float32)
            is_table_component = int(record["id"]) in table_component_id_set
            area_threshold = 0.18 if is_table_component else 0.50
            horizontal_threshold = 0.08 if is_table_component else 0.18
            vertical_height_threshold = 0.14 if is_table_component else 0.45
            horizontal_span_threshold = 0.12 if is_table_component else 0.25
            if float(record["area"]) >= area_threshold:
                retained.append(record)
                continue
            if float(record["horizontal_area"]) >= horizontal_threshold:
                retained.append(record)
                continue
            if (
                float(extents[1]) >= vertical_height_threshold
                and max(float(extents[0]), float(extents[2])) >= horizontal_span_threshold
            ):
                retained.append(record)
        return retained

    def _table_target_size_xy(self) -> np.ndarray:
        target_xy = np.asarray(self._target_table_size_xy, dtype=np.float32)
        if target_xy.shape[0] != 2:
            raise ValueError("target_table_size_m must contain two values")
        return target_xy

    def _asset_to_world_rotation(self) -> np.ndarray:
        return trimesh.transformations.rotation_matrix(
            np.deg2rad(self._scene_x_rotation_degrees), [1.0, 0.0, 0.0]
        ).astype(np.float32)

    def _prepare_positioned_scene(self) -> None:
        assert self.layout is not None
        target_xy = self._table_target_size_xy()
        scale_reference_bounds_min, scale_reference_bounds_max = (
            self._asset_table_scale_reference_bounds
        )
        scale_reference_extent_xy = np.array(
            [
                float(scale_reference_bounds_max[0] - scale_reference_bounds_min[0]),
                float(scale_reference_bounds_max[2] - scale_reference_bounds_min[2]),
            ],
            dtype=np.float32,
        )
        scene_scale = float(
            max(
                float(target_xy[0]) / max(float(scale_reference_extent_xy[0]), 1e-4),
                float(target_xy[1]) / max(float(scale_reference_extent_xy[1]), 1e-4),
            )
        )

        asset_to_world = self._asset_to_world_rotation()
        scale_transform = np.diag([scene_scale, scene_scale, scene_scale, 1.0]).astype(np.float32)
        pre_translation = asset_to_world @ scale_transform
        startup_table_center_world = _transform_points(
            np.asarray(self._asset_startup_table_patch["center"], dtype=np.float32).reshape(1, 3),
            pre_translation,
        )[0]
        translation = (
            np.asarray(self.layout.table_top_center, dtype=np.float32) - startup_table_center_world
        )
        world_transform = trimesh.transformations.translation_matrix(translation).astype(np.float32) @ pre_translation

        self._full_scene_mesh_world = self._full_asset_mesh.copy()
        self._full_scene_mesh_world.apply_transform(world_transform)
        self._background_mesh_world = self._full_scene_mesh_world.copy()

        self._table_mesh_world = self._visible_tabletop_asset_mesh.copy()
        self._table_mesh_world.apply_transform(world_transform)

        self._floor_mesh_world = self._floor_asset_mesh.copy()
        self._floor_mesh_world.apply_transform(world_transform)

        self._left_wall_mesh_world = self._left_wall_asset_mesh.copy()
        self._left_wall_mesh_world.apply_transform(world_transform)
        self._right_wall_mesh_world = self._right_wall_asset_mesh.copy()
        self._right_wall_mesh_world.apply_transform(world_transform)
        self._front_back_mesh_world = self._front_back_asset_mesh.copy()
        self._front_back_mesh_world.apply_transform(world_transform)

        full_bounds = self._full_scene_mesh_world.bounds.astype(np.float32)
        floor_z = float(np.median(self._floor_mesh_world.vertices[:, 2]))
        startup_patch_bounds_min = np.asarray(
            self._asset_startup_table_patch["bounds_min"],
            dtype=np.float32,
        )
        startup_patch_bounds_max = np.asarray(
            self._asset_startup_table_patch["bounds_max"],
            dtype=np.float32,
        )
        active_table_world_min, active_table_world_max = _transform_bounds(
            startup_patch_bounds_min,
            startup_patch_bounds_max,
            world_transform,
        )
        active_table_world_bounds = np.stack(
            [active_table_world_min, active_table_world_max],
            axis=0,
        ).astype(np.float32)
        self.layout.active_table_bounds = np.array(active_table_world_bounds, copy=True)
        startup_surface_center_world = _transform_points(
            np.asarray(self._asset_startup_table_patch["center"], dtype=np.float32).reshape(1, 3),
            world_transform,
        )[0].astype(np.float32)
        self.layout.active_table_surface_center = startup_surface_center_world.copy()
        room_center_xy = np.array(
            [
                0.5 * float(full_bounds[0, 0] + full_bounds[1, 0]),
                0.5 * float(full_bounds[0, 1] + full_bounds[1, 1]),
            ],
            dtype=np.float32,
        )
        self.layout.room_center_xy = room_center_xy
        self.layout.room_half_extent = np.array(
            [
                0.5 * float(full_bounds[1, 0] - full_bounds[0, 0]),
                0.5 * float(full_bounds[1, 1] - full_bounds[0, 1]),
            ],
            dtype=np.float32,
        )
        self.layout.floor_z = floor_z
        self.layout.wall_height = float(max(floor_z - float(full_bounds[0, 2]), 0.1))

        visible_table_world_bounds = self._table_mesh_world.bounds.astype(np.float32)
        self._table_world_bounds = (
            visible_table_world_bounds[0].copy(),
            visible_table_world_bounds[1].copy(),
        )
        self.layout.table_size = np.array(
            [
                float(visible_table_world_bounds[1, 0] - visible_table_world_bounds[0, 0]),
                float(visible_table_world_bounds[1, 1] - visible_table_world_bounds[0, 1]),
                float(max(visible_table_world_bounds[1, 2] - visible_table_world_bounds[0, 2], 0.12)),
            ],
            dtype=np.float32,
        )

        self._wall_world_bounds = {}
        for wall_name, wall_mesh in (
            ("left", self._left_wall_mesh_world),
            ("right", self._right_wall_mesh_world),
        ):
            if wall_mesh.vertices.shape[0] == 0 or wall_mesh.faces.shape[0] == 0:
                continue
            bounds = wall_mesh.bounds.astype(np.float32)
            self._wall_world_bounds[wall_name] = (bounds[0].copy(), bounds[1].copy())

        collider_entries: list[dict[str, Any]] = []

        floor_bounds_min = self._floor_mesh_world.bounds[0].astype(np.float32).copy()
        floor_bounds_max = self._floor_mesh_world.bounds[1].astype(np.float32).copy()
        floor_bounds_min[2] = floor_z
        floor_bounds_max[2] = floor_z
        floor_bounds_min, floor_bounds_max = _expand_bounds_min_thickness(
            floor_bounds_min,
            floor_bounds_max,
            min_thickness=FLOOR_COLLIDER_THICKNESS,
            support_axis=2,
        )
        collider_entries.append(
            {
                "category": "floor_slab",
                "component_id": None,
                "box": np.stack([floor_bounds_min, floor_bounds_max], axis=0).astype(np.float32),
            }
        )

        wall_bounds = {
            "left": _transform_bounds(
                np.array([full_bounds[0, 0], full_bounds[0, 1], full_bounds[0, 2]], dtype=np.float32),
                np.array([full_bounds[0, 0], full_bounds[1, 1], floor_z], dtype=np.float32),
                np.eye(4, dtype=np.float32),
            ),
            "right": _transform_bounds(
                np.array([full_bounds[1, 0], full_bounds[0, 1], full_bounds[0, 2]], dtype=np.float32),
                np.array([full_bounds[1, 0], full_bounds[1, 1], floor_z], dtype=np.float32),
                np.eye(4, dtype=np.float32),
            ),
            "back": _transform_bounds(
                np.array([full_bounds[0, 0], full_bounds[0, 1], full_bounds[0, 2]], dtype=np.float32),
                np.array([full_bounds[1, 0], full_bounds[0, 1], floor_z], dtype=np.float32),
                np.eye(4, dtype=np.float32),
            ),
            "front": _transform_bounds(
                np.array([full_bounds[0, 0], full_bounds[1, 1], full_bounds[0, 2]], dtype=np.float32),
                np.array([full_bounds[1, 0], full_bounds[1, 1], floor_z], dtype=np.float32),
                np.eye(4, dtype=np.float32),
            ),
        }
        for wall_name, (wall_min, wall_max) in wall_bounds.items():
            support_axis = 0 if wall_name in {"left", "right"} else 1
            wall_min, wall_max = _expand_bounds_min_thickness(
                wall_min,
                wall_max,
                min_thickness=WALL_COLLIDER_THICKNESS,
                support_axis=support_axis,
            )
            collider_entries.append(
                {
                    "category": "boundary_wall",
                    "component_id": None,
                    "box": np.stack([wall_min, wall_max], axis=0).astype(np.float32),
                }
            )

        for record in self._collision_component_records:
            component_id = int(record["id"])
            for patch in record.get("support_patches", []):
                if not _support_patch_is_collision_usable(patch, self._asset_floor_y):
                    continue
                bounds_min, bounds_max = _expand_bounds_min_thickness(
                    np.asarray(patch["bounds_min"], dtype=np.float32),
                    np.asarray(patch["bounds_max"], dtype=np.float32),
                    min_thickness=COLLIDER_MIN_THICKNESS,
                    support_axis=1,
                )
                world_min, world_max = _transform_bounds(bounds_min, bounds_max, world_transform)
                collider_entries.append(
                    {
                        "category": "support_slab",
                        "component_id": component_id,
                        "box": np.stack([world_min, world_max], axis=0).astype(np.float32),
                    }
                )
            for blocker_group in record.get("blocker_groups", []):
                bounds_min, bounds_max = _expand_bounds_min_thickness(
                    np.asarray(blocker_group["bounds_min"], dtype=np.float32),
                    np.asarray(blocker_group["bounds_max"], dtype=np.float32),
                    min_thickness=COLLIDER_MIN_THICKNESS,
                    support_axis=None,
                )
                world_min, world_max = _transform_bounds(bounds_min, bounds_max, world_transform)
                collider_entries.append(
                    {
                        "category": "blocker_box",
                        "component_id": component_id,
                        "box": np.stack([world_min, world_max], axis=0).astype(np.float32),
                    }
                )

        self._scene_collider_boxes = np.stack(
            [np.asarray(entry["box"], dtype=np.float32) for entry in collider_entries],
            axis=0,
        ).astype(np.float32)
        self.layout.static_collider_boxes = np.array(self._scene_collider_boxes, copy=True)
        support_slab_count = int(
            sum(1 for entry in collider_entries if entry["category"] == "support_slab")
        )
        blocker_box_count = int(
            sum(1 for entry in collider_entries if entry["category"] == "blocker_box")
        )
        boundary_box_count = int(
            sum(
                1
                for entry in collider_entries
                if entry["category"] in {"floor_slab", "boundary_wall"}
            )
        )
        component_box_counts: dict[int, int] = {}
        for entry in collider_entries:
            component_id = entry["component_id"]
            if component_id is None:
                continue
            component_box_counts[int(component_id)] = component_box_counts.get(int(component_id), 0) + 1
        decomposed_component_count = int(
            sum(1 for count in component_box_counts.values() if count > 1)
        )

        scene_up = np.asarray(self.layout.scene_up, dtype=np.float32)
        collider_top_center = np.array(
            [
                0.5 * float(active_table_world_bounds[0, 0] + active_table_world_bounds[1, 0]),
                0.5 * float(active_table_world_bounds[0, 1] + active_table_world_bounds[1, 1]),
                float(active_table_world_bounds[0, 2]),
            ],
            dtype=np.float32,
        )
        collider_top_plane = float(np.dot(collider_top_center, scene_up))
        world_surface_center = startup_surface_center_world.copy()
        world_surface_plane = float(np.dot(world_surface_center, scene_up))
        world_surface_normal = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self._table_alignment_debug = {
            "scene_preset": ILLIXR_SCENE_NAME,
            "asset_transform": {
                "asset_up_axis": self.manifest.get("asset_up_axis", "y"),
                "scene_up_axis": self.manifest.get("scene_up_axis", "-z"),
                "x_rotation_degrees": self._scene_x_rotation_degrees,
                "uniform_scene_scale": scene_scale,
            },
            "table_component_ids": [int(v) for v in self._table_component_ids],
            "support_component_ids": [int(v) for v in self._table_support_component_ids],
            "local_surface_center": np.asarray(
                self._asset_startup_table_patch["center"],
                dtype=np.float32,
            ).tolist(),
            "local_surface_plane_height": float(self._asset_startup_table_patch["y"]),
            "active_table_support_patch_count": int(len(self._asset_visible_tabletop_patches)),
            "local_surface_normal": self._asset_up.astype(np.float32).tolist(),
            "surface_normal_alignment": float(np.dot(world_surface_normal, scene_up)),
            "world_surface_center": world_surface_center.astype(np.float32).tolist(),
            "world_surface_plane_height": world_surface_plane,
            "world_surface_normal": world_surface_normal.astype(np.float32).tolist(),
            "collider_top_center": collider_top_center.astype(np.float32).tolist(),
            "collider_top_plane_height": collider_top_plane,
            "native_table_support_size_xy": scale_reference_extent_xy.astype(np.float32).tolist(),
            "scaled_table_support_size_xy": self.layout.table_size[:2].astype(np.float32).tolist(),
            "active_table_world_bounds": active_table_world_bounds.astype(np.float32).tolist(),
            "visible_table_world_bounds": visible_table_world_bounds.astype(np.float32).tolist(),
            "room_bounds": full_bounds.astype(np.float32).tolist(),
            "collider_box_count": int(self._scene_collider_boxes.shape[0]),
            "support_slab_count": support_slab_count,
            "blocker_box_count": blocker_box_count,
            "boundary_box_count": boundary_box_count,
            "decomposed_component_count": decomposed_component_count,
        }

    def _rebuild_scene_nodes(self) -> None:
        if self.layout is None:
            return
        self._prepare_positioned_scene()
        assert self._background_mesh_world is not None
        assert self._table_mesh_world is not None
        assert self._floor_mesh_world is not None
        assert self._left_wall_mesh_world is not None
        assert self._right_wall_mesh_world is not None
        assert self._front_back_mesh_world is not None

        mesh_by_role = {
            "all": self._background_mesh_world,
            "front_back_walls": self._front_back_mesh_world,
            "left_wall": self._left_wall_mesh_world,
            "right_wall": self._right_wall_mesh_world,
            "floor": self._floor_mesh_world,
        }
        for entry in self._layer_entries.values():
            scene = entry["scene"]
            if entry["table_node"] is not None:
                scene.remove_node(entry["table_node"])
                entry["table_node"] = None
            if entry["floor_node"] is not None:
                scene.remove_node(entry["floor_node"])
                entry["floor_node"] = None
            for node in entry["wall_nodes"]:
                scene.remove_node(node)
            entry["wall_nodes"] = []
            for node in entry.get("background_nodes", []):
                scene.remove_node(node)
            entry["background_nodes"] = []

            if entry.get("table_role") is not None and self._table_mesh_world.faces.shape[0] > 0:
                entry["table_node"] = scene.add(
                    self._pyrender.Mesh.from_trimesh(self._table_mesh_world.copy(), smooth=False)
                )

            background_role = entry.get("background_role")
            if background_role is not None:
                mesh = mesh_by_role.get(background_role)
                if mesh is not None and mesh.faces.shape[0] > 0:
                    node = scene.add(
                        self._pyrender.Mesh.from_trimesh(mesh.copy(), smooth=False)
                    )
                    entry["background_nodes"].append(node)

    def _get_layer_renderer(self, layer_name: str, width: int, height: int):
        entry = self._ensure_layer_entry(layer_name)
        renderer = entry.get("renderer")
        if renderer is None:
            renderer_cls = (
                PyrenderCudaInteropOffscreenRenderer
                if self._pyrender_cuda_interop_supported
                else self._pyrender.OffscreenRenderer
            )
            renderer_kwargs = {}
            if renderer_cls is PyrenderCudaInteropOffscreenRenderer:
                renderer_kwargs["device"] = torch.device("cuda", torch.cuda.current_device())
            renderer = renderer_cls(
                viewport_width=int(width),
                viewport_height=int(height),
                **renderer_kwargs,
            )
            entry["renderer"] = renderer
        else:
            renderer.viewport_width = int(width)
            renderer.viewport_height = int(height)
        self._update_pyrender_readback_state(renderer)
        return renderer

    def _update_pyrender_readback_state(self, renderer) -> None:
        mode = getattr(renderer, "readback_mode", None)
        if mode is not None:
            self._pyrender_readback_mode = str(mode)
        reason = getattr(renderer, "fallback_reason", None)
        if reason is not None:
            self._pyrender_readback_reason = str(reason)

    def _postprocess_render_color(
        self,
        color: np.ndarray | torch.Tensor,
    ) -> np.ndarray | torch.Tensor:
        if self.lighting_mode != ILLIXR_BAKED_LIGHTING_MODE:
            return color
        gamma = 1.0 / 2.2
        if torch.is_tensor(color):
            color_rgb = torch.clamp(color[..., :3] / 255.0, 0.0, 1.0)
            color[..., :3] = torch.pow(color_rgb, gamma) * 255.0
            return color
        color = np.array(color, copy=True)
        color_rgb = np.clip(color[..., :3].astype(np.float32) / 255.0, 0.0, 1.0)
        color[..., :3] = np.power(color_rgb, gamma) * 255.0
        return color

    def _render_layer_eye(
        self,
        layer_name: str,
        camera_pose_world: np.ndarray,
        intrinsic: np.ndarray,
        width: int | None = None,
        height: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[torch.Tensor, torch.Tensor]:
        if self.layout is None:
            raise RuntimeError("Immersive scene layout has not been configured.")
        if width is None:
            width = self.width
        if height is None:
            height = self.height
        entry = self._ensure_layer_entry(layer_name)
        camera = entry["camera"]
        camera.fx = float(intrinsic[0, 0])
        camera.fy = float(intrinsic[1, 1])
        camera.cx = float(intrinsic[0, 2])
        camera.cy = float(intrinsic[1, 2])
        entry["scene"].set_pose(
            entry["camera_node"],
            pose=np.asarray(camera_pose_world, dtype=np.float32),
        )
        renderer = self._get_layer_renderer(layer_name, width, height)
        color, depth = renderer.render(
            entry["scene"],
            flags=self._pyrender.RenderFlags.RGBA,
        )
        self._update_pyrender_readback_state(renderer)
        color = self._postprocess_render_color(color)
        if torch.is_tensor(depth):
            return color, depth.to(dtype=torch.float32)
        return color, depth.astype(np.float32)

    def _render_layer_eye_roi(
        self,
        layer_name: str,
        camera_pose_world: np.ndarray,
        full_intrinsic: np.ndarray,
        roi_bounds: tuple[int, int, int, int],
        render_scale: float = 1.0,
        return_render_info: bool = False,
    ) -> (
        tuple[np.ndarray, np.ndarray]
        | tuple[torch.Tensor, torch.Tensor]
        | tuple[np.ndarray, np.ndarray, dict[str, Any]]
        | tuple[torch.Tensor, torch.Tensor, dict[str, Any]]
    ):
        roi_intrinsic, render_width, render_height, render_info = self._resolve_roi_render_params(
            full_intrinsic,
            roi_bounds,
            render_scale=render_scale,
        )
        render_color, render_depth = self._render_layer_eye(
            layer_name,
            camera_pose_world,
            roi_intrinsic,
            width=render_width,
            height=render_height,
        )
        if return_render_info:
            return render_color, render_depth, render_info
        return render_color, render_depth


def build_illixr_scene_analysis_cache(
    scene_assets_root: str | Path,
    output_path: str | Path | None = None,
    width: int = 64,
    height: int = 64,
) -> tuple[Path, dict[str, Any]]:
    renderer = SimpleLabSceneRenderer(
        scene_assets_root=scene_assets_root,
        width=width,
        height=height,
        lighting_mode=ILLIXR_BAKED_LIGHTING_MODE,
        balanced_render_backend="pyrender",
        scene_analysis_cache_mode="rebuild",
    )
    try:
        cache_path = renderer.write_scene_analysis_cache(output_path)
        debug = renderer.scene_analysis_cache_debug()
    finally:
        renderer.delete()
    return cache_path, debug


ImmersiveSceneRenderer = SimpleLabSceneRenderer
make_immersive_scene_layout = make_illixr_lab_layout
