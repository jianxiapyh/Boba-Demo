from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from PIL import Image
from trimesh.visual.texture import TextureVisuals


MANIFEST_RELATIVE_PATH = Path("data/open_scene_assets/simple_lab/manifest.json")
USER_AGENT = "BobaQuestImmersiveDemo/1.0"


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

    @property
    def scene_down(self) -> np.ndarray:
        return -self.scene_up

    @property
    def table_top_z(self) -> float:
        return float(self.table_top_center[2])

    @property
    def table_box(self) -> SceneColliderBox:
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
        mins = np.array(
            [
                self.table_top_center[0] - self.room_half_extent[0],
                self.table_top_center[1] - self.room_half_extent[1],
                self.floor_z if float(self.scene_up[2]) < 0.0 else self.floor_z - 0.06,
            ],
            dtype=np.float32,
        )
        maxs = np.array(
            [
                self.table_top_center[0] + self.room_half_extent[0],
                self.table_top_center[1] + self.room_half_extent[1],
                self.floor_z + 0.06 if float(self.scene_up[2]) < 0.0 else self.floor_z,
            ],
            dtype=np.float32,
        )
        return SceneColliderBox(mins=mins, maxs=maxs)


def _repo_root_from_assets_root(scene_assets_root: str | Path) -> Path:
    assets_root = Path(scene_assets_root).resolve()
    if assets_root.name == "open_scene_assets":
        return assets_root.parents[1]
    return assets_root.parents[2]


def simple_lab_manifest_path(scene_assets_root: str | Path) -> Path:
    assets_root = Path(scene_assets_root).resolve()
    if assets_root.name == "simple_lab":
        return assets_root / "manifest.json"
    if assets_root.name == "open_scene_assets":
        return assets_root / "simple_lab" / "manifest.json"
    return _repo_root_from_assets_root(assets_root) / MANIFEST_RELATIVE_PATH


def load_simple_lab_manifest(scene_assets_root: str | Path) -> dict[str, Any]:
    manifest_path = simple_lab_manifest_path(scene_assets_root)
    with open(manifest_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(request, timeout=60) as response, open(
        destination, "wb"
    ) as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)


def ensure_simple_lab_assets(scene_assets_root: str | Path) -> Path:
    assets_root = Path(scene_assets_root).resolve()
    manifest = load_simple_lab_manifest(assets_root)
    if assets_root.name == "simple_lab":
        simple_lab_root = assets_root
    else:
        simple_lab_root = assets_root / "simple_lab"
    for entry in manifest["assets"].values():
        relative_path = Path(entry["relative_path"])
        destination = simple_lab_root / relative_path
        expected_md5 = entry.get("md5")
        if destination.exists() and (
            expected_md5 is None or _md5(destination) == expected_md5
        ):
            continue
        _download(entry["url"], destination)
        if expected_md5 is not None and _md5(destination) != expected_md5:
            raise RuntimeError(
                f"Downloaded asset checksum mismatch for {destination}: "
                f"expected {expected_md5}"
            )
    return simple_lab_root


def make_simple_lab_layout(
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
    scene_up_norm = float(np.linalg.norm(scene_up))
    if scene_up_norm < 1e-5:
        raise ValueError("scene_up must have non-zero length")
    scene_up = scene_up / scene_up_norm

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
        table_size=np.array([0.95, 0.68, 0.76], dtype=np.float32),
        floor_z=float(floor_point[2]),
        room_half_extent=np.array([2.7, 2.7], dtype=np.float32),
        wall_height=2.8,
        scene_up=scene_up.astype(np.float32),
    )


def _as_trimesh(mesh_or_scene: trimesh.Trimesh | trimesh.Scene) -> trimesh.Trimesh:
    if isinstance(mesh_or_scene, trimesh.Trimesh):
        return mesh_or_scene.copy()
    geometries = [geom.copy() for geom in mesh_or_scene.dump(concatenate=False)]
    if not geometries:
        raise ValueError("No geometry found in immersive scene asset.")
    return trimesh.util.concatenate(geometries)


def _make_textured_quad(
    width: float,
    height: float,
    texture_path: Path,
    uv_scale: tuple[float, float],
) -> trimesh.Trimesh:
    vertices = np.array(
        [
            [-0.5 * width, -0.5 * height, 0.0],
            [0.5 * width, -0.5 * height, 0.0],
            [0.5 * width, 0.5 * height, 0.0],
            [-0.5 * width, 0.5 * height, 0.0],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    uvs = np.array(
        [
            [0.0, 0.0],
            [uv_scale[0], 0.0],
            [uv_scale[0], uv_scale[1]],
            [0.0, uv_scale[1]],
        ],
        dtype=np.float32,
    )
    image = Image.open(texture_path).convert("RGB")
    visual = TextureVisuals(uv=uvs, image=image)
    return trimesh.Trimesh(vertices=vertices, faces=faces, visual=visual, process=False)


class SimpleLabSceneRenderer:
    def __init__(
        self,
        scene_assets_root: str | Path,
        width: int,
        height: int,
        lighting_mode: str = "full",
    ):
        import pyrender

        self.width = int(width)
        self.height = int(height)
        self.lighting_mode = str(lighting_mode)
        self._pyrender = pyrender
        self.simple_lab_root = ensure_simple_lab_assets(scene_assets_root)
        self.manifest = load_simple_lab_manifest(scene_assets_root)
        self.layout: SimpleLabLayout | None = None
        self.scene = pyrender.Scene(
            bg_color=np.array([243, 244, 246, 255], dtype=np.uint8),
            ambient_light=np.array([0.22, 0.22, 0.22], dtype=np.float32),
        )
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=self.width,
            viewport_height=self.height,
        )
        self.camera = self._pyrender.IntrinsicsCamera(
            fx=1.0,
            fy=1.0,
            cx=float(self.width) * 0.5,
            cy=float(self.height) * 0.5,
            znear=0.02,
            zfar=100.0,
        )
        self.camera_node = self.scene.add(self.camera, pose=np.eye(4, dtype=np.float32))
        self.table_node = None
        self.floor_node = None
        self.wall_nodes: list[Any] = []
        self._table_alignment_debug: dict[str, Any] | None = None
        self._table_transform_debug: dict[str, Any] | None = None
        self._setup_lights()
        self._table_mesh = self._load_table_mesh()
        self._floor_quad = _make_textured_quad(
            width=1.0,
            height=1.0,
            texture_path=self.simple_lab_root / "floor" / "concrete_floor_diff_2k.jpg",
            uv_scale=(4.0, 4.0),
        )
        self._wall_quad = _make_textured_quad(
            width=1.0,
            height=1.0,
            texture_path=self.simple_lab_root / "walls" / "rough_concrete_diff_2k.jpg",
            uv_scale=(3.0, 2.0),
        )

    def delete(self) -> None:
        if self.renderer is not None:
            self.renderer.delete()
            self.renderer = None

    def table_alignment_debug(self) -> dict[str, Any] | None:
        if self._table_alignment_debug is None:
            return None
        return dict(self._table_alignment_debug)

    def set_layout(self, layout: SimpleLabLayout) -> None:
        self.layout = layout
        self._rebuild_scene_nodes()

    def render_eye(
        self,
        camera_pose_world: np.ndarray,
        intrinsic: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.layout is None:
            raise RuntimeError("Simple lab layout has not been configured.")

        self.camera.fx = float(intrinsic[0, 0])
        self.camera.fy = float(intrinsic[1, 1])
        self.camera.cx = float(intrinsic[0, 2])
        self.camera.cy = float(intrinsic[1, 2])
        self.scene.set_pose(self.camera_node, pose=np.asarray(camera_pose_world))
        color, depth = self.renderer.render(self.scene, flags=self._pyrender.RenderFlags.RGBA)
        return color, depth.astype(np.float32)

    def _setup_lights(self) -> None:
        key_light = self._pyrender.DirectionalLight(
            color=np.ones(3, dtype=np.float32),
            intensity=3.5,
        )
        self.scene.add(key_light, pose=trimesh.transformations.euler_matrix(-0.7, 0.35, 0.0))
        if self.lighting_mode == "full":
            fill_light = self._pyrender.DirectionalLight(
                color=np.array([0.86, 0.90, 1.0], dtype=np.float32),
                intensity=1.4,
            )
            point_light = self._pyrender.PointLight(
                color=np.ones(3, dtype=np.float32),
                intensity=22.0,
            )
            self.scene.add(
                fill_light,
                pose=trimesh.transformations.euler_matrix(-1.1, -0.4, 0.0),
            )
            self.scene.add(
                point_light,
                pose=trimesh.transformations.translation_matrix([0.0, 0.0, -0.2]),
            )

    def _load_table_mesh(self) -> trimesh.Trimesh:
        table_path = self.simple_lab_root / "table" / "wooden_table_02_2k.gltf"
        table_mesh = _as_trimesh(trimesh.load(table_path, force="scene", process=False))
        transform_cfg = dict(self.manifest.get("table_mesh_transform", {}))
        x_rotation_degrees = float(
            transform_cfg.get(
                "x_rotation_degrees",
                transform_cfg.get("flip_x_degrees", 0.0),
            )
        )
        if abs(x_rotation_degrees) > 1e-4:
            table_mesh.apply_transform(
                trimesh.transformations.rotation_matrix(
                    np.deg2rad(x_rotation_degrees), [1.0, 0.0, 0.0]
                )
            )
        self._table_transform_debug = {
            "asset_up_axis": transform_cfg.get("asset_up_axis", "unknown"),
            "scene_up_axis": transform_cfg.get("scene_up_axis", "unknown"),
            "x_rotation_degrees": x_rotation_degrees,
        }
        return table_mesh

    def _rebuild_scene_nodes(self) -> None:
        if self.layout is None:
            return
        if self.table_node is not None:
            self.scene.remove_node(self.table_node)
            self.table_node = None
        if self.floor_node is not None:
            self.scene.remove_node(self.floor_node)
            self.floor_node = None
        for node in self.wall_nodes:
            self.scene.remove_node(node)
        self.wall_nodes = []

        self.table_node = self.scene.add(
            self._pyrender.Mesh.from_trimesh(
                self._make_positioned_table_mesh(),
                smooth=False,
            )
        )
        self.floor_node = self.scene.add(
            self._pyrender.Mesh.from_trimesh(
                self._make_floor_mesh(),
                smooth=False,
            )
        )
        for wall_mesh in self._make_wall_meshes():
            self.wall_nodes.append(
                self.scene.add(self._pyrender.Mesh.from_trimesh(wall_mesh, smooth=False))
            )

    def _make_positioned_table_mesh(self) -> trimesh.Trimesh:
        assert self.layout is not None
        mesh = self._table_mesh.copy()
        bounds = mesh.bounds
        extents = np.maximum(bounds[1] - bounds[0], 1e-5)
        scale = np.array(
            [
                self.layout.table_size[0] / extents[0],
                self.layout.table_size[1] / extents[1],
                self.layout.table_size[2] / extents[2],
            ],
            dtype=np.float32,
        )
        mesh.apply_scale(scale)
        tabletop_debug = self._extract_tabletop_surface(mesh, self.layout.scene_up)
        local_surface_center = tabletop_debug["local_surface_center"]
        mesh.apply_translation(-local_surface_center)
        mesh.apply_translation(self.layout.table_top_center)
        collider_top_center = np.asarray(self.layout.table_top_center, dtype=np.float32)
        collider_top_plane = float(np.dot(collider_top_center, self.layout.scene_up))
        world_surface_center = collider_top_center.copy()
        world_surface_plane = float(np.dot(world_surface_center, self.layout.scene_up))
        self._table_alignment_debug = {
            "asset_transform": dict(self._table_transform_debug or {}),
            "local_surface_center": local_surface_center.astype(np.float32).tolist(),
            "local_surface_plane_height": float(tabletop_debug["local_surface_plane_height"]),
            "local_surface_normal": tabletop_debug["local_surface_normal"].astype(np.float32).tolist(),
            "surface_normal_alignment": float(tabletop_debug["surface_normal_alignment"]),
            "surface_face_count": int(tabletop_debug["surface_face_count"]),
            "world_surface_center": world_surface_center.astype(np.float32).tolist(),
            "world_surface_plane_height": world_surface_plane,
            "world_surface_normal": tabletop_debug["local_surface_normal"].astype(np.float32).tolist(),
            "collider_top_center": collider_top_center.astype(np.float32).tolist(),
            "collider_top_plane_height": collider_top_plane,
        }
        if tabletop_debug["surface_normal_alignment"] < 0.92:
            raise RuntimeError(
                "Selected tabletop normal is not aligned with scene up: "
                f"{self._table_alignment_debug}"
            )
        if abs(world_surface_plane - collider_top_plane) > 0.02:
            raise RuntimeError(
                "Visual tabletop does not align with collider top plane: "
                f"{self._table_alignment_debug}"
            )
        return mesh

    def _extract_tabletop_surface(
        self,
        mesh: trimesh.Trimesh,
        scene_up: np.ndarray,
    ) -> dict[str, Any]:
        scene_up = np.asarray(scene_up, dtype=np.float32)
        scene_up_norm = float(np.linalg.norm(scene_up))
        if scene_up_norm < 1e-6:
            raise ValueError("scene_up must have non-zero length")
        scene_up = scene_up / scene_up_norm

        face_normals = np.asarray(mesh.face_normals, dtype=np.float32)
        face_centroids = np.asarray(mesh.triangles_center, dtype=np.float32)
        face_areas = np.asarray(mesh.area_faces, dtype=np.float32)
        if face_normals.size == 0 or face_centroids.size == 0:
            raise RuntimeError("Table mesh has no faces for tabletop extraction.")

        normal_alignment = face_normals @ scene_up
        aligned_mask = normal_alignment >= 0.85
        if not np.any(aligned_mask):
            raise RuntimeError(
                "Could not find any table faces aligned with scene up for tabletop extraction."
            )

        face_support_depth = face_centroids @ scene_up
        max_support_depth = float(face_support_depth[aligned_mask].max())
        bounds = mesh.bounds
        support_extent = max(
            float(np.dot(bounds[1] - bounds[0], np.abs(scene_up))),
            1e-4,
        )
        plane_eps = max(0.006, support_extent * 0.02)
        surface_mask = aligned_mask & (
            face_support_depth >= (max_support_depth - plane_eps)
        )
        if not np.any(surface_mask):
            raise RuntimeError(
                "Could not isolate a tabletop surface near the top support plane."
            )

        selected_areas = face_areas[surface_mask]
        area_sum = float(selected_areas.sum())
        if area_sum <= 1e-8:
            raise RuntimeError("Selected tabletop faces have zero area.")
        weights = selected_areas / area_sum
        selected_centroids = face_centroids[surface_mask]
        selected_normals = face_normals[surface_mask]
        local_surface_center = np.sum(selected_centroids * weights[:, None], axis=0)
        local_surface_normal = np.sum(selected_normals * weights[:, None], axis=0)
        local_surface_normal /= max(float(np.linalg.norm(local_surface_normal)), 1e-6)
        local_surface_plane_height = float(np.sum(face_support_depth[surface_mask] * weights))
        local_surface_center = local_surface_center - scene_up * (
            float(np.dot(local_surface_center, scene_up)) - local_surface_plane_height
        )
        surface_normal_alignment = float(np.dot(local_surface_normal, scene_up))
        return {
            "local_surface_center": local_surface_center.astype(np.float32),
            "local_surface_normal": local_surface_normal.astype(np.float32),
            "local_surface_plane_height": local_surface_plane_height,
            "surface_normal_alignment": surface_normal_alignment,
            "surface_face_count": int(surface_mask.sum()),
        }

    def _make_floor_mesh(self) -> trimesh.Trimesh:
        assert self.layout is not None
        mesh = self._floor_quad.copy()
        width = float(self.layout.room_half_extent[0] * 2.0)
        depth = float(self.layout.room_half_extent[1] * 2.0)
        mesh.apply_scale([width, depth, 1.0])
        mesh.apply_transform(
            trimesh.transformations.translation_matrix(
                [
                    self.layout.table_top_center[0],
                    self.layout.table_top_center[1],
                    self.layout.floor_z,
                ]
            )
        )
        return mesh

    def _make_wall_meshes(self) -> list[trimesh.Trimesh]:
        assert self.layout is not None
        room_width = float(self.layout.room_half_extent[0] * 2.0)
        room_depth = float(self.layout.room_half_extent[1] * 2.0)
        wall_height = float(self.layout.wall_height)
        center_x = float(self.layout.table_top_center[0])
        center_y = float(self.layout.table_top_center[1])
        floor_z = float(self.layout.floor_z)
        wall_z = floor_z - 0.5 * wall_height
        meshes = []

        back_wall = self._wall_quad.copy()
        back_wall.apply_scale([room_width, wall_height, 1.0])
        back_wall.apply_transform(
            trimesh.transformations.rotation_matrix(np.pi / 2.0, [1.0, 0.0, 0.0])
        )
        back_wall.apply_transform(
            trimesh.transformations.translation_matrix(
                [center_x, center_y - self.layout.room_half_extent[1], wall_z]
            )
        )
        meshes.append(back_wall)

        front_wall = self._wall_quad.copy()
        front_wall.apply_scale([room_width, wall_height, 1.0])
        front_wall.apply_transform(
            trimesh.transformations.rotation_matrix(np.pi / 2.0, [1.0, 0.0, 0.0])
        )
        front_wall.apply_transform(
            trimesh.transformations.rotation_matrix(np.pi, [0.0, 0.0, 1.0])
        )
        front_wall.apply_transform(
            trimesh.transformations.translation_matrix(
                [center_x, center_y + self.layout.room_half_extent[1], wall_z]
            )
        )
        meshes.append(front_wall)

        left_wall = self._wall_quad.copy()
        left_wall.apply_scale([room_depth, wall_height, 1.0])
        left_wall.apply_transform(
            trimesh.transformations.rotation_matrix(np.pi / 2.0, [1.0, 0.0, 0.0])
        )
        left_wall.apply_transform(
            trimesh.transformations.rotation_matrix(np.pi / 2.0, [0.0, 0.0, 1.0])
        )
        left_wall.apply_transform(
            trimesh.transformations.translation_matrix(
                [center_x - self.layout.room_half_extent[0], center_y, wall_z]
            )
        )
        meshes.append(left_wall)

        right_wall = self._wall_quad.copy()
        right_wall.apply_scale([room_depth, wall_height, 1.0])
        right_wall.apply_transform(
            trimesh.transformations.rotation_matrix(np.pi / 2.0, [1.0, 0.0, 0.0])
        )
        right_wall.apply_transform(
            trimesh.transformations.rotation_matrix(-np.pi / 2.0, [0.0, 0.0, 1.0])
        )
        right_wall.apply_transform(
            trimesh.transformations.translation_matrix(
                [center_x + self.layout.room_half_extent[0], center_y, wall_z]
            )
        )
        meshes.append(right_wall)
        return meshes
