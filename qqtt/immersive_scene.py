from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh
from PIL import Image
from trimesh.visual.texture import TextureVisuals

from .pyrender_cuda_bridge import (
    PyrenderCudaInteropOffscreenRenderer,
    probe_pyrender_cuda_bridge_support,
)
from .simple_lab_gpu_renderer import SimpleLabGpuRenderer


MANIFEST_RELATIVE_PATH = Path("assets/scenes/simple_lab/manifest.json")
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


def simple_lab_manifest_path(scene_assets_root: str | Path) -> Path:
    assets_root = Path(scene_assets_root).resolve()
    candidates = []
    if assets_root.name == "simple_lab":
        candidates.append(assets_root / "manifest.json")
    candidates.extend(
        [
            assets_root / "simple_lab" / "manifest.json",
            assets_root / "scenes" / "simple_lab" / "manifest.json",
            assets_root / "assets" / "scenes" / "simple_lab" / "manifest.json",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if assets_root.name == "simple_lab":
        return assets_root / "manifest.json"
    if assets_root.name == "scenes":
        return assets_root / "simple_lab" / "manifest.json"
    if assets_root.name == "assets":
        return assets_root / "scenes" / "simple_lab" / "manifest.json"
    return assets_root / "simple_lab" / "manifest.json"


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
    elif assets_root.name == "scenes":
        simple_lab_root = assets_root / "simple_lab"
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
        balanced_render_backend: str = "pyrender",
    ):
        import pyrender

        self.width = int(width)
        self.height = int(height)
        self.lighting_mode = str(lighting_mode)
        self.balanced_render_backend = str(balanced_render_backend)
        self._pyrender = pyrender
        self.simple_lab_root = ensure_simple_lab_assets(scene_assets_root)
        self.manifest = load_simple_lab_manifest(scene_assets_root)
        self.layout: SimpleLabLayout | None = None
        self._scene_clear_color = np.array([243, 244, 246, 255], dtype=np.uint8)
        self._table_clear_color = np.array([0, 0, 0, 0], dtype=np.uint8)
        self._ambient_light = np.array([0.22, 0.22, 0.22], dtype=np.float32)
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
        self._table_transform_debug: dict[str, Any] | None = None
        self._table_world_bounds: tuple[np.ndarray, np.ndarray] | None = None
        self._wall_world_bounds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._pyrender_readback_mode = "cpu_fallback"
        self._pyrender_readback_reason: str | None = None
        self._pyrender_cuda_interop_supported = False
        if self.balanced_render_backend == "pyrender":
            (
                self._pyrender_cuda_interop_supported,
                self._pyrender_readback_reason,
            ) = probe_pyrender_cuda_bridge_support()
            self._pyrender_readback_mode = (
                "gl_cuda_interop"
                if self._pyrender_cuda_interop_supported
                else "cpu_fallback"
            )
        self._gpu_renderer = (
            None
            if self.balanced_render_backend == "pyrender"
            else SimpleLabGpuRenderer(lighting_mode=self.lighting_mode)
        )
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
        for layer_name in self._eager_layer_names:
            self._layer_entries[layer_name] = self._make_layer_entry(layer_name)

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
        if self._gpu_renderer is not None:
            self._gpu_renderer.delete()

    def pyrender_readback_mode(self) -> str:
        return str(self._pyrender_readback_mode)

    def pyrender_readback_reason(self) -> str | None:
        if self._pyrender_readback_reason is None:
            return None
        return str(self._pyrender_readback_reason)

    def uses_gpu_balanced_table_renderer(self) -> bool:
        return bool(
            self._gpu_renderer is not None
            and self._gpu_renderer.table_available
            and self.balanced_render_backend in {"gpu", "gpu_all"}
        )

    def uses_gpu_balanced_plane_renderer(self) -> bool:
        return bool(
            self._gpu_renderer is not None
            and self._gpu_renderer.available
            and self.balanced_render_backend == "gpu_all"
        )

    def uses_gpu_balanced_side_wall_renderer(self) -> bool:
        return bool(
            self._gpu_renderer is not None
            and self._gpu_renderer.available
            and self.balanced_render_backend in {"gpu", "gpu_all"}
        )

    def table_alignment_debug(self) -> dict[str, Any] | None:
        if self._table_alignment_debug is None:
            return None
        return dict(self._table_alignment_debug)

    def table_world_bounds(self) -> tuple[np.ndarray, np.ndarray] | None:
        if self._table_world_bounds is None:
            return None
        bounds_min, bounds_max = self._table_world_bounds
        return np.array(bounds_min, copy=True), np.array(bounds_max, copy=True)

    def wall_world_bounds(self, wall_name: str) -> tuple[np.ndarray, np.ndarray] | None:
        wall_key = str(wall_name)
        if wall_key not in self._wall_world_bounds:
            return None
        bounds_min, bounds_max = self._wall_world_bounds[wall_key]
        return np.array(bounds_min, copy=True), np.array(bounds_max, copy=True)

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
        return self._render_layer_eye(
            "full",
            camera_pose_world,
            intrinsic,
            width=width,
            height=height,
        )

    def render_background_eye(
        self,
        camera_pose_world: np.ndarray,
        intrinsic: np.ndarray,
        width: int | None = None,
        height: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._render_layer_eye(
            "background",
            camera_pose_world,
            intrinsic,
            width=width,
            height=height,
        )

    def render_background_eye_roi(
        self,
        camera_pose_world: np.ndarray,
        full_intrinsic: np.ndarray,
        roi_bounds: tuple[int, int, int, int],
    ) -> tuple[np.ndarray, np.ndarray] | tuple[torch.Tensor, torch.Tensor]:
        return self._render_layer_eye_roi(
            "background",
            camera_pose_world,
            full_intrinsic,
            roi_bounds,
        )

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
        if self.uses_gpu_balanced_table_renderer():
            render_width = self.width if width is None else int(width)
            render_height = self.height if height is None else int(height)
            return self._gpu_renderer.render_table(
                camera_pose_world,
                intrinsic,
                width=render_width,
                height=render_height,
            )
        return self._render_layer_eye(
            "table",
            camera_pose_world,
            intrinsic,
            width=width,
            height=height,
        )

    def render_table_eye_roi(
        self,
        camera_pose_world: np.ndarray,
        full_intrinsic: np.ndarray,
        roi_bounds: tuple[int, int, int, int],
        render_scale: float = 1.0,
        return_render_info: bool = False,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, dict[str, Any]] | tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        if self.uses_gpu_balanced_table_renderer():
            roi_intrinsic, render_width, render_height, render_info = (
                self._resolve_roi_render_params(
                    full_intrinsic,
                    roi_bounds,
                    render_scale=render_scale,
                )
            )
            render_color, render_depth = self._gpu_renderer.render_table(
                camera_pose_world,
                roi_intrinsic,
                width=render_width,
                height=render_height,
            )
            if return_render_info:
                return render_color, render_depth, render_info
            return render_color, render_depth
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
        if self._should_use_gpu_balanced_layer(layer_name):
            render_width = self.width if width is None else int(width)
            render_height = self.height if height is None else int(height)
            try:
                return self._gpu_renderer.render_plane_layer(
                    layer_name,
                    camera_pose_world,
                    intrinsic,
                    width=render_width,
                    height=render_height,
                )
            except Exception:
                pass
        return self._render_layer_eye(
            layer_name,
            camera_pose_world,
            intrinsic,
            width=width,
            height=height,
        )

    def _render_balanced_layer_roi(
        self,
        layer_name: str,
        camera_pose_world: np.ndarray,
        full_intrinsic: np.ndarray,
        roi_bounds: tuple[int, int, int, int],
    ) -> tuple[np.ndarray, np.ndarray] | tuple[torch.Tensor, torch.Tensor]:
        if self._should_use_gpu_balanced_layer(layer_name):
            roi_intrinsic, render_width, render_height, _ = self._resolve_roi_render_params(
                full_intrinsic,
                roi_bounds,
                render_scale=1.0,
            )
            try:
                return self._gpu_renderer.render_plane_layer(
                    layer_name,
                    camera_pose_world,
                    roi_intrinsic,
                    width=render_width,
                    height=render_height,
                )
            except Exception:
                pass
        return self._render_layer_eye_roi(
            layer_name,
            camera_pose_world,
            full_intrinsic,
            roi_bounds,
        )

    def _should_use_gpu_balanced_layer(self, layer_name: str) -> bool:
        layer_name = str(layer_name)
        if layer_name in {"balanced_left_wall", "balanced_right_wall"}:
            return self.uses_gpu_balanced_side_wall_renderer()
        return self.uses_gpu_balanced_plane_renderer()

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
            render_width = max(
                4,
                int(np.ceil((float(roi_width) * render_scale) / 4.0) * 4),
            )
            render_height = max(
                4,
                int(np.ceil((float(roi_height) * render_scale) / 4.0) * 4),
            )
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
            scene.add(
                fill_light,
                pose=trimesh.transformations.euler_matrix(-1.1, -0.4, 0.0),
            )
            scene.add(
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
        positioned_table_mesh = self._make_positioned_table_mesh()
        self._table_world_bounds = (
            positioned_table_mesh.bounds[0].astype(np.float32).copy(),
            positioned_table_mesh.bounds[1].astype(np.float32).copy(),
        )
        floor_mesh = self._make_floor_mesh()
        wall_meshes = self._make_wall_meshes()
        self._wall_world_bounds = {
            wall_name: (
                wall_mesh.bounds[0].astype(np.float32).copy(),
                wall_mesh.bounds[1].astype(np.float32).copy(),
            )
            for wall_name, wall_mesh in wall_meshes.items()
        }
        if self._gpu_renderer is not None:
            self._gpu_renderer.update_scene_geometry(
                positioned_table_mesh,
                floor_mesh,
                wall_meshes,
            )
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

            table_role = entry.get("table_role")
            if table_role is not None:
                table_mesh = positioned_table_mesh if table_role == "full" else None
                if (
                    table_mesh is not None
                    and table_mesh.vertices.shape[0] > 0
                    and table_mesh.faces.shape[0] > 0
                ):
                    entry["table_node"] = scene.add(
                        self._pyrender.Mesh.from_trimesh(
                            table_mesh.copy(),
                            smooth=False,
                        )
                    )
                else:
                    entry["table_node"] = None
            background_role = entry.get("background_role")
            if background_role in {"all", "floor"}:
                entry["floor_node"] = scene.add(
                    self._pyrender.Mesh.from_trimesh(
                        floor_mesh.copy(),
                        smooth=False,
                    )
                )
            if background_role in {"all", "front_back_walls"}:
                for wall_name in ("front", "back"):
                    entry["wall_nodes"].append(
                        scene.add(
                            self._pyrender.Mesh.from_trimesh(
                                wall_meshes[wall_name].copy(),
                                smooth=False,
                            )
                        )
                    )
            if background_role in {"all", "left_wall", "right_wall"}:
                wall_names = (
                    ("left", "right")
                    if background_role == "all"
                    else ("left",)
                    if background_role == "left_wall"
                    else ("right",)
                )
                for wall_name in wall_names:
                    entry["wall_nodes"].append(
                        scene.add(
                            self._pyrender.Mesh.from_trimesh(
                                wall_meshes[wall_name].copy(),
                                smooth=False,
                            )
                        )
                    )

    def _get_layer_renderer(self, layer_name: str, width: int, height: int):
        entry = self._ensure_layer_entry(layer_name)
        renderer = entry.get("renderer")
        if renderer is None:
            if self.balanced_render_backend == "pyrender":
                renderer_cls = (
                    PyrenderCudaInteropOffscreenRenderer
                    if self._pyrender_cuda_interop_supported
                    else self._pyrender.OffscreenRenderer
                )
                renderer_kwargs = {}
                if renderer_cls is PyrenderCudaInteropOffscreenRenderer:
                    renderer_kwargs["device"] = torch.device(
                        "cuda",
                        torch.cuda.current_device(),
                    )
                renderer = renderer_cls(
                    viewport_width=int(width),
                    viewport_height=int(height),
                    **renderer_kwargs,
                )
            else:
                renderer = self._pyrender.OffscreenRenderer(
                    viewport_width=int(width),
                    viewport_height=int(height),
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

    def _render_layer_eye(
        self,
        layer_name: str,
        camera_pose_world: np.ndarray,
        intrinsic: np.ndarray,
        width: int | None = None,
        height: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[torch.Tensor, torch.Tensor]:
        if self.layout is None:
            raise RuntimeError("Simple lab layout has not been configured.")
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
        roi_intrinsic, render_width, render_height, render_info = (
            self._resolve_roi_render_params(
                full_intrinsic,
                roi_bounds,
                render_scale=render_scale,
            )
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

    def _make_wall_meshes(self) -> dict[str, trimesh.Trimesh]:
        assert self.layout is not None
        room_width = float(self.layout.room_half_extent[0] * 2.0)
        room_depth = float(self.layout.room_half_extent[1] * 2.0)
        wall_height = float(self.layout.wall_height)
        center_x = float(self.layout.table_top_center[0])
        center_y = float(self.layout.table_top_center[1])
        floor_z = float(self.layout.floor_z)
        wall_z = floor_z - 0.5 * wall_height
        meshes: dict[str, trimesh.Trimesh] = {}

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
        meshes["back"] = back_wall

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
        meshes["front"] = front_wall

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
        meshes["left"] = left_wall

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
        meshes["right"] = right_wall
        return meshes
