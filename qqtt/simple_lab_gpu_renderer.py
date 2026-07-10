from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from PIL import Image

MeshRasterizer = None
RasterizationSettings = None
TexturesUV = None
Meshes = None
cameras_from_opencv_projection = None
_PYTORCH3D_IMPORT_ERROR = None
_PYTORCH3D_IMPORT_ATTEMPTED = False


def _load_pytorch3d() -> bool:
    """Load the optional legacy GPU scene backend only when it is selected."""
    global MeshRasterizer
    global RasterizationSettings
    global TexturesUV
    global Meshes
    global cameras_from_opencv_projection
    global _PYTORCH3D_IMPORT_ERROR
    global _PYTORCH3D_IMPORT_ATTEMPTED

    if _PYTORCH3D_IMPORT_ATTEMPTED:
        return _PYTORCH3D_IMPORT_ERROR is None
    _PYTORCH3D_IMPORT_ATTEMPTED = True
    try:
        from pytorch3d.renderer import (
            MeshRasterizer as _MeshRasterizer,
            RasterizationSettings as _RasterizationSettings,
            TexturesUV as _TexturesUV,
        )
        from pytorch3d.structures import Meshes as _Meshes
        from pytorch3d.utils import (
            cameras_from_opencv_projection as _cameras_from_opencv_projection,
        )
    except Exception as exc:  # pragma: no cover - availability is environment-specific
        _PYTORCH3D_IMPORT_ERROR = exc
        return False

    MeshRasterizer = _MeshRasterizer
    RasterizationSettings = _RasterizationSettings
    TexturesUV = _TexturesUV
    Meshes = _Meshes
    cameras_from_opencv_projection = _cameras_from_opencv_projection
    _PYTORCH3D_IMPORT_ERROR = None
    return True


@dataclass
class _PlaneRenderData:
    name: str
    p00_world: torch.Tensor
    u_axis_world: torch.Tensor
    v_axis_world: torch.Tensor
    u_axis_len2: float
    v_axis_len2: float
    normal_world: torch.Tensor
    uv00: torch.Tensor
    duv_u: torch.Tensor
    duv_v: torch.Tensor
    texture: torch.Tensor
    shading_rgb: torch.Tensor


@dataclass
class _MeshRenderData:
    vertices_world: torch.Tensor
    faces: torch.Tensor
    face_shading_rgb: torch.Tensor
    mesh: Meshes


@dataclass
class _FocusSubsetRenderData:
    mesh_parts: tuple[_MeshRenderData, ...]
    selection_entry_count: int
    source_mesh_count: int
    build_wall_s: float


class SimpleLabGpuRenderer:
    SCENE_CLEAR_RGBA = [243.0, 244.0, 246.0, 255.0]
    TABLE_CLEAR_RGBA = [0.0, 0.0, 0.0, 0.0]
    AMBIENT_LIGHT = 0.22
    KEY_LIGHT_INTENSITY = 3.5
    KEY_LIGHT_DIFFUSE_SCALE = 0.35
    FLOOR_COLOR_GAIN_RGB = (1.4, 1.4, 1.4)
    TABLE_COLOR_GAIN_RGB = (1.1, 1.1, 1.1)
    CV_FROM_GL = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
        dtype=torch.float32,
    )

    def __init__(
        self,
        lighting_mode: str = "simple",
        device: str | torch.device | None = None,
    ) -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.lighting_mode = str(lighting_mode)
        self._grid_cache: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
        self._plane_layers: dict[str, list[_PlaneRenderData]] = {}
        self._mesh_layers: dict[str, _MeshRenderData | None] = {}
        self._focus_subset_mesh_cache: dict[
            tuple[int, ...], _FocusSubsetRenderData | None
        ] = {}
        self._focus_catalog_world_by_id: dict[int, dict[str, object]] = {}
        self._focus_entry_geometry_world_by_id: dict[int, dict[str, object]] = {}
        self._focus_source_meshes_world: dict[str, trimesh.Trimesh] = {}
        self._raster_settings_cache: dict[tuple[int, int], RasterizationSettings] = {}
        self._last_focus_subset_debug: dict[str, float] | None = None
        self._scene_clear_rgba = torch.tensor(
            self.SCENE_CLEAR_RGBA,
            dtype=torch.float32,
            device=self.device,
        )
        self._table_clear_rgba = torch.tensor(
            self.TABLE_CLEAR_RGBA,
            dtype=torch.float32,
            device=self.device,
        )
        self._table_color_gain_rgb = torch.tensor(
            self.TABLE_COLOR_GAIN_RGB,
            dtype=torch.float32,
            device=self.device,
        )
        self._key_light_direction_world = self._build_key_light_direction_world()

    @property
    def available(self) -> bool:
        return self.device.type == "cuda"

    @property
    def table_available(self) -> bool:
        return self.available and _load_pytorch3d()

    def delete(self) -> None:
        self._grid_cache.clear()
        self._plane_layers.clear()
        self._mesh_layers.clear()
        self._focus_subset_mesh_cache.clear()
        self._focus_catalog_world_by_id.clear()
        self._focus_entry_geometry_world_by_id.clear()
        self._focus_source_meshes_world.clear()
        self._raster_settings_cache.clear()
        self._last_focus_subset_debug = None

    def update_scene_geometry(
        self,
        *,
        background_mesh: trimesh.Trimesh,
        full_scene_mesh: trimesh.Trimesh,
        positioned_table_mesh: trimesh.Trimesh,
        floor_mesh: trimesh.Trimesh,
        wall_meshes: dict[str, trimesh.Trimesh],
        focus_catalog_world_by_id: dict[int, dict[str, object]] | None = None,
        focus_entry_geometry_world_by_id: dict[int, dict[str, object]] | None = None,
        focus_source_meshes_world: dict[str, trimesh.Trimesh] | None = None,
    ) -> None:
        if not self.available:
            return

        self._plane_layers = {}
        self._mesh_layers = {}
        if self.table_available:
            self._mesh_layers["background"] = self._mesh_from_trimesh_or_none(
                background_mesh
            )
            self._mesh_layers["full"] = self._mesh_from_trimesh_or_none(
                full_scene_mesh
            )
            self._mesh_layers["table"] = self._mesh_from_trimesh_or_none(
                positioned_table_mesh
            )
            self._mesh_layers["balanced_near_floor"] = (
                self._mesh_from_trimesh_or_none(floor_mesh)
            )
            self._mesh_layers["balanced_left_wall"] = (
                self._mesh_from_trimesh_or_none(wall_meshes["left"])
            )
            self._mesh_layers["balanced_right_wall"] = (
                self._mesh_from_trimesh_or_none(wall_meshes["right"])
            )
            self._mesh_layers["balanced_far_front_back_walls"] = (
                self._mesh_from_trimesh_or_none(
                    trimesh.util.concatenate(
                        [
                            mesh.copy()
                            for mesh in (wall_meshes["front"], wall_meshes["back"])
                            if mesh is not None and mesh.faces.shape[0] > 0
                        ]
                    )
                    if (
                        wall_meshes["front"] is not None
                        and wall_meshes["back"] is not None
                        and (
                            wall_meshes["front"].faces.shape[0] > 0
                            or wall_meshes["back"].faces.shape[0] > 0
                        )
                    )
                    else (
                        wall_meshes["front"]
                        if wall_meshes["front"] is not None
                        and wall_meshes["front"].faces.shape[0] > 0
                        else wall_meshes["back"]
                    )
                )
            )
        self._focus_subset_mesh_cache.clear()
        self._focus_catalog_world_by_id = {
            int(entry_id): dict(entry)
            for entry_id, entry in (focus_catalog_world_by_id or {}).items()
        }
        self._focus_entry_geometry_world_by_id = {
            int(entry_id): self._copy_focus_geometry_payload(payload)
            for entry_id, payload in (focus_entry_geometry_world_by_id or {}).items()
        }
        self._focus_source_meshes_world = {
            str(source_name): source_mesh.copy()
            for source_name, source_mesh in (focus_source_meshes_world or {}).items()
        }
        self._last_focus_subset_debug = None

    def last_focus_subset_debug(self) -> dict[str, float] | None:
        if self._last_focus_subset_debug is None:
            return None
        return dict(self._last_focus_subset_debug)

    def render_plane_layer(
        self,
        layer_name: str,
        camera_pose_world: np.ndarray,
        intrinsic: np.ndarray,
        width: int,
        height: int,
        clear_rgba: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.available:
            raise RuntimeError("SimpleLabGpuRenderer is not available on this device.")
        planes = self._plane_layers.get(str(layer_name), [])
        if clear_rgba is None:
            clear_rgba = self._scene_clear_rgba
        return self._render_planes(
            planes,
            camera_pose_world=np.asarray(camera_pose_world, dtype=np.float32),
            intrinsic=np.asarray(intrinsic, dtype=np.float32),
            width=int(width),
            height=int(height),
            clear_rgba=clear_rgba,
        )

    def render_table(
        self,
        camera_pose_world: np.ndarray,
        intrinsic: np.ndarray,
        width: int,
        height: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.render_mesh_layer(
            "table",
            camera_pose_world,
            intrinsic,
            width,
            height,
            clear_rgba=self._table_clear_rgba,
        )

    def render_mesh_layer(
        self,
        layer_name: str,
        camera_pose_world: np.ndarray,
        intrinsic: np.ndarray,
        width: int,
        height: int,
        clear_rgba: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.table_available:
            if _PYTORCH3D_IMPORT_ERROR is not None:
                raise RuntimeError(
                    "SimpleLabGpuRenderer mesh path requires PyTorch3D."
                ) from _PYTORCH3D_IMPORT_ERROR
            raise RuntimeError("SimpleLabGpuRenderer is not available on this device.")
        if clear_rgba is None:
            clear_rgba = self._table_clear_rgba
        mesh = self._mesh_layers.get(str(layer_name))
        if mesh is None:
            return self._empty_render(int(width), int(height), clear_rgba)
        return self._render_mesh(
            mesh,
            camera_pose_world=np.asarray(camera_pose_world, dtype=np.float32),
            intrinsic=np.asarray(intrinsic, dtype=np.float32),
            width=int(width),
            height=int(height),
            clear_rgba=clear_rgba,
        )

    def render_focus_subset(
        self,
        selected_entry_ids: list[int] | tuple[int, ...],
        camera_pose_world: np.ndarray,
        intrinsic: np.ndarray,
        width: int,
        height: int,
        clear_rgba: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.table_available:
            if _PYTORCH3D_IMPORT_ERROR is not None:
                raise RuntimeError(
                    "SimpleLabGpuRenderer focus-subset path requires PyTorch3D."
                ) from _PYTORCH3D_IMPORT_ERROR
            raise RuntimeError("SimpleLabGpuRenderer is not available on this device.")
        if clear_rgba is None:
            clear_rgba = self._table_clear_rgba
        selected_ids = tuple(sorted({int(v) for v in selected_entry_ids}))
        if not selected_ids:
            self._last_focus_subset_debug = {
                "focus_selection_cache_hit_ratio": 1.0,
                "focus_selection_cache_miss_ratio": 0.0,
                "focus_selection_cache_build_wall_s": 0.0,
                "focus_selection_entry_count": 0.0,
                "focus_selection_source_mesh_count": 0.0,
            }
            return self._empty_render(int(width), int(height), clear_rgba)
        focus_subset = self._focus_subset_mesh_cache.get(selected_ids)
        cache_hit = focus_subset is not None
        if focus_subset is None and selected_ids not in self._focus_subset_mesh_cache:
            focus_subset = self._build_focus_subset_mesh(selected_ids)
            self._focus_subset_mesh_cache[selected_ids] = focus_subset
            cache_hit = False
        build_wall_s = 0.0 if focus_subset is None else float(focus_subset.build_wall_s)
        source_mesh_count = (
            0.0 if focus_subset is None else float(focus_subset.source_mesh_count)
        )
        self._last_focus_subset_debug = {
            "focus_selection_cache_hit_ratio": 1.0 if cache_hit else 0.0,
            "focus_selection_cache_miss_ratio": 0.0 if cache_hit else 1.0,
            "focus_selection_cache_build_wall_s": float(build_wall_s),
            "focus_selection_entry_count": float(len(selected_ids)),
            "focus_selection_source_mesh_count": float(source_mesh_count),
        }
        if focus_subset is None:
            return self._empty_render(int(width), int(height), clear_rgba)
        return self._render_focus_subset_meshes(
            focus_subset.mesh_parts,
            camera_pose_world=np.asarray(camera_pose_world, dtype=np.float32),
            intrinsic=np.asarray(intrinsic, dtype=np.float32),
            width=int(width),
            height=int(height),
            clear_rgba=clear_rgba,
        )

    def _build_key_light_direction_world(self) -> torch.Tensor:
        rotation = trimesh.transformations.euler_matrix(-0.7, 0.35, 0.0)[:3, :3]
        local_forward = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        direction_world = rotation @ local_forward
        norm = max(float(np.linalg.norm(direction_world)), 1e-6)
        return torch.tensor(
            direction_world / norm,
            dtype=torch.float32,
            device=self.device,
        )

    def _cached_pixel_grid(
        self,
        width: int,
        height: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (int(width), int(height))
        cached = self._grid_cache.get(key)
        if cached is not None:
            return cached
        xs = (
            torch.arange(int(width), dtype=torch.float32, device=self.device)
            .unsqueeze(0)
            .expand(int(height), int(width))
            + 0.5
        )
        ys = (
            torch.arange(int(height), dtype=torch.float32, device=self.device)
            .unsqueeze(1)
            .expand(int(height), int(width))
            + 0.5
        )
        self._grid_cache[key] = (xs, ys)
        return xs, ys

    def _camera_pose_world_to_cv(
        self,
        camera_pose_world: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        camera_pose_world_t = torch.as_tensor(
            camera_pose_world,
            dtype=torch.float32,
            device=self.device,
        )
        pose_world_cv = torch.eye(4, dtype=torch.float32, device=self.device)
        pose_world_cv[:3, :3] = camera_pose_world_t[:3, :3] @ self.CV_FROM_GL.to(
            device=self.device,
        )
        pose_world_cv[:3, 3] = camera_pose_world_t[:3, 3]
        w2c_cv = torch.linalg.inv(pose_world_cv)
        return pose_world_cv, w2c_cv, pose_world_cv[:3, 3]

    def _camera_rays_world(
        self,
        camera_pose_world: np.ndarray,
        intrinsic: np.ndarray,
        width: int,
        height: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        xs, ys = self._cached_pixel_grid(width, height)
        intrinsic_t = torch.as_tensor(
            intrinsic,
            dtype=torch.float32,
            device=self.device,
        )
        pose_world_cv, _, camera_origin_world = self._camera_pose_world_to_cv(
            camera_pose_world
        )
        dirs_cam = torch.stack(
            [
                (xs - intrinsic_t[0, 2]) / max(float(intrinsic_t[0, 0]), 1e-6),
                (ys - intrinsic_t[1, 2]) / max(float(intrinsic_t[1, 1]), 1e-6),
                torch.ones_like(xs),
            ],
            dim=-1,
        )
        dirs_world = dirs_cam @ pose_world_cv[:3, :3].transpose(0, 1)
        return dirs_world, camera_origin_world, intrinsic_t

    def _render_planes(
        self,
        planes: list[_PlaneRenderData],
        camera_pose_world: np.ndarray,
        intrinsic: np.ndarray,
        width: int,
        height: int,
        clear_rgba: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        color = clear_rgba.view(1, 1, 4).expand(height, width, 4).clone()
        depth = torch.zeros((height, width), dtype=torch.float32, device=self.device)
        if not planes:
            return color, depth

        dirs_world, camera_origin_world, _ = self._camera_rays_world(
            camera_pose_world,
            intrinsic,
            width,
            height,
        )
        for plane in planes:
            numerator = torch.dot(
                plane.normal_world,
                plane.p00_world - camera_origin_world,
            )
            denom = torch.sum(dirs_world * plane.normal_world.view(1, 1, 3), dim=-1)
            valid = denom.abs() > 1e-6
            t = torch.where(
                valid,
                numerator / denom.clamp(min=-1.0e6, max=1.0e6),
                torch.zeros_like(denom),
            )
            valid = valid & (t > 1.0e-4)
            if not bool(valid.any().item()):
                continue

            points_world = (
                camera_origin_world.view(1, 1, 3) + t.unsqueeze(-1) * dirs_world
            )
            rel = points_world - plane.p00_world.view(1, 1, 3)
            alpha = torch.sum(rel * plane.u_axis_world.view(1, 1, 3), dim=-1) / max(
                plane.u_axis_len2,
                1.0e-6,
            )
            beta = torch.sum(rel * plane.v_axis_world.view(1, 1, 3), dim=-1) / max(
                plane.v_axis_len2,
                1.0e-6,
            )
            valid = (
                valid
                & (alpha >= 0.0)
                & (alpha <= 1.0)
                & (beta >= 0.0)
                & (beta <= 1.0)
            )
            if not bool(valid.any().item()):
                continue

            uv = (
                plane.uv00.view(1, 1, 2)
                + alpha.unsqueeze(-1) * plane.duv_u.view(1, 1, 2)
                + beta.unsqueeze(-1) * plane.duv_v.view(1, 1, 2)
            )
            rgb = self._sample_texture(plane.texture, uv, wrap=True)
            rgb = torch.clamp(rgb * plane.shading_rgb.view(1, 1, 3), 0.0, 255.0)

            nearer = (depth <= 0.0) | (t < depth)
            update_mask = valid & nearer
            if not bool(update_mask.any().item()):
                continue
            color_rgb = color[..., :3]
            color_rgb[update_mask] = rgb[update_mask]
            color[..., 3][update_mask] = 255.0
            depth[update_mask] = t[update_mask]
        return color.contiguous(), depth.contiguous()

    def _empty_render(
        self,
        width: int,
        height: int,
        clear_rgba: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        color = clear_rgba.view(1, 1, 4).expand(int(height), int(width), 4).clone()
        depth = torch.zeros(
            (int(height), int(width)),
            dtype=torch.float32,
            device=self.device,
        )
        return color.contiguous(), depth.contiguous()

    def _render_mesh(
        self,
        mesh: _MeshRenderData,
        camera_pose_world: np.ndarray,
        intrinsic: np.ndarray,
        width: int,
        height: int,
        clear_rgba: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        color = clear_rgba.view(1, 1, 4).expand(height, width, 4).clone()
        depth = torch.zeros((height, width), dtype=torch.float32, device=self.device)
        intrinsic_t = torch.as_tensor(
            intrinsic,
            dtype=torch.float32,
            device=self.device,
        )
        _, w2c_cv, _ = self._camera_pose_world_to_cv(camera_pose_world)
        image_size = torch.tensor(
            [[float(height), float(width)]],
            dtype=torch.float32,
            device=self.device,
        )
        cameras = cameras_from_opencv_projection(
            R=w2c_cv[:3, :3].unsqueeze(0),
            tvec=w2c_cv[:3, 3].unsqueeze(0),
            camera_matrix=intrinsic_t.unsqueeze(0),
            image_size=image_size,
        )
        rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=self._raster_settings(width, height),
        )
        fragments = rasterizer(mesh.mesh)
        visible = fragments.pix_to_face[0, ..., 0] >= 0
        if not bool(visible.any().item()):
            return color.contiguous(), depth.contiguous()

        face_ids = fragments.pix_to_face[0, ..., 0]
        bary_coords = fragments.bary_coords[0, ..., 0, :]
        sampled_rgb = mesh.mesh.sample_textures(fragments)[0, ..., 0, :]
        rgb = torch.zeros((height, width, 3), dtype=torch.float32, device=self.device)

        vertices_h = torch.cat(
            [
                mesh.vertices_world,
                torch.ones(
                    (mesh.vertices_world.shape[0], 1),
                    dtype=torch.float32,
                    device=self.device,
                ),
            ],
            dim=1,
        )
        vertices_cam_z = (vertices_h @ w2c_cv.transpose(0, 1))[:, 2]
        rgb[visible] = torch.clamp(
            sampled_rgb[visible]
            * mesh.face_shading_rgb[face_ids[visible]]
            * 255.0,
            0.0,
            255.0,
        )

        faces_depth = vertices_cam_z[mesh.faces]
        depth_values = torch.zeros((height, width), dtype=torch.float32, device=self.device)
        depth_values[visible] = torch.sum(
            faces_depth[face_ids[visible]] * bary_coords[visible],
            dim=-1,
        )
        valid_depth = visible & torch.isfinite(depth_values) & (depth_values > 1.0e-4)
        if not bool(valid_depth.any().item()):
            return color.contiguous(), depth.contiguous()

        color[..., :3][valid_depth] = rgb[valid_depth]
        color[..., 3][valid_depth] = 255.0
        depth[valid_depth] = depth_values[valid_depth]
        return color.contiguous(), depth.contiguous()

    def _render_focus_subset_meshes(
        self,
        mesh_parts: tuple[_MeshRenderData, ...],
        camera_pose_world: np.ndarray,
        intrinsic: np.ndarray,
        width: int,
        height: int,
        clear_rgba: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not mesh_parts:
            return self._empty_render(int(width), int(height), clear_rgba)
        if len(mesh_parts) == 1:
            return self._render_mesh(
                mesh_parts[0],
                camera_pose_world=camera_pose_world,
                intrinsic=intrinsic,
                width=int(width),
                height=int(height),
                clear_rgba=clear_rgba,
            )
        color, depth = self._empty_render(int(width), int(height), clear_rgba)
        for mesh_part in mesh_parts:
            part_color, part_depth = self._render_mesh(
                mesh_part,
                camera_pose_world=camera_pose_world,
                intrinsic=intrinsic,
                width=int(width),
                height=int(height),
                clear_rgba=clear_rgba,
            )
            valid_depth = part_depth > 1.0e-4
            update_mask = valid_depth & ((depth <= 0.0) | (part_depth < depth))
            if not bool(update_mask.any().item()):
                continue
            color[update_mask] = part_color[update_mask]
            depth[update_mask] = part_depth[update_mask]
        return color.contiguous(), depth.contiguous()

    def _raster_settings(self, width: int, height: int) -> RasterizationSettings:
        key = (int(width), int(height))
        cached = self._raster_settings_cache.get(key)
        if cached is not None:
            return cached
        settings = RasterizationSettings(
            image_size=(int(height), int(width)),
            blur_radius=0.0,
            faces_per_pixel=1,
            perspective_correct=True,
            cull_backfaces=False,
            cull_to_frustum=True,
        )
        self._raster_settings_cache[key] = settings
        return settings

    def _sample_texture(
        self,
        texture: torch.Tensor,
        uv: torch.Tensor,
        wrap: bool,
    ) -> torch.Tensor:
        uv_x = uv[..., 0]
        uv_y = uv[..., 1]
        if wrap:
            uv_x = uv_x - torch.floor(uv_x)
            uv_y = uv_y - torch.floor(uv_y)
        else:
            uv_x = uv_x.clamp(0.0, 1.0)
            uv_y = uv_y.clamp(0.0, 1.0)
        grid = torch.stack(
            [
                uv_x * 2.0 - 1.0,
                (1.0 - uv_y) * 2.0 - 1.0,
            ],
            dim=-1,
        ).unsqueeze(0)
        sampled = F.grid_sample(
            texture,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return (sampled.squeeze(0).permute(1, 2, 0) * 255.0).contiguous()

    def _mesh_from_trimesh(self, mesh: trimesh.Trimesh) -> _MeshRenderData:
        if not self.table_available:
            raise RuntimeError("PyTorch3D table rasterization is not available.")
        vertices_world = torch.as_tensor(
            np.asarray(mesh.vertices, dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        faces = torch.as_tensor(
            np.asarray(mesh.faces, dtype=np.int64),
            dtype=torch.long,
            device=self.device,
        )
        uvs_np = np.asarray(getattr(mesh.visual, "uv", None), dtype=np.float32)
        if uvs_np.ndim != 2 or uvs_np.shape[0] != vertices_world.shape[0]:
            raise RuntimeError("Expected per-vertex UVs for GPU table rendering.")
        verts_uvs = torch.as_tensor(
            uvs_np,
            dtype=torch.float32,
            device=self.device,
        )
        # Trimesh/PIL textures use top-left image origin, while PyTorch3D UVs
        # are sampled with a bottom-left V origin.
        verts_uvs = verts_uvs.clone()
        verts_uvs[:, 1] = 1.0 - verts_uvs[:, 1]
        texture_maps = self._mesh_texture_tensor(mesh)
        face_normals = torch.as_tensor(
            np.asarray(mesh.face_normals, dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        face_shading_rgb = self._compute_mesh_face_shading(face_normals)
        textures = TexturesUV(
            maps=texture_maps,
            faces_uvs=[faces],
            verts_uvs=[verts_uvs],
            padding_mode="border",
            align_corners=True,
            sampling_mode="bilinear",
        )
        pytorch3d_mesh = Meshes(
            verts=[vertices_world],
            faces=[faces],
            textures=textures,
        )
        return _MeshRenderData(
            vertices_world=vertices_world,
            faces=faces,
            face_shading_rgb=face_shading_rgb,
            mesh=pytorch3d_mesh,
        )

    def _mesh_from_trimesh_or_none(
        self,
        mesh: trimesh.Trimesh | None,
    ) -> _MeshRenderData | None:
        if mesh is None:
            return None
        if mesh.faces.shape[0] <= 0 or mesh.vertices.shape[0] <= 0:
            return None
        return self._mesh_from_trimesh(mesh)

    def _build_focus_subset_mesh(
        self,
        selected_entry_ids: tuple[int, ...],
    ) -> _FocusSubsetRenderData | None:
        build_start = time.perf_counter()
        if self._focus_entry_geometry_world_by_id:
            geometry_entries_by_source: dict[str, list[dict[str, object]]] = {}
            for entry_id in selected_entry_ids:
                geometry_entry = self._focus_entry_geometry_world_by_id.get(int(entry_id))
                if geometry_entry is None:
                    continue
                source_mesh = str(geometry_entry.get("source_mesh", ""))
                geometry_entries_by_source.setdefault(source_mesh, []).append(
                    geometry_entry
                )
            if not geometry_entries_by_source:
                return None
            mesh_parts: list[_MeshRenderData] = []
            for source_mesh, source_entries in sorted(geometry_entries_by_source.items()):
                group_mesh = self._compose_focus_selection_group_mesh(
                    source_mesh,
                    source_entries,
                )
                group_render_data = self._mesh_from_trimesh_or_none(group_mesh)
                if group_render_data is not None:
                    mesh_parts.append(group_render_data)
            if not mesh_parts:
                return None
            return _FocusSubsetRenderData(
                mesh_parts=tuple(mesh_parts),
                selection_entry_count=int(len(selected_entry_ids)),
                source_mesh_count=int(len(mesh_parts)),
                build_wall_s=float(time.perf_counter() - build_start),
            )
        if not self._focus_source_meshes_world or not self._focus_catalog_world_by_id:
            return None
        sliced_meshes: list[trimesh.Trimesh] = []
        for entry_id in selected_entry_ids:
            catalog_entry = self._focus_catalog_world_by_id.get(int(entry_id))
            if catalog_entry is None:
                continue
            source_mesh_name = str(catalog_entry.get("source_mesh", ""))
            source_mesh_world = self._focus_source_meshes_world.get(source_mesh_name)
            if source_mesh_world is None:
                continue
            face_indices = np.asarray(
                catalog_entry.get("face_indices", []),
                dtype=np.int64,
            )
            sliced_mesh = self._slice_mesh_by_face_indices(
                source_mesh_world,
                face_indices,
            )
            if sliced_mesh.faces.shape[0] <= 0 or sliced_mesh.vertices.shape[0] <= 0:
                continue
            sliced_meshes.append(sliced_mesh)
        if not sliced_meshes:
            return None
        subset_mesh = (
            sliced_meshes[0]
            if len(sliced_meshes) == 1
            else trimesh.util.concatenate(sliced_meshes)
        )
        subset_render_data = self._mesh_from_trimesh_or_none(subset_mesh)
        if subset_render_data is None:
            return None
        return _FocusSubsetRenderData(
            mesh_parts=(subset_render_data,),
            selection_entry_count=int(len(selected_entry_ids)),
            source_mesh_count=1,
            build_wall_s=float(time.perf_counter() - build_start),
        )

    def _slice_mesh_by_face_indices(
        self,
        mesh: trimesh.Trimesh,
        face_indices: np.ndarray,
    ) -> trimesh.Trimesh:
        face_indices = np.asarray(face_indices, dtype=np.int64)
        if face_indices.size == 0:
            return trimesh.Trimesh(
                vertices=np.zeros((0, 3), dtype=np.float32),
                faces=np.zeros((0, 3), dtype=np.int64),
                process=False,
            )
        valid_face_indices = face_indices[
            (face_indices >= 0) & (face_indices < mesh.faces.shape[0])
        ]
        if valid_face_indices.size == 0:
            return trimesh.Trimesh(
                vertices=np.zeros((0, 3), dtype=np.float32),
                faces=np.zeros((0, 3), dtype=np.int64),
                process=False,
            )
        face_mask = np.zeros((mesh.faces.shape[0],), dtype=bool)
        face_mask[valid_face_indices] = True
        return mesh.submesh([face_mask], append=True, only_watertight=False)

    def _copy_focus_geometry_payload(
        self,
        payload: dict[str, object],
    ) -> dict[str, object]:
        return {
            "entry_id": int(payload.get("entry_id", -1)),
            "source_mesh": str(payload.get("source_mesh", "")),
            "face_count": int(payload.get("face_count", 0)),
            "vertices": np.asarray(
                payload.get("vertices", np.zeros((0, 3), dtype=np.float32)),
                dtype=np.float32,
            ).copy(),
            "faces": np.asarray(
                payload.get("faces", np.zeros((0, 3), dtype=np.int64)),
                dtype=np.int64,
            ).copy(),
            "visual_kind": str(payload.get("visual_kind", "none")),
            "uv": (
                None
                if payload.get("uv") is None
                else np.asarray(payload.get("uv"), dtype=np.float32).copy()
            ),
            "vertex_colors": (
                None
                if payload.get("vertex_colors") is None
                else np.asarray(payload.get("vertex_colors"), dtype=np.uint8).copy()
            ),
            "face_colors": (
                None
                if payload.get("face_colors") is None
                else np.asarray(payload.get("face_colors"), dtype=np.uint8).copy()
            ),
        }

    def _compose_focus_selection_group_mesh(
        self,
        source_mesh: str,
        geometry_entries: list[dict[str, object]],
    ) -> trimesh.Trimesh | None:
        if not geometry_entries:
            return None
        vertices_parts: list[np.ndarray] = []
        faces_parts: list[np.ndarray] = []
        uv_parts: list[np.ndarray] = []
        vertex_color_parts: list[np.ndarray] = []
        face_color_parts: list[np.ndarray] = []
        vertex_offset = 0
        visual_kind = str(geometry_entries[0].get("visual_kind", "none"))
        for geometry_entry in geometry_entries:
            vertices = np.asarray(
                geometry_entry.get("vertices"),
                dtype=np.float32,
            )
            faces = np.asarray(
                geometry_entry.get("faces"),
                dtype=np.int64,
            )
            if (
                vertices.ndim != 2
                or vertices.shape[1] != 3
                or faces.ndim != 2
                or faces.shape[1] != 3
                or vertices.shape[0] <= 0
                or faces.shape[0] <= 0
            ):
                continue
            vertices_parts.append(vertices.copy())
            faces_parts.append(faces.copy() + int(vertex_offset))
            vertex_offset += int(vertices.shape[0])
            uv = geometry_entry.get("uv")
            if uv is not None:
                uv_parts.append(np.asarray(uv, dtype=np.float32).copy())
            vertex_colors = geometry_entry.get("vertex_colors")
            if vertex_colors is not None:
                vertex_color_parts.append(
                    np.asarray(vertex_colors, dtype=np.uint8).copy()
                )
            face_colors = geometry_entry.get("face_colors")
            if face_colors is not None:
                face_color_parts.append(
                    np.asarray(face_colors, dtype=np.uint8).copy()
                )
        if not vertices_parts or not faces_parts:
            return None
        mesh = trimesh.Trimesh(
            vertices=np.concatenate(vertices_parts, axis=0),
            faces=np.concatenate(faces_parts, axis=0),
            process=False,
        )
        source_visual = getattr(
            self._focus_source_meshes_world.get(str(source_mesh)),
            "visual",
            None,
        )
        if visual_kind == "texture" and uv_parts:
            material = None if source_visual is None else getattr(source_visual, "material", None)
            mesh.visual = trimesh.visual.TextureVisuals(
                uv=np.concatenate(uv_parts, axis=0),
                material=material,
            )
        elif visual_kind == "vertex" and vertex_color_parts:
            mesh.visual = trimesh.visual.ColorVisuals(
                mesh=mesh,
                vertex_colors=np.concatenate(vertex_color_parts, axis=0),
            )
        elif visual_kind == "face" and face_color_parts:
            mesh.visual = trimesh.visual.ColorVisuals(
                mesh=mesh,
                face_colors=np.concatenate(face_color_parts, axis=0),
            )
        return mesh

    def _plane_from_mesh(self, name: str, mesh: trimesh.Trimesh) -> _PlaneRenderData:
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        uvs = np.asarray(getattr(mesh.visual, "uv", None), dtype=np.float32)
        if vertices.shape[0] < 4 or uvs.shape[0] < 4:
            raise RuntimeError(f"Expected quad mesh for plane '{name}'.")
        p00 = vertices[0]
        p10 = vertices[1]
        p01 = vertices[3]
        uv00 = uvs[0]
        uv10 = uvs[1]
        uv01 = uvs[3]
        u_axis = p10 - p00
        v_axis = p01 - p00
        normal = np.cross(u_axis, v_axis)
        normal_norm = max(float(np.linalg.norm(normal)), 1.0e-6)
        normal = normal / normal_norm
        shading_rgb = self._compute_face_shading_tensor(normal)
        if name == "floor" and self.lighting_mode == "simple":
            shading_rgb = shading_rgb * torch.tensor(
                self.FLOOR_COLOR_GAIN_RGB,
                dtype=torch.float32,
                device=self.device,
            )
        return _PlaneRenderData(
            name=name,
            p00_world=torch.tensor(p00, dtype=torch.float32, device=self.device),
            u_axis_world=torch.tensor(u_axis, dtype=torch.float32, device=self.device),
            v_axis_world=torch.tensor(v_axis, dtype=torch.float32, device=self.device),
            u_axis_len2=float(np.dot(u_axis, u_axis)),
            v_axis_len2=float(np.dot(v_axis, v_axis)),
            normal_world=torch.tensor(normal, dtype=torch.float32, device=self.device),
            uv00=torch.tensor(uv00, dtype=torch.float32, device=self.device),
            duv_u=torch.tensor(uv10 - uv00, dtype=torch.float32, device=self.device),
            duv_v=torch.tensor(uv01 - uv00, dtype=torch.float32, device=self.device),
            texture=self._mesh_texture_tensor(mesh).permute(0, 3, 1, 2).contiguous(),
            shading_rgb=shading_rgb,
        )

    def _compute_normal_shading(self, normals_world: torch.Tensor) -> torch.Tensor:
        if self.lighting_mode != "simple":
            return torch.ones_like(normals_world, dtype=torch.float32)
        normals = F.normalize(normals_world, dim=-1, eps=1.0e-6)
        light_dir = (-self._key_light_direction_world).view(
            *((1,) * max(normals.ndim - 1, 0)),
            3,
        )
        light_alignment = torch.clamp(
            torch.sum(
                normals * light_dir,
                dim=-1,
            ),
            min=0.0,
        )
        shade = self.AMBIENT_LIGHT + self.KEY_LIGHT_DIFFUSE_SCALE * light_alignment
        return torch.clamp(shade, 0.0, 1.0).unsqueeze(-1).expand(*normals.shape)

    def _compute_mesh_face_shading(self, normals_world: torch.Tensor) -> torch.Tensor:
        if self.lighting_mode != "simple":
            return torch.ones_like(normals_world, dtype=torch.float32)
        normals = F.normalize(normals_world, dim=-1, eps=1.0e-6)
        light_alignment = torch.abs(
            torch.sum(
                normals * (-self._key_light_direction_world).view(1, 3),
                dim=-1,
            )
        )
        shade = self.AMBIENT_LIGHT + self.KEY_LIGHT_DIFFUSE_SCALE * light_alignment
        shading_rgb = torch.clamp(shade, 0.0, 1.0).unsqueeze(-1).expand_as(normals)
        return shading_rgb * self._table_color_gain_rgb.view(1, 3)

    def _compute_face_shading_tensor(self, normal_world: np.ndarray) -> torch.Tensor:
        if self.lighting_mode != "simple":
            return torch.ones((3,), dtype=torch.float32, device=self.device)
        normal_t = torch.tensor(normal_world, dtype=torch.float32, device=self.device)
        normal_t = F.normalize(normal_t, dim=0, eps=1.0e-6)
        light_alignment = torch.clamp(
            torch.dot(normal_t, -self._key_light_direction_world),
            min=0.0,
        )
        shade = self.AMBIENT_LIGHT + self.KEY_LIGHT_DIFFUSE_SCALE * light_alignment
        shade = torch.clamp(shade, 0.0, 1.0)
        return torch.full((3,), float(shade.item()), dtype=torch.float32, device=self.device)

    def _mesh_texture_tensor(self, mesh: trimesh.Trimesh) -> torch.Tensor:
        visual_material = getattr(mesh.visual, "material", None)
        material = getattr(
            visual_material,
            "baseColorTexture",
            None,
        )
        if material is None:
            material = getattr(visual_material, "image", None)
        if material is None:
            material = getattr(mesh.visual, "image", None)
        if material is None:
            raise RuntimeError("Expected textured mesh for GPU simple-lab rendering.")
        if not isinstance(material, Image.Image):
            material = Image.open(Path(material))
        texture_srgb = np.asarray(material.convert("RGB"), dtype=np.float32) / 255.0
        diffuse_rgb = np.ones((1, 1, 3), dtype=np.float32)
        diffuse = getattr(visual_material, "diffuse", None)
        if diffuse is not None:
            diffuse_np = np.asarray(diffuse, dtype=np.float32).reshape(-1)
            if diffuse_np.size >= 3:
                diffuse_rgb = np.clip(
                    diffuse_np[:3].reshape(1, 1, 3) / 255.0,
                    0.0,
                    1.0,
                )
        # PyTorch3D expects linear texture values, while the shipped PNG assets
        # and trimesh SimpleMaterial diffuse colors are authored in sRGB space.
        texture_np = np.power(
            np.clip(texture_srgb * diffuse_rgb, 0.0, 1.0),
            2.2,
        )
        return torch.from_numpy(texture_np).to(
            self.device,
            dtype=torch.float32,
        ).unsqueeze(0).contiguous()
