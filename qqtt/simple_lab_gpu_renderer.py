from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from PIL import Image

try:
    from pytorch3d.renderer import MeshRasterizer, RasterizationSettings, TexturesUV
    from pytorch3d.structures import Meshes
    from pytorch3d.utils import cameras_from_opencv_projection
except Exception as exc:  # pragma: no cover - availability is environment-specific
    MeshRasterizer = None
    RasterizationSettings = None
    TexturesUV = None
    Meshes = None
    cameras_from_opencv_projection = None
    _PYTORCH3D_IMPORT_ERROR = exc
else:
    _PYTORCH3D_IMPORT_ERROR = None


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
        self._table_mesh: _MeshRenderData | None = None
        self._raster_settings_cache: dict[tuple[int, int], RasterizationSettings] = {}
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
        return self.device.type == "cuda" and self.lighting_mode == "simple"

    @property
    def table_available(self) -> bool:
        return self.available and _PYTORCH3D_IMPORT_ERROR is None

    def delete(self) -> None:
        self._grid_cache.clear()
        self._plane_layers.clear()
        self._table_mesh = None
        self._raster_settings_cache.clear()

    def update_scene_geometry(
        self,
        positioned_table_mesh: trimesh.Trimesh,
        floor_mesh: trimesh.Trimesh,
        wall_meshes: dict[str, trimesh.Trimesh],
    ) -> None:
        if not self.available:
            return

        self._plane_layers = {
            "balanced_far_front_back_walls": [
                self._plane_from_mesh("front", wall_meshes["front"]),
                self._plane_from_mesh("back", wall_meshes["back"]),
            ],
            "balanced_near_floor": [
                self._plane_from_mesh("floor", floor_mesh),
            ],
            "balanced_left_wall": [
                self._plane_from_mesh("left", wall_meshes["left"]),
            ],
            "balanced_right_wall": [
                self._plane_from_mesh("right", wall_meshes["right"]),
            ],
        }
        self._table_mesh = (
            self._mesh_from_trimesh(positioned_table_mesh)
            if self.table_available
            else None
        )

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
        if not self.table_available:
            if _PYTORCH3D_IMPORT_ERROR is not None:
                raise RuntimeError(
                    "SimpleLabGpuRenderer table path requires PyTorch3D."
                ) from _PYTORCH3D_IMPORT_ERROR
            raise RuntimeError("SimpleLabGpuRenderer is not available on this device.")
        if self._table_mesh is None:
            raise RuntimeError("GPU table mesh has not been initialized.")
        return self._render_mesh(
            self._table_mesh,
            camera_pose_world=np.asarray(camera_pose_world, dtype=np.float32),
            intrinsic=np.asarray(intrinsic, dtype=np.float32),
            width=int(width),
            height=int(height),
            clear_rgba=self._table_clear_rgba,
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
        if name == "floor":
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
        material = getattr(
            getattr(mesh.visual, "material", None),
            "baseColorTexture",
            None,
        )
        if material is None:
            material = getattr(getattr(mesh.visual, "material", None), "image", None)
        if material is None:
            material = getattr(mesh.visual, "image", None)
        if material is None:
            raise RuntimeError("Expected textured mesh for GPU simple-lab rendering.")
        if not isinstance(material, Image.Image):
            material = Image.open(Path(material))
        texture_np = np.asarray(material.convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(texture_np).to(
            self.device,
            dtype=torch.float32,
        ).unsqueeze(0).contiguous()
