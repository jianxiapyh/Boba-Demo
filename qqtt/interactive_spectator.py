from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from qqtt.native_gl_scene_renderer import NativeGlSceneRenderer


ILLIXR_HEADSET_OBJ = "headset.obj"
ILLIXR_HEADSET_WORLD_SCALE = 0.50
SPECTATOR_VERTICAL_FOV_DEGREES = 55.0
# Keep the camera on the clear aisle-side sightline, then move it far enough
# toward the mattress to push the foreground cargo net toward the right edge.
AMBULANCE_SPECTATOR_RIGHT_OFFSET_M = -1.80
AMBULANCE_SPECTATOR_FORWARD_OFFSET_M = 0.80
AMBULANCE_SPECTATOR_UP_OFFSET_M = 0.10
AMBULANCE_SPECTATOR_TABLE_TARGET_WEIGHT = 0.55
AMBULANCE_SPECTATOR_TARGET_UP_OFFSET_M = 0.03


def _finite_pose(pose) -> np.ndarray | None:
    if pose is None:
        return None
    pose_np = np.asarray(pose, dtype=np.float32)
    if pose_np.shape != (4, 4) or not np.all(np.isfinite(pose_np)):
        return None
    return pose_np


def _normalized(vector, fallback) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32).reshape(3)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm <= 1.0e-6:
        value = np.asarray(fallback, dtype=np.float32).reshape(3)
        norm = float(np.linalg.norm(value))
    return (value / max(norm, 1.0e-6)).astype(np.float32)


def resolve_center_eye_pose(left_eye_pose_world, right_eye_pose_world) -> np.ndarray | None:
    """Recover the tracked HMD pose represented by the two eye poses."""

    poses = [
        pose
        for pose in (
            _finite_pose(left_eye_pose_world),
            _finite_pose(right_eye_pose_world),
        )
        if pose is not None
    ]
    if not poses:
        return None

    rotation_sum = np.sum(
        np.stack([pose[:3, :3] for pose in poses], axis=0),
        axis=0,
    )
    u, _, vt = np.linalg.svd(rotation_sum)
    rotation = u @ vt
    desired_determinant_sign = -1.0 if np.linalg.det(poses[0][:3, :3]) < 0.0 else 1.0
    if float(np.linalg.det(rotation)) * desired_determinant_sign < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt

    center_pose = np.eye(4, dtype=np.float32)
    center_pose[:3, :3] = rotation.astype(np.float32)
    center_pose[:3, 3] = np.mean(
        np.stack([pose[:3, 3] for pose in poses], axis=0),
        axis=0,
    ).astype(np.float32)
    return center_pose


def build_fixed_spectator_camera_pose(
    headset_pose_world: np.ndarray,
    table_center_world: np.ndarray,
    scene_up_world: np.ndarray,
) -> np.ndarray:
    """Build a three-quarter room camera anchored at session startup.

    ILLIXR debugview uses an independent orbit camera and then renders the
    tracked headset into that view.  Boba uses a fixed orbit point so physical
    headset translation remains visible relative to the room and table.
    """

    headset_pose = _finite_pose(headset_pose_world)
    if headset_pose is None:
        raise ValueError("A finite headset pose is required for the spectator camera.")
    head_position = headset_pose[:3, 3]
    scene_up = _normalized(scene_up_world, headset_pose[:3, 1])
    head_forward = headset_pose[:3, :3] @ np.array(
        [0.0, 0.0, -1.0],
        dtype=np.float32,
    )
    head_forward = head_forward - float(np.dot(head_forward, scene_up)) * scene_up
    head_forward = _normalized(head_forward, np.array([0.0, 1.0, 0.0]))

    head_right = headset_pose[:3, 0]
    head_right = head_right - float(np.dot(head_right, scene_up)) * scene_up
    head_right = _normalized(
        head_right,
        np.cross(scene_up, head_forward),
    )
    table_center = np.asarray(table_center_world, dtype=np.float32).reshape(3)
    target = (
        0.58 * head_position
        + 0.42 * table_center
        + scene_up * 0.08
    ).astype(np.float32)
    camera_position = (
        head_position
        - head_forward * 0.88
        + head_right * 0.72
        + scene_up * 0.30
    ).astype(np.float32)

    look_forward = _normalized(target - camera_position, head_forward)
    camera_back = -look_forward
    camera_right = _normalized(
        np.cross(scene_up, camera_back),
        head_right,
    )
    camera_up = _normalized(
        np.cross(camera_back, camera_right),
        scene_up,
    )
    rotation = np.stack(
        [camera_right, camera_up, camera_back],
        axis=1,
    ).astype(np.float32)
    desired_determinant_sign = (
        -1.0 if np.linalg.det(headset_pose[:3, :3]) < 0.0 else 1.0
    )
    if float(np.linalg.det(rotation)) * desired_determinant_sign < 0.0:
        rotation[:, 0] *= -1.0

    camera_pose = np.eye(4, dtype=np.float32)
    camera_pose[:3, :3] = rotation
    camera_pose[:3, 3] = camera_position
    return camera_pose


def build_ambulance_spectator_camera_pose(
    headset_pose_world: np.ndarray,
    table_center_world: np.ndarray,
    scene_up_world: np.ndarray,
    scene_forward_world: np.ndarray,
    scene_right_world: np.ndarray,
) -> np.ndarray:
    """Build a fixed interior camera that frames the HMD and stretcher.

    The generic Lab orbit camera sits 88 cm behind the seated headset. In the
    narrow Ambulance capture that crosses the rear wall. Instead, this camera
    moves toward the mattress along the clear aisle-side sightline, then looks
    diagonally across both the headset and playable surface while keeping the
    foreground cargo net near the right edge of the image.
    """

    headset_pose = _finite_pose(headset_pose_world)
    if headset_pose is None:
        raise ValueError("A finite headset pose is required for the spectator camera.")
    head_position = headset_pose[:3, 3]
    scene_up = _normalized(scene_up_world, headset_pose[:3, 1])
    scene_forward = np.asarray(scene_forward_world, dtype=np.float32).reshape(3)
    scene_forward = scene_forward - float(np.dot(scene_forward, scene_up)) * scene_up
    scene_forward = _normalized(scene_forward, np.array([0.0, 1.0, 0.0]))
    scene_right = np.asarray(scene_right_world, dtype=np.float32).reshape(3)
    scene_right = scene_right - float(np.dot(scene_right, scene_up)) * scene_up
    scene_right = _normalized(
        scene_right,
        np.cross(scene_up, scene_forward),
    )
    table_center = np.asarray(table_center_world, dtype=np.float32).reshape(3)

    camera_position = (
        head_position
        + scene_right * AMBULANCE_SPECTATOR_RIGHT_OFFSET_M
        + scene_forward * AMBULANCE_SPECTATOR_FORWARD_OFFSET_M
        + scene_up * AMBULANCE_SPECTATOR_UP_OFFSET_M
    ).astype(np.float32)
    target = (
        (1.0 - AMBULANCE_SPECTATOR_TABLE_TARGET_WEIGHT) * head_position
        + AMBULANCE_SPECTATOR_TABLE_TARGET_WEIGHT * table_center
        + scene_up * AMBULANCE_SPECTATOR_TARGET_UP_OFFSET_M
    ).astype(np.float32)

    look_forward = _normalized(target - camera_position, scene_forward)
    camera_back = -look_forward
    camera_right = _normalized(
        np.cross(scene_up, camera_back),
        scene_right,
    )
    camera_up = _normalized(
        np.cross(camera_back, camera_right),
        scene_up,
    )
    rotation = np.stack(
        [camera_right, camera_up, camera_back],
        axis=1,
    ).astype(np.float32)
    desired_determinant_sign = (
        -1.0 if np.linalg.det(headset_pose[:3, :3]) < 0.0 else 1.0
    )
    if float(np.linalg.det(rotation)) * desired_determinant_sign < 0.0:
        rotation[:, 0] *= -1.0

    camera_pose = np.eye(4, dtype=np.float32)
    camera_pose[:3, :3] = rotation
    camera_pose[:3, 3] = camera_position
    return camera_pose


def spectator_intrinsic(width: int, height: int) -> np.ndarray:
    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0:
        raise ValueError("Spectator dimensions must be positive.")
    focal = 0.5 * float(height) / math.tan(
        0.5 * math.radians(SPECTATOR_VERTICAL_FOV_DEGREES)
    )
    return np.array(
        [
            [focal, 0.0, 0.5 * float(width)],
            [0.0, focal, 0.5 * float(height)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


class InteractiveSpectatorRenderer:
    """Render the ILLIXR room and tracked headset for the desktop window."""

    def __init__(
        self,
        scene_assets_root: str | Path,
        width: int,
        height: int,
        *,
        device,
        texture_mode: str,
        anisotropy: int,
        mipmap_lod_bias: float,
        msaa_samples: int,
        depth_format: str,
    ) -> None:
        self.scene_assets_root = Path(scene_assets_root).resolve()
        self.width = int(width)
        self.height = int(height)
        self.device = torch.device(device)
        self.texture_mode = str(texture_mode)
        self.anisotropy = int(anisotropy)
        self.mipmap_lod_bias = float(mipmap_lod_bias)
        self.msaa_samples = int(msaa_samples)
        self.depth_format = str(depth_format)
        self.camera_pose_world: np.ndarray | None = None
        self.camera_profile = "unconfigured"
        self.intrinsic = spectator_intrinsic(self.width, self.height)
        self._layout_signature = None
        self._deleted = False
        self._renderer = NativeGlSceneRenderer(
            scene_assets_root=self.scene_assets_root,
            width=self.width,
            height=self.height,
            render_background=False,
            texture_mode=self.texture_mode,
            anisotropy=self.anisotropy,
            mipmap_lod_bias=self.mipmap_lod_bias,
            msaa_samples=self.msaa_samples,
            depth_format=self.depth_format,
            device=self.device,
            output_ring_size=3,
            tracked_mesh_obj=ILLIXR_HEADSET_OBJ,
        )

    @property
    def deleted(self) -> bool:
        return bool(self._deleted)

    def is_compatible(
        self,
        scene_assets_root: str | Path,
        width: int,
        height: int,
        *,
        device,
        texture_mode: str,
        anisotropy: int,
        mipmap_lod_bias: float,
        msaa_samples: int,
        depth_format: str,
    ) -> bool:
        return bool(
            not self._deleted
            and self.scene_assets_root == Path(scene_assets_root).resolve()
            and self.width == int(width)
            and self.height == int(height)
            and str(self.device) == str(torch.device(device))
            and self.texture_mode == str(texture_mode)
            and self.anisotropy == int(anisotropy)
            and abs(self.mipmap_lod_bias - float(mipmap_lod_bias)) <= 1.0e-9
            and self.msaa_samples == int(msaa_samples)
            and self.depth_format == str(depth_format)
        )

    def configure_layout(
        self,
        layout,
        left_eye_pose_world,
        right_eye_pose_world,
    ) -> None:
        if self._deleted:
            raise RuntimeError("Interactive spectator renderer has been deleted.")
        headset_pose = resolve_center_eye_pose(
            left_eye_pose_world,
            right_eye_pose_world,
        )
        if headset_pose is None:
            raise ValueError("Cannot configure spectator camera without a headset pose.")
        table_center = np.asarray(layout.table_top_center, dtype=np.float32).reshape(3)
        scene_up = np.asarray(layout.scene_up, dtype=np.float32).reshape(3)
        scene_name = str(getattr(layout, "scene_name", "lab") or "lab").strip().lower()
        signature_values = [table_center, scene_up]
        scene_forward = None
        scene_right = None
        if scene_name == "ambulance":
            scene_forward = np.asarray(
                getattr(
                    layout,
                    "scene_forward",
                    headset_pose[:3, :3]
                    @ np.array([0.0, 0.0, -1.0], dtype=np.float32),
                ),
                dtype=np.float32,
            ).reshape(3)
            scene_right = np.asarray(
                getattr(layout, "scene_right", headset_pose[:3, 0]),
                dtype=np.float32,
            ).reshape(3)
            signature_values.extend([scene_forward, scene_right])
        layout_signature = (
            scene_name,
            *np.round(
                np.concatenate(signature_values),
                decimals=5,
            ).tolist(),
        )
        layout_changed = self._layout_signature != layout_signature
        if layout_changed:
            self._renderer.set_layout(layout)
            self._layout_signature = layout_signature
        if layout_changed or self.camera_pose_world is None:
            if scene_name == "ambulance":
                self.camera_pose_world = build_ambulance_spectator_camera_pose(
                    headset_pose,
                    table_center,
                    scene_up,
                    scene_forward,
                    scene_right,
                )
                self.camera_profile = "ambulance_interior"
            else:
                self.camera_pose_world = build_fixed_spectator_camera_pose(
                    headset_pose,
                    table_center,
                    scene_up,
                )
                self.camera_profile = "lab_orbit"

    def render_scene_with_headset(
        self,
        headset_pose_world: np.ndarray,
        *,
        draw_scene: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._deleted or self.camera_pose_world is None:
            raise RuntimeError("Interactive spectator renderer is not configured.")
        headset_model_pose = np.asarray(
            headset_pose_world,
            dtype=np.float32,
        ).copy()
        # The debug asset spans roughly 0.5 m in its authored coordinates.
        # Scale it to a physical headset footprint in Boba's meter-based room.
        headset_model_pose[:3, :3] *= float(ILLIXR_HEADSET_WORLD_SCALE)
        return self._renderer.render_eye(
            self.camera_pose_world,
            self.intrinsic,
            width=self.width,
            height=self.height,
            target_name="center",
            tracked_mesh_pose_world=headset_model_pose,
            draw_scene=bool(draw_scene),
        )

    def delete(self) -> None:
        if self._deleted:
            return
        self._renderer.delete()
        self.camera_pose_world = None
        self.camera_profile = "deleted"
        self._deleted = True


def reuse_or_create_interactive_spectator_renderer(
    existing_renderer,
    scene_assets_root: str | Path,
    width: int,
    height: int,
    *,
    device,
    texture_mode: str,
    anisotropy: int,
    mipmap_lod_bias: float,
    msaa_samples: int,
    depth_format: str,
) -> InteractiveSpectatorRenderer:
    compatibility_kwargs = {
        "device": device,
        "texture_mode": texture_mode,
        "anisotropy": anisotropy,
        "mipmap_lod_bias": mipmap_lod_bias,
        "msaa_samples": msaa_samples,
        "depth_format": depth_format,
    }
    if existing_renderer is not None and existing_renderer.is_compatible(
        scene_assets_root,
        width,
        height,
        **compatibility_kwargs,
    ):
        return existing_renderer
    if existing_renderer is not None:
        existing_renderer.delete()
    return InteractiveSpectatorRenderer(
        scene_assets_root,
        width,
        height,
        **compatibility_kwargs,
    )
