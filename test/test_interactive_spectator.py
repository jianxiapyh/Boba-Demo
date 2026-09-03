from __future__ import annotations

from pathlib import Path

import numpy as np

from qqtt.ambulance_scene import make_ambulance_layout
from qqtt.immersive_scene import make_simple_lab_layout
from qqtt.interactive_spectator import (
    ILLIXR_HEADSET_WORLD_SCALE,
    InteractiveSpectatorRenderer,
    build_ambulance_spectator_camera_pose,
    build_fixed_spectator_camera_pose,
    resolve_center_eye_pose,
    spectator_intrinsic,
)
from qqtt.native_gl_scene_renderer import NativeGlSceneRenderer


REPO_ROOT = Path(__file__).resolve().parents[1]
SPECTATOR_TEST_WIDTH = 960
SPECTATOR_TEST_HEIGHT = 540


def _pose(position, rotation=None):
    result = np.eye(4, dtype=np.float32)
    result[:3, 3] = np.asarray(position, dtype=np.float32)
    if rotation is not None:
        result[:3, :3] = np.asarray(rotation, dtype=np.float32)
    return result


def test_center_eye_pose_averages_eye_positions_and_preserves_orientation():
    left = _pose([-0.032, 0.0, 1.6])
    right = _pose([0.032, 0.0, 1.6])

    center = resolve_center_eye_pose(left, right)

    assert center is not None
    np.testing.assert_allclose(center[:3, 3], [0.0, 0.0, 1.6], atol=1.0e-6)
    np.testing.assert_allclose(center[:3, :3], np.eye(3), atol=1.0e-6)


def test_center_eye_pose_preserves_reflected_runtime_basis():
    reflected = np.diag([1.0, 1.0, -1.0]).astype(np.float32)
    center = resolve_center_eye_pose(
        _pose([-0.03, 0.0, 0.0], reflected),
        _pose([0.03, 0.0, 0.0], reflected),
    )

    assert center is not None
    assert np.linalg.det(center[:3, :3]) < 0.0


def test_spectator_camera_is_independent_and_targets_the_room():
    headset = _pose([0.0, 0.0, 0.0])
    camera = build_fixed_spectator_camera_pose(
        headset,
        table_center_world=np.array([0.0, 0.78, 0.62], dtype=np.float32),
        scene_up_world=np.array([0.0, 0.0, -1.0], dtype=np.float32),
    )

    assert np.all(np.isfinite(camera))
    assert np.linalg.norm(camera[:3, 3] - headset[:3, 3]) > 1.0
    np.testing.assert_allclose(
        camera[:3, :3].T @ camera[:3, :3],
        np.eye(3),
        atol=1.0e-5,
    )


def _project_world_point(camera_pose, intrinsic, world_point):
    cv_from_gl = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    pose_world_cv = np.eye(4, dtype=np.float32)
    pose_world_cv[:3, :3] = camera_pose[:3, :3] @ cv_from_gl
    pose_world_cv[:3, 3] = camera_pose[:3, 3]
    camera_point = np.linalg.inv(pose_world_cv) @ np.append(world_point, 1.0)
    pixel_h = intrinsic @ camera_point[:3]
    return pixel_h[:2] / pixel_h[2], float(camera_point[2])


def test_lab_spectator_frames_the_actual_standing_layout():
    head_position = np.zeros(3, dtype=np.float32)
    scene_up = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    scene_forward = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    scene_right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    layout = make_simple_lab_layout(
        head_position,
        scene_forward,
        scene_up=scene_up,
        start_posture="standing",
    )
    headset = _pose(
        head_position,
        np.column_stack([scene_right, scene_up, -scene_forward]),
    )
    camera = build_fixed_spectator_camera_pose(
        headset,
        layout.table_top_center,
        layout.scene_up,
    )
    intrinsic = spectator_intrinsic(SPECTATOR_TEST_WIDTH, SPECTATOR_TEST_HEIGHT)

    for world_point in (head_position, layout.table_top_center):
        pixel, depth = _project_world_point(camera, intrinsic, world_point)
        assert depth > 0.5
        assert 24.0 < pixel[0] < SPECTATOR_TEST_WIDTH - 24.0
        assert 24.0 < pixel[1] < SPECTATOR_TEST_HEIGHT - 24.0


def test_ambulance_spectator_camera_stays_inside_and_frames_headset():
    scene_up = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    scene_forward = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    scene_right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    headset = _pose(
        [0.0, 0.0, 0.0],
        np.column_stack([scene_right, scene_up, -scene_forward]),
    )
    table_center = np.array([0.1148, 0.6372, 0.6584], dtype=np.float32)
    camera = build_ambulance_spectator_camera_pose(
        headset,
        table_center,
        scene_up,
        scene_forward,
        scene_right,
    )

    np.testing.assert_allclose(camera[:3, 3], [-1.80, 0.80, -0.10], atol=1.0e-6)
    np.testing.assert_allclose(
        camera[:3, :3].T @ camera[:3, :3],
        np.eye(3),
        atol=1.0e-5,
    )

    # In the Ambulance canonical frame the floor covers x=+-2.4, y=+-1.175,
    # and the 1.8 m cabin runs from z=0.436 down to z=-1.364. The spectator is
    # comfortably within all three limits rather than behind the rear wall.
    canonical_head = np.array([-0.013458, -0.737228, -0.763777], dtype=np.float32)
    canonical_camera = canonical_head + camera[:3, 3]
    assert -2.4 < canonical_camera[0] < 2.4
    assert -1.175 < canonical_camera[1] < 1.175
    assert -1.364 < canonical_camera[2] < 0.436

    intrinsic = spectator_intrinsic(SPECTATOR_TEST_WIDTH, SPECTATOR_TEST_HEIGHT)
    headset_pixel, headset_depth = _project_world_point(
        camera,
        intrinsic,
        headset[:3, 3],
    )
    table_pixel, table_depth = _project_world_point(
        camera,
        intrinsic,
        table_center,
    )
    for pixel, depth in (
        (headset_pixel, headset_depth),
        (table_pixel, table_depth),
    ):
        assert depth > 0.5
        assert 24.0 < pixel[0] < SPECTATOR_TEST_WIDTH - 24.0
        assert 24.0 < pixel[1] < SPECTATOR_TEST_HEIGHT - 24.0


def test_ambulance_spectator_frames_the_actual_standing_layout():
    head_position = np.zeros(3, dtype=np.float32)
    scene_up = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    scene_forward = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    layout = make_ambulance_layout(
        head_position,
        scene_forward,
        scene_up=scene_up,
        repo_root=REPO_ROOT,
        start_posture="standing",
    )
    headset = _pose(
        head_position,
        np.column_stack(
            [layout.scene_right, layout.scene_up, -layout.scene_forward]
        ),
    )
    camera = build_ambulance_spectator_camera_pose(
        headset,
        layout.table_top_center,
        layout.scene_up,
        layout.scene_forward,
        layout.scene_right,
    )

    canonical_camera = (
        np.asarray(layout.ambulance_standing_head_canonical)
        + camera[:3, 3]
    )
    assert -2.4 < canonical_camera[0] < 2.4
    assert -1.175 < canonical_camera[1] < 1.175
    assert -1.364 < canonical_camera[2] < 0.436

    intrinsic = spectator_intrinsic(SPECTATOR_TEST_WIDTH, SPECTATOR_TEST_HEIGHT)
    for world_point in (head_position, layout.table_top_center):
        pixel, depth = _project_world_point(camera, intrinsic, world_point)
        assert depth > 0.5
        assert 24.0 < pixel[0] < SPECTATOR_TEST_WIDTH - 24.0
        assert 24.0 < pixel[1] < SPECTATOR_TEST_HEIGHT - 24.0


def test_spectator_renderer_selects_and_keeps_ambulance_interior_camera():
    class _LayoutRenderer:
        def __init__(self):
            self.layouts = []

        def set_layout(self, layout):
            self.layouts.append(layout)

    spectator = InteractiveSpectatorRenderer.__new__(InteractiveSpectatorRenderer)
    spectator._deleted = False
    spectator._renderer = _LayoutRenderer()
    spectator._layout_signature = None
    spectator.camera_pose_world = None
    spectator.camera_profile = "unconfigured"
    layout = type(
        "AmbulanceLayout",
        (),
        {
            "scene_name": "ambulance",
            "table_top_center": np.array([0.1148, 0.6372, 0.6584], dtype=np.float32),
            "scene_up": np.array([0.0, 0.0, -1.0], dtype=np.float32),
            "scene_forward": np.array([0.0, 1.0, 0.0], dtype=np.float32),
            "scene_right": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        },
    )()
    headset = _pose(
        [0.0, 0.0, 0.0],
        np.column_stack(
            [layout.scene_right, layout.scene_up, -layout.scene_forward]
        ),
    )

    spectator.configure_layout(layout, headset, headset)
    fixed_camera = spectator.camera_pose_world.copy()
    assert spectator.camera_profile == "ambulance_interior"
    assert len(spectator._renderer.layouts) == 1

    moved_headset = headset.copy()
    moved_headset[:3, 3] += np.array([0.2, 0.1, 0.0], dtype=np.float32)
    spectator.configure_layout(layout, moved_headset, moved_headset)

    # Head movement must not drag the desktop camera; the tracked headset moves
    # inside the fixed spectator view instead.
    np.testing.assert_allclose(spectator.camera_pose_world, fixed_camera)
    assert len(spectator._renderer.layouts) == 1


def test_ambulance_interior_camera_keeps_complete_standing_headset_mesh_in_frame():
    head_position = np.zeros(3, dtype=np.float32)
    scene_up = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    scene_forward = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    layout = make_ambulance_layout(
        head_position,
        scene_forward,
        scene_up=scene_up,
        repo_root=REPO_ROOT,
        start_posture="standing",
    )
    headset = _pose(
        head_position,
        np.column_stack(
            [layout.scene_right, layout.scene_up, -layout.scene_forward]
        ),
    )
    camera = build_ambulance_spectator_camera_pose(
        headset,
        layout.table_top_center,
        layout.scene_up,
        layout.scene_forward,
        layout.scene_right,
    )
    parser = NativeGlSceneRenderer.__new__(NativeGlSceneRenderer)
    groups = parser._parse_obj_groups(
        REPO_ROOT / "assets" / "scenes" / "ILLIXR_lab" / "headset.obj"
    )
    authored_vertices = np.concatenate(
        [group.vertices[:, :3] for group in groups],
        axis=0,
    )
    world_vertices = (
        authored_vertices
        @ (headset[:3, :3] * ILLIXR_HEADSET_WORLD_SCALE).T
        + headset[:3, 3]
    )
    cv_from_gl = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    pose_world_cv = np.eye(4, dtype=np.float32)
    pose_world_cv[:3, :3] = camera[:3, :3] @ cv_from_gl
    pose_world_cv[:3, 3] = camera[:3, 3]
    world_to_camera = np.linalg.inv(pose_world_cv)
    camera_vertices = (
        np.concatenate(
            [world_vertices, np.ones((world_vertices.shape[0], 1))],
            axis=1,
        )
        @ world_to_camera.T
    )[:, :3]
    intrinsic = spectator_intrinsic(SPECTATOR_TEST_WIDTH, SPECTATOR_TEST_HEIGHT)
    pixel_h = camera_vertices @ intrinsic.T
    pixels = pixel_h[:, :2] / pixel_h[:, 2:3]
    pixel_min = pixels.min(axis=0)
    pixel_max = pixels.max(axis=0)

    assert np.all(camera_vertices[:, 2] > 0.0)
    assert pixel_min[0] > 24.0
    assert pixel_max[0] < SPECTATOR_TEST_WIDTH - 24.0
    assert pixel_min[1] > 24.0
    assert pixel_max[1] < SPECTATOR_TEST_HEIGHT - 24.0
    assert pixel_max[0] - pixel_min[0] > 40.0
    assert pixel_max[1] - pixel_min[1] > 25.0


def test_spectator_intrinsic_uses_square_pixels_for_wide_window():
    intrinsic = spectator_intrinsic(SPECTATOR_TEST_WIDTH, SPECTATOR_TEST_HEIGHT)

    assert intrinsic[0, 0] == intrinsic[1, 1]
    assert intrinsic[0, 2] == SPECTATOR_TEST_WIDTH / 2.0
    assert intrinsic[1, 2] == SPECTATOR_TEST_HEIGHT / 2.0


def test_native_gl_obj_parser_loads_the_complete_illixr_headset():
    parser = NativeGlSceneRenderer.__new__(NativeGlSceneRenderer)
    groups = parser._parse_obj_groups(
        REPO_ROOT / "assets" / "scenes" / "ILLIXR_lab" / "headset.obj"
    )

    assert {group.material_name for group in groups} == {"Headset", "Logo"}
    assert sum(group.triangle_count for group in groups) == 739
    assert sum(group.indices.size for group in groups) == 739 * 3
