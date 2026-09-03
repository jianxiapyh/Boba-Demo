from __future__ import annotations

import math

import numpy as np

from qqtt.immersive_head_alignment import (
    average_eye_rotation,
    build_startup_eye_orientation_registration,
)


def _axis_angle_rotation(axis, degrees):
    axis = np.asarray(axis, dtype=np.float32)
    axis /= np.linalg.norm(axis)
    radians = math.radians(float(degrees))
    cross = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=np.float32,
    )
    return (
        math.cos(radians) * np.eye(3, dtype=np.float32)
        + math.sin(radians) * cross
        + (1.0 - math.cos(radians)) * np.outer(axis, axis)
    ).astype(np.float32)


def test_startup_eye_registration_authors_pitch_without_changing_tracking_basis():
    scene_up = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    scene_forward = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    tracking_to_scene = _axis_angle_rotation([0.0, 1.0, 0.0], 23.0)
    startup_eye = (
        _axis_angle_rotation([0.0, 1.0, 0.0], -31.0)
        @ _axis_angle_rotation([1.0, 0.0, 0.0], 12.0)
        @ _axis_angle_rotation([0.0, 0.0, 1.0], -7.0)
    )
    registration = build_startup_eye_orientation_registration(
        tracking_to_scene,
        [startup_eye, startup_eye],
        scene_up=scene_up,
        scene_forward=scene_forward,
        pitch_down_degrees=30.0,
    )

    local_offset = registration["eye_local_orientation_offset"]
    achieved = tracking_to_scene @ startup_eye @ local_offset
    desired = registration["desired_startup_eye_rotation_world"]
    desired_forward = registration["desired_startup_forward_world"]

    assert np.allclose(achieved, desired, atol=2.0e-6)
    assert np.allclose(achieved.T @ achieved, np.eye(3), atol=2.0e-6)
    assert np.isclose(np.linalg.det(achieved), 1.0, atol=2.0e-6)
    assert np.allclose(-achieved[:, 2], desired_forward, atol=2.0e-6)
    assert np.isclose(np.dot(desired_forward, scene_forward), math.cos(math.radians(30.0)))
    assert np.isclose(np.dot(desired_forward, -scene_up), math.sin(math.radians(30.0)))
    # The helper returns an eye-local offset; the gravity/yaw tracking basis
    # used for headset positions and controllers remains untouched.
    assert np.allclose(
        tracking_to_scene,
        _axis_angle_rotation([0.0, 1.0, 0.0], 23.0),
        atol=1.0e-7,
    )


def test_eye_local_offset_preserves_later_global_tracking_rotations():
    tracking_to_scene = _axis_angle_rotation([0.0, 1.0, 0.0], -17.0)
    startup_eye = (
        _axis_angle_rotation([0.0, 1.0, 0.0], 26.0)
        @ _axis_angle_rotation([1.0, 0.0, 0.0], -9.0)
    )
    registration = build_startup_eye_orientation_registration(
        tracking_to_scene,
        [startup_eye],
        scene_up=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        scene_forward=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        pitch_down_degrees=30.0,
    )
    local_offset = registration["eye_local_orientation_offset"]
    initial_world = tracking_to_scene @ startup_eye @ local_offset

    live_global_delta = _axis_angle_rotation([0.0, 1.0, 0.0], 14.0)
    moved_eye = live_global_delta @ startup_eye
    moved_world = tracking_to_scene @ moved_eye @ local_offset
    expected_world_delta = (
        tracking_to_scene @ live_global_delta @ tracking_to_scene.T
    )

    assert np.allclose(
        moved_world,
        expected_world_delta @ initial_world,
        atol=2.0e-6,
    )


def test_zero_pitch_registration_levels_a_tilted_startup_headset_to_the_floor():
    scene_up = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    scene_forward = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    tracking_to_scene = np.eye(3, dtype=np.float32)
    startup_eye = (
        _axis_angle_rotation([1.0, 0.0, 0.0], 38.0)
        @ _axis_angle_rotation([0.0, 0.0, 1.0], -21.0)
    )
    registration = build_startup_eye_orientation_registration(
        tracking_to_scene,
        [startup_eye],
        scene_up=scene_up,
        scene_forward=scene_forward,
        pitch_down_degrees=0.0,
    )
    achieved = (
        tracking_to_scene
        @ startup_eye
        @ registration["eye_local_orientation_offset"]
    )
    achieved_forward = -achieved[:, 2]
    achieved_up = achieved[:, 1]

    assert np.allclose(achieved_forward, scene_forward, atol=2.0e-6)
    assert np.allclose(achieved_up, scene_up, atol=2.0e-6)
    assert np.isclose(np.dot(achieved_forward, scene_up), 0.0, atol=2.0e-6)


def test_average_eye_rotation_returns_a_proper_rotation():
    left = _axis_angle_rotation([0.0, 1.0, 0.0], -0.2)
    right = _axis_angle_rotation([0.0, 1.0, 0.0], 0.2)
    averaged = average_eye_rotation([left, right])

    assert np.allclose(averaged, np.eye(3), atol=2.0e-6)
    assert np.isclose(np.linalg.det(averaged), 1.0, atol=2.0e-6)
