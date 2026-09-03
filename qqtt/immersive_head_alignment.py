"""Small, testable helpers for registering live headset orientation."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np


def _normalize_vector(value: np.ndarray, *, label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32).reshape(3)
    norm = float(np.linalg.norm(vector))
    if not np.all(np.isfinite(vector)) or norm <= 1.0e-6:
        raise ValueError(f"{label} must be a finite nonzero vector.")
    return (vector / norm).astype(np.float32)


def _closest_proper_rotation(value: np.ndarray, *, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label} must be a finite 3x3 matrix.")
    left, _, right_t = np.linalg.svd(matrix)
    if float(np.linalg.det(left @ right_t)) < 0.0:
        left[:, -1] *= -1.0
    return (left @ right_t).astype(np.float32)


def average_eye_rotation(rotations: Iterable[np.ndarray]) -> np.ndarray:
    """Return the closest proper rotation to one or more eye rotations."""

    matrices = [np.asarray(value, dtype=np.float32) for value in rotations]
    if not matrices:
        raise ValueError("At least one valid startup eye rotation is required.")
    for matrix in matrices:
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise ValueError("Startup eye rotations must be finite 3x3 matrices.")
    return _closest_proper_rotation(
        np.mean(np.stack(matrices, axis=0), axis=0),
        label="averaged startup eye rotation",
    )


def build_startup_eye_orientation_registration(
    tracking_to_scene_rotation: np.ndarray,
    startup_eye_rotations: Iterable[np.ndarray],
    *,
    scene_up: np.ndarray,
    scene_forward: np.ndarray,
    pitch_down_degrees: float,
) -> dict[str, np.ndarray | float]:
    """Build a camera-local offset that authors the initial view direction.

    Translation and controller directions continue to use the gravity/yaw
    aligned ``tracking_to_scene_rotation``.  The returned local offset is
    applied on the right of each live eye rotation, so subsequent physical
    rotations still occur around the correctly mapped tracking-space axes.
    """

    tracking_to_scene = _closest_proper_rotation(
        tracking_to_scene_rotation,
        label="tracking-to-scene rotation",
    )
    pitch_down_degrees = float(pitch_down_degrees)
    if not math.isfinite(pitch_down_degrees) or not 0.0 <= pitch_down_degrees <= 60.0:
        raise ValueError("Startup gaze pitch must be between 0 and 60 degrees down.")

    up = _normalize_vector(scene_up, label="scene up")
    forward_input = _normalize_vector(scene_forward, label="scene forward")
    forward = forward_input - float(np.dot(forward_input, up)) * up
    forward = _normalize_vector(forward, label="horizontal scene forward")
    back = -forward
    right = _normalize_vector(np.cross(up, back), label="scene right")
    back = _normalize_vector(np.cross(right, up), label="scene back")
    forward = -back

    pitch_radians = math.radians(pitch_down_degrees)
    pitch_cos = math.cos(pitch_radians)
    pitch_sin = math.sin(pitch_radians)
    desired_forward = (
        pitch_cos * forward - pitch_sin * up
    ).astype(np.float32)
    desired_up = (pitch_sin * forward + pitch_cos * up).astype(np.float32)
    desired_rotation = np.stack(
        [right, desired_up, -desired_forward],
        axis=1,
    ).astype(np.float32)
    desired_rotation = _closest_proper_rotation(
        desired_rotation,
        label="desired startup eye rotation",
    )

    startup_eye_rotation = average_eye_rotation(startup_eye_rotations)
    local_offset = _closest_proper_rotation(
        startup_eye_rotation.T @ tracking_to_scene.T @ desired_rotation,
        label="startup eye local orientation offset",
    )
    achieved_rotation = (
        tracking_to_scene @ startup_eye_rotation @ local_offset
    ).astype(np.float32)

    return {
        "eye_local_orientation_offset": local_offset,
        "startup_eye_rotation_local": startup_eye_rotation,
        "desired_startup_eye_rotation_world": desired_rotation,
        "achieved_startup_eye_rotation_world": achieved_rotation,
        "desired_startup_forward_world": desired_forward,
        "pitch_down_degrees": pitch_down_degrees,
    }
