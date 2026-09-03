from __future__ import annotations

import numpy as np

from qqtt.immersive_scene import (
    make_simple_lab_layout,
    normalize_immersive_start_posture,
)


def test_lab_standing_layout_preserves_view_and_lowers_room_to_floor_height():
    head_position = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    forward = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    scene_up = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    seated = make_simple_lab_layout(
        head_position,
        forward,
        scene_up=scene_up,
        start_posture="seated",
    )
    standing = make_simple_lab_layout(
        head_position,
        forward,
        scene_up=scene_up,
        start_posture="standing",
    )

    assert standing.start_posture == "standing"
    assert np.isclose(standing.startup_head_height_above_floor_m, 1.55)
    assert np.isclose(standing.floor_z - head_position[2], 1.55)
    assert np.allclose(
        standing.table_top_center - head_position,
        [0.0, 0.78, 0.82],
    )
    assert np.allclose(
        standing.table_top_center - seated.table_top_center,
        [0.0, 0.0, 0.20],
    )
    assert np.isclose(standing.floor_z - seated.floor_z, 0.20)


def test_start_posture_is_explicit_and_has_no_auto_mode():
    assert normalize_immersive_start_posture("standing") == "standing"
    assert normalize_immersive_start_posture("seated") == "seated"
    try:
        normalize_immersive_start_posture("auto")
    except ValueError:
        pass
    else:
        raise AssertionError("auto posture must remain disabled")
