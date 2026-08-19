from __future__ import annotations

import numpy as np

from qqtt.object_selector import (
    RuntimeObjectSelector,
    object_choices_for_scene,
    selector_lines,
    selector_row_from_ray,
)
from qqtt.live_openxr import parse_controller_payload


def _buttons(**overrides):
    state = {
        "left": {
            "menu": False,
            "navigate": False,
            "select": False,
            "vertical": 0.0,
        },
        "right": {
            "menu": False,
            "navigate": False,
            "select": False,
            "vertical": 0.0,
        },
    }
    for key, value in overrides.items():
        source, field = key.split("_", 1)
        state[source][field] = value
    return state


def _controller_payload(**overrides):
    payload = {
        "source": "aim",
        "active": True,
        "position_valid": True,
        "orientation_valid": True,
        "position_tracked": True,
        "orientation_tracked": True,
        "position": [0.0, 0.0, 0.0],
        "orientation": [0.0, 0.0, 0.0, 1.0],
        "select_available": True,
        "select_pressed": False,
        "select_value": 0.0,
        "select_source": "value",
        "anchor_cycle_available": True,
        "anchor_cycle_pressed": False,
        "anchor_cycle_source": "click",
        "snap_assist_available": True,
        "snap_assist_pressed": False,
        "snap_assist_source": "click",
    }
    payload.update(overrides)
    return payload


def test_openxr_thumbstick_payload_and_legacy_defaults_are_parsed():
    sample = parse_controller_payload(
        _controller_payload(
            thumbstick_available=True,
            thumbstick_x=0.25,
            thumbstick_y=-0.8,
        )
    )
    assert sample.thumbstick_available is True
    assert sample.thumbstick_x == 0.25
    assert sample.thumbstick_y == -0.8

    legacy_sample = parse_controller_payload(_controller_payload())
    assert legacy_sample.thumbstick_available is False
    assert legacy_sample.thumbstick_x == 0.0
    assert legacy_sample.thumbstick_y == 0.0


def test_short_menu_tap_requests_reset_without_opening():
    selector = RuntimeObjectSelector("rope_game")
    selector.update(0.0, _buttons())
    selector.update(0.1, _buttons(left_menu=True))
    events = selector.update(0.4, _buttons())

    assert events["reset_requested"] is True
    assert events["opened"] is False
    assert selector.mode == "closed"


def test_menu_hold_opens_without_reset():
    selector = RuntimeObjectSelector("sloth")
    selector.update(0.0, _buttons())
    selector.update(0.1, _buttons(right_menu=True))
    events = selector.update(0.86, _buttons(right_menu=True))

    assert events["opened"] is True
    assert events["reset_requested"] is False
    assert selector.mode == "open"

    release_events = selector.update(0.9, _buttons())
    assert release_events["cancelled"] is False


def test_x_a_navigation_and_trigger_selection():
    selector = RuntimeObjectSelector("rope_game")
    selector.update(0.0, _buttons())
    selector.update(0.1, _buttons(left_menu=True))
    selector.update(0.9, _buttons(left_menu=True))
    selector.update(1.0, _buttons())

    nav_events = selector.update(1.1, _buttons(right_navigate=True))
    assert nav_events["highlighted_index"] == 1
    selector.update(1.2, _buttons())
    select_events = selector.update(1.3, _buttons(right_select=True))

    assert select_events["selected_case"] == "sloth"
    assert selector.mode == "loading"


def test_each_joystick_navigates_up_and_down_once_per_deflection():
    for source in ("left", "right"):
        selector = RuntimeObjectSelector("rope_game")
        selector.update(0.0, _buttons())
        selector.update(0.1, _buttons(left_menu=True))
        selector.update(0.9, _buttons(left_menu=True))
        selector.update(1.0, _buttons())

        up = selector.update(1.1, _buttons(**{f"{source}_vertical": 0.8}))
        assert up["highlighted_index"] == 1

        held = selector.update(1.2, _buttons(**{f"{source}_vertical": 0.9}))
        assert held["highlighted_index"] == 1

        selector.update(1.3, _buttons(**{f"{source}_vertical": 0.1}))
        down = selector.update(1.4, _buttons(**{f"{source}_vertical": -0.8}))
        assert down["highlighted_index"] == 0


def test_joystick_highlight_is_not_overwritten_by_a_resting_ray():
    selector = RuntimeObjectSelector("rope_game")
    selector.update(0.0, _buttons())
    selector.update(0.1, _buttons(left_menu=True))
    selector.update(0.9, _buttons(left_menu=True))
    selector.update(1.0, _buttons(), hovered_index=0)

    moved = selector.update(
        1.1,
        _buttons(right_vertical=-0.8),
        hovered_index=0,
    )
    assert moved["highlighted_index"] == 1

    selector.update(1.2, _buttons(), hovered_index=0)
    selected = selector.update(
        1.3,
        _buttons(right_select=True),
        hovered_index=0,
    )
    assert selected["selected_case"] == "sloth"


def test_carried_joystick_requires_recentering_before_navigation():
    selector = RuntimeObjectSelector("rope_game", blocked_until=0.3)
    selector.update(0.4, _buttons(right_vertical=-0.9))
    selector.update(0.5, _buttons())
    selector.update(0.6, _buttons(left_menu=True))
    selector.update(1.4, _buttons(left_menu=True))

    assert selector.mode == "open"
    assert selector.highlighted_index == 0

    selector.update(1.5, _buttons())
    moved = selector.update(1.6, _buttons(right_vertical=-0.9))
    assert moved["highlighted_index"] == 1


def test_y_b_cancels_only_after_opener_is_released():
    selector = RuntimeObjectSelector("rope_game")
    selector.update(0.0, _buttons())
    selector.update(0.1, _buttons(left_menu=True))
    selector.update(0.9, _buttons(left_menu=True))
    assert selector.update(0.95, _buttons(left_menu=True))["cancelled"] is False
    selector.update(1.0, _buttons())

    assert selector.update(1.1, _buttons(right_menu=True))["cancelled"] is True
    assert selector.mode == "closed"


def test_post_switch_debounce_requires_time_and_neutral_input():
    selector = RuntimeObjectSelector("sloth", blocked_until=1.0)
    selector.update(1.1, _buttons(right_menu=True))
    assert selector.update(1.2, _buttons())["reset_requested"] is False

    selector.update(1.3, _buttons())
    selector.update(1.4, _buttons(right_menu=True))
    assert selector.update(1.5, _buttons())["reset_requested"] is True


def test_controller_ray_selects_each_row_and_rejects_outside_panel():
    corners = np.array(
        [
            [-1.0, 1.0, 2.0],
            [1.0, 1.0, 2.0],
            [1.0, -1.0, 2.0],
            [-1.0, -1.0, 2.0],
        ],
        dtype=np.float32,
    )
    origin = np.zeros(3, dtype=np.float32)

    assert selector_row_from_ray(origin, [0.0, 0.20, 1.0], corners) == 0
    assert selector_row_from_ray(origin, [0.0, -0.05, 1.0], corners) == 1
    assert selector_row_from_ray(origin, [2.0, 0.0, 1.0], corners) is None


def test_ray_hovered_row_is_selected_by_trigger():
    selector = RuntimeObjectSelector("rope_game")
    selector.update(0.0, _buttons())
    selector.update(0.1, _buttons(right_menu=True))
    selector.update(0.9, _buttons(right_menu=True))
    selector.update(1.0, _buttons())

    hover_events = selector.update(1.1, _buttons(), hovered_index=1)
    assert hover_events["highlighted_index"] == 1
    select_events = selector.update(
        1.2,
        _buttons(right_select=True),
        hovered_index=1,
    )
    assert select_events["selected_case"] == "sloth"


def test_selector_copy_marks_active_mode_and_loading_target():
    open_lines = selector_lines("rope_game", 1)
    assert open_lines[1] == "  Rope — Game  [Active]"
    assert open_lines[2] == "> Sloth — Free Play"
    assert "Sticks up/down" in open_lines[3]
    assert selector_lines(
        "rope_game",
        1,
        mode="loading",
        selected_case="sloth",
    ) == ["Loading Sloth…", "Please wait"]


def test_garden_selector_labels_both_objects_as_free_play():
    choices = object_choices_for_scene("garden")
    assert [choice.label for choice in choices] == [
        "Rope — Free Play",
        "Sloth — Free Play",
    ]
    selector = RuntimeObjectSelector("rope_game", scene_name="garden")
    assert selector.highlighted_case == "rope_game"
    lines = selector_lines(
        "rope_game",
        1,
        scene_name="garden",
    )
    assert lines[1] == "  Rope — Free Play  [Active]"
    assert lines[2] == "> Sloth — Free Play"
