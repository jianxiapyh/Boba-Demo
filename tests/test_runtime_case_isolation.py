from __future__ import annotations

import ast
import json
import logging
from pathlib import Path
import tempfile


REPO_ROOT = Path(__file__).resolve().parents[1]


def _manifest(case_name: str) -> dict:
    with (REPO_ROOT / "assets" / case_name / "manifest.json").open(
        "r", encoding="utf-8"
    ) as handle:
        return json.load(handle)


def test_rope_and_sloth_manifests_isolate_gameplay_modes():
    rope = _manifest("rope_game")
    sloth = _manifest("sloth")

    assert rope["display_name"] == "Rope — Game"
    assert rope["game_mode"] == "rope_pick_place_v1"
    assert rope["game_course"] == "course_v1.json"
    assert sloth["display_name"] == "Sloth — Free Play"
    assert sloth["game_mode"] == "free_play"
    assert "game_course" not in sloth
    assert sloth["tutorial_extra_slides"] == []


def test_historical_sloth_scaling_and_anchor_names_are_present():
    source_path = REPO_ROOT / "qqtt" / "engine" / "trainer_warp.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    trainer_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "InvPhyTrainerWarp"
    )
    translation_scale = next(
        ast.literal_eval(node.value)
        for node in trainer_class.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "LIVE_CONTROLLER_CASE_TRANSLATION_SCALE"
            for target in node.targets
        )
    )
    sloth_anchor_function = next(
        node
        for node in trainer_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_build_sloth_interaction_anchors"
    )
    anchor_source = ast.get_source_segment(source, sloth_anchor_function) or ""

    assert translation_scale["sloth"] == 2.0
    assert translation_scale["rope_game"] == 4.0
    assert all(
        name in anchor_source
        for name in (
            "left_leg",
            "right_leg",
            "left_arm",
            "right_arm",
            "torso_center",
        )
    )


def test_case_configuration_reset_discards_dynamic_mode_state():
    # Importing through qqtt is intentional here: this exercises the same singleton
    # used by the launcher in the fully provisioned Boba runtime environment.
    from qqtt.utils.config import cfg

    cfg.reset()
    cfg.demo_case_name = "rope_game"
    cfg.demo_game_course_path = "course_v1.json"
    cfg.self_collision = True
    cfg.reset()

    assert not hasattr(cfg, "demo_case_name")
    assert not hasattr(cfg, "demo_game_course_path")
    assert cfg.self_collision is False


def test_optimal_parameter_loading_does_not_mutate_package_data():
    from qqtt.utils.config import cfg

    cfg.reset()
    optimal = {"global_spring_Y": 1234.0, "drag_damping": 7.0}
    cfg.set_optimal_params(optimal)

    assert optimal == {"global_spring_Y": 1234.0, "drag_damping": 7.0}
    assert cfg.init_spring_Y == 1234.0
    assert cfg.drag_damping == 7.0
    cfg.reset()


def test_runtime_logger_replaces_the_previous_object_file_handler():
    from qqtt.utils.logger import logger

    with (
        tempfile.TemporaryDirectory() as first_dir,
        tempfile.TemporaryDirectory() as second_dir,
    ):
        logger.set_log_file(first_dir, name="rope")
        first_handler = logger.filehandler
        logger.set_log_file(second_dir, name="sloth")

        assert first_handler not in logger.handlers
        assert first_handler.stream is None
        assert logger.filehandler in logger.handlers
        assert sum(
            isinstance(handler, logging.FileHandler)
            for handler in logger.handlers
        ) == 1

        logger.removeHandler(logger.filehandler)
        logger.filehandler.close()
        logger.filehandler = None


def test_launcher_case_configuration_does_not_leak_rope_state_into_sloth():
    import numpy as np

    import boba_quest_immersive as launcher
    from qqtt.utils.config import cfg

    class _NoopLogger:
        @staticmethod
        def set_log_file(**_kwargs):
            return None

    launcher.np = np
    rope = launcher.configure_demo_case_runtime("rope_game", cfg, _NoopLogger())
    assert rope["case_name"] == "rope_game"
    assert cfg.demo_game_mode == "rope_pick_place_v1"
    assert cfg.demo_game_course_path.endswith("course_v1.json")
    assert cfg.self_collision is True

    sloth = launcher.configure_demo_case_runtime("sloth", cfg, _NoopLogger())
    assert sloth["case_name"] == "sloth"
    assert cfg.demo_game_mode == "free_play"
    assert cfg.demo_game_course_path is None
    assert cfg.self_collision is False
    assert cfg.runtime_static_collider_mode == "scene"
    cfg.reset()


def test_garden_forces_both_packages_into_isolated_free_play_mode():
    import numpy as np

    import boba_quest_immersive as launcher
    from qqtt.utils.config import cfg

    class _NoopLogger:
        @staticmethod
        def set_log_file(**_kwargs):
            return None

    launcher.np = np
    rope = launcher.configure_demo_case_runtime(
        "rope_game", cfg, _NoopLogger(), scene_name="garden"
    )
    assert rope["case_name"] == "rope_game"
    assert cfg.immersive_scene_name == "garden"
    assert cfg.demo_game_mode == "free_play"
    assert cfg.demo_game_course_path is None
    assert all("tutorial_rope_game_goal" not in path for path in cfg.demo_tutorial_slide_paths)

    sloth = launcher.configure_demo_case_runtime(
        "sloth", cfg, _NoopLogger(), scene_name="garden"
    )
    assert sloth["case_name"] == "sloth"
    assert cfg.immersive_scene_name == "garden"
    assert cfg.demo_game_mode == "free_play"
    assert cfg.demo_game_course_path is None
    cfg.reset()


def test_scene_parser_defaults_to_lab_and_accepts_garden_quality():
    import boba_quest_immersive as launcher

    defaults = launcher.build_parser().parse_args([])
    assert defaults.scene == "lab"
    assert defaults.garden_quality == "balanced"

    garden = launcher.build_parser().parse_args(
        ["--scene", "garden", "--garden-quality", "performance"]
    )
    assert garden.scene == "garden"
    assert garden.garden_quality == "performance"
    adaptive = launcher.build_parser().parse_args(
        ["--scene", "garden", "--garden-quality", "auto"]
    )
    assert adaptive.garden_quality == "auto"
