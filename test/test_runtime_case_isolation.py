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


def test_rope_and_sloth_share_controller_scaling_and_sloth_anchors_are_present():
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
    scalar_constants = {
        target.id: ast.literal_eval(node.value)
        for node in trainer_class.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance((target := node.targets[0]), ast.Name)
        and target.id
        in {
            "LIVE_CONTROLLER_SLOTH_HEAD_TOP_OFFSET_X_RATIO",
            "LIVE_CONTROLLER_SLOTH_HEAD_TOP_OFFSET_Y_RATIO",
            "LIVE_CONTROLLER_SLOTH_HEAD_TOP_CENTER_NODE_COUNT",
        }
    }

    assert translation_scale["sloth"] == 4.0
    assert translation_scale["rope_game"] == 4.0
    assert scalar_constants["LIVE_CONTROLLER_SLOTH_HEAD_TOP_OFFSET_X_RATIO"] == 0.16
    assert scalar_constants["LIVE_CONTROLLER_SLOTH_HEAD_TOP_OFFSET_Y_RATIO"] == -0.105
    assert scalar_constants["LIVE_CONTROLLER_SLOTH_HEAD_TOP_CENTER_NODE_COUNT"] == 12
    assert "center_region_node_count" in anchor_source
    assert "visual_surface_points" in anchor_source
    assert "_attach_visual_surface_center_to_anchor_def" in anchor_source
    assert "_nearest_object_surface_seed_index" in anchor_source
    assert all(
        name in anchor_source
        for name in (
            "left_leg",
            "right_leg",
            "left_arm",
            "right_arm",
            "head_top",
        )
    )
    assert "torso_center" not in anchor_source


def test_sloth_forehead_center_prefers_the_rendered_surface_point():
    import torch

    from qqtt.engine.trainer_warp import InvPhyTrainerWarp

    trainer = InvPhyTrainerWarp.__new__(InvPhyTrainerWarp)
    object_points = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    visual_surface_points = torch.tensor(
        [[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        dtype=torch.float32,
    )
    anchor = {
        "visual_point_index": 1,
        "region_indices": torch.tensor([0, 1]),
    }

    center = trainer._interaction_anchor_center_world(
        anchor,
        object_points,
        visual_surface_points=visual_surface_points,
    )

    assert torch.equal(center, visual_surface_points[1])


def test_sloth_forehead_physics_seed_uses_nearest_3d_surface_node():
    import torch

    from qqtt.engine.trainer_warp import InvPhyTrainerWarp

    trainer = InvPhyTrainerWarp.__new__(InvPhyTrainerWarp)
    surface_point = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    object_points = torch.tensor(
        [
            [0.0, 0.0, 0.080],  # Rear-head node at the same screen pixel.
            [0.004, 0.0, 0.004],  # Facial node nearest the rendered surface.
            [0.020, 0.0, 0.010],
        ],
        dtype=torch.float32,
    )

    seed_index = trainer._nearest_object_surface_seed_index(
        surface_point,
        object_points,
    )

    assert seed_index == 1


def test_sloth_controller_grab_does_not_precompress_anchor_volume():
    import torch

    from qqtt.engine.trainer_warp import InvPhyTrainerWarp

    trainer = InvPhyTrainerWarp.__new__(InvPhyTrainerWarp)
    point_deltas = torch.tensor(
        [
            [0.005, 0.0, 0.0],
            [0.025, 0.0, 0.0],
            [0.040, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    sloth_rest_lengths = trainer._controller_attachment_rest_lengths(
        point_deltas,
        case_name="sloth",
    )
    rope_rest_lengths = trainer._controller_attachment_rest_lengths(
        point_deltas,
        case_name="rope_game",
    )

    assert torch.allclose(sloth_rest_lengths, torch.tensor([0.005, 0.025, 0.040]))
    assert torch.allclose(rope_rest_lengths, torch.tensor([0.005, 0.010, 0.010]))


def test_direct_ray_target_beats_an_unrelated_surface_hit():
    import torch

    from qqtt.engine.trainer_warp import InvPhyTrainerWarp

    trainer = InvPhyTrainerWarp.__new__(InvPhyTrainerWarp)
    ray_origin = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    ray_direction = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
    surface_hit = torch.tensor([0.0, 0.0, 0.30], dtype=torch.float32)
    unrelated_surface_anchor = {
        "name": "right_arm",
        "center_world": torch.tensor([0.10, 0.0, 0.30], dtype=torch.float32),
        "selection_radius": 0.05,
    }
    deliberately_aimed_anchor = {
        "name": "left_leg",
        "center_world": torch.tensor([0.04, 0.0, 0.50], dtype=torch.float32),
        "selection_radius": 0.05,
    }

    ranked = trainer._rank_predefined_interaction_anchors(
        surface_hit,
        ray_origin,
        ray_direction,
        [unrelated_surface_anchor, deliberately_aimed_anchor],
    )

    assert ranked[0]["name"] == "left_leg"
    assert trainer._ray_targets_predefined_interaction_anchor(
        ray_origin,
        ray_direction,
        deliberately_aimed_anchor,
    )

    oversized_patch_anchor = {
        "name": "oversized_patch",
        "center_world": torch.tensor([0.08, 0.0, 0.50], dtype=torch.float32),
        "selection_radius": 0.50,
    }
    assert not trainer._ray_targets_predefined_interaction_anchor(
        ray_origin,
        ray_direction,
        oversized_patch_anchor,
    )


def test_direct_ray_target_can_select_a_marker_behind_the_first_surface_hit():
    import torch

    from qqtt.engine.trainer_warp import InvPhyTrainerWarp

    trainer = InvPhyTrainerWarp.__new__(InvPhyTrainerWarp)
    attach_anchor = torch.tensor([0.04, 0.0, 0.50], dtype=torch.float32)
    interaction_state = {"projected_anchor_distance": 0.04}
    remap_candidate = {
        "anchor_name": "left_leg",
        "springs": torch.tensor([[0, 1]], dtype=torch.long),
        "rest_lengths": torch.tensor([0.02], dtype=torch.float32),
        "attach_center_world": attach_anchor.clone(),
        "attach_anchor_world": attach_anchor.clone(),
        "attach_radius": 0.02,
        "selected_object_indices": torch.tensor([0], dtype=torch.long),
    }
    hit_world = torch.tensor([0.0, 0.0, 0.30], dtype=torch.float32)
    ray_direction = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
    controller_interaction_state = {"left": None, "right": None}

    rejected_without_target, _ = trainer._validate_new_controller_interaction_candidate(
        "left",
        {},
        interaction_state,
        remap_candidate,
        hit_world,
        ray_direction,
        controller_interaction_state=controller_interaction_state,
    )
    accepted_with_target, debug = trainer._validate_new_controller_interaction_candidate(
        "left",
        {},
        interaction_state,
        remap_candidate,
        hit_world,
        ray_direction,
        controller_interaction_state=controller_interaction_state,
        ray_target_selected=True,
    )

    assert rejected_without_target.startswith("back_facing_patch(")
    assert accepted_with_target is None
    assert debug["validation_path"] == "ray_target"
    assert debug["ray_target_selected"] is True


def test_controller_ray_is_thicker_in_all_overlay_rendering_paths():
    source_path = REPO_ROOT / "qqtt" / "engine" / "trainer_warp.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    trainer_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "InvPhyTrainerWarp"
    )
    constants = {
        target.id: ast.literal_eval(node.value)
        for node in trainer_class.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance((target := node.targets[0]), ast.Name)
        and target.id == "LIVE_CONTROLLER_RAY_RADIUS"
    }

    assert constants["LIVE_CONTROLLER_RAY_RADIUS"] == 2
    assert source.count("radius=self.LIVE_CONTROLLER_RAY_RADIUS") == 3


def test_sloth_visual_cleanup_keeps_only_the_main_connected_component():
    import numpy as np

    from gs_render import _largest_radius_connected_component_mask

    main_component = np.asarray(
        [
            [0.000, 0.000, 0.000],
            [0.004, 0.000, 0.000],
            [0.008, 0.000, 0.000],
            [0.008, 0.004, 0.000],
        ],
        dtype=np.float32,
    )
    detached_artifact = np.asarray(
        [
            [0.100, 0.000, 0.000],
            [0.104, 0.000, 0.000],
        ],
        dtype=np.float32,
    )

    keep_mask, debug = _largest_radius_connected_component_mask(
        np.concatenate([main_component, detached_artifact], axis=0),
        connectivity_radius=0.005,
    )

    assert keep_mask.tolist() == [True, True, True, True, False, False]
    assert debug["component_count"] == 2
    assert debug["largest_component_count"] == 4


def test_sloth_grounded_release_removes_only_planar_center_of_mass_drift():
    import torch

    from qqtt.engine.trainer_warp import InvPhyTrainerWarp

    trainer = InvPhyTrainerWarp.__new__(InvPhyTrainerWarp)
    trainer.num_all_points = 3
    state = {
        "x": torch.zeros((5, 3), dtype=torch.float32),
        "v": torch.tensor(
            [
                [2.0, 2.0, 3.0],
                [-1.0, 1.0, 6.0],
                [2.0, 0.0, 9.0],
                [10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0],
            ],
            dtype=torch.float32,
        ),
    }
    controller_velocities_before = state["v"][3:].clone()

    stabilized, debug = trainer._stabilize_grounded_release_planar_velocity(
        state,
        scene_up=[0.0, 0.0, -1.0],
    )

    object_mean = stabilized["v"][:3].mean(dim=0)
    assert torch.allclose(object_mean[:2], torch.zeros(2), atol=1.0e-6)
    assert torch.isclose(object_mean[2], state["v"][:3, 2].mean())
    assert torch.equal(stabilized["v"][3:], controller_velocities_before)
    assert debug["planar_speed_before_mps"] > 0.0
    assert debug["planar_speed_after_mps"] < 1.0e-6
    assert InvPhyTrainerWarp.IMMERSIVE_POST_RELEASE_PLANAR_DRIFT_CANCEL_CASES == (
        "sloth",
    )


def test_sloth_grounded_release_drift_guard_remains_active_after_a_correction():
    from qqtt.engine.trainer_warp import InvPhyTrainerWarp

    trainer = InvPhyTrainerWarp.__new__(InvPhyTrainerWarp)
    release_state = {
        "active": True,
        "planar_drift_ground_contact_seen": True,
        # A nonzero count represents an earlier correction before the lifted
        # head completed its fall and generated a second impact impulse.
        "planar_drift_correction_count": 1,
    }

    assert trainer._should_stabilize_post_release_planar_drift(
        release_state,
        case_name="sloth",
        ground_contact_now=True,
    )
    assert not trainer._should_stabilize_post_release_planar_drift(
        release_state,
        case_name="rope_game",
        ground_contact_now=True,
    )
    # Once contact has happened, a brief bounce must not disable the guard.
    assert trainer._should_stabilize_post_release_planar_drift(
        release_state,
        case_name="sloth",
        ground_contact_now=False,
    )
    release_state["active"] = False
    assert not trainer._should_stabilize_post_release_planar_drift(
        release_state,
        case_name="sloth",
        ground_contact_now=True,
    )


def test_sloth_grounded_release_removes_planar_position_creep_after_simulation():
    import torch

    from qqtt.engine.trainer_warp import InvPhyTrainerWarp

    trainer = InvPhyTrainerWarp.__new__(InvPhyTrainerWarp)
    trainer.num_all_points = 3
    state = {
        "x": torch.tensor(
            [
                [2.0, 2.0, 1.0],
                [3.0, 4.0, 2.0],
                [4.0, 3.0, 3.0],
                [20.0, 21.0, 22.0],
            ],
            dtype=torch.float32,
        ),
        "v": torch.tensor(
            [
                [0.4, 0.1, 1.0],
                [0.1, 0.3, 2.0],
                [0.7, 0.5, 3.0],
                [9.0, 8.0, 7.0],
            ],
            dtype=torch.float32,
        ),
    }
    original_object_z = state["x"][:3, 2].clone()
    original_controller_position = state["x"][3].clone()
    original_controller_velocity = state["v"][3].clone()

    stabilized, debug = trainer._stabilize_grounded_release_planar_state(
        state,
        scene_up=[0.0, 0.0, -1.0],
        planar_center_target=[1.25, 1.75, 99.0],
    )

    assert torch.allclose(
        stabilized["x"][:3].mean(dim=0)[:2],
        torch.tensor([1.25, 1.75]),
        atol=1.0e-6,
    )
    assert torch.equal(stabilized["x"][:3, 2], original_object_z)
    assert torch.allclose(
        stabilized["v"][:3].mean(dim=0)[:2],
        torch.zeros(2),
        atol=1.0e-6,
    )
    assert torch.equal(stabilized["x"][3], original_controller_position)
    assert torch.equal(stabilized["v"][3], original_controller_velocity)
    assert debug["planar_position_error_before_m"] > 0.0
    assert debug["planar_position_error_after_m"] < 1.0e-6


def test_sloth_broad_support_removes_rigid_angular_velocity_without_rotating_shape():
    import math

    import torch

    from qqtt.engine.trainer_warp import InvPhyTrainerWarp

    trainer = InvPhyTrainerWarp.__new__(InvPhyTrainerWarp)
    trainer.num_all_points = 5
    reference_positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.2, 0.0, 0.1],
            [0.0, 1.8, 0.2],
            [0.1, 0.0, 2.4],
            [0.8, 1.1, 0.9],
        ],
        dtype=torch.float32,
    )
    reference_offsets = reference_positions - reference_positions.mean(
        dim=0,
        keepdim=True,
    )
    angle = 0.35
    rotation = torch.tensor(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    object_center = torch.tensor([2.5, -1.5, 3.0], dtype=torch.float32)
    current_offsets = reference_offsets @ rotation.T
    object_positions = current_offsets + object_center
    angular_velocity = torch.tensor([0.2, -0.1, 1.4], dtype=torch.float32)
    center_velocity = torch.tensor([0.4, -0.2, 0.7], dtype=torch.float32)
    object_velocities = torch.cross(
        angular_velocity.unsqueeze(0).expand_as(current_offsets),
        current_offsets,
        dim=1,
    ) + center_velocity
    controller_position = torch.tensor([[20.0, 21.0, 22.0]])
    controller_velocity = torch.tensor([[9.0, 8.0, 7.0]])
    state = {
        "x": torch.cat((object_positions, controller_position), dim=0),
        "v": torch.cat((object_velocities, controller_velocity), dim=0),
    }

    stabilized, debug = trainer._stabilize_grounded_release_rigid_motion(
        state,
        scene_up=[0.0, 0.0, -1.0],
        planar_center_target=object_center,
    )

    stabilized_positions = stabilized["x"][:5]
    angular_velocity_after = trainer._estimate_object_rigid_angular_velocity(
        stabilized_positions,
        stabilized["v"][:5],
    )
    assert torch.allclose(stabilized_positions, object_positions, atol=2.0e-5)
    assert torch.allclose(
        torch.cdist(stabilized_positions, stabilized_positions),
        torch.cdist(object_positions, object_positions),
        atol=2.0e-5,
    )
    assert torch.allclose(
        stabilized_positions.mean(dim=0)[:2],
        object_center[:2],
        atol=1.0e-6,
    )
    assert torch.allclose(
        stabilized["v"][:5].mean(dim=0),
        torch.tensor([0.0, 0.0, 0.7]),
        atol=2.0e-5,
    )
    assert torch.linalg.norm(angular_velocity_after).item() < 2.0e-5
    assert torch.equal(stabilized["x"][5:], controller_position)
    assert torch.equal(stabilized["v"][5:], controller_velocity)
    assert debug["rigid_angular_speed_before_radps"] > 1.0
    assert debug["rigid_angular_speed_after_radps"] < 2.0e-5


def test_self_collision_rest_map_uses_configured_query_radius():
    import numpy as np
    import torch

    if not torch.cuda.is_available():
        return

    import warp as wp

    from qqtt.model.diff_simulator.spring_mass_warp import (
        build_resting_collision_pairs,
        update_potential_collision_restmap,
    )

    device = "cuda:0"
    collision_dist = 0.04
    rest_positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.9 * collision_dist, 0.0, 0.0],
            [4.1 * collision_dist, 0.0, 0.0],
            [0.5, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    rest_points = wp.array(rest_positions, dtype=wp.vec3, device=device)
    rest_grid = wp.HashGrid(16, 16, 16, device=device)
    rest_grid.build(rest_points, collision_dist)
    resting_pairs = wp.zeros((4, 4), dtype=wp.bool, device=device)
    wp.launch(
        build_resting_collision_pairs,
        dim=4,
        inputs=[rest_points, collision_dist, rest_grid.id],
        outputs=[resting_pairs],
        device=device,
    )
    wp.synchronize_device(device)
    resting_pairs_t = wp.to_torch(resting_pairs).detach().cpu()

    assert bool(resting_pairs_t[0, 1])
    # This pair is inside the trained 5x query but outside Sloth's 1x query.
    assert not bool(resting_pairs_t[0, 2])

    trained_grid = wp.HashGrid(16, 16, 16, device=device)
    trained_grid.build(rest_points, collision_dist * 5.0)
    trained_resting_pairs = wp.zeros((4, 4), dtype=wp.bool, device=device)
    wp.launch(
        build_resting_collision_pairs,
        dim=4,
        inputs=[rest_points, collision_dist * 5.0, trained_grid.id],
        outputs=[trained_resting_pairs],
        device=device,
    )
    wp.synchronize_device(device)
    trained_resting_pairs_t = wp.to_torch(trained_resting_pairs).detach().cpu()
    assert bool(trained_resting_pairs_t[0, 2])

    deformed_positions = rest_positions.copy()
    deformed_positions[2] = [-0.5 * collision_dist, 0.0, 0.0]
    deformed_points = wp.array(deformed_positions, dtype=wp.vec3, device=device)
    collision_grid = wp.HashGrid(16, 16, 16, device=device)
    collision_grid.build(deformed_points, collision_dist * 5.0)
    masks = wp.array(np.arange(4, dtype=np.int32), dtype=wp.int32, device=device)
    collision_indices = wp.zeros((4, 8), dtype=wp.int32, device=device)
    collision_counts = wp.zeros(4, dtype=wp.int32, device=device)
    wp.launch(
        update_potential_collision_restmap,
        dim=4,
        inputs=[
            deformed_points,
            masks,
            collision_dist,
            collision_grid.id,
            resting_pairs,
            4,
        ],
        outputs=[collision_indices, collision_counts],
        device=device,
    )
    wp.synchronize_device(device)
    collision_counts_t = wp.to_torch(collision_counts).detach().cpu()
    collision_indices_t = wp.to_torch(collision_indices).detach().cpu()
    point_zero_candidates = collision_indices_t[0, : collision_counts_t[0]]

    assert 2 in point_zero_candidates.tolist()


def test_controller_tracking_loss_also_arms_the_post_release_guard():
    source_path = REPO_ROOT / "qqtt" / "engine" / "trainer_warp.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    trainer_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "InvPhyTrainerWarp"
    )
    resolve_function = next(
        node
        for node in trainer_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_resolve_live_controller_interaction_anchors"
    )
    resolve_source = ast.get_source_segment(source, resolve_function) or ""
    invalid_start = resolve_source.index("if controller_world is None:")
    invalid_end = resolve_source.index("select_start_edge =", invalid_start)
    invalid_release_source = resolve_source[invalid_start:invalid_end]

    assert 'reason="controller_invalid"' in invalid_release_source
    assert "interaction_release_callback(" in invalid_release_source
    assert "interaction_state=released_interaction_state" in invalid_release_source


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
    assert cfg.self_collision_rest_exclusion_multiplier == 5.0
    assert cfg.runtime_lab_table_divider_merge_enabled is False

    sloth = launcher.configure_demo_case_runtime("sloth", cfg, _NoopLogger())
    assert sloth["case_name"] == "sloth"
    assert cfg.demo_game_mode == "free_play"
    assert cfg.demo_game_course_path is None
    assert cfg.self_collision is True
    assert cfg.self_collision_rest_exclusion_multiplier == 1.0
    assert (
        cfg.runtime_static_collider_mode
        == "scene_replace_active_table_with_smooth_top"
    )
    assert cfg.runtime_smooth_table_top_offset_m == 0.011
    assert cfg.runtime_smooth_table_top_thickness_m == 0.035
    assert cfg.runtime_lab_table_divider_merge_enabled is True
    assert cfg.runtime_lab_table_divider_lateral_inflate_m == 0.012
    assert cfg.runtime_lab_table_divider_surface_overlap_m == 0.006
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


def test_ambulance_forces_both_packages_into_isolated_free_play_mode():
    import numpy as np

    import boba_quest_immersive as launcher
    from qqtt.utils.config import cfg

    class _NoopLogger:
        @staticmethod
        def set_log_file(**_kwargs):
            return None

    launcher.np = np
    for case_name in ("rope_game", "sloth"):
        result = launcher.configure_demo_case_runtime(
            case_name,
            cfg,
            _NoopLogger(),
            scene_name="ambulance",
        )
        assert result["case_name"] == case_name
        assert cfg.immersive_scene_name == "ambulance"
        assert cfg.demo_game_mode == "free_play"
        assert cfg.demo_game_course_path is None
    cfg.reset()


def test_scene_parser_defaults_to_lab_and_accepts_garden_quality():
    import boba_quest_immersive as launcher

    defaults = launcher.build_parser().parse_args([])
    assert defaults.scene == "lab"
    assert defaults.garden_quality == "balanced"
    assert defaults.interactive_window_mode == "visible"
    assert defaults.immersive_start_posture == "standing"
    assert defaults.immersive_controller_max_motion_interval_m == 0.05

    seated = launcher.build_parser().parse_args(
        ["--immersive_start_posture", "seated"]
    )
    assert seated.immersive_start_posture == "seated"

    garden = launcher.build_parser().parse_args(
        ["--scene", "garden", "--garden-quality", "performance"]
    )
    assert garden.scene == "garden"
    assert garden.garden_quality == "performance"
    adaptive = launcher.build_parser().parse_args(
        ["--scene", "garden", "--garden-quality", "auto"]
    )
    assert adaptive.garden_quality == "auto"
    ambulance = launcher.build_parser().parse_args(["--scene", "ambulance"])
    assert ambulance.scene == "ambulance"


def test_launcher_keeps_the_visible_window_default_explicit():
    launcher_source = (REPO_ROOT / "boba_app.sh").read_text(encoding="utf-8")

    assert "--interactive_window_mode visible" in launcher_source
    assert "--interactive_window_mode hidden" not in launcher_source
    assert "--immersive_start_posture standing" in launcher_source
    assert "--immersive_start_posture auto" not in launcher_source
    assert "--immersive_controller_max_motion_interval_m 0.05" in launcher_source


def test_interactive_window_covers_over_half_of_standard_monitor_workarea():
    import boba_quest_immersive as launcher

    for monitor_width, monitor_height in ((1920, 1080), (2560, 1440), (3440, 1440)):
        width, height = launcher.interactive_window_size_for_workarea(
            monitor_width,
            monitor_height,
        )
        coverage = float(width * height) / float(monitor_width * monitor_height)
        assert coverage >= 0.50
        assert width <= int(
            monitor_width * launcher.INTERACTIVE_WINDOW_MAX_WORKAREA_FRACTION
        ) + 1
        assert height <= int(
            monitor_height * launcher.INTERACTIVE_WINDOW_MAX_WORKAREA_FRACTION
        ) + 1
        assert abs((float(width) / height) - (16.0 / 9.0)) < 0.003


def test_large_controller_jump_runs_one_physics_graph_per_rendered_frame():
    source_path = REPO_ROOT / "qqtt" / "engine" / "trainer_warp.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    trainer_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "InvPhyTrainerWarp"
    )
    immersive_function = next(
        node
        for node in trainer_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_run_quest_immersive_balanced"
    )
    immersive_source = ast.get_source_segment(source, immersive_function) or ""

    assert "_advance_immersive_controller_motion_target" in immersive_source
    assert (
        immersive_source.count(
            "wp.capture_launch(self.simulator.forward_graph)"
        )
        == 1
    )
    assert "for motion_interval_index" not in immersive_source
    assert '"simulated_target"' in immersive_source
    physics_launch_index = immersive_source.index(
        "wp.capture_launch(self.simulator.forward_graph)"
    )
    post_step_guard_index = immersive_source.index(
        "x, current_v = _apply_post_release_rigid_drift_guard("
    )
    interpolation_index = immersive_source.index(
        "lbs_with_rotation_reuse(",
        post_step_guard_index,
    )
    assert physics_launch_index < post_step_guard_index < interpolation_index


def test_object_switch_session_retains_the_interactive_preview_renderer():
    trainer_source = (
        REPO_ROOT / "qqtt" / "engine" / "trainer_warp.py"
    ).read_text(encoding="utf-8")

    assert '"interactive_preview_renderer": preview_renderer' in trainer_source
    assert '"interactive_spectator_renderer": (' in trainer_source
    assert "preview_renderer is not None and not keep_preview_renderer" in trainer_source


def test_visible_window_renders_a_scene_space_headset_and_controller_rays():
    source_path = REPO_ROOT / "qqtt" / "engine" / "trainer_warp.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    trainer_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "InvPhyTrainerWarp"
    )
    spectator_function = next(
        node
        for node in trainer_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_render_interactive_window_spectator_frame"
    )
    spectator_source = ast.get_source_segment(source, spectator_function) or ""

    assert "resolve_center_eye_pose" in spectator_source
    assert "render_scene_with_headset" in spectator_source
    assert "self._render_immersive_eye_frame" in spectator_source
    assert "self._project_live_controller_world_overlays_batched" in spectator_source
    assert "self._draw_live_controller_overlay" in spectator_source
    assert "_draw_interactive_window_headset_pose_overlay" not in source


def test_illixr_headset_assets_and_license_are_vendored():
    scene_root = REPO_ROOT / "assets" / "scenes" / "ILLIXR_lab"

    assert (scene_root / "headset.obj").stat().st_size > 50_000
    assert (scene_root / "headset.mtl").is_file()
    assert (scene_root / "HeadsetBake.png").stat().st_size > 100_000
    license_text = (scene_root / "ILLIXR_HEADSET_LICENSE.md").read_text(
        encoding="utf-8"
    )
    assert "github.com/ILLIXR/ILLIXR" in license_text
    assert "9c8f9b0656f51c973f1f4fc016b18a7d49a12195" in license_text
    assert "Board of Trustees of the University of Illinois" in license_text
