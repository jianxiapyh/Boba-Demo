from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from qqtt.ambulance_scene import (
    AmbulanceSceneRenderer,
    _sample_projected_mesh_upper_surface,
    ambulance_mattress_alignment_metrics,
    ambulance_startup_gaze_pitch_down_degrees,
    make_ambulance_layout,
    validate_ambulance_scene,
)
from qqtt.garden_scene import direct_gaussian_scene_enabled
from qqtt.sog_loader import read_sog_metadata
from tools.fetch_demo_case_assets import resolve_shared_runtime_assets


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_bundled_ambulance_sog_matches_its_manifest():
    (
        sog_path,
        manifest_path,
        calibration_path,
        collision_proxy_path,
    ) = validate_ambulance_scene(REPO_ROOT)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = read_sog_metadata(sog_path)

    assert sog_path.suffix == ".sog"
    assert sog_path.stat().st_size > 16_000_000
    assert calibration_path.is_file()
    assert collision_proxy_path.suffix == ".glb"
    assert collision_proxy_path.stat().st_size < 1_200_000
    assert manifest["scene_name"] == "ambulance"
    assert manifest["sog_version"] == 2
    assert manifest["gaussian_count"] == 999_410
    assert manifest["sh_degree"] == 3
    assert metadata["version"] == 2
    assert metadata["count"] == manifest["gaussian_count"]
    assert metadata["shN"]["bands"] == 3
    assert manifest["collision_proxy_vertex_count"] == 30619
    assert manifest["collision_proxy_face_count"] == 57342
    assert ambulance_startup_gaze_pitch_down_degrees(REPO_ROOT) == 0.0


def test_ambulance_layout_places_the_object_surface_on_the_stretcher_and_head_on_bench():
    head_position = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    layout = make_ambulance_layout(
        head_position,
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
        scene_up=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        repo_root=REPO_ROOT,
    )

    assert layout.scene_name == "ambulance"
    canonical_head = np.asarray(layout.ambulance_seated_head_canonical)
    mapped_head = (
        canonical_head @ layout.canonical_to_world_rotation.T
        + layout.canonical_to_world_translation
    )
    assert np.allclose(
        canonical_head,
        [-0.013458, -0.737228, -0.763777],
        atol=2.0e-6,
    )
    assert np.allclose(mapped_head, head_position, atol=1.0e-6)
    assert np.allclose(layout.ambulance_seated_head_world, head_position)
    assert layout.ambulance_seated_view["support_name"] == "squad_bench"
    assert layout.ambulance_seated_view["faces"] == "stretcher"
    assert layout.ambulance_seated_view["startup_gaze_pitch_down_degrees"] == 0.0
    assert layout.ambulance_manifest["scene_name"] == "ambulance"
    assert np.isclose(
        layout.floor_z - head_position[2],
        1.2,
        atol=2.0e-3,
    )
    fitted_plane_center = np.asarray(
        layout.ambulance_mattress_fitted_plane_center_world
    )
    assert np.isclose(
        layout.floor_z - fitted_plane_center[2],
        0.519616,
        atol=3.0e-6,
    )
    assert np.isclose(
        layout.ambulance_mattress_collision_frame_raise_m,
        0.022,
    )
    assert np.isclose(
        layout.ambulance_mattress_surface_center_local_height_m,
        -0.08449035,
        atol=1.0e-7,
    )
    assert np.isclose(
        layout.ambulance_mattress_center_raise_m,
        0.022,
        atol=1.0e-7,
    )
    assert np.isclose(
        layout.table_top_center[1] - head_position[1],
        0.637234,
        atol=3.0e-6,
    )
    gaze_delta = np.asarray(layout.ambulance_gaze_target_world) - head_position
    gaze_horizontal = gaze_delta - np.dot(gaze_delta, layout.scene_up) * layout.scene_up
    gaze_horizontal /= np.linalg.norm(gaze_horizontal)
    assert np.allclose(
        layout.ambulance_gaze_target_world,
        layout.table_top_center,
        atol=1.0e-6,
    )
    assert np.dot(gaze_horizontal, layout.ambulance_seated_forward_world) > 0.98
    assert np.dot(gaze_delta, -layout.scene_up) > 0.65
    assert np.allclose(layout.table_size, [1.8, 0.52, 0.055])
    assert layout.static_collision_mesh_vertices.shape == (30627, 3)
    assert layout.static_collision_mesh_faces.shape == (57354, 3)
    assert layout.ambulance_mattress_collision_mesh_vertices.shape == (24301, 3)
    assert layout.ambulance_mattress_collision_mesh_faces.shape == (46829, 3)
    collision_kinds = [
        entry["kind"] for entry in layout.static_collision_mesh_metadata
    ]
    assert collision_kinds[0] == "source_mesh_full_resolution_surface"
    assert collision_kinds[1:-1] == ["source_mesh_decimated_surface"] * 6
    assert collision_kinds[-1] == "box"
    assert layout.static_collision_mesh_metadata[0]["face_count"] == 46829
    assert (
        sum(
            entry["face_count"]
            for entry in layout.static_collision_mesh_metadata
            if entry["kind"] == "source_mesh_decimated_surface"
        )
        == 10513
    )
    assert [
        entry["name"]
        for entry in layout.static_collision_detail_mesh_metadata
    ] == [
        "mattress_surface_full_resolution",
        "side_hardware_positive_v",
        "side_hardware_negative_v",
        "lower_structure_00",
        "lower_structure_01",
        "lower_structure_02",
        "lower_structure_03",
    ]
    assert layout.static_collision_detail_mesh_vertices.shape == (30619, 3)
    assert layout.static_collision_detail_mesh_faces.shape == (57342, 3)
    assert layout.static_collision_detail_mesh_component_bounds.shape == (
        7,
        2,
        3,
    )
    assert np.isclose(
        layout.static_collision_detail_mesh_contact["query_distance_m"],
        0.05,
    )
    assert layout.static_collision_detail_mesh_contact["substep_interval"] == 16
    assert np.isclose(
        layout.static_collision_detail_mesh_contact["margin_m"],
        0.01,
    )
    assert layout.static_collision_detail_mesh_two_sided is True
    assert layout.static_collision_detail_mesh_source_asset == (
        REPO_ROOT
        / "assets/scenes/ambulance_insta360/stretcher_collision_proxy.glb"
    )
    assert [surface["name"] for surface in layout.static_collision_surfaces] == [
        "ambulance_floor",
    ]
    assert [surface["kind"] for surface in layout.static_collision_surfaces] == [
        "rectangle",
    ]
    mattress_normal = layout.ambulance_mattress_normal_world
    assert np.allclose(
        mattress_normal,
        [0.0620856, 0.000258648, -0.99807084],
        atol=2.0e-6,
    )
    assert np.dot(mattress_normal, layout.scene_up) > 0.998
    fitted_to_collision = (
        np.asarray(layout.table_top_center) - fitted_plane_center
    )
    assert np.isclose(
        np.dot(fitted_to_collision, mattress_normal),
        0.022,
        atol=2.0e-6,
    )
    captured_center_delta = (
        np.asarray(layout.ambulance_mattress_captured_surface_center_world)
        - np.asarray(layout.table_top_center)
    )
    assert np.isclose(
        np.dot(captured_center_delta, mattress_normal),
        -0.08449035,
        atol=2.0e-6,
    )
    spawn_delta = (
        np.asarray(layout.active_table_surface_center)
        - np.asarray(layout.table_top_center)
    )
    assert np.isclose(
        np.dot(spawn_delta, mattress_normal),
        layout.ambulance_mattress_spawn_local_height_m,
        atol=2.0e-6,
    )
    assert np.isclose(
        layout.ambulance_mattress_spawn_local_height_m,
        0.0,
        atol=2.0e-6,
    )
    assert np.allclose(
        layout.active_table_surface_center,
        layout.table_top_center,
        atol=2.0e-6,
    )
    assert np.asarray(layout.support_surface_boxes).shape == (135, 2, 3)
    assert layout.ambulance_mattress_footprint_kind == "source_mesh_projection"
    assert np.isclose(layout.ambulance_mattress_edge_round_radius_m, 0.0)
    assert np.allclose(
        layout.ambulance_mattress_axis_u_world,
        [0.98859984, -0.13745189, 0.06146083],
        atol=2.0e-6,
    )
    assert np.allclose(
        layout.ambulance_mattress_axis_v_world,
        [0.1371708, 0.99050844, 0.00878948],
        atol=2.0e-6,
    )
    edge_counts = {}
    for face in layout.ambulance_mattress_collision_mesh_faces:
        for first, second in (
            (face[0], face[1]),
            (face[1], face[2]),
            (face[2], face[0]),
        ):
            edge = tuple(sorted((int(first), int(second))))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    assert any(count != 2 for count in edge_counts.values())
    detail_edge_counts = {}
    for face in layout.static_collision_detail_mesh_faces:
        for first, second in (
            (face[0], face[1]),
            (face[1], face[2]),
            (face[2], face[0]),
        ):
            edge = tuple(sorted((int(first), int(second))))
            detail_edge_counts[edge] = detail_edge_counts.get(edge, 0) + 1
    assert any(count != 2 for count in detail_edge_counts.values())
    rotation = np.asarray(layout.canonical_to_world_rotation)
    assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-6)
    assert np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-6)


def test_ambulance_standing_layout_places_head_in_aisle_at_authored_height():
    head_position = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    layout = make_ambulance_layout(
        head_position,
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
        scene_up=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        repo_root=REPO_ROOT,
        start_posture="standing",
    )

    assert layout.start_posture == "standing"
    assert layout.ambulance_start_posture == "standing"
    assert layout.ambulance_standing_view["support_name"] == "center_aisle"
    assert layout.ambulance_standing_view["faces"] == "stretcher"
    assert np.isclose(layout.startup_head_height_above_floor_m, 1.45)
    assert np.isclose(layout.floor_z - head_position[2], 1.45, atol=2.0e-6)
    assert np.allclose(layout.ambulance_standing_head_world, head_position)
    assert np.allclose(layout.ambulance_start_head_world, head_position)
    assert not np.allclose(layout.ambulance_seated_head_world, head_position)
    assert np.allclose(
        layout.ambulance_standing_head_canonical,
        [0.039645, -0.535824, -1.013777],
        atol=2.0e-6,
    )
    table_delta = np.asarray(layout.table_top_center) - head_position
    assert np.dot(table_delta, layout.scene_forward) > 0.40
    assert np.dot(table_delta, -layout.scene_up) > 0.85
    assert (
        ambulance_startup_gaze_pitch_down_degrees(
            REPO_ROOT,
            start_posture="standing",
        )
        == 30.0
    )


def test_settled_rope_is_measured_against_the_captured_mattress_surface():
    layout = make_ambulance_layout(
        np.zeros(3, dtype=np.float32),
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
        scene_up=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        repo_root=REPO_ROOT,
    )
    local_vertices = np.asarray(
        layout.ambulance_mattress_collision_mesh_local_vertices
    )
    local_faces = np.asarray(layout.ambulance_mattress_collision_mesh_faces)
    offset_u = 0.9
    offset_v = 0.1
    expected_height, edge_distance, inside = (
        _sample_projected_mesh_upper_surface(
            local_vertices,
            local_faces,
            local_u=offset_u,
            local_v=offset_v,
        )
    )
    assert inside is True
    assert edge_distance == 0.0
    support_center = (
        np.asarray(layout.ambulance_mattress_collision_frame_center_world)
        + np.asarray(layout.ambulance_mattress_axis_u_world) * offset_u
        + np.asarray(layout.ambulance_mattress_axis_v_world) * offset_v
        + np.asarray(layout.ambulance_mattress_normal_world) * expected_height
    )
    metrics = ambulance_mattress_alignment_metrics(layout, support_center)

    assert np.isclose(float(metrics["offset_u_m"]), offset_u, atol=1.0e-6)
    assert np.isclose(float(metrics["offset_v_m"]), offset_v, atol=1.0e-6)
    assert np.isclose(
        float(metrics["surface_height_offset_m"]),
        expected_height,
        atol=2.0e-5,
    )
    assert float(metrics["plane_error_m"]) < 2.0e-5
    assert float(metrics["edge_overrun_m"]) == 0.0
    assert metrics["inside_surface"] is True

    # The captured shell reaches beyond the old analytic +0.9 m end, and its
    # real end profile is strongly curved rather than one flat rectangle.
    end_height, end_distance, end_inside = _sample_projected_mesh_upper_surface(
        local_vertices,
        local_faces,
        local_u=1.0,
        local_v=0.0,
    )
    assert end_inside is True
    assert end_distance == 0.0
    assert abs(end_height - expected_height) > 0.03

    outside_u = 1.15
    outside_center = (
        np.asarray(layout.ambulance_mattress_collision_frame_center_world)
        + np.asarray(layout.ambulance_mattress_axis_u_world) * outside_u
    )
    outside_metrics = ambulance_mattress_alignment_metrics(
        layout,
        outside_center,
    )
    assert outside_metrics["footprint_value"] > 1.0
    assert outside_metrics["edge_overrun_m"] > 0.05
    assert outside_metrics["inside_surface"] is False


def test_ambulance_renderer_uses_the_direct_combined_gaussian_path():
    layout = make_ambulance_layout(
        np.zeros(3, dtype=np.float32),
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
        scene_up=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        repo_root=REPO_ROOT,
    )
    renderer = AmbulanceSceneRenderer(
        REPO_ROOT / "assets/scenes",
        320,
        240,
        repo_root=REPO_ROOT,
    )
    renderer.set_layout(layout)

    assert direct_gaussian_scene_enabled("ambulance", renderer)
    assert not direct_gaussian_scene_enabled("garden", renderer)
    assert renderer._chunk_counts.tolist() == [999_410]
    assert renderer.scene_analysis_cache_debug()["reason"] == "ambulance_sog_v2"
    alignment_debug = renderer.table_alignment_debug()
    assert alignment_debug["runtime_collision_kind"] == (
        "full_mattress_mesh+compact_detail_mesh+finite_support_surfaces"
    )
    assert alignment_debug["runtime_collision_heightfield_face_count"] == 0
    assert alignment_debug["runtime_collision_mattress_mesh_face_count"] == 46829
    assert alignment_debug["runtime_collision_detail_mesh_face_count"] == 57342
    assert alignment_debug["runtime_collision_detail_mesh_component_count"] == 7
    assert alignment_debug["runtime_collision_detail_mesh_two_sided"] is True
    assert alignment_debug["runtime_collision_detail_mesh_source_asset"].endswith(
        "assets/scenes/ambulance_insta360/stretcher_collision_proxy.glb"
    )
    assert [entry["kind"] for entry in renderer.support_surface_entries_ref()] == [
        "table",
        "floor",
    ]


def test_demo_asset_resolver_accepts_ambulance_as_a_public_scene():
    assets = resolve_shared_runtime_assets(REPO_ROOT, scene_name="ambulance")
    paths = {asset.path for asset in assets}

    assert REPO_ROOT / "assets/scenes/ambulance_insta360/ambulance_insta360.sog" in paths
    assert REPO_ROOT / "assets/scenes/ambulance_insta360/manifest.json" in paths
    assert REPO_ROOT / "assets/scenes/ambulance_insta360/calibration.json" in paths
    assert (
        REPO_ROOT
        / "assets/scenes/ambulance_insta360/stretcher_collision_proxy.glb"
        in paths
    )
    assert REPO_ROOT / "assets/scenes/ILLIXR_lab/headset.obj" in paths
