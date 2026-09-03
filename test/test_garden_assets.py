from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from qqtt.garden_assets import (
    GARDEN_PIPELINE_VERSION,
    GardenAssetError,
    _apply_tabletop_opacity_support,
    _make_tabletop_patch,
    _prune_exterior_by_importance,
    _transform_vertex_payload,
    centerpiece_removal_mask,
    gaussian_importance,
    interaction_roi_mask,
    partition_gaussians_into_spatial_chunks,
    record_garden_profile,
    resolve_garden_quality,
    sha256_file,
    sh_rotation_matrix,
    validate_garden_quality,
)
from qqtt.garden_scene import (
    GardenSceneRenderer,
    gaussian_chunk_spheres_in_camera_frustum,
    garden_direct_output_enabled,
    make_garden_layout,
)
from qqtt.immersive_scene import SimpleLabSceneRenderer


REPO_ROOT = Path(__file__).resolve().parents[1]


def _calibration() -> dict:
    with (REPO_ROOT / "assets/scenes/garden/calibration.json").open(
        "r", encoding="utf-8"
    ) as handle:
        return json.load(handle)


def _sample_vertex(count: int = 96) -> np.ndarray:
    fields = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("opacity", "f4"),
        *((f"scale_{index}", "f4") for index in range(3)),
        *((f"rot_{index}", "f4") for index in range(4)),
        *((f"f_dc_{index}", "f4") for index in range(3)),
        *((f"f_rest_{index}", "f4") for index in range(45)),
    ]
    vertex = np.zeros(count, dtype=np.dtype(fields))
    angles = np.arange(count, dtype=np.float32) * (2.0 * np.pi / count)
    vertex["x"] = 0.59 * np.cos(angles)
    vertex["y"] = 0.59 * np.sin(angles)
    vertex["z"] = 0.0
    vertex["opacity"] = 2.0
    for index in range(3):
        vertex[f"scale_{index}"] = np.log(0.004)
    vertex["rot_0"] = 1.0
    return vertex


def test_calibration_transform_maps_table_center_to_origin_and_rotates_sh():
    calibration = _calibration()
    vertex = _sample_vertex(2)
    center = np.asarray(calibration["source_table_center"], dtype=np.float32)
    vertex["x"] = center[0]
    vertex["y"] = center[1]
    vertex["z"] = center[2]
    vertex["f_rest_0"] = [0.2, -0.1]

    transformed, xyz = _transform_vertex_payload(vertex, calibration)
    assert np.allclose(xyz, 0.0, atol=1.0e-6)
    assert np.isfinite(transformed["f_rest_0"]).all()
    rotation = np.asarray(calibration["source_to_canonical_rotation"])
    assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-6)
    assert np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-6)
    assert np.allclose(sh_rotation_matrix(np.eye(3)), np.eye(16), atol=2.0e-5)


def test_removal_patch_and_lod_generation_are_deterministic_and_preserve_roi():
    calibration = _calibration()
    points = np.array(
        [
            [0.0, 0.0, -0.5],
            [0.0, 0.0, 0.1],
            [0.8, 0.0, -0.5],
            [0.0, 0.0, 0.5],
        ],
        dtype=np.float32,
    )
    assert centerpiece_removal_mask(points, calibration).tolist() == [
        True,
        True,
        False,
        False,
    ]

    vertex = _sample_vertex()
    xyz = np.column_stack([vertex["x"], vertex["y"], vertex["z"]])
    first_vertex, first_xyz = _make_tabletop_patch(vertex, xyz, calibration)
    second_vertex, second_xyz = _make_tabletop_patch(vertex, xyz, calibration)
    assert np.array_equal(first_vertex, second_vertex)
    assert np.array_equal(first_xyz, second_xyz)
    assert np.max(np.linalg.norm(first_xyz[:, :2], axis=1)) <= 0.440001
    assert np.allclose(first_xyz[:, 2], -0.004)
    assert all(
        np.count_nonzero(first_vertex[f"f_rest_{index}"]) == 0
        for index in range(45)
    )

    extended_vertex = np.concatenate([vertex, vertex])
    extended_xyz = np.concatenate([xyz * 0.5, xyz + np.array([3.0, 0.0, 0.0])])
    roi_mask = np.zeros(len(extended_vertex), dtype=bool)
    roi_mask[: len(vertex)] = True
    first_keep = _prune_exterior_by_importance(
        extended_vertex,
        extended_xyz,
        roi_mask,
        0.25,
        mode="opacity",
    )
    second_keep = _prune_exterior_by_importance(
        extended_vertex,
        extended_xyz,
        roi_mask,
        0.25,
        mode="opacity",
    )
    assert np.array_equal(first_keep, second_keep)
    assert np.all(first_keep[roi_mask])
    expected_exterior = int(round(0.25 * np.count_nonzero(~roi_mask)))
    assert np.count_nonzero(first_keep & ~roi_mask) == expected_exterior


def test_garden_pruning_matches_activated_opacity_area_top_k():
    vertex = _sample_vertex(5)
    vertex["opacity"] = np.array([0.0, 0.0, 0.0, -4.0, 4.0], dtype=np.float32)
    scales = np.array(
        [
            [0.01, 0.01, 0.01],
            [0.02, 0.02, 0.01],
            [0.04, 0.02, 0.01],
            [0.20, 0.20, 0.01],
            [0.005, 0.005, 0.005],
        ],
        dtype=np.float32,
    )
    for axis in range(3):
        vertex[f"scale_{axis}"] = np.log(scales[:, axis])

    score = gaussian_importance(vertex, "opacity_area")
    expected = (1.0 / (1.0 + np.exp(-vertex["opacity"]))) * np.array(
        [0.0001, 0.0004, 0.0008, 0.04, 0.000025], dtype=np.float32
    )
    assert np.allclose(score, expected)

    roi_mask = np.array([True, False, False, False, False])
    xyz = np.zeros((5, 3), dtype=np.float32)
    keep = _prune_exterior_by_importance(
        vertex,
        xyz,
        roi_mask,
        0.5,
        mode="opacity_area",
    )
    assert keep.tolist() == [True, False, True, True, False]


def test_spatial_chunks_are_deterministic_contiguous_and_conservative():
    vertex = _sample_vertex(7)
    xyz = np.array(
        [
            [2.2, 0.1, 0.0],
            [-0.2, 0.1, 0.0],
            [0.2, 0.1, 0.0],
            [1.7, 0.1, 0.0],
            [0.3, 1.8, 0.0],
            [-0.3, 0.2, 0.0],
            [0.4, 0.2, 1.7],
        ],
        dtype=np.float32,
    )
    vertex["x"], vertex["y"], vertex["z"] = xyz.T
    vertex["f_dc_0"] = np.arange(vertex.shape[0], dtype=np.float32)
    config = {
        "cell_size_m": 1.5,
        "gaussian_extent_sigma": 4.0,
        "frustum_padding_m": 0.1,
        "prefetch_margin_ratio": 0.2,
        "near_plane_m": 0.01,
    }
    first_vertex, first_xyz, first_chunks = (
        partition_gaussians_into_spatial_chunks(vertex, xyz, config)
    )
    second_vertex, second_xyz, second_chunks = (
        partition_gaussians_into_spatial_chunks(vertex, xyz, config)
    )
    assert np.array_equal(first_vertex, second_vertex)
    assert np.array_equal(first_xyz, second_xyz)
    assert first_chunks == second_chunks
    assert sum(chunk["count"] for chunk in first_chunks) == vertex.shape[0]
    assert [chunk["start"] for chunk in first_chunks] == list(
        np.cumsum([0] + [chunk["count"] for chunk in first_chunks[:-1]])
    )
    extent = 4.0 * 0.004
    for chunk in first_chunks:
        start = int(chunk["start"])
        stop = start + int(chunk["count"])
        points = first_xyz[start:stop]
        bounds_min = np.asarray(chunk["bounds_min"])
        bounds_max = np.asarray(chunk["bounds_max"])
        center = np.asarray(chunk["sphere_center"])
        radius = float(chunk["sphere_radius"])
        assert np.all(points - extent >= bounds_min[None, :] - 1.0e-6)
        assert np.all(points + extent <= bounds_max[None, :] + 1.0e-6)
        assert np.all(np.linalg.norm(points - center, axis=1) + extent <= radius + 1.0e-6)


def test_chunk_sphere_frustum_test_is_conservative_and_respects_margin():
    centers = np.array(
        [
            [0.0, 0.0, 2.0],
            [2.5, 0.0, 2.0],
            [2.5, 0.0, 2.0],
            [0.0, 0.0, -2.0],
        ],
        dtype=np.float32,
    )
    radii = np.array([0.1, 0.1, 1.6, 0.1], dtype=np.float32)
    intrinsic = np.array(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    visible = gaussian_chunk_spheres_in_camera_frustum(
        centers,
        radii,
        np.eye(4, dtype=np.float32),
        intrinsic,
        100,
        100,
    )
    assert visible.tolist() == [True, False, True, False]
    prefetched = gaussian_chunk_spheres_in_camera_frustum(
        centers,
        radii,
        np.eye(4, dtype=np.float32),
        intrinsic,
        100,
        100,
        margin_ratio=1.0,
    )
    assert prefetched.tolist() == [True, True, True, False]


def test_garden_chunk_selection_uses_the_union_of_both_eye_frusta():
    renderer = object.__new__(GardenSceneRenderer)
    renderer.runtime_metadata = {
        "spatial_chunk_config": {
            "near_plane_m": 0.01,
            "frustum_padding_m": 0.0,
        }
    }
    renderer._chunk_centers_world = np.array(
        [[-1.5, 0.0, 2.0], [1.5, 0.0, 2.0], [0.0, 0.0, -2.0]],
        dtype=np.float32,
    )
    renderer._chunk_radii = np.full((3,), 0.1, dtype=np.float32)
    intrinsic = np.array(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    left_w2c = np.eye(4, dtype=np.float32)
    left_w2c[0, 3] = 1.5
    right_w2c = np.eye(4, dtype=np.float32)
    right_w2c[0, 3] = -1.5
    common = {"intrinsic_np": intrinsic, "width": 100, "height": 100}
    selected = renderer._stereo_chunk_mask(
        {**common, "w2c_cv_np": left_w2c},
        {**common, "w2c_cv_np": right_w2c},
        margin_ratio=0.0,
    )
    assert selected.tolist() == [True, True, False]


def test_interaction_roi_protects_gaussian_support_not_only_centers():
    calibration = _calibration()
    vertex = _sample_vertex(3)
    xyz = np.array(
        [
            [1.14, 0.0, 0.0],
            [1.20, 0.0, 0.0],
            [2.60, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    for axis in range(3):
        vertex[f"scale_{axis}"] = np.log(0.03)
    assert interaction_roi_mask(vertex, xyz, calibration).tolist() == [
        True,
        True,
        False,
    ]


def test_tabletop_opacity_support_fills_grazing_view_holes_without_new_points():
    calibration = _calibration()
    vertex = _sample_vertex(4)
    vertex["opacity"] = -4.0
    xyz = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.79, 0.0, 0.02],
            [0.81, 0.0, 0.0],
            [0.0, 0.0, 0.05],
        ],
        dtype=np.float32,
    )
    before_shape = vertex.shape
    changed = _apply_tabletop_opacity_support(vertex, xyz, calibration)
    assert changed == 2
    assert vertex.shape == before_shape
    assert np.allclose(vertex["opacity"][:2], 0.0)
    assert np.allclose(vertex["opacity"][2:], -4.0)


def test_layout_aligns_table_and_static_proxy_to_the_placement_frame():
    head = np.array([0.0, -0.78, -0.62], dtype=np.float32)
    layout = make_garden_layout(
        head,
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
        repo_root=REPO_ROOT,
        scene_up=np.array([0.0, 0.0, -1.0], dtype=np.float32),
    )
    assert layout.scene_name == "garden"
    assert np.allclose(layout.table_top_center, 0.0, atol=1.0e-6)
    assert layout.static_collision_mesh_vertices.shape == (186, 3)
    assert layout.static_collision_mesh_faces.shape == (340, 3)
    assert len(layout.support_surface_boxes) == 2
    assert [entry["kind"] for entry in layout.static_collision_surfaces] == [
        "cylinder",
        "rectangle",
    ]
    assert [entry["name"] for entry in layout.static_collision_surfaces] == [
        "round_tabletop",
        "stone_patio",
    ]
    assert layout.static_collision_boxes.shape == (6, 2, 3)
    assert np.allclose(layout.static_collision_surfaces[0]["center"], 0.0)
    assert np.isclose(layout.static_collision_surfaces[0]["extent_u"], 0.78)
    assert np.isclose(layout.static_collision_surfaces[0]["extent_v"], 0.065)
    assert np.allclose(
        layout.static_collision_surfaces[1]["center"],
        [0.0, 0.25, 0.72],
    )

    renderer = GardenSceneRenderer.__new__(GardenSceneRenderer)
    renderer.layout = layout
    with tempfile.TemporaryDirectory() as temporary_dir:
        debug_path = renderer.export_collision_proxy_obj(
            Path(temporary_dir) / "garden_collision.obj"
        )
        debug_payload = debug_path.read_text(encoding="utf-8")
    assert "o collision_proxy" in debug_payload
    assert "o placement_frame" in debug_payload
    assert "# axes: canonical +x right, +y forward, +z down" in debug_payload


def test_direct_gaussian_output_capability_is_garden_only():
    assert GardenSceneRenderer.supports_direct_gaussian_output is True
    assert not hasattr(SimpleLabSceneRenderer, "supports_direct_gaussian_output")
    garden_renderer = GardenSceneRenderer.__new__(GardenSceneRenderer)
    lab_renderer = SimpleLabSceneRenderer.__new__(SimpleLabSceneRenderer)
    assert garden_direct_output_enabled("garden", garden_renderer)
    assert not garden_direct_output_enabled("lab", garden_renderer)
    assert not garden_direct_output_enabled("garden", lab_renderer)


def test_garden_direct_output_preserves_black_composited_rgb_without_reblending():
    renderer = GardenSceneRenderer.__new__(GardenSceneRenderer)
    rgba = torch.zeros((4, 2, 3), dtype=torch.float32)
    rgba[0].fill_(0.20)
    rgba[1].fill_(0.40)
    rgba[2].fill_(0.60)
    rgba[3].fill_(0.25)
    depth = torch.tensor(
        [[float("nan"), -1.0, 0.5], [float("inf"), 1.25, 2.0]],
        dtype=torch.float32,
    )

    frame, clean_depth, metrics = renderer.prepare_direct_gaussian_eye_output(
        rgba,
        depth,
        output_dtype=torch.float32,
    )

    expected_rgb = rgba[:3].permute(1, 2, 0) * 255.0
    assert frame.shape == (2, 3, 4)
    assert frame.dtype is torch.float32
    assert torch.allclose(frame[..., :3], expected_rgb)
    assert torch.all(frame[..., 3] == 255.0)
    assert torch.equal(
        clean_depth,
        torch.tensor([[0.0, 0.0, 0.5], [0.0, 1.25, 2.0]]),
    )
    assert metrics["compose_mode"] == "garden_direct_output"
    assert metrics["garden_direct_output"] is True

    frame_uint8, _, _ = renderer.prepare_direct_gaussian_eye_output(
        rgba,
        depth,
        output_dtype=torch.uint8,
    )
    assert frame_uint8.dtype is torch.uint8
    assert torch.equal(frame_uint8[0, 0], torch.tensor([51, 102, 153, 255]))


def test_dynamic_suffix_sync_preserves_opacity_and_static_prefix():
    renderer = GardenSceneRenderer.__new__(GardenSceneRenderer)
    renderer._static_count = 2
    renderer._dynamic_capacity = 4
    renderer._dynamic_count = 0
    storage = SimpleNamespace(
        _xyz=torch.full((6, 3), -7.0),
        _rotation=torch.zeros((6, 4)),
        _features_dc=torch.zeros((6, 1, 3)),
        _features_rest=torch.zeros((6, 15, 3)),
        _opacity=torch.full((6, 1), -100.0),
        _scaling=torch.zeros((6, 3)),
    )
    storage._xyz[:2] = 9.0
    combined = SimpleNamespace()
    renderer._render_storage = storage
    renderer._combined_gaussians = combined
    renderer._active_static_count = 2
    dynamic = SimpleNamespace(
        _xyz=torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        get_rotation=torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
        ),
        _features_dc=torch.ones((2, 1, 3)),
        _features_rest=torch.ones((2, 15, 3)),
        _opacity=torch.tensor([[2.0], [3.0]]),
        _scaling=torch.full((2, 3), -2.0),
    )
    renderer._copy_dynamic_payload(dynamic, include_appearance=True)
    dynamic._xyz += 10.0
    renderer._copy_dynamic_payload(dynamic, include_appearance=False)

    assert torch.equal(storage._xyz[:2], torch.full((2, 3), 9.0))
    assert torch.equal(combined._xyz[2:4], dynamic._xyz)
    assert torch.equal(combined._opacity[2:4], dynamic._opacity)
    assert combined._xyz.shape[0] == 4
    assert torch.all(combined._features_dc[2:4] == 1.0)


def _write_fake_garden_repo(root: Path) -> None:
    scene_dir = root / "assets/scenes/garden"
    runtime_dir = root / "data/garden/runtime"
    scene_dir.mkdir(parents=True)
    runtime_dir.mkdir(parents=True)
    calibration = scene_dir / "calibration.json"
    collision = scene_dir / "collision.json"
    license_path = scene_dir / "LICENSE.md"
    calibration.write_text("{}\n", encoding="utf-8")
    collision.write_text("{}\n", encoding="utf-8")
    license_path.write_text("test\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "source": {
            "point_cloud_sha256": "source-hash",
            "point_cloud_member": "garden/point_cloud.ply",
            "cameras_member": "garden/cameras.json",
            "archive_filename": "models.zip",
        },
        "local": {
            "source_dir": "data/garden/source",
            "download_dir": "data/garden/downloads",
            "runtime_dir": "data/garden/runtime",
            "profile_cache": "data/garden/profile_cache.json",
        },
        "calibration": "assets/scenes/garden/calibration.json",
        "collision_proxy": "assets/scenes/garden/collision.json",
        "license": "assets/scenes/garden/LICENSE.md",
        "spatial_chunks": {
            "cell_size_m": 1.5,
            "gaussian_extent_sigma": 4.0,
            "frustum_padding_m": 0.1,
            "prefetch_margin_ratio": 0.2,
            "near_plane_m": 0.01,
        },
        "quality_order": ["full", "balanced", "performance"],
        "default_uncalibrated_auto_quality": "balanced",
        "qualities": {},
    }
    for quality in manifest["quality_order"]:
        ply = runtime_dir / f"garden_{quality}.ply"
        metadata = runtime_dir / f"garden_{quality}.json"
        ply.write_bytes(b"ply\nfixture")
        metadata.write_text(
            json.dumps(
                {
                    "pipeline_version": GARDEN_PIPELINE_VERSION,
                    "quality": quality,
                    "retention": 1.0,
                    "pruning_mode": "opacity",
                    "calibration_sha256": sha256_file(calibration),
                    "source_sha256": "source-hash",
                    "gaussian_count": 1,
                    "spatial_chunk_config": manifest["spatial_chunks"],
                    "spatial_chunk_count": 1,
                    "spatial_chunks": [
                        {
                            "cell": [0, 0, 0],
                            "start": 0,
                            "count": 1,
                            "bounds_min": [0.0, 0.0, 0.0],
                            "bounds_max": [0.0, 0.0, 0.0],
                            "sphere_center": [0.0, 0.0, 0.0],
                            "sphere_radius": 0.0,
                        }
                    ],
                    "ply_sha256": sha256_file(ply),
                }
            ),
            encoding="utf-8",
        )
        manifest["qualities"][quality] = {
            "retention": 1.0,
            "pruning_mode": "opacity",
            "ply": str(ply.relative_to(root)),
            "metadata": str(metadata.relative_to(root)),
        }
    (scene_dir / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def test_checksum_validation_and_auto_profile_cache(tmp_path: Path):
    _write_fake_garden_repo(tmp_path)
    ply, _ = validate_garden_quality(
        tmp_path, "balanced", verify_payload_hash=True
    )
    assert ply.name == "garden_balanced.ply"
    metadata_path = tmp_path / "data/garden/runtime/garden_balanced.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["retention"] = 0.5
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    try:
        validate_garden_quality(tmp_path, "balanced")
    except GardenAssetError as exc:
        assert "retention" in str(exc)
    else:
        raise AssertionError("stale Garden LOD settings passed metadata validation")
    metadata["retention"] = 1.0
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    profile_key = "fixture-profile"
    record_garden_profile(
        tmp_path,
        profile_key=profile_key,
        quality="full",
        source_fps=60.0,
        sample_count=120,
    )
    record_garden_profile(
        tmp_path,
        profile_key=profile_key,
        quality="balanced",
        source_fps=73.0,
        sample_count=120,
    )
    assert resolve_garden_quality(
        tmp_path, "auto", profile_key=profile_key, target_fps=72.0
    ) == "balanced"
    record_garden_profile(
        tmp_path,
        profile_key=profile_key,
        quality="full",
        source_fps=74.0,
        sample_count=120,
    )
    assert resolve_garden_quality(
        tmp_path, "auto", profile_key=profile_key, target_fps=72.0
    ) == "full"

    ply.write_bytes(ply.read_bytes() + b"corrupt")
    try:
        validate_garden_quality(tmp_path, "balanced", verify_payload_hash=True)
    except GardenAssetError as exc:
        assert "checksum mismatch" in str(exc)
    else:
        raise AssertionError("corrupted Garden runtime PLY passed checksum validation")
