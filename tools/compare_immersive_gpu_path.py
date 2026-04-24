#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

QQTT_ROOT = REPO_ROOT / "qqtt"
qqtt_pkg = importlib.util.module_from_spec(
    importlib.machinery.ModuleSpec("qqtt", loader=None, is_package=True)
)
qqtt_pkg.__path__ = [str(QQTT_ROOT)]
sys.modules["qqtt"] = qqtt_pkg


def load_qqtt_module(module_name: str, relative_path: str | None = None):
    if relative_path is None:
        module_path = QQTT_ROOT / f"{module_name}.py"
    else:
        module_path = QQTT_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(f"qqtt.{module_name}", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"qqtt.{module_name}"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


immersive_scene = load_qqtt_module("immersive_scene")
SimpleLabSceneRenderer = immersive_scene.SimpleLabSceneRenderer
make_simple_lab_layout = immersive_scene.make_simple_lab_layout

_TRAINER_WARP_MODULE = None
_TRAINER_WARP_IMPORT_ERROR = None


def load_trainer_warp_module():
    global _TRAINER_WARP_MODULE, _TRAINER_WARP_IMPORT_ERROR
    if _TRAINER_WARP_MODULE is not None or _TRAINER_WARP_IMPORT_ERROR is not None:
        return _TRAINER_WARP_MODULE
    try:
        _TRAINER_WARP_MODULE = load_qqtt_module(
            "engine.trainer_warp",
            "engine/trainer_warp.py",
        )
    except Exception as exc:  # pragma: no cover - optional runtime path
        _TRAINER_WARP_IMPORT_ERROR = exc
        _TRAINER_WARP_MODULE = None
    return _TRAINER_WARP_MODULE


BASE_CAMERA_ROTATION = np.array(
    [[-1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
    dtype=np.float32,
)
DEFAULT_INTRINSIC = np.array(
    [[700.0, 0.0, 352.0], [0.0, 700.0, 352.0], [0.0, 0.0, 1.0]],
    dtype=np.float32,
)
DEPTH_EPS = 1.0e-4
DEFAULT_IPD_M = 0.064
STARTUP_PIXEL_MARGIN = 8.0
SUPPORT_ROI_PADDING = 40
SUPPORT_ROI_SNAP = 8
SUPPORT_ROI_MIN_SIZE = 64
SUPPORT_ROI_FULLFRAME_THRESHOLD = 0.70
FOCUS_SUBSET_SELECTION_PADDING = 24
SUPPORT_ROI_RENDER_SCALE = 1.25


def rotation_x(degrees: float) -> np.ndarray:
    radians = np.deg2rad(float(degrees))
    c = float(np.cos(radians))
    s = float(np.sin(radians))
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
        dtype=np.float32,
    )


def rotation_z(degrees: float) -> np.ndarray:
    radians = np.deg2rad(float(degrees))
    c = float(np.cos(radians))
    s = float(np.sin(radians))
    return np.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def make_pose(
    *,
    yaw_deg: float = 0.0,
    pitch_deg: float = 0.0,
    translation_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = rotation_z(yaw_deg) @ rotation_x(pitch_deg) @ BASE_CAMERA_ROTATION
    pose[:3, 3] = np.asarray(translation_xyz, dtype=np.float32)
    return pose


def make_stereo_eye_poses(
    center_pose: np.ndarray,
    ipd_m: float = DEFAULT_IPD_M,
) -> tuple[np.ndarray, np.ndarray]:
    center_pose = np.asarray(center_pose, dtype=np.float32)
    eye_offset = 0.5 * float(ipd_m) * center_pose[:3, 0]
    left_pose = np.array(center_pose, copy=True)
    right_pose = np.array(center_pose, copy=True)
    left_pose[:3, 3] -= eye_offset
    right_pose[:3, 3] += eye_offset
    return left_pose, right_pose


def default_pose_sequence() -> list[tuple[str, np.ndarray]]:
    return [
        ("center_view", make_pose()),
        ("yaw_left_25", make_pose(yaw_deg=-25.0)),
        ("yaw_right_25", make_pose(yaw_deg=25.0)),
        ("yaw_left_70", make_pose(yaw_deg=-70.0)),
        ("yaw_right_70", make_pose(yaw_deg=70.0)),
        ("pitch_down_15", make_pose(pitch_deg=15.0)),
        ("pitch_down_25", make_pose(pitch_deg=25.0)),
        ("lateral_left_12cm", make_pose(translation_xyz=(-0.12, 0.0, 0.0))),
        ("lateral_right_12cm", make_pose(translation_xyz=(0.12, 0.0, 0.0))),
        (
            "forward_lean_table_edge",
            make_pose(pitch_deg=30.0, translation_xyz=(0.0, 0.18, 0.0)),
        ),
    ]


def ensure_uint8_image(image: np.ndarray) -> np.ndarray:
    image_np = np.asarray(image)
    if image_np.dtype != np.uint8:
        image_np = np.clip(np.rint(image_np), 0.0, 255.0).astype(np.uint8)
    return np.ascontiguousarray(image_np)


def color_and_depth_to_numpy(
    color,
    depth,
) -> tuple[np.ndarray, np.ndarray]:
    if torch.is_tensor(color):
        color_np = color.detach().cpu().numpy()
    else:
        color_np = np.asarray(color)
    if torch.is_tensor(depth):
        depth_np = depth.detach().cpu().numpy()
    else:
        depth_np = np.asarray(depth)
    return ensure_uint8_image(color_np), np.asarray(depth_np, dtype=np.float32)


def save_rgb(path: Path, image: np.ndarray) -> None:
    image_rgba = ensure_uint8_image(image)
    if image_rgba.shape[-1] == 4:
        Image.fromarray(image_rgba).save(path)
    else:
        Image.fromarray(image_rgba[..., :3]).save(path)


def save_rgb_diff(path: Path, diff_rgb: np.ndarray) -> None:
    diff_u8 = np.clip(diff_rgb, 0.0, 255.0).astype(np.uint8)
    Image.fromarray(diff_u8).save(path)


def save_depth_diff(path: Path, pyrender_depth: np.ndarray, gpu_depth: np.ndarray) -> None:
    valid = (pyrender_depth > DEPTH_EPS) | (gpu_depth > DEPTH_EPS)
    if not np.any(valid):
        diff_vis = np.zeros((*pyrender_depth.shape, 3), dtype=np.uint8)
        Image.fromarray(diff_vis).save(path)
        return
    depth_diff = np.abs(pyrender_depth - gpu_depth)
    max_diff = float(np.percentile(depth_diff[valid], 99.0))
    max_diff = max(max_diff, 1.0e-6)
    normalized = np.clip(depth_diff / max_diff, 0.0, 1.0)
    diff_vis = (normalized[..., None] * 255.0).astype(np.uint8)
    diff_vis = np.repeat(diff_vis, 3, axis=2)
    Image.fromarray(diff_vis).save(path)


def depth_compose(
    background_color: np.ndarray,
    background_depth: np.ndarray,
    overlay_color: np.ndarray,
    overlay_depth: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    bg_color = np.asarray(background_color, dtype=np.float32)
    bg_depth = np.asarray(background_depth, dtype=np.float32)
    ov_color = np.asarray(overlay_color, dtype=np.float32)
    ov_depth = np.asarray(overlay_depth, dtype=np.float32)
    alpha = ov_color[..., 3:4] / 255.0
    ov_has_depth = ov_depth > DEPTH_EPS
    bg_has_depth = bg_depth > DEPTH_EPS
    visible = ov_has_depth & ((~bg_has_depth) | (ov_depth <= (bg_depth + 5.0e-3)))
    effective_alpha = alpha * visible[..., None].astype(np.float32)
    composed_color = np.array(bg_color, copy=True)
    composed_color[..., :3] = (
        bg_color[..., :3] * (1.0 - effective_alpha)
        + ov_color[..., :3] * effective_alpha
    )
    composed_color[..., 3] = 255.0
    composed_depth = np.array(bg_depth, copy=True)
    composed_depth[visible] = ov_depth[visible]
    return ensure_uint8_image(composed_color), composed_depth


def render_default_balanced_scene_full_eye(
    renderer: SimpleLabSceneRenderer,
    pose_world: np.ndarray,
    intrinsic: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    background_color, background_depth = color_and_depth_to_numpy(
        *renderer.render_background_eye(
            pose_world,
            intrinsic,
            width=width,
            height=height,
        )
    )
    table_color, table_depth = color_and_depth_to_numpy(
        *renderer.render_table_eye(
            pose_world,
            intrinsic,
            width=width,
            height=height,
        )
    )
    return depth_compose(
        background_color,
        background_depth,
        table_color,
        table_depth,
    )


def render_experimental_balanced_scene_full_eye(
    renderer: SimpleLabSceneRenderer,
    pose_world: np.ndarray,
    intrinsic: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    return color_and_depth_to_numpy(
        *renderer.render_eye(
            pose_world,
            intrinsic,
            width=width,
            height=height,
        )
    )


def render_per_eye_reference_scene_full_eye(
    renderer: SimpleLabSceneRenderer,
    pose_world: np.ndarray,
    intrinsic: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    return color_and_depth_to_numpy(
        *renderer.render_eye(
            pose_world,
            intrinsic,
            width=width,
            height=height,
        )
    )


def make_balanced_compare_trainer():
    trainer_module = load_trainer_warp_module()
    if trainer_module is None:
        return None
    trainer_module.cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return trainer_module.InvPhyTrainerWarp.__new__(trainer_module.InvPhyTrainerWarp)


def render_balanced_runtime_scene_pair(
    trainer,
    renderer: SimpleLabSceneRenderer,
    layout,
    left_pose_world: np.ndarray,
    right_pose_world: np.ndarray,
    intrinsic: np.ndarray,
    eye_width: int,
    eye_height: int,
    *,
    background_mode: str,
    side_wall_mode: str,
) -> tuple[
    tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
    dict[str, float] | None,
]:
    if trainer is None:
        raise RuntimeError("Balanced runtime compare path is unavailable in this environment.")
    trainer._immersive_balanced_runtime_state = (
        trainer._prepare_immersive_balanced_runtime_state(
            layout,
            left_pose_world,
            right_pose_world,
            intrinsic,
            intrinsic,
            eye_width,
            eye_height,
            int(renderer.width),
            int(renderer.height),
        )
    )
    trainer._immersive_balanced_runtime_state["background_mode"] = str(background_mode)
    trainer._immersive_balanced_runtime_state["side_wall_mode"] = str(side_wall_mode)
    frame_profile = {"_cuda_spans": []}
    left_scene_color_t, left_scene_depth_t, right_scene_color_t, right_scene_depth_t = (
        trainer._render_immersive_table_roi_scene_frames(
            renderer,
            layout,
            None,
            None,
            None,
            left_pose_world,
            right_pose_world,
            intrinsic,
            intrinsic,
            eye_width,
            eye_height,
            int(renderer.width),
            int(renderer.height),
            shared_scene_compose_cache={},
            reproject_caches={
                "background_source": {},
                "balanced_edge": {},
            },
            render_profile_frame=frame_profile,
        )
    )
    frame_profile = trainer._render_profile_finalize_frame(frame_profile)
    return (
        (
            color_and_depth_to_numpy(left_scene_color_t, left_scene_depth_t),
            color_and_depth_to_numpy(right_scene_color_t, right_scene_depth_t),
        ),
        frame_profile,
    )


def extract_side_repair_metadata(frame_profile: dict[str, float] | None) -> dict[str, float] | None:
    if not frame_profile:
        return None
    keys = [
        "scene_side_roi_left_ratio",
        "scene_side_roi_right_ratio",
        "scene_side_strip_left_width_ratio",
        "scene_side_strip_right_width_ratio",
        "scene_side_fullframe_fallback_left_ratio",
        "scene_side_fullframe_fallback_right_ratio",
        "scene_compose_side_left_cuda",
        "scene_compose_side_right_cuda",
        "scene_repair_background_left_cuda",
        "scene_repair_background_right_cuda",
        "scene_reproject_background_left_cuda",
        "scene_reproject_background_right_cuda",
        "scene_render_side_left_wall",
        "scene_render_side_right_wall",
        "scene_side_warp_left_used",
        "scene_side_warp_right_used",
        "scene_side_render_fallback_left_used",
        "scene_side_render_fallback_right_used",
    ]
    return {
        key: float(frame_profile.get(key, 0.0))
        for key in keys
        if key in frame_profile
    }


def compare_pair(
    pyrender_color: np.ndarray,
    pyrender_depth: np.ndarray,
    gpu_color: np.ndarray,
    gpu_depth: np.ndarray,
) -> dict[str, float]:
    pyrender_color_f = np.asarray(pyrender_color, dtype=np.float32)
    gpu_color_f = np.asarray(gpu_color, dtype=np.float32)
    pyrender_depth_f = np.asarray(pyrender_depth, dtype=np.float32)
    gpu_depth_f = np.asarray(gpu_depth, dtype=np.float32)
    affected_mask = (pyrender_depth_f > DEPTH_EPS) | (gpu_depth_f > DEPTH_EPS)
    rgb_diff = np.abs(pyrender_color_f[..., :3] - gpu_color_f[..., :3])
    depth_diff = np.abs(pyrender_depth_f - gpu_depth_f)
    if np.any(affected_mask):
        rgb_mae_affected = float(rgb_diff[affected_mask].mean())
        depth_mae_affected = float(depth_diff[affected_mask].mean())
    else:
        rgb_mae_affected = 0.0
        depth_mae_affected = 0.0
    return {
        "rgb_mae_all": float(rgb_diff.mean()),
        "rgb_mae_affected": rgb_mae_affected,
        "rgb_max_affected": float(rgb_diff[affected_mask].max()) if np.any(affected_mask) else 0.0,
        "depth_mae_affected": depth_mae_affected,
        "depth_max_affected": float(depth_diff[affected_mask].max()) if np.any(affected_mask) else 0.0,
        "affected_ratio": float(affected_mask.mean()),
        "pyrender_valid_ratio": float((pyrender_depth_f > DEPTH_EPS).mean()),
        "gpu_valid_ratio": float((gpu_depth_f > DEPTH_EPS).mean()),
    }


def time_render(fn, repeats: int) -> tuple[object, float]:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    result = None
    for _ in range(int(repeats)):
        result = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_ms = 1000.0 * (time.perf_counter() - start) / max(int(repeats), 1)
    return result, elapsed_ms


def make_layer_renderers(width: int, height: int):
    return {
        "background": lambda renderer, pose, intrinsic: renderer.render_background_eye(
            pose,
            intrinsic,
            width=width,
            height=height,
        ),
        "full": lambda renderer, pose, intrinsic: renderer.render_eye(
            pose,
            intrinsic,
            width=width,
            height=height,
        ),
        "table": lambda renderer, pose, intrinsic: renderer.render_table_eye(
            pose,
            intrinsic,
            width=width,
            height=height,
        ),
    }


def camera_pose_world_to_cv_w2c(camera_pose_world: np.ndarray) -> np.ndarray:
    camera_pose_world = np.asarray(camera_pose_world, dtype=np.float32)
    cv_from_gl = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    pose_world_cv = np.eye(4, dtype=np.float32)
    pose_world_cv[:3, :3] = camera_pose_world[:3, :3] @ cv_from_gl
    pose_world_cv[:3, 3] = camera_pose_world[:3, 3]
    return np.linalg.inv(pose_world_cv).astype(np.float32)


def object_bounds_corners(bounds_min, bounds_max) -> np.ndarray:
    bounds_min = np.asarray(bounds_min, dtype=np.float32)
    bounds_max = np.asarray(bounds_max, dtype=np.float32)
    return np.array(
        [
            [bounds_min[0], bounds_min[1], bounds_min[2]],
            [bounds_min[0], bounds_min[1], bounds_max[2]],
            [bounds_min[0], bounds_max[1], bounds_min[2]],
            [bounds_min[0], bounds_max[1], bounds_max[2]],
            [bounds_max[0], bounds_min[1], bounds_min[2]],
            [bounds_max[0], bounds_min[1], bounds_max[2]],
            [bounds_max[0], bounds_max[1], bounds_min[2]],
            [bounds_max[0], bounds_max[1], bounds_max[2]],
        ],
        dtype=np.float32,
    )


def project_world_point_into_eye(
    world_point,
    intrinsic: np.ndarray,
    w2c: np.ndarray,
    width: int,
    height: int,
) -> dict[str, object]:
    if world_point is None:
        return {"depth": None, "pixel": None, "in_bounds": False}
    world_point = np.asarray(world_point, dtype=np.float32)
    intrinsic = np.asarray(intrinsic, dtype=np.float32)
    w2c = np.asarray(w2c, dtype=np.float32)
    world_point_h = np.concatenate([world_point, np.array([1.0], dtype=np.float32)])
    camera_point = w2c @ world_point_h
    depth = float(camera_point[2])
    pixel = None
    in_bounds = False
    if abs(depth) > 1.0e-6:
        pixel_h = intrinsic @ camera_point[:3]
        pixel = pixel_h[:2] / max(pixel_h[2], 1.0e-6)
        in_bounds = bool(
            depth > DEPTH_EPS
            and pixel[0] >= -STARTUP_PIXEL_MARGIN
            and pixel[0] <= float(width) + STARTUP_PIXEL_MARGIN
            and pixel[1] >= -STARTUP_PIXEL_MARGIN
            and pixel[1] <= float(height) + STARTUP_PIXEL_MARGIN
        )
    return {"depth": depth, "pixel": pixel, "in_bounds": in_bounds}


def compute_projected_roi_bounds(
    bounds_min,
    bounds_max,
    intrinsic: np.ndarray,
    w2c: np.ndarray,
    width: int,
    height: int,
    *,
    padding: int = 0,
    snap: int = 1,
    min_size: int = 1,
) -> tuple[int, int, int, int] | None:
    if bounds_min is None or bounds_max is None:
        return None
    projected_pixels = []
    for point in object_bounds_corners(bounds_min, bounds_max):
        projection = project_world_point_into_eye(
            point,
            intrinsic,
            w2c,
            width,
            height,
        )
        pixel = projection["pixel"]
        depth = projection["depth"]
        if pixel is None or depth is None or float(depth) <= DEPTH_EPS:
            continue
        projected_pixels.append(
            np.array(
                [
                    np.clip(pixel[0], 0.0, float(width - 1)),
                    np.clip(pixel[1], 0.0, float(height - 1)),
                ],
                dtype=np.float32,
            )
        )
    if not projected_pixels:
        return None
    pixels_np = np.stack(projected_pixels, axis=0)
    x0 = int(np.floor(np.min(pixels_np[:, 0]))) - int(padding)
    x1 = int(np.ceil(np.max(pixels_np[:, 0]))) + int(padding) + 1
    y0 = int(np.floor(np.min(pixels_np[:, 1]))) - int(padding)
    y1 = int(np.ceil(np.max(pixels_np[:, 1]))) + int(padding) + 1
    if int(snap) > 1:
        x0 = int(np.floor(float(x0) / float(snap))) * int(snap)
        x1 = int(np.ceil(float(x1) / float(snap))) * int(snap)
        y0 = int(np.floor(float(y0) / float(snap))) * int(snap)
        y1 = int(np.ceil(float(y1) / float(snap))) * int(snap)
    x0 = max(0, x0)
    x1 = min(int(width), x1)
    y0 = max(0, y0)
    y1 = min(int(height), y1)

    def _expand_axis(lo: int, hi: int, limit: int) -> tuple[int, int]:
        size = int(hi) - int(lo)
        if size >= int(min_size):
            return int(lo), int(hi)
        if int(limit) <= int(min_size):
            return 0, int(limit)
        extra = int(min_size) - size
        lo -= extra // 2
        hi += extra - (extra // 2)
        if lo < 0:
            hi = min(int(limit), hi - lo)
            lo = 0
        if hi > int(limit):
            lo = max(0, lo - (hi - int(limit)))
            hi = int(limit)
        if (hi - lo) < int(min_size):
            if lo <= 0:
                lo = 0
                hi = min(int(limit), int(min_size))
            else:
                hi = int(limit)
                lo = max(0, hi - int(min_size))
        return int(lo), int(hi)

    x0, x1 = _expand_axis(x0, x1, width)
    y0, y1 = _expand_axis(y0, y1, height)
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def roi_area_ratio(
    roi_bounds: tuple[int, int, int, int] | None,
    width: int,
    height: int,
) -> float:
    if roi_bounds is None:
        return 0.0
    x0, y0, x1, y1 = [int(v) for v in roi_bounds]
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(((x1 - x0) * (y1 - y0)) / max(float(int(width) * int(height)), 1.0))


def resolve_support_render_roi(
    support_bounds_min,
    support_bounds_max,
    intrinsic: np.ndarray,
    w2c: np.ndarray,
    width: int,
    height: int,
) -> tuple[tuple[int, int, int, int] | None, float, bool]:
    roi_bounds = compute_projected_roi_bounds(
        support_bounds_min,
        support_bounds_max,
        intrinsic,
        w2c,
        width,
        height,
        padding=SUPPORT_ROI_PADDING,
        snap=SUPPORT_ROI_SNAP,
        min_size=SUPPORT_ROI_MIN_SIZE,
    )
    if roi_bounds is None:
        return None, 1.0, True
    ratio = roi_area_ratio(roi_bounds, width, height)
    if ratio > SUPPORT_ROI_FULLFRAME_THRESHOLD:
        return None, 1.0, True
    return roi_bounds, ratio, False


def roi_bounds_overlap(
    roi_a: tuple[int, int, int, int] | None,
    roi_b: tuple[int, int, int, int] | None,
) -> bool:
    if roi_a is None or roi_b is None:
        return False
    ax0, ay0, ax1, ay1 = [int(v) for v in roi_a]
    bx0, by0, bx1, by1 = [int(v) for v in roi_b]
    return not (ax1 <= bx0 or bx1 <= ax0 or ay1 <= by0 or by1 <= ay0)


def select_focus_subset_entry_ids(
    renderer: SimpleLabSceneRenderer,
    focus_roi_bounds: tuple[int, int, int, int],
    intrinsic: np.ndarray,
    w2c: np.ndarray,
    width: int,
    height: int,
) -> tuple[list[int], float, int, bool]:
    catalog_entries = renderer.focus_render_catalog_world_entries_ref()
    catalog_entries_by_id = renderer.focus_render_catalog_world_by_id_ref()
    bvh_nodes = renderer.focus_render_bvh_world_nodes()
    total_faces = int(renderer.focus_render_catalog_total_faces())
    if not catalog_entries or not bvh_nodes or total_faces <= 0:
        return [], 1.0, 0, True

    selected_entry_ids: set[int] = {
        int(v) for v in renderer.focus_render_active_table_entry_ids()
    }
    traversal_stack = [0]
    while traversal_stack:
        node_index = traversal_stack.pop()
        if node_index < 0 or node_index >= len(bvh_nodes):
            continue
        node = bvh_nodes[node_index]
        projected_bounds = compute_projected_roi_bounds(
            node["bounds_min"],
            node["bounds_max"],
            intrinsic,
            w2c,
            width,
            height,
            padding=FOCUS_SUBSET_SELECTION_PADDING,
            snap=1,
            min_size=1,
        )
        if not roi_bounds_overlap(projected_bounds, focus_roi_bounds):
            continue
        if bool(node.get("is_leaf", False)):
            for entry_id in node.get("leaf_entry_ids", []):
                selected_entry_ids.add(int(entry_id))
            continue
        right_child = int(node.get("right_child", -1))
        left_child = int(node.get("left_child", -1))
        if right_child >= 0:
            traversal_stack.append(right_child)
        if left_child >= 0:
            traversal_stack.append(left_child)

    ordered_selected_entry_ids = sorted(int(v) for v in selected_entry_ids)
    selected_face_count = 0
    for entry_id in ordered_selected_entry_ids:
        entry = catalog_entries_by_id.get(int(entry_id))
        if entry is None:
            return [], 1.0, 0, True
        selected_face_count += int(entry["face_count"])
    face_ratio = float(selected_face_count) / max(float(total_faces), 1.0)
    return (
        ordered_selected_entry_ids,
        face_ratio,
        len(ordered_selected_entry_ids),
        False,
    )


def make_support_focus_subset_cases(
    renderer: SimpleLabSceneRenderer,
    intrinsic: np.ndarray,
    width: int,
    height: int,
    *,
    max_cases: int = 4,
) -> list[dict[str, object]]:
    support_entries = [
        entry
        for entry in renderer.support_surface_entries_ref()
        if str(entry.get("kind", "support")).strip().lower() == "support"
    ]
    support_entries.sort(
        key=lambda entry: (
            -float(entry.get("support_area", 0.0)),
            int(entry.get("support_id", 0)),
        )
    )
    cases: list[dict[str, object]] = []
    seen_selection_keys: set[tuple[int, ...]] = set()
    for pose_name, pose_world in default_pose_sequence():
        w2c = camera_pose_world_to_cv_w2c(pose_world)
        for support_entry in support_entries:
            roi_bounds, roi_ratio, fullframe_fallback = resolve_support_render_roi(
                support_entry.get("render_bounds_min", support_entry["bounds_min"]),
                support_entry.get("render_bounds_max", support_entry["bounds_max"]),
                intrinsic,
                w2c,
                width,
                height,
            )
            if roi_bounds is None or fullframe_fallback:
                continue
            (
                selected_entry_ids,
                face_ratio,
                entry_count,
                selection_failed,
            ) = select_focus_subset_entry_ids(
                renderer,
                roi_bounds,
                intrinsic,
                w2c,
                width,
                height,
            )
            selection_key = tuple(int(v) for v in selected_entry_ids)
            if selection_failed or not selection_key or selection_key in seen_selection_keys:
                continue
            seen_selection_keys.add(selection_key)
            cases.append(
                {
                    "name": f"{pose_name}_support_{int(support_entry['support_id'])}",
                    "pose_world": np.asarray(pose_world, dtype=np.float32).copy(),
                    "support_id": int(support_entry["support_id"]),
                    "roi_bounds": tuple(int(v) for v in roi_bounds),
                    "roi_ratio": float(roi_ratio),
                    "selected_entry_ids": list(selection_key),
                    "face_ratio": float(face_ratio),
                    "entry_count": int(entry_count),
                    "render_scale": float(SUPPORT_ROI_RENDER_SCALE),
                }
            )
            break
        if len(cases) >= int(max_cases):
            break
    return cases


def run_compare(args: argparse.Namespace) -> dict[str, object]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_assets_root = Path(args.scene_assets_root).resolve()

    layout = make_simple_lab_layout(
        np.array([0.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, -1.0], dtype=np.float32),
    )
    intrinsic = np.array(DEFAULT_INTRINSIC, copy=True)
    intrinsic[0, 2] = 0.5 * float(args.width)
    intrinsic[1, 2] = 0.5 * float(args.height)

    pyrender_renderer = SimpleLabSceneRenderer(
        scene_assets_root,
        args.width,
        args.height,
        lighting_mode=immersive_scene.ILLIXR_BAKED_LIGHTING_MODE,
        balanced_render_backend="pyrender",
    )
    gpu_renderer = SimpleLabSceneRenderer(
        scene_assets_root,
        args.width,
        args.height,
        lighting_mode=immersive_scene.ILLIXR_BAKED_LIGHTING_MODE,
        balanced_render_backend="gpu",
    )
    balanced_compare_trainer = make_balanced_compare_trainer()
    pyrender_renderer.set_layout(layout)
    gpu_renderer.set_layout(layout)

    layer_renderers = make_layer_renderers(args.width, args.height)
    summary: dict[str, object] = {
        "scene_assets_root": str(scene_assets_root),
        "width": int(args.width),
        "height": int(args.height),
        "repeats": int(args.repeats),
        "balanced_runtime_compare_available": bool(balanced_compare_trainer is not None),
        "balanced_runtime_compare_error": None
        if _TRAINER_WARP_IMPORT_ERROR is None
        else repr(_TRAINER_WARP_IMPORT_ERROR),
        "poses": {},
        "focus_subsets": {},
    }

    try:
        focus_subset_cases = make_support_focus_subset_cases(
            pyrender_renderer,
            intrinsic,
            args.width,
            args.height,
        )
        for case in focus_subset_cases:
            case_name = str(case["name"])
            case_dir = output_dir / "focus_subsets" / case_name
            case_dir.mkdir(parents=True, exist_ok=True)
            pyrender_result, pyrender_ms = time_render(
                lambda c=case: pyrender_renderer.render_focus_subset_eye_roi(
                    c["pose_world"],
                    intrinsic,
                    c["selected_entry_ids"],
                    c["roi_bounds"],
                    render_scale=float(c["render_scale"]),
                ),
                args.repeats,
            )
            gpu_result, gpu_ms = time_render(
                lambda c=case: gpu_renderer.render_focus_subset_eye_roi(
                    c["pose_world"],
                    intrinsic,
                    c["selected_entry_ids"],
                    c["roi_bounds"],
                    render_scale=float(c["render_scale"]),
                ),
                args.repeats,
            )
            pyrender_color, pyrender_depth = color_and_depth_to_numpy(*pyrender_result)
            gpu_color, gpu_depth = color_and_depth_to_numpy(*gpu_result)
            metrics = compare_pair(
                pyrender_color,
                pyrender_depth,
                gpu_color,
                gpu_depth,
            )
            metrics["pyrender_ms"] = float(pyrender_ms)
            metrics["gpu_ms"] = float(gpu_ms)
            metrics["speedup"] = (
                float(pyrender_ms / gpu_ms) if gpu_ms > 1.0e-6 else 0.0
            )
            metrics["support_id"] = int(case["support_id"])
            metrics["roi_ratio"] = float(case["roi_ratio"])
            metrics["face_ratio"] = float(case["face_ratio"])
            metrics["entry_count"] = int(case["entry_count"])
            metrics["render_scale"] = float(case["render_scale"])
            summary["focus_subsets"][case_name] = metrics

            save_rgb(case_dir / "pyrender.png", pyrender_color)
            save_rgb(case_dir / "gpu.png", gpu_color)
            save_rgb_diff(
                case_dir / "rgb_diff.png",
                np.abs(
                    pyrender_color[..., :3].astype(np.float32)
                    - gpu_color[..., :3].astype(np.float32)
                ),
            )
            save_depth_diff(
                case_dir / "depth_diff.png",
                pyrender_depth,
                gpu_depth,
            )

        for pose_name, pose_world in default_pose_sequence():
            pose_dir = output_dir / pose_name
            pose_dir.mkdir(parents=True, exist_ok=True)
            pose_metrics: dict[str, object] = {"layers": {}}

            for layer_name, render_fn in layer_renderers.items():
                pyrender_result, pyrender_ms = time_render(
                    lambda rf=render_fn, pw=pose_world: rf(
                        pyrender_renderer,
                        pw,
                        intrinsic,
                    ),
                    args.repeats,
                )
                gpu_result, gpu_ms = time_render(
                    lambda rf=render_fn, pw=pose_world: rf(
                        gpu_renderer,
                        pw,
                        intrinsic,
                    ),
                    args.repeats,
                )
                pyrender_color, pyrender_depth = color_and_depth_to_numpy(*pyrender_result)
                gpu_color, gpu_depth = color_and_depth_to_numpy(*gpu_result)
                metrics = compare_pair(
                    pyrender_color,
                    pyrender_depth,
                    gpu_color,
                    gpu_depth,
                )
                metrics["pyrender_ms"] = float(pyrender_ms)
                metrics["gpu_ms"] = float(gpu_ms)
                metrics["speedup"] = (
                    float(pyrender_ms / gpu_ms) if gpu_ms > 1.0e-6 else 0.0
                )
                pose_metrics["layers"][layer_name] = metrics

                layer_dir = pose_dir / layer_name
                layer_dir.mkdir(parents=True, exist_ok=True)
                save_rgb(layer_dir / "pyrender.png", pyrender_color)
                save_rgb(layer_dir / "gpu.png", gpu_color)
                save_rgb_diff(
                    layer_dir / "rgb_diff.png",
                    np.abs(
                        pyrender_color[..., :3].astype(np.float32)
                        - gpu_color[..., :3].astype(np.float32)
                    ),
                )
                save_depth_diff(
                    layer_dir / "depth_diff.png",
                    pyrender_depth,
                    gpu_depth,
                )

            left_pose, right_pose = make_stereo_eye_poses(
                pose_world,
                ipd_m=args.ipd_m,
            )
            if balanced_compare_trainer is not None:
                (
                    (
                        (baseline_left_scene, baseline_left_depth),
                        (baseline_right_scene, baseline_right_depth),
                    ),
                    baseline_profile,
                ), baseline_ms = time_render(
                    lambda: render_balanced_runtime_scene_pair(
                        balanced_compare_trainer,
                        pyrender_renderer,
                        layout,
                        left_pose,
                        right_pose,
                        intrinsic,
                        args.width,
                        args.height,
                        background_mode="mono_center_background",
                        side_wall_mode="per_eye_roi_replace",
                    ),
                    args.repeats,
                )
                (
                    (
                        (repaired_left_scene, repaired_left_depth),
                        (repaired_right_scene, repaired_right_depth),
                    ),
                    repaired_profile,
                ), repaired_ms = time_render(
                    lambda: render_balanced_runtime_scene_pair(
                        balanced_compare_trainer,
                        pyrender_renderer,
                        layout,
                        left_pose,
                        right_pose,
                        intrinsic,
                        args.width,
                        args.height,
                        background_mode="per_eye_background",
                        side_wall_mode="disabled",
                    ),
                    args.repeats,
                )
            else:
                (
                    (baseline_left_scene, baseline_left_depth),
                    (baseline_right_scene, baseline_right_depth),
                ), baseline_ms = time_render(
                    lambda: (
                        render_default_balanced_scene_full_eye(
                            pyrender_renderer,
                            left_pose,
                            intrinsic,
                            args.width,
                            args.height,
                        ),
                        render_default_balanced_scene_full_eye(
                            pyrender_renderer,
                            right_pose,
                            intrinsic,
                            args.width,
                            args.height,
                        ),
                    ),
                    args.repeats,
                )
                repaired_left_scene = repaired_left_depth = None
                repaired_right_scene = repaired_right_depth = None
                repaired_ms = 0.0
                baseline_profile = None
                repaired_profile = None
            _, experimental_ms = time_render(
                lambda: (
                    render_experimental_balanced_scene_full_eye(
                        gpu_renderer,
                        left_pose,
                        intrinsic,
                        args.width,
                        args.height,
                    ),
                    render_experimental_balanced_scene_full_eye(
                        gpu_renderer,
                        right_pose,
                        intrinsic,
                        args.width,
                        args.height,
                    ),
                ),
                args.repeats,
            )
            for eye_name, eye_pose, baseline_scene, baseline_depth, repaired_scene, repaired_depth in (
                (
                    "final_left_eye",
                    left_pose,
                    baseline_left_scene,
                    baseline_left_depth,
                    repaired_left_scene,
                    repaired_left_depth,
                ),
                (
                    "final_right_eye",
                    right_pose,
                    baseline_right_scene,
                    baseline_right_depth,
                    repaired_right_scene,
                    repaired_right_depth,
                ),
            ):
                reference_scene, reference_depth = render_per_eye_reference_scene_full_eye(
                    pyrender_renderer,
                    eye_pose,
                    intrinsic,
                    args.width,
                    args.height,
                )
                experimental_scene, experimental_depth = (
                    render_experimental_balanced_scene_full_eye(
                        gpu_renderer,
                        eye_pose,
                        intrinsic,
                        args.width,
                        args.height,
                    )
                )
                baseline_metrics = compare_pair(
                    reference_scene,
                    reference_depth,
                    baseline_scene,
                    baseline_depth,
                )
                baseline_metrics["path_ms"] = float(baseline_ms)
                experimental_metrics = compare_pair(
                    reference_scene,
                    reference_depth,
                    experimental_scene,
                    experimental_depth,
                )
                experimental_metrics["path_ms"] = float(experimental_ms)
                repaired_metrics = None
                if repaired_scene is not None and repaired_depth is not None:
                    repaired_metrics = compare_pair(
                        reference_scene,
                        reference_depth,
                        repaired_scene,
                        repaired_depth,
                    )
                    repaired_metrics["path_ms"] = float(repaired_ms)
                pose_metrics[eye_name] = {
                    "baseline_vs_reference": baseline_metrics,
                    "repaired_vs_reference": repaired_metrics,
                    "experimental_vs_reference": experimental_metrics,
                }
                scene_dir = pose_dir / eye_name
                scene_dir.mkdir(parents=True, exist_ok=True)
                save_rgb(scene_dir / "reference.png", reference_scene)
                save_rgb(scene_dir / "baseline.png", baseline_scene)
                save_rgb(scene_dir / "experimental.png", experimental_scene)
                save_rgb_diff(
                    scene_dir / "baseline_rgb_diff.png",
                    np.abs(
                        reference_scene[..., :3].astype(np.float32)
                        - baseline_scene[..., :3].astype(np.float32)
                    ),
                )
                save_rgb_diff(
                    scene_dir / "experimental_rgb_diff.png",
                    np.abs(
                        reference_scene[..., :3].astype(np.float32)
                        - experimental_scene[..., :3].astype(np.float32)
                    ),
                )
                save_depth_diff(
                    scene_dir / "baseline_depth_diff.png",
                    reference_depth,
                    baseline_depth,
                )
                save_depth_diff(
                    scene_dir / "experimental_depth_diff.png",
                    reference_depth,
                    experimental_depth,
                )
                if repaired_scene is not None and repaired_depth is not None:
                    save_rgb(scene_dir / "repaired.png", repaired_scene)
                    save_rgb_diff(
                        scene_dir / "repaired_rgb_diff.png",
                        np.abs(
                            reference_scene[..., :3].astype(np.float32)
                            - repaired_scene[..., :3].astype(np.float32)
                        ),
                    )
                    save_depth_diff(
                        scene_dir / "repaired_depth_diff.png",
                        reference_depth,
                        repaired_depth,
                    )

            pose_metrics["runtime_profile"] = {
                "baseline": extract_side_repair_metadata(baseline_profile),
                "repaired": extract_side_repair_metadata(repaired_profile),
            }
            summary["poses"][pose_name] = pose_metrics
    finally:
        pyrender_renderer.delete()
        gpu_renderer.delete()

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def print_summary(summary: dict[str, object]) -> None:
    print(
        "Comparing the shipped pyrender balanced path and the opt-in hybrid GPU balanced backend "
        "against a full per-eye reference."
    )
    if not summary.get("balanced_runtime_compare_available", False):
        print(
            "Balanced runtime compare path is unavailable in this environment; "
            "runtime balanced A/B images/metrics were skipped."
        )
        if summary.get("balanced_runtime_compare_error"):
            print(f"Reason: {summary['balanced_runtime_compare_error']}")
    poses = summary.get("poses", {})
    focus_subsets = summary.get("focus_subsets", {})
    if focus_subsets:
        print("\n[focus_subsets]")
        for case_name, metrics in focus_subsets.items():
            print(
                f"  {case_name}: "
                f"rgb_mae={metrics['rgb_mae_affected']:.3f} "
                f"depth_mae={metrics['depth_mae_affected']:.5f} "
                f"entries={metrics['entry_count']} "
                f"roi_ratio={metrics['roi_ratio']:.3f} "
                f"face_ratio={metrics['face_ratio']:.3f} "
                f"pyrender_ms={metrics['pyrender_ms']:.3f} "
                f"gpu_ms={metrics['gpu_ms']:.3f} "
                f"speedup={metrics['speedup']:.2f}x"
            )
    for pose_name, pose_metrics in poses.items():
        print(f"\n[{pose_name}]")
        for layer_name, metrics in pose_metrics.get("layers", {}).items():
            print(
                f"  {layer_name}: "
                f"rgb_mae={metrics['rgb_mae_affected']:.3f} "
                f"depth_mae={metrics['depth_mae_affected']:.5f} "
                f"pyrender_ms={metrics['pyrender_ms']:.3f} "
                f"gpu_ms={metrics['gpu_ms']:.3f} "
                f"speedup={metrics['speedup']:.2f}x"
            )
        for eye_name in ("final_left_eye", "final_right_eye"):
            if eye_name not in pose_metrics:
                continue
            metrics = pose_metrics[eye_name]
            repaired_metrics = metrics.get("repaired_vs_reference")
            repaired_rgb_mae = (
                "n/a"
                if repaired_metrics is None
                else f"{repaired_metrics['rgb_mae_affected']:.3f}"
            )
            repaired_ms = (
                "n/a"
                if repaired_metrics is None
                else f"{repaired_metrics['path_ms']:.3f}"
            )
            print(
                f"  {eye_name}: "
                f"baseline_rgb_mae={metrics['baseline_vs_reference']['rgb_mae_affected']:.3f} "
                f"repaired_rgb_mae={repaired_rgb_mae} "
                f"experimental_rgb_mae={metrics['experimental_vs_reference']['rgb_mae_affected']:.3f} "
                f"baseline_ms={metrics['baseline_vs_reference']['path_ms']:.3f} "
                f"repaired_ms={repaired_ms}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the shipped pyrender balanced path and the opt-in hybrid GPU balanced backend "
            "against a full per-eye reference, "
            "save background/side-wall/table/final diff images, and report timing/error metrics."
        )
    )
    parser.add_argument(
        "--scene-assets-root",
        default=str(REPO_ROOT / "assets" / "scenes"),
        help="Path to the open scene assets root.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "tools" / "compare_immersive_gpu_path_output"),
        help="Directory where diff images and summary.json will be written.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=704,
        help="Render width for the comparison.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=704,
        help="Render height for the comparison.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=8,
        help="Number of timing repeats for each layer render.",
    )
    parser.add_argument(
        "--ipd-m",
        type=float,
        default=DEFAULT_IPD_M,
        help="Inter-pupillary distance used for final left/right eye scene comparisons.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_compare(args)
    print_summary(summary)
    print(f"\nWrote comparison artifacts to {Path(args.output_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
