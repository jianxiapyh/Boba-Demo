#this is batched version incorporate latest single instance changes (slimmer lbs + mass node morton reordering + spring clustering)
from qqtt.data import RealData
from qqtt.utils import logger, cfg
from qqtt.model.diff_simulator import (
    SpringMassSystemWarp,
)
import copy
import csv
import json
import open3d as o3d
import numpy as np
import torch
import os
import queue
import warp as wp
import pickle
import cv2
import heapq
import threading

import torchvision
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.gaussian_renderer import render as render_gaussian
from gaussian_splatting.dynamic_utils import (
    lbs_with_rotation_reuse,
    build_rotation_reuse_cache,
    knn_weights_sparse,
    get_topk_indices,
)
from gs_render import (
    remove_gaussians_with_low_opacity,
)

import time
from types import SimpleNamespace

import torch.nn.functional as F 

#add visualization imports
import glfw
from OpenGL import GL as gl
import pycuda.driver as cuda_driver

from pathlib import Path
from sklearn.cluster import KMeans
from qqtt.live_openxr import (
    INDEX_TIP_JOINT_INDEX,
    MIDDLE_TIP_JOINT_INDEX,
    PALM_JOINT_INDEX,
    THUMB_TIP_JOINT_INDEX,
    WRIST_JOINT_INDEX,
    EyePoseSample,
    OpenXRControllerStream,
    OpenXRHandJointStream,
    controller_pose_forward,
    controller_pose_position,
    hand_anchor,
)
from qqtt.quest_display import OpenXRFramePanelMirror, OpenXRImmersiveBridge
from qqtt.immersive_scene import (
    SimpleLabSceneRenderer,
    ensure_simple_lab_assets,
    make_simple_lab_layout,
)
from qqtt.pyrender_cuda_bridge import PreviewTextureCudaUploader

TINY_BITMAP_FONT = {
    "0": ("111", "101", "101", "101", "111"),
    "1": ("010", "110", "010", "010", "111"),
    "2": ("111", "001", "111", "100", "111"),
    "3": ("111", "001", "111", "001", "111"),
    "4": ("101", "101", "111", "001", "001"),
    "5": ("111", "100", "111", "001", "111"),
    "6": ("111", "100", "111", "101", "111"),
    "7": ("111", "001", "001", "001", "001"),
    "8": ("111", "101", "111", "101", "111"),
    "9": ("111", "101", "111", "001", "111"),
    ".": ("000", "000", "000", "000", "010"),
    " ": ("000", "000", "000", "000", "000"),
    "m": ("101", "111", "101", "101", "101"),
    "s": ("111", "100", "111", "001", "111"),
    "f": ("111", "100", "110", "100", "100"),
    "p": ("110", "101", "110", "100", "100"),
}
#pyh moving timer class outside 
class Timer:
    def __init__(self, name):
        self.name = name
        self.elapsed = 0
        self.start_time = None
        self.cuda_start_event = None
        self.cuda_end_event = None
        self.use_cuda = torch.cuda.is_available()

    def start(self):
        if self.use_cuda:
            torch.cuda.synchronize()
            self.cuda_start_event = torch.cuda.Event(enable_timing=True)
            self.cuda_end_event = torch.cuda.Event(enable_timing=True)
            self.cuda_start_event.record()
        self.start_time = time.time()

    def stop(self):
        if self.use_cuda:
            self.cuda_end_event.record()
            torch.cuda.synchronize()
            self.elapsed = (
                self.cuda_start_event.elapsed_time(self.cuda_end_event) / 1000
            )  # convert ms to seconds
        else:
            self.elapsed = time.time() - self.start_time
        return self.elapsed

    def reset(self):
        self.elapsed = 0
        self.start_time = None
        self.cuda_start_event = None
        self.cuda_end_event = None


def _execute_immersive_balanced_scene_render_plan(
    renderer,
    render_plan,
    *,
    tensor_validator=None,
):
    def _validate_tensor(value, label):
        if tensor_validator is None:
            return value
        return tensor_validator(value, label=label)

    background_mode = str(render_plan["background_mode"])
    result = {
        "background_mode": background_mode,
        "shared_background": None,
        "left": {},
        "right": {},
    }
    if background_mode == "per_eye_background":
        for eye_label in ("left", "right"):
            eye_plan = render_plan[eye_label]
            background_render_start = time.perf_counter()
            background_color, background_depth = renderer.render_background_eye(
                eye_plan["eye_pose_world"],
                eye_plan["background_scene_intrinsic"],
                width=int(render_plan["scene_width"]),
                height=int(render_plan["scene_height"]),
            )
            result[eye_label]["background_color"] = _validate_tensor(
                background_color,
                f"{eye_label}.background_color",
            )
            result[eye_label]["background_depth"] = _validate_tensor(
                background_depth,
                f"{eye_label}.background_depth",
            )
            result[eye_label]["background_render_wall_s"] = (
                time.perf_counter() - background_render_start
            )
    elif background_mode == "mono_center_background":
        background_render_start = time.perf_counter()
        background_color, background_depth = renderer.render_background_eye(
            render_plan["center_eye_pose_world"],
            render_plan["center_scene_intrinsic"],
            width=int(render_plan["scene_width"]),
            height=int(render_plan["scene_height"]),
        )
        result["shared_background"] = {
            "color": _validate_tensor(
                background_color,
                "shared_background.color",
            ),
            "depth": _validate_tensor(
                background_depth,
                "shared_background.depth",
            ),
            "render_wall_s": time.perf_counter() - background_render_start,
        }
    else:
        raise ValueError(
            f"Unsupported immersive balanced background mode: {background_mode}"
        )

    for eye_label in ("left", "right"):
        eye_plan = render_plan[eye_label]
        table_render_start = time.perf_counter()
        if bool(eye_plan.get("table_fullframe_fallback", True)):
            table_color, table_depth = renderer.render_table_eye(
                eye_plan["eye_pose_world"],
                eye_plan["eye_intrinsic"],
                width=int(eye_plan["eye_width"]),
                height=int(eye_plan["eye_height"]),
            )
            table_render_info = None
        else:
            table_color, table_depth, table_render_info = renderer.render_table_eye_roi(
                eye_plan["eye_pose_world"],
                eye_plan["eye_intrinsic"],
                tuple(int(v) for v in eye_plan["table_roi_bounds"]),
                render_scale=float(eye_plan["table_roi_render_scale"]),
                return_render_info=True,
            )
        result[eye_label]["table_color"] = _validate_tensor(
            table_color,
            f"{eye_label}.table_color",
        )
        result[eye_label]["table_depth"] = _validate_tensor(
            table_depth,
            f"{eye_label}.table_depth",
        )
        result[eye_label]["table_render_info"] = (
            None if table_render_info is None else dict(table_render_info)
        )
        result[eye_label]["table_render_wall_s"] = (
            time.perf_counter() - table_render_start
        )
    return result


class _ImmersiveStaticSceneRenderWorker:
    def __init__(
        self,
        *,
        scene_assets_root,
        scene_width,
        scene_height,
        lighting_mode,
        balanced_render_backend,
        layout,
        cuda_device_index,
    ):
        self._scene_assets_root = scene_assets_root
        self._scene_width = int(scene_width)
        self._scene_height = int(scene_height)
        self._lighting_mode = str(lighting_mode)
        self._balanced_render_backend = str(balanced_render_backend)
        self._layout = copy.deepcopy(layout)
        self._cuda_device_index = int(cuda_device_index)
        self._request_queue: queue.Queue = queue.Queue(maxsize=1)
        self._response_queue: queue.Queue = queue.Queue(maxsize=1)
        self._thread = None
        self._startup_validation_request = None
        self._worker_readback_mode = None
        self._worker_readback_reason = None

    @staticmethod
    def _require_cuda_tensor(value, *, label):
        if not torch.is_tensor(value):
            raise TypeError(
                f"Immersive static-scene worker expected tensor output for {label}, "
                f"got {type(value).__name__}."
            )
        if value.device.type != "cuda":
            raise TypeError(
                f"Immersive static-scene worker expected CUDA tensor output for {label}, "
                f"got device={value.device}."
            )
        return value.contiguous()

    @staticmethod
    def _attach_worker_cuda_context(cuda_device_index):
        cuda_driver.init()
        cuda_device = cuda_driver.Device(int(cuda_device_index))
        worker_cuda_context = cuda_device.retain_primary_context()
        worker_cuda_context.push()
        return worker_cuda_context

    def start(self, validation_request=None):
        if self._thread is not None:
            return {
                "readback_mode": self._worker_readback_mode,
                "readback_reason": self._worker_readback_reason,
            }
        self._startup_validation_request = validation_request
        self._thread = threading.Thread(
            target=self._thread_main,
            name="ImmersiveStaticSceneWorker",
            daemon=True,
        )
        self._thread.start()
        init_result = self._response_queue.get(timeout=120.0)
        self._worker_readback_mode = init_result.get("readback_mode")
        self._worker_readback_reason = init_result.get("readback_reason")
        if not bool(init_result.get("ok", False)):
            raise RuntimeError(
                "Failed to initialize immersive static-scene worker: "
                f"{init_result.get('error', 'unknown error')}"
            )
        return {
            "readback_mode": self._worker_readback_mode,
            "readback_reason": self._worker_readback_reason,
        }

    def stop(self):
        if self._thread is None:
            return
        try:
            self._request_queue.put({"type": "stop"}, timeout=1.0)
        except queue.Full:
            pass
        self._thread.join(timeout=10.0)
        self._thread = None

    def submit(self, request):
        if self._thread is None:
            raise RuntimeError("Immersive static-scene worker has not been started.")
        self._request_queue.put({"type": "render", "request": request}, timeout=10.0)

    def get_result(self, timeout=None):
        if self._thread is None:
            raise RuntimeError("Immersive static-scene worker has not been started.")
        result = self._response_queue.get(timeout=timeout)
        if not bool(result.get("ok", False)):
            raise RuntimeError(
                "Immersive static-scene worker render failed: "
                f"{result.get('error', 'unknown error')}"
            )
        return result["payload"]

    def _thread_main(self):
        renderer = None
        worker_cuda_context = None
        try:
            if torch.cuda.is_available():
                torch.cuda.set_device(self._cuda_device_index)
                worker_cuda_context = self._attach_worker_cuda_context(
                    self._cuda_device_index
                )
            renderer = SimpleLabSceneRenderer(
                scene_assets_root=self._scene_assets_root,
                width=self._scene_width,
                height=self._scene_height,
                lighting_mode=self._lighting_mode,
                balanced_render_backend=self._balanced_render_backend,
                scene_analysis_cache_mode="auto",
            )
            renderer.set_layout(self._layout)
            if self._startup_validation_request is not None:
                _ = _execute_immersive_balanced_scene_render_plan(
                    renderer,
                    self._startup_validation_request,
                    tensor_validator=self._require_cuda_tensor,
                )
            worker_readback_mode = str(renderer.pyrender_readback_mode())
            worker_readback_reason = renderer.pyrender_readback_reason()
            if worker_readback_mode != "gl_cuda_interop":
                raise RuntimeError(
                    "immersive_timewarp scene_depth_reproject requires worker "
                    "pyrender_readback_mode=gl_cuda_interop "
                    f"(mode={worker_readback_mode} reason={worker_readback_reason})"
                )
            self._response_queue.put(
                {
                    "ok": True,
                    "readback_mode": worker_readback_mode,
                    "readback_reason": worker_readback_reason,
                }
            )
            while True:
                item = self._request_queue.get()
                if item.get("type") == "stop":
                    break
                if item.get("type") != "render":
                    continue
                request = item["request"]
                render_start = time.perf_counter()
                payload = _execute_immersive_balanced_scene_render_plan(
                    renderer,
                    request,
                    tensor_validator=self._require_cuda_tensor,
                )
                payload["worker_wall_ms"] = 1000.0 * (
                    time.perf_counter() - render_start
                )
                self._response_queue.put({"ok": True, "payload": payload})
        except Exception as exc:
            self._response_queue.put(
                {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "readback_mode": None if renderer is None else str(renderer.pyrender_readback_mode()),
                    "readback_reason": None if renderer is None else renderer.pyrender_readback_reason(),
                }
            )
        finally:
            if renderer is not None:
                renderer.delete()
            if worker_cuda_context is not None:
                try:
                    worker_cuda_context.pop()
                except Exception:
                    pass
                try:
                    worker_cuda_context.detach()
                except Exception:
                    pass

class InvPhyTrainerWarp:
    LIVE_HAND_MIN_VALID_JOINTS = 6
    LIVE_HAND_MIN_ONSCREEN_JOINTS = 6
    LIVE_HAND_MIN_SPREAD = 0.02
    LIVE_HAND_MOTION_GAIN = 1.25
    LIVE_HAND_PALM_RADIUS_X = 6
    LIVE_HAND_PALM_RADIUS_Y = 4
    LIVE_HAND_TIP_RADIUS = 2
    LIVE_HAND_PINCH_RADIUS = 3
    LIVE_HAND_LINE_RADIUS = 1
    LIVE_HAND_LEFT_COLOR = [255.0, 64.0, 64.0]
    LIVE_HAND_RIGHT_COLOR = [64.0, 160.0, 255.0]
    LIVE_CONTROLLER_RAY_LENGTH = 0.65
    LIVE_CONTROLLER_ORIGIN_RADIUS = 3
    LIVE_CONTROLLER_HIT_RADIUS = 4
    LIVE_CONTROLLER_INDICATOR_RADIUS = 2
    LIVE_CONTROLLER_LEFT_COLOR = [255.0, 96.0, 96.0]
    LIVE_CONTROLLER_RIGHT_COLOR = [96.0, 160.0, 255.0]
    LIVE_CONTROLLER_HIT_COLOR = [96.0, 255.0, 96.0]
    LIVE_CONTROLLER_SELECT_COLOR = [255.0, 255.0, 255.0]
    LIVE_CONTROLLER_SELECT_IDLE_COLOR = [255.0, 220.0, 64.0]
    LIVE_CONTROLLER_ATTACH_CANDIDATE_COLOR = [255.0, 176.0, 64.0]
    LIVE_CONTROLLER_ATTACH_ACTIVE_COLOR = [255.0, 64.0, 255.0]
    LIVE_CONTROLLER_TRANSLATION_SCALE_DEFAULT = 1.0
    LIVE_CONTROLLER_CASE_TRANSLATION_SCALE = {
        "sloth": 2.0,
        "rope": 4,
    }
    IMMERSIVE_LIVE_HEAD_TRANSLATION_SCALE = 1.0
    LIVE_CONTROLLER_HIT_WORLD_RADIUS = 0.03
    LIVE_CONTROLLER_ATTACH_MAX_REST_LENGTH = 0.01
    LIVE_CONTROLLER_PREDEFINED_ANCHOR_NODE_COUNT = 96
    LIVE_CONTROLLER_PREDEFINED_ANCHOR_RADIUS_SCALE = 1.75
    LIVE_CONTROLLER_PREDEFINED_ANCHOR_MIN_RADIUS = 0.05
    LIVE_CONTROLLER_PREVIEW_RADIUS = 3
    LIVE_CONTROLLER_PREVIEW_SELECTED_RADIUS = 5
    LIVE_CONTROLLER_PREVIEW_OCCUPIED_COLOR = [176.0, 176.0, 176.0]
    LIVE_CONTROLLER_CANDIDATE_SQUARE_RADIUS = 6
    LIVE_CONTROLLER_TEMPLATE_RADIUS_SCALE = 0.55
    LIVE_CONTROLLER_TEMPLATE_MIN_RADIUS = 0.012
    LIVE_CONTROLLER_BLOCK_POINT_COUNT = 15
    LIVE_CONTROLLER_BLOCK_SPRINGS_PER_POINT = 6
    LIVE_CONTROLLER_BLOCK_OFFSET_SCALE = 0.25
    LIVE_CONTROLLER_BLOCK_MIN_OFFSET = 0.05
    LIVE_CONTROLLER_SELECT_START_THRESHOLD = 0.20
    LIVE_CONTROLLER_SELECT_HOLD_THRESHOLD = 0.05
    LIVE_CONTROLLER_SELECT_RELEASE_FRAMES = 2
    LIVE_CONTROLLER_EXIT_HOLD_SECONDS = 0.75
    LIVE_CONTROLLER_ACTIVE_DEBUG_LOG_INTERVAL = 20
    LIVE_CONTROLLER_ACTIVE_MOTION_EPS = 1e-3
    LIVE_CONTROLLER_ACTIVE_TARGET_EPS = 1e-4
    LIVE_CONTROLLER_MULTI_POINTS_BACK_DEPTH_THRESHOLD = 0.015
    LIVE_CONTROLLER_MULTI_POINTS_BACK_PENALTY = 4.0
    LIVE_CONTROLLER_MULTI_POINTS_FETCH_SCALE = 4
    QUEST_PRIMARY_COMPOSITE_WIDTH = 2064
    IMMERSIVE_EYE_WIDTH = 1024
    IMMERSIVE_EYE_HEIGHT = 1024
    IMMERSIVE_BALANCED_INTERNAL_STEREO_MODE = (
        "mono_center_background_table_roi_per_eye"
    )
    IMMERSIVE_BALANCED_BACKGROUND_RENDER_SCALE = 0.625
    IMMERSIVE_BALANCED_BACKGROUND_OVERSCAN = 1.10
    IMMERSIVE_BALANCED_REFERENCE_DEPTH_MIN_M = 0.9
    IMMERSIVE_BALANCED_REFERENCE_DEPTH_MAX_M = 1.8
    IMMERSIVE_BALANCED_FAR_REFERENCE_DEPTH_MIN_M = 2.0
    IMMERSIVE_BALANCED_FAR_REFERENCE_DEPTH_MAX_M = 3.4
    IMMERSIVE_RENDER_PRESET_DEFAULTS = {
        "quality": {
            "scene_render_scale": 1.0,
            "scene_stereo_mode": "per_eye",
            "overlay_mode": "full",
            "lighting_mode": "baked_texture_ambient",
        },
        "balanced": {
            "scene_render_scale": 0.875,
            "scene_stereo_mode": IMMERSIVE_BALANCED_INTERNAL_STEREO_MODE,
            "overlay_mode": "minimal",
            "lighting_mode": "baked_texture_ambient",
        },
        "performance": {
            "scene_render_scale": 0.625,
            "scene_stereo_mode": "reproject_from_center",
            "overlay_mode": "minimal",
            "lighting_mode": "baked_texture_ambient",
        },
    }
    IMMERSIVE_SCENE_RENDER_SCALE_MIN = 0.25
    IMMERSIVE_SCENE_REST_SETTLE_STEPS = 90
    IMMERSIVE_SCENE_REST_VELOCITY_EPS = 0.035
    IMMERSIVE_SCENE_REST_POSITION_EPS = 0.015
    IMMERSIVE_IDLE_LOCK_STABLE_FRAMES = 12
    IMMERSIVE_IDLE_LOCK_MAX_SPEED = 0.025
    IMMERSIVE_IDLE_LOCK_MAX_MEAN_FRAME_DELTA = 0.002
    IMMERSIVE_IDLE_LOCK_SUPPORT_XY_MARGIN = 0.01
    IMMERSIVE_IDLE_LOCK_SUPPORT_Z_TOL = 0.015
    IMMERSIVE_IDLE_LOCK_CASE_MIN_SUPPORT_FRACTION = {
        "sloth": 0.18,
        "rope": 0.55,
    }
    IMMERSIVE_STARTUP_PLANE_EPS = 0.03
    IMMERSIVE_STARTUP_CENTER_EPS = 0.08
    IMMERSIVE_STARTUP_ALPHA_EPS = 0.02
    IMMERSIVE_STARTUP_DEPTH_EPS = 1e-4
    IMMERSIVE_STARTUP_PIXEL_MARGIN = 8.0
    IMMERSIVE_STARTUP_YAW_RADIANS = 0.5 * np.pi
    IMMERSIVE_COMPOSE_ALPHA_EPS = 1.0 / 255.0
    IMMERSIVE_COMPOSE_RAW_MIN_COVERAGE_RATIO = 5e-4
    IMMERSIVE_COMPOSE_VISIBLE_MIN_COVERAGE_RATIO = 1e-4
    IMMERSIVE_COMPOSE_MIN_RETENTION_RATIO = 0.08
    IMMERSIVE_SCENE_DEPTH_MIN_FINITE_RATIO = 0.95
    IMMERSIVE_SCENE_DEPTH_MIN_POSITIVE_RATIO = 0.02
    IMMERSIVE_GRAB_START_VALIDATION_DELAY_FRAMES = 1
    IMMERSIVE_GRAB_START_VALIDATION_FRAMES = 2
    IMMERSIVE_GRAB_START_MAX_TARGET_DELTA = 0.12
    IMMERSIVE_GRAB_START_TARGET_DELTA_RADIUS_SCALE = 2.5
    IMMERSIVE_GRAB_START_RELAXED_MAX_TARGET_DELTA = 0.20
    IMMERSIVE_GRAB_START_RELAXED_TARGET_DELTA_RADIUS_SCALE = 4.0
    IMMERSIVE_GRAB_START_MAX_CENTER_DELTA = 0.25
    IMMERSIVE_HEAD_TRANSLATION_EMA_ALPHA = 0.35
    IMMERSIVE_HEAD_RESET_JUMP_THRESHOLD = 0.20
    IMMERSIVE_HEAD_DEBUG_LOG_INTERVAL = 120
    IMMERSIVE_SCENE_CLEAR_RGBA = [255.0, 255.0, 255.0, 255.0]
    IMMERSIVE_REPROJECT_MIN_DEPTH = 1e-4
    IMMERSIVE_REPROJECT_HOLE_FILL_ITERS = 1
    IMMERSIVE_REPROJECT_HOLE_FILL_SECOND_PASS_INVALID_RATIO = 0.08
    IMMERSIVE_REPROJECT_WINNER_ATOL = 1e-5
    IMMERSIVE_REPROJECT_WINNER_RTOL = 1e-4
    IMMERSIVE_REPROJECT_NEAR_SPLAT_DEPTH = 0.9
    IMMERSIVE_REPROJECT_ROI_PADDING = 16
    IMMERSIVE_REPROJECT_ROI_TARGET_COVERAGE = 0.90
    IMMERSIVE_REPROJECT_STARTUP_PATCH_RADIUS = 3
    IMMERSIVE_REPROJECT_STARTUP_MIN_PATCH_COVERAGE = 0.25
    IMMERSIVE_TABLE_ROI_PADDING = 32
    IMMERSIVE_TABLE_ROI_SNAP = 32
    IMMERSIVE_TABLE_ROI_MIN_SIZE = 64
    IMMERSIVE_TABLE_ROI_FULLFRAME_THRESHOLD = 0.70
    IMMERSIVE_SIDE_WALL_ROI_PADDING = 32
    IMMERSIVE_SIDE_WALL_ROI_SNAP = 32
    IMMERSIVE_SIDE_WALL_ROI_MIN_SIZE = 64
    IMMERSIVE_SIDE_WALL_ROI_FULLFRAME_THRESHOLD = 0.35
    IMMERSIVE_BALANCED_SIDE_WALL_STRIP_PADDING = 48
    IMMERSIVE_BALANCED_SIDE_WALL_STRIP_SNAP = 16
    IMMERSIVE_BALANCED_SIDE_WALL_STRIP_MIN_WIDTH = 128
    IMMERSIVE_BALANCED_SIDE_WALL_STRIP_WARP_MAX_WIDTH_RATIO = 0.60
    IMMERSIVE_BALANCED_SIDE_WALL_STRIP_FULLFRAME_WIDTH_RATIO = 0.85
    IMMERSIVE_BALANCED_SIDE_WALL_STRIP_SHRINK_MAX_PX = 32
    IMMERSIVE_BALANCED_SIDE_WALL_WARP_MIN_VALID_COVERAGE = 0.10
    IMMERSIVE_BALANCED_EDGE_WARP_FEATHER_PX = 24
    IMMERSIVE_BALANCED_TABLE_ROI_PADDING = 40
    IMMERSIVE_BALANCED_TABLE_ROI_SNAP = 8
    IMMERSIVE_BALANCED_TABLE_ROI_MIN_SIZE = 64
    IMMERSIVE_BALANCED_TABLE_ROI_SHRINK_MAX_PX = 16
    IMMERSIVE_BALANCED_TABLE_ROI_SUPERSAMPLE_SCALE = 1.25
    IMMERSIVE_STARTUP_KEEPALIVE_INTERVAL_SECONDS = 0.25
    IMMERSIVE_CONTROLLER_HANDNESS_CONFIRM_STREAK = 5
    IMMERSIVE_CONTROLLER_HANDNESS_MAX_VALID_SAMPLES = 90
    IMMERSIVE_STARTUP_KEEPALIVE_RGBA = [232, 232, 232, 255]
    IMMERSIVE_GAUSSIAN_COMPOSE_ROI_PADDING = 24
    TIMING_OVERLAY_TEXT_COLOR = [255.0, 255.0, 255.0]
    TIMING_OVERLAY_BG_COLOR = [0.0, 0.0, 0.0]
    TIMING_OVERLAY_SCALE = 4
    TIMING_OVERLAY_MARGIN = 10
    TIMING_OVERLAY_REFERENCE_HEIGHT = 480
    TIMING_OVERLAY_MAX_SCALE = 12

    #getting called automatically right after you create an instance of the class
    def __init__(
        self,
        data_path,
        base_dir,
    ):
        cfg.data_path= data_path
        cfg.base_dir = base_dir
        cfg.device = "cuda:0"

        self.init_masks = None
        self.init_velocities = None
        # Load the data
        if cfg.data_type == "real":
            self.dataset = RealData(visualize=False, save_gt=False)
            self.structure_points = self.dataset.structure_points
            self.num_all_points = self.dataset.num_all_points
        elif cfg.data_type == "synthetic":
            print(f"synthetic data detected")
            import pdb
            pdb.set_trace()
        else:
            raise ValueError(f"Data type {cfg.data_type} not supported")

        self.controller_points_group = (
            self.dataset.controller_points.unsqueeze(0).contiguous()
        )
        print(
            "[live_openxr_controller] controller trace source=final_data_single_trace "
            f"trajectories={int(self.controller_points_group.shape[0])} "
            f"frames={int(self.controller_points_group.shape[1])} "
            f"points_per_frame={int(self.controller_points_group.shape[2])}",
            flush=True,
        )
        self.check_controller_group_same_start(
            self.controller_points_group,
            atol=1e-5,
        )
        self.frame_len = self.controller_points_group.shape[1]
        self.num_input_trajectories = self.controller_points_group.shape[0]
        first_frame_controller_points = self.controller_points_group[0][0]
        (
            self.init_vertices,
            self.init_springs,
            self.init_rest_lengths,
            self.init_masses,
            self.num_object_springs,
        ) = self._init_start(
            self.structure_points,
            first_frame_controller_points,
            object_radius=cfg.object_radius,
            object_max_neighbours=cfg.object_max_neighbours,
            controller_radius=cfg.controller_radius,
            controller_max_neighbours=cfg.controller_max_neighbours,
            mask=self.init_masks,
        )

        #pyh move gaussian to a class variable
        self.gaussians =  None

    def check_controller_group_same_start(self, controller_points_group: torch.Tensor,
                                        atol: float = 1e-6,
                                        rtol: float = 0.0,
                                        allow_global_translation: bool = False,
                                        translation_mode: str = "mean",  # "mean" or "first"
                                        verbose: bool = True):
        """
        controller_points_group: (N, T, C, 3) float tensor
        N = #instances, T = #frames, C = #controller points

        Checks whether all instances start from the same controller positions at frame 0.
        If allow_global_translation=True, we allow each instance to differ by a single 3D translation
        at frame 0 (useful if instances were pre-shifted).
        """
        assert controller_points_group.ndim == 4 and controller_points_group.shape[-1] == 3, \
            f"Expected (N,T,C,3), got {tuple(controller_points_group.shape)}"

        N, T, C, _ = controller_points_group.shape
        start = controller_points_group[:, 0]          # (N, C, 3)
        ref = start[0:1]                               # (1, C, 3)

        if allow_global_translation:
            if translation_mode == "mean":
                # translation = mean over controllers at frame 0
                t = start.mean(dim=1, keepdim=True) - ref.mean(dim=1, keepdim=True)  # (N,1,3)
            elif translation_mode == "first":
                # translation = use controller 0 at frame 0
                t = start[:, 0:1, :] - ref[:, 0:1, :]                                # (N,1,3)
            else:
                raise ValueError("translation_mode must be 'mean' or 'first'")

            start_aligned = start - t   # (N,C,3)
            diff = start_aligned - ref  # compare to ref
        else:
            diff = start - ref

        # per-instance max abs error over (C,3)
        per_inst_max = diff.abs().amax(dim=(1, 2))  # (N,)
        # per-instance allclose check
        # (torch.allclose is scalar; do vectorized)
        tol = atol + rtol * ref.abs().amax(dim=(1, 2))  # (1,) broadcastable
        ok = (per_inst_max <= tol).tolist()

        all_ok = all(ok)

        if verbose:
            print(f"[Test] N={N}, T={T}, C={C}, allow_translation={allow_global_translation} ({translation_mode})")
            print(f"[Test] all_ok={all_ok}")
            if not all_ok:
                bad = [i for i, v in enumerate(ok) if not v]
                print(f"[Test] mismatching instances: {bad}")
            print(f"[Test] per-instance max abs diff: {per_inst_max.detach().cpu().numpy()}")

        return all_ok, per_inst_max, ok

    def _build_controller_part_masks(self, controller_points, n_ctrl_parts, intrinsic, w2c):
        if n_ctrl_parts == 1:
            return [
                torch.ones(
                    controller_points.shape[0],
                    dtype=torch.bool,
                    device=controller_points.device,
                )
            ]

        controller_points_np = controller_points.detach().cpu().numpy()
        kmeans = KMeans(n_clusters=n_ctrl_parts, random_state=0, n_init=10)
        cluster_labels = kmeans.fit_predict(controller_points_np)

        projection = intrinsic @ w2c[:3, :]
        ordered_masks = []
        projected_x = []
        for cluster_idx in range(n_ctrl_parts):
            mask_np = cluster_labels == cluster_idx
            mask = torch.from_numpy(mask_np).to(device=controller_points.device)
            ordered_masks.append(mask)

            center = controller_points_np[mask_np].mean(axis=0)
            pixel = projection @ np.append(center, 1.0)
            pixel = pixel[:2] / pixel[2]
            projected_x.append(pixel[0])

        order = np.argsort(projected_x)
        return [ordered_masks[idx] for idx in order]

    def _build_controller_source_masks(self, controller_points, intrinsic, w2c):
        return self._build_controller_part_masks(
            controller_points, n_ctrl_parts=2, intrinsic=intrinsic, w2c=w2c
        )

    def _project_world_point_to_startup_screen_x(self, point_world, intrinsic, w2c):
        point_world_np = (
            point_world.detach().cpu().numpy()
            if torch.is_tensor(point_world)
            else np.asarray(point_world, dtype=np.float32)
        )
        intrinsic_np = (
            intrinsic.detach().cpu().numpy()
            if torch.is_tensor(intrinsic)
            else np.asarray(intrinsic, dtype=np.float32)
        )
        w2c_np = (
            w2c.detach().cpu().numpy()
            if torch.is_tensor(w2c)
            else np.asarray(w2c, dtype=np.float32)
        )
        projection = intrinsic_np @ w2c_np[:3, :]
        pixel_h = projection @ np.append(point_world_np.astype(np.float32), 1.0)
        depth = float(pixel_h[2])
        if not np.isfinite(depth) or abs(depth) < 1e-6:
            return None, depth, False
        screen_x = float(pixel_h[0] / depth)
        if not np.isfinite(screen_x):
            return None, depth, False
        return screen_x, depth, True

    def _assign_startup_controller_sources_by_screen_x(
        self,
        controller_source_masks,
        controller_source_anchor_centers,
        intrinsic,
        w2c,
        recorded_anchor_centers=None,
        assignment_camera="startup_camera",
    ):
        raw_projected_x = []
        raw_projected_depth = []
        raw_projected_valid = []
        for center in controller_source_anchor_centers:
            screen_x, depth, valid = self._project_world_point_to_startup_screen_x(
                center,
                intrinsic,
                w2c,
            )
            raw_projected_x.append(screen_x)
            raw_projected_depth.append(depth)
            raw_projected_valid.append(valid)

        raw_order = list(range(len(controller_source_anchor_centers)))
        if (
            len(controller_source_anchor_centers) == 2
            and len(controller_source_masks) == 2
            and all(raw_projected_valid)
        ):
            raw_order = sorted(
                raw_order,
                key=lambda idx: (raw_projected_x[idx], idx),
            )

        reordered_masks = [controller_source_masks[idx] for idx in raw_order]
        reordered_source_anchor_centers = [
            controller_source_anchor_centers[idx] for idx in raw_order
        ]
        reordered_recorded_anchor_centers = None
        if recorded_anchor_centers is not None:
            reordered_recorded_anchor_centers = [
                recorded_anchor_centers[idx] for idx in raw_order
            ]

        return {
            "assignment_camera": str(assignment_camera),
            "controller_source_masks": reordered_masks,
            "controller_source_anchor_centers": reordered_source_anchor_centers,
            "recorded_anchor_centers": reordered_recorded_anchor_centers,
            "raw_projected_x": raw_projected_x,
            "raw_projected_depth": raw_projected_depth,
            "raw_projected_valid": raw_projected_valid,
            "raw_order": raw_order,
            "swap_applied": raw_order != list(range(len(raw_order))),
            "left_raw_index": None if not raw_order else int(raw_order[0]),
            "right_raw_index": None if len(raw_order) < 2 else int(raw_order[1]),
        }

    def _log_startup_controller_source_assignment(
        self,
        prefix,
        assignment_debug,
    ):
        raw_projected_x = [
            None if value is None else float(value)
            for value in assignment_debug["raw_projected_x"]
        ]
        raw_projected_depth = [
            float(value) if value is not None else None
            for value in assignment_debug["raw_projected_depth"]
        ]
        print(
            f"{prefix} startup controller source assignment: "
            f"assignment_camera={assignment_debug.get('assignment_camera', 'startup_camera')} "
            f"raw_projected_x={raw_projected_x} "
            f"raw_projected_depth={raw_projected_depth} "
            f"valid={assignment_debug['raw_projected_valid']} "
            f"left_raw_index={assignment_debug['left_raw_index']} "
            f"right_raw_index={assignment_debug['right_raw_index']} "
            f"swap={int(bool(assignment_debug['swap_applied']))}",
            flush=True,
        )
        print(
            f"{prefix} startup controller source centers: "
            "left="
            f"{assignment_debug['controller_source_anchor_centers'][0].detach().cpu().numpy().tolist()} "
            "right="
            f"{assignment_debug['controller_source_anchor_centers'][1].detach().cpu().numpy().tolist()}",
            flush=True,
        )
        if assignment_debug.get("recorded_anchor_centers") is not None:
            print(
                f"{prefix} startup recorded anchor centers: "
                "left="
                f"{assignment_debug['recorded_anchor_centers'][0].detach().cpu().numpy().tolist()} "
                "right="
                f"{assignment_debug['recorded_anchor_centers'][1].detach().cpu().numpy().tolist()}",
                flush=True,
            )

    def _object_graph_neighbors(self):
        cached = getattr(self, "_object_graph_neighbor_cache", None)
        if cached is not None:
            return cached

        neighbors = [[] for _ in range(self.num_all_points)]
        springs_np = (
            self.init_springs[: self.num_object_springs].detach().cpu().numpy()
        )
        for endpoint0, endpoint1 in springs_np:
            idx0 = int(endpoint0)
            idx1 = int(endpoint1)
            if idx0 >= self.num_all_points or idx1 >= self.num_all_points:
                continue
            neighbors[idx0].append(idx1)
            neighbors[idx1].append(idx0)
        self._object_graph_neighbor_cache = neighbors
        return neighbors

    def _graph_region_from_seed(self, seed_idx, region_node_count, object_points):
        if seed_idx is None or region_node_count <= 0:
            return torch.empty(0, dtype=torch.long, device=object_points.device)

        neighbors = self._object_graph_neighbors()
        points_np = object_points.detach().cpu().numpy()
        best_distance = {int(seed_idx): 0.0}
        visited = set()
        heap = [(0.0, int(seed_idx))]
        ordered = []
        while heap and len(ordered) < region_node_count:
            distance, current = heapq.heappop(heap)
            if current in visited:
                continue
            visited.add(current)
            ordered.append(current)
            current_point = points_np[current]
            for neighbor in neighbors[current]:
                if neighbor in visited:
                    continue
                edge_cost = float(np.linalg.norm(current_point - points_np[neighbor]))
                new_distance = distance + edge_cost
                if new_distance >= best_distance.get(neighbor, float("inf")):
                    continue
                best_distance[neighbor] = new_distance
                heapq.heappush(heap, (new_distance, neighbor))

        if len(ordered) < region_node_count:
            remaining = [
                idx
                for idx in range(self.num_all_points)
                if idx not in visited
            ]
            if remaining:
                remaining_np = np.asarray(remaining, dtype=np.int64)
                seed_point = points_np[int(seed_idx)]
                distances = np.linalg.norm(points_np[remaining_np] - seed_point, axis=1)
                order = np.argsort(distances)[: max(0, region_node_count - len(ordered))]
                ordered.extend(int(remaining_np[idx]) for idx in order)

        return torch.as_tensor(
            ordered[:region_node_count], dtype=torch.long, device=object_points.device
        )

    def _pick_predefined_anchor_seed_index(
        self,
        depth_valid,
        required_mask,
        score_values,
        prefer_largest,
        used_indices,
    ):
        used_mask = torch.zeros_like(depth_valid)
        if used_indices:
            used_mask[list(used_indices)] = True

        mask = depth_valid & required_mask & (~used_mask)
        if not bool(mask.any().item()):
            mask = depth_valid & (~used_mask)
        if not bool(mask.any().item()):
            mask = depth_valid
        candidate_indices = torch.nonzero(mask, as_tuple=False).squeeze(1)
        if candidate_indices.numel() == 0:
            return None

        candidate_scores = score_values[candidate_indices]
        order = torch.argsort(candidate_scores, descending=prefer_largest)
        return int(candidate_indices[order[0]].item())

    def _interaction_anchor_case_name(self):
        return str(getattr(cfg, "demo_case_name", "sloth")).strip().lower()

    def _demo_case_world_scale(self, case_name=None):
        if case_name is None:
            case_name = self._interaction_anchor_case_name()
        case_name = str(case_name).strip().lower()
        active_case_name = self._interaction_anchor_case_name()
        if case_name != active_case_name:
            return 1.0
        return float(getattr(cfg, "demo_case_world_scale", 1.0))

    def _principal_axis_span_torch(self, points):
        if points.ndim != 2 or int(points.shape[0]) <= 1:
            return 0.0
        centered = points - points.mean(dim=0, keepdim=True)
        try:
            _, _, vh = torch.linalg.svd(centered, full_matrices=False)
            axis = vh[0]
            projected = centered @ axis
            return float((projected.max() - projected.min()).item())
        except RuntimeError:
            bounds = points.max(dim=0).values - points.min(dim=0).values
            return float(torch.linalg.norm(bounds).item())

    def _is_rope_family_case(self, case_name=None):
        if case_name is None:
            case_name = self._interaction_anchor_case_name()
        case_name = str(case_name).strip().lower()
        return case_name in {"rope", "hq_rope"}

    def _live_controller_case_profile(self, case_name=None):
        if case_name is None:
            case_name = self._interaction_anchor_case_name()
        case_name = str(case_name).strip().lower()
        default_anchor_names = {"left": None, "right": None}
        translation_case_name = case_name
        if self._is_rope_family_case(case_name):
            default_anchor_names = {"left": "left_end", "right": "right_end"}
            translation_case_name = "rope"
        translation_scale = self.LIVE_CONTROLLER_CASE_TRANSLATION_SCALE.get(
            translation_case_name,
            self.LIVE_CONTROLLER_TRANSLATION_SCALE_DEFAULT,
        )
        return {
            "case_name": case_name,
            "post_select_grab_mode": "translation_only",
            "post_select_translation_only": True,
            "default_anchor_names": dict(default_anchor_names),
            "controller_translation_scale": float(translation_scale),
        }

    def _idle_lock_case_profile(self, case_name=None):
        if case_name is None:
            case_name = self._interaction_anchor_case_name()
        case_name = str(case_name).strip().lower()
        return {
            "case_name": case_name,
            "stable_frames_required": int(self.IMMERSIVE_IDLE_LOCK_STABLE_FRAMES),
            "max_speed": float(self.IMMERSIVE_IDLE_LOCK_MAX_SPEED),
            "max_mean_frame_delta": float(
                self.IMMERSIVE_IDLE_LOCK_MAX_MEAN_FRAME_DELTA
            ),
            "min_support_fraction": float(
                self.IMMERSIVE_IDLE_LOCK_CASE_MIN_SUPPORT_FRACTION.get(
                    case_name,
                    self.IMMERSIVE_IDLE_LOCK_CASE_MIN_SUPPORT_FRACTION["sloth"],
                )
            ),
        }

    def _make_idle_lock_state(self):
        return {
            "active": False,
            "stable_frame_count": 0,
            "locked_state": None,
            "last_object_points": None,
            "last_object_points_prev": None,
            "support_fraction": 0.0,
        }

    def _build_idle_locked_sim_state(self, sim_state):
        return {
            "x": sim_state["x"].detach().clone(),
            "v": torch.zeros_like(sim_state["v"]).detach().clone(),
        }

    def _scene_support_fraction(
        self,
        object_points,
        support_surface_boxes,
        scene_up,
        *,
        xy_margin=None,
        z_tolerance=None,
    ):
        if object_points is None or int(object_points.numel()) == 0:
            return 0.0
        if support_surface_boxes is None:
            return 0.0
        if xy_margin is None:
            xy_margin = float(self.IMMERSIVE_IDLE_LOCK_SUPPORT_XY_MARGIN)
        if z_tolerance is None:
            z_tolerance = float(self.IMMERSIVE_IDLE_LOCK_SUPPORT_Z_TOL)
        if not torch.is_tensor(support_surface_boxes):
            support_surface_boxes = torch.as_tensor(
                support_surface_boxes,
                dtype=object_points.dtype,
                device=object_points.device,
            )
        else:
            support_surface_boxes = support_surface_boxes.to(
                device=object_points.device,
                dtype=object_points.dtype,
            )
        if support_surface_boxes.ndim != 3 or support_surface_boxes.shape[1:] != (2, 3):
            return 0.0
        if int(support_surface_boxes.shape[0]) == 0:
            return 0.0

        scene_up_np = np.asarray(scene_up, dtype=np.float32).reshape(-1)
        vertical_axis = int(np.argmax(np.abs(scene_up_np)))
        top_uses_min = bool(float(scene_up_np[vertical_axis]) < 0.0)
        lateral_axes = [axis for axis in range(3) if axis != vertical_axis]

        box_mins = support_surface_boxes[:, 0, :]
        box_maxs = support_surface_boxes[:, 1, :]
        point_count = int(object_points.shape[0])
        box_count = int(support_surface_boxes.shape[0])
        support_mask = torch.ones(
            (point_count, box_count),
            dtype=torch.bool,
            device=object_points.device,
        )
        for axis in lateral_axes:
            coords = object_points[:, axis].unsqueeze(1)
            support_mask &= coords >= (box_mins[:, axis].unsqueeze(0) - xy_margin)
            support_mask &= coords <= (box_maxs[:, axis].unsqueeze(0) + xy_margin)
        top_face = (
            box_mins[:, vertical_axis]
            if top_uses_min
            else box_maxs[:, vertical_axis]
        )
        support_mask &= (
            torch.abs(
                object_points[:, vertical_axis].unsqueeze(1) - top_face.unsqueeze(0)
            )
            <= z_tolerance
        )
        supported_points = support_mask.any(dim=1)
        return float(supported_points.to(dtype=torch.float32).mean().item())

    def _set_idle_lock_state(
        self,
        idle_lock_state,
        sim_state,
        object_points,
        *,
        action,
        reason,
        support_fraction,
    ):
        locked_state = self._build_idle_locked_sim_state(sim_state)
        idle_lock_state["active"] = True
        idle_lock_state["stable_frame_count"] = 0
        idle_lock_state["locked_state"] = locked_state
        idle_lock_state["last_object_points"] = object_points.detach().clone()
        idle_lock_state["last_object_points_prev"] = object_points.detach().clone()
        idle_lock_state["support_fraction"] = float(support_fraction)
        self._restore_sim_state(locked_state)
        support_center = (
            self._object_support_patch_center(object_points)
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )
        print(
            "[quest_display] "
            f"idle_lock={action} reason={reason} "
            f"support_fraction={float(support_fraction):.3f} "
            f"support_center={support_center}",
            flush=True,
        )
        return locked_state

    def _release_idle_lock_state(self, idle_lock_state, *, reason):
        if not bool(idle_lock_state.get("active", False)):
            return False
        idle_lock_state["active"] = False
        idle_lock_state["stable_frame_count"] = 0
        idle_lock_state["locked_state"] = None
        idle_lock_state["last_object_points"] = None
        idle_lock_state["last_object_points_prev"] = None
        idle_lock_state["support_fraction"] = 0.0
        print(
            "[quest_display] idle_lock=released "
            f"reason={reason}",
            flush=True,
        )
        return True

    def _update_idle_lock_state(
        self,
        idle_lock_state,
        sim_state,
        object_points,
        object_velocities,
        controller_interaction_state,
        idle_lock_case_profile,
        support_surface_boxes,
        scene_up,
    ):
        if (
            controller_interaction_state.get("left") is not None
            or controller_interaction_state.get("right") is not None
        ):
            self._release_idle_lock_state(
                idle_lock_state,
                reason="interaction_started",
            )
            return

        idle_lock_state["last_object_points"] = object_points.detach().clone()
        if bool(idle_lock_state.get("active", False)):
            idle_lock_state["last_object_points_prev"] = object_points.detach().clone()
            return

        support_fraction = self._scene_support_fraction(
            object_points,
            support_surface_boxes,
            scene_up,
        )
        previous_object_points = idle_lock_state.get("last_object_points_prev")
        if previous_object_points is None or previous_object_points.shape != object_points.shape:
            mean_frame_delta = float("inf")
        else:
            mean_frame_delta = float(
                torch.linalg.norm(
                    object_points - previous_object_points,
                    dim=1,
                ).mean().item()
            )
        max_speed = float(torch.linalg.norm(object_velocities, dim=1).max().item())

        stable = (
            max_speed <= float(idle_lock_case_profile["max_speed"])
            and mean_frame_delta
            <= float(idle_lock_case_profile["max_mean_frame_delta"])
            and support_fraction
            >= float(idle_lock_case_profile["min_support_fraction"])
        )
        if stable:
            idle_lock_state["stable_frame_count"] = int(
                idle_lock_state.get("stable_frame_count", 0)
            ) + 1
            if idle_lock_state["stable_frame_count"] >= int(
                idle_lock_case_profile["stable_frames_required"]
            ):
                self._set_idle_lock_state(
                    idle_lock_state,
                    sim_state,
                    object_points,
                    action="engaged",
                    reason="stably_idle",
                    support_fraction=support_fraction,
                )
        else:
            idle_lock_state["stable_frame_count"] = 0
        idle_lock_state["last_object_points_prev"] = object_points.detach().clone()

    def _build_anchor_def_from_seed(
        self,
        name,
        seed_idx,
        object_points,
        region_node_count,
    ):
        if seed_idx is None:
            return None
        region_indices = self._graph_region_from_seed(
            int(seed_idx),
            region_node_count,
            object_points,
        )
        if int(region_indices.numel()) == 0:
            return None
        region_points = object_points[region_indices]
        center_world = region_points.mean(dim=0)
        radius = torch.linalg.norm(
            region_points - center_world.unsqueeze(0), dim=1
        ).max()
        return {
            "name": name,
            "seed_index": int(seed_idx),
            "region_indices": region_indices,
            "rest_center_world": center_world,
            "rest_radius": float(radius.item()),
        }

    def _build_sloth_interaction_anchors(self, object_points, intrinsic, w2c):
        pixels, depth_valid = self._project_points_to_pixels(object_points, intrinsic, w2c)
        if not bool(depth_valid.any().item()):
            return []

        valid_pixels = pixels[depth_valid]
        center_pixel = valid_pixels.mean(dim=0)
        spread = valid_pixels.max(dim=0).values - valid_pixels.min(dim=0).values
        spread_x = float(spread[0].item())
        spread_y = float(spread[1].item())
        upper_mask = pixels[:, 1] <= center_pixel[1]
        lower_mask = pixels[:, 1] > center_pixel[1]
        left_mask = pixels[:, 0] <= center_pixel[0]
        right_mask = pixels[:, 0] > center_pixel[0]
        center_score = torch.linalg.norm(pixels - center_pixel.unsqueeze(0), dim=1)
        torso_half_width = max(spread_x * 0.18, 8.0)
        torso_half_height = max(spread_y * 0.16, 10.0)
        torso_center_mask = (
            (torch.abs(pixels[:, 0] - center_pixel[0]) <= torso_half_width)
            & (torch.abs(pixels[:, 1] - center_pixel[1]) <= torso_half_height)
        )

        anchor_specs = [
            ("left_leg", left_mask & lower_mask, center_score, True),
            ("right_leg", right_mask & lower_mask, center_score, True),
            ("left_arm", left_mask & upper_mask, center_score, True),
            ("right_arm", right_mask & upper_mask, center_score, True),
            ("torso_center", torso_center_mask, center_score, False),
        ]

        used_indices = set()
        anchors = []
        region_node_count = min(
            int(object_points.shape[0]), self.LIVE_CONTROLLER_PREDEFINED_ANCHOR_NODE_COUNT
        )
        for name, required_mask, score_values, prefer_largest in anchor_specs:
            seed_idx = self._pick_predefined_anchor_seed_index(
                depth_valid,
                required_mask,
                score_values,
                prefer_largest,
                used_indices,
            )
            if seed_idx is None:
                continue
            used_indices.add(seed_idx)
            anchor_def = self._build_anchor_def_from_seed(
                name,
                seed_idx,
                object_points,
                region_node_count,
            )
            if anchor_def is not None:
                anchors.append(anchor_def)
        return anchors

    def _graph_shortest_path_indices(self, start_idx, end_idx, object_points):
        if start_idx is None or end_idx is None:
            return []
        start_idx = int(start_idx)
        end_idx = int(end_idx)
        if start_idx == end_idx:
            return [start_idx]

        neighbors = self._object_graph_neighbors()
        points_np = object_points.detach().cpu().numpy()
        best_distance = {start_idx: 0.0}
        predecessors = {}
        visited = set()
        heap = [(0.0, start_idx)]

        while heap:
            distance, current = heapq.heappop(heap)
            if current in visited:
                continue
            visited.add(current)
            if current == end_idx:
                break
            current_point = points_np[current]
            for neighbor in neighbors[current]:
                if neighbor in visited:
                    continue
                edge_cost = float(np.linalg.norm(current_point - points_np[neighbor]))
                new_distance = distance + edge_cost
                if new_distance >= best_distance.get(neighbor, float("inf")):
                    continue
                best_distance[neighbor] = new_distance
                predecessors[neighbor] = current
                heapq.heappush(heap, (new_distance, neighbor))

        if end_idx not in best_distance:
            return []

        path = [end_idx]
        current = end_idx
        while current != start_idx:
            current = predecessors.get(current)
            if current is None:
                return []
            path.append(current)
        path.reverse()
        return path

    def _rope_midpoint_seed_index(self, endpoint_indices, object_points):
        start_idx, end_idx = int(endpoint_indices[0]), int(endpoint_indices[1])
        path = self._graph_shortest_path_indices(start_idx, end_idx, object_points)
        if len(path) >= 3:
            points_np = object_points.detach().cpu().numpy()
            cumulative = [0.0]
            for prev_idx, next_idx in zip(path[:-1], path[1:]):
                edge_length = float(
                    np.linalg.norm(points_np[next_idx] - points_np[prev_idx])
                )
                cumulative.append(cumulative[-1] + edge_length)
            half_length = cumulative[-1] * 0.5
            midpoint_path_index = min(
                range(len(path)),
                key=lambda idx: abs(cumulative[idx] - half_length),
            )
            midpoint_seed = int(path[midpoint_path_index])
            if midpoint_seed not in {start_idx, end_idx}:
                return midpoint_seed

        midpoint_world = (
            object_points[start_idx] + object_points[end_idx]
        ) * 0.5
        distances = torch.linalg.norm(
            object_points - midpoint_world.unsqueeze(0), dim=1
        )
        order = torch.argsort(distances)
        for candidate in order.tolist():
            if candidate not in {start_idx, end_idx}:
                return int(candidate)
        return int(order[0].item()) if int(order.numel()) > 0 else start_idx

    def _build_rope_interaction_anchors(self, object_points, intrinsic, w2c):
        if int(object_points.shape[0]) < 3:
            return []

        points_centered = object_points - object_points.mean(dim=0, keepdim=True)
        covariance = torch.matmul(points_centered.t(), points_centered)
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        rope_axis = eigenvectors[:, int(torch.argmax(eigenvalues).item())]
        projections = torch.matmul(points_centered, rope_axis)
        endpoint_order = torch.argsort(projections)
        left_seed = int(endpoint_order[0].item())
        right_seed = int(endpoint_order[-1].item())
        if left_seed == right_seed:
            return []

        midpoint_seed = self._rope_midpoint_seed_index(
            (left_seed, right_seed),
            object_points,
        )
        region_node_count = min(
            int(object_points.shape[0]),
            self.LIVE_CONTROLLER_PREDEFINED_ANCHOR_NODE_COUNT,
        )

        left_def = self._build_anchor_def_from_seed(
            "endpoint_a",
            left_seed,
            object_points,
            region_node_count,
        )
        right_def = self._build_anchor_def_from_seed(
            "endpoint_b",
            right_seed,
            object_points,
            region_node_count,
        )
        middle_def = self._build_anchor_def_from_seed(
            "middle",
            midpoint_seed,
            object_points,
            region_node_count,
        )
        if left_def is None or right_def is None or middle_def is None:
            return []
        return [left_def, middle_def, right_def]

    def _resolve_rope_endpoint_anchor_defs(self, anchor_defs, intrinsic, w2c):
        debug = {
            "case_name": self._interaction_anchor_case_name(),
            "endpoint_projected_x": {},
            "naming_valid": False,
            "fallback_used": False,
        }
        if not self._is_rope_family_case(debug["case_name"]):
            return anchor_defs, debug

        endpoint_defs = []
        other_defs = []
        for anchor_def in anchor_defs:
            if anchor_def["name"] in {"endpoint_a", "endpoint_b", "left_end", "right_end"}:
                endpoint_defs.append(anchor_def)
            else:
                other_defs.append(anchor_def)
        if len(endpoint_defs) != 2:
            debug["fallback_used"] = True
            return anchor_defs, debug

        endpoint_projected = []
        endpoint_valid = []
        for anchor_def in endpoint_defs:
            screen_x, _, valid = self._project_world_point_to_startup_screen_x(
                anchor_def["rest_center_world"],
                intrinsic,
                w2c,
            )
            debug["endpoint_projected_x"][anchor_def["name"]] = (
                float(screen_x) if valid and screen_x is not None else None
            )
            endpoint_projected.append(screen_x)
            endpoint_valid.append(bool(valid and screen_x is not None))

        if all(endpoint_valid):
            endpoint_order = sorted(
                range(2),
                key=lambda idx: (float(endpoint_projected[idx]), idx),
            )
            debug["naming_valid"] = True
        else:
            endpoint_order = [0, 1]
            debug["fallback_used"] = True

        resolved_left = dict(endpoint_defs[endpoint_order[0]])
        resolved_right = dict(endpoint_defs[endpoint_order[1]])
        resolved_left["name"] = "left_end"
        resolved_right["name"] = "right_end"
        return [resolved_left, *other_defs, resolved_right], debug

    def _build_case_interaction_anchors(self, object_points, intrinsic, w2c):
        if self._is_rope_family_case():
            return self._build_rope_interaction_anchors(object_points, intrinsic, w2c)
        return self._build_sloth_interaction_anchors(object_points, intrinsic, w2c)

    def _compute_predefined_interaction_anchor_states(self, anchor_defs, object_points):
        states = []
        for anchor in anchor_defs:
            region_points = object_points[anchor["region_indices"]]
            center_world = region_points.mean(dim=0)
            radius = torch.linalg.norm(
                region_points - center_world.unsqueeze(0), dim=1
            ).max()
            selection_radius = max(
                self.LIVE_CONTROLLER_PREDEFINED_ANCHOR_MIN_RADIUS,
                float(radius.item()) * self.LIVE_CONTROLLER_PREDEFINED_ANCHOR_RADIUS_SCALE,
            )
            states.append(
                {
                    "name": anchor["name"],
                    "region_indices": anchor["region_indices"],
                    "center_world": center_world,
                    "radius": float(radius.item()),
                    "selection_radius": selection_radius,
                }
            )
        return states

    def _resolve_case_default_controller_anchor_names(
        self,
        anchor_states,
        controller_source_anchor_centers,
        intrinsic,
        w2c,
    ):
        case_profile = self._live_controller_case_profile()
        default_anchor_names = dict(case_profile["default_anchor_names"])
        debug = {
            "case_name": case_profile["case_name"],
            "source_projected_x": {},
            "anchor_projected_x": {},
            "default_anchor_names": dict(default_anchor_names),
            "resolved_default_anchor_names": dict(default_anchor_names),
            "mapping_swapped": False,
            "mapping_valid": False,
        }

        if not self._is_rope_family_case(case_profile["case_name"]):
            return default_anchor_names, debug

        for source, center in zip(("left", "right"), controller_source_anchor_centers):
            screen_x, _, valid = self._project_world_point_to_startup_screen_x(
                center,
                intrinsic,
                w2c,
            )
            debug["source_projected_x"][source] = (
                float(screen_x) if valid and screen_x is not None else None
            )

        for anchor_name in ("left_end", "right_end"):
            anchor_state = self._anchor_state_by_name(anchor_states, anchor_name)
            if anchor_state is None:
                debug["anchor_projected_x"][anchor_name] = None
                continue
            screen_x, _, valid = self._project_world_point_to_startup_screen_x(
                anchor_state["center_world"],
                intrinsic,
                w2c,
            )
            debug["anchor_projected_x"][anchor_name] = (
                float(screen_x) if valid and screen_x is not None else None
            )

        left_source_x = debug["source_projected_x"].get("left")
        right_source_x = debug["source_projected_x"].get("right")
        left_anchor_x = debug["anchor_projected_x"].get("left_end")
        right_anchor_x = debug["anchor_projected_x"].get("right_end")
        if None in (left_source_x, right_source_x, left_anchor_x, right_anchor_x):
            return default_anchor_names, debug

        source_in_order = left_source_x <= right_source_x
        anchor_in_order = left_anchor_x <= right_anchor_x
        debug["mapping_valid"] = True
        debug["mapping_crossed"] = bool(source_in_order != anchor_in_order)

        debug["resolved_default_anchor_names"] = dict(default_anchor_names)
        return default_anchor_names, debug

    def _log_case_controller_anchor_mapping(self, prefix, mapping_debug):
        if not self._is_rope_family_case(mapping_debug.get("case_name")):
            return
        print(
            f"{prefix} rope runtime anchor mapping: "
            f"source_projected_x={mapping_debug['source_projected_x']} "
            f"anchor_projected_x={mapping_debug['anchor_projected_x']} "
            f"default_anchor_names={mapping_debug['default_anchor_names']} "
            f"resolved_default_anchor_names={mapping_debug['resolved_default_anchor_names']} "
            f"mapping_valid={int(bool(mapping_debug['mapping_valid']))} "
            f"crossed={int(bool(mapping_debug.get('mapping_crossed', False)))} "
            f"swap={int(bool(mapping_debug['mapping_swapped']))}",
            flush=True,
        )

    def _make_controller_anchor_preview_state_entry(self):
        return {
            "visible": False,
            "cycle_locked": False,
            "selected_rank_index": 0,
            "selected_anchor_name": None,
            "current_candidate_names": [],
            "current_selected_rank": None,
            "current_candidate_count": 0,
        }

    def _reset_controller_anchor_preview_state(self, preview_state, source):
        preview_state[source] = self._make_controller_anchor_preview_state_entry()

    def _clear_controller_anchor_preview_candidates(self, state):
        state["visible"] = False
        state["selected_anchor_name"] = None
        state["current_candidate_names"] = []
        state["current_selected_rank"] = None
        state["current_candidate_count"] = 0

    def _rank_predefined_interaction_anchors_for_hit(
        self,
        hit_world,
        anchor_states,
        require_selection_radius=False,
    ):
        if hit_world is None or not anchor_states:
            return []

        ranked = []
        for anchor_index, anchor in enumerate(anchor_states):
            distance = float(torch.linalg.norm(anchor["center_world"] - hit_world).item())
            if require_selection_radius and distance > anchor["selection_radius"]:
                continue
            ranked.append((distance, anchor_index, anchor))

        ranked.sort(key=lambda item: (item[0], item[1]))
        return [anchor for _, _, anchor in ranked]

    def _rank_predefined_interaction_anchors_for_ray(
        self,
        origin_world,
        direction_world,
        anchor_states,
    ):
        if origin_world is None or direction_world is None or not anchor_states:
            return []

        direction = direction_world / direction_world.norm().clamp_min(1e-6)
        ranked = []
        for anchor_index, anchor_state in enumerate(anchor_states):
            delta = anchor_state["center_world"] - origin_world
            along = float(torch.dot(delta, direction).item())
            along = max(along, 0.0)
            closest = origin_world + direction * along
            perpendicular = float(
                torch.linalg.norm(anchor_state["center_world"] - closest).item()
            )
            ranked.append((perpendicular, along, anchor_index, anchor_state))

        ranked.sort(key=lambda item: (item[0], item[1], item[2]))
        return [anchor for _, _, _, anchor in ranked]

    def _rank_predefined_interaction_anchors(
        self,
        hit_world,
        origin_world,
        direction_world,
        anchor_states,
        require_selection_radius=False,
    ):
        ranked = self._rank_predefined_interaction_anchors_for_hit(
            hit_world,
            anchor_states,
            require_selection_radius=require_selection_radius,
        )
        if ranked:
            return ranked
        return self._rank_predefined_interaction_anchors_for_ray(
            origin_world,
            direction_world,
            anchor_states,
        )

    def _select_predefined_interaction_anchor(
        self,
        hit_world,
        anchor_states,
        require_selection_radius=True,
    ):
        ranked = self._rank_predefined_interaction_anchors_for_hit(
            hit_world,
            anchor_states,
            require_selection_radius=require_selection_radius,
        )
        return None if not ranked else ranked[0]

    def _anchor_state_by_name(self, anchor_states, anchor_name):
        if anchor_name is None:
            return None
        for anchor in anchor_states:
            if anchor["name"] == anchor_name:
                return anchor
        return None

    def _controller_anchor_cycle_edge(
        self,
        source,
        controller_world,
        cycle_state_cache,
    ):
        pressed = bool(
            self._controller_input_available(controller_world, "anchor_cycle_available")
            and self._controller_input_pressed(controller_world, "anchor_cycle_pressed")
        )
        previous = bool(cycle_state_cache.get(source, False))
        cycle_state_cache[source] = pressed
        return pressed and not previous

    def _controller_input_available(self, controller_state, field_name):
        if controller_state is None:
            return False
        if isinstance(controller_state, dict):
            return bool(controller_state.get(field_name, False))
        return bool(getattr(controller_state, field_name, False))

    def _controller_input_pressed(self, controller_state, field_name):
        if controller_state is None:
            return False
        if isinstance(controller_state, dict):
            return bool(controller_state.get(field_name, False))
        return bool(getattr(controller_state, field_name, False))

    def _controller_snap_edge(
        self,
        source,
        controller_state,
        snap_state_cache,
    ):
        pressed = bool(
            self._controller_input_available(controller_state, "snap_assist_available")
            and self._controller_input_pressed(controller_state, "snap_assist_pressed")
        )
        previous = bool(snap_state_cache.get(source, False))
        snap_state_cache[source] = pressed
        return pressed and not previous

    def _controller_exit_button_label(self, source):
        return "left grip" if source == "left" else "right grip"

    def _other_controller_source(self, source):
        return "right" if source == "left" else "left"

    def _occupied_anchor_name_by_other_source(
        self,
        source,
        controller_interaction_state,
    ):
        if controller_interaction_state is None:
            return None, self._other_controller_source(source)
        other_source = self._other_controller_source(source)
        other_interaction_state = controller_interaction_state.get(other_source)
        if other_interaction_state is None:
            return None, other_source
        return other_interaction_state.get("anchor_name"), other_source

    def _anchor_is_occupied_by_other_source(
        self,
        source,
        anchor_name,
        controller_interaction_state,
        allow_current_anchor_name=None,
    ):
        if anchor_name is None:
            return False, self._other_controller_source(source)
        if (
            allow_current_anchor_name is not None
            and anchor_name == allow_current_anchor_name
        ):
            return False, self._other_controller_source(source)
        occupied_anchor_name, other_source = self._occupied_anchor_name_by_other_source(
            source,
            controller_interaction_state,
        )
        return occupied_anchor_name == anchor_name, other_source

    def _filter_available_predefined_interaction_anchors(
        self,
        source,
        anchor_states,
        controller_interaction_state,
        allow_current_anchor_name=None,
    ):
        if not anchor_states:
            return []
        available_anchor_states = []
        for anchor_state in anchor_states:
            occupied, _ = self._anchor_is_occupied_by_other_source(
                source,
                anchor_state.get("name"),
                controller_interaction_state,
                allow_current_anchor_name=allow_current_anchor_name,
            )
            if not occupied:
                available_anchor_states.append(anchor_state)
        return available_anchor_states

    def _update_controller_anchor_preview_state(
        self,
        source,
        controller_world,
        overlay,
        anchor_states,
        preview_state,
        cycle_edge,
        interaction_state,
        controller_interaction_state,
    ):
        state = preview_state[source]
        if interaction_state is not None:
            selected_anchor_name = interaction_state.get("anchor_name")
            state["visible"] = selected_anchor_name is not None
            state["selected_anchor_name"] = selected_anchor_name
            state["current_candidate_names"] = (
                [] if selected_anchor_name is None else [selected_anchor_name]
            )
            state["current_selected_rank"] = (
                None
                if selected_anchor_name is None
                else int(state.get("selected_rank_index", 0)) + 1
            )
            state["current_candidate_count"] = len(state["current_candidate_names"])
            return self._anchor_state_by_name(anchor_states, state["selected_anchor_name"])

        hit_world = None if overlay is None else overlay.get("hit_world")
        ray_origin_world, ray_direction_world = self._controller_world_ray_pose(
            controller_world
        )
        ranked_anchors_all = self._rank_predefined_interaction_anchors(
            hit_world,
            ray_origin_world,
            ray_direction_world,
            anchor_states,
            require_selection_radius=False,
        )
        if not ranked_anchors_all:
            self._clear_controller_anchor_preview_candidates(state)
            return None
        ranked_anchors = self._filter_available_predefined_interaction_anchors(
            source,
            ranked_anchors_all,
            controller_interaction_state,
        )
        if not ranked_anchors:
            state["visible"] = True
            state["cycle_locked"] = False
            state["selected_anchor_name"] = None
            state["selected_rank_index"] = 0
            state["current_candidate_names"] = [
                anchor["name"] for anchor in ranked_anchors_all
            ]
            state["current_selected_rank"] = None
            state["current_candidate_count"] = len(ranked_anchors_all)
            return None

        candidate_count = len(ranked_anchors)
        cycle_locked = bool(state.get("cycle_locked", False))
        latched_anchor_name = state.get("selected_anchor_name")
        ranked_anchor_names = [anchor["name"] for anchor in ranked_anchors]
        state["current_candidate_names"] = ranked_anchor_names
        state["current_candidate_count"] = candidate_count
        current_rank_index_by_name = {
            anchor["name"]: index for index, anchor in enumerate(ranked_anchors)
        }
        current_latched_rank_index = (
            current_rank_index_by_name.get(latched_anchor_name)
            if latched_anchor_name is not None
            else None
        )
        selected_anchor = None

        if cycle_edge:
            if candidate_count < 2:
                print(
                    "[live_openxr_controller] "
                    f"{source} anchor_cycle_target unavailable reason=no_alternate_target "
                    f"candidates={candidate_count}",
                    flush=True,
                )
            elif not cycle_locked:
                cycle_locked = True
                selected_anchor = ranked_anchors[1]
            else:
                non_nearest_anchors = ranked_anchors[1:]
                if current_latched_rank_index is None or current_latched_rank_index == 0:
                    next_non_nearest_index = 0
                else:
                    next_non_nearest_index = (
                        (current_latched_rank_index - 1 + 1) % len(non_nearest_anchors)
                    )
                selected_anchor = non_nearest_anchors[next_non_nearest_index]

            if selected_anchor is not None:
                latched_anchor_name = selected_anchor["name"]
                current_latched_rank_index = current_rank_index_by_name.get(
                    latched_anchor_name
                )
            elif latched_anchor_name is not None:
                selected_anchor = self._anchor_state_by_name(anchor_states, latched_anchor_name)
        else:
            if not cycle_locked:
                selected_anchor = ranked_anchors[0]
                latched_anchor_name = selected_anchor["name"]
                current_latched_rank_index = 0
            else:
                selected_anchor = self._anchor_state_by_name(anchor_states, latched_anchor_name)
                if selected_anchor is None and candidate_count >= 2:
                    selected_anchor = ranked_anchors[1]
                    latched_anchor_name = selected_anchor["name"]
                    current_latched_rank_index = current_rank_index_by_name.get(
                        latched_anchor_name
                    )

        if selected_anchor is None:
            self._clear_controller_anchor_preview_candidates(state)
            state["cycle_locked"] = cycle_locked
            if cycle_locked and latched_anchor_name is not None:
                state["selected_anchor_name"] = latched_anchor_name
            return None

        selected_rank_index = (
            0 if current_latched_rank_index is None else int(current_latched_rank_index)
        )
        state["cycle_locked"] = cycle_locked
        state["selected_rank_index"] = selected_rank_index
        state["selected_anchor_name"] = selected_anchor["name"]
        state["current_selected_rank"] = selected_rank_index + 1
        state["visible"] = True

        if cycle_edge and candidate_count >= 2:
            print(
                "[live_openxr_controller] "
                f"{source} anchor_cycle_target selected={selected_anchor['name']} "
                f"rank={selected_rank_index + 1}/{candidate_count}",
                flush=True,
            )
        return selected_anchor

    def _sample_anchor_block_point_indices(self, region_points, sample_count):
        point_count = int(region_points.shape[0])
        if point_count <= 0 or sample_count <= 0:
            return torch.empty(0, dtype=torch.long, device=region_points.device)

        center = region_points.mean(dim=0, keepdim=True)
        first_index = int(
            torch.argmax(torch.linalg.norm(region_points - center, dim=1)).item()
        )
        selected = [first_index]
        min_distance = torch.linalg.norm(
            region_points - region_points[first_index].unsqueeze(0), dim=1
        )
        while len(selected) < min(sample_count, point_count):
            next_index = int(torch.argmax(min_distance).item())
            selected.append(next_index)
            candidate_distance = torch.linalg.norm(
                region_points - region_points[next_index].unsqueeze(0), dim=1
            )
            min_distance = torch.minimum(min_distance, candidate_distance)

        if len(selected) < sample_count:
            repeats = list(selected)
            repeat_cursor = 0
            while len(selected) < sample_count:
                selected.append(repeats[repeat_cursor % len(repeats)])
                repeat_cursor += 1

        return torch.as_tensor(
            selected[:sample_count], dtype=torch.long, device=region_points.device
        )

    def _resample_controller_stiffness_template(self, spring_values, target_count):
        if target_count <= 0:
            return torch.empty(0, dtype=torch.float32, device=cfg.device)

        if spring_values is None:
            return torch.ones(target_count, dtype=torch.float32, device=cfg.device)

        values = spring_values.to(device=cfg.device, dtype=torch.float32).flatten()
        if values.numel() == 0:
            return torch.ones(target_count, dtype=torch.float32, device=cfg.device)
        if values.numel() == 1:
            return values.repeat(target_count)
        if values.numel() == target_count:
            return values.clone()

        values = torch.sort(values).values
        sample_positions = torch.linspace(
            0.0,
            float(values.numel() - 1),
            target_count,
            dtype=torch.float32,
            device=cfg.device,
        )
        lower = sample_positions.floor().long()
        upper = sample_positions.ceil().long()
        alpha = sample_positions - lower.to(dtype=torch.float32)
        return values[lower] * (1.0 - alpha) + values[upper] * alpha

    def _build_predefined_controller_block_runtime(
        self,
        rest_object_points,
        recorded_base_target,
        original_springs,
        original_rest_lengths,
        original_spring_y,
        controller_source_masks,
        controller_predefined_anchor_defs,
    ):
        original_source_meta = self._build_controller_attachment_metadata(
            original_springs,
            original_rest_lengths,
            self.num_all_points,
            controller_source_masks,
        )
        anchor_names = [anchor["name"] for anchor in controller_predefined_anchor_defs]
        controller_block_count = self.LIVE_CONTROLLER_BLOCK_POINT_COUNT
        springs_per_block = (
            self.LIVE_CONTROLLER_BLOCK_POINT_COUNT
            * self.LIVE_CONTROLLER_BLOCK_SPRINGS_PER_POINT
        )
        object_center = rest_object_points.mean(dim=0)
        controller_rest_blocks = []
        templates = {"left": {}, "right": {}}

        object_springs = original_springs[: self.num_object_springs].clone()
        object_rest_lengths = original_rest_lengths[: self.num_object_springs].clone()
        spring_y = original_spring_y[: self.num_object_springs].clone()
        source_canonical_offsets = {}
        for source, mask in zip(("left", "right"), controller_source_masks):
            source_points = recorded_base_target[mask]
            sample_local = self._sample_anchor_block_point_indices(
                source_points,
                self.LIVE_CONTROLLER_BLOCK_POINT_COUNT,
            )
            sampled_points = source_points[sample_local]
            source_canonical_offsets[source] = sampled_points - sampled_points.mean(
                dim=0, keepdim=True
            )

        controller_point_offset = 0
        for source in ("left", "right"):
            source_meta = original_source_meta[source]
            source_spring_y = self._resample_controller_stiffness_template(
                original_spring_y[source_meta["spring_indices"]],
                springs_per_block,
            )
            for anchor_def in controller_predefined_anchor_defs:
                region_indices = anchor_def["region_indices"]
                if region_indices.numel() == 0:
                    continue
                region_points = rest_object_points[region_indices]
                anchor_center = anchor_def["rest_center_world"]
                outward = anchor_center - object_center
                outward_norm = float(torch.linalg.norm(outward).item())
                if outward_norm < 1e-6:
                    outward = torch.tensor(
                        [0.0, 0.0, 1.0], dtype=torch.float32, device=cfg.device
                    )
                else:
                    outward = outward / outward_norm
                offset_distance = max(
                    anchor_def["rest_radius"] * self.LIVE_CONTROLLER_BLOCK_OFFSET_SCALE,
                    self.LIVE_CONTROLLER_BLOCK_MIN_OFFSET,
                )
                block_rest_points = (
                    anchor_center.unsqueeze(0)
                    + source_canonical_offsets[source]
                    + outward.unsqueeze(0) * offset_distance
                )
                block_point_indices = torch.arange(
                    controller_point_offset,
                    controller_point_offset + controller_block_count,
                    dtype=torch.long,
                    device=cfg.device,
                )
                controller_rest_blocks.append(block_rest_points)
                controller_point_offset += controller_block_count

                springs = []
                rest_lengths = []
                spring_point_offsets = []
                block_object_indices = []
                for point_offset, control_point in enumerate(block_rest_points):
                    local_distance = torch.linalg.norm(
                        region_points - control_point.unsqueeze(0),
                        dim=1,
                    )
                    nearest_count = min(
                        self.LIVE_CONTROLLER_BLOCK_SPRINGS_PER_POINT,
                        int(region_indices.numel()),
                    )
                    nearest_local = torch.topk(
                        local_distance,
                        k=nearest_count,
                        largest=False,
                    ).indices
                    if nearest_count < self.LIVE_CONTROLLER_BLOCK_SPRINGS_PER_POINT:
                        repeats = (
                            self.LIVE_CONTROLLER_BLOCK_SPRINGS_PER_POINT
                            + nearest_count
                            - 1
                        ) // nearest_count
                        nearest_local = nearest_local.repeat(repeats)[
                            : self.LIVE_CONTROLLER_BLOCK_SPRINGS_PER_POINT
                        ]
                    selected_object_indices = region_indices[nearest_local]
                    block_object_indices.extend(
                        int(object_idx.item()) for object_idx in selected_object_indices
                    )
                    for object_idx in selected_object_indices:
                        springs.append(
                            [
                                self.num_all_points
                                + int(block_point_indices[point_offset].item()),
                                int(object_idx.item()),
                            ]
                        )
                        rest_lengths.append(
                            torch.linalg.norm(
                                control_point - rest_object_points[int(object_idx.item())]
                            ).clamp_min(1e-4)
                        )
                        spring_point_offsets.append(point_offset)

                templates[source][anchor_def["name"]] = {
                    "anchor_name": anchor_def["name"],
                    "block_point_indices": block_point_indices,
                    "source_template_offsets": block_rest_points - anchor_center.unsqueeze(0),
                    "springs": torch.as_tensor(
                        springs,
                        dtype=object_springs.dtype,
                        device=cfg.device,
                    ),
                    "rest_lengths": torch.stack(rest_lengths, dim=0).to(
                        device=cfg.device, dtype=torch.float32
                    ),
                    "spring_point_offsets": torch.as_tensor(
                        spring_point_offsets,
                        dtype=torch.long,
                        device=cfg.device,
                    ),
                    "spring_y": source_spring_y.clone(),
                    "selected_object_indices": region_indices.clone(),
                    "attach_center_rest": anchor_center.clone(),
                    "attach_radius_rest": anchor_def["rest_radius"],
                }

        controller_rest_points = torch.cat(controller_rest_blocks, dim=0)
        source_runtime = {}
        active_controller_springs = []
        active_controller_rest_lengths = []
        active_controller_spring_y = []
        spring_cursor = int(object_springs.shape[0])
        for source in ("left", "right"):
            default_template = templates[source][anchor_names[0]]
            spring_count = int(default_template["springs"].shape[0])
            active_indices = torch.arange(
                spring_cursor,
                spring_cursor + spring_count,
                dtype=torch.long,
                device=cfg.device,
            )
            active_controller_springs.append(default_template["springs"])
            active_controller_rest_lengths.append(default_template["rest_lengths"])
            active_controller_spring_y.append(
                torch.zeros_like(default_template["spring_y"])
            )
            source_runtime[source] = {
                "spring_indices": active_indices,
                "inactive_spring_y": torch.zeros_like(default_template["spring_y"]),
            }
            spring_cursor += spring_count

        if active_controller_springs:
            object_springs = torch.cat([object_springs] + active_controller_springs, dim=0)
            object_rest_lengths = torch.cat(
                [object_rest_lengths] + active_controller_rest_lengths, dim=0
            )
            spring_y = torch.cat([spring_y] + active_controller_spring_y, dim=0)

        return {
            "controller_rest_points": controller_rest_points,
            "templates": templates,
            "source_runtime": source_runtime,
            "init_springs": object_springs,
            "init_rest_lengths": object_rest_lengths,
            "spring_y": spring_y,
            "controller_points_per_source": (
                len(controller_predefined_anchor_defs) * self.LIVE_CONTROLLER_BLOCK_POINT_COUNT
            ),
            "controller_points_per_anchor": self.LIVE_CONTROLLER_BLOCK_POINT_COUNT,
            "controller_springs_per_source": springs_per_block,
        }

    def _build_controller_attachment_metadata(
        self,
        init_springs,
        init_rest_lengths,
        num_object_points,
        controller_source_masks,
    ):
        num_controller_points = int(controller_source_masks[0].numel())
        point_spring_indices = [[] for _ in range(num_controller_points)]
        controller_spring_start = int(self.num_object_springs)
        springs_np = init_springs.detach().cpu().numpy()
        for spring_idx in range(controller_spring_start, springs_np.shape[0]):
            endpoint0, endpoint1 = springs_np[spring_idx]
            ctrl_idx = (endpoint0 if endpoint0 >= num_object_points else endpoint1) - num_object_points
            point_spring_indices[int(ctrl_idx)].append(int(spring_idx))

        metadata = {}
        for source, mask in zip(("left", "right"), controller_source_masks):
            point_indices = torch.nonzero(mask, as_tuple=False).squeeze(1).tolist()
            spring_indices = []
            point_positions = {}
            for point_idx in point_indices:
                local_spring_indices = point_spring_indices[int(point_idx)]
                point_positions[int(point_idx)] = local_spring_indices
                spring_indices.extend(local_spring_indices)
            spring_indices = sorted(spring_indices)
            spring_indices_torch = torch.as_tensor(
                spring_indices, dtype=torch.long, device=cfg.device
            )
            local_lookup = {spring_idx: pos for pos, spring_idx in enumerate(spring_indices)}
            spring_point_offsets = [-1] * len(spring_indices)
            spring_capable_point_offsets = []
            for point_offset, point_idx in enumerate(point_indices):
                local_positions = [
                    local_lookup[spring_idx] for spring_idx in point_positions[point_idx]
                ]
                if local_positions:
                    spring_capable_point_offsets.append(point_offset)
                for local_pos in local_positions:
                    spring_point_offsets[local_pos] = point_offset
            metadata[source] = {
                "point_indices": point_indices,
                "point_spring_positions": {
                    point_idx: [local_lookup[spring_idx] for spring_idx in point_positions[point_idx]]
                    for point_idx in point_indices
                },
                "spring_indices": spring_indices_torch,
                "template_springs": init_springs[spring_indices_torch].clone(),
                "template_rest_lengths": init_rest_lengths[spring_indices_torch].clone(),
                "spring_point_offsets": torch.as_tensor(
                    spring_point_offsets, dtype=torch.long, device=cfg.device
                ),
                "spring_capable_point_offsets": torch.as_tensor(
                    spring_capable_point_offsets, dtype=torch.long, device=cfg.device
                ),
            }
        return metadata

    def _build_two_point_live_controller_runtime(
        self,
        rest_object_points,
        original_spring_y,
        original_controller_source_masks,
        original_controller_source_anchor_centers,
        controller_predefined_anchor_defs,
        default_anchor_names=None,
    ):
        original_source_meta = self._build_controller_attachment_metadata(
            self.init_springs,
            self.init_rest_lengths,
            self.num_all_points,
            original_controller_source_masks,
        )
        controller_rest_points = torch.stack(
            [
                original_controller_source_anchor_centers[0],
                original_controller_source_anchor_centers[1],
            ],
            dim=0,
        ).to(device=cfg.device, dtype=torch.float32)
        controller_source_masks = [
            torch.tensor([True, False], dtype=torch.bool, device=cfg.device),
            torch.tensor([False, True], dtype=torch.bool, device=cfg.device),
        ]
        controller_source_anchor_centers = [
            controller_rest_points[0].clone(),
            controller_rest_points[1].clone(),
        ]

        object_springs = self.init_springs[: self.num_object_springs].clone()
        object_rest_lengths = self.init_rest_lengths[: self.num_object_springs].clone()
        spring_y = original_spring_y[: self.num_object_springs].clone()

        anchor_states = self._compute_predefined_interaction_anchor_states(
            controller_predefined_anchor_defs,
            rest_object_points,
        )
        source_runtime = {}
        controller_springs = []
        controller_rest_lengths = []
        controller_spring_y = []
        spring_cursor = int(object_springs.shape[0])
        case_profile = self._live_controller_case_profile()
        resolved_default_anchor_names = (
            dict(default_anchor_names)
            if default_anchor_names is not None
            else dict(case_profile["default_anchor_names"])
        )

        for source in ("left", "right"):
            source_index = self._controller_source_index(source)
            preferred_anchor_name = resolved_default_anchor_names.get(source)
            default_anchor = (
                self._anchor_state_by_name(anchor_states, preferred_anchor_name)
                if preferred_anchor_name is not None
                else None
            )
            if default_anchor is None:
                default_anchor = self._select_predefined_interaction_anchor(
                    controller_source_anchor_centers[source_index],
                    anchor_states,
                    require_selection_radius=False,
                )
            if default_anchor is None:
                raise ValueError(f"No predefined anchor available for {source} controller")

            object_indices = default_anchor["region_indices"].clone()
            controller_index = source_index
            endpoint0 = torch.full(
                (int(object_indices.numel()),),
                self.num_all_points + controller_index,
                dtype=object_springs.dtype,
                device=cfg.device,
            )
            springs = torch.stack(
                [endpoint0, object_indices.to(dtype=object_springs.dtype)],
                dim=1,
            )
            rest_lengths = torch.linalg.norm(
                rest_object_points[object_indices]
                - controller_source_anchor_centers[source_index].unsqueeze(0),
                dim=1,
            ).clamp_min(1e-4).clamp_max(self.LIVE_CONTROLLER_ATTACH_MAX_REST_LENGTH)
            source_spring_y = self._resample_controller_stiffness_template(
                original_spring_y[original_source_meta[source]["spring_indices"]],
                int(object_indices.numel()),
            )
            active_indices = torch.arange(
                spring_cursor,
                spring_cursor + int(object_indices.numel()),
                dtype=torch.long,
                device=cfg.device,
            )
            spring_cursor += int(object_indices.numel())
            source_runtime[source] = {
                "spring_indices": active_indices,
                "inactive_spring_y": torch.zeros_like(source_spring_y),
                "template_springs": springs.clone(),
                "template_rest_lengths": rest_lengths.clone(),
                "spring_y_template": source_spring_y.clone(),
                "point_indices": [controller_index],
            }
            controller_springs.append(springs)
            controller_rest_lengths.append(rest_lengths)
            controller_spring_y.append(torch.zeros_like(source_spring_y))

        if controller_springs:
            object_springs = torch.cat([object_springs] + controller_springs, dim=0)
            object_rest_lengths = torch.cat(
                [object_rest_lengths] + controller_rest_lengths, dim=0
            )
            spring_y = torch.cat([spring_y] + controller_spring_y, dim=0)

        return {
            "controller_rest_points": controller_rest_points,
            "controller_source_masks": controller_source_masks,
            "controller_source_anchor_centers": controller_source_anchor_centers,
            "source_runtime": source_runtime,
            "init_springs": object_springs,
            "init_rest_lengths": object_rest_lengths,
            "spring_y": spring_y,
        }

    def _build_single_point_controller_attachment_candidate(
        self,
        source,
        anchor_state,
        object_points,
        controller_attachment_metadata,
        hit_world=None,
    ):
        if anchor_state is None:
            return None
        source_index = self._controller_source_index(source)
        source_meta = controller_attachment_metadata[source]
        object_indices = anchor_state["region_indices"].clone()
        if object_indices.numel() == 0:
            return None

        springs_dtype = source_meta["template_springs"].dtype
        controller_endpoint = torch.full(
            (int(object_indices.numel()),),
            self.num_all_points + source_index,
            dtype=springs_dtype,
            device=cfg.device,
        )
        springs = torch.stack(
            [controller_endpoint, object_indices.to(dtype=springs_dtype)],
            dim=1,
        )
        attach_center_world = anchor_state["center_world"].clone()
        attach_anchor_world = attach_center_world.clone()
        if hit_world is not None:
            region_points = object_points[object_indices]
            nearest_idx = torch.argmin(
                torch.linalg.norm(region_points - hit_world.unsqueeze(0), dim=1)
            )
            attach_anchor_world = region_points[nearest_idx].clone()
        rest_lengths = torch.linalg.norm(
            object_points[object_indices] - attach_anchor_world.unsqueeze(0),
            dim=1,
        ).clamp_min(1e-4).clamp_max(self.LIVE_CONTROLLER_ATTACH_MAX_REST_LENGTH)
        hit_to_anchor_distance = None
        if hit_world is not None:
            hit_to_anchor_distance = float(
                torch.linalg.norm(attach_anchor_world - hit_world).item()
            )
        return {
            "anchor_name": anchor_state["name"],
            "springs": springs,
            "rest_lengths": rest_lengths,
            "spring_y": source_meta["spring_y_template"].clone(),
            "attach_center_world": attach_center_world,
            "attach_node_count": int(object_indices.numel()),
            "attach_radius": float(anchor_state["radius"]),
            "attach_anchor_world": attach_anchor_world,
            "source_template_offsets": torch.zeros(
                (1, 3), dtype=torch.float32, device=cfg.device
            ),
            "selected_object_indices": object_indices.clone(),
            "target_point_indices": torch.tensor(
                [source_index], dtype=torch.long, device=cfg.device
            ),
            "hit_to_anchor_distance": hit_to_anchor_distance,
        }

    def _build_predefined_controller_anchor_templates(
        self,
        recorded_base_target,
        rest_object_points,
        controller_source_masks,
        controller_source_anchor_centers,
        controller_attachment_metadata,
        controller_predefined_anchor_defs,
    ):
        templates = {"left": {}, "right": {}}
        for source in ("left", "right"):
            source_index = self._controller_source_index(source)
            source_mask = controller_source_masks[source_index]
            source_meta = controller_attachment_metadata[source]
            source_point_count = int(source_mask.sum().item())
            source_template_offsets = torch.zeros(
                (source_point_count, 3), dtype=torch.float32, device=cfg.device
            )
            spring_capable_offsets = source_meta["spring_capable_point_offsets"]
            if spring_capable_offsets.numel() == 0:
                continue

            for anchor_def in controller_predefined_anchor_defs:
                candidate_pool = anchor_def["region_indices"]
                if candidate_pool.numel() == 0:
                    continue

                anchor_center = anchor_def["rest_center_world"]
                source_offsets = source_template_offsets.clone()
                template_springs = source_meta["template_springs"].clone()
                selected_object_indices = []
                point_assigned_nodes = {}
                local_distances = torch.linalg.norm(
                    rest_object_points[candidate_pool] - anchor_center.unsqueeze(0), dim=1
                )
                ordered_pool = candidate_pool[torch.argsort(local_distances)]
                pool_cursor = 0
                for point_offset in spring_capable_offsets.tolist():
                    point_idx = source_meta["point_indices"][point_offset]
                    spring_positions = source_meta["point_spring_positions"][point_idx]
                    if not spring_positions:
                        continue
                    spring_count = len(spring_positions)
                    if ordered_pool.numel() == 0:
                        break
                    assigned = []
                    for _ in range(spring_count):
                        assigned.append(int(ordered_pool[pool_cursor % ordered_pool.numel()].item()))
                        pool_cursor += 1
                    point_assigned_nodes[point_offset] = assigned
                    selected_object_indices.extend(assigned)
                    assigned_tensor = torch.as_tensor(
                        assigned, dtype=template_springs.dtype, device=cfg.device
                    )
                    for spring_position, object_idx in zip(spring_positions, assigned_tensor):
                        if int(template_springs[spring_position, 0].item()) < int(self.num_all_points):
                            template_springs[spring_position, 0] = object_idx
                        else:
                            template_springs[spring_position, 1] = object_idx

                if not selected_object_indices:
                    continue

                selected_object_indices = torch.as_tensor(
                    selected_object_indices, dtype=torch.long, device=cfg.device
                )
                unique_object_indices = torch.unique(selected_object_indices)
                attach_center_rest = rest_object_points[unique_object_indices].mean(dim=0)
                attach_radius_rest = torch.linalg.norm(
                    rest_object_points[unique_object_indices] - attach_center_rest.unsqueeze(0),
                    dim=1,
                ).max()
                for point_offset, assigned in point_assigned_nodes.items():
                    assigned_tensor = torch.as_tensor(
                        assigned, dtype=torch.long, device=cfg.device
                    )
                    source_offsets[point_offset] = (
                        rest_object_points[assigned_tensor].mean(dim=0) - attach_center_rest
                    )

                templates[source][anchor_def["name"]] = {
                    "anchor_name": anchor_def["name"],
                    "source_template_offsets": source_offsets,
                    "template_springs": template_springs,
                    "selected_object_indices": unique_object_indices,
                    "attach_center_rest": attach_center_rest,
                    "attach_radius_rest": float(attach_radius_rest.item()),
                }
        return templates

    def _instantiate_predefined_controller_anchor_template(
        self,
        source,
        anchor_state,
        object_points,
        controller_anchor_templates,
        controller_attachment_metadata,
    ):
        if anchor_state is None:
            return None

        source_templates = controller_anchor_templates.get(source)
        if source_templates is None:
            return None
        template = source_templates.get(anchor_state["name"])
        if template is None:
            return None

        if "block_point_indices" in template:
            unique_object_indices = template["selected_object_indices"]
            if unique_object_indices.numel() == 0:
                return None
            attach_center_world = object_points[unique_object_indices].mean(dim=0)
            attach_radius = torch.linalg.norm(
                object_points[unique_object_indices] - attach_center_world.unsqueeze(0),
                dim=1,
            ).max()
            source_target_points = (
                attach_center_world.unsqueeze(0) + template["source_template_offsets"]
            )
            spring_control_points = source_target_points[template["spring_point_offsets"]]
            endpoint0 = template["springs"][:, 0]
            endpoint1 = template["springs"][:, 1]
            object_indices = torch.where(
                endpoint0 < self.num_all_points, endpoint0, endpoint1
            ).long()
            rest_lengths = torch.linalg.norm(
                spring_control_points - object_points[object_indices], dim=1
            ).clamp_min(1e-4).clamp_max(self.LIVE_CONTROLLER_ATTACH_MAX_REST_LENGTH)
            return {
                "anchor_name": template["anchor_name"],
                "springs": template["springs"].clone(),
                "rest_lengths": rest_lengths,
                "spring_y": template["spring_y"].clone(),
                "attach_center_world": attach_center_world,
                "attach_node_count": int(unique_object_indices.numel()),
                "attach_radius": float(attach_radius.item()),
                "attach_anchor_world": attach_center_world.clone(),
                "source_template_offsets": template["source_template_offsets"].clone(),
                "selected_object_indices": unique_object_indices.clone(),
                "target_point_indices": template["block_point_indices"].clone(),
            }

        source_meta = controller_attachment_metadata[source]
        source_template_offsets = template["source_template_offsets"]
        template_springs = template["template_springs"].clone()
        unique_object_indices = template["selected_object_indices"]
        if unique_object_indices.numel() == 0:
            return None

        attach_center_world = object_points[unique_object_indices].mean(dim=0)
        attach_radius = torch.linalg.norm(
            object_points[unique_object_indices] - attach_center_world.unsqueeze(0), dim=1
        ).max()
        source_target_points = attach_center_world.unsqueeze(0) + source_template_offsets

        spring_point_offsets = source_meta["spring_point_offsets"]
        spring_control_points = source_target_points[spring_point_offsets]
        endpoint0 = template_springs[:, 0]
        endpoint1 = template_springs[:, 1]
        object_indices = torch.where(
            endpoint0 < self.num_all_points, endpoint0, endpoint1
        ).long()
        rest_lengths = torch.linalg.norm(
            spring_control_points - object_points[object_indices], dim=1
        ).clamp_min(1e-4).clamp_max(self.LIVE_CONTROLLER_ATTACH_MAX_REST_LENGTH)

        return {
            "anchor_name": anchor_state["name"],
            "springs": template_springs,
            "rest_lengths": rest_lengths,
            "spring_y": source_meta["spring_y_template"].clone(),
            "attach_center_world": attach_center_world,
            "attach_node_count": int(unique_object_indices.numel()),
            "attach_radius": float(attach_radius.item()),
            "attach_anchor_world": attach_center_world.clone(),
            "source_template_offsets": source_template_offsets.clone(),
            "selected_object_indices": unique_object_indices.clone(),
            "target_point_indices": torch.as_tensor(
                source_meta["point_indices"],
                dtype=torch.long,
                device=cfg.device,
            ),
        }

    def _refine_attachment_candidate_anchor_world(
        self,
        remap_candidate,
        object_points,
        hit_world=None,
    ):
        if remap_candidate is None or hit_world is None:
            return remap_candidate

        selected_object_indices = remap_candidate.get("selected_object_indices")
        if selected_object_indices is None or int(selected_object_indices.numel()) <= 0:
            return remap_candidate

        selected_points = object_points[selected_object_indices]
        nearest_offset = torch.argmin(
            torch.linalg.norm(selected_points - hit_world.unsqueeze(0), dim=1)
        )
        refined_candidate = dict(remap_candidate)
        refined_candidate["attach_anchor_world"] = selected_points[nearest_offset].clone()
        refined_candidate["hit_to_anchor_distance"] = float(
            torch.linalg.norm(refined_candidate["attach_anchor_world"] - hit_world).item()
        )
        return refined_candidate

    def _select_predefined_interaction_anchor_for_ray(
        self,
        origin_world,
        direction_world,
        anchor_states,
    ):
        ranked = self._rank_predefined_interaction_anchors_for_ray(
            origin_world,
            direction_world,
            anchor_states,
        )
        return None if not ranked else ranked[0]

    def _select_multi_points_seed_index(
        self,
        hit_world,
        ray_direction,
        candidate_pool,
        object_points,
    ):
        if (
            hit_world is None
            or ray_direction is None
            or candidate_pool is None
            or candidate_pool.numel() == 0
        ):
            return None, None

        direction = ray_direction / ray_direction.norm().clamp_min(1e-6)
        candidate_points = object_points[candidate_pool]
        rel = candidate_points - hit_world.unsqueeze(0)
        depth = torch.sum(rel * direction.unsqueeze(0), dim=1)
        lateral = torch.linalg.norm(
            rel - depth.unsqueeze(1) * direction.unsqueeze(0), dim=1
        )
        visible_mask = depth <= self.LIVE_CONTROLLER_MULTI_POINTS_BACK_DEPTH_THRESHOLD
        if not bool(visible_mask.any().item()):
            visible_mask = (
                depth
                <= depth.min() + self.LIVE_CONTROLLER_MULTI_POINTS_BACK_DEPTH_THRESHOLD
            )
        if not bool(visible_mask.any().item()):
            return None, None

        visible_indices = candidate_pool[visible_mask]
        visible_depth = depth[visible_mask]
        visible_lateral = lateral[visible_mask]
        score = visible_lateral + visible_depth.clamp_min(0.0) * (
            self.LIVE_CONTROLLER_MULTI_POINTS_BACK_PENALTY
        )
        best_pos = int(torch.argmin(score).item())
        seed_idx = int(visible_indices[best_pos].item())
        debug = {
            "seed_depth": float(visible_depth[best_pos].item()),
            "seed_lateral": float(visible_lateral[best_pos].item()),
            "candidate_pool_size": int(candidate_pool.numel()),
            "visible_pool_size": int(visible_indices.numel()),
        }
        return seed_idx, debug

    def _multi_points_region_from_seed(
        self,
        seed_idx,
        candidate_pool,
        hit_world,
        ray_direction,
        object_points,
        region_node_count,
    ):
        if (
            seed_idx is None
            or candidate_pool is None
            or candidate_pool.numel() == 0
            or region_node_count <= 0
        ):
            return (
                torch.empty(0, dtype=torch.long, device=object_points.device),
                torch.empty(0, dtype=torch.long, device=object_points.device),
                None,
            )

        neighbors = self._object_graph_neighbors()
        points_np = object_points.detach().cpu().numpy()
        hit_np = hit_world.detach().cpu().numpy()
        ray_np = (
            ray_direction / ray_direction.norm().clamp_min(1e-6)
        ).detach().cpu().numpy()
        candidate_pool_np = candidate_pool.detach().cpu().numpy().astype(np.int64)
        candidate_mask = np.zeros(self.num_all_points, dtype=np.bool_)
        candidate_mask[candidate_pool_np] = True
        threshold = self.LIVE_CONTROLLER_MULTI_POINTS_BACK_DEPTH_THRESHOLD
        fetch_limit = min(
            self.num_all_points,
            max(region_node_count, region_node_count * self.LIVE_CONTROLLER_MULTI_POINTS_FETCH_SCALE),
        )

        best_distance = {int(seed_idx): 0.0}
        visited = set()
        heap = [(0.0, int(seed_idx))]
        ordered = []
        while heap and len(ordered) < fetch_limit:
            distance, current = heapq.heappop(heap)
            if current in visited:
                continue
            visited.add(current)
            current_point = points_np[current]
            depth = float(np.dot(current_point - hit_np, ray_np))
            if candidate_mask[current] and depth <= threshold:
                ordered.append(current)

            for neighbor in neighbors[current]:
                if neighbor in visited:
                    continue
                edge_cost = float(np.linalg.norm(current_point - points_np[neighbor]))
                new_distance = distance + edge_cost
                if new_distance >= best_distance.get(neighbor, float("inf")):
                    continue
                best_distance[neighbor] = new_distance
                heapq.heappush(heap, (new_distance, neighbor))

        if not ordered:
            return (
                torch.empty(0, dtype=torch.long, device=object_points.device),
                torch.empty(0, dtype=torch.long, device=object_points.device),
                {
                    "patch_size": 0,
                    "depth_min": None,
                    "depth_max": None,
                    "sample_node_ids": [],
                },
            )

        unique_patch_np = np.asarray(ordered[:region_node_count], dtype=np.int64)
        unique_patch = torch.as_tensor(
            unique_patch_np, dtype=torch.long, device=object_points.device
        )
        if unique_patch.shape[0] < region_node_count:
            repeats = (region_node_count + unique_patch.shape[0] - 1) // unique_patch.shape[0]
            expanded_patch = unique_patch.repeat(repeats)[:region_node_count]
        else:
            expanded_patch = unique_patch

        patch_points = object_points[unique_patch]
        rel = patch_points - hit_world.unsqueeze(0)
        depth = torch.sum(
            rel * (ray_direction / ray_direction.norm().clamp_min(1e-6)).unsqueeze(0),
            dim=1,
        )
        debug = {
            "patch_size": int(unique_patch.shape[0]),
            "depth_min": float(depth.min().item()),
            "depth_max": float(depth.max().item()),
            "sample_node_ids": [int(idx) for idx in unique_patch_np[:12].tolist()],
        }
        return expanded_patch, unique_patch, debug

    def _build_multi_points_controller_attachment_candidate(
        self,
        source,
        anchor_state,
        object_points,
        controller_attachment_metadata,
        hit_world,
        ray_direction,
    ):
        if anchor_state is None:
            return None
        if hit_world is None:
            hit_world = anchor_state["center_world"].clone()
        if ray_direction is None:
            fallback = self._build_single_point_controller_attachment_candidate(
                source,
                anchor_state,
                object_points,
                controller_attachment_metadata,
                hit_world=hit_world,
            )
            if fallback is not None:
                fallback["multi_points_debug"] = {
                    "fallback_used": True,
                    "reason": "missing_ray_direction",
                }
            return fallback

        source_meta = controller_attachment_metadata[source]
        candidate_pool = anchor_state["region_indices"].clone()
        spring_count = int(source_meta["spring_indices"].numel())
        seed_idx, seed_debug = self._select_multi_points_seed_index(
            hit_world,
            ray_direction,
            candidate_pool,
            object_points,
        )
        if seed_idx is None:
            fallback = self._build_single_point_controller_attachment_candidate(
                source,
                anchor_state,
                object_points,
                controller_attachment_metadata,
                hit_world=hit_world,
            )
            if fallback is not None:
                fallback["multi_points_debug"] = {
                    "fallback_used": True,
                    "reason": "no_surface_seed",
                }
            return fallback

        expanded_indices, unique_indices, patch_debug = self._multi_points_region_from_seed(
            seed_idx,
            candidate_pool,
            hit_world,
            ray_direction,
            object_points,
            spring_count,
        )
        if expanded_indices.numel() == 0 or unique_indices.numel() == 0:
            fallback = self._build_single_point_controller_attachment_candidate(
                source,
                anchor_state,
                object_points,
                controller_attachment_metadata,
                hit_world=hit_world,
            )
            if fallback is not None:
                fallback["multi_points_debug"] = {
                    "fallback_used": True,
                    "reason": "empty_surface_patch",
                }
            return fallback

        source_index = self._controller_source_index(source)
        springs_dtype = source_meta["template_springs"].dtype
        controller_endpoint = torch.full(
            (int(expanded_indices.numel()),),
            self.num_all_points + source_index,
            dtype=springs_dtype,
            device=cfg.device,
        )
        springs = torch.stack(
            [controller_endpoint, expanded_indices.to(dtype=springs_dtype)],
            dim=1,
        )
        selected_points = object_points[unique_indices]
        attach_center_world = selected_points.mean(dim=0)
        attach_anchor_world = object_points[int(seed_idx)].clone()
        if hit_world is not None:
            nearest_patch_offset = torch.argmin(
                torch.linalg.norm(selected_points - hit_world.unsqueeze(0), dim=1)
            )
            attach_anchor_world = selected_points[nearest_patch_offset].clone()
        rest_lengths = torch.linalg.norm(
            object_points[expanded_indices] - attach_anchor_world.unsqueeze(0),
            dim=1,
        ).clamp_min(1e-4).clamp_max(self.LIVE_CONTROLLER_ATTACH_MAX_REST_LENGTH)
        attach_radius = torch.linalg.norm(
            selected_points - attach_center_world.unsqueeze(0), dim=1
        ).max()
        debug = {
            "fallback_used": False,
            "seed_index": int(seed_idx),
            "patch_size": int(unique_indices.numel()),
            "depth_min": patch_debug["depth_min"],
            "depth_max": patch_debug["depth_max"],
            "sample_node_ids": patch_debug["sample_node_ids"],
            "candidate_pool_size": seed_debug["candidate_pool_size"],
            "visible_pool_size": seed_debug["visible_pool_size"],
            "seed_depth": seed_debug["seed_depth"],
            "seed_lateral": seed_debug["seed_lateral"],
        }
        return {
            "anchor_name": anchor_state["name"],
            "springs": springs,
            "rest_lengths": rest_lengths,
            "spring_y": source_meta["spring_y_template"].clone(),
            "attach_center_world": attach_center_world,
            "attach_node_count": int(unique_indices.numel()),
            "attach_radius": float(attach_radius.item()),
            "attach_anchor_world": attach_anchor_world,
            "source_template_offsets": torch.zeros(
                (1, 3), dtype=torch.float32, device=cfg.device
            ),
            "selected_object_indices": unique_indices.clone(),
            "target_point_indices": torch.tensor(
                [source_index], dtype=torch.long, device=cfg.device
            ),
            "multi_points_debug": debug,
            "hit_to_anchor_distance": None
            if hit_world is None
            else float(torch.linalg.norm(attach_anchor_world - hit_world).item()),
        }

    def _compute_live_hand_alignment(
        self,
        live_left_anchor,
        live_right_anchor,
        recorded_left_anchor,
        recorded_right_anchor,
    ):
        live_left_anchor = torch.as_tensor(
            np.asarray(live_left_anchor, dtype=np.float32),
            dtype=torch.float32,
            device=cfg.device,
        )
        live_right_anchor = torch.as_tensor(
            np.asarray(live_right_anchor, dtype=np.float32),
            dtype=torch.float32,
            device=cfg.device,
        )
        recorded_left_anchor = torch.as_tensor(
            recorded_left_anchor,
            dtype=torch.float32,
            device=cfg.device,
        )
        recorded_right_anchor = torch.as_tensor(
            recorded_right_anchor,
            dtype=torch.float32,
            device=cfg.device,
        )

        sign_x = 1.0
        live_x_delta = float((live_right_anchor[0] - live_left_anchor[0]).item())
        recorded_x_delta = float((recorded_right_anchor[0] - recorded_left_anchor[0]).item())
        if live_x_delta * recorded_x_delta < 0.0:
            sign_x = -1.0

        sign = torch.tensor([sign_x, 1.0, 1.0], dtype=torch.float32, device=cfg.device)
        live_left_aligned = live_left_anchor * sign
        live_right_aligned = live_right_anchor * sign

        live_mid = 0.5 * (live_left_aligned + live_right_aligned)
        recorded_mid = 0.5 * (recorded_left_anchor + recorded_right_anchor)
        live_span = torch.linalg.norm(live_right_aligned - live_left_aligned)
        recorded_span = torch.linalg.norm(recorded_right_anchor - recorded_left_anchor)
        scale = recorded_span / live_span.clamp_min(1e-6)
        translation = recorded_mid - scale * live_mid
        reference_scene_left = live_left_aligned * scale + translation
        reference_scene_right = live_right_aligned * scale + translation

        return {
            "sign": sign,
            "scale": scale,
            "translation": translation,
            "motion_gain": torch.tensor(
                self.LIVE_HAND_MOTION_GAIN, dtype=torch.float32, device=cfg.device
            ),
            "reference_live_left": live_left_aligned,
            "reference_live_right": live_right_aligned,
            "reference_scene_left": reference_scene_left,
            "reference_scene_right": reference_scene_right,
        }

    def _compute_single_hand_alignment(
        self,
        live_anchor,
        recorded_anchor,
    ):
        live_anchor = torch.as_tensor(
            np.asarray(live_anchor, dtype=np.float32),
            dtype=torch.float32,
            device=cfg.device,
        )
        recorded_anchor = torch.as_tensor(
            recorded_anchor,
            dtype=torch.float32,
            device=cfg.device,
        )
        sign = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=cfg.device)
        scale = torch.tensor(1.0, dtype=torch.float32, device=cfg.device)
        translation = recorded_anchor - live_anchor
        return {
            "sign": sign,
            "scale": scale,
            "translation": translation,
            "motion_gain": torch.tensor(
                self.LIVE_HAND_MOTION_GAIN, dtype=torch.float32, device=cfg.device
            ),
            "reference_live_left": live_anchor,
            "reference_live_right": live_anchor,
            "reference_scene_left": recorded_anchor,
            "reference_scene_right": recorded_anchor,
        }

    def _hand_is_renderable(self, hand):
        if not hand.active:
            return False
        valid_count = int(np.count_nonzero(hand.valid))
        if valid_count < self.LIVE_HAND_MIN_VALID_JOINTS:
            return False

        valid_joints = hand.joints[hand.valid]
        if valid_joints.shape[0] == 0:
            return False

        spread = np.linalg.norm(valid_joints.max(axis=0) - valid_joints.min(axis=0))
        return float(spread) >= self.LIVE_HAND_MIN_SPREAD

    def _hand_anchor_if_renderable(self, hand):
        if not self._hand_is_renderable(hand):
            return None
        return hand_anchor(hand)

    def _update_live_alignment(
        self,
        alignment,
        alignment_mode,
        live_sample,
        recorded_anchor_centers,
    ):
        left_anchor = self._hand_anchor_if_renderable(live_sample.left)
        right_anchor = self._hand_anchor_if_renderable(live_sample.right)

        if left_anchor is not None and right_anchor is not None:
            if alignment_mode != "dual":
                alignment = self._compute_live_hand_alignment(
                    left_anchor,
                    right_anchor,
                    recorded_anchor_centers[0],
                    recorded_anchor_centers[1],
                )
                alignment_mode = "dual"
            return alignment, alignment_mode

        if alignment is None:
            if left_anchor is not None:
                alignment = self._compute_single_hand_alignment(
                    left_anchor,
                    recorded_anchor_centers[0],
                )
                alignment_mode = "single_left"
            elif right_anchor is not None:
                alignment = self._compute_single_hand_alignment(
                    right_anchor,
                    recorded_anchor_centers[1],
                )
                alignment_mode = "single_right"

        return alignment, alignment_mode

    def _transform_live_points(self, points, alignment):
        return points * alignment["sign"] * alignment["scale"] + alignment["translation"]

    def _convert_live_sample_to_world(self, live_sample, alignment):
        left_points = torch.from_numpy(live_sample.left.joints).to(
            device=cfg.device, dtype=torch.float32
        )
        right_points = torch.from_numpy(live_sample.right.joints).to(
            device=cfg.device, dtype=torch.float32
        )
        left_valid = torch.from_numpy(live_sample.left.valid).to(
            device=cfg.device, dtype=torch.bool
        )
        right_valid = torch.from_numpy(live_sample.right.valid).to(
            device=cfg.device, dtype=torch.bool
        )

        if alignment is None:
            return None, None, None, None, None, None

        left_anchor_np = self._hand_anchor_if_renderable(live_sample.left)
        right_anchor_np = self._hand_anchor_if_renderable(live_sample.right)
        left_world = None
        right_world = None
        left_valid_mask = None
        right_valid_mask = None
        left_anchor = None
        right_anchor = None
        if left_anchor_np is not None:
            left_base_world = self._transform_live_points(left_points, alignment)
            left_valid_mask = left_valid
            left_anchor_base = self._transform_live_points(
                torch.from_numpy(np.asarray(left_anchor_np, dtype=np.float32))
                .to(device=cfg.device)
                .unsqueeze(0),
                alignment,
            )[0]
            left_extra = (
                alignment["motion_gain"] - 1.0
            ) * (left_anchor_base - alignment["reference_scene_left"])
            left_world = left_base_world + left_extra
            left_anchor = left_anchor_base + left_extra
        if right_anchor_np is not None:
            right_base_world = self._transform_live_points(right_points, alignment)
            right_valid_mask = right_valid
            right_anchor_base = self._transform_live_points(
                torch.from_numpy(np.asarray(right_anchor_np, dtype=np.float32))
                .to(device=cfg.device)
                .unsqueeze(0),
                alignment,
            )[0]
            right_extra = (
                alignment["motion_gain"] - 1.0
            ) * (right_anchor_base - alignment["reference_scene_right"])
            right_world = right_base_world + right_extra
            right_anchor = right_anchor_base + right_extra

        return left_world, left_valid_mask, left_anchor, right_world, right_valid_mask, right_anchor

    def _make_live_target_from_anchors(
        self,
        base_target,
        controller_masks,
        current_left_anchor,
        current_right_anchor,
        recorded_anchor_centers,
    ):
        target = base_target.clone()
        if current_left_anchor is not None:
            target[controller_masks[0]] += current_left_anchor - recorded_anchor_centers[0]
        if len(controller_masks) > 1 and current_right_anchor is not None:
            target[controller_masks[1]] += current_right_anchor - recorded_anchor_centers[1]
        return target

    def _apply_live_controller_source_target(
        self,
        target,
        source,
        anchor,
        controller_source_masks,
        controller_source_anchor_centers,
        interaction_state=None,
    ):
        source_index = self._controller_source_index(source)
        source_mask = controller_source_masks[source_index]
        if anchor is None:
            return

        if interaction_state is not None:
            if interaction_state.get("kinematic_only", False):
                return
            source_template_offsets = interaction_state.get("source_template_offsets")
            target_point_indices = interaction_state.get("target_point_indices")
            if (
                source_template_offsets is not None
                and target_point_indices is not None
            ):
                target[target_point_indices] = anchor.unsqueeze(0) + source_template_offsets
                return
            if source_template_offsets is not None:
                target[source_mask] = anchor.unsqueeze(0) + source_template_offsets
                return

        target[source_mask] += anchor - controller_source_anchor_centers[source_index]

    def _make_live_controller_target_from_anchors(
        self,
        base_target,
        current_left_anchor,
        current_right_anchor,
        controller_source_masks,
        controller_source_anchor_centers,
        controller_interaction_state,
    ):
        target = base_target.clone()
        self._apply_live_controller_source_target(
            target,
            "left",
            current_left_anchor,
            controller_source_masks,
            controller_source_anchor_centers,
            controller_interaction_state.get("left"),
        )
        self._apply_live_controller_source_target(
            target,
            "right",
            current_right_anchor,
            controller_source_masks,
            controller_source_anchor_centers,
            controller_interaction_state.get("right"),
        )
        return target

    def _apply_live_controller_anchor_kinematic_overrides(
        self,
        current_left_anchor,
        current_right_anchor,
        controller_interaction_state,
        state_index=0,
    ):
        state = self.simulator.wp_states[state_index]
        state_x = wp.to_torch(state.wp_x, requires_grad=False)
        state_v = wp.to_torch(state.wp_v, requires_grad=False)
        for source, current_anchor in (
            ("left", current_left_anchor),
            ("right", current_right_anchor),
        ):
            interaction_state = controller_interaction_state.get(source)
            if (
                interaction_state is None
                or current_anchor is None
                or not interaction_state.get("kinematic_only", False)
            ):
                continue
            selected_object_indices = interaction_state.get("selected_object_indices")
            reference_positions = interaction_state.get("selected_object_reference_positions")
            attach_anchor_world = interaction_state.get("attach_anchor_world")
            if (
                selected_object_indices is None
                or reference_positions is None
                or attach_anchor_world is None
            ):
                continue
            state_x[selected_object_indices] = reference_positions + (
                current_anchor - attach_anchor_world
            )
            state_v[selected_object_indices] = 0.0

    def _apply_controller_start_prev_target_overrides(
        self,
        prev_target,
        current_target,
        controller_source_masks,
        controller_interaction_state,
    ):
        for source in ("left", "right"):
            interaction_state = controller_interaction_state.get(source)
            if interaction_state is None or not interaction_state.get("just_started", False):
                continue
            target_point_indices = interaction_state.get("target_point_indices")
            if target_point_indices is not None:
                prev_target[target_point_indices] = current_target[target_point_indices]
            else:
                source_mask = controller_source_masks[self._controller_source_index(source)]
                prev_target[source_mask] = current_target[source_mask]
            interaction_state["just_started"] = False

    def _update_controller_select_hold_state(
        self,
        source,
        controller_sample,
        select_hold_state,
    ):
        previous_state = dict(select_hold_state.get(source, {}))
        available = bool(
            controller_sample is not None and controller_sample.select_available
        )
        pressed = bool(
            controller_sample is not None
            and controller_sample.select_available
            and controller_sample.select_pressed
        )
        value = float(
            0.0 if controller_sample is None else controller_sample.select_value
        )
        previous_pressed = bool(previous_state.get("pressed", False))
        previous_value = float(previous_state.get("value", 0.0))
        previous_release_frames = int(previous_state.get("release_frames", 0))
        start_active = available and (
            pressed or value >= self.LIVE_CONTROLLER_SELECT_START_THRESHOLD
        )
        hold_active = available and (
            pressed or value >= self.LIVE_CONTROLLER_SELECT_HOLD_THRESHOLD
        )
        select_start_edge = available and (
            (pressed and not previous_pressed)
            or (
                previous_value < self.LIVE_CONTROLLER_SELECT_START_THRESHOLD
                and value >= self.LIVE_CONTROLLER_SELECT_START_THRESHOLD
            )
        )
        release_ready = available and (not pressed) and (
            value < self.LIVE_CONTROLLER_SELECT_HOLD_THRESHOLD
        )
        release_frames = previous_release_frames + 1 if release_ready else 0
        state = {
            "available": available,
            "pressed": pressed,
            "value": value,
            "start_active": start_active,
            "hold_active": hold_active,
            "start_edge": select_start_edge,
            "release_ready": release_ready,
            "release_frames": release_frames,
        }
        select_hold_state[source] = state
        return state

    def _update_controller_exit_hold_state(
        self,
        source,
        controller_sample,
        controller_exit_hold_state,
        now_wall=None,
    ):
        previous_state = dict(controller_exit_hold_state.get(source, {}))
        available = bool(
            controller_sample is not None and getattr(controller_sample, "exit_available", False)
        )
        pressed = bool(
            controller_sample is not None
            and getattr(controller_sample, "exit_available", False)
            and getattr(controller_sample, "exit_pressed", False)
        )
        if now_wall is None:
            now_wall = time.perf_counter()

        previous_pressed = bool(previous_state.get("pressed", False))
        press_start_time_wall = previous_state.get("press_start_time_wall")
        hold_fired = bool(previous_state.get("hold_fired", False))
        exit_requested = False

        if not available or not pressed:
            state = {
                "available": available,
                "pressed": False,
                "press_start_time_wall": None,
                "hold_fired": False,
            }
            controller_exit_hold_state[source] = state
            return state, exit_requested

        if not previous_pressed or press_start_time_wall is None:
            press_start_time_wall = float(now_wall)
            hold_fired = False

        if (
            not hold_fired
            and float(now_wall) - float(press_start_time_wall)
            >= float(self.LIVE_CONTROLLER_EXIT_HOLD_SECONDS)
        ):
            hold_fired = True
            exit_requested = True

        state = {
            "available": available,
            "pressed": True,
            "press_start_time_wall": float(press_start_time_wall),
            "hold_fired": hold_fired,
        }
        controller_exit_hold_state[source] = state
        return state, exit_requested

    def _update_live_controller_runtime_from_sample(
        self,
        latest_controller_sample,
        live_controller_alignment,
        live_controller_alignment_mode,
        controller_source_anchor_centers,
        w2c,
        controller_select_state_cache,
        controller_select_hold_state,
        controller_select_hold_state_cache,
        controller_anchor_cycle_state_cache,
        controller_snap_state_cache,
        controller_snap_edge_cache,
        controller_exit_hold_state,
        controller_exit_state_cache=None,
        basis_override=None,
        collect_reset_edges=True,
        collect_exit_holds=True,
        alignment_pose_role="selected",
        controller_position_pose_role="selected",
        controller_ray_pose_role=None,
    ):
        controller_reset_sources = []
        controller_exit_sources = []
        if latest_controller_sample is None:
            return {
                "alignment": live_controller_alignment,
                "alignment_mode": live_controller_alignment_mode,
                "left_controller": None,
                "right_controller": None,
                "reset_sources": controller_reset_sources,
                "exit_sources": controller_exit_sources,
                "alignment_acquired": False,
            }

        previous_alignment_available = live_controller_alignment is not None
        now_wall = time.perf_counter()
        live_controller_alignment, live_controller_alignment_mode = (
            self._update_live_controller_alignment(
                live_controller_alignment,
                live_controller_alignment_mode,
                latest_controller_sample,
                controller_source_anchor_centers,
                w2c,
                basis_override=basis_override,
                pose_role=alignment_pose_role,
            )
        )
        current_live_left_controller = self._convert_live_controller_to_world(
            "left",
            latest_controller_sample.left,
            live_controller_alignment,
            position_pose_role=controller_position_pose_role,
            ray_pose_role=controller_ray_pose_role,
        )
        current_live_right_controller = self._convert_live_controller_to_world(
            "right",
            latest_controller_sample.right,
            live_controller_alignment,
            position_pose_role=controller_position_pose_role,
            ray_pose_role=controller_ray_pose_role,
        )
        controller_world_by_source = {
            "left": current_live_left_controller,
            "right": current_live_right_controller,
        }
        for source, controller_sample in (
            ("left", latest_controller_sample.left),
            ("right", latest_controller_sample.right),
        ):
            self._log_controller_select_transition(
                source,
                controller_sample,
                controller_select_state_cache,
                sample_id=latest_controller_sample.sample,
            )
            select_hold_runtime = self._update_controller_select_hold_state(
                source,
                controller_sample,
                controller_select_hold_state,
            )
            self._log_controller_select_hold_transition(
                source,
                select_hold_runtime,
                controller_select_hold_state_cache,
                sample_id=latest_controller_sample.sample,
            )
            controller_world = controller_world_by_source[source]
            if controller_world is not None:
                controller_world.update(
                    {
                        "sample_id": int(latest_controller_sample.sample),
                        "select_start_edge": select_hold_runtime["start_edge"],
                        "select_hold_active": select_hold_runtime["hold_active"],
                        "select_release_ready": select_hold_runtime["release_ready"],
                        "select_release_frames": select_hold_runtime["release_frames"],
                        "select_start_active": select_hold_runtime["start_active"],
                    }
                )
            if collect_exit_holds:
                _, exit_requested = self._update_controller_exit_hold_state(
                    source,
                    controller_sample,
                    controller_exit_hold_state,
                    now_wall=now_wall,
                )
                if exit_requested:
                    controller_exit_sources.append(source)
            if controller_exit_state_cache is not None:
                self._log_controller_exit_transition(
                    source,
                    controller_sample,
                    controller_exit_state_cache,
                )
            self._log_controller_anchor_cycle_transition(
                source,
                controller_sample,
                controller_anchor_cycle_state_cache,
            )
            self._log_controller_snap_transition(
                source,
                controller_sample,
                controller_snap_state_cache,
            )
            if (
                collect_reset_edges
                and self._controller_snap_edge(
                    source,
                    controller_sample,
                    controller_snap_edge_cache,
                )
            ):
                controller_reset_sources.append(source)

        return {
            "alignment": live_controller_alignment,
            "alignment_mode": live_controller_alignment_mode,
            "left_controller": current_live_left_controller,
            "right_controller": current_live_right_controller,
            "reset_sources": controller_reset_sources,
            "exit_sources": controller_exit_sources,
            "alignment_acquired": (
                not previous_alignment_available
                and live_controller_alignment is not None
            ),
        }

    def _compute_next_live_controller_target(
        self,
        controller_runtime_base_target,
        prev_target,
        controller_interaction_state,
        controller_interaction_state_cache,
        controller_source_masks,
        controller_source_anchor_centers,
        controller_attachment_metadata,
        controller_anchor_templates,
        controller_predefined_anchor_states,
        controller_anchor_preview_state,
        controller_overlay_by_source,
        current_live_left_controller,
        current_live_right_controller,
        object_points,
        controller_reset_triggered=False,
        allow_implicit_fallback_start=True,
        post_select_translation_only=False,
        controller_motion_state_cache=None,
        frame_index=0,
        runtime_label="shared",
    ):
        next_target = controller_runtime_base_target.clone()
        if not controller_reset_triggered:
            (
                current_left_interaction_anchor,
                current_right_interaction_anchor,
            ) = self._resolve_live_controller_interaction_anchors(
                current_live_left_controller,
                current_live_right_controller,
                controller_overlay_by_source,
                controller_interaction_state,
                controller_source_anchor_centers,
                controller_attachment_metadata,
                controller_anchor_templates,
                controller_predefined_anchor_states,
                controller_anchor_preview_state,
                object_points,
                allow_implicit_fallback_start=allow_implicit_fallback_start,
                post_select_translation_only=post_select_translation_only,
            )
            self._log_controller_interaction_transition(
                "left",
                controller_interaction_state["left"],
                controller_interaction_state_cache,
            )
            self._log_controller_interaction_transition(
                "right",
                controller_interaction_state["right"],
                controller_interaction_state_cache,
            )
            next_target = self._make_live_controller_target_from_anchors(
                next_target,
                current_left_interaction_anchor,
                current_right_interaction_anchor,
                controller_source_masks,
                controller_source_anchor_centers,
                controller_interaction_state,
            )
            if controller_motion_state_cache is not None:
                self._log_controller_motion_parity(
                    runtime_label,
                    frame_index,
                    "left",
                    current_live_left_controller,
                    current_left_interaction_anchor,
                    next_target,
                    controller_runtime_base_target,
                    controller_source_masks,
                    controller_attachment_metadata,
                    controller_interaction_state.get("left"),
                    controller_motion_state_cache,
                )
                self._log_controller_motion_parity(
                    runtime_label,
                    frame_index,
                    "right",
                    current_live_right_controller,
                    current_right_interaction_anchor,
                    next_target,
                    controller_runtime_base_target,
                    controller_source_masks,
                    controller_attachment_metadata,
                    controller_interaction_state.get("right"),
                    controller_motion_state_cache,
                )
            self._apply_controller_start_prev_target_overrides(
                prev_target,
                next_target,
                controller_source_masks,
                controller_interaction_state,
            )
        else:
            if controller_motion_state_cache is not None:
                controller_motion_state_cache["left"] = None
                controller_motion_state_cache["right"] = None
            self._log_controller_interaction_transition(
                "left",
                controller_interaction_state["left"],
                controller_interaction_state_cache,
            )
            self._log_controller_interaction_transition(
                "right",
                controller_interaction_state["right"],
                controller_interaction_state_cache,
            )
        return next_target

    def _reset_live_controller_runtime(
        self,
        controller_runtime_base_target,
        controller_interaction_state,
        controller_anchor_preview_state,
        controller_attachment_metadata,
        reset_state=None,
    ):
        for source in ("left", "right"):
            interaction_state = controller_interaction_state.get(source)
            if (
                interaction_state is not None
                and interaction_state.get("spring_remap_applied", False)
            ):
                self._restore_controller_attachment_remap(
                    source, controller_attachment_metadata
                )
            controller_interaction_state[source] = None
            self._reset_controller_anchor_preview_state(
                controller_anchor_preview_state,
                source,
            )

        if reset_state is None:
            self.simulator.set_init_state(
                self.simulator.wp_init_vertices,
                self.simulator.wp_init_velocities,
            )
        else:
            self._restore_sim_state(reset_state)
        if self.simulator.object_collision_flag:
            self.simulator.create_resting_case()

        reset_target = controller_runtime_base_target.clone()
        self.simulator.set_controller_interactive(reset_target, reset_target)
        return reset_target

    def _project_points_to_pixels(self, world_points, intrinsic, w2c):
        if torch.is_tensor(world_points):
            target_device = world_points.device
            target_dtype = world_points.dtype
            world_points_t = world_points
        else:
            target_device = torch.device(cfg.device)
            target_dtype = torch.float32
            world_points_t = torch.as_tensor(
                world_points,
                dtype=target_dtype,
                device=target_device,
            )
        if world_points_t.ndim == 1:
            world_points_t = world_points_t.unsqueeze(0)
        intrinsic_t = torch.as_tensor(
            intrinsic,
            dtype=target_dtype,
            device=target_device,
        )
        w2c_t = torch.as_tensor(
            w2c,
            dtype=target_dtype,
            device=target_device,
        )
        ones = torch.ones(
            (world_points_t.shape[0], 1),
            dtype=target_dtype,
            device=target_device,
        )
        world_points_h = torch.cat([world_points_t, ones], dim=1)
        camera_points_h = world_points_h @ w2c_t.T
        camera_points = camera_points_h[:, :3]
        depth_valid = camera_points[:, 2] > 1e-6
        pixel_h = camera_points @ intrinsic_t.T
        pixels = pixel_h[:, :2] / pixel_h[:, 2:3].clamp_min(1e-6)
        return pixels, depth_valid

    def _project_points_to_pixels_multi_eye(
        self,
        world_points,
        intrinsic_by_eye,
        w2c_by_eye,
    ):
        if torch.is_tensor(world_points):
            world_points_t = world_points
        else:
            world_points_t = torch.as_tensor(
                world_points,
                dtype=torch.float32,
                device=cfg.device,
            )
        if world_points_t.ndim == 1:
            world_points_t = world_points_t.unsqueeze(0)
        intrinsic_t = torch.as_tensor(
            intrinsic_by_eye,
            dtype=world_points_t.dtype,
            device=world_points_t.device,
        )
        w2c_t = torch.as_tensor(
            w2c_by_eye,
            dtype=world_points_t.dtype,
            device=world_points_t.device,
        )
        if intrinsic_t.ndim != 3 or w2c_t.ndim != 3:
            raise ValueError("Expected batched intrinsic and w2c tensors.")
        eye_count = int(intrinsic_t.shape[0])
        ones = torch.ones(
            (world_points_t.shape[0], 1),
            dtype=world_points_t.dtype,
            device=world_points_t.device,
        )
        world_points_h = torch.cat([world_points_t, ones], dim=1).unsqueeze(0).expand(
            eye_count,
            -1,
            -1,
        )
        camera_points_h = torch.bmm(world_points_h, w2c_t.transpose(1, 2))
        camera_points = camera_points_h[..., :3]
        depth_valid = camera_points[..., 2] > 1e-6
        pixel_h = torch.bmm(camera_points, intrinsic_t.transpose(1, 2))
        pixels = pixel_h[..., :2] / pixel_h[..., 2:3].clamp_min(1e-6)
        return pixels, depth_valid

    def _project_world_point_to_pixel(self, world_point, intrinsic, w2c, height, width):
        if world_point is None:
            return None
        if torch.is_tensor(intrinsic):
            target_device = intrinsic.device
            target_dtype = intrinsic.dtype
        elif torch.is_tensor(w2c):
            target_device = w2c.device
            target_dtype = w2c.dtype
        elif torch.is_tensor(world_point):
            target_device = world_point.device
            target_dtype = world_point.dtype
        else:
            target_device = torch.device(cfg.device)
            target_dtype = torch.float32
        world_point_t = torch.as_tensor(
            world_point,
            dtype=target_dtype,
            device=target_device,
        )
        pixels, depth_valid = self._project_points_to_pixels(
            world_point_t.unsqueeze(0), intrinsic, w2c
        )
        if not bool(depth_valid[0].item()):
            return None
        x = float(pixels[0, 0].item())
        y = float(pixels[0, 1].item())
        if x < 0.0 or x >= float(width) or y < 0.0 or y >= float(height):
            return None
        return pixels[0]

    def _interaction_joint_if_valid(self, world_points, valid_mask, joint_index):
        if world_points is None or valid_mask is None:
            return None
        if valid_mask.shape[0] <= joint_index or not bool(valid_mask[joint_index].item()):
            return None
        return world_points[joint_index]

    def _build_hand_interaction_repr(self, world_points, valid_mask):
        if world_points is None or valid_mask is None or not bool(valid_mask.any().item()):
            return None

        palm = self._interaction_joint_if_valid(world_points, valid_mask, PALM_JOINT_INDEX)
        if palm is None:
            palm = world_points[valid_mask].mean(dim=0)
        wrist = self._interaction_joint_if_valid(world_points, valid_mask, WRIST_JOINT_INDEX)

        thumb_tip = self._interaction_joint_if_valid(
            world_points, valid_mask, THUMB_TIP_JOINT_INDEX
        )
        index_tip = self._interaction_joint_if_valid(
            world_points, valid_mask, INDEX_TIP_JOINT_INDEX
        )
        middle_tip = self._interaction_joint_if_valid(
            world_points, valid_mask, MIDDLE_TIP_JOINT_INDEX
        )
        pinch_point = None
        if thumb_tip is not None and index_tip is not None:
            pinch_point = 0.5 * (thumb_tip + index_tip)

        return {
            "palm": palm,
            "wrist": wrist,
            "thumb_tip": thumb_tip,
            "index_tip": index_tip,
            "middle_tip": middle_tip,
            "pinch_point": pinch_point,
        }

    def _project_interaction_repr(self, interaction_repr, intrinsic, w2c, height, width):
        ordered_keys = []
        ordered_points = []
        for key in ("palm", "wrist", "thumb_tip", "index_tip", "middle_tip", "pinch_point"):
            point = interaction_repr.get(key)
            if point is None:
                continue
            ordered_keys.append(key)
            ordered_points.append(point.unsqueeze(0))

        if not ordered_points:
            return None

        pixels, depth_valid = self._project_points_to_pixels(
            torch.cat(ordered_points, dim=0), intrinsic, w2c
        )
        projected = {}
        for idx, key in enumerate(ordered_keys):
            if not bool(depth_valid[idx].item()):
                continue

            x = float(pixels[idx, 0].item())
            y = float(pixels[idx, 1].item())
            if x < 0.0 or x >= float(width) or y < 0.0 or y >= float(height):
                continue
            projected[key] = pixels[idx]

        if "palm" not in projected:
            return None
        return projected

    def _blend_marker(self, frame, pixel, color, radius, blend=0.75):
        height, width = frame.shape[:2]
        x = int(round(float(pixel[0].item())))
        y = int(round(float(pixel[1].item())))
        if x < 0 or x >= width or y < 0 or y >= height:
            return

        x0 = max(0, x - radius)
        x1 = min(width, x + radius + 1)
        y0 = max(0, y - radius)
        y1 = min(height, y + radius + 1)
        rgb_frame = frame[..., :3]
        color_tensor = rgb_frame.new_tensor(color[:3])
        rgb_frame[y0:y1, x0:x1].mul_(1.0 - blend).add_(color_tensor * blend)

    def _blend_rect(self, frame, x0, y0, x1, y1, color, blend=0.45):
        height, width = frame.shape[:2]
        x0 = max(0, min(width, int(x0)))
        x1 = max(0, min(width, int(x1)))
        y0 = max(0, min(height, int(y0)))
        y1 = max(0, min(height, int(y1)))
        if x0 >= x1 or y0 >= y1:
            return
        rgb_frame = frame[..., :3]
        color_tensor = rgb_frame.new_tensor(color[:3])
        rgb_frame[y0:y1, x0:x1].mul_(1.0 - blend).add_(color_tensor * blend)

    def _blend_square_marker(self, frame, pixel, color, radius, blend=0.78):
        height, width = frame.shape[:2]
        x = int(round(float(pixel[0].item())))
        y = int(round(float(pixel[1].item())))
        if x < 0 or x >= width or y < 0 or y >= height:
            return

        x0 = x - radius
        x1 = x + radius + 1
        y0 = y - radius
        y1 = y + radius + 1
        self._blend_rect(frame, x0, y0, x1, y0 + 1, color, blend=blend)
        self._blend_rect(frame, x0, y1 - 1, x1, y1, color, blend=blend)
        self._blend_rect(frame, x0, y0, x0 + 1, y1, color, blend=blend)
        self._blend_rect(frame, x1 - 1, y0, x1, y1, color, blend=blend)

    def _bitmap_text_size(self, text, scale=1):
        glyph_width = 3 * scale
        glyph_height = 5 * scale
        advance = glyph_width + scale
        width = 0 if not text else len(text) * advance - scale
        return width, glyph_height

    def _draw_bitmap_text(self, frame, text, x, y, color, scale=1, blend=0.88):
        cursor_x = int(x)
        cursor_y = int(y)
        rgb_frame = frame[..., :3]
        color_tensor = rgb_frame.new_tensor(color[:3])
        advance = 3 * scale + scale
        height, width = frame.shape[:2]
        for char in text:
            glyph = TINY_BITMAP_FONT.get(char.lower(), TINY_BITMAP_FONT[" "])
            for row_idx, row in enumerate(glyph):
                for col_idx, value in enumerate(row):
                    if value != "1":
                        continue
                    x0 = cursor_x + col_idx * scale
                    y0 = cursor_y + row_idx * scale
                    x1 = x0 + scale
                    y1 = y0 + scale
                    if x1 <= 0 or y1 <= 0 or x0 >= width or y0 >= height:
                        continue
                    x0 = max(0, x0)
                    y0 = max(0, y0)
                    x1 = min(width, x1)
                    y1 = min(height, y1)
                    patch = rgb_frame[y0:y1, x0:x1]
                    patch.mul_(1.0 - blend).add_(color_tensor * blend)
            cursor_x += advance

    def _draw_timing_overlay(self, frame, total_time_seconds):
        if total_time_seconds is None or total_time_seconds <= 0.0:
            return

        frame_ms = total_time_seconds * 1000.0
        fps = 1.0 / max(total_time_seconds, 1e-6)
        text = f"{frame_ms:.1f}ms {fps:.1f}fps"
        height, width = frame.shape[:2]
        scale = max(
            self.TIMING_OVERLAY_SCALE,
            int(
                round(
                    height
                    / float(self.TIMING_OVERLAY_REFERENCE_HEIGHT)
                    * self.TIMING_OVERLAY_SCALE
                )
            ),
        )
        scale = min(scale, self.TIMING_OVERLAY_MAX_SCALE)
        margin = max(self.TIMING_OVERLAY_MARGIN, int(round(scale * 2.5)))
        text_width, text_height = self._bitmap_text_size(text, scale=scale)
        pad = scale + 2
        x0 = width - margin - text_width - 2 * pad
        y0 = margin
        self._blend_rect(
            frame,
            x0,
            y0,
            x0 + text_width + 2 * pad,
            y0 + text_height + 2 * pad,
            self.TIMING_OVERLAY_BG_COLOR,
            blend=0.46,
        )
        self._draw_bitmap_text(
            frame,
            text,
            x0 + pad,
            y0 + pad,
            self.TIMING_OVERLAY_TEXT_COLOR,
            scale=scale,
            blend=0.92,
        )

    def _render_profile_new_frame(self, enabled):
        if not enabled:
            return None
        return {"_cuda_spans": []}

    def _render_profile_capture_cuda_memory(self, frame_profile):
        if frame_profile is None or not torch.cuda.is_available():
            return
        bytes_per_gib = float(1024 ** 3)
        frame_profile["cuda_memory_allocated_gib"] = (
            float(torch.cuda.memory_allocated()) / bytes_per_gib
        )
        frame_profile["cuda_memory_reserved_gib"] = (
            float(torch.cuda.memory_reserved()) / bytes_per_gib
        )

    def _render_profile_add_wall_time(self, frame_profile, key, elapsed_seconds):
        if frame_profile is None:
            return
        frame_profile[key] = frame_profile.get(key, 0.0) + float(elapsed_seconds)

    def _render_profile_record_immersive_compose_metrics(
        self,
        frame_profile,
        eye_label,
        compose_metrics,
    ):
        if frame_profile is None or compose_metrics is None:
            return
        eye_label = str(eye_label)
        frame_profile[f"gaussian_raw_{eye_label}_ratio"] = float(
            compose_metrics.get("raw_gaussian_coverage_ratio", 0.0)
        )
        frame_profile[f"gaussian_visible_{eye_label}_ratio"] = float(
            compose_metrics.get("visible_gaussian_coverage_ratio", 0.0)
        )
        frame_profile[f"gaussian_retention_{eye_label}_ratio"] = float(
            compose_metrics.get("visible_retention_ratio", 0.0)
        )
        frame_profile[f"gaussian_compose_roi_{eye_label}_ratio"] = float(
            compose_metrics.get("compose_roi_ratio", 0.0)
        )
        frame_profile[f"scene_depth_finite_{eye_label}_ratio"] = float(
            compose_metrics.get("scene_depth_finite_ratio", 0.0)
        )
        frame_profile[f"scene_depth_positive_{eye_label}_ratio"] = float(
            compose_metrics.get("scene_depth_positive_ratio", 0.0)
        )
        frame_profile[f"scene_depth_invalid_{eye_label}_ratio"] = float(
            1.0 if compose_metrics.get("scene_depth_invalid", False) else 0.0
        )
        frame_profile[f"scene_depth_suppressed_{eye_label}_ratio"] = float(
            1.0 if compose_metrics.get("scene_depth_suppressed", False) else 0.0
        )
        frame_profile["compose_fallback_active_ratio"] = max(
            float(frame_profile.get("compose_fallback_active_ratio", 0.0)),
            float(1.0 if compose_metrics.get("compose_mode") != "depth_aware" else 0.0),
        )

    def _render_profile_begin_cuda_span(self, frame_profile, key):
        if frame_profile is None or not torch.cuda.is_available():
            return None
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        return (key, start_event, end_event)

    def _render_profile_end_cuda_span(self, frame_profile, span):
        if frame_profile is None or span is None:
            return
        key, start_event, end_event = span
        end_event.record()
        frame_profile["_cuda_spans"].append((key, start_event, end_event))

    def _render_profile_finalize_frame(self, frame_profile):
        if frame_profile is None:
            return None
        cuda_spans = frame_profile.pop("_cuda_spans", [])
        for key, start_event, end_event in cuda_spans:
            frame_profile[key] = frame_profile.get(key, 0.0) + (
                start_event.elapsed_time(end_event) / 1000.0
            )
        return frame_profile

    def _render_profile_should_log(self, frame_index, every):
        every = max(int(every), 1)
        return frame_index > 1 and ((frame_index - 1) % every == 0)

    def _render_profile_append_frame(
        self,
        render_profile_series,
        render_profile_rows,
        frame_index,
        frame_profile,
    ):
        if render_profile_series is None or render_profile_rows is None or frame_profile is None:
            return
        row = {"frame": int(frame_index)}
        for key in render_profile_series:
            value = float(frame_profile.get(key, 0.0))
            render_profile_series[key].append(value)
            row[key] = value
        render_profile_rows.append(row)

    def _render_profile_metric_group(self, key):
        if key.endswith("_ratio"):
            return "ratio"
        if key.endswith("_gib"):
            return "memory"
        if (
            key.endswith("_scale")
            or key.endswith("_used")
            or key.endswith("_sample_id")
            or key.endswith("_applied")
            or key.endswith("_dx_px")
            or key.endswith("_dy_px")
        ):
            return "scalar"
        return "time"

    def _format_render_profile_stat(self, key, value):
        metric_group = self._render_profile_metric_group(key)
        if metric_group == "memory":
            return f"{value:.2f} GiB"
        if metric_group == "ratio":
            return f"{value * 100.0:.1f}%"
        if metric_group == "scalar":
            return f"{value:.2f}x" if key.endswith("_scale") else f"{value:.2f}"
        return f"{value * 1000.0:.2f} ms"

    def _format_component_summary_lines(self, average_frame_time, component_rows):
        if not component_rows:
            return []
        component_rows = sorted(component_rows, key=lambda row: row[1], reverse=True)
        formatted_rows = []
        for label, average_component_time in component_rows:
            time_share_percentage = (
                (average_component_time / average_frame_time) * 100.0
                if average_frame_time > 0.0
                else 0.0
            )
            formatted_rows.append(
                (
                    label,
                    f"{average_component_time * 1000.0:.2f} ms",
                    f"{time_share_percentage:.1f}%",
                )
            )
        label_width = max(len("Component"), *(len(row[0]) for row in formatted_rows))
        time_width = max(len("Avg Time"), *(len(row[1]) for row in formatted_rows))
        share_width = max(len("Share"), *(len(row[2]) for row in formatted_rows))
        lines = [
            f"{'Component':<{label_width}}  {'Avg Time':>{time_width}}  {'Share':>{share_width}}"
        ]
        for label, avg_time, share in formatted_rows:
            lines.append(f"{label:<{label_width}}  {avg_time:>{time_width}}  {share:>{share_width}}")
        return lines

    def _render_profile_summary_lines(self, mode, render_profile_series, ordered_keys):
        if not render_profile_series:
            return []
        sample_count = 0
        for key in ordered_keys:
            sample_count = max(sample_count, len(render_profile_series.get(key, [])))
        if sample_count <= 0:
            return []
        lines = [
            f"=== Render Profile Summary ({mode}, avg/p95/max over {sample_count} frames) ==="
        ]
        rows = []
        for key in ordered_keys:
            values = render_profile_series.get(key, [])
            if not values:
                continue
            values_np = np.asarray(values, dtype=np.float64)
            avg_value = float(np.mean(values_np))
            p95_value = float(np.percentile(values_np, 95))
            max_value = float(np.max(values_np))
            rows.append(
                (
                    key,
                    self._render_profile_metric_group(key),
                    avg_value,
                    self._format_render_profile_stat(key, avg_value),
                    self._format_render_profile_stat(key, p95_value),
                    self._format_render_profile_stat(key, max_value),
                )
            )
        if not rows:
            return lines
        group_rank = {"time": 0, "ratio": 1, "scalar": 2, "memory": 3}
        rows.sort(
            key=lambda row: (
                group_rank.get(row[1], 99),
                -row[2] if row[1] == "time" else 0.0,
                row[0],
            )
        )
        metric_width = max(len("metric"), *(len(row[0]) for row in rows))
        avg_width = max(len("avg"), *(len(row[3]) for row in rows))
        p95_width = max(len("p95"), *(len(row[4]) for row in rows))
        max_width = max(len("max"), *(len(row[5]) for row in rows))
        lines.append(
            f"{'metric':<{metric_width}}  {'avg':>{avg_width}}  {'p95':>{p95_width}}  {'max':>{max_width}}"
        )
        for metric, _, _, avg_text, p95_text, max_text in rows:
            lines.append(
                f"{metric:<{metric_width}}  {avg_text:>{avg_width}}  {p95_text:>{p95_width}}  {max_text:>{max_width}}"
            )
        return lines

    def _write_render_profile_outputs(
        self,
        output_dir,
        summary_lines,
        render_profile_rows,
    ):
        if not output_dir:
            return
        os.makedirs(output_dir, exist_ok=True)
        if summary_lines:
            summary_path = os.path.join(output_dir, "render_profile_summary.txt")
            with open(summary_path, "w") as summary_file:
                summary_file.write("\n".join(summary_lines) + "\n")
        if not render_profile_rows:
            return
        fieldnames = ["frame"]
        extra_keys = []
        for row in render_profile_rows:
            for key in row.keys():
                if key != "frame" and key not in extra_keys:
                    extra_keys.append(key)
        fieldnames.extend(extra_keys)
        csv_path = os.path.join(output_dir, "render_profile_frames.csv")
        with open(csv_path, "w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for row in render_profile_rows:
                writer.writerow(row)

    def _log_immersive_render_profile_frame(self, frame_index, frame_profile):
        print(
            "[render_profile][immersive] "
            f"frame={int(frame_index)} "
            f"rendering={frame_profile.get('rendering', 0.0) * 1000.0:.2f}ms "
            f"scene=C{frame_profile.get('scene_render_center_wall', 0.0) * 1000.0:.2f}/"
            f"L{frame_profile.get('scene_render_left_wall', 0.0) * 1000.0:.2f}/"
            f"R{frame_profile.get('scene_render_right_wall', 0.0) * 1000.0:.2f}ms "
            f"bg=C{frame_profile.get('scene_render_background_center_wall', 0.0) * 1000.0:.2f}ms "
            f"bg_prep={frame_profile.get('scene_prepare_background_eye_wall', 0.0) * 1000.0:.2f}ms "
            f"bg_layers=far{frame_profile.get('scene_render_far_center_wall', 0.0) * 1000.0:.2f}/"
            f"near{frame_profile.get('scene_render_near_center_wall', 0.0) * 1000.0:.2f}ms "
            f"bg_layers_warp=farL{frame_profile.get('scene_warp_far_left_cuda', 0.0) * 1000.0:.2f}/"
            f"R{frame_profile.get('scene_warp_far_right_cuda', 0.0) * 1000.0:.2f} "
            f"nearL{frame_profile.get('scene_warp_near_left_cuda', 0.0) * 1000.0:.2f}/"
            f"R{frame_profile.get('scene_warp_near_right_cuda', 0.0) * 1000.0:.2f}ms "
            f"side=L{frame_profile.get('scene_render_side_left_wall', 0.0) * 1000.0:.2f}/"
            f"R{frame_profile.get('scene_render_side_right_wall', 0.0) * 1000.0:.2f}ms "
            f"side_comp=L{frame_profile.get('scene_compose_side_left_cuda', 0.0) * 1000.0:.2f}/"
            f"R{frame_profile.get('scene_compose_side_right_cuda', 0.0) * 1000.0:.2f}ms "
            f"side_roi=L{frame_profile.get('scene_side_roi_left_ratio', 0.0) * 100.0:.1f}/"
            f"R{frame_profile.get('scene_side_roi_right_ratio', 0.0) * 100.0:.1f}% "
            f"side_strip=L{frame_profile.get('scene_side_strip_left_width_ratio', 0.0) * 100.0:.1f}/"
            f"R{frame_profile.get('scene_side_strip_right_width_ratio', 0.0) * 100.0:.1f}% "
            f"side_ff=L{frame_profile.get('scene_side_fullframe_fallback_left_ratio', 0.0) * 100.0:.0f}/"
            f"R{frame_profile.get('scene_side_fullframe_fallback_right_ratio', 0.0) * 100.0:.0f}% "
            f"reproject=L{frame_profile.get('scene_reproject_left_cuda', 0.0) * 1000.0:.2f}/"
            f"R{frame_profile.get('scene_reproject_right_cuda', 0.0) * 1000.0:.2f}ms "
            f"bg_reproject=L{frame_profile.get('scene_reproject_background_left_cuda', 0.0) * 1000.0:.2f}/"
            f"R{frame_profile.get('scene_reproject_background_right_cuda', 0.0) * 1000.0:.2f}ms "
            f"cov=L{frame_profile.get('scene_reproject_valid_pre_left_ratio', 0.0) * 100.0:.0f}->"
            f"{frame_profile.get('scene_reproject_valid_post_left_ratio', 0.0) * 100.0:.0f}%/"
            f"R{frame_profile.get('scene_reproject_valid_pre_right_ratio', 0.0) * 100.0:.0f}->"
            f"{frame_profile.get('scene_reproject_valid_post_right_ratio', 0.0) * 100.0:.0f}% "
            f"bg_cov=L{frame_profile.get('scene_reproject_background_valid_pre_left_ratio', 0.0) * 100.0:.0f}->"
            f"{frame_profile.get('scene_reproject_background_valid_post_left_ratio', 0.0) * 100.0:.0f}%/"
            f"R{frame_profile.get('scene_reproject_background_valid_pre_right_ratio', 0.0) * 100.0:.0f}->"
            f"{frame_profile.get('scene_reproject_background_valid_post_right_ratio', 0.0) * 100.0:.0f}% "
            f"scene_tw={frame_profile.get('scene_timewarp_gpu_ms', 0.0) * 1000.0:.2f}ms "
            f"scene_tw_use={frame_profile.get('scene_timewarp_applied', 0.0):.0f} "
            f"roi=L{frame_profile.get('scene_reproject_roi_pre_left_ratio', 0.0) * 100.0:.0f}->"
            f"{frame_profile.get('scene_reproject_roi_post_left_ratio', 0.0) * 100.0:.0f}%/"
            f"R{frame_profile.get('scene_reproject_roi_pre_right_ratio', 0.0) * 100.0:.0f}->"
            f"{frame_profile.get('scene_reproject_roi_post_right_ratio', 0.0) * 100.0:.0f}% "
            f"bg_roi=L{frame_profile.get('scene_reproject_background_roi_pre_left_ratio', 0.0) * 100.0:.0f}->"
            f"{frame_profile.get('scene_reproject_background_roi_post_left_ratio', 0.0) * 100.0:.0f}%/"
            f"R{frame_profile.get('scene_reproject_background_roi_pre_right_ratio', 0.0) * 100.0:.0f}->"
            f"{frame_profile.get('scene_reproject_background_roi_post_right_ratio', 0.0) * 100.0:.0f}% "
            f"table=L{frame_profile.get('scene_render_table_left_wall', 0.0) * 1000.0:.2f}/"
            f"R{frame_profile.get('scene_render_table_right_wall', 0.0) * 1000.0:.2f}ms "
            f"table_comp=L{frame_profile.get('scene_compose_table_left_cuda', 0.0) * 1000.0:.2f}/"
            f"R{frame_profile.get('scene_compose_table_right_cuda', 0.0) * 1000.0:.2f}ms "
            f"table_roi=L{frame_profile.get('scene_table_roi_left_ratio', 0.0) * 100.0:.1f}/"
            f"R{frame_profile.get('scene_table_roi_right_ratio', 0.0) * 100.0:.1f}% "
            f"table_ss={frame_profile.get('scene_table_roi_supersample_scale', 0.0):.2f} "
            f"table_ff=L{frame_profile.get('scene_table_fullframe_fallback_left_ratio', 0.0) * 100.0:.0f}/"
            f"R{frame_profile.get('scene_table_fullframe_fallback_right_ratio', 0.0) * 100.0:.0f}% "
            f"gaussian=L{frame_profile.get('gaussian_render_left_cuda', 0.0) * 1000.0:.2f}/"
            f"R{frame_profile.get('gaussian_render_right_cuda', 0.0) * 1000.0:.2f}ms "
            f"compose=L{frame_profile.get('compose_left_cuda', 0.0) * 1000.0:.2f}/"
            f"R{frame_profile.get('compose_right_cuda', 0.0) * 1000.0:.2f}ms "
            f"gaussian_roi=L{frame_profile.get('gaussian_compose_roi_left_ratio', 0.0) * 100.0:.1f}/"
            f"R{frame_profile.get('gaussian_compose_roi_right_ratio', 0.0) * 100.0:.1f}% "
            f"diag=rawL{frame_profile.get('gaussian_raw_left_ratio', 0.0) * 100.0:.2f}->"
            f"{frame_profile.get('gaussian_visible_left_ratio', 0.0) * 100.0:.2f}%/"
            f"R{frame_profile.get('gaussian_raw_right_ratio', 0.0) * 100.0:.2f}->"
            f"{frame_profile.get('gaussian_visible_right_ratio', 0.0) * 100.0:.2f}% "
            f"depth=L{frame_profile.get('scene_depth_finite_left_ratio', 0.0) * 100.0:.0f}/"
            f"{frame_profile.get('scene_depth_positive_left_ratio', 0.0) * 100.0:.0f}% "
            f"R{frame_profile.get('scene_depth_finite_right_ratio', 0.0) * 100.0:.0f}/"
            f"{frame_profile.get('scene_depth_positive_right_ratio', 0.0) * 100.0:.0f}% "
            f"fallback={int(frame_profile.get('compose_fallback_active_ratio', 0.0) >= 0.5)} "
            f"overlay=proj{frame_profile.get('overlay_projection_wall', 0.0) * 1000.0:.2f} "
            f"drawL{frame_profile.get('overlay_draw_left_wall', 0.0) * 1000.0:.2f} "
            f"drawR{frame_profile.get('overlay_draw_right_wall', 0.0) * 1000.0:.2f}ms "
            f"publish={frame_profile.get('publish_total_wall', 0.0) * 1000.0:.2f}ms "
            f"preview={frame_profile.get('preview_window_wall', 0.0) * 1000.0:.2f}ms "
            f"validation={frame_profile.get('grab_validation_wall', 0.0) * 1000.0:.2f}ms "
            f"mem=alloc{frame_profile.get('cuda_memory_allocated_gib', 0.0):.2f}/"
            f"res{frame_profile.get('cuda_memory_reserved_gib', 0.0):.2f}GiB",
            flush=True,
        )

    def _log_quest_render_profile_frame(self, mode, frame_index, frame_profile):
        print(
            f"[render_profile][{mode}] "
            f"frame={int(frame_index)} "
            f"rendering={frame_profile.get('rendering', 0.0) * 1000.0:.2f}ms "
            f"frame_comp={frame_profile.get('frame_compositing', 0.0) * 1000.0:.2f}ms "
            f"gpu={frame_profile.get('frame_compositing_gpu_timer', 0.0) * 1000.0:.2f}ms "
            f"publish={frame_profile.get('frame_comp_quest_publish_wall', 0.0) * 1000.0:.2f}ms "
            f"preview={frame_profile.get('frame_comp_preview_path_wall', 0.0) * 1000.0:.2f}ms "
            f"overlay={frame_profile.get('frame_comp_overlay_draw_submit', 0.0) * 1000.0:.2f}ms "
            f"poll={frame_profile.get('frame_comp_glfw_poll_wall', 0.0) * 1000.0:.2f}ms",
            flush=True,
        )

    def _blend_ellipse(
        self,
        frame,
        center_pixel,
        color,
        radius_x,
        radius_y,
        axis_pixel=None,
        blend=0.50,
    ):
        height, width = frame.shape[:2]
        center_x = float(center_pixel[0].item())
        center_y = float(center_pixel[1].item())
        if center_x < 0.0 or center_x >= width or center_y < 0.0 or center_y >= height:
            return

        if axis_pixel is None:
            axis = frame.new_tensor([1.0, 0.0], dtype=torch.float32)
        else:
            axis = axis_pixel - center_pixel
            axis_norm = torch.linalg.norm(axis)
            if float(axis_norm.item()) < 1e-6:
                axis = frame.new_tensor([1.0, 0.0], dtype=torch.float32)
            else:
                axis = axis / axis_norm
        perp = frame.new_tensor([-axis[1].item(), axis[0].item()], dtype=torch.float32)

        max_radius = max(radius_x, radius_y)
        x0 = max(0, int(np.floor(center_x - max_radius - 1)))
        x1 = min(width, int(np.ceil(center_x + max_radius + 2)))
        y0 = max(0, int(np.floor(center_y - max_radius - 1)))
        y1 = min(height, int(np.ceil(center_y + max_radius + 2)))
        if x0 >= x1 or y0 >= y1:
            return

        ys = torch.arange(y0, y1, device=frame.device, dtype=torch.float32)
        xs = torch.arange(x0, x1, device=frame.device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        delta_x = grid_x - center_x
        delta_y = grid_y - center_y
        along = delta_x * axis[0] + delta_y * axis[1]
        across = delta_x * perp[0] + delta_y * perp[1]
        ellipse_mask = ((along / float(radius_x)) ** 2 + (across / float(radius_y)) ** 2) <= 1.0
        if not bool(ellipse_mask.any().item()):
            return

        rgb_frame = frame[..., :3]
        color_tensor = rgb_frame.new_tensor(color[:3])
        patch = rgb_frame[y0:y1, x0:x1]
        patch[ellipse_mask] = patch[ellipse_mask] * (1.0 - blend) + color_tensor * blend

    def _draw_marker_line(self, frame, start_pixel, end_pixel, color, radius=1, blend=0.45):
        height, width = frame.shape[:2]
        start_x = float(start_pixel[0].item())
        start_y = float(start_pixel[1].item())
        end_x = float(end_pixel[0].item())
        end_y = float(end_pixel[1].item())

        min_x = max(0, int(np.floor(min(start_x, end_x) - radius - 1)))
        max_x = min(width, int(np.ceil(max(start_x, end_x) + radius + 2)))
        min_y = max(0, int(np.floor(min(start_y, end_y) - radius - 1)))
        max_y = min(height, int(np.ceil(max(start_y, end_y) + radius + 2)))
        if min_x >= max_x or min_y >= max_y:
            return

        seg_x = end_x - start_x
        seg_y = end_y - start_y
        seg_len_sq = seg_x * seg_x + seg_y * seg_y
        if seg_len_sq < 1e-6:
            self._blend_marker(frame, start_pixel, color, radius=radius, blend=blend)
            return

        ys = torch.arange(min_y, max_y, device=frame.device, dtype=torch.float32)
        xs = torch.arange(min_x, max_x, device=frame.device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        rel_x = grid_x - start_x
        rel_y = grid_y - start_y
        t = (rel_x * seg_x + rel_y * seg_y) / seg_len_sq
        t = torch.clamp(t, 0.0, 1.0)
        closest_x = start_x + t * seg_x
        closest_y = start_y + t * seg_y
        dist_x = torch.abs(grid_x - closest_x)
        dist_y = torch.abs(grid_y - closest_y)
        line_mask = torch.maximum(dist_x, dist_y) <= (float(radius) + 0.5)
        if not bool(line_mask.any().item()):
            return

        rgb_frame = frame[..., :3]
        color_tensor = rgb_frame.new_tensor(color[:3])
        patch = rgb_frame[min_y:max_y, min_x:max_x]
        patch[line_mask] = patch[line_mask] * (1.0 - blend) + color_tensor * blend

    def _mirror_symbol_about_palm(self, projected):
        mirrored = {"palm": projected["palm"]}
        palm = projected["palm"]
        for key in ("thumb_tip", "index_tip", "middle_tip", "pinch_point"):
            point = projected.get(key)
            if point is None:
                mirrored[key] = None
            else:
                mirrored[key] = palm - (point - palm)
        return mirrored

    def _resolve_live_hand_overlay_colors(self, hand_overlays, width, side_memory):
        if not hand_overlays:
            return hand_overlays

        if len(hand_overlays) >= 2:
            ordered = sorted(hand_overlays, key=lambda entry: float(entry["palm_pixel"][0].item()))
            left_entry = ordered[0]
            right_entry = ordered[-1]
            left_entry["color"] = self.LIVE_HAND_LEFT_COLOR
            right_entry["color"] = self.LIVE_HAND_RIGHT_COLOR
            if side_memory[left_entry["source"]] is None:
                side_memory[left_entry["source"]] = "left"
            if side_memory[right_entry["source"]] is None:
                side_memory[right_entry["source"]] = "right"
            return hand_overlays

        entry = hand_overlays[0]
        remembered_side = side_memory[entry["source"]]
        if remembered_side is None:
            remembered_side = (
                "left"
                if float(entry["palm_pixel"][0].item()) < (0.5 * float(width))
                else "right"
            )
            side_memory[entry["source"]] = remembered_side

        entry["color"] = (
            self.LIVE_HAND_LEFT_COLOR
            if remembered_side == "left"
            else self.LIVE_HAND_RIGHT_COLOR
        )
        return hand_overlays

    def _draw_live_hand_overlay(self, frame, hand_overlays):
        for overlay in hand_overlays:
            color = overlay["color"]
            projected = self._mirror_symbol_about_palm(overlay["projected"])
            palm = projected["palm"]
            thumb_tip = projected.get("thumb_tip")
            index_tip = projected.get("index_tip")
            middle_tip = projected.get("middle_tip")
            pinch_point = projected.get("pinch_point")

            palm_axis = pinch_point
            if palm_axis is None:
                palm_axis = index_tip if index_tip is not None else thumb_tip

            self._blend_ellipse(
                frame,
                palm,
                color,
                radius_x=self.LIVE_HAND_PALM_RADIUS_X,
                radius_y=self.LIVE_HAND_PALM_RADIUS_Y,
                axis_pixel=palm_axis,
                blend=0.42,
            )

            for branch_tip, branch_blend in (
                (thumb_tip, 0.35),
                (index_tip, 0.42),
                (middle_tip, 0.24),
            ):
                if branch_tip is None:
                    continue
                self._draw_marker_line(
                    frame,
                    palm,
                    branch_tip,
                    color,
                    radius=self.LIVE_HAND_LINE_RADIUS,
                    blend=branch_blend,
                )

            if thumb_tip is not None:
                self._blend_marker(
                    frame,
                    thumb_tip,
                    color,
                    radius=self.LIVE_HAND_TIP_RADIUS,
                    blend=0.78,
                )
            if index_tip is not None:
                self._blend_marker(
                    frame,
                    index_tip,
                    color,
                    radius=self.LIVE_HAND_TIP_RADIUS,
                    blend=0.82,
                )
            if middle_tip is not None:
                self._blend_marker(
                    frame,
                    middle_tip,
                    color,
                    radius=1,
                    blend=0.55,
                )
            if pinch_point is not None:
                self._blend_marker(
                    frame,
                    pinch_point,
                    color,
                    radius=self.LIVE_HAND_PINCH_RADIUS,
                    blend=0.90,
                )
                self._blend_marker(frame, pinch_point, [255.0, 255.0, 255.0], radius=1, blend=0.95)

    def _controller_sample_has_position(self, controller_sample, pose_role="selected"):
        return controller_pose_position(controller_sample, pose_role) is not None

    def _controller_sample_is_renderable(self, controller_sample, pose_role="selected"):
        return (
            self._controller_sample_has_position(controller_sample, pose_role)
            and controller_pose_forward(controller_sample, pose_role) is not None
        )

    def _controller_world_ray_pose(self, controller_world):
        if controller_world is None:
            return None, None
        origin_world = controller_world.get("ray_origin", controller_world.get("position"))
        direction_world = controller_world.get("ray_direction", controller_world.get("direction"))
        if origin_world is None or direction_world is None:
            return origin_world, direction_world
        norm = torch.linalg.norm(direction_world)
        if float(norm.item()) < 1e-6:
            return origin_world, None
        return origin_world, direction_world / norm

    def _controller_camera_basis(self, w2c):
        if torch.is_tensor(w2c):
            w2c_np = w2c.detach().cpu().numpy()
        else:
            w2c_np = np.asarray(w2c, dtype=np.float32)

        camera_to_world = w2c_np[:3, :3].T
        screen_right_world = camera_to_world[:, 0]
        screen_up_world = -camera_to_world[:, 1]
        scene_forward_world = -camera_to_world[:, 2]
        basis = np.stack(
            [screen_right_world, screen_up_world, scene_forward_world],
            axis=1,
        ).astype(np.float32)
        return torch.as_tensor(basis, dtype=torch.float32, device=cfg.device)

    def _scene_world_up_vector_np(self):
        return np.array(
            [0.0, 0.0, -1.0 if cfg.reverse_z else 1.0],
            dtype=np.float32,
        )

    def _scene_world_up_vector_torch(self, device=None, dtype=torch.float32):
        if device is None:
            device = cfg.device
        return torch.as_tensor(
            self._scene_world_up_vector_np(),
            dtype=dtype,
            device=device,
        )

    def _scene_world_forward_vector_np(self):
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)

    def _normalize_numpy_vector(self, vector, fallback):
        vector = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if norm < 1e-6:
            return np.asarray(fallback, dtype=np.float32).copy()
        return vector / norm

    def _normalize_torch_vector(self, vector, fallback):
        vector = torch.as_tensor(vector, dtype=torch.float32, device=cfg.device)
        fallback = torch.as_tensor(fallback, dtype=vector.dtype, device=vector.device)
        norm = torch.linalg.norm(vector)
        if float(norm.item()) < 1e-6:
            return fallback.clone()
        return vector / norm

    def _axis_angle_rotation_matrix_torch(self, axis, angle, device=None, dtype=torch.float32):
        if device is None:
            device = cfg.device
        axis = torch.as_tensor(axis, dtype=dtype, device=device)
        axis = axis / torch.linalg.norm(axis).clamp_min(1e-6)
        angle = torch.as_tensor(angle, dtype=dtype, device=device)
        x, y, z = axis.unbind()
        c = torch.cos(angle)
        s = torch.sin(angle)
        one_c = 1.0 - c
        return torch.stack(
            [
                torch.stack([c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s]),
                torch.stack([y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s]),
                torch.stack([z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c]),
            ],
            dim=0,
        )

    def _rotate_points_with_matrix(self, points, rotation_matrix, pivot=None):
        points = torch.as_tensor(points, dtype=rotation_matrix.dtype, device=rotation_matrix.device)
        if pivot is None:
            pivot = torch.zeros(3, dtype=rotation_matrix.dtype, device=rotation_matrix.device)
        else:
            pivot = torch.as_tensor(pivot, dtype=rotation_matrix.dtype, device=rotation_matrix.device)
        return (points - pivot.unsqueeze(0)) @ rotation_matrix.T + pivot.unsqueeze(0)

    def _gaussian_quaternion_multiply_wxyz(self, lhs, rhs):
        lhs = torch.as_tensor(lhs, dtype=torch.float32, device=cfg.device)
        rhs = torch.as_tensor(rhs, dtype=torch.float32, device=cfg.device)
        lhs_shape = lhs.shape
        rhs_shape = rhs.shape
        lhs = lhs.reshape(-1, 4)
        rhs = rhs.reshape(-1, 4)
        if lhs.shape[0] == 1 and rhs.shape[0] > 1:
            lhs = lhs.expand(rhs.shape[0], 4)
        if rhs.shape[0] == 1 and lhs.shape[0] > 1:
            rhs = rhs.expand(lhs.shape[0], 4)
        w1, x1, y1, z1 = lhs.unbind(dim=1)
        w2, x2, y2, z2 = rhs.unbind(dim=1)
        product = torch.stack(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dim=1,
        )
        if len(lhs_shape) == 1 and len(rhs_shape) == 1:
            return product[0]
        return product

    def _axis_angle_to_gaussian_quaternion_wxyz(self, axis, angle, device=None, dtype=torch.float32):
        if device is None:
            device = cfg.device
        axis = torch.as_tensor(axis, dtype=dtype, device=device)
        axis = axis / torch.linalg.norm(axis).clamp_min(1e-6)
        half_angle = 0.5 * torch.as_tensor(angle, dtype=dtype, device=device)
        sin_half = torch.sin(half_angle)
        return torch.stack(
            [
                torch.cos(half_angle),
                axis[0] * sin_half,
                axis[1] * sin_half,
                axis[2] * sin_half,
            ],
            dim=0,
        )

    def _rotate_gaussian_quaternions_about_axis(self, quaternions, axis, angle):
        quaternions = torch.as_tensor(quaternions, dtype=torch.float32, device=cfg.device)
        rotation_quaternion = self._axis_angle_to_gaussian_quaternion_wxyz(
            axis,
            angle,
            device=quaternions.device,
            dtype=quaternions.dtype,
        )
        rotated = self._gaussian_quaternion_multiply_wxyz(
            rotation_quaternion.unsqueeze(0),
            quaternions,
        )
        return F.normalize(rotated, dim=-1)

    def _is_finite_tensor(self, tensor):
        return bool(torch.isfinite(torch.as_tensor(tensor)).all().item())

    def _compute_live_controller_alignment(
        self,
        live_left_anchor,
        live_right_anchor,
        recorded_left_anchor,
        recorded_right_anchor,
        w2c,
        basis_override=None,
    ):
        live_left_anchor = torch.as_tensor(
            np.asarray(live_left_anchor, dtype=np.float32),
            dtype=torch.float32,
            device=cfg.device,
        )
        live_right_anchor = torch.as_tensor(
            np.asarray(live_right_anchor, dtype=np.float32),
            dtype=torch.float32,
            device=cfg.device,
        )
        recorded_left_anchor = torch.as_tensor(
            recorded_left_anchor,
            dtype=torch.float32,
            device=cfg.device,
        )
        recorded_right_anchor = torch.as_tensor(
            recorded_right_anchor,
            dtype=torch.float32,
            device=cfg.device,
        )
        basis = (
            basis_override
            if basis_override is not None
            else self._controller_camera_basis(w2c)
        )
        return {
            "basis": basis,
            "translation_scale": torch.tensor(
                self._live_controller_case_profile()["controller_translation_scale"],
                dtype=torch.float32,
                device=cfg.device,
            ),
            "reference_live_left": live_left_anchor,
            "reference_live_right": live_right_anchor,
            "reference_scene_left": recorded_left_anchor,
            "reference_scene_right": recorded_right_anchor,
        }

    def _compute_single_controller_alignment(
        self,
        live_anchor,
        recorded_anchor,
        w2c,
        basis_override=None,
    ):
        live_anchor = torch.as_tensor(
            np.asarray(live_anchor, dtype=np.float32),
            dtype=torch.float32,
            device=cfg.device,
        )
        recorded_anchor = torch.as_tensor(
            recorded_anchor,
            dtype=torch.float32,
            device=cfg.device,
        )
        basis = (
            basis_override
            if basis_override is not None
            else self._controller_camera_basis(w2c)
        )
        return {
            "basis": basis,
            "translation_scale": torch.tensor(
                self._live_controller_case_profile()["controller_translation_scale"],
                dtype=torch.float32,
                device=cfg.device,
            ),
            "reference_live_left": live_anchor,
            "reference_live_right": live_anchor,
            "reference_scene_left": recorded_anchor,
            "reference_scene_right": recorded_anchor,
        }

    def _update_live_controller_alignment(
        self,
        alignment,
        alignment_mode,
        live_sample,
        recorded_anchor_centers,
        w2c,
        basis_override=None,
        pose_role="selected",
    ):
        left_anchor = controller_pose_position(live_sample.left, pose_role)
        right_anchor = controller_pose_position(live_sample.right, pose_role)

        if left_anchor is not None and right_anchor is not None:
            if alignment_mode != "dual":
                alignment = self._compute_live_controller_alignment(
                    left_anchor,
                    right_anchor,
                    recorded_anchor_centers[0],
                    recorded_anchor_centers[1],
                    w2c,
                    basis_override=basis_override,
                )
                alignment_mode = "dual"
            return alignment, alignment_mode

        if left_anchor is not None:
            if alignment_mode != "single_left":
                alignment = self._compute_single_controller_alignment(
                    left_anchor,
                    recorded_anchor_centers[0],
                    w2c,
                    basis_override=basis_override,
                )
                alignment_mode = "single_left"
            return alignment, alignment_mode

        if right_anchor is not None:
            if alignment_mode != "single_right":
                alignment = self._compute_single_controller_alignment(
                    right_anchor,
                    recorded_anchor_centers[1],
                    w2c,
                    basis_override=basis_override,
                )
                alignment_mode = "single_right"
            return alignment, alignment_mode

        return alignment, alignment_mode

    def _transform_live_controller_delta(self, delta, alignment):
        transformed = alignment["basis"] @ delta
        return transformed * alignment["translation_scale"]

    def _transform_live_controller_direction(self, direction, alignment):
        transformed = alignment["basis"] @ direction
        norm = torch.linalg.norm(transformed)
        if float(norm.item()) < 1e-6:
            return None
        return transformed / norm

    def _convert_live_controller_to_world(
        self,
        source,
        controller_sample,
        alignment,
        position_pose_role="selected",
        ray_pose_role=None,
    ):
        if alignment is None or controller_sample is None or not controller_sample.active:
            return None

        if ray_pose_role is None:
            ray_pose_role = position_pose_role

        position_np = controller_pose_position(controller_sample, position_pose_role)
        if position_np is None:
            return None

        ray_origin_np = controller_pose_position(controller_sample, ray_pose_role)
        if ray_origin_np is None:
            ray_origin_np = position_np

        direction_np = controller_pose_forward(controller_sample, ray_pose_role)
        if direction_np is None:
            direction_np = controller_pose_forward(controller_sample, position_pose_role)
        if direction_np is None:
            return None

        position = torch.from_numpy(np.asarray(position_np, dtype=np.float32)).to(
            device=cfg.device, dtype=torch.float32
        )
        ray_origin = torch.from_numpy(np.asarray(ray_origin_np, dtype=np.float32)).to(
            device=cfg.device, dtype=torch.float32
        )
        direction = torch.from_numpy(np.asarray(direction_np, dtype=np.float32)).to(
            device=cfg.device, dtype=torch.float32
        )
        if source == "left":
            reference_live = alignment["reference_live_left"]
            reference_scene = alignment["reference_scene_left"]
        else:
            reference_live = alignment["reference_live_right"]
            reference_scene = alignment["reference_scene_right"]
        world_position = reference_scene + self._transform_live_controller_delta(
            position - reference_live,
            alignment,
        )
        world_ray_origin = reference_scene + self._transform_live_controller_delta(
            ray_origin - reference_live,
            alignment,
        )
        world_ray_direction = self._transform_live_controller_direction(direction, alignment)
        if world_ray_direction is None:
            return None

        return {
            "source": controller_sample.source,
            "position": world_position,
            "direction": world_ray_direction,
            "ray_origin": world_ray_origin,
            "ray_direction": world_ray_direction,
            "select_available": controller_sample.select_available,
            "select_pressed": controller_sample.select_pressed,
            "select_value": controller_sample.select_value,
            "select_source": controller_sample.select_source,
            "anchor_cycle_available": controller_sample.anchor_cycle_available,
            "anchor_cycle_pressed": controller_sample.anchor_cycle_pressed,
            "anchor_cycle_source": controller_sample.anchor_cycle_source,
            "snap_assist_available": controller_sample.snap_assist_available,
            "snap_assist_pressed": controller_sample.snap_assist_pressed,
            "snap_assist_source": controller_sample.snap_assist_source,
            "exit_available": getattr(controller_sample, "exit_available", False),
            "exit_pressed": getattr(controller_sample, "exit_pressed", False),
            "exit_source": getattr(controller_sample, "exit_source", "none"),
            "position_pose_role": position_pose_role,
            "ray_pose_role": ray_pose_role,
            "grip_pose_valid": bool(
                controller_sample.grip_active
                and controller_sample.grip_position_valid
                and controller_sample.grip_orientation_valid
            )
            ,
            "aim_pose_valid": bool(
                controller_sample.aim_active
                and controller_sample.aim_position_valid
                and controller_sample.aim_orientation_valid
            )
            ,
        }

    def _quaternion_xyzw_to_rotation_matrix(self, quaternion):
        x, y, z, w = [float(v) for v in quaternion]
        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        xz = x * z
        yz = y * z
        wx = w * x
        wy = w * y
        wz = w * z
        return np.array(
            [
                [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
                [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
                [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
            ],
            dtype=np.float32,
        )

    def _controller_alignment_reference_centers(self, alignment):
        if alignment is None:
            return None, None
        live_center = 0.5 * (
            alignment["reference_live_left"] + alignment["reference_live_right"]
        )
        scene_center = 0.5 * (
            alignment["reference_scene_left"] + alignment["reference_scene_right"]
        )
        return live_center, scene_center

    def _controller_basis_with_lateral_flip(self, basis, lateral_flip=False):
        basis_t = torch.as_tensor(
            basis,
            dtype=torch.float32,
            device=cfg.device,
        ).clone()
        if lateral_flip:
            basis_t[:, 0] = -basis_t[:, 0]
        return basis_t

    def _make_immersive_controller_basis_state(self, head_basis, intrinsic):
        intrinsic_np = (
            intrinsic.detach().cpu().numpy()
            if torch.is_tensor(intrinsic)
            else np.asarray(intrinsic, dtype=np.float32)
        )
        head_basis_t = self._controller_basis_with_lateral_flip(
            head_basis,
            lateral_flip=False,
        )
        return {
            "state": "pending",
            "resolved": False,
            "basis": head_basis_t.clone(),
            "head_basis": head_basis_t,
            "lateral_flip_applied": False,
            "screen_center_x": float(intrinsic_np[0, 2]),
            "validation_mode": "pending",
            "validation_sample_id": None,
            "live_lateral_delta_x": None,
            "provisional_left_body_x": None,
            "provisional_right_body_x": None,
            "provisional_left_ray_x": None,
            "provisional_right_ray_x": None,
            "final_left_body_x": None,
            "final_right_body_x": None,
            "final_left_ray_x": None,
            "final_right_ray_x": None,
            "candidate_flip": None,
            "candidate_streak": 0,
            "resolution_start_sample": None,
            "resolution_deadline_sample": None,
            "lock_reason": None,
            "lock_sample_id": None,
            "late_controller_seen_after_lock": [],
            "locked_active_sources": [],
        }

    def _project_controller_world_field_to_startup_screen_x(
        self,
        controller_world,
        field_name,
        intrinsic,
        w2c,
    ):
        if controller_world is None:
            return None, None, False
        point_world = controller_world.get(field_name)
        if point_world is None:
            return None, None, False
        return self._project_world_point_to_startup_screen_x(
            point_world,
            intrinsic,
            w2c,
        )

    def _compute_live_controller_alignment_from_sample(
        self,
        latest_controller_sample,
        current_alignment,
        current_alignment_mode,
        controller_source_anchor_centers,
        w2c,
        basis_override=None,
        pose_role="selected",
    ):
        if latest_controller_sample is None:
            return current_alignment, current_alignment_mode

        left_anchor = controller_pose_position(latest_controller_sample.left, pose_role)
        right_anchor = controller_pose_position(latest_controller_sample.right, pose_role)
        if left_anchor is not None and right_anchor is not None:
            return (
                self._compute_live_controller_alignment(
                    left_anchor,
                    right_anchor,
                    controller_source_anchor_centers[0],
                    controller_source_anchor_centers[1],
                    w2c,
                    basis_override=basis_override,
                ),
                "dual",
            )
        if left_anchor is not None:
            return (
                self._compute_single_controller_alignment(
                    left_anchor,
                    controller_source_anchor_centers[0],
                    w2c,
                    basis_override=basis_override,
                ),
                "single_left",
            )
        if right_anchor is not None:
            return (
                self._compute_single_controller_alignment(
                    right_anchor,
                    controller_source_anchor_centers[1],
                    w2c,
                    basis_override=basis_override,
                ),
                "single_right",
            )
        return current_alignment, current_alignment_mode

    def _copy_live_controller_runtime_metadata(
        self,
        recomputed_controller_world,
        original_controller_world,
    ):
        if recomputed_controller_world is None or original_controller_world is None:
            return recomputed_controller_world
        for key in (
            "sample_id",
            "select_start_edge",
            "select_hold_active",
            "select_release_ready",
            "select_release_frames",
            "select_start_active",
        ):
            if key in original_controller_world:
                recomputed_controller_world[key] = original_controller_world[key]
        return recomputed_controller_world

    def _recompute_live_controller_runtime_state_for_basis(
        self,
        latest_controller_sample,
        controller_runtime_state,
        controller_source_anchor_centers,
        w2c,
        basis_override,
        alignment_pose_role="selected",
        controller_position_pose_role="selected",
        controller_ray_pose_role=None,
    ):
        if latest_controller_sample is None or controller_runtime_state is None:
            return controller_runtime_state

        recomputed_alignment, recomputed_alignment_mode = (
            self._compute_live_controller_alignment_from_sample(
                latest_controller_sample,
                controller_runtime_state.get("alignment"),
                controller_runtime_state.get("alignment_mode", "unset"),
                controller_source_anchor_centers,
                w2c,
                basis_override=basis_override,
                pose_role=alignment_pose_role,
            )
        )
        recomputed_left_controller = self._convert_live_controller_to_world(
            "left",
            latest_controller_sample.left,
            recomputed_alignment,
            position_pose_role=controller_position_pose_role,
            ray_pose_role=controller_ray_pose_role,
        )
        recomputed_right_controller = self._convert_live_controller_to_world(
            "right",
            latest_controller_sample.right,
            recomputed_alignment,
            position_pose_role=controller_position_pose_role,
            ray_pose_role=controller_ray_pose_role,
        )
        recomputed_left_controller = self._copy_live_controller_runtime_metadata(
            recomputed_left_controller,
            controller_runtime_state.get("left_controller"),
        )
        recomputed_right_controller = self._copy_live_controller_runtime_metadata(
            recomputed_right_controller,
            controller_runtime_state.get("right_controller"),
        )
        recomputed_runtime_state = dict(controller_runtime_state)
        recomputed_runtime_state["alignment"] = recomputed_alignment
        recomputed_runtime_state["alignment_mode"] = recomputed_alignment_mode
        recomputed_runtime_state["left_controller"] = recomputed_left_controller
        recomputed_runtime_state["right_controller"] = recomputed_right_controller
        return recomputed_runtime_state

    def _log_immersive_controller_handedness_resolution(self, controller_basis_state):
        def _fmt(value):
            return "none" if value is None else f"{float(value):.2f}"

        basis_lateral = (
            controller_basis_state["basis"][:, 0]
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )
        print(
            "[live_openxr_controller] immersive controller handedness validation: "
            f"sample={controller_basis_state['validation_sample_id']} "
            f"state={controller_basis_state.get('state', 'unknown')} "
            f"mode={controller_basis_state['validation_mode']} "
            f"candidate_streak={int(controller_basis_state.get('candidate_streak', 0))} "
            f"lock_reason={controller_basis_state.get('lock_reason')} "
            f"screen_center_x={controller_basis_state['screen_center_x']:.2f} "
            f"live_lateral_delta_x={_fmt(controller_basis_state['live_lateral_delta_x'])} "
            f"provisional_left_body_x={_fmt(controller_basis_state['provisional_left_body_x'])} "
            f"provisional_right_body_x={_fmt(controller_basis_state['provisional_right_body_x'])} "
            f"provisional_left_ray_x={_fmt(controller_basis_state['provisional_left_ray_x'])} "
            f"provisional_right_ray_x={_fmt(controller_basis_state['provisional_right_ray_x'])} "
            f"flip={int(bool(controller_basis_state['lateral_flip_applied']))} "
            f"final_left_body_x={_fmt(controller_basis_state['final_left_body_x'])} "
            f"final_right_body_x={_fmt(controller_basis_state['final_right_body_x'])} "
            f"final_left_ray_x={_fmt(controller_basis_state['final_left_ray_x'])} "
            f"final_right_ray_x={_fmt(controller_basis_state['final_right_ray_x'])} "
            f"controller_basis_lateral={basis_lateral}",
            flush=True,
        )

    def _controller_basis_active_sources(self, controller_runtime_state):
        if controller_runtime_state is None:
            return []
        active_sources = []
        for source in ("left", "right"):
            if controller_runtime_state.get(f"{source}_controller") is not None:
                active_sources.append(source)
        return active_sources

    def _controller_basis_interaction_started(
        self,
        controller_runtime_state,
        controller_interaction_state,
    ):
        if controller_runtime_state is not None:
            for source in ("left", "right"):
                controller_world = controller_runtime_state.get(f"{source}_controller")
                if controller_world is None:
                    continue
                if bool(controller_world.get("select_start_edge", False)):
                    return True
                if bool(controller_world.get("select_hold_active", False)):
                    return True
        if controller_interaction_state is not None:
            for source in ("left", "right"):
                if controller_interaction_state.get(source) is not None:
                    return True
        return False

    def _lock_immersive_controller_basis_state(
        self,
        controller_basis_state,
        sample_id,
        lock_reason,
        validation_mode=None,
        basis=None,
        lateral_flip_applied=None,
        controller_runtime_state=None,
    ):
        updated_controller_basis_state = dict(controller_basis_state)
        updated_controller_basis_state["state"] = "locked"
        updated_controller_basis_state["resolved"] = True
        updated_controller_basis_state["lock_reason"] = lock_reason
        updated_controller_basis_state["lock_sample_id"] = int(sample_id)
        updated_controller_basis_state["validation_sample_id"] = int(sample_id)
        updated_controller_basis_state["locked_active_sources"] = (
            self._controller_basis_active_sources(controller_runtime_state)
        )
        if validation_mode is not None:
            updated_controller_basis_state["validation_mode"] = validation_mode
        if basis is not None:
            updated_controller_basis_state["basis"] = basis
        if lateral_flip_applied is not None:
            updated_controller_basis_state["lateral_flip_applied"] = bool(
                lateral_flip_applied
            )
        return updated_controller_basis_state

    def _maybe_note_late_controller_seen_after_lock(
        self,
        controller_runtime_state,
        controller_basis_state,
    ):
        if (
            controller_runtime_state is None
            or controller_basis_state is None
            or controller_basis_state.get("state") != "locked"
        ):
            return controller_basis_state
        updated_controller_basis_state = dict(controller_basis_state)
        seen_after_lock = list(
            updated_controller_basis_state.get("late_controller_seen_after_lock", [])
        )
        locked_active_sources = set(
            updated_controller_basis_state.get("locked_active_sources", [])
        )
        for source in self._controller_basis_active_sources(controller_runtime_state):
            if source in locked_active_sources or source in seen_after_lock:
                continue
            seen_after_lock.append(source)
            print(
                "[live_openxr_controller] "
                f"immersive late_controller_seen_after_lock source={source} "
                "basis_reused=1",
                flush=True,
            )
        updated_controller_basis_state["late_controller_seen_after_lock"] = (
            seen_after_lock
        )
        return updated_controller_basis_state

    def _maybe_resolve_immersive_controller_handedness(
        self,
        latest_controller_sample,
        controller_runtime_state,
        controller_basis_state,
        controller_source_anchor_centers,
        intrinsic,
        w2c,
        controller_interaction_state=None,
        alignment_pose_role="selected",
        controller_position_pose_role="selected",
        controller_ray_pose_role=None,
    ):
        if (
            latest_controller_sample is None
            or controller_runtime_state is None
            or controller_basis_state is None
        ):
            return controller_runtime_state, controller_basis_state
        if controller_basis_state.get("state") == "locked":
            return (
                controller_runtime_state,
                self._maybe_note_late_controller_seen_after_lock(
                    controller_runtime_state,
                    controller_basis_state,
                ),
            )

        sample_id = int(latest_controller_sample.sample)
        updated_controller_basis_state = dict(controller_basis_state)
        if updated_controller_basis_state.get("resolution_start_sample") is None:
            updated_controller_basis_state["resolution_start_sample"] = sample_id
            updated_controller_basis_state["resolution_deadline_sample"] = (
                sample_id + int(self.IMMERSIVE_CONTROLLER_HANDNESS_MAX_VALID_SAMPLES)
            )

        if self._controller_basis_interaction_started(
            controller_runtime_state,
            controller_interaction_state,
        ):
            updated_controller_basis_state = self._lock_immersive_controller_basis_state(
                updated_controller_basis_state,
                sample_id=sample_id,
                lock_reason="interaction_started",
                validation_mode="interaction_started_default",
                controller_runtime_state=controller_runtime_state,
            )
            self._log_immersive_controller_handedness_resolution(
                updated_controller_basis_state
            )
            return controller_runtime_state, updated_controller_basis_state

        left_grip_position = controller_pose_position(
            latest_controller_sample.left,
            alignment_pose_role,
        )
        right_grip_position = controller_pose_position(
            latest_controller_sample.right,
            alignment_pose_role,
        )
        if left_grip_position is None or right_grip_position is None:
            if sample_id >= int(updated_controller_basis_state["resolution_deadline_sample"]):
                updated_controller_basis_state = self._lock_immersive_controller_basis_state(
                    updated_controller_basis_state,
                    sample_id=sample_id,
                    lock_reason="timeout_default",
                    validation_mode="timeout_default",
                    controller_runtime_state=controller_runtime_state,
                )
                self._log_immersive_controller_handedness_resolution(
                    updated_controller_basis_state
                )
            return controller_runtime_state, updated_controller_basis_state

        live_lateral_delta_x = float(
            np.asarray(right_grip_position, dtype=np.float32)[0]
            - np.asarray(left_grip_position, dtype=np.float32)[0]
        )
        if abs(live_lateral_delta_x) < 1e-5:
            updated_controller_basis_state["live_lateral_delta_x"] = live_lateral_delta_x
            if sample_id >= int(updated_controller_basis_state["resolution_deadline_sample"]):
                updated_controller_basis_state = self._lock_immersive_controller_basis_state(
                    updated_controller_basis_state,
                    sample_id=sample_id,
                    lock_reason="timeout_default",
                    validation_mode="timeout_default",
                    controller_runtime_state=controller_runtime_state,
                )
                self._log_immersive_controller_handedness_resolution(
                    updated_controller_basis_state
                )
            return controller_runtime_state, updated_controller_basis_state

        provisional_left_body_x, _, provisional_left_body_valid = (
            self._project_controller_world_field_to_startup_screen_x(
                controller_runtime_state.get("left_controller"),
                "position",
                intrinsic,
                w2c,
            )
        )
        provisional_right_body_x, _, provisional_right_body_valid = (
            self._project_controller_world_field_to_startup_screen_x(
                controller_runtime_state.get("right_controller"),
                "position",
                intrinsic,
                w2c,
            )
        )
        provisional_left_ray_x, _, provisional_left_ray_valid = (
            self._project_controller_world_field_to_startup_screen_x(
                controller_runtime_state.get("left_controller"),
                "ray_origin",
                intrinsic,
                w2c,
            )
        )
        provisional_right_ray_x, _, provisional_right_ray_valid = (
            self._project_controller_world_field_to_startup_screen_x(
                controller_runtime_state.get("right_controller"),
                "ray_origin",
                intrinsic,
                w2c,
            )
        )

        body_pair_valid = provisional_left_body_valid and provisional_right_body_valid
        ray_pair_valid = provisional_left_ray_valid and provisional_right_ray_valid
        updated_controller_basis_state.update(
            {
                "live_lateral_delta_x": live_lateral_delta_x,
                "provisional_left_body_x": provisional_left_body_x,
                "provisional_right_body_x": provisional_right_body_x,
                "provisional_left_ray_x": provisional_left_ray_x,
                "provisional_right_ray_x": provisional_right_ray_x,
            }
        )
        if not body_pair_valid and not ray_pair_valid:
            if sample_id >= int(updated_controller_basis_state["resolution_deadline_sample"]):
                updated_controller_basis_state = self._lock_immersive_controller_basis_state(
                    updated_controller_basis_state,
                    sample_id=sample_id,
                    lock_reason="timeout_default",
                    validation_mode="timeout_default",
                    controller_runtime_state=controller_runtime_state,
                )
                self._log_immersive_controller_handedness_resolution(
                    updated_controller_basis_state
                )
            return controller_runtime_state, updated_controller_basis_state

        validation_mode = "dual_grip"
        candidate_flip = None
        if body_pair_valid:
            projected_body_delta_x = float(
                provisional_right_body_x - provisional_left_body_x
            )
            candidate_flip = (live_lateral_delta_x * projected_body_delta_x) < 0.0
            validation_mode += "_body"
        elif ray_pair_valid:
            projected_ray_delta_x = float(
                provisional_right_ray_x - provisional_left_ray_x
            )
            candidate_flip = (live_lateral_delta_x * projected_ray_delta_x) < 0.0
            validation_mode += "_ray"
        if candidate_flip is None:
            if sample_id >= int(updated_controller_basis_state["resolution_deadline_sample"]):
                updated_controller_basis_state = self._lock_immersive_controller_basis_state(
                    updated_controller_basis_state,
                    sample_id=sample_id,
                    lock_reason="timeout_default",
                    validation_mode="timeout_default",
                    controller_runtime_state=controller_runtime_state,
                )
                self._log_immersive_controller_handedness_resolution(
                    updated_controller_basis_state
                )
            return controller_runtime_state, updated_controller_basis_state

        candidate_flip = bool(candidate_flip)
        if updated_controller_basis_state.get("candidate_flip") == candidate_flip:
            updated_controller_basis_state["candidate_streak"] = int(
                updated_controller_basis_state.get("candidate_streak", 0)
            ) + 1
        else:
            updated_controller_basis_state["candidate_flip"] = candidate_flip
            updated_controller_basis_state["candidate_streak"] = 1
        updated_controller_basis_state["validation_mode"] = f"pending_{validation_mode}"
        updated_controller_basis_state["validation_sample_id"] = sample_id

        if int(updated_controller_basis_state["candidate_streak"]) < int(
            self.IMMERSIVE_CONTROLLER_HANDNESS_CONFIRM_STREAK
        ):
            if sample_id >= int(updated_controller_basis_state["resolution_deadline_sample"]):
                updated_controller_basis_state = self._lock_immersive_controller_basis_state(
                    updated_controller_basis_state,
                    sample_id=sample_id,
                    lock_reason="timeout_default",
                    validation_mode="timeout_default",
                    controller_runtime_state=controller_runtime_state,
                )
                self._log_immersive_controller_handedness_resolution(
                    updated_controller_basis_state
                )
            return controller_runtime_state, updated_controller_basis_state

        lateral_flip_applied = candidate_flip

        resolved_basis = self._controller_basis_with_lateral_flip(
            updated_controller_basis_state["head_basis"],
            lateral_flip=lateral_flip_applied,
        )
        if lateral_flip_applied:
            controller_runtime_state = self._recompute_live_controller_runtime_state_for_basis(
                latest_controller_sample,
                controller_runtime_state,
                controller_source_anchor_centers,
                w2c,
                resolved_basis,
                alignment_pose_role=alignment_pose_role,
                controller_position_pose_role=controller_position_pose_role,
                controller_ray_pose_role=controller_ray_pose_role,
            )

        final_left_body_x, _, _ = self._project_controller_world_field_to_startup_screen_x(
            controller_runtime_state.get("left_controller"),
            "position",
            intrinsic,
            w2c,
        )
        final_right_body_x, _, _ = self._project_controller_world_field_to_startup_screen_x(
            controller_runtime_state.get("right_controller"),
            "position",
            intrinsic,
            w2c,
        )
        final_left_ray_x, _, _ = self._project_controller_world_field_to_startup_screen_x(
            controller_runtime_state.get("left_controller"),
            "ray_origin",
            intrinsic,
            w2c,
        )
        final_right_ray_x, _, _ = self._project_controller_world_field_to_startup_screen_x(
            controller_runtime_state.get("right_controller"),
            "ray_origin",
            intrinsic,
            w2c,
        )
        updated_controller_basis_state["final_left_body_x"] = final_left_body_x
        updated_controller_basis_state["final_right_body_x"] = final_right_body_x
        updated_controller_basis_state["final_left_ray_x"] = final_left_ray_x
        updated_controller_basis_state["final_right_ray_x"] = final_right_ray_x
        updated_controller_basis_state = self._lock_immersive_controller_basis_state(
            updated_controller_basis_state,
            sample_id=sample_id,
            lock_reason="dual_confirmed",
            validation_mode=validation_mode,
            basis=resolved_basis,
            lateral_flip_applied=lateral_flip_applied,
            controller_runtime_state=controller_runtime_state,
        )
        self._log_immersive_controller_handedness_resolution(
            updated_controller_basis_state
        )
        return controller_runtime_state, updated_controller_basis_state

    def _immersive_live_forward_from_sample(self, sample):
        if sample is None:
            return None

        eye_forwards = []
        for eye_sample in (sample.left_eye, sample.right_eye):
            if eye_sample is None or not eye_sample.pose_valid:
                continue
            rotation_local = self._quaternion_xyzw_to_rotation_matrix(
                eye_sample.orientation
            )
            eye_forwards.append(rotation_local @ np.array([0.0, 0.0, -1.0], dtype=np.float32))
        if not eye_forwards:
            return None
        return self._normalize_numpy_vector(
            np.mean(np.stack(eye_forwards, axis=0), axis=0),
            fallback=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        )

    def _immersive_sample_has_valid_eye_pose(self, sample):
        return bool(
            sample is not None
            and (
                (sample.left_eye is not None and sample.left_eye.pose_valid)
                or (sample.right_eye is not None and sample.right_eye.pose_valid)
            )
        )

    def _immersive_live_head_center_from_sample(self, sample):
        if sample is None:
            return None

        eye_positions = []
        for eye_sample in (sample.left_eye, sample.right_eye):
            if eye_sample is None or not eye_sample.pose_valid:
                continue
            eye_positions.append(
                torch.from_numpy(np.asarray(eye_sample.position, dtype=np.float32)).to(
                    device=cfg.device,
                    dtype=torch.float32,
                )
            )
        if not eye_positions:
            return None
        return torch.stack(eye_positions, dim=0).mean(dim=0)

    def _compute_immersive_head_alignment(self, sample):
        live_head_origin = self._immersive_live_head_center_from_sample(sample)
        if live_head_origin is None:
            return None

        local_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        live_forward = self._immersive_live_forward_from_sample(sample)
        if live_forward is None:
            return None
        live_forward_horizontal = live_forward - float(np.dot(live_forward, local_up)) * local_up
        live_forward_horizontal = self._normalize_numpy_vector(
            live_forward_horizontal,
            fallback=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        )
        live_back = -live_forward_horizontal
        live_right = self._normalize_numpy_vector(
            np.cross(local_up, live_back),
            fallback=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        )
        live_back = self._normalize_numpy_vector(
            np.cross(live_right, local_up),
            fallback=live_back,
        )
        live_basis = np.stack([live_right, local_up, live_back], axis=1)

        scene_up = self._scene_world_up_vector_np()
        scene_forward = self._scene_world_forward_vector_np()
        scene_back = -scene_forward
        scene_right = self._normalize_numpy_vector(
            np.cross(scene_up, scene_back),
            fallback=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        )
        scene_back = self._normalize_numpy_vector(
            np.cross(scene_right, scene_up),
            fallback=scene_back,
        )
        scene_basis = np.stack([scene_right, scene_up, scene_back], axis=1)
        basis_np = (scene_basis @ live_basis.T).astype(np.float32)
        basis_np_t_basis = basis_np.T @ basis_np

        return {
            "basis": torch.as_tensor(basis_np, dtype=torch.float32, device=cfg.device),
            "translation_scale": torch.tensor(
                self.IMMERSIVE_LIVE_HEAD_TRANSLATION_SCALE,
                dtype=torch.float32,
                device=cfg.device,
            ),
            "reference_live_head": live_head_origin,
            "reference_scene_head": torch.zeros(
                3, dtype=torch.float32, device=cfg.device
            ),
            "scene_up": scene_up,
            "scene_forward": scene_forward,
            "scene_right": scene_right,
            "basis_det": float(np.linalg.det(basis_np)),
            "basis_orthogonality_error": float(
                np.max(np.abs(basis_np_t_basis - np.eye(3, dtype=np.float32)))
            ),
            "startup_live_forward": live_forward_horizontal,
        }

    def _map_live_delta_into_scene(self, delta, head_alignment):
        delta = torch.as_tensor(delta, dtype=torch.float32, device=cfg.device)
        return (
            head_alignment["basis"] @ delta
        ) * head_alignment["translation_scale"]

    def _map_live_head_center_into_scene(self, live_head_center, head_alignment):
        live_head_center = torch.as_tensor(
            live_head_center, dtype=torch.float32, device=cfg.device
        )
        return head_alignment["reference_scene_head"] + self._map_live_delta_into_scene(
            live_head_center - head_alignment["reference_live_head"],
            head_alignment,
        )

    def _update_immersive_head_pose_state(
        self,
        sample,
        head_alignment,
        head_pose_state,
        frame_index=0,
    ):
        if sample is None or head_alignment is None:
            return None, None, head_pose_state

        if head_pose_state is None:
            head_pose_state = {
                "smoothed_scene_head_center": None,
                "last_raw_scene_head_center": None,
                "had_valid_pose": False,
                "cached_scene_eye_offsets": {},
                "last_eye_rotations_world": {},
            }

        valid_eye_positions_live = {}
        current_scene_eye_offsets = {}
        current_eye_rotations_world = {}
        basis_np = head_alignment["basis"].detach().cpu().numpy()

        for source, eye_sample in (
            ("left", sample.left_eye),
            ("right", sample.right_eye),
        ):
            if eye_sample is None or not eye_sample.pose_valid:
                continue
            eye_position_live = torch.from_numpy(
                np.asarray(eye_sample.position, dtype=np.float32)
            ).to(device=cfg.device, dtype=torch.float32)
            valid_eye_positions_live[source] = eye_position_live
            rotation_local = self._quaternion_xyzw_to_rotation_matrix(
                eye_sample.orientation
            )
            current_eye_rotations_world[source] = (basis_np @ rotation_local).astype(
                np.float32
            )

        if not valid_eye_positions_live:
            head_pose_state["had_valid_pose"] = False
            return None, None, head_pose_state

        live_head_center = torch.stack(
            list(valid_eye_positions_live.values()), dim=0
        ).mean(dim=0)
        raw_scene_head_center = self._map_live_head_center_into_scene(
            live_head_center,
            head_alignment,
        )

        for source, eye_position_live in valid_eye_positions_live.items():
            current_scene_eye_offsets[source] = self._map_live_delta_into_scene(
                eye_position_live - live_head_center,
                head_alignment,
            )

        cached_scene_eye_offsets = dict(
            head_pose_state.get("cached_scene_eye_offsets", {})
        )
        cached_scene_eye_offsets.update(current_scene_eye_offsets)

        previous_raw_scene_head_center = head_pose_state.get(
            "last_raw_scene_head_center"
        )
        raw_scene_head_jump = None
        if previous_raw_scene_head_center is not None:
            raw_scene_head_jump = float(
                torch.linalg.norm(
                    raw_scene_head_center - previous_raw_scene_head_center
                ).item()
            )

        reset_reason = None
        if head_pose_state.get("smoothed_scene_head_center") is None:
            reset_reason = "startup"
        elif not head_pose_state.get("had_valid_pose", False):
            reset_reason = "tracking_reacquired"
        elif (
            raw_scene_head_jump is not None
            and raw_scene_head_jump > self.IMMERSIVE_HEAD_RESET_JUMP_THRESHOLD
        ):
            reset_reason = f"jump_{raw_scene_head_jump:.3f}"

        if reset_reason is not None:
            smoothed_scene_head_center = raw_scene_head_center.clone()
        else:
            alpha = float(self.IMMERSIVE_HEAD_TRANSLATION_EMA_ALPHA)
            smoothed_scene_head_center = (
                alpha * raw_scene_head_center
                + (1.0 - alpha) * head_pose_state["smoothed_scene_head_center"]
            )

        last_eye_rotations_world = dict(
            head_pose_state.get("last_eye_rotations_world", {})
        )
        last_eye_rotations_world.update(current_eye_rotations_world)

        head_pose_state.update(
            {
                "smoothed_scene_head_center": smoothed_scene_head_center,
                "last_raw_scene_head_center": raw_scene_head_center.clone(),
                "had_valid_pose": True,
                "cached_scene_eye_offsets": cached_scene_eye_offsets,
                "last_eye_rotations_world": last_eye_rotations_world,
            }
        )

        eye_pose_world = {}
        debug_offsets = {}
        for source, eye_sample in (
            ("left", sample.left_eye),
            ("right", sample.right_eye),
        ):
            eye_offset_world = current_scene_eye_offsets.get(
                source,
                cached_scene_eye_offsets.get(source),
            )
            rotation_world = current_eye_rotations_world.get(
                source,
                last_eye_rotations_world.get(source),
            )
            debug_offsets[source] = (
                None
                if eye_offset_world is None
                else eye_offset_world.detach().cpu().numpy().tolist()
            )
            if eye_offset_world is None or rotation_world is None:
                eye_pose_world[source] = None
                continue
            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3] = rotation_world
            pose[:3, 3] = (
                smoothed_scene_head_center + eye_offset_world
            ).detach().cpu().numpy()
            eye_pose_world[source] = pose

        if (
            reset_reason is not None
            or frame_index == 0
            or (
                self.IMMERSIVE_HEAD_DEBUG_LOG_INTERVAL > 0
                and frame_index % self.IMMERSIVE_HEAD_DEBUG_LOG_INTERVAL == 0
            )
        ):
            print(
                "[quest_display] immersive head pose: "
                f"frame={frame_index} "
                f"raw_head_center={raw_scene_head_center.detach().cpu().numpy().tolist()} "
                f"smoothed_head_center={smoothed_scene_head_center.detach().cpu().numpy().tolist()} "
                f"left_offset={debug_offsets['left']} "
                f"right_offset={debug_offsets['right']} "
                f"reset={reset_reason}",
                flush=True,
            )

        return eye_pose_world["left"], eye_pose_world["right"], head_pose_state

    def _format_controller_pose_startup_state(self, controller_sample):
        if controller_sample is None:
            return "missing"
        return (
            f"active={int(bool(controller_sample.active))} "
            f"position_valid={int(bool(controller_sample.position_valid))} "
            f"orientation_valid={int(bool(controller_sample.orientation_valid))} "
            f"position_tracked={int(bool(controller_sample.position_tracked))} "
            f"orientation_tracked={int(bool(controller_sample.orientation_tracked))}"
        )

    def _format_eye_pose_startup_state(self, eye_sample):
        if eye_sample is None:
            return "missing"
        return (
            f"pose_valid={int(bool(eye_sample.pose_valid))} "
            f"pose_tracked={int(bool(eye_sample.pose_tracked))}"
        )

    def _format_immersive_sample_startup_state(self, sample):
        if sample is None:
            return "no immersive sample received yet"
        return (
            f"left_eye({self._format_eye_pose_startup_state(sample.left_eye)}) "
            f"right_eye({self._format_eye_pose_startup_state(sample.right_eye)}) "
            f"left_controller({self._format_controller_pose_startup_state(sample.left)}) "
            f"right_controller({self._format_controller_pose_startup_state(sample.right)})"
        )

    def _wait_for_valid_immersive_startup_sample(
        self,
        immersive_bridge,
        timeout=10.0,
        progress_callback=None,
    ):
        deadline = time.time() + timeout
        last_sample = None
        while time.time() < deadline:
            if progress_callback is not None:
                progress_callback("startup_wait_for_sample")
            sample = immersive_bridge.get_latest_sample()
            if sample is not None:
                last_sample = sample
                if self._immersive_sample_has_valid_eye_pose(sample):
                    return sample
            if (
                immersive_bridge.process is not None
                and immersive_bridge.process.poll() is not None
            ):
                diagnostics = self._format_immersive_sample_startup_state(last_sample)
                raise RuntimeError(
                    "Quest immersive bridge exited before producing a valid eye pose.\n"
                    f"last_sample: {diagnostics}\n"
                    + immersive_bridge.debug_summary()
                )
            time.sleep(0.05)

        diagnostics = self._format_immersive_sample_startup_state(last_sample)
        raise RuntimeError(
            "Timed out waiting for a valid immersive eye pose.\n"
            f"last_sample: {diagnostics}\n"
            + immersive_bridge.debug_summary()
        )

    def _immersive_bridge_process_state(self, immersive_bridge):
        if immersive_bridge is None:
            return "not_started"
        process = getattr(immersive_bridge, "process", None)
        if process is None:
            return "not_started"
        exit_code = process.poll()
        if exit_code is None:
            return "alive"
        return f"exited(code={exit_code})"

    def _make_immersive_startup_timeline(self):
        return {
            "t0": time.perf_counter(),
            "milestones": [],
            "last_milestone": None,
            "bridge_started_ms": None,
            "first_publish_done_ms": None,
        }

    def _record_immersive_startup_milestone(self, startup_timeline, name, immersive_bridge=None):
        elapsed_ms = (time.perf_counter() - startup_timeline["t0"]) * 1000.0
        bridge_state = self._immersive_bridge_process_state(immersive_bridge)
        milestone = {
            "name": str(name),
            "elapsed_ms": float(elapsed_ms),
            "bridge_state": bridge_state,
        }
        startup_timeline["milestones"].append(milestone)
        startup_timeline["last_milestone"] = str(name)
        if name == "bridge_started":
            startup_timeline["bridge_started_ms"] = float(elapsed_ms)
        elif name == "first_publish_done":
            startup_timeline["first_publish_done_ms"] = float(elapsed_ms)
            bridge_started_ms = startup_timeline.get("bridge_started_ms")
            if bridge_started_ms is not None:
                startup_timeline["startup_gap_ms"] = float(elapsed_ms - bridge_started_ms)
        print(
            "[quest_display] immersive startup milestone: "
            f"{name} elapsed_ms={elapsed_ms:.1f} bridge={bridge_state}",
            flush=True,
        )
        return milestone

    def _format_immersive_startup_timeline_failure(
        self,
        startup_timeline,
        immersive_bridge=None,
    ):
        if startup_timeline is None:
            return ""
        bridge_state = self._immersive_bridge_process_state(immersive_bridge)
        bridge_started_ms = startup_timeline.get("bridge_started_ms")
        last_milestone = startup_timeline.get("last_milestone")
        elapsed_ms = (time.perf_counter() - startup_timeline["t0"]) * 1000.0
        parts = [
            "immersive startup timeline failure: "
            f"last_completed_milestone={last_milestone} "
            f"elapsed_ms={elapsed_ms:.1f} "
            f"bridge={bridge_state}"
        ]
        if bridge_started_ms is not None:
            parts.append(
                f"elapsed_since_bridge_started_ms={elapsed_ms - bridge_started_ms:.1f}"
            )
        return " ".join(parts)

    def _make_immersive_startup_keepalive_state(self, eye_width, eye_height):
        color = torch.tensor(
            self.IMMERSIVE_STARTUP_KEEPALIVE_RGBA,
            dtype=torch.uint8,
            device=cfg.device,
        )
        frame = color.view(1, 1, 4).expand(eye_height, eye_width, 4).contiguous().clone()
        return {
            "left_frame": frame,
            "right_frame": frame.clone(),
            "last_publish_time": None,
            "publish_count": 0,
            "enabled": True,
        }

    def _maybe_publish_immersive_startup_keepalive(
        self,
        immersive_bridge,
        keepalive_state,
        *,
        reason,
        startup_timeline=None,
        force=False,
    ):
        if keepalive_state is None or not keepalive_state.get("enabled", False):
            return False
        now = time.perf_counter()
        last_publish_time = keepalive_state.get("last_publish_time")
        if (
            not force
            and last_publish_time is not None
            and (now - last_publish_time)
            < self.IMMERSIVE_STARTUP_KEEPALIVE_INTERVAL_SECONDS
        ):
            return False
        publish_ok, publish_stats = immersive_bridge.publish_stereo_frames(
            keepalive_state["left_frame"],
            keepalive_state["right_frame"],
        )
        keepalive_state["last_publish_time"] = now
        if publish_ok:
            keepalive_state["publish_count"] = int(keepalive_state.get("publish_count", 0)) + 1
        bridge_state = self._immersive_bridge_process_state(immersive_bridge)
        total_wall_ms = float(publish_stats.get("total_wall", 0.0)) * 1000.0
        print(
            "[quest_display] immersive startup keepalive: "
            f"reason={reason} publish_ok={int(bool(publish_ok))} "
            f"count={keepalive_state.get('publish_count', 0)} "
            f"bridge={bridge_state} publish_wall_ms={total_wall_ms:.2f}",
            flush=True,
        )
        if not publish_ok:
            failure_details = self._format_immersive_startup_timeline_failure(
                startup_timeline,
                immersive_bridge,
            )
            raise RuntimeError(
                "Quest immersive bridge stopped accepting stereo frames during startup keepalive.\n"
                + (failure_details + "\n" if failure_details else "")
                + immersive_bridge.debug_summary()
            )
        return True

    def _sample_received_monotonic_s(self, sample):
        if sample is None:
            return None
        received_s = getattr(sample, "received_monotonic_s", None)
        if received_s is None:
            return None
        received_s = float(received_s)
        if not np.isfinite(received_s):
            return None
        return received_s

    def _predict_immersive_eye_poses_for_sample(
        self,
        sample,
        live_head_alignment,
        head_pose_state,
        frame_index,
    ):
        if sample is None or live_head_alignment is None:
            return None, None
        state_copy = None if head_pose_state is None else copy.deepcopy(head_pose_state)
        left_eye_pose_world, right_eye_pose_world, _ = (
            self._update_immersive_head_pose_state(
                sample,
                live_head_alignment,
                state_copy,
                frame_index=frame_index,
            )
        )
        return left_eye_pose_world, right_eye_pose_world

    def _build_immersive_balanced_scene_render_plan(
        self,
        scene_renderer,
        left_eye_pose_world,
        right_eye_pose_world,
        left_intrinsic,
        right_intrinsic,
        eye_width,
        eye_height,
        scene_width,
        scene_height,
        render_profile_frame=None,
    ):
        balanced_runtime_state = getattr(
            self,
            "_immersive_balanced_runtime_state",
            None,
        )
        if balanced_runtime_state is not None:
            background_mode = str(
                balanced_runtime_state.get(
                    "background_mode",
                    "per_eye_background",
                )
            )
            side_wall_mode = str(
                balanced_runtime_state.get(
                    "side_wall_mode",
                    "disabled",
                )
            )
            table_roi_state = balanced_runtime_state.setdefault(
                "table_roi_state",
                {"left": None, "right": None},
            )
            table_roi_render_scale = float(
                balanced_runtime_state.get(
                    "table_roi_render_scale",
                    self.IMMERSIVE_BALANCED_TABLE_ROI_SUPERSAMPLE_SCALE,
                )
            )
        else:
            background_mode = "per_eye_background"
            side_wall_mode = "disabled"
            table_roi_state = {"left": None, "right": None}
            table_roi_render_scale = float(
                self.IMMERSIVE_BALANCED_TABLE_ROI_SUPERSAMPLE_SCALE
            )
        self._initialize_immersive_balanced_render_profile_frame(render_profile_frame)
        render_plan = {
            "scene_width": int(scene_width),
            "scene_height": int(scene_height),
            "background_mode": background_mode,
            "side_wall_mode": side_wall_mode,
            "table_roi_render_scale": float(table_roi_render_scale),
            "center_eye_pose_world": None,
            "center_intrinsic": None,
            "center_scene_intrinsic": None,
            "left": {
                "eye_pose_world": np.asarray(
                    left_eye_pose_world,
                    dtype=np.float32,
                ).copy(),
                "eye_intrinsic": np.asarray(left_intrinsic, dtype=np.float32).copy(),
                "eye_width": int(eye_width),
                "eye_height": int(eye_height),
                "background_scene_intrinsic": None,
                "table_fullframe_fallback": True,
                "table_roi_bounds": None,
                "table_roi_ratio": 1.0,
                "table_roi_render_scale": float(table_roi_render_scale),
            },
            "right": {
                "eye_pose_world": np.asarray(
                    right_eye_pose_world,
                    dtype=np.float32,
                ).copy(),
                "eye_intrinsic": np.asarray(right_intrinsic, dtype=np.float32).copy(),
                "eye_width": int(eye_width),
                "eye_height": int(eye_height),
                "background_scene_intrinsic": None,
                "table_fullframe_fallback": True,
                "table_roi_bounds": None,
                "table_roi_ratio": 1.0,
                "table_roi_render_scale": float(table_roi_render_scale),
            },
        }
        if background_mode == "per_eye_background":
            render_plan["left"]["background_scene_intrinsic"] = (
                self._scale_intrinsic_for_resolution(
                    render_plan["left"]["eye_intrinsic"],
                    eye_width,
                    eye_height,
                    scene_width,
                    scene_height,
                )
            )
            render_plan["right"]["background_scene_intrinsic"] = (
                self._scale_intrinsic_for_resolution(
                    render_plan["right"]["eye_intrinsic"],
                    eye_width,
                    eye_height,
                    scene_width,
                    scene_height,
                )
            )
        elif background_mode == "mono_center_background":
            center_eye_pose_world, center_intrinsic = (
                self._build_immersive_center_scene_view(
                    render_plan["left"]["eye_pose_world"],
                    render_plan["right"]["eye_pose_world"],
                    render_plan["left"]["eye_intrinsic"],
                    render_plan["right"]["eye_intrinsic"],
                )
            )
            render_plan["center_eye_pose_world"] = center_eye_pose_world
            render_plan["center_intrinsic"] = center_intrinsic
            render_plan["center_scene_intrinsic"] = (
                self._scale_intrinsic_for_resolution(
                    center_intrinsic,
                    eye_width,
                    eye_height,
                    scene_width,
                    scene_height,
                )
            )
        else:
            raise ValueError(
                f"Unsupported immersive balanced background mode: {background_mode}"
            )
        table_world_bounds = scene_renderer.table_world_bounds()
        for eye_label in ("left", "right"):
            eye_pose_world = render_plan[eye_label]["eye_pose_world"]
            eye_intrinsic = render_plan[eye_label]["eye_intrinsic"]
            table_roi_bounds = None
            table_roi_ratio = 1.0
            table_fullframe_fallback = True
            if table_world_bounds is not None:
                eye_w2c_cv = self._camera_pose_world_to_cv_w2c(eye_pose_world)
                (
                    table_roi_bounds,
                    table_roi_ratio,
                    table_fullframe_fallback,
                    _,
                ) = self._resolve_immersive_balanced_table_render_roi(
                    table_world_bounds[0],
                    table_world_bounds[1],
                    eye_intrinsic,
                    eye_w2c_cv,
                    eye_width,
                    eye_height,
                    prev_bounds=table_roi_state.get(eye_label),
                )
            render_plan[eye_label]["table_fullframe_fallback"] = bool(
                table_fullframe_fallback
            )
            render_plan[eye_label]["table_roi_bounds"] = (
                None
                if table_fullframe_fallback or table_roi_bounds is None
                else tuple(int(v) for v in table_roi_bounds)
            )
            render_plan[eye_label]["table_roi_ratio"] = float(table_roi_ratio)
            table_roi_state[eye_label] = render_plan[eye_label]["table_roi_bounds"]
            if render_profile_frame is not None:
                render_profile_frame[f"scene_table_roi_{eye_label}_ratio"] = float(
                    table_roi_ratio
                )
                render_profile_frame["scene_table_roi_supersample_scale"] = float(
                    table_roi_render_scale
                )
                render_profile_frame[
                    f"scene_table_fullframe_fallback_{eye_label}_ratio"
                ] = 1.0 if table_fullframe_fallback else 0.0
        return render_plan

    def _record_immersive_balanced_scene_execute_profile(
        self,
        render_profile_frame,
        render_plan,
        render_outputs,
    ):
        if render_profile_frame is None or render_outputs is None:
            return
        background_mode = str(render_plan["background_mode"])
        if background_mode == "per_eye_background":
            render_profile_frame["scene_render_left_wall"] = float(
                render_outputs["left"].get("background_render_wall_s", 0.0)
            )
            render_profile_frame["scene_render_right_wall"] = float(
                render_outputs["right"].get("background_render_wall_s", 0.0)
            )
        elif background_mode == "mono_center_background":
            shared_background = render_outputs.get("shared_background") or {}
            render_profile_frame["scene_render_background_center_wall"] = float(
                shared_background.get("render_wall_s", 0.0)
            )
        render_profile_frame["scene_render_table_left_wall"] = float(
            render_outputs["left"].get("table_render_wall_s", 0.0)
        )
        render_profile_frame["scene_render_table_right_wall"] = float(
            render_outputs["right"].get("table_render_wall_s", 0.0)
        )

    def _assemble_immersive_balanced_scene_for_eye(
        self,
        scene_renderer,
        render_plan,
        render_outputs,
        eye_label,
        eye_height,
        eye_width,
        background_color_t,
        background_depth_t,
        balanced_runtime_state,
        reproject_caches,
        render_profile_frame,
        center_eye_pose_world,
        center_intrinsic,
        shared_background_source_data,
        source_intrinsic_t,
        source_c2w_cv_t,
        background_compose_cache=None,
    ):
        background_mode = str(render_plan["background_mode"])
        eye_plan = render_plan[eye_label]
        eye_outputs = render_outputs[eye_label]
        eye_pose_world = eye_plan["eye_pose_world"]
        eye_intrinsic = eye_plan["eye_intrinsic"]
        if background_mode == "mono_center_background":
            repaired_background_color_t, repaired_background_depth_t = (
                self._repair_immersive_balanced_background_eye(
                    scene_renderer,
                    eye_label,
                    eye_pose_world,
                    eye_intrinsic,
                    eye_width,
                    eye_height,
                    background_color_t,
                    background_depth_t,
                    center_eye_pose_world,
                    center_intrinsic,
                    balanced_runtime_state=balanced_runtime_state,
                    reproject_caches=reproject_caches,
                    render_profile_frame=render_profile_frame,
                    shared_source_data=shared_background_source_data,
                    source_intrinsic_t=source_intrinsic_t,
                    source_c2w_cv_t=source_c2w_cv_t,
                )
            )
        else:
            repaired_background_color_t = background_color_t
            repaired_background_depth_t = background_depth_t

        table_fullframe_fallback = bool(
            eye_plan.get("table_fullframe_fallback", True)
        )
        if table_fullframe_fallback:
            table_color = eye_outputs["table_color"]
            table_depth = eye_outputs["table_depth"]
            table_roi_bounds = None
            table_coverage_mask = None
        else:
            table_render_info = eye_outputs.get("table_render_info") or {}
            table_color, table_depth, table_coverage_mask = (
                self._downsample_immersive_supersampled_overlay_patch(
                    eye_outputs["table_color"],
                    eye_outputs["table_depth"],
                    int(table_render_info["roi_height"]),
                    int(table_render_info["roi_width"]),
                )
            )
            table_roi_bounds = tuple(int(v) for v in eye_plan["table_roi_bounds"])
        table_compose_span = self._render_profile_begin_cuda_span(
            render_profile_frame,
            f"scene_compose_table_{eye_label}_cuda",
        )
        scene_color_t, scene_depth_t = self._compose_immersive_scene_layers(
            repaired_background_color_t,
            repaired_background_depth_t,
            table_color,
            table_depth,
            target_height=eye_height,
            target_width=eye_width,
            background_cache=background_compose_cache,
            overlay_roi_bounds=table_roi_bounds,
            overlay_coverage_mask=table_coverage_mask,
        )
        self._render_profile_end_cuda_span(
            render_profile_frame,
            table_compose_span,
        )
        return scene_color_t, scene_depth_t

    def _assemble_immersive_balanced_scene_from_render_outputs(
        self,
        scene_renderer,
        render_plan,
        render_outputs,
        eye_width,
        eye_height,
        shared_scene_compose_cache=None,
        reproject_caches=None,
        render_profile_frame=None,
    ):
        balanced_runtime_state = getattr(
            self,
            "_immersive_balanced_runtime_state",
            None,
        )
        background_mode = str(render_plan["background_mode"])
        side_wall_mode = str(render_plan.get("side_wall_mode", "disabled"))
        self._record_immersive_balanced_scene_execute_profile(
            render_profile_frame,
            render_plan,
            render_outputs,
        )
        center_eye_pose_world = render_plan.get("center_eye_pose_world")
        center_intrinsic = render_plan.get("center_intrinsic")
        center_intrinsic_t = None
        center_c2w_cv_t = None
        shared_background_source_data = None
        left_background_compose_cache = shared_scene_compose_cache
        right_background_compose_cache = shared_scene_compose_cache
        if background_mode == "per_eye_background":
            per_eye_background_compose_caches = (
                self._get_immersive_balanced_background_compose_caches(
                    balanced_runtime_state
                )
            )
            left_background_compose_cache = per_eye_background_compose_caches["left"]
            right_background_compose_cache = per_eye_background_compose_caches["right"]
            left_background_prepare_start = (
                time.perf_counter() if render_profile_frame is not None else None
            )
            left_background_color_t, left_background_depth_t = (
                self._prepare_immersive_scene_frame_for_compose(
                    render_outputs["left"]["background_color"],
                    render_outputs["left"]["background_depth"],
                    eye_height,
                    eye_width,
                    compose_cache=left_background_compose_cache,
                )
            )
            if left_background_prepare_start is not None:
                self._render_profile_add_wall_time(
                    render_profile_frame,
                    "scene_prepare_background_eye_wall",
                    time.perf_counter() - left_background_prepare_start,
                )
            right_background_prepare_start = (
                time.perf_counter() if render_profile_frame is not None else None
            )
            right_background_color_t, right_background_depth_t = (
                self._prepare_immersive_scene_frame_for_compose(
                    render_outputs["right"]["background_color"],
                    render_outputs["right"]["background_depth"],
                    eye_height,
                    eye_width,
                    compose_cache=right_background_compose_cache,
                )
            )
            if right_background_prepare_start is not None:
                self._render_profile_add_wall_time(
                    render_profile_frame,
                    "scene_prepare_background_eye_wall",
                    time.perf_counter() - right_background_prepare_start,
                )
        elif background_mode == "mono_center_background":
            shared_background = render_outputs.get("shared_background") or {}
            background_prepare_start = (
                time.perf_counter() if render_profile_frame is not None else None
            )
            background_eye_color_t, background_eye_depth_t = (
                self._prepare_immersive_scene_frame_for_compose(
                    shared_background["color"],
                    shared_background["depth"],
                    eye_height,
                    eye_width,
                    compose_cache=shared_scene_compose_cache,
                )
            )
            if background_prepare_start is not None:
                self._render_profile_add_wall_time(
                    render_profile_frame,
                    "scene_prepare_background_eye_wall",
                    time.perf_counter() - background_prepare_start,
                )

            warp_width_max = float(
                self.IMMERSIVE_BALANCED_SIDE_WALL_STRIP_WARP_MAX_WIDTH_RATIO
            )
            should_prepare_shared_background_source = False
            if side_wall_mode in {"edge_warp_roi", "warp_first_hybrid"}:
                for probe_eye_label in ("left", "right"):
                    probe_specs, _ = self._resolve_immersive_balanced_edge_repair_strips(
                        scene_renderer,
                        probe_eye_label,
                        render_plan[probe_eye_label]["eye_pose_world"],
                        render_plan[probe_eye_label]["eye_intrinsic"],
                        eye_width,
                        eye_height,
                        balanced_runtime_state=balanced_runtime_state,
                        update_state=False,
                    )
                    if any(
                        spec["roi_bounds"] is not None
                        and spec["anchor_edge"] in {"left", "right"}
                        and (not spec["fullframe_fallback"])
                        and float(spec["strip_width_ratio"]) <= warp_width_max
                        for spec in probe_specs
                    ):
                        should_prepare_shared_background_source = True
                        break
            if should_prepare_shared_background_source:
                center_intrinsic_t = torch.as_tensor(
                    center_intrinsic,
                    dtype=torch.float32,
                    device=cfg.device,
                )
                center_w2c_cv_t = torch.as_tensor(
                    self._camera_pose_world_to_cv_w2c(center_eye_pose_world),
                    dtype=torch.float32,
                    device=cfg.device,
                )
                center_c2w_cv_t = torch.linalg.inv(center_w2c_cv_t)
                shared_background_source_data = (
                    self._prepare_immersive_reproject_source_data(
                        background_eye_color_t,
                        background_eye_depth_t,
                        center_intrinsic_t,
                        source_cache=None
                        if reproject_caches is None
                        else reproject_caches.get("background_source"),
                    )
                )
            left_background_color_t = background_eye_color_t
            left_background_depth_t = background_eye_depth_t
            right_background_color_t = background_eye_color_t
            right_background_depth_t = background_eye_depth_t
        else:
            raise ValueError(
                f"Unsupported immersive balanced background mode: {background_mode}"
            )
        left_scene_color_t, left_scene_depth_t = (
            self._assemble_immersive_balanced_scene_for_eye(
                scene_renderer,
                render_plan,
                render_outputs,
                "left",
                eye_height,
                eye_width,
                left_background_color_t,
                left_background_depth_t,
                balanced_runtime_state,
                reproject_caches,
                render_profile_frame,
                center_eye_pose_world,
                center_intrinsic,
                shared_background_source_data,
                center_intrinsic_t,
                center_c2w_cv_t,
                background_compose_cache=left_background_compose_cache,
            )
        )
        right_scene_color_t, right_scene_depth_t = (
            self._assemble_immersive_balanced_scene_for_eye(
                scene_renderer,
                render_plan,
                render_outputs,
                "right",
                eye_height,
                eye_width,
                right_background_color_t,
                right_background_depth_t,
                balanced_runtime_state,
                reproject_caches,
                render_profile_frame,
                center_eye_pose_world,
                center_intrinsic,
                shared_background_source_data,
                center_intrinsic_t,
                center_c2w_cv_t,
                background_compose_cache=right_background_compose_cache,
            )
        )
        return (
            left_scene_color_t,
            left_scene_depth_t,
            right_scene_color_t,
            right_scene_depth_t,
        )

    def _latewarp_immersive_scene_eye(
        self,
        scene_color_t,
        scene_depth_t,
        render_eye_pose_world,
        publish_eye_pose_world,
        render_intrinsic,
        publish_intrinsic,
        target_height,
        target_width,
        eye_label,
        reproject_caches=None,
        render_profile_frame=None,
    ):
        source_intrinsic_t = torch.as_tensor(
            render_intrinsic,
            dtype=torch.float32,
            device=cfg.device,
        )
        publish_intrinsic_t = torch.as_tensor(
            publish_intrinsic,
            dtype=torch.float32,
            device=cfg.device,
        )
        render_w2c_cv_t = torch.as_tensor(
            self._camera_pose_world_to_cv_w2c(render_eye_pose_world),
            dtype=torch.float32,
            device=cfg.device,
        )
        publish_w2c_cv_t = torch.as_tensor(
            self._camera_pose_world_to_cv_w2c(publish_eye_pose_world),
            dtype=torch.float32,
            device=cfg.device,
        )
        source_cache = None
        reproject_cache = None
        if reproject_caches is not None:
            source_cache = reproject_caches.setdefault(
                f"timewarp_scene_source_{eye_label}",
                {},
            )
            reproject_cache = reproject_caches.setdefault(
                f"timewarp_scene_{eye_label}",
                {},
            )
        shared_source_data = self._prepare_immersive_reproject_source_data(
            scene_color_t,
            scene_depth_t,
            source_intrinsic_t,
            source_cache=source_cache,
        )
        return self._reproject_immersive_scene_eye_frame(
            scene_color_t,
            scene_depth_t,
            render_intrinsic,
            render_eye_pose_world,
            publish_intrinsic,
            publish_eye_pose_world,
            target_height,
            target_width,
            render_profile_frame=render_profile_frame,
            eye_label=eye_label,
            reproject_cache=reproject_cache,
            shared_source_data=shared_source_data,
            source_intrinsic_t=source_intrinsic_t,
            source_c2w_cv_t=torch.linalg.inv(render_w2c_cv_t),
            target_intrinsic_t=publish_intrinsic_t,
            target_w2c_cv_t=publish_w2c_cv_t,
            profile_key_prefix="scene_reproject_timewarp",
        )

    def _convert_live_eye_to_world_pose(self, eye_sample: EyePoseSample, head_alignment):
        if head_alignment is None or not eye_sample.pose_valid:
            return None

        eye_position = torch.from_numpy(np.asarray(eye_sample.position, dtype=np.float32)).to(
            device=cfg.device, dtype=torch.float32
        )
        world_position = head_alignment["reference_scene_head"] + (
            head_alignment["basis"]
            @ (eye_position - head_alignment["reference_live_head"])
        ) * head_alignment["translation_scale"]

        rotation_local = self._quaternion_xyzw_to_rotation_matrix(eye_sample.orientation)
        basis = head_alignment["basis"].detach().cpu().numpy()
        rotation_world = basis @ rotation_local

        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = rotation_world
        pose[:3, 3] = world_position.detach().cpu().numpy()
        return pose

    def _camera_pose_world_to_cv_w2c(self, camera_pose_world):
        camera_pose_world = np.asarray(camera_pose_world, dtype=np.float32)
        cv_from_gl = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
        pose_world_cv = np.eye(4, dtype=np.float32)
        pose_world_cv[:3, :3] = camera_pose_world[:3, :3] @ cv_from_gl
        pose_world_cv[:3, 3] = camera_pose_world[:3, 3]
        return np.linalg.inv(pose_world_cv).astype(np.float32)

    def _project_world_point_into_eye(self, world_point, intrinsic, w2c, width, height):
        if world_point is None:
            return {
                "depth": None,
                "pixel": None,
                "in_bounds": False,
            }
        world_point = np.asarray(world_point, dtype=np.float32)
        intrinsic = np.asarray(intrinsic, dtype=np.float32)
        w2c = np.asarray(w2c, dtype=np.float32)
        world_point_h = np.concatenate([world_point, np.array([1.0], dtype=np.float32)])
        camera_point = w2c @ world_point_h
        depth = float(camera_point[2])
        pixel = None
        in_bounds = False
        if abs(depth) > 1e-6:
            pixel_h = intrinsic @ camera_point[:3]
            pixel = pixel_h[:2] / max(pixel_h[2], 1e-6)
            margin = float(self.IMMERSIVE_STARTUP_PIXEL_MARGIN)
            in_bounds = (
                depth > self.IMMERSIVE_STARTUP_DEPTH_EPS
                and pixel[0] >= -margin
                and pixel[0] <= float(width) + margin
                and pixel[1] >= -margin
                and pixel[1] <= float(height) + margin
            )
        return {
            "depth": depth,
            "pixel": pixel,
            "in_bounds": in_bounds,
        }

    def _eye_sample_intrinsic(self, eye_sample: EyePoseSample, width: int, height: int):
        tan_left = float(np.tan(eye_sample.fov.angle_left))
        tan_right = float(np.tan(eye_sample.fov.angle_right))
        tan_up = float(np.tan(eye_sample.fov.angle_up))
        tan_down = float(np.tan(eye_sample.fov.angle_down))
        fx = float(width) / max(tan_right - tan_left, 1e-6)
        fy = float(height) / max(tan_up - tan_down, 1e-6)
        cx = -tan_left * fx
        cy = tan_up * fy
        return np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

    def _eye_forward_world(self, camera_pose_world):
        return camera_pose_world[:3, :3] @ np.array([0.0, 0.0, -1.0], dtype=np.float32)

    def _normalize_gaussian_depth(self, gaussian_depth):
        if gaussian_depth is None:
            return None
        if not torch.is_tensor(gaussian_depth):
            gaussian_depth = torch.as_tensor(
                gaussian_depth,
                device=cfg.device,
                dtype=torch.float32,
            )
        else:
            gaussian_depth = gaussian_depth.to(device=cfg.device, dtype=torch.float32)
        gaussian_depth = gaussian_depth.squeeze()
        finite_mask = torch.isfinite(gaussian_depth)
        if not bool(finite_mask.any().item()):
            return torch.zeros_like(gaussian_depth)
        valid_mask = finite_mask & (gaussian_depth.abs() > self.IMMERSIVE_STARTUP_DEPTH_EPS)
        positive_count = int((valid_mask & (gaussian_depth > 0.0)).sum().item())
        negative_count = int((valid_mask & (gaussian_depth < 0.0)).sum().item())
        normalized_depth = gaussian_depth.clone()
        if negative_count > positive_count:
            normalized_depth = -normalized_depth
        normalized_depth = torch.where(
            finite_mask & (normalized_depth > self.IMMERSIVE_STARTUP_DEPTH_EPS),
            normalized_depth,
            torch.zeros_like(normalized_depth),
        )
        return normalized_depth

    def _object_support_patch_center(self, object_points):
        scene_down = torch.tensor(
            [0.0, 0.0, 1.0 if cfg.reverse_z else -1.0],
            dtype=object_points.dtype,
            device=object_points.device,
        )
        support_depth = torch.sum(object_points * scene_down.unsqueeze(0), dim=1)
        support_depth_max = support_depth.max()
        support_mask = support_depth >= (support_depth_max - 0.012)
        support_points = object_points[support_mask]
        if support_points.numel() == 0:
            support_points = object_points
        support_center = support_points.mean(dim=0)
        support_center = support_center.clone()
        support_center[2] = float(
            object_points[:, 2].max().item() if cfg.reverse_z else object_points[:, 2].min().item()
        )
        return support_center

    def _apply_immersive_startup_yaw(
        self,
        object_vertices,
        controller_vertices,
        gaussians,
        recorded_base_target,
        recorded_anchor_centers,
        controller_source_anchor_centers,
        yaw_axis=None,
        yaw_angle=None,
        gaussian_yaw_pivot=None,
    ):
        if yaw_axis is None:
            yaw_axis = self._scene_world_up_vector_torch(
                device=object_vertices.device,
                dtype=object_vertices.dtype,
            )
        else:
            yaw_axis = torch.as_tensor(
                yaw_axis,
                dtype=object_vertices.dtype,
                device=object_vertices.device,
            )
        if yaw_angle is None:
            yaw_angle = self.IMMERSIVE_STARTUP_YAW_RADIANS
        rotation_matrix = self._axis_angle_rotation_matrix_torch(
            yaw_axis,
            yaw_angle,
            device=object_vertices.device,
            dtype=object_vertices.dtype,
        )
        yaw_pivot = self._object_support_patch_center(object_vertices)
        rotated_object_vertices = self._rotate_points_with_matrix(
            object_vertices,
            rotation_matrix,
            pivot=yaw_pivot,
        )
        rotated_controller_vertices = None
        if controller_vertices is not None:
            rotated_controller_vertices = self._rotate_points_with_matrix(
                controller_vertices,
                rotation_matrix,
                pivot=yaw_pivot,
            )
        rotated_recorded_base_target = self._rotate_points_with_matrix(
            recorded_base_target,
            rotation_matrix,
            pivot=yaw_pivot,
        )
        rotated_recorded_anchor_centers = [
            self._rotate_points_with_matrix(
                center.unsqueeze(0),
                rotation_matrix,
                pivot=yaw_pivot,
            )[0]
            for center in recorded_anchor_centers
        ]
        rotated_controller_source_anchor_centers = [
            self._rotate_points_with_matrix(
                center.unsqueeze(0),
                rotation_matrix,
                pivot=yaw_pivot,
            )[0]
            for center in controller_source_anchor_centers
        ]
        gaussian_rotation_pivot = yaw_pivot
        if gaussian_yaw_pivot is not None:
            gaussian_rotation_pivot = torch.as_tensor(
                gaussian_yaw_pivot,
                dtype=gaussians._xyz.dtype,
                device=gaussians._xyz.device,
            )
        else:
            gaussian_rotation_pivot = yaw_pivot.to(
                device=gaussians._xyz.device,
                dtype=gaussians._xyz.dtype,
            )
        gaussians._xyz = self._rotate_points_with_matrix(
            gaussians._xyz,
            rotation_matrix.to(device=gaussians._xyz.device, dtype=gaussians._xyz.dtype),
            pivot=gaussian_rotation_pivot,
        )
        gaussians._rotation = self._rotate_gaussian_quaternions_about_axis(
            gaussians.get_rotation,
            yaw_axis.to(device=gaussians.get_rotation.device, dtype=gaussians.get_rotation.dtype),
            yaw_angle,
        )
        rotated_support_center = self._object_support_patch_center(rotated_object_vertices)
        return {
            "object_vertices": rotated_object_vertices,
            "controller_vertices": rotated_controller_vertices,
            "recorded_base_target": rotated_recorded_base_target,
            "recorded_anchor_centers": rotated_recorded_anchor_centers,
            "controller_source_anchor_centers": rotated_controller_source_anchor_centers,
            "yaw_axis": yaw_axis,
            "yaw_angle": float(torch.as_tensor(yaw_angle).item()),
            "yaw_pivot": yaw_pivot,
            "rotated_support_center": rotated_support_center,
        }

    def _resolve_immersive_startup_yaw_angle(self, object_vertices):
        case_name = self._interaction_anchor_case_name()
        default_yaw = float(self.IMMERSIVE_STARTUP_YAW_RADIANS)
        if not self._is_rope_family_case(case_name):
            return default_yaw

        if int(object_vertices.shape[0]) < 3:
            return default_yaw

        points_centered = object_vertices - object_vertices.mean(dim=0, keepdim=True)
        covariance = torch.matmul(points_centered.t(), points_centered)
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        rope_axis = eigenvectors[:, int(torch.argmax(eigenvalues).item())]
        horizontal_axis = rope_axis[:2]
        if float(torch.linalg.norm(horizontal_axis).item()) <= 1e-6:
            return default_yaw
        if abs(float(horizontal_axis[0].item())) >= abs(float(horizontal_axis[1].item())):
            return 0.0
        return default_yaw

    def _capture_gaussian_runtime_state(self, gaussians):
        return {
            "xyz": gaussians._xyz.detach().clone(),
            "rotation": gaussians._rotation.detach().clone(),
        }

    def _restore_gaussian_runtime_state(self, gaussians, state):
        gaussians._xyz = state["xyz"].clone()
        gaussians._rotation = state["rotation"].clone()

    def _validate_scene_spawn_alignment(
        self,
        object_points,
        layout,
        context,
        table_surface_center_world=None,
    ):
        support_center, table_center, xy_error, z_error = self._scene_spawn_alignment_metrics(
            object_points,
            layout,
            table_surface_center_world=table_surface_center_world,
        )
        if xy_error > self.IMMERSIVE_STARTUP_CENTER_EPS or z_error > self.IMMERSIVE_STARTUP_PLANE_EPS:
            raise RuntimeError(
                f"Immersive scene spawn validation failed during {context}: "
                f"support_center={support_center.detach().cpu().numpy().tolist()} "
                f"table_top_center={table_center.detach().cpu().numpy().tolist()} "
                f"xy_error={xy_error:.4f} z_error={z_error:.4f}"
            )
        return support_center

    def _scene_table_surface_center_world(self, layout):
        active_table_surface_center = getattr(layout, "active_table_surface_center", None)
        if active_table_surface_center is not None:
            return np.asarray(active_table_surface_center, dtype=np.float32)
        active_table_bounds = getattr(layout, "active_table_bounds", None)
        if active_table_bounds is None:
            return np.asarray(layout.table_top_center, dtype=np.float32)
        bounds = np.asarray(active_table_bounds, dtype=np.float32)
        return np.array(
            [
                0.5 * float(bounds[0, 0] + bounds[1, 0]),
                0.5 * float(bounds[0, 1] + bounds[1, 1]),
                float(bounds[0, 2]),
            ],
            dtype=np.float32,
        )

    def _scene_spawn_alignment_metrics(
        self,
        object_points,
        layout,
        table_surface_center_world=None,
    ):
        support_center = self._object_support_patch_center(object_points)
        target_center = (
            self._scene_table_surface_center_world(layout)
            if table_surface_center_world is None
            else table_surface_center_world
        )
        table_center = torch.as_tensor(
            target_center,
            dtype=torch.float32,
            device=object_points.device,
        )
        xy_error = float(torch.linalg.norm(support_center[:2] - table_center[:2]).item())
        z_error = float(torch.abs(support_center[2] - table_center[2]).item())
        return support_center, table_center, xy_error, z_error

    def _save_immersive_startup_debug_images(self, output_dir, frames):
        if not output_dir:
            return
        os.makedirs(output_dir, exist_ok=True)
        for name, frame in frames.items():
            if frame is None:
                continue
            if torch.is_tensor(frame):
                array = frame.detach().cpu().numpy()
            else:
                array = np.asarray(frame)
            array = np.ascontiguousarray(array)
            if array.ndim != 3:
                continue
            if array.dtype != np.uint8:
                array = np.clip(array, 0, 255).astype(np.uint8)
            output_path = os.path.join(output_dir, f"{name}.png")
            if array.shape[2] == 4:
                cv2.imwrite(output_path, cv2.cvtColor(array, cv2.COLOR_RGBA2BGRA))
            elif array.shape[2] == 3:
                cv2.imwrite(output_path, cv2.cvtColor(array, cv2.COLOR_RGB2BGR))

    def _save_immersive_startup_debug_bundle(self, output_dir, frames, metadata=None):
        if not output_dir:
            return
        self._save_immersive_startup_debug_images(output_dir, frames)
        if metadata is None:
            return
        os.makedirs(output_dir, exist_ok=True)
        metadata_path = os.path.join(output_dir, "startup_debug.json")
        with open(metadata_path, "w") as metadata_file:
            json.dump(metadata, metadata_file, indent=2, sort_keys=True)

    def _visualize_immersive_depth_debug_image(self, depth_map):
        if torch.is_tensor(depth_map):
            depth_np = depth_map.detach().cpu().numpy()
        else:
            depth_np = np.asarray(depth_map)
        depth_np = np.asarray(depth_np, dtype=np.float32).squeeze()
        if depth_np.ndim != 2:
            raise ValueError(
                f"Expected a 2D depth map for visualization, got shape {depth_np.shape}"
            )
        height, width = depth_np.shape
        vis = np.zeros((height, width, 3), dtype=np.uint8)
        finite_mask = np.isfinite(depth_np)
        positive_mask = finite_mask & (depth_np > self.IMMERSIVE_STARTUP_DEPTH_EPS)
        vis[~finite_mask] = np.array([255, 0, 255], dtype=np.uint8)
        if np.any(positive_mask):
            valid_depth = depth_np[positive_mask]
            lo = float(np.percentile(valid_depth, 5.0))
            hi = float(np.percentile(valid_depth, 95.0))
            if not np.isfinite(lo):
                lo = float(valid_depth.min())
            if not np.isfinite(hi):
                hi = float(valid_depth.max())
            scale = max(hi - lo, 1e-6)
            normalized = np.zeros_like(depth_np, dtype=np.float32)
            normalized[positive_mask] = np.clip(
                (depth_np[positive_mask] - lo) / scale,
                0.0,
                1.0,
            )
            depth_u8 = np.round((1.0 - normalized) * 255.0).astype(np.uint8)
            vis[positive_mask] = np.stack(
                [depth_u8, depth_u8, depth_u8],
                axis=-1,
            )[positive_mask]
        return vis

    def _visualize_immersive_alpha_debug_image(self, alpha_map):
        if torch.is_tensor(alpha_map):
            alpha_np = alpha_map.detach().cpu().numpy()
        else:
            alpha_np = np.asarray(alpha_map)
        alpha_np = np.asarray(alpha_np, dtype=np.float32).squeeze()
        if alpha_np.ndim != 2:
            raise ValueError(
                f"Expected a 2D alpha map for visualization, got shape {alpha_np.shape}"
            )
        alpha_u8 = np.clip(alpha_np * 255.0, 0.0, 255.0).astype(np.uint8)
        return np.repeat(alpha_u8[..., None], 3, axis=2)

    @torch.no_grad()
    def _validate_immersive_startup_render(
        self,
        head_alignment,
        layout,
        scene_renderer,
        left_eye_sample,
        right_eye_sample,
        left_eye_pose_world,
        right_eye_pose_world,
        eye_width,
        eye_height,
        gaussians,
        render_pipe,
        background_black,
        background_white,
        debug_output_dir,
        save_success_bundle=False,
        scene_stereo_mode="per_eye",
        scene_width=None,
        scene_height=None,
        reproject_caches=None,
        gaussian_compose_roi_padding=None,
        progress_callback=None,
    ):
        if progress_callback is not None:
            progress_callback("startup_validation_enter")
        if scene_width is None:
            scene_width = int(scene_renderer.width)
        if scene_height is None:
            scene_height = int(scene_renderer.height)
        table_top_center = np.asarray(layout.table_top_center, dtype=np.float32)
        object_points = gaussians.get_xyz.detach()
        object_center = object_points.mean(dim=0).detach().cpu().numpy().astype(np.float32)
        object_support_center = (
            self._object_support_patch_center(object_points).detach().cpu().numpy().astype(np.float32)
        )
        object_bounds_min = (
            object_points.min(dim=0).values.detach().cpu().numpy().astype(np.float32)
        )
        object_bounds_max = (
            object_points.max(dim=0).values.detach().cpu().numpy().astype(np.float32)
        )
        debug_renders = {}
        startup_debug = {
            "table_top_center": table_top_center.tolist(),
            "object_center": object_center.tolist(),
            "object_support_center": object_support_center.tolist(),
            "object_bounds_min": object_bounds_min.tolist(),
            "object_bounds_max": object_bounds_max.tolist(),
            "head_alignment_basis_det": head_alignment["basis_det"],
            "head_alignment_orthogonality_error": head_alignment["basis_orthogonality_error"],
            "scene_up": np.asarray(head_alignment["scene_up"], dtype=np.float32).tolist(),
            "scene_forward": np.asarray(head_alignment["scene_forward"], dtype=np.float32).tolist(),
            "scene_right": np.asarray(head_alignment["scene_right"], dtype=np.float32).tolist(),
        }
        gaussian_visible_any = False
        projection_failures = []
        suppressed_by_scene_depth_eyes = []
        invalid_scene_depth_eyes = []
        left_intrinsic = (
            self._eye_sample_intrinsic(left_eye_sample, eye_width, eye_height)
            if left_eye_sample is not None and left_eye_sample.pose_valid
            else None
        )
        right_intrinsic = (
            self._eye_sample_intrinsic(right_eye_sample, eye_width, eye_height)
            if right_eye_sample is not None and right_eye_sample.pose_valid
            else None
        )
        table_world_bounds = scene_renderer.table_world_bounds()
        balanced_runtime_state = getattr(
            self,
            "_immersive_balanced_runtime_state",
            None,
        )
        balanced_center_eye_pose_world = None
        balanced_background_intrinsic = None
        balanced_near_reference_depth_m = None
        balanced_far_reference_depth_m = None
        if scene_stereo_mode == self.IMMERSIVE_BALANCED_INTERNAL_STEREO_MODE:
            balanced_table_world_bounds = scene_renderer.table_world_bounds()
            if balanced_table_world_bounds is not None:
                table_world_bounds = balanced_table_world_bounds
            startup_debug["balanced_background_mode"] = (
                "per_eye_background"
                if balanced_runtime_state is None
                else str(
                    balanced_runtime_state.get(
                        "background_mode",
                        "per_eye_background",
                    )
                )
            )
            startup_debug["balanced_side_wall_mode"] = (
                "disabled"
                if balanced_runtime_state is None
                else str(
                    balanced_runtime_state.get(
                        "side_wall_mode",
                        "disabled",
                    )
                )
            )
            startup_debug["balanced_table_mode"] = "roi_per_eye"
            startup_debug["balanced_table_roi_render_scale"] = float(
                self.IMMERSIVE_BALANCED_TABLE_ROI_SUPERSAMPLE_SCALE
            )
            if balanced_runtime_state is not None:
                balanced_near_reference_depth_m = float(
                    balanced_runtime_state.get("near_reference_depth_m", 0.0)
                )
                balanced_far_reference_depth_m = float(
                    balanced_runtime_state.get("far_reference_depth_m", 0.0)
                )
                startup_debug["balanced_near_reference_depth_m"] = (
                    balanced_near_reference_depth_m
                )
                startup_debug["balanced_far_reference_depth_m"] = (
                    balanced_far_reference_depth_m
                )
            if (
                (left_intrinsic is not None or right_intrinsic is not None)
                and (left_eye_pose_world is not None or right_eye_pose_world is not None)
            ):
                balanced_center_eye_pose_world, center_intrinsic = (
                    self._build_immersive_center_scene_view(
                        left_eye_pose_world,
                        right_eye_pose_world,
                        left_intrinsic,
                        right_intrinsic,
                    )
                )
                if balanced_runtime_state is not None:
                    balanced_background_intrinsic = np.asarray(
                        balanced_runtime_state["background_intrinsic"],
                        dtype=np.float32,
                    )
                else:
                    balanced_background_intrinsic, _, _ = (
                        self._build_immersive_balanced_background_intrinsic(
                            center_intrinsic,
                            eye_width,
                            eye_height,
                        )
                    )
                    balanced_near_reference_depth_m = (
                        self._compute_immersive_balanced_background_reference_depth(
                            layout,
                            balanced_center_eye_pose_world,
                        )
                    )
                    (
                        balanced_far_reference_depth_m,
                        _,
                    ) = self._compute_immersive_balanced_far_reference_depth(
                        layout,
                        balanced_center_eye_pose_world,
                    )
                    startup_debug["balanced_near_reference_depth_m"] = (
                        balanced_near_reference_depth_m
                    )
                    startup_debug["balanced_far_reference_depth_m"] = (
                        balanced_far_reference_depth_m
                    )
        rendered_scene_by_eye = {}
        if (
            left_intrinsic is not None
            and right_intrinsic is not None
            and left_eye_pose_world is not None
            and right_eye_pose_world is not None
        ):
            (
                left_scene_color,
                left_scene_depth,
                right_scene_color,
                right_scene_depth,
            ) = self._render_immersive_scene_frames_for_mode(
                scene_renderer,
                scene_stereo_mode,
                layout,
                object_support_center,
                object_bounds_min,
                object_bounds_max,
                left_eye_pose_world,
                right_eye_pose_world,
                left_intrinsic,
                right_intrinsic,
                eye_width,
                eye_height,
                scene_width,
                scene_height,
                shared_scene_compose_cache=None,
                reproject_caches=reproject_caches,
                render_profile_frame=None,
            )
            rendered_scene_by_eye = {
                "left": (left_scene_color, left_scene_depth),
                "right": (right_scene_color, right_scene_depth),
            }
            if progress_callback is not None:
                progress_callback("startup_validation_scene_frames_ready")

        for eye_name, eye_sample, eye_pose_world in (
            ("left", left_eye_sample, left_eye_pose_world),
            ("right", right_eye_sample, right_eye_pose_world),
        ):
            if progress_callback is not None:
                progress_callback(f"startup_validation_eye_{eye_name}_begin")
            if eye_sample is None or not eye_sample.pose_valid or eye_pose_world is None:
                continue
            startup_debug[f"{eye_name}_eye_pose_world"] = eye_pose_world.tolist()
            intrinsic = self._eye_sample_intrinsic(eye_sample, eye_width, eye_height)
            w2c_cv = self._camera_pose_world_to_cv_w2c(eye_pose_world)
            table_projection = self._project_world_point_into_eye(
                table_top_center,
                intrinsic,
                w2c_cv,
                eye_width,
                eye_height,
            )
            object_projection = self._project_world_point_into_eye(
                object_center,
                intrinsic,
                w2c_cv,
                eye_width,
                eye_height,
            )
            startup_debug[f"{eye_name}_table_projection"] = {
                "depth": table_projection["depth"],
                "pixel": None
                if table_projection["pixel"] is None
                else table_projection["pixel"].astype(np.float32).tolist(),
                "in_bounds": bool(table_projection["in_bounds"]),
            }
            startup_debug[f"{eye_name}_object_projection"] = {
                "depth": object_projection["depth"],
                "pixel": None
                if object_projection["pixel"] is None
                else object_projection["pixel"].astype(np.float32).tolist(),
                "in_bounds": bool(object_projection["in_bounds"]),
            }
            if (
                scene_stereo_mode == self.IMMERSIVE_BALANCED_INTERNAL_STEREO_MODE
                and table_world_bounds is not None
            ):
                (
                    table_roi_bounds,
                    table_roi_ratio,
                    table_fullframe_fallback,
                    table_hysteresis_bounds,
                ) = self._resolve_immersive_balanced_table_render_roi(
                        table_world_bounds[0],
                        table_world_bounds[1],
                        intrinsic,
                        w2c_cv,
                        eye_width,
                        eye_height,
                        prev_bounds=(
                            balanced_runtime_state.get("table_roi_state", {}).get(eye_name)
                            if balanced_runtime_state is not None
                            else None
                        ),
                )
                startup_debug[f"{eye_name}_table_roi_ratio"] = float(table_roi_ratio)
                startup_debug[f"{eye_name}_table_fullframe_fallback"] = bool(
                    table_fullframe_fallback
                )
                startup_debug[f"{eye_name}_table_roi_bounds"] = (
                    None
                    if table_roi_bounds is None
                    else [int(v) for v in table_roi_bounds]
                )
                startup_debug[f"{eye_name}_table_roi_hysteresis_bounds"] = (
                    None
                    if table_hysteresis_bounds is None
                    else [int(v) for v in table_hysteresis_bounds]
                )
                if (
                    balanced_runtime_state is not None
                    and balanced_runtime_state.get("side_wall_mode")
                    in {
                        "per_eye_roi",
                        "per_eye_roi_replace",
                        "edge_warp_roi",
                        "warp_first_hybrid",
                    }
                    and balanced_center_eye_pose_world is not None
                    and balanced_background_intrinsic is not None
                    and balanced_near_reference_depth_m is not None
                    and balanced_far_reference_depth_m is not None
                ):
                    near_shift_dx_px, near_shift_dy_px, _ = (
                        self._compute_immersive_balanced_background_shift(
                            balanced_center_eye_pose_world,
                            eye_pose_world,
                            balanced_background_intrinsic,
                            intrinsic,
                            balanced_near_reference_depth_m,
                        )
                    )
                    far_shift_dx_px, far_shift_dy_px, _ = (
                        self._compute_immersive_balanced_background_shift(
                            balanced_center_eye_pose_world,
                            eye_pose_world,
                            balanced_background_intrinsic,
                            intrinsic,
                            balanced_far_reference_depth_m,
                        )
                    )
                    startup_debug[f"{eye_name}_balanced_near_shift_px"] = {
                        "dx": float(near_shift_dx_px),
                        "dy": float(near_shift_dy_px),
                    }
                    startup_debug[f"{eye_name}_balanced_far_shift_px"] = {
                        "dx": float(far_shift_dx_px),
                        "dy": float(far_shift_dy_px),
                    }
                    side_roi_total = 0.0
                    side_fullframe_fallback = False
                    side_wall_anchor_edges = {}
                    side_wall_strip_bounds = {}
                    for side_name in ("left", "right"):
                        wall_bounds = scene_renderer.wall_world_bounds(side_name)
                        if wall_bounds is None:
                            continue
                        prev_side_bounds = None
                        if balanced_runtime_state is not None:
                            prev_side_bounds = (
                                balanced_runtime_state.get("side_wall_roi_state", {})
                                .get(eye_name, {})
                                .get(side_name)
                            )
                        side_roi_bounds, side_roi_ratio, side_ff, side_debug = (
                            self._resolve_immersive_balanced_side_wall_strip_roi(
                                wall_bounds[0],
                                wall_bounds[1],
                                intrinsic,
                                w2c_cv,
                                eye_width,
                                eye_height,
                                prev_bounds=prev_side_bounds,
                            )
                        )
                        startup_debug[f"{eye_name}_{side_name}_wall_roi_ratio"] = float(
                            side_roi_ratio
                        )
                        startup_debug[f"{eye_name}_{side_name}_side_wall_anchor_edge"] = (
                            side_debug.get("anchor_edge")
                        )
                        side_wall_anchor_edges[side_name] = side_debug.get("anchor_edge")
                        startup_debug[f"{eye_name}_{side_name}_side_wall_strip_bounds"] = (
                            None
                            if side_debug.get("hysteresis_bounds") is None
                            else [int(v) for v in side_debug["hysteresis_bounds"]]
                        )
                        side_wall_strip_bounds[side_name] = (
                            None
                            if side_debug.get("hysteresis_bounds") is None
                            else [int(v) for v in side_debug["hysteresis_bounds"]]
                        )
                        startup_debug[f"{eye_name}_{side_name}_wall_roi_bounds"] = (
                            None
                            if side_roi_bounds is None
                            else [int(v) for v in side_roi_bounds]
                        )
                        startup_debug[
                            f"{eye_name}_{side_name}_wall_fullframe_fallback"
                        ] = bool(side_ff)
                        side_roi_total += float(side_roi_ratio)
                        side_fullframe_fallback = side_fullframe_fallback or bool(side_ff)
                    startup_debug[f"{eye_name}_side_wall_roi_ratio"] = float(
                        min(side_roi_total, 1.0)
                    )
                    startup_debug[f"{eye_name}_side_wall_anchor_edge"] = (
                        dict(side_wall_anchor_edges)
                    )
                    startup_debug[f"{eye_name}_side_wall_strip_bounds"] = (
                        dict(side_wall_strip_bounds)
                    )
                    startup_debug[f"{eye_name}_side_wall_fullframe_fallback"] = bool(
                        side_fullframe_fallback
                    )
            if not table_projection["in_bounds"] or not object_projection["in_bounds"]:
                projection_failures.append(eye_name)

            precomputed_scene = rendered_scene_by_eye.get(eye_name)
            if precomputed_scene is None:
                scene_intrinsic = self._scale_intrinsic_for_resolution(
                    intrinsic,
                    eye_width,
                    eye_height,
                    scene_width,
                    scene_height,
                )
                scene_color, scene_depth = scene_renderer.render_eye(
                    eye_pose_world,
                    scene_intrinsic,
                    width=scene_width,
                    height=scene_height,
                )
            else:
                scene_color, scene_depth = precomputed_scene
            view, _ = self._create_gs_view(
                w2c_cv,
                intrinsic,
                eye_height,
                eye_width,
            )
            gaussian_rgba, gaussian_depth = self._render_gaussian_rgba(
                view,
                gaussians,
                render_pipe,
                background_black,
                background_white,
                use_gsplat=True,
            )
            gaussian_depth = self._normalize_gaussian_depth(gaussian_depth)
            composed, compose_metrics, compose_debug_maps = self._compose_immersive_eye_frame(
                scene_color,
                scene_depth,
                gaussian_rgba,
                gaussian_depth,
                collect_debug=True,
                collect_debug_maps=True,
                compose_roi_padding=gaussian_compose_roi_padding,
            )
            debug_renders[f"{eye_name}_scene"] = scene_color
            debug_renders[f"{eye_name}_scene_depth"] = (
                self._visualize_immersive_depth_debug_image(
                    compose_debug_maps["scene_depth"]
                )
            )
            debug_renders[f"{eye_name}_gaussian_rgba"] = (
                (
                    gaussian_rgba.detach()
                    .permute(1, 2, 0)
                    .contiguous()
                    .clamp(0.0, 1.0)
                    * 255.0
                ).to(torch.uint8)
            )
            debug_renders[f"{eye_name}_gaussian_alpha"] = (
                self._visualize_immersive_alpha_debug_image(
                    compose_debug_maps["raw_alpha"]
                )
            )
            debug_renders[f"{eye_name}_visible_alpha"] = (
                self._visualize_immersive_alpha_debug_image(
                    compose_debug_maps["visible_alpha"]
                )
            )
            debug_renders[f"{eye_name}_composed"] = composed
            gaussian_alpha = gaussian_rgba[3]
            gaussian_alpha_max = float(gaussian_alpha.max().item())
            gaussian_depth_nonzero = int((gaussian_depth > 0.0).sum().item()) if gaussian_depth is not None else 0
            startup_debug[f"{eye_name}_gaussian_alpha_max"] = gaussian_alpha_max
            startup_debug[f"{eye_name}_gaussian_depth_nonzero"] = gaussian_depth_nonzero
            startup_debug[f"{eye_name}_scene_depth_metrics"] = {
                "finite_ratio": float(compose_metrics["scene_depth_finite_ratio"]),
                "positive_ratio": float(compose_metrics["scene_depth_positive_ratio"]),
                "valid_min": float(compose_metrics["scene_depth_valid_min"]),
                "valid_max": float(compose_metrics["scene_depth_valid_max"]),
                "invalid": bool(compose_metrics["scene_depth_invalid"]),
            }
            startup_debug[f"{eye_name}_compose_metrics"] = {
                "compose_mode": str(compose_metrics["compose_mode"]),
                "raw_gaussian_coverage_ratio": float(
                    compose_metrics["raw_gaussian_coverage_ratio"]
                ),
                "visible_gaussian_coverage_ratio": float(
                    compose_metrics["visible_gaussian_coverage_ratio"]
                ),
                "visible_retention_ratio": float(
                    compose_metrics["visible_retention_ratio"]
                ),
                "scene_depth_suppressed": bool(
                    compose_metrics["scene_depth_suppressed"]
                ),
                "composed_luma_mean": float(compose_metrics["composed_luma_mean"]),
                "composed_luma_variance": float(
                    compose_metrics["composed_luma_variance"]
                ),
            }
            if gaussian_alpha_max > self.IMMERSIVE_STARTUP_ALPHA_EPS or gaussian_depth_nonzero > 0:
                gaussian_visible_any = True
            if compose_metrics.get("scene_depth_invalid", False):
                invalid_scene_depth_eyes.append(eye_name)
            if compose_metrics.get("scene_depth_suppressed", False):
                suppressed_by_scene_depth_eyes.append(eye_name)
            if progress_callback is not None:
                progress_callback(f"startup_validation_eye_{eye_name}_done")

        startup_debug["projection_failures"] = projection_failures
        startup_debug["scene_depth_invalid_eyes"] = invalid_scene_depth_eyes
        startup_debug["suppressed_by_scene_depth_eyes"] = suppressed_by_scene_depth_eyes
        compose_fallback_required = bool(
            invalid_scene_depth_eyes or suppressed_by_scene_depth_eyes
        )
        startup_debug["compose_fallback_required"] = compose_fallback_required
        startup_debug["recommended_compose_mode"] = (
            "alpha_overlay" if compose_fallback_required else "depth_aware"
        )
        if projection_failures or not gaussian_visible_any:
            failure_dir = (
                os.path.join(debug_output_dir, "startup_failure")
                if debug_output_dir is not None
                else None
            )
            self._save_immersive_startup_debug_bundle(
                failure_dir,
                debug_renders,
                startup_debug,
            )
            startup_debug["validation_failed"] = True
            startup_debug["validation_warning"] = (
                "initial_sample_out_of_view"
                if projection_failures
                else "initial_sample_gaussian_not_visible"
            )
            print(
                "[quest_display] immersive startup render validation warning: "
                f"{startup_debug['validation_warning']}; continuing so the live demo can recover "
                "as the headset pose settles",
                flush=True,
            )
            return startup_debug
        if (save_success_bundle or compose_fallback_required) and debug_output_dir is not None:
            self._save_immersive_startup_debug_bundle(
                os.path.join(debug_output_dir, "startup_debug"),
                debug_renders,
                startup_debug,
            )
        if progress_callback is not None:
            progress_callback("startup_validation_complete")
        return startup_debug

    def _sources_pending_grab_start_validation(self, controller_interaction_state):
        sources = []
        for source in ("left", "right"):
            interaction_state = controller_interaction_state.get(source)
            if interaction_state is None:
                continue
            if int(interaction_state.get("startup_validation_frames_remaining", 0)) > 0:
                sources.append(source)
        return sources

    def _validate_new_controller_interaction_candidate(
        self,
        source,
        controller_world,
        interaction_state,
        remap_candidate,
        hit_world,
        ray_direction,
        controller_interaction_state=None,
        explicit_preview_selected=False,
    ):
        if interaction_state is None or remap_candidate is None:
            return "missing_interaction_candidate", {}

        candidate_tensors = (
            remap_candidate.get("springs"),
            remap_candidate.get("rest_lengths"),
            remap_candidate.get("attach_center_world"),
            remap_candidate.get("attach_anchor_world"),
        )
        if any(tensor is None or not self._is_finite_tensor(tensor) for tensor in candidate_tensors):
            return "non_finite_candidate", {}

        selected_object_indices = remap_candidate.get("selected_object_indices")
        if selected_object_indices is None or int(selected_object_indices.numel()) <= 0:
            return "empty_patch", {}
        anchor_name = remap_candidate.get("anchor_name")
        anchor_occupied, occupied_other_source = self._anchor_is_occupied_by_other_source(
            source,
            anchor_name,
            controller_interaction_state,
        )
        if anchor_occupied:
            return f"anchor_occupied(other_source={occupied_other_source})", {}

        attach_anchor_world = remap_candidate["attach_anchor_world"]
        attach_radius = float(remap_candidate.get("attach_radius", 0.0))
        hit_distance = None
        if hit_world is not None and not explicit_preview_selected:
            hit_distance = float(torch.linalg.norm(attach_anchor_world - hit_world).item())
            if hit_distance > self.LIVE_CONTROLLER_HIT_WORLD_RADIUS:
                return (
                    f"hit_anchor_mismatch({hit_distance:.4f}>"
                    f"{self.LIVE_CONTROLLER_HIT_WORLD_RADIUS:.4f})",
                    {
                        "hit_distance": hit_distance,
                    },
                )

        projected_anchor_distance = float(
            interaction_state.get("projected_anchor_distance", 0.0)
        )
        strict_projected_anchor_distance_limit = max(
            self.IMMERSIVE_GRAB_START_MAX_TARGET_DELTA,
            self.IMMERSIVE_GRAB_START_TARGET_DELTA_RADIUS_SCALE * attach_radius,
        )
        relaxed_projected_anchor_distance_limit = max(
            self.IMMERSIVE_GRAB_START_RELAXED_MAX_TARGET_DELTA,
            self.IMMERSIVE_GRAB_START_RELAXED_TARGET_DELTA_RADIUS_SCALE * attach_radius,
        )
        projected_anchor_distance_bypassed = bool(
            explicit_preview_selected
            and projected_anchor_distance > strict_projected_anchor_distance_limit
        )
        if (
            not explicit_preview_selected
            and projected_anchor_distance > relaxed_projected_anchor_distance_limit
        ):
            return (
                "projected_anchor_distance_exceeded("
                f"{projected_anchor_distance:.4f}>"
                f"{relaxed_projected_anchor_distance_limit:.4f})",
                {
                    "projected_anchor_distance": projected_anchor_distance,
                    "projected_anchor_distance_limit": relaxed_projected_anchor_distance_limit,
                    "projected_anchor_distance_bypassed": False,
                },
            )

        if hit_world is not None and ray_direction is not None and not explicit_preview_selected:
            direction = ray_direction / ray_direction.norm().clamp_min(1e-6)
            anchor_depth = float(torch.dot(attach_anchor_world - hit_world, direction).item())
            if anchor_depth > self.LIVE_CONTROLLER_MULTI_POINTS_BACK_DEPTH_THRESHOLD:
                return (
                    f"back_facing_patch({anchor_depth:.4f}>"
                    f"{self.LIVE_CONTROLLER_MULTI_POINTS_BACK_DEPTH_THRESHOLD:.4f})",
                    {
                        "anchor_depth": anchor_depth,
                    },
                )

        return None, {
            "projected_anchor_distance": projected_anchor_distance,
            "projected_anchor_distance_limit": relaxed_projected_anchor_distance_limit,
            "projected_anchor_distance_bypassed": projected_anchor_distance_bypassed,
            "strict_projected_anchor_distance_limit": strict_projected_anchor_distance_limit,
            "explicit_preview_selected": bool(explicit_preview_selected),
            "hit_distance": hit_distance,
        }

    def _log_controller_interaction_start_attempt(
        self,
        source,
        interaction_state,
        remap_candidate=None,
        reason=None,
    ):
        preview_anchor_visible = (
            None
            if interaction_state is None
            else int(bool(interaction_state.get("preview_anchor_visible", False)))
        )
        preview_anchor_name = (
            None
            if interaction_state is None
            else interaction_state.get("preview_anchor_name")
        )
        preview_anchor_resolved = (
            None
            if interaction_state is None
            else int(bool(interaction_state.get("preview_anchor_resolved", False)))
        )
        grab_start_mode = (
            None
            if interaction_state is None
            else interaction_state.get("grab_start_mode")
        )
        anchor_name = (
            remap_candidate.get("anchor_name")
            if remap_candidate is not None
            else preview_anchor_name
        )
        hit_distance = (
            None
            if remap_candidate is None
            else remap_candidate.get("hit_to_anchor_distance")
        )
        projected_anchor_distance = (
            None
            if interaction_state is None
            else interaction_state.get("projected_anchor_distance")
        )
        target_delta = (
            None
            if interaction_state is None
            else interaction_state.get("target_delta")
        )
        hit_present = (
            None
            if interaction_state is None
            else int(bool(interaction_state.get("hit_present", False)))
        )
        start_reference = (
            None
            if interaction_state is None
            else interaction_state.get("start_reference")
        )
        print(
            "[live_openxr_controller] "
            f"{source} interaction_start=1 "
            f"mode={grab_start_mode} "
            f"preview_visible={preview_anchor_visible} "
            f"preview_anchor={preview_anchor_name} "
            f"preview_resolved={preview_anchor_resolved} "
            f"hit_present={hit_present} "
            f"start_reference={start_reference} "
            f"anchor={anchor_name} "
            f"hit_to_anchor={None if hit_distance is None else round(float(hit_distance), 4)} "
            f"projected_anchor={None if projected_anchor_distance is None else round(float(projected_anchor_distance), 4)} "
            f"target_delta={None if target_delta is None else round(float(target_delta), 4)} "
            f"reason={reason}",
            flush=True,
        )

    def _log_controller_interaction_rejected(
        self,
        source,
        remap_candidate,
        interaction_state,
        reason,
        action="rejected",
    ):
        multi_points_debug = None if remap_candidate is None else remap_candidate.get("multi_points_debug")
        seed_index = None
        patch_size = None
        if multi_points_debug is not None and not multi_points_debug.get("fallback_used", False):
            seed_index = multi_points_debug.get("seed_index")
            patch_size = multi_points_debug.get("patch_size")
        hit_distance = None
        if remap_candidate is not None:
            hit_distance = remap_candidate.get("hit_to_anchor_distance")
        target_delta = None if interaction_state is None else interaction_state.get("target_delta")
        projected_anchor_distance = (
            None
            if interaction_state is None
            else interaction_state.get("projected_anchor_distance")
        )
        anchor_name = None if remap_candidate is None else remap_candidate.get("anchor_name")
        if anchor_name is None and interaction_state is not None:
            anchor_name = interaction_state.get("preview_anchor_name")
        grab_start_mode = None if interaction_state is None else interaction_state.get("grab_start_mode")
        start_reference = None if interaction_state is None else interaction_state.get("start_reference")
        print(
            "[live_openxr_controller] "
            f"{source} interaction_{action}=1 "
            f"mode={grab_start_mode} "
            f"start_reference={start_reference} "
            f"anchor={anchor_name} "
            f"seed={seed_index} patch={patch_size} "
            f"hit_to_anchor={None if hit_distance is None else round(float(hit_distance), 4)} "
            f"projected_anchor={None if projected_anchor_distance is None else round(float(projected_anchor_distance), 4)} "
            f"target_delta={None if target_delta is None else round(float(target_delta), 4)} "
            f"reason={reason}",
            flush=True,
        )

    def _validate_immersive_grab_start_frame(
        self,
        x,
        current_pos,
        current_rot,
        last_valid_object_center,
        last_immersive_sample,
        last_left_eye_pose_world,
        last_right_eye_pose_world,
        eye_width,
        eye_height,
        left_gaussian_rgba,
        left_gaussian_depth,
        right_gaussian_rgba,
        right_gaussian_depth,
    ):
        if not self._is_finite_tensor(x):
            return "non_finite_sim_state", {}
        if not self._is_finite_tensor(current_pos):
            return "non_finite_gaussian_positions", {}
        if not self._is_finite_tensor(current_rot):
            return "non_finite_gaussian_rotations", {}
        return None, {}

    def _rollback_immersive_grab_start(
        self,
        sources,
        controller_interaction_state,
        controller_anchor_preview_state,
        controller_attachment_metadata,
        last_valid_sim_state,
        last_valid_target,
        gaussians,
        last_valid_gaussian_state,
    ):
        for source in sources:
            self._clear_live_controller_interaction(
                source,
                controller_interaction_state,
                controller_attachment_metadata,
                reason="rollback",
            )
            self._reset_controller_anchor_preview_state(
                controller_anchor_preview_state,
                source,
            )
        self._restore_sim_state(last_valid_sim_state)
        self._restore_gaussian_runtime_state(gaussians, last_valid_gaussian_state)
        restored_target = last_valid_target.clone()
        self.simulator.set_controller_interactive(restored_target, restored_target)
        return restored_target

    def _compute_scene_spawn_shift(self, object_points, table_top_center_world):
        support_center = self._object_support_patch_center(object_points)
        target = torch.as_tensor(
            table_top_center_world,
            dtype=torch.float32,
            device=object_points.device,
        )
        return target - support_center

    def _copy_object_init_vertices_to_simulator(self, object_vertices):
        self.simulator.wp_init_vertices = wp.from_torch(
            object_vertices.contiguous(), dtype=wp.vec3, requires_grad=False
        )

    def _apply_scene_spawn_offset_runtime(
        self,
        spawn_shift,
        gaussians,
        controller_runtime_base_target=None,
        recorded_base_target=None,
        recorded_anchor_centers=None,
        controller_source_anchor_centers=None,
    ):
        spawn_shift = spawn_shift.to(device=cfg.device, dtype=torch.float32)
        self.batch_init_vertices = self.batch_init_vertices + spawn_shift
        self.batch_controller_points = self.batch_controller_points + spawn_shift
        if controller_runtime_base_target is not None:
            controller_runtime_base_target += spawn_shift
        if recorded_base_target is not None:
            recorded_base_target += spawn_shift
        if recorded_anchor_centers is not None:
            for idx in range(len(recorded_anchor_centers)):
                recorded_anchor_centers[idx] = recorded_anchor_centers[idx] + spawn_shift
        if controller_source_anchor_centers is not None:
            for idx in range(len(controller_source_anchor_centers)):
                controller_source_anchor_centers[idx] = (
                    controller_source_anchor_centers[idx] + spawn_shift
                )
        gaussians._xyz = gaussians._xyz + spawn_shift
        object_vertices = self.batch_init_vertices[: self.simulator.object_massnode_total]
        self._copy_object_init_vertices_to_simulator(object_vertices)
        control_points = (
            controller_runtime_base_target
            if controller_runtime_base_target is not None
            else self.batch_controller_points[0]
        )
        control_points = control_points.contiguous()
        self.simulator.wp_original_control_point = wp.from_torch(
            control_points.clone(),
            dtype=wp.vec3,
            requires_grad=False,
        )
        self.simulator.wp_target_control_point = wp.from_torch(
            control_points.clone(),
            dtype=wp.vec3,
            requires_grad=False,
        )
        self.simulator.set_init_state(self.simulator.wp_init_vertices, self.simulator.wp_init_velocities)

    def _capture_sim_state(self):
        return {
            "x": wp.to_torch(self.simulator.wp_states[0].wp_x, requires_grad=False).clone(),
            "v": wp.to_torch(self.simulator.wp_states[0].wp_v, requires_grad=False).clone(),
        }

    def _restore_sim_state(self, state):
        wp_x = wp.from_torch(state["x"].contiguous(), dtype=wp.vec3, requires_grad=False)
        wp_v = wp.from_torch(state["v"].contiguous(), dtype=wp.vec3, requires_grad=False)
        self.simulator.set_init_state(wp_x, wp_v)

    def _set_scene_collider_boxes(self, layout):
        if layout.static_collider_boxes is not None:
            boxes_np = np.asarray(layout.static_collider_boxes, dtype=np.float32)
        else:
            boxes_np = np.array(
                [
                    [layout.table_box.mins, layout.table_box.maxs],
                    [layout.floor_box.mins, layout.floor_box.maxs],
                ],
                dtype=np.float32,
            )
        boxes = torch.as_tensor(boxes_np, dtype=torch.float32, device=cfg.device)
        self.simulator.set_static_collision_boxes(boxes)

    def _settle_scene_rest_state(self, rest_target, progress_callback=None):
        self.simulator.set_controller_interactive(rest_target, rest_target)
        last_state = None
        for step_idx in range(self.IMMERSIVE_SCENE_REST_SETTLE_STEPS):
            if progress_callback is not None:
                progress_callback(f"settle_step_{step_idx}")
            if self.simulator.object_collision_flag:
                self.simulator.update_collision_graph()
            wp.capture_launch(self.simulator.forward_graph)
            wp.synchronize()
            x = wp.to_torch(self.simulator.wp_states[-1].wp_x, requires_grad=False).clone()
            v = wp.to_torch(self.simulator.wp_states[-1].wp_v, requires_grad=False).clone()
            last_state = {"x": x, "v": v}
            self._restore_sim_state(last_state)
            max_speed = float(torch.linalg.norm(v, dim=1).max().item())
            if max_speed <= self.IMMERSIVE_SCENE_REST_VELOCITY_EPS:
                break
        if last_state is None:
            last_state = self._capture_sim_state()
        if progress_callback is not None:
            progress_callback("settle_complete")
        return last_state

    def _snap_to_scene_rest_if_idle(self, scene_rest_state, controller_interaction_state):
        if scene_rest_state is None:
            return False
        if controller_interaction_state["left"] is not None or controller_interaction_state["right"] is not None:
            return False
        current_x = wp.to_torch(self.simulator.wp_states[0].wp_x, requires_grad=False)
        current_v = wp.to_torch(self.simulator.wp_states[0].wp_v, requires_grad=False)
        max_speed = float(torch.linalg.norm(current_v, dim=1).max().item())
        if max_speed > self.IMMERSIVE_SCENE_REST_VELOCITY_EPS:
            return False
        mean_position_delta = float(
            torch.linalg.norm(current_x - scene_rest_state["x"], dim=1).mean().item()
        )
        if mean_position_delta > self.IMMERSIVE_SCENE_REST_POSITION_EPS:
            return False
        self._restore_sim_state(scene_rest_state)
        return True

    def _ray_aabb_intersection(self, origin, direction, bounds_min, bounds_max):
        eps = 1e-6
        parallel = torch.abs(direction) <= eps
        outside_parallel = parallel & ((origin < bounds_min) | (origin > bounds_max))
        if bool(outside_parallel.any().item()):
            return None

        safe_direction = torch.where(parallel, torch.ones_like(direction), direction)
        t1 = (bounds_min - origin) / safe_direction
        t2 = (bounds_max - origin) / safe_direction
        t_lower = torch.where(parallel, torch.full_like(t1, -1e9), torch.minimum(t1, t2))
        t_upper = torch.where(parallel, torch.full_like(t2, 1e9), torch.maximum(t1, t2))
        t_near = torch.max(t_lower)
        t_far = torch.min(t_upper)
        if float(t_far.item()) < max(float(t_near.item()), 0.0):
            return None

        t_hit = t_near if float(t_near.item()) >= 0.0 else t_far
        if float(t_hit.item()) < 0.0:
            return None
        return origin + direction * t_hit

    def _ray_object_intersection(
        self,
        origin,
        direction,
        object_points,
        bounds_min,
        bounds_max,
    ):
        hit_aabb = self._ray_aabb_intersection(origin, direction, bounds_min, bounds_max)
        if hit_aabb is None:
            return None

        delta = object_points - origin.unsqueeze(0)
        t = torch.sum(delta * direction.unsqueeze(0), dim=1)
        valid_t = t >= 0.0
        if not bool(valid_t.any().item()):
            return None

        projected = origin.unsqueeze(0) + t.unsqueeze(1) * direction.unsqueeze(0)
        ray_distance = torch.linalg.norm(object_points - projected, dim=1)
        hit_mask = valid_t & (ray_distance <= self.LIVE_CONTROLLER_HIT_WORLD_RADIUS)
        if not bool(hit_mask.any().item()):
            return None

        hit_t = t[hit_mask].min()
        return origin + direction * hit_t

    def _build_live_controller_overlay(
        self,
        source,
        controller_world,
        intrinsic,
        w2c,
        height,
        width,
        object_points,
        object_bounds_min,
        object_bounds_max,
    ):
        if controller_world is None:
            return None

        origin_world, direction_world = self._controller_world_ray_pose(controller_world)
        if origin_world is None or direction_world is None:
            return None
        hit_world = self._ray_object_intersection(
            origin_world,
            direction_world,
            object_points,
            object_bounds_min,
            object_bounds_max,
        )
        ray_end_world = (
            hit_world
            if hit_world is not None
            else origin_world + direction_world * self.LIVE_CONTROLLER_RAY_LENGTH
        )

        projected_world = [origin_world.unsqueeze(0), ray_end_world.unsqueeze(0)]
        hit_index = None
        if hit_world is not None:
            hit_index = len(projected_world)
            projected_world.append(hit_world.unsqueeze(0))

        pixels, depth_valid = self._project_points_to_pixels(
            torch.cat(projected_world, dim=0), intrinsic, w2c
        )
        if not bool(depth_valid[0].item()) or not bool(depth_valid[1].item()):
            return None

        origin_pixel = pixels[0]
        end_pixel = pixels[1]
        hit_pixel = None
        if hit_index is not None and bool(depth_valid[hit_index].item()):
            hit_pixel = pixels[hit_index]

        return {
            "source": source,
            "origin_pixel": origin_pixel,
            "end_pixel": end_pixel,
            "hit_pixel": hit_pixel,
            "hit_world": hit_world,
            "ray_end_world": ray_end_world,
            "color": (
                self.LIVE_CONTROLLER_LEFT_COLOR
                if source == "left"
                else self.LIVE_CONTROLLER_RIGHT_COLOR
            ),
            "select_available": controller_world["select_available"],
            "select_pressed": controller_world.get(
                "select_hold_active",
                controller_world["select_pressed"],
            ),
            "select_value": controller_world["select_value"],
            "select_source": controller_world["select_source"],
            "anchor_cycle_available": controller_world["anchor_cycle_available"],
            "anchor_cycle_pressed": controller_world["anchor_cycle_pressed"],
            "anchor_cycle_source": controller_world["anchor_cycle_source"],
            "snap_assist_available": controller_world["snap_assist_available"],
            "snap_assist_pressed": controller_world["snap_assist_pressed"],
            "snap_assist_source": controller_world["snap_assist_source"],
        }

    def _controller_source_target_points(
        self,
        source,
        anchor,
        base_target,
        controller_source_masks,
        controller_source_anchor_centers,
    ):
        source_index = self._controller_source_index(source)
        source_mask = controller_source_masks[source_index]
        source_points = base_target[source_mask].clone()
        if anchor is None:
            return source_points
        return source_points + (anchor - controller_source_anchor_centers[source_index])

    def _build_controller_attachment_candidate(
        self,
        source,
        anchor,
        hit_world,
        predefined_anchor,
        object_points,
        base_target,
        controller_source_masks,
        controller_source_anchor_centers,
        controller_attachment_metadata,
    ):
        if predefined_anchor is None:
            return None
        source_meta = controller_attachment_metadata[source]
        source_points = self._controller_source_target_points(
            source,
            anchor,
            base_target,
            controller_source_masks,
            controller_source_anchor_centers,
        )
        if source_points.shape[0] == 0:
            return None

        candidate_pool = predefined_anchor["region_indices"]
        if candidate_pool.numel() == 0:
            return None

        template_springs = source_meta["template_springs"].clone()
        new_springs = template_springs.clone()
        new_rest_lengths = source_meta["template_rest_lengths"].clone()
        selected_object_indices = []
        point_indices = source_meta["point_indices"]
        spring_capable_offsets = [
            point_offset
            for point_offset, point_idx in enumerate(point_indices)
            if source_meta["point_spring_positions"][point_idx]
        ]
        active_point_count = len(spring_capable_offsets)
        if active_point_count <= 0:
            return None
        spring_capable_offsets = torch.as_tensor(
            spring_capable_offsets, dtype=torch.long, device=source_points.device
        )
        active_source_point_offsets = spring_capable_offsets[
            torch.topk(
                torch.linalg.norm(
                    source_points[spring_capable_offsets] - anchor.unsqueeze(0), dim=1
                ),
                k=active_point_count,
                largest=False,
            ).indices
        ]
        active_local_target_points = []
        resolved_active_point_offsets = []
        remaining_candidate_mask = torch.ones(
            candidate_pool.shape[0], dtype=torch.bool, device=candidate_pool.device
        )

        for point_offset in active_source_point_offsets.tolist():
            point_idx = point_indices[point_offset]
            spring_positions = source_meta["point_spring_positions"][point_idx]
            if not spring_positions:
                continue

            control_point = source_points[point_offset]
            active_candidate_pool = candidate_pool[remaining_candidate_mask]
            if active_candidate_pool.numel() == 0:
                active_candidate_pool = candidate_pool
            local_distances = torch.linalg.norm(
                object_points[active_candidate_pool] - control_point.unsqueeze(0), dim=1
            )
            nearest_count = min(len(spring_positions), int(active_candidate_pool.numel()))
            nearest = torch.topk(local_distances, k=nearest_count, largest=False).indices
            if nearest_count < len(spring_positions):
                repeats = (len(spring_positions) + nearest_count - 1) // nearest_count
                nearest = nearest.repeat(repeats)[: len(spring_positions)]
            selected = active_candidate_pool[nearest]
            selected_object_indices.extend(int(idx.item()) for idx in selected)
            attached_points = object_points[selected]
            active_local_target_points.append(attached_points.mean(dim=0))
            resolved_active_point_offsets.append(point_offset)
            if bool(remaining_candidate_mask.any().item()):
                consumed = torch.zeros_like(remaining_candidate_mask)
                selected_view = selected.view(-1, 1)
                consumed |= torch.any(candidate_pool.unsqueeze(1) == selected_view.T, dim=1)
                remaining_candidate_mask &= ~consumed

            for spring_position, object_idx in zip(spring_positions, selected):
                if int(new_springs[spring_position, 0].item()) < int(object_points.shape[0]):
                    new_springs[spring_position, 0] = object_idx.to(new_springs.dtype)
                else:
                    new_springs[spring_position, 1] = object_idx.to(new_springs.dtype)
                rest_length = torch.linalg.norm(
                    control_point - object_points[int(object_idx.item())]
                ).clamp_min(1e-4).clamp_max(self.LIVE_CONTROLLER_ATTACH_MAX_REST_LENGTH)
                new_rest_lengths[spring_position] = rest_length

        if not selected_object_indices:
            return None

        selected_object_indices = torch.as_tensor(
            selected_object_indices, dtype=torch.long, device=cfg.device
        )
        unique_selected_object_indices = torch.unique(selected_object_indices)
        attach_points = object_points[unique_selected_object_indices]
        kinematic_object_indices = predefined_anchor["region_indices"].clone()
        kinematic_reference_positions = object_points[kinematic_object_indices].clone()
        attach_center_world = predefined_anchor["center_world"].clone()
        attach_radius = torch.linalg.norm(
            kinematic_reference_positions - attach_center_world.unsqueeze(0), dim=1
        ).max()
        active_local_target_points = torch.stack(active_local_target_points, dim=0)
        active_source_point_offsets = torch.as_tensor(
            resolved_active_point_offsets,
            dtype=torch.long,
            device=source_points.device,
        )
        attach_anchor_world = (
            hit_world.clone() if anchor is None else anchor.clone()
        )
        active_local_target_offsets = (
            active_local_target_points - attach_anchor_world.unsqueeze(0)
        )

        return {
            "anchor_name": predefined_anchor["name"],
            "springs": new_springs,
            "rest_lengths": new_rest_lengths,
            "attach_center_world": attach_center_world,
            "attach_node_count": int(kinematic_object_indices.numel()),
            "attach_radius": float(attach_radius.item()),
            "attach_anchor_world": attach_anchor_world,
            "active_source_point_offsets": active_source_point_offsets,
            "active_local_target_points": active_local_target_points,
            "active_local_target_offsets": active_local_target_offsets,
            "selected_object_indices": kinematic_object_indices,
            "selected_object_reference_positions": kinematic_reference_positions,
        }

    def _apply_controller_attachment_remap(
        self,
        source,
        remap_candidate,
        controller_attachment_metadata,
    ):
        source_meta = controller_attachment_metadata[source]
        self.simulator.update_local_spring_subset(
            source_meta["spring_indices"],
            remap_candidate["springs"],
            remap_candidate["rest_lengths"],
        )
        spring_y = remap_candidate.get("spring_y")
        if spring_y is not None:
            self.simulator.update_local_spring_stiffness_subset(
                source_meta["spring_indices"],
                spring_y,
            )

    def _restore_controller_attachment_remap(self, source, controller_attachment_metadata):
        source_meta = controller_attachment_metadata[source]
        if (
            "template_springs" in source_meta
            and "template_rest_lengths" in source_meta
        ):
            self.simulator.update_local_spring_subset(
                source_meta["spring_indices"],
                source_meta["template_springs"],
                source_meta["template_rest_lengths"],
            )
        inactive_spring_y = source_meta.get("inactive_spring_y")
        if inactive_spring_y is not None:
            self.simulator.update_local_spring_stiffness_subset(
                source_meta["spring_indices"],
                inactive_spring_y,
            )

    def _controller_source_index(self, source):
        return 0 if source == "left" else 1

    def _start_live_controller_interaction(
        self,
        source,
        controller_world,
        target_anchor_world,
        controller_source_anchor_centers,
        translation_only=False,
    ):
        source_index = self._controller_source_index(source)
        controller_position_world = controller_world["position"]
        ray_origin_world, ray_direction_world = self._controller_world_ray_pose(
            controller_world
        )
        if ray_origin_world is not None and ray_direction_world is not None:
            ray_distance = torch.dot(
                target_anchor_world - ray_origin_world,
                ray_direction_world,
            ).clamp_min(0.0)
            projected_anchor_world = ray_origin_world + ray_direction_world * ray_distance
            anchor_offset = target_anchor_world - projected_anchor_world
            ray_distance_value = float(ray_distance.item())
            projected_anchor_distance = float(
                torch.linalg.norm(target_anchor_world - projected_anchor_world).item()
            )
        else:
            ray_distance_value = 0.0
            projected_anchor_world = controller_position_world
            anchor_offset = target_anchor_world - controller_position_world
            projected_anchor_distance = 0.0
        target_delta = torch.linalg.norm(
            target_anchor_world - controller_source_anchor_centers[source_index]
        )
        interaction_state = {
            "source": source,
            "ray_distance": ray_distance_value,
            "anchor_offset": anchor_offset,
            "target_delta": float(target_delta.item()),
            "projected_anchor_distance": projected_anchor_distance,
            "just_started": True,
            "start_controller_position_world": controller_position_world.clone(),
            "start_anchor_world": target_anchor_world.clone(),
        }
        if translation_only:
            interaction_state.update(
                {
                    "translation_only": True,
                    "grab_controller_position_world": controller_position_world.clone(),
                    "grab_attach_anchor_world": target_anchor_world.clone(),
                }
            )
        return interaction_state

    def _controller_interaction_anchor(self, controller_world, interaction_state):
        if interaction_state.get("translation_only", False):
            grab_controller_position_world = interaction_state.get(
                "grab_controller_position_world"
            )
            grab_attach_anchor_world = interaction_state.get("grab_attach_anchor_world")
            if grab_controller_position_world is None or grab_attach_anchor_world is None:
                return None
            return grab_attach_anchor_world + (
                controller_world["position"] - grab_controller_position_world
            )
        ray_origin_world, ray_direction_world = self._controller_world_ray_pose(
            controller_world
        )
        if ray_origin_world is None or ray_direction_world is None:
            return None
        return (
            ray_origin_world
            + ray_direction_world * float(interaction_state["ray_distance"])
            + interaction_state["anchor_offset"]
        )

    def _current_controller_attach_center_world(self, interaction_state, current_anchor):
        if interaction_state is None:
            return None
        attach_center_world = interaction_state.get("attach_center_world")
        attach_anchor_world = interaction_state.get("attach_anchor_world")
        if (
            attach_center_world is None
            or attach_anchor_world is None
            or current_anchor is None
        ):
            return attach_center_world
        return attach_center_world + (current_anchor - attach_anchor_world)

    def _clear_live_controller_interaction(
        self,
        source,
        controller_interaction_state,
        controller_attachment_metadata,
        reason=None,
    ):
        interaction_state = controller_interaction_state.get(source)
        if interaction_state is None:
            return None
        if interaction_state.get("spring_remap_applied", False):
            self._restore_controller_attachment_remap(
                source,
                controller_attachment_metadata,
            )
        if reason is not None:
            self._log_controller_interaction_end(source, interaction_state, reason)
        controller_interaction_state[source] = None
        return interaction_state

    def _resolve_live_controller_interaction_anchors(
        self,
        left_controller_world,
        right_controller_world,
        controller_overlay_by_source,
        controller_interaction_state,
        controller_source_anchor_centers,
        controller_attachment_metadata,
        controller_anchor_templates,
        controller_predefined_anchor_states,
        controller_anchor_preview_state,
        object_points,
        allow_implicit_fallback_start=True,
        post_select_translation_only=False,
    ):
        anchors = {"left": None, "right": None}
        controller_world_by_source = {
            "left": left_controller_world,
            "right": right_controller_world,
        }
        for source, controller_world in controller_world_by_source.items():
            interaction_state = controller_interaction_state[source]
            if controller_world is None:
                if interaction_state is not None:
                    self._clear_live_controller_interaction(
                        source,
                        controller_interaction_state,
                        controller_attachment_metadata,
                        reason="controller_invalid",
                    )
                self._reset_controller_anchor_preview_state(
                    controller_anchor_preview_state,
                    source,
                )
                continue

            select_start_edge = bool(controller_world.get("select_start_edge", False))
            select_hold_active = bool(controller_world.get("select_hold_active", False))
            select_release_frames = int(
                controller_world.get("select_release_frames", 0)
            )
            if (
                interaction_state is not None
                and (not select_hold_active)
                and select_release_frames >= self.LIVE_CONTROLLER_SELECT_RELEASE_FRAMES
            ):
                self._clear_live_controller_interaction(
                    source,
                    controller_interaction_state,
                    controller_attachment_metadata,
                    reason="released",
                )
                interaction_state = None
            if interaction_state is None and select_start_edge:
                overlay = controller_overlay_by_source.get(source)
                preview_state = controller_anchor_preview_state[source]
                cycle_locked = bool(preview_state.get("cycle_locked", False))
                selected_rank_index = int(preview_state.get("selected_rank_index", 0))
                preview_anchor_name = preview_state.get("selected_anchor_name")
                selected_preview_anchor = self._anchor_state_by_name(
                    controller_predefined_anchor_states,
                    preview_anchor_name,
                )
                selected_preview_anchor_occupied, _ = (
                    self._anchor_is_occupied_by_other_source(
                        source,
                        preview_anchor_name,
                        controller_interaction_state,
                    )
                )
                if selected_preview_anchor_occupied:
                    selected_preview_anchor = None
                hit_world = None if overlay is None else overlay.get("hit_world")
                ray_origin_world, ray_direction_world = self._controller_world_ray_pose(
                    controller_world
                )
                grab_start_mode = "cycled_locked" if cycle_locked else "nearest"
                snapped_anchor = selected_preview_anchor
                if snapped_anchor is None:
                    if not allow_implicit_fallback_start:
                        continue
                    available_anchor_states = (
                        self._filter_available_predefined_interaction_anchors(
                            source,
                            controller_predefined_anchor_states,
                            controller_interaction_state,
                        )
                    )
                    ranked_anchors = self._rank_predefined_interaction_anchors(
                        hit_world,
                        ray_origin_world,
                        ray_direction_world,
                        available_anchor_states,
                        require_selection_radius=False,
                    )
                    snapped_anchor = None if not ranked_anchors else ranked_anchors[0]
                    if snapped_anchor is not None:
                        preview_anchor_name = snapped_anchor["name"]
                        if not cycle_locked:
                            preview_state["selected_rank_index"] = 0
                            preview_state["selected_anchor_name"] = preview_anchor_name
                            preview_state["visible"] = True
                            preview_state["current_candidate_names"] = [
                                anchor["name"] for anchor in ranked_anchors
                            ]
                            preview_state["current_selected_rank"] = 1
                            preview_state["current_candidate_count"] = len(ranked_anchors)
                if snapped_anchor is None:
                    continue
                preview_anchor_visible = True
                preview_anchor_resolved = True
                remap_candidate = self._instantiate_predefined_controller_anchor_template(
                    source,
                    snapped_anchor,
                    object_points,
                    controller_anchor_templates,
                    controller_attachment_metadata,
                )
                if remap_candidate is None:
                    preview_interaction_state = {
                        "grab_start_mode": grab_start_mode,
                        "preview_anchor_visible": preview_anchor_visible,
                        "preview_anchor_name": preview_anchor_name,
                        "preview_anchor_resolved": True,
                        "hit_present": bool(hit_world is not None),
                        "start_reference": (
                            "ray_hit" if hit_world is not None else "anchor_center"
                        ),
                    }
                    reason = (
                        f"selected_preview_anchor_template_missing(name={preview_anchor_name})"
                    )
                    self._log_controller_interaction_start_attempt(
                        source,
                        preview_interaction_state,
                        reason=reason,
                    )
                    self._log_controller_interaction_rejected(
                        source,
                        None,
                        preview_interaction_state,
                        reason,
                    )
                    continue
                if remap_candidate is None:
                    continue
                interaction_state = self._start_live_controller_interaction(
                    source,
                    controller_world,
                    remap_candidate["attach_anchor_world"].clone(),
                    controller_source_anchor_centers,
                    translation_only=bool(post_select_translation_only),
                )
                interaction_state.update(
                    {
                        "grab_start_mode": grab_start_mode,
                        "preview_anchor_visible": preview_anchor_visible,
                        "preview_anchor_name": preview_anchor_name,
                        "preview_anchor_resolved": preview_anchor_resolved,
                        "hit_present": bool(hit_world is not None),
                        "cycle_locked": cycle_locked,
                        "selected_rank_index": selected_rank_index,
                        "start_reference": (
                            "ray_hit" if hit_world is not None else "anchor_center"
                        ),
                    }
                )
                self._log_controller_interaction_start_attempt(
                    source,
                    interaction_state,
                    remap_candidate=remap_candidate,
                    reason="candidate_built",
                )
                rejection_reason, validation_debug = self._validate_new_controller_interaction_candidate(
                    source,
                    controller_world,
                    interaction_state,
                    remap_candidate,
                    hit_world,
                    ray_direction_world,
                    controller_interaction_state=controller_interaction_state,
                    explicit_preview_selected=True,
                )
                if rejection_reason is not None:
                    self._log_controller_interaction_rejected(
                        source,
                        remap_candidate,
                        interaction_state,
                        rejection_reason,
                    )
                    continue
                if validation_debug.get("projected_anchor_distance_bypassed", False):
                    print(
                        "[live_openxr_controller] "
                        f"{source} interaction_projected_anchor_bypass=1 "
                        f"anchor={remap_candidate['anchor_name']} "
                        f"projected_anchor={validation_debug['projected_anchor_distance']:.4f} "
                        f"strict_limit={validation_debug['strict_projected_anchor_distance_limit']:.4f} "
                        f"hit_to_anchor={validation_debug.get('hit_distance')}",
                        flush=True,
                    )
                self._apply_controller_attachment_remap(
                    source, remap_candidate, controller_attachment_metadata
                )
                controller_anchor_preview_state[source]["visible"] = True
                controller_anchor_preview_state[source]["selected_anchor_name"] = (
                    remap_candidate["anchor_name"]
                )
                interaction_state.update(
                    {
                        "kinematic_only": False,
                        "spring_remap_applied": True,
                        "anchor_name": remap_candidate["anchor_name"],
                        "attach_center_world": remap_candidate["attach_center_world"],
                        "attach_node_count": remap_candidate["attach_node_count"],
                        "attach_radius": remap_candidate["attach_radius"],
                        "attach_anchor_world": remap_candidate["attach_anchor_world"],
                        "source_template_offsets": remap_candidate["source_template_offsets"],
                        "selected_object_indices": remap_candidate["selected_object_indices"],
                        "target_point_indices": remap_candidate.get("target_point_indices"),
                        "multi_points_debug": remap_candidate.get("multi_points_debug"),
                        "hit_to_anchor_distance": remap_candidate.get("hit_to_anchor_distance"),
                        "explicit_preview_selected": True,
                        "grab_start_mode": grab_start_mode,
                        "preview_anchor_visible": preview_anchor_visible,
                        "preview_anchor_name": preview_anchor_name,
                        "preview_anchor_resolved": preview_anchor_resolved,
                        "startup_validation_delay_frames": self.IMMERSIVE_GRAB_START_VALIDATION_DELAY_FRAMES,
                        "startup_validation_frames_remaining": self.IMMERSIVE_GRAB_START_VALIDATION_FRAMES,
                    }
                )
                self._log_controller_interaction_rejected(
                    source,
                    remap_candidate,
                    interaction_state,
                    "grab_start_ok",
                    action="accepted",
                )
                controller_interaction_state[source] = interaction_state

            interaction_state = controller_interaction_state[source]
            if interaction_state is not None:
                anchors[source] = self._controller_interaction_anchor(
                    controller_world,
                    interaction_state,
                )

        return anchors["left"], anchors["right"]

    def _preview_live_controller_attachment(
        self,
        source,
        controller_world,
        overlay,
        selected_predefined_anchor,
        predefined_anchor_states,
        object_points,
        controller_source_anchor_centers,
        controller_attachment_metadata,
        controller_anchor_templates,
    ):
        if overlay is None or overlay["hit_world"] is None or controller_world is None:
            return None, None

        predefined_anchor = selected_predefined_anchor
        if predefined_anchor is None:
            predefined_anchor = self._select_predefined_interaction_anchor(
                overlay["hit_world"], predefined_anchor_states
            )
        if predefined_anchor is None:
            return None, None
        remap_candidate = self._instantiate_predefined_controller_anchor_template(
            source,
            predefined_anchor,
            object_points,
            controller_anchor_templates,
            controller_attachment_metadata,
        )
        if remap_candidate is None:
            return None, None
        interaction_state = self._start_live_controller_interaction(
            source,
            controller_world,
            remap_candidate["attach_center_world"],
            controller_source_anchor_centers,
        )
        return interaction_state, remap_candidate

    def _draw_live_controller_overlay(self, frame, controller_overlays):
        for overlay in controller_overlays:
            color = overlay["color"]
            origin_pixel = overlay["origin_pixel"]
            end_pixel = overlay["end_pixel"]
            hit_pixel = overlay["hit_pixel"]
            candidate_pixel = overlay.get("attach_candidate_pixel")
            active_pixel = overlay.get("attach_active_pixel")
            attach_candidate = overlay.get("attach_candidate", False)
            attachment_active = overlay.get("attachment_active", False)
            anchor_preview_entries = overlay.get("anchor_preview_entries", [])

            self._draw_marker_line(
                frame,
                origin_pixel,
                end_pixel,
                color,
                radius=1,
                blend=0.68 if attachment_active else (0.52 if hit_pixel is not None else 0.34),
            )
            self._blend_marker(
                frame,
                origin_pixel,
                color,
                radius=self.LIVE_CONTROLLER_ORIGIN_RADIUS,
                blend=0.85,
            )

            if attach_candidate and candidate_pixel is not None:
                self._blend_square_marker(
                    frame,
                    candidate_pixel,
                    self.LIVE_CONTROLLER_ATTACH_CANDIDATE_COLOR,
                    radius=self.LIVE_CONTROLLER_CANDIDATE_SQUARE_RADIUS,
                    blend=0.86,
                )
                self._blend_marker(
                    frame,
                    candidate_pixel,
                    self.LIVE_CONTROLLER_SELECT_COLOR,
                    radius=1,
                    blend=0.98,
                )

            for preview_entry in anchor_preview_entries:
                preview_pixel = preview_entry["pixel"]
                preview_selected = preview_entry["selected"]
                preview_active = preview_entry["active"]
                preview_occupied = bool(preview_entry.get("occupied", False))
                preview_color = (
                    self.LIVE_CONTROLLER_ATTACH_ACTIVE_COLOR
                    if preview_active
                    else (
                        self.LIVE_CONTROLLER_PREVIEW_OCCUPIED_COLOR
                        if preview_occupied
                        else color
                    )
                )
                preview_radius = (
                    self.LIVE_CONTROLLER_PREVIEW_SELECTED_RADIUS
                    if preview_selected
                    else self.LIVE_CONTROLLER_PREVIEW_RADIUS
                )
                preview_blend = 0.78 if preview_selected else (0.42 if preview_occupied else 0.32)
                self._blend_marker(
                    frame,
                    preview_pixel,
                    preview_color,
                    radius=preview_radius,
                    blend=preview_blend,
                )
                if preview_selected:
                    self._blend_marker(
                        frame,
                        preview_pixel,
                        self.LIVE_CONTROLLER_SELECT_COLOR,
                        radius=1,
                        blend=0.96,
                    )

            if overlay["select_available"]:
                indicator_pixel = origin_pixel + frame.new_tensor([0.0, -10.0])
                indicator_color = (
                    self.LIVE_CONTROLLER_SELECT_COLOR
                    if overlay["select_pressed"]
                    else self.LIVE_CONTROLLER_SELECT_IDLE_COLOR
                )
                self._blend_marker(
                    frame,
                    indicator_pixel,
                    indicator_color,
                    radius=self.LIVE_CONTROLLER_INDICATOR_RADIUS,
                    blend=0.92,
                )
                if overlay["select_pressed"]:
                    self._blend_marker(
                        frame,
                        origin_pixel,
                        self.LIVE_CONTROLLER_SELECT_COLOR,
                        radius=1,
                        blend=0.98,
                )

            if attachment_active:
                marker_pixel = active_pixel
                if marker_pixel is not None:
                    self._blend_marker(
                        frame,
                        marker_pixel,
                        color,
                        radius=self.LIVE_CONTROLLER_HIT_RADIUS + 3,
                        blend=0.82,
                    )
                    self._blend_marker(
                        frame,
                        marker_pixel,
                        self.LIVE_CONTROLLER_SELECT_COLOR,
                        radius=1,
                        blend=0.98,
                    )

    def _log_controller_select_transition(
        self,
        source,
        controller_sample,
        state_cache,
        sample_id=None,
    ):
        if controller_sample is None:
            return

        state = (
            bool(controller_sample.select_available),
            bool(controller_sample.select_pressed),
            round(float(controller_sample.select_value), 3),
            str(controller_sample.select_source),
        )
        if state_cache.get(source) == state:
            return

        print(
            "[live_openxr_controller] "
            + ("" if sample_id is None else f"sample={int(sample_id)} ")
            +
            f"{source} select available={int(state[0])} "
            f"pressed={int(state[1])} value={state[2]:.3f} source={state[3]}"
            ,
            flush=True,
        )
        state_cache[source] = state

    def _log_controller_select_hold_transition(
        self,
        source,
        select_hold_runtime,
        state_cache,
        sample_id=None,
    ):
        if select_hold_runtime is None:
            return

        state = (
            bool(select_hold_runtime["start_edge"]),
            bool(select_hold_runtime["hold_active"]),
            bool(select_hold_runtime["release_ready"]),
            int(select_hold_runtime["release_frames"]),
            round(float(select_hold_runtime["value"]), 3),
        )
        if state_cache.get(source) == state:
            return

        print(
            "[live_openxr_controller] "
            + ("" if sample_id is None else f"sample={int(sample_id)} ")
            +
            f"{source} select_hold start_edge={int(state[0])} "
            f"hold_active={int(state[1])} release_ready={int(state[2])} "
            f"release_frames={state[3]} value={state[4]:.3f}",
            flush=True,
        )
        state_cache[source] = state

    def _log_controller_anchor_cycle_transition(self, source, controller_sample, state_cache):
        if controller_sample is None:
            return

        state = (
            bool(controller_sample.anchor_cycle_available),
            bool(controller_sample.anchor_cycle_pressed),
            str(controller_sample.anchor_cycle_source),
        )
        if state_cache.get(source) == state:
            return

        print(
            "[live_openxr_controller] "
            f"{source} anchor_cycle available={int(state[0])} "
            f"pressed={int(state[1])} source={state[2]}",
            flush=True,
        )
        state_cache[source] = state

    def _log_controller_snap_transition(self, source, controller_sample, state_cache):
        if controller_sample is None:
            return

        state = (
            bool(controller_sample.snap_assist_available),
            bool(controller_sample.snap_assist_pressed),
            str(controller_sample.snap_assist_source),
        )
        if state_cache.get(source) == state:
            return

        print(
            "[live_openxr_controller] "
            f"{source} snap_assist available={int(state[0])} "
            f"pressed={int(state[1])} source={state[2]}",
            flush=True,
        )
        state_cache[source] = state

    def _log_controller_exit_transition(self, source, controller_sample, state_cache):
        if controller_sample is None:
            return

        state = (
            bool(getattr(controller_sample, "exit_available", False)),
            bool(getattr(controller_sample, "exit_pressed", False)),
            round(float(getattr(controller_sample, "exit_value", 0.0)), 3),
            str(getattr(controller_sample, "exit_source", "none")),
        )
        if state_cache.get(source) == state:
            return

        print(
            "[live_openxr_controller] "
            f"{source} exit available={int(state[0])} "
            f"pressed={int(state[1])} value={state[2]:.3f} source={state[3]}",
            flush=True,
        )
        state_cache[source] = state

    def _log_controller_anchor_preview_transition(self, source, preview_state, state_cache):
        visible = bool(preview_state.get("visible", False))
        cycle_locked = bool(preview_state.get("cycle_locked", False))
        state = (
            visible,
            cycle_locked,
            preview_state.get("selected_anchor_name") if visible else None,
            preview_state.get("current_selected_rank") if visible else None,
            preview_state.get("current_candidate_count"),
        )
        if state_cache.get(source) == state:
            return

        if not state[0]:
            print(
                "[live_openxr_controller] "
                f"{source} anchor_preview=0 "
                f"mode={'cycled_locked' if state[1] else 'nearest'}",
                flush=True,
            )
        else:
            print(
                "[live_openxr_controller] "
                f"{source} mode={'cycled_locked' if state[1] else 'nearest'} "
                "anchor_preview=1 "
                f"selected={state[2]} "
                f"rank={state[3]}/{state[4]}",
                flush=True,
            )
        state_cache[source] = state

    def _log_controller_hit_transition(self, source, overlay, state_cache):
        state = bool(overlay is not None and overlay["hit_world"] is not None)
        if state_cache.get(source) == state:
            return

        print(f"[live_openxr_controller] {source} ray_hit={int(state)}", flush=True)
        state_cache[source] = state

    def _log_controller_attach_candidate_transition(self, source, overlay, state_cache):
        if overlay is None or not overlay.get("attach_candidate", False):
            state = None
        else:
            candidate_data = overlay.get("attach_candidate_data")
            if candidate_data is not None:
                state = (
                    candidate_data["anchor_name"],
                    int(candidate_data["attach_node_count"]),
                    round(float(candidate_data["attach_radius"]), 3),
                )
            else:
                state = (
                    overlay.get("attach_candidate_anchor_name"),
                    None,
                    None,
                )
        if state_cache.get(source) == state:
            return

        if state is None:
            print(f"[live_openxr_controller] {source} attach_candidate=0", flush=True)
        else:
            if state[1] is None or state[2] is None:
                print(
                    "[live_openxr_controller] "
                    f"{source} attach_candidate=1 anchor={state[0]}",
                    flush=True,
                )
            else:
                print(
                    "[live_openxr_controller] "
                    f"{source} attach_candidate=1 anchor={state[0]} "
                    f"nodes={state[1]} radius={state[2]:.3f}",
                    flush=True,
                )
        state_cache[source] = state

    def _log_controller_interaction_transition(self, source, interaction_state, state_cache):
        if interaction_state is None:
            state = None
        else:
            multi_points_debug = interaction_state.get("multi_points_debug")
            if multi_points_debug is None:
                multi_points_state = None
            elif multi_points_debug.get("fallback_used", False):
                multi_points_state = ("fallback", multi_points_debug.get("reason"))
            else:
                multi_points_state = (
                    int(multi_points_debug.get("seed_index", -1)),
                    int(multi_points_debug.get("patch_size", 0)),
                    round(float(multi_points_debug.get("depth_min", 0.0)), 3),
                    round(float(multi_points_debug.get("depth_max", 0.0)), 3),
                )
            state = (
                interaction_state.get("anchor_name", "unknown"),
                round(float(interaction_state["ray_distance"]), 3),
                round(float(interaction_state["target_delta"]), 3),
                int(interaction_state.get("attach_node_count", 0)),
                round(float(interaction_state.get("attach_radius", 0.0)), 3),
                multi_points_state,
            )
        if state_cache.get(source) == state:
            return

        if state is None:
            print(f"[live_openxr_controller] {source} interaction=0", flush=True)
        else:
            multi_points_suffix = ""
            if state[5] is not None:
                if state[5][0] == "fallback":
                    multi_points_suffix = f" multi_points_fallback={state[5][1]}"
                else:
                    multi_points_suffix = (
                        f" multi_points_seed={state[5][0]} multi_points_patch={state[5][1]} "
                        f"multi_points_depth=[{state[5][2]:.3f},{state[5][3]:.3f}]"
                    )
            print(
                "[live_openxr_controller] "
                f"{source} interaction=1 anchor={state[0]} "
                f"ray_distance={state[1]:.3f} "
                f"target_delta={state[2]:.3f} "
                f"nodes={state[3]} radius={state[4]:.3f}"
                f"{multi_points_suffix}"
                ,
                flush=True,
            )
        state_cache[source] = state

    def _log_controller_interaction_end(self, source, interaction_state, reason):
        if interaction_state is None:
            return
        print(
            "[live_openxr_controller] "
            f"{source} interaction_end=1 "
            f"anchor={interaction_state.get('anchor_name')} "
            f"reason={reason}",
            flush=True,
        )

    def _controller_target_point_indices_for_state(
        self,
        source,
        controller_source_masks,
        interaction_state=None,
    ):
        if interaction_state is not None:
            target_point_indices = interaction_state.get("target_point_indices")
            if target_point_indices is not None:
                return target_point_indices.to(device=cfg.device, dtype=torch.long)
        source_index = self._controller_source_index(source)
        return torch.nonzero(
            controller_source_masks[source_index],
            as_tuple=False,
        ).squeeze(1)

    def _controller_spring_subset_stats(self, source, controller_attachment_metadata):
        source_meta = controller_attachment_metadata.get(source)
        if source_meta is None:
            return {
                "spring_active": False,
                "spring_mean": 0.0,
                "spring_max": 0.0,
            }
        spring_indices = source_meta.get("spring_indices")
        if (
            spring_indices is None
            or not hasattr(self.simulator, "torch_spring_Y_clamped")
            or int(spring_indices.numel()) <= 0
        ):
            return {
                "spring_active": False,
                "spring_mean": 0.0,
                "spring_max": 0.0,
            }
        spring_values = self.simulator.torch_spring_Y_clamped[
            spring_indices.to(
                device=self.simulator.torch_spring_Y_clamped.device,
                dtype=torch.long,
            )
        ]
        if int(spring_values.numel()) <= 0:
            return {
                "spring_active": False,
                "spring_mean": 0.0,
                "spring_max": 0.0,
            }
        spring_mean = float(spring_values.mean().item())
        spring_max = float(spring_values.max().item())
        return {
            "spring_active": bool((spring_values > 1e-6).any().item()),
            "spring_mean": spring_mean,
            "spring_max": spring_max,
        }

    def _log_controller_motion_parity(
        self,
        runtime_label,
        frame_index,
        source,
        controller_world,
        current_anchor,
        next_target,
        controller_runtime_base_target,
        controller_source_masks,
        controller_attachment_metadata,
        interaction_state,
        state_cache,
    ):
        if (
            interaction_state is None
            or controller_world is None
            or current_anchor is None
            or next_target is None
            or controller_runtime_base_target is None
        ):
            state_cache[source] = None
            return

        target_point_indices = self._controller_target_point_indices_for_state(
            source,
            controller_source_masks,
            interaction_state,
        )
        if int(target_point_indices.numel()) <= 0:
            state_cache[source] = None
            return

        current_controller_position = controller_world["position"]
        current_target_points = next_target[target_point_indices]
        rest_target_points = controller_runtime_base_target[target_point_indices]
        start_controller_position = interaction_state.get(
            "start_controller_position_world",
            current_controller_position,
        )
        start_anchor_world = interaction_state.get("start_anchor_world", current_anchor)

        controller_delta = float(
            torch.linalg.norm(current_controller_position - start_controller_position).item()
        )
        anchor_delta = float(torch.linalg.norm(current_anchor - start_anchor_world).item())
        target_rest_delta = float(
            torch.linalg.norm(
                current_target_points - rest_target_points,
                dim=1,
            ).mean().item()
        )

        spring_stats = self._controller_spring_subset_stats(
            source,
            controller_attachment_metadata,
        )
        sample_id = controller_world.get("sample_id")

        previous_cache = state_cache.get(source) or {}
        previous_controller_position = previous_cache.get("controller_position")
        previous_target_points = previous_cache.get("target_points")
        controller_frame_delta = 0.0
        if previous_controller_position is not None:
            controller_frame_delta = float(
                torch.linalg.norm(
                    current_controller_position - previous_controller_position
                ).item()
            )
        target_frame_delta = 0.0
        if previous_target_points is not None:
            target_frame_delta = float(
                torch.linalg.norm(
                    current_target_points - previous_target_points,
                    dim=1,
                ).mean().item()
            )

        parity_failure = (
            controller_frame_delta > self.LIVE_CONTROLLER_ACTIVE_MOTION_EPS
            and target_frame_delta <= self.LIVE_CONTROLLER_ACTIVE_TARGET_EPS
        )
        state = (
            interaction_state.get("anchor_name"),
            round(controller_delta, 4),
            round(anchor_delta, 4),
            round(target_frame_delta, 5),
            round(target_rest_delta, 4),
            int(spring_stats["spring_active"]),
            round(spring_stats["spring_max"], 4),
        )
        last_state = previous_cache.get("state")
        last_frame = previous_cache.get("frame_index")
        should_log = (
            last_state != state
            or parity_failure
            or last_frame is None
            or (frame_index - last_frame) >= self.LIVE_CONTROLLER_ACTIVE_DEBUG_LOG_INTERVAL
        )
        if should_log:
            log_prefix = (
                "[live_openxr_controller] "
                f"{runtime_label} motion_parity "
            )
            if parity_failure:
                log_prefix += "parity_failure=1 "
            print(
                log_prefix
                + f"frame={frame_index} "
                + ("" if sample_id is None else f"sample={int(sample_id)} ")
                + f"source={source} "
                f"anchor={interaction_state.get('anchor_name')} "
                f"controller_delta={controller_delta:.4f} "
                f"controller_frame_delta={controller_frame_delta:.4f} "
                f"anchor_delta={anchor_delta:.4f} "
                f"target_frame_delta={target_frame_delta:.5f} "
                f"target_rest_delta={target_rest_delta:.4f} "
                f"spring_active={int(spring_stats['spring_active'])} "
                f"spring_mean={spring_stats['spring_mean']:.4f} "
                f"spring_max={spring_stats['spring_max']:.4f}",
                flush=True,
            )

        state_cache[source] = {
            "state": state,
            "frame_index": frame_index,
            "controller_position": current_controller_position.detach().clone(),
            "target_points": current_target_points.detach().clone(),
        }

    @torch.no_grad()
    def _render_gaussian_rgba(
        self,
        view,
        gaussians,
        render_pipe,
        background_black,
        background_white,
        use_gsplat=False,
    ):
        black_results = render_gaussian(
            view,
            gaussians,
            render_pipe,
            background_black,
            use_gsplat=use_gsplat,
        )
        rendering_black = black_results["render"].detach().clamp(0.0, 1.0)
        black_depth = black_results.get("depth")
        if torch.is_tensor(black_depth):
            black_depth = black_depth.detach()

        if rendering_black.shape[0] == 4:
            return rendering_black, black_depth

        white_results = render_gaussian(
            view,
            gaussians,
            render_pipe,
            background_white,
            use_gsplat=use_gsplat,
        )
        rendering_white = white_results["render"].detach().clamp(0.0, 1.0)

        rgb_black = rendering_black[:3]
        rgb_white = rendering_white[:3]
        alpha = (1.0 - (rgb_white - rgb_black).mean(dim=0, keepdim=True)).clamp(0.0, 1.0)
        safe_alpha = alpha > (1.0 / 255.0)
        rgb = torch.where(
            safe_alpha,
            rgb_black / alpha.clamp_min(1e-6),
            torch.zeros_like(rgb_black),
        ).clamp(0.0, 1.0)
        rgba = torch.cat([rgb, alpha], dim=0)
        return rgba, black_depth

    @torch.no_grad()
    def _render_immersive_eye_frame(
        self,
        eye_pose_world,
        intrinsic,
        eye_height,
        eye_width,
        scene_color,
        scene_depth,
        gaussians,
        render_pipe,
        background_black,
        background_white,
        render_profile_frame=None,
        eye_label=None,
        compose_cache=None,
        compose_mode="depth_aware",
        compose_roi_padding=None,
        collect_compose_debug=False,
        collect_debug_maps=False,
        eye_render_state=None,
        output_dtype=torch.uint8,
    ):
        view_setup_start = (
            time.perf_counter()
            if render_profile_frame is not None and eye_render_state is None
            else None
        )
        if eye_render_state is None:
            eye_render_state = self._prepare_immersive_eye_render_state(
                eye_pose_world,
                intrinsic,
                eye_height,
                eye_width,
                eye_label=eye_label,
            )
        eye_view = eye_render_state["view"]
        if view_setup_start is not None:
            self._render_profile_add_wall_time(
                render_profile_frame,
                "render_eye_intrinsics_setup_wall",
                time.perf_counter() - view_setup_start,
            )
        gaussian_span = self._render_profile_begin_cuda_span(
            render_profile_frame,
            f"gaussian_render_{eye_label}_cuda",
        )
        gaussian_rgba, gaussian_depth = self._render_gaussian_rgba(
            eye_view,
            gaussians,
            render_pipe,
            background_black,
            background_white,
            use_gsplat=True,
        )
        self._render_profile_end_cuda_span(render_profile_frame, gaussian_span)
        compose_span = self._render_profile_begin_cuda_span(
            render_profile_frame,
            f"compose_{eye_label}_cuda",
        )
        compose_metrics = None
        compose_debug_maps = None
        if collect_compose_debug or collect_debug_maps:
            composed, compose_metrics, compose_debug_maps = (
                self._compose_immersive_eye_frame(
                    scene_color,
                    scene_depth,
                    gaussian_rgba,
                    gaussian_depth,
                    target_height=eye_height,
                    target_width=eye_width,
                    compose_cache=compose_cache,
                    compose_mode=compose_mode,
                    compose_roi_padding=compose_roi_padding,
                    collect_debug=True,
                    collect_debug_maps=collect_debug_maps,
                    output_dtype=output_dtype,
                )
            )
        else:
            composed = self._compose_immersive_eye_frame(
                scene_color,
                scene_depth,
                gaussian_rgba,
                gaussian_depth,
                target_height=eye_height,
                target_width=eye_width,
                compose_cache=compose_cache,
                compose_mode=compose_mode,
                compose_roi_padding=compose_roi_padding,
                output_dtype=output_dtype,
            )
        self._render_profile_end_cuda_span(render_profile_frame, compose_span)
        return (
            composed,
            gaussian_rgba,
            gaussian_depth,
            compose_metrics,
            compose_debug_maps,
        )

    def _build_live_controller_world_overlay(
        self,
        source,
        controller_world,
        object_points,
        object_bounds_min,
        object_bounds_max,
    ):
        if controller_world is None:
            return None

        origin_world, direction_world = self._controller_world_ray_pose(controller_world)
        if origin_world is None or direction_world is None:
            return None
        hit_world = self._ray_object_intersection(
            origin_world,
            direction_world,
            object_points,
            object_bounds_min,
            object_bounds_max,
        )
        ray_end_world = (
            hit_world
            if hit_world is not None
            else origin_world + direction_world * self.LIVE_CONTROLLER_RAY_LENGTH
        )
        return {
            "source": source,
            "origin_world": origin_world,
            "direction_world": direction_world,
            "hit_world": hit_world,
            "ray_end_world": ray_end_world,
            "color": (
                self.LIVE_CONTROLLER_LEFT_COLOR
                if source == "left"
                else self.LIVE_CONTROLLER_RIGHT_COLOR
            ),
            "select_available": controller_world["select_available"],
            "select_pressed": controller_world.get(
                "select_hold_active",
                controller_world["select_pressed"],
            ),
            "select_value": controller_world["select_value"],
            "select_source": controller_world["select_source"],
            "anchor_cycle_available": controller_world["anchor_cycle_available"],
            "anchor_cycle_pressed": controller_world["anchor_cycle_pressed"],
            "anchor_cycle_source": controller_world["anchor_cycle_source"],
            "snap_assist_available": controller_world["snap_assist_available"],
            "snap_assist_pressed": controller_world["snap_assist_pressed"],
            "snap_assist_source": controller_world["snap_assist_source"],
        }

    def _project_live_controller_world_overlay(
        self,
        overlay_world,
        intrinsic,
        w2c,
        height,
        width,
    ):
        if overlay_world is None:
            return None

        origin_pixel = self._project_world_point_to_pixel(
            overlay_world.get("origin_world"),
            intrinsic,
            w2c,
            height,
            width,
        )
        end_pixel = self._project_world_point_to_pixel(
            overlay_world.get("ray_end_world"),
            intrinsic,
            w2c,
            height,
            width,
        )
        if origin_pixel is None or end_pixel is None:
            return None

        projected = {
            "source": overlay_world["source"],
            "origin_pixel": origin_pixel,
            "end_pixel": end_pixel,
            "hit_pixel": self._project_world_point_to_pixel(
                overlay_world.get("hit_world"),
                intrinsic,
                w2c,
                height,
                width,
            ),
            "attach_candidate_pixel": self._project_world_point_to_pixel(
                overlay_world.get("attach_candidate_world"),
                intrinsic,
                w2c,
                height,
                width,
            ),
            "attach_active_pixel": self._project_world_point_to_pixel(
                overlay_world.get("attach_active_world"),
                intrinsic,
                w2c,
                height,
                width,
            ),
            "attach_candidate": bool(overlay_world.get("attach_candidate", False)),
            "attachment_active": bool(overlay_world.get("attachment_active", False)),
            "color": overlay_world["color"],
            "select_available": overlay_world["select_available"],
            "select_pressed": overlay_world["select_pressed"],
            "select_value": overlay_world["select_value"],
            "select_source": overlay_world["select_source"],
            "anchor_cycle_available": overlay_world["anchor_cycle_available"],
            "anchor_cycle_pressed": overlay_world["anchor_cycle_pressed"],
            "anchor_cycle_source": overlay_world["anchor_cycle_source"],
            "snap_assist_available": overlay_world["snap_assist_available"],
            "snap_assist_pressed": overlay_world["snap_assist_pressed"],
            "snap_assist_source": overlay_world["snap_assist_source"],
            "anchor_preview_entries": [],
        }
        for preview_entry in overlay_world.get("anchor_preview_entries_world", []):
            preview_pixel = self._project_world_point_to_pixel(
                preview_entry.get("world"),
                intrinsic,
                w2c,
                height,
                width,
            )
            if preview_pixel is None:
                continue
            projected["anchor_preview_entries"].append(
                {
                    "pixel": preview_pixel,
                    "name": preview_entry["name"],
                    "selected": preview_entry["selected"],
                    "active": preview_entry["active"],
                    "occupied": bool(preview_entry.get("occupied", False)),
                }
            )
        return projected

    def _project_live_controller_world_overlays_batched(
        self,
        overlay_world_entries,
        eye_render_states,
        height,
        width,
    ):
        eye_items = list(eye_render_states.items())
        if not eye_items:
            return {}
        projected_by_eye = {eye_label: [] for eye_label, _ in eye_items}
        if not overlay_world_entries:
            return projected_by_eye

        world_points = []
        point_refs = []
        for overlay_idx, overlay_world in enumerate(overlay_world_entries):
            for field_name in (
                "origin_world",
                "ray_end_world",
                "hit_world",
                "attach_candidate_world",
                "attach_active_world",
            ):
                world_point = overlay_world.get(field_name)
                if world_point is None:
                    continue
                world_points.append(
                    torch.as_tensor(
                        world_point,
                        dtype=torch.float32,
                        device=cfg.device,
                    ).reshape(1, 3)
                )
                point_refs.append((overlay_idx, field_name, None))
            for preview_idx, preview_entry in enumerate(
                overlay_world.get("anchor_preview_entries_world", [])
            ):
                world_point = preview_entry.get("world")
                if world_point is None:
                    continue
                world_points.append(
                    torch.as_tensor(
                        world_point,
                        dtype=torch.float32,
                        device=cfg.device,
                    ).reshape(1, 3)
                )
                point_refs.append(
                    (overlay_idx, "anchor_preview_entries_world", preview_idx)
                )

        if not world_points:
            return projected_by_eye

        intrinsic_by_eye_t = torch.stack(
            [state["intrinsic_t"] for _, state in eye_items],
            dim=0,
        )
        w2c_by_eye_t = torch.stack(
            [state["w2c_cv_t"] for _, state in eye_items],
            dim=0,
        )
        pixels_by_eye, depth_valid_by_eye = self._project_points_to_pixels_multi_eye(
            torch.cat(world_points, dim=0),
            intrinsic_by_eye_t,
            w2c_by_eye_t,
        )
        onscreen_by_eye = (
            depth_valid_by_eye
            & (pixels_by_eye[..., 0] >= 0.0)
            & (pixels_by_eye[..., 0] < float(width))
            & (pixels_by_eye[..., 1] >= 0.0)
            & (pixels_by_eye[..., 1] < float(height))
        )

        projected_field_by_eye = [
            [dict() for _ in overlay_world_entries] for _ in eye_items
        ]
        projected_preview_by_eye = [
            [dict() for _ in overlay_world_entries] for _ in eye_items
        ]
        for point_idx, (overlay_idx, field_name, preview_idx) in enumerate(point_refs):
            for eye_idx in range(len(eye_items)):
                pixel = (
                    pixels_by_eye[eye_idx, point_idx]
                    if bool(onscreen_by_eye[eye_idx, point_idx].item())
                    else None
                )
                if preview_idx is None:
                    projected_field_by_eye[eye_idx][overlay_idx][field_name] = pixel
                else:
                    projected_preview_by_eye[eye_idx][overlay_idx][preview_idx] = pixel

        for eye_idx, (eye_label, _) in enumerate(eye_items):
            eye_entries = []
            for overlay_idx, overlay_world in enumerate(overlay_world_entries):
                projected_fields = projected_field_by_eye[eye_idx][overlay_idx]
                origin_pixel = projected_fields.get("origin_world")
                end_pixel = projected_fields.get("ray_end_world")
                if origin_pixel is None or end_pixel is None:
                    continue
                projected = {
                    "source": overlay_world["source"],
                    "origin_pixel": origin_pixel,
                    "end_pixel": end_pixel,
                    "hit_pixel": projected_fields.get("hit_world"),
                    "attach_candidate_pixel": projected_fields.get(
                        "attach_candidate_world"
                    ),
                    "attach_active_pixel": projected_fields.get("attach_active_world"),
                    "attach_candidate": bool(overlay_world.get("attach_candidate", False)),
                    "attachment_active": bool(
                        overlay_world.get("attachment_active", False)
                    ),
                    "color": overlay_world["color"],
                    "select_available": overlay_world["select_available"],
                    "select_pressed": overlay_world["select_pressed"],
                    "select_value": overlay_world["select_value"],
                    "select_source": overlay_world["select_source"],
                    "anchor_cycle_available": overlay_world["anchor_cycle_available"],
                    "anchor_cycle_pressed": overlay_world["anchor_cycle_pressed"],
                    "anchor_cycle_source": overlay_world["anchor_cycle_source"],
                    "snap_assist_available": overlay_world["snap_assist_available"],
                    "snap_assist_pressed": overlay_world["snap_assist_pressed"],
                    "snap_assist_source": overlay_world["snap_assist_source"],
                    "anchor_preview_entries": [],
                }
                preview_pixels = projected_preview_by_eye[eye_idx][overlay_idx]
                for preview_idx, preview_entry in enumerate(
                    overlay_world.get("anchor_preview_entries_world", [])
                ):
                    preview_pixel = preview_pixels.get(preview_idx)
                    if preview_pixel is None:
                        continue
                    projected["anchor_preview_entries"].append(
                        {
                            "pixel": preview_pixel,
                            "name": preview_entry["name"],
                            "selected": preview_entry["selected"],
                            "active": preview_entry["active"],
                            "occupied": bool(preview_entry.get("occupied", False)),
                        }
                    )
                eye_entries.append(projected)
            projected_by_eye[eye_label] = eye_entries
        return projected_by_eye

    def _compute_immersive_compose_roi_bounds(
        self,
        alpha_mask,
        depth_map=None,
        height=None,
        width=None,
        padding=None,
    ):
        if padding is None:
            padding = int(self.IMMERSIVE_GAUSSIAN_COMPOSE_ROI_PADDING)
        if torch.is_tensor(alpha_mask):
            roi_mask = alpha_mask > float(self.IMMERSIVE_COMPOSE_ALPHA_EPS)
        else:
            roi_mask = torch.as_tensor(
                np.asarray(alpha_mask) > float(self.IMMERSIVE_COMPOSE_ALPHA_EPS),
                device=cfg.device,
                dtype=torch.bool,
            )
        if depth_map is not None:
            if torch.is_tensor(depth_map):
                roi_mask = roi_mask | (depth_map > float(self.IMMERSIVE_STARTUP_DEPTH_EPS))
            else:
                roi_mask = roi_mask | torch.as_tensor(
                    np.asarray(depth_map) > float(self.IMMERSIVE_STARTUP_DEPTH_EPS),
                    device=cfg.device,
                    dtype=torch.bool,
                )
        if not bool(roi_mask.any().item()):
            return None
        coords = torch.nonzero(roi_mask, as_tuple=False)
        if int(coords.shape[0]) <= 0:
            return None
        if height is None:
            height = int(roi_mask.shape[0])
        if width is None:
            width = int(roi_mask.shape[1])
        y0 = max(0, int(coords[:, 0].min().item()) - int(padding))
        y1 = min(int(height), int(coords[:, 0].max().item()) + int(padding) + 1)
        x0 = max(0, int(coords[:, 1].min().item()) - int(padding))
        x1 = min(int(width), int(coords[:, 1].max().item()) + int(padding) + 1)
        if x1 <= x0 or y1 <= y0:
            return None
        return (x0, y0, x1, y1)

    @torch.no_grad()
    def _downsample_immersive_supersampled_overlay_patch(
        self,
        overlay_color_rgba,
        overlay_depth,
        target_height,
        target_width,
    ):
        target_height = int(target_height)
        target_width = int(target_width)
        if not torch.is_tensor(overlay_color_rgba) or not torch.is_tensor(overlay_depth):
            raise TypeError(
                "Supersampled immersive overlay downsample expects tensor inputs."
            )
        overlay_color_t, overlay_depth_t = self._prepare_immersive_scene_frame_for_compose(
            overlay_color_rgba,
            overlay_depth,
            int(overlay_depth.shape[0]),
            int(overlay_depth.shape[1]),
            compose_cache=None,
        )
        if overlay_color_t.shape[:2] == (target_height, target_width):
            overlay_coverage_mask = overlay_color_t[..., 3] >= 1.0
            return (
                overlay_color_t.contiguous(),
                overlay_depth_t.contiguous(),
                overlay_coverage_mask.contiguous(),
            )
        color_down = (
            F.interpolate(
                overlay_color_t.permute(2, 0, 1).unsqueeze(0),
                size=(target_height, target_width),
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(0)
            .permute(1, 2, 0)
            .contiguous()
        )
        alpha_mask = (
            (overlay_color_t[..., 3] >= 1.0)
            .to(dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        coverage_mask = (
            F.adaptive_max_pool2d(
                alpha_mask,
                output_size=(target_height, target_width),
            )
            .squeeze(0)
            .squeeze(0)
            > 0.0
        )
        overlay_depth_input = overlay_depth_t.unsqueeze(0).unsqueeze(0)
        depth_valid = overlay_depth_input > float(self.IMMERSIVE_STARTUP_DEPTH_EPS)
        neg_depth_for_min = torch.where(
            depth_valid,
            -overlay_depth_input,
            torch.full_like(overlay_depth_input, -1.0e6),
        )
        pooled_neg_depth = F.adaptive_max_pool2d(
            neg_depth_for_min,
            output_size=(target_height, target_width),
        )
        depth_down = torch.where(
            pooled_neg_depth > -1.0e5,
            -pooled_neg_depth,
            torch.zeros_like(pooled_neg_depth),
        ).squeeze(0).squeeze(0).contiguous()
        return color_down, depth_down, coverage_mask.contiguous()

    def _compose_immersive_scene_layers(
        self,
        background_color_rgba,
        background_depth,
        overlay_color_rgba,
        overlay_depth,
        target_height=None,
        target_width=None,
        background_cache=None,
        overlay_roi_bounds=None,
        overlay_coverage_mask=None,
        force_overlay_visible=False,
    ):
        if target_height is None or target_width is None:
            target_height = int(background_color_rgba.shape[0])
            target_width = int(background_color_rgba.shape[1])
        background_color_t, background_depth_t = (
            self._prepare_immersive_scene_frame_for_compose(
                background_color_rgba,
                background_depth,
                target_height,
                target_width,
                compose_cache=background_cache,
            )
        )
        if overlay_roi_bounds is not None:
            x0, y0, x1, y1 = [int(v) for v in overlay_roi_bounds]
            if x1 <= x0 or y1 <= y0:
                raise ValueError(f"Invalid overlay_roi_bounds: {overlay_roi_bounds}")
            roi_height = y1 - y0
            roi_width = x1 - x0
            overlay_color_t, overlay_depth_t = self._prepare_immersive_scene_frame_for_compose(
                overlay_color_rgba,
                overlay_depth,
                roi_height,
                roi_width,
                compose_cache=None,
            )
            background_color_roi = background_color_t[y0:y1, x0:x1]
            background_depth_roi = background_depth_t[y0:y1, x0:x1]
        else:
            overlay_color_t, overlay_depth_t = self._prepare_immersive_scene_frame_for_compose(
                overlay_color_rgba,
                overlay_depth,
                target_height,
                target_width,
                compose_cache=None,
            )
            background_color_roi = background_color_t
            background_depth_roi = background_depth_t
        overlay_coverage_t = None
        if overlay_coverage_mask is not None:
            if torch.is_tensor(overlay_coverage_mask):
                overlay_coverage_t = overlay_coverage_mask.to(
                    device=cfg.device,
                    dtype=torch.bool,
                )
            else:
                overlay_coverage_t = torch.as_tensor(
                    np.asarray(overlay_coverage_mask),
                    device=cfg.device,
                    dtype=torch.bool,
                )
            if overlay_coverage_t.shape[:2] != overlay_depth_t.shape[:2]:
                overlay_coverage_t = (
                    F.interpolate(
                        overlay_coverage_t.to(dtype=torch.float32)
                        .unsqueeze(0)
                        .unsqueeze(0),
                        size=overlay_depth_t.shape[:2],
                        mode="nearest",
                    )
                    .squeeze(0)
                    .squeeze(0)
                    > 0.0
                )
        overlay_alpha = (
            overlay_color_t[..., 3:4].clamp(0.0, 255.0) / 255.0
        )
        overlay_has_depth = overlay_depth_t > float(self.IMMERSIVE_STARTUP_DEPTH_EPS)
        overlay_has_presence = (
            overlay_coverage_t if overlay_coverage_t is not None else overlay_has_depth
        )
        background_has_depth = background_depth_roi > float(self.IMMERSIVE_STARTUP_DEPTH_EPS)
        if force_overlay_visible:
            overlay_visible = overlay_has_presence
        else:
            depth_visible = (~background_has_depth) | (
                overlay_depth_t <= (background_depth_roi + 5e-3)
            )
            if overlay_coverage_t is not None:
                overlay_visible = overlay_has_presence & (
                    (~background_has_depth) | (overlay_has_depth & depth_visible)
                )
            else:
                overlay_visible = overlay_has_depth & depth_visible
        effective_alpha = overlay_alpha * overlay_visible.unsqueeze(-1).to(
            overlay_alpha.dtype
        )
        if background_cache is not None:
            background_cache = self._ensure_immersive_compose_cache(
                background_cache,
                target_height,
                target_width,
            )
            composed_color = background_cache["composed_color"]
            composed_depth = background_cache["composed_depth"]
            composed_color.copy_(background_color_t, non_blocking=True)
            composed_depth.copy_(background_depth_t, non_blocking=True)
        else:
            composed_color = background_color_t.clone()
            composed_depth = background_depth_t.clone()
        if overlay_roi_bounds is not None:
            composed_color_roi = composed_color[y0:y1, x0:x1]
            composed_depth_roi = composed_depth[y0:y1, x0:x1]
        else:
            composed_color_roi = composed_color
            composed_depth_roi = composed_depth
        composed_color_roi[..., :3] = background_color_roi[..., :3] * (1.0 - effective_alpha) + (
            overlay_color_t[..., :3] * effective_alpha
        )
        composed_color_roi[..., 3] = 255.0
        composed_depth_roi.copy_(background_depth_roi)
        composed_depth_roi[overlay_has_depth & overlay_visible] = overlay_depth_t[
            overlay_has_depth & overlay_visible
        ]
        return composed_color.contiguous(), composed_depth.contiguous()

    @torch.no_grad()
    def _compose_immersive_eye_frame(
        self,
        scene_color_rgba,
        scene_depth,
        gaussian_rgba,
        gaussian_depth,
        target_height=None,
        target_width=None,
        compose_cache=None,
        compose_mode="depth_aware",
        collect_debug=False,
        collect_debug_maps=False,
        compose_roi_padding=None,
        output_dtype=torch.uint8,
        return_depth=False,
    ):
        if target_height is None or target_width is None:
            target_height = int(gaussian_rgba.shape[1])
            target_width = int(gaussian_rgba.shape[2])

        scene_color, scene_depth_t = self._prepare_immersive_scene_frame_for_compose(
            scene_color_rgba,
            scene_depth,
            target_height,
            target_width,
            compose_cache=compose_cache,
        )

        object_rgba = gaussian_rgba.detach().permute(1, 2, 0).contiguous().clamp(0.0, 1.0)
        object_alpha = object_rgba[..., 3:4]
        raw_visible = object_alpha[..., 0] > float(self.IMMERSIVE_COMPOSE_ALPHA_EPS)
        gaussian_depth_t = (
            self._normalize_gaussian_depth(gaussian_depth)
            if gaussian_depth is not None
            else None
        )
        compose_roi_bounds = None
        compose_roi_ratio = 1.0
        if compose_roi_padding is not None:
            compose_roi_bounds = self._compute_immersive_compose_roi_bounds(
                object_alpha[..., 0],
                gaussian_depth_t,
                height=target_height,
                width=target_width,
                padding=compose_roi_padding,
            )
            if compose_roi_bounds is None:
                compose_roi_ratio = 0.0
            else:
                x0, y0, x1, y1 = compose_roi_bounds
                compose_roi_ratio = float(
                    ((x1 - x0) * (y1 - y0))
                    / max(float(target_height * target_width), 1.0)
                )

        if compose_roi_bounds is not None:
            x0, y0, x1, y1 = compose_roi_bounds
            scene_color_roi = scene_color[y0:y1, x0:x1]
            scene_depth_roi = scene_depth_t[y0:y1, x0:x1]
            object_rgba_roi = object_rgba[y0:y1, x0:x1]
            object_alpha_roi = object_alpha[y0:y1, x0:x1]
            raw_visible_roi = raw_visible[y0:y1, x0:x1]
            gaussian_depth_roi = (
                gaussian_depth_t[y0:y1, x0:x1] if gaussian_depth_t is not None else None
            )
            if compose_mode == "alpha_overlay":
                effective_alpha_roi = object_alpha_roi
                visible_mask_roi = raw_visible_roi
            else:
                if compose_mode != "depth_aware":
                    raise ValueError(f"Unsupported immersive compose_mode: {compose_mode}")
                if gaussian_depth_roi is None:
                    effective_alpha_roi = object_alpha_roi
                    visible_mask_roi = raw_visible_roi
                else:
                    scene_has_geometry_roi = scene_depth_roi > 0.0
                    object_has_depth_roi = gaussian_depth_roi > 0.0
                    visible_mask_roi = object_has_depth_roi & (
                        (~scene_has_geometry_roi)
                        | (gaussian_depth_roi <= (scene_depth_roi + 5e-3))
                    )
                    effective_alpha_roi = object_alpha_roi * visible_mask_roi.unsqueeze(-1).to(
                        object_alpha_roi.dtype
                    )
            composed_rgb = scene_color[..., :3].clone()
            composed_depth = scene_depth_t.clone()
            visible_alpha = torch.zeros(
                (target_height, target_width),
                dtype=object_alpha.dtype,
                device=cfg.device,
            )
            composed_rgb[y0:y1, x0:x1] = scene_color_roi[..., :3] * (
                1.0 - effective_alpha_roi
            ) + (object_rgba_roi[..., :3] * 255.0) * effective_alpha_roi
            visible_alpha[y0:y1, x0:x1] = effective_alpha_roi[..., 0]
            if gaussian_depth_roi is not None:
                composed_depth_roi = composed_depth[y0:y1, x0:x1]
                composed_depth_roi[visible_mask_roi] = gaussian_depth_roi[visible_mask_roi]
        elif compose_roi_padding is not None:
            composed_rgb = scene_color[..., :3]
            composed_depth = scene_depth_t
            visible_alpha = torch.zeros(
                (target_height, target_width),
                dtype=object_alpha.dtype,
                device=cfg.device,
            )
        else:
            if compose_mode == "alpha_overlay":
                effective_alpha = object_alpha
                visible_mask = raw_visible
            else:
                if compose_mode != "depth_aware":
                    raise ValueError(f"Unsupported immersive compose_mode: {compose_mode}")
                if gaussian_depth_t is None:
                    effective_alpha = object_alpha
                    visible_mask = raw_visible
                else:
                    scene_has_geometry = scene_depth_t > 0.0
                    object_has_depth = gaussian_depth_t > 0.0
                    visible_mask = object_has_depth & (
                        (~scene_has_geometry) | (gaussian_depth_t <= (scene_depth_t + 5e-3))
                    )
                    effective_alpha = object_alpha * visible_mask.unsqueeze(-1).to(
                        object_alpha.dtype
                    )
            composed_rgb = scene_color[..., :3] * (1.0 - effective_alpha) + (
                object_rgba[..., :3] * 255.0
            ) * effective_alpha
            composed_depth = scene_depth_t.clone()
            if gaussian_depth_t is not None:
                composed_depth[visible_mask] = gaussian_depth_t[visible_mask]
            visible_alpha = effective_alpha[..., 0]

        if output_dtype is torch.float32:
            composed = torch.empty(
                scene_color.shape,
                dtype=torch.float32,
                device=cfg.device,
            )
            composed[..., :3] = composed_rgb.clamp(0.0, 255.0)
            composed[..., 3] = 255.0
        else:
            if output_dtype is not torch.uint8:
                raise ValueError(
                    f"Unsupported immersive eye frame dtype: {output_dtype}"
                )
            composed = torch.empty(
                scene_color.shape,
                dtype=torch.uint8,
                device=cfg.device,
            )
            composed[..., :3] = composed_rgb.clamp(0.0, 255.0).to(torch.uint8)
            composed[..., 3] = 255
        if not (collect_debug or collect_debug_maps or return_depth):
            return composed
        if return_depth and not (collect_debug or collect_debug_maps):
            return composed, composed_depth.contiguous()

        finite_mask = torch.isfinite(scene_depth_t)
        positive_mask = finite_mask & (scene_depth_t > self.IMMERSIVE_STARTUP_DEPTH_EPS)
        scene_depth_finite_ratio = float(finite_mask.to(dtype=torch.float32).mean().item())
        scene_depth_positive_ratio = float(
            positive_mask.to(dtype=torch.float32).mean().item()
        )
        if bool(positive_mask.any().item()):
            scene_depth_valid_min = float(scene_depth_t[positive_mask].min().item())
            scene_depth_valid_max = float(scene_depth_t[positive_mask].max().item())
        else:
            scene_depth_valid_min = 0.0
            scene_depth_valid_max = 0.0

        visible_gaussian_coverage_ratio = float(
            (visible_alpha > float(self.IMMERSIVE_COMPOSE_ALPHA_EPS))
            .to(dtype=torch.float32)
            .mean()
            .item()
        )
        raw_gaussian_coverage_ratio = float(
            raw_visible.to(dtype=torch.float32).mean().item()
        )
        visible_retention_ratio = (
            visible_gaussian_coverage_ratio / max(raw_gaussian_coverage_ratio, 1e-6)
            if raw_gaussian_coverage_ratio > 0.0
            else 1.0
        )
        scene_depth_invalid = bool(
            scene_depth_finite_ratio < float(self.IMMERSIVE_SCENE_DEPTH_MIN_FINITE_RATIO)
            or scene_depth_positive_ratio < float(self.IMMERSIVE_SCENE_DEPTH_MIN_POSITIVE_RATIO)
            or (
                positive_mask.any().item()
                and (
                    (not np.isfinite(scene_depth_valid_min))
                    or (not np.isfinite(scene_depth_valid_max))
                    or scene_depth_valid_max <= scene_depth_valid_min
                )
            )
        )
        scene_depth_suppressed = bool(
            compose_mode == "depth_aware"
            and raw_gaussian_coverage_ratio
            >= float(self.IMMERSIVE_COMPOSE_RAW_MIN_COVERAGE_RATIO)
            and visible_gaussian_coverage_ratio
            <= float(self.IMMERSIVE_COMPOSE_VISIBLE_MIN_COVERAGE_RATIO)
            and visible_retention_ratio
            < float(self.IMMERSIVE_COMPOSE_MIN_RETENTION_RATIO)
        )
        composed_rgb_f = composed[..., :3].to(dtype=torch.float32)
        composed_luma = (
            0.2126 * composed_rgb_f[..., 0]
            + 0.7152 * composed_rgb_f[..., 1]
            + 0.0722 * composed_rgb_f[..., 2]
        )
        compose_metrics = {
            "compose_mode": compose_mode,
            "raw_gaussian_coverage_ratio": raw_gaussian_coverage_ratio,
            "visible_gaussian_coverage_ratio": visible_gaussian_coverage_ratio,
            "visible_retention_ratio": float(visible_retention_ratio),
            "scene_depth_finite_ratio": scene_depth_finite_ratio,
            "scene_depth_positive_ratio": scene_depth_positive_ratio,
            "scene_depth_valid_min": float(scene_depth_valid_min),
            "scene_depth_valid_max": float(scene_depth_valid_max),
            "scene_depth_invalid": scene_depth_invalid,
            "scene_depth_suppressed": scene_depth_suppressed,
            "compose_roi_ratio": float(compose_roi_ratio),
            "composed_luma_mean": float(composed_luma.mean().item()),
            "composed_luma_variance": float(
                composed_luma.var(unbiased=False).item()
            ),
        }
        compose_debug_maps = None
        if collect_debug_maps:
            compose_debug_maps = {
                "scene_color": scene_color.detach(),
                "scene_depth": scene_depth_t.detach(),
                "raw_alpha": object_alpha[..., 0].detach(),
                "visible_alpha": visible_alpha.detach(),
            }
        if return_depth:
            return composed, composed_depth.contiguous(), compose_metrics, compose_debug_maps
        return composed, compose_metrics, compose_debug_maps

    #init_start with morton reordering for both mass node and spring (current last working version)
    def _init_start(
        self,
        object_points,
        controller_points,
        object_radius=0.02,
        object_max_neighbours=30,
        controller_radius=0.04,
        controller_max_neighbours=50,
        mask=None,
    ):
        object_points = object_points.cpu().numpy()
        if controller_points is not None:
            controller_points = controller_points.cpu().numpy()
        if mask is None:
            object_pcd = o3d.geometry.PointCloud()
            object_pcd.points = o3d.utility.Vector3dVector(object_points)
            pcd_tree = o3d.geometry.KDTreeFlann(object_pcd)

            # Connect the springs of the objects first
            points = np.asarray(object_pcd.points)
            spring_flags = np.zeros((len(points), len(points)))
            springs = []
            rest_lengths = []
            for i in range(len(points)):
                [k, idx, _] = pcd_tree.search_hybrid_vector_3d(
                    points[i], object_radius, object_max_neighbours
                )
                idx = idx[1:]
                for j in idx:
                    rest_length = np.linalg.norm(points[i] - points[j])
                    if (
                        spring_flags[i, j] == 0
                        and spring_flags[j, i] == 0
                        and rest_length > 1e-4
                    ):
                        spring_flags[i, j] = 1
                        spring_flags[j, i] = 1
                        springs.append([i, j])
                        rest_lengths.append(np.linalg.norm(points[i] - points[j]))

            num_object_springs = len(springs)
            
            # ============================================================
            # 🆕 NEW: Save num_object_points BEFORE adding controllers
            # ============================================================
            num_object_points = len(points)
            print(f" num object springs {num_object_springs}")
            if controller_points is not None:
                # Connect the springs between the controller points and the object points
                # ============================================================
                # ⚠️  REMOVED: num_object_points = len(points)
                # (moved above, before controller springs are added)
                # ============================================================

                points = np.concatenate([points, controller_points], axis=0)
                for i in range(len(controller_points)):
                    [k, idx, _] = pcd_tree.search_hybrid_vector_3d(
                        controller_points[i],
                        controller_radius,
                        controller_max_neighbours,
                    )
                    for j in idx:
                        springs.append([num_object_points + i, j])
                        rest_lengths.append(
                            np.linalg.norm(controller_points[i] - points[j])
                        )

            springs = np.array(springs)
            print(f" total springs {len(springs)}")

            rest_lengths = np.array(rest_lengths)
            masses = np.ones(len(points))
            
            # ============================================================
            # 🆕 NEW: Morton reordering section (entire block is new)
            # ============================================================
            # Convert to torch tensors
            points_torch = torch.tensor(points, dtype=torch.float32, device=cfg.device)
            springs_torch = torch.tensor(springs, dtype=torch.int32, device=cfg.device)
            rest_lengths_torch = torch.tensor(rest_lengths, dtype=torch.float32, device=cfg.device)
            masses_torch = torch.tensor(masses, dtype=torch.float32, device=cfg.device)
            
            # Apply Morton reordering
            logger.info("Applying Morton ordering to improve cache performance...")
            points_torch, springs_torch, rest_lengths_torch, spring_perm = self._apply_morton_reordering(
                points_torch, springs_torch, rest_lengths_torch, num_object_points
            )
            self.spring_permutation = spring_perm

            return (
                points_torch,
                springs_torch,
                rest_lengths_torch,
                masses_torch,
                num_object_springs,
            )
            # ============================================================            
        else:
            mask = mask.cpu().numpy()
            # Get the unique value in masks
            unique_values = np.unique(mask)
            vertices = []
            springs = []
            rest_lengths = []
            index = 0
            # Loop different objects to connect the springs separately
            for value in unique_values:
                temp_points = object_points[mask == value]
                temp_pcd = o3d.geometry.PointCloud()
                temp_pcd.points = o3d.utility.Vector3dVector(temp_points)
                temp_tree = o3d.geometry.KDTreeFlann(temp_pcd)
                temp_spring_flags = np.zeros((len(temp_points), len(temp_points)))
                temp_springs = []
                temp_rest_lengths = []
                for i in range(len(temp_points)):
                    [k, idx, _] = temp_tree.search_hybrid_vector_3d(
                        temp_points[i], object_radius, object_max_neighbours
                    )
                    idx = idx[1:]
                    for j in idx:
                        rest_length = np.linalg.norm(temp_points[i] - temp_points[j])
                        if (
                            temp_spring_flags[i, j] == 0
                            and temp_spring_flags[j, i] == 0
                            and rest_length > 1e-4
                        ):
                            temp_spring_flags[i, j] = 1
                            temp_spring_flags[j, i] = 1
                            temp_springs.append([i + index, j + index])
                            temp_rest_lengths.append(rest_length)
                vertices += temp_points.tolist()
                springs += temp_springs
                rest_lengths += temp_rest_lengths
                index += len(temp_points)

            num_object_springs = len(springs)
            
            # ============================================================
            # 🆕 NEW: Save num_object_points (for multi-object case)
            # ============================================================
            num_object_points = len(vertices)

            vertices = np.array(vertices)
            springs = np.array(springs)
            rest_lengths = np.array(rest_lengths)
            masses = np.ones(len(vertices))
            
            # ============================================================
            # 🆕 NEW: Morton reordering section (entire block is new)
            # ============================================================
            # Convert to torch tensors
            vertices_torch = torch.tensor(vertices, dtype=torch.float32, device=cfg.device)
            springs_torch = torch.tensor(springs, dtype=torch.int32, device=cfg.device)
            rest_lengths_torch = torch.tensor(rest_lengths, dtype=torch.float32, device=cfg.device)
            masses_torch = torch.tensor(masses, dtype=torch.float32, device=cfg.device)
            
            # Apply Morton reordering
            logger.info("Applying Morton ordering to improve cache performance (multi-object)...")
            vertices_torch, springs_torch, rest_lengths_torch, spring_perm = self._apply_morton_reordering(
                vertices_torch, springs_torch, rest_lengths_torch, num_object_points
            )
            self.spring_permutation = spring_perm

            # NEW:
            return (
                vertices_torch,
                springs_torch,
                rest_lengths_torch,
                masses_torch,
                num_object_springs,
            )
            # ============================================================

    def _find_closest_point(self, target_points):
        """Find the closest structure point to any of the target points."""
        dist_matrix = torch.sum(
            (target_points.unsqueeze(1) - self.structure_points.unsqueeze(0)) ** 2,
            dim=2,
        )
        min_dist_per_ctrl_pts, min_indices = torch.min(dist_matrix, dim=1)
        min_idx = min_indices[torch.argmin(min_dist_per_ctrl_pts)]
        return self.structure_points[min_idx].unsqueeze(0)

    def stable_lexsort(self,keys):
        """
        keys: list of 1D tensors, all same length.
            Order is MOST-significant -> LEAST-significant.
        Returns: permutation idx such that keys are sorted lexicographically.
        """
        assert len(keys) > 0
        n = keys[0].numel()
        idx = torch.arange(n, device=keys[0].device)

        # stable sort from least-significant to most-significant
        for k in reversed(keys):
            idx = idx[torch.argsort(k[idx], stable=True)]
        return idx

    def _reorder_springs_spatial_blocking(self, springs, rest_lengths, num_object_points, block_size=32):
        """
        Cluster springs so consecutive springs mostly touch vertices in the same index-block
        (after Morton vertex reordering, index-blocks correspond to spatial regions).
        
        Preserves: object-object springs first, then controller-object springs.
        Returns: springs2, rest2, perm
        """
        N = int(num_object_points)
        s = springs.long()
        i = s[:, 0]
        j = s[:, 1]
        
        # Split types (preserve your num_object_springs prefix assumption)
        obj_obj_mask = (i < N) & (j < N)
        ctrl_mask = ~obj_obj_mask

        if ctrl_mask.any():
            has_obj_ep = (i[ctrl_mask] < N) | (j[ctrl_mask] < N)
            assert has_obj_ep.all(), "Found controller-controller springs!"
                
        # ---- object-object springs ----
        obj_ids = torch.nonzero(obj_obj_mask, as_tuple=False).squeeze(1)
        if obj_ids.numel() > 0:
            io = i[obj_ids]
            jo = j[obj_ids]
            a = torch.minimum(io, jo)
            b = torch.maximum(io, jo)
            
            ablk = a // block_size
            bblk = b // block_size
            
            # Sort by (ablk, bblk, a, b) lexicographically
            perm_obj_local = self.stable_lexsort([ablk, bblk, a, b])
            obj_order = obj_ids[perm_obj_local]
        else:
            obj_order = obj_ids
        
        # ---- controller-object springs ----
        ctrl_ids = torch.nonzero(ctrl_mask, as_tuple=False).squeeze(1)
        if ctrl_ids.numel() > 0:
            ic = i[ctrl_ids]
            jc = j[ctrl_ids]
            
            # object endpoint = the one < N
            obj_ep  = torch.where(ic < N, ic, jc)
            ctrl_ep = torch.where(ic >= N, ic, jc)
            ctrl_id = (ctrl_ep - N).clamp_min(0)
            
            obj_blk = obj_ep // block_size
            
            perm_ctrl_local = self.stable_lexsort([obj_blk, obj_ep, ctrl_id])
            ctrl_order = ctrl_ids[perm_ctrl_local]

        else:
            ctrl_order = ctrl_ids
        
        perm = torch.cat([obj_order, ctrl_order], dim=0)
        
        # Sanity checks
        assert perm.shape[0] == springs.shape[0], "Permutation size mismatch"
        assert torch.unique(perm).shape[0] == springs.shape[0], "Permutation has duplicates"
        
        print(f"Reordered springs with spatial blocking (block_size={block_size}):")
        print(f"  Object-object springs: {obj_ids.numel()}")
        print(f"  Controller-object springs: {ctrl_ids.numel()}")
        
        return springs[perm], rest_lengths[perm], perm

    def _apply_morton_reordering(self, vertices, springs, rest_lengths, num_object_points):
        """
        Reorder object vertices using Morton (Z-order) curve AND reorder springs for coalescing.
        
        Returns:
            new_vertices: reordered vertices
            new_springs: springs with remapped indices AND reordered for coalescing
            new_rest_lengths: rest_lengths reordered to match springs
            spring_permutation: permutation applied to springs (for reordering other per-spring arrays)
        """
        device = vertices.device
        
        # Device assertions
        assert springs.device == device
        assert rest_lengths.device == device
        
        obj_end = num_object_points
        has_controllers = (num_object_points < len(vertices))
        
        # === Step 1: Morton reorder vertices ===
        obj_verts = vertices[:obj_end].detach().cpu().numpy()
        
        # Normalize to [0,1] for Morton encoding
        mins = obj_verts.min(axis=0)
        maxs = obj_verts.max(axis=0)
        range_vals = maxs - mins
        range_vals[range_vals < 1e-8] = 1.0
        normalized = (obj_verts - mins) / range_vals
        
        # Convert to 21-bit integer coordinates
        BITS = 21
        MAX_VAL = (1 << BITS) - 1
        int_coords = (normalized * MAX_VAL).astype(np.uint64)
        
        # Compute Morton codes
        def part1by2(n):
            n = np.uint64(n)
            n = (n | (n << 32)) & np.uint64(0x1f00000000ffff)
            n = (n | (n << 16)) & np.uint64(0x1f0000ff0000ff)
            n = (n | (n << 8))  & np.uint64(0x100f00f00f00f00f)
            n = (n | (n << 4))  & np.uint64(0x10c30c30c30c30c3)
            n = (n | (n << 2))  & np.uint64(0x1249249249249249)
            return n
        
        x = part1by2(int_coords[:, 0])
        y = part1by2(int_coords[:, 1])
        z = part1by2(int_coords[:, 2])
        morton_codes = x | (y << 1) | (z << 2)
        
        # Stable sort
        perm = np.argsort(morton_codes, kind="stable")
        inv_perm = np.empty(obj_end, dtype=np.int64)
        inv_perm[perm] = np.arange(obj_end)

        sorted_morton_codes = morton_codes[perm]
        morton_codes_torch = torch.from_numpy(sorted_morton_codes).to(device).long()  # ✅ Convert to int64

        
        # Convert to torch
        perm_torch = torch.from_numpy(perm).to(device).long()
        inv_perm_torch = torch.from_numpy(inv_perm).to(device).long()
        
        # Reorder vertices
        reordered_obj_verts = torch.index_select(vertices[:obj_end], 0, perm_torch)
        if has_controllers:
            new_vertices = torch.cat([reordered_obj_verts, vertices[obj_end:]], dim=0)
        else:
            new_vertices = reordered_obj_verts
        
        # === Step 2: Remap spring indices ===
        springs_dtype = springs.dtype
        new_springs = springs.clone()
        
        for col in [0, 1]:
            obj_mask = springs[:, col] < obj_end
            obj_indices = springs[obj_mask, col].long()
            remapped_indices = inv_perm_torch[obj_indices].to(springs_dtype)
            new_springs[obj_mask, col] = remapped_indices
        
        logger.info(f"Applied Morton reordering to {obj_end} object vertices")
        
        # === Step 3: Reorder springs for coalescing ===
        new_springs, new_rest_lengths, spring_permutation = self._reorder_springs_spatial_blocking(
            new_springs, rest_lengths, num_object_points, block_size=32
        )
        return new_vertices, new_springs, new_rest_lengths, spring_permutation

    def merge_two_gaussians(self, gaussians1, gaussians2, max_sh_degree=3):
        '''
        Merge two gaussians into one
        '''
        new_gaussians = GaussianModel(max_sh_degree)
        new_gaussians._xyz = torch.cat([gaussians1._xyz, gaussians2._xyz], dim=0)
        new_gaussians._features_dc = torch.cat([gaussians1._features_dc, gaussians2._features_dc], dim=0)
        new_gaussians._features_rest = torch.cat([gaussians1._features_rest, gaussians2._features_rest], dim=0)
        new_gaussians._opacity = torch.cat([gaussians1._opacity, gaussians2._opacity], dim=0)
        new_gaussians._scaling = torch.cat([gaussians1._scaling, gaussians2._scaling], dim=0)
        new_gaussians._rotation = torch.cat([gaussians1._rotation, gaussians2._rotation], dim=0)
        return new_gaussians

    def _run_quest_immersive_balanced(
        self,
        model_path,
        gs_path,
        output_dir,
        window,
        cuda_ctx,
        interactive_window_mode,
        scene_assets_root,
        render_profile=False,
        render_profile_every=30,
        immersive_timewarp="off",
        immersive_static_scene_overlap="off",
    ):
        logger.info(f"Load model from {model_path}")
        checkpoint = torch.load(model_path, map_location=cfg.device)

        trained_spring_Y = checkpoint["spring_Y"][self.spring_permutation]
        trained_collide_elas = checkpoint["collide_elas"]
        trained_collide_fric = checkpoint["collide_fric"]
        trained_collide_object_elas = checkpoint["collide_object_elas"]
        trained_collide_object_fric = checkpoint["collide_object_fric"]

        intrinsic = cfg.intrinsics[0]
        w2c = cfg.w2cs[0]
        intrinsic_torch = torch.tensor(intrinsic, dtype=torch.float32, device=cfg.device)
        w2c_torch = torch.tensor(w2c, dtype=torch.float32, device=cfg.device)
        obj_init_vertices = self.init_vertices[: self.num_all_points]
        ctrl_init_vertices = self.init_vertices[self.num_all_points :]
        init_springs_for_sim = self.init_springs
        init_rest_lengths_for_sim = self.init_rest_lengths
        trained_spring_Y_for_sim = trained_spring_Y
        init_masses_for_sim = self.init_masses[: self.num_all_points].clone()

        recorded_base_target = self.controller_points_group[0][0].clone()
        controller_masks = self._build_controller_part_masks(
            recorded_base_target,
            n_ctrl_parts=2,
            intrinsic=intrinsic,
            w2c=w2c,
        )
        recorded_anchor_centers = [
            recorded_base_target[mask].mean(dim=0) for mask in controller_masks
        ]

        self._object_graph_neighbors()
        original_controller_source_masks = self._build_controller_source_masks(
            recorded_base_target,
            intrinsic=intrinsic,
            w2c=w2c,
        )
        original_controller_source_anchor_centers = [
            recorded_base_target[mask].mean(dim=0)
            for mask in original_controller_source_masks
        ]
        case_name = self._interaction_anchor_case_name()
        gaussians = GaussianModel(sh_degree=3)
        gaussians.load_ply(gs_path)
        raw_gaussian_count = int(gaussians._xyz.shape[0])
        disable_opacity_pruning = case_name == "hq_rope"
        kept_gaussian_count = raw_gaussian_count
        if not disable_opacity_pruning:
            gaussians = remove_gaussians_with_low_opacity(gaussians, 0.1)
            kept_gaussian_count = int(gaussians._xyz.shape[0])
        gaussians.isotropic = True
        if case_name == "hq_rope":
            gaussian_bounds_min = (
                gaussians._xyz.min(dim=0).values.detach().cpu().numpy().tolist()
            )
            gaussian_bounds_max = (
                gaussians._xyz.max(dim=0).values.detach().cpu().numpy().tolist()
            )
            object_bounds_min = (
                obj_init_vertices.min(dim=0).values.detach().cpu().numpy().tolist()
            )
            object_bounds_max = (
                obj_init_vertices.max(dim=0).values.detach().cpu().numpy().tolist()
            )
            scaled_object_span = self._principal_axis_span_torch(obj_init_vertices.detach())
            print(
                "[quest_display] hq rope gaussian import: "
                f"case={case_name} "
                f"gaussian_source={gs_path} "
                f"total_gaussians={raw_gaussian_count} "
                f"kept_gaussians={kept_gaussian_count} "
                "opacity_pruning=disabled "
                f"gaussian_bounds_min={gaussian_bounds_min} "
                f"gaussian_bounds_max={gaussian_bounds_max} "
                f"frame0_object_bounds_min={object_bounds_min} "
                f"frame0_object_bounds_max={object_bounds_max} "
                f"frame0_object_span_after_scale={scaled_object_span:.8f}",
                flush=True,
            )

        startup_yaw_angle = self._resolve_immersive_startup_yaw_angle(obj_init_vertices)
        startup_yaw_debug = self._apply_immersive_startup_yaw(
            obj_init_vertices,
            None,
            gaussians,
            recorded_base_target,
            recorded_anchor_centers,
            original_controller_source_anchor_centers,
            yaw_angle=startup_yaw_angle,
        )
        obj_init_vertices = startup_yaw_debug["object_vertices"]
        recorded_base_target = startup_yaw_debug["recorded_base_target"]
        recorded_anchor_centers = startup_yaw_debug["recorded_anchor_centers"]
        original_controller_source_anchor_centers = startup_yaw_debug[
            "controller_source_anchor_centers"
        ]
        print(
            "[quest_display] immersive startup yaw: "
            f"axis={startup_yaw_debug['yaw_axis'].detach().cpu().numpy().tolist()} "
            f"angle={startup_yaw_debug['yaw_angle']:.4f} "
            f"pivot={startup_yaw_debug['yaw_pivot'].detach().cpu().numpy().tolist()} "
            f"support_center={startup_yaw_debug['rotated_support_center'].detach().cpu().numpy().tolist()} "
            f"case={case_name} "
            "controller_runtime_rotated=1",
            flush=True,
        )
        controller_predefined_anchor_defs = self._build_case_interaction_anchors(
            obj_init_vertices,
            intrinsic_torch,
            w2c_torch,
        )
        live_controller_case_profile = self._live_controller_case_profile()
        controller_runtime_base_target = None
        controller_source_masks = None
        controller_source_anchor_centers = None
        controller_attachment_metadata = None
        controller_anchor_templates = {"left": {}, "right": {}}
        rotation_cache = None

        background_black = torch.tensor(
            [0.0, 0.0, 0.0], dtype=torch.float32, device="cuda"
        )
        background_white = torch.tensor(
            [1.0, 1.0, 1.0], dtype=torch.float32, device="cuda"
        )
        render_pipe = SimpleNamespace(
            debug=False,
            antialiasing=True,
            compute_cov3D_python=False,
            convert_SHs_python=False,
        )

        repo_root = Path(__file__).resolve().parents[2]
        ensure_simple_lab_assets(scene_assets_root)
        eye_width = int(self.IMMERSIVE_EYE_WIDTH)
        eye_height = int(self.IMMERSIVE_EYE_HEIGHT)
        immersive_render_options = self._resolve_immersive_render_options(
            immersive_render_preset="balanced",
            immersive_scene_render_scale=None,
            immersive_scene_stereo_mode=None,
            immersive_overlay_mode=None,
        )
        scene_width, scene_height = self._resolve_immersive_scene_resolution(
            eye_width,
            eye_height,
            immersive_render_options["scene_render_scale"],
        )
        active_scene_stereo_mode = immersive_render_options["scene_stereo_mode"]
        immersive_bridge = None
        scene_renderer = None
        static_scene_worker = None
        preview_tex = None
        preview_uploader = None
        preview_prog = None
        preview_vao = None
        preview_display_active = interactive_window_mode == "visible"
        left_eye_frame = None
        right_eye_frame = None
        shared_scene_compose_cache = {}
        shared_scene_reproject_caches = {
            "source": {},
            "left": {},
            "right": {},
            "background_source": {},
            "background_left": {},
            "background_right": {},
        }
        frame_count = 0

        live_head_alignment = None
        head_pose_state = None
        live_controller_alignment = None
        live_controller_alignment_mode = "unset"
        immersive_controller_basis_state = None
        current_live_left_controller = None
        current_live_right_controller = None
        controller_select_state_cache = {"left": None, "right": None}
        controller_select_hold_state = {"left": {}, "right": {}}
        controller_select_hold_state_cache = {"left": None, "right": None}
        controller_anchor_cycle_state_cache = {"left": None, "right": None}
        controller_anchor_cycle_edge_cache = {"left": False, "right": False}
        controller_snap_state_cache = {"left": None, "right": None}
        controller_snap_edge_cache = {"left": False, "right": False}
        controller_exit_hold_state = {"left": {}, "right": {}}
        controller_exit_state_cache = {"left": None, "right": None}
        controller_anchor_preview_state = {
            "left": self._make_controller_anchor_preview_state_entry(),
            "right": self._make_controller_anchor_preview_state_entry(),
        }
        controller_anchor_preview_state_cache = {"left": None, "right": None}
        controller_interaction_state = {"left": None, "right": None}
        controller_interaction_state_cache = {"left": None, "right": None}
        controller_motion_state_cache = {"left": None, "right": None}
        last_left_eye_pose_world = None
        last_right_eye_pose_world = None
        last_immersive_sample = None
        immersive_compose_mode = "depth_aware"
        gaussian_compose_roi_padding = None
        startup_render_debug = None
        startup_timeline = None
        startup_keepalive_state = None
        first_real_publish_done = False
        self._immersive_balanced_runtime_state = None
        immersive_timewarp_mode = str(immersive_timewarp).strip().lower()
        if immersive_timewarp_mode not in {"off", "scene_depth_reproject"}:
            raise ValueError(
                "immersive_timewarp must be one of "
                "{'off', 'scene_depth_reproject'}"
            )
        immersive_static_scene_overlap_mode = str(
            immersive_static_scene_overlap
        ).strip().lower()
        if immersive_static_scene_overlap_mode not in {"off", "on"}:
            raise ValueError(
                "immersive_static_scene_overlap must be one of "
                "{'off', 'on'}"
            )
        scene_depth_reproject_requested = (
            immersive_timewarp_mode == "scene_depth_reproject"
        )
        static_scene_overlap_requested = (
            immersive_static_scene_overlap_mode == "on"
            and active_scene_stereo_mode
            == self.IMMERSIVE_BALANCED_INTERNAL_STEREO_MODE
        )
        if scene_depth_reproject_requested and not static_scene_overlap_requested:
            raise ValueError(
                "immersive_timewarp scene_depth_reproject requires "
                "--immersive_static_scene_overlap on"
            )
        scene_depth_reproject_enabled = scene_depth_reproject_requested
        static_scene_overlap_enabled = False
        static_scene_overlap_failure_logged = False
        scene_pose_staleness_ms_samples = []

        diagnostic_output_path = output_dir
        diagnostic_view_render_path = None
        if diagnostic_output_path is not None:
            diagnostic_view_render_path = os.path.join(
                diagnostic_output_path,
                "immersive_output",
            )
            os.makedirs(diagnostic_view_render_path, exist_ok=True)

        sim_timer = Timer("Immersive Simulator")
        interp_timer = Timer("Immersive Interpolation")
        render_timer = Timer("Immersive Rendering")
        total_timer = Timer("Immersive Total")
        component_times = {
            "simulator": [],
            "full_motion_interpolation": [],
            "rendering": [],
            "total": [],
        }
        immersive_render_profile_keys = [
            "rendering",
            "render_eye_intrinsics_setup_wall",
            "scene_render_center_wall",
            "scene_render_background_center_wall",
            "scene_prepare_background_eye_wall",
            "scene_render_far_center_wall",
            "scene_render_near_center_wall",
            "scene_render_side_left_wall",
            "scene_render_side_right_wall",
            "scene_side_roi_left_ratio",
            "scene_side_roi_right_ratio",
            "scene_side_strip_left_width_ratio",
            "scene_side_strip_right_width_ratio",
            "scene_side_fullframe_fallback_left_ratio",
            "scene_side_fullframe_fallback_right_ratio",
            "scene_render_left_wall",
            "scene_render_right_wall",
            "scene_render_table_left_wall",
            "scene_render_table_right_wall",
            "scene_table_roi_left_ratio",
            "scene_table_roi_right_ratio",
            "scene_table_roi_supersample_scale",
            "scene_table_fullframe_fallback_left_ratio",
            "scene_table_fullframe_fallback_right_ratio",
            "gaussian_raw_left_ratio",
            "gaussian_visible_left_ratio",
            "gaussian_retention_left_ratio",
            "gaussian_compose_roi_left_ratio",
            "gaussian_raw_right_ratio",
            "gaussian_visible_right_ratio",
            "gaussian_retention_right_ratio",
            "gaussian_compose_roi_right_ratio",
            "scene_depth_finite_left_ratio",
            "scene_depth_positive_left_ratio",
            "scene_depth_invalid_left_ratio",
            "scene_depth_suppressed_left_ratio",
            "scene_depth_finite_right_ratio",
            "scene_depth_positive_right_ratio",
            "scene_depth_invalid_right_ratio",
            "scene_depth_suppressed_right_ratio",
            "compose_fallback_active_ratio",
            "scene_reproject_left_cuda",
            "scene_reproject_right_cuda",
            "scene_reproject_hole_fill_left_cuda",
            "scene_reproject_hole_fill_right_cuda",
            "scene_reproject_valid_pre_left_ratio",
            "scene_reproject_valid_post_left_ratio",
            "scene_reproject_valid_pre_right_ratio",
            "scene_reproject_valid_post_right_ratio",
            "scene_reproject_roi_pre_left_ratio",
            "scene_reproject_roi_post_left_ratio",
            "scene_reproject_roi_pre_right_ratio",
            "scene_reproject_roi_post_right_ratio",
            "scene_reproject_background_left_cuda",
            "scene_reproject_background_right_cuda",
            "scene_reproject_background_hole_fill_left_cuda",
            "scene_reproject_background_hole_fill_right_cuda",
            "scene_reproject_background_valid_pre_left_ratio",
            "scene_reproject_background_valid_post_left_ratio",
            "scene_reproject_background_valid_pre_right_ratio",
            "scene_reproject_background_valid_post_right_ratio",
            "scene_reproject_background_roi_pre_left_ratio",
            "scene_reproject_background_roi_post_left_ratio",
            "scene_reproject_background_roi_pre_right_ratio",
            "scene_reproject_background_roi_post_right_ratio",
            "scene_warp_far_left_cuda",
            "scene_warp_far_right_cuda",
            "scene_warp_near_left_cuda",
            "scene_warp_near_right_cuda",
            "scene_compose_side_left_cuda",
            "scene_compose_side_right_cuda",
            "gaussian_render_left_cuda",
            "gaussian_render_right_cuda",
            "scene_compose_table_left_cuda",
            "scene_compose_table_right_cuda",
            "compose_left_cuda",
            "compose_right_cuda",
            "overlay_projection_wall",
            "overlay_draw_left_wall",
            "overlay_draw_right_wall",
            "grab_validation_wall",
            "publish_total_wall",
            "publish_process_check_wall",
            "publish_pending_drain_nonblock_wall",
            "publish_pending_drain_block_wall",
            "publish_gpu_to_cpu_wait_wall",
            "publish_gpu_to_cpu_copy_cuda",
            "publish_cpu_mmap_copy_wall",
            "publish_header_write_wall",
            "publish_stage_enqueue_wall",
            "publish_fallback_copy_wall",
            "render_sample_id",
            "publish_sample_id",
            "scene_timewarp_applied",
            "scene_timewarp_fallback_left_used",
            "scene_timewarp_fallback_right_used",
            "scene_timewarp_gpu_ms",
            "static_scene_worker_wall_ms",
            "simulation_lbs_wall_ms",
            "overlap_wait_wall_ms",
            "scene_pose_staleness_ms_at_publish",
            "scene_pose_staleness_savings_ms",
            "preview_window_wall",
            "glfw_poll_wall",
            "eval_png_write_wall",
            "cuda_memory_allocated_gib",
            "cuda_memory_reserved_gib",
        ]
        immersive_render_profile_summary_keys = [
            key for key in immersive_render_profile_keys if key != "eval_png_write_wall"
        ]
        immersive_render_profile_series = (
            {key: [] for key in immersive_render_profile_keys}
            if render_profile
            else None
        )
        immersive_render_profile_rows = [] if render_profile else None

        try:
            startup_timeline = self._make_immersive_startup_timeline()
            self._record_immersive_startup_milestone(
                startup_timeline,
                "scene_renderer_construct_begin",
                immersive_bridge,
            )
            scene_renderer = SimpleLabSceneRenderer(
                scene_assets_root=scene_assets_root,
                width=scene_width,
                height=scene_height,
                lighting_mode=immersive_render_options["lighting_mode"],
                balanced_render_backend=(
                    "pyrender"
                    if active_scene_stereo_mode
                    == self.IMMERSIVE_BALANCED_INTERNAL_STEREO_MODE
                    else "auto"
                ),
            )
            self._record_immersive_startup_milestone(
                startup_timeline,
                "scene_renderer_construct_done",
                immersive_bridge,
            )
            scene_analysis_cache_debug = scene_renderer.scene_analysis_cache_debug()
            print(
                "[quest_display] immersive scene analysis cache: "
                f"status={scene_analysis_cache_debug.get('status')} "
                f"reason={scene_analysis_cache_debug.get('reason')} "
                f"schema={scene_analysis_cache_debug.get('schema')} "
                f"input_hash={scene_analysis_cache_debug.get('input_hash')} "
                f"path={scene_analysis_cache_debug.get('path')}",
                flush=True,
            )
            self._record_immersive_startup_milestone(
                startup_timeline,
                "bridge_start_begin",
                immersive_bridge,
            )
            immersive_bridge = OpenXRImmersiveBridge(
                repo_root,
                width=eye_width,
                height=eye_height,
            )
            immersive_bridge.start()
            self._record_immersive_startup_milestone(
                startup_timeline,
                "bridge_started",
                immersive_bridge,
            )
            startup_keepalive_state = self._make_immersive_startup_keepalive_state(
                eye_width,
                eye_height,
            )
            self._maybe_publish_immersive_startup_keepalive(
                immersive_bridge,
                startup_keepalive_state,
                reason="bridge_started",
                startup_timeline=startup_timeline,
                force=True,
            )
            initial_sample = self._wait_for_valid_immersive_startup_sample(
                immersive_bridge,
                timeout=10.0,
                progress_callback=lambda reason: self._maybe_publish_immersive_startup_keepalive(
                    immersive_bridge,
                    startup_keepalive_state,
                    reason=reason,
                    startup_timeline=startup_timeline,
                ),
            )
            self._record_immersive_startup_milestone(
                startup_timeline,
                "initial_sample_ready",
                immersive_bridge,
            )
            self._maybe_publish_immersive_startup_keepalive(
                immersive_bridge,
                startup_keepalive_state,
                reason="initial_sample_ready",
                startup_timeline=startup_timeline,
                force=True,
            )
            last_immersive_sample = initial_sample
            print(
                "[quest_display] immersive bridge target eye resolution="
                f"{eye_width}x{eye_height}",
                flush=True,
            )
            print(
                "[quest_display] immersive render config: "
                f"preset={immersive_render_options['preset']} "
                f"scene_render_scale={immersive_render_options['scene_render_scale']:.3f} "
                f"scene_resolution={scene_width}x{scene_height} "
                f"scene_stereo_mode={active_scene_stereo_mode} "
                f"overlay_mode={immersive_render_options['overlay_mode']} "
                f"lighting_mode={immersive_render_options['lighting_mode']}",
                flush=True,
            )
            if active_scene_stereo_mode == "mono_head_center":
                print(
                    "[quest_display] immersive scene stereo approximation active: "
                    "mono_head_center reuses one room render across both eyes and is known "
                    "to produce near-geometry stereo artifacts",
                    flush=True,
                )
            elif active_scene_stereo_mode == "reproject_from_center":
                print(
                    "[quest_display] immersive scene stereo approximation active: "
                    "reproject_from_center renders the room once and reprojects it into "
                    "left/right eyes",
                    flush=True,
                )
            elif (
                active_scene_stereo_mode
                == self.IMMERSIVE_BALANCED_INTERNAL_STEREO_MODE
            ):
                print(
                    "[quest_display] immersive scene stereo approximation active: "
                    "balanced renders the room background separately for each eye and "
                    "renders only a per-eye table ROI to keep the table crisp",
                    flush=True,
                )
            print(
                "[quest_display] runtime recommended eye sizes: "
                f"left={initial_sample.left_eye.recommended_width}x{initial_sample.left_eye.recommended_height} "
                f"right={initial_sample.right_eye.recommended_width}x{initial_sample.right_eye.recommended_height}",
                flush=True,
            )
            print(
                "[quest_display] immersive startup sample: "
                + self._format_immersive_sample_startup_state(initial_sample),
                flush=True,
            )

            live_head_alignment = self._compute_immersive_head_alignment(initial_sample)
            if live_head_alignment is None:
                raise RuntimeError(
                    "Immersive mode did not receive a valid eye pose for startup.\n"
                    f"last_sample: {self._format_immersive_sample_startup_state(initial_sample)}"
                )
            print(
                "[quest_display] immersive scene frame: "
                f"up={live_head_alignment['scene_up'].tolist()} "
                f"forward={live_head_alignment['scene_forward'].tolist()} "
                f"right={live_head_alignment['scene_right'].tolist()} "
                f"det={live_head_alignment['basis_det']:.5f} "
                f"ortho_err={live_head_alignment['basis_orthogonality_error']:.6f}",
                flush=True,
            )
            immersive_controller_basis_state = (
                self._make_immersive_controller_basis_state(
                    live_head_alignment["basis"],
                    intrinsic,
                )
            )
            print(
                "[live_openxr_controller] immersive controller handedness validation "
                "pending until both controllers have valid grip poses",
                flush=True,
            )

            (
                initial_left_eye_pose_world,
                initial_right_eye_pose_world,
                head_pose_state,
            ) = self._update_immersive_head_pose_state(
                initial_sample,
                live_head_alignment,
                head_pose_state,
                frame_index=0,
            )
            valid_eye_poses = [
                pose
                for pose in (initial_left_eye_pose_world, initial_right_eye_pose_world)
                if pose is not None
            ]
            if not valid_eye_poses:
                raise RuntimeError(
                    "Immersive mode needs at least one valid eye pose from the Quest runtime."
                )
            initial_left_intrinsic = (
                self._eye_sample_intrinsic(initial_sample.left_eye, eye_width, eye_height)
                if initial_sample.left_eye is not None
                and initial_sample.left_eye.pose_valid
                else None
            )
            initial_right_intrinsic = (
                self._eye_sample_intrinsic(initial_sample.right_eye, eye_width, eye_height)
                if initial_sample.right_eye is not None
                and initial_sample.right_eye.pose_valid
                else None
            )
            head_position = np.mean(
                [pose[:3, 3] for pose in valid_eye_poses], axis=0
            ).astype(np.float32)
            head_forward = np.mean(
                [self._eye_forward_world(pose) for pose in valid_eye_poses],
                axis=0,
            ).astype(np.float32)
            forward_norm = float(np.linalg.norm(head_forward))
            if forward_norm < 1e-5:
                head_forward = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            else:
                head_forward /= forward_norm

            layout = make_simple_lab_layout(
                head_position,
                head_forward,
                scene_up=live_head_alignment["scene_up"],
            )
            self._record_immersive_startup_milestone(
                startup_timeline,
                "layout_ready",
                immersive_bridge,
            )
            self._maybe_publish_immersive_startup_keepalive(
                immersive_bridge,
                startup_keepalive_state,
                reason="layout_ready",
                startup_timeline=startup_timeline,
                force=True,
            )
            scene_renderer.set_layout(layout)
            self._record_immersive_startup_milestone(
                startup_timeline,
                "scene_layout_applied",
                immersive_bridge,
            )
            self._maybe_publish_immersive_startup_keepalive(
                immersive_bridge,
                startup_keepalive_state,
                reason="scene_layout_applied",
                startup_timeline=startup_timeline,
            )
            main_renderer_readback_mode = str(scene_renderer.pyrender_readback_mode())
            main_renderer_readback_reason = scene_renderer.pyrender_readback_reason()
            print(
                f"[quest_display] main_renderer_readback_mode={main_renderer_readback_mode}",
                flush=True,
            )
            if main_renderer_readback_reason:
                print(
                    f"[quest_display] main_renderer_readback_reason={main_renderer_readback_reason}",
                    flush=True,
                )
            if main_renderer_readback_mode != "gl_cuda_interop":
                raise RuntimeError(
                    "Quest immersive runtime requires pyrender_readback_mode=gl_cuda_interop "
                    "so scene/background/table compose stays tensor-only."
                )
            table_alignment_debug = scene_renderer.table_alignment_debug()
            table_surface_center_world = self._scene_table_surface_center_world(layout)
            if table_alignment_debug is not None:
                collider_top_plane_height = float(
                    table_alignment_debug["collider_top_plane_height"]
                )
                world_surface_plane_height = float(
                    table_alignment_debug["world_surface_plane_height"]
                )
                table_surface_center_world = np.asarray(
                    table_alignment_debug["world_surface_center"],
                    dtype=np.float32,
                )
                if (
                    abs(world_surface_plane_height - collider_top_plane_height)
                    > self.IMMERSIVE_STARTUP_PLANE_EPS
                ):
                    raise RuntimeError(
                        "Immersive scene table alignment validation failed: "
                        f"{table_alignment_debug}"
                    )
                print(
                    "[quest_display] table alignment: "
                    f"asset_transform={table_alignment_debug['asset_transform']} "
                    f"surface_normal={table_alignment_debug['world_surface_normal']} "
                    f"normal_alignment={table_alignment_debug['surface_normal_alignment']:.4f} "
                    f"surface_center={table_alignment_debug['world_surface_center']} "
                    f"surface_plane={world_surface_plane_height:.4f} "
                    f"collider_plane={collider_top_plane_height:.4f} "
                    f"active_table_patches={table_alignment_debug.get('active_table_support_patch_count', 'n/a')} "
                    f"support_slabs={table_alignment_debug.get('support_slab_count', 'n/a')} "
                    f"blocker_boxes={table_alignment_debug.get('blocker_box_count', 'n/a')} "
                    f"collider_boxes={table_alignment_debug.get('collider_box_count', 'n/a')}",
                    flush=True,
                )
                print(
                    "[quest_display] table timewarp partition: "
                    f"table_component_ids={table_alignment_debug.get('table_render_component_ids', [])} "
                    "background_excludes_active_table="
                    f"{int(bool(table_alignment_debug.get('background_excludes_active_table', False)))} "
                    "table_bounds_source="
                    f"{table_alignment_debug.get('table_render_bounds_source', 'unknown')} "
                    "table_render_bounds="
                    f"{table_alignment_debug.get('table_render_world_bounds', 'n/a')}",
                    flush=True,
                )
            print(
                "[quest_display] immersive layout: "
                f"head_position={head_position.tolist()} "
                f"table_top_center={layout.table_top_center.tolist()}",
                flush=True,
            )

            spawn_shift = self._compute_scene_spawn_shift(
                obj_init_vertices,
                table_surface_center_world,
            )
            spawn_shift = spawn_shift.to(device=cfg.device, dtype=torch.float32)
            obj_init_vertices = obj_init_vertices + spawn_shift
            recorded_base_target = recorded_base_target + spawn_shift
            gaussians._xyz = gaussians._xyz + spawn_shift
            recorded_anchor_centers = [
                center + spawn_shift for center in recorded_anchor_centers
            ]
            original_controller_source_anchor_centers = [
                center + spawn_shift
                for center in original_controller_source_anchor_centers
            ]
            self._record_immersive_startup_milestone(
                startup_timeline,
                "spawn_shift_done",
                immersive_bridge,
            )
            self._maybe_publish_immersive_startup_keepalive(
                immersive_bridge,
                startup_keepalive_state,
                reason="spawn_shift_done",
                startup_timeline=startup_timeline,
            )

            immersive_center_eye_pose_world, immersive_center_intrinsic = (
                self._build_immersive_center_scene_view(
                    initial_left_eye_pose_world,
                    initial_right_eye_pose_world,
                    initial_left_intrinsic,
                    initial_right_intrinsic,
                )
            )
            immersive_center_w2c = self._camera_pose_world_to_cv_w2c(
                immersive_center_eye_pose_world
            )
            controller_predefined_anchor_defs, rope_endpoint_naming_debug = (
                self._resolve_rope_endpoint_anchor_defs(
                    controller_predefined_anchor_defs,
                    immersive_center_intrinsic,
                    immersive_center_w2c,
                )
            )
            if self._is_rope_family_case(rope_endpoint_naming_debug.get("case_name")):
                print(
                    "[quest_display] immersive rope endpoint naming: "
                    f"endpoint_projected_x={rope_endpoint_naming_debug['endpoint_projected_x']} "
                    f"naming_valid={int(bool(rope_endpoint_naming_debug['naming_valid']))} "
                    f"fallback={int(bool(rope_endpoint_naming_debug['fallback_used']))}",
                    flush=True,
                )
            startup_controller_source_assignment = (
                self._assign_startup_controller_sources_by_screen_x(
                    original_controller_source_masks,
                    original_controller_source_anchor_centers,
                    immersive_center_intrinsic,
                    immersive_center_w2c,
                    recorded_anchor_centers=recorded_anchor_centers,
                    assignment_camera="immersive_center_view",
                )
            )
            original_controller_source_masks = startup_controller_source_assignment[
                "controller_source_masks"
            ]
            original_controller_source_anchor_centers = startup_controller_source_assignment[
                "controller_source_anchor_centers"
            ]
            recorded_anchor_centers = startup_controller_source_assignment[
                "recorded_anchor_centers"
            ]
            self._log_startup_controller_source_assignment(
                "[quest_display] immersive",
                startup_controller_source_assignment,
            )
            startup_predefined_anchor_states = (
                self._compute_predefined_interaction_anchor_states(
                    controller_predefined_anchor_defs,
                    obj_init_vertices,
                )
            )
            resolved_default_anchor_names, anchor_mapping_debug = (
                self._resolve_case_default_controller_anchor_names(
                    startup_predefined_anchor_states,
                    original_controller_source_anchor_centers,
                    immersive_center_intrinsic,
                    immersive_center_w2c,
                )
            )
            self._log_case_controller_anchor_mapping(
                "[quest_display] immersive",
                anchor_mapping_debug,
            )
            print(
                "[live_openxr_controller] immersive controller case profile: "
                f"case={live_controller_case_profile['case_name']} "
                f"translation_scale={live_controller_case_profile['controller_translation_scale']:.2f} "
                f"post_select_grab_mode={live_controller_case_profile['post_select_grab_mode']}",
                flush=True,
            )

            two_point_runtime = self._build_two_point_live_controller_runtime(
                obj_init_vertices,
                trained_spring_Y_for_sim,
                original_controller_source_masks,
                original_controller_source_anchor_centers,
                controller_predefined_anchor_defs,
                default_anchor_names=resolved_default_anchor_names,
            )
            controller_runtime_base_target = two_point_runtime["controller_rest_points"].clone()
            controller_source_masks = two_point_runtime["controller_source_masks"]
            controller_source_anchor_centers = two_point_runtime[
                "controller_source_anchor_centers"
            ]
            init_springs_for_sim = two_point_runtime["init_springs"]
            init_rest_lengths_for_sim = two_point_runtime["init_rest_lengths"]
            trained_spring_Y_for_sim = two_point_runtime["spring_y"]
            controller_attachment_metadata = self._build_controller_attachment_metadata(
                init_springs_for_sim,
                init_rest_lengths_for_sim,
                self.num_all_points,
                controller_source_masks,
            )
            for source, runtime_meta in two_point_runtime["source_runtime"].items():
                controller_attachment_metadata[source].update(runtime_meta)
            ctrl_init_vertices = controller_runtime_base_target

            n_vert_single_obj = obj_init_vertices.shape[0]
            n_vert_single_ctrl = ctrl_init_vertices.shape[0]
            n_springs_single_obj = int(self.num_object_springs)
            n_spring_single_ctrl = int(
                init_springs_for_sim.shape[0] - self.num_object_springs
            )
            base_ctrl_vert_offset = n_vert_single_obj

            self.batch_init_vertices = torch.cat(
                [obj_init_vertices, ctrl_init_vertices], dim=0
            )
            self.batch_init_velocities = (
                self.init_velocities.clone() if self.init_velocities is not None else None
            )
            self.batch_controller_points = controller_runtime_base_target.unsqueeze(0).repeat(
                self.frame_len, 1, 1
            )
            self._record_immersive_startup_milestone(
                startup_timeline,
                "sim_init_begin",
                immersive_bridge,
            )
            self.simulator = SpringMassSystemWarp(
                init_springs=init_springs_for_sim,
                init_rest_lengths=init_rest_lengths_for_sim,
                init_masses=init_masses_for_sim,
                init_masks=self.init_masks,
                init_vertices=self.batch_init_vertices,
                init_velocities=self.batch_init_velocities,
                dt=cfg.dt,
                num_substeps=cfg.num_substeps,
                dashpot_damping=cfg.dashpot_damping,
                drag_damping=cfg.drag_damping,
                collision_dist=cfg.collision_dist,
                reverse_z=cfg.reverse_z,
                spring_Y_max=cfg.spring_Y_max,
                spring_Y_min=cfg.spring_Y_min,
                self_collision=cfg.self_collision,
                collide_elas=trained_collide_elas,
                collide_fric=trained_collide_fric,
                collide_object_elas=trained_collide_object_elas,
                collide_object_fric=trained_collide_object_fric,
                spring_Y=trained_spring_Y_for_sim,
                object_massnodes_total=base_ctrl_vert_offset,
                object_massnodes_single=n_vert_single_obj,
                object_springs_total=n_springs_single_obj,
                object_springs_single=n_springs_single_obj,
                controller_massnodes_single=n_vert_single_ctrl,
                controller_springs_single=n_spring_single_ctrl,
                controller_rest_location=self.batch_controller_points[0],
                number_of_instance=1,
                use_ground_plane=False,
            )
            self.simulator.set_init_state(
                self.simulator.wp_init_vertices, self.simulator.wp_init_velocities
            )
            self._record_immersive_startup_milestone(
                startup_timeline,
                "sim_init_done",
                immersive_bridge,
            )
            self._maybe_publish_immersive_startup_keepalive(
                immersive_bridge,
                startup_keepalive_state,
                reason="sim_init_done",
                startup_timeline=startup_timeline,
            )
            prev_x = wp.to_torch(
                self.simulator.wp_states[0].wp_x, requires_grad=False
            ).clone()
            current_pos = gaussians.get_xyz
            current_rot = gaussians.get_rotation
            relations_single = get_topk_indices(prev_x, K=4)
            weights_single, weights_indices_single = knn_weights_sparse(
                prev_x,
                current_pos,
                K=4,
            )
            rotation_cache = build_rotation_reuse_cache(
                weights_indices=weights_indices_single,
                weights=weights_single,
                relations=relations_single,
                mass_nodes_rest=prev_x,
                gaussians_xyz_rest=current_pos,
                gaussians_quat_rest=current_rot,
                device=cfg.device,
                mass_node_per_instance=n_vert_single_obj,
                gaussians_per_instance=current_pos.shape[0],
                number_of_instance=1,
            )
            spawn_support_center = self._validate_scene_spawn_alignment(
                self.batch_init_vertices[: self.num_all_points],
                layout,
                context="spawn shift",
                table_surface_center_world=table_surface_center_world,
            )
            _, _, spawn_xy_error, spawn_z_error = self._scene_spawn_alignment_metrics(
                self.batch_init_vertices[: self.num_all_points],
                layout,
                table_surface_center_world=table_surface_center_world,
            )
            print(
                "[quest_display] immersive spawn shift: "
                f"shift={spawn_shift.detach().cpu().numpy().tolist()} "
                f"support_center={spawn_support_center.detach().cpu().numpy().tolist()} "
                f"xy_error={spawn_xy_error:.4f} z_error={spawn_z_error:.4f}",
                flush=True,
            )
            live_controller_alignment = None
            live_controller_alignment_mode = "unset"
            controller_runtime_state = self._update_live_controller_runtime_from_sample(
                initial_sample,
                live_controller_alignment,
                live_controller_alignment_mode,
                controller_source_anchor_centers,
                w2c,
                controller_select_state_cache,
                controller_select_hold_state,
                controller_select_hold_state_cache,
                controller_anchor_cycle_state_cache,
                controller_snap_state_cache,
                controller_snap_edge_cache,
                controller_exit_hold_state,
                basis_override=immersive_controller_basis_state["basis"],
                collect_reset_edges=False,
                collect_exit_holds=False,
                alignment_pose_role="grip",
                controller_position_pose_role="grip",
                controller_ray_pose_role="aim",
            )
            (
                controller_runtime_state,
                immersive_controller_basis_state,
            ) = self._maybe_resolve_immersive_controller_handedness(
                initial_sample,
                controller_runtime_state,
                immersive_controller_basis_state,
                controller_source_anchor_centers,
                intrinsic,
                w2c,
                controller_interaction_state=controller_interaction_state,
                alignment_pose_role="grip",
                controller_position_pose_role="grip",
                controller_ray_pose_role="aim",
            )
            live_controller_alignment = controller_runtime_state["alignment"]
            live_controller_alignment_mode = controller_runtime_state["alignment_mode"]
            if live_controller_alignment is None:
                print(
                    "[live_openxr_controller] immersive controller alignment still pending "
                    "after scene spawn shift",
                    flush=True,
                )
            else:
                print(
                    "[live_openxr_controller] immersive controller alignment ready "
                    f"after scene spawn shift mode={live_controller_alignment_mode}",
                    flush=True,
                )
            current_live_left_controller = controller_runtime_state["left_controller"]
            current_live_right_controller = controller_runtime_state["right_controller"]
            print(
                "[live_openxr_controller] immersive using shared 2D controller runtime",
                flush=True,
            )
            last_left_eye_pose_world = initial_left_eye_pose_world
            last_right_eye_pose_world = initial_right_eye_pose_world
            if last_left_eye_pose_world is None:
                last_left_eye_pose_world = last_right_eye_pose_world
            if last_right_eye_pose_world is None:
                last_right_eye_pose_world = last_left_eye_pose_world
            if (
                active_scene_stereo_mode
                == self.IMMERSIVE_BALANCED_INTERNAL_STEREO_MODE
            ):
                self._immersive_balanced_runtime_state = (
                    self._prepare_immersive_balanced_runtime_state(
                        layout,
                        last_left_eye_pose_world,
                        last_right_eye_pose_world,
                        initial_left_intrinsic,
                        initial_right_intrinsic,
                        eye_width,
                        eye_height,
                        scene_width,
                        scene_height,
                    )
                )
                print(
                    "[quest_display] immersive balanced background tuning: "
                    f"mode={self._immersive_balanced_runtime_state['background_mode']} "
                    f"background_resolution="
                    f"{self._immersive_balanced_runtime_state['background_width']}x"
                    f"{self._immersive_balanced_runtime_state['background_height']} "
                    "side_wall_mode=disabled "
                    f"table_mode={self._immersive_balanced_runtime_state['table_mode']} "
                    f"table_roi_render_scale="
                    f"{self._immersive_balanced_runtime_state['table_roi_render_scale']:.2f}",
                    flush=True,
                )
                if not static_scene_overlap_requested:
                    print(
                        "[quest_display] immersive static-scene overlap: "
                        "mode=serial_reference",
                        flush=True,
                    )
                if static_scene_overlap_requested:
                    worker_validation_request = self._build_immersive_balanced_scene_render_plan(
                        scene_renderer,
                        last_left_eye_pose_world,
                        last_right_eye_pose_world,
                        initial_left_intrinsic,
                        initial_right_intrinsic,
                        eye_width,
                        eye_height,
                        scene_width,
                        scene_height,
                        render_profile_frame=None,
                    )
                    try:
                        static_scene_worker = _ImmersiveStaticSceneRenderWorker(
                            scene_assets_root=scene_assets_root,
                            scene_width=scene_width,
                            scene_height=scene_height,
                            lighting_mode=immersive_render_options["lighting_mode"],
                            balanced_render_backend="pyrender",
                            layout=layout,
                            cuda_device_index=int(torch.cuda.current_device()),
                        )
                        worker_startup_debug = static_scene_worker.start(
                            validation_request=worker_validation_request,
                        )
                        static_scene_overlap_enabled = True
                        if scene_depth_reproject_enabled:
                            print(
                                "[quest_display] static_scene_worker_readback_mode="
                                f"{worker_startup_debug.get('readback_mode')}",
                                flush=True,
                            )
                            worker_readback_reason = worker_startup_debug.get(
                                "readback_reason"
                            )
                            if worker_readback_reason:
                                print(
                                    "[quest_display] static_scene_worker_readback_reason="
                                    f"{worker_readback_reason}",
                                    flush=True,
                                )
                            print(
                                "[quest_display] immersive time warp worker ready: "
                                "mode=scene_depth_reproject overlap=enabled",
                                flush=True,
                            )
                        else:
                            print(
                                "[quest_display] immersive static-scene overlap: "
                                "mode=worker_parallel",
                                flush=True,
                            )
                    except Exception as exc:
                        if static_scene_worker is not None:
                            try:
                                static_scene_worker.stop()
                            except Exception:
                                pass
                        static_scene_worker = None
                        static_scene_overlap_enabled = False
                        if scene_depth_reproject_requested:
                            raise RuntimeError(
                                "immersive_timewarp scene_depth_reproject requires "
                                "static-scene worker startup; "
                                f"worker_init_error={type(exc).__name__}: {exc}"
                            ) from exc
                        scene_depth_reproject_enabled = False
                        print(
                            "[quest_display] immersive static-scene overlap unavailable; "
                            "falling back to serial room render: "
                            f"{type(exc).__name__}: {exc}",
                            flush=True,
                        )

            self._set_scene_collider_boxes(layout)
            support_surface_boxes_np = None
            if layout.support_surface_boxes is not None:
                support_surface_boxes_np = np.asarray(
                    layout.support_surface_boxes,
                    dtype=np.float32,
                )
            else:
                support_surface_boxes_np = np.array(
                    [
                        [layout.table_box.mins, layout.table_box.maxs],
                        [layout.floor_box.mins, layout.floor_box.maxs],
                    ],
                    dtype=np.float32,
                )
            support_surface_boxes_t = torch.as_tensor(
                support_surface_boxes_np,
                dtype=torch.float32,
                device=cfg.device,
            )
            idle_lock_case_profile = self._idle_lock_case_profile()
            idle_lock_state = self._make_idle_lock_state()
            if self.simulator.object_collision_flag:
                self.simulator.create_resting_case()
            self.simulator.create_cuda_graph()

            current_target = controller_runtime_base_target.clone()
            prev_target = current_target.clone()
            scene_rest_state = self._settle_scene_rest_state(
                current_target.clone(),
                progress_callback=lambda reason: self._maybe_publish_immersive_startup_keepalive(
                    immersive_bridge,
                    startup_keepalive_state,
                    reason=reason,
                    startup_timeline=startup_timeline,
                ),
            )
            self._record_immersive_startup_milestone(
                startup_timeline,
                "settled_rest_done",
                immersive_bridge,
            )
            self._restore_sim_state(scene_rest_state)
            x = wp.to_torch(
                self.simulator.wp_states[0].wp_x,
                requires_grad=False,
            ).clone()
            current_pos, current_rot = lbs_with_rotation_reuse(
                current_mass_nodes=x,
                cache=rotation_cache,
            )
            gaussians._xyz = current_pos
            gaussians._rotation = current_rot
            prev_x = x.clone()
            settled_support_center = self._validate_scene_spawn_alignment(
                x[: self.num_all_points],
                layout,
                context="settled rest state",
                table_surface_center_world=table_surface_center_world,
            )
            _, _, settled_xy_error, settled_z_error = self._scene_spawn_alignment_metrics(
                x[: self.num_all_points],
                layout,
                table_surface_center_world=table_surface_center_world,
            )
            settled_bounds_min = (
                x[: self.num_all_points].min(dim=0).values.detach().cpu().numpy().tolist()
            )
            settled_bounds_max = (
                x[: self.num_all_points].max(dim=0).values.detach().cpu().numpy().tolist()
            )
            print(
                "[quest_display] immersive settled rest state: "
                f"support_center={settled_support_center.detach().cpu().numpy().tolist()} "
                f"xy_error={settled_xy_error:.4f} z_error={settled_z_error:.4f} "
                f"bounds_min={settled_bounds_min} bounds_max={settled_bounds_max}",
                flush=True,
            )
            startup_support_fraction = self._scene_support_fraction(
                x[: self.num_all_points],
                support_surface_boxes_t,
                layout.scene_up,
            )
            self._set_idle_lock_state(
                idle_lock_state,
                scene_rest_state,
                x[: self.num_all_points],
                action="seeded",
                reason="startup_validated",
                support_fraction=startup_support_fraction,
            )
            controller_anchor_templates = self._build_predefined_controller_anchor_templates(
                controller_runtime_base_target,
                x[: self.num_all_points],
                controller_source_masks,
                controller_source_anchor_centers,
                controller_attachment_metadata,
                controller_predefined_anchor_defs,
            )
            startup_scene_validation_mode = "assembled_scene"
            if active_scene_stereo_mode == "reproject_from_center":
                reproject_valid, reproject_debug = (
                    self._validate_immersive_reprojected_scene_startup(
                        layout,
                        scene_renderer,
                        initial_sample.left_eye,
                        initial_sample.right_eye,
                        last_left_eye_pose_world,
                        last_right_eye_pose_world,
                        eye_width,
                        eye_height,
                        scene_width,
                        scene_height,
                        gaussians,
                        shared_scene_compose_cache=shared_scene_compose_cache,
                        reproject_caches=shared_scene_reproject_caches,
                        scene_stereo_mode=active_scene_stereo_mode,
                    )
                )
                print(
                    "[quest_display] immersive reprojection startup validation: "
                    + str(reproject_debug),
                    flush=True,
                )
                if not reproject_valid:
                    active_scene_stereo_mode = "per_eye"
                    startup_scene_validation_mode = "assembled_per_eye_fallback"
                    print(
                        "[quest_display] immersive reprojection startup validation failed; "
                        "falling back to per_eye room rendering for this run",
                        flush=True,
                    )
                else:
                    startup_scene_validation_mode = "assembled_reproject_from_center"
            elif (
                active_scene_stereo_mode
                == self.IMMERSIVE_BALANCED_INTERNAL_STEREO_MODE
            ):
                balanced_side_wall_mode = "disabled"
                if self._immersive_balanced_runtime_state is not None:
                    balanced_side_wall_mode = str(
                        self._immersive_balanced_runtime_state.get(
                            "side_wall_mode",
                            "disabled",
                        )
                    )
                if balanced_side_wall_mode == "disabled":
                    startup_scene_validation_mode = (
                        "assembled_balanced_per_eye_background_table_roi"
                    )
                    print(
                        "[quest_display] immersive balanced side-strip startup validation "
                        "skipped; shipped balanced mode uses per-eye room backgrounds",
                        flush=True,
                    )
                else:
                    balanced_edge_valid, balanced_edge_debug = (
                        self._validate_immersive_balanced_edge_warp_startup(
                            scene_renderer,
                            initial_sample.left_eye,
                            initial_sample.right_eye,
                            last_left_eye_pose_world,
                            last_right_eye_pose_world,
                            eye_width,
                            eye_height,
                            scene_width,
                            scene_height,
                            shared_scene_compose_cache=shared_scene_compose_cache,
                            reproject_caches=shared_scene_reproject_caches,
                        )
                    )
                    print(
                        "[quest_display] immersive balanced side-strip startup validation: "
                        + str(balanced_edge_debug),
                        flush=True,
                    )
                    if not balanced_edge_valid:
                        startup_scene_validation_mode = (
                            "assembled_balanced_center_background_side_strip_replace_fallback"
                        )
                        print(
                            "[quest_display] immersive balanced side-strip startup validation "
                            "reported weak startup coverage; keeping full-room ROI side-strip "
                            "replacement enabled and using exceptional full-frame fallback as needed",
                            flush=True,
                        )
                    else:
                        startup_scene_validation_mode = (
                            "assembled_balanced_center_background_side_strip_replace_table_roi"
                        )
            gaussian_compose_roi_padding = (
                int(self.IMMERSIVE_GAUSSIAN_COMPOSE_ROI_PADDING)
                if active_scene_stereo_mode
                == self.IMMERSIVE_BALANCED_INTERNAL_STEREO_MODE
                else None
            )
            self._record_immersive_startup_milestone(
                startup_timeline,
                "startup_validation_begin",
                immersive_bridge,
            )
            startup_render_debug = self._validate_immersive_startup_render(
                live_head_alignment,
                layout,
                scene_renderer,
                initial_sample.left_eye,
                initial_sample.right_eye,
                last_left_eye_pose_world,
                last_right_eye_pose_world,
                eye_width,
                eye_height,
                gaussians,
                render_pipe,
                background_black,
                background_white,
                diagnostic_view_render_path,
                save_success_bundle=render_profile,
                scene_stereo_mode=active_scene_stereo_mode,
                scene_width=scene_width,
                scene_height=scene_height,
                reproject_caches=shared_scene_reproject_caches,
                gaussian_compose_roi_padding=gaussian_compose_roi_padding,
                progress_callback=lambda reason: self._maybe_publish_immersive_startup_keepalive(
                    immersive_bridge,
                    startup_keepalive_state,
                    reason=reason,
                    startup_timeline=startup_timeline,
                ),
            )
            self._record_immersive_startup_milestone(
                startup_timeline,
                "startup_validation_done",
                immersive_bridge,
            )
            self._maybe_publish_immersive_startup_keepalive(
                immersive_bridge,
                startup_keepalive_state,
                reason="startup_validation_done",
                startup_timeline=startup_timeline,
            )
            startup_render_debug["requested_scene_stereo_mode"] = (
                immersive_render_options["scene_stereo_mode"]
            )
            startup_render_debug["active_scene_stereo_mode"] = active_scene_stereo_mode
            startup_render_debug["startup_scene_validation_mode"] = (
                startup_scene_validation_mode
            )
            immersive_compose_mode = str(
                startup_render_debug.get("recommended_compose_mode", "depth_aware")
            )
            print(
                "[quest_display] immersive startup render validation: "
                + str(startup_render_debug),
                flush=True,
            )
            if immersive_compose_mode != "depth_aware":
                print(
                    "[quest_display] immersive startup compose validation failed; "
                    "falling back to alpha_overlay object compositing for this run",
                    flush=True,
                )
            last_valid_sim_state = self._capture_sim_state()
            last_valid_target = current_target.clone()
            last_valid_gaussian_state = self._capture_gaussian_runtime_state(gaussians)
            last_valid_object_center = x[: self.num_all_points].mean(dim=0).clone()

            glfw.make_context_current(window)
            if preview_display_active:
                preview_tex = gl.glGenTextures(1)
                gl.glBindTexture(gl.GL_TEXTURE_2D, preview_tex)
                gl.glTexParameteri(
                    gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR
                )
                gl.glTexParameteri(
                    gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR
                )
                gl.glTexImage2D(
                    gl.GL_TEXTURE_2D,
                    0,
                    gl.GL_RGBA8,
                    eye_width,
                    eye_height,
                    0,
                    gl.GL_RGBA,
                    gl.GL_UNSIGNED_BYTE,
                    None,
                )
                gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
                preview_uploader = PreviewTextureCudaUploader(
                    preview_tex,
                    eye_width,
                    eye_height,
                    device=torch.device(cfg.device),
                )

                vertex_shader = """
                #version 330 core
                out vec2 uv;
                const vec2 V[4]=vec2[4](vec2(-1,-1),vec2(1,-1),vec2(-1,1),vec2(1,1));
                const vec2 T[4]=vec2[4](vec2(0,0),vec2(1,0),vec2(0,1),vec2(1,1));
                void main(){ gl_Position=vec4(V[gl_VertexID],0,1); uv=T[gl_VertexID]; }
                """
                fragment_shader = """
                #version 330 core
                in vec2 uv; out vec4 frag; uniform sampler2D uTex;
                void main(){ frag = texture(uTex, vec2(uv.x, 1.0 - uv.y)); }
                """

                def _compile_shader(kind, source):
                    shader_id = gl.glCreateShader(kind)
                    gl.glShaderSource(shader_id, source)
                    gl.glCompileShader(shader_id)
                    if not gl.glGetShaderiv(shader_id, gl.GL_COMPILE_STATUS):
                        raise RuntimeError(gl.glGetShaderInfoLog(shader_id).decode())
                    return shader_id

                preview_prog = gl.glCreateProgram()
                gl.glAttachShader(
                    preview_prog, _compile_shader(gl.GL_VERTEX_SHADER, vertex_shader)
                )
                gl.glAttachShader(
                    preview_prog, _compile_shader(gl.GL_FRAGMENT_SHADER, fragment_shader)
                )
                gl.glLinkProgram(preview_prog)
                if not gl.glGetProgramiv(preview_prog, gl.GL_LINK_STATUS):
                    raise RuntimeError(gl.glGetProgramInfoLog(preview_prog).decode())
                gl.glUseProgram(preview_prog)
                gl.glUniform1i(gl.glGetUniformLocation(preview_prog, "uTex"), 0)
                gl.glUseProgram(0)
                preview_vao = gl.glGenVertexArrays(1)

            while True:
                total_timer.start()
                controller_reset_triggered = False
                controller_overlay_by_source = {}

                latest_sample = immersive_bridge.get_latest_sample()
                if latest_sample is not None:
                    last_immersive_sample = latest_sample
                    controller_runtime_state = self._update_live_controller_runtime_from_sample(
                        latest_sample,
                        live_controller_alignment,
                        live_controller_alignment_mode,
                        controller_source_anchor_centers,
                        w2c,
                        controller_select_state_cache,
                        controller_select_hold_state,
                        controller_select_hold_state_cache,
                        controller_anchor_cycle_state_cache,
                        controller_snap_state_cache,
                        controller_snap_edge_cache,
                        controller_exit_hold_state,
                        controller_exit_state_cache=controller_exit_state_cache,
                        basis_override=immersive_controller_basis_state["basis"],
                        alignment_pose_role="grip",
                        controller_position_pose_role="grip",
                        controller_ray_pose_role="aim",
                    )
                    (
                        controller_runtime_state,
                        immersive_controller_basis_state,
                    ) = self._maybe_resolve_immersive_controller_handedness(
                        latest_sample,
                        controller_runtime_state,
                        immersive_controller_basis_state,
                        controller_source_anchor_centers,
                        intrinsic,
                        w2c,
                        controller_interaction_state=controller_interaction_state,
                        alignment_pose_role="grip",
                        controller_position_pose_role="grip",
                        controller_ray_pose_role="aim",
                    )
                    live_controller_alignment = controller_runtime_state["alignment"]
                    live_controller_alignment_mode = controller_runtime_state["alignment_mode"]
                    if controller_runtime_state["alignment_acquired"]:
                        print(
                            "[live_openxr_controller] immersive controller alignment acquired "
                            f"after startup mode={live_controller_alignment_mode}",
                            flush=True,
                        )
                    current_live_left_controller = controller_runtime_state["left_controller"]
                    current_live_right_controller = controller_runtime_state["right_controller"]
                    controller_exit_sources = controller_runtime_state["exit_sources"]
                    if controller_exit_sources:
                        pressed_buttons = [
                            self._controller_exit_button_label(source)
                            for source in controller_exit_sources
                        ]
                        print(
                            "[live_openxr_controller] immersive clean exit requested via "
                            + "/".join(pressed_buttons)
                            + "; shutting down cleanly",
                            flush=True,
                        )
                        break
                    (
                        left_eye_pose_world,
                        right_eye_pose_world,
                        head_pose_state,
                    ) = self._update_immersive_head_pose_state(
                        latest_sample,
                        live_head_alignment,
                        head_pose_state,
                        frame_index=frame_count,
                    )
                    if left_eye_pose_world is not None:
                        last_left_eye_pose_world = left_eye_pose_world
                    if right_eye_pose_world is not None:
                        last_right_eye_pose_world = right_eye_pose_world
                    if last_left_eye_pose_world is None:
                        last_left_eye_pose_world = last_right_eye_pose_world
                    if last_right_eye_pose_world is None:
                        last_right_eye_pose_world = last_left_eye_pose_world

                    controller_reset_sources = controller_runtime_state["reset_sources"]
                    if controller_reset_sources:
                        pressed_buttons = [
                            "Y" if source == "left" else "B"
                            for source in controller_reset_sources
                        ]
                        print(
                            "[live_openxr_controller] immersive reset requested via "
                            + "/".join(pressed_buttons)
                            + "; restoring the settled table pose",
                            flush=True,
                        )
                        reset_target = self._reset_live_controller_runtime(
                            controller_runtime_base_target,
                            controller_interaction_state,
                            controller_anchor_preview_state,
                            controller_attachment_metadata,
                            reset_state=scene_rest_state,
                        )
                        prev_target = reset_target.clone()
                        current_target = reset_target
                        reset_object_points = scene_rest_state["x"][
                            : self.num_all_points
                        ]
                        reset_support_fraction = self._scene_support_fraction(
                            reset_object_points,
                            support_surface_boxes_t,
                            layout.scene_up,
                        )
                        self._set_idle_lock_state(
                            idle_lock_state,
                            scene_rest_state,
                            reset_object_points,
                            action="seeded",
                            reason="reset",
                            support_fraction=reset_support_fraction,
                        )
                        prev_x = scene_rest_state["x"].clone()
                        controller_reset_triggered = True
                else:
                    current_live_left_controller = None
                    current_live_right_controller = None

                render_profile_frame = self._render_profile_new_frame(render_profile)
                render_sample = last_immersive_sample
                render_sample_received_monotonic_s = (
                    self._sample_received_monotonic_s(render_sample)
                )
                intrinsics_setup_start = (
                    time.perf_counter() if render_profile_frame is not None else None
                )
                left_intrinsic = self._eye_sample_intrinsic(
                    last_immersive_sample.left_eye,
                    eye_width,
                    eye_height,
                )
                right_intrinsic = self._eye_sample_intrinsic(
                    last_immersive_sample.right_eye,
                    eye_width,
                    eye_height,
                )
                eye_frame_output_dtype = torch.float32
                if intrinsics_setup_start is not None:
                    self._render_profile_add_wall_time(
                        render_profile_frame,
                        "render_eye_intrinsics_setup_wall",
                        time.perf_counter() - intrinsics_setup_start,
                    )
                static_scene_request_submitted = False
                balanced_scene_render_plan = None
                if last_immersive_sample is None:
                    raise RuntimeError("Immersive bridge stopped providing pose samples.")
                if last_left_eye_pose_world is None or last_right_eye_pose_world is None:
                    raise RuntimeError("Immersive eye poses became unavailable.")
                if (
                    active_scene_stereo_mode
                    == self.IMMERSIVE_BALANCED_INTERNAL_STEREO_MODE
                ):
                    balanced_scene_render_plan = (
                        self._build_immersive_balanced_scene_render_plan(
                            scene_renderer,
                            last_left_eye_pose_world,
                            last_right_eye_pose_world,
                            left_intrinsic,
                            right_intrinsic,
                            eye_width,
                            eye_height,
                            scene_width,
                            scene_height,
                            render_profile_frame=render_profile_frame,
                        )
                    )
                if static_scene_overlap_enabled and static_scene_worker is not None:
                    try:
                        static_scene_worker.submit(balanced_scene_render_plan)
                        static_scene_request_submitted = True
                    except Exception as exc:
                        if not static_scene_overlap_failure_logged:
                            print(
                                "[quest_display] immersive static-scene overlap disabled; "
                                "falling back to serial room render: "
                                f"{type(exc).__name__}: {exc}",
                                flush=True,
                            )
                            static_scene_overlap_failure_logged = True
                        if static_scene_worker is not None:
                            try:
                                static_scene_worker.stop()
                            except Exception:
                                pass
                        static_scene_worker = None
                        static_scene_overlap_enabled = False
                        scene_depth_reproject_enabled = False

                if (
                    idle_lock_state.get("active", False)
                    and (
                        controller_interaction_state.get("left") is not None
                        or controller_interaction_state.get("right") is not None
                    )
                ):
                    self._release_idle_lock_state(
                        idle_lock_state,
                        reason="interaction_started",
                    )

                sim_timer.start()
                if idle_lock_state.get("active", False):
                    locked_state = idle_lock_state.get("locked_state")
                    if locked_state is None:
                        raise RuntimeError(
                            "Idle lock is active but locked_state is unavailable."
                        )
                    self._restore_sim_state(locked_state)
                    x = locked_state["x"].detach().clone()
                    current_v = locked_state["v"].detach().clone()
                else:
                    pre_step_left_anchor = None
                    pre_step_right_anchor = None
                    if (
                        controller_interaction_state["left"] is not None
                        and current_live_left_controller is not None
                    ):
                        pre_step_left_anchor = self._controller_interaction_anchor(
                            current_live_left_controller,
                            controller_interaction_state["left"],
                        )
                    if (
                        controller_interaction_state["right"] is not None
                        and current_live_right_controller is not None
                    ):
                        pre_step_right_anchor = self._controller_interaction_anchor(
                            current_live_right_controller,
                            controller_interaction_state["right"],
                        )
                    self._apply_live_controller_anchor_kinematic_overrides(
                        pre_step_left_anchor,
                        pre_step_right_anchor,
                        controller_interaction_state,
                    )

                    self.simulator.set_controller_interactive(prev_target, current_target)
                    if self.simulator.object_collision_flag:
                        self.simulator.update_collision_graph()
                    wp.capture_launch(self.simulator.forward_graph)
                    wp.synchronize()
                    self._apply_live_controller_anchor_kinematic_overrides(
                        pre_step_left_anchor,
                        pre_step_right_anchor,
                        controller_interaction_state,
                        state_index=-1,
                    )

                    self.simulator.set_init_state(
                        self.simulator.wp_states[-1].wp_x,
                        self.simulator.wp_states[-1].wp_v,
                    )
                    x = wp.to_torch(
                        self.simulator.wp_states[0].wp_x,
                        requires_grad=False,
                    ).clone()
                    current_v = wp.to_torch(
                        self.simulator.wp_states[0].wp_v,
                        requires_grad=False,
                    ).clone()
                sim_time = sim_timer.stop()
                if frame_count > 1:
                    component_times["simulator"].append(sim_time)

                interp_timer.start()
                current_pos, current_rot = lbs_with_rotation_reuse(
                    current_mass_nodes=x,
                    cache=rotation_cache,
                )
                gaussians._xyz = current_pos
                gaussians._rotation = current_rot
                interp_time = interp_timer.stop()
                if frame_count > 1:
                    component_times["full_motion_interpolation"].append(interp_time)
                prev_x = x.clone()

                object_points = x[: self.num_all_points]
                self._update_idle_lock_state(
                    idle_lock_state,
                    {"x": x, "v": current_v},
                    object_points,
                    current_v[: self.num_all_points],
                    controller_interaction_state,
                    idle_lock_case_profile,
                    support_surface_boxes_t,
                    layout.scene_up,
                )
                object_bounds_min = object_points.min(dim=0).values - 0.01
                object_bounds_max = object_points.max(dim=0).values + 0.01
                object_support_center = (
                    self._object_support_patch_center(object_points)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                controller_predefined_anchor_states = (
                    self._compute_predefined_interaction_anchor_states(
                        controller_predefined_anchor_defs,
                        object_points,
                    )
                )
                current_interaction_anchor_by_source = {"left": None, "right": None}
                for source, controller_world in (
                    ("left", current_live_left_controller),
                    ("right", current_live_right_controller),
                ):
                    interaction_state = controller_interaction_state[source]
                    if interaction_state is not None and controller_world is not None:
                        current_interaction_anchor_by_source[source] = (
                            self._controller_interaction_anchor(
                                controller_world,
                                interaction_state,
                            )
                        )
                controller_overlay_by_source = {}
                for source, controller_world in (
                    ("left", current_live_left_controller),
                    ("right", current_live_right_controller),
                ):
                    overlay_entry = self._build_live_controller_world_overlay(
                        source,
                        controller_world,
                        object_points,
                        object_bounds_min,
                        object_bounds_max,
                    )
                    if overlay_entry is not None:
                        cycle_edge = self._controller_anchor_cycle_edge(
                            source,
                            controller_world,
                            controller_anchor_cycle_edge_cache,
                        )
                        selected_preview_anchor = self._update_controller_anchor_preview_state(
                            source,
                            controller_world,
                            overlay_entry,
                            controller_predefined_anchor_states,
                            controller_anchor_preview_state,
                            cycle_edge,
                            controller_interaction_state[source],
                            controller_interaction_state,
                        )
                        available_anchor_states = (
                            self._filter_available_predefined_interaction_anchors(
                                source,
                                controller_predefined_anchor_states,
                                controller_interaction_state,
                            )
                        )
                        overlay_entry["anchor_preview_entries_world"] = []
                        if (
                            immersive_render_options["overlay_mode"] == "full"
                            and controller_anchor_preview_state[source]["visible"]
                        ):
                            for anchor_state in controller_predefined_anchor_states:
                                preview_occupied, _ = self._anchor_is_occupied_by_other_source(
                                    source,
                                    anchor_state["name"],
                                    controller_interaction_state,
                                )
                                overlay_entry["anchor_preview_entries_world"].append(
                                    {
                                        "world": anchor_state["center_world"],
                                        "name": anchor_state["name"],
                                        "selected": (
                                            anchor_state["name"]
                                            == controller_anchor_preview_state[source][
                                                "selected_anchor_name"
                                            ]
                                        ),
                                        "active": (
                                            controller_interaction_state[source] is not None
                                            and anchor_state["name"]
                                            == controller_interaction_state[source].get(
                                                "anchor_name"
                                            )
                                        ),
                                        "occupied": preview_occupied,
                                    }
                                )
                        nearest_anchor = None
                        if overlay_entry["hit_world"] is not None:
                            nearest_anchor = self._select_predefined_interaction_anchor(
                                overlay_entry["hit_world"],
                                available_anchor_states,
                            )
                        ray_origin_world, ray_direction_world = self._controller_world_ray_pose(
                            controller_world
                        )
                        if (
                            nearest_anchor is None
                            and ray_origin_world is not None
                            and ray_direction_world is not None
                        ):
                            nearest_anchor = self._select_predefined_interaction_anchor_for_ray(
                                ray_origin_world,
                                ray_direction_world,
                                available_anchor_states,
                            )
                        preview_state_entry = controller_anchor_preview_state[source]
                        if bool(preview_state_entry.get("cycle_locked", False)):
                            attach_candidate_anchor = selected_preview_anchor
                        else:
                            attach_candidate_anchor = (
                                selected_preview_anchor
                                if selected_preview_anchor is not None
                                else nearest_anchor
                            )
                        overlay_entry["attach_candidate"] = attach_candidate_anchor is not None
                        overlay_entry["attach_candidate_world"] = (
                            attach_candidate_anchor["center_world"]
                            if attach_candidate_anchor is not None
                            else None
                        )
                        interaction_state = controller_interaction_state[source]
                        overlay_entry["attachment_active"] = interaction_state is not None
                        overlay_entry["attach_active_world"] = (
                            self._current_controller_attach_center_world(
                                interaction_state,
                                current_interaction_anchor_by_source[source],
                            )
                        )
                        controller_overlay_by_source[source] = overlay_entry
                    self._log_controller_anchor_preview_transition(
                        source,
                        controller_anchor_preview_state[source],
                        controller_anchor_preview_state_cache,
                    )

                render_timer.start()
                render_sample_id = (
                    -1 if render_sample is None else int(render_sample.sample)
                )
                publish_sample_id = render_sample_id
                scene_pose_sample = render_sample
                scene_timewarp_applied = 0.0
                scene_timewarp_fallback_left_used = 0.0
                scene_timewarp_fallback_right_used = 0.0
                scene_pose_staleness_ms_at_publish = 0.0
                scene_pose_staleness_savings_ms = 0.0
                simulation_lbs_wall_s = float(sim_time + interp_time)
                overlap_wait_wall_s = 0.0
                static_scene_worker_wall_s = 0.0
                if render_profile_frame is not None:
                    render_profile_frame["render_sample_id"] = float(render_sample_id)
                    render_profile_frame["simulation_lbs_wall_ms"] = (
                        simulation_lbs_wall_s
                    )
                eye_state_setup_start = (
                    time.perf_counter() if render_profile_frame is not None else None
                )
                left_eye_render_state = self._prepare_immersive_eye_render_state(
                    last_left_eye_pose_world,
                    left_intrinsic,
                    eye_height,
                    eye_width,
                    eye_label="left",
                )
                right_eye_render_state = self._prepare_immersive_eye_render_state(
                    last_right_eye_pose_world,
                    right_intrinsic,
                    eye_height,
                    eye_width,
                    eye_label="right",
                )
                if eye_state_setup_start is not None:
                    self._render_profile_add_wall_time(
                        render_profile_frame,
                        "render_eye_intrinsics_setup_wall",
                        time.perf_counter() - eye_state_setup_start,
                    )
                left_overlay_eye_render_state = left_eye_render_state
                right_overlay_eye_render_state = right_eye_render_state
                worker_result_ready = False
                overlap_left_gaussian_rgba = None
                overlap_left_gaussian_depth = None
                overlap_right_gaussian_rgba = None
                overlap_right_gaussian_depth = None
                if static_scene_request_submitted:
                    left_gaussian_span = self._render_profile_begin_cuda_span(
                        render_profile_frame,
                        "gaussian_render_left_cuda",
                    )
                    (
                        overlap_left_gaussian_rgba,
                        overlap_left_gaussian_depth,
                    ) = self._render_gaussian_rgba(
                        left_eye_render_state["view"],
                        gaussians,
                        render_pipe,
                        background_black,
                        background_white,
                        use_gsplat=True,
                    )
                    self._render_profile_end_cuda_span(
                        render_profile_frame,
                        left_gaussian_span,
                    )
                    right_gaussian_span = self._render_profile_begin_cuda_span(
                        render_profile_frame,
                        "gaussian_render_right_cuda",
                    )
                    (
                        overlap_right_gaussian_rgba,
                        overlap_right_gaussian_depth,
                    ) = self._render_gaussian_rgba(
                        right_eye_render_state["view"],
                        gaussians,
                        render_pipe,
                        background_black,
                        background_white,
                        use_gsplat=True,
                    )
                    self._render_profile_end_cuda_span(
                        render_profile_frame,
                        right_gaussian_span,
                    )
                if static_scene_request_submitted:
                    overlap_wait_start = (
                        time.perf_counter() if static_scene_request_submitted else None
                    )
                    try:
                        static_scene_result = static_scene_worker.get_result(timeout=60.0)
                        if overlap_wait_start is not None:
                            overlap_wait_wall_s = (
                                time.perf_counter() - overlap_wait_start
                            )
                        static_scene_worker_wall_s = (
                            float(static_scene_result.get("worker_wall_ms", 0.0))
                            / 1000.0
                        )
                        if render_profile_frame is not None:
                            render_profile_frame["static_scene_worker_wall_ms"] = (
                                static_scene_worker_wall_s
                            )
                            render_profile_frame["overlap_wait_wall_ms"] = (
                                overlap_wait_wall_s
                            )
                        worker_result_ready = True
                    except Exception as exc:
                        if not static_scene_overlap_failure_logged:
                            print(
                                "[quest_display] immersive static-scene overlap failed "
                                "mid-run; falling back to serial room render: "
                                f"{type(exc).__name__}: {exc}",
                                flush=True,
                            )
                            static_scene_overlap_failure_logged = True
                        if static_scene_worker is not None:
                            try:
                                static_scene_worker.stop()
                            except Exception:
                                pass
                        static_scene_worker = None
                        static_scene_overlap_enabled = False
                        scene_depth_reproject_enabled = False

                    if worker_result_ready:
                        (
                            left_scene_color,
                            left_scene_depth,
                            right_scene_color,
                            right_scene_depth,
                        ) = self._assemble_immersive_balanced_scene_from_render_outputs(
                            scene_renderer,
                            balanced_scene_render_plan,
                            static_scene_result,
                            eye_width,
                            eye_height,
                            shared_scene_compose_cache=shared_scene_compose_cache,
                            reproject_caches=shared_scene_reproject_caches,
                            render_profile_frame=render_profile_frame,
                        )

                        left_gaussian_rgba = overlap_left_gaussian_rgba
                        left_gaussian_depth = overlap_left_gaussian_depth
                        right_gaussian_rgba = overlap_right_gaussian_rgba
                        right_gaussian_depth = overlap_right_gaussian_depth

                    left_compose_span = self._render_profile_begin_cuda_span(
                        render_profile_frame,
                        "compose_left_cuda",
                    )
                    if render_profile_frame is not None:
                        (
                            left_eye_frame,
                            left_eye_frame_depth,
                            left_compose_metrics,
                            _,
                        ) = self._compose_immersive_eye_frame(
                            left_scene_color,
                            left_scene_depth,
                            left_gaussian_rgba,
                            left_gaussian_depth,
                            target_height=eye_height,
                            target_width=eye_width,
                            compose_mode=immersive_compose_mode,
                            compose_roi_padding=gaussian_compose_roi_padding,
                            collect_debug=True,
                            output_dtype=eye_frame_output_dtype,
                            return_depth=True,
                        )
                    else:
                        left_eye_frame, left_eye_frame_depth = self._compose_immersive_eye_frame(
                            left_scene_color,
                            left_scene_depth,
                            left_gaussian_rgba,
                            left_gaussian_depth,
                            target_height=eye_height,
                            target_width=eye_width,
                            compose_mode=immersive_compose_mode,
                            compose_roi_padding=gaussian_compose_roi_padding,
                            output_dtype=eye_frame_output_dtype,
                            return_depth=True,
                        )
                        left_compose_metrics = None
                    self._render_profile_end_cuda_span(
                        render_profile_frame,
                        left_compose_span,
                    )

                    right_compose_span = self._render_profile_begin_cuda_span(
                        render_profile_frame,
                        "compose_right_cuda",
                    )
                    if render_profile_frame is not None:
                        (
                            right_eye_frame,
                            right_eye_frame_depth,
                            right_compose_metrics,
                            _,
                        ) = self._compose_immersive_eye_frame(
                            right_scene_color,
                            right_scene_depth,
                            right_gaussian_rgba,
                            right_gaussian_depth,
                            target_height=eye_height,
                            target_width=eye_width,
                            compose_mode=immersive_compose_mode,
                            compose_roi_padding=gaussian_compose_roi_padding,
                            collect_debug=True,
                            output_dtype=eye_frame_output_dtype,
                            return_depth=True,
                        )
                    else:
                        right_eye_frame, right_eye_frame_depth = self._compose_immersive_eye_frame(
                            right_scene_color,
                            right_scene_depth,
                            right_gaussian_rgba,
                            right_gaussian_depth,
                            target_height=eye_height,
                            target_width=eye_width,
                            compose_mode=immersive_compose_mode,
                            compose_roi_padding=gaussian_compose_roi_padding,
                            output_dtype=eye_frame_output_dtype,
                            return_depth=True,
                        )
                        right_compose_metrics = None
                    self._render_profile_end_cuda_span(
                        render_profile_frame,
                        right_compose_span,
                    )

                    publish_sample = immersive_bridge.get_latest_sample()
                    publish_left_eye_pose_world = last_left_eye_pose_world
                    publish_right_eye_pose_world = last_right_eye_pose_world
                    publish_left_intrinsic = left_intrinsic
                    publish_right_intrinsic = right_intrinsic
                    if (
                        scene_depth_reproject_enabled
                        and publish_sample is not None
                        and self._immersive_sample_has_valid_eye_pose(publish_sample)
                        and int(publish_sample.sample) > render_sample_id
                    ):
                        (
                            predicted_left_eye_pose_world,
                            predicted_right_eye_pose_world,
                        ) = self._predict_immersive_eye_poses_for_sample(
                            publish_sample,
                            live_head_alignment,
                            head_pose_state,
                            frame_count,
                        )
                        if predicted_left_eye_pose_world is not None:
                            publish_left_eye_pose_world = predicted_left_eye_pose_world
                        if predicted_right_eye_pose_world is not None:
                            publish_right_eye_pose_world = predicted_right_eye_pose_world
                        if (
                            publish_sample.left_eye is not None
                            and publish_sample.left_eye.pose_valid
                        ):
                            publish_left_intrinsic = self._eye_sample_intrinsic(
                                publish_sample.left_eye,
                                eye_width,
                                eye_height,
                            )
                        if (
                            publish_sample.right_eye is not None
                            and publish_sample.right_eye.pose_valid
                        ):
                            publish_right_intrinsic = self._eye_sample_intrinsic(
                                publish_sample.right_eye,
                                eye_width,
                                eye_height,
                            )
                        latewarp_span = self._render_profile_begin_cuda_span(
                            render_profile_frame,
                            "scene_timewarp_gpu_ms",
                        )
                        try:
                            (
                                warped_left_eye_frame,
                                warped_left_eye_depth,
                                _,
                            ) = self._latewarp_immersive_scene_eye(
                                left_eye_frame,
                                left_eye_frame_depth,
                                last_left_eye_pose_world,
                                publish_left_eye_pose_world,
                                left_intrinsic,
                                publish_left_intrinsic,
                                eye_height,
                                eye_width,
                                "left",
                                reproject_caches=shared_scene_reproject_caches,
                                render_profile_frame=render_profile_frame,
                            )
                            if (
                                torch.is_tensor(warped_left_eye_frame)
                                and torch.is_tensor(warped_left_eye_depth)
                                and tuple(warped_left_eye_frame.shape)
                                == (eye_height, eye_width, 4)
                                and tuple(warped_left_eye_depth.shape)
                                == (eye_height, eye_width)
                                and self._is_finite_tensor(warped_left_eye_frame)
                                and self._is_finite_tensor(warped_left_eye_depth)
                            ):
                                left_eye_frame = warped_left_eye_frame
                                left_eye_frame_depth = warped_left_eye_depth
                            else:
                                scene_timewarp_fallback_left_used = 1.0
                        except Exception:
                            scene_timewarp_fallback_left_used = 1.0
                        try:
                            (
                                warped_right_eye_frame,
                                warped_right_eye_depth,
                                _,
                            ) = self._latewarp_immersive_scene_eye(
                                right_eye_frame,
                                right_eye_frame_depth,
                                last_right_eye_pose_world,
                                publish_right_eye_pose_world,
                                right_intrinsic,
                                publish_right_intrinsic,
                                eye_height,
                                eye_width,
                                "right",
                                reproject_caches=shared_scene_reproject_caches,
                                render_profile_frame=render_profile_frame,
                            )
                            if (
                                torch.is_tensor(warped_right_eye_frame)
                                and torch.is_tensor(warped_right_eye_depth)
                                and tuple(warped_right_eye_frame.shape)
                                == (eye_height, eye_width, 4)
                                and tuple(warped_right_eye_depth.shape)
                                == (eye_height, eye_width)
                                and self._is_finite_tensor(warped_right_eye_frame)
                                and self._is_finite_tensor(warped_right_eye_depth)
                            ):
                                right_eye_frame = warped_right_eye_frame
                                right_eye_frame_depth = warped_right_eye_depth
                            else:
                                scene_timewarp_fallback_right_used = 1.0
                        except Exception:
                            scene_timewarp_fallback_right_used = 1.0
                        self._render_profile_end_cuda_span(
                            render_profile_frame,
                            latewarp_span,
                        )
                        scene_timewarp_applied = 1.0
                        scene_pose_sample = publish_sample
                        publish_sample_id = int(publish_sample.sample)
                        left_overlay_eye_render_state = (
                            self._prepare_immersive_eye_render_state(
                                publish_left_eye_pose_world,
                                publish_left_intrinsic,
                                eye_height,
                                eye_width,
                                eye_label="left_publish",
                            )
                        )
                        right_overlay_eye_render_state = (
                            self._prepare_immersive_eye_render_state(
                                publish_right_eye_pose_world,
                                publish_right_intrinsic,
                                eye_height,
                                eye_width,
                                eye_label="right_publish",
                            )
                        )
                if not worker_result_ready:
                    if (
                        active_scene_stereo_mode
                        == self.IMMERSIVE_BALANCED_INTERNAL_STEREO_MODE
                        and balanced_scene_render_plan is not None
                    ):
                        serial_scene_outputs = (
                            _execute_immersive_balanced_scene_render_plan(
                                scene_renderer,
                                balanced_scene_render_plan,
                            )
                        )
                        (
                            left_scene_color,
                            left_scene_depth,
                            right_scene_color,
                            right_scene_depth,
                        ) = self._assemble_immersive_balanced_scene_from_render_outputs(
                            scene_renderer,
                            balanced_scene_render_plan,
                            serial_scene_outputs,
                            eye_width,
                            eye_height,
                            shared_scene_compose_cache=shared_scene_compose_cache,
                            reproject_caches=shared_scene_reproject_caches,
                            render_profile_frame=render_profile_frame,
                        )
                    else:
                        (
                            left_scene_color,
                            left_scene_depth,
                            right_scene_color,
                            right_scene_depth,
                        ) = self._render_immersive_scene_frames_for_mode(
                            scene_renderer,
                            active_scene_stereo_mode,
                            layout,
                            object_support_center,
                            object_bounds_min.detach().cpu().numpy().astype(np.float32),
                            object_bounds_max.detach().cpu().numpy().astype(np.float32),
                            last_left_eye_pose_world,
                            last_right_eye_pose_world,
                            left_intrinsic,
                            right_intrinsic,
                            eye_width,
                            eye_height,
                            scene_width,
                            scene_height,
                            shared_scene_compose_cache=shared_scene_compose_cache,
                            reproject_caches=shared_scene_reproject_caches,
                            render_profile_frame=render_profile_frame,
                        )
                    (
                        left_eye_frame,
                        left_gaussian_rgba,
                        left_gaussian_depth,
                        left_compose_metrics,
                        _,
                    ) = (
                        self._render_immersive_eye_frame(
                            last_left_eye_pose_world,
                            left_intrinsic,
                            eye_height,
                            eye_width,
                            left_scene_color,
                            left_scene_depth,
                            gaussians,
                            render_pipe,
                            background_black,
                            background_white,
                            render_profile_frame=render_profile_frame,
                            eye_label="left",
                            compose_mode=immersive_compose_mode,
                            compose_roi_padding=gaussian_compose_roi_padding,
                            collect_compose_debug=render_profile_frame is not None,
                            eye_render_state=left_eye_render_state,
                            output_dtype=eye_frame_output_dtype,
                        )
                    )
                    (
                        right_eye_frame,
                        right_gaussian_rgba,
                        right_gaussian_depth,
                        right_compose_metrics,
                        _,
                    ) = (
                        self._render_immersive_eye_frame(
                            last_right_eye_pose_world,
                            right_intrinsic,
                            eye_height,
                            eye_width,
                            right_scene_color,
                            right_scene_depth,
                            gaussians,
                            render_pipe,
                            background_black,
                            background_white,
                            render_profile_frame=render_profile_frame,
                            eye_label="right",
                            compose_mode=immersive_compose_mode,
                            compose_roi_padding=gaussian_compose_roi_padding,
                            collect_compose_debug=render_profile_frame is not None,
                            eye_render_state=right_eye_render_state,
                            output_dtype=eye_frame_output_dtype,
                        )
                    )

                if render_profile_frame is not None:
                    render_profile_frame["publish_sample_id"] = float(
                        publish_sample_id
                    )
                    render_profile_frame["scene_timewarp_applied"] = float(
                        scene_timewarp_applied
                    )
                    render_profile_frame["scene_timewarp_fallback_left_used"] = float(
                        scene_timewarp_fallback_left_used
                    )
                    render_profile_frame["scene_timewarp_fallback_right_used"] = float(
                        scene_timewarp_fallback_right_used
                    )
                    self._render_profile_record_immersive_compose_metrics(
                        render_profile_frame,
                        "left",
                        left_compose_metrics,
                    )
                    self._render_profile_record_immersive_compose_metrics(
                        render_profile_frame,
                        "right",
                        right_compose_metrics,
                    )

                overlay_projection_start = (
                    time.perf_counter() if render_profile_frame is not None else None
                )
                projected_overlay_entries = (
                    self._project_live_controller_world_overlays_batched(
                        list(controller_overlay_by_source.values()),
                        {
                            "left": left_overlay_eye_render_state,
                            "right": right_overlay_eye_render_state,
                        },
                        eye_height,
                        eye_width,
                    )
                )
                left_eye_overlay_entries = projected_overlay_entries.get("left", [])
                right_eye_overlay_entries = projected_overlay_entries.get("right", [])
                if overlay_projection_start is not None:
                    self._render_profile_add_wall_time(
                        render_profile_frame,
                        "overlay_projection_wall",
                        time.perf_counter() - overlay_projection_start,
                    )
                if left_eye_overlay_entries:
                    overlay_draw_left_start = (
                        time.perf_counter() if render_profile_frame is not None else None
                    )
                    self._draw_live_controller_overlay(
                        left_eye_frame,
                        left_eye_overlay_entries,
                    )
                    if overlay_draw_left_start is not None:
                        self._render_profile_add_wall_time(
                            render_profile_frame,
                            "overlay_draw_left_wall",
                            time.perf_counter() - overlay_draw_left_start,
                        )
                if right_eye_overlay_entries:
                    overlay_draw_right_start = (
                        time.perf_counter() if render_profile_frame is not None else None
                    )
                    self._draw_live_controller_overlay(
                        right_eye_frame,
                        right_eye_overlay_entries,
                    )
                    if overlay_draw_right_start is not None:
                        self._render_profile_add_wall_time(
                            render_profile_frame,
                            "overlay_draw_right_wall",
                            time.perf_counter() - overlay_draw_right_start,
                        )

                if left_eye_frame.dtype != torch.uint8:
                    left_eye_frame = left_eye_frame.clamp(0.0, 255.0).to(torch.uint8)
                if right_eye_frame.dtype != torch.uint8:
                    right_eye_frame = right_eye_frame.clamp(0.0, 255.0).to(torch.uint8)

                validation_sources = []
                for source in self._sources_pending_grab_start_validation(
                    controller_interaction_state
                ):
                    interaction_state = controller_interaction_state.get(source)
                    if interaction_state is None:
                        continue
                    delay_frames = int(
                        interaction_state.get("startup_validation_delay_frames", 0)
                    )
                    if delay_frames > 0:
                        interaction_state["startup_validation_delay_frames"] = (
                            delay_frames - 1
                        )
                        continue
                    validation_sources.append(source)

                grab_validation_start = (
                    time.perf_counter() if render_profile_frame is not None else None
                )
                if validation_sources:
                    validation_reason, validation_debug = self._validate_immersive_grab_start_frame(
                        x,
                        current_pos,
                        current_rot,
                        last_valid_object_center,
                        last_immersive_sample,
                        last_left_eye_pose_world,
                        last_right_eye_pose_world,
                        eye_width,
                        eye_height,
                        left_gaussian_rgba,
                        left_gaussian_depth,
                        right_gaussian_rgba,
                        right_gaussian_depth,
                    )
                    if validation_reason is not None:
                        print(
                            "[live_openxr_controller] immersive interaction_rollback "
                            f"sources={validation_sources} reason={validation_reason} "
                            f"debug={validation_debug}",
                            flush=True,
                        )
                        current_target = self._rollback_immersive_grab_start(
                            validation_sources,
                            controller_interaction_state,
                            controller_anchor_preview_state,
                            controller_attachment_metadata,
                            last_valid_sim_state,
                            last_valid_target,
                            gaussians,
                            last_valid_gaussian_state,
                        )
                        x = last_valid_sim_state["x"].clone()
                        current_pos = gaussians.get_xyz
                        current_rot = gaussians.get_rotation
                        object_points = x[: self.num_all_points]
                        object_bounds_min = object_points.min(dim=0).values - 0.01
                        object_bounds_max = object_points.max(dim=0).values + 0.01
                        object_support_center = (
                            self._object_support_patch_center(object_points)
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float32)
                        )
                        (
                            left_scene_color,
                            left_scene_depth,
                            right_scene_color,
                            right_scene_depth,
                        ) = self._render_immersive_scene_frames_for_mode(
                            scene_renderer,
                            active_scene_stereo_mode,
                            layout,
                            object_support_center,
                            object_bounds_min.detach().cpu().numpy().astype(np.float32),
                            object_bounds_max.detach().cpu().numpy().astype(np.float32),
                            last_left_eye_pose_world,
                            last_right_eye_pose_world,
                            left_intrinsic,
                            right_intrinsic,
                            eye_width,
                            eye_height,
                            scene_width,
                            scene_height,
                            shared_scene_compose_cache=shared_scene_compose_cache,
                            reproject_caches=shared_scene_reproject_caches,
                        )
                        (
                            left_eye_frame,
                            left_gaussian_rgba,
                            left_gaussian_depth,
                            _,
                            _,
                        ) = (
                            self._render_immersive_eye_frame(
                                last_left_eye_pose_world,
                                left_intrinsic,
                                eye_height,
                                eye_width,
                                left_scene_color,
                                left_scene_depth,
                                gaussians,
                                render_pipe,
                                background_black,
                                background_white,
                                compose_mode=immersive_compose_mode,
                                compose_roi_padding=gaussian_compose_roi_padding,
                                eye_render_state=left_eye_render_state,
                                output_dtype=eye_frame_output_dtype,
                            )
                        )
                        (
                            right_eye_frame,
                            right_gaussian_rgba,
                            right_gaussian_depth,
                            _,
                            _,
                        ) = (
                            self._render_immersive_eye_frame(
                                last_right_eye_pose_world,
                                right_intrinsic,
                                eye_height,
                                eye_width,
                                right_scene_color,
                                right_scene_depth,
                                gaussians,
                                render_pipe,
                                background_black,
                                background_white,
                                compose_mode=immersive_compose_mode,
                                compose_roi_padding=gaussian_compose_roi_padding,
                                eye_render_state=right_eye_render_state,
                                output_dtype=eye_frame_output_dtype,
                            )
                        )
                    else:
                        for source in validation_sources:
                            interaction_state = controller_interaction_state.get(source)
                            if interaction_state is None:
                                continue
                            remaining = int(
                                interaction_state.get(
                                    "startup_validation_frames_remaining",
                                    0,
                                )
                            )
                            if remaining > 0:
                                interaction_state["startup_validation_frames_remaining"] = (
                                    remaining - 1
                                )
                if grab_validation_start is not None:
                    self._render_profile_add_wall_time(
                        render_profile_frame,
                        "grab_validation_wall",
                        time.perf_counter() - grab_validation_start,
                    )

                if left_eye_frame.dtype != torch.uint8:
                    left_eye_frame = left_eye_frame.clamp(0.0, 255.0).to(torch.uint8)
                if right_eye_frame.dtype != torch.uint8:
                    right_eye_frame = right_eye_frame.clamp(0.0, 255.0).to(torch.uint8)

                if scene_depth_reproject_enabled:
                    publish_measurement_s = time.monotonic()
                    render_pose_staleness_ms = 0.0
                    if render_sample_received_monotonic_s is not None:
                        render_pose_staleness_ms = max(
                            0.0,
                            (publish_measurement_s - render_sample_received_monotonic_s)
                            * 1000.0,
                        )
                    scene_pose_received_monotonic_s = self._sample_received_monotonic_s(
                        scene_pose_sample
                    )
                    if scene_pose_received_monotonic_s is None:
                        scene_pose_staleness_ms_at_publish = render_pose_staleness_ms
                    else:
                        scene_pose_staleness_ms_at_publish = max(
                            0.0,
                            (publish_measurement_s - scene_pose_received_monotonic_s)
                            * 1000.0,
                        )
                    scene_pose_staleness_savings_ms = max(
                        0.0,
                        render_pose_staleness_ms
                        - scene_pose_staleness_ms_at_publish,
                    )
                    if frame_count > 1:
                        scene_pose_staleness_ms_samples.append(
                            float(scene_pose_staleness_ms_at_publish)
                        )
                    if render_profile_frame is not None:
                        render_profile_frame[
                            "scene_pose_staleness_ms_at_publish"
                        ] = scene_pose_staleness_ms_at_publish / 1000.0
                        render_profile_frame[
                            "scene_pose_staleness_savings_ms"
                        ] = scene_pose_staleness_savings_ms / 1000.0

                if not first_real_publish_done:
                    self._record_immersive_startup_milestone(
                        startup_timeline,
                        "first_publish_begin",
                        immersive_bridge,
                    )
                publish_ok, publish_stats = immersive_bridge.publish_stereo_frames(
                    left_eye_frame,
                    right_eye_frame,
                )
                if render_profile_frame is not None:
                    render_profile_frame["publish_total_wall"] = float(
                        publish_stats.get("total_wall", 0.0)
                    )
                    render_profile_frame["publish_process_check_wall"] = float(
                        publish_stats.get("process_check_wall", 0.0)
                    )
                    render_profile_frame["publish_pending_drain_nonblock_wall"] = float(
                        publish_stats.get("pending_drain_nonblock_wall", 0.0)
                    )
                    render_profile_frame["publish_pending_drain_block_wall"] = float(
                        publish_stats.get("pending_drain_block_wall", 0.0)
                    )
                    render_profile_frame["publish_gpu_to_cpu_wait_wall"] = float(
                        publish_stats.get("gpu_to_cpu_wait_wall", 0.0)
                    )
                    render_profile_frame["publish_gpu_to_cpu_copy_cuda"] = float(
                        publish_stats.get("gpu_to_cpu_copy_cuda", 0.0)
                    )
                    render_profile_frame["publish_cpu_mmap_copy_wall"] = float(
                        publish_stats.get("cpu_mmap_copy_wall", 0.0)
                    )
                    render_profile_frame["publish_header_write_wall"] = float(
                        publish_stats.get("header_write_wall", 0.0)
                    )
                    render_profile_frame["publish_stage_enqueue_wall"] = float(
                        publish_stats.get("stage_enqueue_wall", 0.0)
                    )
                    render_profile_frame["publish_fallback_copy_wall"] = float(
                        publish_stats.get("fallback_copy_wall", 0.0)
                    )
                if not publish_ok:
                    failure_details = ""
                    if not first_real_publish_done:
                        failure_details = self._format_immersive_startup_timeline_failure(
                            startup_timeline,
                            immersive_bridge,
                        )
                    raise RuntimeError(
                        "Quest immersive bridge stopped accepting stereo frames.\n"
                        + (failure_details + "\n" if failure_details else "")
                        + immersive_bridge.debug_summary()
                    )
                if not first_real_publish_done:
                    self._record_immersive_startup_milestone(
                        startup_timeline,
                        "first_publish_done",
                        immersive_bridge,
                    )
                    first_real_publish_done = True
                    if startup_keepalive_state is not None:
                        startup_keepalive_state["enabled"] = False
                    startup_gap_ms = startup_timeline.get("startup_gap_ms")
                    if startup_gap_ms is not None:
                        print(
                            "[quest_display] immersive startup first real publish: "
                            f"startup_gap_ms={startup_gap_ms:.1f}",
                            flush=True,
                        )

                if preview_display_active and preview_tex is not None:
                    preview_window_start = (
                        time.perf_counter() if render_profile_frame is not None else None
                    )
                    glfw.make_context_current(window)
                    if preview_uploader is None:
                        raise RuntimeError(
                            "Preview window is active but PreviewTextureCudaUploader was not initialized."
                        )
                    preview_uploader.upload(left_eye_frame)
                    fb_width, fb_height = glfw.get_framebuffer_size(window)
                    gl.glViewport(0, 0, fb_width, fb_height)
                    gl.glDisable(gl.GL_DEPTH_TEST)
                    gl.glClear(gl.GL_COLOR_BUFFER_BIT)
                    gl.glUseProgram(preview_prog)
                    gl.glBindVertexArray(preview_vao)
                    gl.glActiveTexture(gl.GL_TEXTURE0)
                    gl.glBindTexture(gl.GL_TEXTURE_2D, preview_tex)
                    gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)
                    gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
                    gl.glBindVertexArray(0)
                    gl.glUseProgram(0)
                    glfw.swap_buffers(window)
                    if preview_window_start is not None:
                        self._render_profile_add_wall_time(
                            render_profile_frame,
                            "preview_window_wall",
                            time.perf_counter() - preview_window_start,
                        )

                glfw_poll_start = time.perf_counter() if render_profile_frame is not None else None
                glfw.poll_events()
                if glfw_poll_start is not None:
                    self._render_profile_add_wall_time(
                        render_profile_frame,
                        "glfw_poll_wall",
                        time.perf_counter() - glfw_poll_start,
                    )
                render_time = render_timer.stop()
                if frame_count > 1:
                    component_times["rendering"].append(render_time)
                if render_profile_frame is not None:
                    render_profile_frame["rendering"] = float(render_time)
                    render_profile_frame = self._render_profile_finalize_frame(
                        render_profile_frame
                    )

                if (
                    self._is_finite_tensor(x)
                    and self._is_finite_tensor(current_pos)
                    and self._is_finite_tensor(current_rot)
                ):
                    last_valid_sim_state = self._capture_sim_state()
                    last_valid_target = current_target.clone()
                    last_valid_gaussian_state = self._capture_gaussian_runtime_state(
                        gaussians
                    )
                    last_valid_object_center = (
                        x[: self.num_all_points].mean(dim=0).detach().clone()
                    )

                total_time = total_timer.stop()
                if frame_count > 1:
                    component_times["total"].append(total_time)

                if render_profile_frame is not None:
                    self._render_profile_capture_cuda_memory(render_profile_frame)
                if render_profile_frame is not None and frame_count > 1:
                    self._render_profile_append_frame(
                        immersive_render_profile_series,
                        immersive_render_profile_rows,
                        frame_count,
                        render_profile_frame,
                    )
                    if self._render_profile_should_log(frame_count, render_profile_every):
                        self._log_immersive_render_profile_frame(
                            frame_count,
                            render_profile_frame,
                        )

                next_prev_target = current_target.clone()
                next_target = self._compute_next_live_controller_target(
                    controller_runtime_base_target,
                    next_prev_target,
                    controller_interaction_state,
                    controller_interaction_state_cache,
                    controller_source_masks,
                    controller_source_anchor_centers,
                    controller_attachment_metadata,
                    controller_anchor_templates,
                    controller_predefined_anchor_states,
                    controller_anchor_preview_state,
                    controller_overlay_by_source,
                    current_live_left_controller,
                    current_live_right_controller,
                    x[: self.num_all_points],
                    controller_reset_triggered=controller_reset_triggered,
                    post_select_translation_only=live_controller_case_profile[
                        "post_select_translation_only"
                    ],
                    controller_motion_state_cache=controller_motion_state_cache,
                    frame_index=frame_count,
                    runtime_label="immersive",
                )
                frame_count += 1
                prev_target = next_prev_target
                current_target = next_target

                if preview_display_active and glfw.window_should_close(window):
                    break

        except RuntimeError as exc:
            bridge_died_pre_first_publish = (
                immersive_bridge is not None
                and not first_real_publish_done
                and immersive_bridge.process is not None
                and immersive_bridge.process.poll() is not None
            )
            if bridge_died_pre_first_publish:
                failure_details = self._format_immersive_startup_timeline_failure(
                    startup_timeline,
                    immersive_bridge,
                )
                message = str(exc)
                if failure_details and failure_details not in message:
                    message = message + "\n" + failure_details
                raise RuntimeError(message) from exc
            raise
        finally:
            if frame_count > 1 and component_times["total"]:
                frames_used_for_stats = len(component_times["total"])
                summary_header = (
                    f"\n=== Immersive Summary (averaged over {frames_used_for_stats} frames) ==="
                )
                print(summary_header)
                log_lines = [summary_header.lstrip("\n")]
                total_frame_times = component_times["total"]
                total_time_seconds = sum(total_frame_times)
                average_fps = frames_used_for_stats / total_time_seconds
                average_frame_time = np.mean(total_frame_times)
                print(f"Average FPS: {average_fps:.2f}")
                print(f"Average Total Frame Time: {average_frame_time * 1000:.2f} ms")
                log_lines.append(f"Average FPS: {average_fps:.2f}")
                log_lines.append(
                    f"Average Total Frame Time: {average_frame_time * 1000:.2f} ms"
                )
                if scene_pose_staleness_ms_samples:
                    average_scene_pose_staleness_ms = float(
                        np.mean(
                            np.asarray(
                                scene_pose_staleness_ms_samples,
                                dtype=np.float64,
                            )
                        )
                    )
                    scene_staleness_line = (
                        "Average Scene Pose Staleness at Publish: "
                        f"{average_scene_pose_staleness_ms:.2f} ms"
                    )
                    print(scene_staleness_line)
                    log_lines.append(scene_staleness_line)
                if startup_render_debug is not None:
                    startup_compose_line = (
                        "Startup compose mode: "
                        f"{startup_render_debug.get('recommended_compose_mode', 'depth_aware')}"
                    )
                    if startup_render_debug.get("compose_fallback_required", False):
                        startup_compose_line += (
                            f" suppressed={startup_render_debug.get('suppressed_by_scene_depth_eyes', [])}"
                            f" invalid_depth={startup_render_debug.get('scene_depth_invalid_eyes', [])}"
                        )
                    print(startup_compose_line)
                    log_lines.append(startup_compose_line)
                component_summary_rows = []
                for component_name in (
                    "simulator",
                    "full_motion_interpolation",
                    "rendering",
                ):
                    component_times_list = component_times.get(component_name, [])
                    if component_times_list:
                        component_summary_rows.append(
                            (
                                component_name.replace("_", " ").capitalize(),
                                float(np.mean(component_times_list)),
                            )
                        )
                for line in self._format_component_summary_lines(
                    average_frame_time,
                    component_summary_rows,
                ):
                    print(line)
                    log_lines.append(line)
                if render_profile:
                    render_profile_lines = self._render_profile_summary_lines(
                        "immersive",
                        immersive_render_profile_series,
                        immersive_render_profile_summary_keys,
                    )
                    for line in render_profile_lines:
                        print(line)
                    log_lines.extend(render_profile_lines)
                    self._write_render_profile_outputs(
                        diagnostic_output_path,
                        render_profile_lines,
                        immersive_render_profile_rows,
                    )
                if diagnostic_output_path is not None:
                    os.makedirs(diagnostic_output_path, exist_ok=True)
                    with open(
                        os.path.join(diagnostic_output_path, "performance_summary.txt"),
                        "w",
                    ) as log_file:
                        log_file.write("\n".join(log_lines) + "\n")
            if immersive_bridge is not None:
                immersive_bridge.stop()
            if window is not None:
                try:
                    glfw.make_context_current(window)
                except Exception:
                    pass
            if preview_uploader is not None:
                preview_uploader.delete()
            if preview_prog is not None:
                gl.glDeleteProgram(preview_prog)
            if preview_tex is not None:
                gl.glDeleteTextures([preview_tex])
            if preview_vao is not None:
                gl.glDeleteVertexArrays(1, [preview_vao])
            if static_scene_worker is not None:
                static_scene_worker.stop()
            if scene_renderer is not None:
                scene_renderer.delete()
            if cuda_ctx is not None:
                cuda_ctx.pop()

    def interactive_playground_quest_immersive_balanced(
        self,
        model_path,
        gs_path,
        output_dir,
        n_dup=0,
        window=None,
        cuda_ctx=None,
        interactive_window_mode="visible",
        scene_assets_root="./assets/scenes",
        render_profile=False,
        render_profile_every=30,
        immersive_timewarp="off",
        immersive_static_scene_overlap="off",
    ):
        if n_dup != 0:
            raise ValueError(
                "The shipped Quest immersive launcher supports only the single-instance case (--n_dup 0)."
            )
        return self._run_quest_immersive_balanced(
            model_path=model_path,
            gs_path=gs_path,
            output_dir=output_dir,
            window=window,
            cuda_ctx=cuda_ctx,
            interactive_window_mode=interactive_window_mode,
            scene_assets_root=scene_assets_root,
            render_profile=render_profile,
            render_profile_every=render_profile_every,
            immersive_timewarp=immersive_timewarp,
            immersive_static_scene_overlap=immersive_static_scene_overlap,
        )

    def _create_gs_view(
        self,
        w2c,
        intrinsic,
        height,
        width,
    ):
        return self._update_cached_gs_view(
            cache_key=f"adhoc_{int(width)}x{int(height)}",
            camera_pose_world=None,
            w2c=w2c,
            intrinsic=intrinsic,
            height=height,
            width=width,
        )

    def _update_cached_gs_view(
        self,
        cache_key,
        camera_pose_world,
        w2c,
        intrinsic,
        height,
        width,
    ):
        cache_store = getattr(self, "_immersive_gs_view_cache", None)
        if cache_store is None:
            cache_store = {}
            self._immersive_gs_view_cache = cache_store

        entry = cache_store.get(cache_key)
        if (
            entry is None
            or int(getattr(entry, "image_width", -1)) != int(width)
            or int(getattr(entry, "image_height", -1)) != int(height)
        ):
            entry = SimpleNamespace(
                image_width=int(width),
                image_height=int(height),
                world_view_transform=torch.empty(
                    (4, 4),
                    dtype=torch.float32,
                    device=cfg.device,
                ),
                K=torch.empty(
                    (3, 3),
                    dtype=torch.float32,
                    device=cfg.device,
                ),
                camera_center=torch.empty(
                    (3,),
                    dtype=torch.float32,
                    device=cfg.device,
                ),
                FoVx=0.0,
                FoVy=0.0,
                image_name="0000",
                uid=str(cache_key),
                colmap_id=str(cache_key),
            )
            cache_store[cache_key] = entry

        w2c_t = torch.as_tensor(
            w2c,
            dtype=torch.float32,
            device=cfg.device,
        )
        intrinsic_t = torch.as_tensor(
            intrinsic,
            dtype=torch.float32,
            device=cfg.device,
        )
        entry.image_width = int(width)
        entry.image_height = int(height)
        entry.world_view_transform.copy_(w2c_t.transpose(0, 1))
        entry.K.copy_(intrinsic_t)
        if camera_pose_world is None:
            entry.camera_center.copy_(torch.linalg.inv(w2c_t)[:3, 3])
        else:
            entry.camera_center.copy_(
                torch.as_tensor(
                    np.asarray(camera_pose_world[:3, 3], dtype=np.float32),
                    dtype=torch.float32,
                    device=cfg.device,
                )
            )
        return entry, entry.K

    def _prepare_immersive_eye_render_state(
        self,
        eye_pose_world,
        intrinsic,
        height,
        width,
        eye_label=None,
    ):
        eye_w2c_cv = self._camera_pose_world_to_cv_w2c(eye_pose_world)
        intrinsic_t = torch.as_tensor(
            intrinsic,
            dtype=torch.float32,
            device=cfg.device,
        )
        eye_w2c_cv_t = torch.as_tensor(
            eye_w2c_cv,
            dtype=torch.float32,
            device=cfg.device,
        )
        eye_view, _ = self._update_cached_gs_view(
            cache_key=f"immersive_{eye_label or 'default'}",
            camera_pose_world=eye_pose_world,
            w2c=eye_w2c_cv,
            intrinsic=intrinsic,
            height=height,
            width=width,
        )
        return {
            "view": eye_view,
            "intrinsic_t": intrinsic_t,
            "w2c_cv_t": eye_w2c_cv_t,
        }

    def _resolve_composited_frame_resolution(
        self,
        base_width,
        base_height,
        quest_display_mode,
    ):
        width = int(base_width)
        height = int(base_height)
        if quest_display_mode != "primary":
            return width, height

        target_width = max(width, int(self.QUEST_PRIMARY_COMPOSITE_WIDTH))
        target_height = int(round(height * (target_width / float(width))))
        if target_height % 2 != 0:
            target_height += 1
        return target_width, target_height

    def _scale_intrinsic_for_resolution(
        self,
        intrinsic,
        base_width,
        base_height,
        target_width,
        target_height,
    ):
        scaled = np.array(intrinsic, dtype=np.float32, copy=True)
        scale_x = float(target_width) / float(base_width)
        scale_y = float(target_height) / float(base_height)
        scaled[0, :] *= scale_x
        scaled[1, :] *= scale_y
        scaled[2, :] = np.array(intrinsic, dtype=np.float32, copy=False)[2, :]
        return scaled

    def _resolve_immersive_render_options(
        self,
        immersive_render_preset="quality",
        immersive_scene_render_scale=None,
        immersive_scene_stereo_mode=None,
        immersive_overlay_mode=None,
    ):
        preset = str(immersive_render_preset or "quality")
        defaults = self.IMMERSIVE_RENDER_PRESET_DEFAULTS.get(preset)
        if defaults is None:
            raise ValueError(f"Unsupported immersive_render_preset: {preset}")

        if immersive_scene_render_scale is None:
            scene_render_scale = float(defaults["scene_render_scale"])
        else:
            scene_render_scale = float(immersive_scene_render_scale)
        if not np.isfinite(scene_render_scale) or scene_render_scale <= 0.0:
            raise ValueError(
                f"immersive_scene_render_scale must be finite and > 0, got {immersive_scene_render_scale}"
            )
        scene_render_scale = max(
            self.IMMERSIVE_SCENE_RENDER_SCALE_MIN,
            min(scene_render_scale, 1.0),
        )

        scene_stereo_mode = (
            str(immersive_scene_stereo_mode)
            if immersive_scene_stereo_mode is not None
            else str(defaults["scene_stereo_mode"])
        )
        if scene_stereo_mode not in {
            "per_eye",
            "mono_head_center",
            "reproject_from_center",
            self.IMMERSIVE_BALANCED_INTERNAL_STEREO_MODE,
        }:
            raise ValueError(
                f"Unsupported immersive_scene_stereo_mode: {scene_stereo_mode}"
            )

        overlay_mode = (
            str(immersive_overlay_mode)
            if immersive_overlay_mode is not None
            else str(defaults["overlay_mode"])
        )
        if overlay_mode not in {"full", "minimal"}:
            raise ValueError(f"Unsupported immersive_overlay_mode: {overlay_mode}")

        return {
            "preset": preset,
            "scene_render_scale": scene_render_scale,
            "scene_stereo_mode": scene_stereo_mode,
            "overlay_mode": overlay_mode,
            "lighting_mode": str(defaults["lighting_mode"]),
        }

    def _resolve_immersive_scene_resolution(
        self,
        eye_width,
        eye_height,
        scene_render_scale,
    ):
        scene_width = max(1, int(round(float(eye_width) * float(scene_render_scale))))
        scene_height = max(
            1,
            int(round(float(eye_height) * float(scene_render_scale))),
        )
        if scene_width % 2 != 0:
            scene_width += 1
        if scene_height % 2 != 0:
            scene_height += 1
        return scene_width, scene_height

    def _orthonormalize_rotation_matrix(self, rotation_matrix):
        rotation_matrix = np.asarray(rotation_matrix, dtype=np.float32)
        u, _, vh = np.linalg.svd(rotation_matrix)
        ortho = u @ vh
        if np.linalg.det(ortho) < 0.0:
            u[:, -1] *= -1.0
            ortho = u @ vh
        return ortho.astype(np.float32)

    def _build_immersive_center_scene_view(
        self,
        left_eye_pose_world,
        right_eye_pose_world,
        left_intrinsic,
        right_intrinsic,
    ):
        valid_poses = [
            pose
            for pose in (left_eye_pose_world, right_eye_pose_world)
            if pose is not None
        ]
        valid_intrinsics = [
            intrinsic
            for intrinsic in (left_intrinsic, right_intrinsic)
            if intrinsic is not None
        ]
        if not valid_poses or not valid_intrinsics:
            raise ValueError("Immersive center scene view requires at least one valid eye")

        center_pose = np.eye(4, dtype=np.float32)
        center_pose[:3, :3] = self._orthonormalize_rotation_matrix(
            np.mean(
                np.stack([pose[:3, :3] for pose in valid_poses], axis=0),
                axis=0,
            )
        )
        center_pose[:3, 3] = np.mean(
            np.stack([pose[:3, 3] for pose in valid_poses], axis=0),
            axis=0,
        ).astype(np.float32)

        center_intrinsic = np.mean(
            np.stack(valid_intrinsics, axis=0),
            axis=0,
        ).astype(np.float32)
        center_intrinsic[2, :] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        return center_pose, center_intrinsic

    def _ensure_immersive_compose_cache(
        self,
        compose_cache,
        target_height,
        target_width,
    ):
        if compose_cache is None:
            return None

        target_shape = (int(target_height), int(target_width))
        if compose_cache.get("target_shape") == target_shape:
            return compose_cache

        height, width = target_shape
        compose_cache.clear()
        compose_cache["target_shape"] = target_shape
        compose_cache["composed_color"] = torch.empty(
            (height, width, 4),
            dtype=torch.float32,
            device=cfg.device,
        )
        compose_cache["composed_depth"] = torch.empty(
            (height, width),
            dtype=torch.float32,
            device=cfg.device,
        )
        return compose_cache

    def _prepare_immersive_scene_tensor_for_compose(
        self,
        scene_color_rgba,
        scene_depth,
        target_height,
        target_width,
    ):
        if not torch.is_tensor(scene_color_rgba) or not torch.is_tensor(scene_depth):
            raise TypeError(
                "Immersive scene compose expects tensor scene/background/table inputs. "
                "CPU/numpy scene staging has been removed from the shipped Quest runtime."
            )
        target_device = torch.device(cfg.device)
        scene_color_t = scene_color_rgba
        scene_depth_t = scene_depth
        if scene_color_t.device != target_device or scene_color_t.dtype != torch.float32:
            scene_color_t = scene_color_t.to(device=target_device, dtype=torch.float32)
        if scene_depth_t.device != target_device or scene_depth_t.dtype != torch.float32:
            scene_depth_t = scene_depth_t.to(device=target_device, dtype=torch.float32)
        if scene_color_t.shape[:2] != (target_height, target_width):
            scene_color_t = (
                F.interpolate(
                    scene_color_t.permute(2, 0, 1).unsqueeze(0),
                    size=(target_height, target_width),
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze(0)
                .permute(1, 2, 0)
                .contiguous()
            )
        elif not scene_color_t.is_contiguous():
            scene_color_t = scene_color_t.contiguous()
        if scene_depth_t.shape[:2] != (target_height, target_width):
            scene_depth_t = (
                F.interpolate(
                    scene_depth_t.unsqueeze(0).unsqueeze(0),
                    size=(target_height, target_width),
                    mode="nearest",
                )
                .squeeze(0)
                .squeeze(0)
                .contiguous()
            )
        elif not scene_depth_t.is_contiguous():
            scene_depth_t = scene_depth_t.contiguous()
        return scene_color_t, scene_depth_t

    def _prepare_immersive_scene_frame_for_compose(
        self,
        scene_color_rgba,
        scene_depth,
        target_height,
        target_width,
        compose_cache=None,
    ):
        target_height = int(target_height)
        target_width = int(target_width)
        _ = compose_cache
        return self._prepare_immersive_scene_tensor_for_compose(
            scene_color_rgba,
            scene_depth,
            target_height,
            target_width,
        )

    def _ensure_immersive_reproject_cache(
        self,
        reproject_cache,
        source_height,
        source_width,
        target_height,
        target_width,
    ):
        if reproject_cache is None:
            return None

        cache_shape = (
            int(source_height),
            int(source_width),
            int(target_height),
            int(target_width),
        )
        if reproject_cache.get("cache_shape") != cache_shape:
            source_height_i, source_width_i, target_height_i, target_width_i = cache_shape
            reproject_cache.clear()
            reproject_cache["cache_shape"] = cache_shape
            target_pixels = target_height_i * target_width_i
            reproject_cache["target_min_depth_flat"] = torch.empty(
                target_pixels,
                dtype=torch.float32,
                device=cfg.device,
            )
            reproject_cache["target_source_index_flat"] = torch.empty(
                target_pixels,
                dtype=torch.long,
                device=cfg.device,
            )
            reproject_cache["target_valid_flat"] = torch.empty(
                target_pixels,
                dtype=torch.bool,
                device=cfg.device,
            )
            reproject_cache["target_color"] = torch.empty(
                (target_height_i, target_width_i, 4),
                dtype=torch.float32,
                device=cfg.device,
            )
            reproject_cache["target_depth"] = torch.empty(
                (target_height_i, target_width_i),
                dtype=torch.float32,
                device=cfg.device,
            )
            reproject_cache["target_valid"] = torch.empty(
                (target_height_i, target_width_i),
                dtype=torch.bool,
                device=cfg.device,
            )
            reproject_cache["near_offsets"] = torch.tensor(
                [[0, 0], [1, 0], [0, 1], [1, 1]],
                dtype=torch.long,
                device=cfg.device,
            )
            reproject_cache["far_offsets"] = torch.tensor(
                [
                    [-1, -1], [0, -1], [1, -1],
                    [-1, 0], [0, 0], [1, 0],
                    [-1, 1], [0, 1], [1, 1],
                ],
                dtype=torch.long,
                device=cfg.device,
            )

        return reproject_cache

    def _ensure_immersive_reproject_source_cache(
        self,
        source_cache,
        source_height,
        source_width,
    ):
        if source_cache is None:
            return None

        cache_shape = (int(source_height), int(source_width))
        if source_cache.get("cache_shape") != cache_shape:
            source_height_i, source_width_i = cache_shape
            source_cache.clear()
            source_cache["cache_shape"] = cache_shape
            ys = torch.arange(source_height_i, device=cfg.device, dtype=torch.float32)
            xs = torch.arange(source_width_i, device=cfg.device, dtype=torch.float32)
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
            source_cache["grid_x_flat"] = grid_x.reshape(-1).contiguous()
            source_cache["grid_y_flat"] = grid_y.reshape(-1).contiguous()
            source_cache["source_intrinsic_key"] = None
            source_cache["ray_x_flat"] = None
            source_cache["ray_y_flat"] = None
        return source_cache

    def _object_bounds_corners(self, bounds_min, bounds_max):
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

    def _tabletop_footprint_points(self, layout):
        half_x = 0.5 * float(layout.table_size[0])
        half_y = 0.5 * float(layout.table_size[1])
        center = np.asarray(layout.table_top_center, dtype=np.float32)
        z = float(center[2])
        return np.array(
            [
                center,
                [center[0] - half_x, center[1] - half_y, z],
                [center[0] - half_x, center[1] + half_y, z],
                [center[0] + half_x, center[1] - half_y, z],
                [center[0] + half_x, center[1] + half_y, z],
            ],
            dtype=np.float32,
        )

    def _compute_immersive_reproject_roi_bounds(
        self,
        layout,
        object_support_center,
        object_bounds_min,
        object_bounds_max,
        intrinsic,
        w2c,
        width,
        height,
        padding=None,
    ):
        if padding is None:
            padding = int(self.IMMERSIVE_REPROJECT_ROI_PADDING)
        world_points = [self._tabletop_footprint_points(layout)]
        if object_bounds_min is not None and object_bounds_max is not None:
            world_points.append(
                self._object_bounds_corners(object_bounds_min, object_bounds_max)
            )
        if object_support_center is not None:
            world_points.append(
                np.asarray(object_support_center, dtype=np.float32).reshape(1, 3)
            )

        projected_pixels = []
        for points in world_points:
            for point in np.asarray(points, dtype=np.float32):
                projection = self._project_world_point_into_eye(
                    point,
                    intrinsic,
                    w2c,
                    width,
                    height,
                )
                pixel = projection["pixel"]
                if pixel is None or projection["depth"] is None:
                    continue
                if projection["depth"] <= self.IMMERSIVE_STARTUP_DEPTH_EPS:
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
        x0 = max(0, int(np.floor(np.min(pixels_np[:, 0]))) - padding)
        x1 = min(int(width), int(np.ceil(np.max(pixels_np[:, 0]))) + padding + 1)
        y0 = max(0, int(np.floor(np.min(pixels_np[:, 1]))) - padding)
        y1 = min(int(height), int(np.ceil(np.max(pixels_np[:, 1]))) + padding + 1)
        if x1 <= x0 or y1 <= y0:
            return None
        return (x0, y0, x1, y1)

    def _compute_immersive_projected_roi_bounds(
        self,
        bounds_min,
        bounds_max,
        intrinsic,
        w2c,
        width,
        height,
        padding=None,
        snap=None,
        min_size=None,
    ):
        if bounds_min is None or bounds_max is None:
            return None
        if padding is None:
            padding = 0
        if snap is None:
            snap = 1
        if min_size is None:
            min_size = 1

        projected_pixels = []
        for point in self._object_bounds_corners(bounds_min, bounds_max):
            projection = self._project_world_point_into_eye(
                point,
                intrinsic,
                w2c,
                width,
                height,
            )
            pixel = projection["pixel"]
            if pixel is None or projection["depth"] is None:
                continue
            if projection["depth"] <= self.IMMERSIVE_STARTUP_DEPTH_EPS:
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

        def _expand_axis(lo, hi, limit):
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

    def _roi_area_ratio(self, roi_bounds, width, height):
        if roi_bounds is None:
            return 0.0
        x0, y0, x1, y1 = [int(v) for v in roi_bounds]
        if x1 <= x0 or y1 <= y0:
            return 0.0
        return float(
            ((x1 - x0) * (y1 - y0))
            / max(float(int(width) * int(height)), 1.0)
        )

    def _resolve_immersive_render_roi(
        self,
        bounds_min,
        bounds_max,
        intrinsic,
        w2c,
        width,
        height,
        *,
        padding,
        snap,
        min_size,
        fullframe_threshold,
    ):
        roi_bounds = self._compute_immersive_projected_roi_bounds(
            bounds_min,
            bounds_max,
            intrinsic,
            w2c,
            width,
            height,
            padding=padding,
            snap=snap,
            min_size=min_size,
        )
        if roi_bounds is None:
            return None, 1.0, True
        roi_ratio = self._roi_area_ratio(roi_bounds, width, height)
        if roi_ratio > float(fullframe_threshold):
            return None, 1.0, True
        return roi_bounds, roi_ratio, False

    def _resolve_immersive_table_render_roi(
        self,
        table_bounds_min,
        table_bounds_max,
        intrinsic,
        w2c,
        width,
        height,
    ):
        return self._resolve_immersive_render_roi(
            table_bounds_min,
            table_bounds_max,
            intrinsic,
            w2c,
            width,
            height,
            padding=int(self.IMMERSIVE_TABLE_ROI_PADDING),
            snap=int(self.IMMERSIVE_TABLE_ROI_SNAP),
            min_size=int(self.IMMERSIVE_TABLE_ROI_MIN_SIZE),
            fullframe_threshold=float(self.IMMERSIVE_TABLE_ROI_FULLFRAME_THRESHOLD),
        )

    def _resolve_immersive_side_wall_render_roi(
        self,
        wall_bounds_min,
        wall_bounds_max,
        intrinsic,
        w2c,
        width,
        height,
    ):
        return self._resolve_immersive_render_roi(
            wall_bounds_min,
            wall_bounds_max,
            intrinsic,
            w2c,
            width,
            height,
            padding=int(self.IMMERSIVE_SIDE_WALL_ROI_PADDING),
            snap=int(self.IMMERSIVE_SIDE_WALL_ROI_SNAP),
            min_size=int(self.IMMERSIVE_SIDE_WALL_ROI_MIN_SIZE),
            fullframe_threshold=float(
                self.IMMERSIVE_SIDE_WALL_ROI_FULLFRAME_THRESHOLD
            ),
        )

    def _apply_immersive_roi_hysteresis(
        self,
        prev_bounds,
        new_bounds,
        width,
        height,
        shrink_limit_px,
    ):
        if new_bounds is None:
            return None
        x0, y0, x1, y1 = [int(v) for v in new_bounds]
        x0 = max(0, min(int(width), x0))
        x1 = max(0, min(int(width), x1))
        y0 = max(0, min(int(height), y0))
        y1 = max(0, min(int(height), y1))
        if prev_bounds is None:
            return (x0, y0, x1, y1)
        px0, py0, px1, py1 = [int(v) for v in prev_bounds]
        shrink_limit_px = max(int(shrink_limit_px), 0)
        if x0 > px0:
            x0 = min(x0, px0 + shrink_limit_px)
        if y0 > py0:
            y0 = min(y0, py0 + shrink_limit_px)
        if x1 < px1:
            x1 = max(x1, px1 - shrink_limit_px)
        if y1 < py1:
            y1 = max(y1, py1 - shrink_limit_px)
        x0 = max(0, min(int(width), x0))
        x1 = max(0, min(int(width), x1))
        y0 = max(0, min(int(height), y0))
        y1 = max(0, min(int(height), y1))
        if x1 <= x0 or y1 <= y0:
            return new_bounds
        return (int(x0), int(y0), int(x1), int(y1))

    def _resolve_immersive_balanced_table_render_roi(
        self,
        table_bounds_min,
        table_bounds_max,
        intrinsic,
        w2c,
        width,
        height,
        prev_bounds=None,
    ):
        roi_bounds, roi_ratio, fullframe_fallback = self._resolve_immersive_render_roi(
            table_bounds_min,
            table_bounds_max,
            intrinsic,
            w2c,
            width,
            height,
            padding=int(self.IMMERSIVE_BALANCED_TABLE_ROI_PADDING),
            snap=int(self.IMMERSIVE_BALANCED_TABLE_ROI_SNAP),
            min_size=int(self.IMMERSIVE_BALANCED_TABLE_ROI_MIN_SIZE),
            fullframe_threshold=float(self.IMMERSIVE_TABLE_ROI_FULLFRAME_THRESHOLD),
        )
        hysteresis_bounds = None
        if not fullframe_fallback and roi_bounds is not None:
            hysteresis_bounds = self._apply_immersive_roi_hysteresis(
                prev_bounds,
                roi_bounds,
                width,
                height,
                shrink_limit_px=int(self.IMMERSIVE_BALANCED_TABLE_ROI_SHRINK_MAX_PX),
            )
            roi_bounds = hysteresis_bounds
            roi_ratio = self._roi_area_ratio(roi_bounds, width, height)
            if roi_ratio > float(self.IMMERSIVE_TABLE_ROI_FULLFRAME_THRESHOLD):
                roi_bounds = None
                roi_ratio = 1.0
                fullframe_fallback = True
                hysteresis_bounds = None
        return roi_bounds, roi_ratio, fullframe_fallback, hysteresis_bounds

    def _resolve_immersive_balanced_side_wall_strip_roi(
        self,
        wall_bounds_min,
        wall_bounds_max,
        intrinsic,
        w2c,
        width,
        height,
        prev_bounds=None,
    ):
        if wall_bounds_min is None or wall_bounds_max is None:
            return None, 0.0, False, {
                "anchor_edge": None,
                "strip_width_ratio": 0.0,
                "hysteresis_bounds": None,
            }
        padding = int(self.IMMERSIVE_BALANCED_SIDE_WALL_STRIP_PADDING)
        snap = max(int(self.IMMERSIVE_BALANCED_SIDE_WALL_STRIP_SNAP), 1)
        min_width = max(int(self.IMMERSIVE_BALANCED_SIDE_WALL_STRIP_MIN_WIDTH), 1)
        projected_pixels = []
        for point in self._object_bounds_corners(wall_bounds_min, wall_bounds_max):
            projection = self._project_world_point_into_eye(
                point,
                intrinsic,
                w2c,
                width,
                height,
            )
            pixel = projection["pixel"]
            if pixel is None or projection["depth"] is None:
                continue
            if projection["depth"] <= self.IMMERSIVE_STARTUP_DEPTH_EPS:
                continue
            projected_pixels.append(float(np.clip(pixel[0], 0.0, float(width - 1))))
        wall_center = 0.5 * (
            np.asarray(wall_bounds_min, dtype=np.float32)
            + np.asarray(wall_bounds_max, dtype=np.float32)
        )
        center_projection = self._project_world_point_into_eye(
            wall_center,
            intrinsic,
            w2c,
            width,
            height,
        )
        center_pixel = center_projection.get("pixel")
        if not projected_pixels or center_pixel is None or center_projection.get("depth") is None:
            return None, 0.0, False, {
                "anchor_edge": None,
                "strip_width_ratio": 0.0,
                "hysteresis_bounds": None,
            }
        if float(center_projection["depth"]) <= self.IMMERSIVE_STARTUP_DEPTH_EPS:
            return None, 0.0, False, {
                "anchor_edge": None,
                "strip_width_ratio": 0.0,
                "hysteresis_bounds": None,
            }
        min_x = int(np.floor(min(projected_pixels))) - padding
        max_x = int(np.ceil(max(projected_pixels))) + padding + 1
        frame_center_x = 0.5 * float(width)
        projected_center_x = float(center_pixel[0])
        anchor_edge = "left" if projected_center_x < frame_center_x else "right"
        if anchor_edge == "left":
            x0 = 0
            x1 = max(max_x, min_width)
            x1 = int(np.ceil(float(x1) / float(snap))) * snap
            x1 = min(int(width), x1)
        else:
            x1 = int(width)
            x0 = min(min_x, int(width) - min_width)
            x0 = int(np.floor(float(x0) / float(snap))) * snap
            x0 = max(0, x0)
        if (x1 - x0) < min_width:
            if anchor_edge == "left":
                x1 = min(int(width), int(np.ceil(float(min_width) / float(snap))) * snap)
            else:
                x0 = max(0, int(width) - int(np.ceil(float(min_width) / float(snap))) * snap)
        roi_bounds = (int(x0), 0, int(x1), int(height))
        hysteresis_bounds = self._apply_immersive_roi_hysteresis(
            prev_bounds,
            roi_bounds,
            width,
            height,
            shrink_limit_px=int(self.IMMERSIVE_BALANCED_SIDE_WALL_STRIP_SHRINK_MAX_PX),
        )
        roi_bounds = hysteresis_bounds
        strip_width_ratio = float(
            (roi_bounds[2] - roi_bounds[0]) / max(float(width), 1.0)
        )
        fullframe_fallback = (
            strip_width_ratio
            > float(self.IMMERSIVE_BALANCED_SIDE_WALL_STRIP_FULLFRAME_WIDTH_RATIO)
        )
        return roi_bounds, strip_width_ratio, fullframe_fallback, {
            "anchor_edge": anchor_edge,
            "strip_width_ratio": strip_width_ratio,
            "hysteresis_bounds": roi_bounds,
        }

    def _resolve_immersive_balanced_background_resolution(
        self,
        eye_width,
        eye_height,
    ):
        render_scale = float(self.IMMERSIVE_BALANCED_BACKGROUND_RENDER_SCALE) * float(
            self.IMMERSIVE_BALANCED_BACKGROUND_OVERSCAN
        )
        background_width = max(1, int(round(float(eye_width) * render_scale)))
        background_height = max(1, int(round(float(eye_height) * render_scale)))
        if background_width % 2 != 0:
            background_width += 1
        if background_height % 2 != 0:
            background_height += 1
        return background_width, background_height

    def _build_immersive_balanced_background_intrinsic(
        self,
        center_intrinsic,
        eye_width,
        eye_height,
    ):
        background_width, background_height = (
            self._resolve_immersive_balanced_background_resolution(
                eye_width,
                eye_height,
            )
        )
        background_intrinsic = self._scale_intrinsic_for_resolution(
            center_intrinsic,
            eye_width,
            eye_height,
            background_width,
            background_height,
        )
        background_intrinsic = np.array(background_intrinsic, dtype=np.float32, copy=True)
        overscan = float(self.IMMERSIVE_BALANCED_BACKGROUND_OVERSCAN)
        background_intrinsic[0, 0] /= overscan
        background_intrinsic[1, 1] /= overscan
        return background_intrinsic, background_width, background_height

    def _compute_immersive_balanced_background_reference_depth(
        self,
        layout,
        center_eye_pose_world,
    ):
        table_top_center = np.asarray(layout.table_top_center, dtype=np.float32)
        table_top_center_h = np.concatenate(
            [table_top_center, np.array([1.0], dtype=np.float32)],
            axis=0,
        )
        center_w2c_cv = self._camera_pose_world_to_cv_w2c(center_eye_pose_world)
        reference_depth_m = float((center_w2c_cv @ table_top_center_h)[2])
        if not np.isfinite(reference_depth_m):
            reference_depth_m = 1.2
        reference_depth_m = min(
            float(self.IMMERSIVE_BALANCED_REFERENCE_DEPTH_MAX_M),
            max(
                float(self.IMMERSIVE_BALANCED_REFERENCE_DEPTH_MIN_M),
                reference_depth_m,
            ),
        )
        return reference_depth_m

    def _compute_immersive_balanced_far_wall_center_world(
        self,
        layout,
        center_eye_pose_world,
    ):
        eye_position = np.asarray(center_eye_pose_world[:3, 3], dtype=np.float32)
        forward_world = self._eye_forward_world(center_eye_pose_world).astype(np.float32)
        forward_xy = np.array([forward_world[0], forward_world[1]], dtype=np.float32)
        forward_xy_norm = float(np.linalg.norm(forward_xy))
        if forward_xy_norm < 1e-6:
            forward_xy = np.array([0.0, 1.0], dtype=np.float32)
        else:
            forward_xy /= forward_xy_norm

        room_center_xy = (
            np.asarray(layout.room_center_xy, dtype=np.float32)
            if getattr(layout, "room_center_xy", None) is not None
            else np.array(
                [layout.table_top_center[0], layout.table_top_center[1]],
                dtype=np.float32,
            )
        )
        room_mins_xy = room_center_xy - np.asarray(layout.room_half_extent, dtype=np.float32)
        room_maxs_xy = room_center_xy + np.asarray(layout.room_half_extent, dtype=np.float32)
        ray_origin_xy = np.array([eye_position[0], eye_position[1]], dtype=np.float32)

        candidate_ts = []
        if abs(float(forward_xy[0])) > 1e-6:
            target_x = room_maxs_xy[0] if float(forward_xy[0]) >= 0.0 else room_mins_xy[0]
            t_x = float((target_x - ray_origin_xy[0]) / forward_xy[0])
            if t_x > 0.0:
                candidate_ts.append(t_x)
        if abs(float(forward_xy[1])) > 1e-6:
            target_y = room_maxs_xy[1] if float(forward_xy[1]) >= 0.0 else room_mins_xy[1]
            t_y = float((target_y - ray_origin_xy[1]) / forward_xy[1])
            if t_y > 0.0:
                candidate_ts.append(t_y)

        if candidate_ts:
            wall_xy = ray_origin_xy + forward_xy * min(candidate_ts)
        else:
            wall_xy = np.array(
                [room_center_xy[0], room_maxs_xy[1]],
                dtype=np.float32,
            )
        wall_xy[0] = np.clip(wall_xy[0], room_mins_xy[0], room_maxs_xy[0])
        wall_xy[1] = np.clip(wall_xy[1], room_mins_xy[1], room_maxs_xy[1])
        wall_z = float(layout.floor_z - 0.5 * layout.wall_height)
        return np.array([wall_xy[0], wall_xy[1], wall_z], dtype=np.float32)

    def _compute_immersive_balanced_far_reference_depth(
        self,
        layout,
        center_eye_pose_world,
    ):
        far_wall_center = self._compute_immersive_balanced_far_wall_center_world(
            layout,
            center_eye_pose_world,
        )
        far_wall_center_h = np.concatenate(
            [far_wall_center, np.array([1.0], dtype=np.float32)],
            axis=0,
        )
        center_w2c_cv = self._camera_pose_world_to_cv_w2c(center_eye_pose_world)
        reference_depth_m = float((center_w2c_cv @ far_wall_center_h)[2])
        if not np.isfinite(reference_depth_m):
            reference_depth_m = 2.6
        reference_depth_m = min(
            float(self.IMMERSIVE_BALANCED_FAR_REFERENCE_DEPTH_MAX_M),
            max(
                float(self.IMMERSIVE_BALANCED_FAR_REFERENCE_DEPTH_MIN_M),
                reference_depth_m,
            ),
        )
        return reference_depth_m, far_wall_center

    def _compute_immersive_balanced_background_shift(
        self,
        center_eye_pose_world,
        target_eye_pose_world,
        source_intrinsic,
        target_intrinsic,
        reference_depth_m,
    ):
        if center_eye_pose_world is None or target_eye_pose_world is None:
            return 0.0, 0.0, np.zeros(3, dtype=np.float32)
        center_w2c_cv = self._camera_pose_world_to_cv_w2c(center_eye_pose_world)
        delta_world = (
            np.asarray(target_eye_pose_world[:3, 3], dtype=np.float32)
            - np.asarray(center_eye_pose_world[:3, 3], dtype=np.float32)
        )
        delta_cam = center_w2c_cv[:3, :3] @ delta_world
        reference_depth = max(float(reference_depth_m), 1e-4)
        dx_px = -(
            float(source_intrinsic[0, 0]) * float(delta_cam[0]) / reference_depth
        ) + (float(target_intrinsic[0, 2]) - float(source_intrinsic[0, 2]))
        dy_px = -(
            float(source_intrinsic[1, 1]) * float(delta_cam[1]) / reference_depth
        ) + (float(target_intrinsic[1, 2]) - float(source_intrinsic[1, 2]))
        return float(dx_px), float(dy_px), delta_cam.astype(np.float32)

    def _prepare_immersive_balanced_runtime_state(
        self,
        layout,
        left_eye_pose_world,
        right_eye_pose_world,
        left_intrinsic,
        right_intrinsic,
        eye_width,
        eye_height,
        scene_width,
        scene_height,
    ):
        center_eye_pose_world, center_intrinsic = self._build_immersive_center_scene_view(
            left_eye_pose_world,
            right_eye_pose_world,
            left_intrinsic,
            right_intrinsic,
        )
        background_intrinsic = self._scale_intrinsic_for_resolution(
            center_intrinsic,
            eye_width,
            eye_height,
            scene_width,
            scene_height,
        )
        near_reference_depth_m = self._compute_immersive_balanced_background_reference_depth(
            layout,
            center_eye_pose_world,
        )
        far_reference_depth_m, far_wall_center_world = (
            self._compute_immersive_balanced_far_reference_depth(
                layout,
                center_eye_pose_world,
            )
        )
        runtime_state = {
            "background_mode": "per_eye_background",
            "side_wall_mode": "disabled",
            "table_mode": "roi_per_eye",
            "near_reference_depth_m": float(near_reference_depth_m),
            "far_reference_depth_m": float(far_reference_depth_m),
            "far_wall_center_world": far_wall_center_world.astype(np.float32).tolist(),
            "background_width": int(scene_width),
            "background_height": int(scene_height),
            "background_intrinsic": background_intrinsic.astype(np.float32).copy(),
            "background_compose_caches": {"left": {}, "right": {}},
            "table_roi_render_scale": float(
                self.IMMERSIVE_BALANCED_TABLE_ROI_SUPERSAMPLE_SCALE
            ),
            "side_wall_roi_state": {
                "left": {"left": None, "right": None},
                "right": {"left": None, "right": None},
            },
            "table_roi_state": {"left": None, "right": None},
            "startup_shifts_px": {
                "near": {
                    "left": {"dx": 0.0, "dy": 0.0},
                    "right": {"dx": 0.0, "dy": 0.0},
                },
                "far": {
                    "left": {"dx": 0.0, "dy": 0.0},
                    "right": {"dx": 0.0, "dy": 0.0},
                },
            },
        }
        return runtime_state

    def _scene_valid_roi_coverage(self, valid_mask, roi_bounds):
        if valid_mask is None or roi_bounds is None:
            return 0.0
        x0, y0, x1, y1 = [int(v) for v in roi_bounds]
        if x1 <= x0 or y1 <= y0:
            return 0.0
        patch = valid_mask[y0:y1, x0:x1]
        if patch.numel() <= 0:
            return 0.0
        return float(patch.to(dtype=torch.float32).mean().item())

    def _prepare_immersive_reproject_source_data(
        self,
        source_color,
        source_depth,
        source_intrinsic_t,
        source_cache=None,
    ):
        device = source_color.device
        dtype = torch.float32
        source_color_t = source_color.to(device=device, dtype=dtype)
        source_depth_t = source_depth.to(device=device, dtype=dtype)
        source_height = int(source_depth_t.shape[0])
        source_width = int(source_depth_t.shape[1])
        source_cache = self._ensure_immersive_reproject_source_cache(
            source_cache,
            source_height,
            source_width,
        )
        if source_cache is not None:
            intrinsic_key = (
                round(float(source_intrinsic_t[0, 0].item()), 5),
                round(float(source_intrinsic_t[1, 1].item()), 5),
                round(float(source_intrinsic_t[0, 2].item()), 5),
                round(float(source_intrinsic_t[1, 2].item()), 5),
            )
            if source_cache.get("source_intrinsic_key") != intrinsic_key:
                source_cache["source_intrinsic_key"] = intrinsic_key
                source_cache["ray_x_flat"] = (
                    (source_cache["grid_x_flat"] - float(source_intrinsic_t[0, 2].item()))
                    / float(source_intrinsic_t[0, 0].item())
                ).contiguous()
                source_cache["ray_y_flat"] = (
                    (source_cache["grid_y_flat"] - float(source_intrinsic_t[1, 2].item()))
                    / float(source_intrinsic_t[1, 1].item())
                ).contiguous()
            ray_x_flat = source_cache["ray_x_flat"]
            ray_y_flat = source_cache["ray_y_flat"]
        else:
            ys = torch.arange(source_height, device=device, dtype=dtype)
            xs = torch.arange(source_width, device=device, dtype=dtype)
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
            ray_x_flat = (
                (grid_x.reshape(-1) - float(source_intrinsic_t[0, 2].item()))
                / float(source_intrinsic_t[0, 0].item())
            ).contiguous()
            ray_y_flat = (
                (grid_y.reshape(-1) - float(source_intrinsic_t[1, 2].item()))
                / float(source_intrinsic_t[1, 1].item())
            ).contiguous()

        source_depth_flat = source_depth_t.reshape(-1)
        source_color_flat = source_color_t.reshape(-1, 4)
        source_valid_idx = torch.nonzero(
            source_depth_flat > float(self.IMMERSIVE_REPROJECT_MIN_DEPTH),
            as_tuple=False,
        ).squeeze(1)
        if int(source_valid_idx.numel()) <= 0:
            return {
                "source_color_t": source_color_t,
                "source_depth_t": source_depth_t,
                "source_points_cv": torch.empty((0, 4), dtype=dtype, device=device),
                "source_color_valid": torch.empty((0, 4), dtype=dtype, device=device),
                "source_point_indices": torch.empty((0,), dtype=torch.long, device=device),
            }

        z = source_depth_flat[source_valid_idx]
        x = ray_x_flat[source_valid_idx] * z
        y = ray_y_flat[source_valid_idx] * z
        ones = torch.ones_like(z)
        return {
            "source_color_t": source_color_t,
            "source_depth_t": source_depth_t,
            "source_points_cv": torch.stack([x, y, z, ones], dim=1),
            "source_color_valid": source_color_flat[source_valid_idx],
            "source_point_indices": torch.arange(
                int(source_valid_idx.numel()),
                dtype=torch.long,
                device=device,
            ),
        }

    def _expand_immersive_reproject_splats(
        self,
        base_x,
        base_y,
        depth,
        source_idx,
        target_width,
        target_height,
        offsets,
    ):
        if int(base_x.numel()) <= 0:
            empty_long = torch.empty((0,), dtype=torch.long, device=base_x.device)
            empty_float = torch.empty((0,), dtype=depth.dtype, device=depth.device)
            return empty_long, empty_float, empty_long
        expanded_x = base_x.unsqueeze(1) + offsets[:, 0].view(1, -1)
        expanded_y = base_y.unsqueeze(1) + offsets[:, 1].view(1, -1)
        valid = (
            (expanded_x >= 0)
            & (expanded_x < int(target_width))
            & (expanded_y >= 0)
            & (expanded_y < int(target_height))
        )
        if not bool(valid.any().item()):
            empty_long = torch.empty((0,), dtype=torch.long, device=base_x.device)
            empty_float = torch.empty((0,), dtype=depth.dtype, device=depth.device)
            return empty_long, empty_float, empty_long
        linear_idx = (expanded_y * int(target_width) + expanded_x)[valid]
        expanded_depth = depth.unsqueeze(1).expand_as(expanded_x)[valid]
        expanded_source_idx = source_idx.unsqueeze(1).expand_as(expanded_x)[valid]
        return linear_idx, expanded_depth, expanded_source_idx

    def _shift_immersive_image(self, tensor, dy, dx):
        shifted = torch.zeros_like(tensor)
        height = int(tensor.shape[0])
        width = int(tensor.shape[1])
        src_y0 = max(0, -int(dy))
        src_y1 = height - max(0, int(dy))
        dst_y0 = max(0, int(dy))
        dst_y1 = height - max(0, -int(dy))
        src_x0 = max(0, -int(dx))
        src_x1 = width - max(0, int(dx))
        dst_x0 = max(0, int(dx))
        dst_x1 = width - max(0, -int(dx))
        if src_y1 <= src_y0 or src_x1 <= src_x0:
            return shifted
        shifted[dst_y0:dst_y1, dst_x0:dst_x1] = tensor[src_y0:src_y1, src_x0:src_x1]
        return shifted

    @torch.no_grad()
    def _fill_immersive_reprojected_scene_holes(
        self,
        color,
        depth,
        valid_mask,
        iterations=None,
        background_rgba=None,
        roi_bounds=None,
        target_coverage=None,
    ):
        if iterations is None:
            iterations = int(self.IMMERSIVE_REPROJECT_HOLE_FILL_ITERS)
        color_filled = color
        depth_filled = depth
        valid_filled = valid_mask
        if background_rgba is None:
            background_rgba = torch.as_tensor(
                self.IMMERSIVE_SCENE_CLEAR_RGBA,
                dtype=color_filled.dtype,
                device=color_filled.device,
            )
        if roi_bounds is not None:
            x0, y0, x1, y1 = [int(v) for v in roi_bounds]
            if x1 <= x0 or y1 <= y0:
                return color_filled, depth_filled, valid_filled
            color_filled = color_filled[y0:y1, x0:x1]
            depth_filled = depth_filled[y0:y1, x0:x1]
            valid_filled = valid_filled[y0:y1, x0:x1]
        fill_passes = max(int(iterations), 1)
        if target_coverage is not None:
            fill_passes = max(fill_passes, 2)
        else:
            invalid_ratio = 1.0 - float(valid_filled.to(dtype=torch.float32).mean().item())
            if invalid_ratio > float(self.IMMERSIVE_REPROJECT_HOLE_FILL_SECOND_PASS_INVALID_RATIO):
                fill_passes = max(fill_passes, 2)

        neighbor_offsets = (
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
        )
        for _ in range(fill_passes):
            if bool(valid_filled.all().item()):
                break
            color_sum = torch.zeros_like(color_filled)
            depth_sum = torch.zeros_like(depth_filled)
            count = torch.zeros_like(depth_filled, dtype=torch.float32)
            for dy, dx in neighbor_offsets:
                shifted_valid = self._shift_immersive_image(valid_filled, dy, dx)
                shifted_color = self._shift_immersive_image(color_filled, dy, dx)
                shifted_depth = self._shift_immersive_image(depth_filled, dy, dx)
                shifted_valid_f = shifted_valid.to(dtype=torch.float32)
                color_sum += shifted_color * shifted_valid_f.unsqueeze(-1)
                depth_sum += shifted_depth * shifted_valid_f
                count += shifted_valid_f
            fill_mask = (~valid_filled) & (count > 0.0)
            if not bool(fill_mask.any().item()):
                break
            count_safe = count.clamp_min(1.0)
            fill_color = color_sum / count_safe.unsqueeze(-1)
            fill_depth = depth_sum / count_safe
            color_filled[fill_mask] = fill_color[fill_mask]
            depth_filled[fill_mask] = fill_depth[fill_mask]
            valid_filled[fill_mask] = True
            if target_coverage is not None:
                current_coverage = float(
                    valid_filled.to(dtype=torch.float32).mean().item()
                )
                if current_coverage >= float(target_coverage):
                    break

        remaining_invalid = ~valid_filled
        if bool(remaining_invalid.any().item()):
            color_filled[remaining_invalid] = background_rgba
            depth_filled[remaining_invalid] = 0.0
        return color, depth, valid_mask

    @torch.no_grad()
    def _reproject_immersive_scene_eye_frame(
        self,
        source_color,
        source_depth,
        source_intrinsic,
        source_eye_pose_world,
        target_intrinsic,
        target_eye_pose_world,
        target_height,
        target_width,
        render_profile_frame=None,
        eye_label=None,
        reproject_cache=None,
        shared_source_data=None,
        source_intrinsic_t=None,
        source_c2w_cv_t=None,
        target_intrinsic_t=None,
        target_w2c_cv_t=None,
        repair_roi_bounds=None,
        target_roi_bounds=None,
        profile_key_prefix="scene_reproject",
    ):
        device = source_color.device
        dtype = torch.float32
        source_color_t = source_color.to(device=device, dtype=dtype)
        source_depth_t = source_depth.to(device=device, dtype=dtype)
        source_height = int(source_depth_t.shape[0])
        source_width = int(source_depth_t.shape[1])
        full_target_height = int(target_height)
        full_target_width = int(target_width)
        roi_origin_x = 0
        roi_origin_y = 0
        local_repair_roi_bounds = repair_roi_bounds
        if target_roi_bounds is not None:
            roi_x0, roi_y0, roi_x1, roi_y1 = [int(v) for v in target_roi_bounds]
            if roi_x1 <= roi_x0 or roi_y1 <= roi_y0:
                raise ValueError(f"Invalid target_roi_bounds: {target_roi_bounds}")
            roi_origin_x = roi_x0
            roi_origin_y = roi_y0
            target_width = roi_x1 - roi_x0
            target_height = roi_y1 - roi_y0
            if repair_roi_bounds is None:
                local_repair_roi_bounds = (0, 0, int(target_width), int(target_height))
            else:
                rx0, ry0, rx1, ry1 = [int(v) for v in repair_roi_bounds]
                local_repair_roi_bounds = (
                    int(rx0 - roi_origin_x),
                    int(ry0 - roi_origin_y),
                    int(rx1 - roi_origin_x),
                    int(ry1 - roi_origin_y),
                )
        else:
            target_height = full_target_height
            target_width = full_target_width
        background_rgba = torch.as_tensor(
            self.IMMERSIVE_SCENE_CLEAR_RGBA,
            dtype=dtype,
            device=device,
        )
        reproject_cache = self._ensure_immersive_reproject_cache(
            reproject_cache,
            source_height,
            source_width,
            target_height,
            target_width,
        )
        if source_intrinsic_t is None:
            source_intrinsic_t = torch.as_tensor(
                source_intrinsic,
                dtype=dtype,
                device=device,
            )
        if target_intrinsic_t is None:
            target_intrinsic_t = torch.as_tensor(
                target_intrinsic,
                dtype=dtype,
                device=device,
            )
        if source_c2w_cv_t is None:
            source_w2c_cv_t = torch.as_tensor(
                self._camera_pose_world_to_cv_w2c(source_eye_pose_world),
                dtype=dtype,
                device=device,
            )
            source_c2w_cv_t = torch.linalg.inv(source_w2c_cv_t)
        if target_w2c_cv_t is None:
            target_w2c_cv_t = torch.as_tensor(
                self._camera_pose_world_to_cv_w2c(target_eye_pose_world),
                dtype=dtype,
                device=device,
            )
        source_to_target_cv_t = target_w2c_cv_t @ source_c2w_cv_t

        if reproject_cache is not None:
            target_min_depth_flat = reproject_cache["target_min_depth_flat"]
            target_source_index_flat = reproject_cache["target_source_index_flat"]
            target_valid_flat = reproject_cache["target_valid_flat"]
            target_color = reproject_cache["target_color"]
            target_depth = reproject_cache["target_depth"]
            target_valid = reproject_cache["target_valid"]
            near_offsets = reproject_cache["near_offsets"]
            far_offsets = reproject_cache["far_offsets"]
        else:
            target_pixels = target_height * target_width
            target_min_depth_flat = torch.empty(
                target_pixels,
                dtype=dtype,
                device=device,
            )
            target_source_index_flat = torch.empty(
                target_pixels,
                dtype=torch.long,
                device=device,
            )
            target_valid_flat = torch.empty(
                target_pixels,
                dtype=torch.bool,
                device=device,
            )
            target_color = torch.empty(
                (target_height, target_width, 4),
                dtype=dtype,
                device=device,
            )
            target_depth = torch.empty(
                (target_height, target_width),
                dtype=dtype,
                device=device,
            )
            target_valid = torch.empty(
                (target_height, target_width),
                dtype=torch.bool,
                device=device,
            )
            near_offsets = torch.tensor(
                [[0, 0], [1, 0], [0, 1], [1, 1]],
                dtype=torch.long,
                device=device,
            )
            far_offsets = torch.tensor(
                [
                    [-1, -1], [0, -1], [1, -1],
                    [-1, 0], [0, 0], [1, 0],
                    [-1, 1], [0, 1], [1, 1],
                ],
                dtype=torch.long,
                device=device,
            )

        reproject_span = self._render_profile_begin_cuda_span(
            render_profile_frame,
            f"{profile_key_prefix}_{eye_label}_cuda",
        )

        target_color[:] = background_rgba
        target_depth.zero_()
        target_valid.zero_()
        target_min_depth_flat.fill_(float("inf"))
        target_source_index_flat.fill_(-1)
        target_valid_flat.zero_()

        if not hasattr(target_min_depth_flat, "scatter_reduce_"):
            raise RuntimeError(
                "reproject_from_center requires torch.Tensor.scatter_reduce_ support"
            )

        if shared_source_data is None:
            shared_source_data = self._prepare_immersive_reproject_source_data(
                source_color_t,
                source_depth_t,
                source_intrinsic_t,
                source_cache=None,
            )

        source_points_cv = shared_source_data["source_points_cv"]
        source_color_valid = shared_source_data["source_color_valid"]
        source_point_indices = shared_source_data["source_point_indices"]

        if int(source_points_cv.shape[0]) > 0:
            target_points_cv = source_points_cv @ source_to_target_cv_t.T

            target_z = target_points_cv[:, 2]
            target_x = target_points_cv[:, 0]
            target_y = target_points_cv[:, 1]
            target_u = (
                (target_x * target_intrinsic_t[0, 0]) / target_z.clamp_min(1e-6)
                + target_intrinsic_t[0, 2]
            )
            target_v = (
                (target_y * target_intrinsic_t[1, 1]) / target_z.clamp_min(1e-6)
                + target_intrinsic_t[1, 2]
            )
            projected_valid = (
                (target_z > float(self.IMMERSIVE_REPROJECT_MIN_DEPTH))
                & (target_u >= float(roi_origin_x) - 2.0)
                & (
                    target_u
                    < float(
                        roi_origin_x + (
                            target_width if target_roi_bounds is not None else full_target_width
                        )
                    )
                    + 2.0
                )
                & (target_v >= float(roi_origin_y) - 2.0)
                & (
                    target_v
                    < float(
                        roi_origin_y + (
                            target_height
                            if target_roi_bounds is not None
                            else full_target_height
                        )
                    )
                    + 2.0
                )
            )
            if int(projected_valid.sum().item()) > 0:
                projected_depth = target_z[projected_valid]
                projected_u = target_u[projected_valid] - float(roi_origin_x)
                projected_v = target_v[projected_valid] - float(roi_origin_y)
                projected_source_idx = source_point_indices[projected_valid]
                near_mask = projected_depth <= float(self.IMMERSIVE_REPROJECT_NEAR_SPLAT_DEPTH)

                expanded_linear_idx_parts = []
                expanded_depth_parts = []
                expanded_source_parts = []
                if int(near_mask.sum().item()) > 0:
                    near_base_x = torch.floor(projected_u[near_mask]).to(torch.long)
                    near_base_y = torch.floor(projected_v[near_mask]).to(torch.long)
                    near_linear_idx, near_depth, near_source_idx = (
                        self._expand_immersive_reproject_splats(
                            near_base_x,
                            near_base_y,
                            projected_depth[near_mask],
                            projected_source_idx[near_mask],
                            target_width,
                            target_height,
                            near_offsets,
                        )
                    )
                    expanded_linear_idx_parts.append(near_linear_idx)
                    expanded_depth_parts.append(near_depth)
                    expanded_source_parts.append(near_source_idx)
                far_mask = ~near_mask
                if int(far_mask.sum().item()) > 0:
                    far_base_x = torch.round(projected_u[far_mask]).to(torch.long)
                    far_base_y = torch.round(projected_v[far_mask]).to(torch.long)
                    far_linear_idx, far_depth, far_source_idx = (
                        self._expand_immersive_reproject_splats(
                            far_base_x,
                            far_base_y,
                            projected_depth[far_mask],
                            projected_source_idx[far_mask],
                            target_width,
                            target_height,
                            far_offsets,
                        )
                    )
                    expanded_linear_idx_parts.append(far_linear_idx)
                    expanded_depth_parts.append(far_depth)
                    expanded_source_parts.append(far_source_idx)

                if expanded_linear_idx_parts:
                    linear_idx = torch.cat(expanded_linear_idx_parts, dim=0)
                    expanded_depth = torch.cat(expanded_depth_parts, dim=0)
                    expanded_source_idx = torch.cat(expanded_source_parts, dim=0)
                    target_min_depth_flat.scatter_reduce_(
                        0,
                        linear_idx,
                        expanded_depth,
                        reduce="amin",
                        include_self=True,
                    )
                    gathered_min_depth = target_min_depth_flat[linear_idx]
                    winner_mask = torch.isclose(
                        expanded_depth,
                        gathered_min_depth,
                        atol=float(self.IMMERSIVE_REPROJECT_WINNER_ATOL),
                        rtol=float(self.IMMERSIVE_REPROJECT_WINNER_RTOL),
                    )
                    if int(winner_mask.sum().item()) > 0:
                        target_source_index_flat.scatter_reduce_(
                            0,
                            linear_idx[winner_mask],
                            expanded_source_idx[winner_mask] + 1,
                            reduce="amax",
                            include_self=True,
                        )
                        target_source_index_flat.sub_(1)
                        target_valid_flat.copy_(target_source_index_flat >= 0)
                        flat_color = target_color.view(-1, 4)
                        flat_depth = target_depth.view(-1)
                        flat_valid = target_valid.view(-1)
                        flat_valid.copy_(target_valid_flat)
                        if bool(flat_valid.any().item()):
                            selected_source_idx = target_source_index_flat[flat_valid]
                            flat_color[flat_valid] = source_color_valid[selected_source_idx]
                            flat_depth[flat_valid] = target_min_depth_flat[flat_valid]

        self._render_profile_end_cuda_span(render_profile_frame, reproject_span)
        pre_fill_ratio = float(target_valid.to(dtype=torch.float32).mean().item())
        if render_profile_frame is not None and eye_label is not None:
            render_profile_frame[
                f"{profile_key_prefix}_valid_pre_{eye_label}_ratio"
            ] = pre_fill_ratio
            if local_repair_roi_bounds is not None:
                render_profile_frame[
                    f"{profile_key_prefix}_roi_pre_{eye_label}_ratio"
                ] = self._scene_valid_roi_coverage(target_valid, local_repair_roi_bounds)

        hole_fill_span = self._render_profile_begin_cuda_span(
            render_profile_frame,
            f"{profile_key_prefix}_hole_fill_{eye_label}_cuda",
        )
        if local_repair_roi_bounds is not None:
            target_color, target_depth, target_valid = self._fill_immersive_reprojected_scene_holes(
                target_color,
                target_depth,
                target_valid,
                background_rgba=background_rgba,
                roi_bounds=local_repair_roi_bounds,
                target_coverage=float(self.IMMERSIVE_REPROJECT_ROI_TARGET_COVERAGE),
            )
        self._render_profile_end_cuda_span(render_profile_frame, hole_fill_span)
        if render_profile_frame is not None and eye_label is not None:
            render_profile_frame[
                f"{profile_key_prefix}_valid_post_{eye_label}_ratio"
            ] = float(target_valid.to(dtype=torch.float32).mean().item())
            if local_repair_roi_bounds is not None:
                render_profile_frame[
                    f"{profile_key_prefix}_roi_post_{eye_label}_ratio"
                ] = self._scene_valid_roi_coverage(target_valid, local_repair_roi_bounds)
        return target_color, target_depth, target_valid

    def _scene_valid_patch_coverage(self, valid_mask, pixel, radius=None):
        if pixel is None or valid_mask is None:
            return 0.0
        if radius is None:
            radius = int(self.IMMERSIVE_REPROJECT_STARTUP_PATCH_RADIUS)
        if torch.is_tensor(pixel):
            pixel_np = pixel.detach().cpu().numpy()
        else:
            pixel_np = np.asarray(pixel, dtype=np.float32)
        x = int(round(float(pixel_np[0])))
        y = int(round(float(pixel_np[1])))
        height = int(valid_mask.shape[0])
        width = int(valid_mask.shape[1])
        x0 = max(0, x - radius)
        x1 = min(width, x + radius + 1)
        y0 = max(0, y - radius)
        y1 = min(height, y + radius + 1)
        if x1 <= x0 or y1 <= y0:
            return 0.0
        patch = valid_mask[y0:y1, x0:x1]
        return float(patch.to(dtype=torch.float32).mean().item())

    def _resolve_immersive_balanced_edge_repair_strips(
        self,
        scene_renderer,
        eye_label,
        eye_pose_world,
        eye_intrinsic,
        eye_width,
        eye_height,
        balanced_runtime_state=None,
        update_state=True,
    ):
        side_wall_roi_state = None
        if balanced_runtime_state is not None:
            side_wall_roi_state = balanced_runtime_state.setdefault(
                "side_wall_roi_state",
                {
                    "left": {"left": None, "right": None},
                    "right": {"left": None, "right": None},
                },
            ).setdefault(
                eye_label,
                {"left": None, "right": None},
            )
        eye_w2c_cv = self._camera_pose_world_to_cv_w2c(eye_pose_world)
        strip_specs = []
        edge_metrics = {
            "left": {"roi_ratio": 0.0, "strip_width_ratio": 0.0},
            "right": {"roi_ratio": 0.0, "strip_width_ratio": 0.0},
        }
        for side_name in ("left", "right"):
            wall_bounds = scene_renderer.wall_world_bounds(side_name)
            prev_side_bounds = (
                None if side_wall_roi_state is None else side_wall_roi_state.get(side_name)
            )
            if wall_bounds is None:
                if side_wall_roi_state is not None and update_state:
                    side_wall_roi_state[side_name] = None
                strip_specs.append(
                    {
                        "side_name": side_name,
                        "anchor_edge": None,
                        "roi_bounds": None,
                        "roi_ratio": 0.0,
                        "strip_width_ratio": 0.0,
                        "fullframe_fallback": False,
                        "hysteresis_bounds": None,
                    }
                )
                continue

            roi_bounds, roi_ratio, side_ff, side_debug = (
                self._resolve_immersive_balanced_side_wall_strip_roi(
                    wall_bounds[0],
                    wall_bounds[1],
                    eye_intrinsic,
                    eye_w2c_cv,
                    eye_width,
                    eye_height,
                    prev_bounds=prev_side_bounds,
                )
            )
            if side_wall_roi_state is not None and update_state:
                side_wall_roi_state[side_name] = (
                    None if roi_bounds is None or side_ff else roi_bounds
                )
            anchor_edge = side_debug.get("anchor_edge")
            strip_width_ratio = float(side_debug.get("strip_width_ratio", roi_ratio))
            if anchor_edge in edge_metrics and roi_bounds is not None:
                edge_metrics[anchor_edge]["roi_ratio"] = max(
                    float(edge_metrics[anchor_edge]["roi_ratio"]),
                    float(roi_ratio),
                )
                edge_metrics[anchor_edge]["strip_width_ratio"] = max(
                    float(edge_metrics[anchor_edge]["strip_width_ratio"]),
                    strip_width_ratio,
                )
            strip_specs.append(
                {
                    "side_name": side_name,
                    "anchor_edge": anchor_edge,
                    "roi_bounds": None if roi_bounds is None else tuple(int(v) for v in roi_bounds),
                    "roi_ratio": float(roi_ratio),
                    "strip_width_ratio": strip_width_ratio,
                    "fullframe_fallback": bool(side_ff),
                    "hysteresis_bounds": None
                    if side_debug.get("hysteresis_bounds") is None
                    else tuple(int(v) for v in side_debug["hysteresis_bounds"]),
                }
            )
        return strip_specs, edge_metrics

    def _build_immersive_balanced_edge_feather_mask(
        self,
        roi_height,
        roi_width,
        anchor_edge,
        device,
        dtype,
    ):
        roi_height = int(roi_height)
        roi_width = int(roi_width)
        if roi_height <= 0 or roi_width <= 0:
            return torch.empty((0, 0), dtype=dtype, device=device)
        feather_px = min(
            max(int(self.IMMERSIVE_BALANCED_EDGE_WARP_FEATHER_PX), 1),
            roi_width,
        )
        if feather_px <= 1 or anchor_edge not in {"left", "right"}:
            return torch.ones((roi_height, roi_width), dtype=dtype, device=device)
        feather = torch.ones((roi_width,), dtype=dtype, device=device)
        if anchor_edge == "left":
            feather[-feather_px:] = torch.linspace(
                1.0,
                0.0,
                steps=feather_px,
                dtype=dtype,
                device=device,
            )
        else:
            feather[:feather_px] = torch.linspace(
                0.0,
                1.0,
                steps=feather_px,
                dtype=dtype,
                device=device,
            )
        return feather.unsqueeze(0).expand(roi_height, roi_width)

    @torch.no_grad()
    def _compose_immersive_balanced_edge_patch(
        self,
        background_color_t,
        background_depth_t,
        patch_color_t,
        patch_depth_t,
        patch_valid_t,
        roi_bounds,
        anchor_edge,
    ):
        if patch_valid_t is None or int(patch_valid_t.numel()) <= 0:
            return background_color_t, background_depth_t
        patch_valid_t = patch_valid_t.to(device=cfg.device, dtype=torch.bool)
        if not bool(patch_valid_t.any().item()):
            return background_color_t, background_depth_t
        x0, y0, x1, y1 = [int(v) for v in roi_bounds]
        if x1 <= x0 or y1 <= y0:
            return background_color_t, background_depth_t
        background_patch = background_color_t[y0:y1, x0:x1]
        background_depth_patch = background_depth_t[y0:y1, x0:x1]
        feather_mask = self._build_immersive_balanced_edge_feather_mask(
            y1 - y0,
            x1 - x0,
            anchor_edge,
            device=background_patch.device,
            dtype=background_patch.dtype,
        )
        effective_alpha = (
            patch_valid_t.to(dtype=background_patch.dtype) * feather_mask
        ).unsqueeze(-1)
        if bool((effective_alpha > 0.0).any().item()):
            background_patch[..., :3] = (
                background_patch[..., :3] * (1.0 - effective_alpha)
                + patch_color_t[..., :3] * effective_alpha
            )
            background_patch[..., 3] = 255.0
        background_depth_patch[patch_valid_t] = patch_depth_t[patch_valid_t]
        return background_color_t, background_depth_t

    @torch.no_grad()
    def _compose_immersive_balanced_replacement_patch(
        self,
        background_color_t,
        background_depth_t,
        overlay_color_rgba,
        overlay_depth,
        anchor_edge,
        roi_bounds=None,
    ):
        target_height = int(background_color_t.shape[0])
        target_width = int(background_color_t.shape[1])
        if roi_bounds is not None:
            x0, y0, x1, y1 = [int(v) for v in roi_bounds]
            if x1 <= x0 or y1 <= y0:
                raise ValueError(f"Invalid side replacement roi_bounds: {roi_bounds}")
            roi_height = y1 - y0
            roi_width = x1 - x0
            overlay_color_t, overlay_depth_t = self._prepare_immersive_scene_frame_for_compose(
                overlay_color_rgba,
                overlay_depth,
                roi_height,
                roi_width,
                compose_cache=None,
            )
            background_patch = background_color_t[y0:y1, x0:x1]
            background_depth_patch = background_depth_t[y0:y1, x0:x1]
            feather_mask = self._build_immersive_balanced_edge_feather_mask(
                roi_height,
                roi_width,
                anchor_edge,
                device=background_patch.device,
                dtype=background_patch.dtype,
            )
        else:
            overlay_color_t, overlay_depth_t = self._prepare_immersive_scene_frame_for_compose(
                overlay_color_rgba,
                overlay_depth,
                target_height,
                target_width,
                compose_cache=None,
            )
            background_patch = background_color_t
            background_depth_patch = background_depth_t
            feather_mask = torch.ones(
                (target_height, target_width),
                dtype=background_patch.dtype,
                device=background_patch.device,
            )

        overlay_has_depth = overlay_depth_t > float(self.IMMERSIVE_STARTUP_DEPTH_EPS)
        overlay_alpha = overlay_color_t[..., 3].clamp(0.0, 255.0) / 255.0
        valid_mask = overlay_has_depth
        if not bool(valid_mask.any().item()):
            valid_mask = overlay_alpha > float(self.IMMERSIVE_COMPOSE_ALPHA_EPS)
        if not bool(valid_mask.any().item()):
            return background_color_t, background_depth_t

        effective_alpha = (
            valid_mask.to(dtype=background_patch.dtype) * feather_mask
        ).unsqueeze(-1)
        if bool((effective_alpha > 0.0).any().item()):
            background_patch[..., :3] = (
                background_patch[..., :3] * (1.0 - effective_alpha)
                + overlay_color_t[..., :3] * effective_alpha
            )
            background_patch[..., 3] = 255.0
        background_depth_patch[valid_mask] = overlay_depth_t[valid_mask]
        return background_color_t, background_depth_t

    @torch.no_grad()
    def _render_immersive_balanced_side_background_replacement(
        self,
        scene_renderer,
        eye_label,
        side_name,
        eye_pose_world,
        eye_intrinsic,
        eye_width,
        eye_height,
        background_color_t,
        background_depth_t,
        anchor_edge,
        roi_bounds=None,
        fullframe_fallback=False,
        render_profile_frame=None,
        fullframe_render_cache=None,
    ):
        side_name = str(side_name)
        if side_name not in {"left", "right"}:
            raise ValueError(f"Unsupported side wall fallback side: {side_name}")
        fullframe_fallback = bool(fullframe_fallback or roi_bounds is None)
        overlay_roi_bounds = roi_bounds
        replace_full_eye = False
        rendered_background = False
        side_render_start = time.perf_counter() if render_profile_frame is not None else None
        if fullframe_fallback:
            cache_ready = (
                fullframe_render_cache is not None
                and fullframe_render_cache.get("color") is not None
                and fullframe_render_cache.get("depth") is not None
            )
            if cache_ready:
                side_color = fullframe_render_cache["color"]
                side_depth = fullframe_render_cache["depth"]
            else:
                side_color, side_depth = scene_renderer.render_background_eye(
                    eye_pose_world,
                    eye_intrinsic,
                    width=eye_width,
                    height=eye_height,
                )
                rendered_background = True
                if fullframe_render_cache is not None:
                    fullframe_render_cache["color"] = side_color
                    fullframe_render_cache["depth"] = side_depth
            if roi_bounds is None:
                overlay_roi_bounds = None
                replace_full_eye = True
            else:
                x0, y0, x1, y1 = [int(v) for v in roi_bounds]
                replace_full_eye = (
                    x0 <= 0
                    and y0 <= 0
                    and x1 >= int(eye_width)
                    and y1 >= int(eye_height)
                )
                if replace_full_eye:
                    overlay_roi_bounds = None
                else:
                    side_color = side_color[y0:y1, x0:x1]
                    side_depth = side_depth[y0:y1, x0:x1]
                    overlay_roi_bounds = roi_bounds
        else:
            side_color, side_depth = scene_renderer.render_background_eye_roi(
                eye_pose_world,
                eye_intrinsic,
                roi_bounds,
            )
            rendered_background = True
        if rendered_background and side_render_start is not None:
            self._render_profile_add_wall_time(
                render_profile_frame,
                f"scene_render_side_{eye_label}_wall",
                time.perf_counter() - side_render_start,
            )
        return self._compose_immersive_balanced_replacement_patch(
            background_color_t,
            background_depth_t,
            side_color,
            side_depth,
            anchor_edge=None if replace_full_eye else anchor_edge,
            roi_bounds=overlay_roi_bounds,
        )

    @torch.no_grad()
    def _repair_immersive_balanced_background_eye(
        self,
        scene_renderer,
        eye_label,
        eye_pose_world,
        eye_intrinsic,
        eye_width,
        eye_height,
        base_background_color_t,
        base_background_depth_t,
        center_eye_pose_world,
        center_intrinsic,
        balanced_runtime_state=None,
        reproject_caches=None,
        render_profile_frame=None,
        shared_source_data=None,
        source_intrinsic_t=None,
        source_c2w_cv_t=None,
    ):
        side_wall_mode = (
            None
            if balanced_runtime_state is None
            else str(balanced_runtime_state.get("side_wall_mode", "disabled"))
        )
        if side_wall_mode not in {
            "edge_warp_roi",
            "warp_first_hybrid",
            "per_eye_roi",
            "per_eye_roi_replace",
        }:
            return base_background_color_t, base_background_depth_t
        force_render_only = side_wall_mode in {
            "per_eye_roi",
            "per_eye_roi_replace",
        }

        strip_specs, edge_metrics = self._resolve_immersive_balanced_edge_repair_strips(
            scene_renderer,
            eye_label,
            eye_pose_world,
            eye_intrinsic,
            eye_width,
            eye_height,
            balanced_runtime_state=balanced_runtime_state,
            update_state=True,
        )
        if render_profile_frame is not None:
            for edge_name in ("left", "right"):
                render_profile_frame[f"scene_side_roi_{edge_name}_ratio"] = max(
                    float(render_profile_frame.get(f"scene_side_roi_{edge_name}_ratio", 0.0)),
                    float(edge_metrics[edge_name]["roi_ratio"]),
                )
                render_profile_frame[
                    f"scene_side_strip_{edge_name}_width_ratio"
                ] = max(
                    float(
                        render_profile_frame.get(
                            f"scene_side_strip_{edge_name}_width_ratio",
                            0.0,
                        )
                    ),
                    float(edge_metrics[edge_name]["strip_width_ratio"]),
                )
                render_profile_frame[
                    f"scene_side_fullframe_fallback_{edge_name}_ratio"
                ] = max(
                    float(
                        render_profile_frame.get(
                            f"scene_side_fullframe_fallback_{edge_name}_ratio",
                            0.0,
                        )
                    ),
                    0.0,
                )

        active_specs = [
            strip_spec
            for strip_spec in strip_specs
            if (
                strip_spec["roi_bounds"] is not None
                and strip_spec["anchor_edge"] in {"left", "right"}
            )
        ]
        if not active_specs:
            return base_background_color_t, base_background_depth_t

        target_intrinsic_t = None
        target_w2c_cv_t = None
        edge_reproject_caches = None
        if reproject_caches is not None:
            edge_reproject_caches = reproject_caches.setdefault("balanced_edge", {})
            edge_reproject_caches = edge_reproject_caches.setdefault(
                eye_label,
                {"left": {}, "right": {}},
            )

        warp_span = None
        compose_span = None
        repaired_background_color_t = base_background_color_t
        repaired_background_depth_t = base_background_depth_t
        composed_any_patch = False
        coverage_min = float(self.IMMERSIVE_BALANCED_SIDE_WALL_WARP_MIN_VALID_COVERAGE)
        warp_width_max = float(
            self.IMMERSIVE_BALANCED_SIDE_WALL_STRIP_WARP_MAX_WIDTH_RATIO
        )
        shared_source_data_local = shared_source_data
        fullframe_background_render_cache = {"color": None, "depth": None}
        for strip_spec in active_specs:
            anchor_edge = str(strip_spec["anchor_edge"])
            strip_width_ratio = float(strip_spec["strip_width_ratio"])
            use_fullframe_fallback = bool(strip_spec["fullframe_fallback"])
            if (
                render_profile_frame is not None
                and anchor_edge in {"left", "right"}
                and use_fullframe_fallback
            ):
                render_profile_frame[
                    f"scene_side_fullframe_fallback_{anchor_edge}_ratio"
                ] = max(
                    float(
                        render_profile_frame.get(
                            f"scene_side_fullframe_fallback_{anchor_edge}_ratio",
                            0.0,
                        )
                    ),
                    1.0,
                )

            should_try_warp = (
                (not force_render_only)
                and (not use_fullframe_fallback)
                and strip_width_ratio <= warp_width_max
            )
            if should_try_warp:
                if target_intrinsic_t is None:
                    target_intrinsic_t = torch.as_tensor(
                        eye_intrinsic,
                        dtype=torch.float32,
                        device=cfg.device,
                    )
                if target_w2c_cv_t is None:
                    target_w2c_cv_t = torch.as_tensor(
                        self._camera_pose_world_to_cv_w2c(eye_pose_world),
                        dtype=torch.float32,
                        device=cfg.device,
                    )
                if shared_source_data_local is None:
                    if source_intrinsic_t is None:
                        source_intrinsic_t = torch.as_tensor(
                            center_intrinsic,
                            dtype=torch.float32,
                            device=cfg.device,
                        )
                    shared_source_data_local = (
                        self._prepare_immersive_reproject_source_data(
                            base_background_color_t,
                            base_background_depth_t,
                            source_intrinsic_t,
                            source_cache=None
                            if reproject_caches is None
                            else reproject_caches.get("background_source"),
                        )
                    )
                if warp_span is None:
                    warp_span = self._render_profile_begin_cuda_span(
                        render_profile_frame,
                        f"scene_repair_background_{eye_label}_cuda",
                    )
                patch_color_t, patch_depth_t, patch_valid_t = (
                    self._reproject_immersive_scene_eye_frame(
                        base_background_color_t,
                        base_background_depth_t,
                        center_intrinsic,
                        center_eye_pose_world,
                        eye_intrinsic,
                        eye_pose_world,
                        eye_height,
                        eye_width,
                        render_profile_frame=render_profile_frame,
                        eye_label=eye_label,
                        reproject_cache=None
                        if edge_reproject_caches is None
                        else edge_reproject_caches.get(strip_spec["side_name"]),
                        shared_source_data=shared_source_data_local,
                        source_intrinsic_t=source_intrinsic_t,
                        source_c2w_cv_t=source_c2w_cv_t,
                        target_intrinsic_t=target_intrinsic_t,
                        target_w2c_cv_t=target_w2c_cv_t,
                        target_roi_bounds=strip_spec["roi_bounds"],
                        profile_key_prefix="scene_reproject_background",
                    )
                )
                patch_coverage = float(
                    patch_valid_t.to(dtype=torch.float32).mean().item()
                )
                if patch_coverage >= coverage_min:
                    if compose_span is None:
                        compose_span = self._render_profile_begin_cuda_span(
                            render_profile_frame,
                            f"scene_compose_side_{eye_label}_cuda",
                        )
                    if not composed_any_patch:
                        repaired_background_color_t = torch.clone(
                            base_background_color_t
                        )
                        repaired_background_depth_t = torch.clone(
                            base_background_depth_t
                        )
                        composed_any_patch = True
                    repaired_background_color_t, repaired_background_depth_t = (
                        self._compose_immersive_balanced_edge_patch(
                            repaired_background_color_t,
                            repaired_background_depth_t,
                            patch_color_t,
                            patch_depth_t,
                            patch_valid_t,
                            strip_spec["roi_bounds"],
                            anchor_edge,
                        )
                    )
                    if render_profile_frame is not None and anchor_edge in {
                        "left",
                        "right",
                    }:
                        render_profile_frame[
                            f"scene_side_warp_{anchor_edge}_used"
                        ] = 1.0
                    continue

            if compose_span is None:
                compose_span = self._render_profile_begin_cuda_span(
                    render_profile_frame,
                    f"scene_compose_side_{eye_label}_cuda",
                )
            if not composed_any_patch:
                repaired_background_color_t = torch.clone(base_background_color_t)
                repaired_background_depth_t = torch.clone(base_background_depth_t)
                composed_any_patch = True
            repaired_background_color_t, repaired_background_depth_t = (
                self._render_immersive_balanced_side_background_replacement(
                    scene_renderer,
                    eye_label,
                    strip_spec["side_name"],
                    eye_pose_world,
                    eye_intrinsic,
                    eye_width,
                    eye_height,
                    repaired_background_color_t,
                    repaired_background_depth_t,
                    anchor_edge=anchor_edge,
                    roi_bounds=strip_spec["roi_bounds"],
                    fullframe_fallback=use_fullframe_fallback,
                    render_profile_frame=render_profile_frame,
                    fullframe_render_cache=fullframe_background_render_cache,
                )
            )
            if render_profile_frame is not None and anchor_edge in {
                "left",
                "right",
            }:
                render_profile_frame[
                    f"scene_side_render_fallback_{anchor_edge}_used"
                ] = 1.0
        self._render_profile_end_cuda_span(render_profile_frame, compose_span)
        self._render_profile_end_cuda_span(render_profile_frame, warp_span)
        if not composed_any_patch:
            return base_background_color_t, base_background_depth_t
        return repaired_background_color_t, repaired_background_depth_t

    @torch.no_grad()
    def _validate_immersive_balanced_edge_warp_startup(
        self,
        scene_renderer,
        left_eye_sample,
        right_eye_sample,
        left_eye_pose_world,
        right_eye_pose_world,
        eye_width,
        eye_height,
        scene_width,
        scene_height,
        shared_scene_compose_cache=None,
        reproject_caches=None,
    ):
        center_eye_pose_world, center_intrinsic = self._build_immersive_center_scene_view(
            left_eye_pose_world,
            right_eye_pose_world,
            self._eye_sample_intrinsic(left_eye_sample, eye_width, eye_height)
            if left_eye_sample is not None and left_eye_sample.pose_valid
            else None,
            self._eye_sample_intrinsic(right_eye_sample, eye_width, eye_height)
            if right_eye_sample is not None and right_eye_sample.pose_valid
            else None,
        )
        center_scene_intrinsic = self._scale_intrinsic_for_resolution(
            center_intrinsic,
            eye_width,
            eye_height,
            scene_width,
            scene_height,
        )
        center_scene_color, center_scene_depth = scene_renderer.render_background_eye(
            center_eye_pose_world,
            center_scene_intrinsic,
            width=scene_width,
            height=scene_height,
        )
        background_eye_color_t, background_eye_depth_t = (
            self._prepare_immersive_scene_frame_for_compose(
                center_scene_color,
                center_scene_depth,
                eye_height,
                eye_width,
                compose_cache=shared_scene_compose_cache,
            )
        )
        balanced_runtime_state = getattr(
            self,
            "_immersive_balanced_runtime_state",
            None,
        )
        side_wall_mode = (
            "disabled"
            if balanced_runtime_state is None
            else str(balanced_runtime_state.get("side_wall_mode", "disabled"))
        )
        validation_runtime_state = None
        if balanced_runtime_state is not None:
            validation_runtime_state = {
                "side_wall_roi_state": {
                    "left": dict(
                        balanced_runtime_state.get("side_wall_roi_state", {})
                        .get("left", {"left": None, "right": None})
                    ),
                    "right": dict(
                        balanced_runtime_state.get("side_wall_roi_state", {})
                        .get("right", {"left": None, "right": None})
                    ),
                }
            }
        if side_wall_mode in {"per_eye_roi", "per_eye_roi_replace"}:
            debug = {
                "mode": "balanced_full_room_roi_side_strip_replace",
                "center_intrinsic": np.asarray(center_intrinsic, dtype=np.float32).tolist(),
            }
            for eye_name, eye_sample, eye_pose_world in (
                ("left", left_eye_sample, left_eye_pose_world),
                ("right", right_eye_sample, right_eye_pose_world),
            ):
                if eye_sample is None or not eye_sample.pose_valid or eye_pose_world is None:
                    continue
                intrinsic = self._eye_sample_intrinsic(eye_sample, eye_width, eye_height)
                strip_specs, edge_metrics = self._resolve_immersive_balanced_edge_repair_strips(
                    scene_renderer,
                    eye_name,
                    eye_pose_world,
                    intrinsic,
                    eye_width,
                    eye_height,
                    balanced_runtime_state=validation_runtime_state,
                    update_state=True,
                )
                debug[f"{eye_name}_edge_strip_ratios"] = {
                    edge_name: float(edge_metrics[edge_name]["strip_width_ratio"])
                    for edge_name in ("left", "right")
                }
                for strip_spec in strip_specs:
                    side_name = str(strip_spec["side_name"])
                    debug[f"{eye_name}_{side_name}_wall_roi_ratio"] = float(
                        strip_spec["roi_ratio"]
                    )
                    debug[f"{eye_name}_{side_name}_side_wall_anchor_edge"] = (
                        strip_spec["anchor_edge"]
                    )
                    debug[f"{eye_name}_{side_name}_wall_roi_bounds"] = (
                        None
                        if strip_spec["roi_bounds"] is None
                        else [int(v) for v in strip_spec["roi_bounds"]]
                    )
                    debug[f"{eye_name}_{side_name}_wall_fullframe_fallback"] = bool(
                        strip_spec["fullframe_fallback"]
                    )
                    if strip_spec["roi_bounds"] is None or strip_spec["anchor_edge"] not in {
                        "left",
                        "right",
                    }:
                        debug[f"{eye_name}_{side_name}_repair_strategy"] = "inactive"
                    elif strip_spec["fullframe_fallback"]:
                        debug[f"{eye_name}_{side_name}_repair_strategy"] = (
                            "background_fullframe_replace"
                        )
                    else:
                        debug[f"{eye_name}_{side_name}_repair_strategy"] = (
                            "background_roi_replace"
                        )
            return True, debug
        center_intrinsic_t = torch.as_tensor(
            center_intrinsic,
            dtype=torch.float32,
            device=cfg.device,
        )
        center_w2c_cv_t = torch.as_tensor(
            self._camera_pose_world_to_cv_w2c(center_eye_pose_world),
            dtype=torch.float32,
            device=cfg.device,
        )
        center_c2w_cv_t = torch.linalg.inv(center_w2c_cv_t)
        shared_source_data = self._prepare_immersive_reproject_source_data(
            background_eye_color_t,
            background_eye_depth_t,
            center_intrinsic_t,
            source_cache=None
            if reproject_caches is None
            else reproject_caches.get("background_source"),
        )
        debug = {
            "mode": "balanced_edge_warp",
            "center_intrinsic": np.asarray(center_intrinsic, dtype=np.float32).tolist(),
        }
        failures = []
        coverage_min = float(self.IMMERSIVE_BALANCED_SIDE_WALL_WARP_MIN_VALID_COVERAGE)
        warp_width_max = float(
            self.IMMERSIVE_BALANCED_SIDE_WALL_STRIP_WARP_MAX_WIDTH_RATIO
        )
        validation_edge_caches = None
        if reproject_caches is not None:
            validation_edge_caches = reproject_caches.setdefault(
                "balanced_edge_validation",
                {},
            )

        for eye_name, eye_sample, eye_pose_world in (
            ("left", left_eye_sample, left_eye_pose_world),
            ("right", right_eye_sample, right_eye_pose_world),
        ):
            if eye_sample is None or not eye_sample.pose_valid or eye_pose_world is None:
                continue
            intrinsic = self._eye_sample_intrinsic(eye_sample, eye_width, eye_height)
            strip_specs, edge_metrics = self._resolve_immersive_balanced_edge_repair_strips(
                scene_renderer,
                eye_name,
                eye_pose_world,
                intrinsic,
                eye_width,
                eye_height,
                balanced_runtime_state=validation_runtime_state,
                update_state=True,
            )
            debug[f"{eye_name}_edge_strip_ratios"] = {
                edge_name: float(edge_metrics[edge_name]["strip_width_ratio"])
                for edge_name in ("left", "right")
            }
            target_intrinsic_t = torch.as_tensor(
                intrinsic,
                dtype=torch.float32,
                device=cfg.device,
            )
            target_w2c_cv_t = torch.as_tensor(
                self._camera_pose_world_to_cv_w2c(eye_pose_world),
                dtype=torch.float32,
                device=cfg.device,
            )
            eye_validation_caches = None
            if validation_edge_caches is not None:
                eye_validation_caches = validation_edge_caches.setdefault(
                    eye_name,
                    {"left": {}, "right": {}},
                )
            for strip_spec in strip_specs:
                side_name = str(strip_spec["side_name"])
                debug[f"{eye_name}_{side_name}_wall_roi_ratio"] = float(
                    strip_spec["roi_ratio"]
                )
                debug[f"{eye_name}_{side_name}_side_wall_anchor_edge"] = (
                    strip_spec["anchor_edge"]
                )
                debug[f"{eye_name}_{side_name}_wall_roi_bounds"] = (
                    None
                    if strip_spec["roi_bounds"] is None
                    else [int(v) for v in strip_spec["roi_bounds"]]
                )
                debug[f"{eye_name}_{side_name}_wall_fullframe_fallback"] = bool(
                    strip_spec["fullframe_fallback"]
                )
                if strip_spec["roi_bounds"] is None or strip_spec["anchor_edge"] not in {
                    "left",
                    "right",
                }:
                    debug[f"{eye_name}_{side_name}_repair_strategy"] = "inactive"
                    continue
                if strip_spec["fullframe_fallback"]:
                    debug[f"{eye_name}_{side_name}_repair_strategy"] = (
                        "pyrender_fullframe_render_fallback"
                    )
                    continue
                if float(strip_spec["strip_width_ratio"]) > warp_width_max:
                    debug[f"{eye_name}_{side_name}_repair_strategy"] = (
                        "pyrender_roi_render_fallback"
                    )
                    continue
                debug[f"{eye_name}_{side_name}_repair_strategy"] = "warp"
                _, _, patch_valid_t = self._reproject_immersive_scene_eye_frame(
                    background_eye_color_t,
                    background_eye_depth_t,
                    center_intrinsic,
                    center_eye_pose_world,
                    intrinsic,
                    eye_pose_world,
                    eye_height,
                    eye_width,
                    reproject_cache=None
                    if eye_validation_caches is None
                    else eye_validation_caches.get(side_name),
                    shared_source_data=shared_source_data,
                    source_intrinsic_t=center_intrinsic_t,
                    source_c2w_cv_t=center_c2w_cv_t,
                    target_intrinsic_t=target_intrinsic_t,
                    target_w2c_cv_t=target_w2c_cv_t,
                    target_roi_bounds=strip_spec["roi_bounds"],
                    profile_key_prefix="scene_reproject_background",
                )
                patch_coverage = float(
                    patch_valid_t.to(dtype=torch.float32).mean().item()
                )
                debug[f"{eye_name}_{side_name}_edge_repair_valid_ratio"] = (
                    patch_coverage
                )
                if patch_coverage < coverage_min:
                    failures.append(
                        {
                            "eye": eye_name,
                            "side": side_name,
                            "anchor_edge": strip_spec["anchor_edge"],
                            "roi_bounds": [int(v) for v in strip_spec["roi_bounds"]],
                            "valid_ratio": patch_coverage,
                        }
                    )

        debug["failures"] = failures
        return len(failures) == 0, debug

    @torch.no_grad()
    def _validate_immersive_reprojected_scene_startup(
        self,
        layout,
        scene_renderer,
        left_eye_sample,
        right_eye_sample,
        left_eye_pose_world,
        right_eye_pose_world,
        eye_width,
        eye_height,
        scene_width,
        scene_height,
        gaussians,
        shared_scene_compose_cache=None,
        reproject_caches=None,
        scene_stereo_mode="reproject_from_center",
    ):
        background_only = (
            scene_stereo_mode == self.IMMERSIVE_BALANCED_INTERNAL_STEREO_MODE
        )
        center_eye_pose_world, center_intrinsic = self._build_immersive_center_scene_view(
            left_eye_pose_world,
            right_eye_pose_world,
            self._eye_sample_intrinsic(left_eye_sample, eye_width, eye_height)
            if left_eye_sample is not None and left_eye_sample.pose_valid
            else None,
            self._eye_sample_intrinsic(right_eye_sample, eye_width, eye_height)
            if right_eye_sample is not None and right_eye_sample.pose_valid
            else None,
        )
        center_scene_intrinsic = self._scale_intrinsic_for_resolution(
            center_intrinsic,
            eye_width,
            eye_height,
            scene_width,
            scene_height,
        )
        if background_only:
            center_scene_color, center_scene_depth = scene_renderer.render_background_eye(
                center_eye_pose_world,
                center_scene_intrinsic,
                width=scene_width,
                height=scene_height,
            )
        else:
            center_scene_color, center_scene_depth = scene_renderer.render_eye(
                center_eye_pose_world,
                center_scene_intrinsic,
                width=scene_width,
                height=scene_height,
            )
        center_scene_color_t, center_scene_depth_t = self._prepare_immersive_scene_frame_for_compose(
            center_scene_color,
            center_scene_depth,
            scene_height,
            scene_width,
            compose_cache=shared_scene_compose_cache,
        )
        center_scene_intrinsic_t = torch.as_tensor(
            center_scene_intrinsic,
            dtype=torch.float32,
            device=cfg.device,
        )
        center_w2c_cv_t = torch.as_tensor(
            self._camera_pose_world_to_cv_w2c(center_eye_pose_world),
            dtype=torch.float32,
            device=cfg.device,
        )
        center_c2w_cv_t = torch.linalg.inv(center_w2c_cv_t)

        object_points = gaussians.get_xyz.detach()
        object_support_center = (
            self._object_support_patch_center(object_points).detach().cpu().numpy().astype(np.float32)
        )
        object_bounds_min = (
            object_points.min(dim=0).values.detach().cpu().numpy().astype(np.float32)
        )
        object_bounds_max = (
            object_points.max(dim=0).values.detach().cpu().numpy().astype(np.float32)
        )
        shared_source_data = self._prepare_immersive_reproject_source_data(
            center_scene_color_t,
            center_scene_depth_t,
            center_scene_intrinsic_t,
            source_cache=None
            if reproject_caches is None
            else reproject_caches.get(
                "background_source" if background_only else "source"
            ),
        )
        debug = {
            "mode": str(scene_stereo_mode),
            "table_top_center": np.asarray(layout.table_top_center, dtype=np.float32).tolist(),
            "object_support_center": object_support_center.tolist(),
        }
        failures = []

        for eye_name, eye_sample, eye_pose_world in (
            ("left", left_eye_sample, left_eye_pose_world),
            ("right", right_eye_sample, right_eye_pose_world),
        ):
            if eye_sample is None or not eye_sample.pose_valid or eye_pose_world is None:
                continue
            intrinsic = self._eye_sample_intrinsic(eye_sample, eye_width, eye_height)
            scene_intrinsic = self._scale_intrinsic_for_resolution(
                intrinsic,
                eye_width,
                eye_height,
                scene_width,
                scene_height,
            )
            scene_w2c_cv = self._camera_pose_world_to_cv_w2c(eye_pose_world)
            repair_roi_bounds = self._compute_immersive_reproject_roi_bounds(
                layout,
                object_support_center,
                object_bounds_min,
                object_bounds_max,
                scene_intrinsic,
                scene_w2c_cv,
                scene_width,
                scene_height,
            )
            scene_color_t, scene_depth_t, scene_valid_t = self._reproject_immersive_scene_eye_frame(
                center_scene_color_t,
                center_scene_depth_t,
                center_scene_intrinsic,
                center_eye_pose_world,
                scene_intrinsic,
                eye_pose_world,
                scene_height,
                scene_width,
                reproject_cache=None
                if reproject_caches is None
                else reproject_caches.get(
                    f"background_{eye_name}" if background_only else eye_name
                ),
                shared_source_data=shared_source_data,
                source_intrinsic_t=center_scene_intrinsic_t,
                source_c2w_cv_t=center_c2w_cv_t,
                target_intrinsic_t=torch.as_tensor(
                    scene_intrinsic,
                    dtype=torch.float32,
                    device=cfg.device,
                ),
                target_w2c_cv_t=torch.as_tensor(
                    scene_w2c_cv,
                    dtype=torch.float32,
                    device=cfg.device,
                ),
                repair_roi_bounds=repair_roi_bounds,
                profile_key_prefix=(
                    "scene_reproject_background"
                    if background_only
                    else "scene_reproject"
                ),
            )
            table_projection = self._project_world_point_into_eye(
                layout.table_top_center,
                scene_intrinsic,
                scene_w2c_cv,
                scene_width,
                scene_height,
            )
            object_projection = self._project_world_point_into_eye(
                object_support_center,
                scene_intrinsic,
                scene_w2c_cv,
                scene_width,
                scene_height,
            )
            table_coverage = self._scene_valid_patch_coverage(
                scene_valid_t,
                table_projection["pixel"],
            )
            object_coverage = self._scene_valid_patch_coverage(
                scene_valid_t,
                object_projection["pixel"],
            )
            debug[f"{eye_name}_table_patch_coverage"] = table_coverage
            debug[f"{eye_name}_object_patch_coverage"] = object_coverage
            debug[f"{eye_name}_table_pixel"] = None if table_projection["pixel"] is None else table_projection["pixel"].astype(np.float32).tolist()
            debug[f"{eye_name}_object_pixel"] = None if object_projection["pixel"] is None else object_projection["pixel"].astype(np.float32).tolist()
            coverage_min = float(self.IMMERSIVE_REPROJECT_STARTUP_MIN_PATCH_COVERAGE)
            if background_only:
                roi_coverage = self._scene_valid_roi_coverage(
                    scene_valid_t,
                    repair_roi_bounds,
                )
                debug[f"{eye_name}_roi_coverage"] = roi_coverage
                if (
                    not object_projection["in_bounds"]
                    or object_coverage < coverage_min
                    or roi_coverage < coverage_min
                ):
                    failures.append(
                        {
                            "eye": eye_name,
                            "object_in_bounds": bool(object_projection["in_bounds"]),
                            "object_patch_coverage": object_coverage,
                            "roi_coverage": roi_coverage,
                        }
                    )
            else:
                if (
                    not table_projection["in_bounds"]
                    or not object_projection["in_bounds"]
                    or table_coverage < coverage_min
                    or object_coverage < coverage_min
                ):
                    failures.append(
                        {
                            "eye": eye_name,
                            "table_in_bounds": bool(table_projection["in_bounds"]),
                            "object_in_bounds": bool(object_projection["in_bounds"]),
                            "table_patch_coverage": table_coverage,
                            "object_patch_coverage": object_coverage,
                        }
                    )

        debug["failures"] = failures
        return len(failures) == 0, debug

    def _initialize_immersive_balanced_render_profile_frame(self, render_profile_frame):
        if render_profile_frame is None:
            return
        for key in (
            "scene_render_center_wall",
            "scene_render_background_center_wall",
            "scene_render_left_wall",
            "scene_render_right_wall",
            "scene_prepare_background_eye_wall",
            "scene_render_far_center_wall",
            "scene_render_near_center_wall",
            "scene_render_side_left_wall",
            "scene_render_side_right_wall",
            "scene_side_roi_left_ratio",
            "scene_side_roi_right_ratio",
            "scene_side_strip_left_width_ratio",
            "scene_side_strip_right_width_ratio",
            "scene_side_fullframe_fallback_left_ratio",
            "scene_side_fullframe_fallback_right_ratio",
            "scene_compose_side_left_cuda",
            "scene_compose_side_right_cuda",
            "scene_reproject_background_left_cuda",
            "scene_reproject_background_right_cuda",
            "scene_reproject_background_hole_fill_left_cuda",
            "scene_reproject_background_hole_fill_right_cuda",
            "scene_reproject_background_valid_pre_left_ratio",
            "scene_reproject_background_valid_post_left_ratio",
            "scene_reproject_background_valid_pre_right_ratio",
            "scene_reproject_background_valid_post_right_ratio",
            "scene_reproject_background_roi_pre_left_ratio",
            "scene_reproject_background_roi_post_left_ratio",
            "scene_reproject_background_roi_pre_right_ratio",
            "scene_reproject_background_roi_post_right_ratio",
            "scene_warp_far_left_cuda",
            "scene_warp_far_right_cuda",
            "scene_warp_near_left_cuda",
            "scene_warp_near_right_cuda",
            "scene_side_warp_left_used",
            "scene_side_warp_right_used",
            "scene_side_render_fallback_left_used",
            "scene_side_render_fallback_right_used",
            "scene_timewarp_applied",
            "scene_timewarp_fallback_left_used",
            "scene_timewarp_fallback_right_used",
            "scene_timewarp_gpu_ms",
            "static_scene_worker_wall_ms",
            "simulation_lbs_wall_ms",
            "overlap_wait_wall_ms",
            "scene_pose_staleness_ms_at_publish",
            "scene_pose_staleness_savings_ms",
            "render_sample_id",
            "publish_sample_id",
        ):
            render_profile_frame[key] = 0.0

    def _get_immersive_balanced_background_compose_caches(self, balanced_runtime_state):
        if balanced_runtime_state is None:
            return {"left": {}, "right": {}}
        return balanced_runtime_state.setdefault(
            "background_compose_caches",
            {"left": {}, "right": {}},
        )

    def _render_immersive_balanced_background_eye(
        self,
        scene_renderer,
        eye_label,
        eye_pose_world,
        eye_intrinsic,
        eye_width,
        eye_height,
        scene_width,
        scene_height,
        compose_cache,
        render_profile_frame=None,
    ):
        scene_intrinsic = self._scale_intrinsic_for_resolution(
            eye_intrinsic,
            eye_width,
            eye_height,
            scene_width,
            scene_height,
        )
        eye_scene_render_start = (
            time.perf_counter() if render_profile_frame is not None else None
        )
        eye_scene_color, eye_scene_depth = scene_renderer.render_background_eye(
            eye_pose_world,
            scene_intrinsic,
            width=scene_width,
            height=scene_height,
        )
        if eye_scene_render_start is not None:
            self._render_profile_add_wall_time(
                render_profile_frame,
                f"scene_render_{eye_label}_wall",
                time.perf_counter() - eye_scene_render_start,
            )
        background_prepare_start = (
            time.perf_counter() if render_profile_frame is not None else None
        )
        background_color_t, background_depth_t = (
            self._prepare_immersive_scene_frame_for_compose(
                eye_scene_color,
                eye_scene_depth,
                eye_height,
                eye_width,
                compose_cache=compose_cache,
            )
        )
        if background_prepare_start is not None:
            self._render_profile_add_wall_time(
                render_profile_frame,
                "scene_prepare_background_eye_wall",
                time.perf_counter() - background_prepare_start,
            )
        return background_color_t, background_depth_t

    def _render_immersive_balanced_table_scene_for_eye(
        self,
        scene_renderer,
        table_world_bounds,
        eye_label,
        eye_pose_world,
        eye_intrinsic,
        eye_width,
        eye_height,
        background_mode,
        background_color_t,
        background_depth_t,
        center_eye_pose_world,
        center_intrinsic,
        balanced_runtime_state,
        reproject_caches,
        render_profile_frame,
        shared_background_source_data,
        source_intrinsic_t,
        source_c2w_cv_t,
        table_roi_state,
        table_roi_render_scale,
        background_compose_cache=None,
    ):
        if background_mode == "mono_center_background":
            repaired_background_color_t, repaired_background_depth_t = (
                self._repair_immersive_balanced_background_eye(
                    scene_renderer,
                    eye_label,
                    eye_pose_world,
                    eye_intrinsic,
                    eye_width,
                    eye_height,
                    background_color_t,
                    background_depth_t,
                    center_eye_pose_world,
                    center_intrinsic,
                    balanced_runtime_state=balanced_runtime_state,
                    reproject_caches=reproject_caches,
                    render_profile_frame=render_profile_frame,
                    shared_source_data=shared_background_source_data,
                    source_intrinsic_t=source_intrinsic_t,
                    source_c2w_cv_t=source_c2w_cv_t,
                )
            )
        else:
            repaired_background_color_t = background_color_t
            repaired_background_depth_t = background_depth_t

        eye_w2c_cv = self._camera_pose_world_to_cv_w2c(eye_pose_world)
        table_roi_bounds = None
        table_roi_ratio = 1.0
        table_fullframe_fallback = True
        if table_world_bounds is not None:
            (
                table_roi_bounds,
                table_roi_ratio,
                table_fullframe_fallback,
                _,
            ) = self._resolve_immersive_balanced_table_render_roi(
                table_world_bounds[0],
                table_world_bounds[1],
                eye_intrinsic,
                eye_w2c_cv,
                eye_width,
                eye_height,
                prev_bounds=table_roi_state.get(eye_label),
            )
        table_roi_state[eye_label] = None if table_fullframe_fallback else table_roi_bounds
        if render_profile_frame is not None:
            render_profile_frame[f"scene_table_roi_{eye_label}_ratio"] = float(
                table_roi_ratio
            )
            render_profile_frame["scene_table_roi_supersample_scale"] = float(
                table_roi_render_scale
            )
            render_profile_frame[
                f"scene_table_fullframe_fallback_{eye_label}_ratio"
            ] = 1.0 if table_fullframe_fallback else 0.0

        table_render_start = (
            time.perf_counter() if render_profile_frame is not None else None
        )
        table_coverage_mask = None
        if table_fullframe_fallback:
            table_color, table_depth = scene_renderer.render_table_eye(
                eye_pose_world,
                eye_intrinsic,
                width=eye_width,
                height=eye_height,
            )
        else:
            table_color_render, table_depth_render, table_render_info = (
                scene_renderer.render_table_eye_roi(
                    eye_pose_world,
                    eye_intrinsic,
                    table_roi_bounds,
                    render_scale=table_roi_render_scale,
                    return_render_info=True,
                )
            )
            table_color, table_depth, table_coverage_mask = (
                self._downsample_immersive_supersampled_overlay_patch(
                    table_color_render,
                    table_depth_render,
                    int(table_render_info["roi_height"]),
                    int(table_render_info["roi_width"]),
                )
            )
        if table_render_start is not None:
            self._render_profile_add_wall_time(
                render_profile_frame,
                f"scene_render_table_{eye_label}_wall",
                time.perf_counter() - table_render_start,
            )

        table_compose_span = self._render_profile_begin_cuda_span(
            render_profile_frame,
            f"scene_compose_table_{eye_label}_cuda",
        )
        scene_color_t, scene_depth_t = self._compose_immersive_scene_layers(
            repaired_background_color_t,
            repaired_background_depth_t,
            table_color,
            table_depth,
            target_height=eye_height,
            target_width=eye_width,
            background_cache=background_compose_cache,
            overlay_roi_bounds=None if table_fullframe_fallback else table_roi_bounds,
            overlay_coverage_mask=table_coverage_mask,
        )
        self._render_profile_end_cuda_span(
            render_profile_frame,
            table_compose_span,
        )
        return scene_color_t, scene_depth_t

    @torch.no_grad()
    def _render_immersive_table_roi_scene_frames(
        self,
        scene_renderer,
        layout,
        object_support_center_world,
        object_bounds_min_world,
        object_bounds_max_world,
        left_eye_pose_world,
        right_eye_pose_world,
        left_intrinsic,
        right_intrinsic,
        eye_width,
        eye_height,
        scene_width,
        scene_height,
        shared_scene_compose_cache=None,
        reproject_caches=None,
        render_profile_frame=None,
    ):
        _ = layout
        _ = object_support_center_world
        _ = object_bounds_min_world
        _ = object_bounds_max_world
        render_plan = self._build_immersive_balanced_scene_render_plan(
            scene_renderer,
            left_eye_pose_world,
            right_eye_pose_world,
            left_intrinsic,
            right_intrinsic,
            eye_width,
            eye_height,
            scene_width,
            scene_height,
            render_profile_frame=render_profile_frame,
        )
        render_outputs = _execute_immersive_balanced_scene_render_plan(
            scene_renderer,
            render_plan,
        )
        return self._assemble_immersive_balanced_scene_from_render_outputs(
            scene_renderer,
            render_plan,
            render_outputs,
            eye_width,
            eye_height,
            shared_scene_compose_cache=shared_scene_compose_cache,
            reproject_caches=reproject_caches,
            render_profile_frame=render_profile_frame,
        )

    @torch.no_grad()
    def _render_immersive_scene_frames_for_mode(
        self,
        scene_renderer,
        scene_stereo_mode,
        layout,
        object_support_center_world,
        object_bounds_min_world,
        object_bounds_max_world,
        left_eye_pose_world,
        right_eye_pose_world,
        left_intrinsic,
        right_intrinsic,
        eye_width,
        eye_height,
        scene_width,
        scene_height,
        shared_scene_compose_cache=None,
        reproject_caches=None,
        render_profile_frame=None,
    ):
        if scene_stereo_mode == "per_eye":
            left_scene_render_start = (
                time.perf_counter() if render_profile_frame is not None else None
            )
            left_scene_intrinsic = self._scale_intrinsic_for_resolution(
                left_intrinsic,
                eye_width,
                eye_height,
                scene_width,
                scene_height,
            )
            left_scene_color, left_scene_depth = scene_renderer.render_eye(
                left_eye_pose_world,
                left_scene_intrinsic,
                width=scene_width,
                height=scene_height,
            )
            if left_scene_render_start is not None:
                self._render_profile_add_wall_time(
                    render_profile_frame,
                    "scene_render_left_wall",
                    time.perf_counter() - left_scene_render_start,
                )

            right_scene_render_start = (
                time.perf_counter() if render_profile_frame is not None else None
            )
            right_scene_intrinsic = self._scale_intrinsic_for_resolution(
                right_intrinsic,
                eye_width,
                eye_height,
                scene_width,
                scene_height,
            )
            right_scene_color, right_scene_depth = scene_renderer.render_eye(
                right_eye_pose_world,
                right_scene_intrinsic,
                width=scene_width,
                height=scene_height,
            )
            if right_scene_render_start is not None:
                self._render_profile_add_wall_time(
                    render_profile_frame,
                    "scene_render_right_wall",
                    time.perf_counter() - right_scene_render_start,
                )
            return left_scene_color, left_scene_depth, right_scene_color, right_scene_depth

        if scene_stereo_mode == self.IMMERSIVE_BALANCED_INTERNAL_STEREO_MODE:
            return self._render_immersive_table_roi_scene_frames(
                scene_renderer,
                layout,
                object_support_center_world,
                object_bounds_min_world,
                object_bounds_max_world,
                left_eye_pose_world,
                right_eye_pose_world,
                left_intrinsic,
                right_intrinsic,
                eye_width,
                eye_height,
                scene_width,
                scene_height,
                shared_scene_compose_cache=shared_scene_compose_cache,
                reproject_caches=reproject_caches,
                render_profile_frame=render_profile_frame,
            )

        center_eye_pose_world, center_intrinsic = self._build_immersive_center_scene_view(
            left_eye_pose_world,
            right_eye_pose_world,
            left_intrinsic,
            right_intrinsic,
        )
        center_scene_intrinsic = self._scale_intrinsic_for_resolution(
            center_intrinsic,
            eye_width,
            eye_height,
            scene_width,
            scene_height,
        )
        center_scene_render_start = (
            time.perf_counter() if render_profile_frame is not None else None
        )
        center_scene_color, center_scene_depth = scene_renderer.render_eye(
            center_eye_pose_world,
            center_scene_intrinsic,
            width=scene_width,
            height=scene_height,
        )
        if center_scene_render_start is not None:
            self._render_profile_add_wall_time(
                render_profile_frame,
                "scene_render_center_wall",
                time.perf_counter() - center_scene_render_start,
            )

        if scene_stereo_mode == "mono_head_center":
            shared_scene_color_t, shared_scene_depth_t = self._prepare_immersive_scene_frame_for_compose(
                center_scene_color,
                center_scene_depth,
                eye_height,
                eye_width,
                compose_cache=shared_scene_compose_cache,
            )
            return (
                shared_scene_color_t,
                shared_scene_depth_t,
                shared_scene_color_t,
                shared_scene_depth_t,
            )

        if scene_stereo_mode != "reproject_from_center":
            raise ValueError(f"Unsupported immersive scene stereo mode: {scene_stereo_mode}")

        center_scene_color_t, center_scene_depth_t = self._prepare_immersive_scene_frame_for_compose(
            center_scene_color,
            center_scene_depth,
            scene_height,
            scene_width,
            compose_cache=shared_scene_compose_cache,
        )
        center_scene_intrinsic_t = torch.as_tensor(
            center_scene_intrinsic,
            dtype=torch.float32,
            device=cfg.device,
        )
        shared_source_data = self._prepare_immersive_reproject_source_data(
            center_scene_color_t,
            center_scene_depth_t,
            center_scene_intrinsic_t,
            source_cache=None if reproject_caches is None else reproject_caches.get("source"),
        )
        center_w2c_cv_t = torch.as_tensor(
            self._camera_pose_world_to_cv_w2c(center_eye_pose_world),
            dtype=torch.float32,
            device=cfg.device,
        )
        center_c2w_cv_t = torch.linalg.inv(center_w2c_cv_t)
        left_scene_intrinsic = self._scale_intrinsic_for_resolution(
            left_intrinsic,
            eye_width,
            eye_height,
            scene_width,
            scene_height,
        )
        right_scene_intrinsic = self._scale_intrinsic_for_resolution(
            right_intrinsic,
            eye_width,
            eye_height,
            scene_width,
            scene_height,
        )
        left_scene_w2c_cv = self._camera_pose_world_to_cv_w2c(left_eye_pose_world)
        right_scene_w2c_cv = self._camera_pose_world_to_cv_w2c(right_eye_pose_world)
        left_repair_roi_bounds = self._compute_immersive_reproject_roi_bounds(
            layout,
            object_support_center_world,
            object_bounds_min_world,
            object_bounds_max_world,
            left_scene_intrinsic,
            left_scene_w2c_cv,
            scene_width,
            scene_height,
        )
        right_repair_roi_bounds = self._compute_immersive_reproject_roi_bounds(
            layout,
            object_support_center_world,
            object_bounds_min_world,
            object_bounds_max_world,
            right_scene_intrinsic,
            right_scene_w2c_cv,
            scene_width,
            scene_height,
        )
        left_scene_color_t, left_scene_depth_t, _ = self._reproject_immersive_scene_eye_frame(
            center_scene_color_t,
            center_scene_depth_t,
            center_scene_intrinsic,
            center_eye_pose_world,
            left_scene_intrinsic,
            left_eye_pose_world,
            scene_height,
            scene_width,
            render_profile_frame=render_profile_frame,
            eye_label="left",
            reproject_cache=None if reproject_caches is None else reproject_caches.get("left"),
            shared_source_data=shared_source_data,
            source_intrinsic_t=center_scene_intrinsic_t,
            source_c2w_cv_t=center_c2w_cv_t,
            target_intrinsic_t=torch.as_tensor(
                left_scene_intrinsic,
                dtype=torch.float32,
                device=cfg.device,
            ),
            target_w2c_cv_t=torch.as_tensor(
                left_scene_w2c_cv,
                dtype=torch.float32,
                device=cfg.device,
            ),
            repair_roi_bounds=left_repair_roi_bounds,
        )
        right_scene_color_t, right_scene_depth_t, _ = self._reproject_immersive_scene_eye_frame(
            center_scene_color_t,
            center_scene_depth_t,
            center_scene_intrinsic,
            center_eye_pose_world,
            right_scene_intrinsic,
            right_eye_pose_world,
            scene_height,
            scene_width,
            render_profile_frame=render_profile_frame,
            eye_label="right",
            reproject_cache=None if reproject_caches is None else reproject_caches.get("right"),
            shared_source_data=shared_source_data,
            source_intrinsic_t=center_scene_intrinsic_t,
            source_c2w_cv_t=center_c2w_cv_t,
            target_intrinsic_t=torch.as_tensor(
                right_scene_intrinsic,
                dtype=torch.float32,
                device=cfg.device,
            ),
            target_w2c_cv_t=torch.as_tensor(
                right_scene_w2c_cv,
                dtype=torch.float32,
                device=cfg.device,
            ),
            repair_roi_bounds=right_repair_roi_bounds,
        )
        return (
            left_scene_color_t,
            left_scene_depth_t,
            right_scene_color_t,
            right_scene_depth_t,
        )

@torch.no_grad()
def get_shadow_masks_batched_downsampled(
    points,          # (N,3) float CUDA
    intrinsic_T,     # (3,3) float CUDA
    w2c_T,           # (4,4) float CUDA
    W: int, H: int,
    image_mask,      # (H,W) bool CUDA
    lights,          # (L,3) float CUDA
    inv_Lz,          # (L,)  float CUDA
    kernel_size: int = 7,     # original full-res k
    scale: int = 2,           # 2 or 4 are typical
    use_half: bool = False,
    upsample_mode: str = "bilinear",   # or "nearest"
    post_blur: bool = False,           # optional small blur after upsample
):
    """
    Returns: (L,H,W) bool CUDA  — masks computed at low-res and upsampled.
    """
    assert scale >= 1
    if scale == 1:
        # fall back to your existing high-res function if you like
        raise NotImplementedError("Use non-downsampled path when scale=1.")

    device = points.device
    dtype  = torch.float16 if use_half else points.dtype

    points      = points.to(dtype)
    intrinsic_T = intrinsic_T.to(dtype)
    w2c_T       = w2c_T.to(dtype)
    lights      = lights.to(dtype)
    inv_Lz      = inv_Lz.to(dtype)

    Hs, Ws = H // scale, W // scale
    N = points.shape[0]
    L = lights.shape[0]

    # ---- 1) Factor projection once: P = w2c[:,:3] @ K
    P = w2c_T[:, :3] @ intrinsic_T                      # (4,3)

    ones  = torch.ones((N,1), device=device, dtype=dtype)
    base4 = torch.cat([points, ones], dim=1)            # (N,4)
    pix3_base = base4 @ P                               # (N,3)

    zeros1 = torch.zeros((L,1), device=device, dtype=dtype)
    light4 = torch.cat([lights, zeros1], dim=1)         # (L,4)
    pix3_dir = light4 @ P                               # (L,3)

    # ---- 2) Project all lights; then scale pixel coords by 1/scale
    t_base = -points[:, 2].to(dtype)                    # (N,)
    t = inv_Lz.view(L, 1) * t_base.view(1, N)           # (L,N)
    pix3 = pix3_base.unsqueeze(0) + t.unsqueeze(-1) * pix3_dir.unsqueeze(1)  # (L,N,3)
    z = torch.clamp(pix3[..., 2:3], min=1e-12)
    pix = pix3[..., :2] / z                             # (L,N,2)

    # Downsampled pixels (divide by scale before flooring)
    x = torch.floor(pix[..., 0] / scale).to(torch.int64)
    y = torch.floor(pix[..., 1] / scale).to(torch.int64)
    valid = (x >= 0) & (x < Ws) & (y >= 0) & (y < Hs)

    idx = (y * Ws + x)
    #idx = torch.where(valid, idx, torch.zeros_like(idx))
    idx = idx.masked_fill_(~valid, 0)

    # ---- 3) Rasterize in low-res space (L, Hs*Ws)
    shadow_flat = torch.zeros((L, Hs * Ws), device=device, dtype=torch.float32)
    if hasattr(shadow_flat, "scatter_reduce_"):
        src = valid.to(torch.float32)
        shadow_flat.scatter_reduce_(dim=1, index=idx, src=src, reduce="amax", include_self=False)
    else:
        vmask = valid.view(-1)
        rows = torch.arange(L, device=device).view(L, 1).expand_as(idx).view(-1)[vmask]
        cols = idx.view(-1)[vmask]
        shadow_flat.index_put_((rows, cols), torch.ones_like(cols, dtype=torch.float32), accumulate=True)
        shadow_flat.clamp_(0, 1.0)

    shadow_lo = shadow_flat.view(1, L, Hs, Ws)   # N=1,C=L, Hs,Ws
    shadow_lo = shadow_lo.contiguous(memory_format=torch.channels_last)

    # ---- 4) Morphology at low-res; scale kernel to keep physical size similar
    k_lo = max(1, int(round(kernel_size / scale)) | 1)
    # separable k×1 then 1×k (flat structuring element)
    shadow_lo = F.max_pool2d(shadow_lo, kernel_size=(k_lo, 1), stride=1, padding=(k_lo // 2, 0))
    shadow_lo = F.max_pool2d(shadow_lo, kernel_size=(1, k_lo), stride=1, padding=(0, k_lo // 2))
    #pyh removed erosion step to get softer shadows
    shadow_lo = 1.0 - F.max_pool2d(1.0 - shadow_lo, kernel_size=(k_lo, 1), stride=1, padding=(k_lo // 2, 0))
    shadow_lo = 1.0 - F.max_pool2d(1.0 - shadow_lo, kernel_size=(1, k_lo), stride=1, padding=(0, k_lo // 2))

    # ---- 5) Apply occlusion at low-res (downsample occ by avg-pooling)
    # (avg-pool then threshold to get a conservative occlusion)
    occ = image_mask.view(1, 1, H, W).float()
    occ_lo = F.avg_pool2d(occ, kernel_size=scale, stride=scale)     # (1,1,Hs,Ws)
    occ_lo = (occ_lo > 0.5).to(shadow_lo.dtype)                      # conservative block
    shadow_lo.mul_(1.0 - occ_lo)

    # ---- 6) Upsample back to (H,W) and re-threshold
    shadow_hi = F.interpolate(shadow_lo, size=(H, W), mode=upsample_mode, align_corners=False if upsample_mode=="bilinear" else None)
    if post_blur:
        # simple 3x3 box blur per-channel (approximate soft edges)
        shadow_hi = F.avg_pool2d(shadow_hi, kernel_size=3, stride=1, padding=1)
    masks = (shadow_hi > 0.5).squeeze(0)   # (L,H,W) bool

    return masks
