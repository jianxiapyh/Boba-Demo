#this is batched version incorporate latest single instance changes (slimmer lbs + mass node morton reordering + spring clustering)
from qqtt.data import RealData
from qqtt.utils import logger, cfg
from qqtt.model.diff_simulator import (
    SpringMassSystemWarp,
)
import csv
import json
import open3d as o3d
import numpy as np
import torch
import os
import warp as wp
import pickle
import cv2
import heapq

import torchvision
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.scene.cameras import Camera
from gaussian_splatting.gaussian_renderer import render as render_gaussian
from gaussian_splatting.dynamic_utils import (
    lbs_with_rotation_reuse,
    build_rotation_reuse_cache,
    knn_weights_sparse,
    get_topk_indices,
)
from gaussian_splatting.utils.graphics_utils import focal2fov
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
from pycuda.gl import RegisteredBuffer, graphics_map_flags

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
    LIVE_CONTROLLER_TRANSLATION_SCALE = 1.0
    LIVE_CONTROLLER_HIT_WORLD_RADIUS = 0.03
    LIVE_CONTROLLER_ATTACH_MAX_REST_LENGTH = 0.01
    LIVE_CONTROLLER_PREDEFINED_ANCHOR_NODE_COUNT = 96
    LIVE_CONTROLLER_PREDEFINED_ANCHOR_RADIUS_SCALE = 1.75
    LIVE_CONTROLLER_PREDEFINED_ANCHOR_MIN_RADIUS = 0.05
    LIVE_CONTROLLER_PREVIEW_RADIUS = 3
    LIVE_CONTROLLER_PREVIEW_SELECTED_RADIUS = 5
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
    LIVE_CONTROLLER_ACTIVE_DEBUG_LOG_INTERVAL = 20
    LIVE_CONTROLLER_ACTIVE_MOTION_EPS = 1e-3
    LIVE_CONTROLLER_ACTIVE_TARGET_EPS = 1e-4
    LIVE_CONTROLLER_MULTI_POINTS_BACK_DEPTH_THRESHOLD = 0.015
    LIVE_CONTROLLER_MULTI_POINTS_BACK_PENALTY = 4.0
    LIVE_CONTROLLER_MULTI_POINTS_FETCH_SCALE = 4
    QUEST_PRIMARY_COMPOSITE_WIDTH = 2064
    IMMERSIVE_EYE_WIDTH = 1024
    IMMERSIVE_EYE_HEIGHT = 1024
    IMMERSIVE_RENDER_PRESET_DEFAULTS = {
        "quality": {
            "scene_render_scale": 1.0,
            "scene_stereo_mode": "per_eye",
            "overlay_mode": "full",
            "lighting_mode": "full",
        },
        "balanced": {
            "scene_render_scale": 0.75,
            "scene_stereo_mode": "per_eye",
            "overlay_mode": "minimal",
            "lighting_mode": "simple",
        },
        "performance": {
            "scene_render_scale": 0.625,
            "scene_stereo_mode": "reproject_from_center",
            "overlay_mode": "minimal",
            "lighting_mode": "simple",
        },
    }
    IMMERSIVE_SCENE_RENDER_SCALE_MIN = 0.25
    IMMERSIVE_SCENE_REST_SETTLE_STEPS = 90
    IMMERSIVE_SCENE_REST_VELOCITY_EPS = 0.035
    IMMERSIVE_SCENE_REST_POSITION_EPS = 0.015
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
            # Get the object points and controller points
            #🆕 NEW: use self.controller_points_group instead
            #self.controller_points = self.dataset.controller_points
            self.structure_points = self.dataset.structure_points
            self.num_all_points = self.dataset.num_all_points
        elif cfg.data_type == "synthetic":
            print(f"synthetic data detected")
            import pdb
            pdb.set_trace()
        else:
            raise ValueError(f"Data type {cfg.data_type} not supported")
        
        #🆕 NEW: instead of loading controller from final_data.pkl, we are loading from multi_ctrl.pkl
        data_dir = Path(cfg.data_path).parent 
        controller_group_path = data_dir / "multi_ctrls.pkl" 
        self.controller_points_group = self.load_controller_points_group_pkl(controller_group_path, device=cfg.device)
        # testing if the first controller points of all instances have the same starting point
        self.check_controller_group_same_start(self.controller_points_group, atol=1e-5)
        self.frame_len = self.controller_points_group.shape[1]
        self.num_input_trajectories = self.controller_points_group.shape[0]
        #🆕 CHANGED:
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

    def load_controller_points_group_pkl(self, pkl_path: str, device="cuda"):
        """
        Load controller_points_group from a .pkl and return a single batched tensor.

        Returns
        -------
        controller_points_group : torch.Tensor
            Shape (N, T, C, 3), dtype float32, on `device`.
            N = number of instances (len(controller_points_group) in the file)

        Notes
        -----
        Expects pkl root is a dict with key "controller_points_group" that is a list of
        numpy arrays, each with shape (T, C, 3).
        """
        with open(pkl_path, "rb") as f:
            root = pickle.load(f)

        if not isinstance(root, dict):
            raise TypeError(f"PKL root must be dict, got {type(root)}")

        if "controller_points_group" not in root:
            raise KeyError(f"Missing key 'controller_points_group' in {pkl_path}. Keys={list(root.keys())}")

        group = root["controller_points_group"]
        if not isinstance(group, list) or len(group) == 0:
            raise TypeError(f"'controller_points_group' must be a non-empty list, got {type(group)} len={len(group) if hasattr(group,'__len__') else '??'}")

        # Validate + normalize
        T0 = C0 = None
        tensors = []
        for i, arr in enumerate(group):
            arr = np.asarray(arr)
            if arr.ndim != 3 or arr.shape[-1] != 3:
                raise ValueError(f"controller_points_group[{i}] shape {arr.shape}, expected (T, C, 3)")
            T, C, _ = arr.shape
            if T0 is None:
                T0, C0 = T, C
            else:
                if (T, C) != (T0, C0):
                    raise ValueError(
                        f"controller_points_group[{i}] has (T,C)=({T},{C}) but expected ({T0},{C0}). "
                        "If you want variable lengths, return a list instead of stacking."
                    )

            t = torch.from_numpy(arr.astype(np.float32, copy=False)).to(device=device)
            tensors.append(t)

        controller_points_group = torch.stack(tensors, dim=0)  # (N, T, C, 3)
        return controller_points_group

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

    def _build_predefined_interaction_anchors(self, object_points, intrinsic, w2c):
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
        upper_target = center_pixel + object_points.new_tensor([0.0, -0.22 * spread_y])
        upper_torso_mask = (
            (torch.abs(pixels[:, 0] - center_pixel[0]) <= max(spread_x * 0.22, 8.0))
            & upper_mask
        )
        upper_torso_score = torch.linalg.norm(pixels - upper_target.unsqueeze(0), dim=1)

        anchor_specs = [
            ("left_leg", left_mask & lower_mask, center_score, True),
            ("right_leg", right_mask & lower_mask, center_score, True),
            ("left_arm", left_mask & upper_mask, center_score, True),
            ("right_arm", right_mask & upper_mask, center_score, True),
            ("upper_torso", upper_torso_mask, upper_torso_score, False),
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
            region_indices = self._graph_region_from_seed(
                seed_idx,
                region_node_count,
                object_points,
            )
            region_points = object_points[region_indices]
            center_world = region_points.mean(dim=0)
            radius = torch.linalg.norm(
                region_points - center_world.unsqueeze(0), dim=1
            ).max()
            anchors.append(
                {
                    "name": name,
                    "seed_index": seed_idx,
                    "region_indices": region_indices,
                    "rest_center_world": center_world,
                    "rest_radius": float(radius.item()),
                }
            )
        return anchors

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

    def _select_predefined_interaction_anchor(
        self,
        hit_world,
        anchor_states,
        require_selection_radius=True,
    ):
        if hit_world is None or not anchor_states:
            return None

        best_anchor = None
        best_distance = None
        for anchor in anchor_states:
            distance = float(torch.linalg.norm(anchor["center_world"] - hit_world).item())
            if best_distance is None or distance < best_distance:
                best_anchor = anchor
                best_distance = distance

        if best_anchor is None or best_distance is None:
            return None
        if require_selection_radius and best_distance > best_anchor["selection_radius"]:
            return None
        return best_anchor

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

    def _update_controller_anchor_preview_state(
        self,
        source,
        controller_world,
        overlay,
        anchor_states,
        preview_state,
        cycle_edge,
        interaction_state,
    ):
        state = preview_state[source]
        if interaction_state is not None:
            state["visible"] = True
            state["selected_anchor_name"] = interaction_state.get("anchor_name")
            return self._anchor_state_by_name(anchor_states, state["selected_anchor_name"])

        if cycle_edge and anchor_states:
            anchor_names = [anchor["name"] for anchor in anchor_states]
            if not state["visible"]:
                state["visible"] = True
                reference_world = None
                if overlay is not None:
                    reference_world = overlay.get("hit_world")
                    if reference_world is None:
                        reference_world = overlay.get("ray_end_world")
                selected_anchor = self._select_predefined_interaction_anchor(
                    reference_world,
                    anchor_states,
                    require_selection_radius=False,
                )
                if selected_anchor is None:
                    selected_anchor = anchor_states[0]
                state["selected_anchor_name"] = selected_anchor["name"]
            else:
                current_name = state.get("selected_anchor_name")
                if current_name not in anchor_names:
                    next_index = 0
                else:
                    next_index = (anchor_names.index(current_name) + 1) % len(anchor_names)
                state["selected_anchor_name"] = anchor_names[next_index]

        if not state["visible"]:
            return None

        selected_anchor = self._anchor_state_by_name(
            anchor_states, state.get("selected_anchor_name")
        )
        if selected_anchor is None and anchor_states:
            selected_anchor = anchor_states[0]
            state["selected_anchor_name"] = selected_anchor["name"]
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
        default_anchor_names = {"left": "left_arm", "right": "right_arm"}

        for source in ("left", "right"):
            source_index = self._controller_source_index(source)
            default_anchor = self._anchor_state_by_name(
                anchor_states, default_anchor_names.get(source)
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
        if origin_world is None or direction_world is None or not anchor_states:
            return None

        best_anchor = None
        best_perpendicular = None
        best_along = None
        for anchor_state in anchor_states:
            delta = anchor_state["center_world"] - origin_world
            along = float(torch.dot(delta, direction_world).item())
            along = max(along, 0.0)
            closest = origin_world + direction_world * along
            perpendicular = float(
                torch.linalg.norm(anchor_state["center_world"] - closest).item()
            )
            if (
                best_anchor is None
                or perpendicular < best_perpendicular
                or (
                    np.isclose(perpendicular, best_perpendicular)
                    and along < best_along
                )
            ):
                best_anchor = anchor_state
                best_perpendicular = perpendicular
                best_along = along
        return best_anchor

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
        basis_override=None,
        collect_reset_edges=True,
        alignment_pose_role="selected",
        controller_position_pose_role="selected",
        controller_ray_pose_role=None,
    ):
        controller_reset_sources = []
        if latest_controller_sample is None:
            return {
                "alignment": live_controller_alignment,
                "alignment_mode": live_controller_alignment_mode,
                "left_controller": None,
                "right_controller": None,
                "reset_sources": controller_reset_sources,
                "alignment_acquired": False,
            }

        previous_alignment_available = live_controller_alignment is not None
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
        preview_selected_translation_only=False,
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
                preview_selected_translation_only=preview_selected_translation_only,
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
            controller_anchor_preview_state[source]["visible"] = False
            controller_anchor_preview_state[source]["selected_anchor_name"] = None

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
        for key in ordered_keys:
            values = render_profile_series.get(key, [])
            if not values:
                continue
            values_np = np.asarray(values, dtype=np.float64)
            if key.endswith("_gib"):
                lines.append(
                    f"{key}: "
                    f"avg={np.mean(values_np):.2f} GiB "
                    f"p95={np.percentile(values_np, 95):.2f} GiB "
                    f"max={np.max(values_np):.2f} GiB"
                )
            elif key.endswith("_ratio"):
                lines.append(
                    f"{key}: "
                    f"avg={np.mean(values_np) * 100.0:.1f}% "
                    f"p95={np.percentile(values_np, 95) * 100.0:.1f}% "
                    f"max={np.max(values_np) * 100.0:.1f}%"
                )
            else:
                lines.append(
                    f"{key}: "
                    f"avg={np.mean(values_np) * 1000.0:.2f} ms "
                    f"p95={np.percentile(values_np, 95) * 1000.0:.2f} ms "
                    f"max={np.max(values_np) * 1000.0:.2f} ms"
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
            f"reproject=L{frame_profile.get('scene_reproject_left_cuda', 0.0) * 1000.0:.2f}/"
            f"R{frame_profile.get('scene_reproject_right_cuda', 0.0) * 1000.0:.2f}ms "
            f"cov=L{frame_profile.get('scene_reproject_valid_pre_left_ratio', 0.0) * 100.0:.0f}->"
            f"{frame_profile.get('scene_reproject_valid_post_left_ratio', 0.0) * 100.0:.0f}%/"
            f"R{frame_profile.get('scene_reproject_valid_pre_right_ratio', 0.0) * 100.0:.0f}->"
            f"{frame_profile.get('scene_reproject_valid_post_right_ratio', 0.0) * 100.0:.0f}% "
            f"roi=L{frame_profile.get('scene_reproject_roi_pre_left_ratio', 0.0) * 100.0:.0f}->"
            f"{frame_profile.get('scene_reproject_roi_post_left_ratio', 0.0) * 100.0:.0f}%/"
            f"R{frame_profile.get('scene_reproject_roi_pre_right_ratio', 0.0) * 100.0:.0f}->"
            f"{frame_profile.get('scene_reproject_roi_post_right_ratio', 0.0) * 100.0:.0f}% "
            f"gaussian=L{frame_profile.get('gaussian_render_left_cuda', 0.0) * 1000.0:.2f}/"
            f"R{frame_profile.get('gaussian_render_right_cuda', 0.0) * 1000.0:.2f}ms "
            f"compose=L{frame_profile.get('compose_left_cuda', 0.0) * 1000.0:.2f}/"
            f"R{frame_profile.get('compose_right_cuda', 0.0) * 1000.0:.2f}ms "
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
                self.LIVE_CONTROLLER_TRANSLATION_SCALE,
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
                self.LIVE_CONTROLLER_TRANSLATION_SCALE,
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
                self.LIVE_CONTROLLER_TRANSLATION_SCALE,
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

    def _wait_for_valid_immersive_startup_sample(self, immersive_bridge, timeout=10.0):
        deadline = time.time() + timeout
        last_sample = None
        while time.time() < deadline:
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
        gaussians._xyz = self._rotate_points_with_matrix(
            gaussians._xyz,
            rotation_matrix.to(device=gaussians._xyz.device, dtype=gaussians._xyz.dtype),
            pivot=yaw_pivot.to(device=gaussians._xyz.device, dtype=gaussians._xyz.dtype),
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
        support_center = self._object_support_patch_center(object_points)
        target_center = (
            layout.table_top_center
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
        if xy_error > self.IMMERSIVE_STARTUP_CENTER_EPS or z_error > self.IMMERSIVE_STARTUP_PLANE_EPS:
            raise RuntimeError(
                f"Immersive scene spawn validation failed during {context}: "
                f"support_center={support_center.detach().cpu().numpy().tolist()} "
                f"table_top_center={table_center.detach().cpu().numpy().tolist()} "
                f"xy_error={xy_error:.4f} z_error={z_error:.4f}"
            )
        return support_center

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
    ):
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

        for eye_name, eye_sample, eye_pose_world in (
            ("left", left_eye_sample, left_eye_pose_world),
            ("right", right_eye_sample, right_eye_pose_world),
        ):
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
            if not table_projection["in_bounds"] or not object_projection["in_bounds"]:
                projection_failures.append(eye_name)

            scene_intrinsic = self._scale_intrinsic_for_resolution(
                intrinsic,
                eye_width,
                eye_height,
                scene_renderer.width,
                scene_renderer.height,
            )
            scene_color, scene_depth = scene_renderer.render_eye(
                eye_pose_world,
                scene_intrinsic,
            )
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
            raise RuntimeError(
                "Immersive startup render validation failed.\n"
                + str(startup_debug)
            )
        if (save_success_bundle or compose_fallback_required) and debug_output_dir is not None:
            self._save_immersive_startup_debug_bundle(
                os.path.join(debug_output_dir, "startup_debug"),
                debug_renders,
                startup_debug,
            )
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
        boxes = torch.tensor(
            [
                [layout.table_box.mins, layout.table_box.maxs],
                [layout.floor_box.mins, layout.floor_box.maxs],
            ],
            dtype=torch.float32,
            device=cfg.device,
        )
        self.simulator.set_static_collision_boxes(boxes)

    def _settle_scene_rest_state(self, rest_target):
        self.simulator.set_controller_interactive(rest_target, rest_target)
        last_state = None
        for _ in range(self.IMMERSIVE_SCENE_REST_SETTLE_STEPS):
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
        preview_selected_translation_only=False,
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
                preview_anchor_visible = bool(
                    controller_anchor_preview_state[source]["visible"]
                )
                preview_anchor_name = controller_anchor_preview_state[source].get(
                    "selected_anchor_name"
                )
                selected_preview_anchor = None
                if controller_predefined_anchor_states and preview_anchor_visible:
                    selected_preview_anchor = self._anchor_state_by_name(
                        controller_predefined_anchor_states,
                        preview_anchor_name,
                    )
                preview_anchor_resolved = selected_preview_anchor is not None
                nearest_anchor = None
                hit_world = None if overlay is None else overlay.get("hit_world")
                ray_origin_world, ray_direction_world = self._controller_world_ray_pose(
                    controller_world
                )
                grab_start_mode = None
                if preview_anchor_visible:
                    grab_start_mode = "preview_selected_template"
                    if not preview_anchor_resolved:
                        preview_interaction_state = {
                            "grab_start_mode": grab_start_mode,
                            "preview_anchor_visible": preview_anchor_visible,
                            "preview_anchor_name": preview_anchor_name,
                            "preview_anchor_resolved": False,
                            "hit_present": False,
                            "start_reference": None,
                        }
                        self._log_controller_interaction_start_attempt(
                            source,
                            preview_interaction_state,
                            reason=f"selected_preview_anchor_unresolved(name={preview_anchor_name})",
                        )
                        self._log_controller_interaction_rejected(
                            source,
                            None,
                            preview_interaction_state,
                            f"selected_preview_anchor_unresolved(name={preview_anchor_name})",
                        )
                        continue
                    snapped_anchor = selected_preview_anchor
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
                else:
                    if not allow_implicit_fallback_start:
                        continue
                    if hit_world is not None and controller_predefined_anchor_states:
                        nearest_anchor = self._select_predefined_interaction_anchor(
                            hit_world,
                            controller_predefined_anchor_states,
                            require_selection_radius=False,
                        )
                    if (
                        nearest_anchor is None
                        and ray_origin_world is not None
                        and ray_direction_world is not None
                        and controller_predefined_anchor_states
                    ):
                        nearest_anchor = self._select_predefined_interaction_anchor_for_ray(
                            ray_origin_world,
                            ray_direction_world,
                            controller_predefined_anchor_states,
                        )
                    snapped_anchor = nearest_anchor
                    grab_start_mode = "implicit_fallback"
                    if snapped_anchor is None:
                        continue
                    remap_candidate = self._build_multi_points_controller_attachment_candidate(
                        source,
                        snapped_anchor,
                        object_points,
                        controller_attachment_metadata,
                        hit_world,
                        ray_direction_world,
                    )
                if remap_candidate is None:
                    continue
                interaction_state = self._start_live_controller_interaction(
                    source,
                    controller_world,
                    remap_candidate["attach_anchor_world"].clone(),
                    controller_source_anchor_centers,
                    translation_only=(
                        bool(selected_preview_anchor is not None)
                        and preview_selected_translation_only
                    ),
                )
                interaction_state.update(
                    {
                        "grab_start_mode": grab_start_mode,
                        "preview_anchor_visible": preview_anchor_visible,
                        "preview_anchor_name": preview_anchor_name,
                        "preview_anchor_resolved": preview_anchor_resolved,
                        "hit_present": bool(hit_world is not None),
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
                    explicit_preview_selected=selected_preview_anchor is not None,
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
                        "explicit_preview_selected": bool(selected_preview_anchor is not None),
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
                preview_color = (
                    self.LIVE_CONTROLLER_ATTACH_ACTIVE_COLOR
                    if preview_active
                    else color
                )
                preview_radius = (
                    self.LIVE_CONTROLLER_PREVIEW_SELECTED_RADIUS
                    if preview_selected
                    else self.LIVE_CONTROLLER_PREVIEW_RADIUS
                )
                preview_blend = 0.78 if preview_selected else 0.32
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

    def _log_controller_anchor_preview_transition(self, source, preview_state, state_cache):
        state = (
            bool(preview_state.get("visible", False)),
            preview_state.get("selected_anchor_name"),
        )
        if state_cache.get(source) == state:
            return

        if not state[0]:
            print(f"[live_openxr_controller] {source} anchor_preview=0", flush=True)
        else:
            print(
                "[live_openxr_controller] "
                f"{source} anchor_preview=1 selected={state[1]}",
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
        collect_compose_debug=False,
        collect_debug_maps=False,
    ):
        view_setup_start = time.perf_counter() if render_profile_frame is not None else None
        eye_w2c_cv = self._camera_pose_world_to_cv_w2c(eye_pose_world)
        eye_view, _ = self._create_gs_view(
            eye_w2c_cv,
            intrinsic,
            eye_height,
            eye_width,
        )
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
                    collect_debug=True,
                    collect_debug_maps=collect_debug_maps,
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
                }
            )
        return projected

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
        if compose_mode == "alpha_overlay":
            effective_alpha = object_alpha
            visible_mask = raw_visible
        else:
            if compose_mode != "depth_aware":
                raise ValueError(f"Unsupported immersive compose_mode: {compose_mode}")
            if gaussian_depth is None:
                effective_alpha = object_alpha
                visible_mask = raw_visible
            else:
                gaussian_depth = self._normalize_gaussian_depth(gaussian_depth)
                scene_has_geometry = scene_depth_t > 0.0
                object_has_depth = gaussian_depth > 0.0
                visible_mask = object_has_depth & (
                    (~scene_has_geometry) | (gaussian_depth <= (scene_depth_t + 5e-3))
                )
                effective_alpha = object_alpha * visible_mask.unsqueeze(-1).to(
                    object_alpha.dtype
                )

        composed_rgb = scene_color[..., :3] * (1.0 - effective_alpha) + (
            object_rgba[..., :3] * 255.0
        ) * effective_alpha
        composed = torch.empty(
            scene_color.shape,
            dtype=torch.uint8,
            device=cfg.device,
        )
        composed[..., :3] = composed_rgb.clamp(0.0, 255.0).to(torch.uint8)
        composed[..., 3] = 255
        if not (collect_debug or collect_debug_maps):
            return composed

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

        visible_alpha = effective_alpha[..., 0]
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

    #pyh this should be the baseline+, where it is implementing MJX style batching
    # one Model and multiple data stream
    def interactive_playground_batched(
        self, model_path, gs_path,
        eval_image_path, n_dup=0):

        #pyh just for easier debugging
        if cfg.self_collision:
            print(f"collision dist {cfg.collision_dist}")
        else:
            print("no collision flag set")
        
        # Load the model
        logger.info(f"Load model from {model_path}")
        checkpoint = torch.load(model_path, map_location=cfg.device)

        #check if we have enough input trajectories for the launch
        assert self.num_input_trajectories >= (n_dup + 1), (
            f"Not enough input trajectories: have {self.num_input_trajectories}, "
            f"need {n_dup + 1} (n_dup={n_dup})."
        )
        if input_source not in {"recorded", "live_openxr"}:
            raise ValueError(f"Unsupported input_source: {input_source}")
        if input_source == "live_openxr" and n_dup != 0:
            raise ValueError("live_openxr input currently supports only the single-instance case (--n_dup 0)")

        #load trained parameter, collide* are 1D tensor of length 1 
        trained_spring_Y = checkpoint["spring_Y"]
        trained_collide_elas = checkpoint["collide_elas"]
        trained_collide_fric = checkpoint["collide_fric"]
        trained_collide_object_elas = checkpoint["collide_object_elas"]
        trained_collide_object_fric = checkpoint["collide_object_fric"]

        #pyh uncomment to use this if we use Morton ordering
        trained_spring_Y = trained_spring_Y[self.spring_permutation]
        
        #pyh just splitting out objects for object and controller parts
        #populating them separately before concating them together with object value first
        # 🆕 in edge coloring mass nodes layout stay the same
        # [ object nodes: inst0, inst1, inst2, ... ]  [ controller nodes: inst0, inst1, inst2, ... ]
        # object nodes are sorted by Morton ordering
        obj_init_vertices = self.init_vertices[: self.num_all_points]  #extract the vertices for object mass node
        ctrl_init_vertices = self.init_vertices[self.num_all_points :] #extract the vertices for controller mass node
        n_vert_single_obj = obj_init_vertices.shape[0]
        n_vert_single_ctrl = ctrl_init_vertices.shape[0]
        n_springs_single_obj  = int(self.num_object_springs)
        n_spring_single_ctrl  = int(self.init_springs.shape[0] - self.num_object_springs)

        #pyh this offset is similar to batching in exiting physics engine where instances 
        #are placed far apart. here we just add a simple offset
        OFFSET = torch.tensor([10, 10, 0], dtype=torch.float32, device=cfg.device)

        #pyh shift each chunk separately and concatenates them all at once at the end
        #at end should be [ all object vertices ... ][ all controller vertices ... ]        
        out_init_vertices      = []
        out_init_velocities     = []
        out_controller_points  = []

        for dup_i in range(n_dup + 1):
            obj_v  = obj_init_vertices + dup_i * OFFSET
            if self.init_velocities is not None:
                obj_v_vel = self.init_velocities
                out_init_velocities.append(obj_v_vel)
                
            out_init_vertices.append(obj_v)
            
        #this is the start of the controller mass nodes
        base_ctrl_vert_offset = (n_dup + 1) * n_vert_single_obj

        #) DUPLICATE CONTROLLERS ONLY
        for dup_i in range(n_dup + 1):
            #also shift controller by OFFSET as well    
            ctrl_v = ctrl_init_vertices + dup_i * OFFSET

            #load from multi-trajectory input, each trajectory is assigned to one instance
            new_controller_points = self.controller_points_group[dup_i] + dup_i * OFFSET
            out_init_vertices.append(ctrl_v)
            out_controller_points.append(new_controller_points)

        #FINALIZE into single global flat arrays)
        self.batch_init_vertices     = torch.cat(out_init_vertices, dim=0)
        self.batch_init_velocities = None
        if self.init_velocities is not None:    
            self.batch_init_velocities    = torch.cat(out_init_velocities, dim=0)               
        self.batch_controller_points = torch.cat(out_controller_points, dim=1) #frames, total numbers of control point, 3)
                
        #pyh intialization check
        expected_total = base_ctrl_vert_offset + (n_dup+1) * n_vert_single_ctrl
        print(f"[Check] single instance object mass node {n_vert_single_obj}, controller mass node {n_vert_single_ctrl}")
        print(f"[CHECK] total mass nodes {self.batch_init_vertices.shape[0]}, expected {expected_total}")
        print("batch_init_vertices:", type(self.batch_init_vertices), self.batch_init_vertices.shape, self.batch_init_vertices.dtype, self.batch_init_vertices.device)
        print("batch_controller_points:", type(self.batch_controller_points), self.batch_controller_points.shape, self.batch_controller_points.dtype, self.batch_controller_points.device)

        self.simulator = SpringMassSystemWarp(
            #the following variables should be shared only one instance is needed (wp_spring_y (based on pring_y) is set later, so not passed here)
            init_springs=self.init_springs,
            init_rest_lengths=self.init_rest_lengths, 
            init_masses=self.init_masses,
            init_masks=self.init_masks,
            #the following is per instance
            init_vertices=self.batch_init_vertices, 
            init_velocities=self.batch_init_velocities,
            #the following should be shared but does not need any change needed (mainly because it is single scalar)
            dt=cfg.dt,
            num_substeps=cfg.num_substeps,
            dashpot_damping=cfg.dashpot_damping,
            drag_damping=cfg.drag_damping,
            collision_dist = cfg.collision_dist,
            reverse_z=cfg.reverse_z,
            spring_Y_max=cfg.spring_Y_max,
            spring_Y_min=cfg.spring_Y_min,
            self_collision=cfg.self_collision,
            #the following should be updated
            collide_elas=trained_collide_elas,
            collide_fric=trained_collide_fric,
            collide_object_elas=trained_collide_object_elas,
            collide_object_fric=trained_collide_object_fric,
            spring_Y = trained_spring_Y,
            #added
            object_massnodes_total=base_ctrl_vert_offset, #original num_object_points
            object_massnodes_single=n_vert_single_obj,
            object_springs_total=n_springs_single_obj * (n_dup + 1),
            object_springs_single=n_springs_single_obj,
            controller_massnodes_single=n_vert_single_ctrl,
            controller_springs_single=n_spring_single_ctrl,
            controller_rest_location = self.batch_controller_points[0],
            number_of_instance = n_dup + 1,
        )

        #move here so we can populate wp_x
        self.simulator.set_init_state(
            self.simulator.wp_init_vertices, self.simulator.wp_init_velocities
        )

        if self.simulator.object_collision_flag:            
            self.simulator.create_resting_case()
            
        self.simulator.create_cuda_graph()

        #gaussian changes
        gaussians = None
        n_gaussians_single_obj = None
        for dup_i in range(n_dup + 1):
            new_gaussians = GaussianModel(sh_degree=3)
            new_gaussians.load_ply(gs_path)
            new_gaussians = remove_gaussians_with_low_opacity(new_gaussians, 0.1)
            new_gaussians._xyz += dup_i * OFFSET

            if n_gaussians_single_obj is None:
                n_gaussians_single_obj = new_gaussians._xyz.shape[0]
            if gaussians is None:
                gaussians = new_gaussians
            else:
                gaussians = self.merge_two_gaussians(gaussians, new_gaussians)
        gaussians.isotropic=True

        torch.cuda.empty_cache()

        prev_x = wp.to_torch(
            self.simulator.wp_states[0].wp_x, requires_grad=False
        ).clone()

        current_pos = gaussians.get_xyz
        current_rot = gaussians.get_rotation

        #relations, weights, and weights_indices  should be shared
        rest_mass_node_single=prev_x[:n_vert_single_obj]
        relations_single = get_topk_indices(rest_mass_node_single, K=3)
        weights_single, weights_indices_single = knn_weights_sparse(rest_mass_node_single, current_pos[:n_gaussians_single_obj], K=3 )
        xyz_rest_single = current_pos[:n_gaussians_single_obj]
        rot_rest_single = current_rot[:n_gaussians_single_obj]

        #pyh updated version
        rotation_cache = build_rotation_reuse_cache( 
            weights_indices = weights_indices_single, 
            weights = weights_single, 
            relations = relations_single, 
            mass_nodes_rest = rest_mass_node_single,
            gaussians_xyz_rest = xyz_rest_single,
            gaussians_quat_rest = rot_rest_single,
            device = cfg.device,
            mass_node_per_instance = n_vert_single_obj,
            gaussians_per_instance= n_gaussians_single_obj,
            number_of_instance=n_dup+1,
        )

        prev_target = self.batch_controller_points[0]
        current_target = prev_target

        sim_timer = Timer("Simulator")
        interp_timer = Timer("Full Motion Interpolation")
        total_timer = Timer("Total Loop")

        # Performance stats
        fps_history = []
        component_times = {
            "simulator": [],
            "full_motion_interpolation": [],
            "total": [],
        }
        frame_count = 0 

        #profiling code
        profile_started = False
        try:
            while True:

                total_timer.start()

                if input_source == "live_openxr":
                    latest_live_sample = live_hand_stream.get_latest_sample()
                    if latest_live_sample is not None:
                        live_alignment, live_alignment_mode = self._update_live_alignment(
                            live_alignment,
                            live_alignment_mode,
                            latest_live_sample,
                            recorded_anchor_centers,
                        )
                        (
                            current_live_left_world,
                            current_live_left_valid,
                            current_live_left_anchor,
                            current_live_right_world,
                            current_live_right_valid,
                            current_live_right_anchor,
                        ) = self._convert_live_sample_to_world(latest_live_sample, live_alignment)

                # 1. Simulator step

                sim_timer.start()

                pre_step_left_anchor = None
                pre_step_right_anchor = None
                if input_source == "live_openxr_controller":
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
                # if (not profile_started) and frame_count > 50:
                #     torch.cuda.nvtx.range_push("PLAYGROUND_LOOP")  # <- only profile after warmup
                #     profile_started = True
                #     self.simulator.step()
                #     wp.synchronize()
                #     torch.cuda.nvtx.range_pop()
                #     profile_started = False
                # else:
                #     self.simulator.step()
                #     wp.synchronize()

                x = wp.to_torch(self.simulator.wp_states[-1].wp_x, requires_grad=False)
            
                # Set the intial state for the next step
                self.simulator.set_init_state(
                    self.simulator.wp_states[-1].wp_x,
                    self.simulator.wp_states[-1].wp_v,
                )

                sim_time = sim_timer.stop()

                #pyh ignore first two frame since 1st frame is skewed during to rendering initialization and 2nd frame is skewed toward getting neighboruing weights
                if frame_count > 1:
                    component_times["simulator"].append(sim_time)

                if prev_x is not None:
                    with torch.no_grad():
                        interp_timer.start()
                        
                        #R reuse with dropping weight
                        current_pos, current_rot= lbs_with_rotation_reuse(
                            current_mass_nodes = x,
                            cache = rotation_cache,
                        )

                        interp_time = interp_timer.stop()               

                    if frame_count > 1:
                        component_times["full_motion_interpolation"].append(interp_time)

                prev_x = x.clone()


                ############### Temporary timer ###############
                # Total loop time
                total_time = total_timer.stop()
                if frame_count > 1:
                    component_times["total"].append(total_time)

                # Calculate FPS
                fps = 1.0 / total_time
                if frame_count > 1:
                    fps_history.append(fps)

                frame_count += 1

                prev_target = current_target
                if input_source == "live_openxr":
                    latest_live_sample = live_hand_stream.get_latest_sample()
                    if latest_live_sample is not None:
                        (
                            next_left_world,
                            next_left_valid,
                            next_left_anchor,
                            next_right_world,
                            next_right_valid,
                            next_right_anchor,
                        ) = self._convert_live_sample_to_world(latest_live_sample, live_alignment)

                        if next_left_anchor is not None:
                            current_live_left_world = next_left_world
                            current_live_left_valid = next_left_valid
                            current_live_left_anchor = next_left_anchor
                        if next_right_anchor is not None:
                            current_live_right_world = next_right_world
                            current_live_right_valid = next_right_valid
                            current_live_right_anchor = next_right_anchor

                    current_target = self._make_live_target_from_anchors(
                        recorded_base_target,
                        controller_masks,
                        current_live_left_anchor,
                        current_live_right_anchor,
                        recorded_anchor_centers,
                    )
                else:
                    #New updated to use the multi-input length
                    if frame_count < self.frame_len:
                        current_target = self.batch_controller_points[frame_count]
                    else:
                        print("Reached end of recorded control sequence")
                        break


        finally:
            if profile_started:
                torch.cuda.nvtx.range_pop()
            #pyh add overall stat printing
            # --- Final Summary Statistics ---
            if frame_count > 1:

                frames_used_for_stats = len(component_times["total"])

                print(f"\n=== Final Summary (averaged over {frames_used_for_stats} frames) ===")
                #pyh save output for file as well
                log_lines = []
                log_lines.append(f"=== Final Summary (averaged over {frames_used_for_stats} frames) ===")

                total_frame_times = component_times["total"]
                total_time_seconds = sum(total_frame_times)
                average_fps = frames_used_for_stats / total_time_seconds
                average_frame_time = np.mean(total_frame_times)

                print(f"Average FPS: {average_fps:.2f}")
                print(f"Average Total Frame Time: {average_frame_time * 1000:.2f} ms")
                log_lines.append(f"Average FPS: {average_fps:.2f}")
                log_lines.append(f"Average Total Frame Time: {average_frame_time * 1000:.2f} ms")
                
                # Detailed breakdown by components
                components_to_report = [
                    "simulator",
                    "full_motion_interpolation",
                ]

                for component_name in components_to_report:
                    component_times_list = component_times.get(component_name, [])
                    if component_times_list:
                        average_component_time = np.mean(component_times_list)
                        time_share_percentage = (average_component_time / average_frame_time) * 100
                        readable_name = component_name.replace('_', ' ').capitalize()
                        print(f"{readable_name}: {average_component_time * 1000:.2f} ms ({time_share_percentage:.1f}%)")
                        log_lines.append(f"{readable_name}: {average_component_time * 1000:.2f} ms ({time_share_percentage:.1f}%)")

                #pyh save the performance log to a file
                os.makedirs(eval_image_path, exist_ok=True)
                log_file_path = os.path.join(eval_image_path, "performance_summary.txt")
                with open(log_file_path, "w") as log_file:
                    log_file.write("\n".join(log_lines))

    def _interactive_playground_batched_visualization_immersive(
        self,
        model_path,
        gs_path,
        eval_image_path,
        render_profile_output_path,
        window,
        cuda_ctx,
        input_source,
        controller_mode,
        interactive_window_mode,
        scene_preset,
        scene_assets_root,
        render_profile=False,
        render_profile_every=30,
        immersive_render_preset="quality",
        immersive_scene_render_scale=None,
        immersive_scene_stereo_mode=None,
        immersive_overlay_mode=None,
    ):
        if input_source != "live_openxr_controller":
            raise ValueError(
                "Immersive Quest mode currently supports only input_source=live_openxr_controller"
            )
        if controller_mode != "multi_points":
            raise ValueError(f"Unsupported controller_mode for immersive mode: {controller_mode}")
        if scene_preset != "simple_lab":
            raise ValueError(
                "Immersive Quest mode currently requires --scene_preset simple_lab"
            )

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
        gaussians = GaussianModel(sh_degree=3)
        gaussians.load_ply(gs_path)
        gaussians = remove_gaussians_with_low_opacity(gaussians, 0.1)
        gaussians.isotropic = True

        startup_yaw_debug = self._apply_immersive_startup_yaw(
            obj_init_vertices,
            None,
            gaussians,
            recorded_base_target,
            recorded_anchor_centers,
            original_controller_source_anchor_centers,
        )
        obj_init_vertices = startup_yaw_debug["object_vertices"]
        recorded_base_target = startup_yaw_debug["recorded_base_target"]
        recorded_anchor_centers = startup_yaw_debug["recorded_anchor_centers"]
        original_controller_source_anchor_centers = startup_yaw_debug[
            "controller_source_anchor_centers"
        ]
        controller_predefined_anchor_defs = self._build_predefined_interaction_anchors(
            obj_init_vertices,
            intrinsic_torch,
            w2c_torch,
        )
        two_point_runtime = self._build_two_point_live_controller_runtime(
            obj_init_vertices,
            trained_spring_Y_for_sim,
            original_controller_source_masks,
            original_controller_source_anchor_centers,
            controller_predefined_anchor_defs,
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
        controller_anchor_templates = {"left": {}, "right": {}}
        ctrl_init_vertices = controller_runtime_base_target
        print(
            "[quest_display] immersive startup yaw: "
            f"axis={startup_yaw_debug['yaw_axis'].detach().cpu().numpy().tolist()} "
            f"angle={startup_yaw_debug['yaw_angle']:.4f} "
            f"pivot={startup_yaw_debug['yaw_pivot'].detach().cpu().numpy().tolist()} "
            f"support_center={startup_yaw_debug['rotated_support_center'].detach().cpu().numpy().tolist()} "
            "controller_runtime_rotated=1",
            flush=True,
        )

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

        prev_x = wp.to_torch(
            self.simulator.wp_states[0].wp_x, requires_grad=False
        ).clone()
        current_pos = gaussians.get_xyz
        current_rot = gaussians.get_rotation
        relations_single = get_topk_indices(prev_x, K=3)
        weights_single, weights_indices_single = knn_weights_sparse(
            prev_x,
            current_pos,
            K=3,
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
            immersive_render_preset=immersive_render_preset,
            immersive_scene_render_scale=immersive_scene_render_scale,
            immersive_scene_stereo_mode=immersive_scene_stereo_mode,
            immersive_overlay_mode=immersive_overlay_mode,
        )
        scene_width, scene_height = self._resolve_immersive_scene_resolution(
            eye_width,
            eye_height,
            immersive_render_options["scene_render_scale"],
        )
        active_scene_stereo_mode = immersive_render_options["scene_stereo_mode"]
        immersive_bridge = None
        scene_renderer = None
        preview_tex = None
        preview_prog = None
        preview_vao = None
        preview_display_active = interactive_window_mode == "visible"
        left_eye_frame = None
        right_eye_frame = None
        shared_scene_compose_cache = {}
        shared_scene_reproject_caches = {"source": {}, "left": {}, "right": {}}
        frame_count = 0

        live_head_alignment = None
        head_pose_state = None
        live_controller_alignment = None
        live_controller_alignment_mode = "unset"
        current_live_left_controller = None
        current_live_right_controller = None
        controller_select_state_cache = {"left": None, "right": None}
        controller_select_hold_state = {"left": {}, "right": {}}
        controller_select_hold_state_cache = {"left": None, "right": None}
        controller_anchor_cycle_state_cache = {"left": None, "right": None}
        controller_anchor_cycle_edge_cache = {"left": False, "right": False}
        controller_snap_state_cache = {"left": None, "right": None}
        controller_snap_edge_cache = {"left": False, "right": False}
        controller_anchor_preview_state = {
            "left": {"visible": False, "selected_anchor_name": None},
            "right": {"visible": False, "selected_anchor_name": None},
        }
        controller_anchor_preview_state_cache = {"left": None, "right": None}
        controller_interaction_state = {"left": None, "right": None}
        controller_interaction_state_cache = {"left": None, "right": None}
        controller_motion_state_cache = {"left": None, "right": None}
        last_left_eye_pose_world = None
        last_right_eye_pose_world = None
        last_immersive_sample = None
        immersive_compose_mode = "depth_aware"
        startup_render_debug = None

        diagnostic_output_path = render_profile_output_path
        if diagnostic_output_path is None and eval_image_path is not None:
            diagnostic_output_path = eval_image_path
        diagnostic_view_render_path = None
        if diagnostic_output_path is not None:
            diagnostic_view_render_path = os.path.join(
                diagnostic_output_path,
                "immersive_output",
            )
            os.makedirs(diagnostic_view_render_path, exist_ok=True)
        if eval_image_path:
            eval_render_path = os.path.join(eval_image_path, "0")
            os.makedirs(eval_render_path, exist_ok=True)

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
            "scene_render_left_wall",
            "scene_render_right_wall",
            "gaussian_raw_left_ratio",
            "gaussian_visible_left_ratio",
            "gaussian_retention_left_ratio",
            "gaussian_raw_right_ratio",
            "gaussian_visible_right_ratio",
            "gaussian_retention_right_ratio",
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
            "gaussian_render_left_cuda",
            "gaussian_render_right_cuda",
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
            "preview_window_wall",
            "glfw_poll_wall",
            "eval_png_write_wall",
            "cuda_memory_allocated_gib",
            "cuda_memory_reserved_gib",
        ]
        immersive_render_profile_series = (
            {key: [] for key in immersive_render_profile_keys}
            if render_profile
            else None
        )
        immersive_render_profile_rows = [] if render_profile else None

        try:
            immersive_bridge = OpenXRImmersiveBridge(
                repo_root,
                width=eye_width,
                height=eye_height,
            )
            immersive_bridge.start()
            initial_sample = self._wait_for_valid_immersive_startup_sample(
                immersive_bridge,
                timeout=10.0,
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
                basis_override=live_head_alignment["basis"],
                collect_reset_edges=False,
                alignment_pose_role="grip",
                controller_position_pose_role="grip",
                controller_ray_pose_role="aim",
            )
            live_controller_alignment = controller_runtime_state["alignment"]
            live_controller_alignment_mode = controller_runtime_state["alignment_mode"]
            if live_controller_alignment is None:
                print(
                    "[live_openxr_controller] immersive controller alignment pending; "
                    "scene startup will continue without controller interaction",
                    flush=True,
                )
            else:
                print(
                    "[live_openxr_controller] immersive controller alignment acquired "
                    f"during startup mode={live_controller_alignment_mode}",
                    flush=True,
                )
            current_live_left_controller = controller_runtime_state["left_controller"]
            current_live_right_controller = controller_runtime_state["right_controller"]

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
            scene_renderer = SimpleLabSceneRenderer(
                scene_assets_root=scene_assets_root,
                width=scene_width,
                height=scene_height,
                lighting_mode=immersive_render_options["lighting_mode"],
            )
            scene_renderer.set_layout(layout)
            table_alignment_debug = scene_renderer.table_alignment_debug()
            table_surface_center_world = layout.table_top_center
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
                    f"collider_plane={collider_top_plane_height:.4f}",
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
                layout.table_top_center,
            )
            self._apply_scene_spawn_offset_runtime(
                spawn_shift,
                gaussians,
                controller_runtime_base_target=controller_runtime_base_target,
                recorded_base_target=recorded_base_target,
                recorded_anchor_centers=recorded_anchor_centers,
                controller_source_anchor_centers=controller_source_anchor_centers,
            )
            spawn_support_center = self._validate_scene_spawn_alignment(
                self.batch_init_vertices[: self.num_all_points],
                layout,
                context="spawn shift",
                table_surface_center_world=table_surface_center_world,
            )
            print(
                "[quest_display] immersive spawn shift: "
                f"shift={spawn_shift.detach().cpu().numpy().tolist()} "
                f"support_center={spawn_support_center.detach().cpu().numpy().tolist()}",
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
                basis_override=live_head_alignment["basis"],
                collect_reset_edges=False,
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
            (
                last_left_eye_pose_world,
                last_right_eye_pose_world,
                head_pose_state,
            ) = self._update_immersive_head_pose_state(
                initial_sample,
                live_head_alignment,
                head_pose_state,
                frame_index=0,
            )
            if last_left_eye_pose_world is None:
                last_left_eye_pose_world = last_right_eye_pose_world
            if last_right_eye_pose_world is None:
                last_right_eye_pose_world = last_left_eye_pose_world

            self._set_scene_collider_boxes(layout)
            if self.simulator.object_collision_flag:
                self.simulator.create_resting_case()
            self.simulator.create_cuda_graph()

            current_target = controller_runtime_base_target.clone()
            prev_target = current_target.clone()
            scene_rest_state = self._settle_scene_rest_state(current_target.clone())
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
            settled_bounds_min = (
                x[: self.num_all_points].min(dim=0).values.detach().cpu().numpy().tolist()
            )
            settled_bounds_max = (
                x[: self.num_all_points].max(dim=0).values.detach().cpu().numpy().tolist()
            )
            print(
                "[quest_display] immersive settled rest state: "
                f"support_center={settled_support_center.detach().cpu().numpy().tolist()} "
                f"bounds_min={settled_bounds_min} bounds_max={settled_bounds_max}",
                flush=True,
            )
            controller_anchor_templates = self._build_predefined_controller_anchor_templates(
                controller_runtime_base_target,
                x[: self.num_all_points],
                controller_source_masks,
                controller_source_anchor_centers,
                controller_attachment_metadata,
                controller_predefined_anchor_defs,
            )
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
                    )
                )
                print(
                    "[quest_display] immersive reprojection startup validation: "
                    + str(reproject_debug),
                    flush=True,
                )
                if not reproject_valid:
                    active_scene_stereo_mode = "per_eye"
                    print(
                        "[quest_display] immersive reprojection startup validation failed; "
                        "falling back to per_eye room rendering for this run",
                        flush=True,
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
            )
            startup_render_debug["requested_scene_stereo_mode"] = (
                immersive_render_options["scene_stereo_mode"]
            )
            startup_render_debug["active_scene_stereo_mode"] = active_scene_stereo_mode
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
                        basis_override=live_head_alignment["basis"],
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
                        controller_reset_triggered = True
                else:
                    current_live_left_controller = None
                    current_live_right_controller = None

                sim_timer.start()

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
                self._snap_to_scene_rest_if_idle(
                    scene_rest_state,
                    controller_interaction_state,
                )
                x = wp.to_torch(
                    self.simulator.wp_states[0].wp_x,
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
                        )
                        overlay_entry["anchor_preview_entries_world"] = []
                        if (
                            immersive_render_options["overlay_mode"] == "full"
                            and controller_anchor_preview_state[source]["visible"]
                        ):
                            for anchor_state in controller_predefined_anchor_states:
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
                                    }
                                )
                        nearest_anchor = None
                        if overlay_entry["hit_world"] is not None:
                            nearest_anchor = self._select_predefined_interaction_anchor(
                                overlay_entry["hit_world"],
                                controller_predefined_anchor_states,
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
                                controller_predefined_anchor_states,
                            )
                        attach_candidate_anchor = (
                            selected_preview_anchor
                            if selected_preview_anchor is not None
                            else nearest_anchor
                        )
                        overlay_entry["attach_candidate"] = (
                            attach_candidate_anchor is not None
                            or overlay_entry["hit_world"] is not None
                        )
                        overlay_entry["attach_candidate_world"] = (
                            attach_candidate_anchor["center_world"]
                            if attach_candidate_anchor is not None
                            else overlay_entry["hit_world"]
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

                if last_immersive_sample is None:
                    raise RuntimeError("Immersive bridge stopped providing pose samples.")
                if last_left_eye_pose_world is None or last_right_eye_pose_world is None:
                    raise RuntimeError("Immersive eye poses became unavailable.")

                render_timer.start()
                render_profile_frame = self._render_profile_new_frame(render_profile)
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
                if intrinsics_setup_start is not None:
                    self._render_profile_add_wall_time(
                        render_profile_frame,
                        "render_eye_intrinsics_setup_wall",
                        time.perf_counter() - intrinsics_setup_start,
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
                        collect_compose_debug=render_profile_frame is not None,
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
                        collect_compose_debug=render_profile_frame is not None,
                    )
                )
                if render_profile_frame is not None:
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
                left_eye_overlay_entries = []
                right_eye_overlay_entries = []
                left_intrinsic_t = torch.as_tensor(
                    left_intrinsic,
                    dtype=torch.float32,
                    device=cfg.device,
                )
                right_intrinsic_t = torch.as_tensor(
                    right_intrinsic,
                    dtype=torch.float32,
                    device=cfg.device,
                )
                left_w2c_cv = torch.as_tensor(
                    self._camera_pose_world_to_cv_w2c(last_left_eye_pose_world),
                    dtype=torch.float32,
                    device=cfg.device,
                )
                right_w2c_cv = torch.as_tensor(
                    self._camera_pose_world_to_cv_w2c(last_right_eye_pose_world),
                    dtype=torch.float32,
                    device=cfg.device,
                )
                for overlay_world in controller_overlay_by_source.values():
                    left_overlay = self._project_live_controller_world_overlay(
                        overlay_world,
                        left_intrinsic_t,
                        left_w2c_cv,
                        eye_height,
                        eye_width,
                    )
                    if left_overlay is not None:
                        left_eye_overlay_entries.append(left_overlay)
                    right_overlay = self._project_live_controller_world_overlay(
                        overlay_world,
                        right_intrinsic_t,
                        right_w2c_cv,
                        eye_height,
                        eye_width,
                    )
                    if right_overlay is not None:
                        right_eye_overlay_entries.append(right_overlay)
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
                    left_eye_frame = left_eye_frame.to(dtype=torch.float32)
                    self._draw_live_controller_overlay(
                        left_eye_frame,
                        left_eye_overlay_entries,
                    )
                    left_eye_frame = left_eye_frame.clamp(0.0, 255.0).to(torch.uint8)
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
                    right_eye_frame = right_eye_frame.to(dtype=torch.float32)
                    self._draw_live_controller_overlay(
                        right_eye_frame,
                        right_eye_overlay_entries,
                    )
                    right_eye_frame = right_eye_frame.clamp(0.0, 255.0).to(torch.uint8)
                    if overlay_draw_right_start is not None:
                        self._render_profile_add_wall_time(
                            render_profile_frame,
                            "overlay_draw_right_wall",
                            time.perf_counter() - overlay_draw_right_start,
                        )

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
                    raise RuntimeError(
                        "Quest immersive bridge stopped accepting stereo frames.\n"
                        + immersive_bridge.debug_summary()
                    )

                if preview_display_active and preview_tex is not None:
                    preview_window_start = (
                        time.perf_counter() if render_profile_frame is not None else None
                    )
                    glfw.make_context_current(window)
                    left_preview = left_eye_frame.detach().cpu().numpy()
                    gl.glBindTexture(gl.GL_TEXTURE_2D, preview_tex)
                    gl.glTexSubImage2D(
                        gl.GL_TEXTURE_2D,
                        0,
                        0,
                        0,
                        eye_width,
                        eye_height,
                        gl.GL_RGBA,
                        gl.GL_UNSIGNED_BYTE,
                        left_preview,
                    )
                    gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
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

                eval_png_write_wall = 0.0
                if eval_image_path:
                    should_save_frame = frame_count < 5 or (frame_count % 30 == 0)
                    if should_save_frame and left_eye_frame is not None:
                        save_start = time.perf_counter()
                        save_path = os.path.join(
                            diagnostic_view_render_path,
                            f"{frame_count:05d}.png",
                        )
                        img_rgb = left_eye_frame[..., :3].permute(2, 0, 1).float() / 255.0
                        torchvision.utils.save_image(img_rgb, save_path)
                        eval_png_write_wall = time.perf_counter() - save_start
                if render_profile_frame is not None:
                    render_profile_frame["eval_png_write_wall"] = float(
                        eval_png_write_wall
                    )
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
                    controller_motion_state_cache=controller_motion_state_cache,
                    frame_index=frame_count,
                    runtime_label="immersive",
                )
                frame_count += 1
                prev_target = next_prev_target
                current_target = next_target

                if preview_display_active and glfw.window_should_close(window):
                    break

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
                for component_name in (
                    "simulator",
                    "full_motion_interpolation",
                    "rendering",
                ):
                    component_times_list = component_times.get(component_name, [])
                    if component_times_list:
                        average_component_time = np.mean(component_times_list)
                        time_share_percentage = (
                            average_component_time / average_frame_time
                        ) * 100.0
                        readable_name = component_name.replace("_", " ").capitalize()
                        print(
                            f"{readable_name}: {average_component_time * 1000:.2f} ms "
                            f"({time_share_percentage:.1f}%)"
                        )
                        log_lines.append(
                            f"{readable_name}: {average_component_time * 1000:.2f} ms "
                            f"({time_share_percentage:.1f}%)"
                        )
                if render_profile:
                    render_profile_lines = self._render_profile_summary_lines(
                        "immersive",
                        immersive_render_profile_series,
                        immersive_render_profile_keys,
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
            if scene_renderer is not None:
                scene_renderer.delete()
            if preview_prog is not None:
                gl.glDeleteProgram(preview_prog)
            if preview_tex is not None:
                gl.glDeleteTextures([preview_tex])
            if preview_vao is not None:
                gl.glDeleteVertexArrays(1, [preview_vao])
            if cuda_ctx is not None:
                cuda_ctx.pop()

    #this is basically baseline + rendering (to verify correctness)
    def interactive_playground_batched_visualization(
        self, model_path, gs_path,
        eval_image_path, n_dup=0,
        render_profile_output_path=None,
        window=None,
        cuda_ctx=None,
        input_source="recorded",
        controller_mode="multi_points",
        quest_display_mode="off",
        interactive_window_mode="visible",
        scene_preset="none",
        scene_assets_root="./data/open_scene_assets",
        render_profile=False,
        render_profile_every=30,
        immersive_render_preset="quality",
        immersive_scene_render_scale=None,
        immersive_scene_stereo_mode=None,
        immersive_overlay_mode=None):

        if cfg.self_collision:
            print(f"collision dist {cfg.collision_dist}")
        else:
            print("no collision flag set")

        # Load the model
        logger.info(f"Load model from {model_path}")
        checkpoint = torch.load(model_path, map_location=cfg.device)

        #check if we have enough input trajectories for the launch
        assert self.num_input_trajectories >= (n_dup + 1), (
            f"Not enough input trajectories: have {self.num_input_trajectories}, "
            f"need {n_dup + 1} (n_dup={n_dup})."
        )
        if input_source not in {"recorded", "live_openxr", "live_openxr_controller"}:
            raise ValueError(f"Unsupported input_source: {input_source}")
        if controller_mode != "multi_points":
            raise ValueError(f"Unsupported controller_mode: {controller_mode}")
        if quest_display_mode not in {"off", "panel", "primary", "immersive"}:
            raise ValueError(f"Unsupported quest_display_mode: {quest_display_mode}")
        if quest_display_mode != "off" and input_source != "live_openxr_controller":
            raise ValueError(
                "Quest display mode currently supports only input_source=live_openxr_controller"
            )
        if scene_preset not in {"none", "simple_lab"}:
            raise ValueError(f"Unsupported scene_preset: {scene_preset}")
        if input_source in {"live_openxr", "live_openxr_controller"} and n_dup != 0:
            raise ValueError(
                f"{input_source} input currently supports only the single-instance case (--n_dup 0)"
            )
        if quest_display_mode == "immersive":
            return self._interactive_playground_batched_visualization_immersive(
                model_path=model_path,
                gs_path=gs_path,
                eval_image_path=eval_image_path,
                render_profile_output_path=render_profile_output_path,
                window=window,
                cuda_ctx=cuda_ctx,
                input_source=input_source,
                controller_mode=controller_mode,
                interactive_window_mode=interactive_window_mode,
                scene_preset=scene_preset,
                scene_assets_root=scene_assets_root,
                render_profile=render_profile,
                render_profile_every=render_profile_every,
                immersive_render_preset=immersive_render_preset,
                immersive_scene_render_scale=immersive_scene_render_scale,
                immersive_scene_stereo_mode=immersive_scene_stereo_mode,
                immersive_overlay_mode=immersive_overlay_mode,
            )

        #load trained parameter, collide* are 1D tensor of length 1 
        trained_spring_Y = checkpoint["spring_Y"]
        trained_collide_elas = checkpoint["collide_elas"]
        trained_collide_fric = checkpoint["collide_fric"]
        trained_collide_object_elas = checkpoint["collide_object_elas"]
        trained_collide_object_fric = checkpoint["collide_object_fric"]

        #pyh uncomment to use this if we use Morton ordering
        trained_spring_Y = trained_spring_Y[self.spring_permutation]

        #pyh just splitting out objects for object and controller parts
        #populating them separately before concating them together with object value first
        # [ object nodes: inst0, inst1, inst2, ... ]  [ controller nodes: inst0, inst1, inst2, ... ]
        intrinsic = cfg.intrinsics[0]
        w2c = cfg.w2cs[0]
        intrinsic_torch = torch.tensor(intrinsic, dtype=torch.float32, device=cfg.device)
        w2c_torch = torch.tensor(w2c, dtype=torch.float32, device=cfg.device)
        obj_init_vertices = self.init_vertices[: self.num_all_points]  #extract the vertices for object mass node
        ctrl_init_vertices = self.init_vertices[self.num_all_points :] #extract the vertices for controller mass node
        init_springs_for_sim = self.init_springs
        init_rest_lengths_for_sim = self.init_rest_lengths
        trained_spring_Y_for_sim = trained_spring_Y
        init_masses_for_sim = self.init_masses[: self.num_all_points].clone()
        recorded_base_target = (
            self.controller_points_group[0][0].clone()
            if input_source in {"live_openxr", "live_openxr_controller"}
            else None
        )
        controller_masks = None
        recorded_anchor_centers = None
        controller_source_masks = None
        controller_source_anchor_centers = None
        controller_attachment_metadata = None
        controller_anchor_templates = {"left": {}, "right": {}}
        controller_predefined_anchor_defs = None
        controller_runtime_base_target = None
        if recorded_base_target is not None:
            controller_masks = self._build_controller_part_masks(
                recorded_base_target,
                n_ctrl_parts=2,
                intrinsic=intrinsic,
                w2c=w2c,
            )
            recorded_anchor_centers = [
                recorded_base_target[mask].mean(dim=0) for mask in controller_masks
            ]
        if input_source == "live_openxr_controller":
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
            controller_predefined_anchor_defs = self._build_predefined_interaction_anchors(
                obj_init_vertices,
                intrinsic_torch,
                w2c_torch,
            )
            two_point_runtime = self._build_two_point_live_controller_runtime(
                obj_init_vertices,
                trained_spring_Y_for_sim,
                original_controller_source_masks,
                original_controller_source_anchor_centers,
                controller_predefined_anchor_defs,
            )
            controller_runtime_base_target = two_point_runtime[
                "controller_rest_points"
            ].clone()
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
            controller_anchor_templates = self._build_predefined_controller_anchor_templates(
                controller_runtime_base_target,
                obj_init_vertices,
                controller_source_masks,
                controller_source_anchor_centers,
                controller_attachment_metadata,
                controller_predefined_anchor_defs,
            )
            ctrl_init_vertices = controller_runtime_base_target

        n_vert_single_obj = obj_init_vertices.shape[0]
        n_vert_single_ctrl = ctrl_init_vertices.shape[0]
        n_springs_single_obj  = int(self.num_object_springs)
        n_spring_single_ctrl = int(
            init_springs_for_sim.shape[0] - self.num_object_springs
        )

        #pyh this offset is similar to batching in exiting physics engine where instances 
        #are placed far apart. here we just add a simple offset
        OFFSET = torch.tensor([0, 1, 0], dtype=torch.float32, device=cfg.device)
        center = 0.5 * n_dup

        #pyh shift each chunk separately and concatenates them all at once at the end
        #at end should be [ all object vertices ... ][ all controller vertices ... ]        
        out_init_vertices      = []
        out_init_velocities     = []
        out_controller_points  = []

        for dup_i in range(n_dup + 1):
            #obj_v  = obj_init_vertices + dup_i * OFFSET 
            shift = (dup_i - center) * OFFSET
            obj_v  = obj_init_vertices + shift
            
            if self.init_velocities is not None:
                obj_v_vel = self.init_velocities
                out_init_velocities.append(obj_v_vel)
                
            out_init_vertices.append(obj_v)
            
        #this is the start of the controller mass nodes
        base_ctrl_vert_offset = (n_dup + 1) * n_vert_single_obj

        #) DUPLICATE CONTROLLERS ONLY
        for dup_i in range(n_dup + 1):
            #also shift controller by OFFSET as well    
            shift = (dup_i - center) * OFFSET
            ctrl_v = ctrl_init_vertices + shift
            
            #load from multi-trajectory input, each trajectory is assigned to one instance
            if input_source == "live_openxr_controller":
                new_controller_points = ctrl_v.unsqueeze(0).repeat(self.frame_len, 1, 1)
            else:
                new_controller_points = self.controller_points_group[dup_i] + dup_i * OFFSET
            out_init_vertices.append(ctrl_v)
            out_controller_points.append(new_controller_points)

        #FINALIZE into single global flat arrays)
        self.batch_init_vertices     = torch.cat(out_init_vertices, dim=0)
        self.batch_init_velocities = None
        if self.init_velocities is not None:    
            self.batch_init_velocities    = torch.cat(out_init_velocities, dim=0)  

        self.batch_controller_points = torch.cat(out_controller_points, dim=1) #frames, total numbers of control point, 3)
                
        #pyh intialization check
        expected_total = base_ctrl_vert_offset + (n_dup+1) * n_vert_single_ctrl
        print(f"[Check] single instance object mass node {n_vert_single_obj}, controller mass node {n_vert_single_ctrl}")
        print(f"[CHECK] total mass nodes {self.batch_init_vertices.shape[0]}, expected {expected_total}")
        print("batch_init_vertices:", type(self.batch_init_vertices), self.batch_init_vertices.shape, self.batch_init_vertices.dtype, self.batch_init_vertices.device)
        print("batch_controller_points:", type(self.batch_controller_points), self.batch_controller_points.shape, self.batch_controller_points.dtype, self.batch_controller_points.device)

        self.simulator = SpringMassSystemWarp(
            #the following variables should be shared only one instance is needed (wp_spring_y (based on pring_y) is set later, so not passed here)
            init_springs=init_springs_for_sim,
            init_rest_lengths=init_rest_lengths_for_sim, 
            init_masses=init_masses_for_sim,
            init_masks=self.init_masks,
            #the following is per instance
            init_vertices=self.batch_init_vertices, 
            init_velocities=self.batch_init_velocities,
            #the following should be shared but does not need any change needed (mainly because it is single scalar)
            dt=cfg.dt,
            num_substeps=cfg.num_substeps,
            dashpot_damping=cfg.dashpot_damping,
            drag_damping=cfg.drag_damping,
            collision_dist = cfg.collision_dist,
            reverse_z=cfg.reverse_z,
            spring_Y_max=cfg.spring_Y_max,
            spring_Y_min=cfg.spring_Y_min,
            self_collision=cfg.self_collision,
            #the following should be updated
            collide_elas=trained_collide_elas,
            collide_fric=trained_collide_fric,
            collide_object_elas=trained_collide_object_elas,
            collide_object_fric=trained_collide_object_fric,
            spring_Y = trained_spring_Y_for_sim,
            #added
            object_massnodes_total=base_ctrl_vert_offset, #original num_object_points
            object_massnodes_single=n_vert_single_obj,
            object_springs_total=n_springs_single_obj * (n_dup + 1),
            object_springs_single=n_springs_single_obj,
            controller_massnodes_single=n_vert_single_ctrl,
            controller_springs_single=n_spring_single_ctrl,
            controller_rest_location = self.batch_controller_points[0],
            number_of_instance = n_dup + 1,
        )

        #move here so we can populate wp_x
        self.simulator.set_init_state(
            self.simulator.wp_init_vertices, self.simulator.wp_init_velocities
        )

        if self.simulator.object_collision_flag:            
            self.simulator.create_resting_case()
            
        self.simulator.create_cuda_graph()

        #gaussian changes
        gaussians = None
        n_gaussians_single_obj = None
        for dup_i in range(n_dup + 1):
            new_gaussians = GaussianModel(sh_degree=3)
            new_gaussians.load_ply(gs_path)
            new_gaussians = remove_gaussians_with_low_opacity(new_gaussians, 0.1)
            #new_gaussians._xyz += dup_i * OFFSET 
            shift = (dup_i - center) * OFFSET
            new_gaussians._xyz += shift

            if n_gaussians_single_obj is None:
                n_gaussians_single_obj = new_gaussians._xyz.shape[0]
            if gaussians is None:
                gaussians = new_gaussians
            else:
                gaussians = self.merge_two_gaussians(gaussians, new_gaussians)
        gaussians.isotropic=True

        torch.cuda.empty_cache()

        prev_x = wp.to_torch(
            self.simulator.wp_states[0].wp_x, requires_grad=False
        ).clone()

        current_pos = gaussians.get_xyz
        current_rot = gaussians.get_rotation

        #relations, weights, and weights_indices  should be shared
        rest_mass_node_single=prev_x[:n_vert_single_obj]
        relations_single = get_topk_indices(rest_mass_node_single, K=3)
        weights_single, weights_indices_single = knn_weights_sparse(rest_mass_node_single, current_pos[:n_gaussians_single_obj], K=3 )
        xyz_rest_single = current_pos[:n_gaussians_single_obj]
        rot_rest_single = current_rot[:n_gaussians_single_obj]

        #pyh updated version
        rotation_cache = build_rotation_reuse_cache( 
            weights_indices = weights_indices_single, 
            weights = weights_single, 
            relations = relations_single, 
            mass_nodes_rest = rest_mass_node_single,
            gaussians_xyz_rest = xyz_rest_single,
            gaussians_quat_rest = rot_rest_single,
            device = cfg.device,
            mass_node_per_instance = n_vert_single_obj,
            gaussians_per_instance= n_gaussians_single_obj,
            number_of_instance=n_dup+1,
        )

        prev_target = self.batch_controller_points[0]
        current_target = prev_target

        ########add visualization initialization
        glfw.make_context_current(window)
        base_width, base_height = cfg.WH
        width, height = self._resolve_composited_frame_resolution(
            base_width,
            base_height,
            quest_display_mode,
        )
        intrinsic = self._scale_intrinsic_for_resolution(
            cfg.intrinsics[0],
            base_width,
            base_height,
            width,
            height,
        )
        w2c = cfg.w2cs[0]
        if quest_display_mode == "primary":
            print(
                "[quest_display] quest-primary compositing resolution "
                f"{width}x{height} (desktop base {base_width}x{base_height})",
                flush=True,
            )

        background_black = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
        background_white = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda")
        view, K_cuda = self._create_gs_view(w2c, intrinsic, height, width)
        image_path = cfg.bg_img_path
        overlay = cv2.imread(image_path)
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        if overlay.shape[1] != width or overlay.shape[0] != height:
            overlay = cv2.resize(overlay, (width, height), interpolation=cv2.INTER_LINEAR)
        overlay = torch.tensor(overlay, dtype=torch.float32, device=cfg.device)
        assert overlay.shape[0] == height and overlay.shape[1] == width, \
            f"overlay {tuple(overlay.shape)} != (H,W,3)=({height},{width},3)"

        lights = torch.tensor([[0, 0, -3],
                    [1, 0.5, -2],
                    [-3, -0.5, -5]], device=cfg.device, dtype=torch.float32)
        coeffs = torch.tensor([0.95, 0.97, 0.98], device=cfg.device, dtype=torch.float32)
        #K_cuda   = torch.tensor(intrinsic, dtype=torch.float32, device=cfg.device)
        w2c_cuda = torch.tensor(w2c, dtype=torch.float32, device=cfg.device)
        coeffs_b = coeffs.view(-1, 1, 1)
        w2c_T = w2c_cuda.T.contiguous()
        intrinsic_T = K_cuda.T.contiguous()
        inv_Lz = 1.0 / lights[:, 2] 
        BYTES_PER_PIXEL = 4
        pbo_size = width * height * BYTES_PER_PIXEL
        row_pitch = width * BYTES_PER_PIXEL  # <-- define this before using Memcpy2D
        preview_display_active = interactive_window_mode == "visible"
        tex = None
        pbo = None
        reg = None
        prog = None
        vao = None
        pbo_stream = None
        cpy2d = None
        if preview_display_active:
            # Texture
            tex = gl.glGenTextures(1)
            gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
            tex_filter = gl.GL_LINEAR if quest_display_mode == "primary" else gl.GL_NEAREST
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, tex_filter)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, tex_filter)
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, width, height, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None)
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

            # PBO
            pbo = gl.glGenBuffers(1)
            gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, pbo)
            gl.glBufferData(gl.GL_PIXEL_UNPACK_BUFFER, pbo_size, None, gl.GL_STREAM_DRAW)
            gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)
            reg = RegisteredBuffer(int(pbo), graphics_map_flags.WRITE_DISCARD)

            # 4) Tiny fullscreen shader (one quad, no VAO needed)
            VS = """
            #version 330 core
            out vec2 uv; const vec2 V[4]=vec2[4](vec2(-1,-1),vec2(1,-1),vec2(-1,1),vec2(1,1));
            const vec2 T[4]=vec2[4](vec2(0,0),vec2(1,0),vec2(0,1),vec2(1,1));
            void main(){ gl_Position=vec4(V[gl_VertexID],0,1); uv=T[gl_VertexID]; }
            """
            FS = """
            #version 330 core
            in vec2 uv; out vec4 frag; uniform sampler2D uTex;
            void main(){ frag = texture(uTex, vec2(uv.x,1.0 - uv.y)); }
            """

            def _compile(kind, src):
                sid = gl.glCreateShader(kind); gl.glShaderSource(sid, src); gl.glCompileShader(sid)
                if not gl.glGetShaderiv(sid, gl.GL_COMPILE_STATUS):
                    raise RuntimeError(gl.glGetShaderInfoLog(sid).decode())
                return sid

            prog = gl.glCreateProgram()
            gl.glAttachShader(prog, _compile(gl.GL_VERTEX_SHADER, VS))
            gl.glAttachShader(prog, _compile(gl.GL_FRAGMENT_SHADER, FS))
            gl.glLinkProgram(prog)
            if not gl.glGetProgramiv(prog, gl.GL_LINK_STATUS):
                raise RuntimeError(gl.glGetProgramInfoLog(prog).decode())
            gl.glUseProgram(prog); gl.glUniform1i(gl.glGetUniformLocation(prog, "uTex"), 0); gl.glUseProgram(0)
            vao = gl.glGenVertexArrays(1)
            gl.glBindVertexArray(vao)

            # Reuse Stream
            pbo_stream = cuda_driver.Stream()

            cpy2d = cuda_driver.Memcpy2D()
            cpy2d.src_pitch = row_pitch
            cpy2d.dst_pitch = row_pitch
            cpy2d.width_in_bytes = row_pitch
            cpy2d.height = height
        else:
            print(
                "[frame_compositing] skipping local preview upload/draw path because "
                "interactive_window_mode=hidden",
                flush=True,
            )

        # These could be pre-allocated once:
        frame_rgba = torch.empty((height, width, 4), dtype=torch.uint8, device=cfg.device)
        frame_rgba[:, :, 3] = 255
        frame = torch.empty_like(overlay)
        rgb_temp = torch.empty((height, width, 3), dtype=overlay.dtype, device=cfg.device)  # Add this
        render_pipe = SimpleNamespace(
            debug=False,
            antialiasing=True,
            compute_cov3D_python=False,
            convert_SHs_python=False,
        )

        live_hand_stream = None
        live_controller_stream = None
        live_alignment = None
        live_alignment_mode = "unset"
        current_live_left_world = None
        current_live_right_world = None
        current_live_left_valid = None
        current_live_right_valid = None
        current_live_left_anchor = None
        current_live_right_anchor = None
        live_hand_side_memory = {"left": None, "right": None}
        live_controller_alignment = None
        live_controller_alignment_mode = "unset"
        current_live_left_controller = None
        current_live_right_controller = None
        controller_select_state_cache = {"left": None, "right": None}
        controller_select_hold_state = {"left": {}, "right": {}}
        controller_select_hold_state_cache = {"left": None, "right": None}
        controller_anchor_cycle_state_cache = {"left": None, "right": None}
        controller_anchor_cycle_edge_cache = {"left": False, "right": False}
        controller_snap_state_cache = {"left": None, "right": None}
        controller_snap_edge_cache = {"left": False, "right": False}
        controller_anchor_preview_state = {
            "left": {"visible": False, "selected_anchor_name": None},
            "right": {"visible": False, "selected_anchor_name": None},
        }
        controller_anchor_preview_state_cache = {"left": None, "right": None}
        controller_hit_state_cache = {"left": None, "right": None}
        controller_attach_candidate_state_cache = {"left": None, "right": None}
        controller_interaction_state = {"left": None, "right": None}
        controller_interaction_state_cache = {"left": None, "right": None}
        controller_motion_state_cache = {"left": None, "right": None}
        quest_frame_panel = None
        repo_root = Path(__file__).resolve().parents[2]

        if quest_display_mode in {"panel", "primary"}:
            quest_frame_panel = OpenXRFramePanelMirror(repo_root, width=width, height=height)
            quest_frame_panel.start()
            if quest_display_mode == "primary":
                print(
                    "[quest_display] presentation path: Quest primary composited frame",
                    flush=True,
                )
            else:
                print(
                    "[quest_display] presentation path: Quest panel mirror",
                    flush=True,
                )
            print(
                f"[quest_display] interactive_window_mode={interactive_window_mode}",
                flush=True,
            )
            print(
                f"[quest_display] quest display resolution={width}x{height}",
                flush=True,
            )
            print(
                "[quest_display] enabled panel mirror from final frame_rgba "
                f"({width}x{height} RGBA8)",
                flush=True,
            )
            if input_source == "live_openxr_controller":
                live_controller_stream = quest_frame_panel
                print(
                    "[quest_display] using the Quest panel viewer session as the live "
                    "controller source",
                    flush=True,
                )

        if input_source in {"live_openxr", "live_openxr_controller"}:
            if input_source == "live_openxr_controller":
                print(
                    "[live_openxr_controller] controller point groups: "
                    f"left={int(controller_source_masks[0].sum().item())} "
                    f"right={int(controller_source_masks[1].sum().item())}"
                )
                print(
                    f"[live_openxr_controller] controller_mode={controller_mode}",
                    flush=True,
                )
                print(
                    "[live_openxr_controller] using 2 live controller points total "
                    f"({int(controller_runtime_base_target.shape[0])} controller nodes total)",
                    flush=True,
                )
                print(
                    "[live_openxr_controller] A/X preview anchors; select starts the "
                    f"{controller_mode} 2-point controller spring path; B/Y reset the sloth",
                    flush=True,
                )
                print(
                    "[live_openxr_controller] predefined anchors: "
                    + ", ".join(anchor["name"] for anchor in controller_predefined_anchor_defs),
                    flush=True,
                )

            if input_source == "live_openxr":
                live_hand_stream = OpenXRHandJointStream(repo_root)
                live_hand_stream.start()
                initial_live_sample = live_hand_stream.wait_for_sample(timeout=10.0)
                live_alignment, live_alignment_mode = self._update_live_alignment(
                    live_alignment,
                    live_alignment_mode,
                    initial_live_sample,
                    recorded_anchor_centers,
                )
                (
                    current_live_left_world,
                    current_live_left_valid,
                    current_live_left_anchor,
                    current_live_right_world,
                    current_live_right_valid,
                    current_live_right_anchor,
                ) = self._convert_live_sample_to_world(initial_live_sample, live_alignment)
            else:
                if live_controller_stream is None:
                    live_controller_stream = OpenXRControllerStream(repo_root)
                    live_controller_stream.start()
                initial_controller_sample = live_controller_stream.wait_for_sample(timeout=10.0)
                controller_runtime_state = self._update_live_controller_runtime_from_sample(
                    initial_controller_sample,
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
                    collect_reset_edges=False,
                )
                live_controller_alignment = controller_runtime_state["alignment"]
                live_controller_alignment_mode = controller_runtime_state["alignment_mode"]
                current_live_left_controller = controller_runtime_state["left_controller"]
                current_live_right_controller = controller_runtime_state["right_controller"]
            if input_source == "live_openxr_controller":
                current_target = controller_runtime_base_target.clone()
            else:
                current_target = recorded_base_target.clone()
            prev_target = current_target.clone()

        #for saving output videos
        if eval_image_path:
            eval_render_path = os.path.join(eval_image_path, '0')
            view_render_path = os.path.join(eval_image_path, 'output')
            os.makedirs(view_render_path, exist_ok=True)
            os.makedirs(eval_render_path, exist_ok=True)


        #############end of visualization initialization

        #timer initialization code
        sim_timer = Timer("Simulator")
        interp_timer = Timer("Full Motion Interpolation")
        render_timer = Timer("Rendering")
        frame_timer = Timer("Frame Compositing")
        total_timer = Timer("Total Loop")

        # Performance stats
        fps_history = []
        component_times = {
            "simulator": [],
            "full_motion_interpolation": [],
            "rendering": [],
            "frame_compositing": [],
            "frame_compositing_gpu_timer": [],
            "frame_comp_overlay_draw_submit": [],
            "frame_comp_timing_overlay_submit": [],
            "frame_comp_rgba_pack_submit": [],
            "frame_comp_quest_publish_wall": [],
            "frame_comp_quest_process_check": [],
            "frame_comp_quest_pending_drain_nonblock": [],
            "frame_comp_quest_pending_drain_block": [],
            "frame_comp_quest_gpu_to_cpu_wait": [],
            "frame_comp_quest_gpu_to_cpu_copy": [],
            "frame_comp_quest_cpu_mmap_copy": [],
            "frame_comp_quest_header_write": [],
            "frame_comp_quest_stage_enqueue": [],
            "frame_comp_quest_fallback_copy": [],
            "frame_comp_preview_path_wall": [],
            "frame_comp_preview_sync_wall": [],
            "frame_comp_preview_copy_wall": [],
            "frame_comp_preview_upload_draw_wall": [],
            "frame_comp_glfw_poll_wall": [],
            "frame_comp_timer_stop_wall": [],
            "eval_png_write_wall": [],
            "total": [],
        }
        quest_render_profile_keys = [
            "rendering",
            "frame_compositing",
            "frame_compositing_gpu_timer",
            "frame_comp_overlay_draw_submit",
            "frame_comp_timing_overlay_submit",
            "frame_comp_rgba_pack_submit",
            "frame_comp_quest_publish_wall",
            "frame_comp_quest_process_check",
            "frame_comp_quest_pending_drain_nonblock",
            "frame_comp_quest_pending_drain_block",
            "frame_comp_quest_gpu_to_cpu_wait",
            "frame_comp_quest_gpu_to_cpu_copy",
            "frame_comp_quest_cpu_mmap_copy",
            "frame_comp_quest_header_write",
            "frame_comp_quest_stage_enqueue",
            "frame_comp_quest_fallback_copy",
            "frame_comp_preview_path_wall",
            "frame_comp_preview_sync_wall",
            "frame_comp_preview_copy_wall",
            "frame_comp_preview_upload_draw_wall",
            "frame_comp_glfw_poll_wall",
            "frame_comp_timer_stop_wall",
            "eval_png_write_wall",
        ]
        quest_render_profile_series = (
            {key: [] for key in quest_render_profile_keys}
            if render_profile
            else None
        )
        quest_render_profile_rows = [] if render_profile else None
        frame_count = 0 
        last_total_time = None

        try:
            while True:

                total_timer.start()
                controller_overlay_by_source = {}
                controller_reset_triggered = False

                if input_source == "live_openxr":
                    latest_live_sample = live_hand_stream.get_latest_sample()
                    if latest_live_sample is not None:
                        live_alignment, live_alignment_mode = self._update_live_alignment(
                            live_alignment,
                            live_alignment_mode,
                            latest_live_sample,
                            recorded_anchor_centers,
                        )
                        (
                            current_live_left_world,
                            current_live_left_valid,
                            current_live_left_anchor,
                            current_live_right_world,
                            current_live_right_valid,
                            current_live_right_anchor,
                        ) = self._convert_live_sample_to_world(
                            latest_live_sample, live_alignment
                        )
                    else:
                        current_live_left_world = None
                        current_live_left_valid = None
                        current_live_left_anchor = None
                        current_live_right_world = None
                        current_live_right_valid = None
                        current_live_right_anchor = None
                elif input_source == "live_openxr_controller":
                    latest_controller_sample = live_controller_stream.get_latest_sample()
                    if latest_controller_sample is not None:
                        controller_runtime_state = (
                            self._update_live_controller_runtime_from_sample(
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
                            )
                        )
                        live_controller_alignment = controller_runtime_state["alignment"]
                        live_controller_alignment_mode = controller_runtime_state["alignment_mode"]
                        current_live_left_controller = controller_runtime_state["left_controller"]
                        current_live_right_controller = controller_runtime_state["right_controller"]
                        controller_reset_sources = controller_runtime_state["reset_sources"]
                    else:
                        current_live_left_controller = None
                        current_live_right_controller = None
                        controller_reset_sources = []

                    if controller_reset_sources:
                        pressed_buttons = [
                            "Y" if source == "left" else "B"
                            for source in controller_reset_sources
                        ]
                        print(
                            "[live_openxr_controller] reset requested via "
                            + "/".join(pressed_buttons)
                            + "; restoring the original object pose",
                            flush=True,
                        )
                        reset_target = self._reset_live_controller_runtime(
                            controller_runtime_base_target,
                            controller_interaction_state,
                            controller_anchor_preview_state,
                            controller_attachment_metadata,
                        )
                        prev_target = reset_target.clone()
                        current_target = reset_target
                        controller_reset_triggered = True

                # 1. Simulator step

                sim_timer.start()

                pre_step_left_anchor = None
                pre_step_right_anchor = None
                if input_source == "live_openxr_controller":
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
                if input_source == "live_openxr_controller":
                    self._apply_live_controller_anchor_kinematic_overrides(
                        pre_step_left_anchor,
                        pre_step_right_anchor,
                        controller_interaction_state,
                        state_index=-1,
                    )

                x = wp.to_torch(self.simulator.wp_states[-1].wp_x, requires_grad=False)
            
                # Set the intial state for the next step
                self.simulator.set_init_state(
                    self.simulator.wp_states[-1].wp_x,
                    self.simulator.wp_states[-1].wp_v,
                )

                sim_time = sim_timer.stop()

                #pyh ignore first two frame since 1st frame is skewed during to rendering initialization and 2nd frame is skewed toward getting neighboruing weights
                if frame_count > 1:
                    component_times["simulator"].append(sim_time)

                # 3. Rendering
                render_timer.start()

                rendering, depth = self._render_gaussian_rgba(
                    view,
                    gaussians,
                    render_pipe,
                    background_black,
                    background_white,
                )
                image = rendering.permute(1, 2, 0).detach()

                render_time = render_timer.stop()
                if frame_count > 1:
                    component_times["rendering"].append(render_time)

                #frame compositing
                frame_timer.start()
                frame_comp_wall_start = time.perf_counter()
                frame_comp_overlay_draw_submit = 0.0
                frame_comp_timing_overlay_submit = 0.0
                frame_comp_rgba_pack_submit = 0.0
                frame_comp_quest_publish_wall = 0.0
                frame_comp_preview_path_wall = 0.0
                frame_comp_preview_sync_wall = 0.0
                frame_comp_preview_copy_wall = 0.0
                frame_comp_preview_upload_draw_wall = 0.0
                frame_comp_glfw_poll_wall = 0.0
                frame_comp_timer_stop_wall = 0.0
                quest_publish_stats = {
                    "process_check_wall": 0.0,
                    "pending_drain_nonblock_wall": 0.0,
                    "pending_drain_block_wall": 0.0,
                    "gpu_to_cpu_wait_wall": 0.0,
                    "gpu_to_cpu_copy_cuda": 0.0,
                    "cpu_mmap_copy_wall": 0.0,
                    "header_write_wall": 0.0,
                    "stage_enqueue_wall": 0.0,
                    "fallback_copy_wall": 0.0,
                    "total_wall": 0.0,
                }
                
                frame.copy_(overlay)
                
                image.clamp_(0, 1)
                image_mask = image[:, :, 3] > (1.0 / 255.0)
                image[..., 3].masked_fill_(~image_mask, 0.0)

                alpha = image[..., 3:4]
                torch.mul(image[..., :3], alpha, out=rgb_temp)  # rgb_temp = image_rgb * alpha
                rgb_temp.mul_(255.0)                            # rgb_temp *= 255
                frame.mul_(1.0 - alpha).add_(rgb_temp)          # frame = frame * (1 - alpha) + rgb_temp        

                masks = get_shadow_masks_batched_downsampled(
                    points=x,
                    intrinsic_T=intrinsic_T,
                    w2c_T=w2c_T,
                    W=width, H=height,
                    image_mask=image_mask,
                    lights=lights,
                    inv_Lz=inv_Lz,
                    kernel_size=7,   # your original full-res k
                    scale=2,         # try 2 first; 4 if you need more speed
                    use_half=False,  # try True later if acceptable
                    upsample_mode="bilinear",   # "nearest" is sharper/aliasy, faster
                    post_blur=False  # set True if you see stair-steps
                )   # (L,H,W) bool
              
                # Turn masks + coeffs into one attenuation map A[H,W]
                M = masks.to(frame.dtype)                                     # float
                A = torch.prod(1.0 - M + M * coeffs_b, dim=0) 
                frame.mul_(A.unsqueeze(-1)) 

                if input_source == "live_openxr":
                    hand_overlays = []
                    for source, world_points, valid_mask in (
                        ("left", current_live_left_world, current_live_left_valid),
                        ("right", current_live_right_world, current_live_right_valid),
                    ):
                        interaction_repr = self._build_hand_interaction_repr(
                            world_points, valid_mask
                        )
                        if interaction_repr is None:
                            continue

                        projected = self._project_interaction_repr(
                            interaction_repr, K_cuda, w2c_cuda, height, width
                        )
                        if projected is None:
                            continue

                        hand_overlays.append(
                            {
                                "source": source,
                                "projected": projected,
                                "palm_pixel": projected["palm"],
                            }
                        )

                    hand_overlays = self._resolve_live_hand_overlay_colors(
                        hand_overlays, width, live_hand_side_memory
                    )
                    self._draw_live_hand_overlay(frame, hand_overlays)
                elif input_source == "live_openxr_controller":
                    object_points = x[: self.num_all_points]
                    object_bounds_min = object_points.min(dim=0).values - 0.01
                    object_bounds_max = object_points.max(dim=0).values + 0.01
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
                    controller_overlays = []
                    for source, controller_world in (
                        ("left", current_live_left_controller),
                        ("right", current_live_right_controller),
                    ):
                        overlay_entry = self._build_live_controller_overlay(
                            source,
                            controller_world,
                            K_cuda,
                            w2c_cuda,
                            height,
                            width,
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
                            selected_preview_anchor = (
                                self._update_controller_anchor_preview_state(
                                    source,
                                    controller_world,
                                    overlay_entry,
                                    controller_predefined_anchor_states,
                                    controller_anchor_preview_state,
                                    cycle_edge,
                                    controller_interaction_state[source],
                                )
                            )
                            overlay_entry["anchor_preview_entries"] = []
                            if controller_anchor_preview_state[source]["visible"]:
                                for anchor_state in controller_predefined_anchor_states:
                                    anchor_pixel = self._project_world_point_to_pixel(
                                        anchor_state["center_world"],
                                        K_cuda,
                                        w2c_cuda,
                                        height,
                                        width,
                                    )
                                    if anchor_pixel is None:
                                        continue
                                    overlay_entry["anchor_preview_entries"].append(
                                        {
                                            "pixel": anchor_pixel,
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
                                        }
                                    )
                            nearest_anchor = None
                            if overlay_entry["hit_world"] is not None:
                                nearest_anchor = self._select_predefined_interaction_anchor(
                                    overlay_entry["hit_world"],
                                    controller_predefined_anchor_states,
                                )
                            ray_origin_world, ray_direction_world = (
                                self._controller_world_ray_pose(controller_world)
                            )
                            if (
                                nearest_anchor is None
                                and ray_origin_world is not None
                                and ray_direction_world is not None
                            ):
                                nearest_anchor = self._select_predefined_interaction_anchor_for_ray(
                                    ray_origin_world,
                                    ray_direction_world,
                                    controller_predefined_anchor_states,
                                )
                            attach_candidate_anchor = (
                                selected_preview_anchor
                                if selected_preview_anchor is not None
                                else nearest_anchor
                            )
                            overlay_entry["attach_candidate"] = (
                                attach_candidate_anchor is not None
                                or overlay_entry["hit_world"] is not None
                            )
                            overlay_entry["attach_candidate_data"] = None
                            overlay_entry["preview_interaction_state"] = None
                            overlay_entry["selected_preview_anchor"] = selected_preview_anchor
                            overlay_entry["snap_edge"] = False
                            overlay_entry["attach_candidate_anchor_name"] = (
                                None
                                if attach_candidate_anchor is None
                                else attach_candidate_anchor["name"]
                            )
                            overlay_entry["attach_candidate_pixel"] = (
                                self._project_world_point_to_pixel(
                                    attach_candidate_anchor["center_world"],
                                    K_cuda,
                                    w2c_cuda,
                                    height,
                                    width,
                                )
                                if attach_candidate_anchor is not None
                                else overlay_entry["hit_pixel"]
                            )
                            interaction_state = controller_interaction_state[source]
                            overlay_entry["attachment_active"] = interaction_state is not None
                            overlay_entry["attach_active_anchor_name"] = (
                                None
                                if interaction_state is None
                                else interaction_state.get("anchor_name")
                            )
                            active_center_world = self._current_controller_attach_center_world(
                                interaction_state,
                                current_interaction_anchor_by_source[source],
                            )
                            overlay_entry["attach_active_pixel"] = self._project_world_point_to_pixel(
                                active_center_world,
                                K_cuda,
                                w2c_cuda,
                                height,
                                width,
                            )
                            controller_overlays.append(overlay_entry)
                            controller_overlay_by_source[source] = overlay_entry
                    for source in ("left", "right"):
                        self._log_controller_anchor_preview_transition(
                            source,
                            controller_anchor_preview_state[source],
                            controller_anchor_preview_state_cache,
                        )
                        self._log_controller_hit_transition(
                            source,
                            controller_overlay_by_source.get(source),
                            controller_hit_state_cache,
                        )
                        self._log_controller_attach_candidate_transition(
                            source,
                            controller_overlay_by_source.get(source),
                            controller_attach_candidate_state_cache,
                        )
                    overlay_draw_start = time.perf_counter()
                    self._draw_live_controller_overlay(frame, controller_overlays)
                    frame_comp_overlay_draw_submit += time.perf_counter() - overlay_draw_start

                timing_overlay_start = time.perf_counter()
                self._draw_timing_overlay(frame, last_total_time)
                frame_comp_timing_overlay_submit += time.perf_counter() - timing_overlay_start
                rgba_pack_start = time.perf_counter()
                frame_u8 = frame.clamp_(0, 255).to(torch.uint8)   # RGB uint8

                ####################pyh new rendering direclty render on GPU n(o GPU to CPU copying)
                # device→device copy into the mapped PBO, then update the texture and draw
                #convert to rgba
                frame_rgba[:, :, :3] = frame_u8
                frame_comp_rgba_pack_submit += time.perf_counter() - rgba_pack_start
                if quest_frame_panel is not None:
                    quest_publish_start = time.perf_counter()
                    publish_ok, quest_publish_stats = quest_frame_panel.publish_frame(frame_rgba)
                    frame_comp_quest_publish_wall += time.perf_counter() - quest_publish_start
                    if not publish_ok:
                        quest_frame_panel.stop()
                        quest_frame_panel = None

                if preview_display_active:
                    preview_path_start = time.perf_counter()
                    preview_sync_start = time.perf_counter()
                    torch.cuda.current_stream().synchronize()  # ensures frame_rgba is ready to be read
                    frame_comp_preview_sync_wall += time.perf_counter() - preview_sync_start
                    preview_copy_start = time.perf_counter()
                    mapping = reg.map()
                    try:
                        ptr, _ = mapping.device_ptr_and_size()
                        cpy2d.set_src_device(frame_rgba.data_ptr())
                        cpy2d.set_dst_device(ptr)
                        cpy2d(pbo_stream)

                        pbo_stream.synchronize()
                    finally:
                        mapping.unmap()
                    frame_comp_preview_copy_wall += time.perf_counter() - preview_copy_start

                    # Upload from PBO to texture (still on GPU)
                    preview_upload_start = time.perf_counter()
                    gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
                    gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, pbo)
                    gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, width, height, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None)
                    gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)

                    # Draw
                    fb_width, fb_height = glfw.get_framebuffer_size(window)
                    gl.glViewport(0, 0, fb_width, fb_height)
                    gl.glDisable(gl.GL_DEPTH_TEST)
                    gl.glClear(gl.GL_COLOR_BUFFER_BIT)
                    gl.glUseProgram(prog)
                    gl.glActiveTexture(gl.GL_TEXTURE0)
                    gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
                    gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)
                    gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
                    gl.glUseProgram(0)

                    glfw.swap_buffers(window)
                    frame_comp_preview_upload_draw_wall += time.perf_counter() - preview_upload_start
                    frame_comp_preview_path_wall += time.perf_counter() - preview_path_start
                glfw_poll_start = time.perf_counter()
                glfw.poll_events()
                frame_comp_glfw_poll_wall += time.perf_counter() - glfw_poll_start

                frame_timer_stop_start = time.perf_counter()
                frame_comp_gpu_time = frame_timer.stop()
                frame_comp_timer_stop_wall += time.perf_counter() - frame_timer_stop_start
                frame_comp_wall_time = time.perf_counter() - frame_comp_wall_start
                if frame_count > 1:
                    # Total frame compositing time
                    component_times["frame_compositing"].append(frame_comp_wall_time)
                    component_times["frame_compositing_gpu_timer"].append(frame_comp_gpu_time)
                    component_times["frame_comp_overlay_draw_submit"].append(frame_comp_overlay_draw_submit)
                    component_times["frame_comp_timing_overlay_submit"].append(frame_comp_timing_overlay_submit)
                    component_times["frame_comp_rgba_pack_submit"].append(frame_comp_rgba_pack_submit)
                    component_times["frame_comp_quest_publish_wall"].append(frame_comp_quest_publish_wall)
                    component_times["frame_comp_quest_process_check"].append(quest_publish_stats["process_check_wall"])
                    component_times["frame_comp_quest_pending_drain_nonblock"].append(quest_publish_stats["pending_drain_nonblock_wall"])
                    component_times["frame_comp_quest_pending_drain_block"].append(quest_publish_stats["pending_drain_block_wall"])
                    component_times["frame_comp_quest_gpu_to_cpu_wait"].append(quest_publish_stats["gpu_to_cpu_wait_wall"])
                    component_times["frame_comp_quest_gpu_to_cpu_copy"].append(quest_publish_stats["gpu_to_cpu_copy_cuda"])
                    component_times["frame_comp_quest_cpu_mmap_copy"].append(quest_publish_stats["cpu_mmap_copy_wall"])
                    component_times["frame_comp_quest_header_write"].append(quest_publish_stats["header_write_wall"])
                    component_times["frame_comp_quest_stage_enqueue"].append(quest_publish_stats["stage_enqueue_wall"])
                    component_times["frame_comp_quest_fallback_copy"].append(quest_publish_stats["fallback_copy_wall"])
                    component_times["frame_comp_preview_path_wall"].append(frame_comp_preview_path_wall)
                    component_times["frame_comp_preview_sync_wall"].append(frame_comp_preview_sync_wall)
                    component_times["frame_comp_preview_copy_wall"].append(frame_comp_preview_copy_wall)
                    component_times["frame_comp_preview_upload_draw_wall"].append(frame_comp_preview_upload_draw_wall)
                    component_times["frame_comp_glfw_poll_wall"].append(frame_comp_glfw_poll_wall)
                    component_times["frame_comp_timer_stop_wall"].append(frame_comp_timer_stop_wall)


                #LBs code
                if prev_x is not None:
                    with torch.no_grad():
                        interp_timer.start()
                        
                        #R reuse with dropping weight
                        current_pos, current_rot= lbs_with_rotation_reuse(
                            current_mass_nodes = x,
                            cache = rotation_cache,
                        )

                        interp_time = interp_timer.stop() 
                        gaussians._xyz = current_pos
                        gaussians._rotation = current_rot              

                    if frame_count > 1:
                        component_times["full_motion_interpolation"].append(interp_time)
                

                prev_x = x.clone()


                ############### Temporary timer ###############
                # Total loop time
                total_time = total_timer.stop()
                if frame_count > 1:
                    component_times["total"].append(total_time)

                # Calculate FPS
                fps = 1.0 / total_time
                if frame_count > 1:
                    fps_history.append(fps)
                last_total_time = total_time

                eval_png_write_wall_current = 0.0
                if eval_image_path and input_source in {"live_openxr", "live_openxr_controller"}:
                    should_save_frame = frame_count < 5 or (frame_count % 30 == 0)
                    if should_save_frame:
                        save_start = time.perf_counter()
                        save_path = os.path.join(view_render_path, f"{frame_count:05d}.png")
                        img_rgb = frame_u8.permute(2, 0, 1).float() / 255.0
                        torchvision.utils.save_image(img_rgb, save_path)
                        eval_png_write_wall_current = time.perf_counter() - save_start
                        if frame_count > 1:
                            component_times["eval_png_write_wall"].append(
                                eval_png_write_wall_current
                            )

                if render_profile and frame_count > 1:
                    quest_render_profile_frame = {
                        "rendering": float(render_time),
                        "frame_compositing": float(frame_comp_wall_time),
                        "frame_compositing_gpu_timer": float(frame_comp_gpu_time),
                        "frame_comp_overlay_draw_submit": float(frame_comp_overlay_draw_submit),
                        "frame_comp_timing_overlay_submit": float(
                            frame_comp_timing_overlay_submit
                        ),
                        "frame_comp_rgba_pack_submit": float(frame_comp_rgba_pack_submit),
                        "frame_comp_quest_publish_wall": float(frame_comp_quest_publish_wall),
                        "frame_comp_quest_process_check": float(
                            quest_publish_stats["process_check_wall"]
                        ),
                        "frame_comp_quest_pending_drain_nonblock": float(
                            quest_publish_stats["pending_drain_nonblock_wall"]
                        ),
                        "frame_comp_quest_pending_drain_block": float(
                            quest_publish_stats["pending_drain_block_wall"]
                        ),
                        "frame_comp_quest_gpu_to_cpu_wait": float(
                            quest_publish_stats["gpu_to_cpu_wait_wall"]
                        ),
                        "frame_comp_quest_gpu_to_cpu_copy": float(
                            quest_publish_stats["gpu_to_cpu_copy_cuda"]
                        ),
                        "frame_comp_quest_cpu_mmap_copy": float(
                            quest_publish_stats["cpu_mmap_copy_wall"]
                        ),
                        "frame_comp_quest_header_write": float(
                            quest_publish_stats["header_write_wall"]
                        ),
                        "frame_comp_quest_stage_enqueue": float(
                            quest_publish_stats["stage_enqueue_wall"]
                        ),
                        "frame_comp_quest_fallback_copy": float(
                            quest_publish_stats["fallback_copy_wall"]
                        ),
                        "frame_comp_preview_path_wall": float(frame_comp_preview_path_wall),
                        "frame_comp_preview_sync_wall": float(frame_comp_preview_sync_wall),
                        "frame_comp_preview_copy_wall": float(frame_comp_preview_copy_wall),
                        "frame_comp_preview_upload_draw_wall": float(
                            frame_comp_preview_upload_draw_wall
                        ),
                        "frame_comp_glfw_poll_wall": float(frame_comp_glfw_poll_wall),
                        "frame_comp_timer_stop_wall": float(frame_comp_timer_stop_wall),
                        "eval_png_write_wall": float(eval_png_write_wall_current),
                    }
                    self._render_profile_append_frame(
                        quest_render_profile_series,
                        quest_render_profile_rows,
                        frame_count,
                        quest_render_profile_frame,
                    )
                    if self._render_profile_should_log(frame_count, render_profile_every):
                        self._log_quest_render_profile_frame(
                            quest_display_mode,
                            frame_count,
                            quest_render_profile_frame,
                        )

                frame_count += 1

                prev_target = current_target
                if input_source in {"live_openxr", "live_openxr_controller"}:
                    if input_source == "live_openxr_controller":
                        current_target = controller_runtime_base_target.clone()
                    else:
                        current_target = recorded_base_target.clone()
                    if input_source == "live_openxr_controller":
                        current_target = self._compute_next_live_controller_target(
                            current_target,
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
                            x[: self.num_all_points],
                            controller_reset_triggered=controller_reset_triggered,
                            controller_motion_state_cache=controller_motion_state_cache,
                            frame_index=frame_count,
                            runtime_label="2d",
                        )
                else:
                    if frame_count < self.frame_len:
                        current_target = self.batch_controller_points[frame_count]
                    else:
                        print("Reached end of recorded control sequence")
                        break

        finally:

            #pyh add overall stat printing
            # --- Final Summary Statistics ---
            if frame_count > 1:

                frames_used_for_stats = len(component_times["total"])

                print(f"\n=== Final Summary (averaged over {frames_used_for_stats} frames) ===")
                #pyh save output for file as well
                log_lines = []
                log_lines.append(f"=== Final Summary (averaged over {frames_used_for_stats} frames) ===")

                total_frame_times = component_times["total"]
                total_time_seconds = sum(total_frame_times)
                average_fps = frames_used_for_stats / total_time_seconds
                average_frame_time = np.mean(total_frame_times)

                print(f"Average FPS: {average_fps:.2f}")
                print(f"Average Total Frame Time: {average_frame_time * 1000:.2f} ms")
                log_lines.append(f"Average FPS: {average_fps:.2f}")
                log_lines.append(f"Average Total Frame Time: {average_frame_time * 1000:.2f} ms")
                
                # Detailed breakdown by components
                components_to_report = [
                    "simulator",
                    "full_motion_interpolation",
                    "rendering",
                    "frame_compositing",
                ]

                for component_name in components_to_report:
                    component_times_list = component_times.get(component_name, [])
                    if component_times_list:
                        average_component_time = np.mean(component_times_list)
                        time_share_percentage = (average_component_time / average_frame_time) * 100
                        readable_name = component_name.replace('_', ' ').capitalize()
                        print(f"{readable_name}: {average_component_time * 1000:.2f} ms ({time_share_percentage:.1f}%)")
                        log_lines.append(f"{readable_name}: {average_component_time * 1000:.2f} ms ({time_share_percentage:.1f}%)")

                frame_comp_breakdown = [
                    ("frame_compositing_gpu_timer", "Frame compositing gpu timer"),
                    ("frame_comp_overlay_draw_submit", "Frame comp overlay draw submit"),
                    ("frame_comp_timing_overlay_submit", "Frame comp timing text submit"),
                    ("frame_comp_rgba_pack_submit", "Frame comp rgba pack submit"),
                    ("frame_comp_quest_publish_wall", "Frame comp quest publish"),
                    ("frame_comp_quest_process_check", "Frame comp quest process check"),
                    ("frame_comp_quest_pending_drain_nonblock", "Frame comp quest pending drain nonblock"),
                    ("frame_comp_quest_pending_drain_block", "Frame comp quest pending drain block"),
                    ("frame_comp_quest_gpu_to_cpu_wait", "Frame comp quest gpu->cpu wait"),
                    ("frame_comp_quest_gpu_to_cpu_copy", "Frame comp quest gpu->cpu copy"),
                    ("frame_comp_quest_cpu_mmap_copy", "Frame comp quest cpu mmap copy"),
                    ("frame_comp_quest_header_write", "Frame comp quest header write"),
                    ("frame_comp_quest_stage_enqueue", "Frame comp quest stage enqueue"),
                    ("frame_comp_quest_fallback_copy", "Frame comp quest fallback copy"),
                    ("frame_comp_preview_path_wall", "Frame comp local preview path"),
                    ("frame_comp_preview_sync_wall", "Frame comp local preview sync"),
                    ("frame_comp_preview_copy_wall", "Frame comp local preview copy"),
                    ("frame_comp_preview_upload_draw_wall", "Frame comp local preview upload draw"),
                    ("frame_comp_glfw_poll_wall", "Frame comp glfw poll"),
                    ("frame_comp_timer_stop_wall", "Frame comp final synchronize wait"),
                    ("eval_png_write_wall", "Eval png write"),
                ]
                print("Frame compositing breakdown:")
                log_lines.append("Frame compositing breakdown:")
                for component_name, readable_name in frame_comp_breakdown:
                    component_times_list = component_times.get(component_name, [])
                    if component_times_list:
                        average_component_time = np.mean(component_times_list)
                        print(f"{readable_name}: {average_component_time * 1000:.2f} ms")
                        log_lines.append(
                            f"{readable_name}: {average_component_time * 1000:.2f} ms"
                        )

                if render_profile:
                    render_profile_lines = self._render_profile_summary_lines(
                        quest_display_mode,
                        quest_render_profile_series,
                        quest_render_profile_keys,
                    )
                    for line in render_profile_lines:
                        print(line)
                    self._write_render_profile_outputs(
                        eval_image_path,
                        render_profile_lines,
                        quest_render_profile_rows,
                    )

                #pyh save the performance log to a file
                if eval_image_path:
                    os.makedirs(eval_image_path, exist_ok=True)
                    log_file_path = os.path.join(eval_image_path, "performance_summary.txt")
                    with open(log_file_path, "w") as log_file:
                        log_file.write("\n".join(log_lines))

            if live_hand_stream is not None:
                live_hand_stream.stop()
            if live_controller_stream is not None and live_controller_stream is not quest_frame_panel:
                live_controller_stream.stop()
            if quest_frame_panel is not None:
                quest_frame_panel.stop()
            if reg is not None:
                reg.unregister()
            if prog is not None:
                gl.glDeleteProgram(prog)
            if tex is not None:
                gl.glDeleteTextures([tex])
            if pbo is not None:
                gl.glDeleteBuffers(1, [pbo])
            if vao is not None:
                gl.glDeleteVertexArrays(1, [vao])
            cuda_ctx.pop()

    
    def _create_gs_view(self, w2c, intrinsic, height, width):
        R = np.transpose(w2c[:3, :3])
        T = w2c[:3, 3]
        K = torch.tensor(intrinsic, dtype=torch.float32, device="cuda")
        zoom_out = 1  # >1 means zoom out (wider angle)
        K[0,0] /= zoom_out
        K[1,1] /= zoom_out
        focal_length_x = K[0, 0]
        focal_length_y = K[1, 1]
        FovY = focal2fov(focal_length_y, height)
        FovX = focal2fov(focal_length_x, width)
        view = Camera(
            (width, height),
            colmap_id="0000",
            R=R,
            T=T,
            FoVx=FovX,
            FoVy=FovY,
            depth_params=None,
            image=None,
            invdepthmap=None,
            image_name="0000",
            uid="0000",
            data_device="cuda",
            train_test_exp=None,
            is_test_dataset=None,
            is_test_view=None,
            K=K,
            normal=None,
            depth=None,
            occ_mask=None,
        )
        return view, K

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
        compose_cache["scene_color_cpu"] = torch.empty(
            (height, width, 4),
            dtype=torch.float32,
            pin_memory=True,
        )
        compose_cache["scene_depth_cpu"] = torch.empty(
            (height, width),
            dtype=torch.float32,
            pin_memory=True,
        )
        compose_cache["scene_color"] = torch.empty(
            (height, width, 4),
            dtype=torch.float32,
            device=cfg.device,
        )
        compose_cache["scene_depth"] = torch.empty(
            (height, width),
            dtype=torch.float32,
            device=cfg.device,
        )
        return compose_cache

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

        if torch.is_tensor(scene_color_rgba) and torch.is_tensor(scene_depth):
            scene_color_t = scene_color_rgba.to(device=cfg.device, dtype=torch.float32)
            scene_depth_t = scene_depth.to(device=cfg.device, dtype=torch.float32)
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
            return scene_color_t, scene_depth_t

        scene_color_np = np.asarray(scene_color_rgba)
        scene_depth_np = np.asarray(scene_depth, dtype=np.float32)
        if scene_color_np.shape[:2] != (target_height, target_width):
            scene_color_np = cv2.resize(
                scene_color_np,
                (target_width, target_height),
                interpolation=cv2.INTER_LINEAR,
            )
        if scene_depth_np.shape[:2] != (target_height, target_width):
            scene_depth_np = cv2.resize(
                scene_depth_np,
                (target_width, target_height),
                interpolation=cv2.INTER_NEAREST,
            )

        scene_color_np = np.ascontiguousarray(scene_color_np)
        scene_depth_np = np.ascontiguousarray(scene_depth_np, dtype=np.float32)
        if compose_cache is None:
            return (
                torch.from_numpy(scene_color_np).to(
                    device=cfg.device,
                    dtype=torch.float32,
                ),
                torch.from_numpy(scene_depth_np).to(
                    device=cfg.device,
                    dtype=torch.float32,
                ),
            )

        compose_cache = self._ensure_immersive_compose_cache(
            compose_cache,
            target_height,
            target_width,
        )
        compose_cache["scene_color_cpu"].copy_(
            torch.as_tensor(scene_color_np, dtype=torch.float32),
            non_blocking=False,
        )
        compose_cache["scene_depth_cpu"].copy_(
            torch.as_tensor(scene_depth_np, dtype=torch.float32),
            non_blocking=False,
        )
        compose_cache["scene_color"].copy_(
            compose_cache["scene_color_cpu"],
            non_blocking=True,
        )
        compose_cache["scene_depth"].copy_(
            compose_cache["scene_depth_cpu"],
            non_blocking=True,
        )
        return compose_cache["scene_color"], compose_cache["scene_depth"]

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
    ):
        device = source_color.device
        dtype = torch.float32
        source_color_t = source_color.to(device=device, dtype=dtype)
        source_depth_t = source_depth.to(device=device, dtype=dtype)
        source_height = int(source_depth_t.shape[0])
        source_width = int(source_depth_t.shape[1])
        target_height = int(target_height)
        target_width = int(target_width)
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
            f"scene_reproject_{eye_label}_cuda",
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
                & (target_u >= -2.0)
                & (target_u < float(target_width) + 2.0)
                & (target_v >= -2.0)
                & (target_v < float(target_height) + 2.0)
            )
            if int(projected_valid.sum().item()) > 0:
                projected_depth = target_z[projected_valid]
                projected_u = target_u[projected_valid]
                projected_v = target_v[projected_valid]
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
                f"scene_reproject_valid_pre_{eye_label}_ratio"
            ] = pre_fill_ratio
            if repair_roi_bounds is not None:
                render_profile_frame[
                    f"scene_reproject_roi_pre_{eye_label}_ratio"
                ] = self._scene_valid_roi_coverage(target_valid, repair_roi_bounds)

        hole_fill_span = self._render_profile_begin_cuda_span(
            render_profile_frame,
            f"scene_reproject_hole_fill_{eye_label}_cuda",
        )
        if repair_roi_bounds is not None:
            target_color, target_depth, target_valid = self._fill_immersive_reprojected_scene_holes(
                target_color,
                target_depth,
                target_valid,
                background_rgba=background_rgba,
                roi_bounds=repair_roi_bounds,
                target_coverage=float(self.IMMERSIVE_REPROJECT_ROI_TARGET_COVERAGE),
            )
        self._render_profile_end_cuda_span(render_profile_frame, hole_fill_span)
        if render_profile_frame is not None and eye_label is not None:
            render_profile_frame[
                f"scene_reproject_valid_post_{eye_label}_ratio"
            ] = float(target_valid.to(dtype=torch.float32).mean().item())
            if repair_roi_bounds is not None:
                render_profile_frame[
                    f"scene_reproject_roi_post_{eye_label}_ratio"
                ] = self._scene_valid_roi_coverage(target_valid, repair_roi_bounds)
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
        center_scene_color, center_scene_depth = scene_renderer.render_eye(
            center_eye_pose_world,
            center_scene_intrinsic,
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
            source_cache=None if reproject_caches is None else reproject_caches.get("source"),
        )
        debug = {
            "mode": "reproject_from_center",
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
                reproject_cache=None if reproject_caches is None else reproject_caches.get(eye_name),
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
            )
            if right_scene_render_start is not None:
                self._render_profile_add_wall_time(
                    render_profile_frame,
                    "scene_render_right_wall",
                    time.perf_counter() - right_scene_render_start,
                )
            return left_scene_color, left_scene_depth, right_scene_color, right_scene_depth

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
