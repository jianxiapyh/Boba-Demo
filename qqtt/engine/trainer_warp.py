# Boba runtime path for single-instance and batched execution.
# instance == 1 uses the single-instance path.
# instance > 1 enables batched optimizations.
import json
import math
import os
import pickle
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
import warp as wp

import open3d as o3d

from demos.demo2.control import add_vectors_clamped, control_vector_to_step
from gaussian_splatting.scene.cameras import Camera
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.dynamic_utils import (
    lbs_with_rotation_reuse,
    build_rotation_reuse_cache,
    knn_weights_sparse,
    get_topk_indices,
)
from gaussian_splatting.utils.graphics_utils import focal2fov
from qqtt.data import RealData
from qqtt.model.diff_simulator import (
    SpringMassSystemWarp,
)
from qqtt.utils import logger, cfg
from qqtt.utils.gaussian import (
    build_batch_images_render_view,
    build_instance_selective_render_view,
    remove_gaussians_with_low_opacity,
    load_shared_batched_gaussians,
    normalize_gaussian_render_mode,
)

SIM_FORCE_MODE_GATHER = "gather"
SIM_FORCE_MODE_TEMPLATE_STATE_BATCHED_ATOMIC = "template_state_batched_atomic"
SIM_FORCE_MODES = (
    SIM_FORCE_MODE_GATHER,
    SIM_FORCE_MODE_TEMPLATE_STATE_BATCHED_ATOMIC,
)


class BatchedReplayCheckError(RuntimeError):
    def __init__(
        self,
        message,
        *,
        frame_idx=None,
        batch_element=None,
        hinted_instance=None,
        original_error=None,
    ):
        super().__init__(message)
        self.frame_idx = frame_idx
        self.batch_element = batch_element
        self.hinted_instance = hinted_instance
        self.original_error = original_error

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


class RenderComponentProfiler:
    COMPONENT_FIELDS = [
        "render_gsplat_total_ms",
        "prepare_inputs_ms",
        "fully_fused_projection_ms",
        "spherical_harmonics_ms",
        "isect_tiles_ms",
        "isect_tiles_count_kernel_ms",
        "isect_tiles_cumsum_ms",
        "isect_tiles_emit_kernel_ms",
        "isect_tiles_sort_ms",
        "isect_tiles_cuda_total_ms",
        "isect_visible_gaussians",
        "isect_total_tile_intersections",
        "isect_avg_tiles_per_gaussian",
        "isect_max_tiles_per_gaussian",
        "isect_offset_encode_ms",
        "rasterize_to_pixels_ms",
        "background_depth_finalize_ms",
        "format_output_ms",
        "shared_projection_loop_ms",
        "shared_template_gather_ms",
        "shared_projection_cat_ms",
        "densify_projection_metadata_ms",
    ]

    def __init__(self):
        self.frames = []
        self.current_frame = None

    def begin_frame(self, total_gaussians=None, gaussians_per_instance=None):
        self.current_frame = {
            "total_gaussians": total_gaussians,
            "gaussians_per_instance": gaussians_per_instance,
            "components": {},
        }

    def end_frame(self):
        if self.current_frame is not None:
            self.frames.append(self.current_frame)
        self.current_frame = None

    def discard_frame(self):
        self.current_frame = None

    class _RecordScope:
        def __init__(self, profiler, name):
            self.profiler = profiler
            self.name = name
            self.start_time = None

        def __enter__(self):
            if self.profiler.current_frame is None:
                return self
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.start_time = time.perf_counter()
            return self

        def __exit__(self, exc_type, exc, tb):
            if self.profiler.current_frame is None or self.start_time is None:
                return False
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - self.start_time) * 1000.0
            components = self.profiler.current_frame["components"]
            components[self.name] = components.get(self.name, 0.0) + elapsed_ms
            return False

    def record(self, name):
        return self._RecordScope(self, name)

    def record_value(self, name, value):
        if self.current_frame is None:
            return
        components = self.current_frame["components"]
        components[name] = components.get(name, 0.0) + float(value)

    def _mean_frame_value(self, field):
        values = [
            frame[field]
            for frame in self.frames
            if frame.get(field) is not None
        ]
        if not values:
            return None
        return float(np.mean(values))

    def _mean_component_value(self, component_name):
        values = [
            frame["components"].get(component_name)
            for frame in self.frames
            if frame["components"].get(component_name) is not None
        ]
        if not values:
            return None
        return float(np.mean(values))

    def summary(self, metadata):
        profile = dict(metadata)
        profile["frames_profiled"] = len(self.frames)
        profile["total_gaussians"] = self._mean_frame_value("total_gaussians")
        profile["gaussians_per_instance"] = self._mean_frame_value(
            "gaussians_per_instance"
        )
        for component_name in self.COMPONENT_FIELDS:
            profile[component_name] = self._mean_component_value(component_name)
        return profile

    def write_json(self, output_dir, metadata):
        if output_dir is None:
            return
        os.makedirs(output_dir, exist_ok=True)
        profile_path = os.path.join(output_dir, "render_component_profile.json")
        with open(profile_path, "w", encoding="utf-8") as f:
            json.dump(self.summary(metadata), f, indent=2)


def perf_component_label(component_name):
    labels = {
        "simulator": "Simulator",
        "full_motion_interpolation": "Linear Blend Skinning",
        "rendering": "Rendering",
        "frame_compositing": "Frame compositing",
    }
    return labels.get(component_name, component_name.replace("_", " ").capitalize())


class InvPhyTrainerWarp:
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
            self.controller_points = self.dataset.controller_points
            self.structure_points = self.dataset.structure_points
            self.num_all_points = self.dataset.num_all_points
        elif cfg.data_type == "synthetic":
            print(f"synthetic data detected")
            import pdb
            pdb.set_trace()
        else:
            raise ValueError(f"Data type {cfg.data_type} not supported")
        
        first_frame_controller_points = self.controller_points[0]
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
        self.controller_points_group = None
        self.multi_frame_len = None
        self.num_input_trajectories = None

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

    def build_signed_incidence_map(self, base_springs, object_massnode_single, device="cuda:0"):
        """
        Build a flattened signed incidence map for the gather-based spring assembly.

        Each row corresponds to one object mass node. Positive entries mean the node
        adds the corresponding spring force; negative entries mean it subtracts it.
        """
        springs_cpu = base_springs.detach().to(dtype=torch.int64, device="cpu")
        per_node_incidence = [[] for _ in range(int(object_massnode_single))]

        for spring_idx, endpoints in enumerate(springs_cpu.tolist()):
            endpoint_a, endpoint_b = endpoints
            signed_spring_id = spring_idx + 1

            if endpoint_a < object_massnode_single:
                per_node_incidence[endpoint_a].append(signed_spring_id)
            if endpoint_b < object_massnode_single:
                per_node_incidence[endpoint_b].append(-signed_spring_id)

        max_incident_springs = max((len(entries) for entries in per_node_incidence), default=0)
        incidence_map = torch.zeros(
            (int(object_massnode_single), max_incident_springs),
            dtype=torch.int32,
        )

        for node_idx, entries in enumerate(per_node_incidence):
            if entries:
                incidence_map[node_idx, : len(entries)] = torch.tensor(entries, dtype=torch.int32)

        logger.info(
            f"Built signed incidence map for gather assembly with max incident spring count {max_incident_springs}"
        )
        return incidence_map.flatten().to(device=device), max_incident_springs

    def check_controller_group_same_start(
        self,
        controller_points_group,
        atol=1e-6,
        rtol=0.0,
        allow_global_translation=False,
        translation_mode="mean",
        verbose=True,
    ):
        assert controller_points_group.ndim == 4 and controller_points_group.shape[-1] == 3, (
            f"Expected (N,T,C,3), got {tuple(controller_points_group.shape)}"
        )

        num_instances, num_frames, num_controllers, _ = controller_points_group.shape
        start = controller_points_group[:, 0]
        ref = start[0:1]

        if allow_global_translation:
            if translation_mode == "mean":
                translation = start.mean(dim=1, keepdim=True) - ref.mean(dim=1, keepdim=True)
            elif translation_mode == "first":
                translation = start[:, 0:1, :] - ref[:, 0:1, :]
            else:
                raise ValueError("translation_mode must be 'mean' or 'first'")
            diff = (start - translation) - ref
        else:
            diff = start - ref

        per_instance_max = diff.abs().amax(dim=(1, 2))
        tol = atol + rtol * ref.abs().amax(dim=(1, 2))
        ok = (per_instance_max <= tol).tolist()
        all_ok = all(ok)

        if verbose:
            print(
                f"[Test] N={num_instances}, T={num_frames}, C={num_controllers}, "
                f"allow_translation={allow_global_translation} ({translation_mode})"
            )
            print(f"[Test] all_ok={all_ok}")
            if not all_ok:
                bad = [i for i, passed in enumerate(ok) if not passed]
                print(f"[Test] mismatching instances: {bad}")
            print(f"[Test] per-instance max abs diff: {per_instance_max.detach().cpu().numpy()}")

        return all_ok, per_instance_max, ok

    def load_controller_points_group_pkl(self, pkl_path, device="cuda"):
        with open(pkl_path, "rb") as handle:
            root = pickle.load(handle)

        if not isinstance(root, dict):
            raise TypeError(f"PKL root must be dict, got {type(root)}")
        if "controller_points_group" not in root:
            raise KeyError(
                f"Missing key 'controller_points_group' in {pkl_path}. Keys={list(root.keys())}"
            )

        group = root["controller_points_group"]
        if not isinstance(group, list) or len(group) == 0:
            raise TypeError(
                f"'controller_points_group' must be a non-empty list, got {type(group)} "
                f"len={len(group) if hasattr(group, '__len__') else '??'}"
            )

        frame_count = controller_count = None
        tensors = []
        for idx, arr in enumerate(group):
            arr = np.asarray(arr)
            if arr.ndim != 3 or arr.shape[-1] != 3:
                raise ValueError(
                    f"controller_points_group[{idx}] shape {arr.shape}, expected (T, C, 3)"
                )
            cur_frames, cur_controllers, _ = arr.shape
            if frame_count is None:
                frame_count, controller_count = cur_frames, cur_controllers
            elif (cur_frames, cur_controllers) != (frame_count, controller_count):
                raise ValueError(
                    f"controller_points_group[{idx}] has (T,C)=({cur_frames},{cur_controllers}) "
                    f"but expected ({frame_count},{controller_count})."
                )
            tensors.append(torch.from_numpy(arr.astype(np.float32, copy=False)).to(device=device))

        return torch.stack(tensors, dim=0)

    def _ensure_controller_points_group_loaded(self):
        if self.controller_points_group is not None:
            return self.controller_points_group

        controller_group_path = Path(cfg.data_path).parent / "multi_ctrls.pkl"
        if not controller_group_path.exists():
            raise FileNotFoundError(
                f"Missing multi-instance controller trajectories: {controller_group_path}"
            )

        controller_points_group = self.load_controller_points_group_pkl(
            controller_group_path, device=cfg.device
        )
        self.check_controller_group_same_start(controller_points_group, atol=1e-5)

        if controller_points_group.shape[2] != self.controller_points.shape[1]:
            raise ValueError(
                "multi_ctrls.pkl controller count does not match single-instance controller count: "
                f"{controller_points_group.shape[2]} vs {self.controller_points.shape[1]}"
            )

        reference_start = self.controller_points[0].to(
            device=controller_points_group.device,
            dtype=controller_points_group.dtype,
        )
        if not torch.allclose(
            controller_points_group[0, 0], reference_start, atol=1e-5, rtol=0.0
        ):
            max_diff = (
                controller_points_group[0, 0] - reference_start
            ).abs().max().item()
            raise ValueError(
                "multi_ctrls.pkl frame-0 controller pose does not match self.controller_points[0]. "
                f"max abs diff: {max_diff:.6e}"
            )

        self.controller_points_group = controller_points_group
        self.multi_frame_len = int(controller_points_group.shape[1])
        self.num_input_trajectories = int(controller_points_group.shape[0])
        return self.controller_points_group

    def _load_render_backend(self):
        import cv2
        import glfw
        from OpenGL import GL as gl
        import pycuda.driver as cuda_driver
        from pycuda.gl import RegisteredBuffer, graphics_map_flags
        from gaussian_splatting.gaussian_renderer import render as render_gaussian

        return SimpleNamespace(
            cv2=cv2,
            glfw=glfw,
            gl=gl,
            cuda_driver=cuda_driver,
            RegisteredBuffer=RegisteredBuffer,
            graphics_map_flags=graphics_map_flags,
            render_gaussian=render_gaussian,
        )

    def _default_diagonal_instance_offsets(self, batch_size, dtype):
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        if batch_size == 1:
            return torch.zeros((1, 3), dtype=dtype, device=cfg.device)

        offset_step = torch.tensor([10, 10, 0], dtype=dtype, device=cfg.device)
        instance_ids = torch.arange(
            batch_size, device=cfg.device, dtype=dtype
        ).unsqueeze(1)
        return (instance_ids * offset_step.view(1, 3)).contiguous()

    def _build_runtime_core(
        self,
        model_path,
        gs_path,
        n_dup=0,
        instance_offsets=None,
        controller_points_group_override=None,
        gaussian_render_mode="shared_template",
        force_shared_batched_gaussians=False,
        sim_force_mode=SIM_FORCE_MODE_GATHER,
    ):
        gaussian_render_mode = normalize_gaussian_render_mode(gaussian_render_mode)
        if sim_force_mode not in SIM_FORCE_MODES:
            raise ValueError(
                f"sim_force_mode must be one of {SIM_FORCE_MODES}. "
                f"Received: {sim_force_mode}"
            )
        if cfg.self_collision:
            print(f"collision dist {cfg.collision_dist}")
        else:
            print("no collision flag set")

        logger.info(f"Load model from {model_path}")
        checkpoint = torch.load(model_path, map_location=cfg.device)

        trained_spring_Y = checkpoint["spring_Y"]
        trained_collide_elas = checkpoint["collide_elas"]
        trained_collide_fric = checkpoint["collide_fric"]
        trained_collide_object_elas = checkpoint["collide_object_elas"]
        trained_collide_object_fric = checkpoint["collide_object_fric"]

        trained_spring_Y = trained_spring_Y[self.spring_permutation]

        obj_init_vertices = self.init_vertices[: self.num_all_points]
        ctrl_init_vertices = self.init_vertices[self.num_all_points :]
        n_vert_single_obj = obj_init_vertices.shape[0]
        n_vert_single_ctrl = ctrl_init_vertices.shape[0]
        frame_len = int(self.dataset.frame_len)
        controller_points_group = None
        batch_size = n_dup + 1
        use_controller_group = controller_points_group_override is not None or n_dup > 0
        if use_controller_group:
            if controller_points_group_override is None:
                controller_points_group = self._ensure_controller_points_group_loaded()
            else:
                controller_points_group = controller_points_group_override.to(
                    device=cfg.device,
                    dtype=self.controller_points.dtype,
                ).contiguous()
                if controller_points_group.ndim != 4 or controller_points_group.shape[-1] != 3:
                    raise ValueError(
                        "controller_points_group_override must have shape (N,T,C,3), "
                        f"got {tuple(controller_points_group.shape)}"
                    )
                if controller_points_group.shape[2] != self.controller_points.shape[1]:
                    raise ValueError(
                        "controller_points_group_override controller count does not match "
                        f"the case data: {controller_points_group.shape[2]} vs "
                        f"{self.controller_points.shape[1]}"
                    )
            required_instances = batch_size
            available_instances = int(controller_points_group.shape[0])
            if available_instances < required_instances:
                raise ValueError(
                    "Controller trajectory bank does not contain enough trajectories "
                    f"for the requested batch size {required_instances}. "
                    f"Found {available_instances}."
                )
            frame_len = self.multi_frame_len
            if controller_points_group_override is not None:
                frame_len = int(controller_points_group.shape[1])

        if instance_offsets is None:
            resolved_instance_offsets = self._default_diagonal_instance_offsets(
                batch_size=batch_size,
                dtype=obj_init_vertices.dtype,
            )
        else:
            resolved_instance_offsets = instance_offsets.to(
                device=cfg.device,
                dtype=obj_init_vertices.dtype,
            ).contiguous()
            expected_shape = (batch_size, 3)
            if tuple(resolved_instance_offsets.shape) != expected_shape:
                raise ValueError(
                    "instance_offsets must have shape "
                    f"{expected_shape}, got {tuple(resolved_instance_offsets.shape)}"
                )

        out_init_vertices = []
        out_init_velocities = []
        out_controller_points = []

        for dup_i in range(batch_size):
            shift = resolved_instance_offsets[dup_i]
            obj_v = obj_init_vertices + shift

            if self.init_velocities is not None:
                out_init_velocities.append(self.init_velocities)

            out_init_vertices.append(obj_v)

        base_ctrl_vert_offset = batch_size * n_vert_single_obj

        for dup_i in range(batch_size):
            shift = resolved_instance_offsets[dup_i]
            ctrl_v = ctrl_init_vertices + shift
            if use_controller_group:
                new_controller_points = controller_points_group[dup_i] + shift
            else:
                new_controller_points = self.controller_points
            out_init_vertices.append(ctrl_v)
            out_controller_points.append(new_controller_points)

        self.batch_init_vertices = torch.cat(out_init_vertices, dim=0)
        self.batch_init_velocities = None
        if self.init_velocities is not None:
            self.batch_init_velocities = torch.cat(out_init_velocities, dim=0)

        self.batch_controller_points = torch.cat(out_controller_points, dim=1)

        expected_total = base_ctrl_vert_offset + batch_size * n_vert_single_ctrl
        print(
            f"[Check] single instance object mass node {n_vert_single_obj}, controller mass node {n_vert_single_ctrl}"
        )
        print(
            f"[CHECK] total mass nodes {self.batch_init_vertices.shape[0]}, expected {expected_total}"
        )
        print(
            "batch_init_vertices:",
            type(self.batch_init_vertices),
            self.batch_init_vertices.shape,
            self.batch_init_vertices.dtype,
            self.batch_init_vertices.device,
        )
        print(
            "batch_controller_points:",
            type(self.batch_controller_points),
            self.batch_controller_points.shape,
            self.batch_controller_points.dtype,
            self.batch_controller_points.device,
        )

        signed_incidence_map_flat = None
        max_incident_springs = 0
        if n_dup > 0 and sim_force_mode == SIM_FORCE_MODE_GATHER:
            signed_incidence_map_flat, max_incident_springs = self.build_signed_incidence_map(
                self.init_springs,
                n_vert_single_obj,
                device=cfg.device,
            )

        self.simulator = SpringMassSystemWarp(
            base_springs=self.init_springs,
            base_rest_lengths=self.init_rest_lengths,
            init_masses=self.init_masses,
            init_masks=self.init_masks,
            signed_incidence_map=signed_incidence_map_flat,
            max_incident_springs=max_incident_springs,
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
            spring_Y=trained_spring_Y,
            object_massnodes_total=base_ctrl_vert_offset,
            object_massnodes_single=n_vert_single_obj,
            controller_massnodes_single=n_vert_single_ctrl,
            controller_rest_location=self.batch_controller_points[0],
            number_of_instance=batch_size,
            sim_force_mode=sim_force_mode,
        )

        self.simulator.set_init_state(
            self.simulator.wp_init_vertices, self.simulator.wp_init_velocities
        )

        if self.simulator.object_collision_flag:
            self.simulator.create_resting_case()

        self.simulator.create_cuda_graph()

        if n_dup == 0 and not force_shared_batched_gaussians:
            gaussians = GaussianModel(sh_degree=3)
            gaussians.load_ply(gs_path)
            gaussians = remove_gaussians_with_low_opacity(gaussians, 0.1)
            gaussians.isotropic = True
            render_gaussians = gaussians
            n_gaussians_single_obj = gaussians._xyz.shape[0]
            xyz_rest_single = gaussians.get_xyz[:n_gaussians_single_obj]
            rot_rest_single = gaussians.get_rotation[:n_gaussians_single_obj]
        else:
            gaussians, render_gaussians = load_shared_batched_gaussians(
                gs_path=gs_path,
                number_of_instance=batch_size,
                instance_offsets=resolved_instance_offsets,
                sh_degree=3,
                opacity_threshold=0.1,
                gaussian_render_mode=gaussian_render_mode,
            )
            n_gaussians_single_obj = gaussians.gaussians_per_instance
            xyz_rest_single = gaussians.xyz_rest_single
            rot_rest_single = gaussians.rotation_rest_single

        torch.cuda.empty_cache()

        prev_x = wp.to_torch(
            self.simulator.wp_states[0].wp_x, requires_grad=False
        ).clone()

        current_pos = gaussians.get_xyz

        rest_mass_node_single = prev_x[:n_vert_single_obj]
        relations_single = get_topk_indices(rest_mass_node_single, K=3)
        weights_single, weights_indices_single = knn_weights_sparse(
            rest_mass_node_single, current_pos[:n_gaussians_single_obj], K=3
        )

        rotation_cache = build_rotation_reuse_cache(
            weights_indices=weights_indices_single,
            weights=weights_single,
            relations=relations_single,
            mass_nodes_rest=rest_mass_node_single,
            gaussians_xyz_rest=xyz_rest_single,
            gaussians_quat_rest=rot_rest_single,
            device=cfg.device,
            mass_node_per_instance=n_vert_single_obj,
            gaussians_per_instance=n_gaussians_single_obj,
            number_of_instance=batch_size,
        )

        instance_offsets = resolved_instance_offsets

        return SimpleNamespace(
            gaussians=gaussians,
            render_gaussians=render_gaussians,
            rotation_cache=rotation_cache,
            prev_x=prev_x,
            prev_target=self.batch_controller_points[0],
            current_target=self.batch_controller_points[0],
            frame_count=0,
            frame_len=frame_len,
            batch_size=batch_size,
            object_nodes_per_instance=n_vert_single_obj,
            controller_nodes_per_instance=n_vert_single_ctrl,
            gaussians_per_instance=n_gaussians_single_obj,
            instance_offsets=instance_offsets,
            gaussian_render_mode=gaussian_render_mode,
            sim_force_mode=sim_force_mode,
        )

    def _validate_batched_render_request(self, render_mode, instance_id, batch_size):
        if render_mode not in ("instance", "batch_images"):
            raise ValueError(
                "render_mode must be 'instance' or 'batch_images'. "
                f"Received: {render_mode}"
            )

        if render_mode == "instance":
            if instance_id is None:
                raise ValueError(
                    "instance_id is required when render_mode='instance'."
                )
            instance_id = int(instance_id)
            if instance_id < 0 or instance_id >= int(batch_size):
                raise ValueError(
                    f"instance_id must be in [0, {int(batch_size) - 1}], got {instance_id}"
                )
            return instance_id

        if instance_id is not None:
            raise ValueError(
                "instance_id can only be provided when render_mode='instance'."
            )
        return None

    def _select_runtime_points(self, points, runtime, render_mode, instance_id):
        if render_mode != "instance":
            raise ValueError(
                "runtime point selection is only valid for render_mode='instance'."
            )

        object_start = instance_id * runtime.object_nodes_per_instance
        object_end = object_start + runtime.object_nodes_per_instance
        controller_block_start = runtime.batch_size * runtime.object_nodes_per_instance
        controller_start = (
            controller_block_start
            + instance_id * runtime.controller_nodes_per_instance
        )
        controller_end = controller_start + runtime.controller_nodes_per_instance

        point_chunks = [points[object_start:object_end]]
        if runtime.controller_nodes_per_instance > 0:
            point_chunks.append(points[controller_start:controller_end])

        selected_points = torch.cat(point_chunks, dim=0)
        return selected_points - runtime.instance_offsets[instance_id]

    def _composite_batch_images_without_shadows(self, batch_rendering, overlay):
        if batch_rendering.ndim != 4 or batch_rendering.shape[1] != 4:
            raise ValueError(
                "batch image rendering expects [B, 4, H, W], got "
                f"{tuple(batch_rendering.shape)}"
            )

        images = batch_rendering.permute(0, 2, 3, 1).detach().clamp(0, 1)
        image_mask = torch.logical_and(
            (images[..., :3] != 1.0).any(dim=3),
            images[..., 3] > 100 / 255,
        )
        alpha = torch.where(
            image_mask[..., None],
            images[..., 3:4],
            torch.zeros_like(images[..., 3:4]),
        )
        frames = overlay.unsqueeze(0) * (1.0 - alpha) + images[..., :3] * alpha * 255.0
        return frames, image_mask

    def _resolve_batch_image_render_size(
        self,
        batch_image_resolution,
        native_width,
        native_height,
    ):
        if batch_image_resolution == "native":
            return int(native_width), int(native_height)
        if batch_image_resolution == "640x480":
            return 640, 480
        raise ValueError(
            "batch_image_resolution must be 'native' or '640x480'. "
            f"Received: {batch_image_resolution}"
        )

    def _scale_intrinsic_for_render_size(
        self,
        intrinsic,
        native_width,
        native_height,
        render_width,
        render_height,
    ):
        scaled = np.asarray(intrinsic, dtype=np.float32).copy()
        x_scale = float(render_width) / float(native_width)
        y_scale = float(render_height) / float(native_height)
        scaled[0, 0] *= x_scale
        scaled[0, 2] *= x_scale
        scaled[1, 1] *= y_scale
        scaled[1, 2] *= y_scale
        return scaled

    def _make_batch_image_grid(self, batch_frames, batch_grid_cols):
        batch_size = int(batch_frames.shape[0])
        if batch_size < 1:
            raise ValueError("batch image grid requires at least one frame.")
        if batch_frames.ndim != 4 or batch_frames.shape[-1] != 3:
            raise ValueError(
                "batch image grid expects [B, H, W, 3], got "
                f"{tuple(batch_frames.shape)}"
            )

        cols = (
            int(batch_grid_cols)
            if batch_grid_cols is not None
            else int(math.ceil(math.sqrt(batch_size)))
        )
        if cols < 1:
            raise ValueError("--batch_grid_cols must be a positive integer.")

        _, tile_height, tile_width, channels = batch_frames.shape
        rows = int(math.ceil(batch_size / float(cols)))
        grid = torch.full(
            (rows * tile_height, cols * tile_width, channels),
            255.0,
            device=batch_frames.device,
            dtype=batch_frames.dtype,
        )
        for batch_idx in range(batch_size):
            row = batch_idx // cols
            col = batch_idx % cols
            grid[
                row * tile_height : (row + 1) * tile_height,
                col * tile_width : (col + 1) * tile_width,
                :,
            ] = batch_frames[batch_idx]
        return grid

    def _parse_linalg_batch_element(self, exc):
        import re

        match = re.search(r"Batch element (\d+)", str(exc))
        if not match:
            return None
        return int(match.group(1))

    def _demo2_replay_end(self, runtime, replay_start, replay_end):
        replay_start = int(replay_start)
        if replay_start < 0 or replay_start >= int(runtime.frame_len):
            raise ValueError(
                f"replay_start must be in [0, {int(runtime.frame_len) - 1}], "
                f"got {replay_start}"
            )
        if replay_end is None:
            replay_end = int(runtime.frame_len)
        replay_end = int(replay_end)
        if replay_end <= replay_start or replay_end > int(runtime.frame_len):
            raise ValueError(
                f"replay_end must be in ({replay_start}, {int(runtime.frame_len)}], "
                f"got {replay_end}"
            )
        return replay_end

    def _snapshot_demo2_reset_state(self, runtime):
        cache = runtime.rotation_cache
        cache_state = {
            "R_cache": cache["R_cache"].detach().clone(),
            "F_prev": cache["F_prev"].detach().clone(),
            "rotation_computed": cache["rotation_computed"].detach().clone(),
            "Q_cache_bm": cache["Q_cache_bm"].detach().clone(),
            "motions_bm_fp32": torch.zeros_like(cache["motions_bm_fp32"]),
        }
        return SimpleNamespace(
            gaussian_xyz=runtime.gaussians._xyz.detach().clone(),
            gaussian_rotation=runtime.gaussians._rotation.detach().clone(),
            object_x=runtime.prev_x.detach().clone(),
            object_v=wp.to_torch(
                self.simulator.wp_init_velocities, requires_grad=False
            ).detach().clone(),
            prev_x=runtime.prev_x.detach().clone(),
            cache_state=cache_state,
        )

    def _restore_demo2_reset_state(self, runtime, reset_state):
        self.simulator.set_init_state(
            self.simulator.wp_init_vertices,
            self.simulator.wp_init_velocities,
        )
        runtime.gaussians._xyz = reset_state.gaussian_xyz.clone()
        runtime.gaussians._rotation = reset_state.gaussian_rotation.clone()
        runtime.prev_x = reset_state.prev_x.clone()

        cache = runtime.rotation_cache
        for key, value in reset_state.cache_state.items():
            cache[key].copy_(value)

    def _restore_demo2_instance_reset_state(
        self,
        runtime,
        reset_state,
        session_ids,
        x_tensor=None,
    ):
        session_ids = sorted(
            {
                int(session_id)
                for session_id in session_ids
                if 0 <= int(session_id) < int(runtime.batch_size)
            }
        )
        if not session_ids:
            return

        state_x = wp.to_torch(self.simulator.wp_states[0].wp_x, requires_grad=False)
        state_v = wp.to_torch(self.simulator.wp_states[0].wp_v, requires_grad=False)
        cache = runtime.rotation_cache
        obj_nodes = int(runtime.object_nodes_per_instance)
        gs_nodes = int(runtime.gaussians_per_instance)

        for session_id in session_ids:
            obj_start = session_id * obj_nodes
            obj_end = obj_start + obj_nodes
            gs_start = session_id * gs_nodes
            gs_end = gs_start + gs_nodes

            state_x[obj_start:obj_end].copy_(reset_state.object_x[obj_start:obj_end])
            state_v[obj_start:obj_end].copy_(reset_state.object_v[obj_start:obj_end])
            runtime.prev_x[obj_start:obj_end].copy_(reset_state.object_x[obj_start:obj_end])
            if x_tensor is not None:
                x_tensor[obj_start:obj_end].copy_(reset_state.object_x[obj_start:obj_end])

            runtime.gaussians._xyz[gs_start:gs_end].copy_(
                reset_state.gaussian_xyz[gs_start:gs_end]
            )
            runtime.gaussians._rotation[gs_start:gs_end].copy_(
                reset_state.gaussian_rotation[gs_start:gs_end]
            )

            cache["R_cache"][obj_start:obj_end].copy_(
                reset_state.cache_state["R_cache"][obj_start:obj_end]
            )
            cache["F_prev"][obj_start:obj_end].copy_(
                reset_state.cache_state["F_prev"][obj_start:obj_end]
            )
            cache["rotation_computed"][obj_start:obj_end].copy_(
                reset_state.cache_state["rotation_computed"][obj_start:obj_end]
            )
            cache["Q_cache_bm"][:, session_id, :].copy_(
                reset_state.cache_state["Q_cache_bm"][:, session_id, :]
            )
            cache["motions_bm_fp32"][:, session_id, :].copy_(
                reset_state.cache_state["motions_bm_fp32"][:, session_id, :]
            )

    def _resolve_demo2_control_masks(self, demo2_control_parts, w2c, intrinsic):
        controller_points = self.controller_points[0].detach()
        num_controller_points = int(controller_points.shape[0])
        all_indices = torch.arange(
            num_controller_points,
            device=controller_points.device,
            dtype=torch.long,
        )
        if int(demo2_control_parts) == 1:
            return [all_indices]
        if num_controller_points < 2:
            return [all_indices, all_indices]

        try:
            from sklearn.cluster import KMeans
        except Exception as exc:
            raise RuntimeError(
                "Demo 2 two-hand control requires scikit-learn for controller "
                "point splitting."
            ) from exc

        points_np = controller_points.detach().cpu().numpy()
        labels = KMeans(n_clusters=2, random_state=0, n_init=10).fit_predict(points_np)
        masks = [labels == 0, labels == 1]
        if not masks[0].any() or not masks[1].any():
            return [all_indices, all_indices]

        centers = [points_np[mask].mean(axis=0) for mask in masks]
        proj_mat = np.asarray(intrinsic, dtype=np.float32) @ np.asarray(
            w2c[:3, :], dtype=np.float32
        )
        projected_x = []
        for center in centers:
            center_h = np.concatenate([center, [1.0]], axis=0)
            projected = proj_mat @ center_h
            projected = projected / projected[-1]
            projected_x.append(float(projected[0]))
        if projected_x[0] > projected_x[1]:
            masks = [masks[1], masks[0]]

        return [
            torch.from_numpy(mask)
            .to(device=controller_points.device, dtype=torch.bool)
            .nonzero(as_tuple=False)
            .squeeze(1)
            for mask in masks
        ]

    def _update_demo2_control_offsets(
        self,
        control_offsets,
        session_snapshot,
        control_step,
        control_max_offset,
        control_parts,
    ):
        occupied_set = {
            int(session_id)
            for session_id in session_snapshot.keys()
            if 0 <= int(session_id) < int(control_offsets.shape[0])
        }
        for session_id in range(int(control_offsets.shape[0])):
            if session_id not in occupied_set:
                control_offsets[session_id].zero_()

        for session_id, claim in session_snapshot.items():
            session_id = int(session_id)
            if session_id not in occupied_set:
                continue
            left = claim.get("left", (0.0, 0.0, 0.0))
            right = claim.get("right", (0.0, 0.0, 0.0))
            if int(control_parts) == 1:
                vectors = [add_vectors_clamped(left, right)]
            else:
                vectors = [left, right]
            for part_idx, control_vector in enumerate(vectors[: int(control_parts)]):
                step = control_vector_to_step(
                    control_vector[0],
                    control_vector[1],
                    control_vector[2],
                    control_step,
                )
                step_tensor = torch.tensor(
                    step,
                    device=control_offsets.device,
                    dtype=control_offsets.dtype,
                )
                control_offsets[session_id, part_idx] += step_tensor

        control_max_offset = abs(float(control_max_offset))
        if control_max_offset > 0.0:
            norms = torch.linalg.vector_norm(control_offsets, dim=2, keepdim=True)
            scales = torch.clamp(
                control_max_offset / torch.clamp(norms, min=1e-8),
                max=1.0,
            )
            control_offsets.mul_(scales)
        return control_offsets

    def _apply_demo2_control_offsets(
        self,
        target,
        runtime,
        control_offsets,
        occupied_sessions,
        control_masks,
    ):
        if not occupied_sessions:
            return target

        target = target.clone()
        controller_nodes = int(runtime.controller_nodes_per_instance)
        for session_id in occupied_sessions:
            session_id = int(session_id)
            if session_id < 0 or session_id >= int(runtime.batch_size):
                continue
            start = session_id * controller_nodes
            for part_idx, indices in enumerate(control_masks):
                if indices.numel() == 0:
                    continue
                world_offset = control_offsets[session_id, part_idx]
                target[start + indices] = target[start + indices] + world_offset.view(1, 3)
        return target

    def _resolve_demo2_hand_anchor_points(self, control_masks):
        base_controller_points = self.controller_points[0].to(device=cfg.device)
        anchors = []
        for indices in control_masks:
            if indices.numel() == 0:
                anchors.append(base_controller_points.mean(dim=0))
                continue
            target_points = base_controller_points[indices]
            anchors.append(self._find_closest_point(target_points).squeeze(0))
        return torch.stack(anchors, dim=0).to(
            device=cfg.device,
            dtype=base_controller_points.dtype,
        )

    def _load_demo2_hand_icons(self, render_backend, dtype, device):
        asset_dir = Path(__file__).resolve().parents[2] / "assets"
        icons = []
        for filename in ("Picture2.png", "Picture1.png"):
            path = asset_dir / filename
            image = render_backend.cv2.imread(
                str(path),
                render_backend.cv2.IMREAD_UNCHANGED,
            )
            if image is None:
                raise FileNotFoundError(f"Missing Demo 2 hand icon asset: {path}")
            if image.ndim != 3 or image.shape[2] != 4:
                raise ValueError(f"Expected RGBA hand icon asset: {path}")
            image = np.ascontiguousarray(image[:, :, [2, 1, 0, 3]])
            icons.append(torch.tensor(image, device=device, dtype=dtype))
        return icons

    def _project_demo2_point_to_tile(
        self,
        point,
        w2c_T,
        intrinsic_T,
        tile_width,
        tile_height,
    ):
        point = point.to(device=w2c_T.device, dtype=w2c_T.dtype)
        point_h = torch.cat((point, point.new_ones(1)), dim=0)
        pix3 = point_h @ (w2c_T[:, :3] @ intrinsic_T)
        if not bool(torch.isfinite(pix3).all().item()):
            return None
        depth = float(pix3[2].detach().item())
        if abs(depth) < 1e-6:
            return None
        xy = pix3[:2] / pix3[2]
        x = int(round(float(xy[0].detach().item())))
        y = int(round(float(xy[1].detach().item())))
        margin = max(12, min(int(tile_width), int(tile_height)) // 20)
        if x < -margin or x >= int(tile_width) + margin:
            return None
        if y < -margin or y >= int(tile_height) + margin:
            return None
        return x, y

    def _demo2_projected_icon_size(
        self,
        point,
        camera_x_axis,
        hand_size,
        w2c_T,
        intrinsic_T,
        tile_width,
        tile_height,
    ):
        pixel_1 = self._project_demo2_point_to_tile(
            point + hand_size * camera_x_axis,
            w2c_T=w2c_T,
            intrinsic_T=intrinsic_T,
            tile_width=tile_width,
            tile_height=tile_height,
        )
        pixel_2 = self._project_demo2_point_to_tile(
            point - hand_size * camera_x_axis,
            w2c_T=w2c_T,
            intrinsic_T=intrinsic_T,
            tile_width=tile_width,
            tile_height=tile_height,
        )
        if pixel_1 is None or pixel_2 is None:
            return 36
        size = int(
            round(
                math.sqrt(
                    float((pixel_1[0] - pixel_2[0]) ** 2)
                    + float((pixel_1[1] - pixel_2[1]) ** 2)
                )
                / 2.0
            )
        )
        return max(1, min(size, 100))

    def _overlay_demo2_hand_icon(self, frame, x, y, icon, icon_size):
        icon_size = int(icon_size)
        if icon_size < 1:
            return
        resized = F.interpolate(
            icon.permute(2, 0, 1).unsqueeze(0),
            size=(icon_size, icon_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).permute(1, 2, 0)
        icon_h, icon_w = int(resized.shape[0]), int(resized.shape[1])
        frame_h, frame_w = int(frame.shape[0]), int(frame.shape[1])
        left = int(round(x)) - icon_w // 2
        top = int(round(y)) - icon_h // 2
        roi_x0 = max(0, left)
        roi_y0 = max(0, top)
        roi_x1 = min(frame_w, left + icon_w)
        roi_y1 = min(frame_h, top + icon_h)
        if roi_x1 <= roi_x0 or roi_y1 <= roi_y0:
            return
        icon_x0 = roi_x0 - left
        icon_y0 = roi_y0 - top
        icon_roi = resized[
            icon_y0 : icon_y0 + (roi_y1 - roi_y0),
            icon_x0 : icon_x0 + (roi_x1 - roi_x0),
        ]
        alpha = (icon_roi[:, :, 3:4] / 255.0).clamp(0.0, 1.0)
        roi = frame[roi_y0:roi_y1, roi_x0:roi_x1, :]
        roi.mul_(1.0 - alpha).add_(icon_roi[:, :, :3] * alpha)

    def _overlay_demo2_interaction_points(
        self,
        batch_frames,
        *,
        runtime,
        control_offsets,
        occupied_sessions,
        control_masks,
        hand_anchor_points,
        hand_icons,
        camera_x_axis,
        w2c_T,
        intrinsic_T,
    ):
        if not occupied_sessions:
            return batch_frames

        tile_height, tile_width = int(batch_frames.shape[1]), int(batch_frames.shape[2])
        hand_size = 0.1
        for session_id in sorted(int(value) for value in occupied_sessions):
            if session_id < 0 or session_id >= int(runtime.batch_size):
                continue
            for part_idx, indices in enumerate(control_masks):
                if indices.numel() == 0:
                    continue
                local_point = hand_anchor_points[part_idx] + control_offsets[
                    session_id,
                    part_idx,
                ]
                pixel = self._project_demo2_point_to_tile(
                    local_point,
                    w2c_T=w2c_T,
                    intrinsic_T=intrinsic_T,
                    tile_width=tile_width,
                    tile_height=tile_height,
                )
                if pixel is None:
                    continue
                icon_size = self._demo2_projected_icon_size(
                    local_point,
                    camera_x_axis,
                    hand_size,
                    w2c_T,
                    intrinsic_T,
                    tile_width,
                    tile_height,
                )
                self._overlay_demo2_hand_icon(
                    batch_frames[session_id],
                    pixel[0],
                    pixel[1],
                    hand_icons[min(part_idx, len(hand_icons) - 1)],
                    icon_size,
                )
        return batch_frames

    def _draw_demo2_rect(self, frame, x0, y0, x1, y1, color):
        height, width = int(frame.shape[0]), int(frame.shape[1])
        x0 = max(0, min(width, int(x0)))
        x1 = max(0, min(width, int(x1)))
        y0 = max(0, min(height, int(y0)))
        y1 = max(0, min(height, int(y1)))
        if x1 <= x0 or y1 <= y0:
            return
        frame[y0:y1, x0:x1, :] = color

    def _draw_demo2_number(self, frame, value, x, y, scale, color):
        segments = {
            "0": "abcfed",
            "1": "bc",
            "2": "abged",
            "3": "abgcd",
            "4": "fgbc",
            "5": "afgcd",
            "6": "afgecd",
            "7": "abc",
            "8": "abcdefg",
            "9": "abfgcd",
        }
        segment_boxes = {
            "a": (1, 0, 4, 1),
            "b": (4, 1, 5, 4),
            "c": (4, 4, 5, 7),
            "d": (1, 7, 4, 8),
            "e": (0, 4, 1, 7),
            "f": (0, 1, 1, 4),
            "g": (1, 3, 4, 4),
        }
        cursor = int(x)
        for digit in str(int(value)):
            for segment in segments.get(digit, ""):
                sx0, sy0, sx1, sy1 = segment_boxes[segment]
                self._draw_demo2_rect(
                    frame,
                    cursor + sx0 * scale,
                    y + sy0 * scale,
                    cursor + sx1 * scale,
                    y + sy1 * scale,
                    color,
                )
            cursor += 6 * scale

    def _overlay_demo2_public_display(
        self,
        frame,
        *,
        batch_size,
        batch_grid_cols,
        occupied_sessions,
        qr_overlay=None,
    ):
        display_height, display_width = int(frame.shape[0]), int(frame.shape[1])
        cols = (
            int(batch_grid_cols)
            if batch_grid_cols is not None
            else int(math.ceil(math.sqrt(batch_size)))
        )
        rows = int(math.ceil(batch_size / float(cols)))
        label_bg = torch.tensor([0.0, 0.0, 0.0], device=frame.device, dtype=frame.dtype)
        label_fg = torch.tensor([255.0, 255.0, 255.0], device=frame.device, dtype=frame.dtype)

        for session_id in sorted(int(value) for value in occupied_sessions):
            if session_id < 0 or session_id >= int(batch_size):
                continue
            row = session_id // cols
            col = session_id % cols
            x0 = int(round(col * display_width / cols))
            x1 = int(round((col + 1) * display_width / cols))
            y0 = int(round(row * display_height / rows))
            y1 = int(round((row + 1) * display_height / rows))

            label_scale = max(2, min((x1 - x0) // 48, (y1 - y0) // 32))
            label_width = (len(str(session_id)) * 6 + 1) * label_scale
            label_height = 9 * label_scale
            lx0 = x0 + label_scale
            ly0 = y0 + label_scale
            self._draw_demo2_rect(
                frame,
                lx0,
                ly0,
                lx0 + label_width,
                ly0 + label_height,
                label_bg,
            )
            self._draw_demo2_number(
                frame,
                session_id,
                lx0 + label_scale,
                ly0 + label_scale,
                label_scale,
                label_fg,
            )

        if qr_overlay is not None:
            qr_height, qr_width = int(qr_overlay.shape[0]), int(qr_overlay.shape[1])
            margin = max(12, min(display_width, display_height) // 80)
            x0 = max(0, display_width - qr_width - margin)
            y0 = margin
            x1 = min(display_width, x0 + qr_width)
            y1 = min(display_height, y0 + qr_height)
            frame[y0:y1, x0:x1, :] = qr_overlay[: y1 - y0, : x1 - x0, :]

    def _update_demo2_motion_debug(
        self,
        motion_debug,
        *,
        runtime,
        current_target,
        batch_frames,
        frame_counter,
        replay_len,
    ):
        if motion_debug is None or int(runtime.batch_size) < 1:
            return
        if int(frame_counter) >= int(replay_len):
            return

        batch_size = int(runtime.batch_size)
        controller_nodes = int(runtime.controller_nodes_per_instance)
        last_session_id = int(motion_debug.last_session_id)
        start = last_session_id * controller_nodes
        end = start + controller_nodes

        if motion_debug.reference_target is None:
            motion_debug.reference_target = current_target.detach().clone()
            motion_debug.reference_tile = batch_frames[last_session_id].detach().clone()
            if motion_debug.enabled:
                motion_debug.target_delta_max_by_session = torch.zeros(
                    batch_size,
                    device=current_target.device,
                    dtype=current_target.dtype,
                )
            return

        target_delta = torch.linalg.vector_norm(
            current_target[start:end] - motion_debug.reference_target[start:end],
            dim=1,
        ).mean()
        pixel_delta = torch.mean(
            torch.abs(batch_frames[last_session_id] - motion_debug.reference_tile)
        )
        target_delta_value = float(target_delta.detach().item())
        pixel_delta_value = float(pixel_delta.detach().item())
        motion_debug.max_target_delta = max(
            motion_debug.max_target_delta,
            target_delta_value,
        )
        motion_debug.max_pixel_delta = max(
            motion_debug.max_pixel_delta,
            pixel_delta_value,
        )

        if motion_debug.enabled:
            reshaped_delta = torch.linalg.vector_norm(
                (
                    current_target.detach()
                    - motion_debug.reference_target
                ).view(batch_size, controller_nodes, 3),
                dim=2,
            ).mean(dim=1)
            motion_debug.target_delta_max_by_session = torch.maximum(
                motion_debug.target_delta_max_by_session,
                reshaped_delta,
            )
            if frame_counter % 10 == 0 or frame_counter + 1 >= replay_len:
                motion_debug.records.append(
                    {
                        "frame": int(frame_counter),
                        "last_session_target_delta": target_delta_value,
                        "last_session_pixel_delta": pixel_delta_value,
                    }
                )

        if frame_counter + 1 < replay_len or motion_debug.first_cycle_checked:
            return

        motion_debug.first_cycle_checked = True
        if motion_debug.max_target_delta > 1e-4 and motion_debug.max_pixel_delta < 0.5:
            motion_debug.warning = (
                "Last Demo 2 session target moved during the first replay cycle, "
                "but its rendered tile stayed nearly static."
            )
            print(
                "[Demo2][WARN] "
                f"{motion_debug.warning} session={last_session_id}, "
                f"target_delta={motion_debug.max_target_delta:.6f}, "
                f"pixel_delta={motion_debug.max_pixel_delta:.6f}"
            )
        elif motion_debug.enabled:
            print(
                "[Demo2][motion-debug] "
                f"last_session={last_session_id}, "
                f"target_delta={motion_debug.max_target_delta:.6f}, "
                f"pixel_delta={motion_debug.max_pixel_delta:.6f}"
            )

        if motion_debug.enabled and motion_debug.target_delta_max_by_session is not None:
            deltas = motion_debug.target_delta_max_by_session.detach().cpu()
            last_value = float(deltas[last_session_id].item())
            print(
                "[Demo2][motion-debug] "
                f"per-session target delta: min={float(deltas.min().item()):.6f}, "
                f"max={float(deltas.max().item()):.6f}, "
                f"last={last_value:.6f}"
            )

    def _write_demo2_motion_debug(self, motion_debug):
        if (
            motion_debug is None
            or not motion_debug.path
            or not motion_debug.enabled
        ):
            return
        target_delta_max_by_session = None
        if motion_debug.target_delta_max_by_session is not None:
            target_delta_max_by_session = (
                motion_debug.target_delta_max_by_session.detach().cpu().tolist()
            )
        payload = {
            "last_session_id": int(motion_debug.last_session_id),
            "max_target_delta": float(motion_debug.max_target_delta),
            "max_pixel_delta": float(motion_debug.max_pixel_delta),
            "warning": motion_debug.warning,
            "records": motion_debug.records,
            "target_delta_max_by_session": target_delta_max_by_session,
        }
        os.makedirs(os.path.dirname(os.path.abspath(motion_debug.path)), exist_ok=True)
        with open(motion_debug.path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"[Demo2][motion-debug] Wrote {motion_debug.path}")

    def _write_headless_summary(self, output_dir, batch_size, component_times):
        frames_used_for_stats = len(component_times["total"])
        if frames_used_for_stats == 0:
            return {}

        total_frame_times = component_times["total"]
        total_time_seconds = sum(total_frame_times)
        average_batch_fps = frames_used_for_stats / total_time_seconds
        average_frame_time = float(np.mean(total_frame_times))
        average_throughput = batch_size * average_batch_fps

        print(f"\n=== Final Summary (averaged over {frames_used_for_stats} frames) ===")
        log_lines = [
            f"=== Final Summary (averaged over {frames_used_for_stats} frames) ===",
            f"Batch Size: {batch_size}",
            f"Average Batch FPS: {average_batch_fps:.2f}",
            f"Average Throughput (instances/s): {average_throughput:.2f}",
            f"Average Sim+LBS Total: {average_frame_time * 1000:.2f} ms",
        ]

        print(f"Batch Size: {batch_size}")
        print(f"Average Batch FPS: {average_batch_fps:.2f}")
        print(f"Average Throughput (instances/s): {average_throughput:.2f}")
        print(f"Average Sim+LBS Total: {average_frame_time * 1000:.2f} ms")

        for component_name in ["simulator", "full_motion_interpolation"]:
            component_times_list = component_times.get(component_name, [])
            if component_times_list:
                average_component_time = float(np.mean(component_times_list))
                time_share_percentage = (
                    average_component_time / average_frame_time
                ) * 100.0
                readable_name = perf_component_label(component_name)
                print(
                    f"{readable_name}: {average_component_time * 1000:.2f} ms ({time_share_percentage:.1f}%)"
                )
                log_lines.append(
                    f"{readable_name}: {average_component_time * 1000:.2f} ms ({time_share_percentage:.1f}%)"
                )

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            log_file_path = os.path.join(output_dir, "performance_summary.txt")
            with open(log_file_path, "w", encoding="utf-8") as log_file:
                log_file.write("\n".join(log_lines))

        return {
            "frames_used_for_stats": frames_used_for_stats,
            "batch_size": batch_size,
            "average_batch_fps": average_batch_fps,
            "average_throughput": average_throughput,
            "average_sim_lbs_total_ms": average_frame_time * 1000.0,
            "average_simulator_ms": float(np.mean(component_times["simulator"])) * 1000.0
            if component_times["simulator"]
            else None,
            "average_full_motion_interpolation_ms": float(
                np.mean(component_times["full_motion_interpolation"])
            )
            * 1000.0
            if component_times["full_motion_interpolation"]
            else None,
        }

    def check_batched_replay_lbs(
        self,
        model_path,
        gs_path,
        controller_points_group,
        replay_start=0,
        replay_end=None,
        gaussian_render_mode="shared_template",
        sim_force_mode=SIM_FORCE_MODE_GATHER,
    ):
        batch_size = int(controller_points_group.shape[0])
        runtime = self._build_runtime_core(
            model_path=model_path,
            gs_path=gs_path,
            n_dup=batch_size - 1,
            controller_points_group_override=controller_points_group,
            gaussian_render_mode=gaussian_render_mode,
            force_shared_batched_gaussians=True,
            sim_force_mode=sim_force_mode,
        )
        replay_end = self._demo2_replay_end(runtime, replay_start, replay_end)

        gaussians = runtime.gaussians
        frame_idx = int(replay_start)
        prev_target = self.batch_controller_points[frame_idx]
        current_target = prev_target
        checked_frames = 0

        try:
            while frame_idx < replay_end:
                if checked_frames > 0:
                    current_target = self.batch_controller_points[frame_idx]

                self.simulator.set_controller_interactive(prev_target, current_target)
                if self.simulator.object_collision_flag:
                    self.simulator.update_collision_graph()
                wp.capture_launch(self.simulator.forward_graph)
                wp.synchronize()

                x = wp.to_torch(self.simulator.wp_states[-1].wp_x, requires_grad=False)
                self.simulator.set_init_state(
                    self.simulator.wp_states[-1].wp_x,
                    self.simulator.wp_states[-1].wp_v,
                )

                if frame_idx + 1 < replay_end:
                    try:
                        current_pos, current_rot = lbs_with_rotation_reuse(
                            current_mass_nodes=x,
                            cache=runtime.rotation_cache,
                        )
                    except torch._C._LinAlgError as exc:
                        batch_element = self._parse_linalg_batch_element(exc)
                        hinted_instance = None
                        if batch_element is not None:
                            hinted_instance = int(batch_element) // int(
                                runtime.object_nodes_per_instance
                            )
                            if hinted_instance < 0 or hinted_instance >= batch_size:
                                hinted_instance = None
                        raise BatchedReplayCheckError(
                            "Batched replay LBS failed with torch.linalg.eigh "
                            f"at replay frame {frame_idx}.",
                            frame_idx=frame_idx,
                            batch_element=batch_element,
                            hinted_instance=hinted_instance,
                            original_error=exc,
                        ) from exc
                    gaussians._xyz = current_pos
                    gaussians._rotation = current_rot

                prev_target = current_target
                frame_idx += 1
                checked_frames += 1
        finally:
            torch.cuda.empty_cache()

        return {
            "batch_size": batch_size,
            "frames_checked": checked_frames,
            "replay_start": int(replay_start),
            "replay_end": int(replay_end),
        }

    def _write_headless_ncu_profile_metrics(self, output_dir, metrics):
        if output_dir is None:
            return
        os.makedirs(output_dir, exist_ok=True)
        metrics_path = os.path.join(output_dir, "ncu_profile_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as metrics_file:
            json.dump(metrics, metrics_file, indent=2)

    def _write_full_runtime_summary(
        self,
        output_dir,
        batch_size,
        render_mode,
        gaussian_render_mode,
        instance_id,
        component_times,
        batch_image_resolution="native",
        render_width=None,
        render_height=None,
        sim_force_mode=SIM_FORCE_MODE_GATHER,
    ):
        frames_used_for_stats = len(component_times["total"])
        total_frame_times = component_times["total"]
        total_time_seconds = sum(total_frame_times) if total_frame_times else 0.0
        average_fps = (
            frames_used_for_stats / total_time_seconds
            if frames_used_for_stats > 0 and total_time_seconds > 0.0
            else 0.0
        )
        average_frame_time = (
            float(np.mean(total_frame_times)) if total_frame_times else 0.0
        )
        average_throughput = batch_size * average_fps

        print(
            f"\n=== Final Summary (averaged over {frames_used_for_stats} frames) ==="
        )
        print(f"Batch Size: {batch_size}")
        print(f"Render Mode: {render_mode}")
        print(f"Gaussian Render Mode: {gaussian_render_mode}")
        print(f"Sim Force Mode: {sim_force_mode}")
        if instance_id is not None:
            print(f"Instance ID: {instance_id}")
        if render_width is not None and render_height is not None:
            print(f"Batch Image Resolution: {batch_image_resolution}")
            print(f"Render Size: {int(render_width)}x{int(render_height)}")
        print(f"Average FPS: {average_fps:.2f}")
        print(f"Average Throughput (instances/s): {average_throughput:.2f}")
        print(f"Average Total Frame Time: {average_frame_time * 1000:.2f} ms")

        log_lines = [
            f"=== Final Summary (averaged over {frames_used_for_stats} frames) ===",
            f"Batch Size: {batch_size}",
            f"Render Mode: {render_mode}",
            f"Gaussian Render Mode: {gaussian_render_mode}",
            f"Sim Force Mode: {sim_force_mode}",
        ]
        if instance_id is not None:
            log_lines.append(f"Instance ID: {instance_id}")
        if render_width is not None and render_height is not None:
            log_lines.append(f"Batch Image Resolution: {batch_image_resolution}")
            log_lines.append(f"Render Size: {int(render_width)}x{int(render_height)}")
        log_lines.extend(
            [
                f"Average FPS: {average_fps:.2f}",
                f"Average Throughput (instances/s): {average_throughput:.2f}",
                f"Average Total Frame Time: {average_frame_time * 1000:.2f} ms",
            ]
        )

        metrics = {
            "frames_used_for_stats": int(frames_used_for_stats),
            "batch_size": int(batch_size),
            "render_mode": render_mode,
            "gaussian_render_mode": gaussian_render_mode,
            "sim_force_mode": sim_force_mode,
            "instance_id": int(instance_id) if instance_id is not None else None,
            "batch_image_resolution": batch_image_resolution,
            "render_width": int(render_width) if render_width is not None else None,
            "render_height": int(render_height) if render_height is not None else None,
            "average_fps": float(average_fps),
            "average_throughput": float(average_throughput),
            "average_total_frame_time_ms": float(average_frame_time * 1000.0),
        }

        for component_name in [
            "simulator",
            "full_motion_interpolation",
            "rendering",
            "frame_compositing",
        ]:
            component_times_list = component_times.get(component_name, [])
            average_component_time = (
                float(np.mean(component_times_list)) if component_times_list else None
            )
            time_share_percentage = None
            if average_component_time is not None and average_frame_time > 0.0:
                time_share_percentage = (
                    average_component_time / average_frame_time
                ) * 100.0
                readable_name = perf_component_label(component_name)
                print(
                    f"{readable_name}: {average_component_time * 1000:.2f} ms ({time_share_percentage:.1f}%)"
                )
                log_lines.append(
                    f"{readable_name}: {average_component_time * 1000:.2f} ms ({time_share_percentage:.1f}%)"
                )
            metrics[f"average_{component_name}_ms"] = (
                float(average_component_time * 1000.0)
                if average_component_time is not None
                else None
            )
            metrics[f"average_{component_name}_share_pct"] = (
                float(time_share_percentage)
                if time_share_percentage is not None
                else None
            )

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            log_file_path = os.path.join(output_dir, "performance_summary.txt")
            with open(log_file_path, "w", encoding="utf-8") as log_file:
                log_file.write("\n".join(log_lines))

            metrics_path = os.path.join(output_dir, "performance_summary.json")
            with open(metrics_path, "w", encoding="utf-8") as metrics_file:
                json.dump(metrics, metrics_file, indent=2)

        return metrics

    def run_headless_sim_lbs(
        self,
        model_path,
        gs_path,
        output_dir=None,
        n_dup=0,
        ncu_profile_loop=False,
        ncu_profile_frame_stride=None,
        ncu_profile_max_frames=3,
        ncu_profile_nvtx_name="sim_lbs_profile_frame",
    ):
        if ncu_profile_loop:
            if ncu_profile_frame_stride is not None:
                ncu_profile_frame_stride = int(ncu_profile_frame_stride)
                if ncu_profile_frame_stride < 1:
                    raise ValueError(
                        "ncu_profile_frame_stride must be a positive integer. "
                        f"Received: {ncu_profile_frame_stride}"
                    )
            ncu_profile_max_frames = int(ncu_profile_max_frames)
            if ncu_profile_max_frames < 1:
                raise ValueError(
                    "ncu_profile_max_frames must be a positive integer. "
                    f"Received: {ncu_profile_max_frames}"
                )

        runtime = self._build_runtime_core(model_path, gs_path, n_dup=n_dup)
        measured_profile_frames = list(range(2, runtime.frame_len))
        if ncu_profile_loop and ncu_profile_frame_stride is not None:
            ncu_profile_frame_selection = "stride"
            ncu_profile_frame_indices_to_capture = measured_profile_frames[
                ::ncu_profile_frame_stride
            ]
        elif ncu_profile_loop:
            ncu_profile_frame_selection = "evenly_spaced"
            if len(measured_profile_frames) <= ncu_profile_max_frames:
                ncu_profile_frame_indices_to_capture = measured_profile_frames
            elif ncu_profile_max_frames == 1:
                middle_idx = round((len(measured_profile_frames) - 1) / 2)
                ncu_profile_frame_indices_to_capture = [
                    measured_profile_frames[middle_idx]
                ]
            else:
                last_idx = len(measured_profile_frames) - 1
                ncu_profile_frame_indices_to_capture = sorted(
                    {
                        measured_profile_frames[
                            round(i * last_idx / (ncu_profile_max_frames - 1))
                        ]
                        for i in range(ncu_profile_max_frames)
                    }
                )
        else:
            ncu_profile_frame_selection = None
            ncu_profile_frame_indices_to_capture = []
        ncu_profile_frame_indices_to_capture = set(
            int(frame_idx) for frame_idx in ncu_profile_frame_indices_to_capture
        )

        sim_timer = Timer("Simulator")
        interp_timer = Timer("Linear Blend Skinning")
        total_timer = Timer("Sim+LBS Total")
        component_times = {
            "simulator": [],
            "full_motion_interpolation": [],
            "total": [],
        }

        summary = {}
        profile_cuda = bool(ncu_profile_loop and torch.cuda.is_available())
        cuda_profiler = torch.cuda.cudart() if profile_cuda else None
        loop_memory_reset = False
        loop_peak_allocated_gb = None
        loop_peak_reserved_gb = None
        ncu_profiled_frame_indices = []
        try:
            while True:
                if profile_cuda and runtime.frame_count == 2 and not loop_memory_reset:
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    loop_memory_reset = True

                profile_this_frame = (
                    profile_cuda
                    and runtime.frame_count in ncu_profile_frame_indices_to_capture
                )
                profiler_started = False
                nvtx_range_id = None

                try:
                    if profile_this_frame:
                        torch.cuda.synchronize()
                        nvtx_range_id = torch.cuda.nvtx.range_start(
                            ncu_profile_nvtx_name
                        )
                        cuda_profiler.cudaProfilerStart()
                        profiler_started = True
                        ncu_profiled_frame_indices.append(int(runtime.frame_count))

                    total_timer.start()

                    sim_timer.start()
                    self.simulator.set_controller_interactive(
                        runtime.prev_target, runtime.current_target
                    )

                    if self.simulator.object_collision_flag:
                        self.simulator.update_collision_graph()
                    wp.capture_launch(self.simulator.forward_graph)
                    wp.synchronize()

                    x = wp.to_torch(
                        self.simulator.wp_states[-1].wp_x, requires_grad=False
                    )
                    self.simulator.set_init_state(
                        self.simulator.wp_states[-1].wp_x,
                        self.simulator.wp_states[-1].wp_v,
                    )
                    sim_time = sim_timer.stop()

                    with torch.no_grad():
                        interp_timer.start()
                        current_pos, current_rot = lbs_with_rotation_reuse(
                            current_mass_nodes=x,
                            cache=runtime.rotation_cache,
                        )
                        interp_time = interp_timer.stop()
                        runtime.gaussians._xyz = current_pos
                        runtime.gaussians._rotation = current_rot

                    total_time = total_timer.stop()
                finally:
                    if profiler_started:
                        try:
                            torch.cuda.synchronize()
                            cuda_profiler.cudaProfilerStop()
                        except Exception as exc:
                            print(f"[WARN] Failed to stop CUDA profiler: {exc}")
                    if nvtx_range_id is not None:
                        try:
                            torch.cuda.nvtx.range_end(nvtx_range_id)
                        except Exception as exc:
                            print(f"[WARN] Failed to end NVTX range: {exc}")

                if runtime.frame_count > 1:
                    component_times["simulator"].append(sim_time)
                    component_times["full_motion_interpolation"].append(interp_time)
                    component_times["total"].append(total_time)

                runtime.prev_x = x.clone()
                runtime.frame_count += 1
                runtime.prev_target = runtime.current_target

                if runtime.frame_count < runtime.frame_len:
                    runtime.current_target = self.batch_controller_points[
                        runtime.frame_count
                    ]
                else:
                    print("Reached end of recorded control sequence")
                    break
        finally:
            summary = self._write_headless_summary(
                output_dir=output_dir,
                batch_size=runtime.batch_size,
                component_times=component_times,
            )
            if ncu_profile_loop:
                if profile_cuda and loop_memory_reset:
                    torch.cuda.synchronize()
                    loop_peak_allocated_gb = torch.cuda.max_memory_allocated() / (
                        1024**3
                    )
                    loop_peak_reserved_gb = torch.cuda.max_memory_reserved() / (
                        1024**3
                    )
                self._write_headless_ncu_profile_metrics(
                    output_dir=output_dir,
                    metrics={
                        "ncu_profile_loop": True,
                        "ncu_profile_frame_stride": int(ncu_profile_frame_stride)
                        if ncu_profile_frame_stride is not None
                        else None,
                        "ncu_profile_max_frames": int(ncu_profile_max_frames),
                        "ncu_profile_frame_selection": ncu_profile_frame_selection,
                        "ncu_profile_nvtx_name": ncu_profile_nvtx_name,
                        "ncu_profiled_frame_indices": ncu_profiled_frame_indices,
                        "ncu_num_profiled_frames": len(ncu_profiled_frame_indices),
                        "loop_peak_allocated_gb": loop_peak_allocated_gb,
                        "loop_peak_reserved_gb": loop_peak_reserved_gb,
                    },
                )
        return summary

    def run_batched_demo2_runtime(
        self,
        model_path,
        gs_path,
        window,
        cuda_ctx,
        controller_points_group,
        batch_size=100,
        gaussian_render_mode="shared_template",
        batch_image_resolution="640x480",
        batch_grid_cols=10,
        replay_start=0,
        replay_end=None,
        sim_force_mode=SIM_FORCE_MODE_GATHER,
        session_snapshot_fn=None,
        input_snapshot_fn=None,
        occupied_sessions_fn=None,
        publish_frame_fn=None,
        qr_overlay_rgb=None,
        phone_stream_size=(640, 480),
        phone_stream_fps=10.0,
        phone_control_step=0.005,
        phone_control_max_offset=0.0,
        demo2_control_parts=1,
        demo2_debug_motion=False,
        demo2_debug_motion_path=None,
        max_frames=None,
    ):
        gaussian_render_mode = normalize_gaussian_render_mode(gaussian_render_mode)
        if sim_force_mode not in SIM_FORCE_MODES:
            raise ValueError(
                f"sim_force_mode must be one of {SIM_FORCE_MODES}. "
                f"Received: {sim_force_mode}"
            )
        batch_size = int(batch_size)
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if batch_image_resolution not in ("native", "640x480"):
            raise ValueError(
                "batch_image_resolution must be 'native' or '640x480'. "
                f"Received: {batch_image_resolution}"
            )
        if batch_grid_cols is not None and int(batch_grid_cols) < 1:
            raise ValueError("batch_grid_cols must be a positive integer.")
        if phone_stream_fps <= 0:
            raise ValueError("phone_stream_fps must be positive.")
        if phone_control_step < 0:
            raise ValueError("phone_control_step must be non-negative.")
        if phone_control_max_offset < 0:
            raise ValueError("phone_control_max_offset must be non-negative.")
        demo2_control_parts = int(demo2_control_parts)
        if demo2_control_parts not in (1, 2):
            raise ValueError("demo2_control_parts must be 1 or 2.")

        runtime = self._build_runtime_core(
            model_path=model_path,
            gs_path=gs_path,
            n_dup=batch_size - 1,
            controller_points_group_override=controller_points_group[:batch_size],
            gaussian_render_mode=gaussian_render_mode,
            force_shared_batched_gaussians=True,
            sim_force_mode=sim_force_mode,
        )
        replay_end = self._demo2_replay_end(runtime, replay_start, replay_end)
        replay_start = int(replay_start)
        replay_len = int(replay_end - replay_start)
        print(
            "[Demo2] Runtime ready: "
            f"batch_size={runtime.batch_size}, replay={replay_start}:{replay_end}, "
            f"gaussian_render_mode={gaussian_render_mode}, sim_force_mode={sim_force_mode}, "
            f"control_parts={demo2_control_parts}"
        )
        reset_state = self._snapshot_demo2_reset_state(runtime)

        render_backend = self._load_render_backend()
        render_backend.glfw.make_context_current(window)

        native_width, native_height = [int(value) for value in cfg.WH]
        width, height = self._resolve_batch_image_render_size(
            batch_image_resolution=batch_image_resolution,
            native_width=native_width,
            native_height=native_height,
        )
        display_width, display_height = render_backend.glfw.get_framebuffer_size(window)
        display_width = int(display_width) if display_width > 0 else width
        display_height = int(display_height) if display_height > 0 else height

        background = torch.tensor([1, 1, 1], dtype=torch.float32, device=cfg.device)
        intrinsic = self._scale_intrinsic_for_render_size(
            cfg.intrinsics[0],
            native_width=native_width,
            native_height=native_height,
            render_width=width,
            render_height=height,
        )
        w2c = cfg.w2cs[0]
        view, K_cuda = self._create_gs_view(w2c, intrinsic, height, width)
        w2c_cuda = torch.tensor(w2c, dtype=torch.float32, device=cfg.device)
        w2c_T = w2c_cuda.T.contiguous()
        intrinsic_T = K_cuda.T.contiguous()
        control_masks = self._resolve_demo2_control_masks(
            demo2_control_parts,
            w2c=w2c,
            intrinsic=intrinsic,
        )
        hand_anchor_points = self._resolve_demo2_hand_anchor_points(control_masks)
        c2w = np.linalg.inv(np.asarray(w2c, dtype=np.float32))
        camera_x_axis = torch.tensor(
            c2w[:3, 0],
            dtype=hand_anchor_points.dtype,
            device=cfg.device,
        )

        overlay = render_backend.cv2.imread(cfg.bg_img_path)
        overlay = render_backend.cv2.cvtColor(
            overlay, render_backend.cv2.COLOR_BGR2RGB
        )
        if overlay.shape[1] != width or overlay.shape[0] != height:
            overlay = render_backend.cv2.resize(
                overlay,
                (width, height),
                interpolation=render_backend.cv2.INTER_LINEAR,
            )
        overlay = torch.tensor(overlay, dtype=torch.float32, device=cfg.device)
        hand_icons = self._load_demo2_hand_icons(
            render_backend,
            dtype=overlay.dtype,
            device=cfg.device,
        )

        render_gaussians = build_batch_images_render_view(
            runtime.gaussians,
            gaussian_render_mode=gaussian_render_mode,
        )

        bytes_per_pixel = 4
        pbo_size = display_width * display_height * bytes_per_pixel
        row_pitch = display_width * bytes_per_pixel

        tex = None
        pbo = None
        reg = None
        prog = None
        vao = None
        try:
            tex = render_backend.gl.glGenTextures(1)
            render_backend.gl.glBindTexture(render_backend.gl.GL_TEXTURE_2D, tex)
            render_backend.gl.glTexParameteri(
                render_backend.gl.GL_TEXTURE_2D,
                render_backend.gl.GL_TEXTURE_MIN_FILTER,
                render_backend.gl.GL_NEAREST,
            )
            render_backend.gl.glTexParameteri(
                render_backend.gl.GL_TEXTURE_2D,
                render_backend.gl.GL_TEXTURE_MAG_FILTER,
                render_backend.gl.GL_NEAREST,
            )
            render_backend.gl.glTexImage2D(
                render_backend.gl.GL_TEXTURE_2D,
                0,
                render_backend.gl.GL_RGBA8,
                display_width,
                display_height,
                0,
                render_backend.gl.GL_RGBA,
                render_backend.gl.GL_UNSIGNED_BYTE,
                None,
            )
            render_backend.gl.glBindTexture(render_backend.gl.GL_TEXTURE_2D, 0)

            pbo = render_backend.gl.glGenBuffers(1)
            render_backend.gl.glBindBuffer(
                render_backend.gl.GL_PIXEL_UNPACK_BUFFER, pbo
            )
            render_backend.gl.glBufferData(
                render_backend.gl.GL_PIXEL_UNPACK_BUFFER,
                pbo_size,
                None,
                render_backend.gl.GL_STREAM_DRAW,
            )
            render_backend.gl.glBindBuffer(
                render_backend.gl.GL_PIXEL_UNPACK_BUFFER, 0
            )
            reg = render_backend.RegisteredBuffer(
                int(pbo), render_backend.graphics_map_flags.WRITE_DISCARD
            )

            vertex_shader_source = """
            #version 330 core
            out vec2 uv; const vec2 V[4]=vec2[4](vec2(-1,-1),vec2(1,-1),vec2(-1,1),vec2(1,1));
            const vec2 T[4]=vec2[4](vec2(0,0),vec2(1,0),vec2(0,1),vec2(1,1));
            void main(){ gl_Position=vec4(V[gl_VertexID],0,1); uv=T[gl_VertexID]; }
            """
            fragment_shader_source = """
            #version 330 core
            in vec2 uv; out vec4 frag; uniform sampler2D uTex;
            void main(){ frag = texture(uTex, vec2(uv.x,1.0 - uv.y)); }
            """

            def _compile(kind, src):
                shader_id = render_backend.gl.glCreateShader(kind)
                render_backend.gl.glShaderSource(shader_id, src)
                render_backend.gl.glCompileShader(shader_id)
                if not render_backend.gl.glGetShaderiv(
                    shader_id, render_backend.gl.GL_COMPILE_STATUS
                ):
                    raise RuntimeError(
                        render_backend.gl.glGetShaderInfoLog(shader_id).decode()
                    )
                return shader_id

            prog = render_backend.gl.glCreateProgram()
            render_backend.gl.glAttachShader(
                prog,
                _compile(render_backend.gl.GL_VERTEX_SHADER, vertex_shader_source),
            )
            render_backend.gl.glAttachShader(
                prog,
                _compile(render_backend.gl.GL_FRAGMENT_SHADER, fragment_shader_source),
            )
            render_backend.gl.glLinkProgram(prog)
            if not render_backend.gl.glGetProgramiv(
                prog, render_backend.gl.GL_LINK_STATUS
            ):
                raise RuntimeError(
                    render_backend.gl.glGetProgramInfoLog(prog).decode()
                )
            render_backend.gl.glUseProgram(prog)
            render_backend.gl.glUniform1i(
                render_backend.gl.glGetUniformLocation(prog, "uTex"), 0
            )
            render_backend.gl.glUseProgram(0)
            vao = render_backend.gl.glGenVertexArrays(1)
            render_backend.gl.glBindVertexArray(vao)

            pbo_stream = render_backend.cuda_driver.Stream()
            cpy2d = render_backend.cuda_driver.Memcpy2D()
            cpy2d.src_pitch = row_pitch
            cpy2d.dst_pitch = row_pitch
            cpy2d.width_in_bytes = row_pitch
            cpy2d.height = display_height

            frame_rgba = torch.empty(
                (display_height, display_width, 4),
                dtype=torch.uint8,
                device=cfg.device,
            )
            frame = torch.empty(
                (display_height, display_width, 3),
                dtype=overlay.dtype,
                device=cfg.device,
            )
            qr_overlay = None
            if qr_overlay_rgb is not None:
                qr_overlay = torch.tensor(
                    qr_overlay_rgb,
                    device=cfg.device,
                    dtype=frame.dtype,
                )

            input_snapshot_fn = input_snapshot_fn or (lambda: {})
            occupied_sessions_fn = occupied_sessions_fn or (lambda: [])

            def legacy_session_snapshot():
                occupied = set(int(value) for value in occupied_sessions_fn())
                inputs = input_snapshot_fn()
                return {
                    session_id: {
                        "claim_id": 1,
                        "left": inputs.get(session_id, (0.0, 0.0, 0.0)),
                        "right": (0.0, 0.0, 0.0),
                    }
                    for session_id in occupied
                }

            session_snapshot_fn = session_snapshot_fn or legacy_session_snapshot
            stream_width, stream_height = [int(value) for value in phone_stream_size]
            stream_interval = 1.0 / float(phone_stream_fps)
            next_stream_publish = 0.0
            control_offsets = torch.zeros(
                (int(runtime.batch_size), demo2_control_parts, 3),
                device=cfg.device,
                dtype=self.batch_controller_points.dtype,
            )
            motion_debug = SimpleNamespace(
                enabled=bool(demo2_debug_motion),
                path=demo2_debug_motion_path,
                last_session_id=int(runtime.batch_size) - 1,
                reference_target=None,
                reference_tile=None,
                target_delta_max_by_session=None,
                max_target_delta=0.0,
                max_pixel_delta=0.0,
                warning=None,
                first_cycle_checked=False,
                records=[],
            )
            replay_cursors = [int(replay_start) for _ in range(int(runtime.batch_size))]
            active_claim_ids = [None for _ in range(int(runtime.batch_size))]

            frame_counter = 0
            prev_target = self.batch_controller_points[replay_start].clone()
            while not render_backend.glfw.window_should_close(window):
                if max_frames is not None and frame_counter >= int(max_frames):
                    break

                raw_snapshot = session_snapshot_fn()
                session_snapshot = {
                    int(session_id): claim
                    for session_id, claim in raw_snapshot.items()
                    if 0 <= int(session_id) < int(runtime.batch_size)
                }
                occupied_sessions = sorted(session_snapshot.keys())
                current_claim_ids = {
                    session_id: int(claim.get("claim_id", 1))
                    for session_id, claim in session_snapshot.items()
                }
                previously_claimed = {
                    session_id
                    for session_id, claim_id in enumerate(active_claim_ids)
                    if claim_id is not None
                }
                newly_claimed = [
                    session_id
                    for session_id, claim_id in current_claim_ids.items()
                    if active_claim_ids[session_id] != claim_id
                ]
                released_sessions = sorted(
                    previously_claimed - set(current_claim_ids.keys())
                )
                reset_sessions = sorted(set(newly_claimed + released_sessions))

                if reset_sessions:
                    self._restore_demo2_instance_reset_state(
                        runtime,
                        reset_state,
                        reset_sessions,
                    )
                    control_offsets[reset_sessions].zero_()
                    for session_id in reset_sessions:
                        replay_cursors[session_id] = replay_start

                for session_id in released_sessions:
                    active_claim_ids[session_id] = None
                for session_id in newly_claimed:
                    active_claim_ids[session_id] = current_claim_ids[session_id]

                self._update_demo2_control_offsets(
                    control_offsets,
                    session_snapshot,
                    float(phone_control_step),
                    float(phone_control_max_offset),
                    demo2_control_parts,
                )

                current_target = self.batch_controller_points[replay_start].clone()
                controller_nodes = int(runtime.controller_nodes_per_instance)
                for session_id in range(int(runtime.batch_size)):
                    start = session_id * controller_nodes
                    end = start + controller_nodes
                    if active_claim_ids[session_id] is None:
                        frame_idx = replay_cursors[session_id]
                    else:
                        frame_idx = replay_start
                    current_target[start:end].copy_(
                        self.batch_controller_points[frame_idx, start:end]
                    )
                current_target = self._apply_demo2_control_offsets(
                    current_target,
                    runtime,
                    control_offsets,
                    occupied_sessions,
                    control_masks,
                )
                if reset_sessions:
                    for session_id in reset_sessions:
                        start = session_id * controller_nodes
                        end = start + controller_nodes
                        prev_target[start:end].copy_(current_target[start:end])

                self.simulator.set_controller_interactive(prev_target, current_target)
                if self.simulator.object_collision_flag:
                    self.simulator.update_collision_graph()
                wp.capture_launch(self.simulator.forward_graph)
                wp.synchronize()

                x = wp.to_torch(self.simulator.wp_states[-1].wp_x, requires_grad=False)
                self.simulator.set_init_state(
                    self.simulator.wp_states[-1].wp_x,
                    self.simulator.wp_states[-1].wp_v,
                )

                results = render_backend.render_gaussian(
                    view,
                    render_gaussians,
                    None,
                    background,
                )
                batch_frames, _ = self._composite_batch_images_without_shadows(
                    results["render"],
                    overlay,
                )
                if int(batch_frames.shape[0]) != int(runtime.batch_size):
                    raise RuntimeError(
                        "Demo 2 batch image renderer returned an unexpected batch "
                        f"count: {int(batch_frames.shape[0])} vs {int(runtime.batch_size)}"
                    )
                self._update_demo2_motion_debug(
                    motion_debug,
                    runtime=runtime,
                    current_target=current_target,
                    batch_frames=batch_frames,
                    frame_counter=frame_counter,
                    replay_len=replay_len,
                )
                self._overlay_demo2_interaction_points(
                    batch_frames,
                    runtime=runtime,
                    control_offsets=control_offsets,
                    occupied_sessions=occupied_sessions,
                    control_masks=control_masks,
                    hand_anchor_points=hand_anchor_points,
                    hand_icons=hand_icons,
                    camera_x_axis=camera_x_axis,
                    w2c_T=w2c_T,
                    intrinsic_T=intrinsic_T,
                )

                batch_grid = self._make_batch_image_grid(
                    batch_frames,
                    batch_grid_cols=batch_grid_cols,
                )
                display_grid = batch_grid
                if (
                    display_grid.shape[0] != display_height
                    or display_grid.shape[1] != display_width
                ):
                    display_grid = F.interpolate(
                        display_grid.permute(2, 0, 1).unsqueeze(0),
                        size=(display_height, display_width),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0).permute(1, 2, 0)
                frame.copy_(display_grid)

                self._overlay_demo2_public_display(
                    frame,
                    batch_size=runtime.batch_size,
                    batch_grid_cols=batch_grid_cols,
                    occupied_sessions=occupied_sessions,
                    qr_overlay=qr_overlay,
                )

                now = time.time()
                if (
                    publish_frame_fn is not None
                    and occupied_sessions
                    and now >= next_stream_publish
                ):
                    occupied_ids = torch.tensor(
                        occupied_sessions,
                        device=batch_frames.device,
                        dtype=torch.long,
                    )
                    stream_tiles = F.interpolate(
                        batch_frames[occupied_ids].permute(0, 3, 1, 2),
                        size=(stream_height, stream_width),
                        mode="bilinear",
                        align_corners=False,
                    ).permute(0, 2, 3, 1)
                    stream_rgbs = (
                        stream_tiles.clamp(0, 255)
                        .to(torch.uint8)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    for idx, session_id in enumerate(occupied_sessions):
                        rgb = stream_rgbs[idx]
                        publish_frame_fn(session_id, rgb)
                    next_stream_publish = now + stream_interval

                frame_u8 = frame.clamp(0, 255).to(torch.uint8)
                frame_rgba[:, :, :3] = frame_u8
                frame_rgba[:, :, 3] = 255
                torch.cuda.current_stream().synchronize()

                mapping = reg.map()
                try:
                    ptr, _ = mapping.device_ptr_and_size()
                    cpy2d.set_src_device(frame_rgba.data_ptr())
                    cpy2d.set_dst_device(ptr)
                    cpy2d(pbo_stream)
                    pbo_stream.synchronize()
                finally:
                    mapping.unmap()

                render_backend.gl.glBindTexture(render_backend.gl.GL_TEXTURE_2D, tex)
                render_backend.gl.glBindBuffer(
                    render_backend.gl.GL_PIXEL_UNPACK_BUFFER, pbo
                )
                render_backend.gl.glTexSubImage2D(
                    render_backend.gl.GL_TEXTURE_2D,
                    0,
                    0,
                    0,
                    display_width,
                    display_height,
                    render_backend.gl.GL_RGBA,
                    render_backend.gl.GL_UNSIGNED_BYTE,
                    None,
                )
                render_backend.gl.glBindBuffer(
                    render_backend.gl.GL_PIXEL_UNPACK_BUFFER, 0
                )

                render_backend.gl.glViewport(0, 0, display_width, display_height)
                render_backend.gl.glDisable(render_backend.gl.GL_DEPTH_TEST)
                render_backend.gl.glClear(render_backend.gl.GL_COLOR_BUFFER_BIT)
                render_backend.gl.glUseProgram(prog)
                render_backend.gl.glActiveTexture(render_backend.gl.GL_TEXTURE0)
                render_backend.gl.glBindTexture(render_backend.gl.GL_TEXTURE_2D, tex)
                render_backend.gl.glDrawArrays(
                    render_backend.gl.GL_TRIANGLE_STRIP, 0, 4
                )
                render_backend.gl.glBindTexture(render_backend.gl.GL_TEXTURE_2D, 0)
                render_backend.gl.glUseProgram(0)

                render_backend.glfw.swap_buffers(window)
                render_backend.glfw.poll_events()

                end_of_replay_sessions = []
                for session_id in range(int(runtime.batch_size)):
                    if active_claim_ids[session_id] is not None:
                        continue
                    if replay_cursors[session_id] + 1 < replay_end:
                        replay_cursors[session_id] += 1
                    else:
                        replay_cursors[session_id] = replay_start
                        end_of_replay_sessions.append(session_id)

                next_prev_target = current_target.clone()
                if end_of_replay_sessions:
                    self._restore_demo2_instance_reset_state(
                        runtime,
                        reset_state,
                        end_of_replay_sessions,
                        x_tensor=x,
                    )
                    for session_id in end_of_replay_sessions:
                        start = session_id * controller_nodes
                        end = start + controller_nodes
                        next_prev_target[start:end].copy_(
                            self.batch_controller_points[replay_start, start:end]
                        )

                try:
                    current_pos, current_rot = lbs_with_rotation_reuse(
                        current_mass_nodes=x,
                        cache=runtime.rotation_cache,
                    )
                except torch._C._LinAlgError as exc:
                    batch_element = self._parse_linalg_batch_element(exc)
                    hinted_instance = None
                    if batch_element is not None:
                        hinted_instance = int(batch_element) // int(
                            runtime.object_nodes_per_instance
                        )
                        if hinted_instance < 0 or hinted_instance >= runtime.batch_size:
                            hinted_instance = None
                    raise BatchedReplayCheckError(
                        "Demo 2 LBS failed with torch.linalg.eigh "
                        f"at runtime frame {frame_counter}. "
                        "Regenerate the filtered trajectory bank with this replay range.",
                        frame_idx=frame_counter,
                        batch_element=batch_element,
                        hinted_instance=hinted_instance,
                        original_error=exc,
                    ) from exc
                runtime.gaussians._xyz = current_pos
                runtime.gaussians._rotation = current_rot

                prev_target = next_prev_target
                frame_counter += 1
        finally:
            self._write_demo2_motion_debug(motion_debug if "motion_debug" in locals() else None)
            if reg is not None:
                reg.unregister()
            if prog is not None:
                render_backend.gl.glDeleteProgram(prog)
            if tex is not None:
                render_backend.gl.glDeleteTextures([tex])
            if pbo is not None:
                render_backend.gl.glDeleteBuffers(1, [pbo])
            if vao is not None:
                render_backend.gl.glDeleteVertexArrays(1, [vao])
            cuda_ctx.pop()

    def run_batched_full_runtime(
        self,
        model_path,
        gs_path,
        output_dir,
        window,
        cuda_ctx,
        batch_size=1,
        num_views=1,
        render_mode="batch_images",
        instance_id=None,
        gaussian_render_mode="shared_template",
        save_video=False,
        save_batch_images=False,
        save_batch_grid=False,
        display_batch_grid=False,
        batch_image_resolution="native",
        batch_grid_cols=None,
        profile_render_components=False,
        sim_force_mode=SIM_FORCE_MODE_GATHER,
    ):
        gaussian_render_mode = normalize_gaussian_render_mode(gaussian_render_mode)
        if sim_force_mode not in SIM_FORCE_MODES:
            raise ValueError(
                f"sim_force_mode must be one of {SIM_FORCE_MODES}. "
                f"Received: {sim_force_mode}"
            )
        if output_dir is None:
            raise ValueError("output_dir is required for batched full-runtime runs.")
        if batch_size < 1:
            raise ValueError(
                f"batch_size must be a positive integer. Received: {batch_size}"
            )
        instance_id = self._validate_batched_render_request(
            render_mode=render_mode,
            instance_id=instance_id,
            batch_size=batch_size,
        )
        if render_mode == "batch_images":
            if num_views != 1:
                raise ValueError(
                    "render_mode='batch_images' currently supports num_views=1 only."
                )
            if batch_image_resolution not in ("native", "640x480"):
                raise ValueError(
                    "batch_image_resolution must be 'native' or '640x480'. "
                    f"Received: {batch_image_resolution}"
                )
            if batch_grid_cols is not None and int(batch_grid_cols) < 1:
                raise ValueError("batch_grid_cols must be a positive integer.")
        elif batch_image_resolution != "native":
            raise ValueError(
                "batch_image_resolution='640x480' can only be used with "
                "render_mode='batch_images'."
            )
        elif save_batch_images or save_batch_grid or display_batch_grid:
            raise ValueError(
                "--save_batch_images, --save_batch_grid, and --display_batch_grid "
                "can only be used with render_mode='batch_images'."
            )

        print(f"[BatchedRender] gaussian_render_mode={gaussian_render_mode}")
        print(f"[BatchedRender] sim_force_mode={sim_force_mode}")
        if batch_size == 1:
            print("[BatchedRender] single-instance/no-offset/no-camera-change")
        elif render_mode == "batch_images":
            print(
                "[BatchedRender] batch-images-mode/per-instance-calibrated-output/"
                f"{gaussian_render_mode}/no-shadows"
            )
        else:
            print(
                "[BatchedRender] instance-mode/diagonal-sim-offset/"
                "output-recenter/no-camera-change"
            )
        profile_path = os.path.join(output_dir, "render_component_profile.json")
        if not profile_render_components and os.path.exists(profile_path):
            os.remove(profile_path)

        runtime = self._build_runtime_core(
            model_path,
            gs_path,
            n_dup=batch_size - 1,
            gaussian_render_mode=gaussian_render_mode,
            force_shared_batched_gaussians=(render_mode == "batch_images"),
            sim_force_mode=sim_force_mode,
        )
        print("[BatchedRender] Runtime core built")

        render_backend = self._load_render_backend()
        render_backend.glfw.make_context_current(window)

        native_width, native_height = [int(value) for value in cfg.WH]
        width, height = self._resolve_batch_image_render_size(
            batch_image_resolution=batch_image_resolution,
            native_width=native_width,
            native_height=native_height,
        )
        display_width = width
        display_height = height
        if display_batch_grid:
            framebuffer_width, framebuffer_height = render_backend.glfw.get_framebuffer_size(
                window
            )
            if framebuffer_width > 0 and framebuffer_height > 0:
                display_width = int(framebuffer_width)
                display_height = int(framebuffer_height)
        available_views = min(len(cfg.intrinsics), len(cfg.w2cs))
        if num_views < 1 or num_views > available_views:
            raise ValueError(
                f"num_views must be between 1 and {available_views}. Received: {num_views}"
            )

        background = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
        intrinsic = self._scale_intrinsic_for_render_size(
            cfg.intrinsics[0],
            native_width=native_width,
            native_height=native_height,
            render_width=width,
            render_height=height,
        )
        w2c = cfg.w2cs[0]
        view, K_cuda = self._create_gs_view(w2c, intrinsic, height, width)
        render_views = [(0, view)]
        for view_idx in range(1, num_views):
            extra_intrinsic = self._scale_intrinsic_for_render_size(
                cfg.intrinsics[view_idx],
                native_width=native_width,
                native_height=native_height,
                render_width=width,
                render_height=height,
            )
            extra_view, _ = self._create_gs_view(
                cfg.w2cs[view_idx], extra_intrinsic, height, width
            )
            render_views.append((view_idx, extra_view))

        overlay = render_backend.cv2.imread(cfg.bg_img_path)
        overlay = render_backend.cv2.cvtColor(
            overlay, render_backend.cv2.COLOR_BGR2RGB
        )
        if overlay.shape[1] != width or overlay.shape[0] != height:
            overlay = render_backend.cv2.resize(
                overlay,
                (width, height),
                interpolation=render_backend.cv2.INTER_LINEAR,
            )
        overlay = torch.tensor(overlay, dtype=torch.float32, device=cfg.device)
        assert overlay.shape[0] == height and overlay.shape[1] == width, (
            f"overlay {tuple(overlay.shape)} != (H,W,3)=({height},{width},3)"
        )

        lights = torch.tensor(
            [[0, 0, -3], [1, 0.5, -2], [-3, -0.5, -5]],
            device=cfg.device,
            dtype=torch.float32,
        )
        coeffs = torch.tensor(
            [0.95, 0.97, 0.98], device=cfg.device, dtype=torch.float32
        )
        w2c_cuda = torch.tensor(w2c, dtype=torch.float32, device=cfg.device)
        coeffs_b = coeffs.view(-1, 1, 1)
        w2c_T = w2c_cuda.T.contiguous()
        intrinsic_T = K_cuda.T.contiguous()
        inv_Lz = 1.0 / lights[:, 2]
        bytes_per_pixel = 4
        pbo_size = display_width * display_height * bytes_per_pixel
        row_pitch = display_width * bytes_per_pixel

        tex = None
        pbo = None
        reg = None
        prog = None
        vao = None
        summary = {}

        try:
            print("[BatchedRender] GL interop setup start")
            tex = render_backend.gl.glGenTextures(1)
            render_backend.gl.glBindTexture(render_backend.gl.GL_TEXTURE_2D, tex)
            render_backend.gl.glTexParameteri(
                render_backend.gl.GL_TEXTURE_2D,
                render_backend.gl.GL_TEXTURE_MIN_FILTER,
                render_backend.gl.GL_NEAREST,
            )
            render_backend.gl.glTexParameteri(
                render_backend.gl.GL_TEXTURE_2D,
                render_backend.gl.GL_TEXTURE_MAG_FILTER,
                render_backend.gl.GL_NEAREST,
            )
            render_backend.gl.glTexImage2D(
                render_backend.gl.GL_TEXTURE_2D,
                0,
                render_backend.gl.GL_RGBA8,
                display_width,
                display_height,
                0,
                render_backend.gl.GL_RGBA,
                render_backend.gl.GL_UNSIGNED_BYTE,
                None,
            )
            render_backend.gl.glBindTexture(render_backend.gl.GL_TEXTURE_2D, 0)

            pbo = render_backend.gl.glGenBuffers(1)
            render_backend.gl.glBindBuffer(
                render_backend.gl.GL_PIXEL_UNPACK_BUFFER, pbo
            )
            render_backend.gl.glBufferData(
                render_backend.gl.GL_PIXEL_UNPACK_BUFFER,
                pbo_size,
                None,
                render_backend.gl.GL_STREAM_DRAW,
            )
            render_backend.gl.glBindBuffer(
                render_backend.gl.GL_PIXEL_UNPACK_BUFFER, 0
            )
            cuda_logic_error = getattr(
                render_backend.cuda_driver, "LogicError", RuntimeError
            )
            try:
                reg = render_backend.RegisteredBuffer(
                    int(pbo), render_backend.graphics_map_flags.WRITE_DISCARD
                )
            except cuda_logic_error as exc:
                if pbo is not None:
                    render_backend.gl.glDeleteBuffers(1, [pbo])
                    pbo = None
                raise RuntimeError(
                    "Failed to register the OpenGL pixel buffer with CUDA in "
                    "batched full-runtime rendering. This usually indicates a "
                    "CUDA/OpenGL interop initialization mismatch in the batched "
                    "render path."
                ) from exc

            vertex_shader_source = """
            #version 330 core
            out vec2 uv; const vec2 V[4]=vec2[4](vec2(-1,-1),vec2(1,-1),vec2(-1,1),vec2(1,1));
            const vec2 T[4]=vec2[4](vec2(0,0),vec2(1,0),vec2(0,1),vec2(1,1));
            void main(){ gl_Position=vec4(V[gl_VertexID],0,1); uv=T[gl_VertexID]; }
            """
            fragment_shader_source = """
            #version 330 core
            in vec2 uv; out vec4 frag; uniform sampler2D uTex;
            void main(){ frag = texture(uTex, vec2(uv.x,1.0 - uv.y)); }
            """

            def _compile(kind, src):
                shader_id = render_backend.gl.glCreateShader(kind)
                render_backend.gl.glShaderSource(shader_id, src)
                render_backend.gl.glCompileShader(shader_id)
                if not render_backend.gl.glGetShaderiv(
                    shader_id, render_backend.gl.GL_COMPILE_STATUS
                ):
                    raise RuntimeError(
                        render_backend.gl.glGetShaderInfoLog(shader_id).decode()
                    )
                return shader_id

            prog = render_backend.gl.glCreateProgram()
            render_backend.gl.glAttachShader(
                prog,
                _compile(render_backend.gl.GL_VERTEX_SHADER, vertex_shader_source),
            )
            render_backend.gl.glAttachShader(
                prog,
                _compile(render_backend.gl.GL_FRAGMENT_SHADER, fragment_shader_source),
            )
            render_backend.gl.glLinkProgram(prog)
            if not render_backend.gl.glGetProgramiv(
                prog, render_backend.gl.GL_LINK_STATUS
            ):
                raise RuntimeError(
                    render_backend.gl.glGetProgramInfoLog(prog).decode()
                )
            render_backend.gl.glUseProgram(prog)
            render_backend.gl.glUniform1i(
                render_backend.gl.glGetUniformLocation(prog, "uTex"), 0
            )
            render_backend.gl.glUseProgram(0)
            vao = render_backend.gl.glGenVertexArrays(1)
            render_backend.gl.glBindVertexArray(vao)

            pbo_stream = render_backend.cuda_driver.Stream()
            cpy2d = render_backend.cuda_driver.Memcpy2D()
            cpy2d.src_pitch = row_pitch
            cpy2d.dst_pitch = row_pitch
            cpy2d.width_in_bytes = row_pitch
            cpy2d.height = display_height
            print("[BatchedRender] GL interop setup complete")

            frame_rgba = torch.empty(
                (display_height, display_width, 4),
                dtype=torch.uint8,
                device=cfg.device,
            )
            frame = torch.empty(
                (display_height, display_width, 3),
                dtype=overlay.dtype,
                device=cfg.device,
            )
            rgb_temp = torch.empty(
                (height, width, 3), dtype=overlay.dtype, device=cfg.device
            )

            save_frame_artifacts = save_video or save_batch_images or save_batch_grid
            eval_render_paths = {}
            view_render_path = None
            batch_image_paths = []
            batch_grid_path = None
            if save_frame_artifacts:
                import torchvision

                if render_mode == "batch_images":
                    if save_video:
                        view_render_path = os.path.join(output_dir, "output")
                        os.makedirs(view_render_path, exist_ok=True)
                    if save_batch_images:
                        batch_image_root = os.path.join(output_dir, "batch_images")
                        batch_image_paths = [
                            os.path.join(batch_image_root, f"instance_{idx:03d}")
                            for idx in range(runtime.batch_size)
                        ]
                        for render_path in batch_image_paths:
                            os.makedirs(render_path, exist_ok=True)
                    if save_batch_grid:
                        batch_grid_path = os.path.join(output_dir, "output_grid")
                        os.makedirs(batch_grid_path, exist_ok=True)
                else:
                    eval_render_paths = {
                        view_idx: os.path.join(output_dir, str(view_idx))
                        for view_idx in range(num_views)
                    }
                    view_render_path = os.path.join(output_dir, "output")
                    os.makedirs(view_render_path, exist_ok=True)
                    for render_path in eval_render_paths.values():
                        os.makedirs(render_path, exist_ok=True)

            sim_timer = Timer("Simulator")
            interp_timer = Timer("Linear Blend Skinning")
            render_timer = Timer("Rendering")
            frame_timer = Timer("Frame Compositing")
            total_timer = Timer("Total Loop")

            component_times = {
                "simulator": [],
                "full_motion_interpolation": [],
                "rendering": [],
                "frame_compositing": [],
                "total": [],
            }
            render_component_profiler = (
                RenderComponentProfiler()
                if profile_render_components
                else None
            )

            gaussians = runtime.gaussians
            render_gaussians = runtime.render_gaussians
            if render_mode == "instance" and runtime.batch_size > 1:
                render_gaussians = build_instance_selective_render_view(
                    runtime.gaussians,
                    instance_id,
                    gaussian_render_mode=gaussian_render_mode,
                )
            elif render_mode == "batch_images":
                render_gaussians = build_batch_images_render_view(
                    runtime.gaussians,
                    gaussian_render_mode=gaussian_render_mode,
                )

            frame_count = runtime.frame_count
            prev_target = runtime.prev_target
            current_target = runtime.current_target
            prev_x = runtime.prev_x

            while True:
                total_timer.start()

                sim_timer.start()
                self.simulator.set_controller_interactive(prev_target, current_target)

                if self.simulator.object_collision_flag:
                    self.simulator.update_collision_graph()
                wp.capture_launch(self.simulator.forward_graph)
                wp.synchronize()

                x = wp.to_torch(self.simulator.wp_states[-1].wp_x, requires_grad=False)
                self.simulator.set_init_state(
                    self.simulator.wp_states[-1].wp_x,
                    self.simulator.wp_states[-1].wp_v,
                )
                sim_time = sim_timer.stop()

                if frame_count > 1:
                    component_times["simulator"].append(sim_time)

                render_timer.start()
                profile_this_frame = bool(profile_render_components and frame_count > 1)
                frame_profiler = (
                    render_component_profiler
                    if render_component_profiler is not None
                    and profile_this_frame
                    else None
                )
                if frame_profiler is not None:
                    if render_mode == "instance":
                        total_render_gaussians = int(runtime.gaussians_per_instance)
                    else:
                        total_render_gaussians = int(
                            runtime.batch_size * runtime.gaussians_per_instance
                        )
                    frame_profiler.begin_frame(
                        total_gaussians=total_render_gaussians,
                        gaussians_per_instance=int(runtime.gaussians_per_instance),
                    )
                render_profile_success = False
                try:
                    results = render_backend.render_gaussian(
                        view,
                        render_gaussians,
                        None,
                        background,
                        profiler=frame_profiler,
                    )
                    render_profile_success = True
                finally:
                    if frame_profiler is not None:
                        if render_profile_success and profile_this_frame:
                            frame_profiler.end_frame()
                        else:
                            frame_profiler.discard_frame()
                rendering = results["render"]
                additional_renderings = {}
                if render_mode != "batch_images" and save_video and num_views > 1:
                    for view_idx, extra_view in render_views[1:]:
                        additional_renderings[view_idx] = render_backend.render_gaussian(
                            extra_view, render_gaussians, None, background
                        )["render"]
                if render_mode != "batch_images":
                    image = rendering.permute(1, 2, 0).detach()
                render_time = render_timer.stop()

                if frame_count > 1:
                    component_times["rendering"].append(render_time)

                frame_timer.start()
                if render_mode == "batch_images":
                    batch_frames, _ = self._composite_batch_images_without_shadows(
                        rendering,
                        overlay,
                    )
                    batch_grid = None
                    if display_batch_grid:
                        batch_grid = self._make_batch_image_grid(
                            batch_frames,
                            batch_grid_cols=batch_grid_cols,
                        )
                        display_grid = batch_grid
                        if (
                            display_grid.shape[0] != display_height
                            or display_grid.shape[1] != display_width
                        ):
                            display_grid = F.interpolate(
                                display_grid.permute(2, 0, 1).unsqueeze(0),
                                size=(display_height, display_width),
                                mode="bilinear",
                                align_corners=False,
                            ).squeeze(0).permute(1, 2, 0)
                        frame.copy_(display_grid)
                    else:
                        frame.copy_(batch_frames[0])
                else:
                    frame.copy_(overlay)

                    image.clamp_(0, 1)
                    image_mask = torch.logical_and(
                        (image != 1.0).any(dim=2), image[:, :, 3] > 100 / 255
                    )
                    image[..., 3].masked_fill_(~image_mask, 0.0)

                    alpha = image[..., 3:4]
                    torch.mul(image[..., :3], alpha, out=rgb_temp)
                    rgb_temp.mul_(255.0)
                    frame.mul_(1.0 - alpha).add_(rgb_temp)

                    composite_points = self._select_runtime_points(
                        points=x,
                        runtime=runtime,
                        render_mode=render_mode,
                        instance_id=instance_id,
                    )
                    masks = get_shadow_masks_batched_downsampled(
                        points=composite_points,
                        intrinsic_T=intrinsic_T,
                        w2c_T=w2c_T,
                        W=width,
                        H=height,
                        image_mask=image_mask,
                        lights=lights,
                        inv_Lz=inv_Lz,
                        kernel_size=7,
                        scale=2,
                        use_half=False,
                        upsample_mode="bilinear",
                        post_blur=False,
                    )

                    attenuation = torch.prod(
                        1.0 - masks.to(frame.dtype) + masks.to(frame.dtype) * coeffs_b,
                        dim=0,
                    )
                    frame.mul_(attenuation.unsqueeze(-1))

                frame_u8 = frame.clamp(0, 255).to(torch.uint8)
                frame_rgba[:, :, :3] = frame_u8
                frame_rgba[:, :, 3] = 255
                torch.cuda.current_stream().synchronize()

                mapping = reg.map()
                try:
                    ptr, _ = mapping.device_ptr_and_size()
                    cpy2d.set_src_device(frame_rgba.data_ptr())
                    cpy2d.set_dst_device(ptr)
                    cpy2d(pbo_stream)
                    pbo_stream.synchronize()
                finally:
                    mapping.unmap()

                render_backend.gl.glBindTexture(render_backend.gl.GL_TEXTURE_2D, tex)
                render_backend.gl.glBindBuffer(
                    render_backend.gl.GL_PIXEL_UNPACK_BUFFER, pbo
                )
                render_backend.gl.glTexSubImage2D(
                    render_backend.gl.GL_TEXTURE_2D,
                    0,
                    0,
                    0,
                    display_width,
                    display_height,
                    render_backend.gl.GL_RGBA,
                    render_backend.gl.GL_UNSIGNED_BYTE,
                    None,
                )
                render_backend.gl.glBindBuffer(
                    render_backend.gl.GL_PIXEL_UNPACK_BUFFER, 0
                )

                render_backend.gl.glViewport(0, 0, display_width, display_height)
                render_backend.gl.glDisable(render_backend.gl.GL_DEPTH_TEST)
                render_backend.gl.glClear(render_backend.gl.GL_COLOR_BUFFER_BIT)
                render_backend.gl.glUseProgram(prog)
                render_backend.gl.glActiveTexture(render_backend.gl.GL_TEXTURE0)
                render_backend.gl.glBindTexture(render_backend.gl.GL_TEXTURE_2D, tex)
                render_backend.gl.glDrawArrays(
                    render_backend.gl.GL_TRIANGLE_STRIP, 0, 4
                )
                render_backend.gl.glBindTexture(render_backend.gl.GL_TEXTURE_2D, 0)
                render_backend.gl.glUseProgram(0)

                render_backend.glfw.swap_buffers(window)
                render_backend.glfw.poll_events()

                frame_comp_time = frame_timer.stop()
                if frame_count > 1:
                    component_times["frame_compositing"].append(frame_comp_time)

                should_prepare_next_frame = frame_count + 1 < runtime.frame_len
                if prev_x is not None and should_prepare_next_frame:
                    with torch.no_grad():
                        interp_timer.start()
                        current_pos, current_rot = lbs_with_rotation_reuse(
                            current_mass_nodes=x,
                            cache=runtime.rotation_cache,
                        )
                        interp_time = interp_timer.stop()
                        gaussians._xyz = current_pos
                        gaussians._rotation = current_rot

                    if frame_count > 1:
                        component_times["full_motion_interpolation"].append(
                            interp_time
                        )

                prev_x = x.clone()

                total_time = total_timer.stop()
                if frame_count > 1:
                    component_times["total"].append(total_time)

                if save_frame_artifacts:
                    if render_mode == "batch_images":
                        if save_video:
                            torchvision.utils.save_image(
                                frame.permute(2, 0, 1).float() / 255.0,
                                os.path.join(view_render_path, f"{frame_count:05d}.png"),
                            )
                        if save_batch_images:
                            for batch_idx, render_path in enumerate(batch_image_paths):
                                torchvision.utils.save_image(
                                    batch_frames[batch_idx]
                                    .permute(2, 0, 1)
                                    .float()
                                    / 255.0,
                                    os.path.join(
                                        render_path, f"{frame_count:05d}.png"
                                    ),
                                )
                        if save_batch_grid:
                            if batch_grid is None:
                                batch_grid = self._make_batch_image_grid(
                                    batch_frames,
                                    batch_grid_cols=batch_grid_cols,
                                )
                            torchvision.utils.save_image(
                                batch_grid.permute(2, 0, 1).float() / 255.0,
                                os.path.join(
                                    batch_grid_path, f"{frame_count:05d}.png"
                                ),
                            )
                    else:
                        torchvision.utils.save_image(
                            rendering,
                            os.path.join(
                                eval_render_paths[0], f"{frame_count:05d}.png"
                            ),
                        )
                        for view_idx, extra_rendering in additional_renderings.items():
                            torchvision.utils.save_image(
                                extra_rendering,
                                os.path.join(
                                    eval_render_paths[view_idx],
                                    f"{frame_count:05d}.png",
                                ),
                            )
                        composite_rgb = frame_rgba[:, :, :3]
                        composite_rgb = composite_rgb.permute(2, 0, 1).float() / 255.0
                        torchvision.utils.save_image(
                            composite_rgb,
                            os.path.join(view_render_path, f"{frame_count:05d}.png"),
                        )

                frame_count += 1
                prev_target = current_target
                if frame_count < runtime.frame_len:
                    current_target = self.batch_controller_points[frame_count]
                else:
                    print("Reached end of recorded control sequence")
                    break
        finally:
            summary = self._write_full_runtime_summary(
                output_dir=output_dir,
                batch_size=runtime.batch_size if "runtime" in locals() else batch_size,
                render_mode=render_mode,
                gaussian_render_mode=gaussian_render_mode,
                instance_id=instance_id,
                component_times=component_times
                if "component_times" in locals()
                else {
                    "simulator": [],
                    "full_motion_interpolation": [],
                    "rendering": [],
                    "frame_compositing": [],
                    "total": [],
                },
                batch_image_resolution=batch_image_resolution,
                render_width=width if "width" in locals() else None,
                render_height=height if "height" in locals() else None,
                sim_force_mode=runtime.sim_force_mode
                if "runtime" in locals()
                else sim_force_mode,
            )
            if profile_render_components and "render_component_profiler" in locals():
                render_component_profiler.write_json(
                    output_dir=output_dir,
                    metadata={
                        "batch_size": int(
                            runtime.batch_size if "runtime" in locals() else batch_size
                        ),
                        "gaussian_render_mode": gaussian_render_mode,
                        "sim_force_mode": runtime.sim_force_mode
                        if "runtime" in locals()
                        else sim_force_mode,
                        "render_mode": render_mode,
                        "instance_id": int(instance_id)
                        if instance_id is not None
                        else None,
                        "batch_image_resolution": batch_image_resolution,
                        "render_width": int(width) if "width" in locals() else None,
                        "render_height": int(height) if "height" in locals() else None,
                    },
                )

            if reg is not None:
                reg.unregister()
            if prog is not None:
                render_backend.gl.glDeleteProgram(prog)
            if tex is not None:
                render_backend.gl.glDeleteTextures([tex])
            if pbo is not None:
                render_backend.gl.glDeleteBuffers(1, [pbo])
            if vao is not None:
                render_backend.gl.glDeleteVertexArrays(1, [vao])
            cuda_ctx.pop()

        return summary


    #this is basically baseline + rendering (to verify correctness)        
    def interactive_playground(
        self,
        model_path,
        gs_path,
        output_dir=None,
        window=None,
        cuda_ctx=None,
        n_dup=0,
        save_eval_artifacts=False,
    ):
        runtime = self._build_runtime_core(model_path, gs_path, n_dup=n_dup)
        gaussians = runtime.gaussians
        render_gaussians = runtime.render_gaussians
        rotation_cache = runtime.rotation_cache
        prev_x = runtime.prev_x
        prev_target = runtime.prev_target
        current_target = runtime.current_target
        frame_count = runtime.frame_count
        render_backend = self._load_render_backend()

        ########add visualization initialization
        render_backend.glfw.make_context_current(window)
        width, height = cfg.WH
        intrinsic = cfg.intrinsics[0]
        w2c = cfg.w2cs[0]

        use_white_background = True  # set to True for white background
        bg_color = [1, 1, 1] if use_white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        view, K_cuda = self._create_gs_view(w2c, intrinsic, height, width)
        image_path = cfg.bg_img_path
        overlay = render_backend.cv2.imread(image_path)
        overlay = render_backend.cv2.cvtColor(overlay, render_backend.cv2.COLOR_BGR2RGB)
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

        # Texture
        tex = render_backend.gl.glGenTextures(1)
        render_backend.gl.glBindTexture(render_backend.gl.GL_TEXTURE_2D, tex)
        render_backend.gl.glTexParameteri(render_backend.gl.GL_TEXTURE_2D, render_backend.gl.GL_TEXTURE_MIN_FILTER, render_backend.gl.GL_NEAREST)
        render_backend.gl.glTexParameteri(render_backend.gl.GL_TEXTURE_2D, render_backend.gl.GL_TEXTURE_MAG_FILTER, render_backend.gl.GL_NEAREST)
        render_backend.gl.glTexImage2D(render_backend.gl.GL_TEXTURE_2D, 0, render_backend.gl.GL_RGBA8, width, height, 0, render_backend.gl.GL_RGBA, render_backend.gl.GL_UNSIGNED_BYTE, None)
        render_backend.gl.glBindTexture(render_backend.gl.GL_TEXTURE_2D, 0)

        # PBO
        pbo = render_backend.gl.glGenBuffers(1)
        render_backend.gl.glBindBuffer(render_backend.gl.GL_PIXEL_UNPACK_BUFFER, pbo)
        render_backend.gl.glBufferData(render_backend.gl.GL_PIXEL_UNPACK_BUFFER, pbo_size, None, render_backend.gl.GL_STREAM_DRAW)
        render_backend.gl.glBindBuffer(render_backend.gl.GL_PIXEL_UNPACK_BUFFER, 0)
        reg = render_backend.RegisteredBuffer(
            int(pbo), render_backend.graphics_map_flags.WRITE_DISCARD
        )

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
            sid = render_backend.gl.glCreateShader(kind); render_backend.gl.glShaderSource(sid, src); render_backend.gl.glCompileShader(sid)
            if not render_backend.gl.glGetShaderiv(sid, render_backend.gl.GL_COMPILE_STATUS):
                raise RuntimeError(render_backend.gl.glGetShaderInfoLog(sid).decode())
            return sid
        
        prog = render_backend.gl.glCreateProgram()
        render_backend.gl.glAttachShader(prog, _compile(render_backend.gl.GL_VERTEX_SHADER, VS))
        render_backend.gl.glAttachShader(prog, _compile(render_backend.gl.GL_FRAGMENT_SHADER, FS))
        render_backend.gl.glLinkProgram(prog)
        if not render_backend.gl.glGetProgramiv(prog, render_backend.gl.GL_LINK_STATUS):
            raise RuntimeError(render_backend.gl.glGetProgramInfoLog(prog).decode())
        render_backend.gl.glUseProgram(prog); render_backend.gl.glUniform1i(render_backend.gl.glGetUniformLocation(prog, "uTex"), 0); render_backend.gl.glUseProgram(0)
        vao = render_backend.gl.glGenVertexArrays(1)
        render_backend.gl.glBindVertexArray(vao)


        # Reuse Stream
        pbo_stream = render_backend.cuda_driver.Stream()

        cpy2d = render_backend.cuda_driver.Memcpy2D()
        cpy2d.src_pitch = row_pitch
        cpy2d.dst_pitch = row_pitch
        cpy2d.width_in_bytes = row_pitch
        cpy2d.height = height

        # These could be pre-allocated once:
        frame_rgba = torch.empty((height, width, 4), dtype=torch.uint8, device=cfg.device)
        frame = torch.empty_like(overlay)
        rgb_temp = torch.empty((height, width, 3), dtype=overlay.dtype, device=cfg.device)  # Add this

        #for saving output videos
        if output_dir and save_eval_artifacts:
            eval_render_path = os.path.join(output_dir, "0")
            view_render_path = os.path.join(output_dir, "output")
            os.makedirs(view_render_path, exist_ok=True)
            os.makedirs(eval_render_path, exist_ok=True)
            #traj_save_path = os.path.join(eval_image_path, "inference.pkl")


        #############end of visualization initialization

        #timer initialization code
        sim_timer = Timer("Simulator")
        interp_timer = Timer("Linear Blend Skinning")
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
            "total": [],
        }
        #uncomment this for generating qualtiy comparision data
        # traj_frames_cpu = []  # list of (N,3) CPU tensors
        # x0 = wp.to_torch(self.simulator.wp_states[0].wp_x, requires_grad=False)
        # traj_frames_cpu.append(x0.detach().cpu())
        try:
            while True:

                total_timer.start()

                # 1. Simulator step
                #uncomment for correct trajectory generation
                # if frame_count == 0:
                #     x = wp.to_torch(self.simulator.wp_states[0].wp_x, requires_grad=False)
                # else:
                #     sim_timer.start()

                #     self.simulator.set_controller_interactive(prev_target, current_target)

                #     if self.simulator.object_collision_flag:
                #         self.simulator.update_collision_graph()
                #     wp.capture_launch(self.simulator.forward_graph)
                #     wp.synchronize()

                #     x = wp.to_torch(self.simulator.wp_states[-1].wp_x, requires_grad=False)
                #     #uncomment to generate quality comparison data
                #     traj_frames_cpu.append(x.detach().cpu())
                
                #     # Set the intial state for the next step
                #     self.simulator.set_init_state(
                #         self.simulator.wp_states[-1].wp_x,
                #         self.simulator.wp_states[-1].wp_v,
                #     )

                #     sim_time = sim_timer.stop()

                #     #pyh ignore first two frame since 1st frame is skewed during to rendering initialization and 2nd frame is skewed toward getting neighboruing weights
                #     if frame_count > 1:
                #         component_times["simulator"].append(sim_time)

                #uncomment for performance                
                sim_timer.start()

                self.simulator.set_controller_interactive(prev_target, current_target)

                if self.simulator.object_collision_flag:
                    self.simulator.update_collision_graph()
                wp.capture_launch(self.simulator.forward_graph)
                wp.synchronize()

                x = wp.to_torch(self.simulator.wp_states[-1].wp_x, requires_grad=False)
                # if frame_count == 0:
                #     max_err = (x.detach() - wp.to_torch(self.simulator.wp_states[0].wp_x, requires_grad=False)).abs().max().item()
                #     print(f"[traj] frame0 post-step max |Δx| = {max_err:e}")
                # #uncomment to generate quality comparison data
                # if frame_count > 0:
                #     traj_frames_cpu.append(x.detach().cpu())
            
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

                # render with gaussians and paste the image on top of the frame
                results = render_backend.render_gaussian(
                    view, render_gaussians, None, background
                )
                rendering = results["render"]  # (4, H, W)
                image = rendering.permute(1, 2, 0).detach()

                render_time = render_timer.stop()
                if frame_count > 1:
                    component_times["rendering"].append(render_time)

                #frame compositing
                frame_timer.start()
                
                frame.copy_(overlay)
                
                image.clamp_(0, 1)
                image_mask = torch.logical_and(
                    (image != 1.0).any(dim=2), image[:, :, 3] > 100 / 255
                )
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

                frame_u8 = frame.clamp_(0, 255).to(torch.uint8)   # RGB uint8

                ####################pyh new rendering direclty render on GPU n(o GPU to CPU copying)
                # device→device copy into the mapped PBO, then update the texture and draw
                #convert to rgba
                frame_rgba[:, :, :3] = frame_u8
                frame_rgba[:, :, 3] = 255
                torch.cuda.current_stream().synchronize()  # ensures frame_rgba is ready to be read
                
                mapping = reg.map()
                try:
                    ptr, _ = mapping.device_ptr_and_size()
                    cpy2d.set_src_device(frame_rgba.data_ptr())
                    cpy2d.set_dst_device(ptr)
                    cpy2d(pbo_stream)

                    pbo_stream.synchronize()
                finally:
                    mapping.unmap()

                # Upload from PBO to texture (still on GPU)
                render_backend.gl.glBindTexture(render_backend.gl.GL_TEXTURE_2D, tex)
                render_backend.gl.glBindBuffer(render_backend.gl.GL_PIXEL_UNPACK_BUFFER, pbo)
                render_backend.gl.glTexSubImage2D(render_backend.gl.GL_TEXTURE_2D, 0, 0, 0, width, height, render_backend.gl.GL_RGBA, render_backend.gl.GL_UNSIGNED_BYTE, None)
                render_backend.gl.glBindBuffer(render_backend.gl.GL_PIXEL_UNPACK_BUFFER, 0)
                
                # Draw
                render_backend.gl.glViewport(0, 0, width, height)
                render_backend.gl.glDisable(render_backend.gl.GL_DEPTH_TEST)
                render_backend.gl.glClear(render_backend.gl.GL_COLOR_BUFFER_BIT)
                render_backend.gl.glUseProgram(prog)
                render_backend.gl.glActiveTexture(render_backend.gl.GL_TEXTURE0)
                render_backend.gl.glBindTexture(render_backend.gl.GL_TEXTURE_2D, tex)
                render_backend.gl.glDrawArrays(render_backend.gl.GL_TRIANGLE_STRIP, 0, 4)
                render_backend.gl.glBindTexture(render_backend.gl.GL_TEXTURE_2D, 0)
                render_backend.gl.glUseProgram(0)
                
                render_backend.glfw.swap_buffers(window)
                render_backend.glfw.poll_events()

                frame_comp_time = frame_timer.stop() 
                if frame_count > 1:
                    # Total frame compositing time
                    component_times["frame_compositing"].append(frame_comp_time)


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

                # Save images before incrementing so frame 0 is written as 00000.png.
                if output_dir and save_eval_artifacts:
                   import torchvision

                   torchvision.utils.save_image(
                           rendering,
                           os.path.join(eval_render_path, "{0:05d}".format(frame_count) + ".png"),
                   )
                   save_path = os.path.join(view_render_path, f"{frame_count:05d}.png")
                   img_rgb = frame_rgba[:, :, :3]
                   img_rgb = img_rgb.permute(2, 0, 1).float() / 255.0
                   torchvision.utils.save_image(img_rgb, save_path)

                frame_count += 1

                prev_target = current_target
                #New updated to use the multi-input length
                if frame_count < runtime.frame_len:
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
                        readable_name = perf_component_label(component_name)
                        print(f"{readable_name}: {average_component_time * 1000:.2f} ms ({time_share_percentage:.1f}%)")
                        log_lines.append(f"{readable_name}: {average_component_time * 1000:.2f} ms ({time_share_percentage:.1f}%)")

                #pyh save the performance log to a file
                if output_dir is not None:
                    os.makedirs(output_dir, exist_ok=True)
                    log_file_path = os.path.join(output_dir, "performance_summary.txt")
                    with open(log_file_path, "w") as log_file:
                        log_file.write("\n".join(log_lines))

            #uncomment to save trajectory data for quality comparison
            # vertices = torch.stack(traj_frames_cpu, dim=0)  # (T, N, 3) on CPU
            # with open(traj_save_path, "wb") as f:
            #     pickle.dump(vertices.numpy(), f)
            # print(f"[Saved] inference.pkl -> {traj_save_path}, shape={tuple(vertices.shape)}")
            
            reg.unregister()
            render_backend.gl.glDeleteProgram(prog)
            render_backend.gl.glDeleteTextures([tex])
            render_backend.gl.glDeleteBuffers(1, [pbo])
            render_backend.gl.glDeleteVertexArrays(1, [vao])
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
