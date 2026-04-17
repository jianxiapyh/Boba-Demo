import numpy as np
import torch
import pickle
from qqtt.utils import logger, visualize_pc, cfg
import matplotlib.pyplot as plt


def _object_support_patch_center_np(object_points: np.ndarray, reverse_z: bool) -> np.ndarray:
    object_points = np.asarray(object_points, dtype=np.float64)
    scene_down = np.array(
        [0.0, 0.0, 1.0 if reverse_z else -1.0],
        dtype=np.float64,
    )
    support_depth = object_points @ scene_down
    support_depth_max = float(np.max(support_depth))
    support_mask = support_depth >= (support_depth_max - 0.012)
    support_points = object_points[support_mask]
    if support_points.size == 0:
        support_points = object_points
    support_center = support_points.mean(axis=0)
    support_center = support_center.astype(np.float64, copy=True)
    support_center[2] = (
        float(np.max(object_points[:, 2]))
        if reverse_z
        else float(np.min(object_points[:, 2]))
    )
    return support_center


def _scale_points_about_pivot(points: np.ndarray, pivot: np.ndarray, scale: float) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    pivot = np.asarray(pivot, dtype=np.float64).reshape(1, 1, 3) if points.ndim == 3 else np.asarray(pivot, dtype=np.float64).reshape(1, 3)
    return pivot + (points - pivot) * float(scale)


def _principal_axis_span_np(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] <= 1:
        return 0.0
    centered = points - points.mean(axis=0, keepdims=True)
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    axis = vh[0]
    proj = centered @ axis
    return float(np.max(proj) - np.min(proj))


class RealData:
    def __init__(self, visualize=False, save_gt=True):
        logger.info(f"[DATA]: loading data from {cfg.data_path}")
        self.data_path = cfg.data_path
        self.base_dir = cfg.base_dir
        with open(self.data_path, "rb") as f:
            data = pickle.load(f)

        object_points = data["object_points"]
        object_colors = data["object_colors"]
        object_visibilities = data["object_visibilities"]
        object_motions_valid = data["object_motions_valid"]
        controller_points = data["controller_points"]
        other_surface_points = data["surface_points"]
        interior_points = data["interior_points"]
        demo_case_name = str(getattr(cfg, "demo_case_name", "")).strip().lower()
        demo_case_world_scale = float(getattr(cfg, "demo_case_world_scale", 1.0))
        self.demo_case_scale_debug = None
        if abs(demo_case_world_scale - 1.0) > 1e-8:
            scale_pivot = _object_support_patch_center_np(
                object_points[0],
                reverse_z=bool(getattr(cfg, "reverse_z", True)),
            )
            object_span_before = _principal_axis_span_np(object_points[0])
            object_points = _scale_points_about_pivot(
                object_points,
                scale_pivot,
                demo_case_world_scale,
            )
            controller_points = _scale_points_about_pivot(
                controller_points,
                scale_pivot,
                demo_case_world_scale,
            )
            other_surface_points = _scale_points_about_pivot(
                other_surface_points,
                scale_pivot,
                demo_case_world_scale,
            )
            interior_points = _scale_points_about_pivot(
                interior_points,
                scale_pivot,
                demo_case_world_scale,
            )
            object_span_after = _principal_axis_span_np(object_points[0])
            self.demo_case_scale_debug = {
                "case_name": demo_case_name,
                "scale": float(demo_case_world_scale),
                "pivot": scale_pivot.tolist(),
                "object_span_before": float(object_span_before),
                "object_span_after": float(object_span_after),
            }
            logger.info(
                "[DATA]: demo case world scale applied "
                f"case={demo_case_name} "
                f"scale={demo_case_world_scale:.8f} "
                f"pivot={scale_pivot.tolist()} "
                f"frame0_object_span_before={object_span_before:.8f} "
                f"frame0_object_span_after={object_span_after:.8f}"
            )

        # Get the rainbow color for the object_colors
        y_min, y_max = np.min(object_points[0, :, 1]), np.max(object_points[0, :, 1])
        y_normalized = (object_points[0, :, 1] - y_min) / (y_max - y_min)
        rainbow_colors = plt.cm.rainbow(y_normalized)[:, :3]

        self.num_original_points = object_points.shape[1]
        self.num_surface_points = (
            self.num_original_points + other_surface_points.shape[0]
        )
        self.num_all_points = self.num_surface_points + interior_points.shape[0]

        # Concatenate the surface points and interior points
        self.structure_points = np.concatenate(
            [object_points[0], other_surface_points, interior_points], axis=0
        )
        self.structure_points = torch.tensor(
            self.structure_points, dtype=torch.float32, device=cfg.device
        )

        self.object_points = torch.tensor(
            object_points, dtype=torch.float32, device=cfg.device
        )
        # self.object_colors = torch.tensor(
        #     object_colors, dtype=torch.float32, device=cfg.device
        # )
        self.original_object_colors = torch.tensor(
            object_colors, dtype=torch.float32, device=cfg.device
        )
        # Apply the rainbow color to the object_colors
        rainbow_colors = torch.tensor(
            rainbow_colors, dtype=torch.float32, device=cfg.device
        )
        # Make the same rainbow color for each frame
        self.object_colors = rainbow_colors.repeat(self.object_points.shape[0], 1, 1)

        # # Apply the first frame color to all frames
        # first_frame_colors = torch.tensor(
        #     object_colors[0], dtype=torch.float32, device=cfg.device
        # )
        # self.object_colors = first_frame_colors.repeat(self.object_points.shape[0], 1, 1)

        self.object_visibilities = torch.tensor(
            object_visibilities, dtype=torch.bool, device=cfg.device
        )
        self.object_motions_valid = torch.tensor(
            object_motions_valid, dtype=torch.bool, device=cfg.device
        )
        self.controller_points = torch.tensor(
            controller_points, dtype=torch.float32, device=cfg.device
        )

        self.frame_len = self.object_points.shape[0]
        # Visualize/save the GT frames
        self.visualize_data(visualize=visualize, save_gt=save_gt)

    def visualize_data(self, visualize=False, save_gt=True):
        if visualize:
            visualize_pc(
                self.object_points,
                self.object_colors,
                self.controller_points,
                self.object_visibilities,
                self.object_motions_valid,
                visualize=True,
            )
        if save_gt:
            visualize_pc(
                self.object_points,
                self.object_colors,
                self.controller_points,
                self.object_visibilities,
                self.object_motions_valid,
                visualize=False,
                save_video=True,
                save_path=f"{self.base_dir}/gt.mp4",
            )
