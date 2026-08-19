#!/usr/bin/env python3
"""Render official Garden training cameras for calibration diagnostics."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from PIL import Image

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.scene.cameras import Camera
from gaussian_splatting.scene.gaussian_model import GaussianModel
from qqtt.garden_assets import (
    centerpiece_removal_mask,
    garden_quality_paths,
    garden_source_cameras_path,
    garden_source_point_cloud_path,
    load_garden_manifest,
    resolve_repo_path,
)


def build_camera(entry: dict, width: int) -> Camera:
    source_width = int(entry["width"])
    source_height = int(entry["height"])
    height = max(1, int(round(width * source_height / source_width)))
    rotation = np.asarray(entry["rotation"], dtype=np.float32)
    position = np.asarray(entry["position"], dtype=np.float32)
    translation = -(rotation.T @ position)
    fov_x = 2.0 * math.atan(source_width / (2.0 * float(entry["fx"])))
    fov_y = 2.0 * math.atan(source_height / (2.0 * float(entry["fy"])))
    return Camera(
        (width, height),
        colmap_id=int(entry["id"]),
        R=rotation,
        T=translation,
        FoVx=fov_x,
        FoVy=fov_y,
        depth_params=None,
        image=None,
        invdepthmap=None,
        image_name=str(entry["img_name"]),
        uid=int(entry["id"]),
        data_device="cuda",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, action="append", required=True)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument(
        "--quality",
        choices=("source", "full", "balanced", "performance"),
        default="source",
    )
    parser.add_argument(
        "--show-removal",
        action="store_true",
        help="render the calibrated centerpiece removal volume in red",
    )
    parser.add_argument(
        "--patch-view",
        choices=("all", "only", "exclude"),
        default="all",
        help="diagnose the generated tabletop patch independently",
    )
    parser.add_argument(
        "--report-largest",
        type=int,
        default=0,
        help="print the largest projected Gaussians for removal calibration",
    )
    parser.add_argument(
        "--sample",
        action="append",
        default=[],
        metavar="U,V",
        help="print a source-world point for a rendered pixel and expected depth",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data/garden/calibration_renders",
    )
    args = parser.parse_args()

    with garden_source_cameras_path(REPO_ROOT).open("r", encoding="utf-8") as handle:
        cameras = {int(entry["id"]): entry for entry in json.load(handle)}
    _, manifest = load_garden_manifest(REPO_ROOT)
    with resolve_repo_path(REPO_ROOT, manifest["calibration"]).open(
        "r", encoding="utf-8"
    ) as handle:
        calibration = json.load(handle)
    model_path = garden_source_point_cloud_path(REPO_ROOT)
    runtime_metadata = None
    if args.quality != "source":
        model_path, metadata_path = garden_quality_paths(REPO_ROOT, args.quality)
        with metadata_path.open("r", encoding="utf-8") as handle:
            runtime_metadata = json.load(handle)
        scene_rotation = np.asarray(
            calibration["source_to_canonical_rotation"], dtype=np.float32
        )
        scene_center = np.asarray(calibration["source_table_center"], dtype=np.float32)
        scene_scale = float(calibration["meters_per_source_unit"])
        for entry in cameras.values():
            entry["position"] = (
                scene_rotation
                @ (np.asarray(entry["position"], dtype=np.float32) - scene_center)
                * scene_scale
            ).tolist()
            entry["rotation"] = (
                scene_rotation @ np.asarray(entry["rotation"], dtype=np.float32)
            ).tolist()
    model = GaussianModel(sh_degree=3)
    model.load_ply(str(model_path))
    model.active_sh_degree = 3
    if args.patch_view != "all":
        if runtime_metadata is None:
            raise ValueError("--patch-view requires a prepared runtime quality.")
        patch_count = int(runtime_metadata["patch_gaussian_count"])
        with torch.no_grad():
            if args.patch_view == "only":
                model._opacity[:-patch_count].fill_(-100.0)
            else:
                model._opacity[-patch_count:].fill_(-100.0)
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    pipeline = SimpleNamespace()
    override_color = None
    if args.show_removal:
        xyz = model.get_xyz
        if args.quality == "source":
            rotation = torch.as_tensor(
                calibration["source_to_canonical_rotation"],
                dtype=xyz.dtype,
                device=xyz.device,
            )
            center = torch.as_tensor(
                calibration["source_table_center"],
                dtype=xyz.dtype,
                device=xyz.device,
            )
            canonical_xyz = (
                (xyz - center.unsqueeze(0)) @ rotation.T
                * float(calibration["meters_per_source_unit"])
            )
        else:
            canonical_xyz = xyz
        selected = torch.as_tensor(
            centerpiece_removal_mask(canonical_xyz.detach().cpu().numpy(), calibration),
            dtype=torch.bool,
            device=xyz.device,
        )
        override_color = torch.full_like(model.get_xyz, 0.22)
        override_color[selected] = torch.tensor(
            [1.0, 0.02, 0.02], dtype=xyz.dtype, device=xyz.device
        )
        print(f"removal_selection={int(selected.sum().item())}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for camera_id in args.camera:
            entry = cameras[camera_id]
            viewpoint = build_camera(entry, args.width)
            result = render(
                viewpoint,
                model,
                pipeline,
                background,
                override_color=override_color,
                antialiased=False,
            )
            if args.report_largest > 0:
                radii = result["radii"].reshape(-1)
                means2d = result["viewspace_points"].reshape(-1, 2)
                visible = radii > 0
                central = (
                    visible
                    & (means2d[:, 0] >= 0.25 * viewpoint.image_width)
                    & (means2d[:, 0] <= 0.75 * viewpoint.image_width)
                    & (means2d[:, 1] >= 0.2 * viewpoint.image_height)
                    & (means2d[:, 1] <= 0.75 * viewpoint.image_height)
                )
                candidate_indices = torch.nonzero(central, as_tuple=False).reshape(-1)
                count = min(int(args.report_largest), int(candidate_indices.numel()))
                selected_indices = candidate_indices[
                    torch.topk(radii[candidate_indices], k=count).indices
                ]
                for gaussian_index in selected_indices.detach().cpu().tolist():
                    print(
                        "largest "
                        f"index={gaussian_index} radius_px={float(radii[gaussian_index]):.3f} "
                        f"mean2d={means2d[gaussian_index].detach().cpu().tolist()} "
                        f"xyz={model.get_xyz[gaussian_index].detach().cpu().tolist()} "
                        f"scale={model.get_scaling[gaussian_index].detach().cpu().tolist()} "
                        f"opacity={float(model.get_opacity[gaussian_index]):.5f}"
                    )
            rgb = (
                result["render"][:3]
                .clamp(0.0, 1.0)
                .mul(255.0)
                .byte()
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )
            output = args.output_dir / (
                f"{args.quality}_{args.patch_view}_camera_"
                f"{camera_id:03d}_{entry['img_name']}.png"
            )
            Image.fromarray(rgb, mode="RGB").save(output)
            print(output)
            for raw_sample in args.sample:
                u, v = [int(value) for value in raw_sample.split(",", maxsplit=1)]
                depth = float(result["depth"][v, u].item())
                source_width = int(entry["width"])
                source_height = int(entry["height"])
                scale_x = viewpoint.image_width / source_width
                scale_y = viewpoint.image_height / source_height
                fx = float(entry["fx"]) * scale_x
                fy = float(entry["fy"]) * scale_y
                camera_point = np.array(
                    [
                        (u - 0.5 * viewpoint.image_width) * depth / fx,
                        (v - 0.5 * viewpoint.image_height) * depth / fy,
                        depth,
                    ],
                    dtype=np.float32,
                )
                world_point = (
                    np.asarray(entry["position"], dtype=np.float32)
                    + np.asarray(entry["rotation"], dtype=np.float32) @ camera_point
                )
                print(
                    f"camera={camera_id} pixel=({u},{v}) depth={depth:.6f} "
                    f"source_world={world_point.tolist()}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
