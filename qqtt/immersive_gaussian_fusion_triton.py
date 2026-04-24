from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - optional runtime dependency
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:

    @triton.jit
    def _fuse_gaussian_scene_kernel(
        scene_color_ptr,
        scene_depth_ptr,
        gaussian_rgba_ptr,
        gaussian_depth_ptr,
        output_color_ptr,
        output_depth_ptr,
        counters_ptr,
        pixel_count: tl.constexpr,
        collect_metrics: tl.constexpr,
        alpha_eps: tl.constexpr,
        depth_eps: tl.constexpr,
        occlusion_eps: tl.constexpr,
        block_size: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
        valid = offsets < pixel_count

        color_base = offsets * 4
        scene_r = tl.load(scene_color_ptr + color_base + 0, mask=valid, other=0).to(tl.float32)
        scene_g = tl.load(scene_color_ptr + color_base + 1, mask=valid, other=0).to(tl.float32)
        scene_b = tl.load(scene_color_ptr + color_base + 2, mask=valid, other=0).to(tl.float32)
        scene_depth = tl.load(scene_depth_ptr + offsets, mask=valid, other=0.0)
        gaussian_depth = tl.load(gaussian_depth_ptr + offsets, mask=valid, other=0.0)

        gaussian_r = tl.load(
            gaussian_rgba_ptr + offsets,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        gaussian_g = tl.load(
            gaussian_rgba_ptr + pixel_count + offsets,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        gaussian_b = tl.load(
            gaussian_rgba_ptr + pixel_count * 2 + offsets,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        gaussian_alpha = tl.load(
            gaussian_rgba_ptr + pixel_count * 3 + offsets,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        gaussian_alpha = tl.minimum(tl.maximum(gaussian_alpha, 0.0), 1.0)

        finite_scene_depth = (
            (scene_depth == scene_depth)
            & (tl.abs(scene_depth) < 3.4028234663852886e38)
        )
        finite_gaussian_depth = (
            (gaussian_depth == gaussian_depth)
            & (tl.abs(gaussian_depth) < 3.4028234663852886e38)
        )
        scene_has_depth = valid & finite_scene_depth & (scene_depth > depth_eps)
        gaussian_has_depth = valid & finite_gaussian_depth & (gaussian_depth > depth_eps)
        raw_present = valid & (gaussian_alpha > alpha_eps)
        depth_visible = (~scene_has_depth) | (
            gaussian_depth <= (scene_depth + occlusion_eps)
        )
        visible = raw_present & gaussian_has_depth & depth_visible
        occluded = raw_present & gaussian_has_depth & scene_has_depth & (~depth_visible)
        scene_depth_invalid = valid & (~scene_has_depth)

        out_r = tl.where(
            visible,
            scene_r * (1.0 - gaussian_alpha) + gaussian_r * 255.0 * gaussian_alpha,
            scene_r,
        )
        out_g = tl.where(
            visible,
            scene_g * (1.0 - gaussian_alpha) + gaussian_g * 255.0 * gaussian_alpha,
            scene_g,
        )
        out_b = tl.where(
            visible,
            scene_b * (1.0 - gaussian_alpha) + gaussian_b * 255.0 * gaussian_alpha,
            scene_b,
        )
        out_r = tl.minimum(tl.maximum(out_r + 0.5, 0.0), 255.0)
        out_g = tl.minimum(tl.maximum(out_g + 0.5, 0.0), 255.0)
        out_b = tl.minimum(tl.maximum(out_b + 0.5, 0.0), 255.0)

        tl.store(output_color_ptr + color_base + 0, out_r, mask=valid)
        tl.store(output_color_ptr + color_base + 1, out_g, mask=valid)
        tl.store(output_color_ptr + color_base + 2, out_b, mask=valid)
        tl.store(output_color_ptr + color_base + 3, 255, mask=valid)
        tl.store(
            output_depth_ptr + offsets,
            tl.where(visible, gaussian_depth, scene_depth),
            mask=valid,
        )

        if collect_metrics:
            tl.atomic_add(
                counters_ptr + 0,
                tl.sum(raw_present.to(tl.int32), axis=0),
            )
            tl.atomic_add(
                counters_ptr + 1,
                tl.sum(visible.to(tl.int32), axis=0),
            )
            tl.atomic_add(
                counters_ptr + 2,
                tl.sum(occluded.to(tl.int32), axis=0),
            )
            tl.atomic_add(
                counters_ptr + 3,
                tl.sum(scene_depth_invalid.to(tl.int32), axis=0),
            )


def triton_gaussian_fusion_available() -> bool:
    return bool(_TRITON_AVAILABLE)


def fuse_gaussian_scene_depth_aware(
    scene_color: torch.Tensor,
    scene_depth: torch.Tensor,
    gaussian_rgba: torch.Tensor,
    gaussian_depth: torch.Tensor,
    *,
    alpha_eps: float,
    depth_eps: float,
    occlusion_eps: float,
    collect_metrics: bool = True,
    block_size: int = 256,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]]:
    if not _TRITON_AVAILABLE:
        return None
    if (
        not torch.is_tensor(scene_color)
        or not torch.is_tensor(scene_depth)
        or not torch.is_tensor(gaussian_rgba)
        or not torch.is_tensor(gaussian_depth)
    ):
        return None
    if (
        not scene_color.is_cuda
        or not scene_depth.is_cuda
        or not gaussian_rgba.is_cuda
        or not gaussian_depth.is_cuda
    ):
        return None
    if scene_color.ndim != 3 or int(scene_color.shape[-1]) < 4:
        return None
    if gaussian_rgba.ndim != 3 or int(gaussian_rgba.shape[0]) < 4:
        return None
    height = int(scene_color.shape[0])
    width = int(scene_color.shape[1])
    if height <= 0 or width <= 0:
        return None
    if tuple(scene_depth.shape[-2:]) != (height, width):
        return None
    if tuple(gaussian_rgba.shape[-2:]) != (height, width):
        return None
    if tuple(gaussian_depth.shape[-2:]) != (height, width):
        return None
    if scene_color.dtype not in (torch.uint8, torch.float32):
        return None
    if scene_depth.dtype != torch.float32 or gaussian_rgba.dtype != torch.float32:
        return None
    if gaussian_depth.dtype != torch.float32:
        return None

    scene_color_t = scene_color.contiguous()
    scene_depth_t = scene_depth.contiguous()
    gaussian_rgba_t = gaussian_rgba.contiguous()
    gaussian_depth_t = gaussian_depth.contiguous()
    output_color = torch.empty(
        (height, width, 4),
        dtype=torch.uint8,
        device=scene_color_t.device,
    )
    output_depth = torch.empty(
        (height, width),
        dtype=torch.float32,
        device=scene_color_t.device,
    )
    counters = torch.zeros(
        (4,),
        dtype=torch.int32,
        device=scene_color_t.device,
    )

    pixel_count = int(height * width)
    block_size = int(block_size)
    if block_size <= 0:
        block_size = 256
    grid = (int(math.ceil(pixel_count / float(block_size))),)
    _fuse_gaussian_scene_kernel[grid](
        scene_color_t,
        scene_depth_t,
        gaussian_rgba_t,
        gaussian_depth_t,
        output_color,
        output_depth,
        counters,
        int(pixel_count),
        bool(collect_metrics),
        float(alpha_eps),
        float(depth_eps),
        float(occlusion_eps),
        block_size,
        num_warps=4,
    )

    metrics: Dict[str, float] = {}
    if collect_metrics:
        raw_count, visible_count, occluded_count, invalid_count = [
            int(value) for value in counters.detach().cpu().tolist()
        ]
        denom = max(float(pixel_count), 1.0)
        raw_ratio = float(raw_count) / denom
        visible_ratio = float(visible_count) / denom
        metrics = {
            "fusion_raw_ratio": raw_ratio,
            "fusion_visible_ratio": visible_ratio,
            "fusion_occluded_ratio": float(occluded_count) / denom,
            "fusion_depth_invalid_ratio": float(invalid_count) / denom,
            "fusion_retention_ratio": (
                visible_ratio / max(raw_ratio, 1e-6) if raw_ratio > 0.0 else 1.0
            ),
            "fusion_raw_pixel_count": float(raw_count),
            "fusion_visible_pixel_count": float(visible_count),
            "fusion_occluded_pixel_count": float(occluded_count),
            "fusion_depth_invalid_pixel_count": float(invalid_count),
            "fusion_total_pixel_count": float(pixel_count),
        }
    return output_color, output_depth, metrics
