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
        metrics_mode: tl.constexpr,
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

        if metrics_mode > 0:
            tl.atomic_add(
                counters_ptr + 0,
                tl.sum(raw_present.to(tl.int32), axis=0),
            )
            tl.atomic_add(
                counters_ptr + 1,
                tl.sum(visible.to(tl.int32), axis=0),
            )
        if metrics_mode >= 2:
            tl.atomic_add(
                counters_ptr + 2,
                tl.sum(occluded.to(tl.int32), axis=0),
            )
            tl.atomic_add(
                counters_ptr + 3,
                tl.sum(scene_depth_invalid.to(tl.int32), axis=0),
            )

    @triton.jit
    def _blend_gaussian_scene_roi_kernel(
        scene_color_ptr,
        scene_depth_ptr,
        gaussian_rgba_ptr,
        gaussian_depth_ptr,
        output_color_ptr,
        output_depth_ptr,
        counters_ptr,
        width,
        height,
        x0,
        y0,
        roi_width,
        roi_height,
        metrics_mode: tl.constexpr,
        alpha_eps: tl.constexpr,
        depth_eps: tl.constexpr,
        occlusion_eps: tl.constexpr,
        write_depth: tl.constexpr,
        block_size: tl.constexpr,
    ):
        roi_offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
        roi_pixel_count = roi_width * roi_height
        valid = roi_offsets < roi_pixel_count

        local_x = roi_offsets % roi_width
        local_y = roi_offsets // roi_width
        global_x = x0 + local_x
        global_y = y0 + local_y
        global_offsets = global_y * width + global_x
        full_pixel_count = width * height
        color_base = global_offsets * 4

        scene_r = tl.load(scene_color_ptr + color_base + 0, mask=valid, other=0).to(tl.float32)
        scene_g = tl.load(scene_color_ptr + color_base + 1, mask=valid, other=0).to(tl.float32)
        scene_b = tl.load(scene_color_ptr + color_base + 2, mask=valid, other=0).to(tl.float32)
        scene_depth = tl.load(scene_depth_ptr + global_offsets, mask=valid, other=0.0)
        gaussian_depth = tl.load(gaussian_depth_ptr + global_offsets, mask=valid, other=0.0)

        gaussian_r = tl.load(
            gaussian_rgba_ptr + global_offsets,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        gaussian_g = tl.load(
            gaussian_rgba_ptr + full_pixel_count + global_offsets,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        gaussian_b = tl.load(
            gaussian_rgba_ptr + full_pixel_count * 2 + global_offsets,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        gaussian_alpha = tl.load(
            gaussian_rgba_ptr + full_pixel_count * 3 + global_offsets,
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

        out_r = scene_r * (1.0 - gaussian_alpha) + gaussian_r * 255.0 * gaussian_alpha
        out_g = scene_g * (1.0 - gaussian_alpha) + gaussian_g * 255.0 * gaussian_alpha
        out_b = scene_b * (1.0 - gaussian_alpha) + gaussian_b * 255.0 * gaussian_alpha
        out_r = tl.minimum(tl.maximum(out_r + 0.5, 0.0), 255.0)
        out_g = tl.minimum(tl.maximum(out_g + 0.5, 0.0), 255.0)
        out_b = tl.minimum(tl.maximum(out_b + 0.5, 0.0), 255.0)

        tl.store(output_color_ptr + color_base + 0, out_r, mask=visible)
        tl.store(output_color_ptr + color_base + 1, out_g, mask=visible)
        tl.store(output_color_ptr + color_base + 2, out_b, mask=visible)
        tl.store(output_color_ptr + color_base + 3, 255, mask=visible)
        if write_depth:
            tl.store(output_depth_ptr + global_offsets, gaussian_depth, mask=visible)

        if metrics_mode > 0:
            tl.atomic_add(
                counters_ptr + 0,
                tl.sum(raw_present.to(tl.int32), axis=0),
            )
            tl.atomic_add(
                counters_ptr + 1,
                tl.sum(visible.to(tl.int32), axis=0),
            )
        if metrics_mode >= 2:
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


def _resolve_metrics_mode(
    collect_metrics: bool,
    profile_metrics: bool,
    metrics_mode: Optional[str],
) -> tuple[str, int, bool]:
    requested_metrics_mode = (
        str(metrics_mode).strip().lower() if metrics_mode is not None else ""
    )
    if not requested_metrics_mode:
        if bool(collect_metrics):
            requested_metrics_mode = "profile" if bool(profile_metrics) else "safety"
        else:
            requested_metrics_mode = "none"
    if requested_metrics_mode not in {
        "none",
        "safety",
        "safety_async",
        "profile",
    }:
        raise ValueError(f"Invalid fusion metrics_mode: {requested_metrics_mode!r}")
    kernel_metrics_mode = 0
    sync_metrics = False
    if requested_metrics_mode == "profile":
        kernel_metrics_mode = 2
        sync_metrics = True
    elif requested_metrics_mode == "safety":
        kernel_metrics_mode = 1
        sync_metrics = True
    elif requested_metrics_mode == "safety_async":
        kernel_metrics_mode = 1
        sync_metrics = False
    return requested_metrics_mode, kernel_metrics_mode, sync_metrics


def _prepare_counter_tensor(
    counters: Optional[torch.Tensor],
    *,
    required_counter_count: int,
    device: torch.device,
) -> torch.Tensor:
    if (
        counters is None
        or not torch.is_tensor(counters)
        or not counters.is_cuda
        or counters.dtype != torch.int32
        or int(counters.numel()) < int(required_counter_count)
        or counters.device != device
    ):
        return torch.zeros(
            (int(required_counter_count),),
            dtype=torch.int32,
            device=device,
        )
    counters_t = counters.reshape(-1)[: int(required_counter_count)]
    counters_t.zero_()
    return counters_t


def _collect_counter_metrics(
    counters_t: torch.Tensor,
    *,
    kernel_metrics_mode: int,
    sync_metrics: bool,
    pixel_count: int,
) -> Dict[str, float]:
    if int(kernel_metrics_mode) <= 0:
        return {}
    if not bool(sync_metrics):
        return {
            "fusion_metrics_pending": 1.0,
            "fusion_total_pixel_count": float(pixel_count),
        }
    counter_values = [
        int(value) for value in counters_t.detach().cpu().tolist()
    ]
    raw_count = int(counter_values[0])
    visible_count = int(counter_values[1])
    denom = max(float(pixel_count), 1.0)
    raw_ratio = float(raw_count) / denom
    visible_ratio = float(visible_count) / denom
    metrics = {
        "fusion_raw_ratio": raw_ratio,
        "fusion_visible_ratio": visible_ratio,
        "fusion_retention_ratio": (
            visible_ratio / max(raw_ratio, 1e-6) if raw_ratio > 0.0 else 1.0
        ),
        "fusion_raw_pixel_count": float(raw_count),
        "fusion_visible_pixel_count": float(visible_count),
        "fusion_total_pixel_count": float(pixel_count),
    }
    if int(kernel_metrics_mode) >= 2:
        occluded_count = int(counter_values[2])
        invalid_count = int(counter_values[3])
        metrics.update(
            {
                "fusion_occluded_ratio": float(occluded_count) / denom,
                "fusion_depth_invalid_ratio": float(invalid_count) / denom,
                "fusion_occluded_pixel_count": float(occluded_count),
                "fusion_depth_invalid_pixel_count": float(invalid_count),
            }
        )
    return metrics


def fuse_gaussian_scene_depth_aware(
    scene_color: torch.Tensor,
    scene_depth: torch.Tensor,
    gaussian_rgba: torch.Tensor,
    gaussian_depth: torch.Tensor,
    *,
    output_color: Optional[torch.Tensor] = None,
    output_depth: Optional[torch.Tensor] = None,
    alpha_eps: float,
    depth_eps: float,
    occlusion_eps: float,
    collect_metrics: bool = True,
    profile_metrics: bool = True,
    metrics_mode: Optional[str] = None,
    counters: Optional[torch.Tensor] = None,
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
    if output_color is None:
        output_color_t = torch.empty(
            (height, width, 4),
            dtype=torch.uint8,
            device=scene_color_t.device,
        )
    else:
        if (
            not torch.is_tensor(output_color)
            or not output_color.is_cuda
            or output_color.dtype != torch.uint8
            or tuple(output_color.shape) != (height, width, 4)
            or output_color.device != scene_color_t.device
        ):
            return None
        output_color_t = output_color.contiguous()
    if output_depth is None:
        output_depth_t = torch.empty(
            (height, width),
            dtype=torch.float32,
            device=scene_color_t.device,
        )
    else:
        if (
            not torch.is_tensor(output_depth)
            or not output_depth.is_cuda
            or output_depth.dtype != torch.float32
            or tuple(output_depth.shape[-2:]) != (height, width)
            or output_depth.device != scene_color_t.device
        ):
            return None
        output_depth_t = output_depth.contiguous()
    requested_metrics_mode = (
        str(metrics_mode).strip().lower() if metrics_mode is not None else ""
    )
    if not requested_metrics_mode:
        if bool(collect_metrics):
            requested_metrics_mode = "profile" if bool(profile_metrics) else "safety"
        else:
            requested_metrics_mode = "none"
    if requested_metrics_mode not in {
        "none",
        "safety",
        "safety_async",
        "profile",
    }:
        return None
    kernel_metrics_mode = 0
    sync_metrics = False
    if requested_metrics_mode == "profile":
        kernel_metrics_mode = 2
        sync_metrics = True
    elif requested_metrics_mode == "safety":
        kernel_metrics_mode = 1
        sync_metrics = True
    elif requested_metrics_mode == "safety_async":
        kernel_metrics_mode = 1
        sync_metrics = False

    required_counter_count = 4 if kernel_metrics_mode >= 2 else 2 if kernel_metrics_mode == 1 else 1
    if (
        counters is None
        or not torch.is_tensor(counters)
        or not counters.is_cuda
        or counters.dtype != torch.int32
        or int(counters.numel()) < required_counter_count
        or counters.device != scene_color_t.device
    ):
        counters_t = torch.zeros(
            (required_counter_count,),
            dtype=torch.int32,
            device=scene_color_t.device,
        )
    else:
        counters_t = counters.reshape(-1)[:required_counter_count]
        counters_t.zero_()

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
        output_color_t,
        output_depth_t,
        counters_t,
        int(pixel_count),
        int(kernel_metrics_mode),
        float(alpha_eps),
        float(depth_eps),
        float(occlusion_eps),
        block_size,
        num_warps=4,
    )

    metrics: Dict[str, float] = {}
    if kernel_metrics_mode > 0 and sync_metrics:
        counter_values = [
            int(value) for value in counters_t.detach().cpu().tolist()
        ]
        raw_count = int(counter_values[0])
        visible_count = int(counter_values[1])
        denom = max(float(pixel_count), 1.0)
        raw_ratio = float(raw_count) / denom
        visible_ratio = float(visible_count) / denom
        metrics = {
            "fusion_raw_ratio": raw_ratio,
            "fusion_visible_ratio": visible_ratio,
            "fusion_retention_ratio": (
                visible_ratio / max(raw_ratio, 1e-6) if raw_ratio > 0.0 else 1.0
            ),
            "fusion_raw_pixel_count": float(raw_count),
            "fusion_visible_pixel_count": float(visible_count),
            "fusion_total_pixel_count": float(pixel_count),
        }
        if kernel_metrics_mode >= 2:
            occluded_count = int(counter_values[2])
            invalid_count = int(counter_values[3])
            metrics.update(
                {
                    "fusion_occluded_ratio": float(occluded_count) / denom,
                    "fusion_depth_invalid_ratio": float(invalid_count) / denom,
                    "fusion_occluded_pixel_count": float(occluded_count),
                    "fusion_depth_invalid_pixel_count": float(invalid_count),
                }
            )
    elif kernel_metrics_mode > 0:
        metrics = {
            "fusion_metrics_pending": 1.0,
            "fusion_total_pixel_count": float(pixel_count),
        }
    return output_color_t, output_depth_t, metrics


def fuse_gaussian_scene_depth_aware_roi(
    scene_color: torch.Tensor,
    scene_depth: torch.Tensor,
    gaussian_rgba: torch.Tensor,
    gaussian_depth: torch.Tensor,
    *,
    roi_bounds: Tuple[int, int, int, int],
    output_color: Optional[torch.Tensor] = None,
    output_depth: Optional[torch.Tensor] = None,
    alpha_eps: float,
    depth_eps: float,
    occlusion_eps: float,
    collect_metrics: bool = True,
    profile_metrics: bool = True,
    metrics_mode: Optional[str] = None,
    counters: Optional[torch.Tensor] = None,
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
    if scene_color.dtype != torch.uint8:
        return None
    if scene_depth.dtype != torch.float32 or gaussian_rgba.dtype != torch.float32:
        return None
    if gaussian_depth.dtype != torch.float32:
        return None
    try:
        x0, y0, x1, y1 = (int(value) for value in roi_bounds)
    except Exception:
        return None
    x0 = max(0, min(width, x0))
    x1 = max(0, min(width, x1))
    y0 = max(0, min(height, y0))
    y1 = max(0, min(height, y1))
    if x1 <= x0 or y1 <= y0:
        return None

    scene_color_t = scene_color.contiguous()
    scene_depth_t = scene_depth.contiguous()
    gaussian_rgba_t = gaussian_rgba.contiguous()
    gaussian_depth_t = gaussian_depth.contiguous()
    if output_color is None:
        output_color_t = scene_color_t.clone()
    else:
        if (
            not torch.is_tensor(output_color)
            or not output_color.is_cuda
            or output_color.dtype != torch.uint8
            or tuple(output_color.shape[:2]) != (height, width)
            or int(output_color.shape[-1]) < 4
            or output_color.device != scene_color_t.device
        ):
            return None
        output_color_t = output_color.contiguous()
    if output_depth is None:
        output_depth_t = scene_depth_t.clone()
    else:
        if (
            not torch.is_tensor(output_depth)
            or not output_depth.is_cuda
            or output_depth.dtype != torch.float32
            or tuple(output_depth.shape[-2:]) != (height, width)
            or output_depth.device != scene_color_t.device
        ):
            return None
        output_depth_t = output_depth.contiguous()

    try:
        _, kernel_metrics_mode, sync_metrics = _resolve_metrics_mode(
            bool(collect_metrics),
            bool(profile_metrics),
            metrics_mode,
        )
    except Exception:
        return None
    required_counter_count = 4 if kernel_metrics_mode >= 2 else 2 if kernel_metrics_mode == 1 else 1
    counters_t = _prepare_counter_tensor(
        counters,
        required_counter_count=required_counter_count,
        device=scene_color_t.device,
    )

    roi_width = int(x1 - x0)
    roi_height = int(y1 - y0)
    roi_pixel_count = int(roi_width * roi_height)
    full_pixel_count = int(height * width)
    block_size = int(block_size)
    if block_size <= 0:
        block_size = 256
    grid = (int(math.ceil(roi_pixel_count / float(block_size))),)
    _blend_gaussian_scene_roi_kernel[grid](
        scene_color_t,
        scene_depth_t,
        gaussian_rgba_t,
        gaussian_depth_t,
        output_color_t,
        output_depth_t,
        counters_t,
        int(width),
        int(height),
        int(x0),
        int(y0),
        int(roi_width),
        int(roi_height),
        int(kernel_metrics_mode),
        float(alpha_eps),
        float(depth_eps),
        float(occlusion_eps),
        True,
        block_size,
        num_warps=4,
    )

    metrics = _collect_counter_metrics(
        counters_t,
        kernel_metrics_mode=int(kernel_metrics_mode),
        sync_metrics=bool(sync_metrics),
        pixel_count=int(full_pixel_count),
    )
    metrics.update(
        {
            "fusion_roi_ratio": float(roi_pixel_count)
            / max(float(full_pixel_count), 1.0),
            "fusion_roi_pixel_count": float(roi_pixel_count),
            "fusion_roi_x0": float(x0),
            "fusion_roi_y0": float(y0),
            "fusion_roi_x1": float(x1),
            "fusion_roi_y1": float(y1),
        }
    )
    return output_color_t, output_depth_t, metrics
