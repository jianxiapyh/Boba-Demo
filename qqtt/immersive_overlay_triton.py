from __future__ import annotations

import math
from typing import Sequence

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - optional runtime dependency
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


COMMAND_LINE = 0
COMMAND_FILL_RECT = 1


if _TRITON_AVAILABLE:

    @triton.jit
    def _draw_overlay_commands_kernel(
        frame_ptr,
        command_ptr,
        width: tl.constexpr,
        height: tl.constexpr,
        max_blocks_per_command: tl.constexpr,
        block_size: tl.constexpr,
    ):
        command_index = tl.program_id(0)
        block_index = tl.program_id(1)
        command_base = command_index * 14
        command_type = tl.load(command_ptr + command_base + 0).to(tl.int32)
        x0 = tl.load(command_ptr + command_base + 1)
        y0 = tl.load(command_ptr + command_base + 2)
        x1 = tl.load(command_ptr + command_base + 3)
        y1 = tl.load(command_ptr + command_base + 4)
        radius = tl.load(command_ptr + command_base + 5)
        blend = tl.load(command_ptr + command_base + 6)
        color_r = tl.load(command_ptr + command_base + 7)
        color_g = tl.load(command_ptr + command_base + 8)
        color_b = tl.load(command_ptr + command_base + 9)
        bbox_x0 = tl.load(command_ptr + command_base + 10).to(tl.int32)
        bbox_y0 = tl.load(command_ptr + command_base + 11).to(tl.int32)
        bbox_w = tl.load(command_ptr + command_base + 12).to(tl.int32)
        bbox_h = tl.load(command_ptr + command_base + 13).to(tl.int32)

        area = bbox_w * bbox_h
        offsets = block_index * block_size + tl.arange(0, block_size)
        active = offsets < area
        local_x = offsets % bbox_w
        local_y = offsets // bbox_w
        px_i = bbox_x0 + local_x
        py_i = bbox_y0 + local_y
        px = px_i.to(tl.float32)
        py = py_i.to(tl.float32)

        seg_x = x1 - x0
        seg_y = y1 - y0
        seg_len_sq = seg_x * seg_x + seg_y * seg_y
        t = ((px - x0) * seg_x + (py - y0) * seg_y) / tl.maximum(seg_len_sq, 1.0e-6)
        t = tl.minimum(tl.maximum(t, 0.0), 1.0)
        closest_x = x0 + t * seg_x
        closest_y = y0 + t * seg_y
        dist_x = tl.abs(px - closest_x)
        dist_y = tl.abs(py - closest_y)
        line_mask = tl.maximum(dist_x, dist_y) <= (radius + 0.5)
        is_line = command_type == 0
        draw_mask = active & ((~is_line) | line_mask)

        pixel_base = ((py_i * width + px_i) * 4).to(tl.int64)
        old_r = tl.load(frame_ptr + pixel_base + 0, mask=draw_mask, other=0).to(tl.float32)
        old_g = tl.load(frame_ptr + pixel_base + 1, mask=draw_mask, other=0).to(tl.float32)
        old_b = tl.load(frame_ptr + pixel_base + 2, mask=draw_mask, other=0).to(tl.float32)
        keep = 1.0 - blend
        new_r = old_r * keep + color_r * blend
        new_g = old_g * keep + color_g * blend
        new_b = old_b * keep + color_b * blend
        tl.store(frame_ptr + pixel_base + 0, new_r, mask=draw_mask)
        tl.store(frame_ptr + pixel_base + 1, new_g, mask=draw_mask)
        tl.store(frame_ptr + pixel_base + 2, new_b, mask=draw_mask)


def triton_overlay_available() -> bool:
    return bool(_TRITON_AVAILABLE)


def draw_overlay_commands(
    frame: torch.Tensor,
    commands: Sequence[Sequence[float]],
    *,
    block_size: int = 256,
) -> bool:
    if not _TRITON_AVAILABLE or not commands:
        return False
    if not torch.is_tensor(frame) or not frame.is_cuda or frame.ndim != 3:
        return False
    if int(frame.shape[-1]) < 3 or frame.dtype not in (torch.uint8, torch.float32):
        return False
    if not frame.is_contiguous():
        return False
    command_tensor = torch.as_tensor(
        commands,
        dtype=torch.float32,
        device=frame.device,
    )
    if command_tensor.ndim != 2 or int(command_tensor.shape[1]) != 14:
        return False
    areas = command_tensor[:, 12] * command_tensor[:, 13]
    max_area = int(torch.max(areas).item()) if int(areas.numel()) > 0 else 0
    if max_area <= 0:
        return False
    max_blocks = max(1, int(math.ceil(max_area / float(block_size))))
    _draw_overlay_commands_kernel[(int(command_tensor.shape[0]), max_blocks)](
        frame,
        command_tensor,
        int(frame.shape[1]),
        int(frame.shape[0]),
        max_blocks,
        int(block_size),
        num_warps=4,
    )
    return True
