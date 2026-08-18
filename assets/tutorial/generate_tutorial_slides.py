#!/usr/bin/env python3
"""Generate deterministic startup tutorial slides for the immersive demos."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[2]
TUTORIAL_DIR = REPO_ROOT / "assets" / "tutorial"
TUTORIAL_SOURCE_DIR = TUTORIAL_DIR / "source"
ROPE_GAME_DIR = REPO_ROOT / "assets" / "rope_game"

SLIDE_WIDTH = 1920
SLIDE_HEIGHT = 1080

PAGE_BG = (12, 17, 31, 255)
SURFACE_BG = (20, 27, 44, 255)
SURFACE_BORDER = (43, 58, 92, 255)
TITLE_COLOR = (244, 248, 255, 255)
BODY_COLOR = (224, 232, 246, 255)
MUTED_COLOR = (171, 186, 212, 255)
PANEL_BG = (21, 52, 106, 255)
PANEL_BORDER = (116, 163, 242, 255)
CARD_BG = (13, 24, 48, 255)
CARD_BORDER = (70, 106, 171, 255)
BUTTON_BG = (64, 146, 255, 255)
BUTTON_BORDER = (146, 200, 255, 255)
ACCENT_BLUE = (88, 172, 255, 255)
ACCENT_RED = (255, 92, 96, 255)
ACCENT_GREEN = (74, 222, 128, 255)
ACCENT_YELLOW = (255, 208, 74, 255)
ACCENT_MAGENTA = (228, 108, 255, 255)
WHITE = (255, 255, 255, 255)
BLACK = (0, 0, 0, 255)

OUTER_MARGIN = 60
SURFACE_BOTTOM_INSET = 12
TOP_BAND_HEIGHT = 110
FOOTER_HEIGHT = 90
FOOTER_SIDE_INSET = 24
FOOTER_BOTTOM_INSET = 24
CONTENT_TOP = 160
CONTENT_FOOTER_GAP = 18
CONTENT_BOTTOM = SLIDE_HEIGHT - FOOTER_BOTTOM_INSET - FOOTER_HEIGHT - CONTENT_FOOTER_GAP
CONTENT_HEIGHT = CONTENT_BOTTOM - CONTENT_TOP
PANEL_GAP = 34
PANEL_TITLE_INSET_X = 28
PANEL_HEADER_TOP_PAD = 24
PANEL_HEADER_MIN_HEIGHT = 92
PANEL_BODY_PAD_X = 36
PANEL_BODY_PAD_TOP = 128
PANEL_BODY_PAD_BOTTOM = 36
CARD_INSET = 20
CARD_TITLE_TOP_PAD = 18
CARD_BODY_TOP = 70
CARD_BOTTOM_PAD = 20
CARD_GAP = 24
TITLE_OFFSET_X = 28
TITLE_OFFSET_Y = 10
TOUCH_PLUS_WHITE_REFERENCE_PATH = TUTORIAL_SOURCE_DIR / "meta_quest_touch_plus_pair_white.png"
CALLOUT_LINE_COLOR = (15, 26, 54, 255)
HEADER_SEPARATOR_COLOR = (117, 173, 255, 255)
CARD_TITLE_INSET_X = CARD_INSET
CARD_HEADER_TOP_PAD = 18
CARD_HEADER_MIN_HEIGHT = 54
CARD_TITLE_LINE_GAP = 2
STEP_VERTICAL_GAP = 10
SLIDE_TWO_STEP_VERTICAL_GAP = 46
SLIDE_THREE_STEP_VERTICAL_GAP = 28
SLIDE_THREE_ANCHOR_T = 0.93


def load_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_names = []
    if bold:
        font_names.extend(
            [
                "DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            ]
        )
    else:
        font_names.extend(
            [
                "DejaVuSans.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            ]
        )
    for font_name in font_names:
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


TITLE_FONT = load_font(74, bold=True)
PANEL_TITLE_FONT = load_font(40, bold=True)
PANEL_TITLE_COMPACT_FONT = load_font(34, bold=True)
SUBTITLE_FONT = load_font(30, bold=True)
BODY_FONT = load_font(28)
BODY_BOLD_FONT = load_font(28, bold=True)
SMALL_FONT = load_font(24)
SMALL_BOLD_FONT = load_font(24, bold=True)
BUTTON_FONT = load_font(30, bold=True)
FOOTER_FONT = load_font(30, bold=True)
STEP_NUMBER_FONT = load_font(34, bold=True)


def rgba(color):
    if len(color) == 4:
        return color
    return tuple(color) + (255,)


def text_size(draw: ImageDraw.ImageDraw, text: str, font) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])


def wrap_lines(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> list[str]:
    lines: list[str] = []
    for paragraph in text.split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            lines.append("")
            continue
        words = paragraph.split()
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if text_size(draw, candidate, font)[0] <= max_width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines


def draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    *,
    font,
    fill,
    max_width: int,
    line_gap: int = 8,
) -> int:
    x, y = xy
    lines = wrap_lines(draw, text, font, max_width)
    _, line_height = text_size(draw, "Ag", font)
    current_y = y
    for line in lines:
        if line:
            draw.text((x, current_y), line, font=font, fill=fill)
        current_y += line_height + line_gap
    return current_y


def rounded_panel(draw: ImageDraw.ImageDraw, rect, *, fill, outline, radius=28, width=4):
    draw.rounded_rectangle(rect, radius=radius, fill=fill, outline=outline, width=width)


def draw_page_surface(draw: ImageDraw.ImageDraw):
    rounded_panel(
        draw,
        (OUTER_MARGIN, OUTER_MARGIN, SLIDE_WIDTH - OUTER_MARGIN, SLIDE_HEIGHT - SURFACE_BOTTOM_INSET),
        fill=SURFACE_BG,
        outline=SURFACE_BORDER,
        radius=34,
        width=4,
    )


def draw_title(draw: ImageDraw.ImageDraw, text: str):
    title_x = OUTER_MARGIN + TITLE_OFFSET_X
    max_width = SLIDE_WIDTH - title_x - OUTER_MARGIN
    title_font = TITLE_FONT
    if text_size(draw, text, title_font)[0] > max_width:
        for size in range(72, 47, -2):
            candidate_font = load_font(size, bold=True)
            if text_size(draw, text, candidate_font)[0] <= max_width:
                title_font = candidate_font
                break
    draw.text((title_x, OUTER_MARGIN + TITLE_OFFSET_Y), text, font=title_font, fill=TITLE_COLOR)


def draw_footer(draw: ImageDraw.ImageDraw, text: str):
    footer_w = SLIDE_WIDTH - OUTER_MARGIN * 2
    footer_h = FOOTER_HEIGHT
    y0 = SLIDE_HEIGHT - FOOTER_BOTTOM_INSET - footer_h
    rounded_panel(
        draw,
        (
            OUTER_MARGIN + FOOTER_SIDE_INSET,
            y0,
            OUTER_MARGIN + FOOTER_SIDE_INSET + footer_w - FOOTER_SIDE_INSET * 2,
            y0 + footer_h,
        ),
        fill=(17, 39, 78, 255),
        outline=(104, 151, 229, 255),
        radius=24,
        width=3,
    )
    text_w, text_h = text_size(draw, text, FOOTER_FONT)
    draw.text(
        ((SLIDE_WIDTH - text_w) // 2, y0 + (footer_h - text_h) // 2 - 2),
        text,
        font=FOOTER_FONT,
        fill=WHITE,
    )


def draw_button_chip(draw: ImageDraw.ImageDraw, rect, text: str):
    rounded_panel(draw, rect, fill=BUTTON_BG, outline=BUTTON_BORDER, radius=22, width=3)
    w, h = text_size(draw, text, BUTTON_FONT)
    x0, y0, x1, y1 = rect
    draw.text(
        (x0 + (x1 - x0 - w) // 2, y0 + (y1 - y0 - h) // 2 - 2),
        text,
        font=BUTTON_FONT,
        fill=WHITE,
    )


def draw_control_row(draw: ImageDraw.ImageDraw, *, x: int, y: int, button_text: str, description: str, width: int) -> int:
    chip_w = 260
    chip_h = 58
    draw_button_chip(draw, (x, y, x + chip_w, y + chip_h), button_text)
    desc_x = x + chip_w + 24
    desc_w = max(1, width - chip_w - 24)
    end_y = draw_wrapped_text(
        draw,
        (desc_x, y + 8),
        description,
        font=BODY_FONT,
        fill=BODY_COLOR,
        max_width=desc_w,
        line_gap=6,
    )
    return max(y + chip_h, end_y) + 22


def draw_panel_header(
    draw: ImageDraw.ImageDraw,
    rect,
    title: str,
    *,
    title_inset_x: int = PANEL_TITLE_INSET_X,
    header_top_pad: int = PANEL_HEADER_TOP_PAD,
    header_min_height: int = PANEL_HEADER_MIN_HEIGHT,
    title_font=None,
    compact_title_font=None,
    line_gap: int = 4,
    separator_color=HEADER_SEPARATOR_COLOR,
    separator_width: int = 3,
):
    x0, y0, x1, _ = rect
    title_max_width = x1 - x0 - title_inset_x * 2
    title_font = PANEL_TITLE_FONT if title_font is None else title_font
    compact_title_font = (
        PANEL_TITLE_COMPACT_FONT if compact_title_font is None else compact_title_font
    )
    title_lines = wrap_lines(draw, title, title_font, title_max_width)
    if len(title_lines) > 2:
        title_font = compact_title_font
        title_lines = wrap_lines(draw, title, title_font, title_max_width)
    line_x = x0 + title_inset_x
    line_y = y0 + header_top_pad
    _, line_height = text_size(draw, "Ag", title_font)
    for line in title_lines[:2]:
        draw.text((line_x, line_y), line, font=title_font, fill=WHITE)
        line_y += line_height + line_gap
    separator_y = max(y0 + header_min_height, line_y + 12)
    draw.line(
        (x0 + title_inset_x, separator_y, x1 - title_inset_x, separator_y),
        fill=separator_color,
        width=separator_width,
    )
    return separator_y


def draw_panel(
    draw: ImageDraw.ImageDraw,
    rect,
    title: str,
    *,
    title_inset_x: int = PANEL_TITLE_INSET_X,
    header_top_pad: int = PANEL_HEADER_TOP_PAD,
    header_min_height: int = PANEL_HEADER_MIN_HEIGHT,
):
    rounded_panel(draw, rect, fill=PANEL_BG, outline=PANEL_BORDER, radius=28, width=4)
    draw_panel_header(
        draw,
        rect,
        title,
        title_inset_x=title_inset_x,
        header_top_pad=header_top_pad,
        header_min_height=header_min_height,
    )


def draw_card(draw: ImageDraw.ImageDraw, rect, *, title: str | None = None):
    rounded_panel(draw, rect, fill=CARD_BG, outline=CARD_BORDER, radius=24, width=3)
    if title:
        draw_panel_header(
            draw,
            rect,
            title,
            title_inset_x=CARD_TITLE_INSET_X,
            header_top_pad=CARD_HEADER_TOP_PAD,
            header_min_height=CARD_HEADER_MIN_HEIGHT,
            title_font=SMALL_BOLD_FONT,
            compact_title_font=SMALL_BOLD_FONT,
            line_gap=CARD_TITLE_LINE_GAP,
            separator_color=(92, 126, 186, 255),
            separator_width=2,
        )


def draw_square_marker(draw: ImageDraw.ImageDraw, center: tuple[float, float], *, color, size=20, width=4):
    cx, cy = center
    x0 = int(round(cx - size / 2))
    y0 = int(round(cy - size / 2))
    x1 = int(round(cx + size / 2))
    y1 = int(round(cy + size / 2))
    draw.rectangle((x0, y0, x1, y1), outline=color, width=width)
    draw.rectangle((cx - 3, cy - 3, cx + 3, cy + 3), fill=WHITE)


def draw_split_square(draw: ImageDraw.ImageDraw, rect):
    x0, y0, x1, y1 = rect
    draw.line((x0, y0, x1, y0), fill=ACCENT_RED, width=3)
    draw.line((x0, y0, x0, y1), fill=ACCENT_RED, width=3)
    draw.line((x0, y1, x1, y1), fill=ACCENT_BLUE, width=3)
    draw.line((x1, y0, x1, y1), fill=ACCENT_BLUE, width=3)
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    draw.rectangle((cx - 5, cy - 5, cx + 5, cy + 5), fill=WHITE)


def draw_callout_chip(
    draw: ImageDraw.ImageDraw,
    *,
    label: str,
    center: tuple[int, int],
    anchor: tuple[int, int],
    line_color=WHITE,
    dot_color=WHITE,
):
    text_w, text_h = text_size(draw, label, SMALL_BOLD_FONT)
    pad_x = 18
    pad_y = 10
    cx, cy = center
    rect = (
        int(cx - text_w / 2 - pad_x),
        int(cy - text_h / 2 - pad_y),
        int(cx + text_w / 2 + pad_x),
        int(cy + text_h / 2 + pad_y),
    )
    rounded_panel(
        draw,
        rect,
        fill=BUTTON_BG,
        outline=BUTTON_BORDER,
        radius=18,
        width=3,
    )
    draw.text(
        (
            rect[0] + (rect[2] - rect[0] - text_w) // 2,
            rect[1] + (rect[3] - rect[1] - text_h) // 2 - 1,
        ),
        label,
        font=SMALL_BOLD_FONT,
        fill=WHITE,
    )
    end_x = min(max(anchor[0], rect[0],), rect[2])
    end_y = min(max(anchor[1], rect[1],), rect[3])
    mid_x = int(round((anchor[0] + end_x) / 2))
    draw.line((anchor[0], anchor[1], mid_x, anchor[1], end_x, end_y), fill=line_color, width=4)
    draw.ellipse(
        (anchor[0] - 5, anchor[1] - 5, anchor[0] + 5, anchor[1] + 5),
        fill=dot_color,
    )


def _fit_rect(src_size: tuple[int, int], dst_rect: tuple[int, int, int, int]) -> tuple[tuple[int, int, int, int], float]:
    src_w, src_h = src_size
    dst_w = dst_rect[2] - dst_rect[0]
    dst_h = dst_rect[3] - dst_rect[1]
    scale = min(dst_w / float(src_w), dst_h / float(src_h))
    fit_w = max(1, int(round(src_w * scale)))
    fit_h = max(1, int(round(src_h * scale)))
    fit_x0 = dst_rect[0] + (dst_w - fit_w) // 2
    fit_y0 = dst_rect[1] + (dst_h - fit_h) // 2
    return (fit_x0, fit_y0, fit_x0 + fit_w, fit_y0 + fit_h), scale


def draw_touch_plus_controller_panel(image: Image.Image, draw: ImageDraw.ImageDraw, rect):
    x0, y0, x1, y1 = rect
    art_bounds = (
        x0 + 28,
        y0 + 18,
        x1 - 28,
        y1 - 18,
    )
    controller_art = Image.open(TOUCH_PLUS_WHITE_REFERENCE_PATH).convert("RGBA")
    fit_rect, fit_scale = _fit_rect(controller_art.size, art_bounds)
    art_image = controller_art.resize(
        (fit_rect[2] - fit_rect[0], fit_rect[3] - fit_rect[1]),
        Image.Resampling.LANCZOS,
    )
    image.alpha_composite(art_image, dest=(fit_rect[0], fit_rect[1]))

    def art_anchor(px: int, py: int) -> tuple[int, int]:
        return (
            fit_rect[0] + int(round(px * fit_scale)),
            fit_rect[1] + int(round(py * fit_scale)),
        )

    ix0, iy0, ix1, iy1 = rect
    callouts = [
        ("Y", (ix0 + 62, iy0 + 44), art_anchor(208, 84)),
        ("X", (ix0 + 56, iy0 + 120), art_anchor(168, 116)),
        ("Stick Press", (ix0 + 176, iy0 + 30), art_anchor(150, 42)),
        ("Select", (ix0 + 82, iy1 - 84), art_anchor(255, 178)),
        ("Grip (hold)", (ix0 + 158, iy1 - 26), art_anchor(170, 236)),
        ("B", (ix1 - 64, iy0 + 44), art_anchor(505, 83)),
        ("A", (ix1 - 58, iy0 + 120), art_anchor(539, 117)),
        ("Stick Press", (ix1 - 176, iy0 + 28), art_anchor(557, 42)),
        ("Select", (ix1 - 84, iy1 - 84), art_anchor(456, 178)),
        ("Grip (hold)", (ix1 - 158, iy1 - 26), art_anchor(541, 236)),
    ]
    for label, center, anchor in callouts:
        draw_callout_chip(
            draw,
            label=label,
            center=center,
            anchor=anchor,
            line_color=CALLOUT_LINE_COLOR,
            dot_color=CALLOUT_LINE_COLOR,
        )


def _rope_segment(rect, mode: str) -> tuple[tuple[int, int], tuple[int, int]]:
    x0, y0, x1, y1 = rect
    w = x1 - x0
    h = y1 - y0
    if mode == "assist":
        return (
            (x0 + int(0.10 * w), y0 + int(0.70 * h)),
            (x0 + int(0.92 * w), y0 + int(0.38 * h)),
        )
    if mode == "goal":
        return (
            (x0 + int(0.16 * w), y0 + int(0.52 * h)),
            (x0 + int(0.84 * w), y0 + int(0.52 * h)),
        )
    return (
        (x0 - int(0.12 * w), y0 + int(0.70 * h)),
        (x0 + int(0.74 * w), y0 + int(0.36 * h)),
    )


def _rope_point_at(rect, mode: str, t: float) -> tuple[int, int]:
    t = max(0.0, min(1.0, float(t)))
    (x0, y0), (x1, y1) = _rope_segment(rect, mode)
    px = int(round(float(x0) + (float(x1) - float(x0)) * t))
    py = int(round(float(y0) + (float(y1) - float(y0)) * t))
    return px, py


def _segment_point(start: tuple[int, int], end: tuple[int, int], t: float) -> tuple[int, int]:
    t = max(0.0, min(1.0, float(t)))
    return (
        int(round(start[0] + (end[0] - start[0]) * t)),
        int(round(start[1] + (end[1] - start[1]) * t)),
    )


def _slide_three_drag_poses(target_rect) -> dict[str, tuple[tuple[int, int], tuple[int, int]] | tuple[int, int]]:
    x0, y0, x1, y1 = target_rect
    w = x1 - x0
    h = y1 - y0
    rope_dx = int(round(0.79 * w))
    rope_dy = -int(round(0.27 * h))
    before_start = (
        x0 - int(round(0.05 * w)),
        y0 + int(round(0.68 * h)),
    )
    before_end = (
        before_start[0] + rope_dx,
        before_start[1] + rope_dy,
    )
    drag_delta = (
        int(round(0.20 * w)),
        0,
    )
    after_start = (
        before_start[0] + drag_delta[0],
        before_start[1] + drag_delta[1],
    )
    after_end = (
        before_end[0] + drag_delta[0],
        before_end[1] + drag_delta[1],
    )
    return {
        "before": (before_start, before_end),
        "after": (after_start, after_end),
        "drag_delta": drag_delta,
    }


def draw_stage(draw: ImageDraw.ImageDraw, rect):
    rounded_panel(draw, rect, fill=(9, 18, 39, 255), outline=(58, 84, 136, 255), radius=22, width=3)
    x0, y0, x1, y1 = rect
    draw.rounded_rectangle(
        (x0 + 18, y0 + 18, x1 - 18, y1 - 18),
        radius=18,
        fill=(14, 26, 54, 255),
        outline=(42, 70, 118, 255),
        width=2,
    )


def draw_target_window(draw: ImageDraw.ImageDraw, rect, *, color, width=6):
    draw.rounded_rectangle(rect, radius=18, outline=color, width=width)


def draw_force_arrow(
    draw: ImageDraw.ImageDraw,
    *,
    start: tuple[int, int],
    end: tuple[int, int],
    color=ACCENT_BLUE,
    shadow_color=(16, 34, 78, 255),
):
    draw.line((start, end), fill=shadow_color, width=14)
    draw.line((start, end), fill=color, width=8)
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = max((dx * dx + dy * dy) ** 0.5, 1.0)
    ux = dx / length
    uy = dy / length
    perp_x = -uy
    perp_y = ux
    head_len = 34
    head_half = 16
    left = (
        int(round(end[0] - ux * head_len + perp_x * head_half)),
        int(round(end[1] - uy * head_len + perp_y * head_half)),
    )
    right = (
        int(round(end[0] - ux * head_len - perp_x * head_half)),
        int(round(end[1] - uy * head_len - perp_y * head_half)),
    )
    draw.polygon([end, left, right], fill=shadow_color)
    inner_left = (
        int(round(end[0] - ux * 26 + perp_x * 12)),
        int(round(end[1] - uy * 26 + perp_y * 12)),
    )
    inner_right = (
        int(round(end[0] - ux * 26 - perp_x * 12)),
        int(round(end[1] - uy * 26 - perp_y * 12)),
    )
    draw.polygon([end, inner_left, inner_right], fill=color)


def draw_drag_rope_card(
    draw: ImageDraw.ImageDraw,
    rect,
    *,
    title: str,
    target_color,
    pose: str,
    show_arrow: bool = True,
):
    draw_card(draw, rect, title=title)
    scene_rect = (
        rect[0] + CARD_INSET,
        rect[1] + CARD_BODY_TOP,
        rect[2] - CARD_INSET,
        rect[3] - CARD_BOTTOM_PAD,
    )
    draw_stage(draw, scene_rect)
    x0, y0, x1, y1 = scene_rect
    stage_pad = 44
    target_rect = (
        x0 + stage_pad,
        y0 + int(0.21 * (y1 - y0)),
        x1 - stage_pad,
        y1 - int(0.18 * (y1 - y0)),
    )
    draw_target_window(draw, target_rect, color=target_color, width=6)

    drag_poses = _slide_three_drag_poses(target_rect)
    rope_start, rope_end = drag_poses[pose]
    draw.line((rope_start, rope_end), fill=(255, 214, 161, 255), width=18)
    draw.line((rope_start, rope_end), fill=(208, 131, 78, 255), width=8)

    anchor_point = _segment_point(rope_start, rope_end, SLIDE_THREE_ANCHOR_T)
    draw_square_marker(draw, anchor_point, color=ACCENT_BLUE, size=22)

    if show_arrow:
        drag_dx, drag_dy = drag_poses["drag_delta"]
        arrow_end = (
            anchor_point[0] + drag_dx,
            anchor_point[1] + drag_dy,
        )
        draw_force_arrow(draw, start=anchor_point, end=arrow_end)


def draw_rope_scene(
    draw: ImageDraw.ImageDraw,
    rect,
    *,
    mode: str,
    target_color,
    show_left=False,
    show_right=False,
    show_split=False,
    active=False,
    show_cycle_hint=False,
    check=False,
):
    draw_stage(draw, rect)
    x0, y0, x1, y1 = rect
    stage_pad = 52
    target_rect = (
        x0 + stage_pad,
        y0 + int(0.21 * (y1 - y0)),
        x1 - stage_pad,
        y1 - int(0.18 * (y1 - y0)),
    )
    draw_target_window(draw, target_rect, color=target_color)
    rope_start, rope_end = _rope_segment(target_rect, mode)
    draw.line((rope_start, rope_end), fill=(255, 214, 161, 255), width=18)
    draw.line((rope_start, rope_end), fill=(208, 131, 78, 255), width=8)

    if show_left:
        draw_square_marker(
            draw,
            _rope_point_at(target_rect, mode, 0.32),
            color=ACCENT_RED,
            size=22,
        )
    if show_right:
        draw_square_marker(
            draw,
            _rope_point_at(target_rect, mode, 0.68),
            color=ACCENT_BLUE,
            size=22,
        )
    if show_split:
        px, py = _rope_point_at(target_rect, mode, 0.50)
        split_rect = (
            int(px - 12),
            int(py - 12),
            int(px + 12),
            int(py + 12),
        )
        draw_split_square(draw, split_rect)
    if active:
        draw_square_marker(
            draw,
            _rope_point_at(target_rect, mode, 0.50),
            color=ACCENT_MAGENTA,
            size=24,
        )
    if show_cycle_hint:
        hint_x0 = x0 + 24
        hint_y0 = y0 + 22
        draw_button_chip(draw, (hint_x0, hint_y0, hint_x0 + 118, hint_y0 + 46), "X / A")
        arrow_y = hint_y0 + 23
        draw.line((hint_x0 + 136, arrow_y, hint_x0 + 238, arrow_y), fill=WHITE, width=4)
        draw.polygon(
            [
                (hint_x0 + 238, arrow_y),
                (hint_x0 + 220, arrow_y - 10),
                (hint_x0 + 220, arrow_y + 10),
            ],
            fill=WHITE,
        )
    if check:
        check_center_x = x1 - 36
        check_center_y = y0 + 42
        draw.ellipse(
            (check_center_x - 22, check_center_y - 22, check_center_x + 22, check_center_y + 22),
            fill=ACCENT_GREEN,
        )
        draw.line(
            (check_center_x - 10, check_center_y + 2, check_center_x - 1, check_center_y + 11),
            fill=WHITE,
            width=5,
        )
        draw.line(
            (check_center_x - 1, check_center_y + 11, check_center_x + 13, check_center_y - 7),
            fill=WHITE,
            width=5,
        )


def make_slide() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGBA", (SLIDE_WIDTH, SLIDE_HEIGHT), PAGE_BG)
    draw = ImageDraw.Draw(image)
    draw_page_surface(draw)
    return image, draw


def slide_one() -> Image.Image:
    image, draw = make_slide()
    draw_title(draw, "Demo: Boba Immersive XR Application Tutorial")

    left_rect = (OUTER_MARGIN + 28, CONTENT_TOP, 900, CONTENT_TOP + CONTENT_HEIGHT)
    right_rect = (934, CONTENT_TOP, SLIDE_WIDTH - OUTER_MARGIN - 28, CONTENT_TOP + CONTENT_HEIGHT)
    slide_one_title_inset = 38
    slide_one_header_top = 32
    slide_one_header_height = 108
    slide_one_body_top = PANEL_BODY_PAD_TOP + 14
    draw_panel(
        draw,
        left_rect,
        "Controller Basics",
        title_inset_x=slide_one_title_inset,
        header_top_pad=slide_one_header_top,
        header_min_height=slide_one_header_height,
    )
    draw_panel(
        draw,
        right_rect,
        "Button locations",
        title_inset_x=slide_one_title_inset,
        header_top_pad=slide_one_header_top,
        header_min_height=slide_one_header_height,
    )

    body_x = left_rect[0] + PANEL_BODY_PAD_X
    body_y = left_rect[1] + slide_one_body_top
    body_w = left_rect[2] - left_rect[0] - PANEL_BODY_PAD_X * 2
    body_y = draw_control_row(
        draw,
        x=body_x,
        y=body_y,
        button_text="Select",
        description="Hold to interact. Trigger selects an object-menu row.",
        width=body_w,
    )
    body_y = draw_control_row(
        draw,
        x=body_x,
        y=body_y,
        button_text="X / A",
        description="Cycle anchors, or move the object selector highlight.",
        width=body_w,
    )
    body_y = draw_control_row(
        draw,
        x=body_x,
        y=body_y,
        button_text="Y / B",
        description="Tap to reset. Hold 0.75 s to switch Rope / Sloth.",
        width=body_w,
    )
    body_y = draw_control_row(
        draw,
        x=body_x,
        y=body_y,
        button_text="Either Stick",
        description="Up/down moves the selector; press exits anchor cycle.",
        width=body_w,
    )
    draw_control_row(
        draw,
        x=body_x,
        y=body_y,
        button_text="Grip (hold)",
        description="Hold to exit the demo.",
        width=body_w,
    )

    panel_inner = (
        right_rect[0] + PANEL_BODY_PAD_X,
        right_rect[1] + slide_one_body_top,
        right_rect[2] - PANEL_BODY_PAD_X,
        right_rect[3] - PANEL_BODY_PAD_BOTTOM,
    )
    draw_touch_plus_controller_panel(image, draw, panel_inner)
    draw_footer(draw, "Press select to continue")
    return image


def draw_step(
    draw: ImageDraw.ImageDraw,
    *,
    number: int,
    x: int,
    y: int,
    width: int,
    text: str,
    vertical_gap: int = STEP_VERTICAL_GAP,
    circle_size: int = 44,
    text_gap: int = 20,
) -> int:
    draw.ellipse((x, y, x + circle_size, y + circle_size), fill=BUTTON_BG, outline=BUTTON_BORDER, width=3)
    num_w, num_h = text_size(draw, str(number), STEP_NUMBER_FONT)
    draw.text(
        (x + (circle_size - num_w) // 2, y + (circle_size - num_h) // 2 - 1),
        str(number),
        font=STEP_NUMBER_FONT,
        fill=WHITE,
    )
    text_x = x + circle_size + text_gap
    return draw_wrapped_text(
        draw,
        (text_x, y + 2),
        text,
        font=BODY_FONT,
        fill=BODY_COLOR,
        max_width=width - circle_size - text_gap,
        line_gap=6,
    ) + int(vertical_gap)


def slide_two() -> Image.Image:
    image, draw = make_slide()
    draw_title(draw, "How Interaction Works")

    left_rect = (OUTER_MARGIN + 28, CONTENT_TOP, 900, CONTENT_TOP + CONTENT_HEIGHT)
    right_rect = (934, CONTENT_TOP, SLIDE_WIDTH - OUTER_MARGIN - 28, CONTENT_TOP + CONTENT_HEIGHT)
    draw_panel(draw, left_rect, "Interaction steps")
    draw_panel(draw, right_rect, "Marker legends")

    body_x = left_rect[0] + PANEL_BODY_PAD_X
    body_y = left_rect[1] + PANEL_BODY_PAD_TOP
    body_w = left_rect[2] - left_rect[0] - PANEL_BODY_PAD_X * 2
    body_y = draw_step(
        draw,
        number=1,
        x=body_x,
        y=body_y,
        width=body_w,
        text="Aim at the rope or object you want to move.",
        vertical_gap=SLIDE_TWO_STEP_VERTICAL_GAP,
    )
    body_y = draw_step(
        draw,
        number=2,
        x=body_x,
        y=body_y,
        width=body_w,
        text="Hold Select to grab the nearest anchor.",
        vertical_gap=SLIDE_TWO_STEP_VERTICAL_GAP,
    )
    body_y = draw_step(
        draw,
        number=3,
        x=body_x,
        y=body_y,
        width=body_w,
        text="If locking is hard, press X / A to cycle anchors.",
        vertical_gap=SLIDE_TWO_STEP_VERTICAL_GAP,
    )
    draw_step(
        draw,
        number=4,
        x=body_x,
        y=body_y,
        width=body_w,
        text="Stick Press exits cycle. Tap Y / B to reset; hold Y / B to switch objects.",
        vertical_gap=SLIDE_TWO_STEP_VERTICAL_GAP,
    )

    legend_x = right_rect[0] + PANEL_BODY_PAD_X
    legend_y = right_rect[1] + PANEL_BODY_PAD_TOP
    legend_w = right_rect[2] - right_rect[0] - PANEL_BODY_PAD_X * 2
    intro_y = draw_wrapped_text(
        draw,
        (legend_x, legend_y),
        "Candidate boxes stay on the object. They show who owns the current anchor.",
        font=BODY_FONT,
        fill=BODY_COLOR,
        max_width=legend_w,
        line_gap=6,
    )
    row_y = intro_y + 22

    def legend_row(color_rect_drawer, label: str, y: int) -> int:
        box_rect = (legend_x, y, legend_x + 60, y + 60)
        color_rect_drawer(box_rect)
        draw_wrapped_text(
            draw,
            (legend_x + 88, y + 7),
            label,
            font=BODY_FONT,
            fill=BODY_COLOR,
            max_width=legend_w - 88,
            line_gap=4,
        )
        return y + 90

    def draw_legend_square(rect, color):
        cx = (rect[0] + rect[2]) * 0.5
        cy = (rect[1] + rect[3]) * 0.5
        draw_square_marker(draw, (cx, cy), color=color, size=60, width=5)

    row_y = legend_row(
        lambda rect: draw_legend_square(rect, ACCENT_RED),
        "Red box: left controller candidate",
        row_y,
    )
    row_y = legend_row(
        lambda rect: draw_legend_square(rect, ACCENT_BLUE),
        "Blue box: right controller candidate",
        row_y,
    )
    row_y = legend_row(
        lambda rect: draw_split_square(draw, rect),
        "Split box: both controllers are on the same anchor",
        row_y,
    )

    draw_footer(draw, "Press select to continue")
    return image


def slide_three() -> Image.Image:
    image, draw = make_slide()
    draw_title(draw, "Rope Game Goal")

    left_rect = (OUTER_MARGIN + 28, CONTENT_TOP, 690, CONTENT_TOP + CONTENT_HEIGHT)
    right_rect = (724, CONTENT_TOP, SLIDE_WIDTH - OUTER_MARGIN - 28, CONTENT_TOP + CONTENT_HEIGHT)
    draw_panel(draw, left_rect, "Objective")
    draw_panel(draw, right_rect, "Example move")

    slide_three_body_pad_x = 12
    body_x = left_rect[0] + slide_three_body_pad_x
    body_y = left_rect[1] + PANEL_BODY_PAD_TOP
    body_w = left_rect[2] - left_rect[0] - slide_three_body_pad_x * 2
    body_y = draw_step(
        draw,
        number=1,
        x=body_x,
        y=body_y,
        width=body_w,
        text="Pull the rope into the target window.",
        vertical_gap=SLIDE_THREE_STEP_VERTICAL_GAP,
        circle_size=40,
        text_gap=10,
    )
    body_y = draw_step(
        draw,
        number=2,
        x=body_x,
        y=body_y,
        width=body_w,
        text="Yellow rectangle: rope still outside.",
        vertical_gap=SLIDE_THREE_STEP_VERTICAL_GAP,
        circle_size=40,
        text_gap=10,
    )
    body_y = draw_step(
        draw,
        number=3,
        x=body_x,
        y=body_y,
        width=body_w,
        text="Green rectangle: rope fully inside.",
        vertical_gap=SLIDE_THREE_STEP_VERTICAL_GAP,
        circle_size=40,
        text_gap=10,
    )
    draw_step(
        draw,
        number=4,
        x=body_x,
        y=body_y,
        width=body_w,
        text="Use X / A if selecting is hard.",
        vertical_gap=SLIDE_THREE_STEP_VERTICAL_GAP,
        circle_size=40,
        text_gap=10,
    )

    panel_inner = (
        right_rect[0] + PANEL_BODY_PAD_X,
        right_rect[1] + PANEL_BODY_PAD_TOP,
        right_rect[2] - PANEL_BODY_PAD_X,
        right_rect[3] - PANEL_BODY_PAD_BOTTOM,
    )
    card_y = panel_inner[1]
    card_gap = CARD_GAP
    card_width = (panel_inner[2] - panel_inner[0] - card_gap) // 2
    card_height = panel_inner[3] - card_y
    card_left = (panel_inner[0], card_y, panel_inner[0] + card_width, card_y + card_height)
    card_right = (
        panel_inner[0] + card_width + card_gap,
        card_y,
        panel_inner[2],
        card_y + card_height,
    )
    draw_drag_rope_card(
        draw,
        card_left,
        title="Before drag",
        target_color=ACCENT_YELLOW,
        pose="before",
        show_arrow=True,
    )
    draw_drag_rope_card(
        draw,
        card_right,
        title="After drag",
        target_color=ACCENT_GREEN,
        pose="after",
        show_arrow=False,
    )
    draw_footer(draw, "Press select to start the demo")
    return image


def main():
    TUTORIAL_DIR.mkdir(parents=True, exist_ok=True)
    ROPE_GAME_DIR.mkdir(parents=True, exist_ok=True)

    slide_one().save(TUTORIAL_DIR / "controls_overview.png")
    slide_two().save(TUTORIAL_DIR / "interaction_tips.png")
    slide_three().save(ROPE_GAME_DIR / "tutorial_rope_game_goal.png")
    print("Generated tutorial slides.")


if __name__ == "__main__":
    main()
