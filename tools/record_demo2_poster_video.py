#!/usr/bin/env python3
"""Record a real Demo 2 public display and phone stream into a poster video.

The script does not synthesize simulation frames. It captures the live GLFW
public-display window through X11 and places real frames into a deterministic
1920x1080 composition. The phone can show either one claimed session's MJPEG
stream or the exact live public-display tile for a prerecorded replay.
"""

from __future__ import annotations

import argparse
import os
import re
import struct
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = REPO_ROOT / "assets"
DEFAULT_FFMPEG = Path(os.environ.get("CONDA_PREFIX", "")) / "bin" / "ffmpeg"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", default="http://127.0.0.1:7860")
    parser.add_argument("--session-id", type=int, default=32)
    parser.add_argument(
        "--phone-source",
        choices=("mjpeg", "public-tile"),
        default="mjpeg",
        help=(
            "Use a claimed live session's MJPEG stream, or mirror the selected "
            "real prerecorded public-display tile without claiming it"
        ),
    )
    parser.add_argument("--duration", type=float, default=45.0)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--display", default=os.environ.get("DISPLAY", ":1"))
    parser.add_argument("--window-title", default="Boba_Batched Playground")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument(
        "--grid-cols",
        type=int,
        default=10,
        help="Number of columns in the live public-display grid",
    )
    parser.add_argument(
        "--grid-rows",
        type=int,
        default=10,
        help="Number of rows in the live public-display grid",
    )
    parser.add_argument(
        "--ffmpeg",
        type=Path,
        default=DEFAULT_FFMPEG,
        help="ffmpeg binary used for the final H.264 transcode",
    )
    return parser


def wait_for_server(server_url: str, timeout_s: float = 60.0) -> None:
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            response = requests.get(f"{server_url}/api/sessions", timeout=2.0)
            response.raise_for_status()
            return
        except Exception as exc:  # pragma: no cover - depends on live runtime
            last_error = exc
            time.sleep(0.5)
    raise RuntimeError(f"Demo 2 server did not become ready: {last_error}")


def fetch_replay_state(server_url: str, session_id: int) -> dict:
    """Fetch the controller action for the frame currently on public display."""
    response = requests.get(
        f"{server_url}/api/replay/{session_id}",
        timeout=2.0,
    )
    response.raise_for_status()
    state = response.json()
    controls = state.get("controls")
    if not isinstance(controls, list):
        raise RuntimeError("Demo 2 replay state omitted its controls")
    control_parts = int(state.get("control_parts", 0))
    if control_parts not in (1, 2):
        raise RuntimeError("Demo 2 replay state omitted its control-part count")
    state["active_controls"] = frozenset(
        (str(item["hand"]), str(item["control"])) for item in controls
    )
    state["control_parts"] = control_parts
    return state


def locate_x11_window(
    title: str,
    display: str,
    timeout_s: float = 60.0,
) -> tuple[str, int, int, int, int]:
    env = {**os.environ, "DISPLAY": display}
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        tree = subprocess.run(
            ["xwininfo", "-root", "-tree"],
            env=env,
            text=True,
            capture_output=True,
            check=True,
        ).stdout
        window_id = None
        for line in tree.splitlines():
            if f'"{title}"' in line:
                match = re.search(r"\b(0x[0-9a-fA-F]+)\b", line)
                if match:
                    window_id = match.group(1)
                    break
        if window_id is None:
            time.sleep(0.5)
            continue

        report = subprocess.run(
            ["xwininfo", "-id", window_id],
            env=env,
            text=True,
            capture_output=True,
            check=True,
        ).stdout

        def value(label: str) -> int:
            match = re.search(rf"^\s*{re.escape(label)}:\s*(-?\d+)\s*$", report, re.MULTILINE)
            if not match:
                raise RuntimeError(f"xwininfo omitted {label!r} for {window_id}")
            return int(match.group(1))

        return (
            window_id,
            value("Absolute upper-left X"),
            value("Absolute upper-left Y"),
            value("Width"),
            value("Height"),
        )
    raise RuntimeError(f"Could not find X11 window titled {title!r}")


def decode_xwd_payload(payload: bytes) -> Image.Image:
    """Decode one 32-bit TrueColor XWD payload."""
    if len(payload) < 100:
        raise RuntimeError("xwd returned a truncated header")

    fields = struct.unpack_from(">25I", payload)
    header_size = fields[0]
    file_version = fields[1]
    pixmap_format = fields[2]
    width = fields[4]
    height = fields[5]
    byte_order = fields[7]
    bits_per_pixel = fields[11]
    bytes_per_line = fields[12]
    red_mask, green_mask, blue_mask = fields[14:17]
    color_count = fields[19]
    if file_version != 7 or pixmap_format != 2:
        raise RuntimeError(
            f"Unsupported XWD format: version={file_version}, format={pixmap_format}"
        )
    if bits_per_pixel != 32 or bytes_per_line % 4:
        raise RuntimeError(
            f"Unsupported XWD pixel layout: bpp={bits_per_pixel}, stride={bytes_per_line}"
        )

    data_offset = header_size + color_count * 12
    words_per_line = bytes_per_line // 4
    expected_size = data_offset + height * bytes_per_line
    if len(payload) < expected_size:
        raise RuntimeError(
            f"xwd returned {len(payload)} bytes; expected at least {expected_size}"
        )
    dtype = np.dtype("<u4" if byte_order == 0 else ">u4")
    pixels = np.frombuffer(
        payload,
        dtype=dtype,
        count=height * words_per_line,
        offset=data_offset,
    ).reshape(height, words_per_line)[:, :width]

    def channel(mask: int) -> np.ndarray:
        if mask == 0:
            return np.zeros((height, width), dtype=np.uint8)
        shift = (mask & -mask).bit_length() - 1
        maximum = mask >> shift
        values = (pixels & mask) >> shift
        return ((values.astype(np.uint64) * 255) // maximum).astype(np.uint8)

    rgb = np.stack(
        (channel(red_mask), channel(green_mask), channel(blue_mask)),
        axis=-1,
    )
    return Image.fromarray(rgb, mode="RGB")


def capture_x11_window(window_id: str, display: str) -> Image.Image:
    """Capture one X11 window's drawable directly, even when it is occluded.

    PIL ImageGrab reads pixels from the root window, so another application can
    accidentally appear in the recording if it covers the demo. ``xwd`` reads
    the requested GLFW drawable instead. X11 can briefly return either a failed
    read or a malformed pixel layout during a GLFW buffer swap, so retry both
    conditions before abandoning the recording.
    """
    errors = []
    for attempt in range(5):
        candidate = subprocess.run(
            ["xwd", "-silent", "-id", window_id],
            env={**os.environ, "DISPLAY": display},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if candidate.returncode == 0:
            try:
                return decode_xwd_payload(candidate.stdout)
            except RuntimeError as exc:
                errors.append(str(exc))
        else:
            errors.append(
                candidate.stderr.decode("utf-8", errors="replace").strip()
            )
        time.sleep(0.02 * (attempt + 1))
    details = "; ".join(error for error in errors if error) or "no X11 error text"
    raise RuntimeError(f"xwd could not capture {window_id} after 5 attempts: {details}")


class LatestMjpegFrame:
    def __init__(self, url: str):
        self.url = url
        self._frame: Image.Image | None = None
        self._error: Exception | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="demo2-mjpeg", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3.0)

    def latest(self) -> Image.Image | None:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def wait_for_first(self, timeout_s: float = 30.0) -> Image.Image:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            frame = self.latest()
            if frame is not None:
                return frame
            if self._error is not None:
                raise RuntimeError(f"MJPEG reader failed: {self._error}")
            time.sleep(0.05)
        raise RuntimeError("Phone MJPEG stream did not publish a frame")

    def _run(self) -> None:
        buffer = bytearray()
        try:
            with requests.get(self.url, stream=True, timeout=(5.0, 60.0)) as response:
                response.raise_for_status()
                for chunk in response.iter_content(chunk_size=65536):
                    if self._stop.is_set():
                        return
                    if not chunk:
                        continue
                    buffer.extend(chunk)
                    while True:
                        start = buffer.find(b"\xff\xd8")
                        if start < 0:
                            if len(buffer) > 2_000_000:
                                del buffer[:-2]
                            break
                        end = buffer.find(b"\xff\xd9", start + 2)
                        if end < 0:
                            if start > 0:
                                del buffer[:start]
                            break
                        jpeg = bytes(buffer[start : end + 2])
                        del buffer[: end + 2]
                        decoded = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if decoded is None:
                            continue
                        rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
                        with self._lock:
                            self._frame = Image.fromarray(rgb)
        except Exception as exc:  # pragma: no cover - depends on live stream
            self._error = exc


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
        if bold
        else Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf")
        if bold
        else Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def fit_exact(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    return image.convert("RGB").resize(size, Image.Resampling.LANCZOS)


def crop_public_tile(
    public_display: Image.Image,
    session_id: int,
    grid_cols: int,
    grid_rows: int,
) -> Image.Image:
    """Return the exact selected tile from a live public-display capture."""
    row, col = divmod(session_id, grid_cols)
    width, height = public_display.size
    x0 = round(col * width / grid_cols)
    x1 = round((col + 1) * width / grid_cols)
    y0 = round(row * height / grid_rows)
    y1 = round((row + 1) * height / grid_rows)
    return public_display.crop((x0, y0, x1, y1))


def control_state(
    elapsed_s: float,
) -> tuple[dict[str, int], dict[str, int], tuple[str, str] | None]:
    """Move two controller groups through distinct bounded 2D poses.

    The live runtime accumulates a held input once per simulation step.  The
    runtime's per-hand radial offset clamp bounds every pose.  Alternating the
    left and right groups creates actual bends instead of translating the whole
    grasp region, and neutral intervals let the rope settle between moves.
    """
    segments: list[
        tuple[
            float,
            float,
            dict[str, int],
            dict[str, int],
            tuple[str, str] | None,
        ]
    ] = [
        (0.0, 2.0, {}, {}, None),
        (2.0, 4.0, {"x": 1}, {}, ("left", "xneg")),
        (4.0, 7.0, {}, {}, None),
        (7.0, 9.0, {}, {"x": -1}, ("right", "xpos")),
        (9.0, 12.0, {}, {}, None),
        (12.0, 14.0, {"y": -1}, {}, ("left", "ypos")),
        (14.0, 17.0, {}, {}, None),
        (17.0, 19.0, {}, {"y": 1}, ("right", "yneg")),
        (19.0, 22.0, {}, {}, None),
        (22.0, 25.0, {"x": -1}, {}, ("left", "xpos")),
        (25.0, 28.0, {}, {}, None),
        (28.0, 31.0, {}, {"x": 1}, ("right", "xneg")),
        (31.0, 34.0, {}, {}, None),
        (34.0, 37.0, {"y": 1}, {}, ("left", "yneg")),
        (37.0, 40.0, {}, {}, None),
        (40.0, 43.0, {}, {"y": -1}, ("right", "ypos")),
        (43.0, 45.0, {}, {}, None),
    ]
    for start, end, left_values, right_values, active in segments:
        if start <= elapsed_s < end:
            left = {"x": 0, "y": 0, "z": 0}
            right = {"x": 0, "y": 0, "z": 0}
            left.update(left_values)
            right.update(right_values)
            return left, right, active
    zero = {"x": 0, "y": 0, "z": 0}
    return dict(zero), dict(zero), None


class PhoneControls:
    def __init__(self) -> None:
        self._sources = {
            "empty": Image.open(ASSET_ROOT / "arrow_empty.png").convert("RGBA"),
            "left": Image.open(ASSET_ROOT / "arrow_1.png").convert("RGBA"),
            "right": Image.open(ASSET_ROOT / "arrow_2.png").convert("RGBA"),
        }
        self._cache: dict[tuple[str, str, int], Image.Image] = {}

    def arrow(self, hand: str, control: str, active: bool, size: int) -> Image.Image:
        key = (hand if active else "empty", control, size)
        if key in self._cache:
            return self._cache[key]
        source = self._sources[hand if active else "empty"].copy()
        rotations = {
            "xneg": 0,
            "xpos": 180,
            "ypos": 90,
            "yneg": -90,
            "zneg": 0,
            "zpos": 180,
        }
        source = source.rotate(rotations[control], expand=True, resample=Image.Resampling.BICUBIC)
        source.thumbnail((size, size), Image.Resampling.LANCZOS)
        self._cache[key] = source
        return source


def paste_center(canvas: Image.Image, item: Image.Image, center: tuple[int, int]) -> None:
    x = int(center[0] - item.width / 2)
    y = int(center[1] - item.height / 2)
    canvas.paste(item, (x, y), item)


def draw_phone(
    canvas: Image.Image,
    stream: Image.Image,
    active_controls: frozenset[tuple[str, str]],
    control_parts: int,
    controls: PhoneControls,
    highlight_color: tuple[int, int, int],
) -> None:
    # Proportions follow phone_demo_overview_capture_preview_v2.png: the phone
    # is roughly 12.5% of the canvas width and 37% of its height, while the
    # controller occupies only the lower third of the usable display.
    outer = (110, 528, 350, 925)
    screen = (122, 541, 338, 913)
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle(outer, radius=42, fill=(17, 18, 20), outline=(60, 62, 64), width=3)
    draw.rounded_rectangle(screen, radius=31, fill=(4, 5, 6))

    sx0, sy0, sx1, sy1 = screen
    screen_w = sx1 - sx0
    header_h = 45
    stream_h = int(round(screen_w * 3 / 4))
    stream_box = (sx0, sy0 + header_h, sx1, sy0 + header_h + stream_h)
    stream_frame = fit_exact(stream, (screen_w, stream_h))
    canvas.paste(stream_frame, stream_box[:2])

    # Matching accent around the exact real phone stream.
    draw.rectangle(stream_box, outline=highlight_color, width=4)
    notch = (sx0 + screen_w // 2 - 31, sy0 + 8, sx0 + screen_w // 2 + 31, sy0 + 23)
    draw.rounded_rectangle(notch, radius=9, fill=(0, 0, 0))

    release = (sx1 - 70, sy0 + 8, sx1 - 8, sy0 + 37)
    draw.rounded_rectangle(release, radius=16, fill=(58, 32, 32), outline=(157, 75, 75), width=2)
    release_font = load_font(10, bold=True)
    label = "Release"
    bbox = draw.textbbox((0, 0), label, font=release_font)
    draw.text(
        ((release[0] + release[2] - (bbox[2] - bbox[0])) / 2, (release[1] + release[3] - (bbox[3] - bbox[1])) / 2 - 1),
        label,
        font=release_font,
        fill=(245, 247, 248),
    )

    control_y0 = stream_box[3] + 18
    control_box = (sx0, control_y0, sx1, sy1)
    draw.rectangle(control_box, fill=(244, 246, 248))
    title_font = load_font(11, bold=True)
    panel_top = control_y0 + 35
    if int(control_parts) == 1:
        title = "Interaction Point"
        title_bbox = draw.textbbox((0, 0), title, font=title_font)
        draw.text(
            (sx0 + (screen_w - (title_bbox[2] - title_bbox[0])) / 2, control_y0 + 9),
            title,
            font=title_font,
            fill=(16, 20, 24),
        )
        arrow_size = 25
        cell = 36
        layouts = {
            "left": {
                "xneg": (1, 0),
                "zneg": (3, 0),
                "ypos": (0, 1),
                "xpos": (1, 1),
                "yneg": (2, 1),
                "zpos": (3, 1),
            }
        }
        hand_starts = (("left", sx0 + (screen_w - 4 * cell) // 2),)
    else:
        draw.text((sx0 + 6, control_y0 + 9), "Left Hand", font=title_font, fill=(16, 20, 24))
        right_text = "Right Hand"
        right_bbox = draw.textbbox((0, 0), right_text, font=title_font)
        draw.text((sx1 - 6 - (right_bbox[2] - right_bbox[0]), control_y0 + 9), right_text, font=title_font, fill=(16, 20, 24))
        panel_w = 96
        gap = screen_w - 2 * panel_w
        arrow_size = 20
        cell = 24
        layouts = {
            "left": {
                "xneg": (1, 0),
                "zneg": (3, 0),
                "ypos": (0, 1),
                "xpos": (1, 1),
                "yneg": (2, 1),
                "zpos": (3, 1),
            },
            "right": {
                "zneg": (0, 0),
                "xneg": (2, 0),
                "zpos": (0, 1),
                "ypos": (1, 1),
                "xpos": (2, 1),
                "yneg": (3, 1),
            },
        }
        hand_starts = (("left", sx0), ("right", sx0 + panel_w + gap))

    for hand, x_start in hand_starts:
        for control, (col, row) in layouts[hand].items():
            icon = controls.arrow(
                hand,
                control,
                active=((hand, control) in active_controls),
                size=arrow_size,
            )
            center = (x_start + col * cell + cell // 2, panel_top + row * 50 + 16)
            paste_center(canvas, icon, center)


def compose_frame(
    public_display: Image.Image,
    phone_stream: Image.Image,
    active_controls: frozenset[tuple[str, str]],
    control_parts: int,
    canvas_size: tuple[int, int],
    session_id: int,
    grid_cols: int,
    grid_rows: int,
    controls: PhoneControls,
) -> Image.Image:
    canvas = Image.new("RGB", canvas_size, (249, 247, 244))
    draw = ImageDraw.Draw(canvas)

    # Monitor hardware and exact public-display capture.
    monitor_outer = (420, 67, 1872, 891)
    monitor_screen = (452, 98, 1840, 865)
    draw.rounded_rectangle(monitor_outer, radius=20, fill=(24, 25, 27), outline=(65, 66, 68), width=3)
    resized_public = fit_exact(
        public_display,
        (monitor_screen[2] - monitor_screen[0], monitor_screen[3] - monitor_screen[1]),
    )
    canvas.paste(resized_public, monitor_screen[:2])

    # Highlight the exact real matching tile with fixed geometry and width.
    row = session_id // grid_cols
    col = session_id % grid_cols
    screen_w = monitor_screen[2] - monitor_screen[0]
    screen_h = monitor_screen[3] - monitor_screen[1]
    tx0 = monitor_screen[0] + round(col * screen_w / grid_cols)
    tx1 = monitor_screen[0] + round((col + 1) * screen_w / grid_cols)
    ty0 = monitor_screen[1] + round(row * screen_h / grid_rows)
    ty1 = monitor_screen[1] + round((row + 1) * screen_h / grid_rows)
    highlight_color = (255, 91, 91)
    draw.rectangle((tx0, ty0, tx1 - 1, ty1 - 1), outline=highlight_color, width=8)

    # Monitor stand.
    draw.rectangle((1073, 891, 1219, 978), fill=(34, 35, 37))
    draw.rounded_rectangle((895, 967, 1397, 1001), radius=13, fill=(43, 44, 46), outline=(74, 75, 77), width=2)

    draw_phone(
        canvas,
        phone_stream,
        active_controls,
        control_parts,
        controls,
        highlight_color,
    )
    return canvas


def send_input(
    server_url: str,
    session_id: int,
    token: str,
    left: dict[str, int],
    right: dict[str, int],
) -> None:
    response = requests.post(
        f"{server_url}/api/sessions/{session_id}/input",
        json={
            "token": token,
            "left": left,
            "right": right,
        },
        timeout=3.0,
    )
    response.raise_for_status()


def encode_h264(intermediate: Path, output: Path, ffmpeg: Path) -> None:
    if not ffmpeg.is_file():
        raise FileNotFoundError(f"ffmpeg not found: {ffmpeg}")
    subprocess.run(
        [
            str(ffmpeg),
            "-y",
            "-i",
            str(intermediate),
            "-an",
            "-c:v",
            "libopenh264",
            "-b:v",
            "10M",
            "-maxrate",
            "12M",
            "-bufsize",
            "20M",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
    )


def main() -> None:
    args = build_parser().parse_args()
    if args.duration <= 0:
        raise ValueError("--duration must be positive")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.width % 2 or args.height % 2:
        raise ValueError("video width and height must both be even")
    if args.grid_cols < 1 or args.grid_rows < 1:
        raise ValueError("--grid-cols and --grid-rows must both be positive")
    if args.session_id < 0 or args.session_id >= args.grid_cols * args.grid_rows:
        raise ValueError("--session-id must fit inside the configured grid")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    wait_for_server(args.server_url)
    window_id, x, y, window_width, window_height = locate_x11_window(
        args.window_title,
        args.display,
    )
    print(
        f"[Recorder] Public display window: {window_id} "
        f"{window_width}x{window_height}+{x}+{y}",
        flush=True,
    )

    token: str | None = None
    mjpeg: LatestMjpegFrame | None = None
    if args.phone_source == "mjpeg":
        claim = requests.post(
            f"{args.server_url}/api/sessions/{args.session_id}/claim",
            timeout=5.0,
        )
        claim.raise_for_status()
        token = claim.json()["token"]
        stream_url = (
            f"{args.server_url}/stream/{args.session_id}.mjpg"
            f"?token={requests.utils.quote(token, safe='')}"
        )
        mjpeg = LatestMjpegFrame(stream_url)
        mjpeg.start()

    fd, intermediate_name = tempfile.mkstemp(
        prefix="boba-demo2-capture-",
        suffix=".mp4",
        dir=str(args.output.parent),
    )
    os.close(fd)
    intermediate = Path(intermediate_name)
    intermediate.unlink()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(intermediate),
        fourcc,
        args.fps,
        (args.width, args.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {intermediate}")

    controls = PhoneControls()
    total_frames = int(round(args.duration * args.fps))
    frame_interval = 1.0 / args.fps
    last_input: tuple[dict[str, int], dict[str, int]] | None = None
    last_input_send = 0.0
    last_heartbeat = 0.0

    try:
        latest_phone = mjpeg.wait_for_first() if mjpeg is not None else None
        print(
            f"[Recorder] Capturing {total_frames} frames at {args.fps:g} fps "
            f"({args.duration:g} seconds), session={args.session_id}, "
            f"phone_source={args.phone_source}",
            flush=True,
        )
        start = time.monotonic()
        for frame_index in range(total_frames):
            target_time = start + frame_index * frame_interval
            delay = target_time - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            elapsed = frame_index * frame_interval
            active_controls: frozenset[tuple[str, str]] = frozenset()
            control_parts = 2
            if token is not None:
                left, right, active_control = control_state(elapsed)
                if active_control is not None:
                    active_controls = frozenset((active_control,))
                now = time.monotonic()
                current_input = (left, right)
                if current_input != last_input or now - last_input_send >= 0.5:
                    send_input(args.server_url, args.session_id, token, left, right)
                    last_input = (dict(left), dict(right))
                    last_input_send = now
                if now - last_heartbeat >= 2.5:
                    heartbeat = requests.post(
                        f"{args.server_url}/api/sessions/{args.session_id}/heartbeat",
                        json={"token": token},
                        timeout=3.0,
                    )
                    heartbeat.raise_for_status()
                    last_heartbeat = now

            else:
                replay_state = fetch_replay_state(args.server_url, args.session_id)
                active_controls = replay_state["active_controls"]
                control_parts = replay_state["control_parts"]

            # The replay state is fetched immediately before X11 capture so the
            # annotated buttons correspond to the frame currently being shown.
            public_display = capture_x11_window(window_id, args.display)
            if public_display.size != (window_width, window_height):
                raise RuntimeError(
                    "Boba window size changed during recording: "
                    f"expected {(window_width, window_height)}, "
                    f"received {public_display.size}"
                )
            if mjpeg is None:
                latest_phone = crop_public_tile(
                    public_display,
                    args.session_id,
                    args.grid_cols,
                    args.grid_rows,
                )
            else:
                phone = mjpeg.latest()
                if phone is not None:
                    latest_phone = phone
            if latest_phone is None:
                raise RuntimeError("No real phone frame is available")
            composed = compose_frame(
                public_display,
                latest_phone,
                active_controls,
                control_parts,
                (args.width, args.height),
                args.session_id,
                args.grid_cols,
                args.grid_rows,
                controls,
            )
            bgr = cv2.cvtColor(np.asarray(composed), cv2.COLOR_RGB2BGR)
            writer.write(bgr)
            if frame_index == 0 or (frame_index + 1) % max(1, int(args.fps * 5)) == 0:
                print(f"[Recorder] frame {frame_index + 1}/{total_frames}", flush=True)
    finally:
        if token is not None:
            try:
                send_input(
                    args.server_url,
                    args.session_id,
                    token,
                    {"x": 0, "y": 0, "z": 0},
                    {"x": 0, "y": 0, "z": 0},
                )
            except Exception:
                pass
            try:
                requests.post(
                    f"{args.server_url}/api/sessions/{args.session_id}/release",
                    json={"token": token},
                    timeout=3.0,
                )
            except Exception:
                pass
        if mjpeg is not None:
            mjpeg.stop()
        writer.release()

    print("[Recorder] Encoding H.264 final...", flush=True)
    encode_h264(intermediate, args.output, args.ffmpeg)
    intermediate.unlink(missing_ok=True)
    print(f"[Recorder] Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
