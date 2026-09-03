#!/usr/bin/env python3
"""Compare Demo 2 adaptive MJPEG and acknowledged WebSocket freshness.

The benchmark uses 640x480 JPEG at quality 80 and a 30 FPS synthetic simulation.
It publishes a frame only when the transport has requested one, matching the
adaptive runtime. A configurable processing delay simulates a slow phone.
"""

import argparse
import hashlib
import json
import queue
import re
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import quote, urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import requests
from simple_websocket import Client

from demos.demo2.session_manager import SessionManager
from demos.demo2.streaming import MjpegFrameStore
from demos.demo2_server import create_app, start_flask_server


QUICK_TUNNEL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")


class FramePublisher:
    def __init__(self, store, fps=30.0):
        self.store = store
        self.interval_s = 1.0 / float(fps)
        self._stop = threading.Event()
        self._timestamps = {}
        self._lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._run,
            name="demo2-transport-benchmark-publisher",
            daemon=True,
        )
        x = np.arange(640, dtype=np.uint16)[None, :]
        y = np.arange(480, dtype=np.uint16)[:, None]
        background = np.empty((480, 640, 3), dtype=np.uint8)
        background[:, :, 0] = (x // 3 + y // 4) % 256
        background[:, :, 1] = (x // 5 + 2 * y // 3) % 256
        background[:, :, 2] = (2 * x // 3 + y // 7) % 256
        for grid_x in range(0, 640, 80):
            cv2.line(background, (grid_x, 0), (grid_x, 479), (245, 245, 245), 1)
        for grid_y in range(0, 480, 80):
            cv2.line(background, (0, grid_y), (639, grid_y), (245, 245, 245), 1)
        cv2.putText(
            background,
            "SYNTHETIC DEMO BENCHMARK",
            (120, 245),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        self._background = background

    def start(self):
        self._thread.start()

    def close(self):
        self._stop.set()
        self._thread.join(timeout=2.0)

    def published_at(self, jpeg):
        digest = hashlib.sha1(jpeg).digest()
        with self._lock:
            return self._timestamps.get(digest)

    def _run(self):
        sequence = 0
        deadline = time.monotonic()
        while not self._stop.is_set():
            if self.store.requested_sessions([0]):
                frame = self._background.copy()
                x = 20 + (sequence * 17) % 560
                cv2.rectangle(frame, (x, 32), (x + 60, 92), (20, 40, 240), -1)
                cv2.putText(
                    frame,
                    str(sequence),
                    (18, 455),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.store.publish_rgb(0, rgb)
                jpeg, _, published_at = self.store.latest_packet(0)
                digest = hashlib.sha1(jpeg).digest()
                with self._lock:
                    self._timestamps[digest] = published_at
                    if len(self._timestamps) > 2000:
                        oldest = next(iter(self._timestamps))
                        del self._timestamps[oldest]

            sequence += 1
            deadline += self.interval_s
            wait_s = deadline - time.monotonic()
            if wait_s <= 0:
                deadline = time.monotonic()
                continue
            self._stop.wait(wait_s)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--warmup", type=float, default=1.0)
    parser.add_argument("--simulation-fps", type=float, default=30.0)
    parser.add_argument(
        "--slow-delay",
        type=float,
        default=0.2,
        help="Seconds of simulated phone processing after each decoded frame.",
    )
    parser.add_argument(
        "--cloudflare",
        action="store_true",
        help="Create a temporary Quick Tunnel and benchmark through it.",
    )
    parser.add_argument("--json-output", type=Path, default=None)
    return parser.parse_args()


def start_quick_tunnel(origin_url):
    process = subprocess.Popen(
        [
            "cloudflared",
            "tunnel",
            "--url",
            origin_url,
            "--no-autoupdate",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    output_queue = queue.Queue()

    def read_output():
        for line in process.stdout:
            output_queue.put(line)

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    deadline = time.monotonic() + 30.0
    captured = []
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                "cloudflared exited before creating a URL:\n" + "".join(captured)
            )
        try:
            line = output_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        captured.append(line)
        match = QUICK_TUNNEL_RE.search(line)
        if match:
            return process, match.group(0)
    process.terminate()
    process.wait(timeout=5.0)
    raise RuntimeError(
        "Timed out waiting for a Cloudflare Quick Tunnel URL:\n"
        + "".join(captured)
    )


def wait_until_reachable(base_url, timeout_s=90.0):
    deadline = time.monotonic() + timeout_s
    last_error = None
    while time.monotonic() < deadline:
        try:
            response = requests.get(base_url + "/", timeout=2.0)
            if response.status_code == 200:
                return
            last_error = RuntimeError(f"HTTP {response.status_code}")
        except requests.RequestException as exc:
            last_error = exc
        time.sleep(0.25)
    raise RuntimeError(f"Benchmark URL did not become reachable: {last_error}")


def claim_session(base_url):
    response = requests.post(base_url + "/api/sessions/0/claim", timeout=5.0)
    response.raise_for_status()
    return response.json()["token"]


def release_session(base_url, token):
    try:
        requests.post(
            base_url + "/api/sessions/0/release",
            json={"token": token},
            timeout=5.0,
        )
    except requests.RequestException:
        pass


def read_mjpeg_frame(response):
    while True:
        line = response.raw.readline()
        if not line:
            raise EOFError("MJPEG stream ended")
        if line.strip() == b"--frame":
            break

    headers = {}
    while True:
        line = response.raw.readline()
        if not line:
            raise EOFError("MJPEG headers ended")
        if line in (b"\r\n", b"\n"):
            break
        key, value = line.decode("latin-1").split(":", 1)
        headers[key.lower()] = value.strip()

    length = int(headers["content-length"])
    jpeg = response.raw.read(length)
    if len(jpeg) != length:
        raise EOFError("MJPEG frame was truncated")
    response.raw.read(2)
    return jpeg


def verify_and_measure(jpeg, publisher):
    image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.shape[:2] != (480, 640):
        raise RuntimeError("Transport changed or corrupted the 640x480 JPEG")
    published_at = publisher.published_at(jpeg)
    if published_at is None:
        raise RuntimeError("Received a frame that was not produced by this benchmark")
    return (time.monotonic() - published_at) * 1000.0, len(jpeg)


def run_mjpeg(base_url, token, publisher, duration_s, warmup_s, delay_s):
    response = requests.get(
        base_url + f"/stream/0.mjpg?token={quote(token)}",
        stream=True,
        timeout=(5.0, 10.0),
    )
    response.raise_for_status()
    ages = []
    sizes = []
    start = time.monotonic()
    measured_start = start + warmup_s
    deadline = measured_start + duration_s
    try:
        while time.monotonic() < deadline:
            jpeg = read_mjpeg_frame(response)
            age_ms, size = verify_and_measure(jpeg, publisher)
            now = time.monotonic()
            if now >= measured_start:
                ages.append(age_ms)
                sizes.append(size)
            if delay_s > 0:
                time.sleep(delay_s)
    finally:
        response.close()
    return summarize("mjpeg", ages, sizes, duration_s)


def websocket_url(base_url, token):
    parsed = urlparse(base_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    return f"{scheme}://{parsed.netloc}/ws/stream/0?token={quote(token)}"


def run_websocket(base_url, token, publisher, duration_s, warmup_s, delay_s):
    socket = Client.connect(websocket_url(base_url, token))
    ages = []
    sizes = []
    start = time.monotonic()
    measured_start = start + warmup_s
    deadline = measured_start + duration_s
    try:
        while time.monotonic() < deadline:
            jpeg = socket.receive(timeout=10.0)
            if jpeg is None:
                raise TimeoutError("Timed out waiting for WebSocket frame")
            age_ms, size = verify_and_measure(jpeg, publisher)
            now = time.monotonic()
            if now >= measured_start:
                ages.append(age_ms)
                sizes.append(size)
            if delay_s > 0:
                time.sleep(delay_s)
            socket.send("next")
    finally:
        socket.close()
    return summarize("websocket", ages, sizes, duration_s)


def percentile(values, fraction):
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def summarize(transport, ages, sizes, duration_s):
    if not ages:
        raise RuntimeError(f"{transport} produced no measured frames")
    return {
        "transport": transport,
        "frames": len(ages),
        "display_fps": len(ages) / duration_s,
        "median_age_ms": statistics.median(ages),
        "p95_age_ms": percentile(ages, 0.95),
        "max_age_ms": max(ages),
        "mean_jpeg_bytes": statistics.mean(sizes),
        "resolution": "640x480",
        "jpeg_quality": 80,
    }


def run_case(base_url, manager, publisher, transport, duration_s, warmup_s, delay_s):
    token = claim_session(base_url)
    try:
        if transport == "mjpeg":
            return run_mjpeg(
                base_url,
                token,
                publisher,
                duration_s,
                warmup_s,
                delay_s,
            )
        return run_websocket(
            base_url,
            token,
            publisher,
            duration_s,
            warmup_s,
            delay_s,
        )
    finally:
        release_session(base_url, token)
        manager.drop_stale()
        time.sleep(0.25)


def print_results(label, results):
    print(f"\n{label}")
    print(
        "transport  frames  shown_fps  median_age_ms  p95_age_ms  "
        "max_age_ms  mean_JPEG_KB"
    )
    for result in results:
        print(
            f"{result['transport']:<10}"
            f"{result['frames']:>8}"
            f"{result['display_fps']:>11.2f}"
            f"{result['median_age_ms']:>15.1f}"
            f"{result['p95_age_ms']:>12.1f}"
            f"{result['max_age_ms']:>12.1f}"
            f"{result['mean_jpeg_bytes'] / 1024:>14.1f}"
        )


def main():
    args = parse_args()
    if (
        args.duration <= 0
        or args.warmup < 0
        or args.slow_delay < 0
        or args.simulation_fps <= 0
    ):
        raise ValueError("Benchmark durations must be non-negative and duration positive")

    manager = SessionManager(num_sessions=1, heartbeat_timeout_s=300.0)
    store = MjpegFrameStore(jpeg_quality=80)
    app = create_app(manager, store)
    server, server_thread = start_flask_server(app, "127.0.0.1", 0)
    port = server.socket.getsockname()[1]
    origin_url = f"http://127.0.0.1:{port}"
    tunnel_process = None
    publisher = FramePublisher(store, fps=args.simulation_fps)
    publisher.start()
    try:
        if args.cloudflare:
            tunnel_process, base_url = start_quick_tunnel(origin_url)
            wait_until_reachable(base_url)
            path_label = "Cloudflare Quick Tunnel"
        else:
            base_url = origin_url
            path_label = "Local loopback"

        all_results = []
        for case_name, delay_s in (("normal consumer", 0.0), ("slow consumer", args.slow_delay)):
            case_results = []
            for transport in ("mjpeg", "websocket"):
                result = run_case(
                    base_url,
                    manager,
                    publisher,
                    transport,
                    args.duration,
                    args.warmup,
                    delay_s,
                )
                result["case"] = case_name
                result["processing_delay_s"] = delay_s
                case_results.append(result)
                all_results.append(result)
            print_results(f"{path_label}: {case_name}", case_results)

        report = {
            "path": path_label,
            "base_url": base_url,
            "simulation_fps": args.simulation_fps,
            "delivery": "adaptive demand-driven",
            "resolution": "640x480",
            "jpeg_quality": 80,
            "results": all_results,
        }
        if args.json_output is not None:
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
            args.json_output.write_text(
                json.dumps(report, indent=2) + "\n",
                encoding="utf-8",
            )
    finally:
        publisher.close()
        if tunnel_process is not None:
            tunnel_process.terminate()
            try:
                tunnel_process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                tunnel_process.kill()
                tunnel_process.wait(timeout=5.0)
        server.shutdown()
        server_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
