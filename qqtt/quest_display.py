from __future__ import annotations

from collections import deque
import json
import mmap
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import threading
import time
from typing import Optional

import numpy as np
import torch

from qqtt.live_openxr import ControllerPoseSample, LiveControllerSample


class OpenXRFramePanelMirror:
    HEADER_STRUCT = struct.Struct("<8sIIIIIIQQ16x")
    HEADER_MAGIC = b"BOBAQST1"
    HEADER_VERSION = 1
    SLOT_COUNT = 2
    STAGING_BUFFER_COUNT = 2

    def __init__(self, repo_root: Path, width: int, height: int):
        self.repo_root = Path(repo_root)
        self.width = int(width)
        self.height = int(height)
        self.channels = 4
        self.frame_bytes = self.width * self.height * self.channels
        self.shared_frame_path: Optional[Path] = None
        self._shared_file = None
        self._shared_mmap: Optional[mmap.mmap] = None
        self._slot_views: list[np.ndarray] = []
        self._frame_counter = 0
        pin_memory = torch.cuda.is_available()
        self._cpu_stage_buffers = [
            torch.empty(
                (self.height, self.width, self.channels),
                dtype=torch.uint8,
                device="cpu",
                pin_memory=pin_memory,
            )
            for _ in range(self.STAGING_BUFFER_COUNT)
        ]
        self._cpu_stage_arrays = [buffer.numpy() for buffer in self._cpu_stage_buffers]
        self._staging_copy_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        self._pending_stage_copies: deque[dict] = deque()
        self._next_stage_index = 0

        self.binary_path = self.repo_root / "linux_pose_probe" / "openxr_frame_panel"
        self.build_script_path = self.repo_root / "linux_pose_probe" / "build_openxr_frame_panel.sh"
        self.source_path = self.repo_root / "linux_pose_probe" / "openxr_frame_panel.cpp"
        self.process: Optional[subprocess.Popen[str]] = None
        self._stdout_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._stdout_tail: deque[str] = deque(maxlen=100)
        self._stderr_tail: deque[str] = deque(maxlen=100)
        self._parse_errors: deque[str] = deque(maxlen=20)
        self._latest_sample: Optional[LiveControllerSample] = None
        self._latest_lock = threading.Lock()
        self._exit_logged = False

    def start(self) -> None:
        rebuilt_binary = self._ensure_binary()
        self._create_shared_frame_file()

        env = os.environ.copy()
        runtime_json = self._default_runtime_json_path()
        if runtime_json is not None:
            env.setdefault("XR_RUNTIME_JSON", runtime_json)

        steamvr_lib_dir = self._default_steamvr_lib_dir()
        if steamvr_lib_dir is not None:
            current_ld = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = (
                f"{steamvr_lib_dir}:{current_ld}" if current_ld else steamvr_lib_dir
            )

        assert self.shared_frame_path is not None
        self.process = subprocess.Popen(
            [str(self.binary_path), "--frame-path", str(self.shared_frame_path)],
            cwd=self.repo_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._stdout_thread = threading.Thread(
            target=self._read_stdout,
            daemon=True,
        )
        self._stderr_thread = threading.Thread(
            target=self._read_stderr,
            daemon=True,
        )
        self._stdout_thread.start()
        self._stderr_thread.start()
        print(
            f"[quest_display] started OpenXR frame panel pid={self.process.pid} "
            f"shared_frame={self.shared_frame_path}",
            flush=True,
        )
        print(
            "[quest_display] viewer binary status: "
            f"rebuilt={int(rebuilt_binary)} "
            f"binary_mtime={int(self.binary_path.stat().st_mtime)} "
            f"source_mtime={int(self.source_path.stat().st_mtime)}",
            flush=True,
        )

        time.sleep(0.5)
        if self.process.poll() is not None:
            raise RuntimeError(
                "Quest frame panel exited during startup.\n" + self.debug_summary()
            )

    def stop(self) -> None:
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5.0)
            self.process = None
        self._exit_logged = False
        self._drain_pending_stage_copies(block=True)

        if self._shared_mmap is not None:
            self._shared_mmap.close()
            self._shared_mmap = None
        if self._shared_file is not None:
            self._shared_file.close()
            self._shared_file = None
        if self.shared_frame_path is not None:
            try:
                self.shared_frame_path.unlink(missing_ok=True)
            except TypeError:
                if self.shared_frame_path.exists():
                    self.shared_frame_path.unlink()
            self.shared_frame_path = None

    def publish_frame(self, frame_rgba: torch.Tensor) -> tuple[bool, dict[str, float]]:
        timing = {
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
        publish_start = time.perf_counter()
        if self._shared_mmap is None:
            raise RuntimeError("Quest frame panel shared buffer is not initialized.")
        if frame_rgba.shape != (self.height, self.width, self.channels):
            raise ValueError(
                f"Quest mirror frame shape {tuple(frame_rgba.shape)} != "
                f"({self.height}, {self.width}, {self.channels})"
            )
        if frame_rgba.dtype != torch.uint8:
            raise ValueError(f"Quest mirror frame dtype {frame_rgba.dtype} != torch.uint8")
        process_check_start = time.perf_counter()
        if self.process is not None and self.process.poll() is not None:
            if not self._exit_logged:
                print(
                    "[quest_display] frame panel exited unexpectedly; "
                    "disabling Quest publishing for this run.\n"
                    + self.debug_summary(),
                    flush=True,
                )
                self._exit_logged = True
            timing["process_check_wall"] = time.perf_counter() - process_check_start
            timing["total_wall"] = time.perf_counter() - publish_start
            return False, timing
        timing["process_check_wall"] = time.perf_counter() - process_check_start

        slot = self._frame_counter % self.SLOT_COUNT
        frame_id = self._frame_counter + 1
        self._frame_counter = frame_id
        drain_nonblock_start = time.perf_counter()
        drain_nonblock_stats = self._drain_pending_stage_copies(block=False)
        timing["pending_drain_nonblock_wall"] = time.perf_counter() - drain_nonblock_start
        timing["gpu_to_cpu_wait_wall"] += drain_nonblock_stats["wait_wall"]
        timing["gpu_to_cpu_copy_cuda"] += drain_nonblock_stats["gpu_to_cpu_copy_cuda"]
        timing["cpu_mmap_copy_wall"] += drain_nonblock_stats["cpu_mmap_copy_wall"]
        timing["header_write_wall"] += drain_nonblock_stats["header_write_wall"]

        if self._staging_copy_stream is None:
            fallback_start = time.perf_counter()
            stage_array = self._cpu_stage_arrays[0]
            np.copyto(stage_array, frame_rgba.cpu().numpy())
            commit_stats = self._commit_stage_array(stage_array, slot=slot, frame_id=frame_id)
            timing["fallback_copy_wall"] = time.perf_counter() - fallback_start
            timing["cpu_mmap_copy_wall"] += commit_stats["cpu_mmap_copy_wall"]
            timing["header_write_wall"] += commit_stats["header_write_wall"]
            timing["total_wall"] = time.perf_counter() - publish_start
            return True, timing

        if len(self._pending_stage_copies) >= self.STAGING_BUFFER_COUNT:
            drain_block_start = time.perf_counter()
            drain_block_stats = self._drain_pending_stage_copies(block=True)
            timing["pending_drain_block_wall"] = time.perf_counter() - drain_block_start
            timing["gpu_to_cpu_wait_wall"] += drain_block_stats["wait_wall"]
            timing["gpu_to_cpu_copy_cuda"] += drain_block_stats["gpu_to_cpu_copy_cuda"]
            timing["cpu_mmap_copy_wall"] += drain_block_stats["cpu_mmap_copy_wall"]
            timing["header_write_wall"] += drain_block_stats["header_write_wall"]

        stage_index = self._next_stage_index
        stage_buffer = self._cpu_stage_buffers[stage_index]
        stage_array = self._cpu_stage_arrays[stage_index]
        copy_start_event = torch.cuda.Event(enable_timing=True)
        copy_end_event = torch.cuda.Event(enable_timing=True)
        enqueue_start = time.perf_counter()
        with torch.cuda.stream(self._staging_copy_stream):
            copy_start_event.record(self._staging_copy_stream)
            stage_buffer.copy_(frame_rgba, non_blocking=True)
            copy_end_event.record(self._staging_copy_stream)
        timing["stage_enqueue_wall"] = time.perf_counter() - enqueue_start
        self._pending_stage_copies.append(
            {
                "start_event": copy_start_event,
                "end_event": copy_end_event,
                "stage_array": stage_array,
                "slot": slot,
                "frame_id": frame_id,
            }
        )
        self._next_stage_index = (stage_index + 1) % self.STAGING_BUFFER_COUNT
        timing["total_wall"] = time.perf_counter() - publish_start
        return True, timing

    def _drain_pending_stage_copies(self, block: bool) -> dict[str, float]:
        stats = {
            "wait_wall": 0.0,
            "gpu_to_cpu_copy_cuda": 0.0,
            "cpu_mmap_copy_wall": 0.0,
            "header_write_wall": 0.0,
        }
        while self._pending_stage_copies:
            pending = self._pending_stage_copies[0]
            end_event = pending["end_event"]
            if not block and not end_event.query():
                break
            wait_start = time.perf_counter()
            end_event.synchronize()
            stats["wait_wall"] += time.perf_counter() - wait_start
            stats["gpu_to_cpu_copy_cuda"] += (
                pending["start_event"].elapsed_time(end_event) / 1000.0
            )
            commit_stats = self._commit_stage_array(
                pending["stage_array"],
                slot=pending["slot"],
                frame_id=pending["frame_id"],
            )
            stats["cpu_mmap_copy_wall"] += commit_stats["cpu_mmap_copy_wall"]
            stats["header_write_wall"] += commit_stats["header_write_wall"]
            self._pending_stage_copies.popleft()
        return stats

    def _commit_stage_array(self, stage_array: np.ndarray, slot: int, frame_id: int) -> dict[str, float]:
        copy_start = time.perf_counter()
        np.copyto(self._slot_views[slot], stage_array)
        copy_wall = time.perf_counter() - copy_start
        header_start = time.perf_counter()
        self._write_header(latest_frame_id=frame_id, latest_slot=slot)
        header_wall = time.perf_counter() - header_start
        return {
            "cpu_mmap_copy_wall": copy_wall,
            "header_write_wall": header_wall,
        }

    def wait_for_sample(self, timeout: float = 10.0) -> LiveControllerSample:
        deadline = time.time() + timeout
        while time.time() < deadline:
            sample = self.get_latest_sample()
            if sample is not None:
                return sample
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(
                    "Quest frame panel exited before producing controller data.\n"
                    + self.debug_summary()
                )
            time.sleep(0.05)
        raise RuntimeError(
            "Timed out waiting for Quest panel controller sample.\n" + self.debug_summary()
        )

    def get_latest_sample(self) -> Optional[LiveControllerSample]:
        with self._latest_lock:
            return self._latest_sample

    def debug_summary(self) -> str:
        parts = []
        if self._stdout_tail:
            parts.append("stdout:\n" + "".join(self._stdout_tail).strip())
        if self._stderr_tail:
            parts.append("stderr:\n" + "".join(self._stderr_tail).strip())
        if self._parse_errors:
            parts.append("parse errors:\n" + "\n".join(self._parse_errors))
        return "\n\n".join(part for part in parts if part) or "no panel diagnostics available"

    def _ensure_binary(self) -> bool:
        if self.binary_path.exists() and self.binary_path.stat().st_mtime >= max(
            self.build_script_path.stat().st_mtime,
            self.source_path.stat().st_mtime,
        ):
            return False

        subprocess.run(
            ["bash", str(self.build_script_path)],
            cwd=self.repo_root,
            check=True,
            text=True,
        )
        return True

    def _create_shared_frame_file(self) -> None:
        total_bytes = self.HEADER_STRUCT.size + self.SLOT_COUNT * self.frame_bytes
        fd, path = tempfile.mkstemp(
            prefix="boba_quest_frame_",
            suffix=".bin",
            dir="/tmp",
        )
        self.shared_frame_path = Path(path)
        self._shared_file = os.fdopen(fd, "r+b", buffering=0)
        self._shared_file.truncate(total_bytes)
        self._shared_mmap = mmap.mmap(self._shared_file.fileno(), total_bytes)
        self._write_header(latest_frame_id=0, latest_slot=0)
        self._slot_views = []
        for slot_index in range(self.SLOT_COUNT):
            offset = self.HEADER_STRUCT.size + slot_index * self.frame_bytes
            self._slot_views.append(
                np.ndarray(
                    (self.height, self.width, self.channels),
                    dtype=np.uint8,
                    buffer=self._shared_mmap,
                    offset=offset,
                )
            )
            self._slot_views[-1].fill(0)

    def _write_header(self, latest_frame_id: int, latest_slot: int) -> None:
        assert self._shared_mmap is not None
        self.HEADER_STRUCT.pack_into(
            self._shared_mmap,
            0,
            self.HEADER_MAGIC,
            self.HEADER_VERSION,
            self.width,
            self.height,
            self.channels,
            self.frame_bytes,
            self.SLOT_COUNT,
            int(latest_frame_id),
            int(latest_slot),
        )

    def _read_stdout(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        for line in self.process.stdout:
            self._stdout_tail.append(line)
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
                sample = self._parse_sample(payload)
            except Exception as exc:
                self._parse_errors.append(f"{exc}: {stripped}")
                continue
            with self._latest_lock:
                self._latest_sample = sample

    def _read_stderr(self) -> None:
        assert self.process is not None and self.process.stderr is not None
        for line in self.process.stderr:
            self._stderr_tail.append(line)
            print(f"[quest_display_viewer:stderr] {line.rstrip()}", flush=True)

    @staticmethod
    def _parse_sample(payload: dict) -> LiveControllerSample:
        return LiveControllerSample(
            sample=int(payload["sample"]),
            left=OpenXRFramePanelMirror._parse_controller(payload["left"]),
            right=OpenXRFramePanelMirror._parse_controller(payload["right"]),
        )

    @staticmethod
    def _parse_controller(payload: dict) -> ControllerPoseSample:
        position = np.asarray(payload["position"], dtype=np.float32)
        orientation = np.asarray(payload["orientation"], dtype=np.float32)
        if position.shape != (3,):
            raise ValueError(f"controller position shape {position.shape} != (3,)")
        if orientation.shape != (4,):
            raise ValueError(f"controller orientation shape {orientation.shape} != (4,)")
        return ControllerPoseSample(
            source=str(payload["source"]),
            active=bool(payload["active"]),
            position_valid=bool(payload["position_valid"]),
            orientation_valid=bool(payload["orientation_valid"]),
            position_tracked=bool(payload["position_tracked"]),
            orientation_tracked=bool(payload["orientation_tracked"]),
            position=position,
            orientation=orientation,
            select_available=bool(payload["select_available"]),
            select_pressed=bool(payload["select_pressed"]),
            select_value=float(payload["select_value"]),
            select_source=str(payload["select_source"]),
            anchor_cycle_available=bool(payload["anchor_cycle_available"]),
            anchor_cycle_pressed=bool(payload["anchor_cycle_pressed"]),
            anchor_cycle_source=str(payload["anchor_cycle_source"]),
            snap_assist_available=bool(payload["snap_assist_available"]),
            snap_assist_pressed=bool(payload["snap_assist_pressed"]),
            snap_assist_source=str(payload["snap_assist_source"]),
        )

    @staticmethod
    def _default_runtime_json_path() -> Optional[str]:
        candidate = (
            Path.home()
            / ".local"
            / "share"
            / "Steam"
            / "steamapps"
            / "common"
            / "SteamVR"
            / "steamxr_linux64.json"
        )
        return str(candidate) if candidate.exists() else None

    @staticmethod
    def _default_steamvr_lib_dir() -> Optional[str]:
        candidate = (
            Path.home()
            / ".local"
            / "share"
            / "Steam"
            / "steamapps"
            / "common"
            / "SteamVR"
            / "bin"
            / "linux64"
        )
        return str(candidate) if candidate.exists() else None
