from __future__ import annotations

from collections import deque
import ctypes
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

from qqtt.live_openxr import (
    ControllerPoseSample,
    EyeFovSample,
    EyePoseSample,
    LiveControllerSample,
    LiveImmersiveSample,
    _ensure_jsoncpp_compat_dir,
    _prepend_env_path,
    parse_controller_payload,
)


class OpenXRFramePanelMirror:
    HEADER_STRUCT = struct.Struct("<8sIIIIIIQQII8x")
    HEADER_MAGIC = b"BOBAQST1"
    HEADER_VERSION = 2
    SLOT_COUNT = 2
    STAGING_BUFFER_COUNT = 2
    FRESHNESS_FIRST_COMMIT = False
    BRIDGE_PUBLISH_SAMPLE_BYTE_COUNT = 16
    PRESENTATION_MODE_STEREO_FULLSCREEN = 0
    PRESENTATION_MODE_MONO_PANEL = 1

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
        self._staging_copy_stream = (
            torch.cuda.Stream(priority=-1) if torch.cuda.is_available() else None
        )
        self._direct_commit_stream = (
            torch.cuda.Stream(priority=-1) if torch.cuda.is_available() else None
        )
        self._cuda_device_index = (
            int(torch.cuda.current_device()) if torch.cuda.is_available() else None
        )
        self._pending_stage_copies: deque[dict] = deque()
        self._next_stage_index = 0
        self._active_stage_copy: Optional[dict] = None
        self._pending_stage_copy: Optional[dict] = None
        self._reserved_stage_copies: dict[int, dict] = {}
        self._retired_stage_copies: list[dict] = []
        self._free_stage_indices: deque[int] = deque(range(self.STAGING_BUFFER_COUNT))
        self._next_commit_slot = 0
        self._bridge_transition_trace: deque[str] = deque(maxlen=64)
        self._stage_commit_failure: Optional[str] = None
        self._stage_condition = threading.Condition()
        self._commit_thread: Optional[threading.Thread] = None
        self._commit_stop_requested = False
        self._direct_commit_enabled = False
        self._direct_commit_warning: Optional[str] = None
        self._direct_commit_cudart = None
        self._direct_commit_registered_slots: list[tuple[int, int]] = []
        self._bridge_publish_sample_indices = np.zeros((0,), dtype=np.int64)
        self._refresh_bridge_publish_sample_indices()
        self._bridge_publish_sample_check_enabled = True
        self._bridge_transition_trace_enabled = True

        self.binary_path = self.repo_root / "linux_pose_probe" / "boba_immersive_demo"
        self.build_script_path = self.repo_root / "linux_pose_probe" / "build_boba_immersive_demo.sh"
        self.source_path = self.repo_root / "linux_pose_probe" / "openxr_frame_panel.cpp"
        self.process: Optional[subprocess.Popen[str]] = None
        self._stdout_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._stdout_tail: deque[str] = deque(maxlen=100)
        self._stderr_tail: deque[str] = deque(maxlen=100)
        self._parse_errors: deque[str] = deque(maxlen=20)
        self._latest_sample: Optional[LiveControllerSample] = None
        self._latest_lock = threading.Lock()
        self._sample_condition = threading.Condition(self._latest_lock)
        self._viewer_stats_lock = threading.Lock()
        self._exit_logged = False
        self._last_published_frame_id = 0
        self._reset_viewer_source_stats()
        self._reset_viewer_render_stats()
        self._reset_bridge_commit_stats()
        self._reset_steady_state_metrics_epoch()

    def _normalize_presentation_mode(self, presentation_mode=None) -> int:
        if presentation_mode is None:
            return int(self.PRESENTATION_MODE_STEREO_FULLSCREEN)
        if isinstance(presentation_mode, str):
            mode_key = presentation_mode.strip().lower()
            if mode_key in ("stereo_fullscreen", "stereo", "fullscreen"):
                return int(self.PRESENTATION_MODE_STEREO_FULLSCREEN)
            if mode_key in ("mono_panel", "panel", "mono"):
                return int(self.PRESENTATION_MODE_MONO_PANEL)
            raise ValueError(f"Unsupported immersive presentation mode: {presentation_mode}")
        mode_value = int(presentation_mode)
        if mode_value not in (
            int(self.PRESENTATION_MODE_STEREO_FULLSCREEN),
            int(self.PRESENTATION_MODE_MONO_PANEL),
        ):
            raise ValueError(f"Unsupported immersive presentation mode: {presentation_mode}")
        return mode_value

    def _refresh_bridge_publish_sample_indices(self) -> None:
        total_bytes = max(0, int(self.frame_bytes))
        sample_count = min(self.BRIDGE_PUBLISH_SAMPLE_BYTE_COUNT, total_bytes)
        if sample_count <= 0:
            self._bridge_publish_sample_indices = np.zeros((0,), dtype=np.int64)
            return
        if sample_count == 1:
            self._bridge_publish_sample_indices = np.array([0], dtype=np.int64)
            return
        self._bridge_publish_sample_indices = np.linspace(
            0,
            total_bytes - 1,
            num=sample_count,
            dtype=np.int64,
        )

    def _capture_bridge_publish_sample_bytes_from_tensor(
        self,
        frame_tensor: torch.Tensor,
    ) -> Optional[bytes]:
        if not self._bridge_publish_sample_check_enabled:
            return None
        sample_indices = self._bridge_publish_sample_indices
        if sample_indices.size == 0:
            return None
        flat = frame_tensor.contiguous().reshape(-1)
        if flat.numel() < int(sample_indices[-1]) + 1:
            return None
        index_tensor = torch.as_tensor(
            sample_indices,
            dtype=torch.long,
            device=flat.device,
        )
        sample_tensor = flat.index_select(0, index_tensor)
        return np.asarray(
            sample_tensor.detach().cpu().numpy(),
            dtype=np.uint8,
        ).tobytes()

    def _capture_bridge_publish_sample_bytes_from_stereo_tensors(
        self,
        left_frame_tensor: torch.Tensor,
        right_frame_tensor: torch.Tensor,
    ) -> Optional[bytes]:
        if not self._bridge_publish_sample_check_enabled:
            return None
        sample_indices = self._bridge_publish_sample_indices
        if sample_indices.size == 0:
            return None
        left_flat = left_frame_tensor.contiguous().reshape(-1)
        right_flat = right_frame_tensor.contiguous().reshape(-1)
        left_count = int(left_flat.numel())
        total_count = left_count + int(right_flat.numel())
        if total_count < int(sample_indices[-1]) + 1:
            return None
        sample_tensors = []
        left_mask = sample_indices < left_count
        if np.any(left_mask):
            left_indices = torch.as_tensor(
                sample_indices[left_mask],
                dtype=torch.long,
                device=left_flat.device,
            )
            sample_tensors.append(left_flat.index_select(0, left_indices))
        if np.any(~left_mask):
            right_indices = torch.as_tensor(
                sample_indices[~left_mask] - left_count,
                dtype=torch.long,
                device=right_flat.device,
            )
            sample_tensors.append(right_flat.index_select(0, right_indices))
        if not sample_tensors:
            return None
        sample_tensor = (
            sample_tensors[0]
            if len(sample_tensors) == 1
            else torch.cat(sample_tensors, dim=0)
        )
        return np.asarray(
            sample_tensor.detach().cpu().numpy(),
            dtype=np.uint8,
        ).tobytes()

    def _capture_bridge_publish_sample_bytes_from_array(
        self,
        frame_array: np.ndarray,
    ) -> Optional[bytes]:
        if not self._bridge_publish_sample_check_enabled:
            return None
        sample_indices = self._bridge_publish_sample_indices
        if sample_indices.size == 0:
            return None
        flat = np.asarray(frame_array, dtype=np.uint8).reshape(-1)
        if flat.size < int(sample_indices[-1]) + 1:
            return None
        return np.asarray(flat[sample_indices], dtype=np.uint8).tobytes()

    def _record_bridge_publish_sample_check(
        self,
        *,
        expected_sample_bytes: Optional[bytes],
        actual_sample_bytes: Optional[bytes],
    ) -> None:
        if not self._bridge_publish_sample_check_enabled:
            return
        if expected_sample_bytes is None or actual_sample_bytes is None:
            return
        mismatch = expected_sample_bytes != actual_sample_bytes
        with self._stage_condition:
            self._bridge_publish_sample_check_count += 1
            if mismatch:
                self._bridge_publish_sample_mismatch_count += 1
            if self._steady_state_bridge_epoch_active:
                self._steady_state_bridge_publish_sample_check_count += 1
                if mismatch:
                    self._steady_state_bridge_publish_sample_mismatch_count += 1

    def configure_runtime_diagnostics(self, *, enabled: bool) -> None:
        enabled = bool(enabled)
        with self._stage_condition:
            self._bridge_publish_sample_check_enabled = enabled
            self._bridge_transition_trace_enabled = enabled
            if not enabled:
                self._bridge_transition_trace.clear()

    def start(self) -> None:
        rebuilt_binary = self._ensure_binary()
        self._create_shared_frame_file()
        self._initialize_direct_commit_path()

        env = os.environ.copy()
        runtime_json = self._default_runtime_json_path()
        if runtime_json is not None:
            env.setdefault("XR_RUNTIME_JSON", runtime_json)

        steamvr_lib_dir = self._default_steamvr_lib_dir()
        if steamvr_lib_dir is not None:
            _prepend_env_path(env, "LD_LIBRARY_PATH", steamvr_lib_dir)
        jsoncpp_compat_dir = _ensure_jsoncpp_compat_dir(self.repo_root)
        if jsoncpp_compat_dir is not None:
            _prepend_env_path(env, "LD_LIBRARY_PATH", jsoncpp_compat_dir)

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
            f"[quest_display] started Boba Immersive Demo pid={self.process.pid} "
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
        if self._direct_commit_warning is not None:
            print(
                "[quest_display] immersive bridge direct commit disabled: "
                f"{self._direct_commit_warning}",
                flush=True,
            )

        time.sleep(0.5)
        if self.process.poll() is not None:
            raise RuntimeError(
                "Boba Immersive Demo exited during startup.\n" + self.debug_summary()
            )
        self._start_stage_commit_thread()

    def stop(self) -> None:
        self._stop_stage_commit_thread()
        self._teardown_direct_commit_path()
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
        self._last_published_frame_id = 0
        self._reset_viewer_source_stats()
        self._reset_viewer_render_stats()
        self._reset_bridge_commit_stats()
        self._reset_steady_state_metrics_epoch()
        with self._stage_condition:
            self._pending_stage_copies.clear()
            self._active_stage_copy = None
            self._pending_stage_copy = None
            self._reserved_stage_copies = {}
            self._retired_stage_copies = []
            self._free_stage_indices = deque(range(self.STAGING_BUFFER_COUNT))
            self._next_stage_index = 0
            self._next_commit_slot = 0
            self._bridge_transition_trace.clear()
            self._stage_commit_failure = None

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
            raise RuntimeError("Boba Immersive Demo shared buffer is not initialized.")
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
                    "[quest_display] Boba Immersive Demo exited unexpectedly; "
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
        submit_wall_s = time.perf_counter()
        with self._stage_condition:
            self._raise_if_stage_commit_failed_locked()
            self._record_bridge_submit_locked(
                frame_id=frame_id,
                submit_wall_s=submit_wall_s,
            )
            if self.FRESHNESS_FIRST_COMMIT:
                self._trace_bridge_transition_locked(
                    "submit",
                    frame_id=frame_id,
                    slot=slot,
                )

        if self._staging_copy_stream is None:
            fallback_start = time.perf_counter()
            stage_array = self._cpu_stage_arrays[0]
            np.copyto(stage_array, frame_rgba.cpu().numpy())
            commit_stats = self._commit_stage_array(stage_array, slot=slot, frame_id=frame_id)
            timing["fallback_copy_wall"] = time.perf_counter() - fallback_start
            timing["cpu_mmap_copy_wall"] += commit_stats["cpu_mmap_copy_wall"]
            timing["header_write_wall"] += commit_stats["header_write_wall"]
            if self.FRESHNESS_FIRST_COMMIT:
                commit_record = {
                    "wait_wall": 0.0,
                    "gpu_to_cpu_copy_cuda": 0.0,
                    "cpu_mmap_copy_wall": commit_stats["cpu_mmap_copy_wall"],
                    "header_write_wall": commit_stats["header_write_wall"],
                    "total_wall": timing["fallback_copy_wall"],
                    "submit_to_commit_start_ms": 0.0,
                    "commit_wait_for_gpu_ready_ms": 0.0,
                    "commit_thread_wake_delay_ms": 0.0,
                    "commit_active_service_ms": timing["fallback_copy_wall"] * 1000.0,
                }
                with self._stage_condition:
                    self._record_bridge_commit_locked(
                        frame_id=frame_id,
                        commit_stats=commit_record,
                    )
            timing["total_wall"] = time.perf_counter() - publish_start
            self._last_published_frame_id = frame_id
            return True, timing

        timing["pending_drain_block_wall"] = self._wait_for_stage_capacity(
            incoming_frame_id=frame_id
        )
        timing["stage_enqueue_wall"] = self._enqueue_stage_copy(
            frame_tensor=frame_rgba,
            slot=slot,
            frame_id=frame_id,
            submit_wall_s=submit_wall_s,
        )
        timing["total_wall"] = time.perf_counter() - publish_start
        self._last_published_frame_id = frame_id
        return True, timing

    def _drain_pending_stage_copies(self, block: bool) -> dict[str, float]:
        stats = {
            "wait_wall": 0.0,
            "gpu_to_cpu_copy_cuda": 0.0,
            "cpu_mmap_copy_wall": 0.0,
            "header_write_wall": 0.0,
        }
        if self.FRESHNESS_FIRST_COMMIT:
            if not block:
                return stats
            self.wait_for_bridge_idle()
            return stats
        while True:
            with self._stage_condition:
                if not self._pending_stage_copies:
                    break
                pending = self._pending_stage_copies[0]
            end_event = pending["end_event"]
            if not block and not end_event.query():
                break
            commit_stats = self._commit_pending_stage_copy(pending)
            stats["cpu_mmap_copy_wall"] += commit_stats["cpu_mmap_copy_wall"]
            stats["header_write_wall"] += commit_stats["header_write_wall"]
            stats["wait_wall"] += commit_stats["wait_wall"]
            stats["gpu_to_cpu_copy_cuda"] += commit_stats["gpu_to_cpu_copy_cuda"]
            with self._stage_condition:
                if self._pending_stage_copies and self._pending_stage_copies[0] is pending:
                    self._pending_stage_copies.popleft()
                self._stage_condition.notify_all()
        return stats

    def _commit_stage_array(
        self,
        stage_array: np.ndarray,
        slot: int,
        frame_id: int,
        *,
        expected_publish_sample_bytes: Optional[bytes] = None,
        presentation_mode: Optional[int] = None,
    ) -> dict[str, float]:
        copy_start = time.perf_counter()
        slot_view = self._slot_views[slot]
        np.copyto(slot_view, stage_array)
        copy_wall = time.perf_counter() - copy_start
        actual_publish_sample_bytes = self._capture_bridge_publish_sample_bytes_from_array(
            slot_view
        )
        self._record_bridge_publish_sample_check(
            expected_sample_bytes=expected_publish_sample_bytes,
            actual_sample_bytes=actual_publish_sample_bytes,
        )
        header_start = time.perf_counter()
        self._write_header(
            latest_frame_id=frame_id,
            latest_slot=slot,
            presentation_mode=presentation_mode,
        )
        header_wall = time.perf_counter() - header_start
        return {
            "cpu_mmap_copy_wall": copy_wall,
            "header_write_wall": header_wall,
        }

    def _commit_pending_stage_copy(self, pending: dict) -> dict[str, float]:
        if bool(pending.get("direct_commit", False)):
            return self._commit_pending_stage_copy_direct(pending)
        total_start = time.perf_counter()
        wait_start = time.perf_counter()
        pending["end_event"].synchronize()
        wait_wall = time.perf_counter() - wait_start
        gpu_to_cpu_copy_cuda = float(
            pending["start_event"].elapsed_time(pending["end_event"])
        )
        commit_stats = self._commit_stage_array(
            pending["stage_array"],
            slot=pending["slot"],
            frame_id=pending["frame_id"],
            expected_publish_sample_bytes=pending.get(
                "expected_publish_sample_bytes"
            ),
            presentation_mode=pending.get("presentation_mode"),
        )
        submit_to_commit_start_ms = float(
            pending.get("submit_to_commit_start_ms", 0.0)
        )
        commit_thread_wake_delay_ms = float(
            pending.get("commit_thread_wake_delay_ms", 0.0)
        )
        commit_active_service_ms = 1000.0 * (
            time.perf_counter() - total_start
        )
        return {
            "wait_wall": wait_wall,
            "gpu_to_cpu_copy_cuda": gpu_to_cpu_copy_cuda,
            "cpu_mmap_copy_wall": commit_stats["cpu_mmap_copy_wall"],
            "header_write_wall": commit_stats["header_write_wall"],
            "total_wall": time.perf_counter() - total_start,
            "submit_to_commit_start_ms": submit_to_commit_start_ms,
            "commit_wait_for_gpu_ready_ms": wait_wall * 1000.0,
            "commit_thread_wake_delay_ms": commit_thread_wake_delay_ms,
            "commit_active_service_ms": commit_active_service_ms,
        }

    def _wait_for_stage_capacity(self, incoming_frame_id: Optional[int] = None) -> float:
        wait_start = time.perf_counter()
        blocked = False
        with self._stage_condition:
            self._raise_if_stage_commit_failed_locked()
            while True:
                if self.FRESHNESS_FIRST_COMMIT:
                    reclaimed_count = self._reclaim_completed_retired_stage_copies_locked()
                    if reclaimed_count > 0:
                        self._finalize_freshness_scheduler_mutation_locked(
                            "reclaim_retired",
                            reclaimed_count=reclaimed_count,
                        )
                    if self._pending_stage_copy is not None:
                        self._retire_pending_stage_copy_locked(
                            incoming_frame_id=incoming_frame_id,
                            reason="submit_pre_capacity",
                        )
                    if self._free_stage_indices:
                        break
                    if not blocked:
                        blocked = True
                        self._bridge_capacity_block_count += 1
                        if self._steady_state_bridge_epoch_active:
                            self._steady_state_bridge_capacity_block_count += 1
                        self._trace_bridge_transition_locked(
                            "capacity_wait_begin",
                            frame_id=incoming_frame_id,
                        )
                elif len(self._pending_stage_copies) < self.STAGING_BUFFER_COUNT:
                    break
                self._stage_condition.wait(timeout=0.05)
                self._raise_if_stage_commit_failed_locked()
        wait_wall = time.perf_counter() - wait_start
        if self.FRESHNESS_FIRST_COMMIT and blocked:
            with self._stage_condition:
                self._bridge_capacity_wait_ms_sum += wait_wall * 1000.0
                if self._steady_state_bridge_epoch_active:
                    self._steady_state_bridge_capacity_wait_ms_sum += (
                        wait_wall * 1000.0
                    )
                self._trace_bridge_transition_locked(
                    "capacity_wait_end",
                    frame_id=incoming_frame_id,
                    wait_ms=f"{wait_wall * 1000.0:.3f}",
                )
        return wait_wall

    def _enqueue_stage_copy(
        self,
        frame_tensor: torch.Tensor,
        slot: int,
        frame_id: int,
        submit_wall_s: Optional[float] = None,
        expected_publish_sample_bytes: Optional[bytes] = None,
    ) -> float:
        if submit_wall_s is None:
            submit_wall_s = time.perf_counter()
        stage_index: Optional[int] = None
        if self.FRESHNESS_FIRST_COMMIT:
            with self._stage_condition:
                stage_index = self._reserve_freshness_stage_copy_locked(
                    frame_id=frame_id,
                    slot=slot,
                )
        else:
            stage_index = self._next_stage_index
        try:
            stage_buffer = self._cpu_stage_buffers[stage_index]
            stage_array = self._cpu_stage_arrays[stage_index]
            producer_ready_event = torch.cuda.Event()
            enqueue_start = time.perf_counter()
            producer_ready_event.record(torch.cuda.current_stream())
            copy_start_event = None
            copy_end_event = None
            if not self._direct_commit_enabled:
                copy_start_event = torch.cuda.Event(enable_timing=True)
                copy_end_event = torch.cuda.Event(enable_timing=True)
                with torch.cuda.stream(self._staging_copy_stream):
                    self._staging_copy_stream.wait_event(producer_ready_event)
                    copy_start_event.record(self._staging_copy_stream)
                    stage_buffer.copy_(frame_tensor, non_blocking=True)
                    copy_end_event.record(self._staging_copy_stream)
            with self._stage_condition:
                pending = {
                    "slot": slot,
                    "frame_id": frame_id,
                    "stage_index": stage_index,
                    "submit_wall_s": submit_wall_s,
                    "expected_publish_sample_bytes": expected_publish_sample_bytes,
                }
                if self._direct_commit_enabled:
                    pending.update(
                        {
                            "producer_ready_event": producer_ready_event,
                            "frame_tensor": frame_tensor.contiguous(),
                            "direct_commit": True,
                        }
                    )
                else:
                    pending.update(
                        {
                            "start_event": copy_start_event,
                            "end_event": copy_end_event,
                            "stage_array": stage_array,
                            "direct_commit": False,
                        }
                    )
                if self.FRESHNESS_FIRST_COMMIT:
                    if self._active_stage_copy is None:
                        reserved = self._reserved_stage_copies.pop(stage_index, None)
                        if reserved is None:
                            raise RuntimeError(
                                "freshness bridge invariant failed: reserved stage missing "
                                f"during enqueue_active (stage_index={stage_index})\n"
                                f"{self._scheduler_snapshot_locked()}\nrecent transitions:\n"
                                + "\n".join(self._bridge_transition_trace)
                            )
                        self._mark_stage_copy_active_locked(pending)
                        self._active_stage_copy = pending
                        self._finalize_freshness_scheduler_mutation_locked(
                            "enqueue_active",
                            frame_id=frame_id,
                            stage_index=stage_index,
                        )
                    else:
                        if self._pending_stage_copy is not None:
                            self._retire_pending_stage_copy_locked(
                                incoming_frame_id=frame_id,
                                reason="enqueue_replace",
                            )
                        reserved = self._reserved_stage_copies.pop(stage_index, None)
                        if reserved is None:
                            raise RuntimeError(
                                "freshness bridge invariant failed: reserved stage missing "
                                f"during enqueue_pending (stage_index={stage_index})\n"
                                f"{self._scheduler_snapshot_locked()}\nrecent transitions:\n"
                                + "\n".join(self._bridge_transition_trace)
                            )
                        self._pending_stage_copy = pending
                        self._finalize_freshness_scheduler_mutation_locked(
                            "enqueue_pending",
                            frame_id=frame_id,
                            stage_index=stage_index,
                        )
                else:
                    self._pending_stage_copies.append(pending)
                    self._next_stage_index = (stage_index + 1) % self.STAGING_BUFFER_COUNT
                self._stage_condition.notify_all()
            return time.perf_counter() - enqueue_start
        except BaseException:
            if self.FRESHNESS_FIRST_COMMIT and stage_index is not None:
                with self._stage_condition:
                    self._abort_reserved_stage_copy_locked(
                        stage_index=stage_index,
                        frame_id=frame_id,
                        reason="enqueue_prepare_exception",
                    )
            raise

    @staticmethod
    def _cudart_status_code(result) -> int:
        if isinstance(result, tuple):
            if not result:
                return 0
            return int(result[0])
        return int(result)

    def _cuda_stream_handle(self, stream: Optional[torch.cuda.Stream]) -> int:
        if stream is None:
            return 0
        stream_handle = getattr(stream, "cuda_stream", None)
        if callable(stream_handle):
            stream_handle = stream_handle()
        if stream_handle is None:
            return 0
        return int(stream_handle)

    def _cudart_host_register(self, ptr: int, size_bytes: int) -> None:
        if self._direct_commit_cudart is None:
            raise RuntimeError("CUDA runtime API unavailable for host registration")
        last_error = None
        arg_variants = (
            (ctypes.c_void_p(int(ptr)), ctypes.c_size_t(int(size_bytes)), ctypes.c_uint(0)),
            (int(ptr), int(size_bytes), 0),
        )
        for args in arg_variants:
            try:
                status = self._cudart_status_code(
                    self._direct_commit_cudart.cudaHostRegister(*args)
                )
                if status == 0:
                    return
                last_error = RuntimeError(
                    f"cudaHostRegister failed with status={status}"
                )
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"{last_error}")

    def _cudart_host_unregister(self, ptr: int) -> None:
        if self._direct_commit_cudart is None:
            return
        arg_variants = (
            (ctypes.c_void_p(int(ptr)),),
            (int(ptr),),
        )
        last_error = None
        for args in arg_variants:
            try:
                status = self._cudart_status_code(
                    self._direct_commit_cudart.cudaHostUnregister(*args)
                )
                if status == 0:
                    return
                last_error = RuntimeError(
                    f"cudaHostUnregister failed with status={status}"
                )
            except Exception as exc:
                last_error = exc
        if last_error is not None:
            raise RuntimeError(f"{last_error}")

    def _cudart_memcpy_device_to_host_async(
        self,
        *,
        dst_ptr: int,
        src_ptr: int,
        size_bytes: int,
        stream: Optional[torch.cuda.Stream],
    ) -> None:
        if self._direct_commit_cudart is None:
            raise RuntimeError("CUDA runtime API unavailable for cudaMemcpyAsync")
        stream_handle = self._cuda_stream_handle(stream)
        arg_variants = (
            (
                ctypes.c_void_p(int(dst_ptr)),
                ctypes.c_void_p(int(src_ptr)),
                ctypes.c_size_t(int(size_bytes)),
                ctypes.c_int(2),
                ctypes.c_void_p(int(stream_handle)),
            ),
            (
                int(dst_ptr),
                int(src_ptr),
                int(size_bytes),
                2,
                int(stream_handle),
            ),
        )
        last_error = None
        for args in arg_variants:
            try:
                status = self._cudart_status_code(
                    self._direct_commit_cudart.cudaMemcpyAsync(*args)
                )
                if status == 0:
                    return
                last_error = RuntimeError(
                    f"cudaMemcpyAsync failed with status={status}"
                )
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"{last_error}")

    def _initialize_direct_commit_path(self) -> None:
        self._teardown_direct_commit_path()
        if not self.FRESHNESS_FIRST_COMMIT:
            self._direct_commit_warning = None
            return
        if (
            not torch.cuda.is_available()
            or self._shared_mmap is None
            or self._direct_commit_stream is None
        ):
            self._direct_commit_warning = "CUDA direct commit prerequisites unavailable"
            return
        cudart_factory = getattr(torch.cuda, "cudart", None)
        if not callable(cudart_factory):
            self._direct_commit_warning = "torch.cuda.cudart() is unavailable"
            return
        try:
            self._direct_commit_cudart = cudart_factory()
        except Exception as exc:
            self._direct_commit_warning = f"failed to load cudart: {exc}"
            self._direct_commit_cudart = None
            return
        try:
            mapping_ptr = int(
                ctypes.addressof(ctypes.c_char.from_buffer(self._shared_mmap))
            )
            mapping_size = int(len(self._shared_mmap))
            self._cudart_host_register(mapping_ptr, mapping_size)
            self._direct_commit_registered_slots.append((mapping_ptr, mapping_size))
            # Warm up the direct path once so runtime failures fall back before gameplay.
            warmup_tensor = torch.zeros(
                (1,),
                dtype=torch.uint8,
                device=f"cuda:{self._cuda_device_index}",
            )
            warmup_start = torch.cuda.Event(enable_timing=True)
            warmup_end = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(self._direct_commit_stream):
                warmup_start.record(self._direct_commit_stream)
                self._cudart_memcpy_device_to_host_async(
                    dst_ptr=int(self._slot_views[0].ctypes.data),
                    src_ptr=int(warmup_tensor.data_ptr()),
                    size_bytes=1,
                    stream=self._direct_commit_stream,
                )
                warmup_end.record(self._direct_commit_stream)
            warmup_end.synchronize()
            self._direct_commit_enabled = True
            self._direct_commit_warning = None
        except Exception as exc:
            self._direct_commit_warning = f"{type(exc).__name__}: {exc}"
            self._teardown_direct_commit_path()

    def _teardown_direct_commit_path(self) -> None:
        registered_slots = list(self._direct_commit_registered_slots)
        self._direct_commit_registered_slots = []
        for slot_ptr, _ in reversed(registered_slots):
            try:
                self._cudart_host_unregister(slot_ptr)
            except Exception:
                pass
        self._direct_commit_enabled = False
        self._direct_commit_cudart = None

    def _commit_pending_stage_copy_direct(self, pending: dict) -> dict[str, float]:
        if not self._direct_commit_enabled:
            raise RuntimeError("Direct immersive commit path is not enabled")
        slot = int(pending["slot"])
        slot_view = self._slot_views[slot]
        frame_tensors = pending.get("frame_tensors")
        frame_tensor = pending.get("frame_tensor")
        if frame_tensors is not None:
            if (
                not isinstance(frame_tensors, (tuple, list))
                or len(frame_tensors) != 2
                or not torch.is_tensor(frame_tensors[0])
                or not torch.is_tensor(frame_tensors[1])
            ):
                raise RuntimeError(
                    "Direct immersive commit is missing stereo frame tensors"
                )
        elif frame_tensor is None or not torch.is_tensor(frame_tensor):
            raise RuntimeError("Direct immersive commit is missing frame tensor")
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        wait_start = time.perf_counter()
        with torch.cuda.stream(self._direct_commit_stream):
            producer_ready_event = pending.get("producer_ready_event")
            if producer_ready_event is not None:
                self._direct_commit_stream.wait_event(producer_ready_event)
            start_event.record(self._direct_commit_stream)
            if frame_tensors is not None:
                left_frame_tensor, right_frame_tensor = frame_tensors
                self._cudart_memcpy_device_to_host_async(
                    dst_ptr=int(slot_view[0].ctypes.data),
                    src_ptr=int(left_frame_tensor.data_ptr()),
                    size_bytes=int(slot_view[0].nbytes),
                    stream=self._direct_commit_stream,
                )
                self._cudart_memcpy_device_to_host_async(
                    dst_ptr=int(slot_view[1].ctypes.data),
                    src_ptr=int(right_frame_tensor.data_ptr()),
                    size_bytes=int(slot_view[1].nbytes),
                    stream=self._direct_commit_stream,
                )
            else:
                self._cudart_memcpy_device_to_host_async(
                    dst_ptr=int(slot_view.ctypes.data),
                    src_ptr=int(frame_tensor.data_ptr()),
                    size_bytes=int(slot_view.nbytes),
                    stream=self._direct_commit_stream,
                )
            end_event.record(self._direct_commit_stream)
        end_event.synchronize()
        wait_wall = time.perf_counter() - wait_start
        direct_copy_ms = float(start_event.elapsed_time(end_event))
        actual_publish_sample_bytes = self._capture_bridge_publish_sample_bytes_from_array(
            slot_view
        )
        self._record_bridge_publish_sample_check(
            expected_sample_bytes=pending.get("expected_publish_sample_bytes"),
            actual_sample_bytes=actual_publish_sample_bytes,
        )
        header_start = time.perf_counter()
        self._write_header(
            latest_frame_id=int(pending["frame_id"]),
            latest_slot=slot,
            presentation_mode=pending.get("presentation_mode"),
        )
        header_wall = time.perf_counter() - header_start
        submit_to_commit_start_ms = float(
            pending.get("submit_to_commit_start_ms", 0.0)
        )
        commit_thread_wake_delay_ms = float(
            pending.get("commit_thread_wake_delay_ms", 0.0)
        )
        return {
            "wait_wall": wait_wall,
            "gpu_to_cpu_copy_cuda": direct_copy_ms,
            "cpu_mmap_copy_wall": 0.0,
            "header_write_wall": header_wall,
            "total_wall": wait_wall + header_wall,
            "direct_path": True,
            "direct_copy_ms": direct_copy_ms,
            "submit_to_commit_start_ms": submit_to_commit_start_ms,
            "commit_wait_for_gpu_ready_ms": wait_wall * 1000.0,
            "commit_thread_wake_delay_ms": commit_thread_wake_delay_ms,
            "commit_active_service_ms": (wait_wall + header_wall) * 1000.0,
        }

    def _start_stage_commit_thread(self) -> None:
        if self._staging_copy_stream is None or self._commit_thread is not None:
            return
        self._commit_stop_requested = False
        self._commit_thread = threading.Thread(
            target=self._stage_commit_thread_main,
            name=f"{self.__class__.__name__}Commit",
            daemon=True,
        )
        self._commit_thread.start()

    def _stop_stage_commit_thread(self) -> None:
        if self._commit_thread is None:
            return
        with self._stage_condition:
            self._commit_stop_requested = True
            self._stage_condition.notify_all()
        self._commit_thread.join()
        self._commit_thread = None
        self._commit_stop_requested = False

    def _stage_commit_thread_main(self) -> None:
        if self._cuda_device_index is not None:
            torch.cuda.set_device(self._cuda_device_index)
        try:
            while True:
                with self._stage_condition:
                    self._raise_if_stage_commit_failed_locked()
                    if self.FRESHNESS_FIRST_COMMIT:
                        reclaimed_count = self._reclaim_completed_retired_stage_copies_locked()
                        if reclaimed_count > 0:
                            self._finalize_freshness_scheduler_mutation_locked(
                                "reclaim_retired",
                                reclaimed_count=reclaimed_count,
                            )
                        while self._active_stage_copy is None and not self._commit_stop_requested:
                            self._stage_condition.wait()
                            self._raise_if_stage_commit_failed_locked()
                            reclaimed_count = self._reclaim_completed_retired_stage_copies_locked()
                            if reclaimed_count > 0:
                                self._finalize_freshness_scheduler_mutation_locked(
                                    "reclaim_retired",
                                    reclaimed_count=reclaimed_count,
                                )
                        if self._active_stage_copy is None and self._commit_stop_requested:
                            break
                        pending = self._active_stage_copy
                        slot = self._next_commit_slot
                        self._next_commit_slot = (self._next_commit_slot + 1) % self.SLOT_COUNT
                        commit_begin_wall_s = time.perf_counter()
                        submit_wall_s = float(
                            pending.get("submit_wall_s", commit_begin_wall_s)
                        )
                        active_ready_wall_s = float(
                            pending.get("active_ready_wall_s", submit_wall_s)
                        )
                        pending["submit_to_commit_start_ms"] = max(
                            0.0,
                            (commit_begin_wall_s - submit_wall_s) * 1000.0,
                        )
                        pending["commit_thread_wake_delay_ms"] = max(
                            0.0,
                            (commit_begin_wall_s - active_ready_wall_s) * 1000.0,
                        )
                        self._trace_bridge_transition_locked(
                            "active_commit_begin",
                            frame_id=int(pending.get("frame_id", -1)),
                            stage_index=int(pending.get("stage_index", -1)),
                            slot=slot,
                        )
                    else:
                        while not self._pending_stage_copies and not self._commit_stop_requested:
                            self._stage_condition.wait()
                        if not self._pending_stage_copies and self._commit_stop_requested:
                            break
                        pending = self._pending_stage_copies[0]
                        slot = pending["slot"]
                try:
                    pending["slot"] = slot
                    commit_stats = self._commit_pending_stage_copy(pending)
                except Exception as exc:
                    self._parse_errors.append(
                        f"stage commit failure frame_id={pending.get('frame_id')}: {exc}"
                    )
                    commit_stats = None
                finally:
                    with self._stage_condition:
                        if self.FRESHNESS_FIRST_COMMIT:
                            self._complete_active_stage_commit_locked(
                                pending,
                                commit_stats,
                            )
                            reclaimed_count = self._reclaim_completed_retired_stage_copies_locked()
                            if reclaimed_count > 0:
                                self._finalize_freshness_scheduler_mutation_locked(
                                    "reclaim_retired",
                                    reclaimed_count=reclaimed_count,
                                )
                        else:
                            if self._pending_stage_copies and self._pending_stage_copies[0] is pending:
                                self._pending_stage_copies.popleft()
                        self._stage_condition.notify_all()
        except BaseException as exc:
            with self._stage_condition:
                if self._stage_commit_failure is None:
                    scheduler_snapshot = self._scheduler_snapshot_locked()
                    transition_trace = "\n".join(self._bridge_transition_trace)
                    self._stage_commit_failure = (
                        "Freshness-first immersive bridge commit thread failed: "
                        f"{type(exc).__name__}: {exc}\n{scheduler_snapshot}\n"
                        f"recent transitions:\n{transition_trace}"
                    )
                    self._parse_errors.append(self._stage_commit_failure)
                self._stage_condition.notify_all()

    def _reset_viewer_source_stats(self) -> None:
        with self._viewer_stats_lock:
            self._viewer_latest_source_frame_id = 0
            self._viewer_applied_update_count = 0
            self._viewer_source_frame_delta_count = 0
            self._viewer_coalesced_source_frame_count = 0
            self._viewer_source_elapsed_s = 0.0
            self._viewer_recent_applied_update_fps = 0.0
            self._viewer_recent_source_delta_fps = 0.0

    def _reset_viewer_render_stats(self) -> None:
        with self._viewer_stats_lock:
            self._viewer_rendered_frame_count = 0
            self._viewer_render_elapsed_s = 0.0
            self._viewer_recent_render_fps = 0.0

    def _reset_bridge_commit_stats(self) -> None:
        with self._stage_condition:
            self._bridge_submitted_frame_count = 0
            self._bridge_submit_first_wall_s = 0.0
            self._bridge_submit_latest_wall_s = 0.0
            self._bridge_submit_elapsed_s = 0.0
            self._bridge_submit_latest_frame_id = 0
            self._bridge_committed_update_count = 0
            self._bridge_committed_source_frame_delta_count = 0
            self._bridge_dropped_pending_count = 0
            self._bridge_commit_direct_count = 0
            self._bridge_commit_fallback_count = 0
            self._bridge_pending_replace_count = 0
            self._bridge_active_promote_count = 0
            self._bridge_capacity_block_count = 0
            self._bridge_capacity_wait_ms_sum = 0.0
            self._bridge_commit_latest_frame_id = 0
            self._bridge_commit_latest_wall_s = 0.0
            self._bridge_commit_elapsed_s = 0.0
            self._bridge_commit_first_wall_s = 0.0
            self._bridge_commit_gpu_to_cpu_ms_sum = 0.0
            self._bridge_commit_direct_copy_ms_sum = 0.0
            self._bridge_commit_cpu_mmap_ms_sum = 0.0
            self._bridge_commit_header_write_ms_sum = 0.0
            self._bridge_commit_total_ms_sum = 0.0
            self._bridge_submit_to_commit_start_ms_sum = 0.0
            self._bridge_commit_wait_for_gpu_ready_ms_sum = 0.0
            self._bridge_commit_thread_wake_delay_ms_sum = 0.0
            self._bridge_commit_active_service_ms_sum = 0.0
            self._bridge_commit_queue_max_depth = 0
            self._bridge_retired_copying_max_depth = 0
            self._bridge_physical_in_use_max_depth = 0
            self._bridge_free_stage_min = int(self.STAGING_BUFFER_COUNT)
            self._bridge_publish_sample_check_count = 0
            self._bridge_publish_sample_mismatch_count = 0

    def _reset_steady_state_metrics_epoch(self) -> None:
        self._reset_steady_state_viewer_stats()
        self._reset_steady_state_bridge_commit_stats()

    def _reset_steady_state_viewer_stats(self) -> None:
        with self._viewer_stats_lock:
            self._steady_state_viewer_epoch_active = False
            self._steady_state_viewer_epoch_sync_pending = False
            self._steady_state_viewer_epoch_wall_s = 0.0
            self._steady_state_viewer_frame_id_boundary = 0
            self._steady_state_viewer_last_seen_frame_id = 0
            self._steady_state_viewer_latest_frame_id = 0
            self._steady_state_viewer_applied_update_count = 0
            self._steady_state_viewer_source_frame_delta_count = 0
            self._steady_state_viewer_coalesced_source_frame_count = 0
            self._steady_state_viewer_render_baseline_count = 0
            self._steady_state_viewer_epoch_baseline_applied_update_count = 0
            self._steady_state_viewer_epoch_baseline_source_frame_delta_count = 0
            self._steady_state_viewer_epoch_baseline_coalesced_source_frame_count = 0
            self._steady_state_viewer_epoch_baseline_latest_frame_id = 0
            self._steady_state_viewer_last_parsed_applied_update_count = 0
            self._steady_state_viewer_last_parsed_source_frame_delta_count = 0
            self._steady_state_viewer_last_parsed_coalesced_source_frame_count = 0
            self._steady_state_viewer_last_parsed_latest_frame_id = 0
            self._steady_state_viewer_accounting_inconsistency_count = 0

    def _reset_steady_state_bridge_commit_stats(self) -> None:
        with self._stage_condition:
            self._steady_state_bridge_epoch_active = False
            self._steady_state_bridge_epoch_wall_s = 0.0
            self._steady_state_bridge_submitted_frame_count = 0
            self._steady_state_bridge_submit_latest_wall_s = 0.0
            self._steady_state_bridge_submit_latest_frame_id = 0
            self._steady_state_bridge_committed_update_count = 0
            self._steady_state_bridge_committed_source_frame_delta_count = 0
            self._steady_state_bridge_dropped_pending_count = 0
            self._steady_state_bridge_commit_direct_count = 0
            self._steady_state_bridge_commit_fallback_count = 0
            self._steady_state_bridge_pending_replace_count = 0
            self._steady_state_bridge_active_promote_count = 0
            self._steady_state_bridge_capacity_block_count = 0
            self._steady_state_bridge_capacity_wait_ms_sum = 0.0
            self._steady_state_bridge_commit_latest_frame_id = 0
            self._steady_state_bridge_commit_latest_wall_s = 0.0
            self._steady_state_bridge_commit_gpu_to_cpu_ms_sum = 0.0
            self._steady_state_bridge_commit_direct_copy_ms_sum = 0.0
            self._steady_state_bridge_commit_cpu_mmap_ms_sum = 0.0
            self._steady_state_bridge_commit_header_write_ms_sum = 0.0
            self._steady_state_bridge_commit_total_ms_sum = 0.0
            self._steady_state_bridge_submit_to_commit_start_ms_sum = 0.0
            self._steady_state_bridge_commit_wait_for_gpu_ready_ms_sum = 0.0
            self._steady_state_bridge_commit_thread_wake_delay_ms_sum = 0.0
            self._steady_state_bridge_commit_active_service_ms_sum = 0.0
            self._steady_state_bridge_commit_queue_max_depth = 0
            self._steady_state_bridge_retired_copying_max_depth = 0
            self._steady_state_bridge_physical_in_use_max_depth = 0
            self._steady_state_bridge_free_stage_min = int(self.STAGING_BUFFER_COUNT)
            self._steady_state_bridge_publish_sample_check_count = 0
            self._steady_state_bridge_publish_sample_mismatch_count = 0

    def begin_steady_state_metrics_epoch(self, *, frame_id_boundary: int = 0) -> None:
        epoch_wall_s = time.perf_counter()
        with self._stage_condition:
            self._steady_state_bridge_epoch_active = True
            self._steady_state_bridge_epoch_wall_s = float(epoch_wall_s)
            self._steady_state_bridge_submitted_frame_count = 0
            self._steady_state_bridge_submit_latest_wall_s = 0.0
            self._steady_state_bridge_submit_latest_frame_id = 0
            self._steady_state_bridge_committed_update_count = 0
            self._steady_state_bridge_committed_source_frame_delta_count = 0
            self._steady_state_bridge_dropped_pending_count = 0
            self._steady_state_bridge_commit_direct_count = 0
            self._steady_state_bridge_commit_fallback_count = 0
            self._steady_state_bridge_pending_replace_count = 0
            self._steady_state_bridge_active_promote_count = 0
            self._steady_state_bridge_capacity_block_count = 0
            self._steady_state_bridge_capacity_wait_ms_sum = 0.0
            self._steady_state_bridge_commit_latest_frame_id = 0
            self._steady_state_bridge_commit_latest_wall_s = 0.0
            self._steady_state_bridge_commit_gpu_to_cpu_ms_sum = 0.0
            self._steady_state_bridge_commit_direct_copy_ms_sum = 0.0
            self._steady_state_bridge_commit_cpu_mmap_ms_sum = 0.0
            self._steady_state_bridge_commit_header_write_ms_sum = 0.0
            self._steady_state_bridge_commit_total_ms_sum = 0.0
            self._steady_state_bridge_submit_to_commit_start_ms_sum = 0.0
            self._steady_state_bridge_commit_wait_for_gpu_ready_ms_sum = 0.0
            self._steady_state_bridge_commit_thread_wake_delay_ms_sum = 0.0
            self._steady_state_bridge_commit_active_service_ms_sum = 0.0
            self._steady_state_bridge_commit_queue_max_depth = 0
            self._steady_state_bridge_retired_copying_max_depth = 0
            self._steady_state_bridge_physical_in_use_max_depth = 0
            self._steady_state_bridge_free_stage_min = int(len(self._free_stage_indices))
            self._steady_state_bridge_publish_sample_check_count = 0
            self._steady_state_bridge_publish_sample_mismatch_count = 0
        with self._viewer_stats_lock:
            baseline_applied_count = int(self._viewer_applied_update_count)
            baseline_source_frame_delta_count = int(
                self._viewer_source_frame_delta_count
            )
            baseline_coalesced_count = int(self._viewer_coalesced_source_frame_count)
            baseline_latest_frame_id = int(self._viewer_latest_source_frame_id)
            boundary = max(
                int(frame_id_boundary),
                baseline_latest_frame_id,
            )
            self._steady_state_viewer_epoch_active = True
            self._steady_state_viewer_epoch_sync_pending = True
            self._steady_state_viewer_epoch_wall_s = float(epoch_wall_s)
            self._steady_state_viewer_frame_id_boundary = boundary
            self._steady_state_viewer_last_seen_frame_id = baseline_latest_frame_id
            self._steady_state_viewer_latest_frame_id = boundary
            self._steady_state_viewer_applied_update_count = 0
            self._steady_state_viewer_source_frame_delta_count = 0
            self._steady_state_viewer_coalesced_source_frame_count = 0
            self._steady_state_viewer_epoch_baseline_applied_update_count = (
                baseline_applied_count
            )
            self._steady_state_viewer_epoch_baseline_source_frame_delta_count = (
                baseline_source_frame_delta_count
            )
            self._steady_state_viewer_epoch_baseline_coalesced_source_frame_count = (
                baseline_coalesced_count
            )
            self._steady_state_viewer_epoch_baseline_latest_frame_id = (
                baseline_latest_frame_id
            )
            self._steady_state_viewer_last_parsed_applied_update_count = (
                baseline_applied_count
            )
            self._steady_state_viewer_last_parsed_source_frame_delta_count = (
                baseline_source_frame_delta_count
            )
            self._steady_state_viewer_last_parsed_coalesced_source_frame_count = (
                baseline_coalesced_count
            )
            self._steady_state_viewer_last_parsed_latest_frame_id = (
                baseline_latest_frame_id
            )
            self._steady_state_viewer_accounting_inconsistency_count = 0
            self._steady_state_viewer_render_baseline_count = int(
                self._viewer_rendered_frame_count
            )

    def _record_bridge_submit_locked(
        self,
        *,
        frame_id: int,
        submit_wall_s: float,
    ) -> None:
        if frame_id <= 0:
            return
        if self._bridge_submitted_frame_count == 0:
            self._bridge_submit_first_wall_s = float(submit_wall_s)
        self._bridge_submitted_frame_count += 1
        self._bridge_submit_latest_frame_id = int(frame_id)
        self._bridge_submit_latest_wall_s = float(submit_wall_s)
        self._bridge_submit_elapsed_s = max(
            0.0,
            float(submit_wall_s) - float(self._bridge_submit_first_wall_s),
        )
        if self._steady_state_bridge_epoch_active:
            self._steady_state_bridge_submitted_frame_count += 1
            self._steady_state_bridge_submit_latest_frame_id = int(frame_id)
            self._steady_state_bridge_submit_latest_wall_s = float(submit_wall_s)

    def _mark_stage_copy_active_locked(self, pending: dict) -> None:
        pending["active_ready_wall_s"] = time.perf_counter()

    def _free_stage_index_locked(self, stage_index: int, *, context: str) -> None:
        if stage_index < 0:
            return
        if self.FRESHNESS_FIRST_COMMIT and stage_index in self._reserved_stage_copies:
            trace = "\n".join(self._bridge_transition_trace)
            raise RuntimeError(
                "freshness bridge invariant failed: reserved stage released directly "
                f"(stage_index={stage_index}) during {context}\n"
                f"{self._scheduler_snapshot_locked()}\nrecent transitions:\n{trace}"
            )
        if stage_index in self._free_stage_indices:
            if self.FRESHNESS_FIRST_COMMIT:
                trace = "\n".join(self._bridge_transition_trace)
                raise RuntimeError(
                    "freshness bridge invariant failed: stage released twice "
                    f"(stage_index={stage_index}) during {context}\n"
                    f"{self._scheduler_snapshot_locked()}\nrecent transitions:\n{trace}"
                )
            return
        self._free_stage_indices.append(stage_index)

    def _stage_copy_brief_locked(self, pending: Optional[dict]) -> str:
        if pending is None:
            return "none"
        frame_id = int(pending.get("frame_id", -1))
        stage_index = int(pending.get("stage_index", -1))
        slot = int(pending.get("slot", -1))
        return f"f{frame_id}@stage{stage_index}/slot{slot}"

    def _scheduler_snapshot_locked(self) -> str:
        reserved = ", ".join(
            self._stage_copy_brief_locked(pending)
            for _, pending in sorted(self._reserved_stage_copies.items())
        )
        if not reserved:
            reserved = "none"
        retired = ", ".join(
            self._stage_copy_brief_locked(pending) for pending in self._retired_stage_copies
        )
        if not retired:
            retired = "none"
        free_stage_indices = list(self._free_stage_indices)
        free_stage_indices.sort()
        return (
            f"active={self._stage_copy_brief_locked(self._active_stage_copy)} "
            f"pending={self._stage_copy_brief_locked(self._pending_stage_copy)} "
            f"reserved=[{reserved}] "
            f"retired=[{retired}] "
            f"free={free_stage_indices}"
        )

    def _trace_bridge_transition_locked(self, event: str, **fields: object) -> None:
        if not self._bridge_transition_trace_enabled:
            return
        tokens = [str(event)]
        for key, value in fields.items():
            if value is None:
                continue
            tokens.append(f"{key}={value}")
        tokens.append(self._scheduler_snapshot_locked())
        self._bridge_transition_trace.append(" ".join(tokens))

    def _check_freshness_scheduler_invariants_locked(self, context: str) -> None:
        if not self.FRESHNESS_FIRST_COMMIT:
            return
        occupied: dict[int, str] = {}
        buckets = {
            "active": [self._active_stage_copy] if self._active_stage_copy is not None else [],
            "pending": [self._pending_stage_copy] if self._pending_stage_copy is not None else [],
            "reserved": list(self._reserved_stage_copies.values()),
            "retired_copying": list(self._retired_stage_copies),
        }
        for bucket_name, entries in buckets.items():
            if bucket_name in {"active", "pending"} and len(entries) > 1:
                raise RuntimeError(f"freshness bridge invariant failed: {bucket_name} > 1")
            for entry in entries:
                stage_index = int(entry.get("stage_index", -1))
                if stage_index < 0:
                    continue
                if stage_index in occupied:
                    previous = occupied[stage_index]
                    trace = "\n".join(self._bridge_transition_trace)
                    raise RuntimeError(
                        "freshness bridge invariant failed: stage index reused across buckets "
                        f"({stage_index}: {previous} and {bucket_name}) during {context}\n"
                        f"{self._scheduler_snapshot_locked()}\nrecent transitions:\n{trace}"
                    )
                occupied[stage_index] = bucket_name
        free_stage_indices = list(self._free_stage_indices)
        if len(free_stage_indices) != len(set(free_stage_indices)):
            trace = "\n".join(self._bridge_transition_trace)
            raise RuntimeError(
                "freshness bridge invariant failed: duplicate free stage index "
                f"during {context}\n{self._scheduler_snapshot_locked()}\n"
                f"recent transitions:\n{trace}"
            )
        if any(stage_index in occupied for stage_index in free_stage_indices):
            trace = "\n".join(self._bridge_transition_trace)
            raise RuntimeError(
                "freshness bridge invariant failed: free stage also occupied "
                f"during {context}\n{self._scheduler_snapshot_locked()}\n"
                f"recent transitions:\n{trace}"
            )
        total_accounted = len(free_stage_indices) + len(occupied)
        if total_accounted != int(self.STAGING_BUFFER_COUNT):
            trace = "\n".join(self._bridge_transition_trace)
            raise RuntimeError(
                "freshness bridge invariant failed: stage accounting mismatch "
                f"({total_accounted} != {self.STAGING_BUFFER_COUNT}) during {context}\n"
                f"{self._scheduler_snapshot_locked()}\nrecent transitions:\n{trace}"
            )

    def _update_bridge_scheduler_depth_stats_locked(self) -> None:
        current_depth = int(self._active_stage_copy is not None) + int(
            self._pending_stage_copy is not None
        )
        reserved_depth = len(self._reserved_stage_copies)
        self._bridge_commit_queue_max_depth = max(
            self._bridge_commit_queue_max_depth,
            current_depth,
        )
        retired_depth = len(self._retired_stage_copies)
        self._bridge_retired_copying_max_depth = max(
            self._bridge_retired_copying_max_depth,
            retired_depth,
        )
        self._bridge_physical_in_use_max_depth = max(
            self._bridge_physical_in_use_max_depth,
            current_depth + reserved_depth + retired_depth,
        )
        self._bridge_free_stage_min = min(
            self._bridge_free_stage_min,
            len(self._free_stage_indices),
        )
        if self._steady_state_bridge_epoch_active:
            self._steady_state_bridge_commit_queue_max_depth = max(
                self._steady_state_bridge_commit_queue_max_depth,
                current_depth,
            )
            retired_depth = len(self._retired_stage_copies)
            self._steady_state_bridge_retired_copying_max_depth = max(
                self._steady_state_bridge_retired_copying_max_depth,
                retired_depth,
            )
            self._steady_state_bridge_physical_in_use_max_depth = max(
                self._steady_state_bridge_physical_in_use_max_depth,
                current_depth + reserved_depth + retired_depth,
            )
            self._steady_state_bridge_free_stage_min = min(
                self._steady_state_bridge_free_stage_min,
                len(self._free_stage_indices),
            )

    def _finalize_freshness_scheduler_mutation_locked(
        self, event: str, **fields: object
    ) -> None:
        self._trace_bridge_transition_locked(event, **fields)
        self._update_bridge_scheduler_depth_stats_locked()
        self._check_freshness_scheduler_invariants_locked(context=event)
        self._stage_condition.notify_all()

    def _raise_if_stage_commit_failed_locked(self) -> None:
        if self._stage_commit_failure is None:
            return
        raise RuntimeError(self._stage_commit_failure)

    def _reserve_freshness_stage_copy_locked(
        self,
        *,
        frame_id: int,
        slot: int,
    ) -> int:
        self._raise_if_stage_commit_failed_locked()
        reclaimed_count = self._reclaim_completed_retired_stage_copies_locked()
        if reclaimed_count > 0:
            self._finalize_freshness_scheduler_mutation_locked(
                "reclaim_retired",
                reclaimed_count=reclaimed_count,
            )
        if not self._free_stage_indices:
            raise RuntimeError(
                "Freshness-first immersive bridge ran out of staging buffers.\n"
                f"{self._scheduler_snapshot_locked()}\nrecent transitions:\n"
                + "\n".join(self._bridge_transition_trace)
            )
        stage_index = int(self._free_stage_indices.popleft())
        self._reserved_stage_copies[stage_index] = {
            "frame_id": int(frame_id),
            "slot": int(slot),
            "stage_index": int(stage_index),
        }
        self._finalize_freshness_scheduler_mutation_locked(
            "reserve_stage",
            frame_id=frame_id,
            slot=slot,
            stage_index=stage_index,
        )
        return stage_index

    def _abort_reserved_stage_copy_locked(
        self,
        *,
        stage_index: int,
        frame_id: Optional[int],
        reason: str,
    ) -> bool:
        reserved = self._reserved_stage_copies.pop(int(stage_index), None)
        if reserved is None:
            return False
        self._free_stage_index_locked(
            int(stage_index),
            context=f"reservation_abort:{reason}",
        )
        self._finalize_freshness_scheduler_mutation_locked(
            "reservation_abort",
            frame_id=frame_id,
            reason=reason,
            stage_index=stage_index,
        )
        return True

    def _retire_stage_copy_locked(self, pending: dict, *, context: str) -> bool:
        end_event = pending.get("end_event")
        if end_event is None or end_event.query():
            self._free_stage_index_locked(
                int(pending.get("stage_index", -1)),
                context=context,
            )
            return True
        self._retired_stage_copies.append(pending)
        return False

    def _retire_pending_stage_copy_locked(
        self,
        *,
        incoming_frame_id: Optional[int],
        reason: str,
    ) -> bool:
        pending = self._pending_stage_copy
        if pending is None:
            return False
        self._pending_stage_copy = None
        self._bridge_dropped_pending_count += 1
        self._bridge_pending_replace_count += 1
        if self._steady_state_bridge_epoch_active:
            self._steady_state_bridge_dropped_pending_count += 1
            self._steady_state_bridge_pending_replace_count += 1
        freed_immediately = self._retire_stage_copy_locked(
            pending,
            context=f"pending_replace:{reason}",
        )
        self._finalize_freshness_scheduler_mutation_locked(
            "pending_replace",
            reason=reason,
            incoming_frame_id=incoming_frame_id,
            dropped_frame_id=int(pending.get("frame_id", -1)),
            dropped_stage_index=int(pending.get("stage_index", -1)),
            freed_immediately=int(bool(freed_immediately)),
        )
        return True

    def _reclaim_completed_retired_stage_copies_locked(self) -> int:
        if not self._retired_stage_copies:
            return 0
        retained = []
        reclaimed_count = 0
        for pending in self._retired_stage_copies:
            end_event = pending.get("end_event")
            if end_event is None or end_event.query():
                reclaimed_count += 1
                self._free_stage_index_locked(
                    int(pending.get("stage_index", -1)),
                    context="reclaim_retired",
                )
            else:
                retained.append(pending)
        self._retired_stage_copies = retained
        return reclaimed_count

    def _complete_active_stage_commit_locked(
        self,
        pending: dict,
        commit_stats: Optional[dict[str, float]],
    ) -> None:
        completed_frame_id = int(pending.get("frame_id", -1))
        completed_stage_index = int(pending.get("stage_index", -1))
        if self._active_stage_copy is not pending:
            trace = "\n".join(self._bridge_transition_trace)
            raise RuntimeError(
                "freshness bridge invariant failed: active commit lost ownership of "
                "the active stage during active_commit_end\n"
                f"expected_active={self._stage_copy_brief_locked(pending)} "
                f"actual_active={self._stage_copy_brief_locked(self._active_stage_copy)}\n"
                f"{self._scheduler_snapshot_locked()}\nrecent transitions:\n{trace}"
            )
        if commit_stats is not None:
            self._record_bridge_commit_locked(
                frame_id=completed_frame_id,
                commit_stats=commit_stats,
            )
        promoted_pending = self._pending_stage_copy
        if promoted_pending is not None:
            self._mark_stage_copy_active_locked(promoted_pending)
        self._active_stage_copy = promoted_pending
        self._pending_stage_copy = None
        self._free_stage_index_locked(
            completed_stage_index,
            context="active_commit_release",
        )
        self._trace_bridge_transition_locked(
            "active_commit_release",
            committed_frame_id=completed_frame_id,
            committed_stage_index=completed_stage_index,
        )
        promoted_frame_id = None
        promoted_stage_index = None
        if promoted_pending is not None:
            promoted_frame_id = int(promoted_pending.get("frame_id", -1))
            promoted_stage_index = int(promoted_pending.get("stage_index", -1))
            self._bridge_active_promote_count += 1
            if self._steady_state_bridge_epoch_active:
                self._steady_state_bridge_active_promote_count += 1
            self._trace_bridge_transition_locked(
                "promote_pending",
                committed_frame_id=completed_frame_id,
                promoted_frame_id=promoted_frame_id,
                promoted_stage_index=promoted_stage_index,
            )
        self._finalize_freshness_scheduler_mutation_locked(
            "active_commit_end",
            committed_frame_id=completed_frame_id,
            committed_stage_index=completed_stage_index,
            promoted_frame_id=promoted_frame_id,
            promoted_stage_index=promoted_stage_index,
        )

    def _record_bridge_commit_locked(self, frame_id: int, commit_stats: dict[str, float]) -> None:
        if frame_id <= 0:
            return
        now = time.perf_counter()
        if self._bridge_committed_update_count == 0:
            self._bridge_commit_first_wall_s = now
        source_frame_delta = frame_id - self._bridge_commit_latest_frame_id
        if source_frame_delta <= 0:
            source_frame_delta = 1
        self._bridge_commit_latest_frame_id = frame_id
        self._bridge_commit_latest_wall_s = now
        self._bridge_committed_update_count += 1
        self._bridge_committed_source_frame_delta_count += source_frame_delta
        self._bridge_commit_elapsed_s = max(0.0, now - self._bridge_commit_first_wall_s)
        self._bridge_commit_gpu_to_cpu_ms_sum += float(
            commit_stats.get("gpu_to_cpu_copy_cuda", 0.0)
        )
        if bool(commit_stats.get("direct_path", False)):
            self._bridge_commit_direct_count += 1
            self._bridge_commit_direct_copy_ms_sum += float(
                commit_stats.get("direct_copy_ms", 0.0)
            )
        else:
            self._bridge_commit_fallback_count += 1
        self._bridge_commit_cpu_mmap_ms_sum += (
            1000.0 * float(commit_stats.get("cpu_mmap_copy_wall", 0.0))
        )
        self._bridge_commit_header_write_ms_sum += (
            1000.0 * float(commit_stats.get("header_write_wall", 0.0))
        )
        self._bridge_commit_total_ms_sum += (
            1000.0 * float(commit_stats.get("total_wall", 0.0))
        )
        self._bridge_submit_to_commit_start_ms_sum += float(
            commit_stats.get("submit_to_commit_start_ms", 0.0)
        )
        self._bridge_commit_wait_for_gpu_ready_ms_sum += float(
            commit_stats.get("commit_wait_for_gpu_ready_ms", 0.0)
        )
        self._bridge_commit_thread_wake_delay_ms_sum += float(
            commit_stats.get("commit_thread_wake_delay_ms", 0.0)
        )
        self._bridge_commit_active_service_ms_sum += float(
            commit_stats.get("commit_active_service_ms", 0.0)
        )
        if self._steady_state_bridge_epoch_active:
            self._steady_state_bridge_commit_latest_frame_id = int(frame_id)
            self._steady_state_bridge_commit_latest_wall_s = float(now)
            self._steady_state_bridge_committed_update_count += 1
            self._steady_state_bridge_committed_source_frame_delta_count += int(
                source_frame_delta
            )
            self._steady_state_bridge_commit_gpu_to_cpu_ms_sum += float(
                commit_stats.get("gpu_to_cpu_copy_cuda", 0.0)
            )
            if bool(commit_stats.get("direct_path", False)):
                self._steady_state_bridge_commit_direct_count += 1
                self._steady_state_bridge_commit_direct_copy_ms_sum += float(
                    commit_stats.get("direct_copy_ms", 0.0)
                )
            else:
                self._steady_state_bridge_commit_fallback_count += 1
            self._steady_state_bridge_commit_cpu_mmap_ms_sum += (
                1000.0 * float(commit_stats.get("cpu_mmap_copy_wall", 0.0))
            )
            self._steady_state_bridge_commit_header_write_ms_sum += (
                1000.0 * float(commit_stats.get("header_write_wall", 0.0))
            )
            self._steady_state_bridge_commit_total_ms_sum += (
                1000.0 * float(commit_stats.get("total_wall", 0.0))
            )
            self._steady_state_bridge_submit_to_commit_start_ms_sum += float(
                commit_stats.get("submit_to_commit_start_ms", 0.0)
            )
            self._steady_state_bridge_commit_wait_for_gpu_ready_ms_sum += float(
                commit_stats.get("commit_wait_for_gpu_ready_ms", 0.0)
            )
            self._steady_state_bridge_commit_thread_wake_delay_ms_sum += float(
                commit_stats.get("commit_thread_wake_delay_ms", 0.0)
            )
            self._steady_state_bridge_commit_active_service_ms_sum += float(
                commit_stats.get("commit_active_service_ms", 0.0)
            )

    def wait_for_bridge_idle(self, timeout: Optional[float] = None) -> bool:
        if not self.FRESHNESS_FIRST_COMMIT:
            self._drain_pending_stage_copies(block=True)
            return True
        deadline = None if timeout is None else (time.perf_counter() + float(timeout))
        with self._stage_condition:
            while True:
                self._raise_if_stage_commit_failed_locked()
                reclaimed_count = self._reclaim_completed_retired_stage_copies_locked()
                if reclaimed_count > 0:
                    self._finalize_freshness_scheduler_mutation_locked(
                        "reclaim_retired",
                        reclaimed_count=reclaimed_count,
                    )
                if (
                    self._active_stage_copy is None
                    and self._pending_stage_copy is None
                    and not self._reserved_stage_copies
                    and not self._retired_stage_copies
                ):
                    return True
                if deadline is not None:
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0.0:
                        return False
                    self._stage_condition.wait(timeout=min(0.05, remaining))
                else:
                    self._stage_condition.wait(timeout=0.05)

    def wait_for_sample(self, timeout: float = 10.0) -> LiveControllerSample:
        deadline = time.monotonic() + timeout
        with self._sample_condition:
            while True:
                sample = self._latest_sample
                if sample is not None:
                    return sample
                if self.process is not None and self.process.poll() is not None:
                    raise RuntimeError(
                        "Boba Immersive Demo exited before producing controller data.\n"
                        + self.debug_summary()
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                self._sample_condition.wait(timeout=min(0.05, remaining))
        raise RuntimeError(
            "Timed out waiting for Quest panel controller sample.\n" + self.debug_summary()
        )

    def get_latest_sample(self) -> Optional[LiveControllerSample]:
        with self._latest_lock:
            return self._latest_sample

    def wait_for_newer_sample(self, min_sample_id: int, timeout_s: float) -> Optional[LiveControllerSample]:
        min_sample_id = int(min_sample_id)
        timeout_s = max(0.0, float(timeout_s))
        deadline = time.monotonic() + timeout_s
        with self._sample_condition:
            while True:
                sample = self._latest_sample
                if sample is not None and int(getattr(sample, "sample", -1)) > min_sample_id:
                    return sample
                if self.process is not None and self.process.poll() is not None:
                    raise RuntimeError(
                        "Boba Immersive Demo exited while waiting for a newer sample.\n"
                        + self.debug_summary()
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return None
                self._sample_condition.wait(timeout=min(0.05, remaining))

    def debug_summary(self) -> str:
        parts = []
        if self._last_published_frame_id > 0:
            parts.append(f"last published frame id: {self._last_published_frame_id}")
        bridge_commit_stats = self.bridge_commit_stats()
        if (
            bridge_commit_stats.get("committed_update_count", 0) > 0
            or bridge_commit_stats.get("pending_replace_count", 0) > 0
            or bridge_commit_stats.get("capacity_block_count", 0) > 0
        ):
            parts.append(
                "bridge direct commit: "
                f"enabled={int(bool(bridge_commit_stats.get('direct_commit_enabled', False)))} "
                f"reason={bridge_commit_stats.get('direct_commit_warning', 'none') or 'none'}"
            )
            parts.append(
                "bridge submits: "
                f"latest_frame_id={bridge_commit_stats.get('latest_submitted_frame_id', 0)} "
                f"submit_fps={bridge_commit_stats.get('bridge_submit_fps', 0.0):.2f}"
            )
            parts.append(
                "bridge commits: "
                f"latest_frame_id={bridge_commit_stats.get('latest_frame_id', 0)} "
                f"update_fps={bridge_commit_stats.get('committed_update_fps', 0.0):.2f} "
                f"source_delta_fps={bridge_commit_stats.get('committed_source_delta_fps', 0.0):.2f} "
                f"drop_ratio={bridge_commit_stats.get('drop_ratio', 0.0) * 100.0:.1f}% "
                f"logical_queue_max_depth={bridge_commit_stats.get('logical_queue_max_depth', 0)} "
                f"retired_copying_max_depth={bridge_commit_stats.get('retired_copying_max_depth', 0)} "
                f"physical_in_use_max_depth={bridge_commit_stats.get('physical_in_use_max_depth', 0)}"
            )
            parts.append(
                "bridge scheduler: "
                f"pending_replace_count={bridge_commit_stats.get('pending_replace_count', 0)} "
                f"active_promote_count={bridge_commit_stats.get('active_promote_count', 0)} "
                f"capacity_block_count={bridge_commit_stats.get('capacity_block_count', 0)} "
                f"capacity_wait_ms={bridge_commit_stats.get('capacity_wait_ms', 0.0):.2f} "
                f"free_stage_min={bridge_commit_stats.get('free_stage_min', 0)}"
            )
            if bridge_commit_stats.get("bridge_publish_sample_check_count", 0) > 0:
                parts.append(
                    "bridge payload integrity: "
                    f"checks={bridge_commit_stats.get('bridge_publish_sample_check_count', 0)} "
                    f"mismatches={bridge_commit_stats.get('bridge_publish_sample_mismatch_count', 0)}"
                )
            parts.append(
                "bridge timing: "
                f"submit_to_commit_start_ms={bridge_commit_stats.get('bridge_submit_to_commit_start_ms', 0.0):.2f} "
                f"wake_delay_ms={bridge_commit_stats.get('bridge_commit_thread_wake_delay_ms', 0.0):.2f} "
                f"wait_for_gpu_ready_ms={bridge_commit_stats.get('bridge_commit_wait_for_gpu_ready_ms', 0.0):.2f} "
                f"active_service_ms={bridge_commit_stats.get('bridge_commit_active_service_ms', 0.0):.2f}"
            )
        viewer_applied_stats = self.viewer_applied_update_stats()
        viewer_source_delta_stats = self.viewer_source_frame_delta_stats()
        viewer_coalescing_stats = self.viewer_source_coalescing_stats()
        viewer_accounting_stats = self.viewer_accounting_stats()
        if (
            viewer_applied_stats.get("count", 0) > 0
            or viewer_source_delta_stats.get("count", 0) > 0
        ):
            parts.append(
                "viewer source frames: "
                f"latest_frame_id={viewer_applied_stats.get('latest_frame_id', 0)} "
                f"applied_updates={viewer_applied_stats.get('count', 0)} "
                f"source_frame_delta={viewer_source_delta_stats.get('count', 0)} "
                f"coalesced={viewer_coalescing_stats.get('count', 0)} "
                f"elapsed_s={viewer_applied_stats.get('elapsed_s', 0.0):.2f} "
                f"applied_recent_fps={viewer_applied_stats.get('recent_fps', 0.0):.2f} "
                f"source_delta_recent_fps={viewer_source_delta_stats.get('recent_fps', 0.0):.2f}"
            )
            if (
                viewer_accounting_stats.get(
                    "viewer_accounting_inconsistency_count", 0
                )
                > 0
                or viewer_accounting_stats.get("aggregate_inconsistent", False)
            ):
                parts.append(
                    "viewer accounting: "
                    f"inconsistency_count="
                    f"{viewer_accounting_stats.get('viewer_accounting_inconsistency_count', 0)} "
                    f"applied_vs_source_delta_gap="
                    f"{viewer_accounting_stats.get('viewer_applied_vs_source_delta_gap', 0)} "
                    f"coalesced={viewer_accounting_stats.get('viewer_coalesced_count', 0)} "
                    f"reconstructed="
                    f"{viewer_accounting_stats.get('viewer_coalesced_reconstructed_count', 0)}"
                )
        viewer_render_stats = self.viewer_render_stats()
        if viewer_render_stats.get("count", 0) > 0:
            parts.append(
                "viewer rendered frames: "
                f"count={viewer_render_stats.get('count', 0)} "
                f"elapsed_s={viewer_render_stats.get('elapsed_s', 0.0):.2f} "
                f"recent_fps={viewer_render_stats.get('recent_fps', 0.0):.2f}"
            )
        if self._stdout_tail:
            parts.append("stdout:\n" + "".join(self._stdout_tail).strip())
        if self._stderr_tail:
            parts.append("stderr:\n" + "".join(self._stderr_tail).strip())
        if self._parse_errors:
            parts.append("parse errors:\n" + "\n".join(self._parse_errors))
        if self._bridge_transition_trace:
            parts.append(
                "recent bridge transitions:\n" + "\n".join(self._bridge_transition_trace)
            )
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

    def _write_header(
        self,
        latest_frame_id: int,
        latest_slot: int,
        *,
        presentation_mode=None,
    ) -> None:
        assert self._shared_mmap is not None
        normalized_presentation_mode = self._normalize_presentation_mode(
            presentation_mode
        )
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
            int(normalized_presentation_mode),
            0,
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
            if hasattr(sample, "received_monotonic_s"):
                sample.received_monotonic_s = time.monotonic()
            with self._sample_condition:
                self._latest_sample = sample
                self._sample_condition.notify_all()

    def _read_stderr(self) -> None:
        assert self.process is not None and self.process.stderr is not None
        for line in self.process.stderr:
            self._stderr_tail.append(line)
            self._maybe_parse_viewer_source_stats_line(line)
            self._maybe_parse_viewer_render_stats_line(line)
            if self._should_echo_viewer_stderr_line(line):
                print(f"[quest_display_viewer:stderr] {line.rstrip()}", flush=True)

    @staticmethod
    def _is_routine_viewer_stderr_line(line: str) -> bool:
        stripped = str(line).strip()
        return stripped.startswith(
            (
                "Immersive bridge received source frame",
                "Immersive bridge viewer_source_stats",
                "Immersive bridge viewer_render_stats",
                "Immersive bridge presentation mode:",
                "Panel received source frame",
                "Presentation path:",
                "OpenGL version:",
                "Session state ->",
                "Opened shared frame file",
            )
        )

    def _should_echo_viewer_stderr_line(self, line: str) -> bool:
        if self._bridge_transition_trace_enabled:
            return True
        return not self._is_routine_viewer_stderr_line(line)

    @staticmethod
    def _parse_stats_payload(line: str, stats_prefix: str) -> Optional[dict[str, str]]:
        stripped = str(line).strip()
        prefix_pos = stripped.find(stats_prefix)
        if prefix_pos < 0:
            return None
        payload = stripped[prefix_pos + len(stats_prefix) :].strip()
        parsed = {}
        for token in payload.split():
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            parsed[key] = value
        return parsed

    def _maybe_parse_viewer_source_stats_line(self, line: str) -> None:
        parsed = self._parse_stats_payload(line, "viewer_source_stats ")
        if parsed is None:
            return
        try:
            with self._viewer_stats_lock:
                previous_latest_frame_id = int(self._viewer_latest_source_frame_id)
                applied_count = int(
                    parsed.get(
                        "update_count",
                        parsed.get("consumed_count", self._viewer_applied_update_count),
                    )
                )
                source_frame_delta_count = int(
                    parsed.get("source_frame_delta_count", applied_count)
                )
                recent_applied_fps = float(
                    parsed.get(
                        "update_recent_fps",
                        parsed.get("recent_fps", self._viewer_recent_applied_update_fps),
                    )
                )
                recent_source_delta_fps = float(
                    parsed.get("source_delta_recent_fps", recent_applied_fps)
                )
                coalesced_count = int(parsed.get("coalesced_frame_count", 0))
                parsed_latest_frame_id = int(
                    parsed.get("latest_frame_id", self._viewer_latest_source_frame_id)
                )
                self._viewer_latest_source_frame_id = parsed_latest_frame_id
                self._viewer_applied_update_count = applied_count
                self._viewer_source_frame_delta_count = source_frame_delta_count
                self._viewer_coalesced_source_frame_count = coalesced_count
                self._viewer_source_elapsed_s = float(
                    parsed.get("elapsed_s", self._viewer_source_elapsed_s)
                )
                self._viewer_recent_applied_update_fps = recent_applied_fps
                self._viewer_recent_source_delta_fps = recent_source_delta_fps
                if self._steady_state_viewer_epoch_active:
                    previous_parsed_applied_count = int(
                        self._steady_state_viewer_last_parsed_applied_update_count
                    )
                    previous_parsed_source_frame_delta_count = int(
                        self._steady_state_viewer_last_parsed_source_frame_delta_count
                    )
                    previous_parsed_coalesced_count = int(
                        self._steady_state_viewer_last_parsed_coalesced_source_frame_count
                    )
                    previous_parsed_latest_frame_id = int(
                        self._steady_state_viewer_last_parsed_latest_frame_id
                    )
                    raw_applied_delta = int(
                        applied_count - previous_parsed_applied_count
                    )
                    raw_source_delta = int(
                        source_frame_delta_count
                        - previous_parsed_source_frame_delta_count
                    )
                    raw_coalesced_delta = int(
                        coalesced_count - previous_parsed_coalesced_count
                    )
                    raw_latest_frame_delta = int(
                        parsed_latest_frame_id - previous_parsed_latest_frame_id
                    )
                    accounting_inconsistent = (
                        raw_applied_delta < 0
                        or raw_source_delta < 0
                        or raw_coalesced_delta < 0
                        or raw_latest_frame_delta < 0
                    )
                    applied_delta = max(0, raw_applied_delta)
                    source_delta = max(0, raw_source_delta)
                    coalesced_delta = max(0, raw_coalesced_delta)
                    expected_coalesced_delta = max(0, source_delta - applied_delta)
                    if (
                        source_delta < applied_delta
                        or source_delta != max(0, raw_latest_frame_delta)
                        or coalesced_delta != expected_coalesced_delta
                    ):
                        accounting_inconsistent = True
                    if accounting_inconsistent:
                        self._steady_state_viewer_accounting_inconsistency_count += 1
                    if (
                        parsed_latest_frame_id
                        > int(self._steady_state_viewer_frame_id_boundary)
                    ):
                        self._steady_state_viewer_applied_update_count += applied_delta
                        self._steady_state_viewer_source_frame_delta_count += source_delta
                        self._steady_state_viewer_coalesced_source_frame_count += (
                            coalesced_delta
                        )
                        self._steady_state_viewer_latest_frame_id = parsed_latest_frame_id
                        self._steady_state_viewer_epoch_sync_pending = False
                    self._steady_state_viewer_last_seen_frame_id = max(
                        int(self._steady_state_viewer_last_seen_frame_id),
                        int(parsed_latest_frame_id),
                        int(previous_latest_frame_id),
                    )
                    self._steady_state_viewer_last_parsed_applied_update_count = int(
                        applied_count
                    )
                    self._steady_state_viewer_last_parsed_source_frame_delta_count = int(
                        source_frame_delta_count
                    )
                    self._steady_state_viewer_last_parsed_coalesced_source_frame_count = int(
                        coalesced_count
                    )
                    self._steady_state_viewer_last_parsed_latest_frame_id = int(
                        parsed_latest_frame_id
                    )
        except Exception as exc:
            self._parse_errors.append(f"{exc}: {str(line).strip()}")

    def _maybe_parse_viewer_render_stats_line(self, line: str) -> None:
        parsed = self._parse_stats_payload(line, "viewer_render_stats ")
        if parsed is None:
            return
        try:
            with self._viewer_stats_lock:
                self._viewer_rendered_frame_count = int(
                    parsed.get("rendered_count", self._viewer_rendered_frame_count)
                )
                self._viewer_render_elapsed_s = float(
                    parsed.get("elapsed_s", self._viewer_render_elapsed_s)
                )
                self._viewer_recent_render_fps = float(
                    parsed.get("recent_fps", self._viewer_recent_render_fps)
                )
        except Exception as exc:
            self._parse_errors.append(f"{exc}: {str(line).strip()}")

    def viewer_applied_update_stats(
        self,
        *,
        scope: str = "lifetime",
        elapsed_override_s: Optional[float] = None,
    ) -> dict[str, float]:
        with self._viewer_stats_lock:
            if str(scope).strip().lower() == "steady_state":
                elapsed_s = (
                    float(elapsed_override_s)
                    if elapsed_override_s is not None
                    else max(
                        0.0,
                        time.perf_counter() - float(self._steady_state_viewer_epoch_wall_s),
                    )
                )
                count = int(self._steady_state_viewer_applied_update_count)
                latest_frame_id = int(self._steady_state_viewer_latest_frame_id)
                recent_fps = float(count) / elapsed_s if elapsed_s > 0.0 and count > 0 else 0.0
            else:
                elapsed_s = float(self._viewer_source_elapsed_s)
                count = int(self._viewer_applied_update_count)
                latest_frame_id = int(self._viewer_latest_source_frame_id)
                recent_fps = float(self._viewer_recent_applied_update_fps)
        average_fps = 0.0
        if elapsed_s > 0.0 and count > 0:
            average_fps = float(count) / elapsed_s
        return {
            "latest_frame_id": latest_frame_id,
            "count": count,
            "elapsed_s": elapsed_s,
            "recent_fps": recent_fps,
            "average_fps": average_fps,
        }

    def viewer_source_frame_delta_stats(
        self,
        *,
        scope: str = "lifetime",
        elapsed_override_s: Optional[float] = None,
    ) -> dict[str, float]:
        with self._viewer_stats_lock:
            if str(scope).strip().lower() == "steady_state":
                elapsed_s = (
                    float(elapsed_override_s)
                    if elapsed_override_s is not None
                    else max(
                        0.0,
                        time.perf_counter() - float(self._steady_state_viewer_epoch_wall_s),
                    )
                )
                count = int(self._steady_state_viewer_source_frame_delta_count)
                latest_frame_id = int(self._steady_state_viewer_latest_frame_id)
                recent_fps = float(count) / elapsed_s if elapsed_s > 0.0 and count > 0 else 0.0
            else:
                elapsed_s = float(self._viewer_source_elapsed_s)
                count = int(self._viewer_source_frame_delta_count)
                latest_frame_id = int(self._viewer_latest_source_frame_id)
                recent_fps = float(self._viewer_recent_source_delta_fps)
        average_fps = 0.0
        if elapsed_s > 0.0 and count > 0:
            average_fps = float(count) / elapsed_s
        return {
            "latest_frame_id": latest_frame_id,
            "count": count,
            "elapsed_s": elapsed_s,
            "recent_fps": recent_fps,
            "average_fps": average_fps,
        }

    def viewer_source_coalescing_stats(self, *, scope: str = "lifetime") -> dict[str, float]:
        with self._viewer_stats_lock:
            if str(scope).strip().lower() == "steady_state":
                source_frame_delta_count = int(
                    self._steady_state_viewer_source_frame_delta_count
                )
                coalesced_count = int(
                    self._steady_state_viewer_coalesced_source_frame_count
                )
            else:
                source_frame_delta_count = int(self._viewer_source_frame_delta_count)
                coalesced_count = int(self._viewer_coalesced_source_frame_count)
        coalesced_ratio = 0.0
        if source_frame_delta_count > 0:
            coalesced_ratio = float(coalesced_count) / float(source_frame_delta_count)
        return {
            "count": coalesced_count,
            "source_frame_delta_count": source_frame_delta_count,
            "ratio": coalesced_ratio,
        }

    def viewer_accounting_stats(self, *, scope: str = "lifetime") -> dict[str, float]:
        with self._viewer_stats_lock:
            if str(scope).strip().lower() == "steady_state":
                applied_count = int(self._steady_state_viewer_applied_update_count)
                source_frame_delta_count = int(
                    self._steady_state_viewer_source_frame_delta_count
                )
                coalesced_count = int(
                    self._steady_state_viewer_coalesced_source_frame_count
                )
                inconsistency_count = int(
                    self._steady_state_viewer_accounting_inconsistency_count
                )
                epoch_sync_pending = bool(self._steady_state_viewer_epoch_sync_pending)
            else:
                applied_count = int(self._viewer_applied_update_count)
                source_frame_delta_count = int(self._viewer_source_frame_delta_count)
                coalesced_count = int(self._viewer_coalesced_source_frame_count)
                inconsistency_count = 0
                epoch_sync_pending = False
        applied_vs_source_delta_gap = int(source_frame_delta_count - applied_count)
        reconstructed_coalesced_count = max(0, applied_vs_source_delta_gap)
        aggregate_inconsistent = (
            applied_vs_source_delta_gap < 0
            or coalesced_count != reconstructed_coalesced_count
        )
        return {
            "viewer_accounting_inconsistency_count": int(inconsistency_count)
            + int(aggregate_inconsistent),
            "raw_inconsistency_count": inconsistency_count,
            "viewer_applied_vs_source_delta_gap": applied_vs_source_delta_gap,
            "viewer_coalesced_reconstructed_count": reconstructed_coalesced_count,
            "viewer_coalesced_count": coalesced_count,
            "aggregate_inconsistent": aggregate_inconsistent,
            "epoch_sync_pending": epoch_sync_pending,
        }

    def viewer_consumed_source_stats(
        self,
        *,
        scope: str = "lifetime",
        elapsed_override_s: Optional[float] = None,
    ) -> dict[str, float]:
        return self.viewer_applied_update_stats(
            scope=scope,
            elapsed_override_s=elapsed_override_s,
        )

    def viewer_render_stats(
        self,
        *,
        scope: str = "lifetime",
        elapsed_override_s: Optional[float] = None,
    ) -> dict[str, float]:
        with self._viewer_stats_lock:
            if str(scope).strip().lower() == "steady_state":
                elapsed_s = (
                    float(elapsed_override_s)
                    if elapsed_override_s is not None
                    else max(
                        0.0,
                        time.perf_counter() - float(self._steady_state_viewer_epoch_wall_s),
                    )
                )
                count = max(
                    0,
                    int(self._viewer_rendered_frame_count)
                    - int(self._steady_state_viewer_render_baseline_count),
                )
                recent_fps = float(count) / elapsed_s if elapsed_s > 0.0 and count > 0 else 0.0
            else:
                elapsed_s = float(self._viewer_render_elapsed_s)
                count = int(self._viewer_rendered_frame_count)
                recent_fps = float(self._viewer_recent_render_fps)
        average_fps = 0.0
        if elapsed_s > 0.0 and count > 0:
            average_fps = float(count) / elapsed_s
        return {
            "count": count,
            "elapsed_s": elapsed_s,
            "recent_fps": recent_fps,
            "average_fps": average_fps,
        }

    def bridge_commit_stats(
        self,
        *,
        scope: str = "lifetime",
        elapsed_override_s: Optional[float] = None,
    ) -> dict[str, float]:
        with self._stage_condition:
            use_steady_state = str(scope).strip().lower() == "steady_state"
            if use_steady_state:
                submitted_frame_count = int(self._steady_state_bridge_submitted_frame_count)
                latest_submitted_frame_id = int(
                    self._steady_state_bridge_submit_latest_frame_id
                )
                committed_update_count = int(
                    self._steady_state_bridge_committed_update_count
                )
                committed_source_frame_delta_count = int(
                    self._steady_state_bridge_committed_source_frame_delta_count
                )
                dropped_pending_count = int(
                    self._steady_state_bridge_dropped_pending_count
                )
                direct_commit_count = int(self._steady_state_bridge_commit_direct_count)
                fallback_commit_count = int(
                    self._steady_state_bridge_commit_fallback_count
                )
                pending_replace_count = int(
                    self._steady_state_bridge_pending_replace_count
                )
                active_promote_count = int(self._steady_state_bridge_active_promote_count)
                capacity_block_count = int(
                    self._steady_state_bridge_capacity_block_count
                )
                logical_queue_max_depth = int(
                    self._steady_state_bridge_commit_queue_max_depth
                )
                retired_copying_max_depth = int(
                    self._steady_state_bridge_retired_copying_max_depth
                )
                physical_in_use_max_depth = int(
                    self._steady_state_bridge_physical_in_use_max_depth
                )
                free_stage_min = int(self._steady_state_bridge_free_stage_min)
                publish_sample_check_count = int(
                    self._steady_state_bridge_publish_sample_check_count
                )
                publish_sample_mismatch_count = int(
                    self._steady_state_bridge_publish_sample_mismatch_count
                )
                latest_frame_id = int(self._steady_state_bridge_commit_latest_frame_id)
                submit_elapsed_s = (
                    float(elapsed_override_s)
                    if elapsed_override_s is not None
                    else max(
                        0.0,
                        float(self._steady_state_bridge_submit_latest_wall_s)
                        - float(self._steady_state_bridge_epoch_wall_s),
                    )
                )
                elapsed_s = (
                    float(elapsed_override_s)
                    if elapsed_override_s is not None
                    else max(
                        0.0,
                        float(self._steady_state_bridge_commit_latest_wall_s)
                        - float(self._steady_state_bridge_epoch_wall_s),
                    )
                )
                average_gpu_to_cpu_ms = (
                    self._steady_state_bridge_commit_gpu_to_cpu_ms_sum
                    / committed_update_count
                    if committed_update_count > 0
                    else 0.0
                )
                average_cpu_mmap_ms = (
                    self._steady_state_bridge_commit_cpu_mmap_ms_sum
                    / committed_update_count
                    if committed_update_count > 0
                    else 0.0
                )
                average_direct_copy_ms = (
                    self._steady_state_bridge_commit_direct_copy_ms_sum
                    / direct_commit_count
                    if direct_commit_count > 0
                    else 0.0
                )
                average_header_write_ms = (
                    self._steady_state_bridge_commit_header_write_ms_sum
                    / committed_update_count
                    if committed_update_count > 0
                    else 0.0
                )
                average_total_ms = (
                    self._steady_state_bridge_commit_total_ms_sum
                    / committed_update_count
                    if committed_update_count > 0
                    else 0.0
                )
                average_submit_to_commit_start_ms = (
                    self._steady_state_bridge_submit_to_commit_start_ms_sum
                    / committed_update_count
                    if committed_update_count > 0
                    else 0.0
                )
                average_commit_wait_for_gpu_ready_ms = (
                    self._steady_state_bridge_commit_wait_for_gpu_ready_ms_sum
                    / committed_update_count
                    if committed_update_count > 0
                    else 0.0
                )
                average_commit_thread_wake_delay_ms = (
                    self._steady_state_bridge_commit_thread_wake_delay_ms_sum
                    / committed_update_count
                    if committed_update_count > 0
                    else 0.0
                )
                average_commit_active_service_ms = (
                    self._steady_state_bridge_commit_active_service_ms_sum
                    / committed_update_count
                    if committed_update_count > 0
                    else 0.0
                )
                average_capacity_wait_ms = (
                    self._steady_state_bridge_capacity_wait_ms_sum
                    / capacity_block_count
                    if capacity_block_count > 0
                    else 0.0
                )
            else:
                submit_elapsed_s = float(self._bridge_submit_elapsed_s)
                submitted_frame_count = int(self._bridge_submitted_frame_count)
                latest_submitted_frame_id = int(self._bridge_submit_latest_frame_id)
                elapsed_s = float(self._bridge_commit_elapsed_s)
                committed_update_count = int(self._bridge_committed_update_count)
                committed_source_frame_delta_count = int(
                    self._bridge_committed_source_frame_delta_count
                )
                dropped_pending_count = int(self._bridge_dropped_pending_count)
                direct_commit_count = int(self._bridge_commit_direct_count)
                fallback_commit_count = int(self._bridge_commit_fallback_count)
                pending_replace_count = int(self._bridge_pending_replace_count)
                active_promote_count = int(self._bridge_active_promote_count)
                capacity_block_count = int(self._bridge_capacity_block_count)
                logical_queue_max_depth = int(self._bridge_commit_queue_max_depth)
                retired_copying_max_depth = int(self._bridge_retired_copying_max_depth)
                physical_in_use_max_depth = int(self._bridge_physical_in_use_max_depth)
                free_stage_min = int(self._bridge_free_stage_min)
                publish_sample_check_count = int(
                    self._bridge_publish_sample_check_count
                )
                publish_sample_mismatch_count = int(
                    self._bridge_publish_sample_mismatch_count
                )
                latest_frame_id = int(self._bridge_commit_latest_frame_id)
                average_gpu_to_cpu_ms = (
                    self._bridge_commit_gpu_to_cpu_ms_sum / committed_update_count
                    if committed_update_count > 0
                    else 0.0
                )
                average_cpu_mmap_ms = (
                    self._bridge_commit_cpu_mmap_ms_sum / committed_update_count
                    if committed_update_count > 0
                    else 0.0
                )
                average_direct_copy_ms = (
                    self._bridge_commit_direct_copy_ms_sum / direct_commit_count
                    if direct_commit_count > 0
                    else 0.0
                )
                average_header_write_ms = (
                    self._bridge_commit_header_write_ms_sum / committed_update_count
                    if committed_update_count > 0
                    else 0.0
                )
                average_total_ms = (
                    self._bridge_commit_total_ms_sum / committed_update_count
                    if committed_update_count > 0
                    else 0.0
                )
                average_submit_to_commit_start_ms = (
                    self._bridge_submit_to_commit_start_ms_sum / committed_update_count
                    if committed_update_count > 0
                    else 0.0
                )
                average_commit_wait_for_gpu_ready_ms = (
                    self._bridge_commit_wait_for_gpu_ready_ms_sum / committed_update_count
                    if committed_update_count > 0
                    else 0.0
                )
                average_commit_thread_wake_delay_ms = (
                    self._bridge_commit_thread_wake_delay_ms_sum / committed_update_count
                    if committed_update_count > 0
                    else 0.0
                )
                average_commit_active_service_ms = (
                    self._bridge_commit_active_service_ms_sum / committed_update_count
                    if committed_update_count > 0
                    else 0.0
                )
                average_capacity_wait_ms = (
                    self._bridge_capacity_wait_ms_sum / capacity_block_count
                    if capacity_block_count > 0
                    else 0.0
                )
            direct_commit_enabled = bool(self._direct_commit_enabled)
            direct_commit_warning = (
                None
                if self._direct_commit_warning is None
                else str(self._direct_commit_warning)
            )
        bridge_submit_fps = 0.0
        if submit_elapsed_s > 0.0 and submitted_frame_count > 0:
            bridge_submit_fps = submitted_frame_count / submit_elapsed_s
        committed_update_fps = 0.0
        committed_source_delta_fps = 0.0
        if elapsed_s > 0.0 and committed_update_count > 0:
            committed_update_fps = committed_update_count / elapsed_s
        if elapsed_s > 0.0 and committed_source_frame_delta_count > 0:
            committed_source_delta_fps = committed_source_frame_delta_count / elapsed_s
        drop_ratio = 0.0
        if submitted_frame_count > 0:
            drop_ratio = dropped_pending_count / submitted_frame_count
        direct_path_ratio = 0.0
        fallback_ratio = 0.0
        if committed_update_count > 0:
            direct_path_ratio = direct_commit_count / committed_update_count
            fallback_ratio = fallback_commit_count / committed_update_count
        return {
            "submit_elapsed_s": submit_elapsed_s,
            "latest_submitted_frame_id": latest_submitted_frame_id,
            "elapsed_s": elapsed_s,
            "latest_frame_id": latest_frame_id,
            "submitted_frame_count": submitted_frame_count,
            "bridge_submit_fps": bridge_submit_fps,
            "committed_update_count": committed_update_count,
            "committed_source_frame_delta_count": committed_source_frame_delta_count,
            "committed_update_fps": committed_update_fps,
            "committed_source_delta_fps": committed_source_delta_fps,
            "dropped_pending_count": dropped_pending_count,
            "drop_ratio": drop_ratio,
            "direct_commit_count": direct_commit_count,
            "fallback_commit_count": fallback_commit_count,
            "bridge_commit_direct_path_ratio": direct_path_ratio,
            "bridge_commit_direct_copy_ms": average_direct_copy_ms,
            "bridge_commit_fallback_ratio": fallback_ratio,
            "pending_replace_count": pending_replace_count,
            "active_promote_count": active_promote_count,
            "capacity_block_count": capacity_block_count,
            "capacity_wait_ms": average_capacity_wait_ms,
            "queue_max_depth": logical_queue_max_depth,
            "logical_queue_max_depth": logical_queue_max_depth,
            "retired_copying_max_depth": retired_copying_max_depth,
            "physical_in_use_max_depth": physical_in_use_max_depth,
            "free_stage_min": free_stage_min,
            "bridge_publish_sample_check_count": publish_sample_check_count,
            "bridge_publish_sample_mismatch_count": publish_sample_mismatch_count,
            "direct_commit_enabled": direct_commit_enabled,
            "direct_commit_warning": direct_commit_warning,
            "bridge_submit_to_commit_start_ms": average_submit_to_commit_start_ms,
            "bridge_commit_wait_for_gpu_ready_ms": average_commit_wait_for_gpu_ready_ms,
            "bridge_commit_thread_wake_delay_ms": average_commit_thread_wake_delay_ms,
            "bridge_commit_active_service_ms": average_commit_active_service_ms,
            "bridge_commit_gpu_to_cpu_ms": average_gpu_to_cpu_ms,
            "bridge_commit_cpu_mmap_ms": average_cpu_mmap_ms,
            "bridge_commit_header_write_ms": average_header_write_ms,
            "bridge_commit_total_ms": average_total_ms,
        }

    @staticmethod
    def _parse_sample(payload: dict) -> LiveControllerSample:
        return LiveControllerSample(
            sample=int(payload["sample"]),
            left=OpenXRFramePanelMirror._parse_controller(payload["left"]),
            right=OpenXRFramePanelMirror._parse_controller(payload["right"]),
        )

    @staticmethod
    def _parse_controller(payload: dict) -> ControllerPoseSample:
        return parse_controller_payload(payload)

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


class OpenXRImmersiveBridge(OpenXRFramePanelMirror):
    HEADER_MAGIC = b"BOBAQIM1"
    EYE_COUNT = 2
    SLOT_COUNT = 4
    STAGING_BUFFER_COUNT = 4
    FRESHNESS_FIRST_COMMIT = True

    def __init__(self, repo_root: Path, width: int, height: int):
        super().__init__(repo_root=repo_root, width=width, height=height)
        self.frame_bytes = self.width * self.height * self.channels * self.EYE_COUNT
        pin_memory = torch.cuda.is_available()
        self._cpu_stage_buffers = [
            torch.empty(
                (self.EYE_COUNT, self.height, self.width, self.channels),
                dtype=torch.uint8,
                device="cpu",
                pin_memory=pin_memory,
            )
            for _ in range(self.STAGING_BUFFER_COUNT)
        ]
        self._cpu_stage_arrays = [buffer.numpy() for buffer in self._cpu_stage_buffers]
        self._refresh_bridge_publish_sample_indices()
        self.binary_path = self.repo_root / "linux_pose_probe" / "boba_immersive_bridge"
        self.build_script_path = (
            self.repo_root / "linux_pose_probe" / "build_boba_immersive_bridge.sh"
        )
        self.source_path = self.repo_root / "linux_pose_probe" / "openxr_frame_panel.cpp"

    def publish_stereo_frames(
        self,
        left_frame_rgba: torch.Tensor,
        right_frame_rgba: torch.Tensor,
        presentation_mode=None,
    ) -> tuple[bool, dict[str, float]]:
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
            raise RuntimeError("Immersive Quest shared buffer is not initialized.")
        expected_shape = (self.height, self.width, self.channels)
        for name, frame in (("left", left_frame_rgba), ("right", right_frame_rgba)):
            if frame.shape != expected_shape:
                raise ValueError(
                    f"{name} immersive frame shape {tuple(frame.shape)} != {expected_shape}"
                )
            if frame.dtype != torch.uint8:
                raise ValueError(
                    f"{name} immersive frame dtype {frame.dtype} != torch.uint8"
                )

        expected_publish_sample_bytes = (
            self._capture_bridge_publish_sample_bytes_from_stereo_tensors(
                left_frame_rgba,
                right_frame_rgba,
            )
        )
        normalized_presentation_mode = self._normalize_presentation_mode(
            presentation_mode
        )
        process_check_start = time.perf_counter()
        if self.process is not None and self.process.poll() is not None:
            if not self._exit_logged:
                print(
                    "[quest_display] immersive bridge exited unexpectedly; "
                    "disabling immersive Quest publishing for this run.\n"
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
        submit_wall_s = time.perf_counter()
        with self._stage_condition:
            self._raise_if_stage_commit_failed_locked()
            self._record_bridge_submit_locked(
                frame_id=frame_id,
                submit_wall_s=submit_wall_s,
            )
            if self.FRESHNESS_FIRST_COMMIT:
                self._trace_bridge_transition_locked(
                    "submit",
                    frame_id=frame_id,
                    slot=slot,
                )

        if self._staging_copy_stream is None:
            fallback_start = time.perf_counter()
            stage_array = self._cpu_stage_arrays[0]
            np.copyto(stage_array[0], left_frame_rgba.cpu().numpy())
            np.copyto(stage_array[1], right_frame_rgba.cpu().numpy())
            commit_stats = self._commit_stage_array(
                stage_array,
                slot=slot,
                frame_id=frame_id,
                expected_publish_sample_bytes=expected_publish_sample_bytes,
                presentation_mode=normalized_presentation_mode,
            )
            timing["fallback_copy_wall"] = time.perf_counter() - fallback_start
            timing["cpu_mmap_copy_wall"] += commit_stats["cpu_mmap_copy_wall"]
            timing["header_write_wall"] += commit_stats["header_write_wall"]
            if self.FRESHNESS_FIRST_COMMIT:
                commit_record = {
                    "wait_wall": 0.0,
                    "gpu_to_cpu_copy_cuda": 0.0,
                    "cpu_mmap_copy_wall": commit_stats["cpu_mmap_copy_wall"],
                    "header_write_wall": commit_stats["header_write_wall"],
                    "total_wall": timing["fallback_copy_wall"],
                    "submit_to_commit_start_ms": 0.0,
                    "commit_wait_for_gpu_ready_ms": 0.0,
                    "commit_thread_wake_delay_ms": 0.0,
                    "commit_active_service_ms": timing["fallback_copy_wall"] * 1000.0,
                }
                with self._stage_condition:
                    self._record_bridge_commit_locked(
                        frame_id=frame_id,
                        commit_stats=commit_record,
                    )
            timing["total_wall"] = time.perf_counter() - publish_start
            self._last_published_frame_id = frame_id
            return True, timing

        timing["pending_drain_block_wall"] = self._wait_for_stage_capacity(
            incoming_frame_id=frame_id
        )
        timing["stage_enqueue_wall"] = self._enqueue_stereo_stage_copy(
            left_frame_tensor=left_frame_rgba,
            right_frame_tensor=right_frame_rgba,
            slot=slot,
            frame_id=frame_id,
            submit_wall_s=submit_wall_s,
            expected_publish_sample_bytes=expected_publish_sample_bytes,
            presentation_mode=normalized_presentation_mode,
        )
        timing["total_wall"] = time.perf_counter() - publish_start
        self._last_published_frame_id = frame_id
        return True, timing

    def _enqueue_stereo_stage_copy(
        self,
        *,
        left_frame_tensor: torch.Tensor,
        right_frame_tensor: torch.Tensor,
        slot: int,
        frame_id: int,
        submit_wall_s: Optional[float] = None,
        expected_publish_sample_bytes: Optional[bytes] = None,
        presentation_mode: Optional[int] = None,
    ) -> float:
        if submit_wall_s is None:
            submit_wall_s = time.perf_counter()
        stage_index: Optional[int] = None
        if self.FRESHNESS_FIRST_COMMIT:
            with self._stage_condition:
                stage_index = self._reserve_freshness_stage_copy_locked(
                    frame_id=frame_id,
                    slot=slot,
                )
        else:
            stage_index = self._next_stage_index
        try:
            left_copy_tensor = (
                left_frame_tensor
                if left_frame_tensor.is_contiguous()
                else left_frame_tensor.contiguous()
            )
            right_copy_tensor = (
                right_frame_tensor
                if right_frame_tensor.is_contiguous()
                else right_frame_tensor.contiguous()
            )
            stage_buffer = self._cpu_stage_buffers[stage_index]
            stage_array = self._cpu_stage_arrays[stage_index]
            producer_ready_event = torch.cuda.Event()
            enqueue_start = time.perf_counter()
            producer_ready_event.record(torch.cuda.current_stream())
            copy_start_event = None
            copy_end_event = None
            if not self._direct_commit_enabled:
                copy_start_event = torch.cuda.Event(enable_timing=True)
                copy_end_event = torch.cuda.Event(enable_timing=True)
                with torch.cuda.stream(self._staging_copy_stream):
                    self._staging_copy_stream.wait_event(producer_ready_event)
                    copy_start_event.record(self._staging_copy_stream)
                    stage_buffer[0].copy_(left_copy_tensor, non_blocking=True)
                    stage_buffer[1].copy_(right_copy_tensor, non_blocking=True)
                    copy_end_event.record(self._staging_copy_stream)
            with self._stage_condition:
                pending = {
                    "slot": slot,
                    "frame_id": frame_id,
                    "stage_index": stage_index,
                    "submit_wall_s": submit_wall_s,
                    "expected_publish_sample_bytes": expected_publish_sample_bytes,
                    "presentation_mode": self._normalize_presentation_mode(
                        presentation_mode
                    ),
                }
                if self._direct_commit_enabled:
                    pending.update(
                        {
                            "producer_ready_event": producer_ready_event,
                            "frame_tensors": (left_copy_tensor, right_copy_tensor),
                            "direct_commit": True,
                        }
                    )
                else:
                    pending.update(
                        {
                            "start_event": copy_start_event,
                            "end_event": copy_end_event,
                            "stage_array": stage_array,
                            "direct_commit": False,
                        }
                    )
                if self.FRESHNESS_FIRST_COMMIT:
                    if self._active_stage_copy is None:
                        reserved = self._reserved_stage_copies.pop(stage_index, None)
                        if reserved is None:
                            raise RuntimeError(
                                "freshness bridge invariant failed: reserved stage missing "
                                f"during enqueue_active (stage_index={stage_index})\n"
                                f"{self._scheduler_snapshot_locked()}\nrecent transitions:\n"
                                + "\n".join(self._bridge_transition_trace)
                            )
                        self._mark_stage_copy_active_locked(pending)
                        self._active_stage_copy = pending
                        self._finalize_freshness_scheduler_mutation_locked(
                            "enqueue_active",
                            frame_id=frame_id,
                            stage_index=stage_index,
                        )
                    else:
                        if self._pending_stage_copy is not None:
                            self._retire_pending_stage_copy_locked(
                                incoming_frame_id=frame_id,
                                reason="enqueue_replace",
                            )
                        reserved = self._reserved_stage_copies.pop(stage_index, None)
                        if reserved is None:
                            raise RuntimeError(
                                "freshness bridge invariant failed: reserved stage missing "
                                f"during enqueue_pending (stage_index={stage_index})\n"
                                f"{self._scheduler_snapshot_locked()}\nrecent transitions:\n"
                                + "\n".join(self._bridge_transition_trace)
                            )
                        self._pending_stage_copy = pending
                        self._finalize_freshness_scheduler_mutation_locked(
                            "enqueue_pending",
                            frame_id=frame_id,
                            stage_index=stage_index,
                        )
                else:
                    self._pending_stage_copies.append(pending)
                    self._next_stage_index = (stage_index + 1) % self.STAGING_BUFFER_COUNT
                self._stage_condition.notify_all()
            return time.perf_counter() - enqueue_start
        except BaseException:
            if self.FRESHNESS_FIRST_COMMIT and stage_index is not None:
                with self._stage_condition:
                    self._abort_reserved_stage_copy_locked(
                        stage_index=stage_index,
                        frame_id=frame_id,
                        reason="enqueue_prepare_exception",
                    )
            raise

    def wait_for_sample(self, timeout: float = 10.0) -> LiveImmersiveSample:
        deadline = time.monotonic() + timeout
        with self._sample_condition:
            while True:
                sample = self._latest_sample
                if sample is not None:
                    return sample
                if self.process is not None and self.process.poll() is not None:
                    raise RuntimeError(
                        "Quest immersive bridge exited before producing pose data.\n"
                        + self.debug_summary()
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                self._sample_condition.wait(timeout=min(0.05, remaining))
        raise RuntimeError(
            "Timed out waiting for Quest immersive pose sample.\n" + self.debug_summary()
        )

    def _create_shared_frame_file(self) -> None:
        total_bytes = self.HEADER_STRUCT.size + self.SLOT_COUNT * self.frame_bytes
        fd, path = tempfile.mkstemp(
            prefix="boba_quest_immersive_",
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
                    (self.EYE_COUNT, self.height, self.width, self.channels),
                    dtype=np.uint8,
                    buffer=self._shared_mmap,
                    offset=offset,
                )
            )
            self._slot_views[-1].fill(0)

    @staticmethod
    def _parse_sample(payload: dict) -> LiveImmersiveSample:
        return LiveImmersiveSample(
            sample=int(payload["sample"]),
            left=OpenXRFramePanelMirror._parse_controller(payload["left"]),
            right=OpenXRFramePanelMirror._parse_controller(payload["right"]),
            left_eye=OpenXRImmersiveBridge._parse_eye(payload["left_eye"]),
            right_eye=OpenXRImmersiveBridge._parse_eye(payload["right_eye"]),
        )

    @staticmethod
    def _parse_eye(payload: dict) -> EyePoseSample:
        position = np.asarray(payload["position"], dtype=np.float32)
        orientation = np.asarray(payload["orientation"], dtype=np.float32)
        if position.shape != (3,):
            raise ValueError(f"eye position shape {position.shape} != (3,)")
        if orientation.shape != (4,):
            raise ValueError(f"eye orientation shape {orientation.shape} != (4,)")
        return EyePoseSample(
            pose_valid=bool(payload["pose_valid"]),
            pose_tracked=bool(payload["pose_tracked"]),
            position=position,
            orientation=orientation,
            fov=EyeFovSample(
                angle_left=float(payload["fov"]["angle_left"]),
                angle_right=float(payload["fov"]["angle_right"]),
                angle_up=float(payload["fov"]["angle_up"]),
                angle_down=float(payload["fov"]["angle_down"]),
            ),
            recommended_width=int(payload["recommended_width"]),
            recommended_height=int(payload["recommended_height"]),
        )
