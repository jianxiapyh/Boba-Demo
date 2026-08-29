from __future__ import annotations

import mmap
import os
from pathlib import Path
import socket
import struct
import threading
import time
from typing import Optional

import numpy as np

from qqtt.live_openxr import (
    ControllerPoseSample,
    EyeFovSample,
    EyePoseSample,
    LiveImmersiveSample,
)
from qqtt.quest_display import OpenXRImmersiveBridge


class ILLIXRImmersiveBridge(OpenXRImmersiveBridge):
    """Boba's existing renderer bridge with ILLIXR as the OpenXR owner.

    The renderer-facing API is intentionally identical to
    :class:`OpenXRImmersiveBridge`. ILLIXR sends fixed-size binary pose/input
    datagrams and consumes the existing stereo mmap ring; no second OpenXR
    instance or session is created here.
    """

    INPUT_MAGIC = b"ILLIXRI1"
    INPUT_VERSION = 1
    INPUT_HEADER_STRUCT = struct.Struct("<8sIIQq")
    INPUT_EYE_STRUCT = struct.Struct("<III3f4f4f")
    INPUT_POSE_STRUCT = struct.Struct("<I3f4f")
    INPUT_BUTTON_STRUCT = struct.Struct("<If")
    INPUT_AXIS_STRUCT = struct.Struct("<I2f")

    FLAG_ACTIVE = 1 << 0
    FLAG_POSITION_VALID = 1 << 1
    FLAG_ORIENTATION_VALID = 1 << 2
    FLAG_POSITION_TRACKED = 1 << 3
    FLAG_ORIENTATION_TRACKED = 1 << 4
    FLAG_PRESSED = 1 << 1

    CONTROLLER_BYTE_COUNT = (
        8
        + 2 * INPUT_POSE_STRUCT.size
        + 5 * INPUT_BUTTON_STRUCT.size
        + INPUT_AXIS_STRUCT.size
    )
    INPUT_PACKET_BYTE_COUNT = (
        INPUT_HEADER_STRUCT.size
        + 2 * INPUT_EYE_STRUCT.size
        + 2 * CONTROLLER_BYTE_COUNT
    )

    def __init__(self, repo_root: Path, width: int, height: int):
        super().__init__(repo_root=repo_root, width=width, height=height)
        self._input_socket_path = self._required_path_env(
            "BOBA_ILLIXR_INPUT_SOCKET"
        )
        self._configured_frame_path = self._required_path_env(
            "BOBA_ILLIXR_FRAME_PATH"
        )
        self._configured_overlay_path = self._required_path_env(
            "BOBA_ILLIXR_OVERLAY_PATH"
        )
        self._configured_modal_path = self._required_path_env(
            "BOBA_ILLIXR_MODAL_PATH"
        )
        self._input_socket: Optional[socket.socket] = None
        self._input_thread: Optional[threading.Thread] = None
        self._input_stop = threading.Event()
        self._input_error: Optional[str] = None
        self._received_packet_count = 0

    @staticmethod
    def _required_path_env(name: str) -> Path:
        value = os.environ.get(name)
        if not value:
            raise RuntimeError(f"{name} must be set for the ILLIXR bridge")
        return Path(value).expanduser().resolve()

    @staticmethod
    def _create_exact_mmap(path: Path, byte_count: int):
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.unlink(missing_ok=True)
        except TypeError:
            if path.exists():
                path.unlink()
        shared_file = open(path, "w+b", buffering=0)
        shared_file.truncate(byte_count)
        return shared_file, mmap.mmap(shared_file.fileno(), byte_count)

    def _create_shared_frame_file(self) -> None:
        metadata_bytes = self._pose_metadata_bytes
        total_bytes = (
            self.HEADER_STRUCT.size
            + metadata_bytes
            + self.SLOT_COUNT * self.frame_bytes
        )
        self.shared_frame_path = self._configured_frame_path
        self._shared_file, self._shared_mmap = self._create_exact_mmap(
            self.shared_frame_path, total_bytes
        )
        self._shared_mmap[
            self.HEADER_STRUCT.size : self.HEADER_STRUCT.size + metadata_bytes
        ] = b"\x00" * metadata_bytes
        self._write_header(latest_frame_id=0, latest_slot=0)
        self._slot_views = []
        for slot_index in range(self.SLOT_COUNT):
            offset = (
                self.HEADER_STRUCT.size
                + metadata_bytes
                + slot_index * self.frame_bytes
            )
            self._slot_views.append(
                np.ndarray(
                    (self.EYE_COUNT, self.height, self.width, self.channels),
                    dtype=np.uint8,
                    buffer=self._shared_mmap,
                    offset=offset,
                )
            )
            self._slot_views[-1].fill(0)

    def _create_shared_overlay_file(self) -> None:
        command_bytes = (
            self.SLOT_COUNT
            * self.EYE_COUNT
            * self.OVERLAY_MAX_COMMANDS_PER_EYE
            * self.OVERLAY_COMMAND_STRIDE_FLOATS
            * np.dtype(np.float32).itemsize
        )
        metadata_bytes = self.SLOT_COUNT * self.OVERLAY_SLOT_METADATA_STRUCT.size
        total_bytes = self.OVERLAY_HEADER_STRUCT.size + metadata_bytes + command_bytes
        self.shared_overlay_path = self._configured_overlay_path
        self._overlay_file, self._overlay_mmap = self._create_exact_mmap(
            self.shared_overlay_path, total_bytes
        )
        self._overlay_mmap[
            self._overlay_slot_metadata_offset : self._overlay_payload_offset
        ] = b"\x00" * metadata_bytes
        self._overlay_command_array = np.ndarray(
            (
                self.SLOT_COUNT,
                self.EYE_COUNT,
                self.OVERLAY_MAX_COMMANDS_PER_EYE,
                self.OVERLAY_COMMAND_STRIDE_FLOATS,
            ),
            dtype=np.float32,
            buffer=self._overlay_mmap,
            offset=self._overlay_payload_offset,
        )
        self._overlay_command_array.fill(0.0)
        self._write_overlay_header(latest_overlay_id=0, left_count=0, right_count=0)

    def _create_shared_overlay_modal_file(self) -> None:
        texture_bytes = (
            self.SLOT_COUNT
            * self.MODAL_MAX_TEXTURE_HEIGHT
            * self.MODAL_MAX_TEXTURE_WIDTH
            * self.channels
            * np.dtype(np.uint8).itemsize
        )
        metadata_bytes = self.SLOT_COUNT * self.MODAL_SLOT_METADATA_STRUCT.size
        total_bytes = self.MODAL_HEADER_STRUCT.size + metadata_bytes + texture_bytes
        self.shared_overlay_modal_path = self._configured_modal_path
        self._overlay_modal_file, self._overlay_modal_mmap = self._create_exact_mmap(
            self.shared_overlay_modal_path, total_bytes
        )
        self._overlay_modal_mmap[
            self._overlay_modal_slot_metadata_offset : self._overlay_modal_payload_offset
        ] = b"\x00" * metadata_bytes
        self._overlay_modal_array = np.ndarray(
            (
                self.SLOT_COUNT,
                self.MODAL_MAX_TEXTURE_HEIGHT,
                self.MODAL_MAX_TEXTURE_WIDTH,
                self.channels,
            ),
            dtype=np.uint8,
            buffer=self._overlay_modal_mmap,
            offset=self._overlay_modal_payload_offset,
        )
        self._overlay_modal_array.fill(0)
        self._write_overlay_modal_header(latest_modal_id=0)

    def start(self) -> None:
        self._create_shared_frame_file()
        self._after_create_shared_frame_file()
        self._initialize_direct_commit_path()

        self._input_socket_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._input_socket_path.unlink(missing_ok=True)
        except TypeError:
            if self._input_socket_path.exists():
                self._input_socket_path.unlink()
        self._input_socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self._input_socket.bind(str(self._input_socket_path))
        self._input_socket.settimeout(0.1)
        self._input_stop.clear()
        self._input_thread = threading.Thread(
            target=self._receive_input,
            name="boba-illixr-input",
            daemon=True,
        )
        self._input_thread.start()
        self._start_stage_commit_thread()
        print(
            "[quest_display] input_source=illixr_switchboard "
            f"socket={self._input_socket_path} shared_frame={self.shared_frame_path}",
            flush=True,
        )

    def stop(self) -> None:
        self._input_stop.set()
        if self._input_socket is not None:
            self._input_socket.close()
            self._input_socket = None
        if self._input_thread is not None:
            self._input_thread.join(timeout=2.0)
            self._input_thread = None
        try:
            self._input_socket_path.unlink(missing_ok=True)
        except TypeError:
            if self._input_socket_path.exists():
                self._input_socket_path.unlink()
        super().stop()

    def _receive_input(self) -> None:
        while not self._input_stop.is_set():
            input_socket = self._input_socket
            if input_socket is None:
                return
            try:
                packet = input_socket.recv(self.INPUT_PACKET_BYTE_COUNT + 1)
            except socket.timeout:
                continue
            except OSError as exc:
                if not self._input_stop.is_set():
                    self._input_error = f"{type(exc).__name__}: {exc}"
                return
            try:
                sample = self._parse_input_packet(packet)
            except Exception as exc:
                self._input_error = f"{type(exc).__name__}: {exc}"
                continue
            sample.received_monotonic_s = time.monotonic()
            with self._sample_condition:
                self._latest_sample = sample
                self._received_packet_count += 1
                self._sample_condition.notify_all()

    @classmethod
    def _parse_input_packet(cls, packet: bytes) -> LiveImmersiveSample:
        if len(packet) != cls.INPUT_PACKET_BYTE_COUNT:
            raise ValueError(
                f"ILLIXR input packet size {len(packet)} != {cls.INPUT_PACKET_BYTE_COUNT}"
            )
        magic, version, byte_count, sequence, _xr_sample_time = (
            cls.INPUT_HEADER_STRUCT.unpack_from(packet, 0)
        )
        if magic != cls.INPUT_MAGIC:
            raise ValueError(f"unexpected ILLIXR input magic {magic!r}")
        if version != cls.INPUT_VERSION:
            raise ValueError(f"unsupported ILLIXR input version {version}")
        if byte_count != cls.INPUT_PACKET_BYTE_COUNT:
            raise ValueError(f"ILLIXR packet declares {byte_count} bytes")

        offset = cls.INPUT_HEADER_STRUCT.size
        left_eye, offset = cls._parse_eye(packet, offset)
        right_eye, offset = cls._parse_eye(packet, offset)
        left, offset = cls._parse_controller(packet, offset)
        right, offset = cls._parse_controller(packet, offset)
        if offset != len(packet):
            raise ValueError(f"ILLIXR input parser stopped at {offset} of {len(packet)}")
        return LiveImmersiveSample(
            sample=int(sequence),
            left=left,
            right=right,
            left_eye=left_eye,
            right_eye=right_eye,
        )

    @classmethod
    def _parse_eye(cls, packet: bytes, offset: int):
        values = cls.INPUT_EYE_STRUCT.unpack_from(packet, offset)
        flags, recommended_width, recommended_height = values[:3]
        return (
            EyePoseSample(
                pose_valid=bool(flags & cls.FLAG_POSITION_VALID)
                and bool(flags & cls.FLAG_ORIENTATION_VALID),
                pose_tracked=bool(flags & cls.FLAG_POSITION_TRACKED)
                and bool(flags & cls.FLAG_ORIENTATION_TRACKED),
                position=np.asarray(values[3:6], dtype=np.float32),
                orientation=np.asarray(values[6:10], dtype=np.float32),
                fov=EyeFovSample(
                    angle_left=float(values[10]),
                    angle_right=float(values[11]),
                    angle_up=float(values[12]),
                    angle_down=float(values[13]),
                ),
                recommended_width=int(recommended_width),
                recommended_height=int(recommended_height),
            ),
            offset + cls.INPUT_EYE_STRUCT.size,
        )

    @classmethod
    def _parse_pose(cls, packet: bytes, offset: int):
        values = cls.INPUT_POSE_STRUCT.unpack_from(packet, offset)
        flags = int(values[0])
        pose = {
            "active": bool(flags & cls.FLAG_ACTIVE),
            "position_valid": bool(flags & cls.FLAG_POSITION_VALID),
            "orientation_valid": bool(flags & cls.FLAG_ORIENTATION_VALID),
            "position_tracked": bool(flags & cls.FLAG_POSITION_TRACKED),
            "orientation_tracked": bool(flags & cls.FLAG_ORIENTATION_TRACKED),
            "position": np.asarray(values[1:4], dtype=np.float32),
            "orientation": np.asarray(values[4:8], dtype=np.float32),
        }
        return pose, offset + cls.INPUT_POSE_STRUCT.size

    @classmethod
    def _parse_button(cls, packet: bytes, offset: int):
        flags, value = cls.INPUT_BUTTON_STRUCT.unpack_from(packet, offset)
        return (
            bool(flags & cls.FLAG_ACTIVE),
            bool(flags & cls.FLAG_PRESSED),
            float(value),
            offset + cls.INPUT_BUTTON_STRUCT.size,
        )

    @classmethod
    def _parse_controller(cls, packet: bytes, offset: int):
        available_flags, _profile = struct.unpack_from("<II", packet, offset)
        offset += 8
        grip, offset = cls._parse_pose(packet, offset)
        aim, offset = cls._parse_pose(packet, offset)
        trigger_active, trigger_pressed, trigger_value, offset = cls._parse_button(
            packet, offset
        )
        squeeze_active, squeeze_pressed, squeeze_value, offset = cls._parse_button(
            packet, offset
        )
        primary_active, primary_pressed, _primary_value, offset = cls._parse_button(
            packet, offset
        )
        secondary_active, secondary_pressed, _secondary_value, offset = (
            cls._parse_button(packet, offset)
        )
        thumb_click_active, thumb_click_pressed, _thumb_click_value, offset = (
            cls._parse_button(packet, offset)
        )
        axis_flags, thumbstick_x, thumbstick_y = cls.INPUT_AXIS_STRUCT.unpack_from(
            packet, offset
        )
        offset += cls.INPUT_AXIS_STRUCT.size

        selected = grip
        source = "grip"
        aim_preferred = (
            aim["active"]
            and (aim["position_valid"] or aim["orientation_valid"])
        ) or (not grip["active"] and aim["active"]) or (
            not grip["position_valid"]
            and not grip["orientation_valid"]
            and (aim["position_valid"] or aim["orientation_valid"])
        )
        if aim_preferred:
            selected = aim
            source = "aim"

        available = bool(available_flags & cls.FLAG_ACTIVE)
        sample = ControllerPoseSample(
            source=source,
            active=bool(selected["active"] and available),
            position_valid=bool(selected["position_valid"]),
            orientation_valid=bool(selected["orientation_valid"]),
            position_tracked=bool(selected["position_tracked"]),
            orientation_tracked=bool(selected["orientation_tracked"]),
            position=selected["position"],
            orientation=selected["orientation"],
            select_available=trigger_active,
            select_pressed=trigger_pressed,
            select_value=trigger_value,
            select_source="illixr_trigger",
            anchor_cycle_available=primary_active,
            anchor_cycle_pressed=primary_pressed,
            anchor_cycle_source="illixr_primary",
            anchor_reset_available=thumb_click_active,
            anchor_reset_pressed=thumb_click_pressed,
            anchor_reset_source="illixr_thumbstick_click",
            snap_assist_available=secondary_active,
            snap_assist_pressed=secondary_pressed,
            snap_assist_source="illixr_secondary",
            exit_available=squeeze_active,
            exit_pressed=squeeze_pressed,
            exit_value=squeeze_value,
            exit_source="illixr_squeeze",
            thumbstick_available=bool(axis_flags & cls.FLAG_ACTIVE),
            thumbstick_x=float(thumbstick_x),
            thumbstick_y=float(thumbstick_y),
            grip_active=grip["active"],
            grip_position_valid=grip["position_valid"],
            grip_orientation_valid=grip["orientation_valid"],
            grip_position_tracked=grip["position_tracked"],
            grip_orientation_tracked=grip["orientation_tracked"],
            grip_position=grip["position"],
            grip_orientation=grip["orientation"],
            aim_active=aim["active"],
            aim_position_valid=aim["position_valid"],
            aim_orientation_valid=aim["orientation_valid"],
            aim_position_tracked=aim["position_tracked"],
            aim_orientation_tracked=aim["orientation_tracked"],
            aim_position=aim["position"],
            aim_orientation=aim["orientation"],
        )
        return sample, offset

    def debug_summary(self) -> str:
        parts = [
            "ILLIXR bridge: "
            f"received_packets={self._received_packet_count} "
            f"socket={self._input_socket_path}"
        ]
        if self._input_error:
            parts.append(f"last input error: {self._input_error}")
        base_summary = super().debug_summary()
        if base_summary and base_summary != "no panel diagnostics available":
            parts.append(base_summary)
        return "\n\n".join(parts)
