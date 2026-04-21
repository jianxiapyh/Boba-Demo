from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import threading
import time
from typing import Optional

import numpy as np

HAND_JOINT_COUNT = 26
PALM_JOINT_INDEX = 0
WRIST_JOINT_INDEX = 1
THUMB_TIP_JOINT_INDEX = 5
INDEX_TIP_JOINT_INDEX = 10
MIDDLE_TIP_JOINT_INDEX = 15
CONTROLLER_FORWARD = np.asarray([0.0, 0.0, -1.0], dtype=np.float32)


def _prepend_env_path(env: dict[str, str], key: str, value: str) -> None:
    current = env.get(key, "")
    env[key] = f"{value}:{current}" if current else value


def _ensure_jsoncpp_compat_dir(repo_root: Path) -> Optional[str]:
    compat_dir = repo_root / "linux_pose_probe" / ".compat_libs"
    compat_link = compat_dir / "libjsoncpp.so.1"
    if compat_link.exists():
        return str(compat_dir)

    candidates = (
        Path("/usr/lib/x86_64-linux-gnu/libjsoncpp.so.1.9.5"),
        Path("/lib/x86_64-linux-gnu/libjsoncpp.so.1.9.5"),
        Path("/usr/lib/x86_64-linux-gnu/libjsoncpp.so.25"),
        Path("/lib/x86_64-linux-gnu/libjsoncpp.so.25"),
    )
    target = next((candidate for candidate in candidates if candidate.exists()), None)
    if target is None:
        return None

    compat_dir.mkdir(parents=True, exist_ok=True)
    try:
        compat_link.symlink_to(target)
    except FileExistsError:
        pass
    return str(compat_dir)


@dataclass
class HandJointSample:
    active: bool
    valid: np.ndarray
    joints: np.ndarray


@dataclass
class LiveHandSample:
    sample: int
    left: HandJointSample
    right: HandJointSample


@dataclass
class ControllerPoseSample:
    source: str
    active: bool
    position_valid: bool
    orientation_valid: bool
    position_tracked: bool
    orientation_tracked: bool
    position: np.ndarray
    orientation: np.ndarray
    select_available: bool
    select_pressed: bool
    select_value: float
    select_source: str
    anchor_cycle_available: bool
    anchor_cycle_pressed: bool
    anchor_cycle_source: str
    anchor_reset_available: bool
    anchor_reset_pressed: bool
    anchor_reset_source: str
    snap_assist_available: bool
    snap_assist_pressed: bool
    snap_assist_source: str
    exit_available: bool
    exit_pressed: bool
    exit_value: float
    exit_source: str
    grip_active: bool = False
    grip_position_valid: bool = False
    grip_orientation_valid: bool = False
    grip_position_tracked: bool = False
    grip_orientation_tracked: bool = False
    grip_position: Optional[np.ndarray] = None
    grip_orientation: Optional[np.ndarray] = None
    aim_active: bool = False
    aim_position_valid: bool = False
    aim_orientation_valid: bool = False
    aim_position_tracked: bool = False
    aim_orientation_tracked: bool = False
    aim_position: Optional[np.ndarray] = None
    aim_orientation: Optional[np.ndarray] = None


@dataclass
class LiveControllerSample:
    sample: int
    left: ControllerPoseSample
    right: ControllerPoseSample


@dataclass
class EyeFovSample:
    angle_left: float
    angle_right: float
    angle_up: float
    angle_down: float


@dataclass
class EyePoseSample:
    pose_valid: bool
    pose_tracked: bool
    position: np.ndarray
    orientation: np.ndarray
    fov: EyeFovSample
    recommended_width: int
    recommended_height: int


@dataclass
class LiveImmersiveSample:
    sample: int
    left: ControllerPoseSample
    right: ControllerPoseSample
    left_eye: EyePoseSample
    right_eye: EyePoseSample
    received_monotonic_s: float | None = None


def hand_anchor(hand: HandJointSample) -> Optional[np.ndarray]:
    if hand.valid.shape[0] > PALM_JOINT_INDEX and hand.valid[PALM_JOINT_INDEX]:
        return hand.joints[PALM_JOINT_INDEX]

    valid_joints = hand.joints[hand.valid]
    if valid_joints.size == 0:
        return None
    return valid_joints.mean(axis=0)


def quaternion_rotate_vector(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    x, y, z, w = [float(v) for v in quaternion]
    q_xyz = np.asarray([x, y, z], dtype=np.float32)
    uv = np.cross(q_xyz, vector)
    uuv = np.cross(q_xyz, uv)
    return vector + 2.0 * (w * uv + uuv)


def controller_forward(sample: ControllerPoseSample) -> Optional[np.ndarray]:
    if not sample.active or not sample.orientation_valid:
        return None

    direction = quaternion_rotate_vector(sample.orientation, CONTROLLER_FORWARD)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        return None
    return direction / norm


def _parse_pose_vector(
    payload: dict,
    key: str,
    *,
    fallback: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    if key in payload:
        value = np.asarray(payload[key], dtype=np.float32)
    elif fallback is not None:
        value = np.asarray(fallback, dtype=np.float32)
    else:
        return None
    if value.shape != (3,) and value.shape != (4,):
        raise ValueError(f"{key} shape {value.shape} is not supported")
    return value


def controller_pose_position(
    sample: Optional[ControllerPoseSample],
    pose_role: str = "selected",
) -> Optional[np.ndarray]:
    if sample is None or not sample.active:
        return None
    if pose_role == "grip":
        if sample.grip_position is not None:
            return sample.grip_position if sample.grip_position_valid else None
    elif pose_role == "aim":
        if sample.aim_position is not None:
            return sample.aim_position if sample.aim_position_valid else None
    return sample.position if sample.position_valid else None


def controller_pose_forward(
    sample: Optional[ControllerPoseSample],
    pose_role: str = "selected",
) -> Optional[np.ndarray]:
    if sample is None or not sample.active:
        return None
    if pose_role == "grip":
        if sample.grip_orientation is not None:
            if not sample.grip_orientation_valid:
                return None
            direction = quaternion_rotate_vector(sample.grip_orientation, CONTROLLER_FORWARD)
        else:
            direction = None
    elif pose_role == "aim":
        if sample.aim_orientation is not None:
            if not sample.aim_orientation_valid:
                return None
            direction = quaternion_rotate_vector(sample.aim_orientation, CONTROLLER_FORWARD)
        else:
            direction = None
    else:
        direction = None
    if direction is None:
        if not sample.orientation_valid:
            return None
        direction = quaternion_rotate_vector(sample.orientation, CONTROLLER_FORWARD)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        return None
    return direction / norm


def parse_controller_payload(payload: dict) -> ControllerPoseSample:
    position = np.asarray(payload["position"], dtype=np.float32)
    orientation = np.asarray(payload["orientation"], dtype=np.float32)
    if position.shape != (3,):
        raise ValueError(f"controller position shape {position.shape} != (3,)")
    if orientation.shape != (4,):
        raise ValueError(f"controller orientation shape {orientation.shape} != (4,)")
    grip_position = _parse_pose_vector(payload, "grip_position")
    grip_orientation = _parse_pose_vector(payload, "grip_orientation")
    aim_position = _parse_pose_vector(payload, "aim_position")
    aim_orientation = _parse_pose_vector(payload, "aim_orientation")
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
        anchor_reset_available=bool(payload.get("anchor_reset_available", False)),
        anchor_reset_pressed=bool(payload.get("anchor_reset_pressed", False)),
        anchor_reset_source=str(payload.get("anchor_reset_source", "none")),
        snap_assist_available=bool(payload["snap_assist_available"]),
        snap_assist_pressed=bool(payload["snap_assist_pressed"]),
        snap_assist_source=str(payload["snap_assist_source"]),
        exit_available=bool(payload.get("exit_available", False)),
        exit_pressed=bool(payload.get("exit_pressed", False)),
        exit_value=float(payload.get("exit_value", 0.0)),
        exit_source=str(payload.get("exit_source", "none")),
        grip_active=bool(payload.get("grip_active", payload["active"])),
        grip_position_valid=bool(payload.get("grip_position_valid", payload["position_valid"])),
        grip_orientation_valid=bool(
            payload.get("grip_orientation_valid", payload["orientation_valid"])
        ),
        grip_position_tracked=bool(
            payload.get("grip_position_tracked", payload["position_tracked"])
        ),
        grip_orientation_tracked=bool(
            payload.get("grip_orientation_tracked", payload["orientation_tracked"])
        ),
        grip_position=grip_position,
        grip_orientation=grip_orientation,
        aim_active=bool(payload.get("aim_active", payload["active"])),
        aim_position_valid=bool(payload.get("aim_position_valid", payload["position_valid"])),
        aim_orientation_valid=bool(
            payload.get("aim_orientation_valid", payload["orientation_valid"])
        ),
        aim_position_tracked=bool(
            payload.get("aim_position_tracked", payload["position_tracked"])
        ),
        aim_orientation_tracked=bool(
            payload.get("aim_orientation_tracked", payload["orientation_tracked"])
        ),
        aim_position=aim_position,
        aim_orientation=aim_orientation,
    )


class OpenXRHandJointStream:
    def __init__(self, repo_root: Path):
        self.repo_root = Path(repo_root)
        self.binary_path = self.repo_root / "linux_pose_probe" / "openxr_hand_joint_stream"
        self.build_script_path = self.repo_root / "linux_pose_probe" / "build_openxr_hand_joint_stream.sh"
        self.source_path = self.repo_root / "linux_pose_probe" / "openxr_hand_joint_stream.cpp"
        self.process: Optional[subprocess.Popen[str]] = None
        self._stdout_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._latest_sample: Optional[LiveHandSample] = None
        self._latest_lock = threading.Lock()
        self._stderr_tail: deque[str] = deque(maxlen=50)
        self._parse_errors: deque[str] = deque(maxlen=10)

    def start(self) -> None:
        self._ensure_binary()
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

        self.process = subprocess.Popen(
            [str(self.binary_path), "0"],
            cwd=self.repo_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._stdout_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self._stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self._stdout_thread.start()
        self._stderr_thread.start()

    def stop(self) -> None:
        if self.process is None:
            return

        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5.0)

        self.process = None

    def wait_for_sample(self, timeout: float = 10.0) -> LiveHandSample:
        deadline = time.time() + timeout
        while time.time() < deadline:
            sample = self.get_latest_sample()
            if sample is not None:
                return sample
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(self.debug_summary())
            time.sleep(0.05)

        raise RuntimeError(f"Timed out waiting for OpenXR hand sample.\n{self.debug_summary()}")

    def get_latest_sample(self) -> Optional[LiveHandSample]:
        with self._latest_lock:
            return self._latest_sample

    def debug_summary(self) -> str:
        stderr_text = "".join(self._stderr_tail).strip()
        parse_errors = "\n".join(self._parse_errors)
        parts = []
        if stderr_text:
            parts.append(f"stderr:\n{stderr_text}")
        if parse_errors:
            parts.append(f"parse errors:\n{parse_errors}")
        return "\n\n".join(parts) if parts else "no stream diagnostics available"

    def _ensure_binary(self) -> None:
        if self.binary_path.exists() and self.binary_path.stat().st_mtime >= max(
            self.build_script_path.stat().st_mtime,
            self.source_path.stat().st_mtime,
        ):
            return

        subprocess.run(
            ["bash", str(self.build_script_path)],
            cwd=self.repo_root,
            check=True,
            text=True,
        )

    def _read_stdout(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        for line in self.process.stdout:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
                sample = self._parse_sample(payload)
            except Exception as exc:  # pragma: no cover - debug path
                self._parse_errors.append(f"{exc}: {stripped}")
                continue

            with self._latest_lock:
                self._latest_sample = sample

    def _read_stderr(self) -> None:
        assert self.process is not None and self.process.stderr is not None
        for line in self.process.stderr:
            self._stderr_tail.append(line)

    def _parse_sample(self, payload: dict) -> LiveHandSample:
        return LiveHandSample(
            sample=int(payload["sample"]),
            left=self._parse_hand(payload, "left"),
            right=self._parse_hand(payload, "right"),
        )

    @staticmethod
    def _parse_hand(payload: dict, prefix: str) -> HandJointSample:
        valid = np.asarray(payload[f"{prefix}_valid"], dtype=bool)
        joints = np.asarray(payload[f"{prefix}_positions"], dtype=np.float32)
        if valid.shape != (HAND_JOINT_COUNT,):
            raise ValueError(f"{prefix} valid shape {valid.shape} != {(HAND_JOINT_COUNT,)}")
        if joints.shape != (HAND_JOINT_COUNT, 3):
            raise ValueError(f"{prefix} joint shape {joints.shape} != {(HAND_JOINT_COUNT, 3)}")
        return HandJointSample(
            active=bool(payload[f"{prefix}_active"]),
            valid=valid,
            joints=joints,
        )

    def _default_runtime_json_path(self) -> Optional[str]:
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

    def _default_steamvr_lib_dir(self) -> Optional[str]:
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


class OpenXRControllerStream:
    def __init__(self, repo_root: Path):
        self.repo_root = Path(repo_root)
        self.binary_path = self.repo_root / "linux_pose_probe" / "openxr_controller_stream"
        self.build_script_path = self.repo_root / "linux_pose_probe" / "build_openxr_controller_stream.sh"
        self.source_path = self.repo_root / "linux_pose_probe" / "openxr_controller_stream.cpp"
        self.process: Optional[subprocess.Popen[str]] = None
        self._stdout_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._latest_sample: Optional[LiveControllerSample] = None
        self._latest_lock = threading.Lock()
        self._stderr_tail: deque[str] = deque(maxlen=50)
        self._parse_errors: deque[str] = deque(maxlen=10)

    def start(self) -> None:
        self._ensure_binary()
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

        self.process = subprocess.Popen(
            [str(self.binary_path), "0"],
            cwd=self.repo_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._stdout_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self._stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self._stdout_thread.start()
        self._stderr_thread.start()

    def stop(self) -> None:
        if self.process is None:
            return

        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5.0)

        self.process = None

    def wait_for_sample(self, timeout: float = 10.0) -> LiveControllerSample:
        deadline = time.time() + timeout
        while time.time() < deadline:
            sample = self.get_latest_sample()
            if sample is not None:
                return sample
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(self.debug_summary())
            time.sleep(0.05)

        raise RuntimeError(
            f"Timed out waiting for OpenXR controller sample.\n{self.debug_summary()}"
        )

    def get_latest_sample(self) -> Optional[LiveControllerSample]:
        with self._latest_lock:
            return self._latest_sample

    def debug_summary(self) -> str:
        stderr_text = "".join(self._stderr_tail).strip()
        parse_errors = "\n".join(self._parse_errors)
        parts = []
        if stderr_text:
            parts.append(f"stderr:\n{stderr_text}")
        if parse_errors:
            parts.append(f"parse errors:\n{parse_errors}")
        return "\n\n".join(parts) if parts else "no stream diagnostics available"

    def _ensure_binary(self) -> None:
        if self.binary_path.exists() and self.binary_path.stat().st_mtime >= max(
            self.build_script_path.stat().st_mtime,
            self.source_path.stat().st_mtime,
        ):
            return

        subprocess.run(
            ["bash", str(self.build_script_path)],
            cwd=self.repo_root,
            check=True,
            text=True,
        )

    def _read_stdout(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        for line in self.process.stdout:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
                sample = self._parse_sample(payload)
            except Exception as exc:  # pragma: no cover - debug path
                self._parse_errors.append(f"{exc}: {stripped}")
                continue

            with self._latest_lock:
                self._latest_sample = sample

    def _read_stderr(self) -> None:
        assert self.process is not None and self.process.stderr is not None
        for line in self.process.stderr:
            self._stderr_tail.append(line)

    def _parse_sample(self, payload: dict) -> LiveControllerSample:
        return LiveControllerSample(
            sample=int(payload["sample"]),
            left=self._parse_controller(payload["left"]),
            right=self._parse_controller(payload["right"]),
        )

    @staticmethod
    def _parse_controller(payload: dict) -> ControllerPoseSample:
        return parse_controller_payload(payload)

    def _default_runtime_json_path(self) -> Optional[str]:
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

    def _default_steamvr_lib_dir(self) -> Optional[str]:
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
