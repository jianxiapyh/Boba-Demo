"""Thread-safe replay state and phone-control annotations for Demo 2."""

from __future__ import annotations

import threading
import time

import numpy as np


VALID_HANDS = ("left", "right")
VALID_CONTROLS = ("xneg", "xpos", "yneg", "ypos", "zneg", "zpos")


def controls_from_world_delta(hand: str, delta, epsilon: float = 1e-7):
    """Map a controller's world-space motion to the matching phone buttons."""
    if hand not in VALID_HANDS:
        raise ValueError(f"Unknown hand: {hand!r}")
    delta = np.asarray(delta, dtype=np.float64)
    if delta.shape != (3,):
        raise ValueError(f"Expected a 3D delta, got shape {delta.shape}")
    epsilon = abs(float(epsilon))

    controls = []
    # Phone X/Y are mirrored relative to this scene's world axes; Z agrees.
    if abs(float(delta[0])) > epsilon:
        controls.append((hand, "xneg" if delta[0] < 0 else "xpos"))
    if abs(float(delta[1])) > epsilon:
        controls.append((hand, "ypos" if delta[1] > 0 else "yneg"))
    if abs(float(delta[2])) > epsilon:
        controls.append((hand, "zneg" if delta[2] < 0 else "zpos"))
    return tuple(controls)


def build_replay_action_table(
    controller_points_group,
    hand_indices,
    epsilon: float = 1e-7,
    motion_epsilon: float = 0.0,
):
    """Return active phone buttons for every trajectory and replay frame.

    ``controller_points_group`` has shape ``(N, T, C, 3)``. The controller
    motion within each interaction region is reduced to its mean translation,
    which is the phone control that best represents that rendered replay step.
    Frame zero is neutral because the runtime restores the instance directly to
    its reset state instead of applying a last-frame-to-first-frame movement.
    """
    points = np.asarray(controller_points_group)
    if points.ndim != 4 or points.shape[-1] != 3:
        raise ValueError(
            "controller_points_group must have shape (N,T,C,3), "
            f"got {points.shape}"
        )
    if len(hand_indices) not in (1, 2):
        raise ValueError("hand_indices must contain one or two interaction regions")
    motion_epsilon = abs(float(motion_epsilon))

    normalized_indices = []
    for indices in hand_indices:
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        if not len(indices):
            raise ValueError("each hand must contain at least one controller index")
        if int(indices.min()) < 0 or int(indices.max()) >= int(points.shape[2]):
            raise IndexError("hand controller index is outside the controller bank")
        normalized_indices.append(indices)

    deltas = np.zeros_like(points, dtype=np.float64)
    deltas[:, 1:] = points[:, 1:].astype(np.float64) - points[:, :-1].astype(
        np.float64
    )
    result = []
    for trajectory_delta in deltas:
        trajectory_actions = []
        for frame_delta in trajectory_delta:
            active = []
            for hand, indices in zip(VALID_HANDS, normalized_indices):
                mean_delta = frame_delta[indices].mean(axis=0)
                if float(np.linalg.norm(mean_delta)) <= motion_epsilon:
                    continue
                active.extend(controls_from_world_delta(hand, mean_delta, epsilon))
            trajectory_actions.append(tuple(active))
        result.append(tuple(trajectory_actions))
    return tuple(result)


class ReplayStateStore:
    """Publish replay cursor/action snapshots from the renderer to Flask."""

    def __init__(self, num_sessions: int, control_parts: int = 1):
        self.num_sessions = int(num_sessions)
        if self.num_sessions < 1:
            raise ValueError("num_sessions must be positive")
        self.control_parts = int(control_parts)
        if self.control_parts not in (1, 2):
            raise ValueError("control_parts must be 1 or 2")
        self._lock = threading.Lock()
        self._runtime_fps = None
        self._states = [None for _ in range(self.num_sessions)]

    def set_runtime_fps(self, runtime_fps: float) -> None:
        runtime_fps = float(runtime_fps)
        if runtime_fps <= 0:
            raise ValueError("runtime_fps must be positive")
        with self._lock:
            self._runtime_fps = runtime_fps

    def publish(self, sequence, replay_cursors, controls_by_session) -> None:
        if len(replay_cursors) != self.num_sessions:
            raise ValueError("replay cursor count does not match num_sessions")
        if len(controls_by_session) != self.num_sessions:
            raise ValueError("replay control count does not match num_sessions")

        published_at = time.monotonic()
        states = []
        for session_id, (frame_idx, controls) in enumerate(
            zip(replay_cursors, controls_by_session)
        ):
            normalized_controls = []
            for hand, control in controls:
                if hand not in VALID_HANDS or control not in VALID_CONTROLS:
                    raise ValueError(f"Invalid replay control: {(hand, control)!r}")
                normalized_controls.append({"hand": hand, "control": control})
            states.append(
                {
                    "session_id": session_id,
                    "sequence": int(sequence),
                    "frame_idx": int(frame_idx),
                    "controls": normalized_controls,
                    "published_at": published_at,
                }
            )
        with self._lock:
            self._states = states

    def get(self, session_id: int):
        session_id = int(session_id)
        if session_id < 0 or session_id >= self.num_sessions:
            raise IndexError(session_id)
        with self._lock:
            state = self._states[session_id]
            if state is None:
                return None
            return {
                **state,
                "controls": [dict(control) for control in state["controls"]],
                "control_parts": self.control_parts,
                "runtime_fps": self._runtime_fps,
            }
