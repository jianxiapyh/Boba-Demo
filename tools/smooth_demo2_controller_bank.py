#!/usr/bin/env python3
"""Convert Demo 2 controller paths into smooth, closed, deliberate loops."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


def parse_keyframes(value: str) -> list[int]:
    try:
        frames = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("keyframes must be comma-separated integers") from exc
    if len(frames) < 2:
        raise argparse.ArgumentTypeError("at least two keyframes are required")
    if frames[0] != 0:
        raise argparse.ArgumentTypeError("the first keyframe must be frame 0")
    if frames != sorted(set(frames)):
        raise argparse.ArgumentTypeError("keyframes must be unique and increasing")
    return frames


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--keyframes", type=parse_keyframes, default=parse_keyframes("0,29,59,82"))
    parser.add_argument("--transition-frames", type=positive_int, default=45)
    parser.add_argument("--return-frames", type=positive_int, default=60)
    parser.add_argument("--start-hold-frames", type=nonnegative_int, default=10)
    parser.add_argument("--keyframe-hold-frames", type=nonnegative_int, default=8)
    parser.add_argument("--final-hold-frames", type=nonnegative_int, default=30)
    return parser


def minimum_jerk_transition(
    start: np.ndarray,
    end: np.ndarray,
    frame_count: int,
) -> list[np.ndarray]:
    """Interpolate with zero velocity and acceleration at both endpoints."""
    u = np.arange(1, frame_count + 1, dtype=np.float64) / float(frame_count)
    blend = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
    delta = end.astype(np.float64) - start.astype(np.float64)
    return [
        (start.astype(np.float64) + amount * delta).astype(start.dtype)
        for amount in blend
    ]


def smooth_trajectory(
    source: np.ndarray,
    keyframes: list[int],
    transition_frames: int,
    return_frames: int,
    start_hold_frames: int,
    keyframe_hold_frames: int,
    final_hold_frames: int,
) -> np.ndarray:
    poses = [source[index].copy() for index in keyframes]
    result = [poses[0].copy() for _ in range(start_hold_frames)]
    if not result:
        result.append(poses[0].copy())

    current = poses[0]
    for target in poses[1:]:
        result.extend(minimum_jerk_transition(current, target, transition_frames))
        result.extend(target.copy() for _ in range(keyframe_hold_frames))
        current = target

    result.extend(minimum_jerk_transition(current, poses[0], return_frames))
    result.extend(poses[0].copy() for _ in range(final_hold_frames))
    if not np.array_equal(result[-1], poses[0]):
        result.append(poses[0].copy())
    return np.stack(result, axis=0)


def main() -> None:
    args = build_parser().parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")

    with args.input.open("rb") as handle:
        source_root = pickle.load(handle)
    if not isinstance(source_root, dict):
        raise TypeError("controller bank root must be a dictionary")
    trajectories = source_root.get("controller_points_group")
    if not isinstance(trajectories, list) or not trajectories:
        raise TypeError("controller_points_group must be a non-empty list")

    arrays = [np.asarray(item) for item in trajectories]
    expected_shape = arrays[0].shape
    if len(expected_shape) != 3 or expected_shape[-1] != 3:
        raise ValueError(f"expected trajectory shape (T,C,3), got {expected_shape}")
    for index, array in enumerate(arrays):
        if array.shape != expected_shape:
            raise ValueError(
                f"trajectory {index} has shape {array.shape}; expected {expected_shape}"
            )
    if args.keyframes[-1] >= expected_shape[0]:
        raise IndexError(
            f"keyframe {args.keyframes[-1]} exceeds input length {expected_shape[0]}"
        )

    smoothed = [
        smooth_trajectory(
            trajectory,
            args.keyframes,
            args.transition_frames,
            args.return_frames,
            args.start_hold_frames,
            args.keyframe_hold_frames,
            args.final_hold_frames,
        )
        for trajectory in arrays
    ]
    output_frames = int(smoothed[0].shape[0])
    smoothing = {
        "method": "minimum_jerk_closed_loop",
        "keyframes": list(args.keyframes),
        "transition_frames": int(args.transition_frames),
        "return_frames": int(args.return_frames),
        "start_hold_frames": int(args.start_hold_frames),
        "keyframe_hold_frames": int(args.keyframe_hold_frames),
        "final_hold_frames": int(args.final_hold_frames),
        "input_frames": int(expected_shape[0]),
        "output_frames": output_frames,
    }

    output_root = dict(source_root)
    output_root["controller_points_group"] = smoothed
    output_root["smoothed_from"] = str(args.input)
    output_root["smoothing"] = smoothing
    output_root["replay_start"] = 0
    output_root["replay_end"] = None
    metadata = dict(source_root.get("meta") or {})
    metadata["frames"] = output_frames
    metadata["smoothing"] = smoothing
    output_root["meta"] = metadata

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump(output_root, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(
        f"Wrote {len(smoothed)} smooth trajectories with {output_frames} frames "
        f"to {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
