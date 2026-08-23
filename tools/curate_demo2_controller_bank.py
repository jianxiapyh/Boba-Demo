#!/usr/bin/env python3
"""Create a smaller Demo 2 controller bank by preserving selected trajectories."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


def parse_indices(value: str) -> list[int]:
    try:
        indices = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("indices must be comma-separated integers") from exc
    if not indices:
        raise argparse.ArgumentTypeError("at least one trajectory index is required")
    if len(indices) != len(set(indices)):
        raise argparse.ArgumentTypeError("trajectory indices must be unique")
    if min(indices) < 0:
        raise argparse.ArgumentTypeError("trajectory indices must be non-negative")
    return indices


def parse_z_scale(value: str) -> tuple[int, float]:
    try:
        index_text, factor_text = value.split(":", maxsplit=1)
        index = int(index_text.strip())
        factor = float(factor_text.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("z scale must use INDEX:FACTOR") from exc
    if index < 0:
        raise argparse.ArgumentTypeError("z-scale index must be non-negative")
    if factor <= 0:
        raise argparse.ArgumentTypeError("z-scale factor must be positive")
    return index, factor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--indices", type=parse_indices, required=True)
    parser.add_argument(
        "--z-scale",
        type=parse_z_scale,
        action="append",
        default=[],
        metavar="INDEX:FACTOR",
        help=(
            "Scale one selected source trajectory's z displacement around its "
            "frame-0 pose; may be repeated"
        ),
    )
    parser.add_argument("--purpose", default="curated Demo 2 controller bank")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")

    with args.input.open("rb") as handle:
        source = pickle.load(handle)
    if not isinstance(source, dict):
        raise TypeError("controller bank root must be a dictionary")
    trajectories = source.get("controller_points_group")
    if not isinstance(trajectories, list) or not trajectories:
        raise TypeError("controller_points_group must be a non-empty list")
    if max(args.indices) >= len(trajectories):
        raise IndexError(
            f"trajectory index {max(args.indices)} exceeds bank size {len(trajectories)}"
        )

    z_scales = dict(args.z_scale)
    if len(z_scales) != len(args.z_scale):
        raise ValueError("each --z-scale source index may be specified only once")
    unselected_scales = sorted(set(z_scales) - set(args.indices))
    if unselected_scales:
        raise ValueError(
            "--z-scale indices must also appear in --indices: "
            + ", ".join(str(index) for index in unselected_scales)
        )

    selected_trajectories = []
    for index in args.indices:
        trajectory = trajectories[index]
        factor = z_scales.get(index)
        if factor is None:
            selected_trajectories.append(trajectory)
            continue
        transformed = np.asarray(trajectory).copy()
        transformed[:, :, 2] = transformed[:1, :, 2] + factor * (
            transformed[:, :, 2] - transformed[:1, :, 2]
        )
        selected_trajectories.append(transformed)

    curated = dict(source)
    curated["controller_points_group"] = selected_trajectories
    original_source_indices = source.get("source_indices")
    if isinstance(original_source_indices, list) and len(original_source_indices) == len(
        trajectories
    ):
        curated["source_indices"] = [original_source_indices[index] for index in args.indices]
    curated["curated_bank_indices"] = list(args.indices)
    curated["curated_z_scales"] = z_scales
    curated["curated_from"] = str(args.input)
    curated["curation_purpose"] = str(args.purpose)

    metadata = dict(source.get("meta") or {})
    metadata["curated_bank_indices"] = list(args.indices)
    metadata["curated_z_scales"] = z_scales
    metadata["curation_purpose"] = str(args.purpose)
    curated["meta"] = metadata

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump(curated, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(
        f"Wrote {len(args.indices)} trajectories to {args.output}: {args.indices}",
        flush=True,
    )


if __name__ == "__main__":
    main()
