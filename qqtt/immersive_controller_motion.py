from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import torch


# Calibrated across all 22 recorded 30 FPS trajectories in
# Boba_Latest/data/different_types. The exact maximum consecutive driven-point
# displacement was 0.047083356017 m (weird_package, frame 24 -> 25, point 23).
# Use a practical 5 cm production bound, leaving 2.92 mm (6.2%) headroom over
# the measured maximum for replay and floating-point variation.
RECORDED_TEST_TRAJECTORY_MAX_CONTROLLER_STEP_M = 0.047083356017
DEFAULT_MAX_CONTROLLER_MOTION_INTERVAL_M = 0.05
CONTROLLER_MOTION_DISTANCE_EPSILON_M = 1.0e-7


@dataclass(frozen=True)
class ControllerMotionIntervalPlan:
    max_distance_m: float
    interval_count: int
    active_point_count: int

    def interpolation_bounds(self, interval_index: int) -> tuple[float, float]:
        interval_index = int(interval_index)
        if interval_index < 0 or interval_index >= self.interval_count:
            raise IndexError(
                f"controller motion interval {interval_index} is outside "
                f"[0, {self.interval_count})"
            )
        inverse_count = 1.0 / float(self.interval_count)
        return (
            float(interval_index) * inverse_count,
            float(interval_index + 1) * inverse_count,
        )


@dataclass(frozen=True)
class ControllerMotionGroupAdvance:
    requested_distance_m: float
    applied_distance_m: float
    remaining_distance_m: float
    catchup_period_count: int
    active_point_count: int
    limited: bool


@dataclass(frozen=True)
class ControllerMotionTargetAdvance:
    target: torch.Tensor
    group_advances: tuple[ControllerMotionGroupAdvance, ...]

    @property
    def limited_group_count(self) -> int:
        return sum(int(group.limited) for group in self.group_advances)

    @property
    def max_requested_distance_m(self) -> float:
        return max(
            (group.requested_distance_m for group in self.group_advances),
            default=0.0,
        )

    @property
    def max_applied_distance_m(self) -> float:
        return max(
            (group.applied_distance_m for group in self.group_advances),
            default=0.0,
        )

    @property
    def max_remaining_distance_m(self) -> float:
        return max(
            (group.remaining_distance_m for group in self.group_advances),
            default=0.0,
        )

    @property
    def max_catchup_period_count(self) -> int:
        return max(
            (group.catchup_period_count for group in self.group_advances),
            default=1,
        )


def _validate_controller_targets(
    previous_target: torch.Tensor,
    current_target: torch.Tensor,
    max_interval_distance_m: float,
) -> float:
    max_interval_distance_m = float(max_interval_distance_m)
    if (
        not math.isfinite(max_interval_distance_m)
        or max_interval_distance_m <= 0.0
    ):
        raise ValueError("max_interval_distance_m must be finite and positive")
    if not torch.is_tensor(previous_target) or not torch.is_tensor(current_target):
        raise TypeError("controller targets must be torch tensors")
    if previous_target.shape != current_target.shape:
        raise ValueError(
            "controller target shapes differ: "
            f"{tuple(previous_target.shape)} != {tuple(current_target.shape)}"
        )
    if previous_target.ndim != 2 or previous_target.shape[1] != 3:
        raise ValueError(
            f"controller target shape {tuple(previous_target.shape)} != (N, 3)"
        )
    if previous_target.device != current_target.device:
        raise ValueError(
            "controller targets must be on the same device: "
            f"{previous_target.device} != {current_target.device}"
        )
    return max_interval_distance_m


def _normalize_active_point_index_groups(
    active_point_index_groups: Iterable[torch.Tensor],
    target: torch.Tensor,
) -> list[torch.Tensor]:
    normalized_groups = []
    point_count = int(target.shape[0])
    for point_indices in active_point_index_groups:
        if point_indices is None:
            continue
        point_indices = torch.as_tensor(
            point_indices,
            device=target.device,
            dtype=torch.long,
        ).reshape(-1)
        if int(point_indices.numel()) <= 0:
            continue
        # Production targets live on CUDA and their indices come from validated
        # controller masks. Avoid an extra device synchronization in that hot
        # path; index_select still rejects invalid CUDA indices. Keep the
        # explicit error for CPU callers and unit tests.
        if point_indices.device.type == "cpu":
            if bool(
                ((point_indices < 0) | (point_indices >= point_count)).any().item()
            ):
                raise IndexError(
                    "active controller target point index is out of range"
                )
        normalized_groups.append(point_indices)
    return normalized_groups


def _motion_period_count(distance_m: float, max_interval_distance_m: float) -> int:
    return max(
        1,
        int(
            math.ceil(
                max(0.0, distance_m - CONTROLLER_MOTION_DISTANCE_EPSILON_M)
                / max_interval_distance_m
            )
        ),
    )


def advance_controller_motion_target(
    simulated_target: torch.Tensor,
    desired_target: torch.Tensor,
    active_point_index_groups: Iterable[torch.Tensor],
    *,
    max_interval_distance_m: float = DEFAULT_MAX_CONTROLLER_MOTION_INTERVAL_M,
) -> ControllerMotionTargetAdvance:
    """Advance active grab targets by at most one bounded physics period.

    Each active group (normally one controller) advances independently toward
    the newest desired pose. Unfinished distance is intentionally left for a
    later rendered frame. Inactive points copy the desired target immediately;
    their attachment springs are disabled, so this also discards stale catch-up
    motion when a grab is released.
    """

    max_interval_distance_m = _validate_controller_targets(
        simulated_target,
        desired_target,
        max_interval_distance_m,
    )
    normalized_groups = _normalize_active_point_index_groups(
        active_point_index_groups,
        simulated_target,
    )
    next_target = desired_target.clone()
    group_advances = []
    for point_indices in normalized_groups:
        previous_points = simulated_target.index_select(0, point_indices)
        desired_points = desired_target.index_select(0, point_indices)
        displacement = desired_points - previous_points
        requested_distance_m = float(
            torch.linalg.vector_norm(displacement, dim=1).max().item()
        )
        if not math.isfinite(requested_distance_m):
            raise ValueError(
                "active controller target motion contains non-finite values"
            )
        catchup_period_count = _motion_period_count(
            requested_distance_m,
            max_interval_distance_m,
        )
        limited = bool(catchup_period_count > 1)
        if limited:
            applied_fraction = max_interval_distance_m / requested_distance_m
            advanced_points = previous_points + displacement * applied_fraction
            applied_distance_m = max_interval_distance_m
            remaining_distance_m = max(
                0.0,
                requested_distance_m - max_interval_distance_m,
            )
        else:
            advanced_points = desired_points
            applied_distance_m = requested_distance_m
            remaining_distance_m = 0.0
        next_target.index_copy_(0, point_indices, advanced_points)
        group_advances.append(
            ControllerMotionGroupAdvance(
                requested_distance_m=requested_distance_m,
                applied_distance_m=applied_distance_m,
                remaining_distance_m=remaining_distance_m,
                catchup_period_count=catchup_period_count,
                active_point_count=int(point_indices.numel()),
                limited=limited,
            )
        )
    return ControllerMotionTargetAdvance(
        target=next_target,
        group_advances=tuple(group_advances),
    )


def plan_controller_motion_intervals(
    previous_target: torch.Tensor,
    current_target: torch.Tensor,
    active_point_index_groups: Iterable[torch.Tensor],
    *,
    max_interval_distance_m: float = DEFAULT_MAX_CONTROLLER_MOTION_INTERVAL_M,
) -> ControllerMotionIntervalPlan:
    """Describe how many bounded periods a target displacement requires.

    This is a calibration/diagnostic helper. The interactive runtime uses
    :func:`advance_controller_motion_target` once per rendered frame; it must not
    execute all of these periods in a blocking loop before presentation.
    """

    max_interval_distance_m = _validate_controller_targets(
        previous_target,
        current_target,
        max_interval_distance_m,
    )
    normalized_groups = _normalize_active_point_index_groups(
        active_point_index_groups,
        previous_target,
    )
    if not normalized_groups:
        return ControllerMotionIntervalPlan(
            max_distance_m=0.0,
            interval_count=1,
            active_point_count=0,
        )

    active_point_indices = torch.cat(normalized_groups, dim=0)
    active_displacements = (
        current_target.index_select(0, active_point_indices)
        - previous_target.index_select(0, active_point_indices)
    )
    max_distance_m = float(
        torch.linalg.vector_norm(active_displacements, dim=1).max().item()
    )
    if not math.isfinite(max_distance_m):
        raise ValueError("active controller target motion contains non-finite values")
    interval_count = _motion_period_count(
        max_distance_m,
        max_interval_distance_m,
    )
    return ControllerMotionIntervalPlan(
        max_distance_m=max_distance_m,
        interval_count=interval_count,
        active_point_count=int(active_point_indices.numel()),
    )
