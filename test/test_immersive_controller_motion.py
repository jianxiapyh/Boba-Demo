from __future__ import annotations

import json
from pathlib import Path

import torch
import unittest

from qqtt.immersive_controller_motion import (
    DEFAULT_MAX_CONTROLLER_MOTION_INTERVAL_M,
    RECORDED_TEST_TRAJECTORY_MAX_CONTROLLER_STEP_M,
    advance_controller_motion_target,
    plan_controller_motion_intervals,
)


class ControllerMotionIntervalPlanTests(unittest.TestCase):
    def test_default_is_five_cm_and_covers_the_recorded_22_case_maximum(self):
        self.assertEqual(DEFAULT_MAX_CONTROLLER_MOTION_INTERVAL_M, 0.05)
        self.assertGreaterEqual(
            DEFAULT_MAX_CONTROLLER_MOTION_INTERVAL_M,
            RECORDED_TEST_TRAJECTORY_MAX_CONTROLLER_STEP_M,
        )
        self.assertLess(
            DEFAULT_MAX_CONTROLLER_MOTION_INTERVAL_M
            - RECORDED_TEST_TRAJECTORY_MAX_CONTROLLER_STEP_M,
            0.003,
        )

    def test_vendored_calibration_records_all_22_case_maxima(self):
        calibration_path = (
            Path(__file__).resolve().parents[1]
            / "assets"
            / "controller_motion_calibration.json"
        )
        with calibration_path.open("r", encoding="utf-8") as handle:
            calibration = json.load(handle)

        self.assertEqual(calibration["source"]["case_count"], 22)
        self.assertEqual(len(calibration["case_maxima_m"]), 22)
        self.assertEqual(calibration["source"]["recording_fps"], 30)
        self.assertAlmostEqual(
            calibration["global_maximum"]["distance_m"],
            RECORDED_TEST_TRAJECTORY_MAX_CONTROLLER_STEP_M,
            places=12,
        )
        self.assertEqual(
            calibration["runtime_limit_m"],
            DEFAULT_MAX_CONTROLLER_MOTION_INTERVAL_M,
        )

    def test_ten_cm_motion_is_split_into_two_five_cm_intervals(self):
        previous = torch.zeros((3, 3), dtype=torch.float32)
        current = previous.clone()
        current[1, 0] = 0.10

        plan = plan_controller_motion_intervals(
            previous,
            current,
            [torch.tensor([1])],
            max_interval_distance_m=0.05,
        )

        self.assertEqual(plan.interval_count, 2)
        self.assertEqual(plan.active_point_count, 1)
        self.assertAlmostEqual(plan.max_distance_m, 0.10)
        self.assertEqual(plan.interpolation_bounds(0), (0.0, 0.5))
        self.assertEqual(plan.interpolation_bounds(1), (0.5, 1.0))

    def test_every_interpolated_interval_respects_the_motion_bound(self):
        previous = torch.zeros((2, 3), dtype=torch.float32)
        current = previous.clone()
        current[0] = torch.tensor([0.30, 0.40, 0.0])
        plan = plan_controller_motion_intervals(
            previous,
            current,
            [torch.tensor([0])],
            max_interval_distance_m=0.20,
        )

        self.assertEqual(plan.interval_count, 3)
        interval_ends = []
        for interval_index in range(plan.interval_count):
            alpha_start, alpha_end = plan.interpolation_bounds(interval_index)
            interval_start = torch.lerp(previous, current, alpha_start)
            interval_end = torch.lerp(previous, current, alpha_end)
            interval_distance = torch.linalg.vector_norm(
                interval_end[0] - interval_start[0]
            ).item()
            self.assertLessEqual(interval_distance, 0.20 + 1.0e-6)
            interval_ends.append(interval_end)

        self.assertTrue(torch.equal(interval_ends[-1], current))

    def test_motion_materially_above_an_exact_multiple_adds_an_interval(self):
        previous = torch.zeros((1, 3), dtype=torch.float32)
        current = previous.clone()
        current[0, 0] = 0.1001

        plan = plan_controller_motion_intervals(
            previous,
            current,
            [torch.tensor([0])],
            max_interval_distance_m=0.05,
        )

        self.assertEqual(plan.interval_count, 3)

    def test_inactive_target_reset_does_not_add_physics_intervals(self):
        previous = torch.zeros((4, 3), dtype=torch.float32)
        current = previous.clone()
        current[0, 0] = 10.0
        current[3, 1] = 0.10

        plan = plan_controller_motion_intervals(
            previous,
            current,
            [torch.tensor([3])],
            max_interval_distance_m=0.25,
        )

        self.assertEqual(plan.interval_count, 1)
        self.assertAlmostEqual(plan.max_distance_m, 0.10, places=6)

    def test_nonfinite_active_motion_is_rejected_before_simulation(self):
        previous = torch.zeros((1, 3), dtype=torch.float32)
        current = previous.clone()
        current[0, 2] = torch.inf

        with self.assertRaisesRegex(ValueError, "non-finite"):
            plan_controller_motion_intervals(
                previous,
                current,
                [torch.tensor([0])],
            )

    def test_half_meter_motion_uses_ten_five_cm_intervals(self):
        previous = torch.zeros((1, 3), dtype=torch.float32)
        current = previous.clone()
        current[0, 0] = 0.50

        plan = plan_controller_motion_intervals(
            previous,
            current,
            [torch.tensor([0])],
        )

        self.assertEqual(plan.interval_count, 10)


class ControllerMotionTargetAdvanceTests(unittest.TestCase):
    def test_half_meter_jump_advances_only_five_cm_in_one_displayed_period(self):
        simulated = torch.zeros((1, 3), dtype=torch.float32)
        desired = simulated.clone()
        desired[0, 0] = 0.50

        advance = advance_controller_motion_target(
            simulated,
            desired,
            [torch.tensor([0])],
        )

        self.assertAlmostEqual(float(advance.target[0, 0]), 0.05, places=6)
        self.assertEqual(advance.limited_group_count, 1)
        self.assertEqual(advance.max_catchup_period_count, 10)
        self.assertAlmostEqual(advance.max_applied_distance_m, 0.05, places=6)
        self.assertAlmostEqual(advance.max_remaining_distance_m, 0.45, places=6)

    def test_half_meter_jump_reaches_the_target_over_ten_displayed_periods(self):
        simulated = torch.zeros((1, 3), dtype=torch.float32)
        desired = simulated.clone()
        desired[0, 0] = 0.50

        displayed_positions = []
        for _ in range(10):
            advance = advance_controller_motion_target(
                simulated,
                desired,
                [torch.tensor([0])],
            )
            simulated = advance.target
            displayed_positions.append(float(simulated[0, 0]))

        for period_index, position in enumerate(displayed_positions, start=1):
            self.assertAlmostEqual(position, period_index * 0.05, places=5)
        self.assertTrue(torch.allclose(simulated, desired, atol=1.0e-6))

    def test_catchup_chases_the_latest_pose_instead_of_an_old_waypoint(self):
        simulated = torch.zeros((1, 3), dtype=torch.float32)
        first_desired = simulated.clone()
        first_desired[0, 0] = 0.50
        first = advance_controller_motion_target(
            simulated,
            first_desired,
            [torch.tensor([0])],
        )
        latest_desired = first.target.clone()
        latest_desired[0] = torch.tensor([0.05, 0.50, 0.0])

        second = advance_controller_motion_target(
            first.target,
            latest_desired,
            [torch.tensor([0])],
        )

        self.assertAlmostEqual(float(second.target[0, 0]), 0.05, places=6)
        self.assertAlmostEqual(float(second.target[0, 1]), 0.05, places=6)

    def test_two_controller_groups_are_limited_independently(self):
        simulated = torch.zeros((2, 3), dtype=torch.float32)
        desired = simulated.clone()
        desired[0, 0] = 0.50
        desired[1, 1] = 0.02

        advance = advance_controller_motion_target(
            simulated,
            desired,
            [torch.tensor([0]), torch.tensor([1])],
        )

        self.assertAlmostEqual(float(advance.target[0, 0]), 0.05, places=6)
        self.assertAlmostEqual(float(advance.target[1, 1]), 0.02, places=6)
        self.assertEqual(advance.limited_group_count, 1)

    def test_inactive_points_snap_to_desired_and_discard_pending_catchup(self):
        simulated = torch.zeros((2, 3), dtype=torch.float32)
        simulated[0, 0] = 0.05
        desired = torch.zeros_like(simulated)

        advance = advance_controller_motion_target(simulated, desired, [])

        self.assertTrue(torch.equal(advance.target, desired))
        self.assertEqual(advance.limited_group_count, 0)

    def test_exact_five_cm_motion_is_not_limited(self):
        simulated = torch.zeros((1, 3), dtype=torch.float32)
        desired = simulated.clone()
        desired[0, 2] = 0.05

        advance = advance_controller_motion_target(
            simulated,
            desired,
            [torch.tensor([0])],
        )

        self.assertTrue(torch.equal(advance.target, desired))
        self.assertEqual(advance.limited_group_count, 0)


if __name__ == "__main__":
    unittest.main()
