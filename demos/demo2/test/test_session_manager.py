import time
import unittest

import torch

from demos.demo2.control import (
    add_vectors_clamped,
    control_vector_to_step,
    joystick_to_interactive_2d_step,
    resolve_demo2_control_parts,
)
from demos.demo2.ground_control import (
    build_ground_offset_limits,
    clamp_control_offsets_to_ground,
)
from demos.demo2.session_manager import SessionManager


class SessionManagerTest(unittest.TestCase):
    def test_claim_reject_release(self):
        manager = SessionManager(num_sessions=2, heartbeat_timeout_s=10.0)
        claim = manager.claim(0)
        self.assertTrue(claim["token"])
        self.assertEqual(claim["claim_id"], 1)
        self.assertIsNone(manager.claim(0))
        self.assertFalse(manager.release(0, "wrong"))
        self.assertTrue(manager.release(0, claim["token"]))
        next_claim = manager.claim(0)
        self.assertEqual(next_claim["claim_id"], 2)

    def test_input_snapshot_and_clamp(self):
        manager = SessionManager(num_sessions=1, input_limit=1.0)
        token = manager.claim(0)["token"]
        self.assertTrue(manager.update_input(0, token, 2.5, -3.0, 4.0))
        self.assertEqual(manager.snapshot_inputs(), {0: (1.0, -1.0, 1.0)})
        self.assertEqual(
            manager.snapshot_sessions()[0]["right"],
            (0.0, 0.0, 0.0),
        )

    def test_nested_hand_snapshot_and_clamp(self):
        manager = SessionManager(num_sessions=1, input_limit=1.0)
        claim = manager.claim(0)
        self.assertTrue(
            manager.update_input(
                0,
                claim["token"],
                left={"x": 2, "y": -2, "z": 0.5},
                right={"x": -0.25, "y": 0.75, "z": 4},
            )
        )
        self.assertEqual(
            manager.snapshot_sessions(),
            {
                0: {
                    "claim_id": claim["claim_id"],
                    "left": (1.0, -1.0, 0.5),
                    "right": (-0.25, 0.75, 1.0),
                }
            },
        )

    def test_legacy_joystick_input_maps_to_interactive_axes(self):
        manager = SessionManager(num_sessions=1, input_limit=1.0)
        token = manager.claim(0)["token"]
        self.assertTrue(manager.update_input(0, token, dx=0.5, dy=-0.25))
        self.assertEqual(manager.snapshot_inputs(), {0: (-0.25, 0.5, 0.0)})

    def test_heartbeat_timeout(self):
        manager = SessionManager(num_sessions=1, heartbeat_timeout_s=0.01)
        token = manager.claim(0)["token"]
        time.sleep(0.02)
        self.assertFalse(manager.validate(0, token))
        self.assertTrue(manager.list_sessions()[0]["available"])
        self.assertEqual(manager.snapshot_sessions(), {})


class Demo2ControlTest(unittest.TestCase):
    def test_default_rope_phone_axes_are_mirrored_to_world_space(self):
        step = 0.005
        self.assertEqual(
            control_vector_to_step(1.0, 0.0, 0.0, step),
            (-step, 0.0, 0.0),
        )
        self.assertEqual(
            control_vector_to_step(-1.0, 0.0, 0.0, step),
            (step, 0.0, 0.0),
        )
        self.assertEqual(
            control_vector_to_step(0.0, -1.0, 0.0, step),
            (0.0, step, 0.0),
        )
        self.assertEqual(
            control_vector_to_step(0.0, 1.0, 0.0, step),
            (0.0, -step, 0.0),
        )

    def test_phone_vertical_axis_keeps_world_sign(self):
        step = 0.005
        self.assertEqual(
            control_vector_to_step(0.0, 0.0, -1.0, step),
            (0.0, 0.0, -step),
        )
        self.assertEqual(
            control_vector_to_step(0.0, 0.0, 1.0, step),
            (0.0, 0.0, step),
        )

    def test_button_vector_mapping_combines_and_clamps(self):
        self.assertEqual(
            control_vector_to_step(1.0, -1.0, 1.0, 0.1),
            (-0.1, 0.1, 0.1),
        )
        self.assertEqual(
            control_vector_to_step(3.0, -4.0, 5.0, 0.1),
            (-0.1, 0.1, 0.1),
        )
        self.assertEqual(
            add_vectors_clamped((1, 0.5, 0), (1, -1, 2)),
            (1.0, -0.5, 1.0),
        )

    def test_legacy_joystick_mapping_still_matches_keyboard_axes(self):
        self.assertEqual(
            joystick_to_interactive_2d_step(0.0, 1.0, 0.005),
            (0.005, 0.0, 0.0),
        )

    def test_control_part_auto_detection(self):
        self.assertEqual(resolve_demo2_control_parts("single_push_rope_4"), 1)
        self.assertEqual(resolve_demo2_control_parts("rope_double_hand"), 2)
        self.assertEqual(resolve_demo2_control_parts("double_stretch_sloth"), 2)
        self.assertEqual(
            resolve_demo2_control_parts(
                "weird_package",
                double_control_cases=["weird_package"],
            ),
            2,
        )
        self.assertEqual(resolve_demo2_control_parts("anything", requested="1"), 1)
        self.assertEqual(resolve_demo2_control_parts("anything", requested="2"), 2)

    def test_reverse_z_phone_down_is_blocked_at_rest_pose(self):
        base_targets = torch.tensor(
            [[
                [0.0, 0.0, -0.100],
                [0.0, 0.0, -0.003],
                [0.0, 0.0, -0.080],
                [0.0, 0.0, -0.010],
            ]],
            dtype=torch.float32,
        )
        masks = [torch.tensor([0, 1]), torch.tensor([2, 3])]
        limits = build_ground_offset_limits(base_targets, masks, -1.0)
        offsets = torch.tensor(
            [[[0.25, -0.50, 0.050], [-0.25, 0.50, 0.050]]],
            dtype=torch.float32,
        )

        clamp_control_offsets_to_ground(offsets, limits, -1.0)

        torch.testing.assert_close(offsets[0, :, 2], torch.zeros(2))
        self.assertEqual(float(offsets[0, 0, 0]), 0.25)
        self.assertEqual(float(offsets[0, 0, 1]), -0.50)
        for part_idx, indices in enumerate(masks):
            target_z = base_targets[0, indices, 2] + offsets[0, part_idx, 2]
            self.assertLessEqual(float(target_z.max()), 1e-7)

        offsets[..., 2] = -0.2
        clamp_control_offsets_to_ground(offsets, limits, -1.0)
        torch.testing.assert_close(offsets[..., 2], torch.full((1, 2), -0.2))

    def test_standard_z_phone_down_is_blocked_at_rest_pose(self):
        base_targets = torch.tensor(
            [[[0.0, 0.0, 0.003], [0.0, 0.0, 0.100]]],
            dtype=torch.float32,
        )
        masks = [torch.tensor([0, 1])]
        limits = build_ground_offset_limits(base_targets, masks, 1.0)
        offsets = torch.tensor([[[0.0, 0.0, -0.050]]], dtype=torch.float32)

        clamp_control_offsets_to_ground(offsets, limits, 1.0)

        torch.testing.assert_close(offsets[..., 2], torch.zeros((1, 1)))
        target_z = base_targets[0, :, 2] + offsets[0, 0, 2]
        self.assertGreaterEqual(float(target_z.min()), -1e-7)


if __name__ == "__main__":
    unittest.main()
