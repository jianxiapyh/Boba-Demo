from __future__ import annotations

import unittest

import numpy as np
import torch

from qqtt.immersive_scene import merge_lab_table_divider_collision_boxes


SOURCE_DIVIDER_BOXES = np.array(
    [
        [
            [-0.0191, 0.6311, 0.6456],
            [0.0017, 1.2031, 0.8137],
        ],
        [
            [-0.0009, 0.6311, 0.6456],
            [0.0198, 1.2031, 0.8137],
        ],
    ],
    dtype=np.float32,
)
SMOOTH_TABLE_TOP = np.array(
    [
        [-0.7103, 0.5763, 0.6090],
        [0.7103, 1.2563, 0.6440],
    ],
    dtype=np.float32,
)
LOWER_SHELF = np.array(
    [
        [
            [-0.7211, 0.5754, 0.8134],
            [0.7218, 1.2558, 0.8342],
        ]
    ],
    dtype=np.float32,
)


class LabTableDividerGeometryTests(unittest.TestCase):
    def test_merge_is_single_padded_watertight_box(self):
        merged, debug = merge_lab_table_divider_collision_boxes(
            SOURCE_DIVIDER_BOXES,
            scene_up=np.array([0.0, 0.0, -1.0], dtype=np.float32),
            smooth_table_top=SMOOTH_TABLE_TOP,
            lower_support_boxes=LOWER_SHELF,
            lateral_inflate_m=0.012,
            surface_overlap_m=0.006,
        )

        self.assertEqual(merged.shape, (2, 3))
        self.assertEqual(debug["input_box_count"], 2)
        self.assertEqual(debug["vertical_axis"], 2)
        self.assertEqual(debug["thin_lateral_axis"], 0)
        self.assertTrue(
            np.isclose(debug["top_gap_before_m"], 0.0016, atol=1.0e-6)
        )
        self.assertTrue(
            np.isclose(debug["lower_gap_before_m"], -0.0003, atol=1.0e-6)
        )

        # The source halves become one collider, with 12 mm of visual clearance
        # on both sides of its thin axis.
        self.assertTrue(
            np.allclose(merged[:, 0], [-0.0311, 0.0318], atol=1.0e-6)
        )
        self.assertTrue(
            np.allclose(merged[:, 1], [0.6311, 1.2031], atol=1.0e-6)
        )

        # Its upper and lower edges overlap the adjacent slabs by 6 mm, closing
        # the original 1.6 mm tabletop seam and the lower-shelf junction.
        self.assertTrue(np.isclose(merged[0, 2], 0.6380, atol=1.0e-6))
        self.assertTrue(np.isclose(merged[1, 2], 0.8194, atol=1.0e-6))
        self.assertLess(float(merged[0, 2]), float(SMOOTH_TABLE_TOP[1, 2]))
        self.assertGreater(float(merged[1, 2]), float(LOWER_SHELF[0, 0, 2]))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA device required")
class LabTableDividerCollisionCudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import warp as wp
        from qqtt.model.diff_simulator.spring_mass_warp import (
            integrate_ground_collision,
        )

        cls.wp = wp
        cls.kernel = integrate_ground_collision
        merged, _ = merge_lab_table_divider_collision_boxes(
            SOURCE_DIVIDER_BOXES,
            scene_up=np.array([0.0, 0.0, -1.0], dtype=np.float32),
            smooth_table_top=SMOOTH_TABLE_TOP,
            lower_support_boxes=LOWER_SHELF,
            lateral_inflate_m=0.012,
            surface_overlap_m=0.006,
        )
        cls.box_mins = wp.array(
            np.ascontiguousarray(merged[None, 0]),
            dtype=wp.vec3,
            device="cuda:0",
        )
        cls.box_maxs = wp.array(
            np.ascontiguousarray(merged[None, 1]),
            dtype=wp.vec3,
            device="cuda:0",
        )

    def _launch(self, position, velocity):
        wp = self.wp
        x = wp.array(
            np.asarray([position], dtype=np.float32),
            dtype=wp.vec3,
            device="cuda:0",
        )
        v = wp.array(
            np.asarray([velocity], dtype=np.float32),
            dtype=wp.vec3,
            device="cuda:0",
        )
        scalar = wp.zeros(1, dtype=float, device="cuda:0")
        integer = wp.zeros(1, dtype=wp.int32, device="cuda:0")
        dummy_vec3 = wp.zeros(1, dtype=wp.vec3, device="cuda:0")
        x_new = wp.zeros_like(x)
        v_new = wp.zeros_like(v)

        wp.launch(
            self.kernel,
            dim=1,
            inputs=[
                x,
                v,
                scalar,
                scalar,
                0.01,
                1.0,
                0,
                self.box_mins,
                self.box_maxs,
                1,
                dummy_vec3,
                dummy_vec3,
                dummy_vec3,
                dummy_vec3,
                scalar,
                scalar,
                integer,
                scalar,
                scalar,
                integer,
                integer,
                integer,
                0,
                0.0,
                0.0,
                0.0,
                0.5,
                0,
                x,
                0,
                0,
                dummy_vec3,
                dummy_vec3,
                0,
                0.1,
                2.0,
                0.5,
                0.0,
                0.0,
                0.5,
            ],
            outputs=[x_new, v_new],
            device="cuda:0",
        )
        wp.synchronize_device("cuda:0")
        return x_new.numpy()[0], v_new.numpy()[0]

    def test_top_seam_and_visual_clearance_are_both_solid(self):
        # z=0.6448 was inside the original 1.6 mm seam. The merged collider now
        # blocks a fast crossing there at the padded face x=-0.0311.
        position, velocity = self._launch(
            [-0.05, 0.9, 0.6448],
            [10.0, 0.0, 0.0],
        )

        self.assertAlmostEqual(float(position[0]), -0.0312, places=4)
        self.assertAlmostEqual(float(velocity[0]), 0.0, places=5)
