from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch

from qqtt.garden_scene import (
    build_garden_collision_proxy_canonical,
    load_garden_collision_proxy,
    make_garden_layout,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_garden_collision_proxy_is_closed_and_compact():
    _, proxy = load_garden_collision_proxy(REPO_ROOT)
    vertices, faces, metadata = build_garden_collision_proxy_canonical(proxy)
    edge_counts: dict[tuple[int, int], int] = {}
    for face in faces:
        for first, second in (
            (face[0], face[1]),
            (face[1], face[2]),
            (face[2], face[0]),
        ):
            edge = tuple(sorted((int(first), int(second))))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1

    assert vertices.shape == (186, 3)
    assert faces.shape == (340, 3)
    assert len(metadata) == 8
    assert all(count == 2 for count in edge_counts.values())


@unittest.skipUnless(torch.cuda.is_available(), "CUDA device required")
class GardenPrimitiveCollisionCudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import warp as wp
        from qqtt.model.diff_simulator.spring_mass_warp import (
            integrate_ground_collision,
        )

        cls.wp = wp
        cls.kernel = integrate_ground_collision
        _, proxy = load_garden_collision_proxy(REPO_ROOT)
        cls.proxy = proxy
        layout = make_garden_layout(
            np.array([0.0, -0.78, -0.62], dtype=np.float32),
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
            repo_root=REPO_ROOT,
            scene_up=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        )
        cls.surfaces = layout.static_collision_surfaces
        cls.box_mins = wp.array(
            np.ascontiguousarray(layout.static_collision_boxes[:, 0]),
            dtype=wp.vec3,
            device="cuda:0",
        )
        cls.box_maxs = wp.array(
            np.ascontiguousarray(layout.static_collision_boxes[:, 1]),
            dtype=wp.vec3,
            device="cuda:0",
        )
        cls.surface_centers = wp.array(
            np.stack([entry["center"] for entry in cls.surfaces]),
            dtype=wp.vec3,
            device="cuda:0",
        )
        cls.surface_normals = wp.array(
            np.stack([entry["normal"] for entry in cls.surfaces]),
            dtype=wp.vec3,
            device="cuda:0",
        )
        cls.surface_axes_u = wp.array(
            np.stack([entry["axis_u"] for entry in cls.surfaces]),
            dtype=wp.vec3,
            device="cuda:0",
        )
        cls.surface_axes_v = wp.array(
            np.stack([entry["axis_v"] for entry in cls.surfaces]),
            dtype=wp.vec3,
            device="cuda:0",
        )
        cls.surface_extents_u = wp.array(
            np.asarray([entry["extent_u"] for entry in cls.surfaces], dtype=np.float32),
            dtype=float,
            device="cuda:0",
        )
        cls.surface_extents_v = wp.array(
            np.asarray([entry["extent_v"] for entry in cls.surfaces], dtype=np.float32),
            dtype=float,
            device="cuda:0",
        )
        cls.surface_kinds = wp.array(
            np.asarray(
                [
                    {"rectangle": 0, "disk": 1, "cylinder": 2}[entry["kind"]]
                    for entry in cls.surfaces
                ],
                dtype=np.int32,
            ),
            dtype=wp.int32,
            device="cuda:0",
        )

    def _launch(
        self,
        position,
        velocity,
        *,
        capture=False,
        surfaces_enabled=True,
        boxes_enabled=True,
    ):
        wp = self.wp
        contact = self.proxy["contact"]
        x = wp.array(np.asarray([position], dtype=np.float32), dtype=wp.vec3, device="cuda:0")
        v = wp.array(np.asarray([velocity], dtype=np.float32), dtype=wp.vec3, device="cuda:0")
        material = wp.zeros(1, dtype=float, device="cuda:0")
        dummy_boxes = wp.zeros(1, dtype=wp.vec3, device="cuda:0")
        x_new = wp.zeros_like(x)
        v_new = wp.zeros_like(v)

        def launch():
            wp.launch(
                self.kernel,
                dim=1,
                inputs=[
                    x,
                    v,
                    material,
                    material,
                    0.01,
                    1.0,
                    0,
                    self.box_mins if boxes_enabled else dummy_boxes,
                    self.box_maxs if boxes_enabled else dummy_boxes,
                    6 if boxes_enabled else 0,
                    self.surface_centers,
                    self.surface_normals,
                    self.surface_axes_u,
                    self.surface_axes_v,
                    self.surface_extents_u,
                    self.surface_extents_v,
                    self.surface_kinds,
                    len(self.surfaces) if surfaces_enabled else 0,
                    float(contact["query_distance_m"]),
                    float(contact["margin_m"]),
                    float(contact["restitution"]),
                    float(contact["friction"]),
                    0,
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

        if capture:
            with wp.ScopedCapture() as capture_state:
                launch()
            wp.capture_launch(capture_state.graph)
        else:
            launch()
        wp.synchronize_device("cuda:0")
        return (
            wp.to_torch(x_new)[0].detach().cpu().numpy(),
            wp.to_torch(v_new)[0].detach().cpu().numpy(),
        )

    def test_swept_table_drop_and_cuda_graph_capture(self):
        position, velocity = self._launch(
            [0.0, 0.0, -0.2],
            [0.0, 0.0, 100.0],
            capture=True,
        )
        self.assertAlmostEqual(position[2], -0.0025, places=4)
        self.assertLess(velocity[2], 0.0)

    def test_disabled_surfaces_preserve_lab_free_integration_path(self):
        position, velocity = self._launch(
            [0.0, 0.0, -0.2],
            [0.0, 0.0, 1.0],
            surfaces_enabled=False,
            boxes_enabled=False,
        )
        self.assertTrue(np.allclose(position, [0.0, 0.0, -0.19]))
        self.assertTrue(np.allclose(velocity, [0.0, 0.0, 1.0]))

    def test_runtime_setter_keeps_signed_mesh_disabled(self):
        from qqtt.model.diff_simulator.spring_mass_warp import SpringMassSystemWarp

        simulator = SpringMassSystemWarp.__new__(SpringMassSystemWarp)
        simulator.device = "cuda:0"
        simulator.static_mesh_enabled = 0
        simulator.set_static_collision_surfaces(
            self.surfaces,
            query_distance=0.16,
            margin=0.0025,
            restitution=0.04,
            friction=0.62,
        )
        self.assertEqual(simulator.static_surface_count, 2)
        self.assertEqual(simulator.static_mesh_enabled, 0)
        self.assertEqual(
            self.wp.to_torch(simulator.wp_static_surface_kinds).cpu().tolist(),
            [2, 0],
        )

    def test_resting_recovery_and_friction(self):
        position, velocity = self._launch(
            [0.0, 0.0, -0.001],
            [2.0, 0.0, 0.2],
        )
        self.assertLessEqual(position[2], -0.0024)
        self.assertLess(abs(velocity[0]), 2.0)
        self.assertLess(velocity[2], 0.0)

    def test_understructure_uses_cheap_box_contact(self):
        position, velocity = self._launch(
            [-0.6, -0.24, 0.3],
            [20.0, 0.0, 0.0],
        )
        self.assertLess(position[0], -0.5175)
        self.assertAlmostEqual(velocity[0], 0.0, places=5)

    def test_swept_tabletop_rim_blocks_fast_lateral_motion(self):
        position, velocity = self._launch(
            [0.90, 0.0, 0.03],
            [-30.0, 0.0, 0.0],
        )
        self.assertAlmostEqual(position[0], 0.7825, places=4)
        self.assertGreater(velocity[0], 0.0)

    def test_rim_penetration_uses_nearest_side_not_top_face(self):
        position, velocity = self._launch(
            [0.777, 0.0, 0.02],
            [-0.1, 0.0, 0.1],
        )
        self.assertAlmostEqual(position[0], 0.7825, places=4)
        self.assertLess(position[2], 0.022)
        self.assertGreaterEqual(velocity[0], 0.0)

    def test_swept_tabletop_underside_blocks_hanging_rope_swing(self):
        position, velocity = self._launch(
            [0.0, 0.0, 0.12],
            [0.0, 0.0, -20.0],
        )
        self.assertAlmostEqual(position[2], 0.0675, places=4)
        self.assertGreater(velocity[2], 0.0)

    def test_table_edge_fall_hits_patio_but_outside_proxy_keeps_falling(self):
        patio_position, _ = self._launch(
            [1.5, 0.25, 0.4],
            [0.0, 0.0, 100.0],
        )
        outside_position, outside_velocity = self._launch(
            [3.0, 0.25, 0.4],
            [0.0, 0.0, 100.0],
        )
        self.assertAlmostEqual(patio_position[2], 0.7175, places=4)
        self.assertGreater(outside_position[2], 1.0)
        self.assertGreater(outside_velocity[2], 0.0)


if __name__ == "__main__":
    unittest.main()
