from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch

from qqtt.ambulance_scene import (
    _sample_projected_mesh_upper_surface,
    make_ambulance_layout,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(torch.cuda.is_available(), "CUDA device required")
class AmbulanceSourceMeshCollisionCudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import warp as wp

        from qqtt.model.diff_simulator.spring_mass_warp import (
            SpringMassSystemWarp,
            integrate_ground_collision,
        )

        cls.wp = wp
        cls.kernel = integrate_ground_collision
        cls.layout = make_ambulance_layout(
            np.zeros(3, dtype=np.float32),
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
            scene_up=np.array([0.0, 0.0, -1.0], dtype=np.float32),
            repo_root=REPO_ROOT,
        )
        contact = cls.layout.static_collision_mesh_contact
        simulator = SpringMassSystemWarp.__new__(SpringMassSystemWarp)
        simulator.device = "cuda:0"
        simulator.static_mesh_enabled = 0
        simulator.static_mesh_id = 0
        simulator.static_mesh_two_sided = 0
        simulator.static_mesh_component_count = 0
        simulator.static_mesh_query_distance = 0.1
        simulator.static_mesh_winding_accuracy = 2.0
        simulator.static_mesh_winding_threshold = 0.5
        simulator.static_mesh_margin = 0.0
        simulator.static_mesh_restitution = 0.0
        simulator.static_mesh_friction = 0.5
        simulator.set_static_collision_surfaces(
            cls.layout.static_collision_surfaces,
            query_distance=contact["query_distance_m"],
            margin=contact["margin_m"],
            restitution=contact["restitution"],
            friction=contact["friction"],
        )
        detail_contact = cls.layout.static_collision_detail_mesh_contact
        simulator.set_static_collision_mesh(
            cls.layout.static_collision_detail_mesh_vertices,
            cls.layout.static_collision_detail_mesh_faces,
            two_sided=detail_contact["two_sided"],
            component_bounds=(
                cls.layout.static_collision_detail_mesh_component_bounds
            ),
            substep_interval=detail_contact["substep_interval"],
            query_distance=detail_contact["query_distance_m"],
            winding_accuracy=detail_contact["winding_accuracy"],
            winding_threshold=detail_contact["winding_threshold"],
            margin=detail_contact["margin_m"],
            restitution=detail_contact["restitution"],
            friction=detail_contact["friction"],
        )
        cls.simulator = simulator

    def _launch(self, positions, velocities, *, mesh_sweep_start=None):
        wp = self.wp
        simulator = self.simulator
        x = wp.array(
            np.ascontiguousarray(positions, dtype=np.float32),
            dtype=wp.vec3,
            device="cuda:0",
        )
        v = wp.array(
            np.ascontiguousarray(velocities, dtype=np.float32),
            dtype=wp.vec3,
            device="cuda:0",
        )
        mesh_sweep = (
            x
            if mesh_sweep_start is None
            else wp.array(
                np.ascontiguousarray(mesh_sweep_start, dtype=np.float32),
                dtype=wp.vec3,
                device="cuda:0",
            )
        )
        material = wp.zeros(1, dtype=float, device="cuda:0")
        dummy_boxes = wp.zeros(1, dtype=wp.vec3, device="cuda:0")
        x_new = wp.zeros_like(x)
        v_new = wp.zeros_like(v)
        wp.launch(
            self.kernel,
            dim=len(positions),
            inputs=[
                x,
                v,
                material,
                material,
                0.01,
                1.0,
                0,
                dummy_boxes,
                dummy_boxes,
                0,
                simulator.wp_static_surface_centers,
                simulator.wp_static_surface_normals,
                simulator.wp_static_surface_axes_u,
                simulator.wp_static_surface_axes_v,
                simulator.wp_static_surface_extents_u,
                simulator.wp_static_surface_extents_v,
                simulator.wp_static_surface_kinds,
                simulator.wp_static_surface_edge_radii,
                simulator.wp_static_surface_heightfield_offsets,
                simulator.wp_static_surface_heightfield_starts,
                simulator.wp_static_surface_heightfield_cells_u,
                simulator.wp_static_surface_heightfield_cells_v,
                simulator.static_surface_count,
                simulator.static_surface_query_distance,
                simulator.static_surface_margin,
                simulator.static_surface_restitution,
                simulator.static_surface_friction,
                simulator.static_mesh_enabled,
                mesh_sweep,
                simulator.static_mesh_id,
                simulator.static_mesh_two_sided,
                simulator.wp_static_mesh_component_mins,
                simulator.wp_static_mesh_component_maxs,
                simulator.static_mesh_component_count,
                simulator.static_mesh_query_distance,
                simulator.static_mesh_winding_accuracy,
                simulator.static_mesh_winding_threshold,
                simulator.static_mesh_margin,
                simulator.static_mesh_restitution,
                simulator.static_mesh_friction,
            ],
            outputs=[x_new, v_new],
            device="cuda:0",
        )
        wp.synchronize_device("cuda:0")
        return (
            wp.to_torch(x_new).detach().cpu().numpy(),
            wp.to_torch(v_new).detach().cpu().numpy(),
        )

    def test_full_resolution_mattress_blocks_falls_at_center_and_far_end(self):
        layout = self.layout
        self.assertEqual(
            self.simulator.torch_static_surface_kinds.detach().cpu().tolist(),
            [0],
        )
        self.assertEqual(self.simulator.static_mesh_substep_interval, 16)
        frame_center = np.asarray(
            layout.ambulance_mattress_collision_frame_center_world,
            dtype=np.float32,
        )
        normal = np.asarray(
            layout.ambulance_mattress_normal_world,
            dtype=np.float32,
        )
        axis_u = np.asarray(
            layout.ambulance_mattress_axis_u_world,
            dtype=np.float32,
        )
        axis_v = np.asarray(
            layout.ambulance_mattress_axis_v_world,
            dtype=np.float32,
        )
        local_vertices = np.asarray(
            layout.ambulance_mattress_collision_mesh_local_vertices
        )
        local_faces = np.asarray(layout.ambulance_mattress_collision_mesh_faces)
        local_points = (
            (0.0, 0.0),
            (-0.93, 0.0),
            (0.9, 0.0),
            (1.0, 0.0),
        )
        heights = []
        plane_points = []
        for local_u, local_v in local_points:
            height, edge_distance, inside = _sample_projected_mesh_upper_surface(
                local_vertices,
                local_faces,
                local_u=local_u,
                local_v=local_v,
            )
            self.assertTrue(inside)
            self.assertEqual(edge_distance, 0.0)
            heights.append(height)
            plane_points.append(
                frame_center + axis_u * local_u + axis_v * local_v
            )
        heights = np.asarray(heights, dtype=np.float32)
        plane_points = np.asarray(plane_points, dtype=np.float32)
        positions = plane_points + normal[None, :] * (heights[:, None] + 0.05)
        velocities = np.broadcast_to(-normal * 10.0, positions.shape).copy()

        positions_out, velocities_out = self._launch(positions, velocities)
        output_heights = np.sum(
            (positions_out - plane_points) * normal[None, :],
            axis=1,
        )
        expected_heights = heights + float(
            layout.static_collision_detail_mesh_contact["margin_m"]
        )

        # The 1 cm shell margin is applied along each captured triangle's own
        # normal, so its component along the fitted mattress normal varies
        # slightly on wrinkles and slopes.
        self.assertTrue(np.allclose(output_heights, expected_heights, atol=4.0e-3))
        self.assertTrue(np.all(velocities_out @ normal > -2.0))
        self.assertGreater(float(np.ptp(output_heights)), 0.05)

        # The old 1.8 m analytic mattress stopped at +u=0.9 m. The captured
        # shell extends beyond +u=1.08 m, so +u=1.0 m must now be supported.
        self.assertGreater(output_heights[3], heights[3])

        outside_plane_point = frame_center + axis_u * 1.20
        outside_position = outside_plane_point + normal * 0.10
        outside_velocity = -normal * 10.0
        outside_position_out, outside_velocity_out = self._launch(
            outside_position[None, :],
            outside_velocity[None, :],
        )
        outside_height_out = float(
            np.dot(outside_position_out[0] - outside_plane_point, normal)
        )
        self.assertAlmostEqual(outside_height_out, 0.0, places=4)
        self.assertLess(float(np.dot(outside_velocity_out[0], normal)), 0.0)

    def test_full_resolution_mattress_preserves_the_measured_width_profile(self):
        layout = self.layout
        frame_center = np.asarray(
            layout.ambulance_mattress_collision_frame_center_world,
            dtype=np.float32,
        )
        normal = np.asarray(
            layout.ambulance_mattress_normal_world,
            dtype=np.float32,
        )
        axis_u = np.asarray(
            layout.ambulance_mattress_axis_u_world,
            dtype=np.float32,
        )
        axis_v = np.asarray(
            layout.ambulance_mattress_axis_v_world,
            dtype=np.float32,
        )
        local_vertices = np.asarray(
            layout.ambulance_mattress_collision_mesh_local_vertices
        )
        local_faces = np.asarray(layout.ambulance_mattress_collision_mesh_faces)
        local_u = 0.5
        local_vs = (0.0, 0.2)
        heights = np.asarray(
            [
                _sample_projected_mesh_upper_surface(
                    local_vertices,
                    local_faces,
                    local_u=local_u,
                    local_v=local_v,
                )[0]
                for local_v in local_vs
            ],
            dtype=np.float32,
        )
        self.assertGreater(abs(float(heights[1] - heights[0])), 0.05)
        plane_points = np.stack(
            [
                frame_center + axis_u * local_u + axis_v * local_v
                for local_v in local_vs
            ],
            axis=0,
        )
        positions = plane_points + normal[None, :] * (heights[:, None] + 0.05)
        positions_out, velocities_out = self._launch(
            positions,
            np.broadcast_to(-normal * 10.0, positions.shape).copy(),
        )
        output_heights = np.sum(
            (positions_out - plane_points) * normal[None, :],
            axis=1,
        )
        expected = heights + float(
            layout.static_collision_detail_mesh_contact["margin_m"]
        )
        self.assertTrue(np.allclose(output_heights, expected, atol=2.0e-3))
        self.assertTrue(np.all(velocities_out @ normal > -0.1))

    def test_mesh_sweep_spans_skipped_spring_substeps(self):
        layout = self.layout
        frame_center = np.asarray(
            layout.ambulance_mattress_collision_frame_center_world,
            dtype=np.float32,
        )
        normal = np.asarray(
            layout.ambulance_mattress_normal_world,
            dtype=np.float32,
        )
        height = _sample_projected_mesh_upper_surface(
            layout.ambulance_mattress_collision_mesh_local_vertices,
            layout.ambulance_mattress_collision_mesh_faces,
            local_u=0.0,
            local_v=0.0,
        )[0]
        surface_point = frame_center + normal * height
        current_position = surface_point - normal * 0.04
        prior_query_position = surface_point + normal * 0.04

        positions_out, _ = self._launch(
            current_position[None, :],
            np.zeros((1, 3), dtype=np.float32),
            mesh_sweep_start=prior_query_position[None, :],
        )
        output_height = float(np.dot(positions_out[0] - surface_point, normal))
        self.assertAlmostEqual(
            output_height,
            layout.static_collision_detail_mesh_contact["margin_m"],
            delta=2.0e-3,
        )

    def test_source_mesh_side_hardware_blocks_fast_crossings_from_both_sides(self):
        layout = self.layout
        center = np.asarray(
            layout.ambulance_mattress_collision_frame_center_world,
            dtype=np.float32,
        )
        normal = np.asarray(
            layout.ambulance_mattress_normal_world,
            dtype=np.float32,
        )
        axis_u = np.asarray(
            layout.ambulance_mattress_axis_u_world,
            dtype=np.float32,
        )
        axis_v = np.asarray(
            layout.ambulance_mattress_axis_v_world,
            dtype=np.float32,
        )
        # Cross the raised +v handle at its measured center. A collision may
        # rebound or slide over the curved surface, but it must not arrive at
        # the unconstrained endpoint on the opposite side in one period.
        contact_center = (
            center + axis_u * 0.25 + axis_v * 0.31 + normal * 0.07
        )

        for side in (-1.0, 1.0):
            start = contact_center + axis_v * side * 0.08
            velocity = -axis_v * side * 16.0
            position_out, velocity_out = self._launch(
                start[None, :],
                velocity[None, :],
            )
            output_side_distance = float(
                np.dot(position_out[0] - contact_center, axis_v) * side
            )
            output_side_velocity = float(
                np.dot(velocity_out[0], axis_v) * side
            )

            self.assertGreaterEqual(output_side_distance, 0.015)
            self.assertGreater(output_side_velocity, -2.5)


if __name__ == "__main__":
    unittest.main()
