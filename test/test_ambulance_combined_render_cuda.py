from __future__ import annotations

import json
import math
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(torch.cuda.is_available(), "CUDA device required")
class AmbulanceCombinedRenderCudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from gaussian_splatting.scene.gaussian_model import GaussianModel
        from qqtt.ambulance_scene import AmbulanceSceneRenderer, make_ambulance_layout

        cls.GaussianModel = GaussianModel
        cls.renderer = AmbulanceSceneRenderer(
            REPO_ROOT / "assets/scenes",
            192,
            128,
            repo_root=REPO_ROOT,
            eye_resolution=1344,
        )
        calibration = json.loads(
            (
                REPO_ROOT
                / "assets/scenes/ambulance_insta360/calibration.json"
            ).read_text(encoding="utf-8")
        )
        source_rotation = np.asarray(
            calibration["source_to_canonical_rotation"],
            dtype=np.float32,
        )
        source_center = np.asarray(
            calibration["source_interaction_surface_center"],
            dtype=np.float32,
        )
        source_head = np.asarray(
            calibration["seated_view"]["source_head_position"],
            dtype=np.float32,
        )
        cls.head_position = (
            (source_head - source_center)
            @ source_rotation.T
            * float(calibration["meters_per_source_unit"])
        ).astype(np.float32)
        # Using the authored bench-head point as the live head makes the
        # legacy SOG reference coincide with world zero. The established
        # conservative spawn plane remains fixed while hidden startup settling
        # lowers the object onto the matching captured mattress mesh.
        cls.layout = make_ambulance_layout(
            cls.head_position,
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
            scene_up=np.array([0.0, 0.0, -1.0], dtype=np.float32),
            repo_root=REPO_ROOT,
        )
        cls.renderer.set_layout(cls.layout)
        cls.dynamic = cls._make_dynamic()
        cls.combined = cls.renderer.bind_dynamic_gaussians(cls.dynamic)
        cls.camera = cls._make_camera()

    @classmethod
    def tearDownClass(cls):
        renderer = getattr(cls, "renderer", None)
        if renderer is not None:
            renderer.delete()

    @classmethod
    def _make_dynamic(cls):
        model = cls.GaussianModel(sh_degree=3)
        model.active_sh_degree = 3
        model.isotropic = False
        object_center = (
            np.asarray(cls.layout.table_top_center)
            + np.asarray(cls.layout.ambulance_mattress_normal_world) * 0.04
        )
        model._xyz = torch.as_tensor(
            object_center[None, :],
            dtype=torch.float32,
            device="cuda",
        )
        model._features_dc = torch.zeros((1, 1, 3), device="cuda")
        model._features_dc[0, 0, 0] = (0.9 - 0.5) / 0.28209479177387814
        model._features_rest = torch.zeros((1, 15, 3), device="cuda")
        model._opacity = torch.full(
            (1, 1),
            math.log(0.95 / 0.05),
            device="cuda",
        )
        model._scaling = torch.full((1, 3), math.log(0.025), device="cuda")
        model._rotation = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]],
            dtype=torch.float32,
            device="cuda",
        )
        return model

    @classmethod
    def _make_camera(cls):
        width = 192
        height = 128
        focal = 118.0
        intrinsic = torch.tensor(
            [
                [focal, 0.0, 0.5 * width],
                [0.0, focal, 0.5 * height],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
            device="cuda",
        )
        camera_position = cls.head_position
        # CV camera axes in Boba world coordinates: right=+x, down=+z,
        # forward=+y. This looks from the aisle toward the stretcher.
        world_to_camera = np.eye(4, dtype=np.float32)
        world_to_camera[:3, :3] = np.array(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        )
        world_to_camera[:3, 3] = -(
            world_to_camera[:3, :3] @ camera_position
        )
        return SimpleNamespace(
            image_width=width,
            image_height=height,
            world_view_transform=torch.as_tensor(
                world_to_camera.T,
                dtype=torch.float32,
                device="cuda",
            ),
            K=intrinsic,
            camera_center=torch.as_tensor(camera_position, device="cuda"),
            FoVx=0.0,
            FoVy=0.0,
            image_name="ambulance_smoke",
            uid="ambulance_smoke",
            colmap_id="ambulance_smoke",
        )

    def test_sog_scene_and_dynamic_object_render_as_one_direct_frame(self):
        from gaussian_splatting.gaussian_renderer import render

        from qqtt.ambulance_scene import AMBULANCE_MAX_PROJECTED_RADIUS_PX

        pipeline = SimpleNamespace(
            absgrad=False,
            radius_clip=0.25,
            max_projected_radius=AMBULANCE_MAX_PROJECTED_RADIUS_PX,
        )
        background = torch.zeros(3, dtype=torch.float32, device="cuda")
        with torch.inference_mode():
            result = render(
                self.camera,
                self.combined,
                pipeline,
                background,
                antialiased=True,
            )
            frame, depth, metrics = self.renderer.prepare_direct_gaussian_eye_output(
                result["render"],
                result["depth"],
                output_dtype=torch.uint8,
            )

        self.assertEqual(int(self.combined._xyz.shape[0]), 999_411)
        self.assertTrue(
            np.allclose(
                self.layout.canonical_to_world_translation,
                np.zeros(3),
                atol=1.0e-6,
            )
        )
        self.assertTrue(
            np.allclose(
                self.layout.table_top_center,
                [0.10136589, -0.09999431, -0.10535091],
                atol=2.0e-6,
            )
        )
        self.assertTrue(torch.isfinite(result["render"]).all().item())
        self.assertLessEqual(
            result["radii"].max().item(),
            min(128, int(AMBULANCE_MAX_PROJECTED_RADIUS_PX)),
        )
        self.assertGreater(float(result["render"][3].mean().item()), 0.20)
        self.assertGreater(float(result["render"][:3].mean().item()), 0.02)
        self.assertEqual(tuple(frame.shape), (128, 192, 4))
        self.assertEqual(tuple(depth.shape), (128, 192))
        self.assertTrue(torch.all(frame[..., 3] == 255).item())
        self.assertEqual(metrics["compose_mode"], "garden_direct_output")
        self.assertTrue(metrics["direct_gaussian_output"])
        self.assertEqual(metrics["direct_gaussian_scene_name"], "ambulance")


if __name__ == "__main__":
    unittest.main()
