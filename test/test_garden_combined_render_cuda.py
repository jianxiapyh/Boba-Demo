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
class GardenCombinedRenderCudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from gaussian_splatting.scene.cameras import Camera
        from gaussian_splatting.scene.gaussian_model import GaussianModel
        from qqtt.garden_assets import (
            GardenAssetError,
            garden_source_cameras_path,
            validate_garden_quality,
        )
        from qqtt.garden_scene import GardenSceneRenderer, make_garden_layout

        try:
            validate_garden_quality(REPO_ROOT, "performance")
            camera_path = garden_source_cameras_path(REPO_ROOT)
            if not camera_path.is_file():
                raise GardenAssetError("official Garden cameras.json is missing")
        except GardenAssetError as exc:
            raise unittest.SkipTest(str(exc)) from exc

        cls.GaussianModel = GaussianModel
        cls.Camera = Camera
        cls.renderer = GardenSceneRenderer(
            REPO_ROOT / "assets/scenes",
            160,
            107,
            repo_root=REPO_ROOT,
            garden_quality="performance",
            eye_resolution=1344,
        )
        cls.layout = make_garden_layout(
            np.array([0.0, -0.78, -0.62], dtype=np.float32),
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
            repo_root=REPO_ROOT,
            scene_up=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        )
        cls.renderer.set_layout(cls.layout)
        cls.dynamic = cls._make_dynamic(2)
        cls.combined = cls.renderer.bind_dynamic_gaussians(cls.dynamic)

        with camera_path.open("r", encoding="utf-8") as handle:
            camera_entries = {int(entry["id"]): entry for entry in json.load(handle)}
        with (REPO_ROOT / "assets/scenes/garden/calibration.json").open(
            "r", encoding="utf-8"
        ) as handle:
            calibration = json.load(handle)
        source_rotation = np.asarray(
            calibration["source_to_canonical_rotation"], dtype=np.float32
        )
        source_center = np.asarray(
            calibration["source_table_center"], dtype=np.float32
        )
        source_scale = float(calibration["meters_per_source_unit"])
        cls.cameras = []
        for camera_id in (0, 1):
            entry = dict(camera_entries[camera_id])
            entry["position"] = (
                source_rotation
                @ (np.asarray(entry["position"], dtype=np.float32) - source_center)
                * source_scale
            ).tolist()
            entry["rotation"] = (
                source_rotation @ np.asarray(entry["rotation"], dtype=np.float32)
            ).tolist()
            cls.cameras.append(cls._make_camera(entry, width=160))

    @classmethod
    def tearDownClass(cls):
        renderer = getattr(cls, "renderer", None)
        if renderer is not None:
            renderer.delete()

    @classmethod
    def _make_dynamic(cls, count: int):
        model = cls.GaussianModel(sh_degree=3)
        model.active_sh_degree = 3
        model.isotropic = False
        model._xyz = torch.zeros((count, 3), dtype=torch.float32, device="cuda")
        model._xyz[:, 2] = -0.12
        if count > 1:
            model._xyz[:, 0] = torch.linspace(-0.04, 0.04, count, device="cuda")
        model._features_dc = torch.zeros(
            (count, 1, 3), dtype=torch.float32, device="cuda"
        )
        model._features_dc[:, 0, 0] = (0.9 - 0.5) / 0.28209479177387814
        model._features_rest = torch.zeros(
            (count, 15, 3), dtype=torch.float32, device="cuda"
        )
        model._opacity = torch.full(
            (count, 1), math.log(0.95 / 0.05), device="cuda"
        )
        model._scaling = torch.full((count, 3), math.log(0.025), device="cuda")
        model._rotation = torch.zeros((count, 4), device="cuda")
        model._rotation[:, 0] = 1.0
        return model

    @classmethod
    def _make_camera(cls, entry: dict, width: int):
        source_width = int(entry["width"])
        source_height = int(entry["height"])
        height = max(1, int(round(width * source_height / source_width)))
        rotation = np.asarray(entry["rotation"], dtype=np.float32)
        position = np.asarray(entry["position"], dtype=np.float32)
        translation = -(rotation.T @ position)
        fov_x = 2.0 * math.atan(source_width / (2.0 * float(entry["fx"])))
        fov_y = 2.0 * math.atan(source_height / (2.0 * float(entry["fy"])))
        return cls.Camera(
            (width, height),
            colmap_id=int(entry["id"]),
            R=rotation,
            T=translation,
            FoVx=fov_x,
            FoVy=fov_y,
            depth_params=None,
            image=None,
            invdepthmap=None,
            image_name=str(entry["img_name"]),
            uid=int(entry["id"]),
            data_device="cuda",
        )

    def test_stereo_batched_matches_two_separate_reference_renders(self):
        from gaussian_splatting.gaussian_renderer import render, render_batch

        pipeline = SimpleNamespace(absgrad=False, radius_clip=0.25)
        background = torch.zeros(3, dtype=torch.float32, device="cuda")
        with torch.inference_mode():
            batched = render_batch(
                self.cameras,
                self.combined,
                pipeline,
                background,
                antialiased=True,
            )
            separate = [
                render(
                    camera,
                    self.combined,
                    pipeline,
                    background,
                    antialiased=True,
                )
                for camera in self.cameras
            ]
        for batched_eye, separate_eye in zip(batched, separate):
            self.assertTrue(
                torch.allclose(
                    batched_eye["render"],
                    separate_eye["render"],
                    rtol=2.0e-3,
                    atol=3.0e-4,
                )
            )
            self.assertTrue(
                torch.allclose(
                    batched_eye["depth"],
                    separate_eye["depth"],
                    rtol=2.0e-3,
                    atol=3.0e-4,
                )
            )

    def test_direct_output_uses_combined_rgb_without_a_second_alpha_blend(self):
        from gaussian_splatting.gaussian_renderer import render_batch

        pipeline = SimpleNamespace(absgrad=False, radius_clip=0.25)
        background = torch.zeros(3, dtype=torch.float32, device="cuda")
        with torch.inference_mode():
            rendered = render_batch(
                self.cameras,
                self.combined,
                pipeline,
                background,
                antialiased=True,
            )
            for eye in rendered:
                frame, depth, metrics = (
                    self.renderer.prepare_direct_gaussian_eye_output(
                        eye["render"],
                        eye["depth"],
                        output_dtype=torch.float32,
                    )
                )
                self.assertTrue(
                    torch.allclose(
                        frame[..., :3] / 255.0,
                        eye["render"][:3].clamp(0.0, 1.0).permute(1, 2, 0),
                    )
                )
                self.assertTrue(torch.all(frame[..., 3] == 255.0))
                self.assertEqual(tuple(depth.shape), tuple(eye["depth"].shape))
                self.assertEqual(metrics["compose_mode"], "garden_direct_output")
                frame_uint8, _, _ = (
                    self.renderer.prepare_direct_gaussian_eye_output(
                        eye["render"],
                        eye["depth"],
                        output_dtype=torch.uint8,
                    )
                )
                expected_uint8 = (
                    eye["render"][:3]
                    .clamp(0.0, 1.0)
                    .permute(1, 2, 0)
                    .mul(255.0)
                    .round()
                    .to(torch.uint8)
                )
                self.assertTrue(torch.equal(frame_uint8[..., :3], expected_uint8))
                self.assertTrue(torch.all(frame_uint8[..., 3] == 255))

    def test_stereo_frustum_chunk_union_matches_full_reference(self):
        from gaussian_splatting.gaussian_renderer import render_batch

        pipeline = SimpleNamespace(absgrad=False, radius_clip=0.25)
        background = torch.zeros(3, dtype=torch.float32, device="cuda")
        all_chunk_ids = tuple(range(int(self.renderer._chunk_starts.size)))
        self.renderer._copy_selected_static_chunks(all_chunk_ids)
        with torch.inference_mode():
            full_reference = render_batch(
                self.cameras,
                self.combined,
                pipeline,
                background,
                antialiased=True,
            )

        eye_states = []
        for camera in self.cameras:
            width = int(camera.image_width)
            height = int(camera.image_height)
            fx = width / (2.0 * math.tan(float(camera.FoVx) * 0.5))
            fy = height / (2.0 * math.tan(float(camera.FoVy) * 0.5))
            intrinsic = np.array(
                [[fx, 0.0, width * 0.5], [0.0, fy, height * 0.5], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
            eye_states.append(
                {
                    "view": camera,
                    "w2c_cv_np": camera.world_view_transform.T.detach().cpu().numpy(),
                    "intrinsic_np": intrinsic,
                    "width": width,
                    "height": height,
                }
            )
        selected, debug = self.renderer.select_stereo_frustum_gaussians(
            eye_states[0],
            eye_states[1],
        )
        self.assertTrue(debug["rebuild_started"])
        self.assertTrue(debug["rebuild_pending"])
        torch.cuda.synchronize()
        selected, debug = self.renderer.select_stereo_frustum_gaussians(
            eye_states[0],
            eye_states[1],
        )
        self.assertIs(selected, self.renderer.combined_gaussians)
        self.__class__.combined = selected
        self.assertTrue(debug["rebuilt"])
        self.assertLess(debug["selected_static_count"], debug["total_static_count"])
        self.assertEqual(
            int(selected._xyz.shape[0]),
            debug["selected_static_count"] + debug["dynamic_count"],
        )
        selected_pointer = int(selected._xyz.data_ptr())
        selected_again, cached_debug = (
            self.renderer.select_stereo_frustum_gaussians(
                eye_states[0],
                eye_states[1],
            )
        )
        self.assertIs(selected_again, selected)
        self.assertFalse(cached_debug["rebuilt"])
        self.assertEqual(int(selected_again._xyz.data_ptr()), selected_pointer)
        self.assertEqual(
            cached_debug["render_gaussian_count"],
            cached_debug["selected_static_count"] + cached_debug["dynamic_count"],
        )
        with torch.inference_mode():
            selected_render = render_batch(
                self.cameras,
                selected,
                pipeline,
                background,
                antialiased=True,
            )
        for reference_eye, selected_eye in zip(full_reference, selected_render):
            self.assertTrue(
                torch.allclose(
                    selected_eye["render"],
                    reference_eye["render"],
                    rtol=3.0e-3,
                    atol=5.0e-4,
                )
            )
            self.assertTrue(
                torch.allclose(
                    selected_eye["depth"],
                    reference_eye["depth"],
                    rtol=3.0e-3,
                    atol=5.0e-4,
                )
            )

    def test_repeated_object_bind_reuses_static_storage_and_hides_old_tail(self):
        pointers = {
            name: getattr(self.combined, name).data_ptr()
            for name in (
                "_xyz",
                "_features_dc",
                "_features_rest",
                "_opacity",
                "_scaling",
                "_rotation",
            )
        }
        static_prefix = self.combined._xyz[:16].clone()
        replacement = self._make_dynamic(1)
        rebound = self.renderer.bind_dynamic_gaussians(replacement)
        self.assertIs(rebound, self.combined)
        self.assertTrue(torch.equal(rebound._xyz[:16], static_prefix))
        self.assertEqual(
            pointers,
            {name: getattr(rebound, name).data_ptr() for name in pointers},
        )
        start = self.renderer._active_static_count
        self.assertGreater(float(rebound._opacity[start].item()), -10.0)
        self.assertEqual(int(rebound._xyz.shape[0]), start + 1)

    def test_actual_rope_sloth_switches_keep_combined_gpu_storage_stable(self):
        rope = self.GaussianModel(sh_degree=3)
        rope.load_ply(str(REPO_ROOT / "assets/rope_game/phystwin_rope.ply"))
        sloth = self.GaussianModel(sh_degree=3)
        sloth.load_ply(str(REPO_ROOT / "assets/sloth/sloth.ply"))
        self.assertLessEqual(
            max(int(rope._xyz.shape[0]), int(sloth._xyz.shape[0])),
            self.renderer._dynamic_capacity,
        )

        pointers = {
            name: getattr(self.combined, name).data_ptr()
            for name in (
                "_xyz",
                "_features_dc",
                "_features_rest",
                "_opacity",
                "_scaling",
                "_rotation",
            )
        }
        static_prefix = self.combined._xyz[:16].clone()
        self.renderer.bind_dynamic_gaussians(rope)
        self.renderer.bind_dynamic_gaussians(sloth)
        torch.cuda.synchronize()
        baseline_allocated = int(torch.cuda.memory_allocated())

        for _ in range(8):
            self.renderer.bind_dynamic_gaussians(rope)
            self.renderer.bind_dynamic_gaussians(sloth)
        torch.cuda.synchronize()
        final_allocated = int(torch.cuda.memory_allocated())

        self.assertEqual(
            pointers,
            {name: getattr(self.combined, name).data_ptr() for name in pointers},
        )
        self.assertTrue(torch.equal(self.combined._xyz[:16], static_prefix))
        self.assertEqual(self.renderer._dynamic_count, int(sloth._xyz.shape[0]))
        self.assertLessEqual(final_allocated, baseline_allocated + 8 * 1024 * 1024)


if __name__ == "__main__":
    unittest.main()
