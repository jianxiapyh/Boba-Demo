from __future__ import annotations

import importlib
import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


class _FakeGaussianModel:
    def __init__(self, count: int = 3):
        self.get_xyz = torch.zeros((count, 3), dtype=torch.float32)
        self.get_opacity = torch.ones((count, 1), dtype=torch.float32)
        self.get_scaling = torch.ones((count, 3), dtype=torch.float32)
        self.get_rotation = torch.zeros((count, 4), dtype=torch.float32)
        self.get_features = torch.zeros((count, 1, 3), dtype=torch.float32)
        self.active_sh_degree = 0


class _FakeCamera:
    def __init__(self, principal_x: float, view_offset: float = 0.0):
        self.image_width = 3
        self.image_height = 2
        self.K = torch.tensor(
            [[4.0, 0.0, principal_x], [0.0, 5.0, 1.0], [0.0, 0.0, 1.0]]
        )
        self.world_view_transform = torch.eye(4)
        self.world_view_transform[3, 0] = view_offset


class CustomGsplatTests(unittest.TestCase):
    def setUp(self):
        os.environ["CONDA_DEFAULT_ENV"] = "phystwin-cu132"

    def test_vendor_replaces_preloaded_external_gsplat(self):
        script = textwrap.dedent(
            """
            import pathlib
            import sys
            import types

            external = types.ModuleType("gsplat")
            external.__file__ = "/tmp/external-gsplat/gsplat/__init__.py"
            sys.modules["gsplat"] = external

            from gaussian_splatting._gsplat_vendor import (
                VENDORED_GSPLAT_PACKAGE,
                gsplat,
            )

            resolved = pathlib.Path(gsplat.__file__).resolve()
            assert resolved.is_relative_to(VENDORED_GSPLAT_PACKAGE.resolve())
            assert gsplat.__version__ == "1.5.3"
            assert callable(gsplat.rasterization)
            assert callable(gsplat.rasterization_shared_template)
            assert callable(gsplat.fully_fused_projection_shared_template)
            print(resolved)
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[1],
            env={**os.environ, "PYTHONNOUSERSITE": "1"},
            check=True,
            text=True,
            capture_output=True,
        )
        expected_path = (
            Path(__file__).resolve().parents[1]
            / "gaussian_splatting/submodules/gsplat/gsplat/__init__.py"
        ).resolve()
        self.assertIn(str(expected_path), result.stdout)

    def test_two_camera_standard_call_and_elliptical_radii(self):
        renderer = importlib.import_module("gaussian_splatting.gaussian_renderer")
        captured = {}

        def fake_rasterization(**kwargs):
            captured.update(kwargs)
            colors = torch.zeros((2, 2, 3, 4), dtype=torch.float32)
            colors[0, ..., 0] = 0.25
            colors[1, ..., 0] = 0.75
            alphas = torch.ones((2, 2, 3, 1), dtype=torch.float32)
            info = {
                "radii": torch.tensor(
                    [
                        [[1, 4], [0, 0], [3, 2]],
                        [[5, 2], [0, 1], [7, 9]],
                    ],
                    dtype=torch.int32,
                ),
                "means2d": torch.zeros((2, 3, 2), dtype=torch.float32),
            }
            return colors, alphas, info

        with mock.patch.object(renderer, "rasterization", fake_rasterization):
            packages = renderer.render_gsplat_batch(
                [_FakeCamera(1.25), _FakeCamera(1.75, view_offset=0.1)],
                _FakeGaussianModel(),
                pipe=SimpleNamespace(max_projected_radius=1024.0),
                bg_color=torch.zeros(3),
                antialiased=True,
            )

        self.assertEqual(tuple(captured["viewmats"].shape), (2, 4, 4))
        self.assertEqual(tuple(captured["Ks"].shape), (2, 3, 3))
        self.assertEqual(captured["Ks"][:, 0, 2].tolist(), [1.25, 1.75])
        self.assertEqual(captured["rasterize_mode"], "antialiased")
        self.assertEqual(captured["render_mode"], "RGB+ED")
        self.assertEqual(captured["max_projected_radius"], 1024.0)
        self.assertEqual(len(packages), 2)
        self.assertEqual(tuple(packages[0]["render"].shape), (4, 2, 3))
        self.assertEqual(packages[0]["radii"].tolist(), [4, 0, 3])
        self.assertEqual(packages[1]["radii"].tolist(), [5, 1, 9])
        self.assertEqual(
            packages[0]["visibility_filter"].tolist(), [True, False, True]
        )
        self.assertEqual(
            packages[1]["visibility_filter"].tolist(), [True, True, True]
        )
        self.assertTrue(torch.all(packages[0]["render"][0] == 0.25))
        self.assertTrue(torch.all(packages[1]["render"][0] == 0.75))

    def test_single_camera_elliptical_radii_are_scalarized(self):
        renderer = importlib.import_module("gaussian_splatting.gaussian_renderer")

        def fake_rasterization(**_kwargs):
            return (
                torch.zeros((1, 2, 3, 4), dtype=torch.float32),
                torch.ones((1, 2, 3, 1), dtype=torch.float32),
                {
                    "radii": torch.tensor([[[2, 6], [0, 0], [4, 3]]]),
                    "means2d": torch.zeros((1, 3, 2), dtype=torch.float32),
                },
            )

        with mock.patch.object(renderer, "rasterization", fake_rasterization):
            package = renderer.render_gsplat(
                _FakeCamera(1.5),
                _FakeGaussianModel(),
                pipe=None,
                bg_color=torch.zeros(3),
            )

        self.assertEqual(package["radii"].tolist(), [6, 0, 4])
        self.assertEqual(
            package["visibility_filter"].tolist(), [True, False, True]
        )

    def test_projected_radius_limit_bounds_and_reculls_offscreen_splats(self):
        importlib.import_module("gaussian_splatting.gaussian_renderer")
        rendering = importlib.import_module("gsplat.rendering")

        radii = torch.tensor(
            [[[809_933, 400_000], [2_000, 3_000], [8, 9], [0, 0]]],
            dtype=torch.int32,
        )
        means2d = torch.tensor(
            [
                [
                    [-5_000.0, 672.0],
                    [672.0, 672.0],
                    [1_400.0, 672.0],
                    [0.0, 0.0],
                ]
            ]
        )
        bounded = rendering._apply_projected_radius_limit(
            radii,
            means2d,
            width=1344,
            height=1344,
            max_projected_radius=1024.0,
        )

        self.assertEqual(
            bounded.tolist(),
            [[[0, 0], [1024, 1024], [0, 0], [0, 0]]],
        )
        self.assertIs(
            rendering._apply_projected_radius_limit(
                radii,
                means2d,
                width=1344,
                height=1344,
                max_projected_radius=0.0,
            ),
            radii,
        )


if __name__ == "__main__":
    unittest.main()
