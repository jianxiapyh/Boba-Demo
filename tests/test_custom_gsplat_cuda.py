from __future__ import annotations

import os
import unittest

import torch


@unittest.skipUnless(torch.cuda.is_available(), "CUDA device required")
class CustomGsplatCudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["CONDA_DEFAULT_ENV"] = "phystwin"
        from gaussian_splatting._gsplat_vendor import rasterization

        cls.rasterization = staticmethod(rasterization)
        cls.device = torch.device("cuda:0")
        cls.width = 24
        cls.height = 18
        cls.viewmats = torch.eye(4, device=cls.device).repeat(2, 1, 1)
        cls.Ks = torch.tensor(
            [
                [[18.0, 0.0, 9.0], [0.0, 18.0, 9.0], [0.0, 0.0, 1.0]],
                [[18.0, 0.0, 14.0], [0.0, 18.0, 9.0], [0.0, 0.0, 1.0]],
            ],
            device=cls.device,
        )

    def _render(self, means: torch.Tensor, rasterize_mode: str = "classic"):
        count = means.shape[0]
        quats = torch.zeros((count, 4), device=self.device)
        if count:
            quats[:, 0] = 1.0
        return self.rasterization(
            means=means,
            quats=quats,
            scales=torch.full((count, 3), 0.08, device=self.device),
            opacities=torch.full((count,), 0.9, device=self.device),
            colors=torch.tensor([[1.0, 0.2, 0.1]], device=self.device).expand(
                count, -1
            ),
            viewmats=self.viewmats,
            Ks=self.Ks,
            backgrounds=torch.zeros((2, 3), device=self.device),
            width=self.width,
            height=self.height,
            packed=False,
            sh_degree=None,
            render_mode="RGB+ED",
            rasterize_mode=rasterize_mode,
        )

    def test_standard_two_camera_classic_and_antialiased(self):
        means = torch.tensor([[0.0, 0.0, 2.0]], device=self.device)
        x_coordinates = torch.arange(self.width, device=self.device, dtype=torch.float32)

        for mode in ("classic", "antialiased"):
            with self.subTest(mode=mode):
                colors, alphas, info = self._render(means, mode)
                self.assertEqual(tuple(colors.shape), (2, self.height, self.width, 4))
                self.assertEqual(tuple(alphas.shape), (2, self.height, self.width, 1))
                self.assertEqual(tuple(info["radii"].shape), (2, 1, 2))
                self.assertTrue(torch.isfinite(colors).all())
                self.assertTrue(torch.isfinite(alphas).all())

                weights = alphas[..., 0].sum(dim=1)
                centroids = (weights * x_coordinates).sum(dim=1) / weights.sum(dim=1)
                self.assertGreater(float(centroids[1] - centroids[0]), 3.0)

    def test_empty_and_offscreen_inputs(self):
        empty_colors, empty_alphas, empty_info = self._render(
            torch.empty((0, 3), device=self.device)
        )
        self.assertEqual(
            tuple(empty_colors.shape), (2, self.height, self.width, 4)
        )
        self.assertEqual(tuple(empty_alphas.shape), (2, self.height, self.width, 1))
        self.assertEqual(tuple(empty_info["radii"].shape), (2, 0, 2))

        _, offscreen_alphas, offscreen_info = self._render(
            torch.tensor([[1_000.0, 0.0, 2.0]], device=self.device)
        )
        self.assertFalse(torch.any(offscreen_info["radii"] > 0))
        self.assertEqual(float(offscreen_alphas.max()), 0.0)


if __name__ == "__main__":
    unittest.main()
