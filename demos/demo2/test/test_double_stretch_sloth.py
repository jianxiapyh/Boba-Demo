import json
import pickle
import unittest
from pathlib import Path

import numpy as np

from demos.demo2.case_assets import resolve_demo2_case_assets
from demos.demo2.control import (
    control_vector_to_step,
    resolve_controller_part_indices,
    resolve_demo2_control_parts,
    resolve_phone_to_world_axis_signs,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


class DoubleStretchSlothCaseTest(unittest.TestCase):
    def _load_camera_geometry(self, case_name):
        assets = resolve_demo2_case_assets(REPO_ROOT, case_name)
        with assets.final_data.open("rb") as handle:
            data = pickle.load(handle)
        with assets.calibrate.open("rb") as handle:
            c2ws = pickle.load(handle)
        metadata = json.loads(assets.metadata.read_text(encoding="utf-8"))
        return (
            np.asarray(data["controller_points"])[0],
            np.linalg.inv(np.asarray(c2ws)[0]),
            np.asarray(metadata["intrinsics"])[0],
        )

    @staticmethod
    def _project(point, w2c, intrinsic):
        projected = intrinsic @ (w2c[:3, :] @ np.concatenate([point, [1.0]]))
        return projected[:2] / projected[2]

    def test_case_is_packaged_with_two_independently_moving_regions(self):
        assets = resolve_demo2_case_assets(REPO_ROOT, "double_stretch_sloth")
        self.assertEqual(resolve_demo2_control_parts(assets.case_name), 2)

        with assets.controller_bank.open("rb") as handle:
            bank = pickle.load(handle)
        self.assertEqual(bank["case_name"], assets.case_name)
        self.assertEqual(bank["meta"]["case_name"], assets.case_name)
        self.assertEqual(len(bank["controller_points_group"]), 100)

        with assets.final_data.open("rb") as handle:
            data = pickle.load(handle)
        with assets.calibrate.open("rb") as handle:
            c2ws = pickle.load(handle)
        metadata = json.loads(assets.metadata.read_text(encoding="utf-8"))
        controller_points = np.asarray(data["controller_points"])
        parts = resolve_controller_part_indices(
            controller_points[0],
            2,
            w2c=np.linalg.inv(np.asarray(c2ws)[0]),
            intrinsic=np.asarray(metadata["intrinsics"])[0],
        )

        self.assertEqual(len(parts), 2)
        self.assertTrue(all(len(indices) > 0 for indices in parts))
        self.assertEqual(
            set(parts[0]).union(parts[1]),
            set(range(controller_points.shape[1])),
        )
        self.assertTrue(set(parts[0]).isdisjoint(parts[1]))
        for indices in parts:
            region_centers = controller_points[:, indices].mean(axis=1)
            self.assertGreater(float(np.linalg.norm(np.ptp(region_centers, axis=0))), 0.01)

    def test_phone_arrows_are_camera_calibrated_for_rope_and_sloth(self):
        expected_signs = {
            "single_push_rope_4": (-1.0, -1.0, 1.0),
            "double_stretch_sloth": (1.0, 1.0, 1.0),
        }
        phone_directions = (
            ("forward", (1.0, 0.0, 0.0), 1, -1.0),
            ("backward", (-1.0, 0.0, 0.0), 1, 1.0),
            ("left", (0.0, -1.0, 0.0), 0, -1.0),
            ("right", (0.0, 1.0, 0.0), 0, 1.0),
            ("up", (0.0, 0.0, -1.0), 1, -1.0),
            ("down", (0.0, 0.0, 1.0), 1, 1.0),
        )

        for case_name, expected in expected_signs.items():
            with self.subTest(case_name=case_name):
                controller_points, w2c, intrinsic = self._load_camera_geometry(case_name)
                axis_signs = resolve_phone_to_world_axis_signs(
                    controller_points,
                    w2c=w2c,
                    intrinsic=intrinsic,
                )
                self.assertEqual(axis_signs, expected)

                region_indices = resolve_controller_part_indices(
                    controller_points,
                    resolve_demo2_control_parts(case_name),
                    w2c=w2c,
                    intrinsic=intrinsic,
                )
                for region_idx, indices in enumerate(region_indices):
                    center = controller_points[indices].mean(axis=0)
                    base_pixel = self._project(center, w2c, intrinsic)
                    for direction, phone_vector, component, desired_sign in phone_directions:
                        step = np.asarray(
                            control_vector_to_step(
                                phone_vector[0],
                                phone_vector[1],
                                phone_vector[2],
                                0.005,
                                axis_signs=axis_signs,
                            )
                        )
                        screen_delta = self._project(
                            center + step,
                            w2c,
                            intrinsic,
                        ) - base_pixel
                        self.assertGreater(
                            float(screen_delta[component] * desired_sign),
                            0.0,
                            msg=(
                                f"{case_name} region {region_idx} {direction} "
                                "arrow moved the opposite way"
                            ),
                        )


if __name__ == "__main__":
    unittest.main()
