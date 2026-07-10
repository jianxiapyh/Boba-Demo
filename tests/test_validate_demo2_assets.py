import hashlib
import json
import pickle
import tempfile
import unittest
from pathlib import Path

from tools.validate_demo2_assets import AssetValidationError, validate_case_assets


CASE_NAME = "single_push_rope_4"
MANIFEST = {
    "best_model": "best_model.pth",
    "calibrate": "calibrate.pkl",
    "config": "configs/real.yaml",
    "final_data": "final_data.pkl",
    "gaussian_ply": "gaussian.ply",
    "metadata": "metadata.json",
    "optimal_params": "optimal_params.pkl",
    "controller_bank": "controller_bank.pkl",
    "background_image": "background.png",
}


class Demo2AssetValidationTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.repo_root = Path(self.temporary_directory.name)
        self.assets_root = self.repo_root / "assets"
        self.case_dir = self.assets_root / CASE_NAME
        self.case_dir.mkdir(parents=True)
        (self.repo_root / "configs").mkdir()
        (self.repo_root / "configs" / "real.yaml").write_text("device: cuda\n")
        (self.case_dir / "manifest.json").write_text(json.dumps(MANIFEST))

        for name in (
            "best_model.pth",
            "calibrate.pkl",
            "final_data.pkl",
            "optimal_params.pkl",
            "background.png",
        ):
            (self.case_dir / name).write_bytes(b"test payload")
        (self.case_dir / "metadata.json").write_text('{"frame_num": 1}')
        (self.case_dir / "gaussian.ply").write_text(
            "\n".join(
                (
                    "ply",
                    "format ascii 1.0",
                    "element vertex 1",
                    "property float x",
                    "property float y",
                    "property float z",
                    "property float f_dc_0",
                    "property float f_dc_1",
                    "property float f_dc_2",
                    "property float opacity",
                    "property float scale_0",
                    "property float rot_0",
                    "property float rot_1",
                    "property float rot_2",
                    "property float rot_3",
                    "end_header",
                    "0 0 0 0 0 0 1 1 1 0 0 0",
                )
            )
            + "\n"
        )
        self._write_controller_bank(100)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _write_controller_bank(self, count):
        bank = {
            "case_name": CASE_NAME,
            "meta": {"case_name": CASE_NAME},
            "controller_points_group": [[] for _ in range(count)],
            "source_indices": list(range(count)),
        }
        with (self.case_dir / "controller_bank.pkl").open("wb") as handle:
            pickle.dump(bank, handle)

    def _validate(self):
        return validate_case_assets(
            case_name=CASE_NAME,
            assets_root=self.assets_root,
            repo_root=self.repo_root,
        )

    def test_accepts_complete_case_without_optional_provenance(self):
        report = self._validate()
        self.assertEqual(report["assets"], 9)
        self.assertEqual(report["gaussian_vertices"], 1)
        self.assertEqual(report["trajectories"], 100)
        self.assertEqual(report["provenance_records"], 0)

    def test_rejects_git_lfs_pointer(self):
        (self.case_dir / "best_model.pth").write_text(
            "version https://git-lfs.github.com/spec/v1\n"
            "oid sha256:" + "0" * 64 + "\nsize 123\n"
        )
        with self.assertRaisesRegex(AssetValidationError, "Git LFS pointer"):
            self._validate()

    def test_rejects_controller_bank_with_wrong_trajectory_count(self):
        self._write_controller_bank(99)
        with self.assertRaisesRegex(AssetValidationError, "99 trajectories"):
            self._validate()

    def test_rejects_recorded_hash_mismatch(self):
        provenance = {
            "schema_version": 1,
            "case_name": CASE_NAME,
            "files": {},
        }
        for key, packaged_name in MANIFEST.items():
            if key == "config":
                continue
            path = self.case_dir / packaged_name
            provenance["files"][packaged_name] = {
                "source_path": packaged_name,
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        provenance["files"]["background.png"]["sha256"] = "0" * 64
        (self.case_dir / "asset_source.json").write_text(json.dumps(provenance))
        with self.assertRaisesRegex(AssetValidationError, "SHA256 mismatch"):
            self._validate()


if __name__ == "__main__":
    unittest.main()
