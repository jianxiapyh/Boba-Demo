import contextlib
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[3]


class RuntimeEnvironmentTest(unittest.TestCase):
    def test_shell_entry_points_reject_legacy_and_missing_environments(self):
        for script in (
            "scripts/run_demo2.sh",
            "scripts/demo2_preflight.sh",
            "env_install/install_demo2_extras.sh",
        ):
            for active_env in (None, "phystwin", "/tmp/envs/phystwin", "base"):
                with self.subTest(script=script, environment=active_env):
                    env = os.environ.copy()
                    env.pop("CONDA_PREFIX", None)
                    env.pop("CONDA_DEFAULT_ENV", None)
                    if active_env is not None:
                        env["CONDA_DEFAULT_ENV"] = active_env
                    result = subprocess.run(
                        ["bash", str(REPO_ROOT / script), "--help"],
                        env=env, cwd=REPO_ROOT, capture_output=True, text=True,
                        timeout=10,
                    )
                    self.assertNotEqual(result.returncode, 0)
                    policy_errors = [
                        line for line in result.stderr.splitlines()
                        if line.startswith("[Demo2")
                    ]
                    self.assertTrue(policy_errors, result.stderr)
                    self.assertIn("activate phystwin-cu132", policy_errors[0])

    @unittest.skipUnless(
        Path(sys.prefix).resolve().name == "phystwin-cu132",
        "Requires the phystwin-cu132 environment",
    )
    def test_launcher_accepts_required_environment_name_and_prefix(self):
        for active_env in ("phystwin-cu132", sys.prefix):
            with self.subTest(environment=active_env):
                env = os.environ.copy()
                env.update(CONDA_DEFAULT_ENV=active_env, CONDA_PREFIX=sys.prefix,
                           CUDA_HOME=sys.prefix)
                result = subprocess.run(
                    ["bash", str(REPO_ROOT / "scripts/run_demo2.sh"), "--help"],
                    env=env, cwd=REPO_ROOT, capture_output=True, text=True,
                    timeout=20,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--case_name", result.stdout)



class LinearAlgebraPolicyTest(unittest.TestCase):
    def load_policy(self, device_name, override=None):
        spec = importlib.util.spec_from_file_location(
            "gaussian_splatting._demo2_dynamic_policy_test",
            REPO_ROOT / "gaussian_splatting/dynamic_utils.py",
        )
        module = importlib.util.module_from_spec(spec)
        environment = {"BOBA_DEVICE": "auto"}
        if override is not None:
            environment["BOBA_LINALG_BACKEND"] = override
        with (
            patch.dict(os.environ, environment, clear=True),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=1),
            patch("torch.cuda.get_device_name", return_value=device_name),
            patch("torch.backends.cuda.preferred_linalg_library") as backend,
            patch("importlib.import_module", return_value=SimpleNamespace(__all__=[])),
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            spec.loader.exec_module(module)
        backend.assert_called_once_with(module.SELECTED_LINALG_BACKEND)
        self.assertIn(f"Linear algebra backend: {module.SELECTED_LINALG_BACKEND}", output.getvalue())
        return module

    def test_all_desktop_gpus_default_to_cusolver(self):
        for device_name in (
            "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition",
            "NVIDIA GeForce RTX 5090",
            "NVIDIA GeForce RTX 4090",
        ):
            for override in (None, "", "auto"):
                with self.subTest(device=device_name, override=override):
                    module = self.load_policy(device_name, override)
                    self.assertEqual(module.SELECTED_LINALG_BACKEND, "cusolver")

    def test_stale_magma_override_does_not_change_production_backend(self):
        module = self.load_policy("NVIDIA RTX PRO 6000 Blackwell", "MAGMA")
        self.assertEqual(module.SELECTED_LINALG_BACKEND, "cusolver")


if __name__ == "__main__":
    unittest.main()
