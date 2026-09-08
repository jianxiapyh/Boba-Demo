import unittest
from unittest.mock import patch

import torch

from gaussian_splatting.rotation_utils import eigh_3x3


class RotationWorkspaceTest(unittest.TestCase):
    def test_eigenpairs_reconstruct_matrices_across_chunk_boundaries(self):
        generator = torch.Generator().manual_seed(42)
        x = torch.randn(137, 3, 3, generator=generator, dtype=torch.float64)
        matrices = x.transpose(-1, -2) @ x
        # Include singular matrices and repeated eigenvalues as in rest poses.
        matrices[31] = 0
        matrices[32] = torch.eye(3, dtype=matrices.dtype)
        matrices[64] = torch.diag(torch.tensor([0., 0., 2.], dtype=matrices.dtype))
        matrices = matrices.transpose(-1, -2)
        values, vectors = eigh_3x3(matrices, chunk_size=32)
        expected_values = torch.linalg.eigvalsh(matrices)
        torch.testing.assert_close(values, expected_values)
        torch.testing.assert_close(
            vectors @ torch.diag_embed(values) @ vectors.transpose(-1, -2),
            matrices,
        )
        torch.testing.assert_close(
            vectors.transpose(-1, -2) @ vectors,
            torch.eye(3, dtype=matrices.dtype).expand_as(matrices),
        )

    def test_empty_batch(self):
        values, vectors = eigh_3x3(torch.empty(0, 3, 3))
        self.assertEqual(values.shape, (0, 3))
        self.assertEqual(vectors.shape, (0, 3, 3))

    def test_failed_matrix_index_refers_to_full_input(self):
        solver = torch.linalg.eigh
        calls = 0

        def fail_second_chunk(matrices):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise torch.linalg.LinAlgError("Batch element 2: failed to converge")
            return solver(matrices)

        with patch("torch.linalg.eigh", side_effect=fail_second_chunk):
            with self.assertRaisesRegex(torch.linalg.LinAlgError, "Batch element 10"):
                eigh_3x3(torch.eye(3).repeat(17, 1, 1), chunk_size=8)

    @unittest.skipUnless(torch.cuda.is_available(), "Requires a CUDA GPU")
    def test_sloth_sized_batches_keep_solver_workspace_below_two_gib(self):
        # The unchunked cu132 solves requested 30/15 GiB at 49/25 instances.
        torch.linalg.eigh(torch.eye(3, device="cuda").unsqueeze(0))
        for count in (100 * 2427, 49 * 2427, 25 * 2427):
            with self.subTest(count=count):
                x = torch.randn(count, 3, 3, device="cuda")
                matrices = x.transpose(-1, -2) @ x
                baseline = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                values, vectors = eigh_3x3(matrices)
                torch.cuda.synchronize()
                workspace_peak = torch.cuda.max_memory_allocated() - baseline
                self.assertLess(workspace_peak, 2 * 1024**3)
                torch.testing.assert_close(
                    vectors @ torch.diag_embed(values) @ vectors.transpose(-1, -2),
                    matrices,
                    rtol=1e-4,
                    atol=1e-4,
                )
                del x, matrices, values, vectors


if __name__ == "__main__":
    unittest.main()
