import unittest

import numpy as np
import torch

from affine import operator_norm, frobenius_norm


class TestOpNorm(unittest.TestCase):
    def test_operator_norm_2x2(self):
        # Should work on a (2, 2) matrix
        A = torch.tensor([[3.0, 0.0], [0.0, 4.0]])
        norm = operator_norm(A)
        assert norm.shape == ()
        assert torch.isclose(norm, torch.tensor(4.0))

    def test_operator_norm_Bx2x2(self):
        # Should work on a batch of matrices
        A = torch.tensor([[3.0, 0.0], [0.0, 4.0]])
        A_batch = torch.stack([A, 2 * A, 3 * A])  # Shape (3, 2, 2)
        norm_batch = operator_norm(A_batch)
        assert norm_batch.shape == (3,)
        assert torch.allclose(norm_batch, torch.tensor([4.0, 8.0, 12.0]))

    def test_operator_norm_3x3(self):
        # Should in principle work on a 3x3 matrix even though we don't use that
        A3 = torch.eye(3)
        norm3 = operator_norm(A3)
        assert norm3.shape == ()
        assert torch.isclose(norm3, torch.tensor(1.0))

    def test_operator_norm_flat(self):
        # Should work on flattened square matrices
        A = torch.tensor([[3.0, 0.0], [0.0, 4.0]])
        A_batch = torch.stack([A, 2 * A, 3 * A])  # Shape (3, 2, 2)
        A_flat = A_batch.view(3, 4)  # Shape (3, 4)
        norm_flat = operator_norm(A_flat)
        assert norm_flat.shape == (3,)
        assert torch.allclose(norm_flat, torch.tensor([4.0, 8.0, 12.0]))

    def test_operator_norm_is_rotation_invariant(self):
        # The operator norm should be invariant under rotations
        A = torch.tensor([[3.0, 0.0], [0.0, 4.0]])
        norm_A = operator_norm(A)

        theta = torch.pi / 4  # 45 degrees
        R = torch.tensor(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
            dtype=torch.float32,
        )

        A_rotated = R @ A @ R.T
        norm_A_rotated = operator_norm(A_rotated)

        assert torch.isclose(norm_A, norm_A_rotated)


class TestFrobNorm(unittest.TestCase):
    def test_frobenius_norm_2x2(self):
        # Should work on a (2, 2) matrix
        A = torch.tensor([[3.0, 0.0], [0.0, 4.0]])
        norm = frobenius_norm(A)
        assert norm.shape == ()
        assert torch.isclose(norm, torch.tensor(5.0))

    def test_frobenius_norm_Bx2x2(self):
        # Should work on a batch of matrices
        A = torch.tensor([[3.0, 0.0], [0.0, 4.0]])
        A_batch = torch.stack([A, 2 * A, 3 * A])  # Shape (3, 2, 2)
        norm_batch = frobenius_norm(A_batch)
        assert norm_batch.shape == (3,)
        assert torch.allclose(norm_batch, torch.tensor([5.0, 10.0, 15.0]))

    def test_frobenius_norm_3x3(self):
        # Should in principle work on a 3x3 matrix even though we don't use that
        A3 = torch.eye(3)
        norm3 = frobenius_norm(A3)
        assert norm3.shape == ()
        assert torch.isclose(norm3, torch.sqrt(torch.tensor(3.0)))

    def test_frobenius_norm_flat(self):
        # Should work on flattened square matrices
        A = torch.tensor([[3.0, 0.0], [0.0, 4.0]])
        A_batch = torch.stack([A, 2 * A, 3 * A])  # Shape (3, 2, 2)
        A_flat = A_batch.view(3, 4)  # Shape (3, 4)
        norm_flat = frobenius_norm(A_flat)
        assert norm_flat.shape == (3,)
        assert torch.allclose(norm_flat, torch.tensor([5.0, 10.0, 15.0]))

    def test_frobenius_norm_is_rotation_invariant(self):
        # The frobenius norm is expected to be invariant under rotations
        A = torch.tensor([[3.0, 0.0], [0.0, 4.0]])
        norm_A = frobenius_norm(A)

        theta = torch.pi / 4  # 45 degrees
        R = torch.tensor(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
            dtype=torch.float32,
        )

        A_rotated = R @ A @ R.T
        norm_A_rotated = frobenius_norm(A_rotated)

        assert torch.isclose(norm_A, norm_A_rotated)
