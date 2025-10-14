import unittest

import numpy as np
import torch
import torch.testing as tt

from affine import (
    operator_norm,
    frobenius_norm,
    matrix_exp,
    operator_diff,
    frobenius_diff,
)


class TestLogParameterization(unittest.TestCase):
    def test_log_flat(self):
        log_A = torch.zeros(3, 4)
        A = matrix_exp(log_A)
        assert A.shape == log_A.shape
        tt.assert_close(A, torch.eye(2).repeat(3, 1, 1).view(3, 4))

    def test_log_2x2(self):
        log_A = torch.zeros(2, 2)
        A = matrix_exp(log_A)
        assert A.shape == log_A.shape
        tt.assert_close(A, torch.eye(2))

    def test_log_Bx2x2(self):
        log_A = torch.zeros(3, 2, 2)
        A = matrix_exp(log_A)
        assert A.shape == log_A.shape
        tt.assert_close(A, torch.eye(2).repeat(3, 1, 1))

    def test_log_Bx3x3(self):
        log_A = torch.zeros(5, 3, 3)
        A = matrix_exp(log_A)
        assert A.shape == log_A.shape
        tt.assert_close(A, torch.eye(3).repeat(5, 1, 1))


class TestOpNorm(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Seed chosen by keyboard-mashing
        torch.manual_seed(1189357)

    def test_operator_norm_2x2(self):
        # Should work on a (2, 2) matrix
        A = torch.tensor([[3.0, 0.0], [0.0, 4.0]])
        norm = operator_norm(A)
        assert norm.shape == ()
        tt.assert_close(norm, torch.tensor(4.0))

    def test_operator_norm_Bx2x2(self):
        # Should work on a batch of matrices
        A = torch.tensor([[3.0, 0.0], [0.0, 4.0]])
        A_batch = torch.stack([A, 2 * A, 3 * A])  # Shape (3, 2, 2)
        norm_batch = operator_norm(A_batch)
        assert norm_batch.shape == (3,)
        tt.assert_close(norm_batch, torch.tensor([4.0, 8.0, 12.0]))

    def test_operator_norm_3x3(self):
        # Should in principle work on a 3x3 matrix even though we don't use that
        A3 = torch.eye(3)
        norm3 = operator_norm(A3)
        assert norm3.shape == ()
        tt.assert_close(norm3, torch.tensor(1.0))

    def test_operator_norm_flat(self):
        # Should work on flattened square matrices
        A = torch.tensor([[3.0, 0.0], [0.0, 4.0]])
        A_batch = torch.stack([A, 2 * A, 3 * A])  # Shape (3, 2, 2)
        A_flat = A_batch.view(3, 4)  # Shape (3, 4)
        norm_flat = operator_norm(A_flat)
        assert norm_flat.shape == (3,)
        tt.assert_close(norm_flat, torch.tensor([4.0, 8.0, 12.0]))

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

    def test_operator_diff_is_symmetric(self):
        # Run a batch of 100 randomized tests at once
        A1 = torch.tensor([[3.0, 0.0], [0.0, 4.0]]).repeat(100, 1, 1) + torch.randn(
            100, 2, 2
        )
        A2 = torch.tensor([[3.0, 0.0], [0.0, 4.0]]).repeat(100, 1, 1) + torch.randn(
            100, 2, 2
        )
        diff_1_2 = operator_diff(A1, A2)
        diff_2_1 = operator_diff(A2, A1)
        tt.assert_close(diff_1_2, diff_2_1)


class TestFrobNorm(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Seed chosen by keyboard-mashing
        torch.manual_seed(17894165)

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
        tt.assert_close(norm_batch, torch.tensor([5.0, 10.0, 15.0]))

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
        tt.assert_close(norm_flat, torch.tensor([5.0, 10.0, 15.0]))

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

        tt.assert_close(norm_A, norm_A_rotated)

    def test_frobenius_diff_is_symmetric(self):
        # Run a batch of 100 randomized tests at once
        A1 = torch.tensor([[3.0, 0.0], [0.0, 4.0]]).repeat(100, 1, 1) + torch.randn(
            100, 2, 2
        )
        A2 = torch.tensor([[3.0, 0.0], [0.0, 4.0]]).repeat(100, 1, 1) + torch.randn(
            100, 2, 2
        )
        diff_1_2 = frobenius_diff(A1, A2)
        diff_2_1 = frobenius_diff(A2, A1)
        tt.assert_close(diff_1_2, diff_2_1)
