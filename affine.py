from functools import wraps
from math import isqrt

import torch

__all__ = [
    "operator_norm",
    "operator_diff",
    "frobenius_norm",
    "frobenius_diff",
]


def _enforce_matrix_shape(A: torch.Tensor) -> torch.Tensor:
    if A.ndim == 1:
        d = isqrt(len(A))
        if d * d != len(A):
            raise ValueError("Input 1D tensor length is not a perfect square")
        return A.view(d, d)
    elif A.ndim >= 2:
        s = A.shape
        if s[-2] != s[-1]:
            # Try treating the last dimension as flattened square matrix
            d = isqrt(s[-1])
            if d * d != s[-1]:
                raise ValueError("Last dimension is not a perfect square")
            return A.view(*s[:-1], d, d)
        else:
            return A
    else:
        raise ValueError(f"Not sure how to handle tensor of shape {A.shape}")


def enforce_args_shapes(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        new_args = [
            _enforce_matrix_shape(arg) if isinstance(arg, torch.Tensor) else arg
            for arg in args
        ]
        new_kwargs = {
            k: _enforce_matrix_shape(v) if isinstance(v, torch.Tensor) else v
            for k, v in kwargs.items()
        }
        return func(*new_args, **new_kwargs)

    return wrapper


@enforce_args_shapes
def operator_norm(A: torch.Tensor) -> torch.Tensor:
    """
    Compute the operator norm (spectral norm) of a matrix A.
    A is expected to be of shape (..., d, d)

    The operator norm treats the matrix as an operator on a vector space and returns the
    magnitude of ||A @ u||_2 for the unit vector u that maximizes this value. This is equivalent
    to the largest singular value of A, which can be computed more stably as the square root of the
    largest eigenvalue of A^T * A.

    This norm is invariant to rotations and translations of the input space.
    """
    # Compute A^T * A
    AtA = torch.einsum("...ij,...jk->...ik", A, A)
    # Compute eigenvalues of A^T * A
    eigvals = torch.linalg.eigvalsh(AtA)  # Shape (..., d)
    # The operator norm is the square root of the largest eigenvalue
    op_norm = torch.sqrt(eigvals[..., -1])  # Shape (...)
    return op_norm


@enforce_args_shapes
def operator_diff(A1: torch.Tensor, A2: torch.Tensor) -> torch.Tensor:
    """
    Compute the operator norm of the difference between two matrices A1 and A2.
    A1 and A2 are expected to be of shape (..., d, d)
    """
    return operator_norm(A1 - A2)


@enforce_args_shapes
def frobenius_norm(A: torch.Tensor) -> torch.Tensor:
    """
    Compute the Frobenius norm of a matrix A.
    A is expected to be of shape (..., d, d)

    The Frobenius norm treats the matrix as a vector in R^(d*d) and returns its Euclidean length.
    Note that this norm is sensitive to rotations of the input space and thus may not be ideal
    for measuring differences between affine transformations.
    """
    return torch.sqrt(torch.sum(A * A, dim=(-2, -1)))  # Shape (...)


@enforce_args_shapes
def frobenius_diff(A1: torch.Tensor, A2: torch.Tensor) -> torch.Tensor:
    """
    Compute the Frobenius norm of the difference between two matrices A1 and A2.
    A1 and A2 are expected to be of shape (..., d, d)
    """
    return frobenius_norm(A1 - A2)


def matrix_exp(log_A: torch.Tensor) -> torch.Tensor:
    """
    Given a batch of matrices log_A, compute the matrix exponential exp(log_A). This allows for
    optimization in the 'unconstrained' logarithm space.
    """
    s = log_A.shape
    return torch.matrix_exp(_enforce_matrix_shape(log_A)).reshape(s)
