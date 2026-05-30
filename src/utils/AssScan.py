import torch
from torch._higher_order_ops.associative_scan import associative_scan


def full_mat_operator(q_i, q_j):
    """Binary operator for parallel scan of linear recurrence. Assumes a full Jacobian matrix A
    Args:
        q_i: tuple containing A_i and b_i at position i       (..., D, D), (..., D)
        q_j: tuple containing A_j and b_j at position j       (..., D, D), (..., D)
    Returns:
        new element (A_out, b_out)
    """
    A_i, b_i = q_i
    A_j, b_j = q_j

    A_out = A_j @ A_i
    b_out = (A_j @ b_i.unsqueeze(-1)).squeeze(-1) + b_j

    return A_out, b_out


def diag_mat_operator(q_i, q_j):
    """Binary operator for parallel scan of linear recurrence. Assumes a DIAGONAL Jacobian matrix A
    Args:
        q_i: tuple containing diag(A_i) and b_i at position i       (..., D), (..., D)
        q_j: tuple containing diag(A_j) and b_j at position j       (..., D), (..., D)
    Returns:
        new element (A_out, b_out)
    """
    A_i, b_i = q_i
    A_j, b_j = q_j

    A_out = A_j * A_i
    b_out = A_j * b_i + b_j

    return A_out, b_out


def full_mat_scan(A, b, dim=0):
    """Associative scan for dense affine recurrences.

    Solves prefix compositions of

        x_t = A_t x_{t-1} + b_t

    Args:
        A: tensor with dense transition matrices, shape (..., T, D, D) if dim is the time dimension
        b: tensor with bias vectors, shape (..., T, D)

    Returns:
        A_prefix: prefix-composed transition matrices
        b_prefix: prefix-composed bias vectors
    """
    return associative_scan(
        full_mat_operator,
        (A, b),
        dim=dim,
        combine_mode="generic",
    )


def diag_mat_scan(A, b, dim=0):
    """Associative scan for diagonal affine recurrences.

    Solves prefix compositions of

        x_t = A_t * x_{t-1} + b_t

    Args:
        A: tensor with diagonal transition entries, shape (..., T, D) if dim is the time dimension
        b: tensor with bias vectors, shape (..., T, D)

    Returns:
        A_prefix: prefix-composed diagonal transition entries
        b_prefix: prefix-composed bias vectors
    """
    return associative_scan(
        diag_mat_operator,
        (A, b),
        dim=dim,
        combine_mode="generic",
    )
