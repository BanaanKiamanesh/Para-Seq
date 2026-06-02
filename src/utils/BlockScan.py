import torch
from torch._higher_order_ops.associative_scan import associative_scan


def block2_mat_operator(q_i, q_j):
    """Associative composition for independent 2x2 affine maps.

    Each element represents

        x_t = A_t x_{t-1} + b_t,

    where x_t has shape (..., 2), A_t has shape (..., 2, 2), and b_t has
    shape (..., 2). Composition q_j o q_i is

        A = A_j A_i,
        b = A_j b_i + b_j.
    """
    A_i, b_i = q_i
    A_j, b_j = q_j

    A_out = A_j @ A_i
    b_out = (A_j @ b_i.unsqueeze(-1)).squeeze(-1) + b_j

    return A_out, b_out


def block2_mat_scan(A: torch.Tensor, b: torch.Tensor, dim: int = 0):
    """Associative scan for independent 2x2 affine recurrences.

    Example LSTM block recurrence:

        A.shape == (B, T, H, 2, 2)
        b.shape == (B, T, H, 2)
        dim == 1
    """
    if A.shape[:-2] != b.shape[:-1]:
        raise ValueError(
            "A and b have incompatible leading dimensions: "
            f"A={tuple(A.shape)}, b={tuple(b.shape)}."
        )

    if A.shape[-2:] != (2, 2):
        raise ValueError(f"A must end with shape (2, 2), got {tuple(A.shape)}.")

    if b.shape[-1] != 2:
        raise ValueError(f"b must end with shape (2,), got {tuple(b.shape)}.")

    return associative_scan(
        block2_mat_operator,
        (A, b),
        dim=dim,
        combine_mode="generic",
    )
