import torch
from torch._higher_order_ops.associative_scan import associative_scan


def block2_mat_operator(q_i, q_j):
    A_i, b_i = q_i
    A_j, b_j = q_j

    ai00 = A_i[..., 0, 0]
    ai01 = A_i[..., 0, 1]
    ai10 = A_i[..., 1, 0]
    ai11 = A_i[..., 1, 1]

    aj00 = A_j[..., 0, 0]
    aj01 = A_j[..., 0, 1]
    aj10 = A_j[..., 1, 0]
    aj11 = A_j[..., 1, 1]

    a00 = aj00 * ai00 + aj01 * ai10
    a01 = aj00 * ai01 + aj01 * ai11
    a10 = aj10 * ai00 + aj11 * ai10
    a11 = aj10 * ai01 + aj11 * ai11

    row0 = torch.stack((a00, a01), dim=-1)
    row1 = torch.stack((a10, a11), dim=-1)
    A_out = torch.stack((row0, row1), dim=-2)

    bi0 = b_i[..., 0]
    bi1 = b_i[..., 1]
    bj0 = b_j[..., 0]
    bj1 = b_j[..., 1]

    b0 = aj00 * bi0 + aj01 * bi1 + bj0
    b1 = aj10 * bi0 + aj11 * bi1 + bj1
    b_out = torch.stack((b0, b1), dim=-1)

    return A_out, b_out


def block2_mat_scan(A: torch.Tensor, b: torch.Tensor, dim: int = 0):
    if A.shape[:-2] != b.shape[:-1]:
        raise ValueError(
            "A and b have incompatible leading dimensions: "
            f"A={tuple(A.shape)}, b={tuple(b.shape)}."
        )
    if A.shape[-2:] != (2, 2):
        raise ValueError(f"A must end with shape (2, 2), got {tuple(A.shape)}.")
    if b.shape[-1] != 2:
        raise ValueError(f"b must end with shape (2,), got {tuple(b.shape)}.")
    return associative_scan(block2_mat_operator, (A, b), dim=dim, combine_mode="generic")

