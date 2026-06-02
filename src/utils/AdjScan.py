import torch

from src.utils.AssScan import diag_mat_scan
from src.utils.AccelScan import diag_mat_scan_accel_batched


def reverse_diag_adjoint_scan(
    jacobian_diag,
    grad_states,
    scan_backend="torch",
    accel_scan_fn=None,
):
    """Parallel reverse-time adjoint scan for diagonal recurrent Jacobians.

    For a recurrence

        h_t = f_t(h_{t-1}),

    assume jacobian_diag[:, t] stores diag(dh_t / dh_{t-1}) and grad_states[:, t]
    stores the direct gradient contribution dL / dh_t.

    The total adjoint satisfies

        lambda_{T-1} = grad_states_{T-1},
        lambda_t = grad_states_t + jacobian_diag_{t+1} * lambda_{t+1}.
    """
    if jacobian_diag.ndim != 3:
        raise ValueError(
            "jacobian_diag must have shape (B, T, D), got "
            f"{tuple(jacobian_diag.shape)}."
        )

    if grad_states.ndim != 3:
        raise ValueError(
            "grad_states must have shape (B, T, D), got "
            f"{tuple(grad_states.shape)}."
        )

    if jacobian_diag.shape != grad_states.shape:
        raise ValueError(
            "jacobian_diag and grad_states must have the same shape, got "
            f"{tuple(jacobian_diag.shape)} and {tuple(grad_states.shape)}."
        )

    _, seq_len, _ = grad_states.shape

    if seq_len == 0:
        return grad_states.clone()

    A_rev = torch.zeros_like(jacobian_diag)
    b_rev = torch.flip(grad_states, dims=[1])

    if seq_len > 1:
        A_rev[:, 1:, :] = torch.flip(jacobian_diag[:, 1:, :], dims=[1])

    if scan_backend == "torch":
        _, lambda_rev = diag_mat_scan(A_rev, b_rev, dim=1)
    elif scan_backend == "accel_scan":
        _, lambda_rev = diag_mat_scan_accel_batched(
            A=A_rev,
            b=b_rev,
            accel_scan_fn=accel_scan_fn,
        )
    else:
        raise ValueError(f"Unknown scan_backend: {scan_backend}")

    return torch.flip(lambda_rev, dims=[1])


def reverse_diag_adjoint_loop(jacobian_diag, grad_states):
    """Sequential reference implementation of reverse_diag_adjoint_scan."""
    if grad_states.ndim != 3:
        raise ValueError(
            "grad_states must have shape (B, T, D), got "
            f"{tuple(grad_states.shape)}."
        )

    if jacobian_diag.shape != grad_states.shape:
        raise ValueError(
            "jacobian_diag and grad_states must have the same shape, got "
            f"{tuple(jacobian_diag.shape)} and {tuple(grad_states.shape)}."
        )

    _, seq_len, _ = grad_states.shape
    lambdas = torch.empty_like(grad_states)

    if seq_len == 0:
        return lambdas

    lambdas[:, -1, :] = grad_states[:, -1, :]

    for time_idx in range(seq_len - 2, -1, -1):
        lambdas[:, time_idx, :] = (
            grad_states[:, time_idx, :]
            + jacobian_diag[:, time_idx + 1, :] * lambdas[:, time_idx + 1, :]
        )

    return lambdas
