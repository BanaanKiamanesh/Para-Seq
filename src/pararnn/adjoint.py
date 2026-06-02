from __future__ import annotations

import torch

from src.algos.DEER import deer_alg_batched
from src.pararnn.config import DeerNewtonConfig
from src.utils.AdjScan import reverse_diag_adjoint_scan


def functional_paragru_input_projection(
    driver: torch.Tensor,
    B: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    return torch.einsum("...i,gij->...gj", driver, B) + b


def functional_paragru_linearization_diag_from_previous(
    previous_states: torch.Tensor,
    drivers: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Functional ParaGRU recurrence values and diagonal Jacobians.

    This is the tensor-only version of the ParaGRU linearization. It is used by
    both ``ParaGRUCell`` and the custom adjoint autograd function so the gate
    equations live in one place.
    """
    if previous_states.ndim != 3:
        raise ValueError(
            "previous_states must have shape (B, T, state_dim), "
            f"got {tuple(previous_states.shape)}."
        )

    if drivers.ndim != 3:
        raise ValueError(
            "drivers must have shape (B, T, input_dim), "
            f"got {tuple(drivers.shape)}."
        )

    if previous_states.shape[:2] != drivers.shape[:2]:
        raise ValueError(
            "previous_states and drivers must share batch/time dimensions, "
            f"got {tuple(previous_states.shape)} and {tuple(drivers.shape)}."
        )

    h_prev = previous_states

    Bx_plus_b = functional_paragru_input_projection(drivers, B, b)

    z_pre = A[0] * h_prev + Bx_plus_b[..., 0, :]
    r_pre = A[1] * h_prev + Bx_plus_b[..., 1, :]

    z = torch.sigmoid(z_pre)
    r = torch.sigmoid(r_pre)

    c_pre = A[2] * (h_prev * r) + Bx_plus_b[..., 2, :]
    c = torch.tanh(c_pre)

    predicted_states = z * c + (1.0 - z) * h_prev

    dz_dpre = z * (1.0 - z)
    dr_dpre = r * (1.0 - r)
    dc_dpre = 1.0 - c * c

    dz_dh = A[0] * dz_dpre
    dr_dh = A[1] * dr_dpre

    dcpre_dh = A[2] * (r + h_prev * dr_dh)
    dc_dh = dc_dpre * dcpre_dh

    jacobian_diag = (
        (1.0 - z)
        + (c - h_prev) * dz_dh
        + z * dc_dh
    )

    return predicted_states, jacobian_diag


def functional_paragru_recurrence_step(
    state: torch.Tensor,
    driver: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """Functional ParaGRU step for tensors with shape (..., D) and (..., I)."""
    Bx_plus_b = functional_paragru_input_projection(driver, B, b)

    z_pre = A[0] * state + Bx_plus_b[..., 0, :]
    r_pre = A[1] * state + Bx_plus_b[..., 1, :]

    z = torch.sigmoid(z_pre)
    r = torch.sigmoid(r_pre)

    c_pre = A[2] * (state * r) + Bx_plus_b[..., 2, :]
    c = torch.tanh(c_pre)

    return z * c + (1.0 - z) * state


def functional_paragru_initial_guess(
    drivers: torch.Tensor,
    state_dim: int,
    guess_type: str,
    A: torch.Tensor,
    B: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    if guess_type == "zero":
        return torch.zeros(
            drivers.shape[0],
            drivers.shape[1],
            state_dim,
            device=drivers.device,
            dtype=drivers.dtype,
        )

    if guess_type == "f0":
        zero_states = torch.zeros(
            drivers.shape[0],
            drivers.shape[1],
            state_dim,
            device=drivers.device,
            dtype=drivers.dtype,
        )

        predicted_states, _ = functional_paragru_linearization_diag_from_previous(
            previous_states=zero_states,
            drivers=drivers,
            A=A,
            B=B,
            b=b,
        )

        return predicted_states

    raise ValueError(f"Unknown initial guess type: {guess_type!r}.")


def _zero_if_none(grad: torch.Tensor | None, reference: torch.Tensor) -> torch.Tensor:
    if grad is None:
        return torch.zeros_like(reference)
    return grad


class _ParaGRUAdjointDEERFunction(torch.autograd.Function):
    """Explicit quasi-DEER forward with ParaGRU adjoint backward.

    The forward pass computes the hidden trajectory by DEER but does not build a
    differentiable graph through the Newton iterations. The backward pass uses
    the reverse-time adjoint recurrence

        lambda_t = grad_t + J_{t+1}^T lambda_{t+1},

    and then accumulates local gradients from

        sum_t lambda_t^T f_theta(h_{t-1}, x_t).

    Because ParaGRU has diagonal recurrent Jacobians, the adjoint recurrence is
    also a diagonal affine scan.
    """

    @staticmethod
    def forward(
        ctx,
        drivers: torch.Tensor,
        initial_state: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        b: torch.Tensor,
        cfg: DeerNewtonConfig,
        accel_scan_fn,
    ):
        state_dim = A.shape[1]

        def f(state, driver):
            return functional_paragru_recurrence_step(
                state=state,
                driver=driver,
                A=A,
                B=B,
                b=b,
            )

        def linearization_fn(previous_states, current_drivers):
            return functional_paragru_linearization_diag_from_previous(
                previous_states=previous_states,
                drivers=current_drivers,
                A=A,
                B=B,
                b=b,
            )

        with torch.no_grad():
            states_guess = functional_paragru_initial_guess(
                drivers=drivers,
                state_dim=state_dim,
                guess_type=cfg.initial_guess,
                A=A,
                B=B,
                b=b,
            )

            states, _ = deer_alg_batched(
                f=f,
                initial_state=initial_state,
                states_guess=states_guess,
                drivers=drivers,
                num_iters=cfg.num_iters,
                tol=cfg.tol,
                quasi=True,
                damping=cfg.damping,
                clip_value=cfg.clip_value,
                return_trace=False,
                scan_backend=cfg.scan_backend,
                accel_scan_fn=accel_scan_fn,
                strict_tol=cfg.strict_tol,
                stopping_criterion=cfg.stopping_criterion,
                linearization_fn=linearization_fn,
            )

        ctx.save_for_backward(drivers, initial_state, states, A, B, b)
        ctx.scan_backend = cfg.scan_backend
        ctx.accel_scan_fn = accel_scan_fn

        return states

    @staticmethod
    def backward(ctx, grad_states: torch.Tensor):
        drivers, initial_state, states, A, B, b = ctx.saved_tensors

        with torch.enable_grad():
            drivers_req = drivers.detach().requires_grad_(True)
            initial_state_req = initial_state.detach().requires_grad_(True)
            A_req = A.detach().requires_grad_(True)
            B_req = B.detach().requires_grad_(True)
            b_req = b.detach().requires_grad_(True)

            states_const = states.detach()
            previous_states = torch.cat(
                [initial_state_req[:, None, :], states_const[:, :-1, :]],
                dim=1,
            )

            predicted_states, jacobian_diag = functional_paragru_linearization_diag_from_previous(
                previous_states=previous_states,
                drivers=drivers_req,
                A=A_req,
                B=B_req,
                b=b_req,
            )

            with torch.no_grad():
                adjoints = reverse_diag_adjoint_scan(
                    jacobian_diag=jacobian_diag.detach(),
                    grad_states=grad_states.detach(),
                    scan_backend=ctx.scan_backend,
                    accel_scan_fn=ctx.accel_scan_fn,
                )

            grads = torch.autograd.grad(
                outputs=predicted_states,
                inputs=(drivers_req, initial_state_req, A_req, B_req, b_req),
                grad_outputs=adjoints,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )

        grad_drivers, grad_initial_state, grad_A, grad_B, grad_b = grads

        return (
            _zero_if_none(grad_drivers, drivers),
            _zero_if_none(grad_initial_state, initial_state),
            _zero_if_none(grad_A, A),
            _zero_if_none(grad_B, B),
            _zero_if_none(grad_b, b),
            None,
            None,
        )


def paragru_adjoint_deer_forward(
    drivers: torch.Tensor,
    initial_state: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    b: torch.Tensor,
    cfg: DeerNewtonConfig,
    accel_scan_fn,
) -> torch.Tensor:
    return _ParaGRUAdjointDEERFunction.apply(
        drivers,
        initial_state,
        A,
        B,
        b,
        cfg,
        accel_scan_fn,
    )
