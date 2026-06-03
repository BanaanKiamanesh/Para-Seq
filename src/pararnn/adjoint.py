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


# === ParaLSTM block-2 adjoint DEER extension ===

from src.utils.BlockScan import block2_mat_scan


def _split_paralstm_flat_state(
    state: torch.Tensor,
    hidden_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    c = state[..., :hidden_size]
    h = state[..., hidden_size:]
    return c, h


def _pack_paralstm_flat_state(c: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    return torch.cat([c, h], dim=-1)


def _paralstm_flat_to_blocks(
    state: torch.Tensor,
    hidden_size: int,
) -> torch.Tensor:
    c, h = _split_paralstm_flat_state(state, hidden_size)
    return torch.stack([c, h], dim=-1)


def _paralstm_blocks_to_flat(block_state: torch.Tensor) -> torch.Tensor:
    if block_state.shape[-1] != 2:
        raise ValueError(
            "block_state must end with dimension 2 containing (c, h), got "
            f"{tuple(block_state.shape)}."
        )

    c = block_state[..., 0]
    h = block_state[..., 1]
    return _pack_paralstm_flat_state(c, h)


def functional_paralstm_input_projection(
    driver: torch.Tensor,
    B: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    return torch.einsum("...i,gij->...gj", driver, B) + b


def functional_paralstm_linearization_blocks_from_previous(
    previous_states: torch.Tensor,
    drivers: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if previous_states.ndim != 4 or previous_states.shape[-1] != 2:
        raise ValueError(
            "previous_states must have shape (B, T, H, 2), got "
            f"{tuple(previous_states.shape)}."
        )

    if drivers.ndim != 3:
        raise ValueError(
            "drivers must have shape (B, T, input_dim), got "
            f"{tuple(drivers.shape)}."
        )

    if previous_states.shape[:2] != drivers.shape[:2]:
        raise ValueError(
            "previous_states and drivers must share batch/time dimensions, "
            f"got {tuple(previous_states.shape)} and {tuple(drivers.shape)}."
        )

    c_prev = previous_states[..., 0]
    h_prev = previous_states[..., 1]

    Bx_plus_b = functional_paralstm_input_projection(drivers, B, b)

    f_pre = A[0] * h_prev + Bx_plus_b[..., 0, :] + C[0] * c_prev
    z_pre = A[1] * h_prev + Bx_plus_b[..., 1, :]

    f = torch.sigmoid(f_pre)
    z = torch.tanh(z_pre)

    c_next = f * c_prev + (1.0 - f) * z

    o_pre = A[2] * h_prev + Bx_plus_b[..., 2, :] + C[1] * c_next
    o = torch.sigmoid(o_pre)

    tanh_c = torch.tanh(c_next)
    h_next = o * tanh_c

    df_dpre = f * (1.0 - f)
    dz_dpre = 1.0 - z * z
    do_dpre = o * (1.0 - o)
    dtanhc_dc = 1.0 - tanh_c * tanh_c

    df_dc_prev = C[0] * df_dpre
    df_dh_prev = A[0] * df_dpre
    dz_dh_prev = A[1] * dz_dpre

    dc_dc = f + (c_prev - z) * df_dc_prev
    dc_dh = (c_prev - z) * df_dh_prev + (1.0 - f) * dz_dh_prev

    do_dc = do_dpre * (C[1] * dc_dc)
    do_dh = do_dpre * (A[2] + C[1] * dc_dh)

    dh_dc = do_dc * tanh_c + o * dtanhc_dc * dc_dc
    dh_dh = do_dh * tanh_c + o * dtanhc_dc * dc_dh

    predicted_states = torch.stack([c_next, h_next], dim=-1)

    row0 = torch.stack([dc_dc, dc_dh], dim=-1)
    row1 = torch.stack([dh_dc, dh_dh], dim=-1)
    jacobian_blocks = torch.stack([row0, row1], dim=-2)

    return predicted_states, jacobian_blocks


def functional_paralstm_initial_guess_blocks(
    drivers: torch.Tensor,
    hidden_size: int,
    guess_type: str,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    zeros = torch.zeros(
        drivers.shape[0],
        drivers.shape[1],
        hidden_size,
        2,
        device=drivers.device,
        dtype=drivers.dtype,
    )

    if guess_type == "zero":
        return zeros

    if guess_type == "f0":
        predicted, _ = functional_paralstm_linearization_blocks_from_previous(
            previous_states=zeros,
            drivers=drivers,
            A=A,
            B=B,
            C=C,
            b=b,
        )
        return predicted

    raise ValueError(f"Unknown initial guess type: {guess_type!r}.")


def _paralstm_block_merit(
    initial_blocks: torch.Tensor,
    states: torch.Tensor,
    drivers: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    previous_states = torch.cat(
        [initial_blocks[:, None, :, :], states[:, :-1, :, :]],
        dim=1,
    )
    predicted, _ = functional_paralstm_linearization_blocks_from_previous(
        previous_states=previous_states,
        drivers=drivers,
        A=A,
        B=B,
        C=C,
        b=b,
    )
    residual = states - predicted
    return 0.5 * torch.sum(residual * residual)


def _paralstm_dtype_default_tol(dtype: torch.dtype) -> float:
    if dtype in (torch.float16, torch.bfloat16, torch.float32):
        return 1e-4
    if dtype == torch.float64:
        return 1e-7
    return 1e-7


def _paralstm_effective_tol(
    dtype: torch.dtype,
    tol: float | None,
    strict_tol: bool,
) -> float:
    if tol is None:
        return _paralstm_dtype_default_tol(dtype)

    tol = float(tol)

    if strict_tol:
        return tol

    return max(tol, _paralstm_dtype_default_tol(dtype))


def _paralstm_block_deer_forward_no_grad(
    drivers: torch.Tensor,
    initial_state: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    b: torch.Tensor,
    cfg: DeerNewtonConfig,
) -> torch.Tensor:
    hidden_size = A.shape[1]
    initial_blocks = _paralstm_flat_to_blocks(initial_state, hidden_size)

    states = functional_paralstm_initial_guess_blocks(
        drivers=drivers,
        hidden_size=hidden_size,
        guess_type=cfg.initial_guess,
        A=A,
        B=B,
        C=C,
        b=b,
    )

    effective_tol = _paralstm_effective_tol(
        dtype=states.dtype,
        tol=cfg.tol,
        strict_tol=cfg.strict_tol,
    )

    for _ in range(cfg.num_iters):
        if cfg.stopping_criterion == "merit":
            current_merit = _paralstm_block_merit(
                initial_blocks=initial_blocks,
                states=states,
                drivers=drivers,
                A=A,
                B=B,
                C=C,
                b=b,
            )
            if current_merit.item() <= effective_tol:
                break

        old_states = states

        previous_states = torch.cat(
            [initial_blocks[:, None, :, :], old_states[:, :-1, :, :]],
            dim=1,
        )

        predicted, jacobian_blocks = functional_paralstm_linearization_blocks_from_previous(
            previous_states=previous_states,
            drivers=drivers,
            A=A,
            B=B,
            C=C,
            b=b,
        )

        b_terms = predicted - (
            jacobian_blocks @ previous_states.unsqueeze(-1)
        ).squeeze(-1)

        A0 = torch.zeros_like(jacobian_blocks[:, 0, :, :, :])
        b0 = predicted[:, 0, :, :]

        if predicted.shape[1] > 1:
            A_scan = torch.cat(
                [A0[:, None, :, :, :], jacobian_blocks[:, 1:, :, :, :]],
                dim=1,
            )
            b_scan = torch.cat(
                [b0[:, None, :, :], b_terms[:, 1:, :, :]],
                dim=1,
            )
        else:
            A_scan = A0[:, None, :, :, :]
            b_scan = b0[:, None, :, :]

        _, states = block2_mat_scan(A_scan, b_scan, dim=1)

        if cfg.clip_value is not None:
            states = torch.clamp(states, -cfg.clip_value, cfg.clip_value)
            states = torch.nan_to_num(states)

        if cfg.stopping_criterion == "update":
            update_error = torch.max(torch.abs(states - old_states))
            if update_error.item() <= effective_tol:
                break

    return _paralstm_blocks_to_flat(states)


def reverse_block2_adjoint_scan(
    jacobian_blocks: torch.Tensor,
    grad_states: torch.Tensor,
) -> torch.Tensor:
    if jacobian_blocks.ndim != 5 or jacobian_blocks.shape[-2:] != (2, 2):
        raise ValueError(
            "jacobian_blocks must have shape (B, T, H, 2, 2), got "
            f"{tuple(jacobian_blocks.shape)}."
        )

    if grad_states.ndim != 4 or grad_states.shape[-1] != 2:
        raise ValueError(
            "grad_states must have shape (B, T, H, 2), got "
            f"{tuple(grad_states.shape)}."
        )

    if jacobian_blocks.shape[:-1] != grad_states.shape:
        raise ValueError(
            "jacobian_blocks and grad_states have incompatible shapes, got "
            f"{tuple(jacobian_blocks.shape)} and {tuple(grad_states.shape)}."
        )

    _, seq_len, _, _ = grad_states.shape

    if seq_len == 0:
        return grad_states.clone()

    A_rev = torch.zeros_like(jacobian_blocks)
    b_rev = torch.flip(grad_states, dims=[1])

    if seq_len > 1:
        A_rev[:, 1:, :, :, :] = torch.flip(
            jacobian_blocks[:, 1:, :, :, :].transpose(-1, -2),
            dims=[1],
        )

    _, lambda_rev = block2_mat_scan(A_rev, b_rev, dim=1)

    return torch.flip(lambda_rev, dims=[1])


class _ParaLSTMBlockAdjointDEERFunction(torch.autograd.Function):
    """Block-2 ParaLSTM DEER forward with explicit adjoint backward."""

    @staticmethod
    def forward(
        ctx,
        drivers: torch.Tensor,
        initial_state: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        b: torch.Tensor,
        cfg: DeerNewtonConfig,
    ):
        with torch.no_grad():
            states = _paralstm_block_deer_forward_no_grad(
                drivers=drivers,
                initial_state=initial_state,
                A=A,
                B=B,
                C=C,
                b=b,
                cfg=cfg,
            )

        ctx.save_for_backward(drivers, initial_state, states, A, B, C, b)

        return states

    @staticmethod
    def backward(ctx, grad_states: torch.Tensor):
        drivers, initial_state, states, A, B, C, b = ctx.saved_tensors
        hidden_size = A.shape[1]

        with torch.enable_grad():
            drivers_req = drivers.detach().requires_grad_(True)
            initial_state_req = initial_state.detach().requires_grad_(True)
            A_req = A.detach().requires_grad_(True)
            B_req = B.detach().requires_grad_(True)
            C_req = C.detach().requires_grad_(True)
            b_req = b.detach().requires_grad_(True)

            states_const = states.detach()
            previous_states_flat = torch.cat(
                [initial_state_req[:, None, :], states_const[:, :-1, :]],
                dim=1,
            )
            previous_states_blocks = _paralstm_flat_to_blocks(
                previous_states_flat,
                hidden_size,
            )

            predicted_blocks, jacobian_blocks = functional_paralstm_linearization_blocks_from_previous(
                previous_states=previous_states_blocks,
                drivers=drivers_req,
                A=A_req,
                B=B_req,
                C=C_req,
                b=b_req,
            )
            predicted_flat = _paralstm_blocks_to_flat(predicted_blocks)

            with torch.no_grad():
                grad_blocks = _paralstm_flat_to_blocks(
                    grad_states.detach(),
                    hidden_size,
                )
                adjoint_blocks = reverse_block2_adjoint_scan(
                    jacobian_blocks=jacobian_blocks.detach(),
                    grad_states=grad_blocks,
                )
                adjoints = _paralstm_blocks_to_flat(adjoint_blocks)

            grads = torch.autograd.grad(
                outputs=predicted_flat,
                inputs=(drivers_req, initial_state_req, A_req, B_req, C_req, b_req),
                grad_outputs=adjoints,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )

        grad_drivers, grad_initial_state, grad_A, grad_B, grad_C, grad_b = grads

        return (
            _zero_if_none(grad_drivers, drivers),
            _zero_if_none(grad_initial_state, initial_state),
            _zero_if_none(grad_A, A),
            _zero_if_none(grad_B, B),
            _zero_if_none(grad_C, C),
            _zero_if_none(grad_b, b),
            None,
        )


def paralstm_block_adjoint_deer_forward(
    drivers: torch.Tensor,
    initial_state: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    b: torch.Tensor,
    cfg: DeerNewtonConfig,
) -> torch.Tensor:
    return _ParaLSTMBlockAdjointDEERFunction.apply(
        drivers,
        initial_state,
        A,
        B,
        C,
        b,
        cfg,
    )

# === End ParaLSTM block-2 adjoint DEER extension ===
