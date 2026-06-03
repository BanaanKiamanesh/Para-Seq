from __future__ import annotations

import torch
from torch.func import vmap

from src.algos.DEER import merit_fxn_batched
from src.utils.AccelScan import diag_mat_scan_accel_batched as _diag_mat_scan_accel_batched


def _dtype_default_tol(dtype: torch.dtype) -> float:
    if dtype in (torch.float16, torch.bfloat16, torch.float32):
        return 1e-4
    if dtype == torch.float64:
        return 1e-7
    return 1e-7


def _effective_tol(dtype: torch.dtype, tol, strict_tol: bool = False) -> float:
    if tol is None:
        return _dtype_default_tol(dtype)

    tol = float(tol)

    if strict_tol:
        return tol

    return max(tol, _dtype_default_tol(dtype))


def _batched_recurrence_eval(
    f,
    previous_states: torch.Tensor,
    drivers: torch.Tensor,
) -> torch.Tensor:
    if previous_states.ndim != 3:
        raise ValueError(
            "previous_states must have shape (B, T, D), got "
            f"{tuple(previous_states.shape)}."
        )

    if drivers.ndim != 3:
        raise ValueError(
            "drivers must have shape (B, T, input_dim), got "
            f"{tuple(drivers.shape)}."
        )

    if previous_states.shape[:2] != drivers.shape[:2]:
        raise ValueError(
            "previous_states and drivers must share batch/time dimensions, got "
            f"{tuple(previous_states.shape)} and {tuple(drivers.shape)}."
        )

    batch_size, seq_len, state_dim = previous_states.shape
    input_dim = drivers.shape[-1]

    flat_prev = previous_states.reshape(batch_size * seq_len, state_dim)
    flat_drivers = drivers.reshape(batch_size * seq_len, input_dim)

    flat_predicted = vmap(f)(flat_prev, flat_drivers)

    return flat_predicted.reshape(batch_size, seq_len, state_dim)


def _fixed_point_affine_scan_accel(
    A: torch.Tensor,
    b: torch.Tensor,
    accel_scan_fn,
) -> torch.Tensor:
    _, out = _diag_mat_scan_accel_batched(
        A=A,
        b=b,
        accel_scan_fn=accel_scan_fn,
    )
    return out


def fixed_point_step_batched(
    f,
    initial_state: torch.Tensor,
    states: torch.Tensor,
    drivers: torch.Tensor,
    *,
    method: str,
    clip_value=None,
    scan_backend: str = "torch",
    accel_scan_fn=None,
) -> torch.Tensor:
    """One batched Jacobi or Picard step.

    Jacobi:
        h_t^{new} = f(h_{t-1}^{old}, x_t)

    Picard:
        h_t^{new}
        =
        h_{t-1}^{new}
        +
        f(h_{t-1}^{old}, x_t)
        -
        h_{t-1}^{old}

    For scan_backend="accel_scan", both are represented as diagonal affine
    scans:

        y_t = A_t * y_{t-1} + b_t.

    Jacobi uses A_t = 0 and b_t = predicted_t.
    Picard uses A_t = 1, b_0 = initial_state + delta_0, and
    b_t = delta_t for t >= 1.
    """
    if method not in ("jacobi", "picard"):
        raise ValueError("method must be 'jacobi' or 'picard'.")

    if scan_backend not in ("torch", "accel_scan"):
        raise ValueError("scan_backend must be 'torch' or 'accel_scan'.")

    if initial_state.ndim != 2:
        raise ValueError(
            "initial_state must have shape (B, D), got "
            f"{tuple(initial_state.shape)}."
        )

    if states.ndim != 3:
        raise ValueError(
            "states must have shape (B, T, D), got "
            f"{tuple(states.shape)}."
        )

    if drivers.ndim != 3:
        raise ValueError(
            "drivers must have shape (B, T, input_dim), got "
            f"{tuple(drivers.shape)}."
        )

    batch_size, seq_len, state_dim = states.shape

    if initial_state.shape != (batch_size, state_dim):
        raise ValueError(
            "initial_state must have shape (B, D), got "
            f"{tuple(initial_state.shape)} for states shape {tuple(states.shape)}."
        )

    if drivers.shape[:2] != (batch_size, seq_len):
        raise ValueError(
            "drivers must share batch/time dimensions with states, got "
            f"states={tuple(states.shape)} and drivers={tuple(drivers.shape)}."
        )

    if seq_len == 0:
        return states.clone()

    previous_states = torch.cat(
        [initial_state[:, None, :], states[:, :-1, :]],
        dim=1,
    )

    predicted_states = _batched_recurrence_eval(
        f=f,
        previous_states=previous_states,
        drivers=drivers,
    )

    if method == "jacobi":
        if scan_backend == "torch":
            new_states = predicted_states
        else:
            A_scan = torch.zeros_like(predicted_states)
            b_scan = predicted_states
            new_states = _fixed_point_affine_scan_accel(
                A=A_scan,
                b=b_scan,
                accel_scan_fn=accel_scan_fn,
            )

    else:
        deltas = predicted_states - previous_states

        if scan_backend == "torch":
            new_states = initial_state[:, None, :] + torch.cumsum(deltas, dim=1)
        else:
            A_scan = torch.ones_like(deltas)
            b_scan = deltas.clone()
            b_scan[:, 0, :] = b_scan[:, 0, :] + initial_state

            new_states = _fixed_point_affine_scan_accel(
                A=A_scan,
                b=b_scan,
                accel_scan_fn=accel_scan_fn,
            )

    if clip_value is not None:
        new_states = torch.clamp(new_states, -clip_value, clip_value)
        new_states = torch.nan_to_num(new_states)

    return new_states


def fixed_point_alg_batched(
    f,
    initial_state: torch.Tensor,
    states_guess: torch.Tensor,
    drivers: torch.Tensor,
    *,
    method: str,
    num_iters: int = 20,
    tol=None,
    clip_value=None,
    return_trace: bool = False,
    strict_tol: bool = False,
    stopping_criterion: str = "merit",
    scan_backend: str = "torch",
    accel_scan_fn=None,
):
    if method not in ("jacobi", "picard"):
        raise ValueError("method must be 'jacobi' or 'picard'.")

    if scan_backend not in ("torch", "accel_scan"):
        raise ValueError("scan_backend must be 'torch' or 'accel_scan'.")

    if stopping_criterion not in ("update", "merit"):
        raise ValueError("stopping_criterion must be either 'update' or 'merit'.")

    if scan_backend == "accel_scan" and accel_scan_fn is None:
        raise ValueError(
            "accel_scan_fn must be provided when scan_backend='accel_scan'."
        )

    if initial_state.ndim != 2:
        raise ValueError(
            "initial_state must have shape (B, D), got "
            f"{tuple(initial_state.shape)}."
        )

    if states_guess.ndim != 3:
        raise ValueError(
            "states_guess must have shape (B, T, D), got "
            f"{tuple(states_guess.shape)}."
        )

    if drivers.ndim != 3:
        raise ValueError(
            "drivers must have shape (B, T, input_dim), got "
            f"{tuple(drivers.shape)}."
        )

    batch_size, seq_len, state_dim = states_guess.shape

    if initial_state.shape != (batch_size, state_dim):
        raise ValueError(
            "initial_state must have shape (B, D), got "
            f"{tuple(initial_state.shape)} for states_guess shape "
            f"{tuple(states_guess.shape)}."
        )

    if drivers.shape[:2] != (batch_size, seq_len):
        raise ValueError(
            "drivers must share batch/time dimensions with states_guess, got "
            f"states_guess={tuple(states_guess.shape)} and drivers={tuple(drivers.shape)}."
        )

    states = states_guess.clone()

    effective_tol = _effective_tol(
        dtype=states.dtype,
        tol=tol,
        strict_tol=strict_tol,
    )

    trace = [states.clone()] if return_trace else None

    if seq_len == 0:
        zero = torch.zeros((), device=states.device, dtype=states.dtype)
        info = {
            "num_iters": 0,
            "initial_merit": zero.detach(),
            "final_merit": zero.detach(),
            "last_update_error": zero.detach(),
            "tol": tol,
            "effective_tol": effective_tol,
            "strict_tol": strict_tol,
            "stopping_criterion": stopping_criterion,
            "solver": method,
            "batched": True,
            "batch_size": batch_size,
            "scan_backend": scan_backend,
        }
        if return_trace:
            info["trace"] = torch.stack(trace, dim=0)
        return states, info

    initial_merit = merit_fxn_batched(f, initial_state, states, drivers)

    num_steps_done = 0
    last_update_error = torch.tensor(
        float("inf"),
        device=states.device,
        dtype=states.dtype,
    )

    for _ in range(num_iters):
        if stopping_criterion == "merit":
            current_merit = merit_fxn_batched(f, initial_state, states, drivers)
            if current_merit.item() <= effective_tol:
                break

        old_states = states

        states = fixed_point_step_batched(
            f=f,
            initial_state=initial_state,
            states=old_states,
            drivers=drivers,
            method=method,
            clip_value=clip_value,
            scan_backend=scan_backend,
            accel_scan_fn=accel_scan_fn,
        )

        last_update_error = torch.max(torch.abs(states - old_states))
        num_steps_done += 1

        if return_trace:
            trace.append(states.clone())

        if stopping_criterion == "update":
            if last_update_error.item() <= effective_tol:
                break

    final_merit = merit_fxn_batched(f, initial_state, states, drivers)

    info = {
        "num_iters": num_steps_done,
        "initial_merit": initial_merit.detach(),
        "final_merit": final_merit.detach(),
        "last_update_error": last_update_error.detach(),
        "tol": tol,
        "effective_tol": effective_tol,
        "strict_tol": strict_tol,
        "stopping_criterion": stopping_criterion,
        "solver": method,
        "batched": True,
        "batch_size": batch_size,
        "jacobian_backend": "none",
        "linearization_backend": "none",
        "scan_backend": scan_backend,
    }

    if return_trace:
        info["trace"] = torch.stack(trace, dim=0)

    return states, info


def jacobi_alg_batched(*args, **kwargs):
    return fixed_point_alg_batched(*args, method="jacobi", **kwargs)


def picard_alg_batched(*args, **kwargs):
    return fixed_point_alg_batched(*args, method="picard", **kwargs)
