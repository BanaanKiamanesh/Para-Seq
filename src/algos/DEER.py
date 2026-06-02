import torch
from torch.func import vmap, jacrev

from src.utils.AssScan import full_mat_scan, diag_mat_scan


_ACCEL_SCAN_MIN_LEN = 32
_ACCEL_SCAN_MAX_LEN = 65536


def _next_power_of_two(n):
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _dtype_default_tol(dtype):
    if dtype in (torch.float16, torch.bfloat16, torch.float32):
        return 1e-4
    if dtype == torch.float64:
        return 1e-7
    return 1e-7


def _effective_tol(dtype, tol, strict_tol=False):
    if tol is None:
        return _dtype_default_tol(dtype)

    tol = float(tol)

    if strict_tol:
        return tol

    return max(tol, _dtype_default_tol(dtype))


def _validate_accel_scan_inputs(A, b, expected_ndim):
    if A.ndim != expected_ndim or b.ndim != expected_ndim:
        if expected_ndim == 2:
            expected_shape = "(T, D)"
        elif expected_ndim == 3:
            expected_shape = "(B, T, D)"
        else:
            expected_shape = f"{expected_ndim}-dimensional tensors"

        raise ValueError(f"Expected A and b with shape {expected_shape}.")

    if A.shape != b.shape:
        raise ValueError(
            f"A and b must have the same shape, got {A.shape} and {b.shape}."
        )

    if A.device.type != "cuda":
        raise ValueError("accelerated_scan backend requires CUDA tensors.")


def _run_accel_scan_chunk_batched(A_chunk, b_chunk, accel_scan_fn):
    """Run accelerated_scan on one batched chunk.

    Args:
        A_chunk:
            Tensor with shape ``(B, Tc, D)``.

        b_chunk:
            Tensor with shape ``(B, Tc, D)``.

        accel_scan_fn:
            Accelerated scan function. Its expected layout is
            ``(B, D, T)`` for both gate and token.

    Returns:
        Tensor with shape ``(B, Tc, D)``, assuming zero incoming state at the
        beginning of this chunk.
    """
    if accel_scan_fn is None:
        raise ValueError(
            "accel_scan_fn must be provided when scan_backend='accel_scan'."
        )

    if A_chunk.ndim != 3 or b_chunk.ndim != 3:
        raise ValueError("Expected A_chunk and b_chunk with shape (B, T, D).")

    if A_chunk.shape != b_chunk.shape:
        raise ValueError(
            "A_chunk and b_chunk must have the same shape, got "
            f"{A_chunk.shape} and {b_chunk.shape}."
        )

    batch_size, original_len, state_dim = A_chunk.shape
    padded_len = _next_power_of_two(max(original_len, _ACCEL_SCAN_MIN_LEN))

    if padded_len > _ACCEL_SCAN_MAX_LEN:
        raise ValueError(
            f"accelerated_scan chunk length must be <= {_ACCEL_SCAN_MAX_LEN}, "
            f"but got padded_len={padded_len}."
        )

    if padded_len != original_len:
        pad_len = padded_len - original_len

        A_pad = torch.ones(
            batch_size,
            pad_len,
            state_dim,
            device=A_chunk.device,
            dtype=A_chunk.dtype,
        )
        b_pad = torch.zeros(
            batch_size,
            pad_len,
            state_dim,
            device=b_chunk.device,
            dtype=b_chunk.dtype,
        )

        A_chunk = torch.cat([A_chunk, A_pad], dim=1)
        b_chunk = torch.cat([b_chunk, b_pad], dim=1)

    gate = A_chunk.transpose(1, 2).contiguous()
    token = b_chunk.transpose(1, 2).contiguous()

    scanned = accel_scan_fn(gate, token)
    b_prefix = scanned.transpose(1, 2).contiguous()

    return b_prefix[:, :original_len, :]


def _diag_mat_scan_accel_batched(A, b, accel_scan_fn):
    """Batched accelerated-scan backend for diagonal affine recurrences.

    Solves the batched recurrence

        h_{b,t} = A_{b,t} * h_{b,t-1} + b_{b,t}

    for every batch item and hidden coordinate in one accelerated_scan call per
    chunk. Unlike the older implementation, this function does not loop over
    the batch dimension.

    Args:
        A:
            Tensor with shape ``(B, T, D)``.

        b:
            Tensor with shape ``(B, T, D)``.

        accel_scan_fn:
            accelerated_scan backend function. The function is called with
            tensors of shape ``(B, D, T_padded)``.

    Returns:
        A_prefix:
            Prefix-composed diagonal transition entries with shape ``(B, T, D)``.

        b_prefix:
            Prefix-composed bias vectors with shape ``(B, T, D)``.
    """
    _validate_accel_scan_inputs(A, b, expected_ndim=3)

    batch_size, seq_len, state_dim = A.shape

    if seq_len == 0:
        return A.clone(), b.clone()

    b_outputs = []
    A_outputs = []

    state_carry = torch.zeros(
        batch_size,
        state_dim,
        device=A.device,
        dtype=A.dtype,
    )
    A_carry = torch.ones(
        batch_size,
        state_dim,
        device=A.device,
        dtype=A.dtype,
    )

    start = 0

    while start < seq_len:
        end = min(start + _ACCEL_SCAN_MAX_LEN, seq_len)

        A_chunk = A[:, start:end, :].contiguous()
        b_chunk = b[:, start:end, :].contiguous()

        b_prefix_zero = _run_accel_scan_chunk_batched(
            A_chunk=A_chunk,
            b_chunk=b_chunk,
            accel_scan_fn=accel_scan_fn,
        )

        A_prefix_local = torch.cumprod(A_chunk, dim=1)

        b_prefix = A_prefix_local * state_carry[:, None, :] + b_prefix_zero
        A_prefix = A_prefix_local * A_carry[:, None, :]

        state_carry = b_prefix[:, -1, :]
        A_carry = A_prefix[:, -1, :]

        b_outputs.append(b_prefix)
        A_outputs.append(A_prefix)

        start = end

    return torch.cat(A_outputs, dim=1), torch.cat(b_outputs, dim=1)


def _run_accel_scan_chunk(A_chunk, b_chunk, accel_scan_fn):
    """Backward-compatible unbatched accelerated_scan chunk helper."""
    if A_chunk.ndim != 2 or b_chunk.ndim != 2:
        raise ValueError("Expected A_chunk and b_chunk with shape (T, D).")

    return _run_accel_scan_chunk_batched(
        A_chunk=A_chunk.unsqueeze(0),
        b_chunk=b_chunk.unsqueeze(0),
        accel_scan_fn=accel_scan_fn,
    ).squeeze(0)


def _diag_mat_scan_accel(A, b, accel_scan_fn):
    """Unbatched accelerated-scan backend for diagonal affine recurrences."""
    _validate_accel_scan_inputs(A, b, expected_ndim=2)

    A_prefix, b_prefix = _diag_mat_scan_accel_batched(
        A=A.unsqueeze(0),
        b=b.unsqueeze(0),
        accel_scan_fn=accel_scan_fn,
    )

    return A_prefix.squeeze(0), b_prefix.squeeze(0)


def sequential_rollout(f, initial_state, drivers):
    states = []
    state = initial_state

    for t in range(drivers.shape[0]):
        state = f(state, drivers[t])
        states.append(state)

    return torch.stack(states, dim=0)


def _batched_recurrence_eval(f, previous_states, drivers):
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


def get_residual(f, initial_state, states, drivers):
    prev_states = torch.cat([initial_state[None, :], states[:-1]], dim=0)
    predicted_states = vmap(f)(prev_states, drivers)

    return states - predicted_states


def get_residual_batched(f, initial_state, states, drivers):
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

    if initial_state.shape != (states.shape[0], states.shape[2]):
        raise ValueError(
            "initial_state must have shape (B, D), got "
            f"{tuple(initial_state.shape)} for states shape {tuple(states.shape)}."
        )

    if states.shape[:2] != drivers.shape[:2]:
        raise ValueError(
            "states and drivers must share batch/time dimensions, got "
            f"{tuple(states.shape)} and {tuple(drivers.shape)}."
        )

    prev_states = torch.cat(
        [initial_state[:, None, :], states[:, :-1, :]],
        dim=1,
    )

    predicted_states = _batched_recurrence_eval(
        f=f,
        previous_states=prev_states,
        drivers=drivers,
    )

    return states - predicted_states


def merit_fxn(f, initial_state, states, drivers):
    residual = get_residual(f, initial_state, states, drivers)
    return 0.5 * torch.sum(residual * residual)


def merit_fxn_batched(f, initial_state, states, drivers):
    residual = get_residual_batched(f, initial_state, states, drivers)
    return 0.5 * torch.sum(residual * residual)


def _validate_linearization_shapes(predicted, jacobian, previous_states, drivers):
    if predicted.shape != previous_states.shape:
        raise ValueError(
            "linearization_fn must return predicted states with the same shape "
            "as previous_states. Got predicted="
            f"{tuple(predicted.shape)} and previous_states={tuple(previous_states.shape)}."
        )

    if jacobian.shape[:-1] != previous_states.shape[:-1]:
        raise ValueError(
            "linearization_fn returned a Jacobian with incompatible batch/time "
            f"shape {tuple(jacobian.shape)} for previous_states "
            f"{tuple(previous_states.shape)} and drivers {tuple(drivers.shape)}."
        )


def deer_step(
    f,
    initial_state,
    states,
    drivers,
    quasi=False,
    damping=0.0,
    clip_value=None,
    scan_backend="torch",
    accel_scan_fn=None,
    jacobian_fn=None,
    linearization_fn=None,
):
    T, _ = states.shape

    if T == 1:
        if linearization_fn is None:
            new_states = f(initial_state, drivers[0])[None, :]
        else:
            predicted_all, _ = linearization_fn(
                initial_state[None, :],
                drivers[:1],
            )
            new_states = predicted_all

        if clip_value is not None:
            new_states = torch.clamp(new_states, -clip_value, clip_value)
            new_states = torch.nan_to_num(new_states)

        return new_states

    previous_all = torch.cat([initial_state[None, :], states[:-1]], dim=0)

    if linearization_fn is not None:
        predicted_all, J_all = linearization_fn(previous_all, drivers)
        _validate_linearization_shapes(
            predicted=predicted_all,
            jacobian=J_all,
            previous_states=previous_all,
            drivers=drivers,
        )
        b0 = predicted_all[0]
        fs = predicted_all[1:]
        Jfs = J_all[1:]
    else:
        old_prev_states = states[:-1]
        current_drivers = drivers[1:]
        fs = vmap(f)(old_prev_states, current_drivers)

        if jacobian_fn is None:
            jac_f = jacrev(f, argnums=0)
            Jfs = vmap(jac_f)(old_prev_states, current_drivers)
        else:
            Jfs = jacobian_fn(old_prev_states, current_drivers)

        b0 = f(initial_state, drivers[0])

    old_prev_states = states[:-1]

    if quasi:
        if Jfs.ndim == 2:
            As = Jfs
        elif Jfs.ndim == 3:
            As = torch.diagonal(Jfs, dim1=-2, dim2=-1)
        else:
            raise ValueError(
                "For quasi-DEER, jacobian_fn or linearization_fn must return "
                "shape (T-1, D) or (T-1, D, D), got "
                f"{tuple(Jfs.shape)}."
            )

        As = (1.0 - damping) * As
        bs = fs - As * old_prev_states

        A0 = torch.zeros_like(As[0])

        A = torch.cat([A0[None, :], As], dim=0)
        b = torch.cat([b0[None, :], bs], dim=0)

        if scan_backend == "torch":
            _, new_states = diag_mat_scan(A, b, dim=0)
        elif scan_backend == "accel_scan":
            _, new_states = _diag_mat_scan_accel(
                A=A,
                b=b,
                accel_scan_fn=accel_scan_fn,
            )
        else:
            raise ValueError(f"Unknown scan_backend: {scan_backend}")

    else:
        if scan_backend != "torch":
            raise ValueError("Full DEER only supports scan_backend='torch'.")

        if Jfs.ndim != 3:
            raise ValueError(
                "Full DEER requires dense Jacobians with shape (T-1, D, D). "
                f"Got {tuple(Jfs.shape)}."
            )

        As = (1.0 - damping) * Jfs
        bs = fs - torch.einsum("tij,tj->ti", As, old_prev_states)

        A0 = torch.zeros_like(As[0])

        A = torch.cat([A0[None, :, :], As], dim=0)
        b = torch.cat([b0[None, :], bs], dim=0)

        _, new_states = full_mat_scan(A, b, dim=0)

    if clip_value is not None:
        new_states = torch.clamp(new_states, -clip_value, clip_value)
        new_states = torch.nan_to_num(new_states)

    return new_states


def deer_step_batched(
    f,
    initial_state,
    states,
    drivers,
    quasi=False,
    damping=0.0,
    clip_value=None,
    scan_backend="torch",
    accel_scan_fn=None,
    jacobian_fn=None,
    linearization_fn=None,
):
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

    if seq_len == 1:
        if linearization_fn is None:
            new_states = vmap(f)(initial_state, drivers[:, 0, :])[:, None, :]
        else:
            predicted_all, _ = linearization_fn(
                initial_state[:, None, :],
                drivers[:, :1, :],
            )
            new_states = predicted_all

        if clip_value is not None:
            new_states = torch.clamp(new_states, -clip_value, clip_value)
            new_states = torch.nan_to_num(new_states)

        return new_states

    previous_all = torch.cat(
        [initial_state[:, None, :], states[:, :-1, :]],
        dim=1,
    )

    if linearization_fn is not None:
        predicted_all, J_all = linearization_fn(previous_all, drivers)
        _validate_linearization_shapes(
            predicted=predicted_all,
            jacobian=J_all,
            previous_states=previous_all,
            drivers=drivers,
        )
        b0 = predicted_all[:, 0, :]
        fs = predicted_all[:, 1:, :]
        Jfs = J_all[:, 1:]
    else:
        old_prev_states = states[:, :-1, :]
        current_drivers = drivers[:, 1:, :]

        fs = _batched_recurrence_eval(
            f=f,
            previous_states=old_prev_states,
            drivers=current_drivers,
        )

        if jacobian_fn is None:
            jac_f = jacrev(f, argnums=0)

            flat_prev = old_prev_states.reshape(
                batch_size * (seq_len - 1),
                state_dim,
            )
            flat_drivers = current_drivers.reshape(
                batch_size * (seq_len - 1),
                drivers.shape[-1],
            )

            Jfs = vmap(jac_f)(flat_prev, flat_drivers).reshape(
                batch_size,
                seq_len - 1,
                state_dim,
                state_dim,
            )
        else:
            Jfs = jacobian_fn(old_prev_states, current_drivers)

        b0 = vmap(f)(initial_state, drivers[:, 0, :])

    old_prev_states = states[:, :-1, :]

    if quasi:
        if Jfs.ndim == 3:
            As = Jfs
        elif Jfs.ndim == 4:
            As = torch.diagonal(Jfs, dim1=-2, dim2=-1)
        else:
            raise ValueError(
                "For batched quasi-DEER, jacobian_fn or linearization_fn must "
                "return shape (B, T-1, D) or (B, T-1, D, D), got "
                f"{tuple(Jfs.shape)}."
            )

        As = (1.0 - damping) * As
        bs = fs - As * old_prev_states

        A0 = torch.zeros_like(As[:, 0, :])

        A = torch.cat([A0[:, None, :], As], dim=1)
        b = torch.cat([b0[:, None, :], bs], dim=1)

        if scan_backend == "torch":
            _, new_states = diag_mat_scan(A, b, dim=1)
        elif scan_backend == "accel_scan":
            _, new_states = _diag_mat_scan_accel_batched(
                A=A,
                b=b,
                accel_scan_fn=accel_scan_fn,
            )
        else:
            raise ValueError(f"Unknown scan_backend: {scan_backend}")

    else:
        if scan_backend != "torch":
            raise ValueError("Full DEER only supports scan_backend='torch'.")

        if Jfs.ndim != 4:
            raise ValueError(
                "Batched full DEER requires dense Jacobians with shape "
                f"(B, T-1, D, D). Got {tuple(Jfs.shape)}."
            )

        As = (1.0 - damping) * Jfs
        bs = fs - torch.einsum("btij,btj->bti", As, old_prev_states)

        A0 = torch.zeros_like(As[:, 0, :, :])

        A = torch.cat([A0[:, None, :, :], As], dim=1)
        b = torch.cat([b0[:, None, :], bs], dim=1)

        _, new_states = full_mat_scan(A, b, dim=1)

    if clip_value is not None:
        new_states = torch.clamp(new_states, -clip_value, clip_value)
        new_states = torch.nan_to_num(new_states)

    return new_states


def deer_alg(
    f,
    initial_state,
    states_guess,
    drivers,
    num_iters=20,
    tol=None,
    quasi=False,
    damping=0.0,
    clip_value=None,
    return_trace=False,
    scan_backend="torch",
    accel_scan_fn=None,
    strict_tol=False,
    stopping_criterion="update",
    jacobian_fn=None,
    linearization_fn=None,
):
    if stopping_criterion not in ("update", "merit"):
        raise ValueError(
            "stopping_criterion must be either 'update' or 'merit'."
        )

    states = states_guess.clone()

    effective_tol = _effective_tol(
        dtype=states.dtype,
        tol=tol,
        strict_tol=strict_tol,
    )

    trace = [states.clone()] if return_trace else None

    initial_merit = merit_fxn(f, initial_state, states, drivers)

    num_steps_done = 0
    last_update_error = torch.tensor(
        float("inf"),
        device=states.device,
        dtype=states.dtype,
    )

    for it in range(num_iters):
        if stopping_criterion == "merit":
            current_merit = merit_fxn(f, initial_state, states, drivers)

            if current_merit.item() <= effective_tol:
                break

        old_states = states

        new_states = deer_step(
            f=f,
            initial_state=initial_state,
            states=old_states,
            drivers=drivers,
            quasi=quasi,
            damping=damping,
            clip_value=clip_value,
            scan_backend=scan_backend,
            accel_scan_fn=accel_scan_fn,
            jacobian_fn=jacobian_fn,
            linearization_fn=linearization_fn,
        )

        last_update_error = torch.max(torch.abs(new_states - old_states))

        states = new_states
        num_steps_done = it + 1

        if return_trace:
            trace.append(states.clone())

        if stopping_criterion == "update":
            if last_update_error.item() <= effective_tol:
                break

    final_merit = merit_fxn(f, initial_state, states, drivers)

    has_custom_jacobian = jacobian_fn is not None or linearization_fn is not None

    info = {
        "num_iters": num_steps_done,
        "initial_merit": initial_merit.detach(),
        "final_merit": final_merit.detach(),
        "last_update_error": last_update_error.detach(),
        "tol": tol,
        "effective_tol": effective_tol,
        "strict_tol": strict_tol,
        "stopping_criterion": stopping_criterion,
        "scan_backend": scan_backend,
        "quasi": quasi,
        "batched": False,
        "jacobian_backend": "custom" if has_custom_jacobian else "autograd",
        "linearization_backend": "custom" if linearization_fn is not None else "separate",
    }

    if return_trace:
        info["trace"] = torch.stack(trace, dim=0)

    return states, info


def deer_alg_batched(
    f,
    initial_state,
    states_guess,
    drivers,
    num_iters=20,
    tol=None,
    quasi=False,
    damping=0.0,
    clip_value=None,
    return_trace=False,
    scan_backend="torch",
    accel_scan_fn=None,
    strict_tol=False,
    stopping_criterion="update",
    jacobian_fn=None,
    linearization_fn=None,
):
    if stopping_criterion not in ("update", "merit"):
        raise ValueError(
            "stopping_criterion must be either 'update' or 'merit'."
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
            f"states_guess={tuple(states_guess.shape)} and "
            f"drivers={tuple(drivers.shape)}."
        )

    states = states_guess.clone()

    effective_tol = _effective_tol(
        dtype=states.dtype,
        tol=tol,
        strict_tol=strict_tol,
    )

    trace = [states.clone()] if return_trace else None

    initial_merit = merit_fxn_batched(f, initial_state, states, drivers)

    num_steps_done = 0
    last_update_error = torch.tensor(
        float("inf"),
        device=states.device,
        dtype=states.dtype,
    )

    for it in range(num_iters):
        if stopping_criterion == "merit":
            current_merit = merit_fxn_batched(
                f,
                initial_state,
                states,
                drivers,
            )

            if current_merit.item() <= effective_tol:
                break

        old_states = states

        new_states = deer_step_batched(
            f=f,
            initial_state=initial_state,
            states=old_states,
            drivers=drivers,
            quasi=quasi,
            damping=damping,
            clip_value=clip_value,
            scan_backend=scan_backend,
            accel_scan_fn=accel_scan_fn,
            jacobian_fn=jacobian_fn,
            linearization_fn=linearization_fn,
        )

        last_update_error = torch.max(torch.abs(new_states - old_states))

        states = new_states
        num_steps_done = it + 1

        if return_trace:
            trace.append(states.clone())

        if stopping_criterion == "update":
            if last_update_error.item() <= effective_tol:
                break

    final_merit = merit_fxn_batched(f, initial_state, states, drivers)

    has_custom_jacobian = jacobian_fn is not None or linearization_fn is not None

    info = {
        "num_iters": num_steps_done,
        "initial_merit": initial_merit.detach(),
        "final_merit": final_merit.detach(),
        "last_update_error": last_update_error.detach(),
        "tol": tol,
        "effective_tol": effective_tol,
        "strict_tol": strict_tol,
        "stopping_criterion": stopping_criterion,
        "scan_backend": scan_backend,
        "quasi": quasi,
        "batched": True,
        "batch_size": batch_size,
        "jacobian_backend": "custom" if has_custom_jacobian else "autograd",
        "linearization_backend": "custom" if linearization_fn is not None else "separate",
    }

    if return_trace:
        info["trace"] = torch.stack(trace, dim=0)

    return states, info
