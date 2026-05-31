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
    """Default update tolerance based on floating-point precision.

    DEER's original experiments use loose precision-aware tolerances:
        float32: about 1e-4
        float64: about 1e-7

    The important point is that using 1e-12 with float32 can make the
    solver run until max_iters even after it is already numerically converged.
    """
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


def _run_accel_scan_chunk(A_chunk, b_chunk, accel_scan_fn):
    """Run accelerated_scan on one chunk.

    accelerated_scan.warp requires sequence length to be a power of two,
    at least 32, and at most 65536. This helper pads the chunk so that
    the kernel receives a valid length.

    Args:
        A_chunk: shape (Tc, D)
        b_chunk: shape (Tc, D)
        accel_scan_fn: accelerated_scan scan function

    Returns:
        b_prefix: shape (Tc, D), assuming zero initial state
    """
    original_len, D = A_chunk.shape

    padded_len = _next_power_of_two(max(original_len, _ACCEL_SCAN_MIN_LEN))

    if padded_len > _ACCEL_SCAN_MAX_LEN:
        raise ValueError(
            f"accelerated_scan chunk length must be <= {_ACCEL_SCAN_MAX_LEN}, "
            f"but got padded_len={padded_len}."
        )

    if padded_len != original_len:
        pad_len = padded_len - original_len

        A_pad = torch.ones(
            pad_len,
            D,
            device=A_chunk.device,
            dtype=A_chunk.dtype,
        )

        b_pad = torch.zeros(
            pad_len,
            D,
            device=b_chunk.device,
            dtype=b_chunk.dtype,
        )

        A_chunk = torch.cat([A_chunk, A_pad], dim=0)
        b_chunk = torch.cat([b_chunk, b_pad], dim=0)

    gate = A_chunk.transpose(0, 1).unsqueeze(0).contiguous()
    token = b_chunk.transpose(0, 1).unsqueeze(0).contiguous()

    scanned = accel_scan_fn(gate, token)

    b_prefix = scanned.squeeze(0).transpose(0, 1).contiguous()

    return b_prefix[:original_len]


def _diag_mat_scan_accel(A, b, accel_scan_fn):
    """Accelerated-scan backend for diagonal affine recurrences.

    Solves

        h_t = A_t * h_{t-1} + b_t

    using accelerated_scan.

    accelerated_scan.warp has the restriction:

        32 <= sequence length <= 65536
        sequence length must be a power of two

    Therefore this function chunks long sequences and pads non-power-of-two
    chunks internally.

    Args:
        A: shape (T, D)
        b: shape (T, D)
        accel_scan_fn: accelerated_scan scan function

    Returns:
        A_prefix: shape (T, D)
        b_prefix: shape (T, D)
    """
    if accel_scan_fn is None:
        raise ValueError(
            "accel_scan_fn must be provided when scan_backend='accel_scan'."
        )

    if A.ndim != 2 or b.ndim != 2:
        raise ValueError("Expected A and b with shape (T, D).")

    if A.shape != b.shape:
        raise ValueError(
            f"A and b must have the same shape, got {A.shape} and {b.shape}.")

    if A.device.type != "cuda":
        raise ValueError("accelerated_scan backend requires CUDA tensors.")

    T, D = A.shape

    b_outputs = []
    A_outputs = []

    state_carry = torch.zeros(D, device=A.device, dtype=A.dtype)
    A_carry = torch.ones(D, device=A.device, dtype=A.dtype)

    start = 0

    while start < T:
        end = min(start + _ACCEL_SCAN_MAX_LEN, T)

        A_chunk = A[start:end].contiguous()
        b_chunk = b[start:end].contiguous()

        b_prefix_zero = _run_accel_scan_chunk(
            A_chunk=A_chunk,
            b_chunk=b_chunk,
            accel_scan_fn=accel_scan_fn,
        )

        A_prefix_local = torch.cumprod(A_chunk, dim=0)

        b_prefix = A_prefix_local * state_carry[None, :] + b_prefix_zero
        A_prefix = A_prefix_local * A_carry[None, :]

        state_carry = b_prefix[-1]
        A_carry = A_prefix[-1]

        b_outputs.append(b_prefix)
        A_outputs.append(A_prefix)

        start = end

    return torch.cat(A_outputs, dim=0), torch.cat(b_outputs, dim=0)


def sequential_rollout(f, initial_state, drivers):
    """Sequentially evaluates h_t = f(h_{t-1}, u_t).

    Args:
        f: callable(state, input) -> next_state
        initial_state: shape (D,)
        drivers: shape (T, input_dim)

    Returns:
        states: shape (T, D)
    """
    states = []
    state = initial_state

    for t in range(drivers.shape[0]):
        state = f(state, drivers[t])
        states.append(state)

    return torch.stack(states, dim=0)


def get_residual(f, initial_state, states, drivers):
    """Computes r_t = h_t - f(h_{t-1}, u_t)."""
    prev_states = torch.cat([initial_state[None, :], states[:-1]], dim=0)
    predicted_states = vmap(f)(prev_states, drivers)

    return states - predicted_states


def merit_fxn(f, initial_state, states, drivers):
    """Computes 0.5 * ||residual||^2."""
    residual = get_residual(f, initial_state, states, drivers)
    return 0.5 * torch.sum(residual * residual)


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
):
    """One DEER/Newton fixed-point iteration."""
    T, _ = states.shape

    if T == 1:
        new_states = f(initial_state, drivers[0])[None, :]

        if clip_value is not None:
            new_states = torch.clamp(new_states, -clip_value, clip_value)
            new_states = torch.nan_to_num(new_states)

        return new_states

    old_prev_states = states[:-1]
    current_drivers = drivers[1:]

    fs = vmap(f)(old_prev_states, current_drivers)

    jac_f = jacrev(f, argnums=0)
    Jfs = vmap(jac_f)(old_prev_states, current_drivers)

    if quasi:
        As = torch.diagonal(Jfs, dim1=-2, dim2=-1)
        As = (1.0 - damping) * As

        bs = fs - As * old_prev_states

        A0 = torch.zeros_like(As[0])
        b0 = f(initial_state, drivers[0])

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

        As = (1.0 - damping) * Jfs

        bs = fs - torch.einsum("tij,tj->ti", As, old_prev_states)

        A0 = torch.zeros_like(As[0])
        b0 = f(initial_state, drivers[0])

        A = torch.cat([A0[None, :, :], As], dim=0)
        b = torch.cat([b0[None, :], bs], dim=0)

        _, new_states = full_mat_scan(A, b, dim=0)

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
):
    """DEER solver for h_t = f(h_{t-1}, u_t).

    Args:
        f: callable(state, input) -> next_state
        initial_state: shape (D,)
        states_guess: initial trajectory guess, shape (T, D)
        drivers: inputs, shape (T, input_dim)
        num_iters: maximum Newton/DEER iterations
        tol: stopping tolerance. If None, uses dtype-aware default.
             If strict_tol=False, tol is clamped upward to a dtype-safe value.
        quasi: if True, use diagonal quasi-DEER
        damping: multiplies Jacobian by (1 - damping)
        clip_value: optional clipping for numerical stability
        return_trace: if True, return all intermediate states
        scan_backend: "torch" or "accel_scan"
        accel_scan_fn: accelerated_scan scan function when scan_backend="accel_scan"
        strict_tol: if True, use tol exactly. If False, avoid impossible fp32 tolerances.
        stopping_criterion:
            "update": stop when max(|h_new - h_old|) <= effective_tol
            "merit": stop when 0.5 * ||residual||^2 <= effective_tol

    Returns:
        final_states: shape (T, D)
        info: dict
    """
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
        float("inf"), device=states.device, dtype=states.dtype)

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
    }

    if return_trace:
        info["trace"] = torch.stack(trace, dim=0)

    return states, info
