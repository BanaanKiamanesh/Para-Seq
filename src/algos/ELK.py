import torch
from torch.func import vmap, jacrev
from torch._higher_order_ops.associative_scan import associative_scan

from src.algos.DEER import merit_fxn


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


def _run_accel_scan_chunk(A_chunk, b_chunk, accel_scan_fn):
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


def _diag_affine_scan_accel(A, b, accel_scan_fn):
    """Accelerated-scan backend for diagonal affine recurrences.

    Solves:

        h_t = A_t * h_{t-1} + b_t

    The accelerated_scan.warp backend requires sequence lengths that are
    powers of two, at least 32, and at most 65536. This function pads and
    chunks internally, so it supports arbitrary T.
    """
    if accel_scan_fn is None:
        raise ValueError(
            "accel_scan_fn must be provided when scan_backend='accel_scan'."
        )

    if A.ndim != 2 or b.ndim != 2:
        raise ValueError("Expected A and b with shape (T, D).")

    if A.shape != b.shape:
        raise ValueError(
            f"A and b must have the same shape, got {A.shape} and {b.shape}."
        )

    if A.device.type != "cuda":
        raise ValueError("accelerated_scan backend requires CUDA tensors.")

    T, D = A.shape

    outputs = []
    state_carry = torch.zeros(D, device=A.device, dtype=A.dtype)

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

        state_carry = b_prefix[-1]

        outputs.append(b_prefix)

        start = end

    return torch.cat(outputs, dim=0)


def _linearize_dynamics(
    f,
    initial_state,
    states,
    drivers,
    quasi=False,
    damping=0.0,
):
    """Linearize h_t = f(h_{t-1}, u_t) around the current trajectory guess."""
    T, D = states.shape

    initial_mean = f(initial_state, drivers[0])

    if T == 1:
        if quasi:
            A = torch.empty(0, D, device=states.device, dtype=states.dtype)
        else:
            A = torch.empty(0, D, D, device=states.device, dtype=states.dtype)

        b = torch.empty(0, D, device=states.device, dtype=states.dtype)
        return initial_mean, A, b

    prev_states = states[:-1]
    current_drivers = drivers[1:]

    fs = vmap(f)(prev_states, current_drivers)

    jac_f = jacrev(f, argnums=0)
    Jfs = vmap(jac_f)(prev_states, current_drivers)

    if quasi:
        A = torch.diagonal(Jfs, dim1=-2, dim2=-1)
        A = (1.0 - damping) * A
        b = fs - A * prev_states
    else:
        A = (1.0 - damping) * Jfs
        b = fs - torch.einsum("tij,tj->ti", A, prev_states)

    return initial_mean, A, b


def _dense_filter_operator(message_i, message_j):
    """Associative operator for dense parallel Kalman filtering.

    Each message has the form:

        A, b, C, J, eta
    """
    A_i, b_i, C_i, J_i, eta_i = message_i
    A_j, b_j, C_j, J_j, eta_j = message_j

    D = C_i.shape[-1]
    eye = torch.eye(D, device=C_i.device, dtype=C_i.dtype)
    eye = eye.expand(C_i.shape)

    left_denom = C_i @ J_j + eye
    left_inv = torch.linalg.inv(left_denom)

    A = A_j @ (left_inv @ A_i)

    b_mid = C_i @ eta_j.unsqueeze(-1) + b_i.unsqueeze(-1)
    b = (A_j @ (left_inv @ b_mid)).squeeze(-1) + b_j

    C = A_j @ (left_inv @ (C_i @ A_j.transpose(-1, -2))) + C_j

    right_denom = J_j @ C_i + eye
    right_inv = torch.linalg.inv(right_denom)

    eta_mid = eta_j.unsqueeze(-1) - J_j @ b_i.unsqueeze(-1)
    eta = (A_i.transpose(-1, -2) @ (right_inv @ eta_mid)).squeeze(-1) + eta_i

    J = A_i.transpose(-1, -2) @ (right_inv @ (J_j @ A_i)) + J_i

    return A, b, C, J, eta


def _initialize_dense_filter_messages(
    initial_mean,
    A,
    b,
    emissions,
    sigmasq=1e8,
    process_noise=1.0,
):
    T, D = emissions.shape

    device = emissions.device
    dtype = emissions.dtype

    q = float(process_noise)
    r = float(sigmasq)

    if q <= 0.0:
        raise ValueError("process_noise must be positive.")

    if r <= 0.0:
        raise ValueError("sigmasq must be positive.")

    eye = torch.eye(D, device=device, dtype=dtype)

    kalman_weight = q / (q + r)
    posterior_cov_value = q * r / (q + r)

    A0 = torch.zeros(D, D, device=device, dtype=dtype)
    b0 = initial_mean + kalman_weight * (emissions[0] - initial_mean)
    C0 = posterior_cov_value * eye
    J0 = (1.0 / q) * eye
    eta0 = torch.zeros(D, device=device, dtype=dtype)

    if T == 1:
        return (
            A0[None, :, :],
            b0[None, :],
            C0[None, :, :],
            J0[None, :, :],
            eta0[None, :],
        )

    A_msg = (1.0 - kalman_weight) * A
    b_msg = b + kalman_weight * (emissions[1:] - b)

    C_msg = posterior_cov_value * eye.expand(T - 1, D, D).clone()

    J_msg = torch.einsum("tji,tjk->tik", A, A) / (q + r)
    eta_msg = torch.einsum("tji,tj->ti", A, emissions[1:] - b) / (q + r)

    return (
        torch.cat([A0[None, :, :], A_msg], dim=0),
        torch.cat([b0[None, :], b_msg], dim=0),
        torch.cat([C0[None, :, :], C_msg], dim=0),
        torch.cat([J0[None, :, :], J_msg], dim=0),
        torch.cat([eta0[None, :], eta_msg], dim=0),
    )


def _dense_parallel_kalman_filter(
    initial_mean,
    A,
    b,
    emissions,
    sigmasq=1e8,
    process_noise=1.0,
):
    messages = _initialize_dense_filter_messages(
        initial_mean=initial_mean,
        A=A,
        b=b,
        emissions=emissions,
        sigmasq=sigmasq,
        process_noise=process_noise,
    )

    _, filtered_means, _, _, _ = associative_scan(
        _dense_filter_operator,
        messages,
        dim=0,
        combine_mode="generic",
    )

    return filtered_means


def _scalar_filter_operator(message_i, message_j):
    """Associative operator for diagonal/scalar parallel Kalman filtering.

    Each message has the form:

        A, b, C, J, eta
    """
    A_i, b_i, C_i, J_i, eta_i = message_i
    A_j, b_j, C_j, J_j, eta_j = message_j

    denominator = C_i * J_j + 1.0

    A = A_j * A_i / denominator
    b = A_j * (C_i * eta_j + b_i) / denominator + b_j
    C = C_i * (A_j * A_j) / denominator + C_j

    eta = A_i * (eta_j - J_j * b_i) / denominator + eta_i
    J = J_j * (A_i * A_i) / denominator + J_i

    return A, b, C, J, eta


def _initialize_scalar_filter_messages(
    initial_mean,
    A,
    b,
    emissions,
    sigmasq=1e8,
    process_noise=1.0,
):
    T, D = emissions.shape

    device = emissions.device
    dtype = emissions.dtype

    q = float(process_noise)
    r = float(sigmasq)

    if q <= 0.0:
        raise ValueError("process_noise must be positive.")

    if r <= 0.0:
        raise ValueError("sigmasq must be positive.")

    kalman_weight = q / (q + r)
    posterior_cov_value = q * r / (q + r)

    A0 = torch.zeros(D, device=device, dtype=dtype)
    b0 = initial_mean + kalman_weight * (emissions[0] - initial_mean)
    C0 = torch.full((D,), posterior_cov_value, device=device, dtype=dtype)
    J0 = torch.full((D,), 1.0 / q, device=device, dtype=dtype)
    eta0 = torch.zeros(D, device=device, dtype=dtype)

    if T == 1:
        return (
            A0[None, :],
            b0[None, :],
            C0[None, :],
            J0[None, :],
            eta0[None, :],
        )

    A_msg = (1.0 - kalman_weight) * A
    b_msg = b + kalman_weight * (emissions[1:] - b)

    C_msg = torch.full_like(A, posterior_cov_value)
    J_msg = (A * A) / (q + r)
    eta_msg = A * (emissions[1:] - b) / (q + r)

    return (
        torch.cat([A0[None, :], A_msg], dim=0),
        torch.cat([b0[None, :], b_msg], dim=0),
        torch.cat([C0[None, :], C_msg], dim=0),
        torch.cat([J0[None, :], J_msg], dim=0),
        torch.cat([eta0[None, :], eta_msg], dim=0),
    )


def _diag_parallel_kalman_filter(
    initial_mean,
    A,
    b,
    emissions,
    sigmasq=1e8,
    process_noise=1.0,
):
    messages = _initialize_scalar_filter_messages(
        initial_mean=initial_mean,
        A=A,
        b=b,
        emissions=emissions,
        sigmasq=sigmasq,
        process_noise=process_noise,
    )

    _, filtered_means, _, _, _ = associative_scan(
        _scalar_filter_operator,
        messages,
        dim=0,
        combine_mode="generic",
    )

    return filtered_means


def _normalize_mobius(alpha, beta, gamma, delta):
    """Normalize Mobius coefficients without changing the represented map.

    The scalar covariance map is

        p -> (alpha * p + beta) / (gamma * p + delta).

    Multiplying all four coefficients by a nonzero scalar does not change
    the map. During long scans with large sigmasq, the raw coefficients can
    overflow in float32. This normalization keeps the coefficients bounded.
    """
    scale = torch.maximum(
        torch.maximum(alpha.abs(), beta.abs()),
        torch.maximum(gamma.abs(), delta.abs()),
    )

    tiny = torch.finfo(alpha.dtype).tiny
    scale = torch.clamp(scale, min=tiny)

    return alpha / scale, beta / scale, gamma / scale, delta / scale


def _mobius_operator(message_i, message_j):
    """Associative composition of scalar Riccati/Mobius covariance maps.

    Each map has the form:

        p -> (alpha * p + beta) / (gamma * p + delta)

    The composition message_j o message_i is another Mobius map.

    The normalization is essential for float32 stability when sigmasq is
    large, for example sigmasq=1e8.
    """
    alpha_i, beta_i, gamma_i, delta_i = message_i
    alpha_j, beta_j, gamma_j, delta_j = message_j

    alpha = alpha_j * alpha_i + beta_j * gamma_i
    beta = alpha_j * beta_i + beta_j * delta_i
    gamma = gamma_j * alpha_i + delta_j * gamma_i
    delta = gamma_j * beta_i + delta_j * delta_i

    return _normalize_mobius(alpha, beta, gamma, delta)


def _compute_scalar_kalman_gains(
    A,
    emissions,
    sigmasq=1e8,
    process_noise=1.0,
):
    """Compute exact scalar Kalman gains for quasi-ELK.

    The posterior covariance recurrence is

        P_t =
            r * (A_t^2 P_{t-1} + q)
            /
            (A_t^2 P_{t-1} + q + r),

    where

        q = process_noise,
        r = sigmasq.

    This is a Mobius transform. We compute all prefix-composed covariance
    transforms with associative_scan. The Mobius operator normalizes its
    coefficients after each composition, which prevents float32 overflow.
    """
    T, D = emissions.shape

    device = emissions.device
    dtype = emissions.dtype

    q = torch.as_tensor(process_noise, device=device, dtype=dtype)
    r = torch.as_tensor(sigmasq, device=device, dtype=dtype)

    if process_noise <= 0.0:
        raise ValueError("process_noise must be positive.")

    if sigmasq <= 0.0:
        raise ValueError("sigmasq must be positive.")

    gains = torch.empty(T, D, device=device, dtype=dtype)

    gains[0] = q / (q + r)

    if T == 1:
        return gains

    p0 = q * r / (q + r)

    A_sq = A * A

    alpha = r * A_sq
    beta = r * q * torch.ones_like(A)
    gamma = A_sq
    delta = (q + r) * torch.ones_like(A)

    alpha, beta, gamma, delta = _normalize_mobius(alpha, beta, gamma, delta)

    alpha_prefix, beta_prefix, gamma_prefix, delta_prefix = associative_scan(
        _mobius_operator,
        (alpha, beta, gamma, delta),
        dim=0,
        combine_mode="generic",
    )

    p_post_tail = (
        alpha_prefix * p0 + beta_prefix
    ) / (
        gamma_prefix * p0 + delta_prefix
    )

    p_post = torch.cat(
        [
            p0.expand(1, D).clone(),
            p_post_tail,
        ],
        dim=0,
    )

    p_pred_tail = A_sq * p_post[:-1] + q

    gains[1:] = p_pred_tail / (p_pred_tail + r)

    return gains


def _diag_parallel_kalman_filter_accel_scan(
    initial_mean,
    A,
    b,
    emissions,
    sigmasq=1e8,
    process_noise=1.0,
    accel_scan_fn=None,
):
    """Quasi-ELK scalar Kalman filter using accelerated_scan for the mean scan.

    Once the scalar Kalman gains K_t are known, the filtered mean satisfies

        m_t = gate_t * m_{t-1} + token_t,

    where

        gate_t = (1 - K_t) * A_t,
        token_t = (1 - K_t) * b_t + K_t * y_t.

    This final mean recurrence is exactly the kind of diagonal affine
    recurrence supported by accelerated_scan.
    """
    T, _ = emissions.shape

    gains = _compute_scalar_kalman_gains(
        A=A,
        emissions=emissions,
        sigmasq=sigmasq,
        process_noise=process_noise,
    )

    gates = torch.zeros_like(emissions)
    tokens = torch.empty_like(emissions)

    tokens[0] = initial_mean + gains[0] * (emissions[0] - initial_mean)

    if T > 1:
        gates[1:] = (1.0 - gains[1:]) * A
        tokens[1:] = (1.0 - gains[1:]) * b + gains[1:] * emissions[1:]

    filtered_means = _diag_affine_scan_accel(
        A=gates,
        b=tokens,
        accel_scan_fn=accel_scan_fn,
    )

    return filtered_means


def elk_step(
    f,
    initial_state,
    states,
    drivers,
    sigmasq=1e8,
    process_noise=1.0,
    quasi=False,
    damping=0.0,
    clip_value=None,
    scan_backend="torch",
    accel_scan_fn=None,
):
    """One ELK or quasi-ELK iteration."""
    initial_mean, A, b = _linearize_dynamics(
        f=f,
        initial_state=initial_state,
        states=states,
        drivers=drivers,
        quasi=quasi,
        damping=damping,
    )

    if quasi:
        if scan_backend == "torch":
            new_states = _diag_parallel_kalman_filter(
                initial_mean=initial_mean,
                A=A,
                b=b,
                emissions=states,
                sigmasq=sigmasq,
                process_noise=process_noise,
            )
        elif scan_backend == "accel_scan":
            new_states = _diag_parallel_kalman_filter_accel_scan(
                initial_mean=initial_mean,
                A=A,
                b=b,
                emissions=states,
                sigmasq=sigmasq,
                process_noise=process_noise,
                accel_scan_fn=accel_scan_fn,
            )
        else:
            raise ValueError(f"Unknown scan_backend: {scan_backend}")
    else:
        if scan_backend != "torch":
            raise ValueError("Full ELK only supports scan_backend='torch'.")

        new_states = _dense_parallel_kalman_filter(
            initial_mean=initial_mean,
            A=A,
            b=b,
            emissions=states,
            sigmasq=sigmasq,
            process_noise=process_noise,
        )

    if clip_value is not None:
        new_states = torch.clamp(new_states, -clip_value, clip_value)
        new_states = torch.nan_to_num(new_states)

    return new_states


def elk_alg(
    f,
    initial_state,
    states_guess,
    drivers,
    sigmasq=1e8,
    process_noise=1.0,
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
    """ELK solver for h_t = f(h_{t-1}, u_t)."""
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

        new_states = elk_step(
            f=f,
            initial_state=initial_state,
            states=old_states,
            drivers=drivers,
            sigmasq=sigmasq,
            process_noise=process_noise,
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
        "sigmasq": sigmasq,
        "process_noise": process_noise,
        "quasi": quasi,
        "scan_backend": scan_backend,
    }

    if return_trace:
        info["trace"] = torch.stack(trace, dim=0)

    return states, info
