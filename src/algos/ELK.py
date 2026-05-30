import torch
from torch.func import vmap, jacrev
from torch._higher_order_ops.associative_scan import associative_scan

from src.algos.DEER import merit_fxn


def _linearize_dynamics(
    f,
    initial_state,
    states,
    drivers,
    quasi=False,
    damping=0.0,
):
    """Linearize h_t = f(h_{t-1}, u_t) around the current trajectory guess.

    This returns the LGSSM parameters used by ELK:

        h_1 ~ N(initial_mean, I)

        h_t = A_t h_{t-1} + b_t + q_t,
        q_t ~ N(0, I)

        y_t = h_t + e_t,
        e_t ~ N(0, sigmasq I)

    Args:
        f: callable(state, driver) -> next_state
        initial_state: (D,)
        states: current trajectory guess, shape (T, D)
        drivers: inputs, shape (T, input_dim)
        quasi: if True, use diagonal Jacobian approximation
        damping: optional multiplicative damping on the Jacobian

    Returns:
        initial_mean: shape (D,)
        A: shape (T - 1, D, D) if quasi=False, or (T - 1, D) if quasi=True
        b: shape (T - 1, D)
    """
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

    This is the dense matrix analogue of the scalar operator in the ELK repo.
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
    """Initialize dense Kalman filtering messages for associative_scan.

    Args:
        initial_mean: (D,)
        A: (T - 1, D, D)
        b: (T - 1, D)
        emissions: previous iterate, shape (T, D)
        sigmasq: emission variance
        process_noise: process variance

    Returns:
        tuple of messages, each with leading dimension T
    """
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
    """Parallel dense Kalman filter for full ELK."""
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

    This is the PyTorch version of elk/utils/parallel_kalman_scalar.py.
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
    """Initialize scalar/diagonal Kalman filtering messages.

    Args:
        initial_mean: (D,)
        A: (T - 1, D)
        b: (T - 1, D)
        emissions: previous iterate, shape (T, D)

    Returns:
        tuple of messages, each with shape (T, D)
    """
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
    """Parallel scalar Kalman filter for quasi-ELK."""
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
):
    """One ELK or quasi-ELK iteration.

    Args:
        f: callable(state, input) -> next_state
        initial_state: (D,)
        states: current trajectory guess, shape (T, D)
        drivers: inputs, shape (T, input_dim)
        sigmasq: emission variance. Large sigmasq approaches DEER.
        process_noise: process covariance scale
        quasi: if True, use diagonal quasi-ELK
        damping: optional multiplicative damping on the Jacobian
        clip_value: optional scalar clipping for numerical safety

    Returns:
        new_states: updated trajectory, shape (T, D)
    """
    initial_mean, A, b = _linearize_dynamics(
        f=f,
        initial_state=initial_state,
        states=states,
        drivers=drivers,
        quasi=quasi,
        damping=damping,
    )

    if quasi:
        new_states = _diag_parallel_kalman_filter(
            initial_mean=initial_mean,
            A=A,
            b=b,
            emissions=states,
            sigmasq=sigmasq,
            process_noise=process_noise,
        )
    else:
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
    tol=5e-8,
    quasi=False,
    damping=0.0,
    clip_value=None,
    return_trace=False,
):
    """ELK solver for h_t = f(h_{t-1}, u_t).

    Args:
        f: callable(state, input) -> next_state
        initial_state: (D,)
        states_guess: initial trajectory guess, shape (T, D)
        drivers: inputs, shape (T, input_dim)
        sigmasq: trust-region variance.
                 Large sigmasq -> weak damping -> closer to DEER.
                 Small sigmasq -> stronger trust-region damping.
        process_noise: process covariance scale used in the Kalman filter
        num_iters: maximum number of ELK iterations
        tol: stop when 0.5 * ||residual||^2 <= tol
        quasi: if True, use quasi-ELK with diagonal Jacobians
        damping: optional multiplicative damping on the Jacobian
        clip_value: optional scalar clipping for numerical stability
        return_trace: if True, return all intermediate states

    Returns:
        final_states: (T, D)
        info: dict
    """
    states = states_guess.clone()
    trace = [states.clone()] if return_trace else None

    initial_merit = merit_fxn(f, initial_state, states, drivers)

    num_steps_done = 0

    for it in range(num_iters):
        current_merit = merit_fxn(f, initial_state, states, drivers)

        if current_merit.item() <= tol:
            break

        states = elk_step(
            f=f,
            initial_state=initial_state,
            states=states,
            drivers=drivers,
            sigmasq=sigmasq,
            process_noise=process_noise,
            quasi=quasi,
            damping=damping,
            clip_value=clip_value,
        )

        num_steps_done = it + 1

        if return_trace:
            trace.append(states.clone())

    final_merit = merit_fxn(f, initial_state, states, drivers)

    info = {
        "num_iters": num_steps_done,
        "initial_merit": initial_merit.detach(),
        "final_merit": final_merit.detach(),
        "sigmasq": sigmasq,
        "process_noise": process_noise,
        "quasi": quasi,
    }

    if return_trace:
        info["trace"] = torch.stack(trace, dim=0)

    return states, info
