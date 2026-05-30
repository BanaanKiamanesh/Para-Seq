import torch
from torch.func import vmap, jacrev

from src.utils.AssScan import full_mat_scan, diag_mat_scan


def sequential_rollout(f, initial_state, drivers):
    """Sequentially evaluates h_t = f(h_{t-1}, u_t).

    Args:
        f: callable(state, input) -> next_state
        initial_state: (D,)
        drivers: (T, input_dim)

    Returns:
        states: (T, D)
    """
    states = []
    state = initial_state

    for t in range(drivers.shape[0]):
        state = f(state, drivers[t])
        states.append(state)

    return torch.stack(states, dim=0)


def get_residual(f, initial_state, states, drivers):
    """Computes r_t = h_t - f(h_{t-1}, u_t).

    Args:
        f: callable(state, input) -> next_state
        initial_state: (D,)
        states: (T, D)
        drivers: (T, input_dim)

    Returns:
        residual: (T, D)
    """
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
):
    """One DEER/Newton fixed-point iteration.

    Args:
        f: callable(state, input) -> next_state
        initial_state: (D,)
        states: current trajectory guess, shape (T, D)
        drivers: inputs, shape (T, input_dim)
        quasi: if True, use diagonal Jacobian approximation
        damping: multiplies Jacobian by (1 - damping)
        clip_value: optional scalar clipping for the new states

    Returns:
        new_states: updated trajectory, shape (T, D)
    """
    T, _ = states.shape

    if T == 1:
        return f(initial_state, drivers[0])[None, :]

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

        _, new_states = diag_mat_scan(A, b, dim=0)

    else:
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
    tol=5e-8,
    quasi=False,
    damping=0.0,
    clip_value=None,
    return_trace=False,
):
    """Simple DEER solver for h_t = f(h_{t-1}, u_t).

    Args:
        f: callable(state, input) -> next_state
        initial_state: (D,)
        states_guess: initial trajectory guess, shape (T, D)
        drivers: inputs, shape (T, input_dim)
        num_iters: maximum Newton/DEER iterations
        tol: stop when 0.5 * ||residual||^2 <= tol
        quasi: if True, use diagonal quasi-DEER
        damping: multiplies Jacobian by (1 - damping)
        clip_value: optional clipping for numerical stability
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

        states = deer_step(
            f=f,
            initial_state=initial_state,
            states=states,
            drivers=drivers,
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
    }

    if return_trace:
        info["trace"] = torch.stack(trace, dim=0)

    return states, info
