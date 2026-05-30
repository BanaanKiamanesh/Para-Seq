import torch
from torch.func import vmap

from src.algos.DEER import merit_fxn


def jacobi_step(
    f,
    initial_state,
    states,
    drivers,
    clip_value=None,
):
    """One Jacobi fixed-point iteration for h_t = f(h_{t-1}, u_t).

    Jacobi uses the zero approximation

        A_t = 0.

    Therefore the update is simply

        h_t^{new} = f(h_{t-1}^{old}, u_t).

    All time steps can be evaluated in parallel because every update only
    depends on the old trajectory.
    """
    prev_states = torch.cat([initial_state[None, :], states[:-1]], dim=0)

    new_states = vmap(f)(prev_states, drivers)

    if clip_value is not None:
        new_states = torch.clamp(new_states, -clip_value, clip_value)
        new_states = torch.nan_to_num(new_states)

    return new_states


def jacobi_alg(
    f,
    initial_state,
    states_guess,
    drivers,
    num_iters=20,
    tol=5e-8,
    clip_value=None,
    return_trace=False,
):
    """Jacobi solver for h_t = f(h_{t-1}, u_t).

    Args:
        f: callable(state, input) -> next_state
        initial_state: (D,)
        states_guess: initial trajectory guess, shape (T, D)
        drivers: inputs, shape (T, input_dim)
        num_iters: maximum Jacobi iterations
        tol: stop when 0.5 * ||residual||^2 <= tol
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

        states = jacobi_step(
            f=f,
            initial_state=initial_state,
            states=states,
            drivers=drivers,
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
