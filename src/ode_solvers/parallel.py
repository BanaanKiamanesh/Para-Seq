from __future__ import annotations

from typing import Callable

import torch
from torch.func import vmap

from src.algos.DEER import deer_alg_batched
from src.algos.ELK import elk_alg_batched
from src.ode_solvers.config import ODESolverConfig
from src.ode_solvers.integrators import make_integrator_step


def _load_accel_scan(accel_module: str):
    if accel_module == "warp":
        from accelerated_scan.warp import scan
        return scan

    if accel_module == "scalar":
        from accelerated_scan.scalar import scan
        return scan

    if accel_module == "ref":
        from accelerated_scan.ref import scan
        return scan

    raise ValueError(f"Unknown accelerated_scan module: {accel_module!r}.")


def _normalize_initial_state(x0: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if x0.ndim == 1:
        return x0.unsqueeze(0), False

    if x0.ndim == 2:
        return x0, True

    raise ValueError(
        "x0 must have shape (state_dim,) or (B, state_dim), "
        f"got {tuple(x0.shape)}."
    )


def _infer_fixed_dt(t: torch.Tensor) -> float:
    if t.ndim != 1:
        raise ValueError(f"t must have shape (num_time_points,), got {tuple(t.shape)}.")

    if t.shape[0] < 2:
        raise ValueError("At least two time points are required.")

    dts = t[1:] - t[:-1]
    dt0 = dts[0]

    if not torch.allclose(dts, dt0.expand_as(dts), rtol=1e-5, atol=1e-8):
        raise ValueError("Only fixed-step time grids are supported.")

    return float(dt0.item())


def make_ode_drivers(
    t: torch.Tensor,
    controls: torch.Tensor | None = None,
    *,
    batch_size: int | None = None,
    device=None,
    dtype=None,
) -> torch.Tensor:
    t = torch.as_tensor(t, device=device, dtype=dtype)

    if t.ndim != 1:
        raise ValueError(f"t must have shape (num_time_points,), got {tuple(t.shape)}.")

    if t.shape[0] < 2:
        raise ValueError("At least two time points are required.")

    step_times = t[:-1]
    num_steps = step_times.shape[0]

    if controls is None:
        if batch_size is None:
            return step_times[:, None]

        return step_times[None, :, None].expand(batch_size, -1, -1).contiguous()

    controls = torch.as_tensor(controls, device=device, dtype=dtype)

    if controls.ndim == 2:
        if controls.shape[0] == t.shape[0]:
            controls = controls[:-1]
        elif controls.shape[0] != num_steps:
            raise ValueError(
                "controls must have time length len(t) or len(t)-1. Got "
                f"{controls.shape[0]} for len(t)={t.shape[0]}."
            )

        if batch_size is None:
            return torch.cat([step_times[:, None], controls], dim=-1)

        controls = controls[None, :, :].expand(batch_size, -1, -1)
        times = step_times[None, :, None].expand(batch_size, -1, -1)
        return torch.cat([times, controls], dim=-1).contiguous()

    if controls.ndim == 3:
        if controls.shape[1] == t.shape[0]:
            controls = controls[:, :-1, :]
        elif controls.shape[1] != num_steps:
            raise ValueError(
                "controls must have time length len(t) or len(t)-1. Got "
                f"{controls.shape[1]} for len(t)={t.shape[0]}."
            )

        if batch_size is not None and controls.shape[0] != batch_size:
            raise ValueError(
                f"controls batch size {controls.shape[0]} does not match {batch_size}."
            )

        times = step_times[None, :, None].expand(controls.shape[0], -1, -1)
        return torch.cat([times, controls], dim=-1).contiguous()

    raise ValueError(
        "controls must have shape (T, control_dim), (T-1, control_dim), "
        "(B, T, control_dim), or (B, T-1, control_dim)."
    )


def sequential_ode_rollout_batched(
    transition: Callable,
    initial_state: torch.Tensor,
    drivers: torch.Tensor,
) -> torch.Tensor:
    if initial_state.ndim != 2:
        raise ValueError(
            "initial_state must have shape (B, D), got "
            f"{tuple(initial_state.shape)}."
        )

    if drivers.ndim != 3:
        raise ValueError(
            "drivers must have shape (B, T, driver_dim), got "
            f"{tuple(drivers.shape)}."
        )

    if drivers.shape[0] != initial_state.shape[0]:
        raise ValueError("initial_state and drivers batch dimensions must match.")

    states = []
    state = initial_state

    for time_idx in range(drivers.shape[1]):
        state = vmap(transition)(state, drivers[:, time_idx, :])
        states.append(state)

    if not states:
        return torch.empty(
            initial_state.shape[0],
            0,
            initial_state.shape[-1],
            device=initial_state.device,
            dtype=initial_state.dtype,
        )

    return torch.stack(states, dim=1)


def make_initial_guess_batched(
    transition: Callable,
    initial_state: torch.Tensor,
    drivers: torch.Tensor,
    *,
    guess_type: str = "f0",
) -> torch.Tensor:
    batch_size, num_steps, _ = drivers.shape
    state_dim = initial_state.shape[-1]

    if guess_type == "zero":
        return torch.zeros(
            batch_size,
            num_steps,
            state_dim,
            device=initial_state.device,
            dtype=initial_state.dtype,
        )

    if guess_type == "constant":
        return initial_state[:, None, :].expand(-1, num_steps, -1).contiguous()

    if guess_type == "f0":
        zero_states = torch.zeros(
            batch_size * num_steps,
            state_dim,
            device=initial_state.device,
            dtype=initial_state.dtype,
        )
        flat_drivers = drivers.reshape(batch_size * num_steps, drivers.shape[-1])
        flat_guess = vmap(transition)(zero_states, flat_drivers)
        return flat_guess.reshape(batch_size, num_steps, state_dim)

    raise ValueError("guess_type must be 'zero', 'constant', or 'f0'.")


def solve_ode_fixed_step(
    rhs: Callable,
    x0: torch.Tensor,
    t: torch.Tensor,
    controls: torch.Tensor | None = None,
    *,
    method: str = "rk4",
    solver: str = "deer",
    num_iters: int = 20,
    tol: float | None = None,
    strict_tol: bool = False,
    stopping_criterion: str = "update",
    initial_guess: str = "f0",
    quasi: bool = True,
    damping: float = 0.0,
    clip_value: float | None = None,
    scan_backend: str = "torch",
    accel_scan_fn=None,
    accel_module: str = "warp",
    sigmasq: float = 1e8,
    process_noise: float = 1.0,
    include_initial: bool = True,
    return_info: bool = True,
):
    x0 = torch.as_tensor(x0)
    initial_state, had_batch_dim = _normalize_initial_state(x0)

    t = torch.as_tensor(t, device=initial_state.device, dtype=initial_state.dtype)
    dt = _infer_fixed_dt(t)

    drivers = make_ode_drivers(
        t=t,
        controls=controls,
        batch_size=initial_state.shape[0],
        device=initial_state.device,
        dtype=initial_state.dtype,
    )

    transition = make_integrator_step(rhs, dt=dt, method=method)

    if drivers.shape[1] == 0:
        states = torch.empty(
            initial_state.shape[0],
            0,
            initial_state.shape[-1],
            device=initial_state.device,
            dtype=initial_state.dtype,
        )
        info = {"solver": solver, "method": method, "dt": dt, "num_steps": 0}
    else:
        solver = solver.lower()

        if scan_backend == "accel_scan" and accel_scan_fn is None:
            accel_scan_fn = _load_accel_scan(accel_module)

        if scan_backend == "accel_scan" and not quasi:
            raise ValueError(
                "scan_backend='accel_scan' requires quasi=True because accelerated_scan "
                "supports diagonal affine scans."
            )

        if solver == "sequential":
            states = sequential_ode_rollout_batched(
                transition=transition,
                initial_state=initial_state,
                drivers=drivers,
            )
            info = {
                "solver": "sequential",
                "method": method,
                "dt": dt,
                "num_steps": drivers.shape[1],
                "scan_backend": None,
            }

        elif solver == "deer":
            states_guess = make_initial_guess_batched(
                transition=transition,
                initial_state=initial_state,
                drivers=drivers,
                guess_type=initial_guess,
            )

            states, info = deer_alg_batched(
                f=transition,
                initial_state=initial_state,
                states_guess=states_guess,
                drivers=drivers,
                num_iters=num_iters,
                tol=tol,
                quasi=quasi,
                damping=damping,
                clip_value=clip_value,
                return_trace=False,
                scan_backend=scan_backend,
                accel_scan_fn=accel_scan_fn,
                strict_tol=strict_tol,
                stopping_criterion=stopping_criterion,
            )
            info = dict(info)
            info["solver"] = "deer"
            info["method"] = method
            info["dt"] = dt

        elif solver == "elk":
            states_guess = make_initial_guess_batched(
                transition=transition,
                initial_state=initial_state,
                drivers=drivers,
                guess_type=initial_guess,
            )

            states, info = elk_alg_batched(
                f=transition,
                initial_state=initial_state,
                states_guess=states_guess,
                drivers=drivers,
                sigmasq=sigmasq,
                process_noise=process_noise,
                num_iters=num_iters,
                tol=tol,
                quasi=quasi,
                damping=damping,
                clip_value=clip_value,
                return_trace=False,
                scan_backend=scan_backend,
                accel_scan_fn=accel_scan_fn,
                strict_tol=strict_tol,
                stopping_criterion=stopping_criterion,
            )
            info = dict(info)
            info["solver"] = "elk"
            info["method"] = method
            info["dt"] = dt

        else:
            raise ValueError("solver must be 'sequential', 'deer', or 'elk'.")

    if include_initial:
        output = torch.cat([initial_state[:, None, :], states], dim=1)
    else:
        output = states

    if not had_batch_dim:
        output = output.squeeze(0)

    if return_info:
        return output, info

    return output


def solve_ode_fixed_step_from_config(
    rhs: Callable,
    x0: torch.Tensor,
    t: torch.Tensor,
    controls: torch.Tensor | None = None,
    config: ODESolverConfig | None = None,
):
    cfg = ODESolverConfig() if config is None else config

    return solve_ode_fixed_step(
        rhs=rhs,
        x0=x0,
        t=t,
        controls=controls,
        method=cfg.method,
        solver=cfg.solver,
        num_iters=cfg.num_iters,
        tol=cfg.tol,
        strict_tol=cfg.strict_tol,
        stopping_criterion=cfg.stopping_criterion,
        initial_guess=cfg.initial_guess,
        quasi=cfg.quasi,
        damping=cfg.damping,
        clip_value=cfg.clip_value,
        scan_backend=cfg.scan_backend,
        accel_module=cfg.accel_module,
        sigmasq=cfg.sigmasq,
        process_noise=cfg.process_noise,
        include_initial=cfg.include_initial,
    )
