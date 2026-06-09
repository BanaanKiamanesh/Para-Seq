from __future__ import annotations

from typing import Callable

import torch


def split_driver(driver: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if driver.shape[-1] < 1:
        raise ValueError(
            "ODE drivers must contain at least the time coordinate in the last dimension."
        )

    return driver[..., :1], driver[..., 1:]


def _dt_tensor(dt: float | torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(dt, device=reference.device, dtype=reference.dtype)


def make_euler_step(rhs: Callable, dt: float | torch.Tensor) -> Callable:
    def step(state: torch.Tensor, driver: torch.Tensor) -> torch.Tensor:
        t, control = split_driver(driver)
        h = _dt_tensor(dt, state)
        return state + h * rhs(t, state, control)

    return step


def make_midpoint_step(rhs: Callable, dt: float | torch.Tensor) -> Callable:
    def step(state: torch.Tensor, driver: torch.Tensor) -> torch.Tensor:
        t, control = split_driver(driver)
        h = _dt_tensor(dt, state)
        half_h = 0.5 * h

        k1 = rhs(t, state, control)
        k2 = rhs(t + half_h, state + half_h * k1, control)

        return state + h * k2

    return step


def make_heun_step(rhs: Callable, dt: float | torch.Tensor) -> Callable:
    def step(state: torch.Tensor, driver: torch.Tensor) -> torch.Tensor:
        t, control = split_driver(driver)
        h = _dt_tensor(dt, state)

        k1 = rhs(t, state, control)
        k2 = rhs(t + h, state + h * k1, control)

        return state + 0.5 * h * (k1 + k2)

    return step


def make_rk4_step(rhs: Callable, dt: float | torch.Tensor) -> Callable:
    def step(state: torch.Tensor, driver: torch.Tensor) -> torch.Tensor:
        t, control = split_driver(driver)
        h = _dt_tensor(dt, state)
        half_h = 0.5 * h

        k1 = rhs(t, state, control)
        k2 = rhs(t + half_h, state + half_h * k1, control)
        k3 = rhs(t + half_h, state + half_h * k2, control)
        k4 = rhs(t + h, state + h * k3, control)

        return state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return step


def make_integrator_step(
    rhs: Callable,
    dt: float | torch.Tensor,
    method: str = "rk4",
) -> Callable:
    method = method.lower()

    if method == "euler":
        return make_euler_step(rhs, dt)

    if method in ("midpoint", "rk2"):
        return make_midpoint_step(rhs, dt)

    if method == "heun":
        return make_heun_step(rhs, dt)

    if method == "rk4":
        return make_rk4_step(rhs, dt)

    raise ValueError(
        f"Unknown integrator method {method!r}. Expected 'euler', 'midpoint', "
        "'heun', 'rk2', or 'rk4'."
    )
