import numpy as np
import pytest
import torch

from src.ode_solvers import lsim, solve_ode_fixed_step


def test_lsim_dense_matches_scipy_lsim_zoh():
    scipy = pytest.importorskip("scipy.signal")

    dtype = torch.float64

    A = torch.tensor([[-0.7, 0.2], [-0.1, -0.4]], dtype=dtype)
    B = torch.tensor([[1.0], [0.3]], dtype=dtype)
    C = torch.tensor([[1.2, -0.4]], dtype=dtype)
    D = torch.tensor([[0.05]], dtype=dtype)

    t = torch.linspace(0.0, 2.0, 101, dtype=dtype)
    U = torch.sin(2.0 * t)[:, None]
    x0 = torch.tensor([0.3, -0.2], dtype=dtype)

    y_torch, x_torch = lsim(A, B, C, D, U, t, x0=x0, scan_backend="torch")

    tout, y_scipy, x_scipy = scipy.lsim(
        (A.numpy(), B.numpy(), C.numpy(), D.numpy()),
        U=U.numpy(),
        T=t.numpy(),
        X0=x0.numpy(),
        interp=False,
    )

    if y_scipy.ndim == 1:
        y_scipy = y_scipy[:, None]

    assert np.allclose(tout, t.numpy())
    assert np.allclose(x_torch.numpy(), x_scipy, rtol=1e-7, atol=1e-8)
    assert np.allclose(y_torch.numpy(), y_scipy, rtol=1e-7, atol=1e-8)


def test_lsim_diagonal_matches_dense_scan():
    dtype = torch.float64

    A_diag = torch.tensor([-0.5, -1.2, -0.1], dtype=dtype)
    A = torch.diag(A_diag)
    B = torch.tensor([[1.0, 0.0], [0.2, 0.3], [-0.4, 0.1]], dtype=dtype)
    C = torch.eye(3, dtype=dtype)
    D = torch.zeros(3, 2, dtype=dtype)

    t = torch.linspace(0.0, 1.0, 64, dtype=dtype)
    U = torch.stack([torch.sin(t), torch.cos(t)], dim=-1)
    x0 = torch.tensor([0.2, -0.5, 0.7], dtype=dtype)

    y_dense, x_dense = lsim(A, B, C, D, U, t, x0=x0, scan_backend="torch")
    y_diag, x_diag = lsim(
        A_diag,
        B,
        C,
        D,
        U,
        t,
        x0=x0,
        diagonal=True,
        scan_backend="torch",
    )

    assert torch.allclose(x_diag, x_dense, rtol=1e-10, atol=1e-10)
    assert torch.allclose(y_diag, y_dense, rtol=1e-10, atol=1e-10)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for accelerated_scan")
def test_lsim_diagonal_accel_scan_matches_torch():
    pytest.importorskip("accelerated_scan")

    dtype = torch.float32
    device = torch.device("cuda")

    A_diag = torch.tensor([-0.5, -1.2, -0.1], dtype=dtype, device=device)
    B = torch.tensor([[1.0], [0.2], [-0.4]], dtype=dtype, device=device)
    C = torch.eye(3, dtype=dtype, device=device)
    D = torch.zeros(3, 1, dtype=dtype, device=device)

    t = torch.linspace(0.0, 1.0, 96, dtype=dtype, device=device)
    U = torch.sin(t)[:, None]
    x0 = torch.tensor([0.2, -0.5, 0.7], dtype=dtype, device=device)

    _, x_torch = lsim(
        A_diag,
        B,
        C,
        D,
        U,
        t,
        x0=x0,
        diagonal=True,
        scan_backend="torch",
    )
    _, x_accel = lsim(
        A_diag,
        B,
        C,
        D,
        U,
        t,
        x0=x0,
        diagonal=True,
        scan_backend="accel_scan",
    )

    assert torch.allclose(x_accel.cpu(), x_torch.cpu(), rtol=1e-5, atol=1e-5)


def test_rk4_sequential_matches_scipy_solve_ivp():
    scipy_integrate = pytest.importorskip("scipy.integrate")

    dtype = torch.float64
    t = torch.linspace(0.0, 2.0, 201, dtype=dtype)
    x0 = torch.tensor([1.0], dtype=dtype)

    def rhs(time, state, control):
        return -0.7 * state + torch.sin(time)

    states, _ = solve_ode_fixed_step(
        rhs=rhs,
        x0=x0,
        t=t,
        method="rk4",
        solver="sequential",
    )

    sol = scipy_integrate.solve_ivp(
        lambda tau, y: -0.7 * y + np.sin(tau),
        (float(t[0]), float(t[-1])),
        x0.numpy(),
        t_eval=t.numpy(),
        rtol=1e-11,
        atol=1e-13,
    )

    assert sol.success
    assert np.allclose(states.squeeze(-1).numpy(), sol.y[0], rtol=1e-6, atol=1e-7)


def test_deer_and_elk_solve_fixed_step_ode_match_sequential():
    dtype = torch.float64
    t = torch.linspace(0.0, 0.75, 24, dtype=dtype)
    x0 = torch.tensor([0.4], dtype=dtype)

    def rhs(time, state, control):
        return -0.3 * state + 0.1 * torch.tanh(state) + torch.cos(time)

    seq_states, _ = solve_ode_fixed_step(
        rhs=rhs,
        x0=x0,
        t=t,
        method="rk4",
        solver="sequential",
    )

    deer_states, deer_info = solve_ode_fixed_step(
        rhs=rhs,
        x0=x0,
        t=t,
        method="rk4",
        solver="deer",
        num_iters=20,
        tol=1e-10,
        strict_tol=True,
        quasi=True,
        scan_backend="torch",
        initial_guess="f0",
    )

    elk_states, elk_info = solve_ode_fixed_step(
        rhs=rhs,
        x0=x0,
        t=t,
        method="rk4",
        solver="elk",
        num_iters=30,
        tol=1e-10,
        strict_tol=True,
        quasi=True,
        scan_backend="torch",
        sigmasq=1e12,
        process_noise=1.0,
        initial_guess="f0",
    )

    assert deer_info["solver"] == "deer"
    assert elk_info["solver"] == "elk"
    assert torch.allclose(deer_states, seq_states, rtol=1e-7, atol=1e-8)
    assert torch.allclose(elk_states, seq_states, rtol=1e-5, atol=1e-6)
