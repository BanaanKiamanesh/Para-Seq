import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
from src.ode_solvers import solve_ode_fixed_step
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Device setup.
#
# The accelerated_scan backend only works on CUDA tensors.
# If you want a CPU version, use scan_backend="torch" instead.
# ---------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type != "cuda":
    raise RuntimeError("scan_backend='accel_scan' requires CUDA, but CUDA is not available.")


# ---------------------------------------------------------------------
# Numerical precision.
#
# float32 is usually the safest and fastest choice for accelerated_scan.warp.
# ---------------------------------------------------------------------

dtype = torch.float32


# ---------------------------------------------------------------------
# Problem setup.
#
# We solve an 8-state nonlinear coupled ODE:
#
#   dx/dt = A x
#          + 0.08 sin(x)
#          + 0.04 tanh(x_left * x_right)
#          + 0.03 sin(x * x_right)
#          + 0.10 sin(omega t)
#
# The linear part A is stable, while the nonlinear terms add state-dependent
# self-nonlinearity, neighbor coupling, and multiplicative nonlinear mixing.
# ---------------------------------------------------------------------

state_dim = 8


# ---------------------------------------------------------------------
# Fixed-step time grid.
#
# The ODE utilities require a fixed-step grid.
# Using arange * dt is safer than linspace for the fixed-step check.
# ---------------------------------------------------------------------

dt = 1.0 / 2**10
num_steps = int(20.0 / dt)

t = torch.arange(
    num_steps + 1,
    dtype=dtype,
    device=device,
) * dt


# Initial condition for the 8-dimensional state.
x0 = torch.tensor(
    [1.0, -0.5, 0.25, 0.75, -1.0, 0.4, -0.2, 0.1],
    dtype=dtype,
    device=device,
)


# Stable weakly coupled linear part.
A = torch.tensor(
    [
        [-0.70,  0.08,  0.00,  0.00,  0.00,  0.00,  0.00,  0.03],
        [-0.04, -0.65,  0.07,  0.00,  0.00,  0.00,  0.00,  0.00],
        [ 0.00, -0.05, -0.60,  0.06,  0.00,  0.00,  0.00,  0.00],
        [ 0.00,  0.00, -0.04, -0.55,  0.05,  0.00,  0.00,  0.00],
        [ 0.00,  0.00,  0.00, -0.03, -0.50,  0.05,  0.00,  0.00],
        [ 0.00,  0.00,  0.00,  0.00, -0.03, -0.45,  0.04,  0.00],
        [ 0.00,  0.00,  0.00,  0.00,  0.00, -0.02, -0.40,  0.04],
        [ 0.02,  0.00,  0.00,  0.00,  0.00,  0.00, -0.02, -0.35],
    ],
    dtype=dtype,
    device=device,
)


# Different forcing frequency for each state component.
frequencies = torch.arange(1, state_dim + 1, dtype=dtype, device=device)


def rhs(time, state, control):
    """Right-hand side of the nonlinear ODE.

    Args:
        time:
            Current time, shape (..., 1) or scalar-compatible.
        state:
            Current state, shape (..., state_dim).
        control:
            Optional control input. Not used in this example.

    Returns:
        Time derivative dx/dt with shape (..., state_dim).
    """

    # Cyclic neighbor states.
    # x_left[i]  = x[i - 1]
    # x_right[i] = x[i + 1]
    x_left = torch.roll(state, shifts=1, dims=-1)
    x_right = torch.roll(state, shifts=-1, dims=-1)

    # Linear stable dynamics.
    linear_part = state @ A.T

    # Elementwise nonlinear self-dynamics.
    nonlinear_self = 0.08 * torch.sin(state)

    # Nonlinear coupling between neighboring states.
    nonlinear_coupling = 0.04 * torch.tanh(x_left * x_right)

    # Multiplicative nonlinear state mixing.
    nonlinear_mixing = 0.03 * torch.sin(state * x_right)

    # Time-dependent forcing with different frequency per state.
    forcing = 0.10 * torch.sin(time * frequencies)

    return linear_part + nonlinear_self + nonlinear_coupling + nonlinear_mixing + forcing


# ---------------------------------------------------------------------
# Solve the fixed-step ODE.
#
# method="rk4":
#   Uses RK4 to turn the continuous-time ODE into a discrete transition map.
#
# solver="deer":
#   Solves the whole trajectory using DEER fixed-point iterations.
#
# quasi=True:
#   Uses diagonal Jacobian approximation. This is required for accel_scan.
#
# scan_backend="accel_scan":
#   Uses the CUDA accelerated diagonal affine scan backend.
# ---------------------------------------------------------------------

states, info_deer = solve_ode_fixed_step(
    rhs=rhs,
    x0=x0,
    t=t,
    method="rk4",
    solver="deer",
    num_iters=20,
    tol=1e-5,
    strict_tol=False,
    quasi=True,
    damping=0.05,
    scan_backend="accel_scan",
    accel_module="warp",
    initial_guess="f0",
    clip_value=1e6,
)


# ---------------------------------------------------------------------
# Move tensors to CPU for plotting.
# ---------------------------------------------------------------------

t_cpu = t.detach().cpu()
states_cpu = states.detach().cpu()


# ---------------------------------------------------------------------
# Plot all 8 state trajectories.
# ---------------------------------------------------------------------

plt.figure(figsize=(10, 6))

for i in range(state_dim):
    plt.plot(t_cpu, states_cpu[:, i], label=f"x{i + 1}")

plt.xlabel("Time")
plt.ylabel("State")
plt.title("8-State Nonlinear Coupled ODE Solved with RK4 + DEER + accel_scan")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# Print solver diagnostics.
print(info_deer)