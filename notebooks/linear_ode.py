import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import time
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.ode_solvers import lsim


# Number of hidden/state variables in the linear system.
NUM_STATES = 100

# We keep one input channel, but B is zero, so the input has no effect.
INPUT_DIM = 1

dtype = torch.float64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(0)
torch.manual_seed(0)


# ---------------------------------------------------------------------
# Build a stable continuous-time linear system:
#
#   dx/dt = A x + B u
#   y     = C x + D u
#
# We construct A = Q diag(lambda) Q^T with negative eigenvalues.
# Therefore all modes decay over time.
# ---------------------------------------------------------------------

Lambda = np.random.random(NUM_STATES) * -10.0

M = np.random.randn(NUM_STATES, NUM_STATES)
Q, _ = np.linalg.qr(M)

A_np = Q @ np.diag(Lambda) @ Q.T

# Zero input matrix. This means the input u(t) does not affect the state.
B_np = np.zeros((NUM_STATES, INPUT_DIM))

# Full-state output: y(t) = x(t).
C_np = np.eye(NUM_STATES)

# No direct feedthrough from input to output.
D_np = np.zeros((NUM_STATES, INPUT_DIM))


# Convert everything to torch tensors.
A = torch.tensor(A_np, dtype=dtype, device=device)
B = torch.tensor(B_np, dtype=dtype, device=device)
C = torch.tensor(C_np, dtype=dtype, device=device)
D = torch.tensor(D_np, dtype=dtype, device=device)


# ---------------------------------------------------------------------
# Build a fixed-step time grid.
#
# The ODE solver utilities expect a fixed-step grid. Using arange * dt
# avoids small spacing issues that can sometimes happen with linspace.
# ---------------------------------------------------------------------

dt = 1.0 / 512.0
final_time = 5.0
num_steps = int(final_time / dt)

t = torch.arange(
    num_steps + 1,
    dtype=dtype,
    device=device,
) * dt


# Input signal. It is zero here, and B is also zero.
U = torch.zeros(
    t.shape[0],
    INPUT_DIM,
    dtype=dtype,
    device=device,
)


# Random initial state.
x0 = torch.randn(
    NUM_STATES,
    dtype=dtype,
    device=device,
)


# ---------------------------------------------------------------------
# Simulate the linear system using the scan-based lsim implementation.
#
# Since A is dense, use:
#
#   diagonal=False
#   scan_backend="torch"
#
# The accelerated_scan backend is only for diagonal affine recurrences.
# ---------------------------------------------------------------------

start = time.time()

y, x = lsim(
    A=A,
    B=B,
    C=C,
    D=D,
    U=U,
    t=t,
    x0=x0,
    diagonal=False,
    scan_backend="torch",
)

elapsed = time.time() - start


# ---------------------------------------------------------------------
# Print basic diagnostics.
# For a stable A and B = 0, the state norm should decay toward zero.
# ---------------------------------------------------------------------

print(f"lsim finished in {elapsed:.6f} seconds")
print(f"x shape: {tuple(x.shape)}")
print(f"y shape: {tuple(y.shape)}")
print(f"initial norm: {torch.linalg.norm(x[0]).item():.6e}")
print(f"final norm:   {torch.linalg.norm(x[-1]).item():.6e}")


# ---------------------------------------------------------------------
# Plot the first few state components.
# ---------------------------------------------------------------------

t_cpu = t.detach().cpu()
x_cpu = x.detach().cpu()

plt.figure(figsize=(10, 6))

for i in range(8):
    plt.plot(t_cpu, x_cpu[:, i], label=f"x{i + 1}")

plt.xlabel("Time")
plt.ylabel("State")
plt.title("100-State Stable Linear System with B = 0")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()