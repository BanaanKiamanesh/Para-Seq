from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ode_solvers import solve_ode_fixed_step


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_call(fn, repeats: int, warmups: int, device: torch.device):
    result = None

    with torch.no_grad():
        for _ in range(warmups):
            result = fn()
            sync(device)

        times = []
        for _ in range(repeats):
            sync(device)
            start = time.perf_counter()
            result = fn()
            sync(device)
            times.append(time.perf_counter() - start)

    return float(np.median(times)), result


def make_fixed_grid(num_points: int, dt: float, *, dtype, device):
    return torch.arange(num_points, dtype=dtype, device=device) * float(dt)


def make_rhs_1d():
    def rhs(time, state, control):
        return -0.7 * state + torch.sin(time)

    return rhs


def make_rhs_coupled(state_dim: int, dtype, device):
    main_diag = -0.35 - 0.02 * torch.arange(
        state_dim,
        dtype=dtype,
        device=device,
    )

    A = torch.diag(main_diag)

    if state_dim > 1:
        idx = torch.arange(state_dim, device=device)
        A[idx, (idx + 1) % state_dim] = 0.04
        A[idx, (idx - 1) % state_dim] = -0.03

    frequencies = torch.arange(
        1,
        state_dim + 1,
        dtype=dtype,
        device=device,
    )

    def rhs(time, state, control):
        x_left = torch.roll(state, shifts=1, dims=-1)
        x_right = torch.roll(state, shifts=-1, dims=-1)

        linear_part = state @ A.T
        nonlinear_self = 0.08 * torch.sin(state)
        nonlinear_coupling = 0.04 * torch.tanh(x_left * x_right)
        nonlinear_mixing = 0.03 * torch.sin(state * x_right)
        forcing = 0.10 * torch.sin(time * frequencies)

        return (
            linear_part
            + nonlinear_self
            + nonlinear_coupling
            + nonlinear_mixing
            + forcing
        )

    return rhs


def make_x0(state_dim: int, *, dtype, device):
    if state_dim == 1:
        return torch.tensor([1.0], dtype=dtype, device=device)

    if state_dim == 8:
        return torch.tensor(
            [1.0, -0.5, 0.25, 0.75, -1.0, 0.4, -0.2, 0.1],
            dtype=dtype,
            device=device,
        )

    base = torch.linspace(
        -0.5,
        0.5,
        state_dim,
        dtype=dtype,
        device=device,
    )
    base[0] = 1.0
    return base


def run_solver(
    *,
    rhs,
    x0,
    t,
    solver_name: str,
    backend: str,
    num_iters: int,
    tol: float,
    damping: float,
    repeats: int,
    warmups: int,
    device: torch.device,
):
    def fn():
        return solve_ode_fixed_step(
            rhs=rhs,
            x0=x0,
            t=t,
            method="rk4",
            solver=solver_name,
            num_iters=num_iters,
            tol=tol,
            strict_tol=False,
            quasi=True,
            damping=damping,
            scan_backend=backend,
            accel_module="warp",
            initial_guess="f0",
            clip_value=1e6,
            include_initial=True,
        )

    return time_call(fn, repeats, warmups, device)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark fixed-step nonlinear ODE solvers."
    )

    parser.add_argument("--T", type=int, default=4096)
    parser.add_argument("--D", type=int, default=1)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=2)

    # Backward-compatible args used by heavy_benchmark.py.
    # These now control ELK only. DEER has its own safer default.
    parser.add_argument("--num-iters", type=int, default=None)
    parser.add_argument("--tol", type=float, default=None)

    # New solver-specific args.
    parser.add_argument("--elk-iters", type=int, default=20)
    parser.add_argument("--deer-iters", type=int, default=256)
    parser.add_argument("--elk-tol", type=float, default=1e-5)
    parser.add_argument("--deer-tol", type=float, default=1e-4)
    parser.add_argument("--deer-damping", type=float, default=0.0)
    parser.add_argument("--max-error", type=float, default=1e-3)

    parser.add_argument("--no-accel", action="store_true")
    parser.add_argument("--no-deer", action="store_true")
    parser.add_argument("--no-elk", action="store_true")
    parser.add_argument("--include-scipy", action="store_true")
    parser.add_argument(
        "--output",
        type=str,
        default="test/bench/ode_solvers/nonlinear_ode_benchmark.csv",
    )

    args = parser.parse_args()

    # Legacy compatibility:
    # heavy_benchmark.py passes --num-iters and --tol.
    # We let those tune ELK, while DEER keeps its safer defaults.
    if args.num_iters is not None:
        args.elk_iters = args.num_iters

    if args.tol is not None:
        args.elk_tol = args.tol

    device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but CUDA is not available.")

    dtype = torch.float32 if device.type == "cuda" else torch.float64
    dt = 1.0 / 1024.0

    t = make_fixed_grid(args.T, dt, dtype=dtype, device=device)
    x0 = make_x0(args.D, dtype=dtype, device=device)

    if args.D == 1:
        rhs = make_rhs_1d()
    else:
        rhs = make_rhs_coupled(args.D, dtype, device)

    rows = []

    def seq_fn():
        return solve_ode_fixed_step(
            rhs=rhs,
            x0=x0,
            t=t,
            method="rk4",
            solver="sequential",
            include_initial=True,
        )

    seq_time, seq_out = time_call(seq_fn, args.repeats, args.warmups, device)
    seq_states, _ = seq_out

    rows.append({
        "suite": "nonlinear",
        "problem": f"rk4_nonlinear_{args.D}d",
        "T": args.T,
        "D": args.D,
        "method": "sequential_rk4",
        "solver": "sequential",
        "backend": "python_loop",
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "time_s": seq_time,
        "error": 0.0,
        "valid": True,
        "reference": "sequential_rk4",
        "notes": "fixed-step RK4 baseline",
    })

    specs = []

    if not args.no_elk:
        specs.append({
            "method": "elk_rk4_torch",
            "solver": "elk",
            "backend": "torch",
            "num_iters": args.elk_iters,
            "tol": args.elk_tol,
            "damping": 0.0,
        })

        if device.type == "cuda" and not args.no_accel:
            specs.append({
                "method": "elk_rk4_accel_scan",
                "solver": "elk",
                "backend": "accel_scan",
                "num_iters": args.elk_iters,
                "tol": args.elk_tol,
                "damping": 0.0,
            })

    if not args.no_deer:
        specs.append({
            "method": "deer_rk4_torch_safe",
            "solver": "deer",
            "backend": "torch",
            "num_iters": args.deer_iters,
            "tol": args.deer_tol,
            "damping": args.deer_damping,
        })

        if device.type == "cuda" and not args.no_accel:
            specs.append({
                "method": "deer_rk4_accel_scan_safe",
                "solver": "deer",
                "backend": "accel_scan",
                "num_iters": args.deer_iters,
                "tol": args.deer_tol,
                "damping": args.deer_damping,
            })

    for spec in specs:
        try:
            par_time, par_out = run_solver(
                rhs=rhs,
                x0=x0,
                t=t,
                solver_name=spec["solver"],
                backend=spec["backend"],
                num_iters=spec["num_iters"],
                tol=spec["tol"],
                damping=spec["damping"],
                repeats=args.repeats,
                warmups=args.warmups,
                device=device,
            )

            par_states, par_info = par_out
            err = float(torch.max(torch.abs(par_states - seq_states)).detach().cpu())
            valid = bool(err <= args.max_error)

            notes = (
                f"num_iters={par_info.get('num_iters', '')}; "
                f"requested_iters={spec['num_iters']}; "
                f"tol={spec['tol']}; "
                f"damping={spec['damping']}; "
                f"max_error_threshold={args.max_error}"
            )

        except Exception as exc:
            par_time = float("nan")
            err = float("nan")
            valid = False
            notes = f"FAILED: {type(exc).__name__}: {exc}"

        rows.append({
            "suite": "nonlinear",
            "problem": f"rk4_nonlinear_{args.D}d",
            "T": args.T,
            "D": args.D,
            "method": spec["method"],
            "solver": spec["solver"],
            "backend": spec["backend"],
            "device": str(device),
            "dtype": str(dtype).replace("torch.", ""),
            "time_s": par_time,
            "error": err,
            "valid": valid,
            "reference": "sequential_rk4",
            "notes": notes,
        })

    if args.include_scipy and args.D == 1:
        try:
            from scipy.integrate import solve_ivp
        except Exception as exc:
            rows.append({
                "suite": "nonlinear",
                "problem": "rk4_nonlinear_1d",
                "T": args.T,
                "D": args.D,
                "method": "scipy_solve_ivp",
                "solver": "scipy",
                "backend": "solve_ivp",
                "device": "cpu",
                "dtype": "float64",
                "time_s": float("nan"),
                "error": float("nan"),
                "valid": False,
                "reference": "sequential_rk4",
                "notes": f"FAILED_IMPORT: {type(exc).__name__}: {exc}",
            })
        else:
            t_np = t.detach().cpu().numpy().astype(np.float64)
            x0_np = x0.detach().cpu().numpy().astype(np.float64)

            def scipy_rhs(t_scalar, y):
                return -0.7 * y + math.sin(t_scalar)

            def scipy_fn():
                sol = solve_ivp(
                    scipy_rhs,
                    (float(t_np[0]), float(t_np[-1])),
                    x0_np,
                    t_eval=t_np,
                    rtol=1e-8,
                    atol=1e-10,
                )
                return sol.y.T

            scipy_time, scipy_np = time_call(
                scipy_fn,
                max(1, args.repeats // 2),
                max(0, args.warmups // 2),
                torch.device("cpu"),
            )

            scipy_states = torch.as_tensor(
                scipy_np,
                dtype=seq_states.dtype,
                device=seq_states.device,
            )

            scipy_err = float(
                torch.max(torch.abs(seq_states - scipy_states)).detach().cpu()
            )

            rows.append({
                "suite": "nonlinear",
                "problem": "rk4_nonlinear_1d",
                "T": args.T,
                "D": args.D,
                "method": "scipy_solve_ivp",
                "solver": "scipy",
                "backend": "solve_ivp",
                "device": "cpu",
                "dtype": "float64",
                "time_s": scipy_time,
                "error": scipy_err,
                "valid": bool(scipy_err <= args.max_error),
                "reference": "sequential_rk4",
                "notes": "adaptive solve_ivp compared to fixed-step RK4",
            })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "suite",
        "problem",
        "T",
        "D",
        "method",
        "solver",
        "backend",
        "device",
        "dtype",
        "time_s",
        "error",
        "valid",
        "reference",
        "notes",
    ]

    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{row['method']:>28s} | T={args.T:<7d} D={args.D:<4d} "
            f"| {row['time_s']:.6e} s | err={row['error']:.3e} "
            f"| valid={row['valid']}"
        )

    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
