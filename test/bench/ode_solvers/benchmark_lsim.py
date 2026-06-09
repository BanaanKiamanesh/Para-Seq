from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ode_solvers import lsim


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


def make_dense_stable_system(state_dim: int, input_dim: int, seed: int):
    rng = np.random.default_rng(seed)

    lambdas = -0.1 - 9.9 * rng.random(state_dim)
    matrix = rng.standard_normal((state_dim, state_dim))
    q, _ = np.linalg.qr(matrix)

    A = q @ np.diag(lambdas) @ q.T
    B = rng.standard_normal((state_dim, input_dim)) * 0.2
    C = np.eye(state_dim)
    D = np.zeros((state_dim, input_dim))
    x0 = rng.standard_normal(state_dim)

    return A, B, C, D, x0


def make_diag_stable_system(state_dim: int, input_dim: int, seed: int):
    rng = np.random.default_rng(seed)

    A_diag = -0.1 - 9.9 * rng.random(state_dim)
    B = rng.standard_normal((state_dim, input_dim)) * 0.2
    C = np.eye(state_dim)
    D = np.zeros((state_dim, input_dim))
    x0 = rng.standard_normal(state_dim)

    return A_diag, B, C, D, x0


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark scan-based lsim.")
    parser.add_argument("--T", type=int, default=4096)
    parser.add_argument("--D", type=int, default=8)
    parser.add_argument("--M", type=int, default=1)
    parser.add_argument("--diagonal", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--no-scipy", action="store_true")
    parser.add_argument("--no-accel", action="store_true")
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but CUDA is not available.")

    dtype = torch.float32 if device.type == "cuda" else torch.float64
    dt = 1.0 / 512.0

    if args.diagonal:
        A_np, B_np, C_np, D_np, x0_np = make_diag_stable_system(args.D, args.M, args.seed)
        A_scipy_np = np.diag(A_np)
        problem = "linear_diag"
    else:
        A_np, B_np, C_np, D_np, x0_np = make_dense_stable_system(args.D, args.M, args.seed)
        A_scipy_np = A_np
        problem = "linear_dense"

    t_np = np.arange(args.T, dtype=np.float64) * dt

    U_np = np.sin(0.7 * t_np)[:, None]
    if args.M > 1:
        U_np = np.concatenate(
            [np.sin((0.7 + 0.1 * i) * t_np)[:, None] for i in range(args.M)],
            axis=1,
        )

    A = torch.as_tensor(A_np, dtype=dtype, device=device)
    B = torch.as_tensor(B_np, dtype=dtype, device=device)
    C = torch.as_tensor(C_np, dtype=dtype, device=device)
    Dmat = torch.as_tensor(D_np, dtype=dtype, device=device)
    x0 = torch.as_tensor(x0_np, dtype=dtype, device=device)
    U = torch.as_tensor(U_np, dtype=dtype, device=device)
    t = make_fixed_grid(args.T, dt, dtype=dtype, device=device)

    rows = []
    scipy_x = None

    if not args.no_scipy:
        try:
            from scipy import signal
        except Exception as exc:
            rows.append({
                "suite": "linear",
                "problem": problem,
                "T": args.T,
                "D": args.D,
                "method": "scipy_lsim",
                "solver": "scipy",
                "backend": "scipy",
                "device": "cpu",
                "dtype": "float64",
                "time_s": float("nan"),
                "error": float("nan"),
                "reference": "none",
                "notes": f"FAILED_IMPORT: {type(exc).__name__}: {exc}",
            })
        else:
            def scipy_fn():
                _, _, x_out = signal.lsim(
                    (A_scipy_np, B_np, C_np, D_np),
                    U=U_np,
                    T=t_np,
                    X0=x0_np,
                    interp=False,
                )
                return x_out

            scipy_time, scipy_x = time_call(
                scipy_fn,
                max(1, args.repeats // 2),
                max(0, args.warmups // 2),
                torch.device("cpu"),
            )

            rows.append({
                "suite": "linear",
                "problem": problem,
                "T": args.T,
                "D": args.D,
                "method": "scipy_lsim",
                "solver": "scipy",
                "backend": "scipy",
                "device": "cpu",
                "dtype": "float64",
                "time_s": scipy_time,
                "error": 0.0,
                "reference": "scipy_lsim",
                "notes": "scipy.signal.lsim interp=False",
            })

    def torch_scan_fn():
        return lsim(
            A=A,
            B=B,
            C=C,
            D=Dmat,
            U=U,
            t=t,
            x0=x0,
            diagonal=args.diagonal,
            scan_backend="torch",
        )

    torch_time, torch_out = time_call(torch_scan_fn, args.repeats, args.warmups, device)
    _, torch_x = torch_out

    if scipy_x is not None:
        scipy_ref = torch.as_tensor(scipy_x, dtype=torch_x.dtype, device=torch_x.device)
        torch_err = float(torch.max(torch.abs(torch_x - scipy_ref)).detach().cpu())
    else:
        torch_err = float("nan")

    rows.append({
        "suite": "linear",
        "problem": problem,
        "T": args.T,
        "D": args.D,
        "method": "torch_scan_lsim",
        "solver": "scan",
        "backend": "torch",
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "time_s": torch_time,
        "error": torch_err,
        "reference": "scipy_lsim" if scipy_x is not None else "none",
        "notes": "scan-based dense/diagonal lsim",
    })

    if args.diagonal and device.type == "cuda" and not args.no_accel:
        def accel_fn():
            return lsim(
                A=A,
                B=B,
                C=C,
                D=Dmat,
                U=U,
                t=t,
                x0=x0,
                diagonal=True,
                scan_backend="accel_scan",
                accel_module="warp",
            )

        try:
            accel_time, accel_out = time_call(accel_fn, args.repeats, args.warmups, device)
            _, accel_x = accel_out
            accel_err = float(torch.max(torch.abs(accel_x - torch_x)).detach().cpu())
            notes = "accelerated_scan.warp compared to torch diagonal scan"
        except Exception as exc:
            accel_time = float("nan")
            accel_err = float("nan")
            notes = f"FAILED: {type(exc).__name__}: {exc}"

        rows.append({
            "suite": "linear",
            "problem": problem,
            "T": args.T,
            "D": args.D,
            "method": "accel_scan_lsim",
            "solver": "scan",
            "backend": "accel_scan",
            "device": str(device),
            "dtype": str(dtype).replace("torch.", ""),
            "time_s": accel_time,
            "error": accel_err,
            "reference": "torch_scan_lsim",
            "notes": notes,
        })

    if args.output:
        output = Path(args.output)
    else:
        output = Path(__file__).with_name("lsim_benchmark.csv")

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
        "reference",
        "notes",
    ]

    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{row['method']:>20s} | T={args.T:<7d} D={args.D:<4d} "
            f"| {row['time_s']:.6e} s | err={row['error']:.3e}"
        )

    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
