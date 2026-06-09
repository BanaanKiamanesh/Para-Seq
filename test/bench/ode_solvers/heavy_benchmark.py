from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
LOG_DIR = SCRIPT_DIR / "logs"


def run_command(command: list[str]) -> None:
    print()
    print(" ".join(command))
    subprocess.run(command, check=True)


def profile_cases(profile: str):
    if profile == "quick":
        return {
            "linear_dense": [(4096, 8), (16384, 8)],
            "linear_diag": [(4096, 8), (16384, 8), (65536, 32)],
            "nonlinear": [(1024, 1), (4096, 1), (1024, 8)],
            "repeats": 3,
            "warmups": 1,
        }

    if profile == "heavy":
        return {
            "linear_dense": [(4096, 8), (16384, 8), (65536, 4), (4096, 32), (1024, 100)],
            "linear_diag": [(4096, 8), (16384, 8), (65536, 4), (65536, 32), (65536, 100)],
            "nonlinear": [(1024, 1), (4096, 1), (16384, 1), (1024, 8), (4096, 8), (16384, 8)],
            "repeats": 5,
            "warmups": 2,
        }

    if profile == "extreme":
        return {
            "linear_dense": [(4096, 8), (16384, 8), (65536, 4), (8192, 32), (2048, 100)],
            "linear_diag": [(4096, 8), (16384, 8), (65536, 4), (65536, 32), (65536, 100), (262144, 100)],
            "nonlinear": [(1024, 1), (4096, 1), (16384, 1), (65536, 1), (1024, 8), (4096, 8), (16384, 8), (65536, 8)],
            "repeats": 7,
            "warmups": 2,
        }

    raise ValueError(f"Unknown profile: {profile}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full ODE benchmark suite.")
    parser.add_argument("--profile", choices=["quick", "heavy", "extreme"], default="heavy")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-accel", action="store_true")
    parser.add_argument("--linear-only", action="store_true")
    parser.add_argument("--nonlinear-only", action="store_true")
    parser.add_argument("--include-scipy-nonlinear", action="store_true")
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--warmups", type=int, default=None)
    parser.add_argument("--num-iters", type=int, default=20)
    parser.add_argument("--tol", type=float, default=1e-5)
    args = parser.parse_args()

    if args.linear_only and args.nonlinear_only:
        raise ValueError("Cannot use both --linear-only and --nonlinear-only.")

    cfg = profile_cases(args.profile)
    repeats = cfg["repeats"] if args.repeats is None else args.repeats
    warmups = cfg["warmups"] if args.warmups is None else args.warmups

    device_args = []
    if args.cpu:
        device_args = ["--device", "cpu"]
    elif args.device:
        device_args = ["--device", args.device]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    all_csvs = []

    if not args.nonlinear_only:
        for T, D in cfg["linear_dense"]:
            out = LOG_DIR / f"lsim_dense_T{T}_D{D}_{timestamp}.csv"
            command = [
                sys.executable,
                str(SCRIPT_DIR / "benchmark_lsim.py"),
                "--T", str(T),
                "--D", str(D),
                "--repeats", str(repeats),
                "--warmups", str(warmups),
                "--output", str(out),
                *device_args,
            ]
            run_command(command)
            all_csvs.append(out)

        for T, D in cfg["linear_diag"]:
            out = LOG_DIR / f"lsim_diag_T{T}_D{D}_{timestamp}.csv"
            command = [
                sys.executable,
                str(SCRIPT_DIR / "benchmark_lsim.py"),
                "--T", str(T),
                "--D", str(D),
                "--diagonal",
                "--repeats", str(repeats),
                "--warmups", str(warmups),
                "--output", str(out),
                *device_args,
            ]
            if args.no_accel:
                command.append("--no-accel")
            run_command(command)
            all_csvs.append(out)

    if not args.linear_only:
        for T, D in cfg["nonlinear"]:
            out = LOG_DIR / f"nonlinear_T{T}_D{D}_{timestamp}.csv"
            command = [
                sys.executable,
                str(SCRIPT_DIR / "benchmark_nonlinear_ode.py"),
                "--T", str(T),
                "--D", str(D),
                "--repeats", str(repeats),
                "--warmups", str(warmups),
                "--num-iters", str(args.num_iters),
                "--tol", str(args.tol),
                "--output", str(out),
                *device_args,
            ]
            if args.no_accel:
                command.append("--no-accel")
            if args.include_scipy_nonlinear:
                command.append("--include-scipy")
            run_command(command)
            all_csvs.append(out)

    frames = []

    for csv_path in all_csvs:
        df = pd.read_csv(csv_path)
        df.insert(0, "source_file", csv_path.name)
        df.insert(0, "timestamp", timestamp)
        df.insert(0, "profile", args.profile)
        frames.append(df)

    combined = LOG_DIR / f"heavy_ode_benchmark_{timestamp}.csv"

    if frames:
        out_df = pd.concat(frames, ignore_index=True)
        out_df.to_csv(combined, index=False)
        shutil.copyfile(combined, LOG_DIR / "latest_heavy_ode_benchmark.csv")

        print()
        print(f"Wrote {combined}")
        print(f"Updated {LOG_DIR / 'latest_heavy_ode_benchmark.csv'}")
    else:
        print("No benchmark rows were generated.")


if __name__ == "__main__":
    main()
