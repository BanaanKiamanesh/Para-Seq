# ODE Solver Benchmarks

Benchmarks scan-based linear simulation and fixed-step nonlinear ODE solvers.

## Full Suite

```bash
python test/bench/ode_solvers/heavy_benchmark.py --profile quick
```

The shell wrapper forwards arguments to `heavy_benchmark.py`:

```bash
ODE_BENCH_PROFILE=heavy bash test/bench/ode_solvers/run_ode_benchmarks.sh
```

Use `ODE_BENCH_PROFILE` to choose `quick`, `heavy`, or `extreme`. Any extra
arguments are passed through to `heavy_benchmark.py`.

Examples:

```bash
ODE_BENCH_PROFILE=quick bash test/bench/ode_solvers/run_ode_benchmarks.sh
ODE_BENCH_PROFILE=heavy bash test/bench/ode_solvers/run_ode_benchmarks.sh --no-accel
```

## `heavy_benchmark.py` Arguments

| Argument | Default | Notes |
| --- | --- | --- |
| `--profile` | `heavy` | Choices: `quick`, `heavy`, `extreme`. |
| `--device` | auto | Torch device string. |
| `--cpu` | off | Force CPU. |
| `--no-accel` | off | Skip accelerated scan runs. |
| `--linear-only` | off | Run only `benchmark_lsim.py`. |
| `--nonlinear-only` | off | Run only `benchmark_nonlinear_ode.py`. |
| `--include-scipy-nonlinear` | off | Include SciPy nonlinear baseline. |
| `--repeats` | profile default | Timed runs per case. |
| `--warmups` | profile default | Warmup runs per case. |
| `--num-iters` | `20` | Shared nonlinear solver iterations. |
| `--tol` | `1e-5` | Shared nonlinear solver tolerance. |

## `benchmark_lsim.py` Arguments

| Argument | Default | Notes |
| --- | --- | --- |
| `--T` | `4096` | Number of time samples. |
| `--D` | `8` | State dimension. |
| `--M` | `1` | Input dimension. |
| `--diagonal` | off | Use diagonal dynamics. |
| `--device` | auto | `cuda` when available, else `cpu`. |
| `--repeats` | `5` | Timed runs per method. |
| `--warmups` | `2` | Warmup runs per method. |
| `--seed` | `1234` | Random seed. |
| `--no-scipy` | off | Skip SciPy baseline. |
| `--no-accel` | off | Skip accelerated scan. |
| `--output` | auto | Override CSV output path. |

Example:

```bash
python test/bench/ode_solvers/benchmark_lsim.py --T 16384 --D 32 --diagonal
```

## `benchmark_nonlinear_ode.py` Arguments

| Argument | Default | Notes |
| --- | --- | --- |
| `--T` | `4096` | Number of time samples. |
| `--D` | `1` | State dimension. |
| `--device` | auto | `cuda` when available, else `cpu`. |
| `--repeats` | `5` | Timed runs per method. |
| `--warmups` | `2` | Warmup runs per method. |
| `--num-iters` | unset | Shared alias for solver iterations. |
| `--tol` | unset | Shared alias for solver tolerance. |
| `--elk-iters` | `20` | ELK iterations. |
| `--deer-iters` | `256` | DEER iterations. |
| `--elk-tol` | `1e-5` | ELK tolerance. |
| `--deer-tol` | `1e-4` | DEER tolerance. |
| `--deer-damping` | `0.0` | DEER damping. |
| `--max-error` | `1e-3` | Validity threshold. |
| `--no-accel` | off | Skip accelerated scan. |
| `--no-deer` | off | Skip DEER methods. |
| `--no-elk` | off | Skip ELK methods. |
| `--include-scipy` | off | Include SciPy baseline. |
| `--output` | `test/bench/ode_solvers/nonlinear_ode_benchmark.csv` | CSV output path. |

Example:

```bash
python test/bench/ode_solvers/benchmark_nonlinear_ode.py --T 8192 --D 8 --no-accel
```
