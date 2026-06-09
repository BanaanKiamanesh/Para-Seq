# Benchmarks

Run all benchmark commands from the repository root. Use `PYTHONPATH=.` for
scripts that import `src`.

## Suites

| Path | Purpose |
| --- | --- |
| `test/bench/algos` | Core Picard, Jacobi, DEER, and ELK solver timing. |
| `test/bench/baseline` | ParaRNN, ParaGRU, and ParaLSTM reference comparisons. |
| `test/bench/ode_solvers` | Linear system and nonlinear ODE solver timing. |

## Quick Runs

```bash
PYTHONPATH=. python test/bench/algos/benchmarking.py --max-seq-len 4096
PYTHONPATH=. python test/bench/baseline/strong_baselines.py --repeats 2
python test/bench/ode_solvers/heavy_benchmark.py --profile quick
```

## Bash Wrappers

```bash
bash test/bench/algos/benchmarking.sh
ODE_BENCH_PROFILE=quick bash test/bench/ode_solvers/run_ode_benchmarks.sh
```

The baseline suite has no folder-local bash wrapper; run its Python script
directly.

See each subdirectory README for the full argument list.
