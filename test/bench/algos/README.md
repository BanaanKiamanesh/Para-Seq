# Algorithm Benchmarks

Benchmarks the standalone sequential, Picard, Jacobi, DEER, quasi-DEER, ELK,
and quasi-ELK solvers over increasing sequence lengths.

## Run

```bash
PYTHONPATH=. python test/bench/algos/benchmarking.py --algorithms all
```

Outputs default to `test/bench/algos/logs/benchmarking_<timestamp>.log` and
`test/bench/algos/logs/benchmarking_<timestamp>.csv`.

## Arguments

| Argument | Default | Notes |
| --- | --- | --- |
| `--run-name` | `manual` | Label written to logs. |
| `--min-seq-len` | `1024` | Smallest sequence length. |
| `--max-seq-len` | `131072` | Largest sequence length. |
| `--state-dim` | `4` | Hidden/state dimension. |
| `--input-dim` | `3` | Driver/input dimension. |
| `--dtype` | `float64` | Torch dtype name. |
| `--device` | `cuda` | Torch device string. |
| `--warmup` | `1` | Warmup runs per case. |
| `--repeats` | `3` | Timed runs per case. |
| `--seed` | `0` | Random seed. |
| `--algorithms` | `all` | Comma list: `sequential`, `deer`, `quasi_deer`, `picard`, `jacobi`, `elk`, `quasi_elk`, `all`. |
| `--scan-backend` | `torch` | Choices: `torch`, `accel_scan`. |
| `--accel-module` | `warp` | Choices: `warp`, `scalar`, `ref`. |
| `--elk-sigmasq` | `1e8` | Full ELK covariance scale. |
| `--quasi-elk-sigmasq` | `1e8` | Quasi-ELK covariance scale. |
| `--elk-process-noise` | `1.0` | ELK process noise. |
| `--max-iters-deer` | auto | DEER iteration cap. |
| `--max-iters-quasi-deer` | auto | Quasi-DEER iteration cap. |
| `--max-iters-picard` | `256` | Picard iteration cap. |
| `--max-iters-jacobi` | `256` | Jacobi iteration cap. |
| `--max-iters-elk` | `64` | ELK iteration cap. |
| `--max-iters-quasi-elk` | `64` | Quasi-ELK iteration cap. |
| `--tol` | `1e-12` | Solver tolerance. |
| `--clip-value` | `1e8` | State clipping value. |
| `--stopping-criterion` | `update` | Choices: `update`, `merit`. |
| `--strict-tol` | off | Use `--tol` exactly. |
| `--valid-final-merit-threshold` | `1e-6` | Validity threshold. |
| `--valid-error-threshold` | `1e-4` | Error threshold. |
| `--stop-on-error` | off | Raise instead of logging invalid rows. |
| `--log-file` | timestamped | Override log path. |
| `--csv-file` | timestamped | Override CSV path. |

## Wrapper

```bash
bash test/bench/algos/benchmarking.sh
```

The wrapper runs quick and full passes for both `torch` and `accel_scan`, then
writes logs and CSVs under `test/bench/algos/logs`.

Common environment overrides:

| Variable | Default | Notes |
| --- | --- | --- |
| `RUN_QUICK_TORCH` | `1` | Enable quick `torch` run. |
| `RUN_QUICK_ACCEL` | `1` | Enable quick `accel_scan` run. |
| `RUN_FULL_TORCH` | `1` | Enable full `torch` run. |
| `RUN_FULL_ACCEL` | `1` | Enable full `accel_scan` run. |
| `ALGORITHMS` | `all` | Same values as `--algorithms`. |
| `ACCEL_MODULE` | `warp` | `warp`, `scalar`, or `ref`. |
| `QUICK_MAX_SEQ_LEN` | `8192` | Quick upper sequence length. |
| `FULL_MAX_SEQ_LEN` | `131072` | Full upper sequence length. |
| `TORCH_DTYPE` | `float64` | Dtype for `torch` runs. |
| `ACCEL_DTYPE` | `float32` | Dtype for accelerated runs. |

Example:

```bash
RUN_FULL_TORCH=0 RUN_FULL_ACCEL=0 ALGORITHMS=deer,elk \
  bash test/bench/algos/benchmarking.sh
```
