# Strong Baseline Benchmarks

Compares ParaRNN, ParaGRU, and ParaLSTM parallel modes against matching
sequential references. It records timing, correctness error, gradient status,
and CUDA peak memory when available.

## Run

```bash
PYTHONPATH=. python test/bench/baseline/strong_baselines.py
```

Output defaults to `test/bench/baseline/logs/strong_baselines_<timestamp>.csv`.
This folder does not currently include a bash wrapper.

## Arguments

| Argument | Default | Notes |
| --- | --- | --- |
| `--case` | all cases | Repeat to select specific cases. |
| `--batch-size` | `4` | Batch size. |
| `--seq-len` | `64` | Sequence length. |
| `--input-size` | `16` | Input dimension. |
| `--hidden-size` | `32` | Hidden dimension. |
| `--repeats` | `5` | Timed runs per case. |
| `--warmups` | `2` | Warmup runs per case. |
| `--num-iters` | `8` | Solver iterations for parallel cases. |
| `--device` | auto | Torch device string. |
| `--dtype` | `float32` | Torch dtype name. |
| `--no-backward` | off | Skip backward timing/checks. |
| `--valid-tol` | `5e-3` | Correctness threshold. |
| `--output-csv` | timestamped | Override CSV path. |

## Cases

`paragru_sequential`, `paragru_deer_autograd`, `paragru_deer_adjoint`,
`paragru_quasi_elk`, `pararnn_sequential`, `pararnn_full_deer`,
`pararnn_quasi_deer`, `pararnn_full_elk`, `pararnn_quasi_elk`,
`paralstm_sequential`, `paralstm_block_deer_autograd`,
`paralstm_block_deer_adjoint`, `paralstm_quasi_deer`, `paralstm_quasi_elk`.

Example:

```bash
PYTHONPATH=. python test/bench/baseline/strong_baselines.py \
  --case paragru_deer_adjoint --case paralstm_quasi_deer
```
