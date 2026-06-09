# Para-Seq

Para-Seq is a research-oriented PyTorch codebase for parallelizing nonlinear recurrent and sequential models over the sequence length.

The project implements fixed-point and Newton-style algorithms such as **Jacobi, Picard, DEER, quasi-DEER, ELK, and quasi-ELK**, then wraps them into PyTorch-style recurrent layers. The goal is to keep the familiar RNN/GRU/LSTM interface while replacing the usual sequential unroll with scan-based parallel solvers whenever possible.

## What is included

```text
src/
├── algos/        # DEER, ELK, Jacobi, Picard, fixed-point solvers
├── pararnn/      # PyTorch-style ParaRNN, ParaGRU, ParaLSTM layers
└── utils/        # associative scans, accelerated_scan wrappers, adjoint scans

test/
├── pararnn/      # correctness, gradient, backend, and layer API tests
├── utils/        # scan backend tests
└── bench/        # benchmarking scripts and logs

notebooks/        # small demos and Sequential MNIST experiments
```

## Main features

| Component   | Description                                                                                                  |
| ----------- | ------------------------------------------------------------------------------------------------------------ |
| `ParaRNN`   | Vanilla tanh/relu RNN with sequential and DEER-style modes.                                                  |
| `ParaGRU`   | Diagonal ParaGRU with quasi-DEER, optional adjoint backward, and optional `accelerated_scan` support.        |
| `ParaLSTM`  | CIFG/peephole-style ParaLSTM with structured block and diagonal scan backends.                               |
| `src.algos` | Standalone solvers for sequential rollout, Jacobi, Picard, DEER, quasi-DEER, ELK, and quasi-ELK experiments. |
| `src.utils` | Dense, diagonal, and 2x2 block associative scans, plus reverse adjoint scans.                                |

## Installation

This repo currently uses a source-layout workflow, so run scripts from the repository root or make sure the repo root is on `PYTHONPATH`.

```bash
git clone <repo-url>
cd Para-Seq

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

The CUDA accelerated scan path requires a CUDA build of PyTorch and the `accelerated-scan` package. If the PyTorch version in `requirements.txt` does not match your machine, install the correct PyTorch wheel for your CUDA setup first.

## Quick example

```python
import torch
from src.pararnn import ParaGRU

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rnn = ParaGRU(
    input_size=16,
    hidden_size=64,
    batch_first=True,
    mode="deer",
    backend="adjoint",
    scan_backend="torch",
    num_iters=4,
    dtype=torch.float32,
).to(device)

x = torch.randn(8, 512, 16, device=device)

output, h_n = rnn(x)

print(output.shape)  # (8, 512, 64)
print(h_n.shape)     # (1, 8, 64)
```

For the CUDA accelerated diagonal scan backend:

```python
rnn = ParaGRU(
    input_size=16,
    hidden_size=64,
    batch_first=True,
    mode="deer",
    backend="adjoint",
    scan_backend="accel_scan",
    accel_module="warp",
    num_iters=4,
).to("cuda")
```

## Running tests

```bash
pytest test/pararnn test/utils -q
```

The tests check forward correctness, gradient equivalence, layer API behavior, scan backends, adjoint scans, and several DEER/ELK configurations.

## Running benchmarks

```bash
python test/bench/algos/benchmarking.py \
    --algorithms sequential,deer,quasi_deer,picard,jacobi,elk,quasi_elk \
    --device cuda \
    --dtype float32 \
    --scan-backend accel_scan
```

Benchmark logs and CSV files are written under:

```text
test/bench/algos/logs/
```

## Demos

The `notebooks/` folder contains small training and comparison demos, including:

* synthetic drift classification,
* synthetic memory with ParaLSTM,
* Sequential MNIST and permuted Sequential MNIST experiments,
* ParaGRU and ParaLSTM examples using quasi-DEER and `accelerated_scan`.

## Notes and limitations

This is research code. The implementation is useful for experimenting with parallel-in-time sequential model evaluation, but it is not yet packaged as a polished library.

Current limitations include:

* single recurrent layer only,
* unidirectional models only,
* dropout is not implemented for the ParaRNN/ParaGRU/ParaLSTM layers,
* full dense DEER is mainly practical for small hidden sizes,
* `accelerated_scan` requires CUDA,
* imports currently assume the repo root is available as a Python path.

For larger hidden sizes or long sequences, the diagonal, quasi, block-structured, or adjoint-backed paths are usually the most practical ones.

## Background

The code follows the idea that a sequential model can be written as one large nonlinear residual system. Instead of unrolling the recurrence step by step, the solvers repeatedly linearize or approximate the system and solve the resulting linear recurrence with associative scan operations.

This connects nonlinear sequential model evaluation to parallel prefix scans and linear dynamical system solvers, making it possible to experiment with sequence-parallel versions of otherwise sequential recurrent models.

## References

[1] Y. H. Lim, Q. Zhu, J. Selfridge, and M. F. Kasim, “Parallelizing non-linear sequential models over the sequence length,” in *Proc. International Conference on Learning Representations (ICLR)*, 2024. [Online]. Available: https://arxiv.org/abs/2309.12252

[2] Machine Discovery Ltd., “deer,” GitHub repository. [Online]. Available: https://github.com/machine-discovery/deer

[3] X. Gonzalez, A. Warrington, J. T. H. Smith, and S. W. Linderman, “Towards scalable and stable parallelization of nonlinear RNNs,” in *Proc. Advances in Neural Information Processing Systems (NeurIPS)*, 2024. [Online]. Available: https://arxiv.org/abs/2407.19115

[4] S. W. Linderman Lab, “elk,” GitHub repository. [Online]. Available: https://github.com/lindermanlab/elk

[5] X. Gonzalez, E. K. Buchanan, H. D. Lee, J. W. Liu, K. A. Wang, D. M. Zoltowski, L. Kozachkov, C. Ré, and S. W. Linderman, “A unifying framework for parallelizing sequential models with linear dynamical systems,” *Transactions on Machine Learning Research*, 2026. [Online]. Available: https://arxiv.org/abs/2509.21716

[6] S. W. Linderman Lab, “parallelizing_with_lds,” GitHub repository. [Online]. Available: https://github.com/lindermanlab/parallelizing_with_lds

[7] F. Danieli, P. Rodríguez, M. Sarabia, X. Suau, and L. Zappella, “ParaRNN: Unlocking parallel training of nonlinear RNNs for large language models,” Apple, 2025. [Online]. Available: https://arxiv.org/abs/2510.21450

[8] Apple, “ml-pararnn,” GitHub repository, 2025. [Online]. Available: https://github.com/apple/ml-pararnn
