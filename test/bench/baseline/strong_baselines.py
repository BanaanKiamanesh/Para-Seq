from __future__ import annotations

import argparse
import csv
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Literal

import torch

from src.pararnn import ParaGRU, ParaLSTM, ParaRNN
from src.pararnn.cells.para_gru import make_paragru_deer_config
from src.pararnn.cells.para_lstm import make_paralstm_deer_config, make_paralstm_elk_config
from src.pararnn.cells.para_rnn import make_pararnn_deer_config, make_pararnn_elk_config


Family = Literal["paragru", "pararnn", "paralstm"]


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    family: Family
    solver_label: str
    factory: Callable[[int, int, torch.device, torch.dtype], torch.nn.Module]
    reference_factory: Callable[[int, int, torch.device, torch.dtype], torch.nn.Module]
    supports_backward: bool = True


@dataclass
class BenchmarkResult:
    name: str
    family: str
    solver_label: str
    device: str
    dtype: str
    batch_size: int
    seq_len: int
    input_size: int
    hidden_size: int
    repeats: int
    warmups: int
    forward_median_s: float
    forward_min_s: float
    fw_bw_median_s: float | None
    fw_bw_min_s: float | None
    max_error_output: float
    max_error_hidden: float
    max_error_cell: float | None
    final_merit: float | None
    grad_finite: bool | None
    param_grad_finite: bool | None
    cuda_peak_memory_bytes: int | None
    valid_solution: bool
    status: str


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _cuda_peak_memory(device: torch.device) -> int | None:
    if device.type != "cuda":
        return None
    return int(torch.cuda.max_memory_allocated(device))


def _reset_cuda_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _median_min(times: list[float]) -> tuple[float, float]:
    if not times:
        return float("nan"), float("nan")
    return float(statistics.median(times)), float(min(times))


def _time_repeated(
    fn: Callable[[], None],
    *,
    device: torch.device,
    repeats: int,
    warmups: int,
) -> tuple[float, float, int | None]:
    for _ in range(warmups):
        fn()
    _sync(device)

    times: list[float] = []
    peak = 0

    for _ in range(repeats):
        _reset_cuda_peak_memory(device)
        _sync(device)
        start = time.perf_counter()
        fn()
        _sync(device)
        elapsed = time.perf_counter() - start
        times.append(float(elapsed))

        mem = _cuda_peak_memory(device)
        if mem is not None:
            peak = max(peak, mem)

    median_s, min_s = _median_min(times)
    return median_s, min_s, peak if device.type == "cuda" else None


def _make_gru_reference(input_size: int, hidden_size: int, device: torch.device, dtype: torch.dtype):
    return ParaGRU(
        input_size=input_size,
        hidden_size=hidden_size,
        batch_first=True,
        mode="sequential",
        solver="deer",
        dtype=dtype,
        device=device,
    )


def _make_rnn_reference(input_size: int, hidden_size: int, device: torch.device, dtype: torch.dtype):
    return ParaRNN(
        input_size=input_size,
        hidden_size=hidden_size,
        batch_first=True,
        mode="sequential",
        solver="deer",
        backend="autograd",
        dtype=dtype,
        device=device,
    )


def _make_lstm_reference(input_size: int, hidden_size: int, device: torch.device, dtype: torch.dtype):
    return ParaLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        batch_first=True,
        mode="sequential",
        solver="deer",
        dtype=dtype,
        device=device,
        recurrent_init_scale=0.025,
        peephole_init_scale=0.025,
        input_init_scale=0.10,
        forget_bias_init_value=0.12,
    )


def build_strong_baseline_cases(*, num_iters: int = 8) -> list[BenchmarkCase]:
    """Return the benchmark catalog.

    The catalog compares each parallel solver against a sequential model with
    identical parameters. The benchmark is deliberately small and honest:
    it records time, solution error, final merit, gradient finiteness, and CUDA
    peak memory when CUDA is available.
    """

    def gru_seq(input_size, hidden_size, device, dtype):
        return _make_gru_reference(input_size, hidden_size, device, dtype)

    def gru_deer_autograd(input_size, hidden_size, device, dtype):
        return ParaGRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            mode="deer",
            solver="deer",
            deer_config=make_paragru_deer_config(
                backend="autograd",
                num_iters=num_iters,
                tol=1e-10,
                strict_tol=True,
                scan_backend="torch",
            ),
            dtype=dtype,
            device=device,
        )

    def gru_deer_adjoint(input_size, hidden_size, device, dtype):
        return ParaGRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            mode="deer",
            solver="deer",
            deer_config=make_paragru_deer_config(
                backend="adjoint",
                num_iters=num_iters,
                tol=1e-10,
                strict_tol=True,
                scan_backend="torch",
            ),
            dtype=dtype,
            device=device,
        )

    def gru_quasi_elk(input_size, hidden_size, device, dtype):
        from src.pararnn.cells.para_gru import make_paragru_elk_config

        return ParaGRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            mode="elk",
            solver="elk",
            deer_config=make_paragru_elk_config(
                num_iters=num_iters,
                tol=1e-10,
                strict_tol=True,
                scan_backend="torch",
                sigmasq=1e8,
                process_noise=1.0,
            ),
            dtype=dtype,
            device=device,
        )

    def rnn_seq(input_size, hidden_size, device, dtype):
        return _make_rnn_reference(input_size, hidden_size, device, dtype)

    def rnn_full_deer(input_size, hidden_size, device, dtype):
        return ParaRNN(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            mode="deer",
            solver="deer",
            deer_config=make_pararnn_deer_config(
                backend="autograd",
                num_iters=num_iters,
                tol=1e-10,
                strict_tol=True,
            ),
            dtype=dtype,
            device=device,
        )

    def rnn_quasi_deer(input_size, hidden_size, device, dtype):
        return ParaRNN(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            mode="deer",
            solver="deer",
            deer_config=make_pararnn_deer_config(
                backend="quasi_autograd",
                num_iters=num_iters,
                tol=1e-10,
                strict_tol=True,
                scan_backend="torch",
            ),
            dtype=dtype,
            device=device,
        )

    def rnn_full_elk(input_size, hidden_size, device, dtype):
        return ParaRNN(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            mode="elk",
            solver="elk",
            deer_config=make_pararnn_elk_config(
                backend="elk",
                num_iters=num_iters,
                tol=1e-10,
                strict_tol=True,
                scan_backend="torch",
                sigmasq=1e8,
                process_noise=1.0,
            ),
            dtype=dtype,
            device=device,
        )

    def rnn_quasi_elk(input_size, hidden_size, device, dtype):
        return ParaRNN(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            mode="elk",
            solver="quasi_elk",
            deer_config=make_pararnn_elk_config(
                backend="quasi_elk",
                num_iters=num_iters,
                tol=1e-10,
                strict_tol=True,
                scan_backend="torch",
                sigmasq=1e8,
                process_noise=1.0,
            ),
            dtype=dtype,
            device=device,
        )

    def lstm_seq(input_size, hidden_size, device, dtype):
        return _make_lstm_reference(input_size, hidden_size, device, dtype)

    def lstm_block_deer_autograd(input_size, hidden_size, device, dtype):
        return ParaLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            mode="deer",
            solver="deer",
            deer_config=make_paralstm_deer_config(
                backend="autograd",
                num_iters=num_iters,
                tol=1e-10,
                strict_tol=True,
            ),
            dtype=dtype,
            device=device,
            recurrent_init_scale=0.025,
            peephole_init_scale=0.025,
            input_init_scale=0.10,
            forget_bias_init_value=0.12,
        )

    def lstm_block_deer_adjoint(input_size, hidden_size, device, dtype):
        return ParaLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            mode="deer",
            solver="deer",
            deer_config=make_paralstm_deer_config(
                backend="adjoint",
                num_iters=num_iters,
                tol=1e-10,
                strict_tol=True,
            ),
            dtype=dtype,
            device=device,
            recurrent_init_scale=0.025,
            peephole_init_scale=0.025,
            input_init_scale=0.10,
            forget_bias_init_value=0.12,
        )

    def lstm_quasi_deer(input_size, hidden_size, device, dtype):
        return ParaLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            mode="deer",
            solver="deer",
            deer_config=make_paralstm_deer_config(
                backend="quasi_autograd",
                num_iters=num_iters,
                tol=1e-10,
                strict_tol=True,
                scan_backend="torch",
            ),
            dtype=dtype,
            device=device,
            recurrent_init_scale=0.025,
            peephole_init_scale=0.025,
            input_init_scale=0.10,
            forget_bias_init_value=0.12,
        )

    def lstm_quasi_elk(input_size, hidden_size, device, dtype):
        return ParaLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            mode="elk",
            solver="quasi_elk",
            deer_config=make_paralstm_elk_config(
                num_iters=num_iters,
                tol=1e-10,
                strict_tol=True,
                scan_backend="torch",
                sigmasq=1e8,
                process_noise=1.0,
            ),
            dtype=dtype,
            device=device,
            recurrent_init_scale=0.025,
            peephole_init_scale=0.025,
            input_init_scale=0.10,
            forget_bias_init_value=0.12,
        )

    return [
        BenchmarkCase("paragru_sequential", "paragru", "sequential", gru_seq, _make_gru_reference),
        BenchmarkCase("paragru_deer_autograd", "paragru", "DEER autograd", gru_deer_autograd, _make_gru_reference),
        BenchmarkCase("paragru_deer_adjoint", "paragru", "DEER adjoint", gru_deer_adjoint, _make_gru_reference),
        BenchmarkCase("paragru_quasi_elk", "paragru", "quasi-ELK", gru_quasi_elk, _make_gru_reference),
        BenchmarkCase("pararnn_sequential", "pararnn", "sequential", rnn_seq, _make_rnn_reference),
        BenchmarkCase("pararnn_full_deer", "pararnn", "full DEER", rnn_full_deer, _make_rnn_reference),
        BenchmarkCase("pararnn_quasi_deer", "pararnn", "quasi-DEER", rnn_quasi_deer, _make_rnn_reference),
        BenchmarkCase("pararnn_full_elk", "pararnn", "full ELK", rnn_full_elk, _make_rnn_reference),
        BenchmarkCase("pararnn_quasi_elk", "pararnn", "quasi-ELK", rnn_quasi_elk, _make_rnn_reference),
        BenchmarkCase("paralstm_sequential", "paralstm", "sequential", lstm_seq, _make_lstm_reference),
        BenchmarkCase("paralstm_block_deer_autograd", "paralstm", "block-DEER autograd", lstm_block_deer_autograd, _make_lstm_reference),
        BenchmarkCase("paralstm_block_deer_adjoint", "paralstm", "block-DEER adjoint", lstm_block_deer_adjoint, _make_lstm_reference),
        BenchmarkCase("paralstm_quasi_deer", "paralstm", "quasi-DEER", lstm_quasi_deer, _make_lstm_reference),
        BenchmarkCase("paralstm_quasi_elk", "paralstm", "quasi-ELK", lstm_quasi_elk, _make_lstm_reference),
    ]


def _contract_model(model: torch.nn.Module) -> None:
    with torch.no_grad():
        if isinstance(model, ParaGRU):
            model.A.mul_(0.035)
            model.B.mul_(0.10)
            if hasattr(model, "b") and model.b is not None:
                model.b.mul_(0.02)

        elif isinstance(model, ParaRNN):
            model.weight_hh.mul_(0.035)
            model.weight_ih.mul_(0.10)
            if model.bias_ih is not None:
                model.bias_ih.mul_(0.02)
            if model.bias_hh is not None:
                model.bias_hh.mul_(0.02)

        elif isinstance(model, ParaLSTM):
            model.A.mul_(0.025)
            model.B.mul_(0.10)
            model.C.mul_(0.025)
            if hasattr(model, "b") and model.b is not None:
                model.b.mul_(0.02)
                model.b[0].fill_(0.12)


def _make_inputs(
    family: Family,
    *,
    batch_size: int,
    seq_len: int,
    input_size: int,
    hidden_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, object]:
    x = 0.10 * torch.randn(
        batch_size,
        seq_len,
        input_size,
        device=device,
        dtype=dtype,
    )

    if family == "paralstm":
        h0 = 0.02 * torch.randn(1, batch_size, hidden_size, device=device, dtype=dtype)
        c0 = 0.02 * torch.randn(1, batch_size, hidden_size, device=device, dtype=dtype)
        return x, (h0, c0)

    h0 = 0.02 * torch.randn(1, batch_size, hidden_size, device=device, dtype=dtype)
    return x, h0


def _clone_hx(hx: object, *, requires_grad: bool) -> object:
    if isinstance(hx, tuple):
        return tuple(t.detach().clone().requires_grad_(requires_grad) for t in hx)
    assert isinstance(hx, torch.Tensor)
    return hx.detach().clone().requires_grad_(requires_grad)


def _run_gru_or_rnn_model(model: torch.nn.Module, x: torch.Tensor, hx: object):
    """Run ParaGRU/ParaRNN using benchmark-safe hidden-state dispatch.

    The public sequence API accepts hx with PyTorch shape (1, B, H), while the
    base-cell methods accept initial_state with shape (B, H). Some local patched
    class versions can route model(x, hx) through BaseParaRNNCell.forward, so the
    benchmark avoids that ambiguity and dispatches explicitly by mode.
    """
    if not isinstance(hx, torch.Tensor):
        raise TypeError("ParaGRU/ParaRNN benchmark hx must be a tensor.")

    if hx.ndim == 3:
        if hx.shape[0] != 1:
            raise ValueError(
                "Single-layer benchmark expects hx shape (1, B, H), got "
                f"{tuple(hx.shape)}."
            )
        initial_state = hx[0]
    elif hx.ndim == 2:
        initial_state = hx
    else:
        raise ValueError(
            "ParaGRU/ParaRNN benchmark hx must have shape (1, B, H) or (B, H), "
            f"got {tuple(hx.shape)}."
        )

    mode = getattr(model, "mode", "sequential")

    if mode == "sequential":
        output = model.forward_sequential(x, initial_state=initial_state)
    elif mode == "deer":
        output = model.forward_deer(x, initial_state=initial_state)
    elif mode == "elk":
        output = model.forward_elk(x, initial_state=initial_state)
    elif mode == "jacobi":
        output = model.forward_jacobi(x, initial_state=initial_state)
    elif mode == "picard":
        output = model.forward_picard(x, initial_state=initial_state)
    else:
        raise ValueError(
            f"Unknown benchmark mode {mode!r}. Expected sequential/deer/elk/jacobi/picard."
        )

    if hasattr(model, "_make_h_n"):
        h_n = model._make_h_n(output, unbatched_input=False)
    else:
        h_n = output[:, -1, :].unsqueeze(0)

    return output, h_n


def _run_lstm_model(model: torch.nn.Module, x: torch.Tensor, hx: object):
    """Run ParaLSTM using benchmark-safe hidden-state dispatch.

    LSTM public hx is (h0, c0), each with PyTorch shape (1, B, H). Internally
    the ParaLSTM solver uses concat(c0, h0) with shape (B, 2H).
    """
    if not isinstance(hx, tuple) or len(hx) != 2:
        raise TypeError("ParaLSTM benchmark hx must be a tuple (h0, c0).")

    h0, c0 = hx

    if h0.ndim == 3:
        if h0.shape[0] != 1:
            raise ValueError(
                "Single-layer benchmark expects h0 shape (1, B, H), got "
                f"{tuple(h0.shape)}."
            )
        h_init = h0[0]
    elif h0.ndim == 2:
        h_init = h0
    else:
        raise ValueError(f"h0 must have shape (1, B, H) or (B, H), got {tuple(h0.shape)}.")

    if c0.ndim == 3:
        if c0.shape[0] != 1:
            raise ValueError(
                "Single-layer benchmark expects c0 shape (1, B, H), got "
                f"{tuple(c0.shape)}."
            )
        c_init = c0[0]
    elif c0.ndim == 2:
        c_init = c0
    else:
        raise ValueError(f"c0 must have shape (1, B, H) or (B, H), got {tuple(c0.shape)}.")

    initial_state = torch.cat([c_init, h_init], dim=-1)

    x_batched, had_batch_dim = model._normalize_input(x)
    unbatched_input = not had_batch_dim

    mode = getattr(model, "mode", "sequential")

    if mode == "sequential":
        states = model.batched_sequential_rollout(
            initial_state=initial_state,
            drivers=x_batched,
        )
    elif mode == "deer":
        states = model.forward_deer_states(
            x_batched=x_batched,
            initial_state=initial_state,
            deer_config=model.config.deer,
        )
    elif mode == "elk":
        states = model.forward_elk_states(
            x_batched=x_batched,
            initial_state=initial_state,
            elk_config=model.config.deer,
        )
    elif mode in ("jacobi", "picard"):
        states = model.forward_fixed_point_states(
            x_batched=x_batched,
            initial_state_batched=initial_state,
            method=mode,
            fixed_config=model.config.deer,
        )
    else:
        raise ValueError(
            f"Unknown benchmark mode {mode!r}. Expected sequential/deer/elk/jacobi/picard."
        )

    output_batched = model.post_process(states)
    output = model._restore_output_layout(
        output_batched,
        had_batch_dim=had_batch_dim,
    )
    h_n, c_n = model._make_h_c_n(states, unbatched_input=unbatched_input)

    return output, (h_n, c_n)


def _run_gru_or_rnn_model(model: torch.nn.Module, x: torch.Tensor, hx: object):
    """Run ParaGRU/ParaRNN using benchmark-safe hidden-state dispatch.

    The public sequence API accepts hx with PyTorch shape (1, B, H), while the
    base-cell methods accept initial_state with shape (B, H). Some local patched
    class versions can route model(x, hx) through BaseParaRNNCell.forward, so the
    benchmark avoids that ambiguity and dispatches explicitly by mode.
    """
    if not isinstance(hx, torch.Tensor):
        raise TypeError("ParaGRU/ParaRNN benchmark hx must be a tensor.")

    if hx.ndim == 3:
        if hx.shape[0] != 1:
            raise ValueError(
                "Single-layer benchmark expects hx shape (1, B, H), got "
                f"{tuple(hx.shape)}."
            )
        initial_state = hx[0]
    elif hx.ndim == 2:
        initial_state = hx
    else:
        raise ValueError(
            "ParaGRU/ParaRNN benchmark hx must have shape (1, B, H) or (B, H), "
            f"got {tuple(hx.shape)}."
        )

    mode = getattr(model, "mode", "sequential")

    if mode == "sequential":
        output = model.forward_sequential(x, initial_state=initial_state)
    elif mode == "deer":
        output = model.forward_deer(x, initial_state=initial_state)
    elif mode == "elk":
        output = model.forward_elk(x, initial_state=initial_state)
    elif mode == "jacobi":
        output = model.forward_jacobi(x, initial_state=initial_state)
    elif mode == "picard":
        output = model.forward_picard(x, initial_state=initial_state)
    else:
        raise ValueError(
            f"Unknown benchmark mode {mode!r}. Expected sequential/deer/elk/jacobi/picard."
        )

    if hasattr(model, "_make_h_n"):
        h_n = model._make_h_n(output, unbatched_input=False)
    else:
        h_n = output[:, -1, :].unsqueeze(0)

    return output, h_n


def _run_lstm_model(model: torch.nn.Module, x: torch.Tensor, hx: object):
    """Run ParaLSTM using benchmark-safe hidden-state dispatch.

    LSTM public hx is (h0, c0), each with PyTorch shape (1, B, H). Internally
    the ParaLSTM solver uses concat(c0, h0) with shape (B, 2H).
    """
    if not isinstance(hx, tuple) or len(hx) != 2:
        raise TypeError("ParaLSTM benchmark hx must be a tuple (h0, c0).")

    h0, c0 = hx

    if h0.ndim == 3:
        if h0.shape[0] != 1:
            raise ValueError(
                "Single-layer benchmark expects h0 shape (1, B, H), got "
                f"{tuple(h0.shape)}."
            )
        h_init = h0[0]
    elif h0.ndim == 2:
        h_init = h0
    else:
        raise ValueError(f"h0 must have shape (1, B, H) or (B, H), got {tuple(h0.shape)}.")

    if c0.ndim == 3:
        if c0.shape[0] != 1:
            raise ValueError(
                "Single-layer benchmark expects c0 shape (1, B, H), got "
                f"{tuple(c0.shape)}."
            )
        c_init = c0[0]
    elif c0.ndim == 2:
        c_init = c0
    else:
        raise ValueError(f"c0 must have shape (1, B, H) or (B, H), got {tuple(c0.shape)}.")

    initial_state = torch.cat([c_init, h_init], dim=-1)

    x_batched, had_batch_dim = model._normalize_input(x)
    unbatched_input = not had_batch_dim

    mode = getattr(model, "mode", "sequential")

    if mode == "sequential":
        states = model.batched_sequential_rollout(
            initial_state=initial_state,
            drivers=x_batched,
        )
    elif mode == "deer":
        states = model.forward_deer_states(
            x_batched=x_batched,
            initial_state=initial_state,
            deer_config=model.config.deer,
        )
    elif mode == "elk":
        states = model.forward_elk_states(
            x_batched=x_batched,
            initial_state=initial_state,
            elk_config=model.config.deer,
        )
    elif mode in ("jacobi", "picard"):
        states = model.forward_fixed_point_states(
            x_batched=x_batched,
            initial_state_batched=initial_state,
            method=mode,
            fixed_config=model.config.deer,
        )
    else:
        raise ValueError(
            f"Unknown benchmark mode {mode!r}. Expected sequential/deer/elk/jacobi/picard."
        )

    output_batched = model.post_process(states)
    output = model._restore_output_layout(
        output_batched,
        had_batch_dim=had_batch_dim,
    )
    h_n, c_n = model._make_h_c_n(states, unbatched_input=unbatched_input)

    return output, (h_n, c_n)


def _run_model(model: torch.nn.Module, x: torch.Tensor, hx: object):
    if isinstance(model, ParaLSTM):
        return _run_lstm_model(model, x, hx)

    if isinstance(model, (ParaGRU, ParaRNN)):
        return _run_gru_or_rnn_model(model, x, hx)

    if isinstance(hx, tuple):
        return model(x, hx)

    return model(x, hx)

def _compare_outputs(family: Family, actual, expected) -> tuple[float, float, float | None]:
    y_actual = actual[0]
    y_expected = expected[0]

    max_output = float(torch.max(torch.abs(y_actual - y_expected)).item())

    if family == "paralstm":
        h_actual, c_actual = actual[1]
        h_expected, c_expected = expected[1]
        max_hidden = float(torch.max(torch.abs(h_actual - h_expected)).item())
        max_cell = float(torch.max(torch.abs(c_actual - c_expected)).item())
        return max_output, max_hidden, max_cell

    h_actual = actual[1]
    h_expected = expected[1]
    max_hidden = float(torch.max(torch.abs(h_actual - h_expected)).item())
    return max_output, max_hidden, None


def _loss_from_output(family: Family, output) -> torch.Tensor:
    y = output[0]
    loss = y.square().mean()

    if family == "paralstm":
        h, c = output[1]
        return loss + 0.25 * h.square().mean() + 0.25 * c.square().mean()

    h = output[1]
    return loss + 0.25 * h.square().mean()


def _all_param_grads_finite(model: torch.nn.Module) -> bool:
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    if not grads:
        return True
    return all(g is not None and torch.isfinite(g).all().item() for g in grads)


def _input_grads_finite(x: torch.Tensor, hx: object) -> bool:
    tensors = [x]

    if isinstance(hx, tuple):
        tensors.extend(hx)
    elif isinstance(hx, torch.Tensor):
        tensors.append(hx)

    return all(t.grad is not None and torch.isfinite(t.grad).all().item() for t in tensors)


def _final_merit(model: torch.nn.Module) -> float | None:
    infos = getattr(model, "last_deer_infos", None)
    if not infos:
        return None

    value = infos[-1].get("final_merit", None)
    if value is None:
        return None

    if torch.is_tensor(value):
        return float(value.detach().cpu().item())

    try:
        return float(value)
    except Exception:
        return None


def run_one_case(
    case: BenchmarkCase,
    *,
    batch_size: int,
    seq_len: int,
    input_size: int,
    hidden_size: int,
    repeats: int,
    warmups: int,
    device: torch.device,
    dtype: torch.dtype,
    include_backward: bool,
    valid_tol: float,
) -> BenchmarkResult:
    torch.manual_seed(12345)

    reference = case.reference_factory(input_size, hidden_size, device, dtype)
    _contract_model(reference)

    model = case.factory(input_size, hidden_size, device, dtype)
    model.load_state_dict(reference.state_dict())
    model.to(device=device, dtype=dtype)

    x_base, hx_base = _make_inputs(
        case.family,
        batch_size=batch_size,
        seq_len=seq_len,
        input_size=input_size,
        hidden_size=hidden_size,
        device=device,
        dtype=dtype,
    )

    reference.eval()
    model.eval()

    with torch.no_grad():
        expected = _run_model(reference, x_base, hx_base)
        actual = _run_model(model, x_base, hx_base)

    max_error_output, max_error_hidden, max_error_cell = _compare_outputs(
        case.family,
        actual,
        expected,
    )

    def forward_only() -> None:
        with torch.no_grad():
            _run_model(model, x_base, hx_base)

    forward_median_s, forward_min_s, forward_peak_mem = _time_repeated(
        forward_only,
        device=device,
        repeats=repeats,
        warmups=warmups,
    )

    fw_bw_median_s = None
    fw_bw_min_s = None
    fw_bw_peak_mem = None
    grad_finite = None
    param_grad_finite = None

    if include_backward and case.supports_backward:
        model.train()

        def fw_bw() -> None:
            model.zero_grad(set_to_none=True)
            x = x_base.detach().clone().requires_grad_(True)
            hx = _clone_hx(hx_base, requires_grad=True)
            output = _run_model(model, x, hx)
            loss = _loss_from_output(case.family, output)
            loss.backward()

        fw_bw_median_s, fw_bw_min_s, fw_bw_peak_mem = _time_repeated(
            fw_bw,
            device=device,
            repeats=repeats,
            warmups=warmups,
        )

        model.zero_grad(set_to_none=True)
        x = x_base.detach().clone().requires_grad_(True)
        hx = _clone_hx(hx_base, requires_grad=True)
        output = _run_model(model, x, hx)
        loss = _loss_from_output(case.family, output)
        loss.backward()

        grad_finite = _input_grads_finite(x, hx)
        param_grad_finite = _all_param_grads_finite(model)
        model.eval()

    final_merit = _final_merit(model)
    peak_memory = max(
        mem for mem in [forward_peak_mem, fw_bw_peak_mem] if mem is not None
    ) if device.type == "cuda" else None

    valid_solution = (
        math.isfinite(max_error_output)
        and math.isfinite(max_error_hidden)
        and max_error_output <= valid_tol
        and max_error_hidden <= valid_tol
        and (max_error_cell is None or max_error_cell <= valid_tol)
    )

    status = "ok" if valid_solution else "large_error"

    return BenchmarkResult(
        name=case.name,
        family=case.family,
        solver_label=case.solver_label,
        device=str(device),
        dtype=str(dtype).replace("torch.", ""),
        batch_size=batch_size,
        seq_len=seq_len,
        input_size=input_size,
        hidden_size=hidden_size,
        repeats=repeats,
        warmups=warmups,
        forward_median_s=forward_median_s,
        forward_min_s=forward_min_s,
        fw_bw_median_s=fw_bw_median_s,
        fw_bw_min_s=fw_bw_min_s,
        max_error_output=max_error_output,
        max_error_hidden=max_error_hidden,
        max_error_cell=max_error_cell,
        final_merit=final_merit,
        grad_finite=grad_finite,
        param_grad_finite=param_grad_finite,
        cuda_peak_memory_bytes=peak_memory,
        valid_solution=valid_solution,
        status=status,
    )


def run_strong_baseline_benchmark(
    *,
    case_names: Iterable[str] | None = None,
    batch_size: int = 4,
    seq_len: int = 64,
    input_size: int = 16,
    hidden_size: int = 32,
    repeats: int = 5,
    warmups: int = 2,
    num_iters: int = 8,
    include_backward: bool = True,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    valid_tol: float = 5e-3,
    output_csv: str | Path | None = None,
) -> list[BenchmarkResult]:
    selected_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    cases = build_strong_baseline_cases(num_iters=num_iters)
    if case_names is not None:
        wanted = set(case_names)
        cases = [case for case in cases if case.name in wanted]

    if not cases:
        raise ValueError("No benchmark cases selected.")

    results: list[BenchmarkResult] = []

    for case in cases:
        try:
            result = run_one_case(
                case,
                batch_size=batch_size,
                seq_len=seq_len,
                input_size=input_size,
                hidden_size=hidden_size,
                repeats=repeats,
                warmups=warmups,
                device=selected_device,
                dtype=dtype,
                include_backward=include_backward,
                valid_tol=valid_tol,
            )
        except Exception as exc:
            result = BenchmarkResult(
                name=case.name,
                family=case.family,
                solver_label=case.solver_label,
                device=str(selected_device),
                dtype=str(dtype).replace("torch.", ""),
                batch_size=batch_size,
                seq_len=seq_len,
                input_size=input_size,
                hidden_size=hidden_size,
                repeats=repeats,
                warmups=warmups,
                forward_median_s=float("nan"),
                forward_min_s=float("nan"),
                fw_bw_median_s=None,
                fw_bw_min_s=None,
                max_error_output=float("nan"),
                max_error_hidden=float("nan"),
                max_error_cell=None,
                final_merit=None,
                grad_finite=None,
                param_grad_finite=None,
                cuda_peak_memory_bytes=None,
                valid_solution=False,
                status=f"error: {type(exc).__name__}: {exc}",
            )

        results.append(result)

    if output_csv is not None:
        write_results_csv(results, output_csv)

    return results


def write_results_csv(results: list[BenchmarkResult], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def _dtype_from_string(name: str) -> torch.dtype:
    normalized = name.lower()
    if normalized in ("float32", "fp32"):
        return torch.float32
    if normalized in ("float64", "fp64", "double"):
        return torch.float64
    if normalized in ("bfloat16", "bf16"):
        return torch.bfloat16
    if normalized in ("float16", "fp16", "half"):
        return torch.float16
    raise ValueError(f"Unknown dtype {name!r}.")


def _print_table(results: list[BenchmarkResult]) -> None:
    header = (
        "name,family,solver,forward_median_s,fw_bw_median_s,"
        "max_error_output,max_error_hidden,final_merit,grad_finite,status"
    )
    print(header)
    for r in results:
        print(
            f"{r.name},{r.family},{r.solver_label},"
            f"{r.forward_median_s:.6g},"
            f"{'' if r.fw_bw_median_s is None else f'{r.fw_bw_median_s:.6g}'},"
            f"{r.max_error_output:.6g},"
            f"{r.max_error_hidden:.6g},"
            f"{'' if r.final_merit is None else f'{r.final_merit:.6g}'},"
            f"{r.grad_finite},"
            f"{r.status}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Strong ParaRNN baseline benchmark.")
    parser.add_argument("--case", action="append", dest="cases", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--input-size", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--num-iters", type=int, default=8)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--no-backward", action="store_true")
    parser.add_argument("--valid-tol", type=float, default=5e-3)
    parser.add_argument(
        "--output-csv",
        type=str,
        default=f"test/bench/baseline/logs/strong_baselines_{int(time.time())}.csv",
    )

    args = parser.parse_args()

    results = run_strong_baseline_benchmark(
        case_names=args.cases,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        repeats=args.repeats,
        warmups=args.warmups,
        num_iters=args.num_iters,
        include_backward=not args.no_backward,
        device=args.device,
        dtype=_dtype_from_string(args.dtype),
        valid_tol=args.valid_tol,
        output_csv=args.output_csv,
    )

    _print_table(results)
    print(f"\nWrote CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
