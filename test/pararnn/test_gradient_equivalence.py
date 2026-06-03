import torch

from src.pararnn import ParaLSTM, ParaRNN


def _max_abs(actual, expected):
    return torch.max(torch.abs(actual - expected)).item()


def _assert_close(name, actual, expected, tol):
    error = _max_abs(actual, expected)
    assert error < tol, f"{name}: max abs error {error} >= {tol}"


def _clone_or_none(tensor):
    if tensor is None:
        return None
    return tensor.detach().clone()


def _collect_pararnn_gradients(model, x_base, h0_base):
    x = x_base.detach().clone().requires_grad_(True)
    h0 = h0_base.detach().clone().requires_grad_(True)

    output, h_n = model(x, h0)

    output_weight = torch.linspace(
        -0.7,
        0.9,
        output.numel(),
        device=output.device,
        dtype=output.dtype,
    ).reshape_as(output)

    hn_weight = torch.linspace(
        0.4,
        -0.3,
        h_n.numel(),
        device=h_n.device,
        dtype=h_n.dtype,
    ).reshape_as(h_n)

    loss = (output * output_weight).mean() + 0.25 * (h_n * hn_weight).sum()
    loss.backward()

    return {
        "output": output.detach(),
        "h_n": h_n.detach(),
        "x_grad": x.grad.detach().clone(),
        "h0_grad": h0.grad.detach().clone(),
        "weight_ih_grad": model.weight_ih.grad.detach().clone(),
        "weight_hh_grad": model.weight_hh.grad.detach().clone(),
        "bias_ih_grad": _clone_or_none(model.bias_ih.grad),
        "bias_hh_grad": _clone_or_none(model.bias_hh.grad),
    }


def _collect_paralstm_gradients(model, x_base, h0_base, c0_base):
    x = x_base.detach().clone().requires_grad_(True)
    h0 = h0_base.detach().clone().requires_grad_(True)
    c0 = c0_base.detach().clone().requires_grad_(True)

    output, (h_n, c_n) = model(x, (h0, c0))

    output_weight = torch.linspace(
        -0.6,
        0.8,
        output.numel(),
        device=output.device,
        dtype=output.dtype,
    ).reshape_as(output)

    hn_weight = torch.linspace(
        0.3,
        -0.5,
        h_n.numel(),
        device=h_n.device,
        dtype=h_n.dtype,
    ).reshape_as(h_n)

    cn_weight = torch.linspace(
        -0.2,
        0.4,
        c_n.numel(),
        device=c_n.device,
        dtype=c_n.dtype,
    ).reshape_as(c_n)

    loss = (
        (output * output_weight).mean()
        + 0.20 * (h_n * hn_weight).sum()
        + 0.15 * (c_n * cn_weight).sum()
    )
    loss.backward()

    return {
        "output": output.detach(),
        "h_n": h_n.detach(),
        "c_n": c_n.detach(),
        "x_grad": x.grad.detach().clone(),
        "h0_grad": h0.grad.detach().clone(),
        "c0_grad": c0.grad.detach().clone(),
        "A_grad": model.A.grad.detach().clone(),
        "B_grad": model.B.grad.detach().clone(),
        "C_grad": model.C.grad.detach().clone(),
        "b_grad": model.b.grad.detach().clone(),
    }


def test_pararnn_full_deer_gradients_match_sequential():
    torch.manual_seed(1001)

    seq_rnn = ParaRNN(
        input_size=2,
        hidden_size=3,
        batch_first=True,
        mode="sequential",
        backend="autograd",
        num_iters=40,
        tol=1e-12,
        strict_tol=True,
        dtype=torch.float64,
    )

    deer_rnn = ParaRNN(
        input_size=2,
        hidden_size=3,
        batch_first=True,
        mode="deer",
        backend="autograd",
        num_iters=40,
        tol=1e-12,
        strict_tol=True,
        dtype=torch.float64,
    )

    with torch.no_grad():
        seq_rnn.weight_hh.mul_(0.07)
        seq_rnn.weight_ih.mul_(0.18)
        seq_rnn.bias_ih.mul_(0.05)
        seq_rnn.bias_hh.mul_(0.05)

    deer_rnn.load_state_dict(seq_rnn.state_dict())

    x = 0.18 * torch.randn(2, 7, 2, dtype=torch.float64)
    h0 = 0.04 * torch.randn(1, 2, 3, dtype=torch.float64)

    sequential = _collect_pararnn_gradients(seq_rnn, x, h0)
    parallel = _collect_pararnn_gradients(deer_rnn, x, h0)

    for key in (
        "output",
        "h_n",
        "x_grad",
        "h0_grad",
        "weight_ih_grad",
        "weight_hh_grad",
        "bias_ih_grad",
        "bias_hh_grad",
    ):
        _assert_close(f"ParaRNN full-DEER {key}", parallel[key], sequential[key], 2e-6)


def test_pararnn_scalar_quasi_deer_gradients_match_sequential():
    torch.manual_seed(1002)

    seq_rnn = ParaRNN(
        input_size=2,
        hidden_size=3,
        batch_first=True,
        mode="sequential",
        backend="quasi_autograd",
        num_iters=30,
        tol=1e-12,
        strict_tol=True,
        dtype=torch.float64,
    )

    deer_rnn = ParaRNN(
        input_size=2,
        hidden_size=3,
        batch_first=True,
        mode="deer",
        backend="quasi_autograd",
        scan_backend="torch",
        num_iters=30,
        tol=1e-12,
        strict_tol=True,
        dtype=torch.float64,
    )

    with torch.no_grad():
        seq_rnn.weight_hh.mul_(0.03)
        seq_rnn.weight_ih.mul_(0.13)
        seq_rnn.bias_ih.mul_(0.02)
        seq_rnn.bias_hh.mul_(0.02)

    deer_rnn.load_state_dict(seq_rnn.state_dict())

    x = 0.10 * torch.randn(2, 5, 2, dtype=torch.float64)
    h0 = 0.02 * torch.randn(1, 2, 3, dtype=torch.float64)

    sequential = _collect_pararnn_gradients(seq_rnn, x, h0)
    parallel = _collect_pararnn_gradients(deer_rnn, x, h0)

    for key in (
        "output",
        "h_n",
        "x_grad",
        "h0_grad",
        "weight_ih_grad",
        "weight_hh_grad",
        "bias_ih_grad",
        "bias_hh_grad",
    ):
        _assert_close(
            f"ParaRNN scalar quasi-DEER {key}",
            parallel[key],
            sequential[key],
            2e-6,
        )


def test_paralstm_block_deer_gradients_match_sequential():
    torch.manual_seed(1003)

    seq_lstm = ParaLSTM(
        input_size=2,
        hidden_size=3,
        batch_first=True,
        mode="sequential",
        backend="autograd",
        num_iters=40,
        tol=1e-12,
        strict_tol=True,
        dtype=torch.float64,
        recurrent_init_scale=0.035,
        peephole_init_scale=0.035,
        forget_bias_init_value=0.15,
    )

    deer_lstm = ParaLSTM(
        input_size=2,
        hidden_size=3,
        batch_first=True,
        mode="deer",
        backend="autograd",
        num_iters=40,
        tol=1e-12,
        strict_tol=True,
        dtype=torch.float64,
        recurrent_init_scale=0.035,
        peephole_init_scale=0.035,
        forget_bias_init_value=0.15,
    )

    deer_lstm.load_state_dict(seq_lstm.state_dict())

    x = 0.12 * torch.randn(2, 6, 2, dtype=torch.float64)
    h0 = 0.03 * torch.randn(1, 2, 3, dtype=torch.float64)
    c0 = 0.03 * torch.randn(1, 2, 3, dtype=torch.float64)

    sequential = _collect_paralstm_gradients(seq_lstm, x, h0, c0)
    parallel = _collect_paralstm_gradients(deer_lstm, x, h0, c0)

    for key in (
        "output",
        "h_n",
        "c_n",
        "x_grad",
        "h0_grad",
        "c0_grad",
        "A_grad",
        "B_grad",
        "C_grad",
        "b_grad",
    ):
        _assert_close(f"ParaLSTM block-DEER {key}", parallel[key], sequential[key], 2e-6)


def test_paralstm_scalar_quasi_deer_gradients_match_sequential():
    torch.manual_seed(1004)

    seq_lstm = ParaLSTM(
        input_size=2,
        hidden_size=3,
        batch_first=True,
        mode="sequential",
        backend="quasi_autograd",
        num_iters=30,
        tol=1e-12,
        strict_tol=True,
        dtype=torch.float64,
        recurrent_init_scale=0.025,
        peephole_init_scale=0.025,
        forget_bias_init_value=0.12,
    )

    deer_lstm = ParaLSTM(
        input_size=2,
        hidden_size=3,
        batch_first=True,
        mode="deer",
        backend="quasi_autograd",
        scan_backend="torch",
        num_iters=30,
        tol=1e-12,
        strict_tol=True,
        dtype=torch.float64,
        recurrent_init_scale=0.025,
        peephole_init_scale=0.025,
        forget_bias_init_value=0.12,
    )

    deer_lstm.load_state_dict(seq_lstm.state_dict())

    x = 0.08 * torch.randn(2, 5, 2, dtype=torch.float64)
    h0 = 0.02 * torch.randn(1, 2, 3, dtype=torch.float64)
    c0 = 0.02 * torch.randn(1, 2, 3, dtype=torch.float64)

    sequential = _collect_paralstm_gradients(seq_lstm, x, h0, c0)
    parallel = _collect_paralstm_gradients(deer_lstm, x, h0, c0)

    for key in (
        "output",
        "h_n",
        "c_n",
        "x_grad",
        "h0_grad",
        "c0_grad",
        "A_grad",
        "B_grad",
        "C_grad",
        "b_grad",
    ):
        _assert_close(
            f"ParaLSTM scalar quasi-DEER {key}",
            parallel[key],
            sequential[key],
            2e-6,
        )
