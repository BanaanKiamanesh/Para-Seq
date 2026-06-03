import torch

from src.pararnn import ParaGRU, ParaLSTM, ParaRNN


def _max_abs(actual, expected):
    return torch.max(torch.abs(actual - expected)).item()


def _assert_close(name, actual, expected, tol):
    error = _max_abs(actual, expected)
    assert error < tol, f"{name}: max abs error {error} >= {tol}"


def _clone_or_none(tensor):
    if tensor is None:
        return None
    return tensor.detach().clone()


def _collect_gru_gradients(model, x_base, h0_base):
    model.zero_grad(set_to_none=True)

    x = x_base.detach().clone().requires_grad_(True)
    h0 = h0_base.detach().clone().requires_grad_(True)

    output, h_n = model(x, h0)

    output_weight = torch.linspace(
        -0.6,
        0.8,
        output.numel(),
        device=output.device,
        dtype=output.dtype,
    ).reshape_as(output)

    hn_weight = torch.linspace(
        0.3,
        -0.2,
        h_n.numel(),
        device=h_n.device,
        dtype=h_n.dtype,
    ).reshape_as(h_n)

    loss = (output * output_weight).mean() + 0.20 * (h_n * hn_weight).sum()
    loss.backward()

    return {
        "output": output.detach(),
        "h_n": h_n.detach(),
        "x_grad": x.grad.detach().clone(),
        "h0_grad": h0.grad.detach().clone(),
        "A_grad": model.A.grad.detach().clone(),
        "B_grad": model.B.grad.detach().clone(),
        "b_grad": model.b.grad.detach().clone(),
    }


def _collect_rnn_gradients(model, x_base, h0_base):
    model.zero_grad(set_to_none=True)

    x = x_base.detach().clone().requires_grad_(True)
    h0 = h0_base.detach().clone().requires_grad_(True)

    output, h_n = model(x, h0)

    output_weight = torch.linspace(
        -0.6,
        0.8,
        output.numel(),
        device=output.device,
        dtype=output.dtype,
    ).reshape_as(output)

    hn_weight = torch.linspace(
        0.3,
        -0.2,
        h_n.numel(),
        device=h_n.device,
        dtype=h_n.dtype,
    ).reshape_as(h_n)

    loss = (output * output_weight).mean() + 0.20 * (h_n * hn_weight).sum()
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


def _collect_lstm_gradients(model, x_base, h0_base, c0_base):
    model.zero_grad(set_to_none=True)

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
        -0.2,
        h_n.numel(),
        device=h_n.device,
        dtype=h_n.dtype,
    ).reshape_as(h_n)

    cn_weight = torch.linspace(
        -0.1,
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


def test_paragru_quasi_elk_gradients_match_sequential_contracting_case():
    torch.manual_seed(6201)

    seq = ParaGRU(
        input_size=2,
        hidden_size=3,
        batch_first=True,
        mode="sequential",
        dtype=torch.float64,
    )

    elk = ParaGRU(
        input_size=2,
        hidden_size=3,
        batch_first=True,
        solver="elk",
        mode="elk",
        scan_backend="torch",
        num_iters=40,
        tol=1e-12,
        strict_tol=True,
        dtype=torch.float64,
    )

    with torch.no_grad():
        seq.A.mul_(0.03)
        seq.B.mul_(0.10)
        seq.b.mul_(0.02)

    elk.load_state_dict(seq.state_dict())

    x = 0.10 * torch.randn(2, 5, 2, dtype=torch.float64)
    h0 = 0.02 * torch.randn(1, 2, 3, dtype=torch.float64)

    sequential = _collect_gru_gradients(seq, x, h0)
    parallel = _collect_gru_gradients(elk, x, h0)

    for key in sequential:
        _assert_close(f"ParaGRU quasi-ELK {key}", parallel[key], sequential[key], 2e-5)


def test_pararnn_full_elk_gradients_match_sequential_contracting_case():
    torch.manual_seed(6202)

    seq = ParaRNN(
        input_size=2,
        hidden_size=3,
        batch_first=True,
        mode="sequential",
        backend="autograd",
        dtype=torch.float64,
    )

    elk = ParaRNN(
        input_size=2,
        hidden_size=3,
        batch_first=True,
        solver="elk",
        mode="elk",
        backend="elk",
        scan_backend="torch",
        num_iters=40,
        tol=1e-12,
        strict_tol=True,
        dtype=torch.float64,
    )

    with torch.no_grad():
        seq.weight_hh.mul_(0.03)
        seq.weight_ih.mul_(0.10)
        seq.bias_ih.mul_(0.02)
        seq.bias_hh.mul_(0.02)

    elk.load_state_dict(seq.state_dict())

    x = 0.10 * torch.randn(2, 5, 2, dtype=torch.float64)
    h0 = 0.02 * torch.randn(1, 2, 3, dtype=torch.float64)

    sequential = _collect_rnn_gradients(seq, x, h0)
    parallel = _collect_rnn_gradients(elk, x, h0)

    for key in sequential:
        _assert_close(f"ParaRNN full-ELK {key}", parallel[key], sequential[key], 2e-5)


def test_paralstm_quasi_elk_gradients_match_sequential_contracting_case():
    torch.manual_seed(6203)

    seq = ParaLSTM(
        input_size=2,
        hidden_size=3,
        batch_first=True,
        mode="sequential",
        dtype=torch.float64,
        recurrent_init_scale=0.025,
        peephole_init_scale=0.025,
        forget_bias_init_value=0.12,
    )

    elk = ParaLSTM(
        input_size=2,
        hidden_size=3,
        batch_first=True,
        solver="quasi_elk",
        mode="elk",
        scan_backend="torch",
        num_iters=40,
        tol=1e-12,
        strict_tol=True,
        dtype=torch.float64,
        recurrent_init_scale=0.025,
        peephole_init_scale=0.025,
        forget_bias_init_value=0.12,
    )

    elk.load_state_dict(seq.state_dict())

    x = 0.08 * torch.randn(2, 5, 2, dtype=torch.float64)
    h0 = 0.02 * torch.randn(1, 2, 3, dtype=torch.float64)
    c0 = 0.02 * torch.randn(1, 2, 3, dtype=torch.float64)

    sequential = _collect_lstm_gradients(seq, x, h0, c0)
    parallel = _collect_lstm_gradients(elk, x, h0, c0)

    for key in sequential:
        _assert_close(f"ParaLSTM quasi-ELK {key}", parallel[key], sequential[key], 1e-4)
