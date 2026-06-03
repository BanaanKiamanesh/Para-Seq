import torch

from src.pararnn import ParaLSTM


def _assert_close(name, actual, expected, tol):
    error = torch.max(torch.abs(actual - expected)).item()
    assert error < tol, f"{name}: max abs error {error} >= {tol}"


def _collect_gradients(model, x_base, h0_base, c0_base):
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


def test_paralstm_block_adjoint_gradients_match_sequential():
    torch.manual_seed(2001)

    seq_lstm = ParaLSTM(
        input_size=2,
        hidden_size=3,
        batch_first=True,
        mode="sequential",
        backend="adjoint",
        num_iters=40,
        tol=1e-12,
        strict_tol=True,
        dtype=torch.float64,
        recurrent_init_scale=0.025,
        peephole_init_scale=0.025,
        forget_bias_init_value=0.12,
    )

    adjoint_lstm = ParaLSTM(
        input_size=2,
        hidden_size=3,
        batch_first=True,
        mode="deer",
        backend="adjoint",
        num_iters=40,
        tol=1e-12,
        strict_tol=True,
        dtype=torch.float64,
        recurrent_init_scale=0.025,
        peephole_init_scale=0.025,
        forget_bias_init_value=0.12,
    )

    adjoint_lstm.load_state_dict(seq_lstm.state_dict())

    x = 0.08 * torch.randn(2, 6, 2, dtype=torch.float64)
    h0 = 0.02 * torch.randn(1, 2, 3, dtype=torch.float64)
    c0 = 0.02 * torch.randn(1, 2, 3, dtype=torch.float64)

    sequential = _collect_gradients(seq_lstm, x, h0, c0)
    adjoint = _collect_gradients(adjoint_lstm, x, h0, c0)

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
            f"ParaLSTM block adjoint {key}",
            adjoint[key],
            sequential[key],
            2e-6,
        )

    info = adjoint_lstm.last_deer_infos[-1]
    assert info["backward_backend"] == "adjoint"
    assert info["paralstm_deer_kind"] == "block2"


def test_paralstm_block_adjoint_alias_works():
    model = ParaLSTM(
        input_size=2,
        hidden_size=3,
        batch_first=True,
        mode="deer",
        backend="block_deer_adjoint_torch",
        num_iters=3,
        dtype=torch.float64,
    )

    x = torch.randn(2, 4, 2, dtype=torch.float64)
    output, (h_n, c_n) = model(x)

    assert output.shape == (2, 4, 3)
    assert h_n.shape == (1, 2, 3)
    assert c_n.shape == (1, 2, 3)
    assert model.last_deer_infos[-1]["backward_backend"] == "adjoint"
