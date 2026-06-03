import torch

from src.pararnn import ParaGRU, ParaLSTM, ParaRNN


def _assert_close(name, actual, expected, tol):
    error = torch.max(torch.abs(actual - expected)).item()
    assert error < tol, f"{name}: max abs error {error} >= {tol}"


def test_paragru_elk_layer_api_matches_sequential_small_contracting_case():
    torch.manual_seed(3001)

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
        num_iters=12,
        tol=1e-12,
        strict_tol=True,
        dtype=torch.float64,
    )

    with torch.no_grad():
        seq.A.mul_(0.04)
        seq.B.mul_(0.10)
        seq.b.mul_(0.02)

    elk.load_state_dict(seq.state_dict())

    x = 0.10 * torch.randn(2, 6, 2, dtype=torch.float64)
    h0 = 0.02 * torch.randn(1, 2, 3, dtype=torch.float64)

    y_seq, hn_seq = seq(x, h0)
    y_elk, hn_elk = elk(x, h0)

    _assert_close("ParaGRU ELK output", y_elk, y_seq, 5e-5)
    _assert_close("ParaGRU ELK h_n", hn_elk, hn_seq, 5e-5)

    info = elk.last_deer_infos[-1]
    assert info["solver"] == "elk"
    assert info["quasi"] is True


def test_pararnn_full_elk_layer_api_matches_sequential_small_contracting_case():
    torch.manual_seed(3002)

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
        num_iters=12,
        tol=1e-12,
        strict_tol=True,
        dtype=torch.float64,
    )

    with torch.no_grad():
        seq.weight_hh.mul_(0.04)
        seq.weight_ih.mul_(0.12)
        seq.bias_ih.mul_(0.02)
        seq.bias_hh.mul_(0.02)

    elk.load_state_dict(seq.state_dict())

    x = 0.10 * torch.randn(2, 6, 2, dtype=torch.float64)
    h0 = 0.02 * torch.randn(1, 2, 3, dtype=torch.float64)

    y_seq, hn_seq = seq(x, h0)
    y_elk, hn_elk = elk(x, h0)

    _assert_close("ParaRNN full ELK output", y_elk, y_seq, 5e-5)
    _assert_close("ParaRNN full ELK h_n", hn_elk, hn_seq, 5e-5)

    info = elk.last_deer_infos[-1]
    assert info["solver"] == "elk"
    assert info["quasi"] is False
    assert info["pararnn_elk_kind"] == "full_dense"


def test_pararnn_quasi_elk_layer_api_runs_and_backprops():
    torch.manual_seed(3003)

    model = ParaRNN(
        input_size=2,
        hidden_size=4,
        batch_first=True,
        solver="quasi_elk",
        mode="elk",
        backend="quasi_elk",
        scan_backend="torch",
        num_iters=6,
        dtype=torch.float64,
    )

    with torch.no_grad():
        model.weight_hh.mul_(0.03)
        model.weight_ih.mul_(0.10)

    x = torch.randn(2, 5, 2, dtype=torch.float64, requires_grad=True)
    y, h_n = model(x)

    loss = y.square().mean() + h_n.square().mean()
    loss.backward()

    assert y.shape == (2, 5, 4)
    assert h_n.shape == (1, 2, 4)
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(model.weight_ih.grad).all()
    assert torch.isfinite(model.weight_hh.grad).all()

    info = model.last_deer_infos[-1]
    assert info["solver"] == "elk"
    assert info["quasi"] is True
    assert info["pararnn_elk_kind"] == "scalar_quasi"


def test_paralstm_quasi_elk_layer_api_runs_and_backprops():
    torch.manual_seed(3004)

    model = ParaLSTM(
        input_size=2,
        hidden_size=3,
        batch_first=True,
        solver="quasi_elk",
        mode="elk",
        scan_backend="torch",
        num_iters=6,
        dtype=torch.float64,
        recurrent_init_scale=0.025,
        peephole_init_scale=0.025,
        forget_bias_init_value=0.12,
    )

    x = torch.randn(2, 5, 2, dtype=torch.float64, requires_grad=True)
    y, (h_n, c_n) = model(x)

    loss = y.square().mean() + h_n.square().mean() + c_n.square().mean()
    loss.backward()

    assert y.shape == (2, 5, 3)
    assert h_n.shape == (1, 2, 3)
    assert c_n.shape == (1, 2, 3)
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(model.A.grad).all()
    assert torch.isfinite(model.B.grad).all()
    assert torch.isfinite(model.C.grad).all()
    assert torch.isfinite(model.b.grad).all()

    info = model.last_deer_infos[-1]
    assert info["solver"] == "elk"
    assert info["quasi"] is True
    assert info["paralstm_elk_kind"] == "scalar_quasi"
