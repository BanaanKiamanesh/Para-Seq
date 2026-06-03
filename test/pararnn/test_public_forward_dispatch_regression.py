import pytest
import torch

from src.pararnn import ParaGRU, ParaLSTM, ParaRNN


@pytest.mark.parametrize("cls", [ParaGRU, ParaRNN])
@pytest.mark.parametrize("mode", ["sequential", "deer", "jacobi", "picard"])
def test_gru_and_rnn_public_forward_returns_output_and_hn_for_all_core_modes(cls, mode):
    torch.manual_seed(9101)

    kwargs = dict(
        input_size=2,
        hidden_size=3,
        batch_first=True,
        mode=mode,
        dtype=torch.float64,
    )

    if mode in ("jacobi", "picard"):
        kwargs.pop("mode")
        kwargs["solver"] = mode
        kwargs["num_iters"] = 3
    elif cls is ParaRNN and mode == "deer":
        kwargs["backend"] = "quasi_autograd"
        kwargs["num_iters"] = 4
    elif cls is ParaGRU and mode == "deer":
        kwargs["backend"] = "adjoint"
        kwargs["num_iters"] = 4

    model = cls(**kwargs)

    x = torch.randn(2, 4, 2, dtype=torch.float64)
    h0 = torch.randn(1, 2, 3, dtype=torch.float64)

    y, h_n = model(x, h0)

    assert tuple(y.shape) == (2, 4, 3)
    assert tuple(h_n.shape) == (1, 2, 3)


@pytest.mark.parametrize("cls,solver,backend", [
    (ParaGRU, "elk", None),
    (ParaRNN, "elk", "elk"),
    (ParaRNN, "quasi_elk", "quasi_elk"),
])
def test_gru_and_rnn_public_forward_returns_output_and_hn_for_elk_modes(cls, solver, backend):
    torch.manual_seed(9102)

    kwargs = dict(
        input_size=2,
        hidden_size=3,
        batch_first=True,
        solver=solver,
        mode="elk",
        scan_backend="torch",
        num_iters=4,
        dtype=torch.float64,
    )

    if backend is not None:
        kwargs["backend"] = backend

    model = cls(**kwargs)

    x = torch.randn(2, 4, 2, dtype=torch.float64)
    h0 = torch.randn(1, 2, 3, dtype=torch.float64)

    y, h_n = model(x, h0)

    assert tuple(y.shape) == (2, 4, 3)
    assert tuple(h_n.shape) == (1, 2, 3)


@pytest.mark.parametrize("mode", ["sequential", "deer", "jacobi", "picard"])
def test_lstm_public_forward_returns_output_hn_cn_for_all_core_modes(mode):
    torch.manual_seed(9103)

    kwargs = dict(
        input_size=2,
        hidden_size=3,
        batch_first=True,
        mode=mode,
        dtype=torch.float64,
        recurrent_init_scale=0.025,
        peephole_init_scale=0.025,
    )

    if mode in ("jacobi", "picard"):
        kwargs.pop("mode")
        kwargs["solver"] = mode
        kwargs["num_iters"] = 3
    elif mode == "deer":
        kwargs["backend"] = "adjoint"
        kwargs["num_iters"] = 4

    model = ParaLSTM(**kwargs)

    x = torch.randn(2, 4, 2, dtype=torch.float64)
    h0 = torch.randn(1, 2, 3, dtype=torch.float64)
    c0 = torch.randn(1, 2, 3, dtype=torch.float64)

    y, (h_n, c_n) = model(x, (h0, c0))

    assert tuple(y.shape) == (2, 4, 3)
    assert tuple(h_n.shape) == (1, 2, 3)
    assert tuple(c_n.shape) == (1, 2, 3)


def test_lstm_public_forward_returns_output_hn_cn_for_elk_mode():
    torch.manual_seed(9104)

    model = ParaLSTM(
        input_size=2,
        hidden_size=3,
        batch_first=True,
        solver="quasi_elk",
        mode="elk",
        scan_backend="torch",
        num_iters=4,
        dtype=torch.float64,
        recurrent_init_scale=0.025,
        peephole_init_scale=0.025,
    )

    x = torch.randn(2, 4, 2, dtype=torch.float64)
    h0 = torch.randn(1, 2, 3, dtype=torch.float64)
    c0 = torch.randn(1, 2, 3, dtype=torch.float64)

    y, (h_n, c_n) = model(x, (h0, c0))

    assert tuple(y.shape) == (2, 4, 3)
    assert tuple(h_n.shape) == (1, 2, 3)
    assert tuple(c_n.shape) == (1, 2, 3)


def test_mode_override_still_works_after_public_forward_fix():
    torch.manual_seed(9105)

    model = ParaGRU(
        input_size=2,
        hidden_size=3,
        batch_first=True,
        mode="sequential",
        dtype=torch.float64,
    )

    x = torch.randn(2, 4, 2, dtype=torch.float64)

    y_seq, h_seq = model(x, mode="sequential")
    y_jac, h_jac = model(x, mode="jacobi")
    y_pic, h_pic = model(x, mode="picard")

    assert tuple(y_seq.shape) == (2, 4, 3)
    assert tuple(h_seq.shape) == (1, 2, 3)
    assert tuple(y_jac.shape) == (2, 4, 3)
    assert tuple(h_jac.shape) == (1, 2, 3)
    assert tuple(y_pic.shape) == (2, 4, 3)
    assert tuple(h_pic.shape) == (1, 2, 3)
