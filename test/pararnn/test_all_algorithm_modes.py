import pytest
import torch

from src.pararnn import ParaGRU, ParaLSTM, ParaRNN
from src.pararnn.cells.para_gru import (
    make_paragru_deer_config,
    make_paragru_elk_config,
    make_paragru_jacobi_config,
    make_paragru_picard_config,
)
from src.pararnn.cells.para_lstm import (
    make_paralstm_deer_config,
    make_paralstm_elk_config,
    make_paralstm_jacobi_config,
    make_paralstm_picard_config,
)
from src.pararnn.cells.para_rnn import (
    make_pararnn_deer_config,
    make_pararnn_elk_config,
    make_pararnn_jacobi_config,
    make_pararnn_picard_config,
)


def _assert_shape(tensor, expected):
    assert tuple(tensor.shape) == tuple(expected), (
        f"got {tuple(tensor.shape)}, expected {tuple(expected)}"
    )


@pytest.mark.parametrize("mode", ["jacobi", "picard"])
def test_paragru_supports_jacobi_and_picard_forward_backward(mode):
    torch.manual_seed(8101)

    model = ParaGRU(
        input_size=2,
        hidden_size=3,
        batch_first=True,
        solver=mode,
        num_iters=4,
        dtype=torch.float64,
    )

    x = torch.randn(2, 5, 2, dtype=torch.float64, requires_grad=True)
    h0 = torch.randn(1, 2, 3, dtype=torch.float64, requires_grad=True)

    y, h_n = model(x, h0)

    _assert_shape(y, (2, 5, 3))
    _assert_shape(h_n, (1, 2, 3))
    assert model.last_deer_infos[-1]["solver"] == mode

    loss = y.square().mean() + h_n.square().mean()
    loss.backward()

    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(h0.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("mode", ["jacobi", "picard"])
def test_pararnn_supports_jacobi_and_picard_forward_backward(mode):
    torch.manual_seed(8102)

    model = ParaRNN(
        input_size=2,
        hidden_size=3,
        batch_first=True,
        solver=mode,
        num_iters=4,
        dtype=torch.float64,
    )

    x = torch.randn(2, 5, 2, dtype=torch.float64, requires_grad=True)
    h0 = torch.randn(1, 2, 3, dtype=torch.float64, requires_grad=True)

    y, h_n = model(x, h0)

    _assert_shape(y, (2, 5, 3))
    _assert_shape(h_n, (1, 2, 3))
    assert model.last_deer_infos[-1]["solver"] == mode

    loss = y.square().mean() + h_n.square().mean()
    loss.backward()

    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(h0.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("mode", ["jacobi", "picard"])
def test_paralstm_supports_jacobi_and_picard_forward_backward(mode):
    torch.manual_seed(8103)

    model = ParaLSTM(
        input_size=2,
        hidden_size=3,
        batch_first=True,
        solver=mode,
        num_iters=4,
        dtype=torch.float64,
        recurrent_init_scale=0.025,
        peephole_init_scale=0.025,
    )

    x = torch.randn(2, 5, 2, dtype=torch.float64, requires_grad=True)
    h0 = torch.randn(1, 2, 3, dtype=torch.float64, requires_grad=True)
    c0 = torch.randn(1, 2, 3, dtype=torch.float64, requires_grad=True)

    y, (h_n, c_n) = model(x, (h0, c0))

    _assert_shape(y, (2, 5, 3))
    _assert_shape(h_n, (1, 2, 3))
    _assert_shape(c_n, (1, 2, 3))
    assert model.last_deer_infos[-1]["solver"] == mode

    loss = y.square().mean() + h_n.square().mean() + c_n.square().mean()
    loss.backward()

    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(h0.grad).all()
    assert torch.isfinite(c0.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_algorithm_mode_helpers_cover_torch_and_accel_scan_quasi_configs():
    cfg = make_paragru_deer_config(
        backend="quasi_deer_adjoint_accel_scan",
        scan_backend="torch",
    )
    assert cfg.quasi is True
    assert cfg.scan_backend == "accel_scan"
    assert cfg.backward_backend == "adjoint"

    cfg = make_paragru_deer_config(
        backend="quasi_deer_autograd_torch",
        scan_backend="accel_scan",
    )
    assert cfg.quasi is True
    assert cfg.scan_backend == "torch"
    assert cfg.backward_backend == "autograd"

    cfg = make_paragru_elk_config(scan_backend="accel_scan")
    assert cfg.quasi is True
    assert cfg.scan_backend == "accel_scan"
    assert cfg.solver == "elk"

    cfg = make_pararnn_deer_config(
        backend="quasi_deer_autograd_accel_scan",
        scan_backend="torch",
    )
    assert cfg.quasi is True
    assert cfg.scan_backend == "accel_scan"

    cfg = make_pararnn_deer_config(
        backend="quasi_deer_autograd_torch",
        scan_backend="accel_scan",
    )
    assert cfg.quasi is True
    assert cfg.scan_backend == "torch"

    cfg = make_pararnn_elk_config(
        backend="quasi_elk",
        scan_backend="accel_scan",
    )
    assert cfg.quasi is True
    assert cfg.scan_backend == "accel_scan"
    assert cfg.solver == "elk"

    cfg = make_paralstm_deer_config(
        backend="quasi_deer_autograd_accel_scan",
        scan_backend="torch",
    )
    assert cfg.quasi is True
    assert cfg.scan_backend == "accel_scan"

    cfg = make_paralstm_deer_config(
        backend="quasi_deer_autograd_torch",
        scan_backend="accel_scan",
    )
    assert cfg.quasi is True
    assert cfg.scan_backend == "torch"

    cfg = make_paralstm_elk_config(scan_backend="accel_scan")
    assert cfg.quasi is True
    assert cfg.scan_backend == "accel_scan"
    assert cfg.solver == "elk"


def test_algorithm_mode_helpers_cover_jacobi_and_picard_configs():
    helpers = [
        make_paragru_jacobi_config,
        make_paragru_picard_config,
        make_pararnn_jacobi_config,
        make_pararnn_picard_config,
        make_paralstm_jacobi_config,
        make_paralstm_picard_config,
    ]

    expected_solvers = [
        "jacobi",
        "picard",
        "jacobi",
        "picard",
        "jacobi",
        "picard",
    ]

    for helper, expected_solver in zip(helpers, expected_solvers):
        cfg = helper(num_iters=3, tol=1e-6, strict_tol=True)
        assert cfg.solver == expected_solver
        assert cfg.num_iters == 3
        assert cfg.tol == 1e-6
        assert cfg.strict_tol is True
        assert cfg.backward_backend == "autograd"


@pytest.mark.parametrize(
    "model",
    [
        ParaGRU(input_size=2, hidden_size=3, batch_first=True, dtype=torch.float64),
        ParaRNN(input_size=2, hidden_size=3, batch_first=True, dtype=torch.float64),
        ParaLSTM(
            input_size=2,
            hidden_size=3,
            batch_first=True,
            dtype=torch.float64,
            recurrent_init_scale=0.025,
            peephole_init_scale=0.025,
        ),
    ],
)
def test_mode_override_can_run_jacobi_and_picard_without_constructor_solver(model):
    torch.manual_seed(8104)

    x = torch.randn(2, 4, 2, dtype=torch.float64)

    y_jacobi = model(x, mode="jacobi")[0]
    y_picard = model(x, mode="picard")[0]

    _assert_shape(y_jacobi, (2, 4, 3))
    _assert_shape(y_picard, (2, 4, 3))
