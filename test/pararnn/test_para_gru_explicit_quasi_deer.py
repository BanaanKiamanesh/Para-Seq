from src.algos.DEER import deer_alg
from src.pararnn.cells.para_gru import ParaGRUCell, ParaGRUConfig
from src.pararnn import DeerNewtonConfig
from torch import nn
import torch
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def make_config(
    input_dim=3,
    state_dim=4,
    batch_first=True,
    num_iters=24,
    jacobian_backend="explicit",
    quasi=True,
):
    return ParaGRUConfig(
        input_dim=input_dim,
        state_dim=state_dim,
        mode="deer",
        batch_first=batch_first,
        dtype=torch.float64,
        recurrent_init_scale=0.20,
        deer=DeerNewtonConfig(
            num_iters=num_iters,
            tol=1e-11,
            strict_tol=True,
            stopping_criterion="update",
            initial_guess="f0",
            quasi=quasi,
            scan_backend="torch",
            jacobian_backend=jacobian_backend,
        ),
    )


def test_deer_alg_accepts_custom_diagonal_jacobian_callback():
    torch.manual_seed(0)

    config = make_config(input_dim=2, state_dim=3, num_iters=20)
    cell = ParaGRUCell(config).to(dtype=torch.float64)

    x = 0.5 * torch.randn(13, 2, dtype=torch.float64)
    h0 = torch.zeros(3, dtype=torch.float64)
    states_guess = cell.assemble_initial_guess(
        drivers=x,
        initial_state=h0,
        guess_type="f0",
    )

    def jacobian_fn(previous_states, drivers):
        return cell._compute_jacobians_diag_from_previous(
            previous_states=previous_states.unsqueeze(0),
            drivers=drivers.unsqueeze(0),
        ).squeeze(0)

    states, info = deer_alg(
        f=cell.recurrence_step,
        initial_state=h0,
        states_guess=states_guess,
        drivers=x,
        num_iters=20,
        tol=1e-11,
        strict_tol=True,
        stopping_criterion="update",
        quasi=True,
        scan_backend="torch",
        jacobian_fn=jacobian_fn,
    )

    sequential_states = cell(x, mode="sequential")

    assert states.shape == (13, 3)
    assert torch.max(torch.abs(states - sequential_states)).item() < 1e-7
    assert info["jacobian_backend"] == "custom"
    assert info["quasi"] is True


def test_explicit_quasi_deer_matches_sequential_batched_float64():
    torch.manual_seed(1)

    config = make_config(input_dim=3, state_dim=5, num_iters=24)
    cell = ParaGRUCell(config).to(dtype=torch.float64)

    x = 0.5 * torch.randn(4, 19, 3, dtype=torch.float64)

    y_seq = cell(x, mode="sequential")
    y_explicit = cell(x, mode="deer")

    assert y_seq.shape == (4, 19, 5)
    assert y_explicit.shape == (4, 19, 5)
    assert torch.max(torch.abs(y_seq - y_explicit)).item() < 1e-7
    assert all(info["jacobian_backend"] ==
               "custom" for info in cell.last_deer_infos)


def test_explicit_quasi_deer_matches_generic_quasi_deer():
    torch.manual_seed(2)

    explicit_config = make_config(
        input_dim=3,
        state_dim=4,
        num_iters=20,
        jacobian_backend="explicit",
        quasi=True,
    )
    autograd_config = make_config(
        input_dim=3,
        state_dim=4,
        num_iters=20,
        jacobian_backend="autograd",
        quasi=True,
    )

    explicit_cell = ParaGRUCell(explicit_config).to(dtype=torch.float64)
    autograd_cell = ParaGRUCell(autograd_config).to(dtype=torch.float64)
    autograd_cell.load_state_dict(explicit_cell.state_dict())

    x = 0.5 * torch.randn(3, 15, 3, dtype=torch.float64)

    y_explicit = explicit_cell(x, mode="deer")
    y_autograd = autograd_cell(x, mode="deer")

    assert y_explicit.shape == y_autograd.shape
    assert torch.max(torch.abs(y_explicit - y_autograd)).item() < 1e-7


def test_explicit_backend_requires_quasi_deer():
    torch.manual_seed(3)

    config = make_config(
        input_dim=2,
        state_dim=3,
        num_iters=8,
        jacobian_backend="explicit",
        quasi=False,
    )
    cell = ParaGRUCell(config).to(dtype=torch.float64)
    x = 0.5 * torch.randn(2, 7, 2, dtype=torch.float64)

    try:
        cell(x, mode="deer")
    except ValueError as error:
        assert "quasi=True" in str(error)
    else:
        raise AssertionError(
            "Expected explicit backend with quasi=False to fail.")


def test_explicit_quasi_deer_time_first_layout():
    torch.manual_seed(4)

    batch_first_config = make_config(
        input_dim=2, state_dim=4, batch_first=True)
    time_first_config = make_config(
        input_dim=2, state_dim=4, batch_first=False)

    batch_first_cell = ParaGRUCell(batch_first_config).to(dtype=torch.float64)
    time_first_cell = ParaGRUCell(time_first_config).to(dtype=torch.float64)
    time_first_cell.load_state_dict(batch_first_cell.state_dict())

    x_batch_first = 0.5 * torch.randn(3, 13, 2, dtype=torch.float64)
    x_time_first = x_batch_first.transpose(0, 1).contiguous()

    y_batch_first = batch_first_cell(x_batch_first, mode="deer")
    y_time_first = time_first_cell(x_time_first, mode="deer")

    assert y_time_first.shape == (13, 3, 4)
    assert torch.max(
        torch.abs(y_batch_first - y_time_first.transpose(0, 1))).item() < 1e-7


def test_explicit_quasi_deer_backward_smoke():
    torch.manual_seed(5)

    config = make_config(input_dim=2, state_dim=3, num_iters=10)
    cell = ParaGRUCell(config).to(dtype=torch.float64)

    x = torch.randn(3, 9, 2, dtype=torch.float64, requires_grad=True)

    y = cell(x, mode="deer")
    loss = y.square().mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    for param in cell.parameters():
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()


class TinyExplicitQuasiDEERClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        config = make_config(input_dim=2, state_dim=4, num_iters=8)
        self.rnn = ParaGRUCell(config).to(dtype=torch.float64)
        self.readout = nn.Linear(4, 2).to(dtype=torch.float64)

    def forward(self, x):
        states = self.rnn(x, mode="deer")
        return self.readout(states[:, -1, :])


def test_explicit_quasi_deer_tiny_training_smoke():
    torch.manual_seed(6)

    x = 0.5 * torch.randn(6, 4, 2, dtype=torch.float64)
    y = (x[:, -1, 0] > 0.0).long()

    model = TinyExplicitQuasiDEERClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-2)
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        initial_loss = criterion(model(x), y).item()

    for _ in range(6):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_loss = criterion(model(x), y).item()

    assert final_loss < initial_loss
