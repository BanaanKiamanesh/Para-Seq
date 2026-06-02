from src.pararnn import DeerNewtonConfig, ParaRNNConfig, ParaRNNDeerConfig, TanhDeerRNNCell
import torch
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def make_config(input_dim=3, state_dim=4, batch_first=True):
    return ParaRNNConfig(
        input_dim=input_dim,
        state_dim=state_dim,
        mode="deer",
        batch_first=batch_first,
        dtype=torch.float64,
        deer=DeerNewtonConfig(
            num_iters=16,
            tol=1e-10,
            strict_tol=True,
            stopping_criterion="update",
            initial_guess="f0",
            quasi=False,
            scan_backend="torch",
        ),
    )


def test_backward_compatible_config_alias():
    config = ParaRNNDeerConfig(
        input_dim=3,
        state_dim=4,
        dtype=torch.float64,
    )

    assert isinstance(config, ParaRNNConfig)
    assert config.output_dim == config.state_dim


def test_deer_forward_matches_sequential_unbatched_float64():
    torch.manual_seed(0)

    config = make_config(input_dim=3, state_dim=4)
    cell = TanhDeerRNNCell(config).to(dtype=torch.float64)
    x = torch.randn(32, 3, dtype=torch.float64)

    y_seq = cell(x, mode="sequential")
    y_deer = cell(x, mode="deer")

    assert y_seq.shape == (32, 4)
    assert y_deer.shape == (32, 4)

    max_error = torch.max(torch.abs(y_seq - y_deer)).item()
    assert max_error < 1e-7


def test_deer_forward_matches_sequential_batched_float64():
    torch.manual_seed(1)

    config = make_config(input_dim=3, state_dim=4)
    cell = TanhDeerRNNCell(config).to(dtype=torch.float64)
    x = torch.randn(2, 24, 3, dtype=torch.float64)

    y_seq = cell(x, mode="sequential")
    y_deer = cell(x, mode="deer")

    assert y_seq.shape == (2, 24, 4)
    assert y_deer.shape == (2, 24, 4)

    max_error = torch.max(torch.abs(y_seq - y_deer)).item()
    assert max_error < 1e-7


def test_time_first_layout_matches_batch_first_layout():
    torch.manual_seed(2)

    batch_first_config = make_config(
        input_dim=3, state_dim=4, batch_first=True)
    time_first_config = make_config(
        input_dim=3, state_dim=4, batch_first=False)

    batch_first_cell = TanhDeerRNNCell(
        batch_first_config).to(dtype=torch.float64)
    time_first_cell = TanhDeerRNNCell(
        time_first_config).to(dtype=torch.float64)
    time_first_cell.load_state_dict(batch_first_cell.state_dict())

    x_batch_first = torch.randn(2, 20, 3, dtype=torch.float64)
    x_time_first = x_batch_first.transpose(0, 1).contiguous()

    y_batch_first = batch_first_cell(x_batch_first, mode="deer")
    y_time_first = time_first_cell(x_time_first, mode="deer")

    assert y_time_first.shape == (20, 2, 4)

    max_error = torch.max(
        torch.abs(y_batch_first - y_time_first.transpose(0, 1))).item()
    assert max_error < 1e-7


def test_initial_state_broadcast_and_batched_initial_state():
    torch.manual_seed(3)

    config = make_config(input_dim=2, state_dim=3)
    cell = TanhDeerRNNCell(config).to(dtype=torch.float64)

    x = torch.randn(2, 16, 2, dtype=torch.float64)
    h0_shared = torch.randn(3, dtype=torch.float64)
    h0_batched = h0_shared.expand(2, -1).clone()

    y_shared = cell(x, initial_state=h0_shared, mode="sequential")
    y_batched = cell(x, initial_state=h0_batched, mode="sequential")

    max_error = torch.max(torch.abs(y_shared - y_batched)).item()
    assert max_error == 0.0


def test_negative_residuals_are_small_on_sequential_solution():
    torch.manual_seed(4)

    config = make_config(input_dim=2, state_dim=3)
    cell = TanhDeerRNNCell(config).to(dtype=torch.float64)

    x = torch.randn(2, 18, 2, dtype=torch.float64)
    states = cell.forward_states_sequential(x)
    negative_residuals = cell.compute_negative_residuals(
        states=states, drivers=x)

    assert negative_residuals.shape == states.shape
    assert torch.max(torch.abs(negative_residuals)).item() < 1e-12


def test_autograd_jacobian_shape_and_values_are_finite():
    torch.manual_seed(5)

    config = make_config(input_dim=2, state_dim=3)
    cell = TanhDeerRNNCell(config).to(dtype=torch.float64)

    x = torch.randn(2, 7, 2, dtype=torch.float64)
    states = cell.forward_states_sequential(x)
    jac = cell.compute_jacobians_autograd(states=states, drivers=x)

    assert jac.shape == (2, 7, 3, 3)
    assert torch.isfinite(jac).all()


def test_deer_forward_backward_smoke():
    torch.manual_seed(6)

    config = ParaRNNConfig(
        input_dim=2,
        state_dim=3,
        mode="deer",
        dtype=torch.float64,
        deer=DeerNewtonConfig(
            num_iters=8,
            tol=1e-8,
            strict_tol=True,
            stopping_criterion="update",
            initial_guess="f0",
            quasi=False,
            scan_backend="torch",
        ),
    )

    cell = TanhDeerRNNCell(config).to(dtype=torch.float64)
    x = torch.randn(2, 12, 2, dtype=torch.float64, requires_grad=True)

    y = cell(x, mode="deer")
    loss = y.square().mean()
    loss.backward()

    assert torch.isfinite(x.grad).all()

    for param in cell.parameters():
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()
