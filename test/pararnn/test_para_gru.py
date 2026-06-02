from src.pararnn.cells.para_gru import ParaGRU, ParaGRUCell, ParaGRUConfig
from src.pararnn import DeerNewtonConfig
import torch
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def make_config(
    input_dim=3,
    state_dim=4,
    batch_first=True,
    num_iters=20,
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
            quasi=False,
            scan_backend="torch",
        ),
    )


def test_para_gru_alias():
    assert ParaGRU is ParaGRUCell


def test_para_gru_sequential_shape_unbatched():
    torch.manual_seed(0)

    config = make_config(input_dim=3, state_dim=5)
    cell = ParaGRUCell(config).to(dtype=torch.float64)

    x = 0.5 * torch.randn(17, 3, dtype=torch.float64)
    y = cell(x, mode="sequential")

    assert y.shape == (17, 5)
    assert torch.isfinite(y).all()


def test_para_gru_sequential_shape_batched():
    torch.manual_seed(1)

    config = make_config(input_dim=3, state_dim=5)
    cell = ParaGRUCell(config).to(dtype=torch.float64)

    x = 0.5 * torch.randn(2, 17, 3, dtype=torch.float64)
    y = cell(x, mode="sequential")

    assert y.shape == (2, 17, 5)
    assert torch.isfinite(y).all()


def test_para_gru_deer_matches_sequential_unbatched_float64():
    torch.manual_seed(2)

    config = make_config(input_dim=3, state_dim=4, num_iters=24)
    cell = ParaGRUCell(config).to(dtype=torch.float64)

    x = 0.5 * torch.randn(20, 3, dtype=torch.float64)

    y_seq = cell(x, mode="sequential")
    y_deer = cell(x, mode="deer")

    assert y_seq.shape == (20, 4)
    assert y_deer.shape == (20, 4)

    max_error = torch.max(torch.abs(y_seq - y_deer)).item()

    assert max_error < 1e-7


def test_para_gru_deer_matches_sequential_batched_float64():
    torch.manual_seed(3)

    config = make_config(input_dim=3, state_dim=4, num_iters=24)
    cell = ParaGRUCell(config).to(dtype=torch.float64)

    x = 0.5 * torch.randn(2, 18, 3, dtype=torch.float64)

    y_seq = cell(x, mode="sequential")
    y_deer = cell(x, mode="deer")

    assert y_seq.shape == (2, 18, 4)
    assert y_deer.shape == (2, 18, 4)

    max_error = torch.max(torch.abs(y_seq - y_deer)).item()

    assert max_error < 1e-7


def test_para_gru_negative_residuals_are_small_on_sequential_solution():
    torch.manual_seed(4)

    config = make_config(input_dim=2, state_dim=3)
    cell = ParaGRUCell(config).to(dtype=torch.float64)

    x = 0.5 * torch.randn(2, 16, 2, dtype=torch.float64)

    states = cell.forward_states_sequential(x)
    negative_residuals = cell.compute_negative_residuals(
        states=states,
        drivers=x,
    )

    assert negative_residuals.shape == states.shape
    assert torch.max(torch.abs(negative_residuals)).item() < 1e-12


def test_para_gru_initial_guess_f0_matches_zero_state_recurrence():
    torch.manual_seed(5)

    config = make_config(input_dim=2, state_dim=3)
    cell = ParaGRUCell(config).to(dtype=torch.float64)

    x = 0.5 * torch.randn(11, 2, dtype=torch.float64)
    h0 = torch.randn(3, dtype=torch.float64)

    guess = cell.assemble_initial_guess(
        drivers=x,
        initial_state=h0,
        guess_type="f0",
    )

    zero_states = torch.zeros(11, 3, dtype=torch.float64)
    expected = cell.recurrence_step(zero_states, x)

    max_error = torch.max(torch.abs(guess - expected)).item()

    assert guess.shape == (11, 3)
    assert max_error == 0.0


def test_para_gru_explicit_diagonal_jacobian_matches_autograd_diagonal():
    torch.manual_seed(6)

    config = make_config(input_dim=2, state_dim=3)
    cell = ParaGRUCell(config).to(dtype=torch.float64)

    x = 0.5 * torch.randn(2, 9, 2, dtype=torch.float64)

    states = cell.forward_states_sequential(x)

    jac_diag = cell.compute_jacobians_diag(
        states=states,
        drivers=x,
    )

    jac_dense = cell.compute_jacobians_autograd(
        states=states,
        drivers=x,
    )

    jac_diag_from_dense = torch.diagonal(
        jac_dense,
        dim1=-2,
        dim2=-1,
    )

    assert jac_diag.shape == (2, 9, 3)
    assert jac_diag_from_dense.shape == (2, 9, 3)

    max_error = torch.max(torch.abs(jac_diag - jac_diag_from_dense)).item()

    assert max_error < 1e-8


def test_para_gru_backward_diag_shape_and_first_entry_zero():
    torch.manual_seed(7)

    config = make_config(input_dim=2, state_dim=3)
    cell = ParaGRUCell(config).to(dtype=torch.float64)

    x = 0.5 * torch.randn(2, 10, 2, dtype=torch.float64)
    states = cell.forward_states_sequential(x)

    jac_bwd = cell.compute_jacobians_bwd_diag(
        states=states,
        drivers=x,
    )

    assert jac_bwd.shape == (2, 10, 3)
    assert torch.isfinite(jac_bwd).all()
    assert torch.max(torch.abs(jac_bwd[:, 0, :])).item() == 0.0


def test_para_gru_deer_forward_backward_smoke():
    torch.manual_seed(8)

    config = make_config(input_dim=2, state_dim=3, num_iters=12)
    cell = ParaGRUCell(config).to(dtype=torch.float64)

    x_leaf = torch.randn(
        2,
        10,
        2,
        dtype=torch.float64,
        requires_grad=True,
    )

    x = 0.5 * x_leaf

    y = cell(x, mode="deer")
    loss = y.square().mean()
    loss.backward()

    assert x_leaf.grad is not None
    assert torch.isfinite(x_leaf.grad).all()

    for param in cell.parameters():
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()
