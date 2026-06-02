from src.pararnn.cells.para_gru import ParaGRUCell, ParaGRUConfig
from src.pararnn import DeerNewtonConfig, ParaRNNConfig, TanhDeerRNNCell
from src.algos.DEER import sequential_rollout
import torch
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def make_tanh_config(input_dim=3, state_dim=4, batch_first=True):
    return ParaRNNConfig(
        input_dim=input_dim,
        state_dim=state_dim,
        mode="sequential",
        batch_first=batch_first,
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


def make_gru_config(input_dim=3, state_dim=4, batch_first=True):
    return ParaGRUConfig(
        input_dim=input_dim,
        state_dim=state_dim,
        mode="sequential",
        batch_first=batch_first,
        dtype=torch.float64,
        recurrent_init_scale=0.20,
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


def reference_per_sample_states(cell, x, initial_state=None):
    """Old Phase-1/3 behavior: loop over batch, then over time."""
    x_batched, had_batch_dim = cell._normalize_input(x)
    initial_state_batched = cell._normalize_initial_state(
        x_batched=x_batched,
        initial_state=initial_state,
    )

    state_outputs = []
    for batch_idx in range(x_batched.shape[0]):
        states_i = sequential_rollout(
            f=cell.recurrence_step,
            initial_state=initial_state_batched[batch_idx],
            drivers=x_batched[batch_idx],
        )
        state_outputs.append(states_i)

    states = torch.stack(state_outputs, dim=0)
    return cell._restore_output_layout(states, had_batch_dim=had_batch_dim)


def test_tanh_batched_sequential_matches_old_per_sample_batched():
    torch.manual_seed(0)

    cell = TanhDeerRNNCell(make_tanh_config(input_dim=3, state_dim=5)).to(
        dtype=torch.float64
    )
    x = torch.randn(7, 19, 3, dtype=torch.float64)
    h0 = torch.randn(7, 5, dtype=torch.float64)

    y_new = cell.forward_states_sequential(x, initial_state=h0)
    y_ref = reference_per_sample_states(cell, x, initial_state=h0)

    assert y_new.shape == (7, 19, 5)
    assert torch.max(torch.abs(y_new - y_ref)).item() < 1e-12


def test_tanh_batched_sequential_matches_old_per_sample_unbatched():
    torch.manual_seed(1)

    cell = TanhDeerRNNCell(make_tanh_config(input_dim=3, state_dim=5)).to(
        dtype=torch.float64
    )
    x = torch.randn(19, 3, dtype=torch.float64)
    h0 = torch.randn(5, dtype=torch.float64)

    y_new = cell.forward_states_sequential(x, initial_state=h0)
    y_ref = reference_per_sample_states(cell, x, initial_state=h0)

    assert y_new.shape == (19, 5)
    assert torch.max(torch.abs(y_new - y_ref)).item() < 1e-12


def test_para_gru_batched_sequential_matches_old_per_sample_batched():
    torch.manual_seed(2)

    cell = ParaGRUCell(make_gru_config(input_dim=4, state_dim=6)).to(
        dtype=torch.float64
    )
    x = 0.5 * torch.randn(5, 17, 4, dtype=torch.float64)
    h0 = torch.randn(5, 6, dtype=torch.float64)

    y_new = cell.forward_states_sequential(x, initial_state=h0)
    y_ref = reference_per_sample_states(cell, x, initial_state=h0)

    assert y_new.shape == (5, 17, 6)
    assert torch.max(torch.abs(y_new - y_ref)).item() < 1e-12


def test_para_gru_batched_sequential_matches_old_per_sample_time_first():
    torch.manual_seed(3)

    batch_first_cell = ParaGRUCell(
        make_gru_config(input_dim=4, state_dim=6, batch_first=True)
    ).to(dtype=torch.float64)
    time_first_cell = ParaGRUCell(
        make_gru_config(input_dim=4, state_dim=6, batch_first=False)
    ).to(dtype=torch.float64)
    time_first_cell.load_state_dict(batch_first_cell.state_dict())

    x_batch_first = 0.5 * torch.randn(5, 17, 4, dtype=torch.float64)
    x_time_first = x_batch_first.transpose(0, 1).contiguous()
    h0 = torch.randn(5, 6, dtype=torch.float64)

    y_batch_first = batch_first_cell.forward_states_sequential(
        x_batch_first,
        initial_state=h0,
    )
    y_time_first = time_first_cell.forward_states_sequential(
        x_time_first,
        initial_state=h0,
    )

    assert y_time_first.shape == (17, 5, 6)
    assert torch.max(
        torch.abs(y_batch_first - y_time_first.transpose(0, 1))).item() < 1e-12


def test_batched_sequential_backward_smoke():
    torch.manual_seed(4)

    cell = ParaGRUCell(make_gru_config(input_dim=3, state_dim=4)).to(
        dtype=torch.float64
    )
    x = torch.randn(6, 11, 3, dtype=torch.float64, requires_grad=True)

    y = cell(x, mode="sequential")
    loss = y.square().mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    for param in cell.parameters():
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()
