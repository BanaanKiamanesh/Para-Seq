from src.pararnn.cells.para_gru import ParaGRUCell, ParaGRUConfig
from src.pararnn import DeerNewtonConfig, ParaRNNConfig, TanhDeerRNNCell
from src.algos.DEER import deer_alg, deer_alg_batched
import torch
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def make_tanh_config(input_dim=3, state_dim=4, num_iters=20):
    return ParaRNNConfig(
        input_dim=input_dim,
        state_dim=state_dim,
        mode="deer",
        batch_first=True,
        dtype=torch.float64,
        deer=DeerNewtonConfig(
            num_iters=num_iters,
            tol=1e-10,
            strict_tol=True,
            stopping_criterion="update",
            initial_guess="f0",
            quasi=False,
            scan_backend="torch",
            jacobian_backend="autograd",
        ),
    )


def make_para_gru_config(
    input_dim=3,
    state_dim=4,
    batch_first=True,
    num_iters=24,
    quasi=True,
    jacobian_backend="explicit",
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


def run_per_sample_deer(
    cell,
    x,
    h0,
    states_guess,
    quasi,
    num_iters,
    jacobian_fn=None,
):
    outputs = []
    infos = []

    for batch_idx in range(x.shape[0]):
        if jacobian_fn is None:
            jacobian_fn_i = None
        else:
            def jacobian_fn_i(previous_states, drivers, batch_idx=batch_idx):
                return jacobian_fn(
                    previous_states.unsqueeze(0),
                    drivers.unsqueeze(0),
                ).squeeze(0)

        states_i, info_i = deer_alg(
            f=cell.recurrence_step,
            initial_state=h0[batch_idx],
            states_guess=states_guess[batch_idx],
            drivers=x[batch_idx],
            num_iters=num_iters,
            tol=1e-11,
            strict_tol=True,
            stopping_criterion="update",
            quasi=quasi,
            scan_backend="torch",
            jacobian_fn=jacobian_fn_i,
        )

        outputs.append(states_i)
        infos.append(info_i)

    return torch.stack(outputs, dim=0), infos


def test_batched_full_deer_matches_old_per_sample_full_deer():
    torch.manual_seed(0)

    config = make_tanh_config(
        input_dim=2,
        state_dim=3,
        num_iters=18,
    )
    cell = TanhDeerRNNCell(config).to(dtype=torch.float64)

    x = 0.5 * torch.randn(4, 11, 2, dtype=torch.float64)
    h0 = torch.randn(4, 3, dtype=torch.float64)

    states_guess = cell.assemble_initial_guess_batched(
        drivers=x,
        initial_state=h0,
        guess_type="f0",
    )

    states_batched, info_batched = deer_alg_batched(
        f=cell.recurrence_step,
        initial_state=h0,
        states_guess=states_guess,
        drivers=x,
        num_iters=18,
        tol=1e-11,
        strict_tol=True,
        stopping_criterion="update",
        quasi=False,
        scan_backend="torch",
    )

    states_ref, _ = run_per_sample_deer(
        cell=cell,
        x=x,
        h0=h0,
        states_guess=states_guess,
        quasi=False,
        num_iters=18,
    )

    assert states_batched.shape == (4, 11, 3)
    assert info_batched["batched"] is True
    assert info_batched["batch_size"] == 4
    assert torch.max(torch.abs(states_batched - states_ref)).item() < 1e-10


def test_batched_quasi_deer_matches_old_per_sample_quasi_deer_autograd():
    torch.manual_seed(1)

    config = make_tanh_config(
        input_dim=2,
        state_dim=3,
        num_iters=12,
    )
    cell = TanhDeerRNNCell(config).to(dtype=torch.float64)

    x = 0.5 * torch.randn(3, 9, 2, dtype=torch.float64)
    h0 = torch.randn(3, 3, dtype=torch.float64)

    states_guess = cell.assemble_initial_guess_batched(
        drivers=x,
        initial_state=h0,
        guess_type="f0",
    )

    states_batched, info_batched = deer_alg_batched(
        f=cell.recurrence_step,
        initial_state=h0,
        states_guess=states_guess,
        drivers=x,
        num_iters=12,
        tol=1e-11,
        strict_tol=True,
        stopping_criterion="update",
        quasi=True,
        scan_backend="torch",
    )

    states_ref, _ = run_per_sample_deer(
        cell=cell,
        x=x,
        h0=h0,
        states_guess=states_guess,
        quasi=True,
        num_iters=12,
    )

    assert states_batched.shape == (3, 9, 3)
    assert info_batched["batched"] is True
    assert info_batched["quasi"] is True
    assert torch.max(torch.abs(states_batched - states_ref)).item() < 1e-10


def test_batched_explicit_quasi_deer_matches_old_per_sample_explicit_quasi_deer():
    torch.manual_seed(2)

    config = make_para_gru_config(
        input_dim=3,
        state_dim=4,
        num_iters=20,
    )
    cell = ParaGRUCell(config).to(dtype=torch.float64)

    x = 0.5 * torch.randn(4, 13, 3, dtype=torch.float64)
    h0 = torch.zeros(4, 4, dtype=torch.float64)

    states_guess = cell.assemble_initial_guess_batched(
        drivers=x,
        initial_state=h0,
        guess_type="f0",
    )

    def explicit_jacobian_fn(previous_states, drivers):
        return cell._compute_jacobians_diag_from_previous(
            previous_states=previous_states,
            drivers=drivers,
        )

    states_batched, info_batched = deer_alg_batched(
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
        jacobian_fn=explicit_jacobian_fn,
    )

    states_ref, _ = run_per_sample_deer(
        cell=cell,
        x=x,
        h0=h0,
        states_guess=states_guess,
        quasi=True,
        num_iters=20,
        jacobian_fn=explicit_jacobian_fn,
    )

    assert states_batched.shape == (4, 13, 4)
    assert info_batched["batched"] is True
    assert info_batched["jacobian_backend"] == "custom"
    assert torch.max(torch.abs(states_batched - states_ref)).item() < 1e-10


def test_base_forward_deer_uses_single_batched_solver_call():
    torch.manual_seed(3)

    config = make_para_gru_config(
        input_dim=3,
        state_dim=5,
        num_iters=24,
    )
    cell = ParaGRUCell(config).to(dtype=torch.float64)

    x = 0.5 * torch.randn(4, 17, 3, dtype=torch.float64)

    y_seq = cell(x, mode="sequential")
    y_deer = cell(x, mode="deer")

    assert y_seq.shape == (4, 17, 5)
    assert y_deer.shape == (4, 17, 5)
    assert torch.max(torch.abs(y_seq - y_deer)).item() < 1e-7
    assert len(cell.last_deer_infos) == 1
    assert cell.last_deer_infos[0]["batched"] is True
    assert cell.last_deer_infos[0]["batch_size"] == 4
    assert cell.last_deer_infos[0]["jacobian_backend"] == "custom"


def test_batched_deer_return_trace_shape():
    torch.manual_seed(4)

    config = make_para_gru_config(
        input_dim=2,
        state_dim=3,
        num_iters=5,
    )
    cell = ParaGRUCell(config).to(dtype=torch.float64)

    x = 0.5 * torch.randn(2, 8, 2, dtype=torch.float64)
    h0 = torch.zeros(2, 3, dtype=torch.float64)

    states_guess = cell.assemble_initial_guess_batched(
        drivers=x,
        initial_state=h0,
        guess_type="f0",
    )

    def explicit_jacobian_fn(previous_states, drivers):
        return cell._compute_jacobians_diag_from_previous(
            previous_states=previous_states,
            drivers=drivers,
        )

    states, info = deer_alg_batched(
        f=cell.recurrence_step,
        initial_state=h0,
        states_guess=states_guess,
        drivers=x,
        num_iters=5,
        tol=1e-12,
        strict_tol=True,
        stopping_criterion="update",
        quasi=True,
        scan_backend="torch",
        jacobian_fn=explicit_jacobian_fn,
        return_trace=True,
    )

    assert states.shape == (2, 8, 3)
    assert "trace" in info
    assert info["trace"].shape[1:] == (2, 8, 3)
    assert info["trace"].shape[0] == info["num_iters"] + 1


def test_batched_explicit_quasi_deer_time_first_layout():
    torch.manual_seed(5)

    batch_first_config = make_para_gru_config(
        input_dim=2,
        state_dim=4,
        batch_first=True,
        num_iters=24,
    )
    time_first_config = make_para_gru_config(
        input_dim=2,
        state_dim=4,
        batch_first=False,
        num_iters=24,
    )

    batch_first_cell = ParaGRUCell(batch_first_config).to(dtype=torch.float64)
    time_first_cell = ParaGRUCell(time_first_config).to(dtype=torch.float64)
    time_first_cell.load_state_dict(batch_first_cell.state_dict())

    x_batch_first = 0.5 * torch.randn(3, 12, 2, dtype=torch.float64)
    x_time_first = x_batch_first.transpose(0, 1).contiguous()

    y_batch_first = batch_first_cell(x_batch_first, mode="deer")
    y_time_first = time_first_cell(x_time_first, mode="deer")

    assert y_time_first.shape == (12, 3, 4)
    assert torch.max(
        torch.abs(y_batch_first - y_time_first.transpose(0, 1))
    ).item() < 1e-7


def test_batched_explicit_quasi_deer_backward_smoke():
    torch.manual_seed(6)

    config = make_para_gru_config(
        input_dim=2,
        state_dim=3,
        num_iters=10,
    )
    cell = ParaGRUCell(config).to(dtype=torch.float64)

    x = torch.randn(
        3,
        9,
        2,
        dtype=torch.float64,
        requires_grad=True,
    )

    y = cell(x, mode="deer")
    loss = y.square().mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    for param in cell.parameters():
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()
