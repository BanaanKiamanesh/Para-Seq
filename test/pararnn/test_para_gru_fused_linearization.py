from src.pararnn.cells.para_gru import ParaGRUCell, ParaGRUConfig
from src.pararnn import DeerNewtonConfig
from src.algos.DEER import deer_alg_batched
import torch
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def make_config(
    input_dim=3,
    state_dim=4,
    batch_first=True,
    num_iters=12,
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


def test_default_deer_num_iters_is_four():
    cfg = DeerNewtonConfig()
    assert cfg.num_iters == 4


def test_fused_linearization_matches_recurrence_and_explicit_jacobian():
    torch.manual_seed(0)

    config = make_config(input_dim=3, state_dim=5)
    cell = ParaGRUCell(config).to(dtype=torch.float64)

    previous_states = 0.5 * torch.randn(4, 13, 5, dtype=torch.float64)
    drivers = 0.5 * torch.randn(4, 13, 3, dtype=torch.float64)

    predicted_fused, jac_fused = cell.compute_linearization_diag_from_previous(
        previous_states=previous_states,
        drivers=drivers,
    )

    predicted_reference = cell.batched_recurrence_step(
        previous_states=previous_states,
        drivers=drivers,
    )
    jac_reference = cell._compute_jacobians_diag_from_previous(
        previous_states=previous_states,
        drivers=drivers,
    )

    assert predicted_fused.shape == (4, 13, 5)
    assert jac_fused.shape == (4, 13, 5)
    assert torch.max(torch.abs(predicted_fused -
                     predicted_reference)).item() < 1e-14
    assert torch.max(torch.abs(jac_fused - jac_reference)).item() == 0.0


def test_fused_linearization_jacobian_matches_autograd_diagonal():
    torch.manual_seed(1)

    config = make_config(input_dim=2, state_dim=3)
    cell = ParaGRUCell(config).to(dtype=torch.float64)

    x = 0.5 * torch.randn(2, 7, 2, dtype=torch.float64)
    states = cell.forward_states_sequential(x)
    previous_states = cell.roll_state(
        states=states,
        initial_state=torch.zeros(2, 3, dtype=torch.float64),
    )

    _, jac_fused = cell.compute_linearization_diag_from_previous(
        previous_states=previous_states,
        drivers=x,
    )

    jac_dense = cell.compute_jacobians_autograd(
        states=states,
        drivers=x,
    )
    jac_diag_autograd = torch.diagonal(
        jac_dense,
        dim1=-2,
        dim2=-1,
    )

    assert jac_fused.shape == (2, 7, 3)
    assert torch.max(torch.abs(jac_fused - jac_diag_autograd)).item() < 1e-8


def test_batched_deer_with_fused_linearization_matches_separate_jacobian_path():
    torch.manual_seed(2)

    config = make_config(input_dim=3, state_dim=4, num_iters=16)
    cell = ParaGRUCell(config).to(dtype=torch.float64)

    x = 0.5 * torch.randn(3, 12, 3, dtype=torch.float64)
    h0 = torch.zeros(3, 4, dtype=torch.float64)
    states_guess = cell.assemble_initial_guess_batched(
        drivers=x,
        initial_state=h0,
        guess_type="f0",
    )

    def jacobian_fn(previous_states, drivers):
        return cell._compute_jacobians_diag_from_previous(
            previous_states=previous_states,
            drivers=drivers,
        )

    def linearization_fn(previous_states, drivers):
        return cell.compute_linearization_diag_from_previous(
            previous_states=previous_states,
            drivers=drivers,
        )

    states_separate, info_separate = deer_alg_batched(
        f=cell.recurrence_step,
        initial_state=h0,
        states_guess=states_guess,
        drivers=x,
        num_iters=16,
        tol=1e-11,
        strict_tol=True,
        stopping_criterion="update",
        quasi=True,
        scan_backend="torch",
        jacobian_fn=jacobian_fn,
    )

    states_fused, info_fused = deer_alg_batched(
        f=cell.recurrence_step,
        initial_state=h0,
        states_guess=states_guess,
        drivers=x,
        num_iters=16,
        tol=1e-11,
        strict_tol=True,
        stopping_criterion="update",
        quasi=True,
        scan_backend="torch",
        linearization_fn=linearization_fn,
    )

    assert states_fused.shape == states_separate.shape
    assert info_separate["linearization_backend"] == "separate"
    assert info_fused["linearization_backend"] == "custom"
    assert info_fused["jacobian_backend"] == "custom"
    assert torch.max(torch.abs(states_fused - states_separate)).item() < 1e-10


def test_base_forward_deer_uses_fused_linearization_for_explicit_paragru():
    torch.manual_seed(3)

    config = make_config(input_dim=3, state_dim=5, num_iters=20)
    cell = ParaGRUCell(config).to(dtype=torch.float64)

    x = 0.5 * torch.randn(4, 14, 3, dtype=torch.float64)

    y_seq = cell(x, mode="sequential")
    y_deer = cell(x, mode="deer")

    assert y_seq.shape == (4, 14, 5)
    assert y_deer.shape == (4, 14, 5)
    assert torch.max(torch.abs(y_seq - y_deer)).item() < 1e-7
    assert len(cell.last_deer_infos) == 1
    assert cell.last_deer_infos[0]["batched"] is True
    assert cell.last_deer_infos[0]["jacobian_backend"] == "custom"
    assert cell.last_deer_infos[0]["linearization_backend"] == "custom"


def test_fused_linearization_backward_smoke():
    torch.manual_seed(4)

    config = make_config(input_dim=2, state_dim=3, num_iters=10)
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
