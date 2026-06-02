import pytest
import torch
from torch import nn

from src.algos.DEER import deer_alg, deer_alg_batched, sequential_rollout
from src.pararnn import (
    DeerNewtonConfig,
    ParaGRU,
    ParaGRUCell,
    ParaRNNConfig,
    ParaRNNDeerConfig,
    TanhDeerRNNCell,
    make_paragru_deer_config,
)
from src.utils.AdjScan import (
    reverse_diag_adjoint_loop,
    reverse_diag_adjoint_scan,
)


def tanh_config():
    return ParaRNNConfig(
        input_dim=3,
        state_dim=4,
        mode="deer",
        batch_first=True,
        dtype=torch.float64,
        deer=DeerNewtonConfig(
            num_iters=16,
            tol=1e-10,
            strict_tol=True,
            stopping_criterion="update",
            initial_guess="f0",
            scan_backend="torch",
        ),
    )


def make_gru(
    *,
    input_size=3,
    hidden_size=4,
    mode="deer",
    num_iters=24,
    backend="autograd",
    scan_backend="torch",
    batch_first=True,
    dtype=torch.float64,
    recurrent_init_scale=0.20,
):
    return ParaGRU(
        input_size=input_size,
        hidden_size=hidden_size,
        mode=mode,
        batch_first=batch_first,
        backend=backend,
        scan_backend=scan_backend,
        num_iters=num_iters,
        tol=1e-11 if dtype == torch.float64 else 1e-4,
        strict_tol=(dtype == torch.float64),
        dtype=dtype,
        recurrent_init_scale=recurrent_init_scale,
    ).to(dtype=dtype)


def collect_grads(rnn, x, hx=None, mode=None):
    output, h_n = rnn(x, hx, mode=mode)
    loss = output.square().mean() + 0.1 * h_n[-1].sum()
    loss.backward()

    result = {
        "output": output.detach(),
        "h_n": h_n.detach(),
        "x_grad": x.grad.detach().clone(),
        "A_grad": rnn.A.grad.detach().clone(),
        "B_grad": rnn.B.grad.detach().clone(),
        "b_grad": rnn.b.grad.detach().clone(),
    }
    if hx is not None:
        result["hx_grad"] = hx.grad.detach().clone()

    return result


def test_public_aliases_and_config_compatibility():
    config = ParaRNNDeerConfig(input_dim=3, state_dim=4, dtype=torch.float64)
    cell = ParaGRUCell(input_size=3, hidden_size=4, dtype=torch.float64)
    rnn = ParaGRU(input_size=3, hidden_size=4, dtype=torch.float64)

    assert isinstance(config, ParaRNNConfig)
    assert config.output_dim == config.state_dim
    assert isinstance(cell, nn.Module)
    assert isinstance(rnn, nn.Module)


def test_paragru_cell_is_single_step_and_backpropagates():
    torch.manual_seed(0)

    cell = ParaGRUCell(input_size=3, hidden_size=4, dtype=torch.float64)
    x = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    h = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)

    h_next = cell(x, h)
    loss = h_next.square().mean()
    loss.backward()

    assert h_next.shape == (5, 4)
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert h.grad is not None and torch.isfinite(h.grad).all()
    assert cell.A.grad is not None

    x_single = torch.randn(3, dtype=torch.float64)
    h_single = torch.randn(4, dtype=torch.float64)
    assert cell(x_single, h_single).shape == (4,)


def test_tanh_deer_matches_sequential_and_backpropagates():
    torch.manual_seed(1)

    cell = TanhDeerRNNCell(tanh_config()).to(dtype=torch.float64)
    x = torch.randn(2, 24, 3, dtype=torch.float64, requires_grad=True)

    y_seq = cell(x.detach(), mode="sequential")
    y_deer = cell(x, mode="deer")
    y_deer.square().mean().backward()

    assert torch.max(torch.abs(y_seq - y_deer.detach())).item() < 1e-7
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_paragru_sequence_module_returns_output_and_hn_like_torch_gru():
    torch.manual_seed(2)

    rnn = make_gru(input_size=3, hidden_size=4,
                   mode="sequential", batch_first=True)
    x = torch.randn(2, 7, 3, dtype=torch.float64)
    output, h_n = rnn(x)

    assert output.shape == (2, 7, 4)
    assert h_n.shape == (1, 2, 4)
    assert torch.max(torch.abs(h_n[0] - output[:, -1, :])).item() < 1e-12

    rnn_time_first = make_gru(
        input_size=3,
        hidden_size=4,
        mode="sequential",
        batch_first=False,
    )
    rnn_time_first.load_state_dict(rnn.state_dict())
    output_tf, h_n_tf = rnn_time_first(x.transpose(0, 1).contiguous())

    assert output_tf.shape == (7, 2, 4)
    assert h_n_tf.shape == (1, 2, 4)
    assert torch.max(
        torch.abs(output_tf.transpose(0, 1) - output)).item() < 1e-12
    assert torch.max(torch.abs(h_n_tf - h_n)).item() < 1e-12

    x_unbatched = torch.randn(7, 3, dtype=torch.float64)
    output_u, h_n_u = rnn_time_first(x_unbatched)
    assert output_u.shape == (7, 4)
    assert h_n_u.shape == (1, 4)


def test_paragru_deer_matches_sequential_and_explicit_jacobian():
    torch.manual_seed(3)

    rnn = make_gru(input_size=3, hidden_size=4,
                   mode="deer", backend="autograd")
    x_leaf = torch.randn(2, 18, 3, dtype=torch.float64, requires_grad=True)
    x = 0.5 * x_leaf

    y_seq, h_seq = rnn(x.detach(), mode="sequential")
    y_deer, h_deer = rnn(x, mode="deer")
    y_deer.square().mean().backward()

    states = rnn.forward_states_sequential(x.detach())
    jac_diag = rnn.compute_jacobians_diag(states=states, drivers=x.detach())
    jac_dense = rnn.compute_jacobians_autograd(
        states=states,
        drivers=x.detach(),
    )

    assert torch.max(torch.abs(y_seq - y_deer.detach())).item() < 1e-7
    assert torch.max(torch.abs(h_seq - h_deer.detach())).item() < 1e-7
    assert x_leaf.grad is not None
    assert torch.max(
        torch.abs(jac_diag - torch.diagonal(jac_dense, dim1=-2, dim2=-1))
    ).item() < 1e-8


def test_batched_rollout_and_deer_match_per_sample_references():
    torch.manual_seed(4)

    rnn = make_gru(
        input_size=3,
        hidden_size=4,
        mode="deer",
        backend="autograd",
        num_iters=16,
    )
    x = 0.5 * torch.randn(3, 12, 3, dtype=torch.float64)
    h0 = torch.zeros(3, 4, dtype=torch.float64)

    seq_batched = rnn.forward_states_sequential(x, initial_state=h0)
    seq_ref = torch.stack(
        [
            sequential_rollout(rnn.recurrence_step, h0[i], x[i])
            for i in range(x.shape[0])
        ],
        dim=0,
    )

    states_guess = rnn.assemble_initial_guess_batched(
        drivers=x,
        initial_state=h0,
        guess_type="f0",
    )

    def jacobian_fn(previous_states, drivers):
        return rnn._compute_jacobians_diag_from_previous(
            previous_states=previous_states,
            drivers=drivers,
        )

    deer_batched, info = deer_alg_batched(
        f=rnn.recurrence_step,
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

    deer_ref = []
    for i in range(x.shape[0]):
        states_i, _ = deer_alg(
            f=rnn.recurrence_step,
            initial_state=h0[i],
            states_guess=states_guess[i],
            drivers=x[i],
            num_iters=16,
            tol=1e-11,
            strict_tol=True,
            stopping_criterion="update",
            quasi=True,
            scan_backend="torch",
            jacobian_fn=lambda states, drivers, i=i: jacobian_fn(
                states.unsqueeze(0),
                drivers.unsqueeze(0),
            ).squeeze(0),
        )
        deer_ref.append(states_i)

    assert torch.max(torch.abs(seq_batched - seq_ref)).item() < 1e-12
    assert info["batched"] is True
    assert torch.max(
        torch.abs(deer_batched - torch.stack(deer_ref, dim=0))
    ).item() < 1e-10


def test_explicit_quasi_deer_and_fused_linearization():
    torch.manual_seed(5)

    explicit_rnn = make_gru(
        input_size=3,
        hidden_size=4,
        mode="deer",
        backend="autograd",
        num_iters=16,
    )
    autograd_rnn = make_gru(
        input_size=3,
        hidden_size=4,
        mode="deer",
        backend="autograd",
        num_iters=16,
    )
    autograd_rnn.load_state_dict(explicit_rnn.state_dict())

    x = 0.5 * torch.randn(3, 12, 3, dtype=torch.float64)
    h0 = torch.zeros(3, 4, dtype=torch.float64)
    states_guess = explicit_rnn.assemble_initial_guess_batched(
        drivers=x,
        initial_state=h0,
        guess_type="f0",
    )

    y_explicit, _ = explicit_rnn(x, mode="deer")
    y_autograd, _ = autograd_rnn(x, mode="deer")
    states_fused, info_fused = deer_alg_batched(
        f=explicit_rnn.recurrence_step,
        initial_state=h0,
        states_guess=states_guess,
        drivers=x,
        num_iters=16,
        tol=1e-11,
        strict_tol=True,
        stopping_criterion="update",
        quasi=True,
        scan_backend="torch",
        linearization_fn=explicit_rnn.compute_linearization_diag_from_previous,
    )

    assert torch.max(torch.abs(y_explicit - y_autograd)).item() < 1e-7
    assert info_fused["linearization_backend"] == "custom"
    assert torch.max(torch.abs(states_fused - y_explicit)).item() < 1e-7


def test_adjoint_scan_and_backward_match_references():
    torch.manual_seed(6)

    jacobian_diag = 0.5 * torch.randn(3, 11, 4, dtype=torch.float64)
    grad_states = torch.randn(3, 11, 4, dtype=torch.float64)
    assert torch.max(
        torch.abs(
            reverse_diag_adjoint_scan(
                jacobian_diag,
                grad_states,
                scan_backend="torch",
            )
            - reverse_diag_adjoint_loop(jacobian_diag, grad_states)
        )
    ).item() < 1e-12

    seq_rnn = make_gru(
        input_size=2,
        hidden_size=4,
        mode="sequential",
        backend="autograd",
    )
    adj_rnn = make_gru(
        input_size=2,
        hidden_size=4,
        mode="deer",
        backend="adjoint",
        num_iters=32,
    )
    adj_rnn.load_state_dict(seq_rnn.state_dict())

    x_base = 0.35 * torch.randn(3, 9, 2, dtype=torch.float64)
    hx_base = 0.20 * torch.randn(1, 3, 4, dtype=torch.float64)

    seq = collect_grads(
        seq_rnn,
        x_base.clone().requires_grad_(True),
        hx_base.clone().requires_grad_(True),
        mode="sequential",
    )
    adj = collect_grads(
        adj_rnn,
        x_base.clone().requires_grad_(True),
        hx_base.clone().requires_grad_(True),
        mode="deer",
    )

    assert torch.max(torch.abs(adj["output"] - seq["output"])).item() < 1e-7
    assert torch.max(torch.abs(adj["h_n"] - seq["h_n"])).item() < 1e-7
    assert torch.max(torch.abs(adj["x_grad"] - seq["x_grad"])).item() < 1e-6
    assert torch.max(torch.abs(adj["hx_grad"] - seq["hx_grad"])).item() < 1e-6
    assert torch.max(torch.abs(adj["A_grad"] - seq["A_grad"])).item() < 1e-6
    assert torch.max(torch.abs(adj["B_grad"] - seq["B_grad"])).item() < 1e-6
    assert torch.max(torch.abs(adj["b_grad"] - seq["b_grad"])).item() < 1e-6


def test_invalid_explicit_and_adjoint_configs_raise():
    with pytest.raises(NotImplementedError, match="num_layers=1"):
        ParaGRU(input_size=3, hidden_size=4, num_layers=2)

    with pytest.raises(NotImplementedError, match="bidirectional=False"):
        ParaGRU(input_size=3, hidden_size=4, bidirectional=True)

    bad_cfg = make_paragru_deer_config(backend="adjoint")
    bad_cfg.quasi = False
    rnn = ParaGRU(input_size=3, hidden_size=4,
                  deer_config=bad_cfg, dtype=torch.float64)
    x = 0.5 * torch.randn(2, 7, 3, dtype=torch.float64)

    with pytest.raises(ValueError, match="quasi=True"):
        rnn(x)
