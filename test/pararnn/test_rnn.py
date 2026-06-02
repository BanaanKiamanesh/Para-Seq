import pytest
import torch
from torch import nn
from torch.func import jacrev

from src.pararnn import ParaRNN, ParaRNNCell, make_pararnn_deer_config


def make_rnn(
    *,
    input_size=3,
    hidden_size=4,
    mode="sequential",
    nonlinearity="tanh",
    num_iters=24,
    batch_first=True,
    dtype=torch.float64,
):
    rnn = ParaRNN(
        input_size=input_size,
        hidden_size=hidden_size,
        mode=mode,
        batch_first=batch_first,
        nonlinearity=nonlinearity,
        num_iters=num_iters,
        tol=1e-11 if dtype == torch.float64 else 1e-4,
        strict_tol=(dtype == torch.float64),
        dtype=dtype,
    ).to(dtype=dtype)

    # Keep Newton/DEER tests contractive and stable.
    with torch.no_grad():
        rnn.weight_hh.mul_(0.15)
        rnn.weight_ih.mul_(0.35)
        if rnn.bias_ih is not None:
            rnn.bias_ih.mul_(0.15)
        if rnn.bias_hh is not None:
            rnn.bias_hh.mul_(0.15)

    return rnn


def copy_to_torch_rnn(pararnn: ParaRNN, torch_rnn: nn.RNN) -> None:
    with torch.no_grad():
        torch_rnn.weight_ih_l0.copy_(pararnn.weight_ih)
        torch_rnn.weight_hh_l0.copy_(pararnn.weight_hh)
        if pararnn.bias_ih is not None:
            torch_rnn.bias_ih_l0.copy_(pararnn.bias_ih)
            torch_rnn.bias_hh_l0.copy_(pararnn.bias_hh)


def test_pararnn_cell_is_single_step_and_backpropagates():
    torch.manual_seed(0)

    cell = ParaRNNCell(input_size=3, hidden_size=4, dtype=torch.float64)
    x = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    h = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)

    h_next = cell(x, h)
    loss = h_next.square().mean()
    loss.backward()

    assert h_next.shape == (5, 4)
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert h.grad is not None and torch.isfinite(h.grad).all()
    assert cell.weight_ih.grad is not None
    assert cell.weight_hh.grad is not None
    assert cell.bias_ih is not None and cell.bias_ih.grad is not None
    assert cell.bias_hh is not None and cell.bias_hh.grad is not None

    x_single = torch.randn(3, dtype=torch.float64)
    h_single = torch.randn(4, dtype=torch.float64)
    assert cell(x_single, h_single).shape == (4,)


def test_pararnn_matches_torch_rnn_sequential_tanh():
    torch.manual_seed(1)

    rnn = make_rnn(
        input_size=3,
        hidden_size=4,
        mode="sequential",
        batch_first=True,
    )
    torch_rnn = nn.RNN(
        input_size=3,
        hidden_size=4,
        nonlinearity="tanh",
        batch_first=True,
        dtype=torch.float64,
    )
    copy_to_torch_rnn(rnn, torch_rnn)

    x = torch.randn(2, 7, 3, dtype=torch.float64)
    h0 = torch.randn(1, 2, 4, dtype=torch.float64)

    y_para, h_para = rnn(x, h0, mode="sequential")
    y_torch, h_torch = torch_rnn(x, h0)

    assert torch.max(torch.abs(y_para - y_torch)).item() < 1e-12
    assert torch.max(torch.abs(h_para - h_torch)).item() < 1e-12


def test_pararnn_explicit_dense_jacobian_matches_autograd():
    torch.manual_seed(2)

    cell = ParaRNNCell(input_size=3, hidden_size=4, dtype=torch.float64)
    with torch.no_grad():
        cell.weight_hh.mul_(0.15)
        cell.weight_ih.mul_(0.35)

    previous_states = 0.15 * torch.randn(2, 5, 4, dtype=torch.float64)
    drivers = 0.20 * torch.randn(2, 5, 3, dtype=torch.float64)

    _, jac_dense = cell.compute_linearization_dense_from_previous(
        previous_states=previous_states,
        drivers=drivers,
    )

    def one_step(state, driver):
        return cell.recurrence_step(state, driver)

    jac_autograd = jacrev(one_step, argnums=0)(previous_states[0, 0], drivers[0, 0])

    assert torch.max(torch.abs(jac_dense[0, 0] - jac_autograd)).item() < 1e-10


def test_pararnn_sequence_module_returns_output_and_hn_like_torch_rnn():
    torch.manual_seed(3)

    rnn = make_rnn(
        input_size=3,
        hidden_size=4,
        mode="sequential",
        batch_first=True,
    )
    x = torch.randn(2, 7, 3, dtype=torch.float64)
    output, h_n = rnn(x)

    assert output.shape == (2, 7, 4)
    assert h_n.shape == (1, 2, 4)
    assert torch.max(torch.abs(h_n[0] - output[:, -1, :])).item() < 1e-12

    rnn_time_first = make_rnn(
        input_size=3,
        hidden_size=4,
        mode="sequential",
        batch_first=False,
    )
    rnn_time_first.load_state_dict(rnn.state_dict())
    output_tf, h_n_tf = rnn_time_first(x.transpose(0, 1).contiguous())

    assert output_tf.shape == (7, 2, 4)
    assert h_n_tf.shape == (1, 2, 4)
    assert torch.max(torch.abs(output_tf.transpose(0, 1) - output)).item() < 1e-12
    assert torch.max(torch.abs(h_n_tf - h_n)).item() < 1e-12

    x_unbatched = torch.randn(7, 3, dtype=torch.float64)
    output_u, h_n_u = rnn_time_first(x_unbatched)
    assert output_u.shape == (7, 4)
    assert h_n_u.shape == (1, 4)


def test_pararnn_deer_matches_sequential_and_backpropagates():
    torch.manual_seed(4)

    seq_rnn = make_rnn(
        input_size=3,
        hidden_size=4,
        mode="sequential",
        num_iters=32,
    )
    deer_rnn = make_rnn(
        input_size=3,
        hidden_size=4,
        mode="deer",
        num_iters=32,
    )
    deer_rnn.load_state_dict(seq_rnn.state_dict())

    x_base = 0.20 * torch.randn(2, 10, 3, dtype=torch.float64)
    h0_base = 0.05 * torch.randn(1, 2, 4, dtype=torch.float64)

    seq_out, seq_h = seq_rnn(
        x_base.detach(),
        h0_base.detach(),
        mode="sequential",
    )

    x = x_base.clone().requires_grad_(True)
    h0 = h0_base.clone().requires_grad_(True)

    deer_out, deer_h = deer_rnn(x, h0, mode="deer")
    loss = deer_out.square().mean() + 0.1 * deer_h.square().mean()
    loss.backward()

    assert torch.max(torch.abs(deer_out.detach() - seq_out)).item() < 1e-6
    assert torch.max(torch.abs(deer_h.detach() - seq_h)).item() < 1e-6
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert h0.grad is not None and torch.isfinite(h0.grad).all()
    assert deer_rnn.weight_hh.grad is not None
    assert torch.isfinite(deer_rnn.weight_hh.grad).all()
    assert deer_rnn.last_deer_infos[0]["jacobian_backend"] == "explicit_dense"
    assert deer_rnn.last_deer_infos[0]["cell_variant"] == "dense_vanilla_tanh"


def test_pararnn_tiny_training_step():
    torch.manual_seed(5)

    class TinyRNNClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = ParaRNN(
                input_size=3,
                hidden_size=5,
                batch_first=True,
                mode="deer",
                num_iters=16,
                tol=1e-7,
                dtype=torch.float64,
            )
            with torch.no_grad():
                self.rnn.weight_hh.mul_(0.10)
                self.rnn.weight_ih.mul_(0.30)
            self.head = nn.Linear(5, 2).to(dtype=torch.float64)

        def forward(self, x):
            _, h_n = self.rnn(x)
            return self.head(h_n[-1])

    x = 0.4 * torch.randn(16, 8, 3, dtype=torch.float64)
    y = (x[:, -1, 0] + x[:, :, 1].mean(dim=1) > 0.0).long()
    model = TinyRNNClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-2)
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        initial_loss = criterion(model(x), y).item()

    for _ in range(20):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_loss = criterion(model(x), y).item()

    assert final_loss < initial_loss


def test_invalid_pararnn_configs_raise():
    with pytest.raises(NotImplementedError, match="num_layers=1"):
        ParaRNN(input_size=3, hidden_size=4, num_layers=2)

    with pytest.raises(NotImplementedError, match="bidirectional=False"):
        ParaRNN(input_size=3, hidden_size=4, bidirectional=True)

    with pytest.raises(ValueError, match="nonlinearity"):
        ParaRNN(input_size=3, hidden_size=4, nonlinearity="sigmoid")

    bad_cfg = make_pararnn_deer_config()
    bad_cfg.quasi = True
    rnn = ParaRNN(input_size=3, hidden_size=4, deer_config=bad_cfg, dtype=torch.float64)
    x = 0.5 * torch.randn(2, 7, 3, dtype=torch.float64)

    with pytest.raises(ValueError, match="quasi=False"):
        rnn(x, mode="deer")
