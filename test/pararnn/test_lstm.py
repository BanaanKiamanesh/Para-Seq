import pytest
import torch
from torch import nn
from torch.func import jacrev

from src.pararnn import ParaLSTM, ParaLSTMCell, make_paralstm_deer_config
from src.pararnn.cells.para_lstm import _flat_to_blocks
from src.utils.BlockScan import block2_mat_scan


def make_lstm(
    *,
    input_size=3,
    hidden_size=4,
    mode="sequential",
    num_iters=16,
    batch_first=True,
    dtype=torch.float64,
    recurrent_init_scale=0.08,
    peephole_init_scale=0.08,
):
    return ParaLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        mode=mode,
        batch_first=batch_first,
        num_iters=num_iters,
        tol=1e-11 if dtype == torch.float64 else 1e-4,
        strict_tol=(dtype == torch.float64),
        dtype=dtype,
        recurrent_init_scale=recurrent_init_scale,
        peephole_init_scale=peephole_init_scale,
        forget_bias_init_value=0.25,
    ).to(dtype=dtype)


def test_paralstm_cell_is_single_step_and_backpropagates():
    torch.manual_seed(0)

    cell = ParaLSTMCell(input_size=3, hidden_size=4, dtype=torch.float64)
    x = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    h = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)
    c = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)

    h_next, c_next = cell(x, (h, c))
    loss = h_next.square().mean() + 0.1 * c_next.square().mean()
    loss.backward()

    assert h_next.shape == (5, 4)
    assert c_next.shape == (5, 4)
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert h.grad is not None and torch.isfinite(h.grad).all()
    assert c.grad is not None and torch.isfinite(c.grad).all()
    assert cell.A.grad is not None
    assert cell.B.grad is not None
    assert cell.C.grad is not None
    assert cell.b.grad is not None

    assert cell.A.shape == (3, 4)
    assert cell.B.shape == (3, 3, 4)
    assert cell.C.shape == (2, 4)
    assert cell.b.shape == (3, 4)

    x_single = torch.randn(3, dtype=torch.float64)
    h_single = torch.randn(4, dtype=torch.float64)
    c_single = torch.randn(4, dtype=torch.float64)
    h_out, c_out = cell(x_single, (h_single, c_single))
    assert h_out.shape == (4,)
    assert c_out.shape == (4,)


def test_block2_scan_matches_sequential_loop():
    torch.manual_seed(1)

    B, T, H = 2, 9, 3
    A = 0.1 * torch.randn(B, T, H, 2, 2, dtype=torch.float64)
    b = torch.randn(B, T, H, 2, dtype=torch.float64)

    _, scanned = block2_mat_scan(A, b, dim=1)

    state = torch.zeros(B, H, 2, dtype=torch.float64)
    ref = []
    for t in range(T):
        state = (A[:, t] @ state.unsqueeze(-1)).squeeze(-1) + b[:, t]
        ref.append(state)
    ref = torch.stack(ref, dim=1)

    assert torch.max(torch.abs(scanned - ref)).item() < 1e-12


def test_paralstm_explicit_block_jacobian_matches_autograd():
    torch.manual_seed(2)

    cell = ParaLSTMCell(input_size=3, hidden_size=4, dtype=torch.float64)
    previous_flat = 0.15 * torch.randn(2, 5, 8, dtype=torch.float64)
    drivers = 0.20 * torch.randn(2, 5, 3, dtype=torch.float64)

    previous_blocks = _flat_to_blocks(previous_flat, hidden_size=4)
    _, blocks = cell.compute_linearization_blocks_from_previous(
        previous_states=previous_blocks,
        drivers=drivers,
    )

    def one_step(state, driver):
        return cell.recurrence_step(state, driver)

    jac = jacrev(one_step, argnums=0)(previous_flat[0, 0], drivers[0, 0])

    for hidden_idx in range(4):
        rows = [hidden_idx, 4 + hidden_idx]
        cols = [hidden_idx, 4 + hidden_idx]
        block_dense = jac[rows][:, cols]
        assert torch.max(torch.abs(blocks[0, 0, hidden_idx] - block_dense)).item() < 1e-10

    # Off-block derivatives should be zero because A and C are diagonal.
    dense_from_blocks = torch.zeros_like(jac)
    for hidden_idx in range(4):
        rows = [hidden_idx, 4 + hidden_idx]
        cols = [hidden_idx, 4 + hidden_idx]
        dense_from_blocks[rows[0], cols[0]] = blocks[0, 0, hidden_idx, 0, 0]
        dense_from_blocks[rows[0], cols[1]] = blocks[0, 0, hidden_idx, 0, 1]
        dense_from_blocks[rows[1], cols[0]] = blocks[0, 0, hidden_idx, 1, 0]
        dense_from_blocks[rows[1], cols[1]] = blocks[0, 0, hidden_idx, 1, 1]

    assert torch.max(torch.abs(jac - dense_from_blocks)).item() < 1e-10


def test_paralstm_sequence_module_returns_output_and_hc_like_torch_lstm():
    torch.manual_seed(3)

    lstm = make_lstm(
        input_size=3,
        hidden_size=4,
        mode="sequential",
        batch_first=True,
    )
    x = torch.randn(2, 7, 3, dtype=torch.float64)
    output, (h_n, c_n) = lstm(x)

    assert output.shape == (2, 7, 4)
    assert h_n.shape == (1, 2, 4)
    assert c_n.shape == (1, 2, 4)
    assert torch.max(torch.abs(h_n[0] - output[:, -1, :])).item() < 1e-12

    lstm_time_first = make_lstm(
        input_size=3,
        hidden_size=4,
        mode="sequential",
        batch_first=False,
    )
    lstm_time_first.load_state_dict(lstm.state_dict())
    output_tf, (h_n_tf, c_n_tf) = lstm_time_first(
        x.transpose(0, 1).contiguous()
    )

    assert output_tf.shape == (7, 2, 4)
    assert h_n_tf.shape == (1, 2, 4)
    assert c_n_tf.shape == (1, 2, 4)
    assert torch.max(torch.abs(output_tf.transpose(0, 1) - output)).item() < 1e-12
    assert torch.max(torch.abs(h_n_tf - h_n)).item() < 1e-12
    assert torch.max(torch.abs(c_n_tf - c_n)).item() < 1e-12

    x_unbatched = torch.randn(7, 3, dtype=torch.float64)
    output_u, (h_n_u, c_n_u) = lstm_time_first(x_unbatched)
    assert output_u.shape == (7, 4)
    assert h_n_u.shape == (1, 4)
    assert c_n_u.shape == (1, 4)


def test_paralstm_deer_matches_sequential_and_backpropagates():
    torch.manual_seed(4)

    seq_lstm = make_lstm(
        input_size=3,
        hidden_size=4,
        mode="sequential",
        num_iters=32,
    )
    deer_lstm = make_lstm(
        input_size=3,
        hidden_size=4,
        mode="deer",
        num_iters=32,
    )
    deer_lstm.load_state_dict(seq_lstm.state_dict())

    x_base = 0.20 * torch.randn(2, 10, 3, dtype=torch.float64)
    h0_base = 0.05 * torch.randn(1, 2, 4, dtype=torch.float64)
    c0_base = 0.05 * torch.randn(1, 2, 4, dtype=torch.float64)

    seq_out, (seq_h, seq_c) = seq_lstm(
        x_base.detach(),
        (h0_base.detach(), c0_base.detach()),
        mode="sequential",
    )

    x = x_base.clone().requires_grad_(True)
    h0 = h0_base.clone().requires_grad_(True)
    c0 = c0_base.clone().requires_grad_(True)

    deer_out, (deer_h, deer_c) = deer_lstm(x, (h0, c0), mode="deer")
    loss = (
        deer_out.square().mean()
        + 0.1 * deer_h.square().mean()
        + 0.1 * deer_c.square().mean()
    )
    loss.backward()

    assert torch.max(torch.abs(deer_out.detach() - seq_out)).item() < 1e-6
    assert torch.max(torch.abs(deer_h.detach() - seq_h)).item() < 1e-6
    assert torch.max(torch.abs(deer_c.detach() - seq_c)).item() < 1e-6
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert h0.grad is not None and torch.isfinite(h0.grad).all()
    assert c0.grad is not None and torch.isfinite(c0.grad).all()
    assert deer_lstm.C.grad is not None and torch.isfinite(deer_lstm.C.grad).all()
    assert deer_lstm.last_deer_infos[0]["jacobian_backend"] == "explicit_block2"
    assert deer_lstm.last_deer_infos[0]["cell_variant"] == "cifg_peephole_diag"


def test_paralstm_tiny_training_step():
    torch.manual_seed(5)

    class TinyLSTMClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = ParaLSTM(
                input_size=3,
                hidden_size=5,
                batch_first=True,
                mode="deer",
                num_iters=12,
                tol=1e-7,
                dtype=torch.float64,
                recurrent_init_scale=0.05,
                peephole_init_scale=0.05,
                forget_bias_init_value=0.25,
            )
            self.head = nn.Linear(5, 2).to(dtype=torch.float64)

        def forward(self, x):
            _, (h_n, _) = self.rnn(x)
            return self.head(h_n[-1])

    x = 0.4 * torch.randn(12, 7, 3, dtype=torch.float64)
    y = (x[:, -1, 0] + x[:, :, 1].mean(dim=1) > 0.0).long()
    model = TinyLSTMClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-2)
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        initial_loss = criterion(model(x), y).item()

    for _ in range(8):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_loss = criterion(model(x), y).item()

    assert final_loss < initial_loss


def test_invalid_paralstm_configs_raise():
    with pytest.raises(NotImplementedError, match="num_layers=1"):
        ParaLSTM(input_size=3, hidden_size=4, num_layers=2)

    with pytest.raises(NotImplementedError, match="bidirectional=False"):
        ParaLSTM(input_size=3, hidden_size=4, bidirectional=True)

    bad_cfg = make_paralstm_deer_config()
    bad_cfg.scan_backend = "accel_scan"  # type: ignore[assignment]
    lstm = ParaLSTM(input_size=3, hidden_size=4, deer_config=bad_cfg, mode="deer")
    x = torch.randn(2, 5, 3)

    with pytest.raises(ValueError, match="scan_backend='torch'"):
        lstm(x)


def test_paralstm_scalar_quasi_diag_matches_block_diagonal():
    torch.manual_seed(51)

    cell = ParaLSTMCell(input_size=3, hidden_size=5, dtype=torch.float64)

    previous_flat = 0.15 * torch.randn(2, 6, 10, dtype=torch.float64)
    drivers = 0.20 * torch.randn(2, 6, 3, dtype=torch.float64)

    previous_blocks = _flat_to_blocks(previous_flat, hidden_size=5)

    predicted_blocks, jac_blocks = cell.compute_linearization_blocks_from_previous(
        previous_states=previous_blocks,
        drivers=drivers,
    )

    predicted_flat, jac_diag = cell.compute_linearization_diag_from_previous_flat(
        previous_states=previous_flat,
        drivers=drivers,
    )

    expected_diag = torch.cat(
        [
            jac_blocks[..., 0, 0],
            jac_blocks[..., 1, 1],
        ],
        dim=-1,
    )

    assert torch.max(torch.abs(predicted_flat[:, :, :5] - predicted_blocks[..., 0])).item() < 1e-12
    assert torch.max(torch.abs(predicted_flat[:, :, 5:] - predicted_blocks[..., 1])).item() < 1e-12
    assert torch.max(torch.abs(jac_diag - expected_diag)).item() < 1e-12


def test_paralstm_scalar_quasi_deer_torch_backend_runs_and_backpropagates():
    torch.manual_seed(52)

    lstm = ParaLSTM(
        input_size=3,
        hidden_size=5,
        batch_first=True,
        mode="deer",
        backend="quasi_autograd",
        scan_backend="torch",
        num_iters=16,
        tol=1e-7,
        dtype=torch.float64,
        recurrent_init_scale=0.03,
        peephole_init_scale=0.03,
        forget_bias_init_value=0.20,
    )

    x = (0.15 * torch.randn(4, 10, 3, dtype=torch.float64)).detach().requires_grad_(True)

    output, (h_n, c_n) = lstm(x)

    loss = (
        output.square().mean()
        + 0.1 * h_n.square().mean()
        + 0.1 * c_n.square().mean()
    )
    loss.backward()

    assert output.shape == (4, 10, 5)
    assert h_n.shape == (1, 4, 5)
    assert c_n.shape == (1, 4, 5)
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert lstm.A.grad is not None and torch.isfinite(lstm.A.grad).all()
    assert lstm.B.grad is not None and torch.isfinite(lstm.B.grad).all()
    assert lstm.C.grad is not None and torch.isfinite(lstm.C.grad).all()

    info = lstm.last_deer_infos[0]
    assert info["quasi"] is True
    assert info["scan_backend"] == "torch"
    assert info["jacobian_backend"] == "explicit_scalar_diag_from_block2"
    assert info["linearization_backend"] == "custom_scalar_diag_from_block2"
    assert info["paralstm_deer_kind"] == "scalar_quasi"


def test_paralstm_scalar_quasi_deer_accel_scan_backend_if_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")

    try:
        import accelerated_scan.warp  # noqa: F401
    except Exception:
        pytest.skip("accelerated_scan.warp is not available.")

    torch.manual_seed(53)

    lstm = ParaLSTM(
        input_size=3,
        hidden_size=4,
        batch_first=True,
        mode="deer",
        backend="quasi_deer_autograd_accel_scan",
        scan_backend="accel_scan",
        num_iters=8,
        tol=1e-4,
        dtype=torch.float32,
        recurrent_init_scale=0.03,
        peephole_init_scale=0.03,
        forget_bias_init_value=0.20,
        device="cuda",
    )

    x = (
        0.10
        * torch.randn(
            2,
            64,
            3,
            device="cuda",
            dtype=torch.float32,
        )
    ).detach().requires_grad_(True)

    output, (h_n, c_n) = lstm(x)

    loss = (
        output.square().mean()
        + 0.1 * h_n.square().mean()
        + 0.1 * c_n.square().mean()
    )
    loss.backward()

    assert output.shape == (2, 64, 4)
    assert h_n.shape == (1, 2, 4)
    assert c_n.shape == (1, 2, 4)
    assert x.grad is not None and torch.isfinite(x.grad).all()

    info = lstm.last_deer_infos[0]
    assert info["quasi"] is True
    assert info["scan_backend"] == "accel_scan"
    assert info["jacobian_backend"] == "explicit_scalar_diag_from_block2"

