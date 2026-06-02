from src.utils.AssScan import diag_mat_scan
from src.pararnn.cells.para_gru import ParaGRUCell, ParaGRUConfig
from src.pararnn import DeerNewtonConfig
from src.algos.DEER import _diag_mat_scan_accel_batched
import src.algos.DEER as deer_module
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="accelerated_scan batched tests require CUDA.",
)


def load_warp_scan():
    warp_module = pytest.importorskip(
        "accelerated_scan.warp",
        reason="accelerated_scan.warp is not installed.",
    )
    return warp_module.scan


def make_config(
    input_dim=3,
    state_dim=4,
    batch_first=True,
    num_iters=4,
    scan_backend="accel_scan",
):
    return ParaGRUConfig(
        input_dim=input_dim,
        state_dim=state_dim,
        mode="deer",
        batch_first=batch_first,
        dtype=torch.float32,
        recurrent_init_scale=0.15,
        deer=DeerNewtonConfig(
            num_iters=num_iters,
            tol=1e-4,
            strict_tol=False,
            stopping_criterion="update",
            initial_guess="f0",
            quasi=True,
            scan_backend=scan_backend,
            accel_module="warp",
            jacobian_backend="explicit",
        ),
    )


def test_vectorized_batched_accel_scan_matches_torch_diag_scan():
    torch.manual_seed(0)
    scan = load_warp_scan()
    device = torch.device("cuda")

    batch_size = 3
    seq_len = 37
    state_dim = 5

    A = 0.25 * torch.rand(
        batch_size,
        seq_len,
        state_dim,
        device=device,
        dtype=torch.float32,
    ) + 0.70
    b = 0.10 * torch.randn(
        batch_size,
        seq_len,
        state_dim,
        device=device,
        dtype=torch.float32,
    )

    A_ref, b_ref = diag_mat_scan(A, b, dim=1)
    A_accel, b_accel = _diag_mat_scan_accel_batched(
        A=A,
        b=b,
        accel_scan_fn=scan,
    )

    torch.cuda.synchronize()

    assert A_accel.shape == (batch_size, seq_len, state_dim)
    assert b_accel.shape == (batch_size, seq_len, state_dim)
    assert torch.max(torch.abs(A_accel - A_ref)).item() < 1e-5
    assert torch.max(torch.abs(b_accel - b_ref)).item() < 1e-5


def test_vectorized_batched_accel_scan_matches_torch_diag_scan_with_chunking():
    torch.manual_seed(1)
    scan = load_warp_scan()
    device = torch.device("cuda")

    batch_size = 2
    seq_len = 100
    state_dim = 4

    A = 0.20 * torch.rand(
        batch_size,
        seq_len,
        state_dim,
        device=device,
        dtype=torch.float32,
    ) + 0.75
    b = 0.05 * torch.randn(
        batch_size,
        seq_len,
        state_dim,
        device=device,
        dtype=torch.float32,
    )

    old_max_len = deer_module._ACCEL_SCAN_MAX_LEN
    deer_module._ACCEL_SCAN_MAX_LEN = 64

    try:
        A_ref, b_ref = diag_mat_scan(A, b, dim=1)
        A_accel, b_accel = _diag_mat_scan_accel_batched(
            A=A,
            b=b,
            accel_scan_fn=scan,
        )
    finally:
        deer_module._ACCEL_SCAN_MAX_LEN = old_max_len

    torch.cuda.synchronize()

    assert A_accel.shape == (batch_size, seq_len, state_dim)
    assert b_accel.shape == (batch_size, seq_len, state_dim)
    assert torch.max(torch.abs(A_accel - A_ref)).item() < 1e-5
    assert torch.max(torch.abs(b_accel - b_ref)).item() < 1e-5


def test_paragru_explicit_quasi_deer_accel_scan_matches_torch_scan():
    torch.manual_seed(2)
    device = torch.device("cuda")

    torch_config = make_config(
        input_dim=3,
        state_dim=5,
        num_iters=4,
        scan_backend="torch",
    )
    accel_config = make_config(
        input_dim=3,
        state_dim=5,
        num_iters=4,
        scan_backend="accel_scan",
    )

    torch_cell = ParaGRUCell(torch_config).to(
        device=device, dtype=torch.float32)
    accel_cell = ParaGRUCell(accel_config).to(
        device=device, dtype=torch.float32)
    accel_cell.load_state_dict(torch_cell.state_dict())

    x = 0.5 * torch.randn(4, 37, 3, device=device, dtype=torch.float32)

    y_torch = torch_cell(x, mode="deer")
    y_accel = accel_cell(x, mode="deer")

    torch.cuda.synchronize()

    assert y_torch.shape == (4, 37, 5)
    assert y_accel.shape == (4, 37, 5)
    assert torch.max(torch.abs(y_accel - y_torch)).item() < 5e-4
    assert len(accel_cell.last_deer_infos) == 1
    assert accel_cell.last_deer_infos[0]["batched"] is True
    assert accel_cell.last_deer_infos[0]["scan_backend"] == "accel_scan"
    assert accel_cell.last_deer_infos[0]["linearization_backend"] == "custom"


def test_paragru_explicit_quasi_deer_accel_scan_backward_smoke_if_supported():
    torch.manual_seed(3)
    device = torch.device("cuda")

    config = make_config(
        input_dim=2,
        state_dim=3,
        num_iters=3,
        scan_backend="accel_scan",
    )
    cell = ParaGRUCell(config).to(device=device, dtype=torch.float32)

    x = torch.randn(
        2,
        33,
        2,
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )

    y = cell(x, mode="deer")
    loss = y.square().mean()

    try:
        loss.backward()
    except RuntimeError as error:
        pytest.skip(
            "accelerated_scan.warp backward is not supported in this environment: "
            f"{error}"
        )

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    for param in cell.parameters():
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()
