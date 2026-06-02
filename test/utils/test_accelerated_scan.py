import pytest
import torch

from src.pararnn import DeerNewtonConfig, ParaGRU
from src.utils.AccelScan import (
    diag_mat_scan_accel_batched,
    next_power_of_two,
    validate_accel_scan_inputs,
)
from src.utils.AssScan import diag_mat_scan


def test_accel_scan_input_helpers():
    assert [next_power_of_two(n) for n in [0, 1, 2, 3, 32, 33]] == [
        1,
        1,
        2,
        4,
        32,
        64,
    ]

    with pytest.raises(ValueError, match="same shape"):
        validate_accel_scan_inputs(
            torch.ones(2, 5, 3),
            torch.zeros(2, 5, 4),
            expected_ndim=3,
        )
    with pytest.raises(ValueError, match="requires CUDA"):
        validate_accel_scan_inputs(
            torch.ones(2, 5, 3),
            torch.zeros(2, 5, 3),
            expected_ndim=3,
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="accelerated_scan tests require CUDA.",
)
def test_accel_scan_backend_matches_torch_scan():
    warp_module = pytest.importorskip(
        "accelerated_scan.warp",
        reason="accelerated_scan.warp is not installed.",
    )

    torch.manual_seed(0)
    device = torch.device("cuda")
    A = 0.25 * torch.rand(3, 37, 5, device=device) + 0.70
    b = 0.10 * torch.randn(3, 37, 5, device=device)
    A_ref, b_ref = diag_mat_scan(A, b, dim=1)
    A_accel, b_accel = diag_mat_scan_accel_batched(
        A=A,
        b=b,
        accel_scan_fn=warp_module.scan,
    )

    def make_rnn(scan_backend):
        return ParaGRU(
            input_size=3,
            hidden_size=5,
            mode="deer",
            batch_first=True,
            dtype=torch.float32,
            recurrent_init_scale=0.15,
            deer_config=DeerNewtonConfig(
                num_iters=4,
                tol=1e-4,
                strict_tol=False,
                stopping_criterion="update",
                initial_guess="f0",
                quasi=True,
                scan_backend=scan_backend,
                accel_module="warp",
                jacobian_backend="explicit",
                backward_backend="adjoint",
            ),
        )

    torch_rnn = make_rnn("torch").to(device=device)
    accel_rnn = make_rnn("accel_scan").to(device=device)
    accel_rnn.load_state_dict(torch_rnn.state_dict())
    x = 0.5 * torch.randn(4, 37, 3, device=device)
    y_torch, _ = torch_rnn(x)
    y_accel, _ = accel_rnn(x)
    torch.cuda.synchronize()

    assert torch.max(torch.abs(A_accel - A_ref)).item() < 1e-5
    assert torch.max(torch.abs(b_accel - b_ref)).item() < 1e-5
    assert torch.max(torch.abs(y_accel - y_torch)).item() < 5e-4
    assert accel_rnn.last_deer_infos[0]["scan_backend"] == "accel_scan"
