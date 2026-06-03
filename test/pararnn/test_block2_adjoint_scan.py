import pytest
import torch

from src.pararnn.adjoint import reverse_block2_adjoint_scan


def _reverse_block2_adjoint_loop(jacobian_blocks, grad_states):
    """Sequential reference for lambda_t = g_t + J_{t+1}^T lambda_{t+1}."""
    if grad_states.shape[1] == 0:
        return grad_states.clone()

    out = torch.empty_like(grad_states)
    lam = torch.zeros_like(grad_states[:, 0])

    for t in range(grad_states.shape[1] - 1, -1, -1):
        if t == grad_states.shape[1] - 1:
            lam = grad_states[:, t]
        else:
            lam = grad_states[:, t] + (
                jacobian_blocks[:, t + 1].transpose(-1, -2)
                @ lam.unsqueeze(-1)
            ).squeeze(-1)

        out[:, t] = lam

    return out


def test_reverse_block2_adjoint_scan_matches_loop_reference():
    torch.manual_seed(6101)

    jacobian_blocks = 0.15 * torch.randn(
        2,
        7,
        3,
        2,
        2,
        dtype=torch.float64,
    )
    grad_states = torch.randn(
        2,
        7,
        3,
        2,
        dtype=torch.float64,
    )

    actual = reverse_block2_adjoint_scan(
        jacobian_blocks=jacobian_blocks,
        grad_states=grad_states,
    )
    expected = _reverse_block2_adjoint_loop(
        jacobian_blocks=jacobian_blocks,
        grad_states=grad_states,
    )

    assert torch.allclose(actual, expected, atol=1e-12, rtol=0.0)


def test_reverse_block2_adjoint_scan_single_step_is_identity_on_grad():
    torch.manual_seed(6102)

    jacobian_blocks = 0.15 * torch.randn(
        2,
        1,
        3,
        2,
        2,
        dtype=torch.float64,
    )
    grad_states = torch.randn(
        2,
        1,
        3,
        2,
        dtype=torch.float64,
    )

    actual = reverse_block2_adjoint_scan(
        jacobian_blocks=jacobian_blocks,
        grad_states=grad_states,
    )

    assert torch.equal(actual, grad_states)


def test_reverse_block2_adjoint_scan_rejects_bad_shapes():
    jacobian_blocks = torch.randn(2, 5, 3, 2, 2, dtype=torch.float64)
    grad_states = torch.randn(2, 5, 3, 2, dtype=torch.float64)

    with pytest.raises(ValueError):
        reverse_block2_adjoint_scan(
            jacobian_blocks=jacobian_blocks[..., 0],
            grad_states=grad_states,
        )

    with pytest.raises(ValueError):
        reverse_block2_adjoint_scan(
            jacobian_blocks=jacobian_blocks,
            grad_states=grad_states[..., 0],
        )

    with pytest.raises(ValueError):
        reverse_block2_adjoint_scan(
            jacobian_blocks=jacobian_blocks[:, :-1],
            grad_states=grad_states,
        )
