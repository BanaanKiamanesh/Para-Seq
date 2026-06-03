import pytest
import torch

import src.algos.FixedPoint as fixed_point_module
from src.algos.FixedPoint import fixed_point_alg_batched, fixed_point_step_batched
from src.pararnn import ParaGRU, ParaLSTM, ParaRNN
from src.pararnn.cells.para_gru import make_paragru_jacobi_config, make_paragru_picard_config
from src.pararnn.cells.para_lstm import make_paralstm_jacobi_config, make_paralstm_picard_config
from src.pararnn.cells.para_rnn import make_pararnn_jacobi_config, make_pararnn_picard_config
from src.utils.AssScan import diag_mat_scan


def _f(state, driver):
    return torch.tanh(0.13 * state + 0.17 * driver[..., : state.shape[-1]])


def _fake_accel_scan_batched(A, b, accel_scan_fn):
    assert accel_scan_fn is not None
    return diag_mat_scan(A, b, dim=1)


@pytest.mark.parametrize("method", ["jacobi", "picard"])
def test_fixed_point_accel_scan_step_matches_torch_backend(monkeypatch, method):
    torch.manual_seed(9301)

    monkeypatch.setattr(
        fixed_point_module,
        "_diag_mat_scan_accel_batched",
        _fake_accel_scan_batched,
    )

    initial_state = torch.randn(2, 3, dtype=torch.float64)
    states = torch.randn(2, 5, 3, dtype=torch.float64)
    drivers = torch.randn(2, 5, 3, dtype=torch.float64)

    expected = fixed_point_step_batched(
        f=_f,
        initial_state=initial_state,
        states=states,
        drivers=drivers,
        method=method,
        scan_backend="torch",
    )

    actual = fixed_point_step_batched(
        f=_f,
        initial_state=initial_state,
        states=states,
        drivers=drivers,
        method=method,
        scan_backend="accel_scan",
        accel_scan_fn=object(),
    )

    assert torch.allclose(actual, expected, atol=1e-12, rtol=0.0)


@pytest.mark.parametrize("method", ["jacobi", "picard"])
def test_fixed_point_accel_scan_algorithm_matches_torch_backend(monkeypatch, method):
    torch.manual_seed(9302)

    monkeypatch.setattr(
        fixed_point_module,
        "_diag_mat_scan_accel_batched",
        _fake_accel_scan_batched,
    )

    initial_state = torch.randn(2, 3, dtype=torch.float64)
    states_guess = torch.randn(2, 5, 3, dtype=torch.float64)
    drivers = torch.randn(2, 5, 3, dtype=torch.float64)

    expected, expected_info = fixed_point_alg_batched(
        f=_f,
        initial_state=initial_state,
        states_guess=states_guess,
        drivers=drivers,
        method=method,
        num_iters=4,
        tol=None,
        strict_tol=True,
        stopping_criterion="update",
        scan_backend="torch",
    )

    actual, actual_info = fixed_point_alg_batched(
        f=_f,
        initial_state=initial_state,
        states_guess=states_guess,
        drivers=drivers,
        method=method,
        num_iters=4,
        tol=None,
        strict_tol=True,
        stopping_criterion="update",
        scan_backend="accel_scan",
        accel_scan_fn=object(),
    )

    assert torch.allclose(actual, expected, atol=1e-12, rtol=0.0)
    assert expected_info["scan_backend"] == "torch"
    assert actual_info["scan_backend"] == "accel_scan"
    assert actual_info["solver"] == method


def test_fixed_point_accel_scan_requires_accel_scan_fn():
    torch.manual_seed(9303)

    initial_state = torch.randn(2, 3, dtype=torch.float64)
    states_guess = torch.randn(2, 5, 3, dtype=torch.float64)
    drivers = torch.randn(2, 5, 3, dtype=torch.float64)

    with pytest.raises(ValueError, match="accel_scan_fn"):
        fixed_point_alg_batched(
            f=_f,
            initial_state=initial_state,
            states_guess=states_guess,
            drivers=drivers,
            method="picard",
            num_iters=2,
            scan_backend="accel_scan",
            accel_scan_fn=None,
        )


def test_jacobi_and_picard_config_helpers_accept_accel_scan():
    helpers = [
        make_paragru_jacobi_config,
        make_paragru_picard_config,
        make_pararnn_jacobi_config,
        make_pararnn_picard_config,
        make_paralstm_jacobi_config,
        make_paralstm_picard_config,
    ]

    for helper in helpers:
        cfg = helper(
            num_iters=7,
            tol=1e-6,
            strict_tol=True,
            scan_backend="accel_scan",
            accel_module="ref",
        )

        assert cfg.num_iters == 7
        assert cfg.tol == 1e-6
        assert cfg.strict_tol is True
        assert cfg.scan_backend == "accel_scan"
        assert cfg.accel_module == "ref"
        assert cfg.solver in ("jacobi", "picard")


@pytest.mark.parametrize("cls", [ParaGRU, ParaRNN, ParaLSTM])
@pytest.mark.parametrize("solver", ["jacobi", "picard"])
def test_constructor_wires_accel_scan_config_without_running_backend(cls, solver):
    kwargs = dict(
        input_size=2,
        hidden_size=3,
        batch_first=True,
        solver=solver,
        scan_backend="accel_scan",
        accel_module="ref",
        num_iters=5,
        dtype=torch.float64,
    )

    if cls is ParaLSTM:
        kwargs.update(
            recurrent_init_scale=0.025,
            peephole_init_scale=0.025,
        )

    model = cls(**kwargs)

    assert model.solver == solver
    assert model.mode == solver
    assert model.config.deer.solver == solver
    assert model.config.deer.scan_backend == "accel_scan"
    assert model.config.deer.accel_module == "ref"


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available.")
@pytest.mark.parametrize("method", ["jacobi", "picard"])
def test_fixed_point_real_accel_scan_matches_torch_on_cuda(method):
    try:
        from accelerated_scan.warp import scan as accel_scan_fn
    except Exception as exc:
        pytest.skip(f"accelerated_scan.warp unavailable: {exc}")

    torch.manual_seed(9304)

    device = torch.device("cuda")
    dtype = torch.float32

    initial_state = torch.randn(2, 4, device=device, dtype=dtype)
    states_guess = torch.randn(2, 37, 4, device=device, dtype=dtype)
    drivers = torch.randn(2, 37, 4, device=device, dtype=dtype)

    expected, _ = fixed_point_alg_batched(
        f=_f,
        initial_state=initial_state,
        states_guess=states_guess,
        drivers=drivers,
        method=method,
        num_iters=3,
        tol=None,
        strict_tol=True,
        stopping_criterion="update",
        scan_backend="torch",
    )

    actual, info = fixed_point_alg_batched(
        f=_f,
        initial_state=initial_state,
        states_guess=states_guess,
        drivers=drivers,
        method=method,
        num_iters=3,
        tol=None,
        strict_tol=True,
        stopping_criterion="update",
        scan_backend="accel_scan",
        accel_scan_fn=accel_scan_fn,
    )

    torch.cuda.synchronize()

    assert info["scan_backend"] == "accel_scan"
    assert torch.allclose(actual, expected, atol=2e-4, rtol=2e-4)
