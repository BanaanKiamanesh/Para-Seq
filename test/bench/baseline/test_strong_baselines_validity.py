import torch

from test.bench.baseline.strong_baselines import run_strong_baseline_benchmark


def test_all_strong_baseline_cases_are_valid_in_tiny_contracting_forward_run():
    results = run_strong_baseline_benchmark(
        batch_size=1,
        seq_len=3,
        input_size=2,
        hidden_size=2,
        repeats=1,
        warmups=0,
        num_iters=3,
        include_backward=False,
        device="cpu",
        dtype=torch.float64,
        valid_tol=1e-1,
    )

    bad = [
        (
            r.name,
            r.status,
            r.max_error_output,
            r.max_error_hidden,
            r.max_error_cell,
        )
        for r in results
        if not r.valid_solution or r.status != "ok"
    ]

    assert not bad, f"Invalid benchmark cases: {bad}"


def test_selected_strong_baseline_backward_runs_are_valid_not_only_finite():
    results = run_strong_baseline_benchmark(
        case_names=[
            "paragru_deer_adjoint",
            "pararnn_quasi_deer",
            "paralstm_block_deer_adjoint",
        ],
        batch_size=2,
        seq_len=4,
        input_size=2,
        hidden_size=3,
        repeats=1,
        warmups=0,
        num_iters=6,
        include_backward=True,
        device="cpu",
        dtype=torch.float64,
        valid_tol=2e-2,
    )

    for result in results:
        assert result.status == "ok", result
        assert result.valid_solution is True, result
        assert result.grad_finite is True, result
        assert result.param_grad_finite is True, result
        assert result.fw_bw_median_s is not None, result
        assert result.fw_bw_min_s is not None, result
