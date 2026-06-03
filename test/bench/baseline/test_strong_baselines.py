from pathlib import Path

import torch

from test.bench.baseline.strong_baselines import (
    build_strong_baseline_cases,
    run_strong_baseline_benchmark,
)


def test_strong_baseline_case_catalog_contains_expected_methods():
    names = {case.name for case in build_strong_baseline_cases(num_iters=2)}

    expected = {
        "paragru_sequential",
        "paragru_deer_autograd",
        "paragru_deer_adjoint",
        "paragru_quasi_elk",
        "pararnn_sequential",
        "pararnn_full_deer",
        "pararnn_quasi_deer",
        "pararnn_full_elk",
        "pararnn_quasi_elk",
        "paralstm_sequential",
        "paralstm_block_deer_autograd",
        "paralstm_block_deer_adjoint",
        "paralstm_quasi_deer",
        "paralstm_quasi_elk",
    }

    missing = expected - names
    assert not missing, f"Missing benchmark cases: {sorted(missing)}"


def test_strong_baseline_smoke_run_writes_csv(tmp_path):
    csv_path = tmp_path / "strong_baselines_smoke.csv"

    results = run_strong_baseline_benchmark(
        case_names=[
            "paragru_sequential",
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
        output_csv=csv_path,
    )

    assert len(results) == 4
    assert csv_path.exists()
    assert csv_path.read_text().startswith("name,family,solver_label")

    for result in results:
        assert result.status in ("ok", "large_error")
        assert result.forward_median_s >= 0.0
        assert result.forward_min_s >= 0.0
        assert result.fw_bw_median_s is not None
        assert result.fw_bw_min_s is not None
        assert result.grad_finite is True
        assert result.param_grad_finite is True
        assert result.max_error_output == result.max_error_output
        assert result.max_error_hidden == result.max_error_hidden


def test_strong_baseline_cli_module_smoke(tmp_path):
    output_csv = tmp_path / "cli_smoke.csv"

    import subprocess
    import sys

    cmd = [
        sys.executable,
        "-m",
        "test.bench.baseline.strong_baselines",
        "--case",
        "paragru_sequential",
        "--case",
        "pararnn_quasi_deer",
        "--batch-size",
        "2",
        "--seq-len",
        "3",
        "--input-size",
        "2",
        "--hidden-size",
        "3",
        "--repeats",
        "1",
        "--warmups",
        "0",
        "--num-iters",
        "4",
        "--device",
        "cpu",
        "--dtype",
        "float64",
        "--output-csv",
        str(output_csv),
    ]

    completed = subprocess.run(cmd, check=True, capture_output=True, text=True)

    assert "paragru_sequential" in completed.stdout
    assert "pararnn_quasi_deer" in completed.stdout
    assert output_csv.exists()
