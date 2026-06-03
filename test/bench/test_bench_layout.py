from pathlib import Path

from test.bench.baseline.strong_baselines import build_strong_baseline_cases


def test_benchmark_layout_has_algos_and_baseline_folders():
    assert Path("test/bench/algos").is_dir()
    assert Path("test/bench/baseline").is_dir()
    assert Path("test/bench/baseline/strong_baselines.py").is_file()

    assert not Path("test/bench/benchmarking.py").exists()
    assert not Path("test/bench/log_summary.ipynb").exists()
    assert not Path("test/bench/logs").exists()


def test_baseline_benchmark_imports_from_new_location():
    names = {case.name for case in build_strong_baseline_cases(num_iters=2)}
    assert "paragru_deer_adjoint" in names
    assert "pararnn_quasi_elk" in names
    assert "paralstm_block_deer_adjoint" in names
