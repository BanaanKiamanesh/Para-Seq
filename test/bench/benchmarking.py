from src.algos.Picard import picard_alg
from src.algos.Jacobi import jacobi_alg
from src.algos.ELK import elk_alg
from src.algos.DEER import deer_alg, sequential_rollout
import argparse
import csv
import gc
import json
import logging
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = Path(__file__).resolve().parent
LOG_DIR = BENCH_DIR / "logs"

sys.path.insert(0, str(PROJECT_ROOT))


class SimpleRNNCell(torch.nn.Module):
    def __init__(self, state_dim, input_dim):
        super().__init__()

        self.W_h = torch.nn.Linear(state_dim, state_dim, bias=False)
        self.W_u = torch.nn.Linear(input_dim, state_dim, bias=True)

        torch.nn.init.normal_(self.W_h.weight, mean=0.0, std=0.25)
        torch.nn.init.normal_(self.W_u.weight, mean=0.0, std=0.25)
        torch.nn.init.zeros_(self.W_u.bias)

    def forward(self, state, driver):
        return torch.tanh(self.W_h(state) + self.W_u(driver))


def freeze_model_for_benchmark(model):
    model.eval()

    for param in model.parameters():
        param.requires_grad_(False)


def parse_dtype(dtype_name):
    dtype_map = {
        "float32": torch.float32,
        "float64": torch.float64,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }

    if dtype_name not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype_name}")

    return dtype_map[dtype_name]


def setup_logging(log_file):
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("benchmarking")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def sync_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def clear_cuda_memory(device):
    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def reset_peak_memory(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def get_peak_memory_mb(device):
    if device.type != "cuda":
        return 0.0, 0.0

    allocated = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
    reserved = torch.cuda.max_memory_reserved(device) / (1024.0 * 1024.0)

    return allocated, reserved


def is_cuda_oom(error):
    if isinstance(error, torch.OutOfMemoryError):
        return True

    message = str(error).lower()
    return "cuda out of memory" in message or "outofmemoryerror" in message


@contextmanager
def maybe_no_grad(enabled):
    if enabled:
        with torch.no_grad():
            yield
    else:
        yield


def time_function(fn, device, use_no_grad=False):
    sync_if_cuda(device)

    start_time = time.perf_counter()

    with maybe_no_grad(use_no_grad):
        result = fn()

    sync_if_cuda(device)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    return result, elapsed_time


def make_states_guess(seq_len, state_dim, device, dtype):
    return torch.zeros(seq_len, state_dim, device=device, dtype=dtype)


def make_sequence_sizes(min_seq_len, max_seq_len):
    sequence_sizes = []

    seq_len = min_seq_len
    while seq_len <= max_seq_len:
        sequence_sizes.append(seq_len)
        seq_len *= 2

    return sequence_sizes


def load_accelerated_scan(accel_module):
    if accel_module == "warp":
        from accelerated_scan.warp import scan
        return scan

    if accel_module == "scalar":
        from accelerated_scan.scalar import scan
        return scan

    if accel_module == "ref":
        from accelerated_scan.ref import scan
        return scan

    raise ValueError(f"Unknown accelerated_scan module: {accel_module}")


def summarize_times(times):
    if len(times) == 0:
        return {
            "time_min_s": None,
            "time_mean_s": None,
            "time_std_s": None,
        }

    if len(times) == 1:
        return {
            "time_min_s": times[0],
            "time_mean_s": times[0],
            "time_std_s": 0.0,
        }

    return {
        "time_min_s": min(times),
        "time_mean_s": mean(times),
        "time_std_s": stdev(times),
    }


def to_python_scalar(value):
    if value is None:
        return None

    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.detach().cpu().item()

        return str(value.detach().cpu())

    if isinstance(value, (int, float, str, bool)):
        return value

    return str(value)


def validate_result_record(
    record,
    valid_final_merit_threshold,
    valid_error_threshold,
):
    method = record.get("method", "")
    status = record.get("status", "")

    if method == "Sequential Evaluation" and status == "ok":
        record["valid_solution"] = True
        record["validation_reason"] = "sequential_baseline"
        return record

    if status != "ok":
        record["valid_solution"] = False
        record["validation_reason"] = f"status_not_ok:{status}"
        return record

    final_merit = record.get("final_merit", None)
    max_error = record.get("max_error_vs_sequential", None)

    if final_merit is None:
        record["valid_solution"] = False
        record["validation_reason"] = "missing_final_merit"
        return record

    if max_error is None:
        record["valid_solution"] = False
        record["validation_reason"] = "missing_max_error"
        return record

    if final_merit >= valid_final_merit_threshold:
        record["valid_solution"] = False
        record["validation_reason"] = (
            f"final_merit_too_large:"
            f"{final_merit}>={valid_final_merit_threshold}"
        )
        return record

    if max_error >= valid_error_threshold:
        record["valid_solution"] = False
        record["validation_reason"] = (
            f"max_error_too_large:"
            f"{max_error}>={valid_error_threshold}"
        )
        return record

    record["valid_solution"] = True
    record["validation_reason"] = "passed"
    return record


def empty_error_record(method, status, error_message):
    return {
        "method": method,
        "status": status,
        "error_message": error_message,
        "valid_solution": False,
        "validation_reason": f"runtime_failure:{status}",
        "successful_repeats": 0,
        "num_iters": None,
        "initial_merit": None,
        "final_merit": None,
        "last_update_error": None,
        "tol": None,
        "effective_tol": None,
        "strict_tol": None,
        "stopping_criterion": None,
        "max_error_vs_sequential": None,
        "time_min_s": None,
        "time_mean_s": None,
        "time_std_s": None,
        "peak_memory_allocated_mb": None,
        "peak_memory_reserved_mb": None,
    }


def extract_info_fields(info):
    fields = {
        "num_iters": None,
        "initial_merit": None,
        "final_merit": None,
        "last_update_error": None,
        "tol": None,
        "effective_tol": None,
        "strict_tol": None,
        "stopping_criterion": None,
    }

    if not isinstance(info, dict):
        return fields

    for key in fields:
        if key in info:
            fields[key] = to_python_scalar(info[key])

    return fields


def benchmark_method(
    name,
    fn_builder,
    device,
    true_states_cpu,
    warmup,
    repeats,
    logger,
    valid_final_merit_threshold,
    valid_error_threshold,
    use_no_grad=False,
    stop_on_error=False,
):
    logger.info(f"Starting method: {name}")

    try:
        for warmup_idx in range(warmup):
            reset_peak_memory(device)

            result, _ = time_function(
                fn_builder,
                device=device,
                use_no_grad=use_no_grad,
            )

            del result
            clear_cuda_memory(device)

            logger.info(f"Warmup {warmup_idx + 1}/{warmup} done for {name}")

    except RuntimeError as error:
        clear_cuda_memory(device)

        status = "oom_warmup" if is_cuda_oom(error) else "runtime_error_warmup"

        logger.error(f"{status} during warmup for {name}: {error}")

        if stop_on_error:
            raise

        return validate_result_record(
            empty_error_record(name, status, str(error)),
            valid_final_merit_threshold=valid_final_merit_threshold,
            valid_error_threshold=valid_error_threshold,
        )

    times = []
    errors = []
    info_values = []
    peak_allocated_values = []
    peak_reserved_values = []

    for repeat_idx in range(repeats):
        try:
            reset_peak_memory(device)

            result, elapsed_time = time_function(
                fn_builder,
                device=device,
                use_no_grad=use_no_grad,
            )

            states, info = result

            peak_allocated_mb, peak_reserved_mb = get_peak_memory_mb(device)

            states_cpu = states.detach().cpu()
            max_error = torch.max(
                torch.abs(states_cpu - true_states_cpu)).item()

            info_fields = extract_info_fields(info)

            times.append(elapsed_time)
            errors.append(max_error)
            info_values.append(info_fields)
            peak_allocated_values.append(peak_allocated_mb)
            peak_reserved_values.append(peak_reserved_mb)

            del states_cpu
            del states
            del info
            del result

            clear_cuda_memory(device)

            logger.info(
                f"Repeat {repeat_idx + 1}/{repeats} done for {name}: "
                f"{elapsed_time:.6f} s | peak allocated {peak_allocated_mb:.2f} MB"
            )

        except RuntimeError as error:
            clear_cuda_memory(device)

            status = "oom_repeat" if is_cuda_oom(
                error) else "runtime_error_repeat"

            logger.error(f"{status} during repeat for {name}: {error}")

            if stop_on_error:
                raise

            if len(times) == 0:
                return validate_result_record(
                    empty_error_record(name, status, str(error)),
                    valid_final_merit_threshold=valid_final_merit_threshold,
                    valid_error_threshold=valid_error_threshold,
                )

            time_summary = summarize_times(times)
            last_info = info_values[-1]

            partial_record = {
                "method": name,
                "status": "partial_" + status,
                "error_message": str(error),
                "successful_repeats": len(times),
                "max_error_vs_sequential": errors[-1],
                **last_info,
                **time_summary,
                "peak_memory_allocated_mb": max(peak_allocated_values),
                "peak_memory_reserved_mb": max(peak_reserved_values),
            }

            return validate_result_record(
                partial_record,
                valid_final_merit_threshold=valid_final_merit_threshold,
                valid_error_threshold=valid_error_threshold,
            )

    time_summary = summarize_times(times)
    last_info = info_values[-1]

    result_record = {
        "method": name,
        "status": "ok",
        "error_message": "",
        "successful_repeats": len(times),
        "max_error_vs_sequential": errors[-1],
        **last_info,
        **time_summary,
        "peak_memory_allocated_mb": max(peak_allocated_values),
        "peak_memory_reserved_mb": max(peak_reserved_values),
    }

    result_record = validate_result_record(
        result_record,
        valid_final_merit_threshold=valid_final_merit_threshold,
        valid_error_threshold=valid_error_threshold,
    )

    logger.info(f"Finished method: {name}")
    logger.info(json.dumps(result_record, indent=2))

    return result_record


def benchmark_sequential(
    f,
    initial_state,
    drivers,
    device,
    warmup,
    repeats,
    logger,
    valid_final_merit_threshold,
    valid_error_threshold,
    stop_on_error=False,
):
    logger.info("Starting sequential baseline")

    try:
        for warmup_idx in range(warmup):
            reset_peak_memory(device)

            states, _ = time_function(
                lambda: sequential_rollout(f, initial_state, drivers),
                device=device,
                use_no_grad=True,
            )

            del states
            clear_cuda_memory(device)

            logger.info(f"Sequential warmup {warmup_idx + 1}/{warmup} done")

    except RuntimeError as error:
        clear_cuda_memory(device)

        status = "oom_warmup" if is_cuda_oom(error) else "runtime_error_warmup"

        logger.error(f"{status} during sequential warmup: {error}")

        if stop_on_error:
            raise

        return None, validate_result_record(
            empty_error_record("Sequential Evaluation", status, str(error)),
            valid_final_merit_threshold=valid_final_merit_threshold,
            valid_error_threshold=valid_error_threshold,
        )

    times = []
    peak_allocated_values = []
    peak_reserved_values = []
    true_states_cpu = None

    for repeat_idx in range(repeats):
        try:
            reset_peak_memory(device)

            states, elapsed_time = time_function(
                lambda: sequential_rollout(f, initial_state, drivers),
                device=device,
                use_no_grad=True,
            )

            peak_allocated_mb, peak_reserved_mb = get_peak_memory_mb(device)

            if true_states_cpu is not None:
                del true_states_cpu

            true_states_cpu = states.detach().cpu()

            times.append(elapsed_time)
            peak_allocated_values.append(peak_allocated_mb)
            peak_reserved_values.append(peak_reserved_mb)

            del states
            clear_cuda_memory(device)

            logger.info(
                f"Sequential repeat {repeat_idx + 1}/{repeats} done: "
                f"{elapsed_time:.6f} s | peak allocated {peak_allocated_mb:.2f} MB"
            )

        except RuntimeError as error:
            clear_cuda_memory(device)

            status = "oom_repeat" if is_cuda_oom(
                error) else "runtime_error_repeat"

            logger.error(f"{status} during sequential repeat: {error}")

            if stop_on_error:
                raise

            return None, validate_result_record(
                empty_error_record("Sequential Evaluation",
                                   status, str(error)),
                valid_final_merit_threshold=valid_final_merit_threshold,
                valid_error_threshold=valid_error_threshold,
            )

    time_summary = summarize_times(times)

    result_record = {
        "method": "Sequential Evaluation",
        "status": "ok",
        "error_message": "",
        "successful_repeats": len(times),
        "num_iters": None,
        "initial_merit": 0.0,
        "final_merit": 0.0,
        "last_update_error": 0.0,
        "tol": None,
        "effective_tol": None,
        "strict_tol": None,
        "stopping_criterion": None,
        "max_error_vs_sequential": 0.0,
        **time_summary,
        "peak_memory_allocated_mb": max(peak_allocated_values),
        "peak_memory_reserved_mb": max(peak_reserved_values),
    }

    result_record = validate_result_record(
        result_record,
        valid_final_merit_threshold=valid_final_merit_threshold,
        valid_error_threshold=valid_error_threshold,
    )

    logger.info("Finished sequential baseline")
    logger.info(json.dumps(result_record, indent=2))

    return true_states_cpu, result_record


def write_row(csv_writer, csv_file_handle, base_row, record):
    row = {
        **base_row,
        **record,
    }

    csv_writer.writerow(row)
    csv_file_handle.flush()


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser()

    parser.add_argument("--run-name", type=str, default="manual")
    parser.add_argument("--min-seq-len", type=int, default=1024)
    parser.add_argument("--max-seq-len", type=int, default=128 * 1024)
    parser.add_argument("--state-dim", type=int, default=4)
    parser.add_argument("--input-dim", type=int, default=3)
    parser.add_argument("--dtype", type=str, default="float64")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--algorithms",
        type=str,
        default="all",
        help=(
            "Comma-separated list. Options: sequential,deer,quasi_deer,picard,"
            "jacobi,elk,quasi_elk,all"
        ),
    )

    parser.add_argument(
        "--scan-backend",
        type=str,
        default="torch",
        choices=["torch", "accel_scan"],
    )

    parser.add_argument(
        "--accel-module",
        type=str,
        default="warp",
        choices=["warp", "scalar", "ref"],
    )

    parser.add_argument("--elk-sigmasq", type=float, default=1e8)
    parser.add_argument("--quasi-elk-sigmasq", type=float, default=1e8)
    parser.add_argument("--elk-process-noise", type=float, default=1.0)

    parser.add_argument("--max-iters-deer", type=int, default=None)
    parser.add_argument("--max-iters-quasi-deer", type=int, default=None)
    parser.add_argument("--max-iters-picard", type=int, default=256)
    parser.add_argument("--max-iters-jacobi", type=int, default=256)
    parser.add_argument("--max-iters-elk", type=int, default=64)
    parser.add_argument("--max-iters-quasi-elk", type=int, default=64)

    parser.add_argument("--tol", type=float, default=1e-12)
    parser.add_argument("--clip-value", type=float, default=1e8)

    parser.add_argument(
        "--stopping-criterion",
        type=str,
        default="update",
        choices=["update", "merit"],
    )

    parser.add_argument(
        "--strict-tol",
        action="store_true",
        help="Use --tol exactly instead of clamping to a dtype-safe tolerance.",
    )

    parser.add_argument(
        "--valid-final-merit-threshold",
        type=float,
        default=1e-6,
    )

    parser.add_argument(
        "--valid-error-threshold",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Raise runtime errors instead of logging invalid method rows.",
    )

    parser.add_argument(
        "--log-file",
        type=str,
        default=str(LOG_DIR / f"benchmarking_{timestamp}.log"),
    )

    parser.add_argument(
        "--csv-file",
        type=str,
        default=str(LOG_DIR / f"benchmarking_{timestamp}.csv"),
    )

    args = parser.parse_args()

    logger = setup_logging(args.log_file)

    logger.info("Benchmark arguments:")
    logger.info(json.dumps(vars(args), indent=2))

    dtype = parse_dtype(args.dtype)
    device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is False.")

    torch.manual_seed(args.seed)

    accel_scan_fn = None

    if args.scan_backend == "accel_scan":
        accel_scan_fn = load_accelerated_scan(args.accel_module)
        logger.info(f"Loaded accelerated_scan.{args.accel_module}.scan")

    if args.algorithms == "all":
        algorithms = [
            "sequential",
            "deer",
            "quasi_deer",
            "picard",
            "jacobi",
            "elk",
            "quasi_elk",
        ]
    else:
        algorithms = [algorithm.strip()
                      for algorithm in args.algorithms.split(",")]

    if "sequential" not in algorithms:
        raise ValueError(
            "The benchmark needs the sequential baseline for correctness errors. "
            "Include 'sequential' in --algorithms."
        )

    sequence_sizes = make_sequence_sizes(args.min_seq_len, args.max_seq_len)

    logger.info(f"Project root: {PROJECT_ROOT}")
    logger.info(f"Benchmark directory: {BENCH_DIR}")
    logger.info(f"Logs directory: {LOG_DIR}")
    logger.info(f"Sequence sizes: {sequence_sizes}")
    logger.info(f"Using torch version: {torch.__version__}")

    if device.type == "cuda":
        logger.info(f"CUDA device: {torch.cuda.get_device_name(device)}")

    csv_file = Path(args.csv_file)
    csv_file.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "run_name",
        "seq_len",
        "state_dim",
        "input_dim",
        "dtype",
        "device",
        "scan_backend_requested",
        "scan_backend_applied",
        "method",
        "status",
        "error_message",
        "valid_solution",
        "validation_reason",
        "successful_repeats",
        "num_iters",
        "initial_merit",
        "final_merit",
        "last_update_error",
        "tol",
        "effective_tol",
        "strict_tol",
        "stopping_criterion",
        "max_error_vs_sequential",
        "time_min_s",
        "time_mean_s",
        "time_std_s",
        "peak_memory_allocated_mb",
        "peak_memory_reserved_mb",
        "valid_final_merit_threshold",
        "valid_error_threshold",
        "warmup",
        "repeats",
        "seed",
    ]

    with open(csv_file, mode="w", newline="") as csv_file_handle:
        csv_writer = csv.DictWriter(csv_file_handle, fieldnames=fieldnames)
        csv_writer.writeheader()
        csv_file_handle.flush()

        for seq_len in sequence_sizes:
            logger.info("=" * 80)
            logger.info(f"Starting sequence length: {seq_len}")
            logger.info("=" * 80)

            cell = SimpleRNNCell(args.state_dim, args.input_dim).to(
                device=device,
                dtype=dtype,
            )
            freeze_model_for_benchmark(cell)

            initial_state = torch.zeros(
                args.state_dim,
                device=device,
                dtype=dtype,
            )

            drivers = torch.randn(
                seq_len,
                args.input_dim,
                device=device,
                dtype=dtype,
            )

            def f(state, driver):
                return cell(state, driver)

            base_row = {
                "run_name": args.run_name,
                "seq_len": seq_len,
                "state_dim": args.state_dim,
                "input_dim": args.input_dim,
                "dtype": args.dtype,
                "device": str(device),
                "scan_backend_requested": args.scan_backend,
                "valid_final_merit_threshold": args.valid_final_merit_threshold,
                "valid_error_threshold": args.valid_error_threshold,
                "warmup": args.warmup,
                "repeats": args.repeats,
                "seed": args.seed,
            }

            true_states_cpu, sequential_record = benchmark_sequential(
                f=f,
                initial_state=initial_state,
                drivers=drivers,
                device=device,
                warmup=args.warmup,
                repeats=args.repeats,
                logger=logger,
                valid_final_merit_threshold=args.valid_final_merit_threshold,
                valid_error_threshold=args.valid_error_threshold,
                stop_on_error=args.stop_on_error,
            )

            write_row(
                csv_writer,
                csv_file_handle,
                {
                    **base_row,
                    "scan_backend_applied": "none",
                },
                sequential_record,
            )

            if true_states_cpu is None:
                logger.error(
                    f"Skipping sequence length {seq_len} because sequential baseline failed."
                )

                del drivers
                del initial_state
                del cell
                clear_cuda_memory(device)

                continue

            if "deer" in algorithms:
                max_iters = args.max_iters_deer if args.max_iters_deer is not None else seq_len

                record = benchmark_method(
                    name="Full DEER / Newton",
                    fn_builder=lambda: deer_alg(
                        f=f,
                        initial_state=initial_state,
                        states_guess=make_states_guess(
                            seq_len, args.state_dim, device, dtype),
                        drivers=drivers,
                        num_iters=max_iters,
                        tol=args.tol,
                        quasi=False,
                        damping=0.0,
                        clip_value=args.clip_value,
                        return_trace=False,
                        scan_backend="torch",
                        accel_scan_fn=None,
                        strict_tol=args.strict_tol,
                        stopping_criterion=args.stopping_criterion,
                    ),
                    device=device,
                    true_states_cpu=true_states_cpu,
                    warmup=args.warmup,
                    repeats=args.repeats,
                    logger=logger,
                    valid_final_merit_threshold=args.valid_final_merit_threshold,
                    valid_error_threshold=args.valid_error_threshold,
                    use_no_grad=False,
                    stop_on_error=args.stop_on_error,
                )

                write_row(
                    csv_writer,
                    csv_file_handle,
                    {
                        **base_row,
                        "scan_backend_applied": "torch_dense_associative_scan",
                    },
                    record,
                )

            if "quasi_deer" in algorithms:
                max_iters = (
                    args.max_iters_quasi_deer
                    if args.max_iters_quasi_deer is not None
                    else seq_len
                )

                record = benchmark_method(
                    name="Quasi-DEER / Quasi-Newton",
                    fn_builder=lambda: deer_alg(
                        f=f,
                        initial_state=initial_state,
                        states_guess=make_states_guess(
                            seq_len, args.state_dim, device, dtype),
                        drivers=drivers,
                        num_iters=max_iters,
                        tol=args.tol,
                        quasi=True,
                        damping=0.0,
                        clip_value=args.clip_value,
                        return_trace=False,
                        scan_backend=args.scan_backend,
                        accel_scan_fn=accel_scan_fn,
                        strict_tol=args.strict_tol,
                        stopping_criterion=args.stopping_criterion,
                    ),
                    device=device,
                    true_states_cpu=true_states_cpu,
                    warmup=args.warmup,
                    repeats=args.repeats,
                    logger=logger,
                    valid_final_merit_threshold=args.valid_final_merit_threshold,
                    valid_error_threshold=args.valid_error_threshold,
                    use_no_grad=False,
                    stop_on_error=args.stop_on_error,
                )

                applied_backend = (
                    f"accelerated_scan.{args.accel_module}"
                    if args.scan_backend == "accel_scan"
                    else "torch_associative_scan"
                )

                write_row(
                    csv_writer,
                    csv_file_handle,
                    {
                        **base_row,
                        "scan_backend_applied": applied_backend,
                    },
                    record,
                )

            if "picard" in algorithms:
                record = benchmark_method(
                    name="Picard",
                    fn_builder=lambda: picard_alg(
                        f=f,
                        initial_state=initial_state,
                        states_guess=make_states_guess(
                            seq_len, args.state_dim, device, dtype),
                        drivers=drivers,
                        num_iters=args.max_iters_picard,
                        tol=args.tol,
                        clip_value=args.clip_value,
                        return_trace=False,
                    ),
                    device=device,
                    true_states_cpu=true_states_cpu,
                    warmup=args.warmup,
                    repeats=args.repeats,
                    logger=logger,
                    valid_final_merit_threshold=args.valid_final_merit_threshold,
                    valid_error_threshold=args.valid_error_threshold,
                    use_no_grad=True,
                    stop_on_error=args.stop_on_error,
                )

                write_row(
                    csv_writer,
                    csv_file_handle,
                    {
                        **base_row,
                        "scan_backend_applied": "torch_cumsum",
                    },
                    record,
                )

            if "jacobi" in algorithms:
                record = benchmark_method(
                    name="Jacobi",
                    fn_builder=lambda: jacobi_alg(
                        f=f,
                        initial_state=initial_state,
                        states_guess=make_states_guess(
                            seq_len, args.state_dim, device, dtype),
                        drivers=drivers,
                        num_iters=args.max_iters_jacobi,
                        tol=args.tol,
                        clip_value=args.clip_value,
                        return_trace=False,
                    ),
                    device=device,
                    true_states_cpu=true_states_cpu,
                    warmup=args.warmup,
                    repeats=args.repeats,
                    logger=logger,
                    valid_final_merit_threshold=args.valid_final_merit_threshold,
                    valid_error_threshold=args.valid_error_threshold,
                    use_no_grad=True,
                    stop_on_error=args.stop_on_error,
                )

                write_row(
                    csv_writer,
                    csv_file_handle,
                    {
                        **base_row,
                        "scan_backend_applied": "none_parallel_map",
                    },
                    record,
                )

            if "elk" in algorithms:
                record = benchmark_method(
                    name="ELK",
                    fn_builder=lambda: elk_alg(
                        f=f,
                        initial_state=initial_state,
                        states_guess=make_states_guess(
                            seq_len, args.state_dim, device, dtype),
                        drivers=drivers,
                        sigmasq=args.elk_sigmasq,
                        process_noise=args.elk_process_noise,
                        num_iters=args.max_iters_elk,
                        tol=args.tol,
                        quasi=False,
                        damping=0.0,
                        clip_value=args.clip_value,
                        return_trace=False,
                        scan_backend="torch",
                        accel_scan_fn=None,
                        strict_tol=args.strict_tol,
                        stopping_criterion=args.stopping_criterion,
                    ),
                    device=device,
                    true_states_cpu=true_states_cpu,
                    warmup=args.warmup,
                    repeats=args.repeats,
                    logger=logger,
                    valid_final_merit_threshold=args.valid_final_merit_threshold,
                    valid_error_threshold=args.valid_error_threshold,
                    use_no_grad=False,
                    stop_on_error=args.stop_on_error,
                )

                write_row(
                    csv_writer,
                    csv_file_handle,
                    {
                        **base_row,
                        "scan_backend_applied": "torch_dense_kalman_associative_scan",
                    },
                    record,
                )

            if "quasi_elk" in algorithms:
                record = benchmark_method(
                    name="Quasi-ELK",
                    fn_builder=lambda: elk_alg(
                        f=f,
                        initial_state=initial_state,
                        states_guess=make_states_guess(
                            seq_len, args.state_dim, device, dtype),
                        drivers=drivers,
                        sigmasq=args.quasi_elk_sigmasq,
                        process_noise=args.elk_process_noise,
                        num_iters=args.max_iters_quasi_elk,
                        tol=args.tol,
                        quasi=True,
                        damping=0.0,
                        clip_value=args.clip_value,
                        return_trace=False,
                        scan_backend=args.scan_backend,
                        accel_scan_fn=accel_scan_fn,
                        strict_tol=args.strict_tol,
                        stopping_criterion=args.stopping_criterion,
                    ),
                    device=device,
                    true_states_cpu=true_states_cpu,
                    warmup=args.warmup,
                    repeats=args.repeats,
                    logger=logger,
                    valid_final_merit_threshold=args.valid_final_merit_threshold,
                    valid_error_threshold=args.valid_error_threshold,
                    use_no_grad=False,
                    stop_on_error=args.stop_on_error,
                )

                applied_backend = (
                    f"torch_covariance_scan_plus_accelerated_scan.{args.accel_module}_mean_scan"
                    if args.scan_backend == "accel_scan"
                    else "torch_scalar_kalman_associative_scan"
                )

                write_row(
                    csv_writer,
                    csv_file_handle,
                    {
                        **base_row,
                        "scan_backend_applied": applied_backend,
                    },
                    record,
                )

            del true_states_cpu
            del drivers
            del initial_state
            del cell
            clear_cuda_memory(device)

            logger.info(f"Finished sequence length: {seq_len}")

    logger.info("Benchmark complete.")
    logger.info(f"Log saved to: {args.log_file}")
    logger.info(f"CSV saved to: {args.csv_file}")


if __name__ == "__main__":
    main()
