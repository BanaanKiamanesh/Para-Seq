#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

mkdir -p "$SCRIPT_DIR/logs"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

ACCEL_MODULE="${ACCEL_MODULE:-warp}"

QUICK_MIN_SEQ_LEN="${QUICK_MIN_SEQ_LEN:-1024}"
QUICK_MAX_SEQ_LEN="${QUICK_MAX_SEQ_LEN:-8192}"
QUICK_WARMUP="${QUICK_WARMUP:-0}"
QUICK_REPEATS="${QUICK_REPEATS:-1}"

FULL_MIN_SEQ_LEN="${FULL_MIN_SEQ_LEN:-1024}"
FULL_MAX_SEQ_LEN="${FULL_MAX_SEQ_LEN:-131072}"
FULL_WARMUP="${FULL_WARMUP:-1}"
FULL_REPEATS="${FULL_REPEATS:-3}"

STATE_DIM="${STATE_DIM:-4}"
INPUT_DIM="${INPUT_DIM:-3}"
SEED="${SEED:-0}"
ALGORITHMS="${ALGORITHMS:-all}"

TORCH_DTYPE="${TORCH_DTYPE:-float64}"
ACCEL_DTYPE="${ACCEL_DTYPE:-float32}"

TOL="${TOL:-1e-12}"
CLIP_VALUE="${CLIP_VALUE:-1e8}"

STOPPING_CRITERION="${STOPPING_CRITERION:-update}"
STRICT_TOL="${STRICT_TOL:-0}"
STOP_ON_ERROR="${STOP_ON_ERROR:-0}"

ELK_SIGMASQ="${ELK_SIGMASQ:-1e8}"
QUASI_ELK_SIGMASQ="${QUASI_ELK_SIGMASQ:-1e8}"
ELK_PROCESS_NOISE="${ELK_PROCESS_NOISE:-1.0}"

VALID_FINAL_MERIT_THRESHOLD="${VALID_FINAL_MERIT_THRESHOLD:-1e-6}"
VALID_ERROR_THRESHOLD="${VALID_ERROR_THRESHOLD:-1e-4}"

RUN_QUICK_TORCH="${RUN_QUICK_TORCH:-1}"
RUN_QUICK_ACCEL="${RUN_QUICK_ACCEL:-1}"
RUN_FULL_TORCH="${RUN_FULL_TORCH:-1}"
RUN_FULL_ACCEL="${RUN_FULL_ACCEL:-1}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

build_optional_flags () {
    local FLAGS=()

    if [[ "$STRICT_TOL" == "1" || "$STRICT_TOL" == "true" || "$STRICT_TOL" == "True" ]]; then
        FLAGS+=("--strict-tol")
    fi

    if [[ "$STOP_ON_ERROR" == "1" || "$STOP_ON_ERROR" == "true" || "$STOP_ON_ERROR" == "True" ]]; then
        FLAGS+=("--stop-on-error")
    fi

    printf '%s\n' "${FLAGS[@]}"
}

run_one_benchmark () {
    local RUN_KIND="$1"
    local SCAN_BACKEND="$2"
    local DTYPE="$3"
    local MIN_SEQ_LEN="$4"
    local MAX_SEQ_LEN="$5"
    local WARMUP="$6"
    local REPEATS="$7"

    local RUN_NAME="${RUN_KIND}_${SCAN_BACKEND}_${ACCEL_MODULE}_${DTYPE}_${TIMESTAMP}"
    local LOG_FILE="$SCRIPT_DIR/logs/${RUN_NAME}.log"
    local CSV_FILE="$SCRIPT_DIR/logs/${RUN_NAME}.csv"

    local OPTIONAL_FLAGS=()
    while IFS= read -r FLAG; do
        if [[ -n "$FLAG" ]]; then
            OPTIONAL_FLAGS+=("$FLAG")
        fi
    done < <(build_optional_flags)

    echo ""
    echo "================================================================================"
    echo "Running benchmark: $RUN_NAME"
    echo "Project root: $PROJECT_ROOT"
    echo "Benchmark script: $SCRIPT_DIR/benchmarking.py"
    echo "Run kind: $RUN_KIND"
    echo "Scan backend: $SCAN_BACKEND"
    echo "Accelerated-scan module: $ACCEL_MODULE"
    echo "Dtype: $DTYPE"
    echo "Seq length: $MIN_SEQ_LEN -> $MAX_SEQ_LEN"
    echo "Warmup: $WARMUP"
    echo "Repeats: $REPEATS"
    echo "Algorithms: $ALGORITHMS"
    echo "Tol: $TOL"
    echo "Stopping criterion: $STOPPING_CRITERION"
    echo "Strict tol: $STRICT_TOL"
    echo "Validation final merit threshold: $VALID_FINAL_MERIT_THRESHOLD"
    echo "Validation error threshold: $VALID_ERROR_THRESHOLD"
    echo "Log file: $LOG_FILE"
    echo "CSV file: $CSV_FILE"
    echo "PYTORCH_CUDA_ALLOC_CONF: $PYTORCH_CUDA_ALLOC_CONF"
    echo "================================================================================"
    echo ""

    PYTHONPATH="$PROJECT_ROOT" python -u "$SCRIPT_DIR/benchmarking.py" \
        --run-name "$RUN_NAME" \
        --min-seq-len "$MIN_SEQ_LEN" \
        --max-seq-len "$MAX_SEQ_LEN" \
        --state-dim "$STATE_DIM" \
        --input-dim "$INPUT_DIM" \
        --dtype "$DTYPE" \
        --device cuda \
        --warmup "$WARMUP" \
        --repeats "$REPEATS" \
        --seed "$SEED" \
        --algorithms "$ALGORITHMS" \
        --scan-backend "$SCAN_BACKEND" \
        --accel-module "$ACCEL_MODULE" \
        --elk-sigmasq "$ELK_SIGMASQ" \
        --quasi-elk-sigmasq "$QUASI_ELK_SIGMASQ" \
        --elk-process-noise "$ELK_PROCESS_NOISE" \
        --tol "$TOL" \
        --clip-value "$CLIP_VALUE" \
        --stopping-criterion "$STOPPING_CRITERION" \
        --valid-final-merit-threshold "$VALID_FINAL_MERIT_THRESHOLD" \
        --valid-error-threshold "$VALID_ERROR_THRESHOLD" \
        --log-file "$LOG_FILE" \
        --csv-file "$CSV_FILE" \
        "${OPTIONAL_FLAGS[@]}"

    echo ""
    echo "Finished benchmark: $RUN_NAME"
    echo "Log saved to: $LOG_FILE"
    echo "CSV saved to: $CSV_FILE"
}

echo "Starting complete benchmark suite."
echo "Logs will be saved under: $SCRIPT_DIR/logs"

if [[ "$RUN_QUICK_TORCH" == "1" ]]; then
    run_one_benchmark \
        "quick" \
        "torch" \
        "$TORCH_DTYPE" \
        "$QUICK_MIN_SEQ_LEN" \
        "$QUICK_MAX_SEQ_LEN" \
        "$QUICK_WARMUP" \
        "$QUICK_REPEATS"
fi

if [[ "$RUN_QUICK_ACCEL" == "1" ]]; then
    run_one_benchmark \
        "quick" \
        "accel_scan" \
        "$ACCEL_DTYPE" \
        "$QUICK_MIN_SEQ_LEN" \
        "$QUICK_MAX_SEQ_LEN" \
        "$QUICK_WARMUP" \
        "$QUICK_REPEATS"
fi

if [[ "$RUN_FULL_TORCH" == "1" ]]; then
    run_one_benchmark \
        "full" \
        "torch" \
        "$TORCH_DTYPE" \
        "$FULL_MIN_SEQ_LEN" \
        "$FULL_MAX_SEQ_LEN" \
        "$FULL_WARMUP" \
        "$FULL_REPEATS"
fi

if [[ "$RUN_FULL_ACCEL" == "1" ]]; then
    run_one_benchmark \
        "full" \
        "accel_scan" \
        "$ACCEL_DTYPE" \
        "$FULL_MIN_SEQ_LEN" \
        "$FULL_MAX_SEQ_LEN" \
        "$FULL_WARMUP" \
        "$FULL_REPEATS"
fi

echo ""
echo "================================================================================"
echo "All requested benchmarks finished."
echo "Logs and CSVs are in: $SCRIPT_DIR/logs"
echo "================================================================================"