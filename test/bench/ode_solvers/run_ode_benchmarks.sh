#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILE="${ODE_BENCH_PROFILE:-heavy}"

echo "Running ODE benchmarks from: ${SCRIPT_DIR}"
echo "Profile: ${PROFILE}"

python "${SCRIPT_DIR}/heavy_benchmark.py" --profile "${PROFILE}" "$@"
