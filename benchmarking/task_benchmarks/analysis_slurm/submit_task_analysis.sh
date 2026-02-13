#!/bin/bash
set -euo pipefail

ROOT_DIR="$(git -C "$(pwd)" rev-parse --show-toplevel 2>/dev/null || true)"
if [ -z "$ROOT_DIR" ]; then
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
fi

sbatch "$ROOT_DIR/benchmarking/task_benchmarks/analysis_slurm/evaluate_and_plot_task_benchmarks.sh"
