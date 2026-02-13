#!/bin/bash
#SBATCH --job-name=task_eval_plot
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=0:30:00
#SBATCH --output=/dev/null
#SBATCH --partition=peerd

ROOT_DIR="$(git -C "${SLURM_SUBMIT_DIR:-$(pwd)}" rev-parse --show-toplevel 2>/dev/null || true)"
if [ -z "$ROOT_DIR" ]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
fi

LOG_DIR="$ROOT_DIR/benchmarking/task_benchmarks/analysis_slurm/logs"
mkdir -p "$LOG_DIR"
LOG_PATH="$LOG_DIR/evaluate_plot_task_benchmarks_${SLURM_JOB_ID:-0}.log"
exec > "$LOG_PATH" 2>&1

RESULTS_DIR="${RESULTS_DIR:-$ROOT_DIR/benchmarking/task_benchmarks/results}"
ANALYSIS_DIR="${ANALYSIS_DIR:-$ROOT_DIR/benchmarking/task_benchmarks/analysis}"
PLOTS_DIR="${PLOTS_DIR:-$ANALYSIS_DIR/plots}"
SUMMARY_JSON="${SUMMARY_JSON:-$ANALYSIS_DIR/task_benchmark_summary.json}"
SUMMARY_CSV="${SUMMARY_CSV:-$RESULTS_DIR/summary.csv}"
SUMMARY_RAW_JSON="${SUMMARY_RAW_JSON:-$RESULTS_DIR/summary.json}"

rm -f "$SUMMARY_CSV" "$SUMMARY_RAW_JSON" "$SUMMARY_JSON"
rm -rf "$PLOTS_DIR"
mkdir -p "$ANALYSIS_DIR" "$PLOTS_DIR"

python "$ROOT_DIR/benchmarking/task_benchmarks/src/results_collector.py" \
  --results-dir "$RESULTS_DIR" \
  --output-json "$SUMMARY_RAW_JSON" \
  --output-csv "$SUMMARY_CSV"

python "$ROOT_DIR/benchmarking/task_benchmarks/analysis/compare_qc_results.py" \
  --results-dir "$RESULTS_DIR" \
  --output-dir "$ANALYSIS_DIR"

python "$ROOT_DIR/benchmarking/task_benchmarks/analysis/plot_task_benchmark_summary.py" \
  --summary-json "$SUMMARY_JSON" \
  --output-dir "$PLOTS_DIR"

echo "Results summary: $SUMMARY_RAW_JSON"
echo "Results CSV: $SUMMARY_CSV"
echo "Analysis summary: $SUMMARY_JSON"
echo "Plots saved to: $PLOTS_DIR"
