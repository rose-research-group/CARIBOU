#!/bin/bash
#SBATCH --job-name=meta_eval_plot
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=0:30:00
#SBATCH --output=/dev/null
#SBATCH --partition=peerd

ROOT_DIR="$(git -C "${SLURM_SUBMIT_DIR:-$(pwd)}" rev-parse --show-toplevel 2>/dev/null || true)"
if [ -z "$ROOT_DIR" ]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
fi

LOG_DIR="$ROOT_DIR/benchmarking/metadata_benchmarks/results/logs/metadata"
mkdir -p "$LOG_DIR"
LOG_PATH="$LOG_DIR/evaluate_plot_metadata_${SLURM_JOB_ID:-0}.log"
exec > "$LOG_PATH" 2>&1

RESULTS_DIR="${RESULTS_DIR:-$ROOT_DIR/benchmarking/metadata_benchmarks/results}"
SCORES_CSV="${SCORES_CSV:-$ROOT_DIR/benchmarking/metadata_benchmarks/metadata_benchmark_scores.csv}"
PLOTS_DIR="${PLOTS_DIR:-$ROOT_DIR/benchmarking/metadata_benchmarks/results/plots}"

python "$ROOT_DIR/benchmarking/metadata_benchmarks/evaluate_metadata_results.py" \
  --results-dir "$RESULTS_DIR" \
  --output "$SCORES_CSV"

python "$ROOT_DIR/benchmarking/metadata_benchmarks/plot_metadata_benchmark_scores.py" \
  --scores-csv "$SCORES_CSV" \
  --output-dir "$PLOTS_DIR"

echo "Evaluation complete. Scores: $SCORES_CSV"
echo "Plots saved to: $PLOTS_DIR"
