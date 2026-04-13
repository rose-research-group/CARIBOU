#!/bin/bash
# Run plot.py for tsp_large_intestine only.
#
# Usage:
#   sbatch slurm/tsp_large_intestine/run_plot.sh
#
#SBATCH --job-name=plot_tsp_large_intestine
#SBATCH --partition=peerd
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=../logs/plot_tsp_large_intestine_%j.out
#SBATCH --error=../logs/plot_tsp_large_intestine_%j.err

set -euo pipefail

DATASET_ID="tsp_large_intestine"
CARIBOU_ROOT="/data1/peerd/riffled/riffled/Olaf_project/CARIBOU"
ANALYSIS_DIR="$CARIBOU_ROOT/benchmarking/celltyping_benchmarks/analysis"
PLOTS_DIR="$CARIBOU_ROOT/benchmarking/celltyping_benchmarks/analysis/plots"

mkdir -p "$PLOTS_DIR" "$CARIBOU_ROOT/benchmarking/celltyping_benchmarks/slurm/logs"

echo "========================================================="
echo "  CARIBOU Plotting"
echo "  Dataset : $DATASET_ID"
echo "  Plots   : $PLOTS_DIR"
echo "  Started : $(date)"
echo "========================================================="

cd "$CARIBOU_ROOT"

python3 "$ANALYSIS_DIR/plot.py" \
    --output-dir "$PLOTS_DIR" \
    --dataset "$DATASET_ID"

echo ""
echo "========================================================="
echo "  Plotting complete: $(date)"
echo "========================================================="
