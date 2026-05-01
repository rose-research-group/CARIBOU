#!/bin/bash
# Run plot.py for aba_hippocampus only.
#
# Usage:
#   sbatch slurm/aba_hippocampus/run_plot.sh
#
#SBATCH --job-name=plot_aba_hippocampus
#SBATCH --partition=peerd
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=../logs/plot_aba_hippocampus_%j.out
#SBATCH --error=../logs/plot_aba_hippocampus_%j.err

set -euo pipefail

DATASET_ID="aba_hippocampus"
CARIBOU_ROOT="/data1/peerd/riffled/riffled/Olaf_project/CARIBOU"
ANALYSIS_DIR="$CARIBOU_ROOT/benchmarking/celltyping_benchmarks/analysis"
PLOTS_DIR="$ANALYSIS_DIR/plots"
CLEAN_PLOTS_DIR="$CARIBOU_ROOT/benchmarking/celltyping_benchmarks/clean_plots"

mkdir -p "$PLOTS_DIR" "$CARIBOU_ROOT/benchmarking/celltyping_benchmarks/slurm/logs"

echo "========================================================="
echo "  CARIBOU Plotting"
echo "  Dataset : $DATASET_ID"
echo "  Plots   : $PLOTS_DIR"
echo "  Clean   : $CLEAN_PLOTS_DIR"
echo "  Started : $(date)"
echo "========================================================="

cd "$CARIBOU_ROOT"

python3 "$ANALYSIS_DIR/plot.py" \
    --output-dir "$PLOTS_DIR" \
    --dataset "$DATASET_ID"

echo ""
echo "========================================================="
echo "  Clean plots (SCIB + marker panels)"
echo "========================================================="

python3 "$CLEAN_PLOTS_DIR/make_clean_plots.py" \
    --dataset "$DATASET_ID"

echo ""
echo "========================================================="
echo "  Plotting complete: $(date)"
echo "========================================================="
