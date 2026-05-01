#!/bin/bash
# Run the full evaluation pipeline on completed integration benchmark results:
#   1. run_evaluation.py  — compute per-run metrics + reference baseline
#   2. collect_results.py — aggregate into summary.json/csv
#   3. plot.py            — generate figures
#
# Usage:
#   sbatch run_analysis.sh                              # all datasets
#   sbatch run_analysis.sh --dataset aba_hippocampus    # one dataset
#
#SBATCH --job-name=intg_analysis
#SBATCH --partition=peerd
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
DATASET_FILTER=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset) DATASET_FILTER="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR="$(git -C "${SLURM_SUBMIT_DIR:-$(pwd)}" rev-parse --show-toplevel 2>/dev/null || true)"
if [ -z "$ROOT_DIR" ]; then
    ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
fi

INTBENCH_DIR="$ROOT_DIR/benchmarking/integration_benchmarks"
PYTHON="/data1/peerd/riffled/riffled/conda_envs/olaf/bin/python"

ANALYSIS_DIR="$INTBENCH_DIR/analysis"
OUTPUT_DIR="$ANALYSIS_DIR/outputs"
PLOTS_DIR="$ANALYSIS_DIR/plots"
LOG_DIR="$INTBENCH_DIR/slurm/logs"

mkdir -p "$OUTPUT_DIR" "$PLOTS_DIR" "$LOG_DIR"
LOG_PATH="$LOG_DIR/intg_analysis_${SLURM_JOB_ID:-0}.log"
exec > "$LOG_PATH" 2>&1

echo "========================================================="
echo "  CARIBOU Integration Benchmark Analysis"
echo "  Dataset filter : ${DATASET_FILTER:-all}"
echo "  Outputs        : $OUTPUT_DIR"
echo "  Plots          : $PLOTS_DIR"
echo "  Started        : $(date)"
echo "========================================================="

DATASET_ARGS=()
if [[ -n "$DATASET_FILTER" ]]; then
    read -r -a DS_LIST <<< "$DATASET_FILTER"
    DATASET_ARGS=(--dataset "${DS_LIST[@]}")
fi

cd "$INTBENCH_DIR"

# ---------------------------------------------------------------------------
# Step 1 — Evaluate
# ---------------------------------------------------------------------------
echo ""
echo "---------------------------------------------------------"
echo "  Step 1/3: run_evaluation.py"
echo "---------------------------------------------------------"
"$PYTHON" "$ANALYSIS_DIR/run_evaluation.py" \
    --output-dir "$OUTPUT_DIR" \
    "${DATASET_ARGS[@]}"

# ---------------------------------------------------------------------------
# Step 2 — Collect
# ---------------------------------------------------------------------------
echo ""
echo "---------------------------------------------------------"
echo "  Step 2/3: collect_results.py"
echo "---------------------------------------------------------"
"$PYTHON" "$ANALYSIS_DIR/collect_results.py" \
    --output-dir "$OUTPUT_DIR" \
    "${DATASET_ARGS[@]}"

# ---------------------------------------------------------------------------
# Step 3 — Plot
# ---------------------------------------------------------------------------
echo ""
echo "---------------------------------------------------------"
echo "  Step 3/3: plot.py"
echo "---------------------------------------------------------"
"$PYTHON" "$ANALYSIS_DIR/plot.py" \
    --output-dir "$PLOTS_DIR" \
    --outputs-dir "$OUTPUT_DIR" \
    "${DATASET_ARGS[@]}"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "========================================================="
echo "  Analysis complete: $(date)"
echo ""
echo "  Outputs : $OUTPUT_DIR"
echo "  Plots   : $PLOTS_DIR"
echo ""
echo "  Result files:"
find "$OUTPUT_DIR" -name "*.json" -o -name "*.csv" 2>/dev/null | sort | sed 's|^|    |'
echo ""
echo "  Plots:"
find "$PLOTS_DIR"  -name "*.png" 2>/dev/null | sort | sed 's|^|    |'
echo "========================================================="
