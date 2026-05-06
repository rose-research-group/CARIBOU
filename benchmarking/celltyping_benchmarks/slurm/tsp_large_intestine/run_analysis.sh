#!/bin/bash
# Run evaluate.py for tsp_large_intestine only.
#
# Usage:
#   sbatch slurm/tsp_large_intestine/run_analysis.sh
#
#SBATCH --job-name=analysis_tsp_large_intestine
#SBATCH --partition=peerd
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=../logs/analysis_tsp_large_intestine_%j.out
#SBATCH --error=../logs/analysis_tsp_large_intestine_%j.err

set -euo pipefail

DATASET_ID="tsp_large_intestine"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CARIBOU_ROOT="$(git -C "${SLURM_SUBMIT_DIR:-$(pwd)}" rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$CARIBOU_ROOT" ]]; then
    CARIBOU_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || true)"
fi
if [[ -z "$CARIBOU_ROOT" ]]; then
    CARIBOU_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
fi
ANALYSIS_DIR="$CARIBOU_ROOT/benchmarking/celltyping_benchmarks/analysis"
OUTPUT_DIR="$ANALYSIS_DIR/outputs"

mkdir -p "$OUTPUT_DIR" "$CARIBOU_ROOT/benchmarking/celltyping_benchmarks/slurm/logs"

echo "========================================================="
echo "  CARIBOU Analysis"
echo "  Dataset : $DATASET_ID"
echo "  Outputs : $OUTPUT_DIR"
echo "  Started : $(date)"
echo "========================================================="

cd "$CARIBOU_ROOT"

python3 "$ANALYSIS_DIR/evaluate.py" \
    --output-dir "$OUTPUT_DIR" \
    --dataset "$DATASET_ID"

echo ""
echo "========================================================="
echo "  Analysis complete: $(date)"
echo "========================================================="
