#!/bin/bash
# Submit all LLM providers for aba_hippocampus (each as a 3-mode array job).
#
# Usage:
#   cd slurm/aba_hippocampus/
#   ./submit_all.sh
#   ./submit_all.sh --llms "chatgpt deepseek"   # subset of LLMs
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET="aba_hippocampus"

LLMS=(chatgpt deepseek claude)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --llms) IFS=' ' read -r -a LLMS <<< "$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

echo "========================================================"
echo "  $DATASET — submitting ${#LLMS[@]} LLM(s) × 3 modes"
echo "  LLMs: ${LLMS[*]}"
echo "========================================================"
echo ""

SUBMITTED=()
for LLM in "${LLMS[@]}"; do
    SCRIPT="$SCRIPT_DIR/run_${LLM}.sh"
    if [[ ! -f "$SCRIPT" ]]; then
        echo "  WARNING: No script for LLM '$LLM' (expected $SCRIPT)"
        continue
    fi
    JOB_ID=$(sbatch --parsable "$SCRIPT")
    echo "  $LLM → job array $JOB_ID (tasks 1-3 = full_system / single_agent / one_shot)"
    SUBMITTED+=("$JOB_ID")
done

echo ""
echo "========================================================"
echo "  Submitted ${#SUBMITTED[@]} job array(s): ${SUBMITTED[*]}"
echo ""
echo "Monitor:"
echo "  squeue -j $(IFS=,; echo "${SUBMITTED[*]}")"
echo ""
echo "After completion:"
echo "  cd ../../analysis && python evaluate.py --dataset $DATASET"
echo "========================================================"
