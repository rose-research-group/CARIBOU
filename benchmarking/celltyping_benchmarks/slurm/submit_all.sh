#!/bin/bash
# Submit all comparison runs for every configured dataset × LLM × mode combination.
#
# Edit the three arrays below to control what gets launched.
#
# Usage:
#   cd slurm/
#   ./submit_all.sh
#   ./submit_all.sh --datasets "aba_hippocampus tsp_large_intestine"   # override datasets
#   ./submit_all.sh --llms "chatgpt deepseek"                          # override LLMs
#   ./submit_all.sh --modes "full_system single_agent"                 # override modes
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMP_DIR="$(dirname "$SCRIPT_DIR")"

# ---------------------------------------------------------------------------
# Configuration — edit these
# ---------------------------------------------------------------------------
DATASETS=(aba_hippocampus tsp_large_intestine)
LLMS=(chatgpt deepseek)
MODES=(full_system single_agent one_shot full_system_no_mem)

# ---------------------------------------------------------------------------
# Allow overrides from CLI
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --datasets) IFS=' ' read -r -a DATASETS <<< "$2"; shift 2 ;;
        --llms)     IFS=' ' read -r -a LLMS     <<< "$2"; shift 2 ;;
        --modes)    IFS=' ' read -r -a MODES    <<< "$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

mkdir -p "$SCRIPT_DIR/logs"

TOTAL=$(( ${#DATASETS[@]} * ${#LLMS[@]} * ${#MODES[@]} ))
echo "========================================================"
echo "  Submitting $TOTAL CARIBOU comparison runs"
echo "  Datasets : ${DATASETS[*]}"
echo "  LLMs     : ${LLMS[*]}"
echo "  Modes    : ${MODES[*]}"
echo "========================================================"
echo ""

SUBMITTED=()

for DATASET in "${DATASETS[@]}"; do
    CONFIG="$COMP_DIR/datasets/$DATASET/config.json"
    if [[ ! -f "$CONFIG" ]]; then
        echo "  WARNING: Skipping $DATASET — no config.json found"
        continue
    fi
    MEM=$(python3 -c "import json; c=json.load(open('$CONFIG')); print(c.get('slurm_mem','64G'))")
    TIME=$(python3 -c "import json; c=json.load(open('$CONFIG')); print(c.get('slurm_time','12:00:00'))")

    for LLM in "${LLMS[@]}"; do
        for MODE in "${MODES[@]}"; do
            JOB_ID=$(sbatch \
                --parsable \
                --partition=peerd \
                --mem="$MEM" \
                --time="$TIME" \
                --job-name="cmp_${DATASET}_${LLM}_${MODE}" \
                "$SCRIPT_DIR/run_celltyping.sh" \
                    --dataset "$DATASET" \
                    --llm     "$LLM" \
                    --mode    "$MODE")
            echo "  Submitted: $DATASET / $LLM / $MODE  (job $JOB_ID)"
            SUBMITTED+=("$JOB_ID")
        done
    done
done

echo ""
echo "========================================================"
echo "  All $TOTAL jobs submitted"
echo "  Job IDs: ${SUBMITTED[*]}"
echo ""
echo "Monitor:"
echo "  squeue -j $(IFS=,; echo "${SUBMITTED[*]}")"
echo ""
echo "After completion:"
echo "  cd ../analysis && python evaluate.py"
echo "========================================================"
