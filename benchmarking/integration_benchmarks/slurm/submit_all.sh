#!/bin/bash
# Submit all integration benchmark runs for every configured dataset × LLM × mode combination.
#
# Usage:
#   cd slurm/
#   ./submit_all.sh
#   ./submit_all.sh --datasets "aba_hippocampus"
#   ./submit_all.sh --llms "chatgpt deepseek"
#   ./submit_all.sh --modes "full_system single_agent"
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INTBENCH_DIR="$(dirname "$SCRIPT_DIR")"

# ---------------------------------------------------------------------------
# Configuration — edit these
# ---------------------------------------------------------------------------
DATASETS=(aba_hippocampus)
LLMS=(chatgpt deepseek)
MODES=(full_system full_system_no_mem single_agent one_shot)

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
echo "  Submitting $TOTAL CARIBOU integration benchmark runs"
echo "  Datasets : ${DATASETS[*]}"
echo "  LLMs     : ${LLMS[*]}"
echo "  Modes    : ${MODES[*]}"
echo "========================================================"
echo ""

SUBMITTED=()

for DATASET in "${DATASETS[@]}"; do
    CONFIG="$INTBENCH_DIR/datasets/$DATASET/config.json"
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
                --job-name="intg_${DATASET}_${LLM}_${MODE}" \
                "$SCRIPT_DIR/run_integration.sh" \
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
echo "After completion, run analysis:"
echo "  sbatch run_analysis.sh"
echo "========================================================"
