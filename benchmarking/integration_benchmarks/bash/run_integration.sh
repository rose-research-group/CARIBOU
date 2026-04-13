#!/bin/bash
# Local (non-SLURM) runner for a single integration benchmark run.
#
# Usage:
#   LLM=chatgpt MODE=full_system ./bash/run_integration.sh
#   LLM=deepseek MODE=single_agent DATASET=aba_hippocampus ./bash/run_integration.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INTBENCH_DIR="$(dirname "$SCRIPT_DIR")"

DATASET="${DATASET:-aba_hippocampus}"
LLM="${LLM:-chatgpt}"
MODE="${MODE:-full_system}"

echo "Running: DATASET=$DATASET  LLM=$LLM  MODE=$MODE"

bash "$INTBENCH_DIR/slurm/run_integration.sh" \
    --dataset "$DATASET" \
    --llm     "$LLM" \
    --mode    "$MODE"
