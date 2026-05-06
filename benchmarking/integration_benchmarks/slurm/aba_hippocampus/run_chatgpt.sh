#!/bin/bash
# Run all 4 modes for aba_hippocampus × ChatGPT as a SLURM array job.
#
# Usage:
#   sbatch slurm/aba_hippocampus/run_chatgpt.sh
#
# Array task → mode mapping:
#   1 → full_system
#   2 → full_system_no_mem
#   3 → single_agent
#   4 → one_shot
#
#SBATCH --job-name=intg_aba_chatgpt
#SBATCH --partition=peerd
#SBATCH --array=1-4
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=../logs/aba_chatgpt_%A_%a.out
#SBATCH --error=../logs/aba_chatgpt_%A_%a.err

set -euo pipefail

MODES=("full_system" "full_system_no_mem" "single_agent" "one_shot")
MODE="${MODES[$((SLURM_ARRAY_TASK_ID - 1))]}"

CARIBOU_ROOT="$(git -C "${SLURM_SUBMIT_DIR:-$(pwd)}" rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$CARIBOU_ROOT" ]]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    CARIBOU_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || true)"
fi
if [[ -z "$CARIBOU_ROOT" ]]; then
    CARIBOU_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
fi
RUN_SCRIPT="$CARIBOU_ROOT/benchmarking/integration_benchmarks/slurm/run_integration.sh"
bash "$RUN_SCRIPT" --dataset aba_hippocampus --llm chatgpt --mode "$MODE"
