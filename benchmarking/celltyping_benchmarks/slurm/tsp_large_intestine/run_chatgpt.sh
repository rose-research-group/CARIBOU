#!/bin/bash
# Run all three modes for tsp_large_intestine × ChatGPT as a SLURM array job.
#
# Usage:
#   sbatch slurm/tsp_large_intestine/run_chatgpt.sh
#
# Array task → mode mapping:
#   1 → full_system
#   2 → single_agent
#   3 → one_shot
#   4 → full_system_no_mem
#
#SBATCH --job-name=cmp_tsp_chatgpt
#SBATCH --partition=peerd
#SBATCH --array=1-4
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=../logs/tsp_chatgpt_%A_%a.out
#SBATCH --error=../logs/tsp_chatgpt_%A_%a.err

set -euo pipefail

MODES=("full_system" "single_agent" "one_shot" "full_system_no_mem")
MODE="${MODES[$((SLURM_ARRAY_TASK_ID - 1))]}"

RUN_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/run_celltyping.sh"
bash "$RUN_SCRIPT" --dataset tsp_large_intestine --llm chatgpt --mode "$MODE"
