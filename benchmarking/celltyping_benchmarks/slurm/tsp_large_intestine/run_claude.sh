#!/bin/bash
# Run all three modes for tsp_large_intestine × Claude as a SLURM array job.
#
# Usage:
#   sbatch slurm/tsp_large_intestine/run_claude.sh
#
# Array task → mode mapping:
#   1 → full_system
#   2 → single_agent
#   3 → one_shot
#   4 → full_system_no_mem
#
#SBATCH --job-name=cmp_tsp_claude
#SBATCH --partition=peerd
#SBATCH --array=1-4
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=../logs/tsp_claude_%A_%a.out
#SBATCH --error=../logs/tsp_claude_%A_%a.err

set -euo pipefail

MODES=("full_system" "single_agent" "one_shot" "full_system_no_mem")
MODE="${MODES[$((SLURM_ARRAY_TASK_ID - 1))]}"

RUN_SCRIPT="/data1/peerd/riffled/riffled/Olaf_project/CARIBOU/benchmarking/celltyping_benchmarks/slurm/run_celltyping.sh"
bash "$RUN_SCRIPT" --dataset tsp_large_intestine --llm claude --mode "$MODE"
