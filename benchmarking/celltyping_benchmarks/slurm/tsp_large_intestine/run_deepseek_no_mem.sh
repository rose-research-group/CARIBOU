#!/bin/bash
# Run full_system_no_mem for tsp_large_intestine × DeepSeek as a single SLURM job.
#
# Usage:
#   sbatch slurm/tsp_large_intestine/run_deepseek_no_mem.sh
#
#SBATCH --job-name=cmp_tsp_deepseek_no_mem
#SBATCH --partition=peerd
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=../logs/tsp_deepseek_no_mem_%j.out
#SBATCH --error=../logs/tsp_deepseek_no_mem_%j.err

set -euo pipefail

RUN_SCRIPT="/data1/peerd/riffled/riffled/Olaf_project/CARIBOU/benchmarking/celltyping_benchmarks/slurm/run_celltyping.sh"
bash "$RUN_SCRIPT" --dataset tsp_large_intestine --llm deepseek --mode full_system_no_mem
