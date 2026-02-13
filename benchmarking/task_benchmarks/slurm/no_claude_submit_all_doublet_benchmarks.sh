#!/bin/bash
ROOT_DIR="$(git -C "${SLURM_SUBMIT_DIR:-$(pwd)}" rev-parse --show-toplevel 2>/dev/null || true)"
if [ -z "$ROOT_DIR" ]; then
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
fi
# Submit all doublet benchmark jobs

SLURM_DIR="$ROOT_DIR/benchmarking/task_benchmarks/slurm"
LOG_DIR="$ROOT_DIR/benchmarking/task_benchmarks/results/logs/doublet"

echo "Submitting doublet Benchmark Jobs... (no Claude)"

mkdir -p "$LOG_DIR"

# One-shot tests
sbatch --chdir "$ROOT_DIR" "$SLURM_DIR/one_shot_chatgpt_doublet.sh"
sbatch --chdir "$ROOT_DIR" "$SLURM_DIR/one_shot_deepseek_doublet.sh"

# Single-agent tests
sbatch --chdir "$ROOT_DIR" "$SLURM_DIR/single_agent_chatgpt_doublet.sh"
sbatch --chdir "$ROOT_DIR" "$SLURM_DIR/single_agent_deepseek_doublet.sh"

# Full-system tests
sbatch --chdir "$ROOT_DIR" "$SLURM_DIR/full_system_chatgpt_doublet.sh"
sbatch --chdir "$ROOT_DIR" "$SLURM_DIR/full_system_deepseek_doublet.sh"

echo "All doublet jobs submitted. Use 'squeue -u $USER' to monitor."
