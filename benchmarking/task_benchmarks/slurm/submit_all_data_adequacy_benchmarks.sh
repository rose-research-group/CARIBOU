#!/bin/bash
ROOT_DIR="$(git -C "${SLURM_SUBMIT_DIR:-$(pwd)}" rev-parse --show-toplevel 2>/dev/null || true)"
if [ -z "$ROOT_DIR" ]; then
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
fi
# Submit all data adequacy benchmark jobs

SLURM_DIR="$ROOT_DIR/benchmarking/task_benchmarks/slurm"
LOG_DIR="$ROOT_DIR/benchmarking/task_benchmarks/results/logs/data_adequacy"

echo "Submitting Data Adequacy Benchmark Jobs..."

mkdir -p "$LOG_DIR"

# One-shot tests
sbatch --chdir "$ROOT_DIR" "$SLURM_DIR/one_shot_chatgpt_data_adequacy.sh"
sbatch --chdir "$ROOT_DIR" "$SLURM_DIR/one_shot_claude_data_adequacy.sh"
sbatch --chdir "$ROOT_DIR" "$SLURM_DIR/one_shot_deepseek_data_adequacy.sh"

# Single-agent tests
sbatch --chdir "$ROOT_DIR" "$SLURM_DIR/single_agent_chatgpt_data_adequacy.sh"
sbatch --chdir "$ROOT_DIR" "$SLURM_DIR/single_agent_claude_data_adequacy.sh"
sbatch --chdir "$ROOT_DIR" "$SLURM_DIR/single_agent_deepseek_data_adequacy.sh"

# Full-system tests
sbatch --chdir "$ROOT_DIR" "$SLURM_DIR/full_system_chatgpt_data_adequacy.sh"
sbatch --chdir "$ROOT_DIR" "$SLURM_DIR/full_system_claude_data_adequacy.sh"
sbatch --chdir "$ROOT_DIR" "$SLURM_DIR/full_system_deepseek_data_adequacy.sh"

echo "All jobs submitted. Use 'squeue -u $USER' to monitor."
