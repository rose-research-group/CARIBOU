#!/bin/bash
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# Submit all QC benchmark jobs

SLURM_DIR="$ROOT_DIR/benchmarking/task_benchmarks/slurm"
LOG_DIR="$ROOT_DIR/benchmarking/task_benchmarks/results/logs/qc"

echo "Submitting QC Benchmark Jobs..."

mkdir -p "$LOG_DIR"

# One-shot tests
sbatch "$SLURM_DIR/one_shot_chatgpt_qc.sh"
sbatch "$SLURM_DIR/one_shot_claude_qc.sh"
sbatch "$SLURM_DIR/one_shot_deepseek_qc.sh"

# Single-agent tests
sbatch "$SLURM_DIR/single_agent_chatgpt_qc.sh"
sbatch "$SLURM_DIR/single_agent_claude_qc.sh"
sbatch "$SLURM_DIR/single_agent_deepseek_qc.sh"

# Full-system tests
sbatch "$SLURM_DIR/full_system_chatgpt_qc.sh"
sbatch "$SLURM_DIR/full_system_claude_qc.sh"
sbatch "$SLURM_DIR/full_system_deepseek_qc.sh"

echo "All jobs submitted. Use 'squeue -u $USER' to monitor."
PROMPT_PATH="${PROMPT_PATH:-}"
if [ -z "$PROMPT_PATH" ]; then
  if [ -t 0 ]; then
    read -r -p "Enter prompt file path: " PROMPT_PATH
  else
    echo "PROMPT_PATH is required. Export PROMPT_PATH=<path_to_prompt.txt>."
    exit 1
  fi
fi
if [ ! -f "$PROMPT_PATH" ]; then
  echo "Prompt file not found: $PROMPT_PATH"
  exit 1
fi
INITIAL_PROMPT="$(cat "$PROMPT_PATH")"
