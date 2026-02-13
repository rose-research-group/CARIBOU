#!/bin/bash
ROOT_DIR="$(git -C "${SLURM_SUBMIT_DIR:-$(pwd)}" rev-parse --show-toplevel 2>/dev/null || true)"
if [ -z "$ROOT_DIR" ]; then
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
fi
# Submit all one-shot benchmark jobs across all tasks

SLURM_DIR="$ROOT_DIR/benchmarking/task_benchmarks/slurm"

echo "Submitting All One-Shot Benchmark Jobs... (no Claude)"

# Load data
sbatch --chdir "$ROOT_DIR" "$SLURM_DIR/one_shot_chatgpt_load.sh"
sbatch --chdir "$ROOT_DIR" "$SLURM_DIR/one_shot_deepseek_load.sh"

# QC
sbatch --chdir "$ROOT_DIR" "$SLURM_DIR/one_shot_chatgpt_qc.sh"
sbatch --chdir "$ROOT_DIR" "$SLURM_DIR/one_shot_deepseek_qc.sh"

# Doublet
sbatch --chdir "$ROOT_DIR" "$SLURM_DIR/one_shot_chatgpt_doublet.sh"
sbatch --chdir "$ROOT_DIR" "$SLURM_DIR/one_shot_deepseek_doublet.sh"

# Full QC
sbatch --chdir "$ROOT_DIR" "$SLURM_DIR/one_shot_chatgpt_full_qc.sh"
sbatch --chdir "$ROOT_DIR" "$SLURM_DIR/one_shot_deepseek_full_qc.sh"

# Batch correction
sbatch --chdir "$ROOT_DIR" "$SLURM_DIR/one_shot_chatgpt_batch_correction.sh"
sbatch --chdir "$ROOT_DIR" "$SLURM_DIR/one_shot_deepseek_batch_correction.sh"

# DEG
sbatch --chdir "$ROOT_DIR" "$SLURM_DIR/one_shot_chatgpt_deg.sh"
sbatch --chdir "$ROOT_DIR" "$SLURM_DIR/one_shot_deepseek_deg.sh"

# Data adequacy
sbatch --chdir "$ROOT_DIR" "$SLURM_DIR/one_shot_chatgpt_data_adequacy.sh"
sbatch --chdir "$ROOT_DIR" "$SLURM_DIR/one_shot_deepseek_data_adequacy.sh"

echo "All one-shot jobs submitted. Use 'squeue -u $USER' to monitor."
