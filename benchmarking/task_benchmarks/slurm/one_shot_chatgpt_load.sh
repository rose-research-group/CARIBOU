#!/bin/bash
#SBATCH --job-name=load_one_shot_chatgpt
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=1:00:00
#SBATCH --output=/dev/null
#SBATCH --partition=peerd
ROOT_DIR="$(git -C "${SLURM_SUBMIT_DIR:-$(pwd)}" rev-parse --show-toplevel 2>/dev/null || true)"
if [ -z "$ROOT_DIR" ]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
fi
LOG_DIR="$ROOT_DIR/benchmarking/task_benchmarks/results/logs/load_data"
mkdir -p "$LOG_DIR"
LOG_PATH="$LOG_DIR/one_shot_chatgpt_load_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}.log"
exec > "$LOG_PATH" 2>&1

# Configuration
PROMPT_PATH="${PROMPT_PATH:-$ROOT_DIR/benchmarking/task_benchmarks/prompts/load_prompt.txt}"
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
DATASET_PATH="$ROOT_DIR/dev/datasets/pbmc_1k_v2_v3_combined.h5ad"
OUTPUT_BASE="$ROOT_DIR/benchmarking/task_benchmarks/results/load_data/one_shot"
BENCHMARK_MODULE="$ROOT_DIR/caribou/src/caribou/auto_metrics/LoadDataMetric.py"
LLM_BACKEND="chatgpt"
NUM_TRIALS=3

mkdir -p "$OUTPUT_BASE/logs"

for trial in $(seq 1 "$NUM_TRIALS"); do
    echo "================================================================================"
    echo "Starting One-Shot Load Trial $trial of $NUM_TRIALS"
    echo "LLM: $LLM_BACKEND | Mode: one_shot"
    echo "================================================================================"

    JOB_ID=${SLURM_JOB_ID:-$$}
    RUN_DIR="$OUTPUT_BASE/${LLM_BACKEND}_${JOB_ID}_trial${trial}"

    python $ROOT_DIR/benchmarking/task_benchmarks/src/one_shot_runner.py \
        --dataset "$DATASET_PATH" \
        --output-dir "$RUN_DIR" \
        --llm "$LLM_BACKEND" \
        --sandbox singularity \
        --benchmark-module "$BENCHMARK_MODULE" \
        --prompt-module load_prompt \
        --prompt-var LOAD_PROMPT

    echo "Trial $trial completed"
done

echo "All one-shot load trials complete"
