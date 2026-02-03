#!/bin/bash
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

# Configuration
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
DATASET_PATH="$ROOT_DIR/benchmarking/datasets/pbmc_1k_v2_v3_combined.h5ad"
OUTPUT_BASE="$ROOT_DIR/benchmarking/task_benchmarks/results/qc_task/one_shot"
BENCHMARK_MODULE="$ROOT_DIR/caribou/src/caribou/auto_metrics/QCBenchmarkMetric.py"
LLM_BACKEND="deepseek"
NUM_TRIALS=3

# Create output directories
mkdir -p "$OUTPUT_BASE/logs"

# Run trials
for trial in $(seq 1 "$NUM_TRIALS"); do
    echo "================================================================================"
    echo "Starting One-Shot Trial $trial of $NUM_TRIALS"
    echo "LLM: $LLM_BACKEND | Mode: one_shot"
    echo "================================================================================"

    JOB_ID=${SLURM_JOB_ID:-$$}
    RUN_DIR="$OUTPUT_BASE/${LLM_BACKEND}_${JOB_ID}_trial${trial}"

    python $ROOT_DIR/benchmarking/task_benchmarks/src/one_shot_runner.py \
        --dataset "$DATASET_PATH" \
        --output-dir "$RUN_DIR" \
        --llm "$LLM_BACKEND" \
        --sandbox singularity

    echo "Trial $trial completed"
done

echo "All one-shot trials complete"
