#!/bin/bash
#SBATCH --job-name=meta_single_agent_deepseek
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=2:00:00
#SBATCH --output=/dev/null
#SBATCH --partition=peerd
#SBATCH --array=1-10
ROOT_DIR="$(git -C "${SLURM_SUBMIT_DIR:-$(pwd)}" rev-parse --show-toplevel 2>/dev/null || true)"
if [ -z "$ROOT_DIR" ]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
fi
LOG_DIR="$ROOT_DIR/benchmarking/metadata_benchmarks/results/logs/metadata"
mkdir -p "$LOG_DIR"
LOG_PATH="$LOG_DIR/single_agent_deepseek_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}.log"
exec > "$LOG_PATH" 2>&1

# Configuration
BLUEPRINT_PATH="$ROOT_DIR/benchmarking/metadata_benchmarks/configs/metadata_single_agent.json"
MANIFEST_PATH="$ROOT_DIR/benchmarking/metadata_benchmarks/benchmark_data/benchmark_manifest.csv"
OUTPUT_BASE="$ROOT_DIR/benchmarking/metadata_benchmarks/results/metadata_task/single_agent"
SANDBOX_BACKEND="singularity"
LLM_BACKEND="deepseek"
NUM_TURNS=6
NUM_TRIALS=3

mkdir -p "$ROOT_DIR/benchmarking/metadata_benchmarks/results/logs/metadata"

DATASET_NAME=$(awk -F, "NR==${SLURM_ARRAY_TASK_ID}+1 {print \$1}" "$MANIFEST_PATH")
DATASET_PATH=$(awk -F, "NR==${SLURM_ARRAY_TASK_ID}+1 {print \$2}" "$MANIFEST_PATH")

if [[ -z "$DATASET_NAME" || -z "$DATASET_PATH" ]]; then
    echo "Failed to resolve dataset for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
    exit 1
fi

PROMPT_PATH="${PROMPT_PATH:-$ROOT_DIR/benchmarking/metadata_benchmarks/prompts/metadata_prompt.txt}"
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

for trial in $(seq 1 "$NUM_TRIALS"); do
    echo "================================================================================"
    echo "Starting Single Agent Metadata Trial $trial of $NUM_TRIALS"
    echo "LLM: $LLM_BACKEND | Dataset: $DATASET_NAME | Turns: $NUM_TURNS"
    echo "================================================================================"

    JOB_ID=${SLURM_JOB_ID:-$$}
    RUN_DIR="$OUTPUT_BASE/${DATASET_NAME}/${LLM_BACKEND}_${NUM_TURNS}turns_${JOB_ID}_trial${trial}"

    mkdir -p "$RUN_DIR"
    echo "BLUEPRINT_PATH: $BLUEPRINT_PATH" > "$RUN_DIR/params.txt"
    echo "DATASET_PATH: $DATASET_PATH" >> "$RUN_DIR/params.txt"
    echo "LLM_BACKEND: $LLM_BACKEND" >> "$RUN_DIR/params.txt"
    echo "NUM_TURNS: $NUM_TURNS" >> "$RUN_DIR/params.txt"
    echo "TRIAL: $trial" >> "$RUN_DIR/params.txt"

    caribou run auto \
        --blueprint "$BLUEPRINT_PATH" \
        --dataset "$DATASET_PATH" \
        --sandbox "$SANDBOX_BACKEND" \
        --llm "$LLM_BACKEND" \
        --turns "$NUM_TURNS" \
        --prompt "$INITIAL_PROMPT" \
        --driver-agent "metadata_agent" \
        --output-dir "$RUN_DIR" \
        --make-report

    echo "Trial $trial completed"
done

echo "All single-agent metadata trials complete"
