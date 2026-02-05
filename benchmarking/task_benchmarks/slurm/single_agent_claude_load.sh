#!/bin/bash
#SBATCH --job-name=load_single_agent_claude
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=2:00:00
#SBATCH --output=/dev/null
#SBATCH --partition=peerd
ROOT_DIR="$(git -C "${SLURM_SUBMIT_DIR:-$(pwd)}" rev-parse --show-toplevel 2>/dev/null || true)"
if [ -z "$ROOT_DIR" ]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
fi
LOG_DIR="$ROOT_DIR/benchmarking/task_benchmarks/results/logs/load_data"
mkdir -p "$LOG_DIR"
LOG_PATH="$LOG_DIR/single_agent_claude_load_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}.log"
exec > "$LOG_PATH" 2>&1

# Configuration
BLUEPRINT_PATH="$ROOT_DIR/benchmarking/task_benchmarks/configs/load_single_agent.json"
DATASET_PATH="$ROOT_DIR/dev/datasets/pbmc_1k_v2_v3_combined.h5ad"
OUTPUT_BASE="$ROOT_DIR/benchmarking/task_benchmarks/results/load_data/single_agent"
SANDBOX_BACKEND="singularity"
BENCHMARK_ID="load_data"
LLM_BACKEND="claude"
NUM_TURNS=6
NUM_TRIALS=3

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

mkdir -p "$OUTPUT_BASE/logs"

for trial in $(seq 1 "$NUM_TRIALS"); do
    echo "================================================================================"
    echo "Starting Single Agent Load Trial $trial of $NUM_TRIALS"
    echo "LLM: $LLM_BACKEND | Mode: single_agent | Turns: $NUM_TURNS"
    echo "================================================================================"

    JOB_ID=${SLURM_JOB_ID:-$$}
    RUN_DIR="$OUTPUT_BASE/${LLM_BACKEND}_${NUM_TURNS}turns_${JOB_ID}_trial${trial}"

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
        --driver-agent "load_agent" \
        --output-dir "$RUN_DIR" \
        --benchmark-id "$BENCHMARK_ID" \
        --make-report

    echo "Trial $trial completed"
done

echo "All single-agent load trials complete"
