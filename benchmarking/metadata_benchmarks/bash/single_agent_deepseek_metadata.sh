#!/bin/bash
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

# Configuration
BLUEPRINT_PATH="$ROOT_DIR/benchmarking/metadata_benchmarks/configs/metadata_single_agent.json"
MANIFEST_PATH="$ROOT_DIR/benchmarking/metadata_benchmarks/benchmark_data/benchmark_manifest.csv"
OUTPUT_BASE="$ROOT_DIR/benchmarking/metadata_benchmarks/results/metadata_task/single_agent"
SANDBOX_BACKEND="singularity"
LLM_BACKEND="deepseek"
NUM_TURNS=6
NUM_TRIALS=3

mkdir -p "$ROOT_DIR/benchmarking/metadata_benchmarks/results/logs/metadata"



while IFS=, read -r DATASET_NAME DATASET_PATH; do
    if [ "$DATASET_NAME" = "dataset_name" ]; then
      continue
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
  import sys
  sys.path.insert(0, "$ROOT_DIR")
  from dev.metadata_benchmarks.metadata_prompt import METADATA_PROMPT
  print(METADATA_PROMPT)
  PY
  )"
  
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
done < "$MANIFEST_PATH"
