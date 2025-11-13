#!/bin/bash
#SBATCH --job-name=auto_run  # More descriptive job name
#SBATCH --cpus-per-task=8             # Number of CPU cores
#SBATCH --mem=16GB                    # Total RAM requested
#SBATCH --time=24:00:00               # Max wall time (hh:mm:ss)
#SBATCH --output=auto_run.log  # Updated log file name
#SBATCH --partition=cpu

BLUEPRINT_PATH="full blue print path"
DATASET_PATH="full dataset path"
SANDBOX_BACKEND="singularity"
LLM_BACKEND="deepseek"
NUM_TURNS=15
OUTPUT_DIR="full output dir path"
DESCRIPTION="logs_F_mem_deep"
BENCHMARK_PATH="full benchmark module path"
JOB_ID=${SLURM_JOB_ID:-$$}
for i in {1..10}
do
  echo "--- Starting Run $i ---"

  INITIAL_PROMPT="Prompt goes here"
  RUN_DIR="$OUTPUT_DIR/${DESCRIPTION}_${JOB_ID}_$i"
  # save params in run directory
  mkdir -p "$RUN_DIR"
  echo "BLUEPRINT_PATH: $BLUEPRINT_PATH" > "$RUN_DIR/params.txt"
  echo "DATASET_PATH: $DATASET_PATH" >> "$RUN_DIR/params.txt"
  echo "SANDBOX_BACKEND: $SANDBOX_BACKEND" >> "$RUN_DIR/params.txt"
  echo "LLM_BACKEND: $LLM_BACKEND" >> "$RUN_DIR/params.txt"
  echo "NUM_TURNS: $NUM_TURNS" >> "$RUN_DIR/params.txt"
  echo "INITIAL_PROMPT: $INITIAL_PROMPT" >> "$RUN_DIR/params.txt"

  caribou run auto \
    --blueprint "$BLUEPRINT_PATH" \
    --dataset "$DATASET_PATH" \
    --sandbox "$SANDBOX_BACKEND" \
    --llm "$LLM_BACKEND" \
    --turns "$NUM_TURNS" \
    --prompt "$INITIAL_PROMPT" \
    --driver-agent "master_agent" \
    --output-dir "$RUN_DIR" \
    --benchmark-module "$BENCHMARK_PATH"

  echo "--- Run $i Finished ---"
  echo "" 
done

echo "--- All Runs Complete ---"