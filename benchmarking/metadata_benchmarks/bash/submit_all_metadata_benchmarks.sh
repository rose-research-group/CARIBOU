#!/bin/bash
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

sbatch "${SCRIPT_DIR}/single_agent_chatgpt_metadata.sh"
sbatch "${SCRIPT_DIR}/single_agent_deepseek_metadata.sh"
sbatch "${SCRIPT_DIR}/full_system_chatgpt_metadata.sh"
sbatch "${SCRIPT_DIR}/full_system_deepseek_metadata.sh"
while IFS=, read -r DATASET_NAME DATASET_PATH; do
    if [ "$DATASET_NAME" = "dataset_name" ]; then
      continue
    fi
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
done < "$MANIFEST_PATH"
