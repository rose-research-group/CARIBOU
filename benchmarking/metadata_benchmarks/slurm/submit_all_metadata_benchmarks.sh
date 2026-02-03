#!/bin/bash
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

sbatch --chdir "$ROOT_DIR" "${SCRIPT_DIR}/single_agent_chatgpt_metadata.sh"
sbatch --chdir "$ROOT_DIR" "${SCRIPT_DIR}/single_agent_deepseek_metadata.sh"
sbatch --chdir "$ROOT_DIR" "${SCRIPT_DIR}/full_system_chatgpt_metadata.sh"
sbatch --chdir "$ROOT_DIR" "${SCRIPT_DIR}/full_system_deepseek_metadata.sh"
