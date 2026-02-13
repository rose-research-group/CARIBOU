#!/bin/bash
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

BASH_DIR="$ROOT_DIR/benchmarking/task_benchmarks/bash"
LOG_DIR="$ROOT_DIR/benchmarking/task_benchmarks/results/logs/data_adequacy"

echo "Running Data Adequacy Benchmark Scripts..."

mkdir -p "$LOG_DIR"

$BASH_DIR/one_shot_chatgpt_data_adequacy.sh
$BASH_DIR/one_shot_claude_data_adequacy.sh
$BASH_DIR/one_shot_deepseek_data_adequacy.sh

$BASH_DIR/single_agent_chatgpt_data_adequacy.sh
$BASH_DIR/single_agent_claude_data_adequacy.sh
$BASH_DIR/single_agent_deepseek_data_adequacy.sh

$BASH_DIR/full_system_chatgpt_data_adequacy.sh
$BASH_DIR/full_system_claude_data_adequacy.sh
$BASH_DIR/full_system_deepseek_data_adequacy.sh

