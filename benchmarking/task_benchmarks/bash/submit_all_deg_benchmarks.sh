#!/bin/bash
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

BASH_DIR="$ROOT_DIR/benchmarking/task_benchmarks/bash"
LOG_DIR="$ROOT_DIR/benchmarking/task_benchmarks/results/logs/deg"

echo "Running DEG Benchmark Scripts..."

mkdir -p "$LOG_DIR"

$BASH_DIR/one_shot_chatgpt_deg.sh
$BASH_DIR/one_shot_claude_deg.sh
$BASH_DIR/one_shot_deepseek_deg.sh

$BASH_DIR/single_agent_chatgpt_deg.sh
$BASH_DIR/single_agent_claude_deg.sh
$BASH_DIR/single_agent_deepseek_deg.sh

$BASH_DIR/full_system_chatgpt_deg.sh
$BASH_DIR/full_system_claude_deg.sh
$BASH_DIR/full_system_deepseek_deg.sh

