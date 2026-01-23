# Task Benchmarking Test Suite: Comparative Execution Mode Testing

## Overview

This testing framework compares three execution paradigms on the same scRNA-seq tasks:
1. **One-Shot API** - Single LLM call with complete instructions (no agent framework)
2. **Single Agent** - One agent running in auto mode (no delegation)
3. **Full Agent System** - Multi-agent system with delegation (existing OLAF architecture)

All three modes use the **same initial prompt** and run in **auto mode** via **SLURM** for fair comparison.

## Layout

```
dev/task_benchmarks/
├── analysis/
│   └── compare_qc_results.py
├── configs/
│   ├── qc_single_agent.json
│   └── shared_params.json
├── results/
│   ├── logs/
│   ├── one_shot/
│   ├── single_agent/
│   └── full_system/
├── slurm/
│   ├── one_shot_chatgpt_qc.sh
│   ├── one_shot_claude_qc.sh
│   ├── one_shot_deepseek_qc.sh
│   ├── one_shot_chatgpt_load.sh
│   ├── one_shot_claude_load.sh
│   ├── one_shot_deepseek_load.sh
│   ├── single_agent_chatgpt_qc.sh
│   ├── single_agent_claude_qc.sh
│   ├── single_agent_deepseek_qc.sh
│   ├── single_agent_chatgpt_load.sh
│   ├── single_agent_claude_load.sh
│   ├── single_agent_deepseek_load.sh
│   ├── full_system_chatgpt_qc.sh
│   ├── full_system_claude_qc.sh
│   ├── full_system_deepseek_qc.sh
│   ├── full_system_chatgpt_load.sh
│   ├── full_system_claude_load.sh
│   ├── full_system_deepseek_load.sh
│   └── submit_all_qc_benchmarks.sh
│   └── submit_all_load_benchmarks.sh
└── src/
    ├── one_shot_runner.py
    ├── load_prompt.py
    ├── qc_prompt.py
    └── results_collector.py
```

## Running the Benchmarks

- Submit all jobs:
  - `bash dev/task_benchmarks/slurm/submit_all_qc_benchmarks.sh`
- Submit all load-data jobs:
  - `bash dev/task_benchmarks/slurm/submit_all_load_benchmarks.sh`
- Submit a single script:
  - `sbatch dev/task_benchmarks/slurm/one_shot_chatgpt_qc.sh`

Each SLURM script writes logs to `dev/task_benchmarks/results/logs` and run outputs to the task-specific mode subdirectory (for example `dev/task_benchmarks/results/load_data/one_shot` or `dev/task_benchmarks/results/qc_task/one_shot`).

## Collecting Results

- Collect run summaries:
  - `python dev/task_benchmarks/src/results_collector.py --results-dir dev/task_benchmarks/results --output-csv dev/task_benchmarks/results/summary.csv`
- Generate a grouped comparison table:
  - `python dev/task_benchmarks/analysis/compare_qc_results.py --results-dir dev/task_benchmarks/results`

The comparison script writes a JSON summary to `dev/task_benchmarks/analysis/task_benchmark_summary.json`.

## Additional Task Prompts

Two additional prompt files are available:
- `benchmarking/task_benchmarks/prompts/doublet_prompt.txt` (doublet detection + filtering)
- `benchmarking/task_benchmarks/prompts/full_qc_prompt.txt` (end-to-end QC workflow)

To run them with the migrated scripts, pass a prompt file via `PROMPT_PATH`, for example:
  - `PROMPT_PATH=benchmarking/task_benchmarks/prompts/doublet_prompt.txt`
  - `PROMPT_PATH=benchmarking/task_benchmarks/prompts/full_qc_prompt.txt`
