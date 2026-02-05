# Task Benchmarking Test Suite: Comparative Execution Mode Testing

## Overview

This testing framework compares three execution paradigms on the same scRNA-seq tasks:
1. **One-Shot API** - Single LLM call with complete instructions (no agent framework)
2. **Single Agent** - One agent running in auto mode (no delegation)
3. **Full Agent System** - Multi-agent system with delegation (existing OLAF architecture)

All three modes use the **same initial prompt** and run in **auto mode** via **SLURM** for fair comparison.

## Layout

```
benchmarking/task_benchmarks/
├── analysis/
│   └── compare_qc_results.py
│   └── plot_task_benchmark_summary.py
├── configs/
│   ├── qc_single_agent.json
│   └── shared_params.json
├── prompts/
│   ├── qc_prompt.txt
│   ├── load_prompt.txt
│   ├── doublet_prompt.txt
│   └── full_qc_prompt.txt
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
├── bash/
│   └── (non-Slurm equivalents of slurm scripts)
└── src/
    ├── one_shot_runner.py
    └── results_collector.py
```

## Running the Benchmarks

- Submit all jobs:
  - `bash benchmarking/task_benchmarks/slurm/submit_all_qc_benchmarks.sh`
- Submit all load-data jobs:
  - `bash benchmarking/task_benchmarks/slurm/submit_all_load_benchmarks.sh`
- Submit all batch-correction jobs:
  - `bash benchmarking/task_benchmarks/slurm/submit_all_batch_correction_benchmarks.sh`
- Submit all data-adequacy jobs:
  - `bash benchmarking/task_benchmarks/slurm/submit_all_data_adequacy_benchmarks.sh`
- Submit a single script:
  - `sbatch benchmarking/task_benchmarks/slurm/one_shot_chatgpt_qc.sh`

Each SLURM script writes logs to `benchmarking/task_benchmarks/results/logs` and run outputs to the task-specific mode subdirectory (for example `benchmarking/task_benchmarks/results/load_data/one_shot` or `benchmarking/task_benchmarks/results/qc_task/one_shot`).

## Collecting Results

- Collect run summaries:
  - `python benchmarking/task_benchmarks/src/results_collector.py --results-dir benchmarking/task_benchmarks/results --output-csv benchmarking/task_benchmarks/results/summary.csv`
- Generate a grouped comparison table:
  - `python benchmarking/task_benchmarks/analysis/compare_qc_results.py --results-dir benchmarking/task_benchmarks/results`

The comparison script writes a JSON summary to `benchmarking/task_benchmarks/analysis/task_benchmark_summary.json`.

## Additional Task Prompts

Two additional prompt files are available:
- `benchmarking/task_benchmarks/prompts/doublet_prompt.txt` (doublet detection + filtering)
- `benchmarking/task_benchmarks/prompts/full_qc_prompt.txt` (end-to-end QC workflow)
- `benchmarking/task_benchmarks/prompts/batch_correction_prompt.txt` (batch effect analysis + correction)
- `benchmarking/task_benchmarks/prompts/data_adequacy_prompt.txt` (input data adequacy assessment)

To run them with the migrated scripts, pass a prompt file via `PROMPT_PATH`, for example:
  - `PROMPT_PATH=benchmarking/task_benchmarks/prompts/doublet_prompt.txt`
  - `PROMPT_PATH=benchmarking/task_benchmarks/prompts/full_qc_prompt.txt`

Standard prompts:
- `benchmarking/task_benchmarks/prompts/qc_prompt.txt`
- `benchmarking/task_benchmarks/prompts/load_prompt.txt`
