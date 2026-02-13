# Task Benchmarks Analysis SLURM Scripts

This directory contains SLURM job scripts for analyzing and visualizing task benchmark results.

## Scripts

### Full Analysis (All LLMs)

**`submit_task_analysis.sh`**
- Submits the full analysis job including all LLM backends (ChatGPT, Claude, DeepSeek)
- Wrapper script that calls `evaluate_and_plot_task_benchmarks.sh`

**`evaluate_and_plot_task_benchmarks.sh`**
- Main SLURM job script for complete analysis pipeline
- **Steps:**
  1. Collects raw results from all benchmark runs
  2. Aggregates results into summary statistics
  3. Generates visualization plots
- **Outputs:**
  - `results/summary.json` - Raw collected results
  - `results/summary.csv` - CSV format of raw results
  - `analysis/task_benchmark_summary.json` - Aggregated summary
  - `analysis/plots/` - Visualization plots

### Analysis Without Claude

**`submit_task_analysis_no_claude.sh`**
- Submits analysis job excluding Claude results
- Useful when Claude benchmarks are incomplete or have errors
- Wrapper script that calls `evaluate_and_plot_task_benchmarks_no_claude.sh`

**`evaluate_and_plot_task_benchmarks_no_claude.sh`**
- Analysis pipeline that filters out Claude LLM backend
- **Steps:**
  1. Collects raw results from all benchmark runs
  2. Filters out Claude results
  3. Aggregates remaining results (ChatGPT & DeepSeek)
  4. Generates visualization plots
- **Outputs:**
  - `results/summary_no_claude_raw.json` - Filtered raw results
  - `results/summary_no_claude.csv` - CSV format of filtered results
  - `analysis/task_benchmark_summary_no_claude.json` - Aggregated summary
  - `analysis/plots_no_claude/` - Visualization plots

## Usage

### Run Full Analysis
```bash
cd /path/to/CARIBOU/benchmarking/task_benchmarks/analysis_slurm
./submit_task_analysis.sh
```

### Run Analysis Without Claude
```bash
cd /path/to/CARIBOU/benchmarking/task_benchmarks/analysis_slurm
./submit_task_analysis_no_claude.sh
```

### Monitor Jobs
```bash
squeue -u $USER
```

### Check Logs
Logs are written to `analysis_slurm/logs/`:
- `evaluate_plot_task_benchmarks_<JOB_ID>.log` - Full analysis log
- `evaluate_plot_task_benchmarks_no_claude_<JOB_ID>.log` - No-Claude analysis log

## Configuration

Environment variables can be set to customize output locations:

### Full Analysis
- `RESULTS_DIR` - Base results directory (default: `benchmarking/task_benchmarks/results`)
- `ANALYSIS_DIR` - Analysis output directory (default: `benchmarking/task_benchmarks/analysis`)
- `PLOTS_DIR` - Plot output directory (default: `analysis/plots`)
- `SUMMARY_JSON` - Summary JSON path (default: `analysis/task_benchmark_summary.json`)
- `SUMMARY_CSV` - Summary CSV path (default: `results/summary.csv`)
- `SUMMARY_RAW_JSON` - Raw results JSON path (default: `results/summary.json`)

### No-Claude Analysis
Same variables as above, with different defaults:
- `PLOTS_DIR` - Default: `analysis/plots_no_claude`
- `SUMMARY_JSON` - Default: `analysis/task_benchmark_summary_no_claude.json`
- `SUMMARY_CSV` - Default: `results/summary_no_claude.csv`
- `SUMMARY_RAW_JSON` - Default: `results/summary_no_claude_raw.json`

## Example with Custom Paths
```bash
export RESULTS_DIR=/custom/results/path
export PLOTS_DIR=/custom/plots/path
./submit_task_analysis.sh
```

## SLURM Configuration

Both analysis scripts use:
- **Job name:** `task_eval_plot` or `task_eval_plot_no_claude`
- **CPUs:** 2
- **Memory:** 8GB
- **Time limit:** 30 minutes
- **Partition:** peerd

## Dependencies

The analysis pipeline requires:
- Python 3.8+
- matplotlib
- numpy
- anndata (optional, for h5ad metrics)
