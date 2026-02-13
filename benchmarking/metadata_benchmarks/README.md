# Metadata Benchmarks

This directory contains scripts to prepare anonymized datasets, run agent
inference, and score results on the host.

1) Prepare anonymized datasets
   python benchmarking/metadata_benchmarks/prepare_benchmark.py --continue-on-error

   If a dataset id is not present in the default census version, the script will
   probe other Census versions automatically (up to 3 tries). Disable with:
   python benchmarking/metadata_benchmarks/prepare_benchmark.py --no-probe-versions

   Adjust the probe limit:
   python benchmarking/metadata_benchmarks/prepare_benchmark.py --max-versions 3

   To pin a specific Census version:
   python benchmarking/metadata_benchmarks/prepare_benchmark.py --census-version <version>

   Outputs:
   - benchmarking/metadata_benchmarks/benchmark_data/*_raw.h5ad
   - benchmarking/metadata_benchmarks/benchmark_data/*_blind.h5ad
   - benchmarking/metadata_benchmarks/benchmark_data/ground_truth_master.csv
   - benchmarking/metadata_benchmarks/benchmark_data/benchmark_manifest.csv

2) Run CARIBOU against anonymized datasets
   Example for one dataset:
   caribou run auto \
     --dataset benchmarking/metadata_benchmarks/benchmark_data/human_lung_blind.h5ad \
     --prompt "$(cat benchmarking/metadata_benchmarks/prompts/metadata_prompt.txt)" \
     --turns 3 \
     --output-dir benchmarking/metadata_benchmarks/results/human_lung

   Use the dataset_name (e.g., human_lung) in the output path so evaluation
   can map results to ground truth without embedding dataset ids in the h5ad.

3) Score results on host
   python benchmarking/metadata_benchmarks/evaluate_metadata_results.py \
     --results-dir benchmarking/metadata_benchmarks/results \
     --output benchmarking/metadata_benchmarks/metadata_benchmark_scores.csv

   This extracts:
   - Metadata accuracy (species, organ, cell count, transcript count)
   - Runtime metrics (from runtime.json files)
   - Overall scores per run

4) Generate plots
   python benchmarking/metadata_benchmarks/plot_metadata_benchmark_scores.py \
     --scores-csv benchmarking/metadata_benchmarks/metadata_benchmark_scores.csv

   Creates:
   - Basic comparison plots (accuracy by setup, runtime by setup, heatmaps)
   - Scalability plots:
     * Accuracy vs Dataset Size (demonstrates maintained accuracy at scale)
     * Runtime vs Dataset Size (shows scaling efficiency)
     * Efficiency Frontier (accuracy vs speed trade-offs)
     * Scalability Dashboard (2x2 comprehensive view)

5) Generate summary statistics
   python benchmarking/metadata_benchmarks/summarize_scalability_metrics.py \
     --scores-csv benchmarking/metadata_benchmarks/metadata_benchmark_scores.csv

   Produces scalability_summary.csv with:
   - Mean/std accuracy and runtime by setup
   - Throughput (datasets/hour)
   - Dataset scale range (min/max cells)
   - Task-specific accuracy breakdown

6) Slurm scripts
   Submit all metadata benchmark jobs:
   bash benchmarking/metadata_benchmarks/slurm/submit_all_metadata_benchmarks.sh

   Individual scripts:
   - benchmarking/metadata_benchmarks/slurm/single_agent_chatgpt_metadata.sh
   - benchmarking/metadata_benchmarks/slurm/single_agent_deepseek_metadata.sh
   - benchmarking/metadata_benchmarks/slurm/full_system_chatgpt_metadata.sh
   - benchmarking/metadata_benchmarks/slurm/full_system_deepseek_metadata.sh
   Full-system scripts use the dataset-specific agent system:
   caribou/src/caribou/agents/dataset_metadata_agent.json

7) Bash scripts (non-Slurm)
   Set a prompt file (required):
   export PROMPT_PATH=/path/to/metadata_prompt.txt

   Run single-agent (chatgpt):
   bash benchmarking/metadata_benchmarks/bash/single_agent_chatgpt_metadata.sh

   Run full-system (deepseek):
   bash benchmarking/metadata_benchmarks/bash/full_system_deepseek_metadata.sh

   Defaults: if PROMPT_PATH is not set, scripts use:
   - benchmarking/metadata_benchmarks/prompts/metadata_prompt.txt
   - benchmarking/metadata_benchmarks/prompts/full_system_metadata_prompt.txt

   Note: The bash scripts iterate over benchmark_manifest.csv.
   If your dataset paths differ, edit MANIFEST_PATH or the manifest file.

Notes:
- The anonymized h5ad removes obs/uns/obsm/varm/obsp and keeps only gene names.
- Mean transcript count is computed from counts layer if present, otherwise X.
- The agent must write /workspace/outputs/metadata_inference.json.
