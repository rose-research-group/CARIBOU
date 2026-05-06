# CARIBOU Cell-Typing Benchmarks

Compare CARIBOU's cell-type annotation performance against expert-curated reference datasets.
Ported from `dev/comparisons` for manuscript-ready figures under the shared `benchmarking/` tree.

## Datasets

| ID | Organism | Tissue | Reference | Cells |
|----|----------|--------|-----------|-------|
| `aba_hippocampus` | Mouse | Hippocampus | Allen Brain Atlas (Yao et al., Nature 2023) | ~86k |
| `tsp_large_intestine` | Human | Large intestine | Tabula Sapiens (Science 2022) | ~30k |

## Quick start

### Run CARIBOU (SLURM)

Single run:
```bash
sbatch benchmarking/celltyping_benchmarks/slurm/run_celltyping.sh \
  --dataset aba_hippocampus --llm chatgpt --mode full_system
```

All combinations (2 datasets × 2 LLMs × 4 modes):
```bash
cd benchmarking/celltyping_benchmarks/slurm/
./submit_all.sh
# or with overrides:
./submit_all.sh --datasets "aba_hippocampus" --llms "chatgpt deepseek" --modes "full_system_no_mem single_agent"
```

Per-dataset, per-LLM array jobs:
```bash
sbatch benchmarking/celltyping_benchmarks/slurm/aba_hippocampus/run_chatgpt.sh
sbatch benchmarking/celltyping_benchmarks/slurm/tsp_large_intestine/run_deepseek.sh
```

### Evaluate and plot (SLURM)

```bash
sbatch benchmarking/celltyping_benchmarks/slurm/aba_hippocampus/run_analysis.sh
sbatch benchmarking/celltyping_benchmarks/slurm/aba_hippocampus/run_plot.sh
```

### Evaluate and plot (local)

```bash
PYTHON="${PYTHON:-python}"
cd benchmarking/celltyping_benchmarks/analysis/

$PYTHON evaluate.py --dataset aba_hippocampus
$PYTHON plot.py --dataset aba_hippocampus

# Clean publication plots (no claude, full_system excluded):
cd ../clean_plots/
$PYTHON make_clean_plots.py --dataset aba_hippocampus
```

## Analysis pipeline

```
results/{dataset}/{llm}_{mode}_{job_id}/
    annotated_dataset.h5ad       ← CARIBOU output
    run_metadata.json            ← runtime info
    run.log

        ↓  analysis/evaluate.py
analysis/outputs/{dataset}/
    results.json / results.csv   ← per-run metrics
    confusion_{run}.json         ← per-run confusion matrices
    all_results.json / .csv      ← combined across datasets

        ↓  analysis/plot.py
analysis/plots/{dataset}/
    completeness_heatmap.{png,svg}
    metrics_overview.{png,svg}
    celltyping_summary.{png,svg}
    celltyping_*_heatmap.{png,svg}
    per_type_f1.{png,svg}
    confusion_*.png
    gene_expression_fidelity.{png,svg}
    population_metrics.{png,svg}
    quality_vs_speed.{png,svg}
    radar_quality.{png,svg}
    dashboard.{png,svg}

        ↓  clean_plots/make_clean_plots.py
clean_plots/{dataset}/
    weighted_f1_heatmap.png      ← manuscript figure
    scib_metrics_panel.png       ← manuscript figure
    marker_jaccard_heatmap.png   ← manuscript figure
    marker_gene_recovery.png     ← manuscript figure
    runtime_panel.png            ← manuscript figure
```

## Metrics

| Metric | Description |
|--------|-------------|
| `ari` | Adjusted Rand Index (clustering agreement) |
| `nmi` | Normalized Mutual Information |
| `macro_f1` / `weighted_f1` | Per-cell-type F1 averaged across types |
| `gene_expr_spearman_r` | Spearman r of per-gene mean log-expression vs reference |
| `hvg_jaccard` | Jaccard overlap of highly variable genes |
| `umap_knn_overlap` / `pca_knn_overlap` | Fraction of k-NN shared with reference embeddings |
| `qc_filtering_rate` | Fraction of input cells removed by QC |
| `celltype_name_overlap_coarse` | Fraction of coarse reference types recovered by name |
| `celltype_prop_corr_coarse` | Pearson r of cell-type proportions (coarse mapping) |

Cell-type labels are harmonised to a shared coarse vocabulary via `coarse_celltype_mapping`
(reference fine → coarse) and `caribou_celltype_mapping` (CARIBOU output → same coarse vocab),
both defined per-dataset in `datasets/{id}/config.json`.

## Execution modes

| Mode | Turns | Description |
|------|-------|-------------|
| `one_shot` | 1 | Single prompt, no iteration |
| `single_agent` | 20 | Solo agent, compressed memory |
| `full_system` | 30 | Multi-agent with memory reports |
| `full_system_no_mem` | 30 | Multi-agent without memory reporting |

## Adding a new dataset

1. Create `datasets/{id}/config.json` (see existing configs for schema)
2. Create `datasets/{id}/prompt.txt`
3. Add `{id}` to `DATASETS` in `slurm/submit_all.sh`
4. Optionally create `slurm/{id}/` array scripts for per-dataset submission

## Notes

- SLURM scripts derive the CARIBOU root from the script location or Git checkout.
- Set `PYTHON=/path/to/python` before running analysis scripts if the default `python` is not the desired environment.
- Source of truth for this module is `dev/comparisons` — this is a read-only port.
  Do not delete or modify anything in `dev/comparisons`.
