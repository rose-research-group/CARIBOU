# CARIBOU Integration Benchmarks

Run CARIBOU on integration-focused single-cell datasets, then score outputs
against a reference pipeline (raw + processed) with standardized metrics.

## Directory layout

- `analysis/`: evaluation, aggregation, and plotting scripts
- `bash/`: local (non-SLURM) run wrapper
- `datasets/`: dataset configs and prompts
- `results/`: raw CARIBOU run outputs (per dataset and run)
- `slurm/`: SLURM launchers for runs and analysis
- `src/`: shared evaluation utilities (data loading, metrics)

## Quick start (local)

Run a single dataset/LLM/mode locally (still uses `caribou run` and the
Singularity sandbox configured in the scripts).

```bash
LLM=chatgpt MODE=full_system DATASET=aba_hippocampus \
  bash benchmarking/integration_benchmarks/bash/run_integration.sh
```

Evaluate, aggregate, and plot:

```bash
python benchmarking/integration_benchmarks/analysis/run_evaluation.py \
  --dataset aba_hippocampus
python benchmarking/integration_benchmarks/analysis/collect_results.py \
  --dataset aba_hippocampus
python benchmarking/integration_benchmarks/analysis/plot.py \
  --dataset aba_hippocampus
```

## Quick start (SLURM)

Single run:

```bash
sbatch benchmarking/integration_benchmarks/slurm/run_integration.sh \
  --dataset aba_hippocampus --llm chatgpt --mode full_system
```

All runs (dataset x LLM x mode) from the default config:

```bash
cd benchmarking/integration_benchmarks/slurm
./submit_all.sh
```

Dataset-specific arrays (example: ABA x ChatGPT):

```bash
sbatch benchmarking/integration_benchmarks/slurm/aba_hippocampus/run_chatgpt.sh
```

Run the full analysis pipeline on completed outputs:

```bash
sbatch benchmarking/integration_benchmarks/slurm/run_analysis.sh \
  --dataset aba_hippocampus
```

## Analysis pipeline

1. `analysis/run_evaluation.py` loads references, evaluates each run, and
   writes per-run metrics plus a reference baseline.
2. `analysis/collect_results.py` aggregates per-run `metrics.json` into
   `summary.json` and `summary.csv`.
3. `analysis/plot.py` generates two manuscript-quality figures from `summary.json`.

## Outputs

CARIBOU run outputs:

- `results/<dataset>/<run_name>/` (created by `slurm/run_integration.sh`)
  - `run_metadata.json`, `run.log`, and CARIBOU outputs
  - The evaluator looks for one of these h5ad names (in the run dir or
    a subdir): `annotated_dataset.h5ad`, `dataset_annotated.h5ad`,
    `data_annotated.h5ad`, `adata_final_annotated.h5ad`,
    `adata_final_complete.h5ad`, `adata_annotated.h5ad`,
    `adata_downstream_processed.h5ad`, `adata_downstream.h5ad`

Evaluation outputs:

- `analysis/outputs/<dataset>/reference_baseline.json`
- `analysis/outputs/<dataset>/<run_name>/metrics.json`
- `analysis/outputs/<dataset>/summary.json`
- `analysis/outputs/<dataset>/summary.csv`
- `analysis/plots/<dataset>/umap_comparison.png` — UMAP panels for quick visual comparison
- `analysis/plots/<dataset>/integration_comparison.png` — main result figure
- `analysis/plots/<dataset>/quality_panel.png` — supporting quality figure

## Figures

**`umap_comparison.png`** (visual overview)

One UMAP panel per successful CARIBOU run beside the ABA reference UMAP.
All panels share the same cell-type color palette (from reference labels projected
onto aligned CARIBOU cells). Panel borders are colored by execution mode.
At a glance: if cluster patterns match the reference, integration succeeded.

**`integration_comparison.png`** (main result)

Two-row panel comparing CARIBOU runs against the ABA reference baseline
(dashed line). Bars are colored by execution mode (project palette from
`dev/colors.py`). Metrics:

- Top row (batch correction): ASW Batch, Graph Connectivity, iLISI
- Bottom row (bio conservation): ASW Cell Type, cLISI

**`quality_panel.png`** (supporting)

Three-panel quality summary:

- Panel A: Gene expression fidelity (Spearman r of mean log-expression vs reference)
- Panel B: Embedding fidelity (PCA kNN overlap vs reference)
- Panel C: QC filtering rate

All figures are saved as 300 dpi PNG + SVG.

## Metrics

- QC: `qc_filtering_rate`, `median_pct_mt`, `median_genes_per_cell`
- Expression: `gene_expr_spearman_r` — Spearman r of per-gene mean log-expression
- Embedding: `pca_knn_overlap` — fraction of k-NN shared with reference PCA embedding
- scib batch correction: `car_asw_batch`, `car_graph_connectivity`, `car_ilisi`
- scib bio conservation: `car_asw_celltype`, `car_clisi`

Bio-conservation metrics use **reference cell type labels** projected onto aligned
CARIBOU cells, not CARIBOU's own labels. This is the principled approach:
it asks whether CARIBOU's embedding preserves ground-truth ABA cell structure.

iLISI and cLISI use a pure-Python implementation that reads the pre-computed
scanpy kNN graph directly, bypassing scib's C++ binary (which requires GLIBC ≥ 2.34).
Values are scaled to [0, 1] following scib's convention: higher is better for both.

The reference baseline is computed once per dataset using the processed reference
embedding defined in each dataset config.

## Dataset configuration

Each dataset lives in `datasets/<id>/` and must include:

- `config.json` with:
  - `input_dataset_path`
  - `reference_path` and `processed_reference_path`
  - `reference_celltype_key` (column with ground-truth cell type labels)
  - `barcode_join` and `metadata_join` (if needed for alignment)
  - `scib` config (optional, keys: `caribou_batch_key`, `caribou_embedding_key`,
    `reference_batch_key`, `reference_embedding_key`)
  - `slurm_mem` and `slurm_time` (optional, used by `submit_all.sh`)
- `prompt.txt` for the CARIBOU task instructions

Minimal `config.json` skeleton:

```json
{
  "id": "my_dataset",
  "input_dataset_path": "/path/to/input.h5ad",
  "reference_path": "/path/to/reference_raw.h5ad",
  "processed_reference_path": "/path/to/reference_processed.h5ad",
  "reference_celltype_key": "cell_type",
  "barcode_join": {
    "caribou_strip_suffix": true,
    "caribou_sample_col": "sample",
    "reference_barcode_col": "cell_barcode",
    "reference_library_col": "library_label"
  }
}
```

## Notes

- The SLURM scripts use absolute paths (see `slurm/run_integration.sh` and
  `slurm/run_analysis.sh`). Update them if the repo location or Python
  environment changes.
- `slurm/run_analysis.sh` uses a fixed Python path; adjust `PYTHON` if you
  run in a different environment.
