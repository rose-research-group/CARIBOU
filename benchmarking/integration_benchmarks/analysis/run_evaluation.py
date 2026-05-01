#!/usr/bin/env python3
"""
Evaluate CARIBOU integration benchmark runs against the ABA reference.

Loads raw reference + processed reference once per dataset, then iterates
over all run directories, computes metrics, and writes per-run outputs.
Also computes the reference baseline (scib metrics on the ABA-processed
embedding) once and saves it separately.

Usage:
    python run_evaluation.py --dataset aba_hippocampus
    python run_evaluation.py --dataset aba_hippocampus --run chatgpt_full_system_99999
    python run_evaluation.py --dataset aba_hippocampus --output-dir /custom/path
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Allow running as a script from the analysis/ directory
INTBENCH_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(INTBENCH_DIR))

from src.data_loader import (
    load_dataset_config,
    available_datasets,
    load_raw_reference,
    load_processed_reference,
)
from src.evaluate import evaluate_run, compute_reference_baseline

RESULTS_DIR  = INTBENCH_DIR / "results"
ANALYSIS_DIR = INTBENCH_DIR / "analysis"


def evaluate_dataset(dataset_id: str, output_dir: Path, run_filter: str | None = None):
    print(f"\n{'='*60}")
    print(f"  Dataset: {dataset_id}")
    print(f"{'='*60}")

    cfg = load_dataset_config(dataset_id)
    dataset_results_dir = RESULTS_DIR / dataset_id

    if not dataset_results_dir.exists():
        raise FileNotFoundError(
            f"No results directory for dataset '{dataset_id}': {dataset_results_dir}\n"
            f"Run CARIBOU integration jobs first (see slurm/run_integration.sh)."
        )

    run_dirs = sorted(d for d in dataset_results_dir.iterdir() if d.is_dir())
    if run_filter:
        run_dirs = [d for d in run_dirs if d.name == run_filter]
        if not run_dirs:
            raise ValueError(
                f"Run '{run_filter}' not found in {dataset_results_dir}. "
                f"Available: {[d.name for d in sorted(dataset_results_dir.iterdir()) if d.is_dir()]}"
            )

    print(f"  Found {len(run_dirs)} run(s) to evaluate")

    # ── Load reference data (done once per dataset) ───────────────────────────
    print("\n  Loading raw reference …")
    raw_ref = load_raw_reference(cfg)

    print("\n  Loading processed reference …")
    proc_ref = load_processed_reference(cfg)

    out = output_dir / dataset_id
    out.mkdir(parents=True, exist_ok=True)

    # ── Reference baseline (scib on ABA-processed embedding) ─────────────────
    baseline_path = out / "reference_baseline.json"
    if baseline_path.exists():
        print(f"\n  Reference baseline already computed: {baseline_path}")
        baseline = json.loads(baseline_path.read_text())
    else:
        print("\n  Computing reference baseline …")
        baseline = compute_reference_baseline(cfg, proc_ref)
        baseline_path.write_text(json.dumps(baseline, indent=2))
        print(f"  Saved → {baseline_path.relative_to(INTBENCH_DIR)}")

    # ── Save reference UMAP for visualization (once, subsampled) ─────────────
    ref_umap_path = out / "reference_umap.npz"
    if not ref_umap_path.exists() and "X_umap" in proc_ref.obsm:
        ref_ct_key = cfg["reference_celltype_key"]
        n_ref = proc_ref.n_obs
        max_plot = 50_000
        if n_ref > max_plot:
            rng = np.random.default_rng(0)
            idx = rng.choice(n_ref, max_plot, replace=False)
            ref_plot = proc_ref[idx]
        else:
            ref_plot = proc_ref
        labels = (
            ref_plot.obs[ref_ct_key].astype(str).values
            if ref_ct_key in ref_plot.obs.columns
            else np.full(len(ref_plot), "")
        )
        np.savez(ref_umap_path, coords=ref_plot.obsm["X_umap"], labels=labels)
        print(f"  Reference UMAP saved ({len(ref_plot):,} cells) → {ref_umap_path.relative_to(INTBENCH_DIR)}")

    # ── Per-run evaluation ────────────────────────────────────────────────────
    print(f"\n  Evaluating {len(run_dirs)} run(s) …\n")
    results = []
    for run_dir in run_dirs:
        print(f"  → {run_dir.name}")
        r = evaluate_run(run_dir, cfg, raw_ref, proc_ref, out)
        # Merge reference baseline into every row for easy comparison
        r.update({f"baseline_{k}": v for k, v in baseline.items()})
        results.append(r)
        status = "OK" if r.get("success") else f"FAILED ({r.get('error','')})"
        print(f"    {status}")
        if r.get("success"):
            for m in ("ari", "nmi", "macro_f1", "gene_expr_spearman_r",
                      "car_asw_batch", "car_asw_celltype", "pca_knn_overlap"):
                if m in r and r[m] is not None:
                    print(f"    {m}: {r[m]:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate CARIBOU integration benchmark runs"
    )
    parser.add_argument(
        "--dataset", nargs="+", default=None,
        help="Dataset IDs to evaluate (default: all available)"
    )
    parser.add_argument(
        "--run", default=None,
        help="Evaluate a single named run directory (optional)"
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ANALYSIS_DIR / "outputs",
        help="Directory to write per-run metrics and baselines"
    )
    args = parser.parse_args()

    datasets = args.dataset or available_datasets()
    if not datasets:
        raise RuntimeError(f"No datasets found in {INTBENCH_DIR / 'datasets'}")

    if args.run and len(datasets) > 1:
        raise ValueError("--run can only be used with a single --dataset")

    print(f"Evaluating datasets: {datasets}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for ds in datasets:
        evaluate_dataset(ds, args.output_dir, run_filter=args.run)

    print(f"\n✓  Evaluation complete. Run collect_results.py to aggregate.")


if __name__ == "__main__":
    main()
