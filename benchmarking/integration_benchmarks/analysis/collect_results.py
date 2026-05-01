#!/usr/bin/env python3
"""
Aggregate per-run metrics into summary files.

Walks outputs/{dataset}/*/metrics.json, merges the reference baseline
into each row, and writes summary.json + summary.csv.

Usage:
    python collect_results.py --dataset aba_hippocampus
    python collect_results.py                          # all datasets
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

INTBENCH_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(INTBENCH_DIR))

from src.data_loader import available_datasets

ANALYSIS_DIR = INTBENCH_DIR / "analysis"


def collect_dataset(dataset_id: str, output_dir: Path) -> list:
    ds_dir = output_dir / dataset_id
    if not ds_dir.exists():
        raise FileNotFoundError(
            f"No outputs for dataset '{dataset_id}': {ds_dir}\n"
            f"Run run_evaluation.py first."
        )

    # Load baseline once
    baseline_path = ds_dir / "reference_baseline.json"
    baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else {}

    records = []
    for metrics_path in sorted(ds_dir.glob("*/metrics.json")):
        rec = json.loads(metrics_path.read_text())
        # Ensure baseline fields are present (prefix 'baseline_')
        for k, v in baseline.items():
            bk = f"baseline_{k}"
            if bk not in rec:
                rec[bk] = v
        records.append(rec)

    if not records:
        raise RuntimeError(
            f"No metrics.json files found under {ds_dir}. "
            f"Run run_evaluation.py first."
        )

    print(f"  {dataset_id}: {len(records)} run(s) collected")
    return records


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate integration benchmark metrics into summary files"
    )
    parser.add_argument(
        "--dataset", nargs="+", default=None,
        help="Dataset IDs to aggregate (default: all available)"
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ANALYSIS_DIR / "outputs",
        help="Directory containing per-run outputs (and where summary files are written)"
    )
    args = parser.parse_args()

    datasets = args.dataset or available_datasets()
    if not datasets:
        raise RuntimeError(f"No datasets found in {INTBENCH_DIR / 'datasets'}")

    print(f"Collecting results for datasets: {datasets}")

    all_records = []
    for ds in datasets:
        records = collect_dataset(ds, args.output_dir)
        ds_dir = args.output_dir / ds
        (ds_dir / "summary.json").write_text(json.dumps(records, indent=2))
        pd.DataFrame(records).to_csv(ds_dir / "summary.csv", index=False)
        print(f"  Saved → {(ds_dir / 'summary.json').relative_to(INTBENCH_DIR)}")
        all_records.extend(records)

    if len(datasets) > 1 and all_records:
        (args.output_dir / "all_summary.json").write_text(json.dumps(all_records, indent=2))
        pd.DataFrame(all_records).to_csv(args.output_dir / "all_summary.csv", index=False)
        print(f"  Saved → {(args.output_dir / 'all_summary.json').relative_to(INTBENCH_DIR)}")

    print(f"\n✓  Collection complete. Run plot.py to generate figures.")


if __name__ == "__main__":
    main()
