#!/usr/bin/env python3
"""
Generate scalability summary statistics for metadata benchmarks.
"""
import argparse
from pathlib import Path

import pandas as pd


def _parse_run_path(run_path: str):
    parts = Path(run_path).parts
    if "metadata_task" in parts:
        idx = parts.index("metadata_task")
        if len(parts) > idx + 2:
            mode = parts[idx + 1]
            dataset = parts[idx + 2]
            llm = None
            if len(parts) > idx + 3:
                llm = parts[idx + 3].split("_", 1)[0]
            return mode, llm or "unknown"
    return None, None


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    modes = []
    llms = []
    for path in df["run_path"].astype(str):
        mode, llm = _parse_run_path(path)
        modes.append(mode or "unknown")
        llms.append(llm or "unknown")
    df = df.copy()
    df["mode"] = modes
    df["llm"] = llms
    df["setup"] = df["mode"] + "/" + df["llm"]
    return df


def generate_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Generate scalability summary statistics."""
    # Ensure setup column exists
    if "setup" not in df.columns:
        df = _ensure_columns(df)

    summary_data = []

    for setup in df["setup"].unique():
        subset = df[df["setup"] == setup]

        # Basic stats
        n_runs = len(subset)
        mean_score = subset["score"].mean() if "score" in subset.columns else None
        std_score = subset["score"].std() if "score" in subset.columns else None

        # Runtime stats
        mean_runtime = None
        std_runtime = None
        throughput = None
        if "runtime_seconds" in subset.columns and subset["runtime_seconds"].notna().any():
            mean_runtime = subset["runtime_seconds"].mean()
            std_runtime = subset["runtime_seconds"].std()
            throughput = 3600 / mean_runtime if mean_runtime > 0 else None

        # Dataset scale range
        min_cells = None
        max_cells = None
        if "gt_cell_count" in subset.columns and subset["gt_cell_count"].notna().any():
            min_cells = int(subset["gt_cell_count"].min())
            max_cells = int(subset["gt_cell_count"].max())

        # Task accuracy breakdown
        species_acc = None
        organ_acc = None
        if "species_match" in subset.columns:
            species_acc = subset["species_match"].mean()
        if "organ_match" in subset.columns:
            organ_acc = subset["organ_match"].mean()

        summary_data.append(
            {
                "Setup": setup,
                "Runs": n_runs,
                "Mean Score": f"{mean_score:.3f}" if mean_score is not None else "N/A",
                "Std Score": f"{std_score:.3f}" if std_score is not None else "N/A",
                "Mean Runtime (s)": f"{mean_runtime:.1f}" if mean_runtime is not None else "N/A",
                "Std Runtime (s)": f"{std_runtime:.1f}" if std_runtime is not None else "N/A",
                "Throughput (datasets/hr)": f"{throughput:.1f}" if throughput is not None else "N/A",
                "Min Cells": f"{min_cells:,}" if min_cells is not None else "N/A",
                "Max Cells": f"{max_cells:,}" if max_cells is not None else "N/A",
                "Species Accuracy": f"{species_acc:.3f}" if species_acc is not None else "N/A",
                "Organ Accuracy": f"{organ_acc:.3f}" if organ_acc is not None else "N/A",
            }
        )

    summary_df = pd.DataFrame(summary_data)
    # Sort by mean score descending
    if "Mean Score" in summary_df.columns:
        summary_df["_sort_score"] = pd.to_numeric(
            summary_df["Mean Score"].replace("N/A", "0"), errors="coerce"
        )
        summary_df = summary_df.sort_values("_sort_score", ascending=False)
        summary_df = summary_df.drop(columns=["_sort_score"])

    return summary_df


def main():
    parser = argparse.ArgumentParser(
        description="Generate scalability summary statistics."
    )
    parser.add_argument(
        "--scores-csv",
        required=True,
        help="Path to metadata_benchmark_scores.csv.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (defaults to scalability_summary.csv in same dir).",
    )
    args = parser.parse_args()

    scores_path = Path(args.scores_csv).expanduser()
    output_path = (
        Path(args.output).expanduser()
        if args.output
        else scores_path.parent / "scalability_summary.csv"
    )

    df = pd.read_csv(scores_path)
    if df.empty:
        print("No records found in scores CSV.")
        return

    summary = generate_summary(df)

    # Save to CSV
    summary.to_csv(output_path, index=False)
    print(f"\nScalability Summary Statistics saved to: {output_path}\n")

    # Print to console
    print(summary.to_string(index=False))

    # Additional insights
    print("\n" + "=" * 80)
    print("KEY SCALABILITY INSIGHTS:")
    print("=" * 80)

    if "runtime_seconds" in df.columns and df["runtime_seconds"].notna().any():
        min_runtime = df["runtime_seconds"].min()
        max_runtime = df["runtime_seconds"].max()
        print(f"Runtime range: {min_runtime:.1f}s - {max_runtime:.1f}s")

    if "gt_cell_count" in df.columns and df["gt_cell_count"].notna().any():
        min_cells = int(df["gt_cell_count"].min())
        max_cells = int(df["gt_cell_count"].max())
        scale_factor = max_cells / min_cells if min_cells > 0 else 0
        print(f"Dataset scale range: {min_cells:,} - {max_cells:,} cells ({scale_factor:.1f}x)")

    if "score" in df.columns:
        high_acc = (df["score"] >= 0.9).sum()
        total = len(df)
        print(f"High accuracy (≥90%): {high_acc}/{total} runs ({100*high_acc/total:.1f}%)")


if __name__ == "__main__":
    main()
