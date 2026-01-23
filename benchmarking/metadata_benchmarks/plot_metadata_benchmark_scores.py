import argparse
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd


def _parse_run_path(run_path: str) -> Tuple[Optional[str], Optional[str]]:
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


def _plot_overall_bar(df: pd.DataFrame, output_path: Path) -> None:
    grouped = df.groupby("setup")["score"].agg(["mean", "std", "count"]).reset_index()
    grouped = grouped.sort_values("mean", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(grouped))
    ax.bar(x, grouped["mean"], yerr=grouped["std"], color="#4C78A8", capsize=4)
    ax.set_ylabel("Mean score")
    ax.set_title("Metadata Benchmark: Mean Score by Setup")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(grouped["setup"], rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_subtask_bar(df: pd.DataFrame, column: str, title: str, output_path: Path) -> None:
    if column not in df.columns:
        print(f"Skipping {output_path.name}: missing column {column}")
        return
    grouped = df.groupby("setup")[column].mean().reset_index()
    grouped = grouped.sort_values(column, ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(grouped))
    ax.bar(x, grouped[column], color="#59A14F")
    ax.set_ylabel("Match rate")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(grouped["setup"], rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_dataset_heatmap(df: pd.DataFrame, output_path: Path) -> None:
    pivot = df.pivot_table(
        index="dataset_name",
        columns="setup",
        values="score",
        aggfunc="mean",
    )
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    cax = ax.imshow(pivot.values, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_title("Metadata Benchmark: Mean Score by Dataset and Setup")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    fig.colorbar(cax, ax=ax, label="Mean score")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot metadata benchmark scores.")
    parser.add_argument(
        "--scores-csv",
        required=True,
        help="Path to metadata_benchmark_scores.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save plots (defaults to CSV directory).",
    )
    args = parser.parse_args()

    scores_path = Path(args.scores_csv).expanduser()
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else scores_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(scores_path)
    if df.empty:
        print("No records found in scores CSV.")
        return

    df = _ensure_columns(df)

    _plot_overall_bar(df, output_dir / "metadata_scores_by_setup.png")
    _plot_dataset_heatmap(df, output_dir / "metadata_scores_heatmap.png")
    _plot_subtask_bar(
        df,
        "species_match",
        "Metadata Benchmark: Species Match Rate by Setup",
        output_dir / "metadata_species_match_by_setup.png",
    )
    _plot_subtask_bar(
        df,
        "organ_match",
        "Metadata Benchmark: Organ Match Rate by Setup",
        output_dir / "metadata_organ_match_by_setup.png",
    )
    _plot_subtask_bar(
        df,
        "cell_count_match",
        "Metadata Benchmark: Cell Count Match Rate by Setup",
        output_dir / "metadata_cell_count_match_by_setup.png",
    )
    _plot_subtask_bar(
        df,
        "mean_transcript_match",
        "Metadata Benchmark: Mean Transcript Match Rate by Setup",
        output_dir / "metadata_mean_transcript_match_by_setup.png",
    )
    print(f"Saved plots to {output_dir}")


if __name__ == "__main__":
    main()
