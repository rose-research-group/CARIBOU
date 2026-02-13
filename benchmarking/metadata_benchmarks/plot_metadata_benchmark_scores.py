import argparse
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
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


def _plot_runtime_bar(df: pd.DataFrame, output_path: Path) -> None:
    """Plot mean runtime by setup with error bars."""
    grouped = df.groupby("setup")["runtime_seconds"].agg(["mean", "std", "count"]).reset_index()
    grouped = grouped.sort_values("mean", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(grouped))
    ax.barh(x, grouped["mean"], xerr=grouped["std"], color="#E15759", capsize=4)
    ax.set_xlabel("Runtime (seconds)")
    ax.set_title("Metadata Benchmark: Mean Runtime by Setup")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.set_yticks(x)
    ax.set_yticklabels(grouped["setup"])
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_accuracy_vs_scale(df: pd.DataFrame, output_path: Path) -> None:
    """Plot accuracy vs dataset size to demonstrate scalability."""
    if "gt_cell_count" not in df.columns:
        print(f"Skipping {output_path.name}: missing gt_cell_count column")
        return

    df_plot = df[df["gt_cell_count"].notna() & df["score"].notna()].copy()
    if df_plot.empty:
        return

    setups = df_plot["setup"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(setups)))

    fig, ax = plt.subplots(figsize=(10, 6))

    for setup, color in zip(setups, colors):
        subset = df_plot[df_plot["setup"] == setup]
        ax.scatter(
            subset["gt_cell_count"],
            subset["score"],
            label=setup,
            alpha=0.6,
            s=80,
            color=color,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Dataset Size (cells)", fontsize=12)
    ax.set_ylabel("Accuracy Score", fontsize=12)
    ax.set_title("Accuracy vs Dataset Size: Scalability Performance", fontsize=14, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(y=0.9, color="gray", linestyle="--", alpha=0.5, label="90% threshold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_runtime_vs_scale(df: pd.DataFrame, output_path: Path) -> None:
    """Plot runtime vs dataset size to show scaling behavior."""
    if "gt_cell_count" not in df.columns or "runtime_seconds" not in df.columns:
        print(f"Skipping {output_path.name}: missing required columns")
        return

    df_plot = df[
        df["gt_cell_count"].notna() & df["runtime_seconds"].notna()
    ].copy()
    if df_plot.empty:
        return

    setups = df_plot["setup"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(setups)))

    fig, ax = plt.subplots(figsize=(10, 6))

    for setup, color in zip(setups, colors):
        subset = df_plot[df_plot["setup"] == setup]
        ax.scatter(
            subset["gt_cell_count"],
            subset["runtime_seconds"],
            label=setup,
            alpha=0.6,
            s=80,
            color=color,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Dataset Size (cells)", fontsize=12)
    ax.set_ylabel("Runtime (seconds)", fontsize=12)
    ax.set_title("Runtime vs Dataset Size: Scaling Efficiency", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_efficiency_frontier(df: pd.DataFrame, output_path: Path) -> None:
    """Plot efficiency frontier: accuracy vs runtime with dataset size as bubble size."""
    if "runtime_seconds" not in df.columns:
        print(f"Skipping {output_path.name}: missing runtime_seconds column")
        return

    df_plot = df[df["runtime_seconds"].notna() & df["score"].notna()].copy()
    if df_plot.empty:
        return

    # Aggregate by setup
    grouped = df_plot.groupby("setup").agg(
        {
            "score": "mean",
            "runtime_seconds": "mean",
            "gt_cell_count": "mean",
        }
    ).reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Normalize bubble sizes
    if "gt_cell_count" in grouped.columns and grouped["gt_cell_count"].notna().any():
        sizes = (grouped["gt_cell_count"] / grouped["gt_cell_count"].max()) * 500 + 100
    else:
        sizes = [200] * len(grouped)

    scatter = ax.scatter(
        grouped["runtime_seconds"],
        grouped["score"],
        s=sizes,
        alpha=0.6,
        c=range(len(grouped)),
        cmap="viridis",
        edgecolors="black",
        linewidths=1.5,
    )

    for idx, row in grouped.iterrows():
        ax.annotate(
            row["setup"],
            (row["runtime_seconds"], row["score"]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
        )

    ax.set_xlabel("Mean Runtime (seconds)", fontsize=12)
    ax.set_ylabel("Mean Accuracy Score", fontsize=12)
    ax.set_title("Efficiency Frontier: Accuracy vs Speed", fontsize=14, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # Add quadrant lines
    ax.axhline(y=0.9, color="green", linestyle="--", alpha=0.3, label="High accuracy (>90%)")
    ax.axvline(x=grouped["runtime_seconds"].median(), color="blue", linestyle="--", alpha=0.3, label="Median runtime")

    ax.legend(loc="lower left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_scalability_dashboard(df: pd.DataFrame, output_path: Path) -> None:
    """Create a 2x2 dashboard showing complete scalability story."""
    if "runtime_seconds" not in df.columns or "gt_cell_count" not in df.columns:
        print(f"Skipping {output_path.name}: missing required columns")
        return

    df_plot = df[
        df["runtime_seconds"].notna() & df["score"].notna() & df["gt_cell_count"].notna()
    ].copy()

    if df_plot.empty:
        return

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    setups = df_plot["setup"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(setups)))

    # Top-left: Accuracy by setup
    grouped_score = df_plot.groupby("setup")["score"].agg(["mean", "std"]).reset_index()
    grouped_score = grouped_score.sort_values("mean", ascending=False)
    x = range(len(grouped_score))
    ax1.bar(x, grouped_score["mean"], yerr=grouped_score["std"], color="#4C78A8", capsize=4)
    ax1.set_ylabel("Mean Accuracy Score", fontsize=11)
    ax1.set_title("Accuracy by Configuration", fontsize=12, fontweight="bold")
    ax1.set_ylim(0, 1.05)
    ax1.set_xticks(x)
    ax1.set_xticklabels(grouped_score["setup"], rotation=30, ha="right")
    ax1.grid(axis="y", alpha=0.3)
    ax1.axhline(y=0.9, color="gray", linestyle="--", alpha=0.5)

    # Top-right: Runtime by setup
    grouped_runtime = df_plot.groupby("setup")["runtime_seconds"].agg(["mean", "std"]).reset_index()
    grouped_runtime = grouped_runtime.sort_values("mean", ascending=True)
    x = range(len(grouped_runtime))
    ax2.barh(x, grouped_runtime["mean"], xerr=grouped_runtime["std"], color="#E15759", capsize=4)
    ax2.set_xlabel("Mean Runtime (seconds)", fontsize=11)
    ax2.set_title("Runtime by Configuration", fontsize=12, fontweight="bold")
    ax2.set_yticks(x)
    ax2.set_yticklabels(grouped_runtime["setup"])
    ax2.grid(axis="x", alpha=0.3)

    # Bottom-left: Accuracy vs Scale
    for setup, color in zip(setups, colors):
        subset = df_plot[df_plot["setup"] == setup]
        ax3.scatter(subset["gt_cell_count"], subset["score"], label=setup, alpha=0.6, s=60, color=color)
    ax3.set_xscale("log")
    ax3.set_xlabel("Dataset Size (cells)", fontsize=11)
    ax3.set_ylabel("Accuracy Score", fontsize=11)
    ax3.set_title("Accuracy vs Dataset Size", fontsize=12, fontweight="bold")
    ax3.set_ylim(-0.05, 1.05)
    ax3.axhline(y=0.9, color="gray", linestyle="--", alpha=0.5)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="best", fontsize=8, framealpha=0.9)

    # Bottom-right: Runtime vs Scale
    for setup, color in zip(setups, colors):
        subset = df_plot[df_plot["setup"] == setup]
        ax4.scatter(subset["gt_cell_count"], subset["runtime_seconds"], label=setup, alpha=0.6, s=60, color=color)
    ax4.set_xscale("log")
    ax4.set_yscale("log")
    ax4.set_xlabel("Dataset Size (cells)", fontsize=11)
    ax4.set_ylabel("Runtime (seconds)", fontsize=11)
    ax4.set_title("Runtime vs Dataset Size", fontsize=12, fontweight="bold")
    ax4.grid(True, alpha=0.3, which="both")
    ax4.legend(loc="best", fontsize=8, framealpha=0.9)

    fig.suptitle("CARIBOU Scalability Dashboard", fontsize=16, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
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

    # Plot runtime if available
    if "runtime_seconds" in df.columns and df["runtime_seconds"].notna().any():
        _plot_runtime_bar(df, output_dir / "metadata_runtime_by_setup.png")

    # Scalability plots
    _plot_accuracy_vs_scale(df, output_dir / "scalability_accuracy_vs_size.png")
    _plot_runtime_vs_scale(df, output_dir / "scalability_runtime_vs_size.png")
    _plot_efficiency_frontier(df, output_dir / "scalability_efficiency_frontier.png")
    _plot_scalability_dashboard(df, output_dir / "scalability_dashboard.png")

    print(f"Saved plots to {output_dir}")


if __name__ == "__main__":
    main()
