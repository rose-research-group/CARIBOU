import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


META_BENCH_DIR = Path(__file__).resolve().parent
CARIBOU_ROOT = META_BENCH_DIR.parent.parent
sys.path.insert(0, str(CARIBOU_ROOT / "dev"))

CIVIDIS_CMAP = plt.get_cmap("cividis")

SETUP_ORDER = [
    "full_system/chatgpt",
    "full_system/deepseek",
    "single_agent/chatgpt",
    "single_agent/deepseek",
]
MODE_COLORS = {
    "full_system": CIVIDIS_CMAP(0.78),
    "single_agent": CIVIDIS_CMAP(0.38),
}
LLM_COLORS = {
    "chatgpt": CIVIDIS_CMAP(0.22),
    "deepseek": CIVIDIS_CMAP(0.92),
}
LLM_MARKERS = {
    "chatgpt": "o",
    "deepseek": "s",
}
SETUP_LABELS = {
    "full_system/chatgpt": "Full system\nChatGPT",
    "full_system/deepseek": "Full system\nDeepSeek",
    "single_agent/chatgpt": "Single agent\nChatGPT",
    "single_agent/deepseek": "Single agent\nDeepSeek",
}

plt.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
})


def _savefig(fig: plt.Figure, output_path: Path, **kwargs) -> None:
    fig.savefig(output_path, **kwargs)
    svg_kwargs = {k: v for k, v in kwargs.items() if k != "dpi"}
    fig.savefig(output_path.with_suffix(".svg"), **svg_kwargs)


def _parse_run_path(run_path: str) -> Tuple[Optional[str], Optional[str]]:
    parts = Path(run_path).parts
    if "metadata_task" in parts:
        idx = parts.index("metadata_task")
        if len(parts) > idx + 2:
            mode = parts[idx + 1]
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


def _setup_sort_key(setup: str) -> Tuple[int, str]:
    return (SETUP_ORDER.index(setup), setup) if setup in SETUP_ORDER else (len(SETUP_ORDER), setup)


def _ordered_setups(values) -> list[str]:
    return sorted(pd.Series(values).dropna().unique(), key=_setup_sort_key)


def _setup_label(setup: str) -> str:
    return SETUP_LABELS.get(setup, str(setup).replace("/", "\n"))


def _short_setup_label(setup: str) -> str:
    return {
        "full_system/chatgpt": "FS · ChatGPT",
        "full_system/deepseek": "FS · DeepSeek",
        "single_agent/chatgpt": "SA · ChatGPT",
        "single_agent/deepseek": "SA · DeepSeek",
    }.get(setup, str(setup).replace("/", " · "))


def _mode_from_setup(setup: str) -> str:
    return str(setup).split("/", 1)[0]


def _llm_from_setup(setup: str) -> str:
    return str(setup).split("/", 1)[1] if "/" in str(setup) else "unknown"


def _mode_color(setup: str) -> str:
    return MODE_COLORS.get(_mode_from_setup(setup), CIVIDIS_CMAP(0.5))


def _llm_color(setup: str) -> str:
    return LLM_COLORS.get(_llm_from_setup(setup), CIVIDIS_CMAP(0.5))


def _llm_marker(setup: str) -> str:
    return LLM_MARKERS.get(_llm_from_setup(setup), "o")


def _style_axes(ax) -> None:
    ax.spines["left"].set_color("#bbbbbb")
    ax.spines["bottom"].set_color("#bbbbbb")


def _setup_legends(ax, mode_anchor=(1.02, 1.0), llm_anchor=(1.02, 0.72)) -> None:
    mode_handles = [
        Line2D([0], [0], marker="s", linestyle="None", markersize=9,
               markerfacecolor=MODE_COLORS["full_system"], markeredgecolor=MODE_COLORS["full_system"], label="Full system"),
        Line2D([0], [0], marker="s", linestyle="None", markersize=9,
               markerfacecolor=MODE_COLORS["single_agent"], markeredgecolor=MODE_COLORS["single_agent"], label="Single agent"),
    ]
    llm_handles = [
        Line2D([0], [0], marker="o", linestyle="None", markersize=8,
               markerfacecolor="white", markeredgecolor="black", label="ChatGPT"),
        Line2D([0], [0], marker="s", linestyle="None", markersize=8,
               markerfacecolor="white", markeredgecolor="black", label="DeepSeek"),
    ]
    leg1 = ax.legend(handles=mode_handles, title="Mode", loc="upper left", bbox_to_anchor=mode_anchor)
    ax.add_artist(leg1)
    ax.legend(handles=llm_handles, title="LLM", loc="upper left", bbox_to_anchor=llm_anchor)


def _add_bar_markers(ax, x_positions, values, setups) -> None:
    for x, value, setup in zip(x_positions, values, setups):
        ax.scatter(
            x,
            min(float(value) + 0.03, 1.03),
            s=50,
            color=_llm_color(setup),
            marker=_llm_marker(setup),
            edgecolors="white",
            linewidths=0.8,
            zorder=4,
        )


def _plot_overall_bar(df: pd.DataFrame, output_path: Path) -> None:
    grouped = df.groupby("setup")["score"].agg(["mean", "std", "count"]).reset_index()
    grouped = grouped.sort_values("setup", key=lambda s: s.map(_setup_sort_key))

    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    x = np.arange(len(grouped))
    ax.bar(
        x,
        grouped["mean"],
        yerr=grouped["std"],
        color=[_mode_color(setup) for setup in grouped["setup"]],
        capsize=4,
        width=0.72,
    )
    _add_bar_markers(ax, x, grouped["mean"], grouped["setup"])
    ax.set_ylabel("Mean score")
    ax.set_title("Metadata benchmark: mean score by setup")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.22)
    ax.set_xticks(x)
    ax.set_xticklabels([_setup_label(s) for s in grouped["setup"]])
    _style_axes(ax)
    fig.tight_layout()
    _savefig(fig, output_path, dpi=150)
    plt.close(fig)


def _plot_subtask_bar(df: pd.DataFrame, column: str, title: str, output_path: Path) -> None:
    if column not in df.columns:
        print(f"Skipping {output_path.name}: missing column {column}")
        return

    grouped = df.groupby("setup")[column].mean().reset_index()
    grouped = grouped.sort_values("setup", key=lambda s: s.map(_setup_sort_key))

    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    x = np.arange(len(grouped))
    ax.bar(
        x,
        grouped[column],
        color=[_mode_color(setup) for setup in grouped["setup"]],
        width=0.72,
    )
    _add_bar_markers(ax, x, grouped[column], grouped["setup"])
    ax.set_ylabel("Match rate")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.22)
    ax.set_xticks(x)
    ax.set_xticklabels([_setup_label(s) for s in grouped["setup"]])
    _style_axes(ax)
    fig.tight_layout()
    _savefig(fig, output_path, dpi=150)
    plt.close(fig)


def _plot_dataset_heatmap(df: pd.DataFrame, output_path: Path) -> None:
    pivot = df.pivot_table(
        index="dataset_name",
        columns="setup",
        values="score",
        aggfunc="mean",
    )
    pivot = pivot.reindex(columns=[s for s in SETUP_ORDER if s in pivot.columns])
    if pivot.empty:
        return

    cmap = CIVIDIS_CMAP.copy()
    cmap.set_bad("#f2f2f2")
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    cax = ax.imshow(pivot.values, aspect="auto", vmin=0, vmax=1, cmap=cmap)
    ax.set_title("Mean metadata score by dataset and setup")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([_setup_label(s) for s in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(name).replace("_", " ") for name in pivot.index])

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8.5, color="black")

    fig.colorbar(cax, ax=ax, label="Mean score", fraction=0.046, pad=0.04)
    _style_axes(ax)
    fig.tight_layout()
    _savefig(fig, output_path, dpi=150)
    plt.close(fig)


def _plot_runtime_bar(df: pd.DataFrame, output_path: Path) -> None:
    grouped = df.groupby("setup")["runtime_seconds"].agg(["mean", "std", "count"]).reset_index()
    grouped = grouped.sort_values("setup", key=lambda s: s.map(_setup_sort_key))

    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    y = np.arange(len(grouped))
    ax.barh(
        y,
        grouped["mean"],
        xerr=grouped["std"],
        color=[_mode_color(setup) for setup in grouped["setup"]],
        capsize=4,
    )
    ax.set_xlabel("Runtime (seconds)")
    ax.set_title("Metadata benchmark: mean runtime by setup")
    ax.grid(axis="x", linestyle="--", alpha=0.22)
    ax.set_yticks(y)
    ax.set_yticklabels([_setup_label(s) for s in grouped["setup"]])
    _style_axes(ax)
    fig.tight_layout()
    _savefig(fig, output_path, dpi=150)
    plt.close(fig)


def _plot_accuracy_vs_scale(df: pd.DataFrame, output_path: Path) -> None:
    if "gt_cell_count" not in df.columns:
        print(f"Skipping {output_path.name}: missing gt_cell_count column")
        return

    df_plot = df[df["gt_cell_count"].notna() & df["score"].notna()].copy()
    if df_plot.empty:
        return

    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    for setup in _ordered_setups(df_plot["setup"]):
        subset = df_plot[df_plot["setup"] == setup]
        ax.scatter(
            subset["gt_cell_count"],
            subset["score"],
            alpha=0.72,
            s=68,
            color=_mode_color(setup),
            marker=_llm_marker(setup),
            edgecolors="white",
            linewidths=0.6,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Dataset size (cells)", fontsize=12)
    ax.set_ylabel("Accuracy score", fontsize=12)
    ax.set_title("Accuracy vs dataset size", fontsize=13, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(y=0.9, color="#9e9e9e", linestyle="--", alpha=0.6)
    ax.grid(True, alpha=0.22)
    _setup_legends(ax)
    _style_axes(ax)
    fig.tight_layout()
    _savefig(fig, output_path, dpi=150)
    plt.close(fig)


def _plot_runtime_vs_scale(df: pd.DataFrame, output_path: Path) -> None:
    if "gt_cell_count" not in df.columns or "runtime_seconds" not in df.columns:
        print(f"Skipping {output_path.name}: missing required columns")
        return

    df_plot = df[df["gt_cell_count"].notna() & df["runtime_seconds"].notna()].copy()
    if df_plot.empty:
        return

    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    for setup in _ordered_setups(df_plot["setup"]):
        subset = df_plot[df_plot["setup"] == setup]
        ax.scatter(
            subset["gt_cell_count"],
            subset["runtime_seconds"],
            alpha=0.72,
            s=68,
            color=_mode_color(setup),
            marker=_llm_marker(setup),
            edgecolors="white",
            linewidths=0.6,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Dataset size (cells)", fontsize=12)
    ax.set_ylabel("Runtime (seconds)", fontsize=12)
    ax.set_title("Runtime vs dataset size", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.22, which="both")
    _setup_legends(ax)
    _style_axes(ax)
    fig.tight_layout()
    _savefig(fig, output_path, dpi=150)
    plt.close(fig)


def _plot_efficiency_frontier(df: pd.DataFrame, output_path: Path) -> None:
    if "runtime_seconds" not in df.columns:
        print(f"Skipping {output_path.name}: missing runtime_seconds column")
        return

    df_plot = df[df["runtime_seconds"].notna() & df["score"].notna()].copy()
    if df_plot.empty:
        return

    grouped = df_plot.groupby("setup").agg(
        score=("score", "mean"),
        runtime_seconds=("runtime_seconds", "mean"),
        gt_cell_count=("gt_cell_count", "mean"),
    ).reset_index()
    grouped = grouped.sort_values("setup", key=lambda s: s.map(_setup_sort_key))

    if grouped["gt_cell_count"].notna().any():
        sizes = (grouped["gt_cell_count"] / grouped["gt_cell_count"].max()) * 420 + 120
    else:
        sizes = np.full(len(grouped), 180)

    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    for (_, row), size in zip(grouped.iterrows(), sizes):
        ax.scatter(
            row["runtime_seconds"],
            row["score"],
            s=size,
            alpha=0.72,
            color=_mode_color(row["setup"]),
            marker=_llm_marker(row["setup"]),
            edgecolors="black",
            linewidths=1.0,
        )
        ax.annotate(
            row["setup"].replace("/", " · "),
            (row["runtime_seconds"], row["score"]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.8),
        )

    ax.set_xlabel("Mean runtime (seconds)", fontsize=12)
    ax.set_ylabel("Mean accuracy score", fontsize=12)
    ax.set_title("Efficiency frontier: accuracy vs speed", fontsize=13, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.22)
    ax.axhline(y=0.9, color="#9e9e9e", linestyle="--", alpha=0.6)
    ax.axvline(x=grouped["runtime_seconds"].median(), color="#9e9e9e", linestyle="--", alpha=0.6)
    _setup_legends(ax)
    _style_axes(ax)
    fig.tight_layout()
    _savefig(fig, output_path, dpi=150)
    plt.close(fig)


def _plot_scalability_dashboard(df: pd.DataFrame, output_path: Path) -> None:
    if "runtime_seconds" not in df.columns or "gt_cell_count" not in df.columns:
        print(f"Skipping {output_path.name}: missing required columns")
        return

    df_plot = df[
        df["runtime_seconds"].notna() & df["score"].notna() & df["gt_cell_count"].notna()
    ].copy()
    if df_plot.empty:
        return

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 11))

    grouped_score = df_plot.groupby("setup")["score"].agg(["mean", "std"]).reset_index()
    grouped_score = grouped_score.sort_values("setup", key=lambda s: s.map(_setup_sort_key))
    x = np.arange(len(grouped_score))
    ax1.bar(
        x,
        grouped_score["mean"],
        yerr=grouped_score["std"],
        color=[_mode_color(s) for s in grouped_score["setup"]],
        capsize=4,
        width=0.72,
    )
    _add_bar_markers(ax1, x, grouped_score["mean"], grouped_score["setup"])
    ax1.set_ylabel("Mean accuracy score", fontsize=11)
    ax1.set_title("Accuracy by configuration", fontsize=12, fontweight="bold")
    ax1.set_ylim(0, 1.05)
    ax1.set_xticks(x)
    ax1.set_xticklabels([_setup_label(s) for s in grouped_score["setup"]])
    ax1.grid(axis="y", alpha=0.22)
    ax1.axhline(y=0.9, color="#9e9e9e", linestyle="--", alpha=0.6)
    _style_axes(ax1)

    grouped_runtime = df_plot.groupby("setup")["runtime_seconds"].agg(["mean", "std"]).reset_index()
    grouped_runtime = grouped_runtime.sort_values("setup", key=lambda s: s.map(_setup_sort_key))
    y = np.arange(len(grouped_runtime))
    ax2.barh(
        y,
        grouped_runtime["mean"],
        xerr=grouped_runtime["std"],
        color=[_mode_color(s) for s in grouped_runtime["setup"]],
        capsize=4,
    )
    ax2.set_xlabel("Mean runtime (seconds)", fontsize=11)
    ax2.set_title("Runtime by configuration", fontsize=12, fontweight="bold")
    ax2.set_yticks(y)
    ax2.set_yticklabels([_setup_label(s) for s in grouped_runtime["setup"]])
    ax2.grid(axis="x", alpha=0.22)
    _style_axes(ax2)

    for setup in _ordered_setups(df_plot["setup"]):
        subset = df_plot[df_plot["setup"] == setup]
        ax3.scatter(
            subset["gt_cell_count"],
            subset["score"],
            alpha=0.72,
            s=58,
            color=_mode_color(setup),
            marker=_llm_marker(setup),
            edgecolors="white",
            linewidths=0.6,
        )
    ax3.set_xscale("log")
    ax3.set_xlabel("Dataset size (cells)", fontsize=11)
    ax3.set_ylabel("Accuracy score", fontsize=11)
    ax3.set_title("Accuracy vs dataset size", fontsize=12, fontweight="bold")
    ax3.set_ylim(-0.05, 1.05)
    ax3.axhline(y=0.9, color="#9e9e9e", linestyle="--", alpha=0.6)
    ax3.grid(True, alpha=0.22)
    _style_axes(ax3)

    for setup in _ordered_setups(df_plot["setup"]):
        subset = df_plot[df_plot["setup"] == setup]
        ax4.scatter(
            subset["gt_cell_count"],
            subset["runtime_seconds"],
            alpha=0.72,
            s=58,
            color=_mode_color(setup),
            marker=_llm_marker(setup),
            edgecolors="white",
            linewidths=0.6,
        )
    ax4.set_xscale("log")
    ax4.set_yscale("log")
    ax4.set_xlabel("Dataset size (cells)", fontsize=11)
    ax4.set_ylabel("Runtime (seconds)", fontsize=11)
    ax4.set_title("Runtime vs dataset size", fontsize=12, fontweight="bold")
    ax4.grid(True, alpha=0.22, which="both")
    _setup_legends(ax4, mode_anchor=(1.02, 1.0), llm_anchor=(1.02, 0.72))
    _style_axes(ax4)

    fig.suptitle("CARIBOU scalability dashboard", fontsize=15, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    _savefig(fig, output_path, dpi=150)
    plt.close(fig)


def _plot_descriptive_summary(df: pd.DataFrame, output_path: Path) -> None:
    if "runtime_seconds" not in df.columns or "gt_cell_count" not in df.columns:
        print(f"Skipping {output_path.name}: missing required columns")
        return

    df_plot = df[
        df["runtime_seconds"].notna() & df["score"].notna() & df["gt_cell_count"].notna()
    ].copy()
    if df_plot.empty:
        return

    pivot = df_plot.pivot_table(
        index="dataset_name",
        columns="setup",
        values="score",
        aggfunc="mean",
    )
    pivot = pivot.reindex(columns=[s for s in SETUP_ORDER if s in pivot.columns])

    grouped = df_plot.groupby("setup").agg(
        mean_score=("score", "mean"),
        score_std=("score", "std"),
    ).reset_index()
    grouped = grouped.sort_values("setup", key=lambda s: s.map(_setup_sort_key))

    fig = plt.figure(figsize=(13.6, 6.2))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.75, 1.0],
        height_ratios=[1.0, 1.12],
        wspace=0.28,
        hspace=0.35,
    )
    ax_heat = fig.add_subplot(gs[:, 0])
    ax_score = fig.add_subplot(gs[0, 1])
    ax_scale = fig.add_subplot(gs[1, 1])

    cmap = CIVIDIS_CMAP.copy()
    cmap.set_bad("#f2f2f2")
    im = ax_heat.imshow(pivot.values, aspect="auto", vmin=0, vmax=1, cmap=cmap)
    ax_heat.set_title("Per-dataset metadata score", fontsize=12, fontweight="bold", pad=8)
    ax_heat.set_xticks(range(len(pivot.columns)))
    ax_heat.set_xticklabels([_setup_label(s) for s in pivot.columns], fontsize=9)
    ax_heat.set_yticks(range(len(pivot.index)))
    ax_heat.set_yticklabels([str(name).replace("_", " ") for name in pivot.index], fontsize=10)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            if pd.notna(val):
                ax_heat.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8.5, color="black")
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
    cbar.set_label("Mean score", rotation=270, labelpad=14)
    _style_axes(ax_heat)

    y = np.arange(len(grouped))
    ax_score.hlines(
        y,
        grouped["mean_score"] - grouped["score_std"],
        grouped["mean_score"] + grouped["score_std"],
        color="#9e9e9e",
        linewidth=1.4,
        zorder=1,
    )
    for yy, (_, row) in zip(y, grouped.iterrows()):
        ax_score.scatter(
            row["mean_score"],
            yy,
            s=90,
            color=_mode_color(row["setup"]),
            marker=_llm_marker(row["setup"]),
            edgecolors="black",
            linewidths=0.8,
            zorder=3,
        )
    ax_score.axvline(0.9, color="#9e9e9e", linestyle="--", linewidth=1.0, alpha=0.6)
    ax_score.set_xlim(0.7, 1.02)
    ax_score.set_yticks(y)
    ax_score.set_yticklabels([_short_setup_label(s) for s in grouped["setup"]], fontsize=9)
    ax_score.set_xlabel("Mean score")
    ax_score.set_title("Overall accuracy", fontsize=12, fontweight="bold", pad=6)
    ax_score.grid(axis="x", alpha=0.22)
    _style_axes(ax_score)

    for setup in _ordered_setups(df_plot["setup"]):
        subset = df_plot[df_plot["setup"] == setup]
        ax_scale.scatter(
            subset["gt_cell_count"],
            subset["runtime_seconds"],
            s=60,
            alpha=0.72,
            color=_mode_color(setup),
            marker=_llm_marker(setup),
            edgecolors="white",
            linewidths=0.6,
        )
    ax_scale.set_xscale("log")
    ax_scale.set_yscale("log")
    ax_scale.set_xlabel("Dataset size (cells)")
    ax_scale.set_ylabel("Runtime (seconds)")
    ax_scale.set_title("Scaling cost", fontsize=12, fontweight="bold", pad=6)
    ax_scale.grid(True, which="both", alpha=0.22)
    _setup_legends(ax_scale, mode_anchor=(1.02, 1.0), llm_anchor=(1.02, 0.72))
    _style_axes(ax_scale)

    fig.suptitle("Metadata benchmark summary", fontsize=15, fontweight="bold", y=0.99)
    fig.subplots_adjust(left=0.06, right=0.88, top=0.86, bottom=0.11)
    _savefig(fig, output_path, dpi=180, bbox_inches="tight")
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
        "Metadata benchmark: species match rate by setup",
        output_dir / "metadata_species_match_by_setup.png",
    )
    _plot_subtask_bar(
        df,
        "organ_match",
        "Metadata benchmark: organ match rate by setup",
        output_dir / "metadata_organ_match_by_setup.png",
    )
    _plot_subtask_bar(
        df,
        "cell_count_match",
        "Metadata benchmark: cell count match rate by setup",
        output_dir / "metadata_cell_count_match_by_setup.png",
    )
    _plot_subtask_bar(
        df,
        "mean_transcript_match",
        "Metadata benchmark: mean transcript match rate by setup",
        output_dir / "metadata_mean_transcript_match_by_setup.png",
    )

    if "runtime_seconds" in df.columns and df["runtime_seconds"].notna().any():
        _plot_runtime_bar(df, output_dir / "metadata_runtime_by_setup.png")

    _plot_accuracy_vs_scale(df, output_dir / "scalability_accuracy_vs_size.png")
    _plot_runtime_vs_scale(df, output_dir / "scalability_runtime_vs_size.png")
    _plot_efficiency_frontier(df, output_dir / "scalability_efficiency_frontier.png")
    _plot_scalability_dashboard(df, output_dir / "scalability_dashboard.png")
    _plot_descriptive_summary(df, output_dir / "metadata_summary_panel.png")

    print(f"Saved plots to {output_dir}")


if __name__ == "__main__":
    main()
