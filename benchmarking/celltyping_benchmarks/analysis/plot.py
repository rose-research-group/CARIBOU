#!/usr/bin/env python3
"""
Universal CARIBOU comparison plotter.

Reads all_results.json (or per-dataset results.json) from the analysis/outputs/
directory and generates publication-quality figures.

Usage:
    python plot.py                                       # all results
    python plot.py --dataset aba_hippocampus             # one dataset
    python plot.py --results-json outputs/all_results.json
"""

import argparse
import json
from pathlib import Path
from math import pi

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import seaborn as sns

# Custom colormap: white → plasma warm end (pink → orange → yellow).
_plasma_base = plt.get_cmap("plasma")
WHITE_PLASMA = LinearSegmentedColormap.from_list(
    "white_plasma",
    [(1, 1, 1, 1)] + [_plasma_base(x) for x in np.linspace(0.35, 1.0, 255)],
)

COMP_DIR   = Path(__file__).parent.parent
OUTPUT_DIR = Path(__file__).parent / "outputs"
PLOTS_DIR  = Path(__file__).parent / "plots"


# ---------------------------------------------------------------------------
# Shared constants and helpers
# ---------------------------------------------------------------------------

METRIC_LABELS = {
    "ari":                   "Adjusted Rand Index",
    "nmi":                   "Normalized Mutual Info",
    "macro_f1":              "Macro F1 (cell type)",
    "weighted_f1":           "Weighted F1 (cell type)",
    "hvg_jaccard":           "HVG Jaccard",
    "gene_expr_spearman_r":  "Gene Expr. Spearman r",
    "umap_knn_overlap":      "UMAP kNN Overlap",
    "pca_knn_overlap":       "PCA kNN Overlap",
    "qc_filtering_rate":     "QC Filtering Rate",
    "runtime_s":             "Runtime (seconds)",
    "cell_count_ratio":      "Cell Count Ratio",
    "celltype_name_overlap_coarse": "Cell-Type Recall (coarse)",
    "celltype_prop_corr_coarse":    "Proportion Corr. (coarse)",
}

MODE_ORDER  = ["one_shot", "single_agent", "full_system", "full_system_no_mem"]
MODE_COLORS = {
    "one_shot":          "#E15759",
    "single_agent":      "#F28E2B",
    "full_system":       "#4E79A7",
    "full_system_no_mem":"#76B7B2",
}
LLM_COLORS = {"chatgpt": "#4E79A7", "deepseek": "#F28E2B", "claude": "#59A14F"}


def _save(fig, path: Path, dpi: int = 300):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    svg_path = path.with_suffix(".svg")
    if svg_path != path:
        fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.relative_to(COMP_DIR)}")
    if svg_path != path:
        print(f"  Saved: {svg_path.relative_to(COMP_DIR)}")


def _success_df(df: pd.DataFrame) -> pd.DataFrame:
    return df[df.get("success", True) == True].copy() if "success" in df.columns else df.copy()


def _mode_color(mode):
    return MODE_COLORS.get(mode, "#aaa")


def _short_name(run_name: str) -> str:
    """Shorten a run name for axis labels."""
    parts = run_name.split("_")
    # Keep last numeric token as the job ID suffix
    job_id = parts[-1] if parts[-1].isdigit() else ""
    # Find mode-like tokens
    mode_tokens = [p for p in parts if p in ("one", "single", "full", "shot", "agent", "system", "mem")]
    label = "_".join(mode_tokens) + (f"\n{job_id}" if job_id else "")
    return label or run_name


def _metric_available(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns and df[col].dropna().size > 0


def plot_celltyping_summary(df: pd.DataFrame, dataset_id: str, out: Path):
    """Summary boxplots for cell typing metrics across runs."""
    metrics = [
        ("macro_f1", "Macro F1"),
        ("weighted_f1", "Weighted F1"),
        ("macro_precision", "Macro Precision"),
        ("macro_recall", "Macro Recall"),
        ("celltype_name_overlap", "Celltype Name Overlap"),
        ("celltype_prop_corr", "Celltype Prop Corr"),
        ("celltype_name_overlap_coarse", "Celltype Name Overlap (coarse)"),
        ("celltype_prop_corr_coarse", "Celltype Prop Corr (coarse)"),
    ]
    available = [(c, label) for c, label in metrics if _metric_available(df, c)]
    if not available:
        return

    plot_df = df.copy()
    has_llm = "llm" in plot_df.columns and plot_df["llm"].notna().any()
    n = len(available)
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3.6 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax, (col, label) in zip(axes, available):
        sns.boxplot(
            data=plot_df,
            x="mode",
            y=col,
            hue="llm" if has_llm else None,
            order=[m for m in MODE_ORDER if m in plot_df["mode"].unique()],
            palette=LLM_COLORS if has_llm else None,
            ax=ax,
        )
        ax.set_title(label, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=20)
        if has_llm:
            ax.legend(title="LLM", fontsize=8, title_fontsize=9, loc="best")
        else:
            ax.get_legend().remove() if ax.get_legend() else None

    for ax in axes[len(available):]:
        ax.axis("off")

    fig.suptitle(f"{dataset_id}: Cell Typing Summary", fontsize=13, fontweight="bold")
    _save(fig, out / "celltyping_summary.png")


def plot_celltyping_heatmaps(df: pd.DataFrame, dataset_id: str, out: Path):
    """Heatmaps of mean cell typing metrics by mode × LLM."""
    if "mode" not in df.columns or "llm" not in df.columns:
        return

    metrics = [
        ("macro_f1", "Macro F1"),
        ("weighted_f1", "Weighted F1"),
        ("celltype_name_overlap_coarse", "Celltype Name Overlap (coarse)"),
    ]
    for col, label in metrics:
        if not _metric_available(df, col):
            continue
        pivot = (
            df.pivot_table(index="mode", columns="llm", values=col, aggfunc="mean")
            .reindex(index=MODE_ORDER)
        )
        if pivot.shape[0] < 1 or pivot.shape[1] < 1:
            continue
        fig, ax = plt.subplots(figsize=(4 + 1.2 * pivot.shape[1], 2.5 + 0.6 * pivot.shape[0]))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap=WHITE_PLASMA, linewidths=0.5, ax=ax)
        ax.set_title(f"{dataset_id}: {label} (mean)")
        ax.set_xlabel("LLM")
        ax.set_ylabel("Mode")
        _save(fig, out / f"celltyping_{col}_heatmap.png")


# ---------------------------------------------------------------------------
# Figure 1 – Pipeline completeness (step-by-step heatmap)
# ---------------------------------------------------------------------------

def plot_completeness_heatmap(df: pd.DataFrame, dataset_id: str, out: Path):
    """Heatmap: which pipeline output keys are present per run."""
    key_cols = [c for c in df.columns if c.startswith("has_")]
    if not key_cols:
        return

    heat = df[["run_name"] + key_cols].set_index("run_name")
    heat = heat.dropna(how="all")
    if heat.empty:
        return
    heat.columns = [c.replace("has_", "").replace("_", " ").title() for c in heat.columns]
    heat = heat.infer_objects(copy=False).fillna(0).astype(float)

    fig, ax = plt.subplots(figsize=(max(6, len(heat.columns) * 1.4), max(4, len(heat) * 0.5 + 1)))
    sns.heatmap(heat, annot=True, fmt=".0f", cmap="RdYlGn", vmin=0, vmax=1,
                linewidths=0.5, linecolor="#ddd",
                cbar_kws={"label": "Present (1) / Missing (0)"}, ax=ax)
    ax.set_title(f"{dataset_id}: Pipeline Output Completeness", fontweight="bold", pad=12)
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=30, labelsize=9)
    _save(fig, out / "completeness_heatmap.png")


# ---------------------------------------------------------------------------
# Figure 2 – Quality metrics overview (ARI, NMI, HVG, runtime)
# ---------------------------------------------------------------------------

def plot_metrics_overview(df: pd.DataFrame, dataset_id: str, out: Path):
    """Grid of box + strip plots for core quantitative metrics."""
    candidate = ["ari", "nmi", "macro_f1", "weighted_f1",
                 "hvg_jaccard", "gene_expr_spearman_r", "runtime_s"]
    metrics = [m for m in candidate if m in df.columns and df[m].notna().any()]
    if not metrics:
        return

    ncols = 3
    nrows = (len(metrics) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    llm_vals = df["llm"].dropna().unique() if "llm" in df.columns else []
    llm_palette = {llm: LLM_COLORS.get(llm, "#aaa") for llm in llm_vals}

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        plot_df = df.dropna(subset=[metric]).copy()
        if plot_df.empty:
            ax.set_visible(False); continue

        order = [m for m in MODE_ORDER if m in plot_df.get("mode", pd.Series()).values]
        if not order:
            ax.set_visible(False); continue

        sns.boxplot(data=plot_df, x="mode", y=metric, order=order,
                    hue="mode", palette={m: _mode_color(m) for m in order},
                    legend=False, ax=ax, width=0.5, fliersize=0)
        if "llm" in plot_df.columns:
            sns.stripplot(data=plot_df, x="mode", y=metric, order=order,
                          hue="llm", palette=llm_palette, ax=ax,
                          size=7, alpha=0.85, dodge=True, jitter=True)

        label = METRIC_LABELS.get(metric, metric)
        ax.set_title(label, fontweight="bold", fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel(label, fontsize=9)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.tick_params(axis="x", rotation=20, labelsize=8)

        if metric in ("ari", "nmi", "macro_f1", "weighted_f1", "hvg_jaccard",
                      "gene_expr_spearman_r", "umap_knn_overlap", "pca_knn_overlap"):
            ax.axhline(0.6, color="orange", ls="--", lw=1, alpha=0.6)
            ax.axhline(0.8, color="green",  ls="--", lw=1, alpha=0.6)

        legend = ax.get_legend()
        if legend and idx > 0:
            legend.remove()

    for idx in range(len(metrics), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f"{dataset_id}: Quality Metrics by Mode", fontsize=13, fontweight="bold")
    _save(fig, out / "metrics_overview.png")


# ---------------------------------------------------------------------------
# Figure 3 – Population-level metrics
# ---------------------------------------------------------------------------

def plot_population_metrics(df: pd.DataFrame, dataset_id: str, out: Path):
    """Horizontal bar charts for population-level similarity metrics."""
    pop_cols = [c for c in (
        "celltype_name_overlap", "celltype_name_overlap_coarse",
        "celltype_prop_corr",    "celltype_prop_corr_coarse",
        "cell_count_ratio",
    ) if c in df.columns and df[c].notna().any()]
    if not pop_cols:
        return

    plot_df = df.dropna(subset=pop_cols, how="all").copy()
    if plot_df.empty:
        return

    labels = {
        "celltype_name_overlap":        "Cell-Type Recall\n(name overlap / ref types)",
        "celltype_name_overlap_coarse": "Cell-Type Recall (coarse)\n(mapped groups / ref groups)",
        "celltype_prop_corr":           "Proportion Correlation\n(Pearson r, exact names)",
        "celltype_prop_corr_coarse":    "Proportion Correlation (coarse)\n(ontology-mapped)",
        "cell_count_ratio":             "Cell Count Ratio\n(CARIBOU output / reference)",
    }

    n = len(pop_cols)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, max(4, len(plot_df) * 0.45 + 1.5)))
    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, pop_cols):
        sub = plot_df.dropna(subset=[col])
        if sub.empty:
            ax.set_visible(False); continue

        colors = [_mode_color(m) for m in sub.get("mode", pd.Series([""] * len(sub)))]
        bars = ax.barh(sub["run_name"], sub[col], color=colors, edgecolor="white", linewidth=0.5)
        ax.set_xlabel(labels.get(col, col), fontweight="bold", fontsize=9)
        ax.set_title(labels.get(col, col).split("\n")[0], fontweight="bold", fontsize=10)
        ax.grid(axis="x", alpha=0.3, linestyle="--")
        ax.tick_params(axis="y", labelsize=8)

        if col in ("celltype_name_overlap", "celltype_name_overlap_coarse",
                   "celltype_prop_corr",    "celltype_prop_corr_coarse"):
            ax.axvline(1.0, color="green",  ls="--", lw=1, alpha=0.7)
            ax.axvline(0.6, color="orange", ls="--", lw=1, alpha=0.7)
            lo = min(0, float(sub[col].min()) - 0.05)
            ax.set_xlim(left=lo, right=max(1.05, float(sub[col].max()) + 0.05))
        elif col == "cell_count_ratio":
            ax.axvline(1.0, color="green", ls="--", lw=1, alpha=0.7, label="1:1 ratio")

        for bar, val in zip(bars, sub[col]):
            ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}", va="center", fontsize=8)

    handles = [mpatches.Patch(color=c, label=m) for m, c in MODE_COLORS.items()]
    fig.legend(handles=handles, title="Mode", bbox_to_anchor=(1.01, 0.5),
               loc="center left", fontsize=8)
    fig.suptitle(f"{dataset_id}: Population-Level Comparison", fontsize=13, fontweight="bold")
    _save(fig, out / "population_metrics.png")


# ---------------------------------------------------------------------------
# Figure 4 – Confusion matrix per run
# ---------------------------------------------------------------------------

def plot_confusion_matrix(cm_path: Path, dataset_id: str, run_name: str, out: Path):
    """Row-normalised confusion heatmap: reference cell type vs CARIBOU prediction."""
    try:
        cm_data = json.loads(cm_path.read_text())
    except Exception:
        return

    labels = cm_data.get("labels", [])
    cm     = np.array(cm_data.get("confusion_matrix", []), dtype=float)
    if cm.size == 0 or len(labels) == 0:
        return

    # Drop zero-support reference rows so the y-axis reflects reference labels only.
    per_type = cm_data.get("per_type", {})
    row_keep = [
        i for i, label in enumerate(labels)
        if per_type.get(label, {}).get("support", 0) > 0
    ]
    if row_keep:
        cm = cm[row_keep, :]
        row_labels = [labels[i] for i in row_keep]
    else:
        row_labels = labels

    # Row-normalise (each row = fraction of reference-type cells assigned to each CARIBOU type)
    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        cm_norm = np.where(row_sums > 0, cm / row_sums, 0.0)

    n_rows = len(row_labels)
    n_cols = len(labels)
    cell_size = max(0.7, min(1.2, 12 / max(n_rows, n_cols)))
    fig, ax = plt.subplots(figsize=(n_cols * cell_size + 2, n_rows * cell_size + 1.5))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap=WHITE_PLASMA,
                xticklabels=labels, yticklabels=row_labels,
                vmin=0, vmax=1, linewidths=0.3, linecolor="#eee",
                cbar_kws={"label": "Fraction of reference cells"}, ax=ax)
    ax.set_xlabel("CARIBOU predicted cell type", fontweight="bold")
    ax.set_ylabel("Reference cell type", fontweight="bold")
    ax.set_title(f"{dataset_id} / {run_name}\nCell-Type Confusion (row = recall)",
                 fontweight="bold", pad=10)
    ax.tick_params(axis="x", rotation=40, labelsize=8)
    ax.tick_params(axis="y", rotation=0,  labelsize=8)
    _save(fig, out / f"confusion_{run_name}.png")


# ---------------------------------------------------------------------------
# Figure 5 – Per-cell-type F1 across runs
# ---------------------------------------------------------------------------

def plot_per_type_f1(output_dir: Path, dataset_id: str, out: Path):
    """Horizontal bar chart: F1 per cell type, one bar per run, grouped by type."""
    cm_files = sorted(output_dir.glob(f"confusion_*.json"))
    if not cm_files:
        return

    rows = []
    for cm_path in cm_files:
        run_name = cm_path.stem.replace("confusion_", "")
        try:
            cm_data = json.loads(cm_path.read_text())
        except Exception:
            continue
        for ct, vals in cm_data.get("per_type", {}).items():
            rows.append({
                "run_name":  run_name,
                "cell_type": ct,
                "f1":        vals["f1"],
                "precision": vals["precision"],
                "recall":    vals["recall"],
                "support":   vals["support"],
            })

    if not rows:
        return

    df = pd.DataFrame(rows)
    # Only show types with at least some support
    df = df[df["support"] > 0]
    if df.empty:
        return

    cell_types = sorted(df["cell_type"].unique())
    runs = sorted(df["run_name"].unique())
    n_ct, n_r = len(cell_types), len(runs)

    fig, ax = plt.subplots(figsize=(10, max(4, n_ct * 0.55 + 1.5)))
    bar_h = 0.8 / max(n_r, 1)
    y_pos = np.arange(n_ct)

    for i, run in enumerate(runs):
        sub = df[df["run_name"] == run].set_index("cell_type")
        f1s = [float(sub.loc[ct, "f1"]) if ct in sub.index else 0.0 for ct in cell_types]
        offset = (i - n_r / 2 + 0.5) * bar_h
        bars = ax.barh(y_pos + offset, f1s, height=bar_h * 0.9,
                       label=run, alpha=0.85)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(cell_types, fontsize=9)
    ax.set_xlabel("F1 Score", fontweight="bold")
    ax.set_title(f"{dataset_id}: Per-Cell-Type F1", fontweight="bold")
    ax.axvline(0.6, color="orange", ls="--", lw=1, alpha=0.7, label="F1 = 0.6")
    ax.axvline(0.8, color="green",  ls="--", lw=1, alpha=0.7, label="F1 = 0.8")
    ax.set_xlim(0, 1.05)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    fig.suptitle(f"{dataset_id}: Cell-Type Annotation Accuracy", fontsize=12, fontweight="bold")
    _save(fig, out / "per_type_f1.png")


# ---------------------------------------------------------------------------
# Figure 6 – Gene expression scatter
# ---------------------------------------------------------------------------

def plot_gene_scatter(df: pd.DataFrame, dataset_id: str, out: Path):
    """Placeholder scatter reference mean vs CARIBOU mean per gene.

    The actual per-gene data is not stored in results.json (too large);
    this function plots the Spearman r summary as a bar chart instead.
    """
    col = "gene_expr_spearman_r"
    if col not in df.columns or df[col].dropna().empty:
        return

    sub = df.dropna(subset=[col]).copy()
    sub = sub.sort_values(col, ascending=True)

    fig, ax = plt.subplots(figsize=(8, max(3, len(sub) * 0.45 + 1)))
    colors = [_mode_color(m) for m in sub.get("mode", pd.Series([""] * len(sub)))]
    bars = ax.barh(sub["run_name"], sub[col], color=colors)
    ax.axvline(0.9, color="green",  ls="--", lw=1.2, alpha=0.8, label="r = 0.9 (excellent)")
    ax.axvline(0.7, color="orange", ls="--", lw=1.2, alpha=0.8, label="r = 0.7 (acceptable)")
    ax.set_xlabel("Spearman r  (per-gene mean log-expression: CARIBOU vs reference)",
                  fontweight="bold")
    ax.set_title(f"{dataset_id}: Gene Expression Fidelity", fontweight="bold")
    ax.set_xlim(0, 1.05)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.legend(fontsize=9)

    for bar, val in zip(bars, sub[col]):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)

    handles = [mpatches.Patch(color=c, label=m) for m, c in MODE_COLORS.items()
               if m in sub.get("mode", pd.Series()).values]
    ax.legend(handles=handles + [
        mpatches.Patch(color="none", label=""),
        plt.Line2D([0],[0], color="green",  ls="--", label="r = 0.9 (excellent)"),
        plt.Line2D([0],[0], color="orange", ls="--", label="r = 0.7 (acceptable)"),
    ], bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)

    _save(fig, out / "gene_expression_fidelity.png")


# ---------------------------------------------------------------------------
# Figure 7 – Radar / spider chart (summary across all metrics)
# ---------------------------------------------------------------------------

def plot_radar(df: pd.DataFrame, dataset_id: str, out: Path):
    """Radar chart showing 5 key quality dimensions per run.

    Dimensions (all normalised to 0–1):
      1. QC filtering rate    (capped at 0.9, higher = more filtered = closer to ref)
      2. HVG Jaccard          (direct)
      3. Gene expression r    (direct)
      4. Weighted F1          (direct, uses coarse mapping)
      5. UMAP kNN overlap     (direct)
    """
    DIMS = [
        ("qc_filtering_rate",    "QC\nFiltering"),
        ("hvg_jaccard",          "HVG\nJaccard"),
        ("gene_expr_spearman_r", "Gene Expr\nCorr."),
        ("weighted_f1",          "Cell-Type\nF1"),
        ("umap_knn_overlap",     "Embedding\nOverlap"),
    ]

    present = [(col, lbl) for col, lbl in DIMS if col in df.columns and df[col].notna().any()]
    if len(present) < 3:
        return

    ok = _success_df(df).dropna(subset=[col for col, _ in present], how="all")
    if ok.empty:
        return

    cols   = [col for col, _ in present]
    labels = [lbl for _, lbl in present]
    N = len(present)
    angles = [n / N * 2 * pi for n in range(N)] + [0]  # close the polygon

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7, color="grey")
    ax.grid(color="grey", alpha=0.3)

    cmap = plt.get_cmap("tab10")
    for i, (_, row) in enumerate(ok.iterrows()):
        vals = [min(1.0, max(0.0, float(row[c]) if pd.notna(row[c]) else 0.0)) for c in cols]
        vals_closed = vals + [vals[0]]
        color = cmap(i % 10)
        ax.plot(angles, vals_closed, color=color, linewidth=2, alpha=0.85)
        ax.fill(angles, vals_closed, color=color, alpha=0.08)

    # Legend
    handles = [mpatches.Patch(color=cmap(i % 10), label=row["run_name"])
               for i, (_, row) in enumerate(ok.iterrows())]
    ax.legend(handles=handles, bbox_to_anchor=(1.3, 1.1), loc="upper left", fontsize=8)
    ax.set_title(f"{dataset_id}: Multi-Dimensional Quality\n", fontweight="bold", pad=20)
    _save(fig, out / "radar_quality.png")


# ---------------------------------------------------------------------------
# Figure 8 – Runtime vs quality scatter
# ---------------------------------------------------------------------------

def plot_runtime_vs_quality(df: pd.DataFrame, dataset_id: str, out: Path):
    """Scatter: runtime vs best available quality metric, coloured by mode."""
    quality_col = next((c for c in ("weighted_f1", "macro_f1", "ari")
                        if c in df.columns and df[c].notna().any()), None)
    if quality_col is None or "runtime_s" not in df.columns:
        return

    plot_df = df.dropna(subset=[quality_col, "runtime_s"])
    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    for mode in MODE_ORDER:
        sub = plot_df[plot_df["mode"] == mode] if "mode" in plot_df.columns else pd.DataFrame()
        if sub.empty: continue
        ax.scatter(sub["runtime_s"], sub[quality_col],
                   label=mode, color=_mode_color(mode),
                   s=80, alpha=0.8, edgecolors="black", linewidth=0.5, zorder=3)
        for _, row in sub.iterrows():
            ax.annotate(row.get("llm", ""), (row["runtime_s"], row[quality_col]),
                        textcoords="offset points", xytext=(5, 3), fontsize=7, alpha=0.8)

    ax.axhline(0.6, color="orange", ls="--", lw=1, alpha=0.7, label="0.6 threshold")
    ax.axhline(0.8, color="green",  ls="--", lw=1, alpha=0.7, label="0.8 threshold")
    ax.set_xlabel("Runtime (seconds)", fontweight="bold")
    ax.set_ylabel(METRIC_LABELS.get(quality_col, quality_col), fontweight="bold")
    ax.set_title(f"{dataset_id}: Quality vs Speed", fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3, linestyle="--")
    _save(fig, out / "quality_vs_speed.png")


# ---------------------------------------------------------------------------
# Figure 9 – Dashboard (success rate + ARI bar + runtime)
# ---------------------------------------------------------------------------

def plot_dataset_dashboard(df: pd.DataFrame, dataset_id: str, out: Path):
    """3-panel summary dashboard."""
    ok = _success_df(df)
    if ok.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"{dataset_id}: Summary Dashboard", fontsize=13, fontweight="bold")

    # Panel 1: success rate by mode
    ax = axes[0]
    sr = df.groupby("mode").apply(
        lambda x: (x.get("success", True) == True).mean() * 100,
        include_groups=False,
    )
    order = [m for m in MODE_ORDER if m in sr.index]
    order += [m for m in sorted(sr.index) if m not in MODE_ORDER]
    ax.bar(order, [sr[m] for m in order],
           color=[_mode_color(m) for m in order], edgecolor="white")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Success Rate by Mode", fontweight="bold")
    ax.set_ylim(0, 108)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.tick_params(axis="x", rotation=25, labelsize=8)

    # Panel 2: best quality metric by mode × LLM
    ax = axes[1]
    quality_col = next((c for c in ("weighted_f1", "macro_f1", "ari")
                        if c in ok.columns and ok[c].notna().any()), None)
    if quality_col and "mode" in ok.columns and "llm" in ok.columns:
        pivot = ok.pivot_table(values=quality_col, index="mode", columns="llm", aggfunc="mean")
        order_idx = [m for m in MODE_ORDER if m in pivot.index]
        order_idx += [m for m in pivot.index if m not in MODE_ORDER]
        reindexed = pivot.reindex(index=order_idx)
        if not reindexed.empty and reindexed.notna().any().any():
            reindexed.plot(kind="bar", ax=ax, colormap="Set2", width=0.7, edgecolor="white")
            ax.set_title(f"Mean {METRIC_LABELS.get(quality_col, quality_col)} (Mode × LLM)",
                         fontweight="bold", fontsize=9)
            ax.set_ylabel(METRIC_LABELS.get(quality_col, quality_col))
            ax.set_xlabel("")
            ax.set_ylim(0, 1.08)
            ax.axhline(0.6, color="orange", ls="--", lw=1, alpha=0.6)
            ax.axhline(0.8, color="green",  ls="--", lw=1, alpha=0.6)
            ax.grid(axis="y", alpha=0.3, linestyle="--")
            ax.legend(title="LLM", fontsize=8)
            ax.tick_params(axis="x", rotation=25, labelsize=8)
        else:
            ax.set_visible(False)
    else:
        ax.set_visible(False)

    # Panel 3: runtime by mode
    ax = axes[2]
    if "runtime_s" in ok.columns:
        rt = ok.groupby("mode")["runtime_s"].mean()
        order_r = [m for m in MODE_ORDER if m in rt.index]
        order_r += [m for m in sorted(rt.index) if m not in MODE_ORDER]
        ax.barh(order_r, [rt[m] for m in order_r],
                color=[_mode_color(m) for m in order_r], edgecolor="white")
        ax.set_xlabel("Mean Runtime (s)")
        ax.set_title("Runtime by Mode", fontweight="bold")
        ax.grid(axis="x", alpha=0.3, linestyle="--")
        ax.tick_params(axis="y", labelsize=8)

    _save(fig, out / "dashboard.png")


# ---------------------------------------------------------------------------
# Cross-dataset figure
# ---------------------------------------------------------------------------

def plot_cross_dataset_comparison(df: pd.DataFrame, out: Path):
    """Cross-dataset comparison of key metrics."""
    if "dataset" not in df.columns:
        return

    candidate = ["weighted_f1", "macro_f1", "ari", "gene_expr_spearman_r",
                 "hvg_jaccard", "runtime_s"]
    metrics = [m for m in candidate if m in df.columns and df[m].notna().any()]
    if not metrics:
        return

    ncols = min(3, len(metrics))
    nrows = (len(metrics) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.array(axes).flatten()

    datasets = df["dataset"].dropna().unique()
    palette  = sns.color_palette("Set2", len(datasets))

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        plot_df = df.dropna(subset=[metric])
        if plot_df.empty:
            ax.set_visible(False); continue
        ds_palette = {ds: palette[i] for i, ds in enumerate(datasets)}
        sns.boxplot(data=plot_df, x="dataset", y=metric,
                    hue="dataset", palette=ds_palette, legend=False,
                    ax=ax, width=0.5)
        sns.stripplot(data=plot_df, x="dataset", y=metric, color="black",
                      ax=ax, alpha=0.5, size=5, jitter=True)
        label = METRIC_LABELS.get(metric, metric)
        ax.set_title(label, fontweight="bold")
        ax.set_xlabel("Dataset")
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.tick_params(axis="x", rotation=20)

    for idx in range(len(metrics), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Cross-Dataset Comparison", fontsize=13, fontweight="bold")
    _save(fig, out / "cross_dataset_comparison.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot CARIBOU comparison results")
    parser.add_argument("--dataset", nargs="+", default=None)
    parser.add_argument("--results-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=PLOTS_DIR)
    parser.add_argument("--eval-output-dir", type=Path, default=OUTPUT_DIR,
                        help="Directory containing evaluate.py outputs (for confusion matrices)")
    args = parser.parse_args()

    if args.results_json:
        df = pd.DataFrame(json.loads(args.results_json.read_text()))
    else:
        all_path = args.eval_output_dir / "all_results.json"
        if not all_path.exists():
            print(f"No results found at {all_path}. Run: python evaluate.py")
            return
        df = pd.DataFrame(json.loads(all_path.read_text()))

    if args.dataset:
        df = df[df["dataset"].isin(args.dataset)]

    if df.empty:
        print("No data to plot.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Plotting {len(df)} results → {args.output_dir}")

    for ds_id in df["dataset"].dropna().unique():
        all_ds_df = df[df["dataset"] == ds_id]
        ds_df     = _success_df(all_ds_df)
        ds_out    = args.output_dir / ds_id
        eval_out  = args.eval_output_dir / ds_id
        ds_out.mkdir(parents=True, exist_ok=True)
        print(f"\n  {ds_id} ({len(ds_df)} successful runs)")

        plot_completeness_heatmap(all_ds_df, ds_id, ds_out)
        plot_metrics_overview(all_ds_df,     ds_id, ds_out)
        plot_population_metrics(all_ds_df,   ds_id, ds_out)
        plot_gene_scatter(all_ds_df,         ds_id, ds_out)
        plot_radar(all_ds_df,                ds_id, ds_out)
        plot_runtime_vs_quality(ds_df,       ds_id, ds_out)
        plot_dataset_dashboard(all_ds_df,    ds_id, ds_out)
        plot_celltyping_summary(ds_df,       ds_id, ds_out)
        plot_celltyping_heatmaps(ds_df,      ds_id, ds_out)

        # Confusion matrix figures (one per run, one aggregated per-type F1)
        plot_per_type_f1(eval_out, ds_id, ds_out)
        for run_row in all_ds_df.itertuples():
            run_name = run_row.run_name
            cm_path  = eval_out / f"confusion_{run_name}.json"
            if cm_path.exists():
                plot_confusion_matrix(cm_path, ds_id, run_name, ds_out)

    if df["dataset"].nunique() > 1:
        print("\n  Cross-dataset")
        plot_cross_dataset_comparison(_success_df(df), args.output_dir)

    print(f"\n✓  All plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
