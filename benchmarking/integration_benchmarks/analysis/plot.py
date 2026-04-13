#!/usr/bin/env python3
"""
Manuscript-quality plots for integration benchmark results.

Produces three figures:
  umap_comparison.png         — UMAP panels: reference + one per run, colored by cell type
  integration_comparison.png  — main result: scib batch + bio metrics per run vs reference
  quality_panel.png           — supporting: gene expression fidelity, embedding fidelity, QC

Usage:
    python plot.py --dataset aba_hippocampus
    python plot.py                          # all available datasets
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

INTBENCH_DIR = Path(__file__).parent.parent
CARIBOU_ROOT = INTBENCH_DIR.parent.parent
sys.path.insert(0, str(INTBENCH_DIR))
sys.path.insert(0, str(CARIBOU_ROOT / "dev"))

from src.data_loader import available_datasets
from colors import MODE_COLORS as _MC, GREEN, BLUE, RED

ANALYSIS_DIR = INTBENCH_DIR / "analysis"

# ---------------------------------------------------------------------------
# Publication style
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.size":        10,
    "font.family":      "sans-serif",
    "axes.linewidth":   0.8,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "legend.frameon":   False,
})

MODE_ORDER  = ["one_shot", "single_agent", "full_system", "full_system_no_mem"]
MODE_COLORS = _MC  # from dev/colors.py
MODE_LABELS = {
    "one_shot":           "One-shot",
    "single_agent":       "Single agent",
    "full_system":        "Full system",
    "full_system_no_mem": "Full system\n(no mem)",
}


def _bar_color(mode):
    return MODE_COLORS.get(str(mode) if mode else "", "#aaaaaa")


def _save(fig, path: Path, dpi: int = 300):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.relative_to(INTBENCH_DIR)}")


def _ok(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["success"] == True].copy() if "success" in df.columns else df.copy()


def _avail(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns and df[col].notna().any()


def _run_label(row: pd.Series) -> str:
    """Short display label: llm + mode abbreviation."""
    llm  = str(row.get("llm", "")).capitalize()
    mode = str(row.get("mode", ""))
    abbrev = {"one_shot": "1S", "single_agent": "SA",
               "full_system": "FS", "full_system_no_mem": "FS-NM"}.get(mode, mode[:4])
    return f"{llm}\n{abbrev}" if llm and llm != "None" else abbrev


def _annotate_bars(ax, bars, vals, fmt=".2f", offset=0.01):
    """Place value labels above each bar."""
    for bar, val in zip(bars, vals):
        if val is not None and not np.isnan(float(val)):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + offset,
                f"{val:{fmt}}",
                ha="center", va="bottom", fontsize=8,
            )


def _mode_legend(fig, modes_present):
    patches = [
        mpatches.Patch(color=MODE_COLORS[m], label=MODE_LABELS.get(m, m))
        for m in MODE_ORDER if m in modes_present
    ]
    if patches:
        fig.legend(handles=patches, loc="lower center", ncol=len(patches),
                   bbox_to_anchor=(0.5, -0.04), fontsize=9, handlelength=1.2)


# ---------------------------------------------------------------------------
# Figure 1: integration_comparison
# ---------------------------------------------------------------------------

def plot_integration_comparison(df: pd.DataFrame, dataset_id: str, out: Path):
    """Main manuscript figure: scib batch + bio metrics, CARIBOU vs reference baseline."""

    batch_metrics = [
        ("car_asw_batch",          "baseline_ref_asw_batch",          "ASW Batch"),
        ("car_graph_connectivity", "baseline_ref_graph_connectivity", "Graph\nConnectivity"),
        ("car_ilisi",              "baseline_ref_ilisi",              "iLISI"),
    ]
    bio_metrics = [
        ("car_asw_celltype", "baseline_ref_asw_celltype", "ASW Cell Type"),
        ("car_clisi",        "baseline_ref_clisi",        "cLISI"),
    ]

    avail_batch = [(c, b, lb) for c, b, lb in batch_metrics if _avail(df, c)]
    avail_bio   = [(c, b, lb) for c, b, lb in bio_metrics   if _avail(df, c)]

    if not avail_batch and not avail_bio:
        print(f"  [{dataset_id}] No scib metrics available — skipping integration_comparison")
        return

    nrows = int(bool(avail_batch)) + int(bool(avail_bio))
    ncols = max(len(avail_batch), len(avail_bio), 1)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3.5 * nrows),
                             squeeze=False)

    rows_data = []
    if avail_batch: rows_data.append((avail_batch, "Batch correction"))
    if avail_bio:   rows_data.append((avail_bio,   "Bio conservation"))

    modes_present = set()
    annotate = len(df) <= 8

    for row_i, (metrics, row_title) in enumerate(rows_data):
        axes[row_i, 0].set_ylabel(row_title, fontsize=10, fontweight="bold", labelpad=8)
        for col_i, (car_col, ref_col, label) in enumerate(metrics):
            ax = axes[row_i, col_i]
            vals = df[car_col].values if car_col in df.columns else np.full(len(df), np.nan)
            colors = [_bar_color(m) for m in df.get("mode", pd.Series([""] * len(df)))]
            x = np.arange(len(df))
            bars = ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.4, zorder=3)
            if annotate:
                _annotate_bars(ax, bars, vals)

            # Reference baseline
            if ref_col in df.columns:
                ref_val = df[ref_col].dropna().iloc[0] if df[ref_col].notna().any() else None
                if ref_val is not None:
                    ax.axhline(ref_val, color="black", linewidth=1.5,
                               linestyle="--", zorder=4, label="ABA reference")

            ax.set_title(label, fontsize=10, pad=4)
            ax.set_ylim(0, 1.09)
            ax.set_xticks(x)
            xlabels = [_run_label(row) for _, row in df.iterrows()]
            ax.set_xticklabels(xlabels, fontsize=8)
            ax.yaxis.grid(True, linewidth=0.4, color="#e0e0e0", zorder=0)
            ax.set_axisbelow(True)
            modes_present.update(df.get("mode", pd.Series()).dropna().unique())

        # Hide unused columns in this row
        for col_i in range(len(metrics), ncols):
            axes[row_i, col_i].set_visible(False)

    fig.suptitle(f"Integration Quality — {dataset_id}", fontsize=12,
                 fontweight="bold", y=1.01)
    _mode_legend(fig, modes_present)
    fig.tight_layout()
    _save(fig, out / "integration_comparison.png")


# ---------------------------------------------------------------------------
# Figure 2: quality_panel
# ---------------------------------------------------------------------------

def plot_quality_panel(df: pd.DataFrame, dataset_id: str, out: Path):
    """Supporting figure: gene expression fidelity, embedding fidelity, QC."""

    panels = [
        ("gene_expr_spearman_r", "Gene Expression\nFidelity (Spearman r)", None),
        ("pca_knn_overlap",      "Embedding Fidelity\n(PCA kNN Overlap)",  1.0),
        ("qc_filtering_rate",    "QC Filtering Rate",                       None),
    ]
    avail = [(col, title, ref) for col, title, ref in panels if _avail(df, col)]
    if not avail:
        return

    fig, axes = plt.subplots(1, len(avail), figsize=(3.5 * len(avail), 3.8),
                             squeeze=False)
    panel_labels = "ABCDEF"
    modes_present = set()
    annotate = len(df) <= 8

    for i, (col, title, ref_val) in enumerate(avail):
        ax = axes[0, i]
        vals = df[col].values
        colors = [_bar_color(m) for m in df.get("mode", pd.Series([""] * len(df)))]
        x = np.arange(len(df))
        bars = ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.4, zorder=3)
        if annotate:
            _annotate_bars(ax, bars, vals)
        if ref_val is not None:
            ax.axhline(ref_val, color="black", linewidth=1.5,
                       linestyle="--", zorder=4)
        ax.set_title(title, fontsize=10, pad=4)
        ax.set_xticks(x)
        ax.set_xticklabels([_run_label(r) for _, r in df.iterrows()], fontsize=8)
        ax.yaxis.grid(True, linewidth=0.4, color="#e0e0e0", zorder=0)
        ax.set_axisbelow(True)
        # Panel label
        ax.text(-0.12, 1.02, panel_labels[i], transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="top")
        modes_present.update(df.get("mode", pd.Series()).dropna().unique())

    fig.suptitle(f"Data Quality — {dataset_id}", fontsize=12,
                 fontweight="bold", y=1.04)
    _mode_legend(fig, modes_present)
    fig.tight_layout()
    _save(fig, out / "quality_panel.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Figure 0: umap_comparison
# ---------------------------------------------------------------------------

# Cell-type palette for UMAPs — up to 28 distinct categories using project
# primary colours anchored at positions 0/1/2 and tab20c filling the rest.
_UMAP_BASE = [RED, GREEN, BLUE]


def _celltype_palette(labels: list[str]) -> dict[str, tuple]:
    unique = sorted(set(labels) - {"nan", "None", ""})
    n = len(unique)
    base_rgba = [matplotlib.colors.to_rgba(c) for c in _UMAP_BASE]
    if n <= 3:
        palette = base_rgba[:n]
    else:
        extra = plt.cm.tab20c(np.linspace(0, 1, max(n - 3, 1)))
        palette = base_rgba + list(extra[: n - 3])
    return {lb: palette[i] for i, lb in enumerate(unique)}


def plot_umap_comparison(df: pd.DataFrame, dataset_id: str, outputs_dir: Path, out: Path):
    """One UMAP panel per run + reference, all colored by cell type."""
    ref_path = outputs_dir / dataset_id / "reference_umap.npz"
    if not ref_path.exists():
        print(f"  [{dataset_id}] No reference_umap.npz — skipping UMAP comparison")
        return

    ref = np.load(ref_path, allow_pickle=True)
    ref_coords = ref["coords"]
    ref_labels = ref["labels"].astype(str)

    # Load per-run UMAP data
    run_panels = []
    for _, row in df.iterrows():
        npz = outputs_dir / dataset_id / str(row.get("run_name", "")) / "umap_coords.npz"
        if not npz.exists():
            continue
        d = np.load(npz, allow_pickle=True)
        run_panels.append({
            "coords": d["coords"],
            "labels": d["labels"].astype(str),
            "title":  _run_label(row),
            "mode":   str(row.get("mode", "")),
        })

    if not run_panels:
        print(f"  [{dataset_id}] No per-run umap_coords.npz found — skipping UMAP comparison")
        return

    # Shared cell-type palette across reference + all runs
    all_labels: list[str] = ref_labels.tolist()
    for rp in run_panels:
        all_labels.extend(rp["labels"].tolist())
    palette = _celltype_palette(all_labels)
    unknown = (0.75, 0.75, 0.75, 1.0)

    n_panels = 1 + len(run_panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(3.6 * n_panels, 3.6), squeeze=False)
    axes = axes[0]

    def _panel(ax, coords, labels, title, border_color=None):
        c = [palette.get(lb, unknown) for lb in labels]
        ax.scatter(coords[:, 0], coords[:, 1], c=c, s=0.5,
                   linewidths=0, rasterized=True, alpha=0.7)
        ax.set_title(title, fontsize=10, pad=5)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("UMAP 1", fontsize=8, labelpad=2)
        ax.set_ylabel("UMAP 2", fontsize=8, labelpad=2)
        if border_color:
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(border_color)
                spine.set_linewidth(2)

    _panel(axes[0], ref_coords, ref_labels, "ABA Reference")
    for i, rp in enumerate(run_panels):
        _panel(axes[i + 1], rp["coords"], rp["labels"], rp["title"],
               border_color=MODE_COLORS.get(rp["mode"]))

    # Legend (skip if too many labels to fit)
    unique_shown = [lb for lb in sorted(palette) if lb not in {"nan", "None", ""}]
    if len(unique_shown) <= 30:
        patches = [mpatches.Patch(color=palette[lb], label=lb) for lb in unique_shown]
        ncol = min(len(patches), 6)
        fig.legend(handles=patches, loc="lower center", ncol=ncol,
                   bbox_to_anchor=(0.5, -0.05), fontsize=7,
                   handlelength=1.0, handleheight=0.8, columnspacing=0.8)

    fig.suptitle(f"UMAP — {dataset_id}", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, out / "umap_comparison.png")


# ---------------------------------------------------------------------------
# Clean-plot filter
# ---------------------------------------------------------------------------

def _prepare_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a filtered, relabelled copy for clean publication plots.

    Rules
    -----
    - Drop any runs whose LLM backend is 'claude'.
    - Drop full_system (with memory) runs entirely.
    - Rename full_system_no_mem → full_system so existing colour/label
      mappings apply without modification.
    """
    d = df.copy()
    if "llm" in d.columns:
        d = d[d["llm"].str.lower() != "claude"]
    if "mode" in d.columns:
        d = d[d["mode"] != "full_system"]
        d["mode"] = d["mode"].replace("full_system_no_mem", "full_system")
    return d.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def plot_dataset(dataset_id: str, output_dir: Path, outputs_dir: Path):
    summary_path = outputs_dir / dataset_id / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"No summary.json for '{dataset_id}': {summary_path}\n"
            f"Run collect_results.py first."
        )

    df  = pd.DataFrame(json.loads(summary_path.read_text()))
    ok  = _ok(df)
    out = output_dir / dataset_id
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n  Plotting {dataset_id} ({len(ok)} successful run(s)) …")
    plot_umap_comparison(ok, dataset_id, outputs_dir, out)
    plot_integration_comparison(ok, dataset_id, out)
    plot_quality_panel(ok, dataset_id, out)

    # Clean plots — no claude, full_system_no_mem relabelled as full_system
    ok_clean = _prepare_clean(ok)
    if len(ok_clean) > 0:
        out_clean = out / "clean_plots"
        out_clean.mkdir(parents=True, exist_ok=True)
        print(f"  Clean plots ({len(ok_clean)} run(s)) …")
        plot_umap_comparison(ok_clean, dataset_id, outputs_dir, out_clean)
        plot_integration_comparison(ok_clean, dataset_id, out_clean)
        plot_quality_panel(ok_clean, dataset_id, out_clean)


def main():
    parser = argparse.ArgumentParser(
        description="Generate manuscript-quality integration benchmark plots"
    )
    parser.add_argument("--dataset", nargs="+", default=None)
    parser.add_argument("--output-dir", type=Path, default=ANALYSIS_DIR / "plots")
    parser.add_argument("--outputs-dir", type=Path, default=ANALYSIS_DIR / "outputs")
    args = parser.parse_args()

    datasets = args.dataset or available_datasets()
    if not datasets:
        raise RuntimeError(f"No datasets found in {INTBENCH_DIR / 'datasets'}")

    print(f"Plotting datasets: {datasets}")
    for ds in datasets:
        plot_dataset(ds, args.output_dir, args.outputs_dir)

    print(f"\n✓  Plots written to {args.output_dir}")


if __name__ == "__main__":
    main()
