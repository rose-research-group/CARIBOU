#!/usr/bin/env python3
"""
Clean publication-quality plots for CARIBOU comparisons.

Reads results from analysis/outputs/ and produces polished figures
for use in manuscripts and presentations.

Usage:
    python make_clean_plots.py
    python make_clean_plots.py --dataset tsp_large_intestine
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

COMP_DIR   = Path(__file__).parent.parent
OUTPUT_DIR = COMP_DIR / "analysis" / "outputs"
PLOTS_DIR  = Path(__file__).parent
RESULTS_DIR = COMP_DIR / "results"
DATASETS_DIR = COMP_DIR / "datasets"

# ---------------------------------------------------------------------------
# Display configuration
# ---------------------------------------------------------------------------

# Modes to EXCLUDE from clean plots
EXCLUDED_MODES = {"full_system"}

# Human-readable labels for mode names
MODE_DISPLAY_NAMES = {
    "one_shot":           "One Shot",
    "single_agent":       "Single Agent",
    "full_system_no_mem": "Full Agent System\nNo Memory Compression",
    "full_system":        "Full Agent System",
}

# Ordered list of modes to show (top → bottom on y-axis / left → right on x)
MODE_ORDER = ["one_shot", "single_agent", "full_system_no_mem"]

CIVIDIS_CMAP = plt.get_cmap("cividis")

# Categorical colors per mode
PLASMA_CATS = {
    "one_shot":           CIVIDIS_CMAP(0.18),
    "single_agent":       CIVIDIS_CMAP(0.52),
    "full_system_no_mem": CIVIDIS_CMAP(0.84),
}

# LLM colors
LLM_COLORS_PLASMA = {
    "chatgpt":  CIVIDIS_CMAP(0.15),
    "deepseek": CIVIDIS_CMAP(0.50),
    "claude":   CIVIDIS_CMAP(0.85),
}

LLM_DISPLAY = {
    "chatgpt":  "ChatGPT",
    "deepseek": "DeepSeek",
    "claude":   "Claude",
}

# SCIB-like metrics to include in the review panel
SCIB_METRICS = [
    ("ari",                  "Adjusted Rand Index"),
    ("nmi",                  "Normalized Mutual Information"),
    ("weighted_f1",          "Weighted F1 Score"),
    ("gene_expr_spearman_r", "Gene Expr. Spearman r"),
    ("hvg_jaccard",          "HVG Jaccard"),
    ("umap_knn_overlap",     "UMAP k-nearest-neighbor overlap"),
    ("pca_knn_overlap",      "PCA k-nearest-neighbor overlap"),
]

# Custom colormap: white → cividis.
WHITE_CIVIDIS = LinearSegmentedColormap.from_list(
    "white_cividis",
    [(1, 1, 1, 1)] + [CIVIDIS_CMAP(x) for x in np.linspace(0.2, 1.0, 255)],
)

# Global matplotlib style
plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.labelsize":   11,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "figure.dpi":       150,
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(fig, path: Path, dpi: int = 300):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    svg_path = path.with_suffix(".svg")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.relative_to(COMP_DIR)}")
    print(f"  Saved: {svg_path.relative_to(COMP_DIR)}")


def _load_dataset(dataset_id: str) -> pd.DataFrame:
    results_path = OUTPUT_DIR / dataset_id / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"No results at {results_path}. Run evaluate.py first.")
    df = pd.DataFrame(json.loads(results_path.read_text()))
    if "mode" in df.columns:
        df = df[~df["mode"].isin(EXCLUDED_MODES)].copy()
    return df


def _display_mode(mode: str) -> str:
    return MODE_DISPLAY_NAMES.get(mode, mode)


def _display_llm(llm: str) -> str:
    return LLM_DISPLAY.get(str(llm), str(llm).title())


def _mode_color(mode: str) -> str:
    return PLASMA_CATS.get(mode, "#aaa")


def _best_run_per_mode_llm(df: pd.DataFrame, metric: str = "weighted_f1") -> pd.DataFrame:
    """Return the single best run per (mode, llm) by a given metric."""
    if metric not in df.columns:
        return df
    return (
        df.dropna(subset=[metric])
          .sort_values(metric, ascending=False)
          .groupby(["mode", "llm"], as_index=False)
          .first()
    )


# ---------------------------------------------------------------------------
# Plot 1: Weighted F1 heatmap (mode × LLM)
# ---------------------------------------------------------------------------

def plot_weighted_f1_heatmap(df: pd.DataFrame, dataset_id: str, out: Path):
    """
    Heatmap of mean Weighted F1 by mode (y) × LLM (x).
    Uses plasma colormap. Excludes full_system.
    Renames full_system_no_mem to its full human-readable label.
    """
    col = "weighted_f1"
    if col not in df.columns or df[col].dropna().empty:
        print(f"  [skip] weighted_f1 not available for {dataset_id}")
        return

    plot_df = df.dropna(subset=[col]).copy()
    if "mode" not in plot_df.columns or "llm" not in plot_df.columns:
        print(f"  [skip] mode/llm columns missing for {dataset_id}")
        return

    pivot = plot_df.pivot_table(
        index="mode", columns="llm", values=col, aggfunc="mean"
    )

    present_modes = [m for m in MODE_ORDER if m in pivot.index]
    pivot = pivot.reindex(index=present_modes)
    pivot.index = [_display_mode(m) for m in pivot.index]
    pivot.columns = [_display_llm(c) for c in pivot.columns]

    if pivot.empty:
        return

    n_rows, n_cols = pivot.shape
    fig_w = 3.5 + 1.8 * n_cols
    fig_h = 2.0 + 1.1 * n_rows

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap=WHITE_CIVIDIS,
        vmin=0.0,
        vmax=1.0,
        linewidths=0.8,
        linecolor="#ffffff",
        annot_kws={"size": 13, "weight": "bold"},
        cbar_kws={"label": "Weighted F1", "shrink": 0.85},
        ax=ax,
    )
    ax.set_title(
        f"{dataset_id.replace('_', ' ').title()}: Weighted F1 (mean)",
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel("LLM", fontweight="bold", labelpad=8)
    ax.set_ylabel("Mode", fontweight="bold", labelpad=8)
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)

    _save(fig, out / dataset_id / "weighted_f1_heatmap.png")


# ---------------------------------------------------------------------------
# Plot 2: Marker gene comparison (reference top markers vs CARIBOU top markers)
# ---------------------------------------------------------------------------

def _compute_reference_markers(config: dict, top_n: int = 5) -> pd.DataFrame:
    """
    Compute top marker genes per coarse cell type from the reference h5ad.
    Returns a DataFrame with columns: [cell_type, rank, gene].
    Results are cached to avoid recomputation.
    """
    import scanpy as sc
    import warnings

    ref_path = Path(config["reference_path"])
    coarse_map = config.get("coarse_celltype_mapping", {})
    celltype_col = config.get("reference_celltype_key", "cell_type")
    cache_path = ref_path.parent / "ref_top_markers_coarse.csv"

    if cache_path.exists():
        return pd.read_csv(cache_path)

    print(f"    Computing reference markers from {ref_path.name} (this may take a moment)...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adata = sc.read_h5ad(ref_path)

    if celltype_col not in adata.obs.columns:
        _attach_metadata_celltypes(adata, config)
    if celltype_col not in adata.obs.columns:
        raise KeyError(f"Missing reference cell type column: '{celltype_col}'")

    # Apply coarse mapping
    adata.obs["coarse_celltype"] = (
        adata.obs[celltype_col].map(coarse_map).fillna("Other")
    )
    adata = adata[adata.obs["coarse_celltype"] != "Other"].copy()

    # Need log-normalised counts for rank_genes_groups
    if "log1p" not in adata.uns.get("log1p", {}) and adata.X.max() > 30:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc.tl.rank_genes_groups(
            adata,
            groupby="coarse_celltype",
            method="wilcoxon",
            n_genes=top_n,
            use_raw=False,
        )

    # Build Ensembl → gene symbol mapping if reference uses Ensembl IDs
    ensembl_to_symbol = {}
    if "feature_name" in adata.var.columns:
        ensembl_to_symbol = adata.var["feature_name"].to_dict()

    rows = []
    groups = adata.uns["rank_genes_groups"]["names"].dtype.names
    for grp in groups:
        for rank, gene in enumerate(adata.uns["rank_genes_groups"]["names"][grp], start=1):
            # Convert Ensembl ID to gene symbol if available
            gene_sym = ensembl_to_symbol.get(gene, gene)
            rows.append({"cell_type": grp, "rank": rank, "gene": gene_sym})

    result = pd.DataFrame(rows)
    result.to_csv(cache_path, index=False)
    print(f"    Cached reference markers to {cache_path.name}")
    return result


def _attach_metadata_celltypes(adata, config: dict) -> bool:
    """Attach cell types from metadata CSV if configured. Returns True if applied."""
    mj = config.get("metadata_join")
    if not mj:
        return False

    csv_path = Path(mj["csv_path"])
    if not csv_path.exists():
        print(f"    [skip marker plot] metadata_join csv not found: {csv_path}")
        return False

    ct_col = mj["csv_celltype_col"]
    split_on = mj.get("csv_index_split_on")
    barcode_col = mj.get("csv_barcode_col", "cell_barcode")
    library_col = mj.get("csv_library_col", "library_label")
    ref_ct_key = config.get("reference_celltype_key", "cell_type")
    ref_bc_col = config.get("barcode_join", {}).get("reference_barcode_col", "cell_barcode")
    ref_lib_col = config.get("barcode_join", {}).get("reference_library_col", "library_label")

    if ref_bc_col not in adata.obs.columns or ref_lib_col not in adata.obs.columns:
        print(f"    [skip marker plot] reference obs missing '{ref_bc_col}' or '{ref_lib_col}'")
        return False

    meta = pd.read_csv(csv_path, index_col=0, low_memory=False)

    if split_on:
        def _split(s):
            idx = s.find(split_on)
            return (s[:idx], s[idx + 1:]) if idx != -1 else (s, None)
        meta[[barcode_col, library_col]] = pd.DataFrame(
            [_split(s) for s in meta.index], index=meta.index
        )

    meta["_jk"] = meta[barcode_col].astype(str) + "|" + meta[library_col].astype(str)
    meta_ct = meta.set_index("_jk")[ct_col]

    ref_jk = (
        adata.obs[ref_bc_col].astype(str)
        + "|"
        + adata.obs[ref_lib_col].astype(str)
    )
    adata.obs[ref_ct_key] = ref_jk.map(meta_ct).values

    n_annotated = adata.obs[ref_ct_key].notna().sum()
    print(f"    Metadata join: {n_annotated:,}/{adata.n_obs:,} cells annotated via '{ct_col}'")
    return True


def plot_marker_gene_comparison(
    df: pd.DataFrame,
    dataset_id: str,
    out: Path,
    top_n: int = 5,
):
    """
    Two-panel marker gene comparison figure:

    Panel A (top): Jaccard overlap heatmap — for each cell type × CARIBOU run,
      the fraction of reference top-N markers that appear in CARIBOU's top-N.
      Plasma colormap, cell types on y-axis, CARIBOU runs on x-axis.

    Panel B (bottom): Gene-level dot plot — for each reference cell type, shows
      which top-N reference genes are recovered by each CARIBOU run.
      One subplot per cell type; rows = sources (Ref + runs), cols = genes.
    """
    config_path = DATASETS_DIR / dataset_id / "config.json"
    if not config_path.exists():
        print(f"  [skip marker plot] No config at {config_path}")
        return

    config = json.loads(config_path.read_text())

    # --- Reference markers ---
    try:
        ref_markers = _compute_reference_markers(config, top_n=top_n)
    except Exception as e:
        print(f"  [skip marker plot] Could not compute reference markers: {e}")
        return

    ct_order = sorted(ref_markers["cell_type"].unique())

    # --- CARIBOU markers: best run per (mode, llm) ---
    best_runs = _best_run_per_mode_llm(df, metric="weighted_f1")
    run_dir = RESULTS_DIR / dataset_id

    # ref_gene_sets[ct] = ordered list of reference top-N genes
    ref_gene_sets = {}
    for ct in ct_order:
        genes = (ref_markers[ref_markers["cell_type"] == ct]
                 .nsmallest(top_n, "rank")["gene"].tolist())
        ref_gene_sets[ct] = genes

    # caribou_gene_sets[run_label][ct] = set of CARIBOU top-N genes
    caribou_sources = {}   # run_label → {ct: set(genes)}
    run_labels_ordered = []
    for _, row in best_runs.iterrows():
        run_name = row["run_name"]
        mode_label = _display_mode(row["mode"]).replace("\n", " ")
        llm_label = _display_llm(row.get("llm", ""))
        label = f"{llm_label} / {mode_label}"
        csv_path = run_dir / run_name / "top_markers_by_cell_type.csv"
        if not csv_path.exists():
            continue
        try:
            car_df = pd.read_csv(csv_path)
        except Exception:
            continue
        src = {}
        for ct in ct_order:
            genes = (car_df[car_df["cell_type"] == ct]
                     .nsmallest(top_n, "rank")["gene"].tolist())
            src[ct] = set(genes)
        caribou_sources[label] = src
        run_labels_ordered.append(label)

    if not caribou_sources:
        print(f"  [skip marker plot] No CARIBOU marker CSVs found")
        return

    # -----------------------------------------------------------------------
    # Panel A: Jaccard overlap heatmap (cell types × CARIBOU runs)
    # -----------------------------------------------------------------------
    jaccard_data = {}
    for run_label in run_labels_ordered:
        col_vals = []
        for ct in ct_order:
            ref_set = set(ref_gene_sets[ct])
            car_set = caribou_sources[run_label].get(ct, set())
            if not ref_set and not car_set:
                col_vals.append(np.nan)
            else:
                j = len(ref_set & car_set) / len(ref_set | car_set) if (ref_set | car_set) else np.nan
                col_vals.append(j)
        jaccard_data[run_label] = col_vals

    jaccard_df = pd.DataFrame(jaccard_data, index=ct_order)

    fig_a_h = max(4.5, len(ct_order) * 0.55 + 1.5)
    fig_a_w = max(6.0, len(run_labels_ordered) * 2.0 + 2.5)
    fig_a, ax_a = plt.subplots(figsize=(fig_a_w, fig_a_h))

    sns.heatmap(
        jaccard_df,
        annot=True,
        fmt=".2f",
        cmap=WHITE_CIVIDIS,
        vmin=0.0,
        vmax=1.0,
        linewidths=0.6,
        linecolor="#ffffff",
        annot_kws={"size": 10, "weight": "bold"},
        cbar_kws={"label": "Jaccard overlap\n(ref top-" + str(top_n) + " ∩ CARIBOU top-" + str(top_n) + ")",
                  "shrink": 0.8},
        ax=ax_a,
    )
    ax_a.set_title(
        f"{dataset_id.replace('_', ' ').title()}: Marker Gene Overlap (Jaccard)\n"
        f"Reference top-{top_n} vs CARIBOU top-{top_n} per cell type",
        fontweight="bold",
        pad=12,
    )
    ax_a.set_xlabel("CARIBOU run", fontweight="bold", labelpad=8)
    ax_a.set_ylabel("Cell type", fontweight="bold", labelpad=8)
    ax_a.tick_params(axis="x", rotation=15, labelsize=9)
    ax_a.tick_params(axis="y", rotation=0, labelsize=9)

    _save(fig_a, out / dataset_id / "marker_jaccard_heatmap.png")

    # -----------------------------------------------------------------------
    # Panel B: Per-cell-type dot plot — which reference genes are recovered
    # -----------------------------------------------------------------------
    # Layout: each cell type gets one small subplot; sources on y, genes on x
    n_ct = len(ct_order)
    ncols_b = 4
    nrows_b = int(np.ceil(n_ct / ncols_b))
    n_sources = 1 + len(run_labels_ordered)  # Reference + CARIBOU runs
    source_labels_b = ["Reference"] + run_labels_ordered

    cell_h = max(0.55 * n_sources + 1.2, 2.5)
    cell_w = max(top_n * 0.7 + 1.0, 3.5)
    fig_b, axes_b = plt.subplots(
        nrows_b, ncols_b,
        figsize=(cell_w * ncols_b, cell_h * nrows_b),
        squeeze=False,
    )

    cmap = WHITE_CIVIDIS
    dot_max = 280

    for ci, ct in enumerate(ct_order):
        ax = axes_b[ci // ncols_b][ci % ncols_b]
        ref_genes = ref_gene_sets[ct]  # ordered list, rank 1 first

        for si, src_label in enumerate(source_labels_b):
            if src_label == "Reference":
                gene_ranks = {g: r + 1 for r, g in enumerate(ref_genes)}
            else:
                # Check each reference gene's presence in this CARIBOU run
                car_set = caribou_sources[src_label].get(ct, set())
                # For recovered genes show rank 1 (present), absent = NaN
                gene_ranks = {g: (1 if g in car_set else None) for g in ref_genes}

            for gi, gene in enumerate(ref_genes):
                rank_val = gene_ranks.get(gene)
                if src_label == "Reference":
                    # Show actual rank as color strength
                    strength = (top_n + 1 - rank_val) / top_n
                    color = cmap(strength)
                    size = dot_max * strength
                    ax.scatter(gi, si, s=size, color=color,
                               edgecolors="white", linewidth=0.4,
                               zorder=3, alpha=0.95)
                elif rank_val is not None:
                    # Gene recovered in CARIBOU — bright plasma yellow
                    color = cmap(0.85)
                    ax.scatter(gi, si, s=dot_max * 0.7, color=color,
                               edgecolors="white", linewidth=0.4,
                               zorder=3, alpha=0.95, marker="D")
                else:
                    # Gene absent in CARIBOU
                    ax.scatter(gi, si, s=25, color="#e0e0e0",
                               edgecolors="#cccccc", linewidth=0.3,
                               zorder=2, marker="o")

        ax.set_title(ct, fontweight="bold", fontsize=9, pad=4)
        ax.set_xticks(range(len(ref_genes)))
        ax.set_xticklabels(ref_genes, rotation=40, ha="right", fontsize=7)
        ax.set_yticks(range(n_sources))
        ax.set_yticklabels(
            [s.replace(" / ", "\n") for s in source_labels_b],
            fontsize=6.5,
        )
        ax.set_xlim(-0.6, len(ref_genes) - 0.4)
        ax.set_ylim(-0.6, n_sources - 0.4)
        ax.grid(axis="x", alpha=0.2, linestyle=":", zorder=0)

    # Hide unused subplots
    for ci in range(n_ct, nrows_b * ncols_b):
        axes_b[ci // ncols_b][ci % ncols_b].set_visible(False)

    # Legend
    legend_elements = [
        plt.scatter([], [], s=dot_max * 0.9, color=cmap(0.9),
                    label=f"Ref rank 1 (strongest)", edgecolors="white"),
        plt.scatter([], [], s=dot_max * 0.3, color=cmap(0.3),
                    label=f"Ref rank {top_n} (weakest)", edgecolors="white"),
        plt.scatter([], [], s=dot_max * 0.7, color=cmap(0.85), marker="D",
                    label="Recovered by CARIBOU", edgecolors="white"),
        plt.scatter([], [], s=25, color="#e0e0e0",
                    label="Not recovered", edgecolors="#cccccc"),
    ]
    fig_b.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=4,
        fontsize=8.5,
        bbox_to_anchor=(0.5, -0.02),
        frameon=True,
        title="Reference marker gene strength  |  CARIBOU recovery",
        title_fontsize=9,
    )

    fig_b.suptitle(
        f"{dataset_id.replace('_', ' ').title()}: Reference Top-{top_n} Markers — CARIBOU Recovery",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    _save(fig_b, out / dataset_id / "marker_gene_recovery.png")


# ---------------------------------------------------------------------------
# Plot 3: SCIB-like metrics review panel
# ---------------------------------------------------------------------------

def plot_scib_metrics_panel(df: pd.DataFrame, dataset_id: str, out: Path):
    """
    Review panel of SCIB-like metrics comparing each CARIBOU mode to the
    ground-truth reference (perfect score = 1.0).

    One sub-panel per metric; each shows:
      - Strip + box plot of individual run scores per mode
      - A dashed reference line at 1.0 (perfect alignment with ground truth)
      - Plasma-derived colors per mode
    """
    available = [(col, label) for col, label in SCIB_METRICS
                 if col in df.columns and df[col].dropna().size > 0]
    if not available:
        print(f"  [skip scib panel] No SCIB-like metrics available for {dataset_id}")
        return

    n = len(available)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.2 * nrows))
    axes = np.array(axes).reshape(-1)

    present_modes = [m for m in MODE_ORDER if m in df["mode"].values]
    palette = {m: _mode_color(m) for m in present_modes}
    x_labels = [_display_mode(m) for m in present_modes]

    for ax, (col, label) in zip(axes, available):
        plot_df = df[df["mode"].isin(present_modes)].dropna(subset=[col]).copy()
        if plot_df.empty:
            ax.set_visible(False)
            continue

        # Box plot
        sns.boxplot(
            data=plot_df,
            x="mode",
            y=col,
            order=present_modes,
            hue="mode",
            palette=palette,
            legend=False,
            ax=ax,
            width=0.5,
            fliersize=0,
            linewidth=1.2,
        )

        # Strip plot (individual runs), colored by LLM
        if "llm" in plot_df.columns:
            llm_palette = {k: LLM_COLORS_PLASMA.get(k, "#aaa") for k in plot_df["llm"].dropna().unique()}
            sns.stripplot(
                data=plot_df,
                x="mode",
                y=col,
                order=present_modes,
                hue="llm",
                palette=llm_palette,
                ax=ax,
                size=7,
                alpha=0.9,
                dodge=True,
                jitter=True,
                zorder=4,
            )
            legend = ax.get_legend()
            if legend:
                legend.set_title("LLM", prop={"size": 8})
                for text in legend.get_texts():
                    text.set_text(_display_llm(text.get_text()))
                    text.set_fontsize(8)

        # Reference baseline
        ax.axhline(1.0, color="#2d2d2d", lw=1.5, ls="--", alpha=0.75,
                   label="Reference (perfect)", zorder=2)

        ax.set_title(label, fontweight="bold", fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel(label, fontsize=9)
        ax.set_xticks(range(len(present_modes)))
        ax.set_xticklabels(x_labels, rotation=20, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.3, linestyle="--", zorder=0)

        # Y-axis limits
        ymin = max(0.0, plot_df[col].min() - 0.08)
        ax.set_ylim(ymin, 1.08)

    # Hide unused panels
    for ax in axes[len(available):]:
        ax.set_visible(False)

    # Legend for reference line
    ref_patch = plt.Line2D([0], [0], color="#2d2d2d", lw=1.5, ls="--",
                            label="Reference baseline (= 1.0)")
    fig.legend(
        handles=[ref_patch],
        loc="lower center",
        ncol=1,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.01),
        frameon=True,
    )

    fig.suptitle(
        f"{dataset_id.replace('_', ' ').title()}: SCIB-like Metrics vs Reference Baseline",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    _save(fig, out / dataset_id / "scib_metrics_panel.png")


# ---------------------------------------------------------------------------
# Plot 4: Runtime panel
# ---------------------------------------------------------------------------

def plot_runtime_panel(df: pd.DataFrame, dataset_id: str, out: Path):
    """
    Two sub-panels showing wall-clock runtime and number of agent turns per mode.

    - Left: runtime in minutes (box + strip, colored by LLM via plasma palette)
    - Right: number of turns (box + strip, same scheme)
    All runs included regardless of success; failed runs shown at lower opacity.
    """
    if "runtime_s" not in df.columns:
        print(f"  [skip runtime] runtime_s not available for {dataset_id}")
        return

    plot_df = df[df["mode"].isin(MODE_ORDER)].copy()
    plot_df = plot_df.dropna(subset=["runtime_s"])
    if plot_df.empty:
        return

    plot_df["runtime_min"] = plot_df["runtime_s"] / 60.0

    present_modes = [m for m in MODE_ORDER if m in plot_df["mode"].values]
    mode_palette  = {m: _mode_color(m) for m in present_modes}
    x_labels      = [_display_mode(m) for m in present_modes]
    llm_palette   = {k: LLM_COLORS_PLASMA.get(k, "#aaa")
                     for k in plot_df["llm"].dropna().unique()}

    panels = [("runtime_min", "Runtime (minutes)")]
    if "num_turns" in df.columns and df["num_turns"].dropna().size > 0:
        panels.append(("num_turns", "Agent turns"))

    fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 4.8))
    if len(panels) == 1:
        axes = [axes]

    for ax, (col, label) in zip(axes, panels):
        pdata = plot_df.dropna(subset=[col])

        sns.boxplot(
            data=pdata,
            x="mode",
            y=col,
            order=present_modes,
            hue="mode",
            palette=mode_palette,
            legend=False,
            ax=ax,
            width=0.5,
            fliersize=0,
            linewidth=1.2,
        )

        sns.stripplot(
            data=pdata,
            x="mode",
            y=col,
            order=present_modes,
            hue="llm",
            palette=llm_palette,
            ax=ax,
            size=8,
            alpha=0.9,
            dodge=True,
            jitter=True,
            zorder=4,
        )

        legend = ax.get_legend()
        if legend:
            legend.set_title("LLM", prop={"size": 8})
            for text in legend.get_texts():
                text.set_text(_display_llm(text.get_text()))
                text.set_fontsize(8)

        ax.set_title(label, fontweight="bold", fontsize=11)
        ax.set_xlabel("")
        ax.set_ylabel(label, fontsize=10)
        ax.set_xticks(range(len(present_modes)))
        ax.set_xticklabels(x_labels, rotation=20, ha="right", fontsize=9)
        ax.grid(axis="y", alpha=0.3, linestyle="--", zorder=0)
        ymin = max(0.0, pdata[col].min() * 0.85)
        ax.set_ylim(ymin, pdata[col].max() * 1.12)

    fig.suptitle(
        f"{dataset_id.replace('_', ' ').title()}: Runtime Comparison",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    _save(fig, out / dataset_id / "runtime_panel.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate clean CARIBOU plots")
    parser.add_argument(
        "--dataset", nargs="+", default=None,
        help="Dataset ID(s) to plot. Defaults to all available."
    )
    parser.add_argument(
        "--top-n", type=int, default=5,
        help="Number of top marker genes per cell type (default: 5)"
    )
    args = parser.parse_args()

    # Discover datasets
    if args.dataset:
        dataset_ids = args.dataset
    else:
        dataset_ids = [p.name for p in OUTPUT_DIR.iterdir()
                       if p.is_dir() and (p / "results.json").exists()]
        if not dataset_ids:
            print(f"No results found under {OUTPUT_DIR}. Run evaluate.py first.")
            return

    print(f"Generating clean plots → {PLOTS_DIR}")

    for ds_id in sorted(dataset_ids):
        print(f"\n  [{ds_id}]")
        try:
            df = _load_dataset(ds_id)
        except FileNotFoundError as e:
            print(f"  [skip] {e}")
            continue

        plot_weighted_f1_heatmap(df, ds_id, PLOTS_DIR)
        plot_scib_metrics_panel(df, ds_id, PLOTS_DIR)
        plot_runtime_panel(df, ds_id, PLOTS_DIR)
        plot_marker_gene_comparison(df, ds_id, PLOTS_DIR, top_n=args.top_n)

    print(f"\n✓  Done.")


if __name__ == "__main__":
    main()
