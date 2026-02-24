import scanpy as sc
import numpy as np
import pandas as pd
from pathlib import Path

def run_dge_scanpy(
    adata,
    groupby: str,
    group: str,
    reference: str | None = None,
    subset_query: str | None = None,
    counts_layer: str | None = "counts",
    normalize: bool = True,
    log1p: bool = True,
    target_sum: float = 1e4,
    method: str = "wilcoxon",     # "wilcoxon", "t-test", "logreg"
    n_genes: int = 200,
    key_added: str = "rank_genes_groups",
    export_csv: str | Path | None = None,
    plot_rank: bool = True,
    plot_dotplot: bool = False,
    top_plot: int = 25,
    show: bool = True,
):
    """
    Agent-ready DGE in Scanpy using sc.tl.rank_genes_groups.

    Workflow:
      1) Optional subset (adata.obs.eval)
      2) Choose expression source:
         - if counts_layer exists: copy to X (so we don't mutate the original), else leave X
      3) Optional normalize_total + log1p
      4) Store processed matrix in adata.raw (for reproducibility & plotting)
      5) Run rank_genes_groups
      6) Return tidy DataFrame; optionally export CSV; optionally plot

    Returns:
      (adata_sub, df)  # adata_sub is a copy if subset_query is used
    """

    # --- basic checks ---
    if groupby not in adata.obs:
        raise ValueError(f"'{groupby}' not found in adata.obs")

    # --- optional subset ---
    ad = adata
    if subset_query:
        mask = adata.obs.eval(subset_query)
        if mask.sum() == 0:
            raise ValueError(f"subset_query returned 0 cells: {subset_query}")
        ad = adata[mask].copy()
        print(f"Subset: kept {ad.n_obs}/{adata.n_obs} cells via: {subset_query}")
    else:
        # don't accidentally mutate upstream object if we're about to overwrite X
        ad = adata.copy()

    # --- choose expression source ---
    if counts_layer and counts_layer in ad.layers:
        ad.X = ad.layers[counts_layer].copy()
        print(f"Using layer '{counts_layer}' as starting expression (copied into .X).")
    else:
        if counts_layer:
            print(f"Note: counts_layer='{counts_layer}' not found. Using existing ad.X as-is.")

    # --- normalize / log ---
    if normalize:
        sc.pp.normalize_total(ad, target_sum=target_sum)
    if log1p:
        sc.pp.log1p(ad)

    # --- store processed matrix in raw for reproducibility ---
    # raw is commonly used for plotting & downstream ops; this captures the processed state.
    ad.raw = ad

    # --- categorical + membership checks ---
    ad.obs[groupby] = ad.obs[groupby].astype("category")
    cats = list(ad.obs[groupby].cat.categories)

    if group not in cats:
        raise ValueError(f"group='{group}' not in obs['{groupby}'] categories: {cats}")

    if reference is not None and reference not in cats:
        raise ValueError(f"reference='{reference}' not in obs['{groupby}'] categories: {cats}")

    # --- DGE ---
    print(f"DGE: {group} vs {reference if reference else 'rest'} | method={method}")
    sc.tl.rank_genes_groups(
        ad,
        groupby=groupby,
        groups=[group],
        reference=reference,
        method=method,
        n_genes=n_genes,
        key_added=key_added,
        use_raw=True,   # use the processed ad.raw we just set
    )

    # --- tidy results ---
    df = sc.get.rank_genes_groups_df(ad, group=group, key=key_added)

    # add some convenience columns for agents
    df["comparison"] = f"{group} vs {reference if reference else 'rest'}"
    df["groupby"] = groupby
    if subset_query:
        df["subset_query"] = subset_query

    # --- export ---
    if export_csv is not None:
        export_csv = Path(export_csv)
        export_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(export_csv, index=False)
        print(f"Saved DGE table → {export_csv}")

    # --- plots ---
    if plot_rank:
        sc.pl.rank_genes_groups(
            ad,
            key=key_added,
            groups=[group],
            n_genes=min(top_plot, n_genes),
            sharey=False,
            show=show
        )

    if plot_dotplot:
        # Dotplot of top genes
        top_genes = df["names"].head(min(top_plot, len(df))).tolist()
        sc.pl.dotplot(
            ad,
            var_names=top_genes,
            groupby=groupby,
            standard_scale="var",
            show=show
        )

    return ad, df
