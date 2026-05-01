import os

# *** MUST BE FIRST — before any celltypist import ***
# CellTypist reads CELLTYPIST_DATA_DIR at import time to set its model cache path.
# If not set, it falls back to $HOME/.celltypist_cache which is /tmp inside the
# sandbox container and will run out of space.
os.environ["CELLTYPIST_DATA_DIR"] = "/workspace/celltypist_models"

import scanpy as sc
import celltypist
from celltypist import models


def ensure_gene_symbols(adata):
    """
    CellTypist requires adata.var_names to be HGNC gene symbols (e.g. 'CD3D', 'MS4A1').
    If var_names look like Ensembl IDs (ENSG.../ENSM...), this function remaps them
    to gene symbols from a var column.

    Call this BEFORE any celltypist call.  Raises ValueError if no symbol column found.
    """
    sample = str(adata.var_names[0])
    if not (sample.startswith("ENS") or sample.startswith("ens")):
        print(f"var_names look like gene symbols already (e.g. '{sample}'). No remapping needed.")
        return adata

    # Ensembl IDs detected — find a gene symbol column in adata.var
    SYMBOL_COLS = ["gene_symbols", "gene_symbol", "symbol", "gene_name",
                   "feature_name", "hgnc_symbol", "Gene"]
    col = next((c for c in SYMBOL_COLS if c in adata.var.columns), None)
    if col is None:
        raise ValueError(
            f"var_names appear to be Ensembl IDs (e.g. '{sample}') but no gene symbol "
            f"column found in adata.var. Checked: {SYMBOL_COLS}. "
            f"Available columns: {list(adata.var.columns)}"
        )

    symbols = adata.var[col].astype(str)
    # Make unique: duplicate symbols get a numeric suffix
    seen = {}
    unique = []
    for s in symbols:
        if s in seen:
            seen[s] += 1
            unique.append(f"{s}_{seen[s]}")
        else:
            seen[s] = 0
            unique.append(s)

    adata.var_names = unique
    print(f"Remapped var_names from Ensembl IDs to gene symbols via adata.var['{col}'] "
          f"({len(unique)} genes).")
    return adata


def annotate_with_celltypist(
    adata,
    model_name=None,
    custom_markers=None,
    update_counts_layer=True,
    plot_umap=True,
    show=True
):
    """
    Annotate cells in AnnData using a CellTypist model or custom marker genes.

    Parameters:
        adata: AnnData object.
        model_name: Name of CellTypist model (e.g. 'Immune_All_Low.pkl').
        custom_markers: Dict of manual marker gene sets.
        update_counts_layer: If True and 'counts' in adata.layers, reset .X to raw counts.
        plot_umap: Plot UMAP colored by predicted cell type.
        show: Whether to immediately display plots.

    Returns:
        Annotated AnnData object with 'cell_type' column in .obs.
    """
    # Step 1: ensure gene symbols (CellTypist requirement)
    adata = ensure_gene_symbols(adata)

    # Step 2: reset to raw counts then normalize+log for CellTypist
    if update_counts_layer and 'counts' in adata.layers:
        adata.X = adata.layers['counts'].copy()
        print("Reset .X to .layers['counts'] (raw counts).")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    if model_name:
        print(f"Annotating with CellTypist model: {model_name}")
        # *** MANDATORY: always pass model= argument. ***
        # download_models() with NO model= downloads ALL 60 models (~10 GB) and
        # will crash with "No space left on device".
        models.download_models(model=model_name, force_update=False)
        model = models.Model.load(model=model_name)
        predictions = celltypist.annotate(adata, model=model, majority_voting=True)
        adata.obs['cell_type'] = predictions.predicted_labels['majority_voting'].values
        print("CellTypist annotation complete.")
        print(adata.obs['cell_type'].value_counts())

    elif custom_markers:
        print("Annotating using custom marker genes...")
        adata.obs['cell_type'] = 'Unknown'
        for cell_type, markers in custom_markers.items():
            valid = [g for g in markers if g in adata.var_names]
            if not valid:
                print(f"Warning: No valid markers for {cell_type}.")
                continue
            expr = adata[:, valid].X
            if hasattr(expr, 'toarray'):
                expr = expr.toarray()
            mask = expr.sum(axis=1) > 0
            adata.obs.loc[mask, 'cell_type'] = cell_type
        print(adata.obs['cell_type'].value_counts())
    else:
        raise ValueError("Specify either model_name or custom_markers.")

    if plot_umap and 'X_umap' in adata.obsm:
        sc.pl.umap(adata, color='cell_type', title='Cell Type Annotation', show=show)
    return adata


# Example usage (DO NOT IMPORT — rewrite this logic in your own code block):
#
# adata = ensure_gene_symbols(adata)   # always call before celltypist
# adata = annotate_with_celltypist(adata, model_name='Immune_All_Low.pkl')
