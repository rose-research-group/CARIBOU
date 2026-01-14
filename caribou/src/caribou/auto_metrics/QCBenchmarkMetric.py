from typing import Dict

import anndata

from AutoMetric import AutoMetric


class QCBenchmarkMetric(AutoMetric):
    """
    Evaluates QC workflow results for benchmark comparison.
    """

    def metric(self, adata: anndata.AnnData) -> Dict:
        results = {
            "final_cell_count": adata.n_obs,
            "final_gene_count": adata.n_vars,
        }

        required_obs = [
            "doublet_score",
            "predicted_doublet",
            "n_genes_by_counts",
            "total_counts",
            "pct_counts_mt",
        ]
        results["obs_columns_present"] = {col: col in adata.obs.columns for col in required_obs}
        results["counts_layer_present"] = "counts" in adata.layers
        results["pca_present"] = "X_pca" in adata.obsm
        results["umap_present"] = "X_umap" in adata.obsm
        results["hvg_calculated"] = "highly_variable" in adata.var.columns

        return results

    def requirements(self) -> str:
        return "Requires QC'd AnnData with standard obs columns and embeddings."
