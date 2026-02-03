from typing import Dict

import anndata

from caribou.auto_metrics.AutoMetric import AutoMetric
from caribou.auto_metrics.registry import MetricSpec, register_metric


class FullQCBenchmarkMetric(AutoMetric):
    """
    Evaluates full QC workflow results for benchmark comparison.
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
            "pct_counts_in_top_20_genes",
            "log1p_total_counts",
            "log1p_n_genes_by_counts",
        ]
        results["obs_columns_present"] = {col: col in adata.obs.columns for col in required_obs}
        results["counts_layer_present"] = "counts" in adata.layers
        results["pca_present"] = "X_pca" in adata.obsm
        results["umap_present"] = "X_umap" in adata.obsm
        results["hvg_calculated"] = "highly_variable" in adata.var.columns

        return results

    def requirements(self) -> str:
        return "Requires QC'd AnnData with doublet/QC obs columns, embeddings, and HVGs."


register_metric(
    MetricSpec(
        id="full_qc_benchmark",
        name="Full QC Benchmark",
        description="Checks for complete QC outputs including doublets, metrics, and embeddings.",
        inputs={
            "obs": "doublet_score,predicted_doublet,n_genes_by_counts,total_counts,pct_counts_mt,pct_counts_in_top_20_genes,log1p_total_counts,log1p_n_genes_by_counts",
            "obsm": "X_pca,X_umap",
            "layers": "counts",
            "var": "highly_variable",
        },
        outputs={
            "final_cell_count": "Number of observations after QC.",
            "final_gene_count": "Number of variables after QC.",
            "obs_columns_present": "Presence map for required .obs columns.",
            "counts_layer_present": "Whether .layers['counts'] exists.",
            "pca_present": "Whether .obsm['X_pca'] exists.",
            "umap_present": "Whether .obsm['X_umap'] exists.",
            "hvg_calculated": "Whether .var['highly_variable'] exists.",
        },
        tags=["qc", "doublet", "benchmark"],
    ),
    FullQCBenchmarkMetric,
)
