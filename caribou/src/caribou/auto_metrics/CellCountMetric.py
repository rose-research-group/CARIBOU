from caribou.auto_metrics.AutoMetric import AutoMetric
from caribou.auto_metrics.registry import MetricSpec, register_metric
import scanpy as sc

class CellCountMetric(AutoMetric):
    """
    A simple metric to count the number of cells and genes.
    """
    def metric(self, adata) -> dict:
        num_cells = adata.n_obs
        num_genes = adata.n_vars
        
        return {
            "Number of Cells": num_cells,
            "Number of Genes": num_genes
        }
    
    def requirements(self) -> str:
        return "Requires an AnnData object with .n_obs and .n_vars attributes."


register_metric(
    MetricSpec(
        id="cell_count",
        name="Cell/Gene Count",
        description="Counts cells and genes in the AnnData object.",
        inputs={"adata": ".n_obs,.n_vars"},
        outputs={
            "Number of Cells": "Total number of observations.",
            "Number of Genes": "Total number of variables.",
        },
        tags=["qc", "summary"],
    ),
    CellCountMetric,
)
