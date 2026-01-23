from typing import Dict, List

import anndata

from caribou.auto_metrics.AutoMetric import AutoMetric
from caribou.auto_metrics.registry import MetricSpec, register_metric


class LoadDataMetric(AutoMetric):
    """
    Evaluates basic data loading expectations.
    """

    def metric(self, adata: anndata.AnnData) -> Dict:
        layers: List[str] = list(adata.layers.keys())
        return {
            "n_obs": adata.n_obs,
            "n_vars": adata.n_vars,
            "layers": layers,
            "counts_layer_present": "counts" in adata.layers,
        }

    def requirements(self) -> str:
        return "Requires loaded AnnData with .n_obs/.n_vars and layers."


register_metric(
    MetricSpec(
        id="load_data",
        name="Load Data",
        description="Checks basic AnnData loading expectations.",
        inputs={
            "adata": ".n_obs,.n_vars,.layers",
        },
        outputs={
            "n_obs": "Number of observations.",
            "n_vars": "Number of variables.",
            "layers": "List of available layers.",
            "counts_layer_present": "Whether .layers['counts'] exists.",
        },
        tags=["load", "qc"],
    ),
    LoadDataMetric,
)
