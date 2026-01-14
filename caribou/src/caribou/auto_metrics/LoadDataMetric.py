from typing import Dict, List

import anndata

from AutoMetric import AutoMetric


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
