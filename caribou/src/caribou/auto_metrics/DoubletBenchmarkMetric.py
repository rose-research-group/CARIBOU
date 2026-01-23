from typing import Dict

import anndata
import numpy as np

from caribou.auto_metrics.AutoMetric import AutoMetric
from caribou.auto_metrics.registry import MetricSpec, register_metric


class DoubletBenchmarkMetric(AutoMetric):
    """
    Evaluates doublet detection outputs for benchmark comparison.
    """

    def metric(self, adata: anndata.AnnData) -> Dict:
        results = {
            "n_obs": adata.n_obs,
            "n_vars": adata.n_vars,
            "doublet_score_present": "doublet_score" in adata.obs.columns,
            "predicted_doublet_present": "predicted_doublet" in adata.obs.columns,
        }

        if results["predicted_doublet_present"]:
            preds = adata.obs["predicted_doublet"]
            try:
                doublet_rate = float(np.mean(preds.astype(bool)))
            except Exception:
                doublet_rate = None
            results["predicted_doublet_rate"] = doublet_rate
        else:
            results["predicted_doublet_rate"] = None

        if results["doublet_score_present"]:
            try:
                results["doublet_score_mean"] = float(np.mean(adata.obs["doublet_score"]))
            except Exception:
                results["doublet_score_mean"] = None
        else:
            results["doublet_score_mean"] = None

        return results

    def requirements(self) -> str:
        return "Requires AnnData with 'doublet_score' and 'predicted_doublet' in .obs."


register_metric(
    MetricSpec(
        id="doublet_benchmark",
        name="Doublet Benchmark",
        description="Checks presence of doublet score/predictions and summarizes rates.",
        inputs={"obs": "doublet_score,predicted_doublet"},
        outputs={
            "n_obs": "Number of observations.",
            "n_vars": "Number of variables.",
            "doublet_score_present": "Whether .obs['doublet_score'] exists.",
            "predicted_doublet_present": "Whether .obs['predicted_doublet'] exists.",
            "predicted_doublet_rate": "Mean predicted doublet rate.",
            "doublet_score_mean": "Mean doublet score.",
        },
        tags=["doublet", "qc", "benchmark"],
    ),
    DoubletBenchmarkMetric,
)
