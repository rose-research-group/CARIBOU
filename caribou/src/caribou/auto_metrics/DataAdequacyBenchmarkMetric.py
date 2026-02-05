from __future__ import annotations

from typing import Dict, Optional

import anndata

from caribou.auto_metrics.AutoMetric import AutoMetric
from caribou.auto_metrics.registry import MetricSpec, register_metric


class DataAdequacyBenchmarkMetric(AutoMetric):
    """
    Evaluate whether the input data has the minimum structure to fairly assess agents.
    Produces a simple adequacy score and success flag.
    """

    def __init__(
        self,
        batch_key: str = "batch",
        label_key: str = "cell_type",
        min_batches: int = 2,
        min_labels: int = 2,
        min_obs: int = 1,
        min_vars: int = 1,
        success_threshold: float = 0.8,
    ) -> None:
        self.batch_key = batch_key
        self.label_key = label_key
        self.min_batches = min_batches
        self.min_labels = min_labels
        self.min_obs = min_obs
        self.min_vars = min_vars
        self.success_threshold = success_threshold

    def _resolve_obs_key(self, adata: anndata.AnnData, primary: str, fallbacks: list[str]) -> Optional[str]:
        if primary in adata.obs:
            return primary
        for key in fallbacks:
            if key in adata.obs:
                return key
        return None

    def metric(self, adata: anndata.AnnData) -> Dict:
        batch_key = self._resolve_obs_key(adata, self.batch_key, ["batch", "sample", "donor", "batch_id"])
        label_key = self._resolve_obs_key(adata, self.label_key, ["cell_type", "celltype", "majority_voting"])

        n_obs = int(adata.n_obs)
        n_vars = int(adata.n_vars)
        counts_layer_present = "counts" in adata.layers
        raw_present = adata.raw is not None

        n_batches = None
        n_labels = None
        if batch_key is not None:
            n_batches = int(adata.obs[batch_key].nunique())
        if label_key is not None:
            n_labels = int(adata.obs[label_key].nunique())

        checks = {
            "has_batch_key": batch_key is not None,
            "has_label_key": label_key is not None,
            "counts_layer_present": counts_layer_present or raw_present,
            "min_obs": n_obs >= self.min_obs,
            "min_vars": n_vars >= self.min_vars,
            "min_batches": n_batches is not None and n_batches >= self.min_batches,
            "min_labels": n_labels is not None and n_labels >= self.min_labels,
        }

        passed = sum(1 for ok in checks.values() if ok)
        total = len(checks)
        adequacy_score = passed / total if total else 0.0
        success = adequacy_score >= self.success_threshold

        return {
            "success": success,
            "adequacy_score": adequacy_score,
            "batch_key": batch_key,
            "label_key": label_key,
            "n_obs": n_obs,
            "n_vars": n_vars,
            "n_batches": n_batches,
            "n_labels": n_labels,
            "counts_layer_present": counts_layer_present,
            "raw_present": raw_present,
            "checks": checks,
            "thresholds": {
                "min_batches": self.min_batches,
                "min_labels": self.min_labels,
                "min_obs": self.min_obs,
                "min_vars": self.min_vars,
                "success_threshold": self.success_threshold,
            },
        }

    def requirements(self) -> str:
        return "Requires AnnData with batch/cell-type labels and basic structure."


register_metric(
    MetricSpec(
        id="data_adequacy_benchmark",
        name="Data Adequacy Benchmark",
        description="Checks whether inputs are sufficient for fair agent evaluation.",
        inputs={
            "obs": "batch,cell_type",
            "layers": "counts",
        },
        outputs={
            "success": "Whether adequacy score meets threshold.",
            "adequacy_score": "Fraction of adequacy checks passed.",
            "n_obs": "Number of observations.",
            "n_vars": "Number of variables.",
            "n_batches": "Number of batch groups.",
            "n_labels": "Number of cell-type groups.",
        },
        tags=["data", "benchmark", "adequacy"],
    ),
    DataAdequacyBenchmarkMetric,
)
