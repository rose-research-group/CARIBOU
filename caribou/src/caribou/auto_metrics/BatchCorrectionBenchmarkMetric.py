from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from scib_metrics.benchmark import Benchmarker

from caribou.auto_metrics.AutoMetric import AutoMetric
from caribou.auto_metrics.registry import MetricSpec, register_metric


class BatchCorrectionBenchmarkMetric(AutoMetric):
    """
    Compare baseline vs integrated embeddings for batch correction quality.

    Success rule:
      - batch_silhouette improves by >= min_batch_improvement (lower is better)
      - celltype_silhouette does not drop by more than max_celltype_drop
      - isolated_label_f1 does not drop by more than max_isolated_drop
    """

    def __init__(
        self,
        baseline_key: str = "X_pca",
        integrated_key: str = "integration",
        batch_key: str = "batch",
        label_key: str = "cell_type",
        min_batch_improvement: float = 0.05,
        max_celltype_drop: float = 0.02,
        max_isolated_drop: float = 0.05,
    ) -> None:
        self.baseline_key = baseline_key
        self.integrated_key = integrated_key
        self.batch_key = batch_key
        self.label_key = label_key
        self.min_batch_improvement = min_batch_improvement
        self.max_celltype_drop = max_celltype_drop
        self.max_isolated_drop = max_isolated_drop

    def _resolve_obsm_key(self, adata, primary: str, fallbacks: list[str]) -> Optional[str]:
        if primary in adata.obsm:
            return primary
        for key in fallbacks:
            if key in adata.obsm:
                return key
        return None

    def _resolve_obs_key(self, adata, primary: str, fallbacks: list[str]) -> Optional[str]:
        if primary in adata.obs:
            return primary
        for key in fallbacks:
            if key in adata.obs:
                return key
        return None

    def _safe_float(self, value: object) -> Optional[float]:
        try:
            if value is None:
                return None
            if isinstance(value, (int, float)):
                if np.isnan(value):
                    return None
                return float(value)
            return float(value)
        except Exception:
            return None

    def _extract_embedding_metrics(self, results_df, baseline_key: str, integrated_key: str) -> Optional[Dict[str, Dict[str, object]]]:
        if hasattr(results_df, "columns") and baseline_key in results_df.columns and integrated_key in results_df.columns:
            baseline = results_df[baseline_key].to_dict()
            integrated = results_df[integrated_key].to_dict()
            return {"baseline": baseline, "integrated": integrated}
        if hasattr(results_df, "index") and baseline_key in results_df.index and integrated_key in results_df.index:
            baseline = results_df.loc[baseline_key].to_dict()
            integrated = results_df.loc[integrated_key].to_dict()
            return {"baseline": baseline, "integrated": integrated}
        return None

    def metric(self, adata) -> Dict:
        baseline_key = self._resolve_obsm_key(
            adata,
            self.baseline_key,
            ["X_pca"],
        )
        integrated_key = self._resolve_obsm_key(
            adata,
            self.integrated_key,
            ["integration", "X_pca_harmony", "X_scVI"],
        )
        batch_key = self._resolve_obs_key(adata, self.batch_key, ["batch", "sample", "donor"])
        label_key = self._resolve_obs_key(adata, self.label_key, ["cell_type", "celltype", "majority_voting"])

        results: Dict[str, object] = {
            "baseline_embedding": baseline_key,
            "integrated_embedding": integrated_key,
            "batch_key": batch_key,
            "label_key": label_key,
            "thresholds": {
                "min_batch_improvement": self.min_batch_improvement,
                "max_celltype_drop": self.max_celltype_drop,
                "max_isolated_drop": self.max_isolated_drop,
            },
        }

        missing = [
            name
            for name, value in (
                ("baseline_embedding", baseline_key),
                ("integrated_embedding", integrated_key),
                ("batch_key", batch_key),
                ("label_key", label_key),
            )
            if value is None
        ]
        if missing:
            results["error"] = f"Missing required keys: {', '.join(missing)}"
            results["success"] = False
            return results

        bm = Benchmarker(
            adata,
            batch_key=batch_key,
            label_key=label_key,
            embedding_obsm_keys=[baseline_key, integrated_key],
        )
        bm.prepare()
        bm.benchmark()
        results_df = bm.get_results()
        extracted = self._extract_embedding_metrics(results_df, baseline_key, integrated_key)
        if extracted is None:
            results["error"] = "Unable to extract embedding metrics from scib_metrics output."
            results["success"] = False
            return results

        baseline_metrics = extracted["baseline"]
        integrated_metrics = extracted["integrated"]

        batch_base = self._safe_float(baseline_metrics.get("batch_silhouette"))
        batch_int = self._safe_float(integrated_metrics.get("batch_silhouette"))
        cell_base = self._safe_float(baseline_metrics.get("celltype_silhouette"))
        cell_int = self._safe_float(integrated_metrics.get("celltype_silhouette"))
        iso_base = self._safe_float(baseline_metrics.get("isolated_label_f1"))
        iso_int = self._safe_float(integrated_metrics.get("isolated_label_f1"))

        batch_delta = batch_base - batch_int if batch_base is not None and batch_int is not None else None
        cell_delta = cell_int - cell_base if cell_base is not None and cell_int is not None else None
        iso_delta = iso_int - iso_base if iso_base is not None and iso_int is not None else None

        results.update(
            {
                "batch_silhouette_baseline": batch_base,
                "batch_silhouette_integrated": batch_int,
                "batch_silhouette_delta": batch_delta,
                "celltype_silhouette_baseline": cell_base,
                "celltype_silhouette_integrated": cell_int,
                "celltype_silhouette_delta": cell_delta,
                "isolated_label_f1_baseline": iso_base,
                "isolated_label_f1_integrated": iso_int,
                "isolated_label_f1_delta": iso_delta,
            }
        )

        if None in (batch_delta, cell_delta, iso_delta):
            results["success"] = False
            results["error"] = "Incomplete SCIB metrics; cannot score success."
            return results

        results["success"] = bool(
            batch_delta >= self.min_batch_improvement
            and cell_delta >= -self.max_celltype_drop
            and iso_delta >= -self.max_isolated_drop
        )
        return results

    def requirements(self) -> str:
        return (
            "Requires AnnData with batch/cell type labels in .obs and baseline/integration embeddings in .obsm."
        )


register_metric(
    MetricSpec(
        id="batch_correction_benchmark",
        name="Batch Correction Benchmark",
        description="Compares baseline vs integrated embeddings using SCIB metrics.",
        inputs={
            "obs": "batch,cell_type",
            "obsm": "X_pca,integration",
        },
        outputs={
            "batch_silhouette_baseline": "Batch silhouette for baseline embedding (lower is better).",
            "batch_silhouette_integrated": "Batch silhouette for integrated embedding (lower is better).",
            "batch_silhouette_delta": "Baseline minus integrated batch silhouette (positive is improvement).",
            "celltype_silhouette_baseline": "Cell type silhouette for baseline embedding (higher is better).",
            "celltype_silhouette_integrated": "Cell type silhouette for integrated embedding (higher is better).",
            "celltype_silhouette_delta": "Integrated minus baseline cell type silhouette (positive is improvement).",
            "isolated_label_f1_baseline": "Isolated label F1 for baseline embedding (higher is better).",
            "isolated_label_f1_integrated": "Isolated label F1 for integrated embedding (higher is better).",
            "isolated_label_f1_delta": "Integrated minus baseline isolated label F1 (positive is improvement).",
            "success": "Whether integration passes the quality thresholds.",
        },
        tags=["batch", "integration", "benchmark"],
    ),
    BatchCorrectionBenchmarkMetric,
)
