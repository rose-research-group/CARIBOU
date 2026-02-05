from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from caribou.auto_metrics.AutoMetric import AutoMetric
from caribou.auto_metrics.registry import MetricSpec, register_metric


class DEGBenchmarkMetric(AutoMetric):
    """
    Evaluate DEG output quality using schema checks and signal sanity tests.
    Expects /workspace/outputs/deg_results.csv to exist.
    """

    def __init__(
        self,
        results_path: str = "/workspace/outputs/deg_results.csv",
        min_clusters: int = 2,
        min_genes_per_cluster: int = 10,
        min_significant_fraction: float = 0.05,
        min_median_abs_logfc: float = 0.25,
        max_top_jaccard: float = 0.5,
        top_n: int = 10,
    ) -> None:
        self.results_path = results_path
        self.min_clusters = min_clusters
        self.min_genes_per_cluster = min_genes_per_cluster
        self.min_significant_fraction = min_significant_fraction
        self.min_median_abs_logfc = min_median_abs_logfc
        self.max_top_jaccard = max_top_jaccard
        self.top_n = top_n

    def _safe_float(self, value: object) -> Optional[float]:
        try:
            if value is None:
                return None
            return float(value)
        except Exception:
            return None

    def _load_results(self) -> Optional[pd.DataFrame]:
        try:
            df = pd.read_csv(self.results_path)
        except Exception:
            return None
        return df

    def metric(self, adata) -> Dict:
        results: Dict[str, object] = {
            "results_path": self.results_path,
        }

        df = self._load_results()
        if df is None or df.empty:
            results["success"] = False
            results["error"] = "deg_results_missing_or_empty"
            return results

        # Normalize column names
        cols = {c.lower(): c for c in df.columns}
        required = ["cluster", "gene", "pval", "pval_adj", "scores"]
        missing = [col for col in required if col not in cols]
        if missing:
            results["success"] = False
            results["error"] = f"missing_columns: {missing}"
            return results

        logfc_col = None
        for candidate in ("log2fc", "logfoldchange", "logfc", "log2_fold_change"):
            if candidate in cols:
                logfc_col = cols[candidate]
                break
        if logfc_col is None:
            results["success"] = False
            results["error"] = "missing_logfc"
            return results

        cluster_col = cols["cluster"]
        gene_col = cols["gene"]
        pval_col = cols["pval"]
        pval_adj_col = cols["pval_adj"]

        df = df.copy()
        df["_logfc"] = pd.to_numeric(df[logfc_col], errors="coerce")
        df["_pval_adj"] = pd.to_numeric(df[pval_adj_col], errors="coerce")

        cluster_counts = df[cluster_col].value_counts()
        n_clusters = int(cluster_counts.shape[0])
        min_genes_cluster = int(cluster_counts.min()) if not cluster_counts.empty else 0

        significant_fraction = float((df["_pval_adj"] < 0.05).mean()) if df["_pval_adj"].notna().any() else 0.0
        median_abs_logfc = float(df["_logfc"].abs().median()) if df["_logfc"].notna().any() else 0.0

        # Top-N Jaccard overlap
        top_sets = []
        for cluster, group in df.groupby(cluster_col):
            group_sorted = group.sort_values(by=pval_adj_col, ascending=True)
            top_genes = list(group_sorted[gene_col].astype(str).head(self.top_n))
            top_sets.append(set(top_genes))

        max_jaccard = 0.0
        for i in range(len(top_sets)):
            for j in range(i + 1, len(top_sets)):
                a, b = top_sets[i], top_sets[j]
                if not a and not b:
                    continue
                denom = len(a | b)
                if denom == 0:
                    continue
                jaccard = len(a & b) / denom
                max_jaccard = max(max_jaccard, jaccard)

        checks = {
            "min_clusters": n_clusters >= self.min_clusters,
            "min_genes_per_cluster": min_genes_cluster >= self.min_genes_per_cluster,
            "significant_fraction": significant_fraction >= self.min_significant_fraction,
            "median_abs_logfc": median_abs_logfc >= self.min_median_abs_logfc,
            "max_top_jaccard": max_jaccard <= self.max_top_jaccard,
        }

        success = all(checks.values())

        results.update(
            {
                "success": success,
                "n_clusters": n_clusters,
                "min_genes_per_cluster": min_genes_cluster,
                "significant_fraction": significant_fraction,
                "median_abs_logfc": median_abs_logfc,
                "max_top_jaccard": max_jaccard,
                "checks": checks,
                "thresholds": {
                    "min_clusters": self.min_clusters,
                    "min_genes_per_cluster": self.min_genes_per_cluster,
                    "min_significant_fraction": self.min_significant_fraction,
                    "min_median_abs_logfc": self.min_median_abs_logfc,
                    "max_top_jaccard": self.max_top_jaccard,
                    "top_n": self.top_n,
                },
            }
        )
        return results

    def requirements(self) -> str:
        return "Requires /workspace/outputs/deg_results.csv with DEG columns."


register_metric(
    MetricSpec(
        id="deg_benchmark",
        name="DEG Benchmark",
        description="Validates DEG output quality using schema and signal checks.",
        inputs={
            "files": "/workspace/outputs/deg_results.csv",
        },
        outputs={
            "success": "Whether DEG results pass quality checks.",
            "n_clusters": "Number of clusters with DEG entries.",
            "min_genes_per_cluster": "Minimum gene count per cluster.",
            "significant_fraction": "Fraction of genes with pval_adj < 0.05.",
            "median_abs_logfc": "Median absolute log fold-change.",
            "max_top_jaccard": "Max Jaccard overlap of top genes between clusters.",
        },
        tags=["deg", "benchmark"],
    ),
    DEGBenchmarkMetric,
)
