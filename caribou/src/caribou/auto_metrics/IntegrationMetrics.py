# --- New metric class using scib-metrics ------------------------------------
from scib_metrics.benchmark import Benchmarker
from typing import Dict

from caribou.auto_metrics.AutoMetric import AutoMetric
from caribou.auto_metrics.registry import MetricSpec, register_metric

BATCH_KEY = "batch"     # The batch key in adata.obs
LABEL_KEY = "cell_type" # The cell type key in adata.obs

class IntegrationMetric(AutoMetric):
    """
    Compute SCIB integration quality metrics on an AnnData object using scib_metrics.
    Returns a dictionary with three metrics:
        • batch_silhouette: How well batches mix (lower ≈ better)
        • celltype_silhouette: How well cell types separate (higher ≈ better)
        • isolated_label_f1: Label preservation in isolated clusters (higher ≈ better)
    """
    def __init__(self, embedding_key: str = "X_scVI"):
        self.embedding_key = embedding_key

    def metric(self, adata):
        bm = Benchmarker(
            adata,
            batch_key=BATCH_KEY,
            label_key=LABEL_KEY,
            embedding_obsm_keys=[self.embedding_key],        # list of embeddings to evaluate
        )
        bm.prepare()     # computes neighbors
        bm.benchmark()   # runs selected metrics
        results = bm.get_results()

        return results.to_dict()
    
    def requirements(self) -> str:
        return (
            f"Requires an AnnData object with '{BATCH_KEY}' and '{LABEL_KEY}' in .obs "
            f"and an embedding '{self.embedding_key}' in .obsm."
        )


register_metric(
    MetricSpec(
        id="integration_scvi",
        name="Integration (scVI)",
        description="SCIB metrics on the X_scVI embedding.",
        inputs={"obs": f"{BATCH_KEY},{LABEL_KEY}", "obsm": "X_scVI"},
        outputs={
            "batch_silhouette": "Batch mixing score (lower is better).",
            "celltype_silhouette": "Cell type separation (higher is better).",
            "isolated_label_f1": "Isolated label preservation (higher is better).",
        },
        tags=["integration", "scib"],
    ),
    IntegrationMetric,
)

register_metric(
    MetricSpec(
        id="integration_generic",
        name="Integration (Generic Embedding)",
        description="SCIB metrics on a generic 'integration' embedding.",
        inputs={"obs": f"{BATCH_KEY},{LABEL_KEY}", "obsm": "integration"},
        outputs={
            "batch_silhouette": "Batch mixing score (lower is better).",
            "celltype_silhouette": "Cell type separation (higher is better).",
            "isolated_label_f1": "Isolated label preservation (higher is better).",
        },
        tags=["integration", "scib"],
    ),
    IntegrationMetric,
    init_kwargs={"embedding_key": "integration"},
)
