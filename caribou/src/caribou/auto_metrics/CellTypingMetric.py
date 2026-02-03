from caribou.auto_metrics.AutoMetric import AutoMetric
from caribou.auto_metrics.registry import MetricSpec, register_metric
import scanpy as sc
import celltypist
from celltypist import models
from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection
import scanpy.external as sce

class CellTypingMetric(AutoMetric):
    """
    This is a class that computes cell typing using CellTypist
    Then, it evaluates using metrics from Benchmarker class from SCIB's Metric Module.
    """
    def metric(self, adata):
        #scib_metrics Benchmarker
        bm = Benchmarker(
            adata,
            batch_key="batch",
            label_key="majority_voting",
            bio_conservation_metrics=BioConservation(nmi_ari_cluster_labels_leiden=True),
            batch_correction_metrics=None,
            embedding_obsm_keys=["X_pca","X_pca_harmony"], #need to check if it has such a label -> if it doesnt perform pca
            n_jobs=6,
        )
        bm.prepare()
        bm.benchmark()
        bm.plot_results_table(min_max_scale=False)
        bm.get_results()
    
    def requirements(self) -> str:
        return "Requires an AnnData object with 'batch' and 'majority_voting' in .obs and PCA embeddings in .obsm."


register_metric(
    MetricSpec(
        id="cell_typing",
        name="Cell Typing Benchmark",
        description="Runs CellTypist labeling and SCIB Benchmarker metrics.",
        inputs={"obs": "batch,majority_voting", "obsm": "X_pca,X_pca_harmony"},
        outputs={"scib_metrics": "Benchmarker output table serialized by scib-metrics."},
        tags=["celltyping", "benchmark"],
    ),
    CellTypingMetric,
)
