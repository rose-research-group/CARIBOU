FULL_QC_PROMPT = """
Perform a full QC pipeline on the single-cell RNA-seq dataset at /workspace/dataset.h5ad.

Steps:
1) Load and inspect data; report initial cell and gene counts; preserve raw counts.
2) Compute QC metrics (n_genes_by_counts, total_counts, pct_counts_mt, pct_counts_in_top_20_genes) and log1p variants.
3) Run Scrublet with expected_doublet_rate=0.06; add doublet_score and predicted_doublet.
4) Apply MAD-based filtering: 5 MADs for log1p_total_counts, log1p_n_genes_by_counts, pct_counts_in_top_20_genes; 3 MADs upper-only for pct_counts_mt.
5) Remove predicted doublets.
6) Re-normalize and log1p after filtering; select HVGs; run PCA; neighbors; UMAP.
7) Save QC plots and a summary JSON to /workspace/outputs/.
8) Save filtered AnnData to /workspace/outputs/qc_filtered.h5ad and report final counts.

Show your code and report results.
"""
