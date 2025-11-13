import numpy as np
from scipy.stats import median_abs_deviation as mad

def mad_outlier(adata, metric, nmads, upper_only=False):
    M = adata.obs[metric]
    med = np.median(M)
    mad_val = mad(M)  

    if not upper_only:
        return (M < med - nmads * mad_val) | (M > med + nmads * mad_val)
    
    return M > med + nmads * mad_val

def auto_QC(adata):
    outliers = (
        mad_outlier(adata, 'log1p_total_counts', 5)
        | mad_outlier(adata, 'log1p_n_genes_by_counts', 5)
        | mad_outlier(adata, 'pct_counts_in_top_20_genes', 5)
        | mad_outlier(adata, 'pct_counts_mt', 3, upper_only=True)
    )

    adata = adata[~outliers].copy()
    return adata
