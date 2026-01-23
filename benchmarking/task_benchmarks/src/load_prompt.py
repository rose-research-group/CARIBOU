LOAD_PROMPT = """
Load the single-cell RNA-seq dataset at /workspace/dataset.h5ad using scanpy.

Steps:
1. Load the h5ad file.
2. Report the initial cell count (n_obs) and gene count (n_vars).
3. List available layers and confirm whether a 'counts' layer exists.
4. If no 'counts' layer exists, preserve raw counts in .layers['counts'].
5. Save a short summary JSON to /workspace/outputs/load_summary.json with:
   - n_obs
   - n_vars
   - layers
   - counts_layer_present
6. Print the summary.

Return executable Python code only, wrapped in ```python ... ``` blocks.
"""
