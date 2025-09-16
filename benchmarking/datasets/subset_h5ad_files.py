import os
import scanpy as sc
import anndata
import numpy as np

def subset_h5ad_files_in_dir(directory=".", subset_frac=0.1, seed=42):
    np.random.seed(seed)

    for filename in os.listdir(directory):
        if filename.endswith(".h5ad") and not filename.endswith(f"_{subset_frac}_subset.h5ad"):
            filepath = os.path.join(directory, filename)
            print(f"Processing {filename}")

            adata = sc.read_h5ad(filepath)
            n_cells = adata.n_obs
            n_subset = int(n_cells * subset_frac)

            subset_indices = np.random.choice(n_cells, size=n_subset, replace=False)
            adata_subset = adata[subset_indices].copy()

            subset_filename = filename.replace(".h5ad", "_subset.h5ad")
            subset_path = os.path.join(directory, subset_filename)

            adata_subset.write(subset_path)
            print(f"Saved subset to {subset_filename}")

if __name__ == "__main__":
    subset_h5ad_files_in_dir()