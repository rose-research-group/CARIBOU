"""
Metric computation for integration benchmarks.

Focused on scib integration quality metrics only — no cell-typing metrics.
"""

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

import scib


# ---------------------------------------------------------------------------
# Pure-Python LISI (replaces scib's C++ knn_graph binary)
# ---------------------------------------------------------------------------

def _hbeta(D: np.ndarray, beta: float):
    """Shannon entropy H and softmax probabilities P for bandwidth beta."""
    P = np.exp(-D * beta)
    sumP = P.sum()
    if sumP == 0:
        return 0.0, np.zeros_like(D)
    P /= sumP
    H = -np.dot(P, np.log(P + 1e-300))
    return H, P


def _lisi_scores_from_knn(
    knn_indices: np.ndarray,
    knn_distances: np.ndarray,
    labels: np.ndarray,
    perplexity: float,
    tol: float = 1e-5,
) -> np.ndarray:
    """Compute per-cell LISI scores (= 1 / Simpson's index) in pure Python.

    Parameters
    ----------
    knn_indices   (n_cells, k) integer array of neighbor indices (0-based)
    knn_distances (n_cells, k) float array of neighbor distances
    labels        (n_cells,)   integer-coded label vector
    perplexity    target perplexity (≈ effective neighborhood size)
    """
    n_cells, k = knn_indices.shape
    n_labels = int(labels.max()) + 1
    logU = np.log(perplexity)
    lisi = np.zeros(n_cells)

    for i in range(n_cells):
        D = knn_distances[i].astype(float)
        idx = knn_indices[i]

        beta = 1.0
        betamin, betamax = -np.inf, np.inf
        H, P = _hbeta(D, beta)
        Hdiff = H - logU

        for _ in range(50):
            if abs(Hdiff) < tol:
                break
            if Hdiff > 0:
                betamin = beta
                beta = beta * 2 if betamax == np.inf else (beta + betamax) / 2
            else:
                betamax = beta
                beta = beta / 2 if betamin == -np.inf else (beta + betamin) / 2
            H, P = _hbeta(D, beta)
            Hdiff = H - logU

        if H == 0:
            lisi[i] = 1.0
            continue

        # Sum probabilities per label → Simpson's index
        batch_codes = labels[idx]
        sumP = np.zeros(n_labels)
        np.add.at(sumP, batch_codes, P)
        simpson = np.dot(sumP, sumP)
        lisi[i] = 1.0 / simpson if simpson > 0 else 1.0

    return lisi


def _extract_knn(adata: sc.AnnData, k: int):
    """Extract top-k neighbor indices and distances from scanpy neighbors graph."""
    dist_mat = adata.obsp["distances"]  # CSR, (n_cells, n_cells)
    n_cells = dist_mat.shape[0]
    indices = np.zeros((n_cells, k), dtype=int)
    distances = np.zeros((n_cells, k), dtype=float)

    for i in range(n_cells):
        row = dist_mat.getrow(i)
        row_idx = row.indices
        row_data = row.data
        if len(row_idx) == 0:
            continue
        # sort by distance, take top-k
        order = np.argsort(row_data)[:k]
        actual_k = len(order)
        indices[i, :actual_k] = row_idx[order]
        distances[i, :actual_k] = row_data[order]
        if actual_k < k:
            # fill remaining with first neighbor (shouldn't happen with n_neighbors≥k)
            indices[i, actual_k:] = row_idx[order[0]]
            distances[i, actual_k:] = row_data[order[0]]

    return indices, distances


def lisi_py(adata: sc.AnnData, obs_key: str, scale: bool, n_labels: int) -> float:
    """Pure-Python LISI (iLISI or cLISI) using the pre-computed scanpy kNN graph.

    Uses the k stored in the neighbors graph (typically 15). Perplexity = k/3.
    Scaled to [0,1] following scib convention:
      iLISI: (median - 1) / (n_batches - 1)  — higher is better
      cLISI: (n_labels - median) / (n_labels - 1)  — higher is better
    """
    if "neighbors" not in adata.uns:
        raise AttributeError("Pre-compute sc.pp.neighbors before calling lisi_py.")

    k = int(adata.uns["neighbors"]["params"].get("n_neighbors", 15))
    perplexity = max(1.0, k / 3)

    knn_idx, knn_dist = _extract_knn(adata, k)

    labels_cat = adata.obs[obs_key].astype("category")
    label_codes = labels_cat.cat.codes.values.astype(int)

    scores = _lisi_scores_from_knn(knn_idx, knn_dist, label_codes, perplexity)
    median_lisi = float(np.nanmedian(scores))

    if scale:
        if n_labels <= 1:
            return 0.0
        # iLISI uses (median - 1)/(n-1); cLISI uses (n - median)/(n-1)
        # Caller chooses direction via n_labels sign convention
        return (median_lisi - 1) / (n_labels - 1)
    return median_lisi


# ---------------------------------------------------------------------------
# Presence checks
# ---------------------------------------------------------------------------

def check_required_keys(car: sc.AnnData) -> Dict[str, bool]:
    return {
        "has_batch":       "batch"       in car.obs.columns,
        "has_integration": "integration" in car.obsm,
        "has_umap":        "X_umap"      in car.obsm,
        "has_pca":         "X_pca"       in car.obsm,
        "has_counts":      "counts"      in car.layers,
    }


# ---------------------------------------------------------------------------
# QC metrics
# ---------------------------------------------------------------------------

def qc_metrics(car: sc.AnnData, input_n_cells: Optional[int] = None) -> Dict:
    result: Dict = {}
    if input_n_cells and input_n_cells > 0:
        result["qc_filtering_rate"] = round(1.0 - car.n_obs / input_n_cells, 4)
    for col, key in [
        ("pct_counts_mt",      "median_pct_mt"),
        ("n_genes_by_counts",  "median_genes_per_cell"),
        ("total_counts",       "median_total_counts"),
    ]:
        if col in car.obs.columns:
            result[key] = float(np.median(car.obs[col].values))
    if "predicted_doublet" in car.obs.columns:
        n_doublets = int(car.obs["predicted_doublet"].astype(bool).sum())
        result["doublet_removal_rate"] = round(n_doublets / car.n_obs, 4) if car.n_obs else None
    return result


# ---------------------------------------------------------------------------
# Gene expression fidelity
# ---------------------------------------------------------------------------

def gene_expression_metrics(ref_aligned: sc.AnnData, car_aligned: sc.AnnData) -> Dict:
    """Spearman r of per-gene mean log-expression (CARIBOU vs reference)."""
    def _mat(adata):
        m = adata.layers["counts"] if "counts" in adata.layers else adata.X
        return m.toarray() if issparse(m) else m

    if ref_aligned.n_vars < 10:
        return {}

    ref_mean = np.log1p(_mat(ref_aligned).mean(axis=0).flatten())
    car_mean = np.log1p(_mat(car_aligned).mean(axis=0).flatten())
    r, p = spearmanr(ref_mean, car_mean)
    return {
        "gene_expr_spearman_r": round(float(r), 4),
        "gene_expr_spearman_p": float(p),
    }


# ---------------------------------------------------------------------------
# Embedding kNN overlap
# ---------------------------------------------------------------------------

def embedding_knn_overlap(
    ref_aligned: sc.AnnData,
    car_aligned: sc.AnnData,
    k: int = 15,
) -> Dict:
    """Fraction of k-NN shared between CARIBOU and reference embeddings.

    Asks: does CARIBOU reproduce the same neighbourhood structure as the
    ABA-processed reference?
    """
    result = {}
    for key in ("X_pca", "X_umap"):
        if key not in ref_aligned.obsm or key not in car_aligned.obsm:
            continue
        ref_emb = ref_aligned.obsm[key]
        car_emb = car_aligned.obsm[key]
        if ref_emb.shape[0] < k + 1 or car_emb.shape[0] < k + 1:
            continue
        nn_ref = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(ref_emb)
        nn_car = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(car_emb)
        _, idx_ref = nn_ref.kneighbors(ref_emb)
        _, idx_car = nn_car.kneighbors(car_emb)
        overlaps = [
            len(set(r.tolist()) & set(c.tolist())) / k
            for r, c in zip(idx_ref, idx_car)
        ]
        label = key.replace("X_", "")
        result[f"{label}_knn_overlap"] = round(float(np.mean(overlaps)), 4)
    return result


# ---------------------------------------------------------------------------
# scib integration metrics
# ---------------------------------------------------------------------------

def scib_integration_metrics(
    adata: sc.AnnData,
    batch_key: str,
    celltype_key: Optional[str],
    embedding_key: str,
    label: str,
) -> Dict:
    """Compute scib batch-correction and bio-conservation metrics.

    Parameters
    ----------
    adata         AnnData with embedding in obsm[embedding_key]
    batch_key     obs column with batch/sample labels
    celltype_key  obs column with cell-type labels (for bio metrics; None to skip)
    embedding_key obsm key to evaluate
    label         prefix for result keys, e.g. 'car' or 'ref'

    Graph-based metrics (iLISI, cLISI) use lisi_py() — a pure-Python LISI
    implementation that reads the pre-computed sc.pp.neighbors graph directly,
    bypassing scib's C++ knn_graph binary (which requires GLIBC ≥ 2.34).
    """
    result: Dict = {}

    if batch_key not in adata.obs.columns:
        raise KeyError(f"batch_key '{batch_key}' not found in adata.obs")
    if embedding_key not in adata.obsm:
        raise KeyError(f"embedding_key '{embedding_key}' not found in adata.obsm")

    n_batches = adata.obs[batch_key].nunique()
    if n_batches < 2 or adata.n_obs < 50:
        print(f"  [{label}] Skipping scib: n_batches={n_batches}, n_cells={adata.n_obs}")
        return result

    adata_work = adata.copy()
    sc.pp.neighbors(adata_work, use_rep=embedding_key, n_neighbors=15)

    # ── Batch correction ──────────────────────────────────────────────────────
    try:
        asw_b = scib.metrics.silhouette_batch(
            adata_work,
            batch_key=batch_key,
            label_key=celltype_key if celltype_key and celltype_key in adata_work.obs.columns
                      else batch_key,
            embed=embedding_key,
            verbose=False,
        )
        result[f"{label}_asw_batch"] = round(float(asw_b), 4)
    except Exception as e:
        print(f"  [{label}] silhouette_batch failed: {e}")

    try:
        gc = scib.metrics.graph_connectivity(adata_work, label_key=batch_key)
        result[f"{label}_graph_connectivity"] = round(float(gc), 4)
    except Exception as e:
        print(f"  [{label}] graph_connectivity failed: {e}")

    try:
        n_batches = adata_work.obs[batch_key].nunique()
        ilisi = lisi_py(adata_work, batch_key, scale=True, n_labels=n_batches)
        result[f"{label}_ilisi"] = round(float(ilisi), 4)
    except Exception as e:
        print(f"  [{label}] ilisi failed: {e}")

    # ── Bio conservation ──────────────────────────────────────────────────────
    if celltype_key and celltype_key in adata_work.obs.columns:
        ct_vals = adata_work.obs[celltype_key].dropna()
        if ct_vals.nunique() >= 2:
            try:
                asw_ct = scib.metrics.silhouette(
                    adata_work, label_key=celltype_key, embed=embedding_key,
                )
                result[f"{label}_asw_celltype"] = round(float(asw_ct), 4)
            except Exception as e:
                print(f"  [{label}] silhouette (celltype) failed: {e}")

            try:
                n_ct = adata_work.obs[celltype_key].nunique()
                # cLISI: (n_labels - median) / (n_labels - 1) — use negative n_labels
                # to distinguish from iLISI; compute manually:
                k_val = int(adata_work.uns["neighbors"]["params"].get("n_neighbors", 15))
                perp = max(1.0, k_val / 3)
                knn_idx, knn_dist = _extract_knn(adata_work, k_val)
                ct_codes = adata_work.obs[celltype_key].astype("category").cat.codes.values.astype(int)
                scores = _lisi_scores_from_knn(knn_idx, knn_dist, ct_codes, perp)
                median_lisi = float(np.nanmedian(scores))
                clisi = (n_ct - median_lisi) / (n_ct - 1) if n_ct > 1 else 0.0
                result[f"{label}_clisi"] = round(float(clisi), 4)
            except Exception as e:
                print(f"  [{label}] clisi failed: {e}")

    return result


# ---------------------------------------------------------------------------
# Single-run evaluation
# ---------------------------------------------------------------------------

def evaluate_run(
    run_dir: Path,
    cfg: Dict,
    raw_ref: sc.AnnData,
    proc_ref: sc.AnnData,
    output_dir: Path,
) -> Dict:
    """Evaluate one CARIBOU run. Writes metrics.json into output_dir/{run_name}/."""
    from .data_loader import (
        load_run_metadata,
        find_output_h5ad,
        align,
        align_processed_ref,
    )

    ref_ct_key   = cfg["reference_celltype_key"]
    barcode_join = cfg.get("barcode_join")
    input_n_cells = cfg.get("input_n_cells")
    scib_cfg     = cfg.get("scib", {})

    meta = load_run_metadata(run_dir)
    base = {
        "run_name":  run_dir.name,
        "dataset":   cfg.get("id"),
        "llm":       meta.get("llm_backend"),
        "mode":      meta.get("mode"),
        "runtime_s": meta.get("runtime_seconds"),
        "num_turns": meta.get("num_turns"),
    }

    output_h5ad = find_output_h5ad(run_dir)
    if output_h5ad is None:
        return {**base, "success": False, "error": "No output h5ad found"}

    try:
        car = sc.read_h5ad(output_h5ad)
    except Exception as e:
        return {**base, "success": False, "error": str(e)}

    keys   = check_required_keys(car)
    qc_met = qc_metrics(car, input_n_cells)

    # ── Align with raw reference (gene expression + ref-label projection) ─────
    try:
        ref_al, car_al, n_cells, n_genes = align(raw_ref, car, barcode_join)
    except ValueError as e:
        return {
            **base, "success": False, "error": str(e), **keys, **qc_met,
            "n_cells_reference": raw_ref.n_obs, "n_cells_caribou": car.n_obs,
        }

    result = {
        **base,
        "success":           True,
        "n_cells_reference": raw_ref.n_obs,
        "n_cells_caribou":   car.n_obs,
        "n_common_cells":    n_cells,
        "n_genes_reference": raw_ref.n_vars,
        "n_genes_caribou":   car.n_vars,
        "n_common_genes":    n_genes,
        "cell_count_ratio":  car.n_obs / raw_ref.n_obs if raw_ref.n_obs else None,
        **keys,
        **qc_met,
    }

    if n_genes > 0:
        result.update(gene_expression_metrics(ref_al, car_al))

    # ── Aligned processed reference (embedding kNN overlap) ───────────────────
    try:
        proc_al, car_proc_al, n_common_proc = align_processed_ref(
            proc_ref, car, barcode_join
        )
        result["n_common_cells_proc_ref"] = n_common_proc
        result.update(embedding_knn_overlap(proc_al, car_proc_al))
    except ValueError as e:
        print(f"    WARNING: processed reference alignment failed: {e}")

    # ── scib on CARIBOU embedding (bio metrics use reference cell type labels) ─
    car_batch_key = scib_cfg.get("caribou_batch_key", "sample")
    car_embed_key = scib_cfg.get("caribou_embedding_key", "X_pca")
    if "integration" in car.obsm:
        car_embed_key = "integration"

    if car_batch_key in car_al.obs.columns and car_embed_key in car_al.obsm:
        # Project reference cell type labels onto aligned CARIBOU cells
        car_for_bio = car_al.copy()
        if ref_ct_key in ref_al.obs.columns:
            car_for_bio.obs["_ref_celltype"] = ref_al.obs[ref_ct_key].values
            car_for_bio = car_for_bio[car_for_bio.obs["_ref_celltype"].notna()].copy()
            bio_ct_key = "_ref_celltype"
        else:
            bio_ct_key = None

        result.update(
            scib_integration_metrics(
                car_for_bio, car_batch_key, bio_ct_key, car_embed_key, label="car"
            )
        )

        # Save UMAP coordinates for visualization
        if "X_umap" in car_for_bio.obsm:
            run_out_dir_early = output_dir / run_dir.name
            run_out_dir_early.mkdir(parents=True, exist_ok=True)
            np.savez(
                run_out_dir_early / "umap_coords.npz",
                coords=car_for_bio.obsm["X_umap"],
                labels=car_for_bio.obs["_ref_celltype"].astype(str).values
                       if bio_ct_key == "_ref_celltype"
                       else np.full(len(car_for_bio), ""),
                batch=car_for_bio.obs[car_batch_key].astype(str).values
                      if car_batch_key in car_for_bio.obs else np.full(len(car_for_bio), ""),
            )

    # Write per-run metrics
    run_out_dir = output_dir / run_dir.name
    run_out_dir.mkdir(parents=True, exist_ok=True)
    (run_out_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    return result


# ---------------------------------------------------------------------------
# Reference baseline
# ---------------------------------------------------------------------------

def compute_reference_baseline(cfg: Dict, proc_ref: sc.AnnData) -> Dict:
    """Compute scib metrics on the consortium-processed reference embedding.

    Uses reference cell type labels for bio-conservation metrics.
    NaN-labeled cells are dropped before computing silhouette.
    """
    scib_cfg      = cfg.get("scib", {})
    ref_batch_key = scib_cfg.get("reference_batch_key", "library_label")
    ref_ct_key    = cfg["reference_celltype_key"]
    ref_embed_key = scib_cfg.get("reference_embedding_key", "X_pca")

    # Drop cells with no cell type label before computing bio metrics
    proc = proc_ref.copy()
    if ref_ct_key in proc.obs.columns:
        proc = proc[proc.obs[ref_ct_key].notna()].copy()

    print(f"  Computing reference baseline scib metrics "
          f"(embedding={ref_embed_key}, n_cells={proc.n_obs:,}) …")
    baseline = scib_integration_metrics(
        proc, ref_batch_key, ref_ct_key, ref_embed_key, label="ref"
    )
    print(f"  Reference baseline: {baseline}")
    return baseline
