#!/usr/bin/env python3
"""
Universal CARIBOU comparison evaluator.

Reads dataset configs from ../datasets/<id>/config.json, finds completed runs
under ../results/<dataset>/, and computes quality metrics against the reference.

Usage:
    python evaluate.py                          # all datasets
    python evaluate.py --dataset aba_hippocampus
    python evaluate.py --dataset tsp_large_intestine
    python evaluate.py --dataset aba_hippocampus tsp_large_intestine
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    classification_report,
    confusion_matrix,
)
from sklearn.neighbors import NearestNeighbors


COMP_DIR     = Path(__file__).parent.parent
DATASETS_DIR = COMP_DIR / "datasets"
RESULTS_DIR  = COMP_DIR / "results"


# ---------------------------------------------------------------------------
# Dataset config helpers
# ---------------------------------------------------------------------------

def load_dataset_config(dataset_id: str) -> Dict:
    path = DATASETS_DIR / dataset_id / "config.json"
    if not path.exists():
        raise FileNotFoundError(f"No config for dataset '{dataset_id}': {path}")
    return json.loads(path.read_text())


def available_datasets() -> List[str]:
    return sorted(d.name for d in DATASETS_DIR.iterdir() if d.is_dir() and (d / "config.json").exists())


# ---------------------------------------------------------------------------
# Run metadata helpers
# ---------------------------------------------------------------------------

def _infer_meta_from_run_name(run_dir: Path) -> Dict:
    """Infer mode and llm_backend from the run directory name convention.

    Expected pattern: {llm}_{mode}_{slurm_job_id}
    e.g. deepseek_single_agent_11187432, chatgpt_full_system_no_mem_99999
    """
    MODES = ("full_system_no_mem", "full_system", "single_agent", "one_shot")
    LLMS  = ("deepseek", "chatgpt", "claude", "gpt4", "gpt-4")
    name  = run_dir.name
    meta: Dict = {}
    for mode in MODES:
        if mode in name:
            meta["mode"] = mode
            break
    for llm in LLMS:
        if name.lower().startswith(llm):
            meta["llm_backend"] = llm
            break
    return meta


def load_run_metadata(run_dir: Path) -> Dict:
    """Load run_metadata.json (new format) or fall back to legacy config.json + runtime.json.

    Any fields still missing after file-based lookup are inferred from the directory name.
    """
    meta_path = run_dir / "run_metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    else:
        meta = {}
        for fname in ("config.json", "runtime.json"):
            p = run_dir / fname
            if p.exists():
                meta.update(json.loads(p.read_text()))

    # Fill in any missing mode/llm from the run directory name
    inferred = _infer_meta_from_run_name(run_dir)
    for key in ("mode", "llm_backend"):
        if not meta.get(key):
            meta[key] = inferred.get(key)
    return meta


def find_output_h5ad(run_dir: Path) -> Optional[Path]:
    """Find the CARIBOU output — try root then recurse one level into subdirectories.

    Filename variants ordered by preference (most final first).
    """
    NAMES = (
        "annotated_dataset.h5ad",
        "dataset_annotated.h5ad",
        "data_annotated.h5ad",
        "adata_final_annotated.h5ad",
        "adata_final_complete.h5ad",
        "adata_annotated.h5ad",
        "adata_downstream_processed.h5ad",
        "adata_downstream.h5ad",
    )
    for name in NAMES:
        p = run_dir / name
        if p.exists():
            return p
    for subdir in sorted(run_dir.iterdir()):
        if subdir.is_dir() and not subdir.name.startswith("."):
            for name in NAMES:
                p = subdir / name
                if p.exists():
                    return p
    return None


# ---------------------------------------------------------------------------
# Cell alignment
# ---------------------------------------------------------------------------

def _build_compound_key(adata: sc.AnnData, barcode_col: Optional[str],
                         library_col: Optional[str], strip_suffix: bool,
                         index_extract: Optional[str] = None) -> "pd.Series":
    """Build a per-cell string key for joining two AnnData objects.

    For datasets where the cell index format differs between CARIBOU output and
    the reference, specify how to reconstruct a shared key:
      - strip_suffix:   strip the trailing '-<number>' from the index
      - barcode_col:    obs column containing the raw barcode (overrides index)
      - library_col:    obs column to append as '<barcode>|||<library>'
      - index_extract:  how to pull the raw barcode from the index when no
                        barcode_col is provided:
          "split_hyphen_first"    – take the part before the first '-'
                                    e.g. 'ACGT-1-SampleA' → 'ACGT'
          "split_underscore_last" – take the part after the last '_'
                                    e.g. 'SampleA_ACGT' → 'ACGT'
    """
    if barcode_col and barcode_col in adata.obs.columns:
        base = adata.obs[barcode_col].astype(str)
    else:
        base = pd.Series(adata.obs.index, index=adata.obs.index).astype(str)
        if strip_suffix:
            base = base.str.rsplit("-", n=1).str[0]
        elif index_extract == "split_hyphen_first":
            base = base.str.split("-").str[0]
        elif index_extract == "split_underscore_last":
            base = base.str.rsplit("_", n=1).str[1]

    if library_col and library_col in adata.obs.columns:
        key = base + "|||" + adata.obs[library_col].astype(str).values
    else:
        key = base

    return key


def align(ref: sc.AnnData, car: sc.AnnData, barcode_join: Optional[Dict] = None):
    """Align ref and car on common cells and genes.

    barcode_join (optional dict from dataset config) controls normalisation:
      {
        "caribou_strip_suffix":    true,             # strip trailing -N from CARIBOU index
        "caribou_index_extract":   "split_hyphen_first",  # extract barcode from CARIBOU index
        "caribou_sample_col":      "batch",          # obs col with library label in CARIBOU
        "reference_barcode_col":   "cell_barcode",   # obs col with raw barcode in reference
        "reference_index_extract": "split_underscore_last",  # extract barcode from ref index
        "reference_library_col":   "sample_id"       # obs col with library label in reference
      }
    When not provided, a direct index intersection is used.
    """
    if barcode_join:
        car_key = _build_compound_key(
            car,
            barcode_col=None,
            library_col=barcode_join.get("caribou_sample_col"),
            strip_suffix=barcode_join.get("caribou_strip_suffix", False),
            index_extract=barcode_join.get("caribou_index_extract"),
        )
        ref_key = _build_compound_key(
            ref,
            barcode_col=barcode_join.get("reference_barcode_col"),
            library_col=barcode_join.get("reference_library_col"),
            strip_suffix=False,
            index_extract=barcode_join.get("reference_index_extract"),
        )
        car_idx_map = pd.Series(car.obs.index, index=car_key.values)
        ref_idx_map = pd.Series(ref.obs.index, index=ref_key.values)
        common_keys = set(car_idx_map.index) & set(ref_idx_map.index)
        if not common_keys:
            raise ValueError("No common cells after barcode normalisation — check barcode_join config.")
        car_cells = car_idx_map[list(common_keys)].values
        ref_cells = ref_idx_map[list(common_keys)].values
    else:
        common_cells = ref.obs.index.intersection(car.obs.index)
        if len(common_cells) == 0:
            raise ValueError("No common cells after alignment — check barcode format.")
        car_cells = ref_cells = common_cells

    common_genes = ref.var.index.intersection(car.var.index)
    return (
        ref[ref_cells, common_genes].copy(),
        car[car_cells, common_genes].copy(),
        len(ref_cells),
        len(common_genes),
    )


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------

def check_required_keys(car: sc.AnnData) -> Dict[str, bool]:
    return {
        "has_batch":       "batch"       in car.obs.columns,
        "has_cluster":     "cluster"     in car.obs.columns,
        "has_cell_type":   "cell_type"   in car.obs.columns,
        "has_integration": "integration" in car.obsm,
        "has_umap":        "X_umap"      in car.obsm,
        "has_counts":      "counts"      in car.layers,
    }


def clustering_metrics(ref_labels, car_labels) -> Dict:
    return {
        "ari": adjusted_rand_score(ref_labels, car_labels),
        "nmi": normalized_mutual_info_score(ref_labels, car_labels),
    }


def hvg_overlap(ref: sc.AnnData, car: sc.AnnData) -> Dict:
    if "highly_variable" not in ref.var.columns or "highly_variable" not in car.var.columns:
        return {"hvg_jaccard": None, "hvg_ref": None, "hvg_caribou": None, "hvg_common": None}
    ref_hvg = set(ref.var_names[ref.var["highly_variable"]])
    car_hvg = set(car.var_names[car.var["highly_variable"]])
    inter = ref_hvg & car_hvg
    union = ref_hvg | car_hvg
    return {
        "hvg_jaccard": len(inter) / len(union) if union else 0,
        "hvg_ref":     len(ref_hvg),
        "hvg_caribou": len(car_hvg),
        "hvg_common":  len(inter),
    }


def qc_metrics(car: sc.AnnData, input_n_cells: Optional[int] = None) -> Dict:
    """QC-level metrics extracted from the CARIBOU output's .obs table.

    Returns filtering rate (cells removed vs input), median MT%, median genes/cell,
    and doublet removal rate where those columns are present.
    """
    result: Dict = {}

    if input_n_cells and input_n_cells > 0:
        result["qc_filtering_rate"] = round(1.0 - car.n_obs / input_n_cells, 4)

    for col, key in [
        ("pct_counts_mt",  "median_pct_mt"),
        ("n_genes_by_counts", "median_genes_per_cell"),
        ("total_counts",   "median_total_counts"),
    ]:
        if col in car.obs.columns:
            result[key] = float(np.median(car.obs[col].values))

    if "predicted_doublet" in car.obs.columns:
        n_doublets = int(car.obs["predicted_doublet"].astype(bool).sum())
        result["doublet_removal_rate"] = round(n_doublets / car.n_obs, 4) if car.n_obs else None

    return result


def gene_expression_metrics(ref_aligned: sc.AnnData, car_aligned: sc.AnnData) -> Dict:
    """Spearman correlation of per-gene mean log-expression between reference and CARIBOU.

    Uses raw counts from .layers['counts'] if available, otherwise falls back to .X.
    Applies log1p(mean) to get a comparable scale regardless of normalisation.
    A high correlation (r > 0.9) demonstrates that CARIBOU's data processing
    preserves the relative expression structure of the reference.
    """
    def _get_matrix(adata):
        if "counts" in adata.layers:
            mat = adata.layers["counts"]
        else:
            mat = adata.X
        if issparse(mat):
            mat = mat.toarray()
        return mat

    if ref_aligned.n_vars < 10:
        return {}

    try:
        ref_mat = _get_matrix(ref_aligned)
        car_mat = _get_matrix(car_aligned)
        ref_mean = np.log1p(ref_mat.mean(axis=0).flatten())
        car_mean = np.log1p(car_mat.mean(axis=0).flatten())
        r, p = spearmanr(ref_mean, car_mean)
        return {
            "gene_expr_spearman_r": round(float(r), 4),
            "gene_expr_spearman_p": float(p),
        }
    except Exception:
        return {}


def celltype_confusion_metrics(
    ref_labels: pd.Series,
    car_labels: pd.Series,
    coarse_mapping: Optional[Dict[str, str]] = None,
    caribou_mapping: Optional[Dict[str, str]] = None,
) -> Tuple[Dict, Dict]:
    """Per-cell-type precision, recall, F1 and confusion matrix.

    When coarse_mapping is provided, reference fine labels are collapsed to
    coarse groups before comparison (essential for TSP where the reference uses
    CellXGene ontology names but CARIBOU produces colloquial names).
    When caribou_mapping is provided, CARIBOU labels are normalised to the same
    coarse vocabulary so that e.g. 'Colonocytes' matches 'Colonocyte'.

    Returns:
        summary  – scalar metrics (macro_f1, weighted_f1) for results.json
        cm_data  – full confusion matrix dict for a separate per-run JSON file
    """
    # Reset indices so both Series are 0-based aligned (they come from different AnnData objects)
    ref = ref_labels.astype(str).reset_index(drop=True)
    car = car_labels.astype(str).reset_index(drop=True)

    if caribou_mapping:
        # Map CARIBOU labels to canonical coarse names; unmapped labels kept as-is
        car = car.map(lambda x: caribou_mapping.get(x, x))

    if coarse_mapping:
        ref = ref.map(lambda x: coarse_mapping.get(x, "Other"))
        # Drop cells mapped to "Other" — they have no valid coarse equivalent
        mask = (ref != "Other").values  # numpy array to avoid index-alignment issues
        ref = ref[mask]
        car = car[mask]

    # Drop remaining CARIBOU cells labelled "Other" from both sides
    if caribou_mapping or coarse_mapping:
        car_mask = (car != "Other").values
        ref = ref[car_mask]
        car = car[car_mask]

    if len(ref) == 0:
        return {}, {}

    all_labels = sorted(set(ref) | set(car))
    report = classification_report(ref, car, labels=all_labels,
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(ref, car, labels=all_labels)

    summary = {
        "macro_f1":    round(report["macro avg"]["f1-score"], 4),
        "weighted_f1": round(report["weighted avg"]["f1-score"], 4),
        "macro_precision": round(report["macro avg"]["precision"], 4),
        "macro_recall":    round(report["macro avg"]["recall"], 4),
    }

    per_type = {
        k: {
            "precision": round(v["precision"], 4),
            "recall":    round(v["recall"], 4),
            "f1":        round(v["f1-score"], 4),
            "support":   int(v["support"]),
        }
        for k, v in report.items()
        if k not in ("accuracy", "macro avg", "weighted avg")
    }

    cm_data = {
        "labels":           all_labels,
        "confusion_matrix": cm.tolist(),
        "per_type":         per_type,
    }

    return summary, cm_data


def embedding_metrics(
    ref_aligned: sc.AnnData,
    car_aligned: sc.AnnData,
    k: int = 15,
) -> Dict:
    """k-nearest-neighbour overlap between reference and CARIBOU embeddings.

    For each aligned cell, computes the fraction of its k-NN in the CARIBOU
    embedding that are also k-NN in the reference embedding.  This metric is
    rotation-invariant and more biologically meaningful than axis correlation.

    Computed for X_umap and X_pca where both objects have the key.
    """
    result = {}
    for key in ("X_umap", "X_pca"):
        if key not in ref_aligned.obsm or key not in car_aligned.obsm:
            continue
        ref_emb = ref_aligned.obsm[key]
        car_emb = car_aligned.obsm[key]
        if ref_emb.shape[0] < k + 1 or car_emb.shape[0] < k + 1:
            continue
        try:
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
        except Exception:
            pass
    return result


def population_metrics(
    ref: sc.AnnData, car: sc.AnnData, ref_ct_key: str,
    coarse_mapping: Optional[Dict[str, str]] = None,
) -> Dict:
    """Population-level comparison that doesn't require cell-level alignment.

    Computes metrics directly from the two obs tables, so it works even when
    barcode formats differ and align() fails.

    When `coarse_mapping` is provided (a dict from reference fine-label →
    coarse group name, stored in datasets config as "coarse_celltype_mapping"),
    the reference labels are collapsed before comparison.  This handles
    datasets like TSP where ontology-level reference names differ from the
    high-level labels an agent naturally produces.
    """
    result: Dict = {
        "cell_count_ratio": car.n_obs / ref.n_obs if ref.n_obs else None,
    }

    if ref_ct_key not in ref.obs.columns:
        return result

    ref_props = ref.obs[ref_ct_key].value_counts(normalize=True)
    ref_types = set(ref_props.index)
    result["ref_n_celltypes"] = len(ref_types)

    if "cell_type" not in car.obs.columns:
        result["car_n_celltypes"] = None
        return result

    car_props = car.obs["cell_type"].value_counts(normalize=True)
    car_types = set(car_props.index)
    result["car_n_celltypes"] = len(car_types)

    shared = ref_types & car_types
    result["n_shared_celltypes"]    = len(shared)
    result["celltype_name_overlap"] = len(shared) / len(ref_types) if ref_types else None

    if len(shared) >= 2:
        shared_list = sorted(shared)
        r_val, _ = pearsonr(
            [float(ref_props.get(ct, 0)) for ct in shared_list],
            [float(car_props.get(ct, 0)) for ct in shared_list],
        )
        result["celltype_prop_corr"] = float(r_val)

    if coarse_mapping:
        coarse_ref: Dict[str, float] = {}
        for fine_label, prop in ref_props.items():
            coarse = coarse_mapping.get(fine_label)
            if coarse and coarse != "Other":
                coarse_ref[coarse] = coarse_ref.get(coarse, 0.0) + float(prop)
        total = sum(coarse_ref.values()) or 1.0
        coarse_ref = {k: v / total for k, v in coarse_ref.items()}

        coarse_types = set(coarse_ref.keys())
        coarse_shared = coarse_types & car_types
        result["n_shared_celltypes_coarse"]    = len(coarse_shared)
        result["celltype_name_overlap_coarse"] = (
            len(coarse_shared) / len(coarse_types) if coarse_types else None
        )

        if len(coarse_shared) >= 2:
            shared_list = sorted(coarse_shared)
            r_val, _ = pearsonr(
                [coarse_ref.get(ct, 0.0)     for ct in shared_list],
                [float(car_props.get(ct, 0)) for ct in shared_list],
            )
            result["celltype_prop_corr_coarse"] = float(r_val)

    return result


# ---------------------------------------------------------------------------
# Single-run evaluation
# ---------------------------------------------------------------------------

def evaluate_run(
    run_dir: Path,
    reference: sc.AnnData,
    ref_celltype_key: str,
    barcode_join: Optional[Dict] = None,
    coarse_mapping: Optional[Dict[str, str]] = None,
    caribou_mapping: Optional[Dict[str, str]] = None,
    input_n_cells: Optional[int] = None,
    output_dir: Optional[Path] = None,
) -> Dict:
    meta = load_run_metadata(run_dir)
    output = find_output_h5ad(run_dir)

    base = {
        "run_name":  run_dir.name,
        "dataset":   meta.get("dataset"),
        "llm":       meta.get("llm_backend"),
        "mode":      meta.get("mode"),
        "runtime_s": meta.get("runtime_seconds"),
        "num_turns": meta.get("num_turns"),
    }

    if output is None:
        return {**base, "success": False, "error": "No output h5ad found"}

    try:
        car = sc.read_h5ad(output)
    except Exception as e:
        return {**base, "success": False, "error": str(e)}

    # Normalise cell_type column: accept common alternative names
    CELLTYPE_ALIASES = ("celltypist_prediction", "cell_type_predicted", "predicted_cell_type")
    if "cell_type" not in car.obs.columns:
        for alias in CELLTYPE_ALIASES:
            if alias in car.obs.columns:
                car.obs["cell_type"] = car.obs[alias]
                break

    keys    = check_required_keys(car)
    qc_met  = qc_metrics(car, input_n_cells)
    pop_met = population_metrics(reference, car, ref_celltype_key, coarse_mapping)

    try:
        ref_aligned, car_aligned, n_cells, n_genes = align(reference, car, barcode_join)
    except ValueError as e:
        return {**base, "success": False, "error": str(e), **keys, **qc_met, **pop_met,
                "n_cells_reference": reference.n_obs, "n_cells_caribou": car.n_obs}

    result = {
        **base,
        "success":           True,
        "n_cells_reference": reference.n_obs,
        "n_cells_caribou":   car.n_obs,
        "n_common_cells":    n_cells,
        "n_genes_reference": reference.n_vars,
        "n_genes_caribou":   car.n_vars,
        "n_common_genes":    n_genes,
        **keys,
        **qc_met,
        **pop_met,
    }

    # ── Gene expression correlation (requires aligned genes) ────────────────
    if n_genes > 0:
        result.update(gene_expression_metrics(ref_aligned, car_aligned))

    # ── Clustering agreement ────────────────────────────────────────────────
    has_ref_ct = ref_celltype_key in ref_aligned.obs.columns
    has_car_ct = "cell_type" in car_aligned.obs.columns

    if has_ref_ct and has_car_ct:
        cm_summary, cm_data = celltype_confusion_metrics(
            ref_aligned.obs[ref_celltype_key],
            car_aligned.obs["cell_type"],
            coarse_mapping,
            caribou_mapping,
        )
        result.update(cm_summary)

        # Save full confusion matrix separately to avoid bloating results.json
        if output_dir and cm_data:
            cm_path = output_dir / f"confusion_{run_dir.name}.json"
            cm_path.write_text(json.dumps(cm_data, indent=2))

        # Also compute ARI/NMI (on coarse-mapped labels for TSP)
        ref_lbl = ref_aligned.obs[ref_celltype_key].astype(str).reset_index(drop=True)
        car_lbl = car_aligned.obs["cell_type"].astype(str).reset_index(drop=True)
        if caribou_mapping:
            car_lbl = car_lbl.map(lambda x: caribou_mapping.get(x, x))
        if coarse_mapping:
            ref_lbl = ref_lbl.map(lambda x: coarse_mapping.get(x, "Other"))
            mask = (ref_lbl != "Other").values
            ref_lbl = ref_lbl[mask]
            car_lbl = car_lbl[mask]
        # Drop unresolved CARIBOU "Other" clusters from ARI too
        other_mask = (car_lbl != "Other").values
        ref_lbl = ref_lbl[other_mask]
        car_lbl = car_lbl[other_mask]
        if len(ref_lbl) > 0:
            cm = clustering_metrics(ref_lbl, car_lbl)
            result.update(cm)
            print(f"      ARI={cm['ari']:.3f}  NMI={cm['nmi']:.3f}  "
                  f"F1(macro)={result.get('macro_f1', float('nan')):.3f}")

    # ── HVG overlap ─────────────────────────────────────────────────────────
    result.update(hvg_overlap(ref_aligned, car_aligned))

    # ── Embedding quality ────────────────────────────────────────────────────
    result.update(embedding_metrics(ref_aligned, car_aligned))

    return result


# ---------------------------------------------------------------------------
# Dataset evaluation loop
# ---------------------------------------------------------------------------

def _attach_metadata_celltypes(reference: sc.AnnData, cfg: Dict) -> None:
    """
    Join cell-type annotations from an external CSV onto reference.obs in-place.

    Triggered by a ``metadata_join`` block in the dataset config, e.g.::

        "metadata_join": {
            "csv_path": "/path/to/metadata.csv",
            "csv_celltype_col": "subclass_label",
            "csv_index_split_on": "-L",   // split index "BARCODE-LIBRARY" on first "-L"
            "csv_barcode_col": "cell_barcode",
            "csv_library_col": "library_label"
        }

    The reference h5ad must already have ``cell_barcode`` and ``library_label`` obs columns
    (as specified in ``barcode_join.reference_barcode_col/library_col``).
    After the join, ``reference.obs[ref_celltype_key]`` is populated.
    """
    mj = cfg.get("metadata_join")
    if not mj:
        return

    csv_path       = Path(mj["csv_path"])
    ct_col         = mj["csv_celltype_col"]
    split_on       = mj.get("csv_index_split_on")
    barcode_col    = mj.get("csv_barcode_col", "cell_barcode")
    library_col    = mj.get("csv_library_col", "library_label")
    ref_ct_key     = cfg["reference_celltype_key"]
    ref_bc_col     = cfg.get("barcode_join", {}).get("reference_barcode_col", "cell_barcode")
    ref_lib_col    = cfg.get("barcode_join", {}).get("reference_library_col", "library_label")

    if not csv_path.exists():
        print(f"  WARNING: metadata_join csv not found: {csv_path}"); return

    print(f"  Loading metadata CSV for cell-type join: {csv_path.name}")
    meta = pd.read_csv(csv_path, index_col=0, low_memory=False)

    # Split index into barcode + library if needed
    if split_on:
        def _split(s):
            idx = s.find(split_on)
            return (s[:idx], s[idx + 1:]) if idx != -1 else (s, None)
        meta[[barcode_col, library_col]] = pd.DataFrame(
            [_split(s) for s in meta.index], index=meta.index
        )

    meta["_jk"] = meta[barcode_col] + "|" + meta[library_col]
    meta_ct = meta.set_index("_jk")[ct_col]

    ref_jk = (
        reference.obs[ref_bc_col].astype(str)
        + "|"
        + reference.obs[ref_lib_col].astype(str)
    )
    reference.obs[ref_ct_key] = ref_jk.map(meta_ct).values

    n_annotated = reference.obs[ref_ct_key].notna().sum()
    print(f"  Metadata join: {n_annotated:,}/{reference.n_obs:,} cells annotated "
          f"({100*n_annotated/reference.n_obs:.1f}%) via '{ct_col}'")


def evaluate_dataset(dataset_id: str, output_dir: Path):
    print(f"\n{'='*60}")
    print(f"  Dataset: {dataset_id}")
    print(f"{'='*60}")

    config = load_dataset_config(dataset_id)
    ref_path         = Path(config["reference_path"])
    ref_ct_key       = config["reference_celltype_key"]
    barcode_join     = config.get("barcode_join")
    coarse_mapping   = config.get("coarse_celltype_mapping")
    caribou_mapping  = config.get("caribou_celltype_mapping")
    input_n_cells    = config.get("input_n_cells")
    gene_sym_col     = config.get("reference_gene_symbol_col")
    dataset_results_dir = RESULTS_DIR / dataset_id

    if not ref_path.exists():
        print(f"  ERROR: Reference not found: {ref_path}"); return []
    if not dataset_results_dir.exists():
        print(f"  No results yet: {dataset_results_dir}"); return []

    run_dirs = sorted(d for d in dataset_results_dir.iterdir() if d.is_dir())
    print(f"  Found {len(run_dirs)} runs | Loading reference…")
    reference = sc.read_h5ad(ref_path)

    # Phase 1: remap reference gene IDs to symbols if needed
    if gene_sym_col and gene_sym_col in reference.var.columns:
        new_names = reference.var[gene_sym_col].astype(str)
        if new_names.is_unique:
            reference.var_names = new_names
            print(f"  Remapped reference var_names via '{gene_sym_col}'")
        else:
            # Make unique by appending Ensembl suffix to duplicates
            counts = new_names.value_counts()
            seen: Dict[str, int] = {}
            unique_names = []
            for orig, ensembl in zip(new_names, reference.var_names):
                if counts[orig] > 1:
                    seen[orig] = seen.get(orig, 0) + 1
                    unique_names.append(f"{orig}_{ensembl}")
                else:
                    unique_names.append(orig)
            reference.var_names = unique_names
            print(f"  Remapped reference var_names via '{gene_sym_col}' (disambiguated duplicates)")

    # Phase 2: attach cell types from external metadata CSV if configured
    _attach_metadata_celltypes(reference, config)

    print(f"  Reference: {reference.n_obs} cells × {reference.n_vars} genes")
    if barcode_join:
        print(f"  Barcode join: {barcode_join}")
    if coarse_mapping:
        print(f"  Coarse cell-type mapping: {len(coarse_mapping)} entries")
    if caribou_mapping:
        print(f"  CARIBOU cell-type mapping: {len(caribou_mapping)} entries")
    if input_n_cells:
        print(f"  Input dataset size: {input_n_cells:,} cells")

    out = output_dir / dataset_id
    out.mkdir(parents=True, exist_ok=True)

    results = []
    for run_dir in run_dirs:
        print(f"  → {run_dir.name}")
        r = evaluate_run(
            run_dir, reference, ref_ct_key, barcode_join, coarse_mapping,
            caribou_mapping, input_n_cells, out,
        )
        r["dataset"] = dataset_id          # always set from directory path
        r["dataset_name"] = config["name"]
        results.append(r)

    (out / "results.json").write_text(json.dumps(results, indent=2))
    pd.DataFrame(results).to_csv(out / "results.csv", index=False)
    print(f"  Saved → {out}/results.{{json,csv}}")

    df = pd.DataFrame(results)
    ok = df[df.get("success", False) == True] if "success" in df else df
    if len(ok):
        for m in ("ari", "nmi", "macro_f1", "weighted_f1",
                  "hvg_jaccard", "gene_expr_spearman_r", "runtime_s"):
            if m in ok.columns:
                vals = ok[m].dropna()
                if len(vals):
                    print(f"  {m}: {vals.mean():.3f} ± {vals.std():.3f}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate CARIBOU comparison runs")
    parser.add_argument(
        "--dataset", nargs="+", default=None,
        help="Dataset IDs to evaluate (default: all)"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=COMP_DIR / "analysis" / "outputs",
        help="Directory to write result JSON/CSV files"
    )
    args = parser.parse_args()

    datasets = args.dataset or available_datasets()
    if not datasets:
        print("No datasets found in", DATASETS_DIR); return

    print(f"Evaluating datasets: {datasets}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for ds in datasets:
        results = evaluate_dataset(ds, args.output_dir)
        all_results.extend(results)

    if all_results:
        (args.output_dir / "all_results.json").write_text(json.dumps(all_results, indent=2))
        pd.DataFrame(all_results).to_csv(args.output_dir / "all_results.csv", index=False)
        print(f"\n✓  Combined results → {args.output_dir}/all_results.{{json,csv}}")
        print(f"   Generate plots:  python plot.py")


if __name__ == "__main__":
    main()
