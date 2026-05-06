"""
Data loading and cell-alignment utilities for integration benchmarks.

Ported and extended from dev/comparisons/analysis/evaluate.py.
"""

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import scanpy as sc


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

INTBENCH_DIR = Path(__file__).parent.parent
DATASETS_DIR = INTBENCH_DIR / "datasets"
RESULTS_DIR  = INTBENCH_DIR / "results"
REPO_ROOT    = INTBENCH_DIR.parent.parent


def _repo_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def load_dataset_config(dataset_id: str) -> Dict:
    path = DATASETS_DIR / dataset_id / "config.json"
    if not path.exists():
        raise FileNotFoundError(f"No config for dataset '{dataset_id}': {path}")
    return json.loads(path.read_text())


def available_datasets() -> list:
    return sorted(
        d.name for d in DATASETS_DIR.iterdir()
        if d.is_dir() and (d / "config.json").exists()
    )


# ---------------------------------------------------------------------------
# Run metadata
# ---------------------------------------------------------------------------

def _infer_meta_from_run_name(run_dir: Path) -> Dict:
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
    meta_path = run_dir / "run_metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    else:
        meta = {}
        for fname in ("config.json", "runtime.json"):
            p = run_dir / fname
            if p.exists():
                meta.update(json.loads(p.read_text()))

    inferred = _infer_meta_from_run_name(run_dir)
    for key in ("mode", "llm_backend"):
        if not meta.get(key):
            meta[key] = inferred.get(key)
    return meta


def find_output_h5ad(run_dir: Path) -> Optional[Path]:
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
# Reference loading
# ---------------------------------------------------------------------------

def _remap_gene_symbols(reference: sc.AnnData, gene_sym_col: str) -> None:
    """Remap reference var_names from Ensembl IDs to gene symbols in-place."""
    if gene_sym_col not in reference.var.columns:
        return
    new_names = reference.var[gene_sym_col].astype(str)
    if new_names.is_unique:
        reference.var_names = new_names
        print(f"  Remapped reference var_names via '{gene_sym_col}'")
    else:
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


def attach_metadata_celltypes(reference: sc.AnnData, cfg: Dict) -> None:
    """Join cell-type labels from an external CSV onto reference.obs in-place.

    Reads the ``metadata_join`` block from the dataset config. The CSV is
    expected to have cell_barcode + library_label columns (or derivable from
    the index), and a cell-type column named by ``csv_celltype_col``.
    """
    mj = cfg.get("metadata_join")
    if not mj:
        return

    csv_path    = _repo_path(mj["csv_path"])
    ct_col      = mj["csv_celltype_col"]
    split_on    = mj.get("csv_index_split_on")
    barcode_col = mj.get("csv_barcode_col", "cell_barcode")
    library_col = mj.get("csv_library_col", "library_label")
    ref_ct_key  = cfg["reference_celltype_key"]
    ref_bc_col  = cfg.get("barcode_join", {}).get("reference_barcode_col", "cell_barcode")
    ref_lib_col = cfg.get("barcode_join", {}).get("reference_library_col", "library_label")

    if not csv_path.exists():
        raise FileNotFoundError(f"metadata_join csv not found: {csv_path}")

    print(f"  Loading metadata CSV for cell-type join: {csv_path.name}")
    meta = pd.read_csv(csv_path, index_col=0, low_memory=False)

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
          f"({100 * n_annotated / reference.n_obs:.1f}%) via '{ct_col}'")


def load_raw_reference(cfg: Dict) -> sc.AnnData:
    """Load the raw reference h5ad, remap gene symbols, attach cell types from CSV."""
    ref_path = _repo_path(cfg["reference_path"])
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference not found: {ref_path}")

    print(f"  Loading raw reference: {ref_path.name}")
    reference = sc.read_h5ad(ref_path)

    gene_sym_col = cfg.get("reference_gene_symbol_col")
    if gene_sym_col:
        _remap_gene_symbols(reference, gene_sym_col)

    attach_metadata_celltypes(reference, cfg)
    print(f"  Raw reference: {reference.n_obs:,} cells × {reference.n_vars:,} genes")
    return reference


def load_processed_reference(cfg: Dict) -> sc.AnnData:
    """Load the consortium-processed reference (with X_pca/X_umap) and attach cell types."""
    proc_path = _repo_path(cfg["processed_reference_path"])
    if not proc_path.exists():
        raise FileNotFoundError(f"Processed reference not found: {proc_path}")

    print(f"  Loading processed reference: {proc_path.name}")
    proc_ref = sc.read_h5ad(proc_path)

    # Attach cell types from metadata CSV (processed ref has no cell_type column)
    ref_ct_key = cfg["reference_celltype_key"]
    if ref_ct_key not in proc_ref.obs.columns:
        attach_metadata_celltypes(proc_ref, cfg)

    print(f"  Processed reference: {proc_ref.n_obs:,} cells × {proc_ref.n_vars:,} genes  "
          f"| obsm: {list(proc_ref.obsm.keys())}")
    return proc_ref


# ---------------------------------------------------------------------------
# Barcode alignment
# ---------------------------------------------------------------------------

def _build_compound_key(
    adata: sc.AnnData,
    barcode_col: Optional[str],
    library_col: Optional[str],
    strip_suffix: bool,
    index_extract: Optional[str] = None,
) -> pd.Series:
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


def align(
    ref: sc.AnnData,
    car: sc.AnnData,
    barcode_join: Optional[Dict] = None,
):
    """Align ref and car on common cells and genes.

    Returns (ref_aligned, car_aligned, n_common_cells, n_common_genes).
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


def align_processed_ref(
    proc_ref: sc.AnnData,
    car: sc.AnnData,
    barcode_join: Optional[Dict] = None,
):
    """Align the processed reference (basic_pp) with a CARIBOU run on common cells.

    The processed reference has a different gene set (2000 HVGs) so we align
    on cells only; the caller is responsible for using obsm embeddings rather
    than gene expression for subsequent metrics.

    Returns (proc_ref_aligned, car_aligned_cells_only, n_common_cells).
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
            proc_ref,
            barcode_col=barcode_join.get("reference_barcode_col"),
            library_col=barcode_join.get("reference_library_col"),
            strip_suffix=False,
            index_extract=barcode_join.get("reference_index_extract"),
        )
        car_idx_map = pd.Series(car.obs.index, index=car_key.values)
        ref_idx_map = pd.Series(proc_ref.obs.index, index=ref_key.values)
        common_keys = set(car_idx_map.index) & set(ref_idx_map.index)
        if not common_keys:
            raise ValueError(
                "No common cells between processed reference and CARIBOU output — "
                "check barcode_join config."
            )
        car_cells = car_idx_map[list(common_keys)].values
        ref_cells = ref_idx_map[list(common_keys)].values
    else:
        common_cells = proc_ref.obs.index.intersection(car.obs.index)
        if len(common_cells) == 0:
            raise ValueError("No common cells between processed reference and CARIBOU output.")
        car_cells = ref_cells = np.array(common_cells)

    return (
        proc_ref[ref_cells].copy(),
        car[car_cells].copy(),
        len(ref_cells),
    )
