"""Deterministic baseline for the metadata inference benchmark.

Implements the same strategy described in the agent prompts (Ensembl-prefix
counting for species, hard-coded marker-panel scoring for organ, direct
computation for cell count and mean transcript count) but without any LLM.

This establishes a floor for what a simple rule-based script can achieve,
so the LLM-agent results can be contextualized.

Usage
-----
Run on all benchmark datasets::

    python deterministic_baseline.py --manifest benchmark_data/benchmark_manifest.csv

Run on a single dataset::

    python deterministic_baseline.py \
        --dataset benchmark_data/human_lung_blind.h5ad \
        --output-dir results/metadata_task/deterministic_baseline/human_lung/run
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

# ---------------------------------------------------------------------------
# Species inference via Ensembl gene-ID prefixes
# ---------------------------------------------------------------------------
ENSEMBL_PREFIX_TO_SPECIES = {
    "ENSG": "Homo sapiens",
    "ENSMUSG": "Mus musculus",
    "ENSMMUG": "Macaca mulatta",
    "ENSPTRG": "Pan troglodytes",
    "ENSCJAG": "Callithrix jacchus",
    "ENSDARG": "Danio rerio",
}

_PREFIX_RE = re.compile(r"^(ENS[A-Z]*G)\d")


def _infer_species(var_names: list[str]) -> tuple[str, float, list[str]]:
    """Return (species, confidence, evidence) from Ensembl prefix fractions."""
    counts: Counter[str] = Counter()
    for name in var_names:
        m = _PREFIX_RE.match(name)
        if m:
            counts[m.group(1)] += 1
    if not counts:
        return "Unknown", 0.0, ["no Ensembl-style gene IDs found"]

    total = sum(counts.values())
    best_prefix, best_count = counts.most_common(1)[0]
    fraction = best_count / total
    species = ENSEMBL_PREFIX_TO_SPECIES.get(best_prefix, "Unknown")
    evidence = [
        f"{best_prefix}: {best_count}/{total} ({fraction:.1%})",
        *(f"{p}: {c}/{total} ({c/total:.1%})" for p, c in counts.most_common() if p != best_prefix),
    ]
    return species, round(fraction, 4), evidence


# ---------------------------------------------------------------------------
# Organ inference via hard-coded marker panels (Ensembl IDs)
# ---------------------------------------------------------------------------
# Each panel maps an organ to a dict of {Ensembl_ID: gene_symbol} for the
# three benchmark species.  We keep ~10 well-known markers per organ.
# IDs are given WITHOUT version suffixes so they match stripped var_names.

MARKER_PANELS = {
    "lung": {
        # Human
        "ENSG00000168484": "SFTPC",   # surfactant protein C
        "ENSG00000185303": "SFTPB",   # surfactant protein B
        "ENSG00000148344": "PTGS2",
        "ENSG00000164867": "NOS3",
        "ENSG00000122641": "INHBA",
        "ENSG00000091527": "CDH16",
        "ENSG00000168214": "RBPJ",
        # Mouse
        "ENSMUSG00000021795": "Sftpc",
        "ENSMUSG00000035790": "Sftpb",
        "ENSMUSG00000032487": "Ptgs2",
        "ENSMUSG00000024145": "Nos3",
        "ENSMUSG00000041324": "Aqp5",
        "ENSMUSG00000029378": "Scgb1a1",
    },
    "pancreas": {
        # Human
        "ENSG00000254647": "INS",     # insulin
        "ENSG00000115263": "GCG",     # glucagon
        "ENSG00000137941": "TTR",
        "ENSG00000105697": "HAMP",
        "ENSG00000108849": "GPC3",
        "ENSG00000169903": "NKX2-2",
        "ENSG00000204305": "PDX1",
        # Mouse
        "ENSMUSG00000000215": "Ins2",
        "ENSMUSG00000000394": "Gcg",
        "ENSMUSG00000061808": "Ttr",
        "ENSMUSG00000008393": "Pdx1",
        "ENSMUSG00000027984": "Isl1",
        "ENSMUSG00000041147": "Nkx2-2",
    },
    "brain": {
        # Human
        "ENSG00000197971": "MBP",     # myelin basic protein
        "ENSG00000120549": "GFAP",    # glial fibrillary acidic protein
        "ENSG00000104888": "SLC17A7", # VGLUT1
        "ENSG00000070748": "CHAT",    # choline acetyltransferase
        "ENSG00000079215": "SLC1A3",  # GLAST
        "ENSG00000157542": "KCNJ6",
        "ENSG00000132639": "SNAP25",
        # Mouse
        "ENSMUSG00000041607": "Mbp",
        "ENSMUSG00000020932": "Gfap",
        "ENSMUSG00000070570": "Slc17a7",
        "ENSMUSG00000027204": "Snap25",
        "ENSMUSG00000026787": "Gad2",
        "ENSMUSG00000030500": "Slc1a3",
    },
    "kidney": {
        # Human
        "ENSG00000136872": "ALDOB",
        "ENSG00000113361": "CDH6",
        "ENSG00000074803": "SLC12A1", # NKCC2
        "ENSG00000100063": "SLC22A6",
        "ENSG00000143839": "AQP2",
        "ENSG00000168481": "AQP1",
        "ENSG00000137331": "IER3",
        # Mouse
        "ENSMUSG00000028307": "Aqp2",
        "ENSMUSG00000024867": "Aqp1",
        "ENSMUSG00000020044": "Slc12a1",
        "ENSMUSG00000023087": "Slc22a6",
        "ENSMUSG00000063626": "Umod",
        "ENSMUSG00000031490": "Slc12a3",
    },
    "liver": {
        # Human
        "ENSG00000163631": "ALB",
        "ENSG00000080618": "AFP",
        "ENSG00000026508": "CD36",
        "ENSG00000105697": "HAMP",
        "ENSG00000137941": "TTR",
        "ENSG00000084674": "APOB",
        # Mouse
        "ENSMUSG00000029368": "Alb",
        "ENSMUSG00000054932": "Afp",
        "ENSMUSG00000024164": "Ttr",
        "ENSMUSG00000031722": "Apob",
    },
    "blood": {
        # Human
        "ENSG00000170180": "GYPA",    # glycophorin A
        "ENSG00000196565": "HBB",
        "ENSG00000206172": "HBA1",
        "ENSG00000244734": "HBB",
        "ENSG00000169442": "CD52",
        "ENSG00000153563": "CD8A",
        # Mouse
        "ENSMUSG00000052305": "Hbb-bs",
        "ENSMUSG00000069919": "Hba-a1",
        "ENSMUSG00000000409": "Lck",
    },
    "heart": {
        # Human
        "ENSG00000092054": "MYH7",
        "ENSG00000129991": "TNNI3",
        "ENSG00000118194": "TNNT2",
        "ENSG00000106631": "MYL2",
        "ENSG00000159251": "ACTC1",
        # Mouse
        "ENSMUSG00000053093": "Myh7",
        "ENSMUSG00000026414": "Tnni3",
        "ENSMUSG00000026180": "Tnnt2",
    },
    "pharyngeal arch": {
        # Zebrafish — developmental / craniofacial markers
        "ENSDARG00000040944": "dlx2a",
        "ENSDARG00000011993": "hand2",
        "ENSDARG00000039564": "sox9a",
        "ENSDARG00000037819": "col2a1a",
        "ENSDARG00000003732": "barx1",
        "ENSDARG00000076380": "nkx3.2",
        "ENSDARG00000101599": "dlx5a",
        "ENSDARG00000019949": "edn1",
        "ENSDARG00000020850": "fgf8a",
        "ENSDARG00000042816": "mef2ca",
    },
}


def _strip_version(gene_id: str) -> str:
    """Remove Ensembl version suffix (e.g. ENSG00000123456.3 -> ENSG00000123456)."""
    return gene_id.split(".")[0]


def _infer_organ(
    adata: sc.AnnData,
    species: str,
) -> tuple[str, float, list[str]]:
    """Score each organ panel against the dataset and return the best match."""
    stripped_names = [_strip_version(v) for v in adata.var_names]
    var_set = set(stripped_names)
    name_to_idx = {n: i for i, n in enumerate(stripped_names)}

    matrix = adata.layers["counts"] if "counts" in adata.layers else adata.X

    best_organ = "Unknown"
    best_score = -1.0
    best_evidence: list[str] = []
    all_scores: dict[str, float] = {}

    for organ, panel in MARKER_PANELS.items():
        present_ids = [eid for eid in panel if eid in var_set]
        if not present_ids:
            all_scores[organ] = 0.0
            continue

        indices = [name_to_idx[eid] for eid in present_ids]
        col_subset = matrix[:, indices]
        mean_expr = float(np.asarray(col_subset.mean(axis=0)).ravel().mean())
        frac_present = len(present_ids) / len(panel)
        score = mean_expr * frac_present

        all_scores[organ] = score
        if score > best_score:
            best_score = score
            best_organ = organ
            best_evidence = [
                f"markers present: {len(present_ids)}/{len(panel)}",
                f"mean expression: {mean_expr:.4f}",
                f"detected: {', '.join(panel[eid] for eid in present_ids[:5])}",
            ]

    if best_score <= 0:
        return "Unknown", 0.0, ["no marker panel matched"]

    sorted_scores = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
    if len(sorted_scores) > 1 and sorted_scores[1][1] > 0:
        margin = best_score / sorted_scores[1][1]
        confidence = min(1.0, margin / 10.0)
    else:
        confidence = 1.0

    return best_organ, round(confidence, 4), best_evidence


# ---------------------------------------------------------------------------
# Cell count and mean transcript count
# ---------------------------------------------------------------------------
def _compute_counts(adata: sc.AnnData) -> tuple[int, float]:
    cell_count = int(adata.n_obs)
    matrix = adata.layers["counts"] if "counts" in adata.layers else adata.X
    summed = matrix.sum(axis=1)
    mean_transcript = float(np.asarray(summed).ravel().mean())
    return cell_count, mean_transcript


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_baseline(dataset_path: Path, output_dir: Path) -> dict:
    """Run deterministic baseline on a single blind h5ad and write results."""
    adata = sc.read_h5ad(dataset_path)

    species, species_conf, species_evidence = _infer_species(list(adata.var_names))
    organ, organ_conf, organ_evidence = _infer_organ(adata, species)
    cell_count, mean_transcript = _compute_counts(adata)

    result = {
        "species": species,
        "organ": organ,
        "cell_count": cell_count,
        "mean_transcript_count": mean_transcript,
        "confidence": {
            "species": species_conf,
            "organ": organ_conf,
        },
        "evidence": {
            "species": species_evidence,
            "organ": organ_evidence,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "metadata_inference.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"  -> {out_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deterministic baseline for metadata inference benchmark.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Path to benchmark_manifest.csv.  Runs baseline on every dataset.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Path to a single blind h5ad file.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (required with --dataset; ignored with --manifest).",
    )
    args = parser.parse_args()

    if args.manifest:
        manifest = pd.read_csv(args.manifest)
        base_dir = Path(args.manifest).resolve().parent.parent
        results_base = base_dir / "results" / "metadata_task" / "deterministic_baseline"
        for _, row in manifest.iterrows():
            name = row["dataset_name"]
            ds_path = Path(row["blind_h5ad_path"].strip())
            out_dir = results_base / name / "run"
            print(f"--- {name} ---")
            run_baseline(ds_path, out_dir)
        print("\nDeterministic baseline complete.")
    elif args.dataset:
        if not args.output_dir:
            parser.error("--output-dir is required when using --dataset")
        run_baseline(Path(args.dataset), Path(args.output_dir))
    else:
        parser.error("Provide either --manifest or --dataset")


if __name__ == "__main__":
    main()
