import argparse
import json
import numbers
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


SPECIES_SYNONYMS = {
    "human": "homo sapiens",
    "h. sapiens": "homo sapiens",
    "homo sapiens": "homo sapiens",
    "unknown": "unknown",
    "mouse": "mus musculus",
    "m. musculus": "mus musculus",
    "mus musculus": "mus musculus",
    "chimp": "pan troglodytes",
    "chimpanzee": "pan troglodytes",
    "pan troglodytes": "pan troglodytes",
    "marmoset": "callithrix jacchus",
    "c. jacchus": "callithrix jacchus",
    "callithrix jacchus": "callithrix jacchus",
    "rhesus": "macaca mulatta",
    "rhesus macaque": "macaca mulatta",
    "macaca mulatta": "macaca mulatta",
    "zebrafish": "danio rerio",
    "d. rerio": "danio rerio",
    "danio rerio": "danio rerio",
}

ORGAN_SYNONYMS = {
    # Blood
    "blood": "blood",
    "pbmc": "blood",
    "peripheral blood": "blood",
    "bone marrow": "blood",
    # Lung
    "lung": "lung",
    "lung parenchyma": "lung",
    "pulmonary": "lung",
    "respiratory": "lung",
    "airway": "lung",
    # Heart
    "heart": "heart",
    "cardiac": "heart",
    "myocardium": "heart",
    # Pancreas
    "pancreas": "pancreas",
    "islet of langerhans": "pancreas",
    "islets of langerhans": "pancreas",
    "pancreatic islet": "pancreas",
    "pancreatic islets": "pancreas",
    "islet": "pancreas",
    "islets": "pancreas",
    "endocrine pancreas": "pancreas",
    # Brain
    "brain": "brain",
    "primary visual cortex": "brain",
    "visual cortex": "brain",
    "cortex": "brain",
    "cerebral cortex": "brain",
    "cerebrum": "brain",
    "cerebellum": "brain",
    "hippocampus": "brain",
    "hypothalamus": "brain",
    "nervous system": "brain",
    "central nervous system": "brain",
    "cns": "brain",
    # Kidney
    "kidney": "kidney",
    "renal": "kidney",
    "nephron": "kidney",
    # Liver
    "liver": "liver",
    "hepatic": "liver",
    # Muscle
    "muscle": "muscle",
    "skeletal muscle": "muscle",
    "smooth muscle": "muscle",
    # Epithelium
    "epithelium": "epithelium",
    "epithelial": "epithelium",
    # Lens
    "lens": "lens",
    "eye lens": "lens",
    "ocular lens": "lens",
    # Pharyngeal arch
    "pharyngeal arch": "pharyngeal arch",
    "pharyngeal arches": "pharyngeal arch",
    "branchial arch": "pharyngeal arch",
    "branchial arches": "pharyngeal arch",
    "gill arch": "pharyngeal arch",
    "gill arches": "pharyngeal arch",
    "pharyngeal pouch": "pharyngeal arch",
    "gill": "pharyngeal arch",
    "pharyngeal": "pharyngeal arch",
}


def _normalize(value: Optional[str]) -> str:
    if not value:
        return ""
    return " ".join(value.strip().lower().replace("_", " ").split())


def _canonical_species(value: Optional[str]) -> str:
    normalized = _normalize(value)
    return SPECIES_SYNONYMS.get(normalized, normalized)


def _canonical_organ(value: Optional[str]) -> str:
    normalized = _normalize(value)
    return ORGAN_SYNONYMS.get(normalized, normalized)


def _find_dataset_name(path: Path, dataset_names: List[str]) -> Optional[str]:
    for part in path.parts:
        if part in dataset_names:
            return part
    return None


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text())


def _score_record(pred: Dict, gt: Dict, transcript_tol: float) -> Dict:
    pred_species = _canonical_species(pred.get("species"))
    pred_organ = _canonical_organ(pred.get("organ"))
    gt_species = _canonical_species(gt.get("actual_species"))
    gt_organ = _canonical_organ(gt.get("actual_organ"))

    species_match = None
    if gt_species and gt_species != "unknown":
        species_match = pred_species == gt_species and bool(pred_species)

    organ_match = None
    if gt_organ and gt_organ != "unknown":
        organ_match = pred_organ == gt_organ and bool(pred_organ)

    pred_cells = pred.get("cell_count")
    if isinstance(pred_cells, float) and pred_cells.is_integer():
        pred_cells = int(pred_cells)
    gt_cells = gt.get("cell_count")
    cell_count_match = None
    if isinstance(gt_cells, numbers.Integral):
        cell_count_match = isinstance(pred_cells, numbers.Integral) and pred_cells == gt_cells

    pred_transcript = pred.get("mean_transcript_count")
    gt_transcript = gt.get("mean_transcript_count")
    transcript_match = None
    rel_error = None
    abs_error = None
    if isinstance(pred_transcript, numbers.Real) and isinstance(gt_transcript, numbers.Real):
        abs_error = float(abs(pred_transcript - gt_transcript))
        rel_error = abs_error / gt_transcript if gt_transcript else None
        transcript_match = rel_error is not None and rel_error <= transcript_tol

    matches = [species_match, organ_match, cell_count_match, transcript_match]
    available = [m for m in matches if m is not None]
    score = sum(int(m) for m in available) / len(available) if available else 0.0

    species_match_val = float(species_match) if species_match is not None else None
    organ_match_val = float(organ_match) if organ_match is not None else None
    cell_count_match_val = float(cell_count_match) if cell_count_match is not None else None
    transcript_match_val = float(transcript_match) if transcript_match is not None else None

    return {
        "species_match": species_match_val,
        "organ_match": organ_match_val,
        "cell_count_match": cell_count_match_val,
        "mean_transcript_match": transcript_match_val,
        "mean_transcript_abs_error": abs_error,
        "mean_transcript_rel_error": rel_error,
        "score": score,
    }


def evaluate_results(results_dir: Path, ground_truth_path: Path, transcript_tol: float) -> pd.DataFrame:
    gt_df = pd.read_csv(ground_truth_path)
    dataset_names = gt_df["dataset_name"].tolist()
    gt_lookup = {row["dataset_name"]: row for _, row in gt_df.iterrows()}

    records: List[Dict] = []
    for json_path in results_dir.rglob("metadata_inference.json"):
        dataset_name = _find_dataset_name(json_path, dataset_names)
        if not dataset_name:
            continue
        gt = gt_lookup.get(dataset_name)
        if gt is None:
            continue
        pred = _load_json(json_path)
        score = _score_record(pred, gt, transcript_tol)
        records.append(
            {
                "dataset_name": dataset_name,
                "run_path": str(json_path),
                "pred_species": pred.get("species"),
                "pred_organ": pred.get("organ"),
                "pred_cell_count": pred.get("cell_count"),
                "pred_mean_transcript_count": pred.get("mean_transcript_count"),
                "gt_species": gt.get("actual_species"),
                "gt_organ": gt.get("actual_organ"),
                "gt_cell_count": gt.get("cell_count"),
                "gt_mean_transcript_count": gt.get("mean_transcript_count"),
                **score,
            }
        )

    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate metadata inference outputs.")
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Directory containing run outputs with metadata_inference.json files.",
    )
    parser.add_argument(
        "--ground-truth",
        default=str(Path(__file__).resolve().parent / "benchmark_data" / "ground_truth_master.csv"),
        help="Path to ground truth CSV.",
    )
    parser.add_argument(
        "--transcript-tol",
        type=float,
        default=0.05,
        help="Relative tolerance for mean transcript count match.",
    )
    parser.add_argument(
        "--output",
        default="metadata_benchmark_scores.csv",
        help="Output CSV path for per-run scores.",
    )
    args = parser.parse_args()

    df = evaluate_results(
        Path(args.results_dir).expanduser(),
        Path(args.ground_truth).expanduser(),
        args.transcript_tol,
    )
    if df.empty:
        print("No matching metadata_inference.json files found.")
        return

    output_path = Path(args.output).expanduser()
    df.to_csv(output_path, index=False)
    print(f"Saved scores to {output_path}")
    summary = df.groupby("dataset_name")["score"].mean().reset_index()
    print("\nMean score by dataset:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
