import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


SPECIES_SYNONYMS = {
    "human": "homo sapiens",
    "h. sapiens": "homo sapiens",
    "homo sapiens": "homo sapiens",
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
}

ORGAN_SYNONYMS = {
    "blood": "blood",
    "pbmc": "blood",
    "lung": "lung",
    "heart": "heart",
    "pancreas": "pancreas",
    "brain": "brain",
    "kidney": "kidney",
    "liver": "liver",
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

    species_match = pred_species == gt_species and bool(pred_species)
    organ_match = pred_organ == gt_organ and bool(pred_organ)

    pred_cells = pred.get("cell_count")
    if isinstance(pred_cells, float) and pred_cells.is_integer():
        pred_cells = int(pred_cells)
    gt_cells = gt.get("cell_count")
    cell_count_match = isinstance(pred_cells, int) and pred_cells == gt_cells

    pred_transcript = pred.get("mean_transcript_count")
    gt_transcript = gt.get("mean_transcript_count")
    transcript_match = False
    rel_error = None
    abs_error = None
    if isinstance(pred_transcript, (int, float)) and isinstance(gt_transcript, (int, float)):
        abs_error = float(abs(pred_transcript - gt_transcript))
        rel_error = abs_error / gt_transcript if gt_transcript else None
        transcript_match = rel_error is not None and rel_error <= transcript_tol

    score = (
        int(species_match)
        + int(organ_match)
        + int(cell_count_match)
        + int(transcript_match)
    ) / 4.0

    return {
        "species_match": species_match,
        "organ_match": organ_match,
        "cell_count_match": cell_count_match,
        "mean_transcript_match": transcript_match,
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
