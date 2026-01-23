"""
Collect and summarize task benchmark results across modes.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional


REQUIRED_OBS = [
    "doublet_score",
    "predicted_doublet",
    "n_genes_by_counts",
    "total_counts",
    "pct_counts_mt",
]


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text())


def _latest_file(path: Path, pattern: str) -> Optional[Path]:
    candidates = list(path.glob(pattern))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _parse_params(params_path: Path) -> Dict[str, str]:
    params = {}
    for line in params_path.read_text().splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        params[key.strip().lower()] = value.strip()
    return params


def _find_output_h5ad(run_dir: Path) -> Optional[Path]:
    candidates = [
        run_dir / "outputs" / "qc_filtered.h5ad",
        run_dir / "qc_filtered.h5ad",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    for pattern in ("outputs/*.h5ad", "*.h5ad"):
        matches = list(run_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def _collect_h5ad_metrics(h5ad_path: Path) -> Dict:
    try:
        import anndata  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        return {"error": f"anndata_unavailable: {exc}"}

    try:
        adata = anndata.read_h5ad(h5ad_path)
    except Exception as exc:
        return {"error": f"failed_to_load_h5ad: {exc}"}

    obs_present = {col: col in adata.obs.columns for col in REQUIRED_OBS}
    metrics = {
        "final_cell_count": adata.n_obs,
        "final_gene_count": adata.n_vars,
        "obs_columns_present": obs_present,
        "counts_layer_present": "counts" in adata.layers,
        "pca_present": "X_pca" in adata.obsm,
        "umap_present": "X_umap" in adata.obsm,
        "hvg_calculated": "highly_variable" in adata.var.columns,
    }
    return metrics


def _load_latest_benchmark_results(run_dir: Path) -> Optional[Dict]:
    ledger_path = run_dir / "benchmark_results.jsonl"
    if not ledger_path.exists():
        return None
    lines = [line for line in ledger_path.read_text().splitlines() if line.strip()]
    if not lines:
        return None
    try:
        record = json.loads(lines[-1])
    except json.JSONDecodeError:
        return None
    results = record.get("results")
    return results if isinstance(results, dict) else None


def _infer_autometric_success(results: Optional[Dict]) -> Optional[bool]:
    if not results:
        return None
    if "obs_columns_present" in results:
        obs_ok = all(results.get("obs_columns_present", {}).values())
        return bool(
            obs_ok
            and results.get("counts_layer_present")
            and results.get("pca_present")
            and results.get("umap_present")
            and results.get("hvg_calculated")
        )
    if "n_obs" in results and "n_vars" in results:
        return bool(results.get("n_obs", 0) > 0 and results.get("n_vars", 0) > 0)
    return None


def _flatten_record(record: Dict) -> Dict:
    flat = {
        k: v
        for k, v in record.items()
        if k not in {"output_metrics", "autometric_results"}
    }
    metrics = record.get("output_metrics") or {}
    obs_columns = metrics.get("obs_columns_present") or {}
    flat["final_cell_count"] = metrics.get("final_cell_count")
    flat["final_gene_count"] = metrics.get("final_gene_count")
    flat["counts_layer_present"] = metrics.get("counts_layer_present")
    flat["pca_present"] = metrics.get("pca_present")
    flat["umap_present"] = metrics.get("umap_present")
    flat["hvg_calculated"] = metrics.get("hvg_calculated")
    for col in REQUIRED_OBS:
        flat[f"obs_{col}_present"] = obs_columns.get(col)
    if "error" in metrics:
        flat["output_metrics_error"] = metrics["error"]
    flat["autometric_present"] = record.get("autometric_present")
    flat["autometric_success"] = record.get("autometric_success")
    flat["code_exec_attempts"] = record.get("code_exec_attempts")
    flat["code_exec_failures"] = record.get("code_exec_failures")
    return flat


def collect_results(results_dir: Path, include_h5ad_metrics: bool = True) -> List[Dict]:
    records: List[Dict] = []
    modes = ["one_shot", "single_agent", "full_system"]

    mode_dirs = [results_dir / mode for mode in modes if (results_dir / mode).exists()]
    task_roots: Dict[str, Path] = {}
    if mode_dirs:
        task_roots["qc"] = results_dir
    for child in results_dir.iterdir():
        if not child.is_dir() or child.name == "logs":
            continue
        if any((child / mode).exists() for mode in modes):
            task_roots.setdefault(child.name, child)

    for task_name, task_root in task_roots.items():
        for mode in modes:
            mode_dir = task_root / mode
            if not mode_dir.exists():
                continue
            for run_dir in sorted(mode_dir.iterdir()):
                if not run_dir.is_dir() or run_dir.name == "logs":
                    continue

                record: Dict[str, object] = {
                    "run_dir": str(run_dir),
                    "task": task_name,
                    "mode": mode,
                    "llm_backend": None,
                    "model_name": None,
                    "success": None,
                    "end_reason": None,
                    "num_api_calls": None,
                    "api_time_seconds": None,
                    "exec_time_seconds": None,
                    "total_time_seconds": None,
                    "duration_seconds": None,
                    "agent_turns": None,
                    "output_h5ad": None,
                    "output_present": False,
                    "output_metrics": None,
                    "autometric_present": False,
                    "autometric_success": None,
                    "autometric_results": None,
                }

                params_path = run_dir / "params.txt"
                params = _parse_params(params_path) if params_path.exists() else {}
                if params.get("llm_backend"):
                    record["llm_backend"] = params.get("llm_backend")

                metrics_path = run_dir / "metrics.json"
                if metrics_path.exists():
                    metrics = _load_json(metrics_path)
                    record["success"] = metrics.get("success")
                    record["llm_backend"] = record["llm_backend"] or metrics.get("llm_backend")
                    record["model_name"] = metrics.get("model_name")
                    record["num_api_calls"] = metrics.get("num_api_calls")
                    record["api_time_seconds"] = metrics.get("api_time_seconds")
                    record["exec_time_seconds"] = metrics.get("exec_time_seconds")
                    record["total_time_seconds"] = metrics.get("total_time_seconds")
                    record["code_exec_attempts"] = metrics.get("code_exec_attempts")
                    record["code_exec_failures"] = metrics.get("code_exec_failures")

                report_path = _latest_file(run_dir / "reports", "session_report_*.json")
                if report_path and report_path.exists():
                    report = _load_json(report_path)
                    record["model_name"] = report.get("model", record["model_name"])
                    record["agent_turns"] = report.get("agent_turns")
                    record["duration_seconds"] = report.get("duration_seconds")
                    record["end_reason"] = report.get("end_reason")
                    record["code_exec_attempts"] = report.get("code_exec_attempts", record.get("code_exec_attempts"))
                    record["code_exec_failures"] = report.get("code_exec_failures", record.get("code_exec_failures"))
                    if record["success"] is None:
                        record["success"] = report.get("end_reason") in {"completed", "max_turns_reached"}

                output_h5ad = _find_output_h5ad(run_dir)
                if output_h5ad:
                    record["output_h5ad"] = str(output_h5ad)
                    record["output_present"] = True
                    if include_h5ad_metrics:
                        record["output_metrics"] = _collect_h5ad_metrics(output_h5ad)

                autometric_results = _load_latest_benchmark_results(run_dir)
                if autometric_results:
                    record["autometric_present"] = True
                    record["autometric_results"] = autometric_results
                    record["autometric_success"] = _infer_autometric_success(autometric_results)

                records.append(record)

    return records


def _write_csv(path: Path, records: Iterable[Dict]) -> None:
    rows = [_flatten_record(record) for record in records]
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            for key in fieldnames:
                row.setdefault(key, None)
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect task benchmark results.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("dev/task_benchmarks/results"),
        help="Base directory for task benchmark results.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Path to write JSON summary (defaults to results_dir/summary.json).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional CSV output path.",
    )
    parser.add_argument(
        "--skip-h5ad-metrics",
        action="store_true",
        help="Skip loading h5ad files for output metrics.",
    )
    args = parser.parse_args()

    output_json = args.output_json or args.results_dir / "summary.json"
    records = collect_results(args.results_dir, include_h5ad_metrics=not args.skip_h5ad_metrics)
    output_json.write_text(json.dumps(records, indent=2))
    if args.output_csv:
        _write_csv(args.output_csv, records)

    print(f"Collected {len(records)} runs.")
    print(f"Wrote summary to {output_json}")


if __name__ == "__main__":
    main()
