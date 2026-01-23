"""
Aggregate task benchmark results into summary tables.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from results_collector import collect_results


def _safe_mean(values: Iterable[Optional[float]]) -> Optional[float]:
    cleaned = [v for v in values if isinstance(v, (int, float))]
    return mean(cleaned) if cleaned else None


def _format_float(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}"


def _group_key(record: Dict) -> tuple:
    return (record.get("task"), record.get("mode"), record.get("llm_backend"))


def _summarize(records: List[Dict]) -> List[Dict]:
    summary = []
    grouped: Dict[tuple, List[Dict]] = {}
    for record in records:
        grouped.setdefault(_group_key(record), []).append(record)

    for (task, mode, llm_backend), group in grouped.items():
        runtime_values = []
        for record in group:
            duration = record.get("duration_seconds")
            total_time = record.get("total_time_seconds")
            runtime_values.append(duration if isinstance(duration, (int, float)) else total_time)
        success_rate = _safe_mean([1.0 if r.get("success") else 0.0 for r in group])
        output_rate = _safe_mean([1.0 if r.get("output_present") else 0.0 for r in group])
        autometric_present_rate = _safe_mean([1.0 if r.get("autometric_present") else 0.0 for r in group])
        autometric_success_rate = _safe_mean([1.0 if r.get("autometric_success") else 0.0 for r in group])
        summary.append(
            {
                "task": task,
                "mode": mode,
                "llm_backend": llm_backend,
                "runs": len(group),
                "success_rate": success_rate,
                "output_present_rate": output_rate,
                "autometric_present_rate": autometric_present_rate,
                "autometric_success_rate": autometric_success_rate,
                "avg_api_time_seconds": _safe_mean(r.get("api_time_seconds") for r in group),
                "avg_exec_time_seconds": _safe_mean(r.get("exec_time_seconds") for r in group),
                "avg_total_time_seconds": _safe_mean(r.get("total_time_seconds") for r in group),
                "avg_duration_seconds": _safe_mean(r.get("duration_seconds") for r in group),
                "avg_runtime_seconds": _safe_mean(runtime_values),
                "avg_agent_turns": _safe_mean(r.get("agent_turns") for r in group),
                "avg_final_cell_count": _safe_mean(
                    r.get("output_metrics", {}).get("final_cell_count") if r.get("output_metrics") else None
                    for r in group
                ),
            }
        )
    return sorted(summary, key=lambda r: (r["task"] or "", r["mode"], r["llm_backend"] or ""))


def _print_summary(summary: List[Dict]) -> None:
    headers = [
        "task",
        "mode",
        "llm",
        "runs",
        "success_rate",
        "output_rate",
        "autometric_rate",
        "avg_total_time",
        "avg_duration",
        "avg_turns",
    ]
    print("\t".join(headers))
    for row in summary:
        print(
            "\t".join(
                [
                    str(row.get("task")),
                    str(row.get("mode")),
                    str(row.get("llm_backend")),
                    str(row.get("runs")),
                    _format_float(row.get("success_rate")),
                    _format_float(row.get("output_present_rate")),
                    _format_float(row.get("autometric_success_rate")),
                    _format_float(row.get("avg_total_time_seconds")),
                    _format_float(row.get("avg_duration_seconds")),
                    _format_float(row.get("avg_agent_turns")),
                ]
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare task benchmark results.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("dev/task_benchmarks/results"),
        help="Base results directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dev/task_benchmarks/analysis"),
        help="Directory to write summary outputs.",
    )
    parser.add_argument(
        "--skip-h5ad-metrics",
        action="store_true",
        help="Skip loading h5ad files for output metrics.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = collect_results(args.results_dir, include_h5ad_metrics=not args.skip_h5ad_metrics)
    summary = _summarize(records)

    summary_path = args.output_dir / "task_benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    _print_summary(summary)
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
