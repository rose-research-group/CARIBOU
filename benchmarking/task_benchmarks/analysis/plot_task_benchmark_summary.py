import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def _load_summary(path: Path) -> List[Dict]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError("Expected a JSON list of summary records.")
    return data


def _group_by_task(records: List[Dict]) -> Dict[str, List[Dict]]:
    grouped: Dict[str, List[Dict]] = {}
    for rec in records:
        task = rec.get("task") or "unknown"
        grouped.setdefault(task, []).append(rec)
    return grouped


def _sort_key(rec: Dict) -> tuple:
    return (rec.get("mode") or "", rec.get("llm_backend") or "")


def _safe_mean(values: Iterable[Optional[float]]) -> Optional[float]:
    cleaned = [v for v in values if isinstance(v, (int, float))]
    return mean(cleaned) if cleaned else None


def _summarize_raw(records: List[Dict]) -> List[Dict]:
    grouped: Dict[tuple, List[Dict]] = {}
    for record in records:
        key = (record.get("task"), record.get("mode"), record.get("llm_backend"))
        grouped.setdefault(key, []).append(record)

    summary = []
    for (task, mode, llm_backend), group in grouped.items():
        summary.append(
            {
                "task": task,
                "mode": mode,
                "llm_backend": llm_backend,
                "runs": len(group),
                "success_rate": _safe_mean([1.0 if r.get("success") else 0.0 for r in group]),
                "output_present_rate": _safe_mean([1.0 if r.get("output_present") else 0.0 for r in group]),
                "autometric_success_rate": _safe_mean([1.0 if r.get("autometric_success") else 0.0 for r in group]),
                "avg_total_time_seconds": _safe_mean(r.get("total_time_seconds") for r in group),
                "avg_duration_seconds": _safe_mean(r.get("duration_seconds") for r in group),
                "avg_runtime_seconds": _safe_mean(
                    [
                        r.get("duration_seconds")
                        if isinstance(r.get("duration_seconds"), (int, float))
                        else r.get("total_time_seconds")
                        for r in group
                    ]
                ),
            }
        )
    return summary


def _plot_grouped_bars(
    title: str,
    records: List[Dict],
    value_key: str,
    output_path: Path,
    ylabel: str,
    ylim: Optional[tuple] = None,
) -> None:
    records = sorted(records, key=_sort_key)
    labels = [f"{r.get('mode')}\n{r.get('llm_backend')}" for r in records]
    values = [r.get(value_key) for r in records]
    cleaned = [
        float(v) if isinstance(v, (int, float)) else np.nan
        for v in values
    ]
    if all(np.isnan(v) for v in cleaned):
        print(f"Skipping {output_path.name}: no numeric values for {value_key}")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    ax.bar(x, cleaned, color="#4C78A8")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_summary(summary_path: Path, output_dir: Path) -> None:
    records = _load_summary(summary_path)
    if records and "success_rate" not in records[0]:
        records = _summarize_raw(records)
    grouped = _group_by_task(records)
    output_dir.mkdir(parents=True, exist_ok=True)

    for task, task_records in grouped.items():
        _plot_grouped_bars(
            title=f"{task}: success rate",
            records=task_records,
            value_key="success_rate",
            output_path=output_dir / f"{task}_success_rate.png",
            ylabel="Success rate",
            ylim=(0, 1.05),
        )
        _plot_grouped_bars(
            title=f"{task}: autometric success rate",
            records=task_records,
            value_key="autometric_success_rate",
            output_path=output_dir / f"{task}_autometric_rate.png",
            ylabel="Autometric success rate",
            ylim=(0, 1.05),
        )
        _plot_grouped_bars(
            title=f"{task}: avg duration (s)",
            records=task_records,
            value_key="avg_duration_seconds",
            output_path=output_dir / f"{task}_avg_duration.png",
            ylabel="Avg duration (s)",
            ylim=None,
        )
        _plot_grouped_bars(
            title=f"{task}: avg runtime (s)",
            records=task_records,
            value_key="avg_runtime_seconds",
            output_path=output_dir / f"{task}_avg_runtime.png",
            ylabel="Avg runtime (s)",
            ylim=None,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot task benchmark summary JSON.")
    parser.add_argument(
        "--summary-json",
        required=True,
        type=Path,
        help="Path to task_benchmark_summary.json.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        type=Path,
        help="Directory to save plots (defaults to summary JSON directory).",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or args.summary_json.parent
    plot_summary(args.summary_json, output_dir)
    print(f"Saved plots to {output_dir}")


if __name__ == "__main__":
    main()
