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
                "avg_correction_count": _safe_mean(
                    r.get("correction_count")
                    if r.get("correction_count") is not None
                    else r.get("code_exec_failures")
                    for r in group
                ),
                "avg_data_adequacy_score": _safe_mean(
                    (
                        r.get("autometric_results", {}).get("adequacy_score")
                        if isinstance(r.get("autometric_results"), dict)
                        else None
                        for r in group
                    )
                ),
                "avg_predicted_doublet_rate": _safe_mean(
                    (
                        r.get("autometric_results", {}).get("predicted_doublet_rate")
                        if isinstance(r.get("autometric_results"), dict)
                        else None
                        for r in group
                    )
                ),
                "avg_doublet_score_mean": _safe_mean(
                    (
                        r.get("autometric_results", {}).get("doublet_score_mean")
                        if isinstance(r.get("autometric_results"), dict)
                        else None
                        for r in group
                    )
                ),
                "avg_batch_silhouette_baseline": _safe_mean(
                    (
                        r.get("autometric_results", {}).get("batch_silhouette_baseline")
                        if isinstance(r.get("autometric_results"), dict)
                        else None
                        for r in group
                    )
                ),
                "avg_batch_silhouette_integrated": _safe_mean(
                    (
                        r.get("autometric_results", {}).get("batch_silhouette_integrated")
                        if isinstance(r.get("autometric_results"), dict)
                        else None
                        for r in group
                    )
                ),
                "avg_batch_silhouette_delta": _safe_mean(
                    (
                        r.get("autometric_results", {}).get("batch_silhouette_delta")
                        if isinstance(r.get("autometric_results"), dict)
                        else None
                        for r in group
                    )
                ),
                "avg_celltype_silhouette_baseline": _safe_mean(
                    (
                        r.get("autometric_results", {}).get("celltype_silhouette_baseline")
                        if isinstance(r.get("autometric_results"), dict)
                        else None
                        for r in group
                    )
                ),
                "avg_celltype_silhouette_integrated": _safe_mean(
                    (
                        r.get("autometric_results", {}).get("celltype_silhouette_integrated")
                        if isinstance(r.get("autometric_results"), dict)
                        else None
                        for r in group
                    )
                ),
                "avg_celltype_silhouette_delta": _safe_mean(
                    (
                        r.get("autometric_results", {}).get("celltype_silhouette_delta")
                        if isinstance(r.get("autometric_results"), dict)
                        else None
                        for r in group
                    )
                ),
                "avg_isolated_label_f1_baseline": _safe_mean(
                    (
                        r.get("autometric_results", {}).get("isolated_label_f1_baseline")
                        if isinstance(r.get("autometric_results"), dict)
                        else None
                        for r in group
                    )
                ),
                "avg_isolated_label_f1_integrated": _safe_mean(
                    (
                        r.get("autometric_results", {}).get("isolated_label_f1_integrated")
                        if isinstance(r.get("autometric_results"), dict)
                        else None
                        for r in group
                    )
                ),
                "avg_isolated_label_f1_delta": _safe_mean(
                    (
                        r.get("autometric_results", {}).get("isolated_label_f1_delta")
                        if isinstance(r.get("autometric_results"), dict)
                        else None
                        for r in group
                    )
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


def _plot_matrix(
    title: str,
    records: List[Dict],
    value_key: str,
    output_path: Path,
    ylabel: str,
    ylim: Optional[tuple] = None,
) -> None:
    if not records:
        return
    modes = sorted({r.get("mode") or "" for r in records})
    llms = sorted({r.get("llm_backend") or "" for r in records})
    if not modes or not llms:
        return

    values = np.full((len(modes), len(llms)), np.nan)
    for r in records:
        mode = r.get("mode") or ""
        llm = r.get("llm_backend") or ""
        try:
            m = modes.index(mode)
            l = llms.index(llm)
        except ValueError:
            continue
        v = r.get(value_key)
        values[m, l] = float(v) if isinstance(v, (int, float)) else np.nan

    if np.all(np.isnan(values)):
        print(f"Skipping {output_path.name}: no numeric values for {value_key}")
        return

    fig, ax = plt.subplots(figsize=(max(5, len(llms) * 1.2), max(3, len(modes) * 0.8)))
    im = ax.imshow(values, aspect="auto", cmap="viridis", vmin=None if not ylim else ylim[0], vmax=None if not ylim else ylim[1])
    ax.set_title(title)
    ax.set_xticks(np.arange(len(llms)))
    ax.set_xticklabels(llms, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(modes)))
    ax.set_yticklabels(modes)
    ax.set_xlabel("LLM")
    ax.set_ylabel(ylabel)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
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
        _plot_grouped_bars(
            title=f"{task}: avg correction count",
            records=task_records,
            value_key="avg_correction_count",
            output_path=output_dir / f"{task}_avg_corrections.png",
            ylabel="Avg corrections (failed executions)",
            ylim=None,
        )
        if "data_adequacy" in task:
            _plot_grouped_bars(
                title=f"{task}: avg data adequacy score",
                records=task_records,
                value_key="avg_data_adequacy_score",
                output_path=output_dir / f"{task}_avg_data_adequacy.png",
                ylabel="Avg adequacy score",
                ylim=(0, 1.05),
            )
        if task == "doublet_task":
            _plot_grouped_bars(
                title=f"{task}: avg predicted doublet rate",
                records=task_records,
                value_key="avg_predicted_doublet_rate",
                output_path=output_dir / f"{task}_avg_doublet_rate.png",
                ylabel="Avg predicted doublet rate",
                ylim=(0, 1.0),
            )
            _plot_grouped_bars(
                title=f"{task}: avg doublet score",
                records=task_records,
                value_key="avg_doublet_score_mean",
                output_path=output_dir / f"{task}_avg_doublet_score.png",
                ylabel="Avg doublet score",
                ylim=None,
            )
            _plot_matrix(
                title=f"{task}: predicted doublet rate by mode/LLM",
                records=task_records,
                value_key="avg_predicted_doublet_rate",
                output_path=output_dir / f"{task}_avg_doublet_rate_matrix.png",
                ylabel="Mode",
                ylim=(0, 1.0),
            )
            _plot_matrix(
                title=f"{task}: doublet score by mode/LLM",
                records=task_records,
                value_key="avg_doublet_score_mean",
                output_path=output_dir / f"{task}_avg_doublet_score_matrix.png",
                ylabel="Mode",
                ylim=None,
            )
        if "batch" in task and "correction" in task:
            for metric in ("batch_silhouette", "celltype_silhouette", "isolated_label_f1"):
                _plot_grouped_bars(
                    title=f"{task}: {metric.replace('_', ' ')} (baseline)",
                    records=task_records,
                    value_key=f"avg_{metric}_baseline",
                    output_path=output_dir / f"{task}_{metric}_baseline.png",
                    ylabel=metric.replace("_", " "),
                    ylim=None,
                )
                _plot_grouped_bars(
                    title=f"{task}: {metric.replace('_', ' ')} (integrated)",
                    records=task_records,
                    value_key=f"avg_{metric}_integrated",
                    output_path=output_dir / f"{task}_{metric}_integrated.png",
                    ylabel=metric.replace("_", " "),
                    ylim=None,
                )
                _plot_grouped_bars(
                    title=f"{task}: {metric.replace('_', ' ')} delta",
                    records=task_records,
                    value_key=f"avg_{metric}_delta",
                    output_path=output_dir / f"{task}_{metric}_delta.png",
                    ylabel=f"delta {metric.replace('_', ' ')}",
                    ylim=None,
                )
                _plot_matrix(
                    title=f"{task}: {metric.replace('_', ' ')} delta by mode/LLM",
                    records=task_records,
                    value_key=f"avg_{metric}_delta",
                    output_path=output_dir / f"{task}_{metric}_delta_matrix.png",
                    ylabel="Mode",
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
