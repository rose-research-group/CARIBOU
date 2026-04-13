import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _savefig(fig: plt.Figure, output_path: Path, **kwargs) -> None:
    fig.savefig(output_path, **kwargs)
    svg_kwargs = {k: v for k, v in kwargs.items() if k != "dpi"}
    fig.savefig(output_path.with_suffix(".svg"), **svg_kwargs)


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
    ax.bar(x, cleaned, color="#A1F289")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    _savefig(fig, output_path, dpi=150)
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
    im = ax.imshow(values, aspect="auto", cmap="plasma", vmin=None if not ylim else ylim[0], vmax=None if not ylim else ylim[1])
    ax.set_title(title)
    ax.set_xticks(np.arange(len(llms)))
    ax.set_xticklabels(llms, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(modes)))
    ax.set_yticklabels(modes)
    ax.set_xlabel("LLM")
    ax.set_ylabel(ylabel)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    _savefig(fig, output_path, dpi=150)
    plt.close(fig)


def _plot_performance_time_combined(task: str, records: List[Dict], output_path: Path) -> None:
    """Create a mode × LLM matrix with color = success and size = runtime."""
    if not records:
        return

    modes = sorted({r.get("mode") or "" for r in records})
    llms = sorted({r.get("llm_backend") or "" for r in records})
    if not modes or not llms:
        return

    success = np.full((len(modes), len(llms)), np.nan)
    runtime = np.full((len(modes), len(llms)), np.nan)

    for r in records:
        mode = r.get("mode") or ""
        llm = r.get("llm_backend") or ""
        if mode not in modes or llm not in llms:
            continue
        m = modes.index(mode)
        l = llms.index(llm)
        success_val = r.get("autometric_success_rate")
        runtime_val = r.get("avg_runtime_seconds")
        success[m, l] = float(success_val) if isinstance(success_val, (int, float)) else np.nan
        runtime[m, l] = float(runtime_val) if isinstance(runtime_val, (int, float)) else np.nan

    if np.all(np.isnan(success)) and np.all(np.isnan(runtime)):
        print(f"Skipping {output_path.name}: no numeric values for success/runtime")
        return

    fig, ax = plt.subplots(figsize=(max(5, len(llms) * 1.2), max(3.5, len(modes) * 0.9)))
    im = ax.imshow(success, aspect="auto", cmap="plasma", vmin=0, vmax=1)

    ax.set_title(f"{task}: Autometric Success (color) & Runtime (size)", fontsize=12, fontweight="bold")
    ax.set_xticks(np.arange(len(llms)))
    ax.set_xticklabels(llms, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(modes)))
    ax.set_yticklabels(modes)
    ax.set_xlabel("LLM", fontsize=11)
    ax.set_ylabel("Mode", fontsize=11)

    runtime_vals = runtime[~np.isnan(runtime)]
    size_min, size_max = 60, 420
    if runtime_vals.size:
        r_min = float(runtime_vals.min())
        r_max = float(runtime_vals.max())
        denom = r_max - r_min if r_max > r_min else 1.0
        for i in range(len(modes)):
            for j in range(len(llms)):
                if np.isnan(runtime[i, j]):
                    continue
                size = size_min + (runtime[i, j] - r_min) / denom * (size_max - size_min)
                ax.scatter(j, i, s=size, facecolor="none", edgecolor="black", linewidth=0.8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Autometric Success Rate", rotation=270, labelpad=16, fontsize=10)

    fig.tight_layout()
    _savefig(fig, output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ==================== SCALABILITY PLOTTING FUNCTIONS ====================

def _get_task_complexity() -> Dict[str, int]:
    """Define task complexity scores (excluding qc_task)."""
    return {
        "load_data_task": 1,
        "doublet_task": 2,
        "data_adequacy_task": 2,
        "full_qc_task": 3,
        "deg_task": 4,
        "batch_correction_task": 5,
    }


def _plot_complexity_ladder(records: List[Dict], output_path: Path, llm_backend: Optional[str] = None) -> None:
    """Plot success rate vs task complexity to show scalability."""
    complexity_map = _get_task_complexity()

    # Filter records to only include tasks with defined complexity (exclude qc_task)
    filtered = [r for r in records if r.get("task") in complexity_map]

    # Filter by LLM backend if specified
    if llm_backend:
        filtered = [r for r in filtered if r.get("llm_backend") == llm_backend]

    if not filtered:
        print(f"Skipping complexity ladder for {llm_backend or 'all'}: no tasks with defined complexity")
        return

    # Group by task and mode
    df_data = []
    for rec in filtered:
        df_data.append({
            "task": rec.get("task"),
            "mode": rec.get("mode"),
            "llm_backend": rec.get("llm_backend"),
            "complexity": complexity_map.get(rec.get("task"), 0),
            "autometric_rate": rec.get("autometric_success_rate", 0),
            "runtime": rec.get("avg_runtime_seconds", 0),
        })

    df = pd.DataFrame(df_data)

    # Group by complexity and mode, average autometric rate
    grouped = df.groupby(["complexity", "mode"]).agg({
        "autometric_rate": "mean",
        "task": "first",
    }).reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))

    modes = sorted(grouped["mode"].unique())
    colors = {"one_shot": "#BFF1F5", "single_agent": "#A1F289", "full_system": "#FF9898"}

    for mode in modes:
        mode_data = grouped[grouped["mode"] == mode].sort_values("complexity")
        ax.plot(mode_data["complexity"], mode_data["autometric_rate"],
                marker='o', linewidth=2, markersize=8,
                label=mode, color=colors.get(mode, "#A1F289"))

    ax.axhline(y=0.9, color='#FF9898', linestyle='--', alpha=0.5, linewidth=1, label='90% threshold')
    ax.set_xlabel("Task Complexity", fontsize=12, fontweight='bold')
    ax.set_ylabel("Autometric Success Rate", fontsize=12, fontweight='bold')
    ax.set_title("Task Complexity Ladder: Success Rate vs Complexity", fontsize=13, fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(sorted(complexity_map.values()))
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best')

    fig.tight_layout()
    _savefig(fig, output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_efficiency_frontier(records: List[Dict], output_path: Path, llm_backend: Optional[str] = None) -> None:
    """Plot efficiency frontier: success vs runtime scatter."""
    complexity_map = _get_task_complexity()

    # Filter records to only include tasks with defined complexity
    filtered = [r for r in records if r.get("task") in complexity_map
                and r.get("autometric_success_rate") is not None
                and r.get("avg_runtime_seconds") is not None]

    # Filter by LLM backend if specified
    if llm_backend:
        filtered = [r for r in filtered if r.get("llm_backend") == llm_backend]

    if not filtered:
        print(f"Skipping efficiency frontier for {llm_backend or 'all'}: no valid data")
        return

    df_data = []
    for rec in filtered:
        df_data.append({
            "task": rec.get("task"),
            "mode": rec.get("mode"),
            "llm_backend": rec.get("llm_backend"),
            "complexity": complexity_map.get(rec.get("task"), 0),
            "autometric_rate": rec.get("autometric_success_rate", 0),
            "runtime": rec.get("avg_runtime_seconds", 0),
        })

    df = pd.DataFrame(df_data)

    fig, ax = plt.subplots(figsize=(10, 7))

    modes = sorted(df["mode"].unique())
    colors = {"one_shot": "#BFF1F5", "single_agent": "#A1F289", "full_system": "#FF9898"}

    for mode in modes:
        mode_data = df[df["mode"] == mode]
        scatter = ax.scatter(mode_data["runtime"], mode_data["autometric_rate"],
                            s=mode_data["complexity"]*100, alpha=0.6,
                            c=[colors.get(mode, "#A1F289")]*len(mode_data),
                            label=mode, edgecolors='black', linewidth=0.5)

    # Add quadrant lines
    if len(df) > 0:
        median_runtime = df["runtime"].median()
        ax.axhline(y=0.9, color='#FF9898', linestyle='--', alpha=0.5, linewidth=1.5, label='High accuracy (90%)')
        ax.axvline(x=median_runtime, color='#BFF1F5', linestyle='--', alpha=0.5, linewidth=1.5, label=f'Median runtime ({median_runtime:.0f}s)')

    ax.set_xlabel("Average Runtime (seconds)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Autometric Success Rate", fontsize=12, fontweight='bold')
    ax.set_title("Efficiency Frontier: Accuracy vs Runtime\n(Bubble size = task complexity)", fontsize=13, fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', framealpha=0.9)

    fig.tight_layout()
    _savefig(fig, output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_reliability_heatmap(records: List[Dict], output_path: Path, llm_backend: Optional[str] = None) -> None:
    """Plot reliability heatmap: task × mode."""
    complexity_map = _get_task_complexity()

    # Filter records to only include tasks with defined complexity
    filtered = [r for r in records if r.get("task") in complexity_map]

    # Filter by LLM backend if specified
    if llm_backend:
        filtered = [r for r in filtered if r.get("llm_backend") == llm_backend]

    if not filtered:
        print(f"Skipping reliability heatmap for {llm_backend or 'all'}: no valid data")
        return

    df_data = []
    for rec in filtered:
        df_data.append({
            "task": rec.get("task"),
            "mode": rec.get("mode"),
            "autometric_rate": rec.get("autometric_success_rate", 0),
        })

    df = pd.DataFrame(df_data)

    # Aggregate by task and mode
    pivot = df.pivot_table(values="autometric_rate", index="task", columns="mode", aggfunc="mean")

    # Sort tasks by complexity
    task_order = sorted(pivot.index, key=lambda t: complexity_map.get(t, 0))
    pivot = pivot.reindex(task_order)

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(pivot.values, aspect='auto', cmap='plasma', vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha='right')
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                text = ax.text(j, i, f'{val:.2f}',
                             ha="center", va="center", color="black" if val > 0.5 else "white",
                             fontsize=10, fontweight='bold')

    ax.set_title("Reliability Scaling Heatmap: Task × Mode", fontsize=13, fontweight='bold')
    ax.set_xlabel("Execution Mode", fontsize=11, fontweight='bold')
    ax.set_ylabel("Task (by complexity)", fontsize=11, fontweight='bold')

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Autometric Success Rate", rotation=270, labelpad=20, fontweight='bold')

    fig.tight_layout()
    _savefig(fig, output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_cost_benefit_analysis(records: List[Dict], output_path: Path, llm_backend: Optional[str] = None) -> None:
    """Plot cost-benefit analysis per task: runtime vs success."""
    complexity_map = _get_task_complexity()

    # Filter records to only include tasks with defined complexity
    filtered = [r for r in records if r.get("task") in complexity_map
                and r.get("autometric_success_rate") is not None
                and r.get("avg_runtime_seconds") is not None]

    # Filter by LLM backend if specified
    if llm_backend:
        filtered = [r for r in filtered if r.get("llm_backend") == llm_backend]

    if not filtered:
        print(f"Skipping cost-benefit analysis for {llm_backend or 'all'}: no valid data")
        return

    df_data = []
    for rec in filtered:
        df_data.append({
            "task": rec.get("task"),
            "mode": rec.get("mode"),
            "complexity": complexity_map.get(rec.get("task"), 0),
            "autometric_rate": rec.get("autometric_success_rate", 0),
            "runtime": rec.get("avg_runtime_seconds", 0),
        })

    df = pd.DataFrame(df_data)

    # Group by task, average across modes
    task_summary = df.groupby("task").agg({
        "complexity": "first",
        "autometric_rate": "mean",
        "runtime": "mean",
    }).reset_index()

    # Sort by complexity
    task_summary = task_summary.sort_values("complexity")

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(task_summary))
    width = 0.35

    ax2 = ax.twinx()

    bars1 = ax.bar(x - width/2, task_summary["autometric_rate"], width,
                   label='Autometric Rate', color='#A1F289', alpha=0.8)
    bars2 = ax2.bar(x + width/2, task_summary["runtime"], width,
                    label='Avg Runtime (s)', color='#BFF1F5', alpha=0.8)

    ax.set_xlabel("Task (by complexity)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Autometric Success Rate", fontsize=11, fontweight='bold', color='#A1F289')
    ax2.set_ylabel("Average Runtime (seconds)", fontsize=11, fontweight='bold', color='#BFF1F5')

    ax.set_title("Cost-Benefit Analysis: Success vs Runtime by Task", fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(task_summary["task"], rotation=30, ha='right')
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis='y', labelcolor='#A1F289')
    ax2.tick_params(axis='y', labelcolor='#BFF1F5')

    ax.axhline(y=0.9, color='#FF9898', linestyle='--', alpha=0.5, linewidth=1)

    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    ax.grid(True, alpha=0.3, linestyle='--', axis='y')

    fig.tight_layout()
    _savefig(fig, output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_complexity_resilience(records: List[Dict], output_path: Path, llm_backend: Optional[str] = None) -> None:
    """Plot complexity vs resilience: how success degrades with complexity."""
    complexity_map = _get_task_complexity()

    # Filter records to only include tasks with defined complexity
    filtered = [r for r in records if r.get("task") in complexity_map]

    # Filter by LLM backend if specified
    if llm_backend:
        filtered = [r for r in filtered if r.get("llm_backend") == llm_backend]

    if not filtered:
        print(f"Skipping complexity resilience for {llm_backend or 'all'}: no valid data")
        return

    df_data = []
    for rec in filtered:
        df_data.append({
            "task": rec.get("task"),
            "mode": rec.get("mode"),
            "llm_backend": rec.get("llm_backend"),
            "complexity": complexity_map.get(rec.get("task"), 0),
            "autometric_rate": rec.get("autometric_success_rate", 0),
        })

    df = pd.DataFrame(df_data)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot by LLM backend
    llms = sorted(df["llm_backend"].unique())
    colors = {"chatgpt": "#FF9898", "deepseek": "#BFF1F5", "claude": "#A1F289"}

    for llm in llms:
        llm_data = df[df["llm_backend"] == llm].groupby("complexity").agg({
            "autometric_rate": "mean",
        }).reset_index()

        ax.plot(llm_data["complexity"], llm_data["autometric_rate"],
                marker='o', linewidth=2.5, markersize=10,
                label=llm, color=colors.get(llm, "#A1F289"))

    ax.axhline(y=0.9, color='#FF9898', linestyle='--', alpha=0.5, linewidth=1.5, label='90% threshold')
    ax.set_xlabel("Task Complexity", fontsize=12, fontweight='bold')
    ax.set_ylabel("Autometric Success Rate", fontsize=12, fontweight='bold')
    ax.set_title("Complexity vs Resilience: How Success Degrades with Complexity", fontsize=13, fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(sorted(complexity_map.values()))
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', framealpha=0.9)

    fig.tight_layout()
    _savefig(fig, output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_execution_mode_dashboard(records: List[Dict], output_path: Path, llm_backend: Optional[str] = None) -> None:
    """Create a dashboard comparing execution modes across metrics."""
    complexity_map = _get_task_complexity()

    # Filter records to only include tasks with defined complexity
    filtered = [r for r in records if r.get("task") in complexity_map]

    # Filter by LLM backend if specified
    if llm_backend:
        filtered = [r for r in filtered if r.get("llm_backend") == llm_backend]

    if not filtered:
        print(f"Skipping execution mode dashboard for {llm_backend or 'all'}: no valid data")
        return

    df_data = []
    for rec in filtered:
        df_data.append({
            "task": rec.get("task"),
            "mode": rec.get("mode"),
            "complexity": complexity_map.get(rec.get("task"), 0),
            "autometric_rate": rec.get("autometric_success_rate", 0),
            "runtime": rec.get("avg_runtime_seconds", 0),
            "corrections": rec.get("avg_correction_count", 0),
        })

    df = pd.DataFrame(df_data)

    # Aggregate by mode
    mode_summary = df.groupby("mode").agg({
        "autometric_rate": "mean",
        "runtime": "mean",
        "corrections": "mean",
    }).reset_index()

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Execution Mode Comparison Dashboard", fontsize=15, fontweight='bold')

    modes = sorted(mode_summary["mode"].unique())
    colors_map = {"one_shot": "#BFF1F5", "single_agent": "#A1F289", "full_system": "#FF9898"}
    colors = [colors_map.get(m, "#A1F289") for m in modes]

    # Top-left: Autometric rate by mode
    ax = axes[0, 0]
    bars = ax.bar(modes, mode_summary["autometric_rate"], color=colors, alpha=0.8)
    ax.set_ylabel("Autometric Success Rate", fontweight='bold')
    ax.set_title("Average Success Rate by Mode", fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.9, color='#FF9898', linestyle='--', alpha=0.5)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    for bar, val in zip(bars, mode_summary["autometric_rate"]):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10)

    # Top-middle: Runtime by mode
    ax = axes[0, 1]
    bars = ax.bar(modes, mode_summary["runtime"], color=colors, alpha=0.8)
    ax.set_ylabel("Average Runtime (seconds)", fontweight='bold')
    ax.set_title("Average Runtime by Mode", fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    for bar, val in zip(bars, mode_summary["runtime"]):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.0f}s', ha='center', va='bottom', fontsize=10)

    # Top-right: Corrections by mode
    ax = axes[0, 2]
    bars = ax.bar(modes, mode_summary["corrections"], color=colors, alpha=0.8)
    ax.set_ylabel("Average Corrections", fontweight='bold')
    ax.set_title("Average Corrections by Mode", fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    for bar, val in zip(bars, mode_summary["corrections"]):
        height = bar.get_height()
        if not np.isnan(val) and val > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=10)

    # Bottom-left: Success by complexity and mode
    ax = axes[1, 0]
    for mode in modes:
        mode_data = df[df["mode"] == mode].groupby("complexity").agg({
            "autometric_rate": "mean",
        }).reset_index()
        ax.plot(mode_data["complexity"], mode_data["autometric_rate"],
                marker='o', linewidth=2, markersize=8,
                label=mode, color=colors_map.get(mode, "#A1F289"))
    ax.axhline(y=0.9, color='#FF9898', linestyle='--', alpha=0.5)
    ax.set_xlabel("Task Complexity", fontweight='bold')
    ax.set_ylabel("Autometric Success Rate", fontweight='bold')
    ax.set_title("Success vs Complexity by Mode", fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(sorted(complexity_map.values()))
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best')

    # Bottom-middle: Runtime vs complexity
    ax = axes[1, 1]
    for mode in modes:
        mode_data = df[df["mode"] == mode].groupby("complexity").agg({
            "runtime": "mean",
        }).reset_index()
        ax.plot(mode_data["complexity"], mode_data["runtime"],
                marker='o', linewidth=2, markersize=8,
                label=mode, color=colors_map.get(mode, "#A1F289"))
    ax.set_xlabel("Task Complexity", fontweight='bold')
    ax.set_ylabel("Average Runtime (seconds)", fontweight='bold')
    ax.set_title("Runtime vs Complexity by Mode", fontweight='bold')
    ax.set_xticks(sorted(complexity_map.values()))
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best')

    # Bottom-right: Efficiency scatter (success vs runtime)
    ax = axes[1, 2]
    for mode in modes:
        mode_data = df[df["mode"] == mode]
        ax.scatter(mode_data["runtime"], mode_data["autometric_rate"],
                  s=mode_data["complexity"]*100, alpha=0.6,
                  c=[colors_map.get(mode, "#A1F289")]*len(mode_data),
                  label=mode, edgecolors='black', linewidth=0.5)
    ax.axhline(y=0.9, color='#FF9898', linestyle='--', alpha=0.5)
    if len(df) > 0:
        median_runtime = df["runtime"].median()
        ax.axvline(x=median_runtime, color='#BFF1F5', linestyle='--', alpha=0.5)
    ax.set_xlabel("Average Runtime (seconds)", fontweight='bold')
    ax.set_ylabel("Autometric Success Rate", fontweight='bold')
    ax.set_title("Efficiency Frontier by Mode", fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best')

    fig.tight_layout()
    _savefig(fig, output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ==================== END SCALABILITY PLOTTING FUNCTIONS ====================


def plot_summary(summary_path: Path, output_dir: Path) -> None:
    records = _load_summary(summary_path)
    if records and "success_rate" not in records[0]:
        records = _summarize_raw(records)
    grouped = _group_by_task(records)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create performance_time subfolder
    performance_time_dir = output_dir / "performance_time"
    performance_time_dir.mkdir(parents=True, exist_ok=True)

    # Create scalability subfolder
    scalability_dir = output_dir / "scalability"
    scalability_dir.mkdir(parents=True, exist_ok=True)

    # Generate scalability plots (excluding qc_task)
    print("Generating scalability plots...")

    # Get all unique LLM backends
    llm_backends = sorted({r.get("llm_backend") for r in records if r.get("llm_backend")})

    # Generate plots for each LLM backend
    for llm in llm_backends:
        llm_dir = scalability_dir / llm
        llm_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Generating plots for {llm}...")
        _plot_complexity_ladder(records, llm_dir / "complexity_ladder.png", llm_backend=llm)
        _plot_efficiency_frontier(records, llm_dir / "efficiency_frontier.png", llm_backend=llm)
        _plot_reliability_heatmap(records, llm_dir / "reliability_heatmap.png", llm_backend=llm)
        _plot_cost_benefit_analysis(records, llm_dir / "cost_benefit_analysis.png", llm_backend=llm)
        _plot_complexity_resilience(records, llm_dir / "complexity_resilience.png", llm_backend=llm)
        _plot_execution_mode_dashboard(records, llm_dir / "execution_mode_dashboard.png", llm_backend=llm)

    # Also generate combined plots across all LLMs
    combined_dir = scalability_dir / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)
    print("  Generating combined plots across all LLMs...")
    _plot_complexity_ladder(records, combined_dir / "complexity_ladder.png")
    _plot_efficiency_frontier(records, combined_dir / "efficiency_frontier.png")
    _plot_reliability_heatmap(records, combined_dir / "reliability_heatmap.png")
    _plot_cost_benefit_analysis(records, combined_dir / "cost_benefit_analysis.png")
    _plot_complexity_resilience(records, combined_dir / "complexity_resilience.png")
    _plot_execution_mode_dashboard(records, combined_dir / "execution_mode_dashboard.png")

    for task, task_records in grouped.items():
        # Create combined performance/time plot for presentations
        _plot_performance_time_combined(
            task=task,
            records=task_records,
            output_path=performance_time_dir / f"{task}_performance_time.png"
        )

        # Note: success_rate plots removed - only meaningful for one_shot mode
        # Note: avg_duration plots removed - overlap with runtime in non-meaningful way
        _plot_grouped_bars(
            title=f"{task}: autometric success rate",
            records=task_records,
            value_key="autometric_success_rate",
            output_path=output_dir / f"{task}_autometric_rate.png",
            ylabel="Autometric success rate",
            ylim=(0, 1.05),
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
    print(f"\n✓ Saved plots to {output_dir}")
    print(f"✓ Combined performance/time plots: {output_dir / 'performance_time'}")
    print(f"✓ Scalability plots by provider: {output_dir / 'scalability'}/{{provider}}")
    print(f"✓ Combined scalability plots: {output_dir / 'scalability'}/combined")


if __name__ == "__main__":
    main()
