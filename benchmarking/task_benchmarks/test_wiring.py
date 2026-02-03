import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TASK_DIR = ROOT / "benchmarking" / "task_benchmarks"


def _check_exists(label: str, path: Path) -> int:
    if path.exists():
        print(f"[ok] {label}: {path}")
        return 0
    print(f"[missing] {label}: {path}")
    return 1


def _check_slurm_prompts(slurm_dir: Path) -> int:
    failures = 0
    for script in sorted(slurm_dir.glob("*.sh")):
        text = script.read_text()
        if "PROMPT_PATH" not in text or "INITIAL_PROMPT" not in text:
            print(f"[missing] prompt wiring in {script}")
            failures += 1
    if failures == 0:
        print(f"[ok] prompt wiring present in {slurm_dir}")
    return failures


def _check_metrics_registry(metric_ids: list[str]) -> int:
    failures = 0
    try:
        from caribou.auto_metrics.registry import list_metrics
    except Exception as exc:
        print(f"[error] failed to import metrics registry: {exc}")
        return len(metric_ids)
    available = {m.id for m in list_metrics()}
    for metric_id in metric_ids:
        if metric_id not in available:
            print(f"[missing] metric id not registered: {metric_id}")
            failures += 1
        else:
            print(f"[ok] metric id registered: {metric_id}")
    return failures


def main() -> int:
    failures = 0

    failures += _check_exists("task benchmarks README", TASK_DIR / "README.md")
    failures += _check_exists("doublet prompt", TASK_DIR / "prompts" / "doublet_prompt.txt")
    failures += _check_exists("full QC prompt", TASK_DIR / "prompts" / "full_qc_prompt.txt")
    failures += _check_exists("QC prompt", TASK_DIR / "src" / "qc_prompt.py")
    failures += _check_exists("load prompt", TASK_DIR / "src" / "load_prompt.py")
    failures += _check_exists("one_shot_runner", TASK_DIR / "src" / "one_shot_runner.py")
    failures += _check_exists("results_collector", TASK_DIR / "src" / "results_collector.py")

    failures += _check_slurm_prompts(TASK_DIR / "slurm")
    failures += _check_slurm_prompts(TASK_DIR / "bash")

    failures += _check_metrics_registry(
        ["qc_benchmark", "load_data", "doublet_benchmark", "full_qc_benchmark"]
    )

    if failures:
        print(f"\nFAILED: {failures} issue(s) detected.")
        return 1
    print("\nPASSED: wiring checks look good.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
