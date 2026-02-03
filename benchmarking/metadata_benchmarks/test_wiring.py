import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
META_DIR = ROOT / "benchmarking" / "metadata_benchmarks"


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


def main() -> int:
    failures = 0

    failures += _check_exists("metadata benchmarks README", META_DIR / "README.md")
    failures += _check_exists("prepare_benchmark", META_DIR / "prepare_benchmark.py")
    failures += _check_exists("metadata prompt", META_DIR / "prompts" / "metadata_prompt.txt")
    failures += _check_exists("full system prompt", META_DIR / "prompts" / "full_system_metadata_prompt.txt")
    failures += _check_exists("evaluate script", META_DIR / "evaluate_metadata_results.py")
    failures += _check_exists("plot script", META_DIR / "plot_metadata_benchmark_scores.py")
    failures += _check_exists("benchmark manifest", META_DIR / "benchmark_data" / "benchmark_manifest.csv")
    failures += _check_exists("ground truth", META_DIR / "benchmark_data" / "ground_truth_master.csv")
    failures += _check_slurm_prompts(META_DIR / "slurm")
    failures += _check_slurm_prompts(META_DIR / "bash")

    if failures:
        print(f"\nFAILED: {failures} issue(s) detected.")
        return 1
    print("\nPASSED: metadata benchmark wiring looks good.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
