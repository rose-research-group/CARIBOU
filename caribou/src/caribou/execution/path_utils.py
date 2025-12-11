"""
Path management utilities for CARIBOU execution outputs.

This module handles:
- Default directory paths for runs, snippets, and reports
- Initializing output directory structures
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

from caribou.config import CARIBOU_HOME


# Default output directories if --output-dir is NOT specified
_DEFAULT_RUNS_DIR = CARIBOU_HOME / "runs"
_DEFAULT_SNIPPET_DIR = _DEFAULT_RUNS_DIR / "snippets"
_DEFAULT_BENCHMARK_LEDGER_PATH = _DEFAULT_RUNS_DIR / f"benchmark_history_{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}.jsonl"


def _init_paths(output_dir: Optional[Path] = None) -> None:
    """Ensure output directories exist before writing."""
    snippet_dir = output_dir / "snippets" if output_dir else _DEFAULT_SNIPPET_DIR
    ledger_path = output_dir / "benchmark_results.jsonl" if output_dir else _DEFAULT_BENCHMARK_LEDGER_PATH

    snippet_dir.mkdir(exist_ok=True, parents=True)
    ledger_path.parent.mkdir(exist_ok=True, parents=True)


def get_default_runs_dir() -> Path:
    """Get the default runs directory."""
    return _DEFAULT_RUNS_DIR


def get_default_snippet_dir() -> Path:
    """Get the default snippet directory."""
    return _DEFAULT_SNIPPET_DIR
