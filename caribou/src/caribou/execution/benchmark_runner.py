"""
Benchmark execution and result tracking for CARIBOU agents.

This module handles:
- Executing benchmark modules inside sandboxes
- Saving benchmark results to JSONL ledgers
- Dumping code snippets for reproducibility
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from rich.console import Console
from rich.table import Table


def _dump_code_snippet(run_id: str, code: str, output_dir: Path) -> str:
    """Write <run_id>.py under the appropriate snippets dir and return the relative path."""
    snippet_dir = output_dir / "snippets"
    snippet_dir.mkdir(exist_ok=True, parents=True)

    snippet_path = snippet_dir / f"{run_id}.py"
    snippet_path.write_text(code, encoding="utf-8")
    # Return path relative to the main output directory for consistency in the log
    return str(snippet_path.relative_to(output_dir))


def _save_benchmark_record(
    *,
    run_id: str,
    results: dict,
    meta: dict,
    code: str | None,
    output_dir: Path
) -> None:
    """Append a JSONL record for the benchmark run to the correct ledger file."""
    record = {
        "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "run": run_id,
        "dataset": meta.get("name"),
        "results": results,
    }
    if code:
        record["code_path"] = _dump_code_snippet(run_id, code, output_dir)

    # Determine the correct ledger path
    ledger_path = output_dir / "benchmark_results.jsonl"

    with ledger_path.open("a") as fh:
        fh.write(json.dumps(record) + "\n")


def run_benchmark(
    console: Console,
    mgr: object,  # SandboxManager
    benchmark_module: Path,
    *,
    is_auto: bool,
    output_dir: Path,
    metadata: Optional[Dict] = None,
    agent_name: Optional[str] = None,
    code_snippet: Optional[str] = None,
) -> str:
    """
    Execute a benchmark module inside the sandbox.
    In auto mode, saves results and returns a result string for the history.
    In interactive mode, prints results to the console.
    """
    console.print(f"\n[bold cyan]Running benchmark module: {benchmark_module.name}[/bold cyan]")
    autometric_base_path = benchmark_module.parent / "AutoMetric.py"
    try:
        with open(autometric_base_path, "r") as f:
            autometric_code = f.read()
        with open(benchmark_module, "r") as f:
            benchmark_code = f.read()
    except FileNotFoundError as e:
        err = f"Benchmark module or AutoMetric.py not found: {e}"
        console.print(f"[red]{err}[/red]")
        return err if is_auto else ""

    console.print("[cyan]Executing benchmark code...[/cyan]")

    payload = f"""
import json, sys, types
try:
    import anndata
except ImportError:
    anndata = None

# Load AutoMetric into a module to satisfy imports inside the metric script
_auto_mod = types.ModuleType("AutoMetric")
sys.modules["AutoMetric"] = _auto_mod
exec({autometric_code!r}, _auto_mod.__dict__)

# Load the benchmark module code into its own module namespace
_metric_mod = types.ModuleType("{benchmark_module.stem}")
sys.modules["{benchmark_module.stem}"] = _metric_mod
exec({benchmark_code!r}, _metric_mod.__dict__)

# Identify the first AutoMetric subclass defined in the benchmark module
_AM = _auto_mod.__dict__.get("AutoMetric")
_metric_cls = None
for _name, _obj in list(_metric_mod.__dict__.items()):
    if _AM and isinstance(_obj, type) and issubclass(_obj, _AM):
        _metric_cls = _obj
        break

if _metric_cls is None:
    raise RuntimeError("No AutoMetric subclass found in benchmark module.")
if "adata" not in globals():
    raise RuntimeError("No adata available in the sandbox session for benchmark execution.")
if anndata is not None and not isinstance(adata, anndata.AnnData):
    raise RuntimeError(f"'adata' is {{type(adata)}}; expected an AnnData object.")

_metric = _metric_cls()
_results = _metric.metric(adata)
print(json.dumps(_results))
"""

    try:
        exec_result = mgr.exec_code(payload, timeout=300)

        table = Table(title="Benchmark Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        stdout = exec_result.get("stdout", "")
        result_dict = {}
        try:
            result_dict = json.loads(stdout.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError) as e:
            console.print(f"[yellow]Warning: Could not parse JSON from stdout: {e}[/yellow]")

        if exec_result.get("status") == "ok" and isinstance(result_dict, dict):
            for key, value in result_dict.items():
                table.add_row(str(key), str(value))
            if is_auto:
                _save_benchmark_record(
                    run_id=f"{benchmark_module.stem}:{agent_name}:{int(time.time())}",
                    results=result_dict,
                    meta=metadata if metadata else {},
                    code=code_snippet,
                    output_dir=output_dir,
                )
        else:
            error_message = exec_result.get("stderr") or "An unknown error occurred."
            table.add_row("Error", error_message)

        console.print(table)
        return "Benchmark results:\n" + json.dumps(result_dict)

    except Exception as exc:
        err_msg = f"Benchmark execution failed: {exc}"
        console.print(f"[red]{err_msg}[/red]")
        return err_msg
