"""
Filter benchmark results by excluding specific LLM backends.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List


def filter_results(
    input_json: Path,
    output_json: Path,
    exclude_llms: List[str],
) -> None:
    """
    Filter results JSON to exclude specific LLM backends.

    Args:
        input_json: Path to input summary JSON
        output_json: Path to output filtered JSON
        exclude_llms: List of LLM backend names to exclude (e.g., ['claude'])
    """
    records = json.loads(input_json.read_text())

    if not isinstance(records, list):
        raise ValueError("Expected a JSON list of records")

    # Filter out excluded LLMs
    filtered = [
        rec for rec in records
        if rec.get("llm_backend") not in exclude_llms
    ]

    print(f"Original records: {len(records)}")
    print(f"Filtered records: {len(filtered)}")
    print(f"Excluded {len(records) - len(filtered)} records from LLMs: {', '.join(exclude_llms)}")

    # Write filtered results
    output_json.write_text(json.dumps(filtered, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter benchmark results by excluding specific LLM backends."
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        required=True,
        help="Path to input summary JSON",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="Path to output filtered JSON",
    )
    parser.add_argument(
        "--exclude-llm",
        action="append",
        dest="exclude_llms",
        help="LLM backend to exclude (can be specified multiple times)",
    )
    args = parser.parse_args()

    if not args.exclude_llms:
        print("No LLMs to exclude specified. Copying input to output.")
        args.output_json.write_text(args.input_json.read_text())
        return

    filter_results(
        args.input_json,
        args.output_json,
        args.exclude_llms,
    )


if __name__ == "__main__":
    main()
