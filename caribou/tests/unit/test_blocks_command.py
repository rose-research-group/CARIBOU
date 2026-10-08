"""The `/blocks` REPL command reads blocks.json beside the work-items store."""

from __future__ import annotations

from pathlib import Path

import pytest
from rich.console import Console

from caribou.execution.blocks import BlockError
from caribou.execution.user_commands import (
    USER_COMMANDS,
    UserCommandContext,
    dispatch_user_command,
    format_block_line,
)
from caribou.execution.work_items import WorkItemPolicy, WorkItemStore

from .test_blocks_route import _block, _write_blocks


def _ctx(tmp_path: Path) -> UserCommandContext:
    store = WorkItemStore(
        tmp_path / "work-items", session_id="source-id", policy=WorkItemPolicy()
    )
    return UserCommandContext(
        console=Console(force_terminal=False, width=200, record=True),
        run_id="source-id",
        turn=1,
        history=[],
        artifacts=None,
        agent_system=None,
        current_agent=None,
        llm_client=None,
        model_name="fake",
        memory_manager=None,
        report_memory=None,
        benchmark_modules=None,
        sandbox_manager=None,
        output_dir=tmp_path,
        evaluator_runtime=None,
        work_items=store,
    )


def test_format_matches_the_contract_line() -> None:
    block = _block(
        1,
        action_ids=["a1", "a2", "a3", "a4"],
        failed_action_ids=["a2"],
        artifact_paths=["figures/qc.png"],
    )
    assert format_block_line(block) == (
        "blk-0001 ✓ QC  [work item #2, attempt 1]  turns 3–5  4 actions, 1 artifact"
    )


@pytest.mark.parametrize(
    ("status", "glyph"),
    [("running", "▶"), ("ok", "✓"), ("warn", "⚠"), ("error", "✗")],
)
def test_status_glyphs(status: str, glyph: str) -> None:
    assert format_block_line(_block(1, status=status)).startswith(f"blk-0001 {glyph} ")


def test_implicit_block_and_singular_plural_counts() -> None:
    block = _block(
        3,
        work_item_id=None,
        implicit=True,
        title="analyst (no work item)",
        turn_start=7,
        turn_end=7,
        action_ids=["a1"],
        artifact_paths=[],
    )
    assert format_block_line(block) == (
        "blk-0003 ✓ analyst (no work item)  [no work item]  turns 7–7  "
        "1 action, 0 artifacts"
    )


def test_command_prints_one_line_per_block(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    _write_blocks(
        tmp_path / "blocks.json",
        [_block(1), _block(2, implicit=True, work_item_id=None, status="running")],
    )

    assert dispatch_user_command("/blocks", ctx) is True

    out = ctx.console.export_text()
    # Rendered as plain text, so the brackets survive Rich markup parsing.
    assert "blk-0001 ✓ QC  [work item #2, attempt 1]  turns 3–5" in out
    assert "blk-0002 ▶ QC  [no work item]  turns 3–5" in out


def test_command_without_blocks_file_says_so(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    USER_COMMANDS["/blocks"].handler("", ctx)
    assert "No blocks recorded for this session" in ctx.console.export_text()


def test_command_with_empty_blocks_list_says_so(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    _write_blocks(tmp_path / "blocks.json", [])
    USER_COMMANDS["/blocks"].handler("", ctx)
    assert "No blocks recorded for this session" in ctx.console.export_text()


def test_command_raises_on_a_malformed_blocks_file(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    (tmp_path / "blocks.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(BlockError):
        USER_COMMANDS["/blocks"].handler("", ctx)


def test_help_lists_blocks(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    USER_COMMANDS["/help"].handler("", ctx)
    assert "/blocks" in ctx.console.export_text()
