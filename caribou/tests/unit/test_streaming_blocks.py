"""Web loop: block_id on code and artifact events, block_changed events, and
blocks.json next to work-items/ (output_dir.parent)."""

from __future__ import annotations

from pathlib import Path

import pytest

from caribou.execution.blocks import BlockError, load_blocks
from caribou.execution.event_ids import make_action_id

from .test_streaming_artifacts import WritingSandbox, _write
from .test_streaming_runner_step0 import RecordingLLM, _run_auto, _stub_rag


def _of(events, event_type):
    return [e["data"] for e in events if e["type"] == event_type]


def test_code_and_artifact_events_carry_the_block_id(tmp_path, monkeypatch):
    _stub_rag(monkeypatch)

    def produce(out: Path):
        _write(out / "umap.png", "x", 1_000)

    events = _run_auto(
        tmp_path,
        RecordingLLM(["```python\npass\n```"]),
        session_id="sess",
        sandbox_manager=WritingSandbox(tmp_path / "outputs", [produce]),
    )

    action_id = make_action_id("sess", 1, 1)
    assert _of(events, "code_submitted")[0]["block_id"] == "blk-0001"
    assert _of(events, "code_result")[0]["block_id"] == "blk-0001"
    artifact = _of(events, "artifact")[0]
    assert artifact["block_id"] == "blk-0001"
    assert artifact["artifact"]["action_id"] == action_id

    types = [e["type"] for e in events]
    assert types.index("block_changed") < types.index("code_submitted")
    final = _of(events, "block_changed")[-1]["block"]
    assert final["status"] == "ok"
    assert final["artifact_paths"] == ["umap.png"]
    assert final["action_ids"] == [action_id]

    index = load_blocks(tmp_path / "blocks.json")
    assert index["blocks"] == [final]


def test_failed_sandbox_call_still_finishes_and_closes_the_block(tmp_path, monkeypatch):
    _stub_rag(monkeypatch)

    class ExplodingSandbox:
        def exec_code(self, _code, timeout):
            raise RuntimeError("sandbox gone")

    events = _run_auto(
        tmp_path,
        RecordingLLM(["```python\npass\n```"]),
        sandbox_manager=ExplodingSandbox(),
    )

    assert _of(events, "code_result")[0]["block_id"] == "blk-0001"
    final = _of(events, "block_changed")[-1]["block"]
    assert final["status"] == "error"
    assert final["failed_action_ids"] == final["action_ids"]


def test_work_item_block_closes_when_the_item_is_done(tmp_path, monkeypatch):
    _stub_rag(monkeypatch)
    events = _run_auto(
        tmp_path,
        RecordingLLM(
            [
                'open_work_item "QC" "filter low quality cells"',
                "```python\npass\n```",
                'close_work_item 0 "filtered"',
            ]
        ),
        max_turns=3,
    )

    blocks = [d["block"] for d in _of(events, "block_changed")]
    assert {b["block_id"] for b in blocks} == {"blk-0001"}
    assert blocks[0]["work_item_id"] == 0
    assert blocks[0]["implicit"] is False
    assert blocks[-1]["status"] == "ok"
    done_index = next(
        i
        for i, e in enumerate(events)
        if e["type"] == "work_item_changed" and e["data"]["item"]["status"] == "Done"
    )
    assert events[done_index + 1]["type"] == "block_changed"


def test_resumed_web_session_continues_block_ids(tmp_path, monkeypatch):
    _stub_rag(monkeypatch)
    _run_auto(tmp_path, RecordingLLM(["```python\npass\n```"]), session_id="sess")
    events = _run_auto(
        tmp_path,
        RecordingLLM(["```python\npass\n```"]),
        session_id="sess",
        resume_state={"turns_completed": 1},
        max_turns=2,
    )
    assert _of(events, "code_submitted")[0]["block_id"] == "blk-0002"
    index = load_blocks(tmp_path / "blocks.json")
    assert [b["block_id"] for b in index["blocks"]] == ["blk-0001", "blk-0002"]


def test_malformed_blocks_file_fails_loudly(tmp_path, monkeypatch):
    _stub_rag(monkeypatch)
    (tmp_path / "blocks.json").write_text("{not json")
    with pytest.raises(BlockError):
        _run_auto(tmp_path, RecordingLLM(["```python\npass\n```"]))
