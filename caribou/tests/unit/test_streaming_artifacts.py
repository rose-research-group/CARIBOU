"""Web `artifact` events: dedupe by (path, mtime_ns), recursive scan with
internal locations excluded, and attribution to the producing code block."""

from __future__ import annotations

import os
from pathlib import Path

from caribou.execution.event_ids import make_action_id
from caribou.execution.session_recovery import LIVE_DATASET

from .test_streaming_runner_step0 import RecordingLLM, _run_auto, _stub_rag


class WritingSandbox:
    """Each exec runs the next scripted writer against output_dir."""

    def __init__(self, output_dir: Path, writers):
        self.output_dir = output_dir
        self.writers = list(writers)

    def exec_code(self, _code, timeout):
        self.writers.pop(0)(self.output_dir)
        return {"status": "ok", "stdout": "", "stderr": ""}


def _write(path: Path, text: str, mtime_ns: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    os.utime(path, ns=(mtime_ns, mtime_ns))


def _artifacts(events):
    return [ev["data"]["artifact"] for ev in events if ev["type"] == "artifact"]


def _run(tmp_path, monkeypatch, writers, *, blocks_per_turn=1, max_turns=1):
    _stub_rag(monkeypatch)
    output_dir = tmp_path / "outputs"
    response = "\n".join(["```python\npass\n```"] * blocks_per_turn)
    llm = RecordingLLM([response])
    return _run_auto(
        tmp_path,
        llm,
        session_id="sess",
        max_turns=max_turns,
        sandbox_manager=WritingSandbox(output_dir, writers),
    )


def test_overwritten_file_is_reported_again_with_new_mtime(tmp_path, monkeypatch):
    def first(out):
        _write(out / "umap.png", "v1", 1_000)

    def unchanged(out):
        pass

    def overwrite(out):
        _write(out / "umap.png", "v2!", 2_000)

    events = _run(tmp_path, monkeypatch, [first, unchanged, overwrite], blocks_per_turn=3)

    arts = _artifacts(events)
    assert [(a["path"], a["mtime_ns"], a["action_id"]) for a in arts] == [
        ("umap.png", 1_000, make_action_id("sess", 1, 1)),
        ("umap.png", 2_000, make_action_id("sess", 1, 3)),
    ]
    assert arts[1]["size_bytes"] == 3
    assert arts[1]["filename"] == "umap.png"
    assert arts[1]["local_path"] == str(tmp_path / "outputs" / "umap.png")


def test_mtime_rollback_is_also_a_change(tmp_path, monkeypatch):
    def first(out):
        _write(out / "table.csv", "a", 5_000)

    def rolled_back(out):
        _write(out / "table.csv", "b", 4_000)

    events = _run(tmp_path, monkeypatch, [first, rolled_back], blocks_per_turn=2)

    assert [a["mtime_ns"] for a in _artifacts(events)] == [5_000, 4_000]


def test_subfolder_files_are_reported_with_relative_posix_path(tmp_path, monkeypatch):
    def produce(out):
        _write(out / "plots" / "qc" / "violin.png", "x", 1)
        _write(out / "summary.txt", "x", 1)

    events = _run(tmp_path, monkeypatch, [produce])

    arts = _artifacts(events)
    assert sorted((a["path"], a["filename"]) for a in arts) == [
        ("plots/qc/violin.png", "violin.png"),
        ("summary.txt", "summary.txt"),
    ]
    assert {a["type"] for a in arts} == {"plot", "report"}


def test_internal_locations_are_not_reported(tmp_path, monkeypatch):
    def produce(out):
        _write(out / LIVE_DATASET, "x", 1)
        _write(out / (LIVE_DATASET + ".tmp"), "x", 1)
        _write(out / ".cache" / "fontconfig" / "x.cache-9", "x", 1)
        _write(out / "work-items" / "index.json", "{}", 1)
        _write(out / "work-items" / ".git" / "HEAD", "ref", 1)
        _write(out / "real.png", "x", 1)

    events = _run(tmp_path, monkeypatch, [produce])

    assert [a["path"] for a in _artifacts(events)] == ["real.png"]


def test_known_artifacts_are_not_reemitted_after_resume(tmp_path, monkeypatch):
    output_dir = tmp_path / "outputs"
    _write(output_dir / "old.png", "x", 1_000)
    _write(output_dir / "changed.png", "x", 1_000)

    def produce(out):
        _write(out / "changed.png", "y", 2_000)
        _write(out / "new.csv", "z", 3_000)

    _stub_rag(monkeypatch)
    events = _run_auto(
        tmp_path,
        RecordingLLM(["```python\npass\n```"]),
        session_id="sess",
        sandbox_manager=WritingSandbox(output_dir, [produce]),
        known_artifacts={"old.png": 1_000, "changed.png": 1_000},
    )

    assert sorted((a["path"], a["mtime_ns"]) for a in _artifacts(events)) == [
        ("changed.png", 2_000),
        ("new.csv", 3_000),
    ]
