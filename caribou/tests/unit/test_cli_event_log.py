from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console

import caribou.execution.path_utils as path_utils
import caribou.execution.runner as runner_module
from caribou.cli import run_cli
from caribou.cli.run_cli import (
    CLI_EVENT_LOG_FILENAME,
    AppContext,
    _cli_event_log_callback,
    _setup_and_run_session,
)


def _event(run_id: str, event_type: str, payload: dict[str, object]) -> dict:
    return {
        "schema_version": "caribou.runner_event.v1",
        "event_type": event_type,
        "occurred_at": "2026-07-15T12:00:00Z",
        "run_id": run_id,
        "turn": 1,
        "agent_name": "analyst",
        "payload": payload,
    }


def test_event_log_appends_one_flushed_json_line_per_event(tmp_path: Path) -> None:
    record = _cli_event_log_callback(tmp_path)
    first = _event("run_a", "turn_started", {"model_name": "m"})
    second = _event("run_a", "assistant_message", {"role": "assistant", "content": "é"})

    record(first)
    log_path = tmp_path / CLI_EVENT_LOG_FILENAME
    # Readable immediately: nothing is buffered until session end.
    assert [json.loads(line) for line in log_path.read_text().splitlines()] == [first]

    record(second)
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == [first, second]


def test_event_log_appends_to_an_existing_log(tmp_path: Path) -> None:
    first = _event("run_a", "turn_started", {"model_name": "m"})
    second = _event("run_b", "turn_started", {"model_name": "m"})
    _cli_event_log_callback(tmp_path)(first)
    _cli_event_log_callback(tmp_path)(second)

    lines = (tmp_path / CLI_EVENT_LOG_FILENAME).read_text().splitlines()
    assert [json.loads(line) for line in lines] == [first, second]


def test_event_log_without_output_dir_uses_the_runner_session_notes_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(path_utils, "_DEFAULT_RUNS_DIR", tmp_path / "runs")
    record = _cli_event_log_callback(None)
    event = _event("run_x", "turn_started", {"model_name": "m"})

    record(event)

    log_path = tmp_path / "runs" / "session_notes" / "run_x" / CLI_EVENT_LOG_FILENAME
    assert json.loads(log_path.read_text()) == event


def test_event_log_rejects_a_run_id_change(tmp_path: Path) -> None:
    record = _cli_event_log_callback(tmp_path)
    record(_event("run_a", "turn_started", {"model_name": "m"}))
    with pytest.raises(RuntimeError, match="does not match"):
        record(_event("run_b", "turn_started", {"model_name": "m"}))


def test_event_log_raises_on_a_non_serializable_payload(tmp_path: Path) -> None:
    record = _cli_event_log_callback(tmp_path)
    with pytest.raises(TypeError):
        record(_event("run_a", "turn_started", {"path": Path("x")}))
    assert not (tmp_path / CLI_EVENT_LOG_FILENAME).exists()


class _ExecBackend:
    """Mimics the Singularity backend: outputs are listed from the host dir."""

    def __init__(self) -> None:
        self.output_dir: Path | None = None

    def set_data(self, _: list, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir

    def start_container(self) -> bool:
        return True

    def stop_container(self) -> None:
        return None

    def list_output_files(self) -> list:
        assert self.output_dir is not None
        return [
            {"name": f.name, "size": "0.00 MB"}
            for f in self.output_dir.iterdir()
            if f.is_file()
        ]


def test_cli_session_wires_the_event_log_into_the_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    emitted = [
        _event("run_cli", "turn_started", {"model_name": "m"}),
        _event("run_cli", "session_end", {"end_reason": "completed"}),
    ]

    def fake_run_agent_session(**kwargs: object) -> None:
        callback = kwargs["event_callback"]
        assert callable(callback)
        for event in emitted:
            callback(event)

    monkeypatch.setattr(runner_module, "run_agent_session", fake_run_agent_session)
    monkeypatch.setattr(
        "caribou.core.io_helpers.save_chat_history_as_notebook",
        lambda *args: None,
    )

    output_dir = tmp_path / "outputs"
    context = AppContext()
    terminal = io.StringIO()
    context.console = Console(file=terminal, width=400)
    context.output_dir = output_dir
    context.sandbox_manager = _ExecBackend()
    context.sandbox_details = {"is_exec_mode": True}
    context.dataset_path = tmp_path / "data.h5ad"
    context.agent_system = SimpleNamespace(get_agent=lambda name: name)
    context.driver_agent_name = "analyst"
    context.analysis_context = "ctx"
    context.model_name = "m"

    _setup_and_run_session(context, history=[], is_auto=True, max_turns=1)

    lines = (output_dir / CLI_EVENT_LOG_FILENAME).read_text().splitlines()
    assert [json.loads(line) for line in lines] == emitted
    assert run_cli.CLI_EVENT_LOG_FILENAME == "events.jsonl"
    # The event log is not an agent output: a session that wrote nothing still
    # reports that no output files were generated.
    assert "No output files were generated." in terminal.getvalue()
    assert "Session outputs saved in" not in terminal.getvalue()
