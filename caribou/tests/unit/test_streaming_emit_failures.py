"""Failures in the session's event_callback must reach the runner, not just the asyncio log."""

import asyncio
import threading
import time

import pytest

from caribou.server import streaming_runner


def _run_async(monkeypatch, fake_sync, callback):
    monkeypatch.setattr(streaming_runner, "run_session_sync", fake_sync)
    return asyncio.run(
        streaming_runner.run_session_async(
            session_id="s",
            agent_system=None,
            driver_agent=None,
            analysis_context="",
            llm_client=None,
            sandbox_manager=None,
            history=[],
            is_auto=True,
            max_turns=1,
            model_name="m",
            output_dir=None,
            event_callback=callback,
            stop_flag=threading.Event(),
        )
    )


def _failing_callback(event):
    if event["type"] == "bad":
        raise RuntimeError("code_result with no matching code_submitted")


def test_callback_failure_raises_on_next_emit(monkeypatch):
    seen = []

    def fake_sync(*, emit, **_):
        emit({"type": "bad"})
        # Give the loop time to run the failing delivery.
        time.sleep(0.2)
        try:
            emit({"type": "next"})
        except RuntimeError as exc:
            seen.append(exc)
            # The failure is reported once; the runner's error event goes through.
            emit({"type": "error"})
            return
        raise AssertionError("emit did not raise after a callback failure")

    _run_async(monkeypatch, fake_sync, _failing_callback)
    assert len(seen) == 1
    assert "no matching code_submitted" in str(seen[0])


def test_callback_failure_after_last_emit_raises_from_run_session_async(monkeypatch):
    def fake_sync(*, emit, **_):
        emit({"type": "bad"})

    with pytest.raises(RuntimeError, match="Session event handling failed"):
        _run_async(monkeypatch, fake_sync, _failing_callback)
