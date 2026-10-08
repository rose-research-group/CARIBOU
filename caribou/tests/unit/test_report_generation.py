from __future__ import annotations

import io
from types import SimpleNamespace

import pytest
from rich.console import Console

from caribou.execution.report_generation import _generate_agent_report


class ReportLlm:
    def __init__(self, outcome):
        self.outcome = outcome
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **_kwargs):
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        message = SimpleNamespace(content=self.outcome)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _generate(llm, history_slice=None):
    return _generate_agent_report(
        Console(file=io.StringIO()),
        llm_client=llm,
        model_name="fake",
        agent_name="planner",
        history_slice=history_slice or [{"role": "assistant", "content": "did work"}],
    )


def test_returns_the_report_text():
    assert _generate(ReportLlm("planner made umap.png")) == "planner made umap.png"


def test_llm_errors_propagate():
    with pytest.raises(ConnectionError, match="provider down"):
        _generate(ReportLlm(ConnectionError("provider down")))


@pytest.mark.parametrize("content", [None, ""])
def test_empty_report_is_an_error(content):
    with pytest.raises(RuntimeError, match="'planner' returned no content"):
        _generate(ReportLlm(content))
