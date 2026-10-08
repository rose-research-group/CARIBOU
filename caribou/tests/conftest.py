"""
Pytest configuration and shared fixtures for CARIBOU tests.
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

# Add the caribou/src directory to the Python path so imports work
caribou_src = Path(__file__).parent.parent / "src"
if str(caribou_src) not in sys.path:
    sys.path.insert(0, str(caribou_src))

# Isolate the suite from the user's real CARIBOU home (and its .env) before
# any caribou module is imported: CARIBOU_HOME and ENV_FILE are module-level
# constants computed at import time in caribou.config.
_ISOLATED_CARIBOU_HOME = Path(tempfile.mkdtemp(prefix="caribou-test-home-"))
os.environ["CARIBOU_HOME"] = str(_ISOLATED_CARIBOU_HOME)

import pytest

import caribou.config as caribou_config

if caribou_config.CARIBOU_HOME != _ISOLATED_CARIBOU_HOME:
    raise RuntimeError(
        "caribou.config was imported before tests/conftest.py could isolate "
        f"CARIBOU_HOME (resolved to {caribou_config.CARIBOU_HOME})"
    )

TEST_SLURM_PARTITION = "peerd"


def pytest_unconfigure(config):
    shutil.rmtree(_ISOLATED_CARIBOU_HOME)


@pytest.fixture(autouse=True)
def isolated_caribou_settings(tmp_path_factory, monkeypatch):
    """Pin CARIBOU settings so no test reads the user's real configuration.

    The Slurm partition is set explicitly (tests that need it unset or
    different override it), and caribou.config.ENV_FILE points at a per-test
    path that does not exist so no real .env value can leak in.
    """
    monkeypatch.setenv("CARIBOU_SLURM_PARTITION", TEST_SLURM_PARTITION)
    env_dir = tmp_path_factory.mktemp("caribou-env")
    monkeypatch.setattr(caribou_config, "ENV_FILE", env_dir / ".env")
from types import SimpleNamespace
from typing import List, Dict, Any


@pytest.fixture
def mock_anthropic_response():
    """Create a mock Anthropic API response."""
    def _create_response(content: str, stop_reason: str = "end_turn"):
        text_block = SimpleNamespace(type="text", text=content)
        return SimpleNamespace(
            content=[text_block],
            stop_reason=stop_reason,
            id="msg_123",
            model="claude-sonnet-4-5-20250929",
            role="assistant",
            type="message",
        )
    return _create_response


@pytest.fixture
def mock_openai_response():
    """Create a mock OpenAI API response."""
    def _create_response(content: str, finish_reason: str = "stop"):
        message = SimpleNamespace(content=content, role="assistant")
        choice = SimpleNamespace(message=message, index=0, finish_reason=finish_reason)
        return SimpleNamespace(
            choices=[choice],
            id="chatcmpl-123",
            model="gpt-4",
            object="chat.completion",
        )
    return _create_response


@pytest.fixture
def sample_messages():
    """Sample message history for testing."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you!"},
        {"role": "user", "content": "Can you help me with a task?"},
    ]


@pytest.fixture
def sample_agent_system():
    """Sample agent system configuration."""
    return {
        "global_policy": "Always be helpful and accurate.",
        "agents": {
            "planner": {
                "prompt": "You are a planning agent.",
                "neighbors": {
                    "delegate_to_coder": {
                        "target_agent": "coder",
                        "description": "Delegate coding tasks"
                    }
                },
                "code_samples": [],
                "rag": {"enabled": False}
            },
            "coder": {
                "prompt": "You are a coding agent.",
                "neighbors": {
                    "delegate_to_planner": {
                        "target_agent": "planner",
                        "description": "Go back to planning"
                    }
                },
                "code_samples": [],
                "rag": {"enabled": False}
            }
        }
    }


class MockLLMClient:
    """Mock LLM client for testing."""

    def __init__(self, responses: List[str] = None):
        self.responses = responses or ["Mock response"]
        self.call_count = 0
        self.calls = []

        # Mock the nested structure: client.chat.completions.create()
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    def _create(self, **kwargs):
        """Mock the chat.completions.create method."""
        self.calls.append(kwargs)

        response_text = self.responses[min(self.call_count, len(self.responses) - 1)]
        self.call_count += 1

        message = SimpleNamespace(content=response_text, role="assistant")
        choice = SimpleNamespace(message=message, index=0, finish_reason="stop")
        return SimpleNamespace(choices=[choice])


@pytest.fixture
def mock_llm_client():
    """Fixture for mock LLM client."""
    return lambda responses=None: MockLLMClient(responses)
