"""
Pytest configuration and shared fixtures for CARIBOU tests.
"""
import sys
from pathlib import Path

# Add the caribou/src directory to the Python path so imports work
caribou_src = Path(__file__).parent.parent / "src"
if str(caribou_src) not in sys.path:
    sys.path.insert(0, str(caribou_src))

import pytest
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
