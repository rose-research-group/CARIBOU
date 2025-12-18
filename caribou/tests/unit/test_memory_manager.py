"""
Unit tests for MemoryManager.

Tests conversation history management, episodic summarization,
and context assembly.
"""
import pytest
from unittest.mock import Mock, MagicMock
from types import SimpleNamespace

from caribou.execution.MemoryManager import MemoryManager


class TestMemoryManagerInitialization:
    """Test MemoryManager initialization."""

    def test_init_with_default_params(self, mock_llm_client):
        """Test initialization with default parameters."""
        client = mock_llm_client()
        initial_history = [
            {"role": "system", "content": "Global policy"},
            {"role": "system", "content": "Agent prompt"},
        ]

        manager = MemoryManager(
            llm_client=client,
            model_name="gpt-4",
            initial_history=initial_history
        )

        assert manager.config["working_history_size"] == 4
        assert manager.config["summarization_threshold"] == 20
        assert manager.config["chunk_size_to_summarize"] == 10
        assert len(manager._full_history) == 2
        assert len(manager._pinned_messages) == 2

    def test_init_with_custom_params(self, mock_llm_client):
        """Test initialization with custom parameters."""
        client = mock_llm_client()
        initial_history = [{"role": "system", "content": "System"}]

        manager = MemoryManager(
            llm_client=client,
            model_name="gpt-4",
            initial_history=initial_history,
            working_history_size=8,
            summarization_threshold=30,
            chunk_size_to_summarize=15
        )

        assert manager.config["working_history_size"] == 8
        assert manager.config["summarization_threshold"] == 30
        assert manager.config["chunk_size_to_summarize"] == 15

    def test_init_pins_early_messages(self, mock_llm_client):
        """Test that early messages are pinned."""
        client = mock_llm_client()
        initial_history = [
            {"role": "system", "content": "Message 1"},
            {"role": "system", "content": "Message 2"},
            {"role": "system", "content": "Message 3"},
            {"role": "user", "content": "Message 4"},
        ]

        manager = MemoryManager(
            llm_client=client,
            model_name="gpt-4",
            initial_history=initial_history
        )

        # Should pin first 3 messages
        assert len(manager._pinned_messages) == 3
        assert manager._pinned_messages[0]["content"] == "Message 1"
        assert manager._pinned_messages[1]["content"] == "Message 2"
        assert manager._pinned_messages[2]["content"] == "Message 3"

    def test_init_with_short_history(self, mock_llm_client):
        """Test initialization with history shorter than pin count."""
        client = mock_llm_client()
        initial_history = [{"role": "system", "content": "Only one"}]

        manager = MemoryManager(
            llm_client=client,
            model_name="gpt-4",
            initial_history=initial_history
        )

        # Should only pin what exists
        assert len(manager._pinned_messages) == 1


class TestMemoryManagerMessageOperations:
    """Test adding and updating messages."""

    def test_add_message(self, mock_llm_client):
        """Test adding a new message."""
        client = mock_llm_client()
        manager = MemoryManager(
            llm_client=client,
            model_name="gpt-4",
            initial_history=[{"role": "system", "content": "Init"}]
        )

        manager.add_message("user", "Hello")
        manager.add_message("assistant", "Hi there")

        assert len(manager._full_history) == 3
        assert manager._full_history[-2] == {"role": "user", "content": "Hello"}
        assert manager._full_history[-1] == {"role": "assistant", "content": "Hi there"}

    def test_add_pivotal_code(self, mock_llm_client):
        """Test adding pivotal code."""
        client = mock_llm_client()
        manager = MemoryManager(
            llm_client=client,
            model_name="gpt-4",
            initial_history=[{"role": "system", "content": "Init"}]
        )

        code = "def foo():\n    return 42"
        manager.add_pivotal_code(code)

        assert len(manager._pivotal_code) == 1
        pivotal_msg = manager._pivotal_code[0]
        assert pivotal_msg["role"] == "system"
        assert "PIVOTAL CODE" in pivotal_msg["content"]
        assert code in pivotal_msg["content"]
        assert "```python" in pivotal_msg["content"]

    def test_update_system_prompt(self, mock_llm_client):
        """Test updating system prompt."""
        client = mock_llm_client()
        initial_history = [
            {"role": "system", "content": "Global policy"},
            {"role": "system", "content": "Old agent prompt"},
        ]
        manager = MemoryManager(
            llm_client=client,
            model_name="gpt-4",
            initial_history=initial_history
        )

        new_prompt = "New agent prompt"
        manager.update_system_prompt(new_prompt)

        # Should update second pinned message
        assert manager._pinned_messages[1]["content"] == new_prompt
        # Should also update in full history
        assert manager._full_history[1]["content"] == new_prompt

    def test_update_system_prompt_with_short_history(self, mock_llm_client):
        """Test updating system prompt when history is too short."""
        client = mock_llm_client()
        initial_history = [{"role": "system", "content": "Only one"}]
        manager = MemoryManager(
            llm_client=client,
            model_name="gpt-4",
            initial_history=initial_history
        )

        new_prompt = "New prompt"
        manager.update_system_prompt(new_prompt)

        # Should insert at index 1
        assert len(manager._pinned_messages) == 2
        assert manager._pinned_messages[1]["content"] == new_prompt


class TestMemoryManagerContextAssembly:
    """Test context assembly without summarization."""

    def test_get_context_basic(self, mock_llm_client):
        """Test basic context retrieval."""
        client = mock_llm_client()
        initial_history = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Hello"},
        ]
        manager = MemoryManager(
            llm_client=client,
            model_name="gpt-4",
            initial_history=initial_history,
            working_history_size=4
        )

        context = manager.get_context()

        # Should return pinned + recent working history
        assert len(context) >= 2
        assert context[0]["content"] == "System"

    def test_get_context_with_working_history(self, mock_llm_client):
        """Test context includes working history."""
        client = mock_llm_client()
        manager = MemoryManager(
            llm_client=client,
            model_name="gpt-4",
            initial_history=[{"role": "system", "content": "System"}],
            working_history_size=3
        )

        # Add several messages
        for i in range(10):
            manager.add_message("user", f"Message {i}")

        context = manager.get_context()

        # Should include pinned + last 3 messages
        # Last 3 should be messages 7, 8, 9
        working_history_part = [msg for msg in context if "Message" in msg.get("content", "")]
        assert len(working_history_part) == 3
        assert "Message 8" in working_history_part[-2]["content"]
        assert "Message 9" in working_history_part[-1]["content"]

    def test_get_context_with_pivotal_code(self, mock_llm_client):
        """Test context includes pivotal code."""
        client = mock_llm_client()
        manager = MemoryManager(
            llm_client=client,
            model_name="gpt-4",
            initial_history=[{"role": "system", "content": "System"}]
        )

        manager.add_pivotal_code("x = 1")
        manager.add_pivotal_code("y = 2")

        context = manager.get_context()

        # Count system messages with PIVOTAL CODE
        pivotal_count = sum(1 for msg in context if "PIVOTAL CODE" in msg.get("content", ""))
        assert pivotal_count == 2


class TestMemoryManagerSummarization:
    """Test episodic summarization logic."""

    def test_should_summarize_false_initially(self, mock_llm_client):
        """Test that summarization is not triggered initially."""
        client = mock_llm_client()
        manager = MemoryManager(
            llm_client=client,
            model_name="gpt-4",
            initial_history=[{"role": "system", "content": "System"}],
            summarization_threshold=20,
            working_history_size=4
        )

        # Add a few messages (not enough to trigger)
        for i in range(10):
            manager.add_message("user", f"Message {i}")

        assert not manager._should_summarize()

    def test_should_summarize_true_when_threshold_exceeded(self, mock_llm_client):
        """Test that summarization is triggered when threshold exceeded."""
        client = mock_llm_client()
        manager = MemoryManager(
            llm_client=client,
            model_name="gpt-4",
            initial_history=[{"role": "system", "content": "System"}],
            summarization_threshold=5,
            working_history_size=4
        )

        # Add enough messages to exceed threshold
        # Pinned: 1, Working tail: 4, Threshold: 5
        # Need at least 1 (pinned) + 5 (threshold) + 4 (working) = 10 messages
        for i in range(15):
            manager.add_message("user", f"Message {i}")

        assert manager._should_summarize()

    def test_summarize_chunk_creates_summary(self, mock_llm_client):
        """Test that summarization creates a summary message."""
        client = mock_llm_client(responses=["This is a summary of the conversation."])
        manager = MemoryManager(
            llm_client=client,
            model_name="gpt-4",
            initial_history=[{"role": "system", "content": "System"}],
            chunk_size_to_summarize=5,
            summarization_threshold=5,
            working_history_size=2
        )

        # Add enough messages to allow summarization
        for i in range(12):
            manager.add_message("user", f"Message {i}")

        # Manually trigger summarization
        manager._summarize_chunk()

        # Should have created one summary
        assert len(manager._summarized_log) == 1
        assert "EPISODIC SUMMARY" in manager._summarized_log[0]["content"]
        assert "This is a summary" in manager._summarized_log[0]["content"]

    def test_summarize_chunk_calls_llm(self, mock_llm_client):
        """Test that summarization calls the LLM."""
        client = mock_llm_client(responses=["Summary"])
        manager = MemoryManager(
            llm_client=client,
            model_name="test-model",
            initial_history=[{"role": "system", "content": "System"}],
            chunk_size_to_summarize=3,
            working_history_size=2
        )

        for i in range(10):
            manager.add_message("user", f"Message {i}")

        manager._summarize_chunk()

        # Check that LLM was called
        assert client.call_count >= 1
        call_kwargs = client.calls[0]
        assert call_kwargs["model"] == "test-model"
        assert call_kwargs["temperature"] == 0.0

    def test_get_context_triggers_summarization(self, mock_llm_client):
        """Test that get_context triggers summarization when needed."""
        client = mock_llm_client(responses=["Summary"])
        manager = MemoryManager(
            llm_client=client,
            model_name="gpt-4",
            initial_history=[{"role": "system", "content": "System"}],
            summarization_threshold=5,
            chunk_size_to_summarize=5,
            working_history_size=2
        )

        # Add enough messages to trigger summarization
        for i in range(15):
            manager.add_message("user", f"Message {i}")

        context = manager.get_context()

        # Should have triggered summarization
        assert len(manager._summarized_log) >= 1

        # Summary should be in context
        summary_in_context = any("EPISODIC SUMMARY" in msg.get("content", "") for msg in context)
        assert summary_in_context

    def test_summarization_preserves_working_history(self, mock_llm_client):
        """Test that working history is not summarized."""
        client = mock_llm_client(responses=["Summary"])
        manager = MemoryManager(
            llm_client=client,
            model_name="gpt-4",
            initial_history=[{"role": "system", "content": "System"}],
            chunk_size_to_summarize=5,
            working_history_size=3
        )

        for i in range(15):
            manager.add_message("user", f"Message {i}")

        # Force summarization
        while manager._should_summarize():
            manager._summarize_chunk()

        # Full history should still contain all messages
        assert len(manager._full_history) == 16  # 1 initial + 15 added

        # Last 3 messages should be preserved in context
        context = manager.get_context()
        working_msgs = [msg for msg in context if "Message" in msg.get("content", "")]
        assert len(working_msgs) >= 3
        assert "Message 14" in working_msgs[-1]["content"]

    def test_multiple_summarization_rounds(self, mock_llm_client):
        """Test multiple rounds of summarization."""
        client = mock_llm_client(responses=["Summary 1", "Summary 2", "Summary 3"])
        manager = MemoryManager(
            llm_client=client,
            model_name="gpt-4",
            initial_history=[{"role": "system", "content": "System"}],
            summarization_threshold=3,
            chunk_size_to_summarize=5,
            working_history_size=2
        )

        # Add many messages to trigger multiple summarizations
        for i in range(30):
            manager.add_message("user", f"Message {i}")
            # Trigger summarization periodically
            if manager._should_summarize():
                manager._summarize_chunk()

        # Should have multiple summaries
        assert len(manager._summarized_log) >= 2


class TestMemoryManagerErrorHandling:
    """Test error handling in summarization."""

    def test_summarization_error_does_not_crash(self, mock_llm_client, capsys):
        """Test that summarization errors are handled gracefully."""
        client = mock_llm_client()
        # Make the LLM call raise an error
        client.chat.completions.create = Mock(side_effect=Exception("API Error"))

        manager = MemoryManager(
            llm_client=client,
            model_name="gpt-4",
            initial_history=[{"role": "system", "content": "System"}],
            chunk_size_to_summarize=3,
            working_history_size=2
        )

        for i in range(10):
            manager.add_message("user", f"Message {i}")

        # Should not raise exception
        try:
            manager._summarize_chunk()
        except Exception:
            pytest.fail("Summarization error should be caught")

        # Should print warning
        captured = capsys.readouterr()
        assert "Warning" in captured.out or "Could not summarize" in captured.out


class TestMemoryManagerContextLayout:
    """Test the context layout structure."""

    def test_context_layout_order(self, mock_llm_client):
        """Test that context has correct order: pinned + pivotal + summaries + working."""
        client = mock_llm_client(responses=["Summary"])
        initial_history = [
            {"role": "system", "content": "Global"},
            {"role": "system", "content": "Agent"},
        ]
        manager = MemoryManager(
            llm_client=client,
            model_name="gpt-4",
            initial_history=initial_history,
            chunk_size_to_summarize=3,
            summarization_threshold=3,
            working_history_size=2
        )

        # Add pivotal code
        manager.add_pivotal_code("important_code()")

        # Add messages to trigger summarization
        for i in range(10):
            manager.add_message("user", f"Message {i}")

        context = manager.get_context()

        # Find indices of different sections
        pinned_idx = None
        pivotal_idx = None
        summary_idx = None
        working_idx = None

        for i, msg in enumerate(context):
            content = msg.get("content", "")
            if "Global" in content:
                pinned_idx = i
            elif "PIVOTAL CODE" in content:
                pivotal_idx = i
            elif "EPISODIC SUMMARY" in content:
                summary_idx = i
            elif "Message" in content:
                if working_idx is None:
                    working_idx = i

        # Verify order (indices should increase)
        assert pinned_idx is not None
        if pivotal_idx is not None:
            assert pivotal_idx > pinned_idx
        if summary_idx is not None:
            assert summary_idx > pivotal_idx if pivotal_idx else summary_idx > pinned_idx

    def test_empty_sections_not_included(self, mock_llm_client):
        """Test that empty sections are not included in context."""
        client = mock_llm_client()
        manager = MemoryManager(
            llm_client=client,
            model_name="gpt-4",
            initial_history=[{"role": "system", "content": "System"}],
            working_history_size=2
        )

        manager.add_message("user", "Hello")

        context = manager.get_context()

        # Should not have pivotal code or summaries
        pivotal_count = sum(1 for msg in context if "PIVOTAL CODE" in msg.get("content", ""))
        summary_count = sum(1 for msg in context if "EPISODIC SUMMARY" in msg.get("content", ""))

        assert pivotal_count == 0
        assert summary_count == 0


class TestMemoryManagerEdgeCases:
    """Test edge cases."""

    def test_zero_working_history_size(self, mock_llm_client):
        """Test with zero working history size."""
        client = mock_llm_client()
        manager = MemoryManager(
            llm_client=client,
            model_name="gpt-4",
            initial_history=[{"role": "system", "content": "System"}],
            working_history_size=0
        )

        manager.add_message("user", "Test")
        context = manager.get_context()

        # Should still work, just no working history included
        assert len(context) >= 1

    def test_very_large_working_history(self, mock_llm_client):
        """Test with working history larger than actual history."""
        client = mock_llm_client()
        manager = MemoryManager(
            llm_client=client,
            model_name="gpt-4",
            initial_history=[{"role": "system", "content": "System"}],
            working_history_size=1000
        )

        manager.add_message("user", "Test")
        context = manager.get_context()

        # Should include all messages without error
        assert len(context) >= 2
