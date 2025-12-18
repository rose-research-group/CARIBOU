"""
Unit tests for AnthropicClient wrapper.

Tests the OpenAI API compatibility layer and message conversion.
"""
import pytest
from types import SimpleNamespace
from unittest.mock import Mock, patch, MagicMock

from caribou.core.anthropic_wrapper import AnthropicClient


class TestAnthropicClientInitialization:
    """Test AnthropicClient initialization."""

    def test_init_with_minimal_params(self):
        """Test initialization with only required parameters."""
        with patch("caribou.core.anthropic_wrapper.anthropic.Anthropic") as mock_anthropic:
            client = AnthropicClient(api_key="test-key")

            assert client._default_model == "claude-sonnet-4-5-20250929"
            assert client._max_output_tokens == 1024
            mock_anthropic.assert_called_once_with(api_key="test-key")

    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        with patch("caribou.core.anthropic_wrapper.anthropic.Anthropic") as mock_anthropic:
            client = AnthropicClient(
                api_key="test-key",
                model="claude-opus-4",
                max_output_tokens=2048,
                base_url="https://custom.api.com"
            )

            assert client._default_model == "claude-opus-4"
            assert client._max_output_tokens == 2048
            mock_anthropic.assert_called_once_with(
                api_key="test-key",
                base_url="https://custom.api.com"
            )

    def test_chat_completions_interface_exists(self):
        """Test that the OpenAI-compatible interface exists."""
        with patch("caribou.core.anthropic_wrapper.anthropic.Anthropic"):
            client = AnthropicClient(api_key="test-key")

            assert hasattr(client, "chat")
            assert hasattr(client.chat, "completions")
            assert hasattr(client.chat.completions, "create")
            assert callable(client.chat.completions.create)


class TestAnthropicClientMessageConversion:
    """Test message format conversion from OpenAI to Anthropic."""

    def test_system_message_extraction(self):
        """Test that system messages are extracted and combined."""
        with patch("caribou.core.anthropic_wrapper.anthropic.Anthropic") as mock_anthropic:
            mock_instance = Mock()
            mock_response = Mock(
                content=[SimpleNamespace(type="text", text="Response")],
                stop_reason="end_turn"
            )
            mock_instance.messages.create.return_value = mock_response
            mock_anthropic.return_value = mock_instance

            client = AnthropicClient(api_key="test-key")

            messages = [
                {"role": "system", "content": "Global policy"},
                {"role": "system", "content": "Agent prompt"},
                {"role": "user", "content": "Hello"},
            ]

            client.chat.completions.create(messages=messages)

            call_args = mock_instance.messages.create.call_args
            assert call_args[1]["system"] == "Global policy\n\nAgent prompt"
            assert len(call_args[1]["messages"]) == 1
            assert call_args[1]["messages"][0] == {"role": "user", "content": "Hello"}

    def test_system_message_absent(self):
        """Test handling when no system messages present."""
        with patch("caribou.core.anthropic_wrapper.anthropic.Anthropic") as mock_anthropic:
            mock_instance = Mock()
            mock_response = Mock(
                content=[SimpleNamespace(type="text", text="Response")],
                stop_reason="end_turn"
            )
            mock_instance.messages.create.return_value = mock_response
            mock_anthropic.return_value = mock_instance

            client = AnthropicClient(api_key="test-key")

            messages = [
                {"role": "user", "content": "Hello"},
            ]

            client.chat.completions.create(messages=messages)

            call_args = mock_instance.messages.create.call_args
            assert call_args[1]["system"] is None

    def test_role_filtering(self):
        """Test that only assistant and user roles are preserved."""
        with patch("caribou.core.anthropic_wrapper.anthropic.Anthropic") as mock_anthropic:
            mock_instance = Mock()
            mock_response = Mock(
                content=[SimpleNamespace(type="text", text="Response")],
                stop_reason="end_turn"
            )
            mock_instance.messages.create.return_value = mock_response
            mock_anthropic.return_value = mock_instance

            client = AnthropicClient(api_key="test-key")

            messages = [
                {"role": "user", "content": "Message 1"},
                {"role": "assistant", "content": "Message 2"},
                {"role": "function", "content": "Message 3"},  # Should be converted to user
                {"role": "tool", "content": "Message 4"},  # Should be converted to user
            ]

            client.chat.completions.create(messages=messages)

            call_args = mock_instance.messages.create.call_args
            converted_messages = call_args[1]["messages"]

            assert len(converted_messages) == 4
            assert converted_messages[0]["role"] == "user"
            assert converted_messages[1]["role"] == "assistant"
            assert converted_messages[2]["role"] == "user"  # function -> user
            assert converted_messages[3]["role"] == "user"  # tool -> user


class TestAnthropicClientAPICall:
    """Test actual API call behavior."""

    def test_api_call_with_default_params(self):
        """Test API call with default parameters."""
        with patch("caribou.core.anthropic_wrapper.anthropic.Anthropic") as mock_anthropic:
            mock_instance = Mock()
            mock_response = Mock(
                content=[SimpleNamespace(type="text", text="Test response")],
                stop_reason="end_turn"
            )
            mock_instance.messages.create.return_value = mock_response
            mock_anthropic.return_value = mock_instance

            client = AnthropicClient(api_key="test-key")

            messages = [{"role": "user", "content": "Hello"}]
            client.chat.completions.create(messages=messages)

            mock_instance.messages.create.assert_called_once()
            call_args = mock_instance.messages.create.call_args[1]

            assert call_args["model"] == "claude-sonnet-4-5-20250929"
            assert call_args["max_tokens"] == 1024
            assert call_args["temperature"] is None

    def test_api_call_with_override_params(self):
        """Test API call with parameter overrides."""
        with patch("caribou.core.anthropic_wrapper.anthropic.Anthropic") as mock_anthropic:
            mock_instance = Mock()
            mock_response = Mock(
                content=[SimpleNamespace(type="text", text="Test response")],
                stop_reason="end_turn"
            )
            mock_instance.messages.create.return_value = mock_response
            mock_anthropic.return_value = mock_instance

            client = AnthropicClient(api_key="test-key", model="default-model")

            messages = [{"role": "user", "content": "Hello"}]
            client.chat.completions.create(
                messages=messages,
                model="override-model",
                temperature=0.7,
                max_output_tokens=512
            )

            call_args = mock_instance.messages.create.call_args[1]

            assert call_args["model"] == "override-model"
            assert call_args["max_tokens"] == 512
            assert call_args["temperature"] == 0.7

    def test_extra_kwargs_ignored(self):
        """Test that extra unknown kwargs are ignored."""
        with patch("caribou.core.anthropic_wrapper.anthropic.Anthropic") as mock_anthropic:
            mock_instance = Mock()
            mock_response = Mock(
                content=[SimpleNamespace(type="text", text="Test response")],
                stop_reason="end_turn"
            )
            mock_instance.messages.create.return_value = mock_response
            mock_anthropic.return_value = mock_instance

            client = AnthropicClient(api_key="test-key")

            messages = [{"role": "user", "content": "Hello"}]
            # Should not raise error for unknown kwargs
            client.chat.completions.create(
                messages=messages,
                unknown_param="should_be_ignored",
                another_unknown=123
            )

            mock_instance.messages.create.assert_called_once()


class TestAnthropicClientResponseFormatting:
    """Test response formatting to OpenAI-compatible structure."""

    def test_response_structure(self):
        """Test that response has correct OpenAI-compatible structure."""
        with patch("caribou.core.anthropic_wrapper.anthropic.Anthropic") as mock_anthropic:
            mock_instance = Mock()
            mock_response = Mock(
                content=[SimpleNamespace(type="text", text="Test response")],
                stop_reason="end_turn"
            )
            mock_instance.messages.create.return_value = mock_response
            mock_anthropic.return_value = mock_instance

            client = AnthropicClient(api_key="test-key")

            messages = [{"role": "user", "content": "Hello"}]
            response = client.chat.completions.create(messages=messages)

            # Check OpenAI-compatible structure
            assert hasattr(response, "choices")
            assert len(response.choices) == 1
            assert hasattr(response.choices[0], "message")
            assert hasattr(response.choices[0].message, "content")
            assert hasattr(response.choices[0].message, "role")
            assert response.choices[0].message.content == "Test response"
            assert response.choices[0].message.role == "assistant"

    def test_multiple_text_blocks(self):
        """Test combining multiple text blocks from Anthropic response."""
        with patch("caribou.core.anthropic_wrapper.anthropic.Anthropic") as mock_anthropic:
            mock_instance = Mock()
            mock_response = Mock(
                content=[
                    SimpleNamespace(type="text", text="Part 1 "),
                    SimpleNamespace(type="text", text="Part 2 "),
                    SimpleNamespace(type="text", text="Part 3"),
                ],
                stop_reason="end_turn"
            )
            mock_instance.messages.create.return_value = mock_response
            mock_anthropic.return_value = mock_instance

            client = AnthropicClient(api_key="test-key")

            messages = [{"role": "user", "content": "Hello"}]
            response = client.chat.completions.create(messages=messages)

            assert response.choices[0].message.content == "Part 1 Part 2 Part 3"

    def test_non_text_blocks_ignored(self):
        """Test that non-text blocks are filtered out."""
        with patch("caribou.core.anthropic_wrapper.anthropic.Anthropic") as mock_anthropic:
            mock_instance = Mock()
            mock_response = Mock(
                content=[
                    SimpleNamespace(type="text", text="Text content"),
                    SimpleNamespace(type="tool_use", id="tool_123"),  # Should be ignored
                    SimpleNamespace(type="image", source="data:image"),  # Should be ignored
                ],
                stop_reason="end_turn"
            )
            mock_instance.messages.create.return_value = mock_response
            mock_anthropic.return_value = mock_instance

            client = AnthropicClient(api_key="test-key")

            messages = [{"role": "user", "content": "Hello"}]
            response = client.chat.completions.create(messages=messages)

            assert response.choices[0].message.content == "Text content"

    def test_finish_reason_mapping(self):
        """Test that stop_reason is mapped to finish_reason."""
        with patch("caribou.core.anthropic_wrapper.anthropic.Anthropic") as mock_anthropic:
            mock_instance = Mock()
            mock_response = Mock(
                content=[SimpleNamespace(type="text", text="Response")],
                stop_reason="max_tokens"
            )
            mock_instance.messages.create.return_value = mock_response
            mock_anthropic.return_value = mock_instance

            client = AnthropicClient(api_key="test-key")

            messages = [{"role": "user", "content": "Hello"}]
            response = client.chat.completions.create(messages=messages)

            assert response.choices[0].finish_reason == "max_tokens"


class TestAnthropicClientEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_messages_list(self):
        """Test handling of empty messages list."""
        with patch("caribou.core.anthropic_wrapper.anthropic.Anthropic") as mock_anthropic:
            mock_instance = Mock()
            mock_response = Mock(
                content=[SimpleNamespace(type="text", text="Response")],
                stop_reason="end_turn"
            )
            mock_instance.messages.create.return_value = mock_response
            mock_anthropic.return_value = mock_instance

            client = AnthropicClient(api_key="test-key")

            # Should not raise error, just pass empty list
            client.chat.completions.create(messages=[])

            call_args = mock_instance.messages.create.call_args[1]
            assert call_args["messages"] == []

    def test_missing_content_key(self):
        """Test handling of messages missing content key."""
        with patch("caribou.core.anthropic_wrapper.anthropic.Anthropic") as mock_anthropic:
            mock_instance = Mock()
            mock_response = Mock(
                content=[SimpleNamespace(type="text", text="Response")],
                stop_reason="end_turn"
            )
            mock_instance.messages.create.return_value = mock_response
            mock_anthropic.return_value = mock_instance

            client = AnthropicClient(api_key="test-key")

            messages = [
                {"role": "user"},  # Missing content
                {"role": "assistant", "content": "Hello"}
            ]

            client.chat.completions.create(messages=messages)

            call_args = mock_instance.messages.create.call_args[1]
            assert call_args["messages"][0]["content"] == ""

    def test_empty_response_content(self):
        """Test handling of empty response from Anthropic."""
        with patch("caribou.core.anthropic_wrapper.anthropic.Anthropic") as mock_anthropic:
            mock_instance = Mock()
            mock_response = Mock(
                content=[],  # Empty content
                stop_reason="end_turn"
            )
            mock_instance.messages.create.return_value = mock_response
            mock_anthropic.return_value = mock_instance

            client = AnthropicClient(api_key="test-key")

            messages = [{"role": "user", "content": "Hello"}]
            response = client.chat.completions.create(messages=messages)

            assert response.choices[0].message.content == ""
