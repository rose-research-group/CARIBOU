"""
Unit tests for OllamaClient wrapper.

Tests the OpenAI API compatibility layer and ND-JSON response parsing.
"""
import pytest
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch, MagicMock

from caribou.core.ollama_wrapper import OllamaClient


class TestOllamaClientInitialization:
    """Test OllamaClient initialization."""

    def test_init_with_default_params(self):
        """Test initialization with default parameters."""
        client = OllamaClient()

        assert client._host == "http://localhost:11434"
        assert client._default_model == "deepseek-r1:70b"

    def test_init_with_custom_host(self):
        """Test initialization with custom host."""
        client = OllamaClient(host="http://ollama.local:8080", model="llama3")

        assert client._host == "http://ollama.local:8080"
        assert client._default_model == "llama3"

    def test_init_adds_http_prefix(self):
        """Test that http:// is added if missing."""
        client = OllamaClient(host="localhost:11434")

        assert client._host == "http://localhost:11434"

    def test_init_strips_trailing_slash(self):
        """Test that trailing slashes are removed."""
        client = OllamaClient(host="http://localhost:11434/")

        assert client._host == "http://localhost:11434"

    def test_chat_completions_interface_exists(self):
        """Test that the OpenAI-compatible interface exists."""
        client = OllamaClient()

        assert hasattr(client, "chat")
        assert hasattr(client.chat, "completions")
        assert hasattr(client.chat.completions, "create")
        assert callable(client.chat.completions.create)


class TestOllamaClientAPICall:
    """Test Ollama API call behavior."""

    @patch("caribou.core.ollama_wrapper.requests.post")
    def test_basic_api_call(self, mock_post):
        """Test a basic API call with default parameters."""
        # Mock response with ND-JSON format
        mock_response = Mock()
        mock_response.text = '{"message": {"role": "assistant", "content": "Hello!"}}\n'
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        client = OllamaClient(host="http://localhost:11434", model="llama3")
        messages = [{"role": "user", "content": "Hi"}]

        response = client.chat.completions.create(messages=messages)

        # Verify API call
        mock_post.assert_called_once()
        call_args = mock_post.call_args

        assert call_args[0][0] == "http://localhost:11434/api/chat"
        assert call_args[1]["json"]["model"] == "llama3"
        assert call_args[1]["json"]["messages"] == messages
        assert call_args[1]["json"]["stream"] is False
        assert call_args[1]["timeout"] == 300

        # Verify response structure
        assert hasattr(response, "choices")
        assert len(response.choices) == 1
        assert response.choices[0].message.content == "Hello!"
        assert response.choices[0].message.role == "assistant"

    @patch("caribou.core.ollama_wrapper.requests.post")
    def test_api_call_with_temperature(self, mock_post):
        """Test API call with temperature parameter."""
        mock_response = Mock()
        mock_response.text = '{"message": {"role": "assistant", "content": "Response"}}\n'
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        client = OllamaClient()
        messages = [{"role": "user", "content": "Test"}]

        client.chat.completions.create(messages=messages, temperature=0.7)

        call_args = mock_post.call_args
        payload = call_args[1]["json"]

        assert "options" in payload
        assert payload["options"]["temperature"] == 0.7

    @patch("caribou.core.ollama_wrapper.requests.post")
    def test_api_call_without_temperature(self, mock_post):
        """Test API call without temperature (should not include options)."""
        mock_response = Mock()
        mock_response.text = '{"message": {"role": "assistant", "content": "Response"}}\n'
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        client = OllamaClient()
        messages = [{"role": "user", "content": "Test"}]

        client.chat.completions.create(messages=messages)

        call_args = mock_post.call_args
        payload = call_args[1]["json"]

        assert "options" not in payload

    @patch("caribou.core.ollama_wrapper.requests.post")
    def test_extra_kwargs_ignored(self, mock_post):
        """Test that extra unknown kwargs don't break the call."""
        mock_response = Mock()
        mock_response.text = '{"message": {"role": "assistant", "content": "Response"}}\n'
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        client = OllamaClient()
        messages = [{"role": "user", "content": "Test"}]

        # Should not raise error for unknown kwargs
        client.chat.completions.create(
            messages=messages,
            unknown_param="ignored",
            another_param=123
        )

        mock_post.assert_called_once()


class TestOllamaClientResponseParsing:
    """Test ND-JSON response parsing."""

    @patch("caribou.core.ollama_wrapper.requests.post")
    def test_parse_single_line_response(self, mock_post):
        """Test parsing single-line ND-JSON response."""
        mock_response = Mock()
        mock_response.text = '{"message": {"role": "assistant", "content": "Single line"}}\n'
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        client = OllamaClient()
        response = client.chat.completions.create(messages=[{"role": "user", "content": "Test"}])

        assert response.choices[0].message.content == "Single line"

    @patch("caribou.core.ollama_wrapper.requests.post")
    def test_parse_multiline_ndjson_response(self, mock_post):
        """Test parsing multi-line ND-JSON response (takes first message)."""
        mock_response = Mock()
        mock_response.text = (
            '{"other": "data"}\n'
            '{"message": {"role": "assistant", "content": "First message"}}\n'
            '{"message": {"role": "assistant", "content": "Second message"}}\n'
        )
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        client = OllamaClient()
        response = client.chat.completions.create(messages=[{"role": "user", "content": "Test"}])

        # Should take the first message found
        assert response.choices[0].message.content == "First message"

    @patch("caribou.core.ollama_wrapper.requests.post")
    def test_parse_response_with_metadata(self, mock_post):
        """Test parsing response that includes metadata lines."""
        mock_response = Mock()
        mock_response.text = (
            '{"model": "llama3", "created_at": "2024-01-01T00:00:00Z"}\n'
            '{"message": {"role": "assistant", "content": "Hello!"}}\n'
            '{"done": true}\n'
        )
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        client = OllamaClient()
        response = client.chat.completions.create(messages=[{"role": "user", "content": "Test"}])

        assert response.choices[0].message.content == "Hello!"

    @patch("caribou.core.ollama_wrapper.requests.post")
    def test_response_structure_matches_openai(self, mock_post):
        """Test that response structure matches OpenAI format."""
        mock_response = Mock()
        mock_response.text = '{"message": {"role": "assistant", "content": "Test"}}\n'
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        client = OllamaClient()
        response = client.chat.completions.create(messages=[{"role": "user", "content": "Test"}])

        # Check OpenAI-compatible structure
        assert hasattr(response, "choices")
        assert len(response.choices) == 1
        assert hasattr(response.choices[0], "message")
        assert hasattr(response.choices[0], "index")
        assert hasattr(response.choices[0], "finish_reason")
        assert hasattr(response.choices[0].message, "content")
        assert hasattr(response.choices[0].message, "role")

        assert response.choices[0].index == 0
        assert response.choices[0].finish_reason == "stop"
        assert response.choices[0].message.role == "assistant"


class TestOllamaClientErrorHandling:
    """Test error handling scenarios."""

    @patch("caribou.core.ollama_wrapper.requests.post")
    def test_http_error_propagates(self, mock_post):
        """Test that HTTP errors are propagated."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = Exception("HTTP 500")
        mock_post.return_value = mock_response

        client = OllamaClient()

        with pytest.raises(Exception, match="HTTP 500"):
            client.chat.completions.create(messages=[{"role": "user", "content": "Test"}])

    @patch("caribou.core.ollama_wrapper.requests.post")
    def test_no_message_in_response(self, mock_post):
        """Test error when no message object found in response."""
        mock_response = Mock()
        mock_response.text = '{"model": "llama3"}\n{"done": true}\n'
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        client = OllamaClient()

        with pytest.raises(ValueError, match="No message object found"):
            client.chat.completions.create(messages=[{"role": "user", "content": "Test"}])

    @patch("caribou.core.ollama_wrapper.requests.post")
    def test_invalid_json_in_response(self, mock_post):
        """Test error when response contains invalid JSON."""
        mock_response = Mock()
        mock_response.text = 'not valid json\n'
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        client = OllamaClient()

        with pytest.raises(json.JSONDecodeError):
            client.chat.completions.create(messages=[{"role": "user", "content": "Test"}])

    @patch("caribou.core.ollama_wrapper.requests.post")
    def test_empty_response(self, mock_post):
        """Test error when response is empty."""
        mock_response = Mock()
        mock_response.text = ''
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        client = OllamaClient()

        with pytest.raises(ValueError, match="No message object found"):
            client.chat.completions.create(messages=[{"role": "user", "content": "Test"}])

    @patch("caribou.core.ollama_wrapper.requests.post")
    def test_timeout_parameter(self, mock_post):
        """Test that timeout is set correctly."""
        mock_response = Mock()
        mock_response.text = '{"message": {"role": "assistant", "content": "Test"}}\n'
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        client = OllamaClient()
        client.chat.completions.create(messages=[{"role": "user", "content": "Test"}])

        call_args = mock_post.call_args
        assert call_args[1]["timeout"] == 300


class TestOllamaClientEdgeCases:
    """Test edge cases."""

    @patch("caribou.core.ollama_wrapper.requests.post")
    def test_empty_messages_list(self, mock_post):
        """Test with empty messages list."""
        mock_response = Mock()
        mock_response.text = '{"message": {"role": "assistant", "content": "Response"}}\n'
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        client = OllamaClient()
        client.chat.completions.create(messages=[])

        call_args = mock_post.call_args
        assert call_args[1]["json"]["messages"] == []

    @patch("caribou.core.ollama_wrapper.requests.post")
    def test_empty_content_in_message(self, mock_post):
        """Test parsing message with empty content."""
        mock_response = Mock()
        mock_response.text = '{"message": {"role": "assistant", "content": ""}}\n'
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        client = OllamaClient()
        response = client.chat.completions.create(messages=[{"role": "user", "content": "Test"}])

        assert response.choices[0].message.content == ""

    @patch("caribou.core.ollama_wrapper.requests.post")
    def test_whitespace_in_response(self, mock_post):
        """Test parsing response with extra whitespace."""
        mock_response = Mock()
        mock_response.text = '\n\n  {"message": {"role": "assistant", "content": "Test"}}  \n\n'
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        client = OllamaClient()
        response = client.chat.completions.create(messages=[{"role": "user", "content": "Test"}])

        assert response.choices[0].message.content == "Test"
