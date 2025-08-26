"""
Tests for LLM sampling capabilities in MCP clients.

This module tests server-initiated LLM sampling requests and integration
with the LLM service.
"""

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from llmring.schemas import LLMResponse
from llmring.mcp.client.llm_client import AsyncMCPClientWithLLM, MCPClientWithLLM


class TestMCPClientWithLLM:
    """Test LLM sampling in sync MCP client."""

    def test_initialization_default_llm_service(self):
        """Test client initialization with default LLM service."""
        with patch("llmring.mcp.client.llm_client.LLMRing") as mock_service:
            client = MCPClientWithLLM("http://test.example.com")
            mock_service.assert_called_once()
            assert client.enable_sampling is True
            assert client.max_sampling_tokens == 1000
            assert client.default_temperature == 0.7

    def test_initialization_custom_config(self):
        """Test client initialization with custom sampling config."""
        config = {
            "enabled": False,
            "max_tokens": 500,
            "temperature": 0.5,
            "allowed_models": ["gpt-4", "claude-3"],
        }

        client = MCPClientWithLLM(
            "http://test.example.com", default_model="gpt-4", sampling_config=config
        )

        assert client.enable_sampling is False
        assert client.max_sampling_tokens == 500
        assert client.default_temperature == 0.5
        assert client.allowed_models == ["gpt-4", "claude-3"]
        assert client.default_model == "gpt-4"

    def test_sampling_disabled(self):
        """Test sampling request when sampling is disabled."""
        config = {"enabled": False}
        client = MCPClientWithLLM("http://test.example.com", sampling_config=config)

        request = {
            "method": "sampling/createMessage",
            "params": {"messages": [{"role": "user", "content": "Hello"}]},
            "id": "test-123",
        }

        response = client.handle_sampling_request(request)

        assert "error" in response
        assert response["error"]["code"] == -32601
        assert "disabled" in response["error"]["message"].lower()

    def test_unsupported_method(self):
        """Test sampling request with unsupported method."""
        client = MCPClientWithLLM("http://test.example.com")

        request = {"method": "sampling/unsupportedMethod", "params": {}, "id": "test-123"}

        response = client.handle_sampling_request(request)

        assert response["id"] == "test-123"
        assert "error" in response
        assert response["error"]["code"] == -32601
        assert "not found" in response["error"]["message"].lower()

    def test_missing_messages(self):
        """Test sampling request without messages."""
        client = MCPClientWithLLM("http://test.example.com")

        request = {"method": "sampling/createMessage", "params": {}, "id": "test-123"}

        response = client.handle_sampling_request(request)

        assert response["id"] == "test-123"
        assert "error" in response
        assert response["error"]["code"] == -32602
        assert "required" in response["error"]["data"].lower()

    def test_invalid_message_format(self):
        """Test sampling request with invalid message format."""
        client = MCPClientWithLLM("http://test.example.com")

        request = {
            "method": "sampling/createMessage",
            "params": {"messages": [{"role": "user"}]},  # Missing content
            "id": "test-123",
        }

        response = client.handle_sampling_request(request)

        assert response["id"] == "test-123"
        assert "error" in response
        assert response["error"]["code"] == -32602
        assert "role" in response["error"]["data"] and "content" in response["error"]["data"]

    def test_missing_model(self):
        """Test sampling request without model and no default."""
        client = MCPClientWithLLM("http://test.example.com")

        request = {
            "method": "sampling/createMessage",
            "params": {"messages": [{"role": "user", "content": "Hello"}]},
            "id": "test-123",
        }

        response = client.handle_sampling_request(request)

        assert response["id"] == "test-123"
        assert "error" in response
        assert response["error"]["code"] == -32602
        assert "model" in response["error"]["data"].lower()

    def test_forbidden_model(self):
        """Test sampling request with model not in allowed list."""
        config = {"allowed_models": ["gpt-4"]}
        client = MCPClientWithLLM("http://test.example.com", sampling_config=config)

        request = {
            "method": "sampling/createMessage",
            "params": {
                "messages": [{"role": "user", "content": "Hello"}],
                "model": "forbidden-model",
            },
            "id": "test-123",
        }

        response = client.handle_sampling_request(request)

        assert response["id"] == "test-123"
        assert "error" in response
        assert response["error"]["code"] == -32603
        assert "forbidden" in response["error"]["message"].lower()

    def test_successful_sampling_request(self):
        """Test successful sampling request."""
        mock_llm_service = MagicMock()
        mock_response = LLMResponse(
            content="Hello! How can I help you?",
            model="gpt-4",
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            finish_reason="stop",
        )

        # Mock the async chat method
        async def mock_chat(request):
            return mock_response

        mock_llm_service.chat = Mock(return_value=mock_response)

        client = MCPClientWithLLM(
            "http://test.example.com", llmring=mock_llm_service, default_model="gpt-4"
        )

        request = {
            "method": "sampling/createMessage",
            "params": {
                "messages": [{"role": "user", "content": "Hello"}],
                "temperature": 0.8,
                "max_tokens": 100,
            },
            "id": "test-123",
        }

        with patch.object(client, "_run_async", return_value=mock_response):
            response = client.handle_sampling_request(request)

        assert response["id"] == "test-123"
        assert "result" in response
        result = response["result"]
        assert result["role"] == "assistant"
        assert result["content"]["type"] == "text"
        assert result["content"]["text"] == "Hello! How can I help you?"
        assert result["model"] == "gpt-4"
        assert result["stopReason"] == "stop"
        assert "usage" in result
        assert result["usage"]["inputTokens"] == 10
        assert result["usage"]["outputTokens"] == 20
        assert result["usage"]["totalTokens"] == 30

    def test_max_tokens_capping(self):
        """Test that max_tokens is capped at client limit."""
        mock_llm_service = MagicMock()
        mock_response = LLMResponse(content="Response", model="gpt-4")

        client = MCPClientWithLLM(
            "http://test.example.com",
            llmring=mock_llm_service,
            default_model="gpt-4",
            sampling_config={"max_tokens": 100},
        )

        request = {
            "method": "sampling/createMessage",
            "params": {
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 500,  # Exceeds client limit
            },
            "id": "test-123",
        }

        with patch.object(client, "_run_async", return_value=mock_response):
            with patch("llmring.mcp.client.llm_client.logger") as mock_logger:
                # Set up the warning method explicitly as a MagicMock
                mock_logger.warning = MagicMock()
                client.handle_sampling_request(request)
                mock_logger.warning.assert_called_once()
                assert "capped" in mock_logger.warning.call_args[0][0]

    def test_llm_service_exception(self):
        """Test handling of LLM service exceptions."""
        mock_llm_service = MagicMock()

        client = MCPClientWithLLM(
            "http://test.example.com", llmring=mock_llm_service, default_model="gpt-4"
        )

        request = {
            "method": "sampling/createMessage",
            "params": {"messages": [{"role": "user", "content": "Hello"}]},
            "id": "test-123",
        }

        with patch.object(client, "_run_async", side_effect=Exception("LLM error")):
            response = client.handle_sampling_request(request)

        assert response["id"] == "test-123"
        assert "error" in response
        assert response["error"]["code"] == -32603
        assert "internal error" in response["error"]["message"].lower()
        assert "LLM error" in response["error"]["data"]


class TestAsyncMCPClientWithLLM:
    """Test LLM sampling in async MCP client."""

    @pytest.mark.asyncio
    async def test_successful_async_sampling_request(self):
        """Test successful async sampling request."""
        mock_llm_service = MagicMock()
        mock_response = LLMResponse(
            content="Async response!",
            model="claude-3",
            usage={"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
            finish_reason="stop",
        )
        mock_llm_service.chat = AsyncMock(return_value=mock_response)

        client = AsyncMCPClientWithLLM(
            "http://test.example.com", llmring=mock_llm_service, default_model="claude-3"
        )

        request = {
            "method": "sampling/createMessage",
            "params": {"messages": [{"role": "user", "content": "Hello async"}]},
            "id": "async-123",
        }

        response = await client.handle_sampling_request(request)

        assert response["id"] == "async-123"
        assert "result" in response
        result = response["result"]
        assert result["content"]["text"] == "Async response!"
        assert result["model"] == "claude-3"
        assert "usage" in result

    @pytest.mark.asyncio
    async def test_async_sampling_disabled(self):
        """Test async sampling when disabled."""
        config = {"enabled": False}
        client = AsyncMCPClientWithLLM("http://test.example.com", sampling_config=config)

        request = {
            "method": "sampling/createMessage",
            "params": {"messages": [{"role": "user", "content": "Hello"}]},
            "id": "async-123",
        }

        response = await client.handle_sampling_request(request)

        assert "error" in response
        assert response["error"]["code"] == -32601

    @pytest.mark.asyncio
    async def test_async_llm_service_exception(self):
        """Test handling of async LLM service exceptions."""
        mock_llm_service = MagicMock()
        mock_llm_service.chat = AsyncMock(side_effect=Exception("Async LLM error"))

        client = AsyncMCPClientWithLLM(
            "http://test.example.com", llmring=mock_llm_service, default_model="gpt-4"
        )

        request = {
            "method": "sampling/createMessage",
            "params": {"messages": [{"role": "user", "content": "Hello"}]},
            "id": "async-123",
        }

        response = await client.handle_sampling_request(request)

        assert response["id"] == "async-123"
        assert "error" in response
        assert response["error"]["code"] == -32603
        assert "Async LLM error" in response["error"]["data"]


class TestLLMSamplingIntegration:
    """Test integration between MCP clients and LLM sampling."""

    def test_message_conversion(self):
        """Test conversion from request messages to Message objects."""
        mock_llm_service = MagicMock()
        mock_response = LLMResponse(content="Response", model="gpt-4")

        # Capture the LLM request that was made
        captured_request = None

        def capture_chat(request):
            nonlocal captured_request
            captured_request = request
            return mock_response

        mock_llm_service.chat = Mock(side_effect=capture_chat)

        client = MCPClientWithLLM(
            "http://test.example.com", llmring=mock_llm_service, default_model="gpt-4"
        )

        request = {
            "method": "sampling/createMessage",
            "params": {
                "messages": [
                    {"role": "system", "content": "You are helpful"},
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                    {"role": "user", "content": "How are you?"},
                ]
            },
            "id": "test-123",
        }

        with patch.object(client, "_run_async", return_value=mock_response):
            client.handle_sampling_request(request)

        # Verify that the chat method was called and capture the request
        mock_llm_service.chat.assert_called_once()
        captured_request = mock_llm_service.chat.call_args[0][0]

        # Verify the messages were converted correctly
        messages = captured_request.messages
        assert len(messages) == 4
        assert messages[0].role == "system"
        assert messages[0].content == "You are helpful"
        assert messages[1].role == "user"
        assert messages[1].content == "Hello"
        assert messages[2].role == "assistant"
        assert messages[2].content == "Hi there!"
        assert messages[3].role == "user"
        assert messages[3].content == "How are you?"

    def test_parameter_passing(self):
        """Test that sampling parameters are passed correctly to LLM service."""
        mock_llm_service = MagicMock()
        mock_response = LLMResponse(content="Response", model="custom-model")

        mock_llm_service.chat = Mock(return_value=mock_response)

        client = MCPClientWithLLM(
            "http://test.example.com", llmring=mock_llm_service, default_model="gpt-4"
        )

        request = {
            "method": "sampling/createMessage",
            "params": {
                "messages": [{"role": "user", "content": "Hello"}],
                "model": "custom-model",
                "temperature": 0.9,
                "max_tokens": 200,
                "response_format": {"type": "json"},
                "tools": [{"name": "test_tool"}],
                "tool_choice": "auto",
            },
            "id": "test-123",
        }

        with patch.object(client, "_run_async", return_value=mock_response):
            client.handle_sampling_request(request)

        # Verify that the chat method was called and capture the request
        mock_llm_service.chat.assert_called_once()
        captured_request = mock_llm_service.chat.call_args[0][0]

        # Verify all parameters were passed
        assert captured_request.model == "custom-model"
        assert captured_request.temperature == 0.9
        assert captured_request.max_tokens == 200
        assert captured_request.response_format == {"type": "json"}
        assert captured_request.tools == [{"name": "test_tool"}]
        assert captured_request.tool_choice == "auto"

    def test_response_format_mcp_compliant(self):
        """Test that responses are formatted according to MCP specification."""
        mock_llm_service = MagicMock()
        mock_response = LLMResponse(
            content="This is the LLM response",
            model="test-model",
            usage={"prompt_tokens": 15, "completion_tokens": 25, "total_tokens": 40},
            finish_reason="length",
        )

        client = MCPClientWithLLM(
            "http://test.example.com", llmring=mock_llm_service, default_model="test-model"
        )

        request = {
            "method": "sampling/createMessage",
            "params": {"messages": [{"role": "user", "content": "Test"}]},
            "id": "format-test",
        }

        with patch.object(client, "_run_async", return_value=mock_response):
            response = client.handle_sampling_request(request)

        # Verify MCP-compliant response format
        assert response["id"] == "format-test"
        assert "result" in response

        result = response["result"]
        assert result["role"] == "assistant"
        assert result["content"]["type"] == "text"
        assert result["content"]["text"] == "This is the LLM response"
        assert result["model"] == "test-model"
        assert result["stopReason"] == "length"

        usage = result["usage"]
        assert usage["inputTokens"] == 15
        assert usage["outputTokens"] == 25
        assert usage["totalTokens"] == 40
