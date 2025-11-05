"""Tests for server-side usage logging functionality."""

from unittest.mock import AsyncMock, patch

import pytest

from llmring.schemas import LLMRequest, LLMResponse, Message
from llmring.service import LLMRing


@pytest.mark.asyncio
async def test_llmring_with_server_logs_usage():
    """Test that LLMRing logs usage to server when configured."""
    # Mock server client
    mock_server_client = AsyncMock()
    mock_server_client.post = AsyncMock(return_value={"log_id": "test-log-id"})

    # Create LLMRing with mocked server client
    with patch("llmring.server_client.ServerClient", return_value=mock_server_client):
        ring = LLMRing(
            origin="test",
            server_url="http://test-server",
            api_key="test-key",
        )

        # Mock provider response
        mock_response = LLMResponse(
            content="Test response",
            model="openai:gpt-4o-mini",
            usage={
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
            finish_reason="stop",
        )

        # Mock the provider's chat method
        with patch.object(ring, "get_provider") as mock_get_provider:
            mock_provider = AsyncMock()
            mock_provider.chat = AsyncMock(return_value=mock_response)
            mock_get_provider.return_value = mock_provider

            # Mock registry for cost calculation
            with patch.object(ring, "get_model_from_registry", return_value=None):
                # Mock calculate_cost to return cost info
                with patch.object(
                    ring,
                    "calculate_cost",
                    return_value={"total_cost": 0.001, "input_cost": 0.0005, "output_cost": 0.0005},
                ):
                    # Make a chat request
                    request = LLMRequest(
                        model="openai:gpt-4o-mini",
                        messages=[Message(role="user", content="Hello")],
                    )

                    response = await ring.chat(request)

                    # Verify response
                    assert response.content == "Test response"
                    assert response.usage["prompt_tokens"] == 100

                    # Verify usage was logged to server
                    mock_server_client.post.assert_called_once()
                    call_args = mock_server_client.post.call_args

                    assert call_args[0][0] == "/api/v1/log"
                    log_data = call_args[1]["json"]

                    assert log_data["model"] == "gpt-4o-mini"
                    assert log_data["provider"] == "openai"
                    assert log_data["input_tokens"] == 100
                    assert log_data["output_tokens"] == 50
                    assert log_data["origin"] == "test"
                    assert log_data["cost"] == 0.001


@pytest.mark.asyncio
async def test_llmring_without_server_no_logging():
    """Test that LLMRing without server doesn't attempt logging."""
    # Create LLMRing without server
    ring = LLMRing(origin="test")

    # Verify server_client is None
    assert ring.server_client is None

    # Mock provider response
    mock_response = LLMResponse(
        content="Test response",
        model="openai:gpt-4o-mini",
        usage={
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        },
        finish_reason="stop",
    )

    # Mock the provider's chat method
    with patch.object(ring, "get_provider") as mock_get_provider:
        mock_provider = AsyncMock()
        mock_provider.chat = AsyncMock(return_value=mock_response)
        mock_get_provider.return_value = mock_provider

        # Mock registry for cost calculation
        with patch.object(ring, "get_model_from_registry", return_value=None):
            with patch.object(ring, "calculate_cost", return_value=None):
                # Make a chat request
                request = LLMRequest(
                    model="openai:gpt-4o-mini",
                    messages=[Message(role="user", content="Hello")],
                )

                response = await ring.chat(request)

                # Verify response (no errors)
                assert response.content == "Test response"


@pytest.mark.asyncio
async def test_usage_logging_with_alias():
    """Test that usage logging includes alias information."""
    # Mock server client
    mock_server_client = AsyncMock()
    mock_server_client.post = AsyncMock(return_value={"log_id": "test-log-id"})

    # Create LLMRing with mocked server client
    with patch("llmring.server_client.ServerClient", return_value=mock_server_client):
        ring = LLMRing(
            origin="test",
            server_url="http://test-server",
            api_key="test-key",
        )

        # Mock provider response
        mock_response = LLMResponse(
            content="Test response",
            model="openai:gpt-4o-mini",
            usage={
                "prompt_tokens": 100,
                "completion_tokens": 50,
            },
            finish_reason="stop",
        )

        # Mock the provider and alias resolution
        with patch.object(ring, "get_provider") as mock_get_provider:
            mock_provider = AsyncMock()
            mock_provider.chat = AsyncMock(return_value=mock_response)
            mock_get_provider.return_value = mock_provider

            # Mock alias resolution to return a model string
            with patch.object(ring, "resolve_alias", return_value="openai:gpt-4o-mini"):
                with patch.object(ring, "get_model_from_registry", return_value=None):
                    with patch.object(
                        ring,
                        "calculate_cost",
                        return_value={
                            "total_cost": 0.001,
                            "input_cost": 0.0005,
                            "output_cost": 0.0005,
                        },
                    ):
                        # Make a chat request with an alias
                        request = LLMRequest(
                            model="fast",  # Using alias instead of direct model
                            messages=[Message(role="user", content="Hello")],
                        )

                        await ring.chat(request)

                        # Verify usage was logged with alias
                        mock_server_client.post.assert_called_once()
                        log_data = mock_server_client.post.call_args[1]["json"]

                        # Should include alias since "fast" doesn't contain ":"
                        assert log_data["alias"] == "fast"
                        assert log_data["model"] == "gpt-4o-mini"
                        assert log_data["provider"] == "openai"


@pytest.mark.asyncio
async def test_usage_logging_failure_doesnt_break_request():
    """Test that if logging to server fails, the request still succeeds."""
    # Mock server client that fails
    mock_server_client = AsyncMock()
    mock_server_client.post = AsyncMock(side_effect=Exception("Server error"))

    # Create LLMRing with mocked server client
    with patch("llmring.server_client.ServerClient", return_value=mock_server_client):
        ring = LLMRing(
            origin="test",
            server_url="http://test-server",
            api_key="test-key",
        )

        # Mock provider response
        mock_response = LLMResponse(
            content="Test response",
            model="openai:gpt-4o-mini",
            usage={
                "prompt_tokens": 100,
                "completion_tokens": 50,
            },
            finish_reason="stop",
        )

        # Mock the provider
        with patch.object(ring, "get_provider") as mock_get_provider:
            mock_provider = AsyncMock()
            mock_provider.chat = AsyncMock(return_value=mock_response)
            mock_get_provider.return_value = mock_provider

            with patch.object(ring, "get_model_from_registry", return_value=None):
                with patch.object(ring, "calculate_cost", return_value=None):
                    # Make a chat request
                    request = LLMRequest(
                        model="openai:gpt-4o-mini",
                        messages=[Message(role="user", content="Hello")],
                    )

                    # Should not raise exception despite logging failure
                    response = await ring.chat(request)

                    # Verify response is returned successfully
                    assert response.content == "Test response"


@pytest.mark.asyncio
async def test_streaming_usage_logging():
    """Test that streaming responses also log usage."""
    # Mock server client
    mock_server_client = AsyncMock()
    mock_server_client.post = AsyncMock(return_value={"log_id": "test-log-id"})

    # Create LLMRing with mocked server client
    with patch("llmring.server_client.ServerClient", return_value=mock_server_client):
        ring = LLMRing(
            origin="test",
            server_url="http://test-server",
            api_key="test-key",
        )

        # Mock streaming response
        async def mock_stream():
            from llmring.schemas import StreamChunk

            # Yield some chunks
            yield StreamChunk(delta="Hello", model="openai:gpt-4o-mini")
            yield StreamChunk(delta=" world", model="openai:gpt-4o-mini")
            # Final chunk with usage
            yield StreamChunk(
                delta="",
                model="openai:gpt-4o-mini",
                usage={"prompt_tokens": 100, "completion_tokens": 50},
                finish_reason="stop",
            )

        # Mock the provider
        with patch.object(ring, "get_provider") as mock_get_provider:
            mock_provider = AsyncMock()
            mock_provider.chat_stream = AsyncMock(return_value=mock_stream())
            mock_get_provider.return_value = mock_provider

            with patch.object(ring, "get_model_from_registry", return_value=None):
                with patch.object(ring, "calculate_cost", return_value={"total_cost": 0.001}):
                    # Make a streaming request
                    request = LLMRequest(
                        model="openai:gpt-4o-mini",
                        messages=[Message(role="user", content="Hello")],
                    )

                    # Consume the stream
                    chunks = []
                    async for chunk in ring.chat_stream(request):
                        chunks.append(chunk)

                    # Verify we got chunks
                    assert len(chunks) == 3

                    # Verify usage was logged after streaming completed
                    mock_server_client.post.assert_called_once()
                    log_data = mock_server_client.post.call_args[1]["json"]

                    assert log_data["input_tokens"] == 100
                    assert log_data["output_tokens"] == 50
